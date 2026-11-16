"""The same whole-brain LIF model as brain.py, on an NVIDIA GPU (CuPy + CUDA).

Same equations, parameters, delays, refractory period, adaptation and rest
tolerance as the numba kernel; what changes is how the work is laid out:

- Every step integrates all 138,639 neurons in parallel (one thread each). Neurons
  at rest don't move under the equations, and every 10 steps they are snapped back
  to rest exactly like the CPU version does, so the result is the same model.
- Spikes are delivered one warp per spiking neuron, 32 synapses at a time
  (atomicAdd on the target conductances).
- Poisson drive uses a counter-based hash (seed, step, input) instead of a
  sequential RNG, so it's parallel and reproducible.
- The 100 steps of a game tick are recorded once as a CUDA Graph and replayed
  with a single launch: the CPU only touches the GPU once per tick.

Weights live on the GPU, but the host copy (`self.weights`) stays the reference
for the mushroom-body memory, which edits KC -> MBON synapses: those edges are
registered with `watch_weights` and re-uploaded before every step() call.
"""
import warnings

import numpy as np

with warnings.catch_warnings():               # CuPy warns when CUDA_PATH isn't set even
    warnings.simplefilter("ignore")           # though the pip CUDA libraries work fine
    import cupy as cp

from .brain import Brain

_SRC = r"""
extern "C" {

__device__ __forceinline__ float urand(unsigned long long seed, long long t, int k) {
    unsigned long long x = seed * 0x9E3779B97F4A7C15ULL
                         ^ ((unsigned long long)t * 0xD1B54A32D192ED03ULL)
                         ^ ((unsigned long long)k * 0x8CB92BA72F3D8DD7ULL);
    x += 0x9E3779B97F4A7C15ULL;                     // splitmix64 finaliser
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return (float)(x >> 40) * (1.0f / 16777216.0f);  // [0, 1)
}

// Spike ring with S = D + 1 slots: spikes of step t go to slot t % S and are delivered
// at step t + D, when (t + D) % S == (t - 1) % S... i.e. the slot read at step t is
// (t + 1) % S, which is never the one being written (t % S). Two kernels per step.

// 1) Poisson sensory drive + deliver the spikes emitted D steps ago (one warp per spiking
//    neuron, 32 synapses at a time)
__global__ void inputs(const long long* tp, int S, int n, const long long* indptr,
                       const int* indices, const float* w, float* g, const int* spk_buf,
                       const int* spk_n, const unsigned long long* seedp, const int* sidx,
                       const float* sp, const int* ns, float w_poi, unsigned char* live) {
    long long t = tp[0];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nthreads = gridDim.x * blockDim.x;
    int m = ns[0];
    for (int k = tid; k < m; k += nthreads)
        if (urand(seedp[0], t, k) < sp[k]) {
            int j = sidx[k];
            atomicAdd(&g[j], w_poi);
            live[j] = 1;
        }
    int slot = (int)((t + 1) % S);
    int cnt = spk_n[slot];
    int lane = threadIdx.x & 31;
    for (int k = tid >> 5; k < cnt; k += nthreads >> 5) {
        int i = spk_buf[(long long)slot * n + k];
        long long end = indptr[i + 1];
        for (long long e = indptr[i] + lane; e < end; e += 32) {
            int j = indices[e];
            atomicAdd(&g[j], w[e]);
            live[j] = 1;
        }
    }
}

// 2) integrate every neuron away from rest (live[j] == 1; a neuron at rest has v = v0,
//    g = a = 0, refr = 0, so the equations leave it unchanged and it is skipped); the last block to finish advances the clock and empties the
//    slot that was just delivered (it is the one written next step)
__global__ void integrate(long long* tp, int S, int n, float* v, float* g, float* a,
                          short* refr, int* spk_buf, int* spk_n, int* count, int* out, int* out_n,
                          int cap, float a_v, float a_g, float a_a, float a_inc, float v0,
                          float vrst, float vth, short rfc, float tol, unsigned int* done,
                          unsigned char* live) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    long long t = tp[0];
    if (j < n && live[j]) {
    int slot = (int)(t % S);
    short r = refr[j];
    float vj = v[j], gj = g[j], aj = a[j];
    if (r > 0) {
        r -= 1;                                   // refractory: nothing integrates
    } else {
        vj += a_v * (v0 - vj + gj - aj);
        gj *= a_g;
        aj *= a_a;
        if (vj > vth) {
            vj = vrst; gj = 0.0f; r = rfc; aj += a_inc;
            count[j] += 1;
            int p = atomicAdd(&spk_n[slot], 1);
            spk_buf[(long long)slot * n + p] = j;
            int q = atomicAdd(out_n, 1);
            if (q < cap) out[q] = j;
        }
    }
    // every 10 steps, neurons that relaxed back are put exactly at rest (as on the CPU)
    if (t % 10 == 0 && r <= 0 && fabsf(gj) <= tol && fabsf(vj - v0) <= tol && aj <= tol) {
        vj = v0; gj = 0.0f; aj = 0.0f;
        live[j] = 0;
    }
    v[j] = vj; g[j] = gj; a[j] = aj; refr[j] = r;
    }
    __shared__ bool last;
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        last = atomicInc(done, gridDim.x - 1) == gridDim.x - 1;   // wraps to 0 for next step
    }
    __syncthreads();
    if (last && threadIdx.x == 0) {
        spk_n[(t + 1) % S] = 0;
        tp[0] = t + 1;
    }
}

__global__ void probe(long long* tp) { }

__global__ void set_refr(short* refr, unsigned char* live, const long long* idx, int m, short val) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < m) {
        refr[idx[k]] = val;
        if (val > 0) live[idx[k]] = 1;            // refractory neurons must count down
    }
}

__global__ void not_at_rest(const float* v, const float* g, const short* refr, const float* a,
                            int n, float v0, float tol, int* total) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n && (fabsf(g[j]) > tol || fabsf(v[j] - v0) > tol || refr[j] > 0 || a[j] > tol))
        atomicAdd(total, 1);
}

}
"""

STIM_CAP = 1 << 18      # Poisson inputs per step (all sensory groups together: a few thousand)
OUT_CAP = 1 << 22       # spikes returned per step() call
GRAPH_STEPS = 100       # steps recorded per CUDA graph (one game tick at dt=0.2 ms)
TPB = 256


def available():
    """True if there is a usable NVIDIA GPU (and the kernels compile)."""
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            return False
        cp.RawModule(code=_SRC).get_function("probe")
        return True
    except Exception:  # no driver, no GPU, no NVRTC...
        return False


def device_name():
    name = cp.cuda.runtime.getDeviceProperties(0)["name"]
    return name.decode() if isinstance(name, bytes) else str(name)


class GpuBrain(Brain):
    backend = "gpu"

    def __init__(self, *a, **kw):
        self._mod = cp.RawModule(code=_SRC)
        self._k = {f: self._mod.get_function(f) for f in
                   ("inputs", "integrate", "set_refr", "not_at_rest")}
        self.stream = cp.cuda.Stream(non_blocking=True)
        self._plastic = np.empty(0, np.int64)
        self._graphs = {}
        self._dev = None
        super().__init__(*a, **kw)          # loads the connectome, then calls reset()

    # ----------------------------------------------------------------- device state
    def _alloc(self):
        n, S = self.n, self.delay_steps + 1      # ring of D + 1 slots (see the kernels)
        with self.stream:
            self.d_indptr = cp.asarray(self.indptr.astype(np.int64))
            self.d_indices = cp.asarray(self.indices.astype(np.int32))
            self.d_w = cp.asarray(self.weights.astype(np.float32))
            self.d_v = cp.empty(n, cp.float32)
            self.d_g = cp.zeros(n, cp.float32)
            self.d_a = cp.zeros(n, cp.float32)
            self.d_refr = cp.zeros(n, cp.int16)
            self.d_count = cp.zeros(n, cp.int32)
            self.d_spk_buf = cp.zeros(S * n, cp.int32)
            self.d_spk_n = cp.zeros(S, cp.int32)
            self.d_done = cp.zeros(1, cp.uint32)
            self.d_live = cp.zeros(n, cp.uint8)
            self.d_out = cp.zeros(OUT_CAP, cp.int32)
            self.d_out_n = cp.zeros(1, cp.int32)
            self.d_t = cp.zeros(1, cp.int64)
            self.d_seed = cp.zeros(1, cp.uint64)
            self.d_sidx = cp.zeros(STIM_CAP, cp.int32)
            self.d_sp = cp.zeros(STIM_CAP, cp.float32)
            self.d_ns = cp.zeros(1, cp.int32)
            self.d_tmp = cp.zeros(1, cp.int32)
        self._dev = True

    def reset(self):
        if self._dev is None:
            self._alloc()
        p = self.p
        with self.stream:
            self.d_v.fill(p["v_0"])
            for x in (self.d_g, self.d_a, self.d_refr, self.d_count, self.d_spk_n, self.d_t, self.d_ns,
                      self.d_done, self.d_live):
                x.fill(0)
        self.t = 0
        self.stim = {}
        self.live = np.empty(0, np.int64)   # kept for API compatibility; use n_live()

    def seed(self, s):
        self.rng = np.random.default_rng(s)
        with self.stream:
            self.d_seed.fill(int(s) & 0xFFFFFFFFFFFFFFFF)

    @property
    def spike_count(self):
        with self.stream:
            out = self.d_count.get(stream=self.stream)
        self.stream.synchronize()
        return out

    def watch_weights(self, edge_idx):
        """Edges whose host weights may change (plasticity): re-uploaded every step()."""
        self._plastic = np.union1d(self._plastic, np.asarray(edge_idx, np.int64))
        with self.stream:
            self._d_plastic = cp.asarray(self._plastic)

    def hold(self, idx, steps=30000):
        self._set_refr(idx, steps)

    def release(self, idx):
        self._set_refr(idx, 0)

    def _set_refr(self, idx, val):
        idx = np.asarray(idx, np.int64)
        if len(idx):
            with self.stream:
                d = cp.asarray(idx)
                self._k["set_refr"](((len(idx) + TPB - 1) // TPB,), (TPB,),
                                    (self.d_refr, self.d_live, d, np.int32(len(idx)), np.int16(val)))

    def n_live(self):
        with self.stream:
            self.d_tmp.fill(0)
            self._k["not_at_rest"](((self.n + TPB - 1) // TPB,), (TPB,),
                                   (self.d_v, self.d_g, self.d_refr, self.d_a, np.int32(self.n),
                                    np.float32(self.p["v_0"]), np.float32(self.rest_tol), self.d_tmp))
            r = int(self.d_tmp.get(stream=self.stream)[0])
        return r

    # ----------------------------------------------------------------- stepping
    def _launch_steps(self, m):
        """Queue m simulation steps on self.stream (2 kernels per step)."""
        p, dt, n, k = self.p, self.dt, self.n, self._k
        S = np.int32(self.delay_steps + 1)
        grid_n = ((n + TPB - 1) // TPB,)
        a_v, a_g, a_a = np.float32(dt / p["t_mbr"]), np.float32(1 - dt / p["tau"]), np.float32(1 - dt / p["tau_a"])
        inputs_args = (self.d_t, S, np.int32(n), self.d_indptr, self.d_indices, self.d_w, self.d_g,
                       self.d_spk_buf, self.d_spk_n, self.d_seed, self.d_sidx, self.d_sp, self.d_ns,
                       np.float32(p["w_syn"] * p["f_poi"]), self.d_live)
        integrate_args = (self.d_t, S, np.int32(n), self.d_v, self.d_g, self.d_a, self.d_refr, self.d_spk_buf,
                          self.d_spk_n, self.d_count, self.d_out, self.d_out_n, np.int32(OUT_CAP), a_v, a_g, a_a,
                          np.float32(p["a_inc"]), np.float32(p["v_0"]), np.float32(p["v_rst"]),
                          np.float32(p["v_th"]), np.int16(self.rfc_steps), np.float32(self.rest_tol), self.d_done,
                          self.d_live)
        for _ in range(m):
            k["inputs"]((80,), (TPB,), inputs_args)
            k["integrate"](grid_n, (TPB,), integrate_args)

    def _graph(self, m):
        g = self._graphs.get(m)
        if g is None:
            with self.stream:
                self.stream.begin_capture()
                self._launch_steps(m)
                g = self._graphs[m] = self.stream.end_capture()
        return g

    def step(self, n_steps=1):
        dt = self.dt
        if self.stim:
            sidx = np.concatenate([ix for ix, _ in self.stim.values()]).astype(np.int32)
            sp = np.concatenate([np.full(len(ix), r * dt * 1e-3, np.float32) for ix, r in self.stim.values()])
            if len(sidx) > STIM_CAP:
                raise ValueError(f"demasiadas entradas sensoriales a la vez ({len(sidx)} > {STIM_CAP})")
        else:
            sidx, sp = np.zeros(0, np.int32), np.zeros(0, np.float32)
        with self.stream:
            if len(sidx):
                self.d_sidx[:len(sidx)].set(sidx, stream=self.stream)
                self.d_sp[:len(sp)].set(sp, stream=self.stream)
            self.d_ns.fill(len(sidx))
            self.d_out_n.fill(0)
            if len(self._plastic):            # memory (KC -> MBON) may have changed these weights
                self.d_w[self._d_plastic] = cp.asarray(self.weights[self._plastic])
        full, rest = divmod(n_steps, GRAPH_STEPS)
        if full:
            g = self._graph(GRAPH_STEPS)
            for _ in range(full):
                g.launch(self.stream)
        if rest:
            self._graph(rest).launch(self.stream)
        self.t += n_steps
        with self.stream:
            nout = min(int(self.d_out_n.get(stream=self.stream)[0]), OUT_CAP)
            spikes = self.d_out[:nout].get(stream=self.stream) if nout else np.zeros(0, np.int32)
        self.stream.synchronize()
        return spikes.astype(np.int64)
