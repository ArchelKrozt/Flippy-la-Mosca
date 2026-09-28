"""Whole-brain leaky integrate-and-fire model of Drosophila (FlyWire v783).

Same equations/parameters as Shiu et al. 2024 (Nature, "A Drosophila computational
brain model reveals sensorimotor processing"), reimplemented in plain numpy with
event-driven spike propagation so it runs step-by-step (for a closed loop game)
on a laptop with 8 GB RAM.

    dv/dt = (v_0 - v + g) / t_mbr
    dg/dt = -g / tau
    spike: v > v_th  -> v = v_rst, g = 0, refractory t_rfc
    synapse: g_post += w_syn * (sign * n_synapses), after delay t_dly
"""
from pathlib import Path
import numpy as np

try:  # compiled kernel (~10x faster); falls back to plain numpy if numba is missing
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None

DATA = Path(__file__).resolve().parent.parent / "data"
CACHE = DATA / "brain_csr.npz"

PARAMS = dict(
    v_0=-52.0, v_rst=-52.0, v_th=-45.0,  # mV
    t_mbr=20.0, tau=5.0, t_rfc=2.2, t_dly=1.8,  # ms
    w_syn=0.275,  # mV per synapse
    # spike-frequency adaptation (not in the original model; 0 = off). Each spike adds
    # a_inc mV of hyperpolarising current that decays with tau_a. Breaks up the
    # self-sustained states a fixed-weight network otherwise gets stuck in.
    a_inc=0.0, tau_a=300.0,
    f_poi=250.0,  # Poisson input synapse scaling
)


def build_cache():
    """Convert the parquet edge list to a compact CSR (pre -> post) cache."""
    import pandas as pd
    comp = pd.read_csv(DATA / "Completeness_783.csv", index_col=0)
    con = pd.read_parquet(DATA / "Connectivity_783.parquet",
                          columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
    pre = con["Presynaptic_Index"].to_numpy(np.int32)
    post = con["Postsynaptic_Index"].to_numpy(np.int32)
    w = con["Excitatory x Connectivity"].to_numpy(np.float32)
    del con
    order = np.argsort(pre, kind="stable")
    pre, post, w = pre[order], post[order], w[order]
    n = len(comp)
    indptr = np.zeros(n + 1, np.int64)
    np.cumsum(np.bincount(pre, minlength=n), out=indptr[1:])
    np.savez(CACHE, indptr=indptr, indices=post, weights=w, root_ids=comp.index.to_numpy(np.int64))


class Brain:
    def __init__(self, dt=0.1, params=None, seed=0, fix_signs=True):
        if not CACHE.exists():
            build_cache()
        z = np.load(CACHE)
        self.indptr, self.indices = z["indptr"], z["indices"]
        self.p = dict(PARAMS, **(params or {}))
        self.weights = z["weights"] * np.float32(self.p["w_syn"])
        self.root_ids = z["root_ids"]
        self.id2idx = {int(r): i for i, r in enumerate(self.root_ids)}
        self.n = len(self.root_ids)
        if fix_signs:
            self.apply_sign_fixes()
        self.dt = dt
        self.rng = np.random.default_rng(seed)
        self.delay_steps = max(1, round(self.p["t_dly"] / dt))
        self.rfc_steps = round(self.p["t_rfc"] / dt)
        self.rest_tol = 0.2  # mV: below this a neuron is considered back at rest
        self.reset()

    def apply_sign_fixes(self):
        """Antennal-lobe local neurons (ALLN) whose predicted transmitter is a
        modulator (dopamine/serotonin/octopamine) are treated as excitatory by the
        original model; they form an excitatory loop that ignites a seizure-like
        state from any olfactory/thermo input. Biologically these LNs are mostly
        GABAergic, so make their outputs inhibitory."""
        ann = DATA / "annotations.tsv"
        if not ann.exists():
            return
        import pandas as pd
        a = pd.read_csv(ann, sep="\t", usecols=["root_id", "cell_class", "top_nt"], low_memory=False)
        bad = a[(a.cell_class == "ALLN") & a.top_nt.isin(["dopamine", "serotonin", "octopamine"])]
        for i in self.idx(bad.root_id):
            sl = slice(self.indptr[i], self.indptr[i + 1])
            self.weights[sl] = -np.abs(self.weights[sl])
        self.n_sign_fixed = len(bad)

    def seed(self, s):
        self.rng = np.random.default_rng(s)
        if njit is not None:
            _seed(int(s))

    def reset(self):
        n, p = self.n, self.p
        if njit is not None:  # buffers for the compiled kernel
            self._live_buf = np.zeros(n, np.int64)
            self._n_live = np.zeros(1, np.int64)
            self._spk_buf = np.zeros((self.delay_steps, n), np.int32)  # spikes waiting t_dly
            self._spk_n = np.zeros(self.delay_steps, np.int64)
            self._out = np.zeros(1 << 21, np.int64)
            self._out_n = np.zeros(1, np.int64)
        self.v = np.full(n, p["v_0"], np.float32)
        self.g = np.zeros(n, np.float32)
        self.a = np.zeros(n, np.float32)  # adaptation current (mV)
        self.refr = np.zeros(n, np.int16)  # remaining refractory steps
        # delayed synaptic input: per ring slot, list of (targets, weights)
        self.pending = [[] for _ in range(self.delay_steps)]
        self.live = np.empty(0, np.int64)  # neurons not at rest (only these are integrated)
        self.is_live = np.zeros(n, bool)
        self.t = 0
        self.stim = {}  # name -> (idx array, rate Hz) Poisson drive
        self.spike_count = np.zeros(n, np.int32)

    def idx(self, root_ids):
        return np.array([self.id2idx[int(r)] for r in root_ids if int(r) in self.id2idx], np.int64)

    def set_stim(self, name, idx, rate_hz):
        """Poisson drive (like Shiu et al.) on a group of neurons. rate 0 removes it."""
        if rate_hz <= 0:
            self.stim.pop(name, None)
        else:
            self.stim[name] = (np.asarray(idx, np.int64), float(rate_hz))

    def _targets(self, spk):
        starts, ends = self.indptr[spk], self.indptr[spk + 1]
        lens = ends - starts
        tot = int(lens.sum())
        # flat indices of all outgoing edges of spiking neurons
        offs = np.repeat(starts - np.concatenate(([0], np.cumsum(lens)[:-1])), lens)
        e = offs + np.arange(tot)
        return self.indices[e], self.weights[e]

    def step(self, n_steps=1):
        """Advance n_steps (event driven: neurons at rest are skipped).
        Returns indices of neurons that spiked (concatenated, may repeat)."""
        if njit is None:
            return self._step_numpy(n_steps)
        p, dt = self.p, self.dt
        if self.stim:
            stim_idx = np.concatenate([ix for ix, _ in self.stim.values()])
            stim_p = np.concatenate([np.full(len(ix), r * dt * 1e-3) for ix, r in self.stim.values()])
        else:
            stim_idx, stim_p = np.zeros(0, np.int64), np.zeros(0)
        self._out_n[0] = 0
        self.t = _kernel(n_steps, self.t, self.delay_steps, self.indptr, self.indices, self.weights,
                         self.v, self.g, self.a, self.refr, self.is_live, self._live_buf, self._n_live,
                         self._spk_buf, self._spk_n, stim_idx, stim_p,
                         np.float32(p["w_syn"] * p["f_poi"]), np.float32(dt / p["t_mbr"]),
                         np.float32(1 - dt / p["tau"]), np.float32(1 - dt / p["tau_a"]),
                         np.float32(p["a_inc"]), np.float32(p["v_0"]), np.float32(p["v_rst"]),
                         np.float32(p["v_th"]), self.rfc_steps, np.float32(self.rest_tol),
                         self.spike_count, self._out, self._out_n)
        self.live = self._live_buf[:self._n_live[0]]
        return self._out[:self._out_n[0]].copy()

    def _step_numpy(self, n_steps=1):
        p, dt = self.p, self.dt
        a_v = np.float32(dt / p["t_mbr"])
        a_g = np.float32(1 - dt / p["tau"])
        a_a = np.float32(1 - dt / p["tau_a"])
        adapt = p["a_inc"] > 0
        v0, vth = np.float32(p["v_0"]), np.float32(p["v_th"])
        w_poi = np.float32(p["w_syn"] * p["f_poi"])
        v, g, refr = self.v, self.g, self.refr
        spikes = []
        for _ in range(n_steps):
            slot = self.t % self.delay_steps
            new = []
            if self.pending[slot]:
                tg = np.concatenate([x[0] for x in self.pending[slot]])
                wg = np.concatenate([x[1] for x in self.pending[slot]])
                self.pending[slot] = []
                np.add.at(g, tg, wg)
                new.append(tg)
            for idx, rate in self.stim.values():
                hit = idx[self.rng.random(len(idx)) < rate * dt * 1e-3]
                if len(hit):
                    g[hit] += w_poi
                    new.append(hit)
            L = self.live
            if new:
                nw = np.concatenate(new)
                nw = nw[~self.is_live[nw]]
                if len(nw):
                    nw = np.unique(nw)
                    self.is_live[nw] = True
                    L = self.live = np.concatenate((L, nw))
            if len(L) == 0:
                self.t += 1
                continue
            r = refr[L]
            act = r <= 0
            La = L[act]
            va, ga = v[La], g[La]
            if adapt:
                aa = self.a[La]
                va += a_v * (v0 - va + ga - aa)
                self.a[La] = aa * a_a
            else:
                va += a_v * (v0 - va + ga)
            ga *= a_g
            v[La], g[La] = va, ga
            if not act.all():
                refr[L[~act]] -= 1
            spk = La[va > vth]
            if len(spk):
                v[spk] = p["v_rst"]
                if adapt:
                    self.a[spk] += p["a_inc"]
                g[spk] = 0
                refr[spk] = self.rfc_steps
                self.spike_count[spk] += 1
                tg, wg = self._targets(spk)
                if len(tg):
                    self.pending[slot].append((tg, wg))  # arrives delay_steps later
                spikes.append(spk)
            if self.t % 10 == 0:  # drop neurons that relaxed back to rest
                keep = (np.abs(g[L]) > self.rest_tol) | (np.abs(v[L] - v0) > self.rest_tol) | (refr[L] > 0) | (self.a[L] > self.rest_tol)
                rest = L[~keep]
                v[rest], g[rest], self.a[rest] = v0, 0, 0
                self.is_live[rest] = False
                self.live = L[keep]
            self.t += 1
        return np.concatenate(spikes) if spikes else np.empty(0, np.int64)


if njit is not None:
    @njit(cache=True)
    def _seed(s):
        np.random.seed(s)

    @njit(cache=True)
    def _kernel(n_steps, t, D, indptr, indices, weights, v, g, a, refr, is_live, live, n_live_arr,
                spk_buf, spk_n, stim_idx, stim_p, w_poi, a_v, a_g, a_a, a_inc, v0, vrst, vth, rfc,
                tol, count, out, out_n_arr):
        """Same equations as _step_numpy, compiled. Spikes are queued per ring slot and
        delivered D steps later; only neurons away from rest ("live") are integrated."""
        n_live = n_live_arr[0]
        out_n = 0
        cap = out.shape[0]
        for _ in range(n_steps):
            slot = t % D
            # 1) deliver spikes emitted D steps ago
            for k in range(spk_n[slot]):
                i = spk_buf[slot, k]
                for e in range(indptr[i], indptr[i + 1]):
                    j = indices[e]
                    g[j] += weights[e]
                    if not is_live[j]:
                        is_live[j] = True
                        live[n_live] = j
                        n_live += 1
            spk_n[slot] = 0
            # 2) Poisson sensory drive
            for k in range(stim_idx.shape[0]):
                if np.random.random() < stim_p[k]:
                    j = stim_idx[k]
                    g[j] += w_poi
                    if not is_live[j]:
                        is_live[j] = True
                        live[n_live] = j
                        n_live += 1
            # 3) integrate live neurons
            for k in range(n_live):
                j = live[k]
                if refr[j] > 0:
                    refr[j] -= 1
                    continue
                v[j] += a_v * (v0 - v[j] + g[j] - a[j])
                g[j] *= a_g
                a[j] *= a_a
                if v[j] > vth:
                    v[j] = vrst
                    g[j] = 0.0
                    refr[j] = rfc
                    a[j] += a_inc
                    count[j] += 1
                    spk_buf[slot, spk_n[slot]] = j
                    spk_n[slot] += 1
                    if out_n < cap:
                        out[out_n] = j
                        out_n += 1
            # 4) every 10 steps drop neurons that relaxed back to rest
            if t % 10 == 0:
                m = 0
                for k in range(n_live):
                    j = live[k]
                    if abs(g[j]) > tol or abs(v[j] - v0) > tol or refr[j] > 0 or a[j] > tol:
                        live[m] = j
                        m += 1
                    else:
                        v[j] = v0
                        g[j] = 0.0
                        a[j] = 0.0
                        is_live[j] = False
                n_live = m
            t += 1
        n_live_arr[0] = n_live
        out_n_arr[0] = out_n
        return t
