"""Associative memory in the mushroom body (real plasticity in the connectome).

Rule (Hige et al. 2015, Neuron): when a Kenyon cell (KC) was active in the last
~second AND the dopamine neurons (DANs) of a compartment fire, the KC -> MBON
synapses of that compartment are depressed. Nothing else changes.

- PPL1 compartments (punishment): their MBONs promote approach, so depressing
  them for an odour makes the fly avoid that odour.
- PAM compartments (reward): their MBONs promote avoidance, so depressing them
  makes the fly approach.

Which DAN feeds which MBON is read from the connectome (DAN -> MBON synapses);
it matches the known compartment map (e.g. MBON11 <- PPL101, gamma1pedc).
"""
from pathlib import Path
import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"
CACHE = DATA / "mb_cache.npz"

TAU_ELIG = 1.0       # s, KC eligibility window
LEARN_RATE = 0.08    # per (DAN Hz * s) at full eligibility
W_FLOOR = 0.05       # a synapse never goes below 5 % of its original weight
TAU_FORGET = 600.0   # s of simulated time to recover (slow forgetting)
EXTINCTION = 0.05    # per s at full eligibility: smelling her with no dopamine undoes the memory
MIN_KCS = 8          # distinct KCs needed before the memory has an opinion about someone


def _build_cache(brain, ann):
    import pandas as pd
    kc = ann[ann.cell_class == "Kenyon_Cell"].idx.to_numpy()
    mb = ann[ann.cell_class == "MBON"]
    dan = ann[ann.cell_class == "DAN"]
    con = pd.read_parquet(DATA / "Connectivity_783.parquet",
                          columns=["Presynaptic_Index", "Postsynaptic_Index", "Connectivity"])
    dm = con[con.Presynaptic_Index.isin(set(dan.idx)) & con.Postsynaptic_Index.isin(set(mb.idx))]
    del con
    dtype = dan.set_index("idx").cell_type
    dm = dm.assign(dan=dtype.reindex(dm.Presynaptic_Index).values)
    # compartment of each MBON = the DAN type that sends it most synapses
    comp_of = (dm.groupby(["Postsynaptic_Index", "dan"]).Connectivity.sum().reset_index()
               .sort_values("Connectivity", ascending=False).drop_duplicates("Postsynaptic_Index")
               .set_index("Postsynaptic_Index").dan)
    comps = sorted(set(comp_of))
    comp_id = {c: i for i, c in enumerate(comps)}
    mb_comp = {int(m): comp_id[c] for m, c in comp_of.items()}
    # every KC -> MBON edge in the CSR
    e_idx, e_pre, e_comp = [], [], []
    for k in kc:
        s, e = brain.indptr[k], brain.indptr[k + 1]
        post = brain.indices[s:e]
        for off in np.flatnonzero(np.isin(post, list(mb_comp))):
            e_idx.append(s + off)
            e_pre.append(k)
            e_comp.append(mb_comp[int(post[off])])
    fam = np.array([0 if c.startswith("PPL1") else 1 if c.startswith("PAM") else 2 for c in comps])
    dan_of_comp = [dan[dan.cell_type == c].idx.to_numpy() for c in comps]
    np.savez(CACHE, kc=kc, e_idx=np.array(e_idx), e_pre=np.array(e_pre), e_comp=np.array(e_comp),
             fam=fam, comps=np.array(comps), dan_idx=np.concatenate(dan_of_comp),
             dan_len=np.array([len(x) for x in dan_of_comp]))


class MushroomMemory:
    def __init__(self, brain, ann):
        if not CACHE.exists():
            _build_cache(brain, ann)
        z = np.load(CACHE)
        self.brain = brain
        self.kc = z["kc"]
        self.e_idx, self.e_comp = z["e_idx"], z["e_comp"]
        kc_pos = {int(k): i for i, k in enumerate(self.kc)}
        self.e_kc = np.array([kc_pos[int(k)] for k in z["e_pre"]])   # edge -> KC position
        self.comps = [str(c) for c in z["comps"]]
        self.comp_fam = z["fam"]                    # 0 PPL1 (approach MBON), 1 PAM (avoid MBON)
        self.e_fam = self.comp_fam[self.e_comp]
        splits = np.cumsum(z["dan_len"])[:-1]
        self.dan_of_comp = np.split(z["dan_idx"], splits)
        self.ppl1 = np.concatenate([d for d, f in zip(self.dan_of_comp, self.comp_fam) if f == 0])
        self.pam = np.concatenate([d for d, f in zip(self.dan_of_comp, self.comp_fam) if f == 1])
        self.w0 = brain.weights[self.e_idx].copy()
        self.reset()

    def reset(self):
        self.brain.weights[self.e_idx] = self.w0
        self.elig = np.zeros(len(self.kc))
        self.profiles = {}                          # doll id -> KC activity profile of her odour

    # ----------------------------------------------------------------- learning
    def update(self, counts, dt, odor_owner=None, forget=True):
        """counts: spikes per neuron in the last tick. odor_owner: the doll whose odour
        clearly dominates right now, to remember which KCs code for her."""
        kc_spk = counts[self.kc] if counts is not None else np.zeros(len(self.kc))
        self.elig = self.elig * np.exp(-dt / TAU_ELIG) + kc_spk
        if odor_owner is not None and kc_spk.any():
            p = self.profiles.setdefault(odor_owner, np.zeros(len(self.kc)))
            p += np.minimum(kc_spk, 3)
        w = self.brain.weights
        dan_hz = (np.array([counts[d].sum() / max(1, len(d)) / dt for d in self.dan_of_comp])
                  if counts is not None else np.zeros(len(self.dan_of_comp)))
        elig = np.minimum(1.0, self.elig[self.e_kc])
        drive = dan_hz[self.e_comp] * elig
        hit = drive > 0
        if hit.any():                                # KC active + dopamine: depression
            e = self.e_idx[hit]
            w[e] = np.maximum(W_FLOOR * self.w0[hit], w[e] * np.exp(-LEARN_RATE * drive[hit] * dt))
        ext = (elig > 0.05) & ~hit                   # KC active, no dopamine: extinction
        if ext.any():
            e = self.e_idx[ext]
            w[e] += (self.w0[ext] - w[e]) * np.minimum(1.0, EXTINCTION * elig[ext] * dt)
        # slow forgetting toward the original weights (paused during sleep: consolidation)
        if forget:
            cur = w[self.e_idx]
            w[self.e_idx] = cur + (self.w0 - cur) * (dt / TAU_FORGET)

    # ----------------------------------------------------------------- readout
    def opinion(self, owner):
        """How the memory centre now responds to this doll's odour, from the current
        KC -> MBON weights: +1 = strongly attractive, -1 = strongly aversive, 0 = neutral.
        (MBON input for her KC pattern, relative to the naive weights.)"""
        p = self.profiles.get(owner)
        if p is None or (p > 0).sum() < MIN_KCS:     # not smelled enough to know her
            return None
        wk = p[self.e_kc]
        ratio = self.brain.weights[self.e_idx] / self.w0
        r = [(wk * ratio)[self.e_fam == f].sum() / max(1e-9, wk[self.e_fam == f].sum()) for f in (0, 1)]
        # punishment depresses approach MBONs (f=0), reward depresses avoid MBONs (f=1)
        return float(np.clip((r[0] - r[1]) * 1.5, -1.0, 1.0))

    # ----------------------------------------------------------------- save/load
    def to_save(self):
        ratio = self.brain.weights[self.e_idx] / self.w0
        changed = np.flatnonzero(ratio < 0.999)
        return dict(edges=changed.tolist(), ratio=np.round(ratio[changed], 4).tolist(),
                    profiles={str(k): {str(i): float(v) for i, v in zip(np.flatnonzero(p), p[p > 0])}
                              for k, p in self.profiles.items()})

    def load_save(self, s):
        self.reset()
        e = np.array(s["edges"], dtype=np.int64)
        if len(e):
            self.brain.weights[self.e_idx[e]] = self.w0[e] * np.array(s["ratio"])
        for k, d in s["profiles"].items():
            p = np.zeros(len(self.kc))
            p[np.array(list(map(int, d)), dtype=np.int64)] = list(d.values())
            self.profiles[int(k) if k.lstrip("-").isdigit() else k] = p   # doll ids / "p<place id>"
