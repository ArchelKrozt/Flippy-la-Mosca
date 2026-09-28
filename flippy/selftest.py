"""Self-test: checks that the connectome still produces the key results.

    python3 -m flippy.selftest

Each check stimulates sensory/internal neurons of the whole-brain model and
reads the neurons the game relies on. Takes ~1-2 minutes.
"""
import json
import sys
import time

import numpy as np

from .brain import Brain
from .memory import MushroomMemory
from .neurons import load_groups

results = []


def check(name, ok, detail):
    results.append(ok)
    print(f"{'✔' if ok else '✖'} {name}: {detail}", flush=True)


def main():
    t0 = time.time()
    b = Brain(dt=0.2, params=dict(a_inc=0.5))
    ann, S, M = load_groups(b)
    cat = lambda *k: np.concatenate([S[x] for x in k])   # noqa: E731
    g = lambda q: ann.query(q).idx.to_numpy()             # noqa: E731

    def run(stims, steps=5000, seed=0):
        b.seed(seed)
        b.reset()
        for i, (ix, r) in enumerate(stims):
            b.set_stim(str(i), ix, r)
        b.step(steps)
        return b.spike_count / (steps * b.dt / 1000)      # Hz

    print(f"Cerebro cargado: {b.n:,} neuronas, {len(b.indices):,} conexiones "
          f"(motor {'numba' if hasattr(b, '_live_buf') else 'numpy'})\n")

    hz = run([(cat("sugar_L", "sugar_R"), 100)])
    check("Azúcar → MN9 (comer)", hz[M["feed"]][:2].mean() > 20,
          f"MN9 {hz[M['feed']][:2].mean():.0f} Hz (Shiu et al. 2024)")

    hz = run([(cat("loom_L", "loom_R"), 100)], steps=2500)
    check("Sombra que se acerca → fibra gigante (escape)", hz[M["escape"]].mean() > 20,
          f"DNp01 {hz[M['escape']].mean():.0f} Hz")

    L, R = g("cell_type=='DNa02' and side=='left'"), g("cell_type=='DNa02' and side=='right'")
    hz = run([(S["fly_seen_L"], 40)])
    check("Mosca a la izquierda → gira a la izquierda", hz[L].mean() > hz[R].mean() + 10,
          f"DNa02 izq {hz[L].mean():.0f} Hz / der {hz[R].mean():.0f} Hz")

    hz = run([(S["obstacle_L"], 30)])
    check("Obstáculo a la izquierda → gira a la derecha (LPLC1)", hz[R].mean() > hz[L].mean() + 5,
          f"DNa02 izq {hz[L].mean():.0f} Hz / der {hz[R].mean():.0f} Hz")

    lo = np.mean([run([(S["mating_drive"], 10)], seed=s)[M["receptive"]].mean() for s in range(3)])
    hi = np.mean([run([(S["mating_drive"], 10), (S["song"], 100), (S["song_vpoEN"], 60)], seed=s)[M["receptive"]].mean()
                  for s in range(3)])
    check("Época + canto → receptiva (vpoDN)", hi > lo + 4, f"vpoDN {lo:.1f} → {hi:.1f} Hz con canto")

    hz = run([(S["song"], 100), (S["song_vpoEN"], 60)])
    check("Fecundada + canto → rechazo (DNp13 > vpoDN)", hz[M["reject"]].mean() > hz[M["receptive"]].mean(),
          f"DNp13 {hz[M['reject']].mean():.1f} Hz vs vpoDN {hz[M['receptive']].mean():.1f} Hz (Wang et al. 2021)")

    hz = run([(cat("odor_bad_L", "odor_bad_R"), 40), (S["heat"], 40)], steps=2500)
    check("CO₂ y calor sin convulsión (corrección de signos ALLN)", (hz > 0).sum() < 5000,
          f"{(hz > 0).sum():,} neuronas activas (sin la corrección: ~10.000)")

    # associative memory: odour + punishment vs odour alone
    mem = MushroomMemory(b, ann)
    free = sorted({t for t in ann.cell_type.dropna() if str(t).startswith("ORN_")}
                  - {"ORN_DM1", "ORN_DM4", "ORN_DP1m", "ORN_VM7d", "ORN_V", "ORN_DA2", "ORN_DA1", "ORN_VA1v", "ORN_VA1d"})
    rng = np.random.default_rng(7)
    odor = {k: ann[ann.cell_type.isin([str(x) for x in rng.choice(free, 10, replace=False)])].idx.to_numpy()
            for k in "AB"}

    def present(k, dan=None):
        mem.elig[:] = 0
        b.reset()
        b.set_stim("o", odor[k], 80)
        for tick in range(25):
            if dan is not None and tick >= 10:
                b.set_stim("d", dan, 60)
            spk = b.step(100)
            mem.update(np.bincount(spk, minlength=b.n) if len(spk) else None, 0.02, k)
        b.set_stim("d", [], 0)

    for _ in range(3):
        present("A", mem.ppl1)
        present("B")
    a, bb = mem.opinion("A"), mem.opinion("B")
    check("Memoria: olor + dolor → aversión, olor solo → neutral",
          a is not None and bb is not None and a < -0.5 and abs(bb) < 0.3,
          f"opinión A {a:+.2f}, B {bb:+.2f} (Hige et al. 2015)")
    mem.reset()
    del b, mem                                  # one brain at a time: 8 GB machines

    # the game world: runs, and save/load is exact
    from .world import World
    w = World()
    w.set_heat(True)
    w.add("doll", 185, 100, name="Tenor", sex="M", mating=True, song=1, persistence=1)
    w.add("doll", 120, 150, name="Rosa", sex="F", mating=True)
    t1 = time.time()
    for _ in range(250):
        w.tick()
    speed = 5 / (time.time() - t1)
    s = json.loads(json.dumps(w.to_save(), default=float))
    before = json.dumps((w.time_ms, [d.traits for d in w.dolls], list(w.family.values()), w.chronicle), default=float)
    w.reset()
    w.load_save(s)
    after = json.dumps((w.time_ms, [d.traits for d in w.dolls], list(w.family.values()), w.chronicle), default=float)
    same = before == after
    check("Mundo: 5 s de simulación + guardar/cargar idéntico", same, f"velocidad {speed:.2f}× tiempo real")

    print(f"\n{sum(results)}/{len(results)} comprobaciones correctas en {time.time() - t0:.0f} s")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
