"""2D arena + fly body in closed loop with the connectome brain.

Loop every tick (TICK_MS of brain time):
  world -> sensory firing rates -> Poisson drive on sensory neurons
  brain.step()  (138k LIF neurons, 15M connections)
  descending-neuron spikes -> smoothed rates -> body motion

The FlyWire brain is from an adult female, so the real fly is female.
"""
import math
import numpy as np

from .brain import Brain
from .dolls import Doll, DEFAULT_TRAITS
from .neurons import load_groups, brain_map, ATTRACTIVE_ORN, AVERSIVE_ORN
from .memory import MushroomMemory
from . import experiments

W, H = 480, 300  # arena in pixels (changeable: see World.set_arena)
ARENAS = {"S": (320, 200), "M": (480, 300), "L": (640, 400)}
AREA_PER_FLY = 6400.0   # px² per adult fly: ~10 / 22 / 40 in the three arenas
TICK_MS = 20.0
ODOR_SIGMA = 45.0


class Thing:
    _id = 0

    def __init__(self, kind, x, y, **kw):
        Thing._id += 1
        self.id, self.kind, self.x, self.y = Thing._id, kind, float(x), float(y)
        self.r = kw.get("r", {"food": 7, "bitter": 7, "co2": 5, "heat": 22, "fan": 6, "stone": 9, "lamp": 5,
                              "leaf": 18}.get(kind, 6))
        self.amount = kw.get("amount", 100.0)
        self.label = kw.get("label", "")      # "Hoja 2", "Comida 1"... (places Flippy can remember)
        self.scent = kw.get("scent", [])      # a place's own odour (glomeruli), for place memory

    def to_json(self):
        return dict(id=self.id, kind=self.kind, x=round(self.x, 1), y=round(self.y, 1),
                    r=self.r, amount=round(self.amount, 1), label=self.label, scent=self.scent)


NAMES_F = ["Lola", "Pepa", "Rocío", "Maruja", "Chelo", "Nube", "Luna", "Sol", "Mimi", "Kiki", "Tula", "Brisa"]
NAMES_M = ["Toño", "Paco", "Chema", "Rulo", "Zumbi", "Pancho", "Tito", "Bruno", "Coco", "Nico", "Lalo", "Pipo"]
# life cycle, compressed (a real fly takes ~10 days from egg to adult)
EGG_S, LARVA_GROWTH, PUPA_S, MATURE_S = 8.0, 12.0, 8.0, 20.0
LIFESPAN_S = 150.0   # offspring only; like real flies, adults live ~5x their development time
GENERATION_S = EGG_S + LARVA_GROWTH + PUPA_S + MATURE_S   # rest after mating / laying
MAX_MATINGS = 3      # a doll dies after its third copulation
CLUTCH = 5           # eggs Flippy matures per copulation
SAVE_VERSION = 2
# glomeruli not used by any other sense: each fly gets 10 of them as its individual odour
SOCIAL_ORN = {"ORN_DA1", "ORN_VA1v", "ORN_VA1d"}
SCENT_SIZE, SCENT_HZ, SCENT_SIGMA = 10, 80.0, 25.0
PLACE_KINDS = {"leaf": "Hoja", "food": "Comida", "heat": "Calor"}   # places with their own odour
PLACE_SCENT_SIZE = 8
def max_dolls():
    return max(4, round(W * H / AREA_PER_FLY))


def max_brood():
    return round(max_dolls() * 1.3)
DAY_S, NIGHT_S, TWILIGHT_S = 150.0, 90.0, 15.0   # simulated seconds
VISUAL = ("loom_L", "loom_R", "fly_seen_L", "fly_seen_R", "obstacle_L", "obstacle_R", "light_L", "light_R")
INTERNAL = ("mating_drive", "egg_drive", "pain", "reward")


class World:
    MATING_S = 6.0

    def __init__(self, dt=0.2):
        self.brain = Brain(dt=dt, params=dict(a_inc=0.5))  # adaptation: see brain.PARAMS
        self.ann, self.sens_idx, self.mot_idx = load_groups(self.brain)
        self.map_xy = brain_map(self.brain, self.ann)
        self.mem = MushroomMemory(self.brain, self.ann)
        self.fn_idx = experiments.build(self.brain, self.ann)     # experiment mode catalogue
        self.silenced = []                                       # active silencing experiments
        self.silenced_idx = np.empty(0, np.int64)
        self.silence_id = 0
        # memory-centre outputs, to watch learning happen
        self.mot_idx["mbon_approach"] = np.unique(self.brain.indices[self.mem.e_idx[self.mem.e_fam == 0]])
        self.mot_idx["mbon_avoid"] = np.unique(self.brain.indices[self.mem.e_idx[self.mem.e_fam == 1]])
        used = set(ATTRACTIVE_ORN) | set(AVERSIVE_ORN) | SOCIAL_ORN
        orn = self.ann[self.ann.cell_type.astype(str).str.startswith("ORN_")]
        self.scent_orn = {t: g.idx.to_numpy() for t, g in orn.groupby("cell_type") if t not in used}
        self.scent_pool = sorted(self.scent_orn)
        self.steps_per_tick = round(TICK_MS / dt)
        self.gain = {"sugar": 1.0, "bitter": 1.0, "odor_good": 1.0, "odor_bad": 1.0,
                     "loom": 1.0, "wind": 1.0, "touch": 1.0, "heat": 1.0}
        self.base_walk = 0.35  # intrinsic exploratory drive (body CPG), brain modulates it
        # Assisted navigation: the brain decides *whether* (motivation), a body reflex
        # decides *where* from the left/right antenna difference. Needed because in
        # this model the side of an odour never reaches the descending neurons.
        self.assist = True
        self.reset()

    # ---------------------------------------------------------------- setup
    def reset(self):
        self.brain.reset()
        self.mem.reset()
        self.opinions = {}        # doll id -> what Flippy's memory centre thinks of her (-1..1)
        self.place_opinions = {}  # thing id -> what she thinks of that place (-1..1)
        self.learn_log = []       # what she has been learning, for the memory window
        self.learn_last = {}
        self.reward_pulse = 0.0   # "relief" after sleeping well under a leaf
        self.scent_owner = None
        self.pain, self.pain_from = 0.0, None
        self.flippy_scent = list(np.random.choice(self.scent_pool, SCENT_SIZE, replace=False))
        self.fly = dict(x=W / 2, y=H / 2, heading=0.0, speed=0.0, energy=60.0,
                        state="walk", state_t=0.0, proboscis=0.0, z=0.0, ovi=0.0)
        self.fly_in_heat = False  # "época de apareamiento" of the real fly -> pC1 drive
        self.fly_mated = False
        self.fly_sperm = None     # father of Flippy's eggs: dict(id, traits)
        self.eggs_ready = 0       # mature eggs waiting (internal state)
        self.eggs_pending = 0     # eggs still maturing from the last copulation
        self.egg_progress = 0.0
        self.fly_rest = 0.0       # out of season for one generation after laying
        self.fly_matings = 0
        self.chronicle = []       # births, copulations, eggs, deaths: the family history
        self.eggs_laid = 0
        self.ovi_acc = 0.0        # integrated oviDN activity
        self.brood = []           # eggs, larvae and pupae
        self.brood_id = 0
        self.family = {0: dict(id=0, name="Flippy", sex="F", gen=0, mother=None, father=None,
                               real=True, alive=True, born=0.0, died=None, matings=0, cause=None)}
        kx, ky = W / 320, H / 200
        self.things = []
        for kind, x, y in (("food", 60, 50), ("bitter", 260, 150), ("co2", 270, 40)):
            self.add(kind, x * kx, y * ky)
        self.dolls = []
        self.shadows = []  # looming threats: dict(x, y, r, t)
        self.sens_rates = {k: 0.0 for k in self.sens_idx}
        self.mot_rates = {k: 0.0 for k in self.mot_idx}
        self.orn_slow = {}  # slow trace of odour concentration per antenna (ORN adaptation)
        self.odor_slow = 0.0
        self.avoid_slow = 0.0
        self.chemo = "—"
        self.vision = {}          # what each eye detects (0..1), also shown in the UI
        self.light_slow = 0.0
        self.conc = {"good": (0.0, 0.0), "bad": (0.0, 0.0)}
        self.conc_slow = 0.0
        self.touch_recent = 0.0
        self.spike_buf = []
        self.time_ms = 0.0
        self.log = []
        self.wander = 0.0
        self.groom_block = 0.0
        self.mate_block = 0.0
        self.habituation = 0.0    # repeated harmless threats raise the take-off threshold
        self.clock = 30.0         # time of day (s since dawn); a day lasts DAY_S + NIGHT_S
        self.daynight = True
        self.sleep_p = 0.2        # sleep pressure 0..1: grows awake, drains asleep
        self.seek_t = 0.0
        self.awake_block = 0.0    # after a fright she stays awake a while (no sleep/wake flicker)
        self.touch_adapt = {"L": 0.0, "R": 0.0}   # mechanoreceptor adaptation to sustained contact
        self.startle_side = 0.0

    def add(self, kind, x, y, **kw):
        if kind == "shadow":
            self.shadows.append(dict(x=float(x), y=float(y), r=4.0, t=0.0))
        elif kind == "doll":
            d = Doll(x, y, **kw)
            self.dolls.append(d)
            self._register(d)
            t = d.traits
            self.event(f"Nace {t['name']} ({'macho' if d.male else 'hembra'})")
            return d
        else:
            t = Thing(kind, x, y, **kw)
            if kind in PLACE_KINDS and not t.label and not (kind == "food" and t.r < 5):  # crumbs aren't places
                n = 1 + sum(o.kind == kind for o in self.things)
                while any(o.label == f"{PLACE_KINDS[kind]} {n}" for o in self.things):
                    n += 1
                t.label = f"{PLACE_KINDS[kind]} {n}"
                t.scent = [str(g) for g in np.random.choice(self.scent_pool, PLACE_SCENT_SIZE, replace=False)]
            self.things.append(t)
            return t

    # ---------------------------------------------------------------- family
    def record(self, kind, text, gen=None, ids=()):
        """ids: every fly involved, so the history can be filtered per fly."""
        self.chronicle.append(dict(t=round(self.time_ms / 1000, 1), kind=kind, text=text, gen=gen,
                                   ids=[i for i in ids if i is not None]))
        if len(self.chronicle) > 3000:
            self.chronicle = self.chronicle[-3000:]

    def set_heat(self, on):
        """Manual toggle of Flippy's mating season. Turning it on forces it: it ends her
        rest and makes her receptive again even if she was fecundated."""
        self.fly_in_heat = bool(on)
        if on:
            self.fly_rest = 0.0
            self.fly_mated = False

    def _random_scent(self):
        return [str(x) for x in np.random.choice(self.scent_pool, SCENT_SIZE, replace=False)]

    def _child_scent(self, mom_scent, dad_scent):
        """Half of each parent's glomeruli: relatives smell alike (and share KCs)."""
        ms = list(mom_scent or self._random_scent())
        pick = [str(g) for g in np.random.choice(ms, SCENT_SIZE // 2, replace=False)]
        rest = [g for g in (dad_scent or self._random_scent()) if g not in pick]
        if len(rest) < SCENT_SIZE - len(pick):      # related parents share glomeruli: top up
            rest += [g for g in self.scent_pool if g not in pick and g not in rest]
        pick += [str(g) for g in np.random.choice(rest, SCENT_SIZE - len(pick), replace=False)]
        return pick

    def owner_name(self, key):
        """Doll id or "p<thing id>" -> readable name."""
        if isinstance(key, str) and key.startswith("p"):
            t = next((o for o in self.things if f"p{o.id}" == key), None)
            return t.label if t else "un lugar que ya no existe"
        return self.label(key) if key in self.family else "?"

    def learned(self, kind, owner, text):
        """Note what she is learning (at most once every 6 s per owner and kind)."""
        k = (kind, owner)
        if self.time_ms - self.learn_last.get(k, -1e9) < 6000:
            return
        self.learn_last[k] = self.time_ms
        self.learn_log = (self.learn_log + [dict(t=round(self.time_ms / 1000, 1), kind=kind, text=text)])[-40:]

    def hurt(self, doll):
        """A lunge that lands: pain (internal signal onto PPL1 dopamine neurons)."""
        self.pain, self.pain_from = 0.5, doll.id
        self.event(f"{doll.traits['name']} embiste a Flippy: dolor → dopamina PPL1")

    def _register(self, d):
        if not d.scent:
            d.scent = self._random_scent()
        self.family[d.id] = dict(id=d.id, name=d.traits["name"], sex=d.traits["sex"], gen=d.gen,
                                 mother=d.mother, father=d.father, real=False, alive=True,
                                 born=round(self.time_ms / 1000, 1), died=None, matings=0, cause=None)

    def mark_gone(self, ids, cause="retirada de la arena"):
        for i in ids:
            if i in self.family and self.family[i]["alive"]:
                self.family[i].update(alive=False, died=round(self.time_ms / 1000, 1), cause=cause)

    def lay_egg(self, x, y, mother, father):
        """father: dict(id, traits, scent) stored at copulation. The mother's traits and
        scent are copied into the egg too: she may die before it hatches."""
        if len(self.brood) >= max_brood():
            return False
        mom = next((d for d in self.dolls if d.id == mother), None)
        self.brood_id += 1
        self.brood.append(dict(id=self.brood_id, stage="egg", x=float(x), y=float(y), t=0.0,
                               grow=0.0, mother=mother, father=father, heading=0.0,
                               mom=dict(traits=dict(mom.traits), scent=list(mom.scent)) if mom else
                               dict(traits=None, scent=list(self.flippy_scent))))
        return True

    def label(self, fid):
        """Name, plus the generation when another fly in the history has the same name."""
        m = self.family.get(fid)
        if m is None:
            return "?"
        twin = any(o["name"] == m["name"] and o["id"] != fid for o in self.family.values())
        return f"{m['name']} (gen {m['gen']})" if twin else m["name"]

    def _inherit(self, mom, father, gen=None):
        """Child traits: mix of both parents plus a small mutation. Flippy has no trait
        sliders (her behaviour is her brain), so she passes on neutral values."""
        mt = (mom or {}).get("traits") or dict(DEFAULT_TRAITS)
        ft = (father or {}).get("traits") or dict(DEFAULT_TRAITS)
        sex = "F" if np.random.random() < 0.5 else "M"
        child = dict(sex=sex, mating=False)
        for k in ("aggression", "helper", "social", "speed", "song", "persistence", "size"):
            lo, hi = (0.7, 1.4) if k == "size" else (0.0, 1.0)
            v = 0.5 * (mt[k] + ft[k]) + np.random.normal(0, 0.1 * (hi - lo))
            child[k] = float(round(min(hi, max(lo, v)), 2))
        child["name"] = self._fresh_name(NAMES_F if sex == "F" else NAMES_M, gen)
        return child

    def _fresh_name(self, names, gen=None):
        """1) a name no fly in the history has had; 2) else one not used in this
        generation (shown as "Sol (gen 2)"); 3) else a numbered one (Sol II, Sol III...)."""
        used = [m["name"] for m in self.family.values()] + [d.traits["name"] for d in self.dolls]
        free = [n for n in names if n not in used]
        if free:
            return str(np.random.choice(free))
        in_gen = {m["name"] for m in self.family.values() if m["gen"] == gen}
        free = [n for n in names if n not in in_gen]
        if gen is not None and free:
            return str(min(free, key=lambda n: sum(u == n for u in used)))
        base = min(names, key=lambda n: sum(u == n or u.startswith(n + " ") for u in used))
        k = 2
        roman = ["", "", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
        while f"{base} {roman[k] if k < len(roman) else k}" in used:
            k += 1
        return f"{base} {roman[k] if k < len(roman) else k}"

    def update_brood(self, dt):
        for b in list(self.brood):
            b["t"] += dt
            if b["stage"] == "egg" and b["t"] > EGG_S:
                b["stage"], b["t"] = "larva", 0.0
                self.event("Eclosiona un huevo: sale una larva")
            elif b["stage"] == "larva":
                # larvae crawl to the nearest food and grow much faster while eating
                food = min((f for f in self.things if f.kind == "food" and f.amount > 0),
                           key=lambda f: math.hypot(f.x - b["x"], f.y - b["y"]), default=None)
                eating = False
                if food is not None:
                    d = math.hypot(food.x - b["x"], food.y - b["y"])
                    if d < food.r + 2:
                        eating = True
                        food.amount -= 0.4 * dt
                    elif d < 120:
                        b["heading"] = math.atan2(food.y - b["y"], food.x - b["x"])
                        b["x"] += math.cos(b["heading"]) * 5 * dt
                        b["y"] += math.sin(b["heading"]) * 5 * dt
                b["grow"] += dt * (1.0 if eating else 0.3)
                if b["grow"] > LARVA_GROWTH:
                    b["stage"], b["t"] = "pupa", 0.0
                    self.event("Una larva se convierte en pupa")
            elif b["stage"] == "pupa" and b["t"] > PUPA_S and len(self.dolls) < max_dolls():
                self.brood.remove(b)
                mother_gen = 0 if b["mother"] == 0 else self.family.get(b["mother"], {}).get("gen", 0)
                father_gen = self.family.get((b["father"] or {}).get("id"), {}).get("gen", 0)
                traits = self._inherit(b.get("mom"), b["father"], max(mother_gen, father_gen) + 1)
                d = Doll(b["x"], b["y"], **traits)
                d.gen = max(mother_gen, father_gen) + 1
                d.mother, d.father = b["mother"], (b["father"] or {}).get("id")
                d.scent = self._child_scent((b.get("mom") or {}).get("scent"), (b["father"] or {}).get("scent"))
                for z in self.dolls:                    # family ties
                    kin = 50 if z.id in (d.mother, d.father) else \
                        30 if (z.mother, z.father) == (d.mother, d.father) else 0
                    if kin:
                        z.like(d.id, kin)
                        d.like(z.id, kin)
                d.adult_at = MATURE_S
                d.lifespan = LIFESPAN_S + np.random.uniform(-20, 20)
                d.traits["mating"] = bool(np.random.random() < 0.6)  # will court/breed once adult
                self.dolls.append(d)
                self._register(d)
                mom, dad = self.label(d.mother), self.label(d.father)
                msg = (f"¡Nace {self.label(d.id)} ({'macho' if d.male else 'hembra'}), "
                       f"hij{'o' if d.male else 'a'} de {mom} y {dad}!")
                self.event(msg)
                self.record("birth", msg.strip("¡!"), d.gen, ids=(d.id, d.mother, d.father))

    def _solid(self, obj_x, obj_y, rad):
        """Push a body out of stones. Returns the corrected position."""
        for s in self.things:
            if s.kind == "stone":
                dx, dy = obj_x - s.x, obj_y - s.y
                d, need = math.hypot(dx, dy), s.r + rad
                if d < need:
                    if d < 1e-6:
                        dx, dy, d = 1.0, 0.0, 1.0
                    obj_x, obj_y = s.x + dx / d * need, s.y + dy / d * need
        return obj_x, obj_y

    def _separate(self):
        """Bodies don't overlap: push flies apart (copulating pairs and fighters excepted)."""
        bodies = [d for d in self.dolls if d.state not in ("copulate", "fight")]
        f = self.fly
        for i, a in enumerate(bodies):
            for b in bodies[i + 1:]:
                dx, dy = b.x - a.x, b.y - a.y
                dist, need = math.hypot(dx, dy), 5.0 * (a.traits["size"] + b.traits["size"]) / 2
                if 1e-6 < dist < need:
                    push = (need - dist) / 2 / dist
                    a.x, a.y, b.x, b.y = a.x - dx * push, a.y - dy * push, b.x + dx * push, b.y + dy * push
            if f["state"] not in ("mate", "fly"):
                dx, dy = a.x - f["x"], a.y - f["y"]
                dist = math.hypot(dx, dy)
                if 1e-6 < dist < 4.0:                   # touching is allowed, overlapping is not
                    a.x, a.y = f["x"] + dx / dist * 4.0, f["y"] + dy / dist * 4.0
            a.x, a.y = self._solid(a.x, a.y, 3.0)
            a.x, a.y = min(max(a.x, 4), W - 4), min(max(a.y, 4), H - 4)
        if f["state"] != "fly":
            f["x"], f["y"] = self._solid(f["x"], f["y"], 3.0)

    def _relations(self, dt):
        """Dolls who spend time together get used to each other; feelings fade slowly."""
        for a in self.dolls:
            for k in list(a.affinity):
                a.affinity[k] *= math.exp(-dt / 400.0)
            for b in self.dolls:
                if a is not b and math.hypot(a.x - b.x, a.y - b.y) < 25 and a.affinity.get(b.id, 0) < 30:
                    a.like(b.id, 1.0 * dt)

    def update_ages(self):
        """Offspring die of old age; every doll dies after its third copulation (a female
        first lays the eggs of that copulation)."""
        for d in list(self.dolls):
            if d.state == "copulate":
                continue
            o = "o" if d.male else "a"
            if d.lifespan and d.age > d.lifespan:
                why = f"muere de viej{o}"
            elif d.matings >= MAX_MATINGS and (d.male or d.eggs_left == 0):
                why = f"muere tras su {MAX_MATINGS}.ª cópula"
            else:
                continue
            self.dolls.remove(d)
            self.mark_gone([d.id], cause=why)
            self.event(f"{d.traits['name']} {why}")
            self.record("death", f"{d.traits['name']} {why}", d.gen, ids=(d.id,))

    def remove_near(self, x, y):
        self.things = [t for t in self.things if math.hypot(t.x - x, t.y - y) > t.r + 3]
        gone = [d.id for d in self.dolls if math.hypot(d.x - x, d.y - y) <= 6]
        self.mark_gone(gone)
        self.dolls = [d for d in self.dolls if d.id not in gone]
        self.brood = [b for b in self.brood if math.hypot(b["x"] - x, b["y"] - y) > 4]

    def doll(self, did):
        return next((d for d in self.dolls if d.id == did), None)

    def event(self, msg):
        self.log = (self.log + [f"{self.time_ms / 1000:6.1f}s  {msg}"])[-9:]

    # ---------------------------------------------------------------- senses
    def _field(self, kind, x, y, sigma=ODOR_SIGMA):
        c = 0.0
        for t in self.things:
            if t.kind == kind and t.amount > 0:
                d2 = (t.x - x) ** 2 + (t.y - y) ** 2
                c += math.exp(-d2 / (2 * sigma ** 2)) * min(1.0, t.amount / 50)
        return min(c, 1.5)

    def _doll_field(self, x, y, weight, sigma=25.0):
        c = 0.0
        for d in self.dolls:
            w = weight(d)
            if w:
                c += w * math.exp(-((d.x - x) ** 2 + (d.y - y) ** 2) / (2 * sigma ** 2))
        return min(c, 1.5)

    def _touching(self, kind, x, y):
        return any(t.kind == kind and t.amount > 0 and math.hypot(t.x - x, t.y - y) < t.r + 1
                   for t in self.things)

    def _orn(self, key, c, gain, tonic=0.25, phasic=4.0):
        """Olfactory receptor neurons are strongly phasic: they fire mostly when the
        concentration rises. slow = adapted baseline (tau 0.6 s)."""
        slow = self.orn_slow.get(key, c)
        slow += (c - slow) * (TICK_MS / 600)
        self.orn_slow[key] = slow
        return min(90.0, 40.0 * gain * (tonic * c + phasic * max(0.0, c - slow)))

    def sense(self):
        f = self.fly
        h, x, y = f["heading"], f["x"], f["y"]
        # antennae / front legs 4 px ahead, +-40 deg
        pts = {s: (x + 4 * math.cos(h + a), y + 4 * math.sin(h + a)) for s, a in (("L", -0.7), ("R", 0.7))}
        r = {}
        airborne = f["state"] == "fly"
        hunger = 1 - f["energy"] / 100
        self.conc = {"good": tuple(self._field("food", *pts[s]) for s in "LR"),
                     "bad": tuple(self._field("co2", *pts[s], 30) + self._heat_at(*pts[s]) for s in "LR")}
        for s, (px, py) in pts.items():
            r[f"sugar_{s}"] = 0 if airborne else 110.0 * self._touching("food", px, py) * self.gain["sugar"]
            r[f"bitter_{s}"] = 0 if airborne else 110.0 * self._touching("bitter", px, py) * self.gain["bitter"]
            r[f"odor_good_{s}"] = self._orn(f"good{s}", self._field("food", px, py),
                                            self.gain["odor_good"] * (0.6 + 0.9 * hunger))
            r[f"odor_bad_{s}"] = self._orn(f"bad{s}", self._field("co2", px, py, 30), self.gain["odor_bad"])
            wall = px < 2 or py < 2 or px > W - 2 or py > H - 2
            r[f"touch_{s}"] = 80.0 * wall * self.gain["touch"]
        # vision: shadows (loom) and other flies (small objects -> LC10a/LC11; fast approach -> loom)
        loom = {"L": 0.0, "R": 0.0}
        seen = {"L": 0.0, "R": 0.0}
        for sh in self.shadows:
            ang = _ang(math.atan2(sh["y"] - y, sh["x"] - x) - h)
            dist = max(4.0, math.hypot(sh["x"] - x, sh["y"] - y))
            loom["L" if ang < 0 else "R"] += min(1.0, sh["r"] / dist * 1.5)
        for d in self.dolls:
            dx, dy = d.x - x, d.y - y
            dist = max(3.0, math.hypot(dx, dy))
            theta = 3.0 * d.traits["size"] / dist            # angular size (rad)
            ang = _ang(math.atan2(dy, dx) - h)
            side = "L" if ang < 0 else "R"
            if abs(ang) < 2.6 and dist < 120:                  # fly eyes see ~300 deg
                seen[side] += min(1.0, theta * 4)
                if d.theta_prev is not None:
                    # LC4/LPLC2 are tuned to fast approaches: a lunge (60-120 px/s) looms,
                    # a fly walking toward her (<35 px/s) does not. Only the other fly's own
                    # approach counts: flies discount the image motion they cause themselves
                    # (otherwise running away from one doll "looms" every other doll).
                    closing = d.speed * math.cos(d.heading - math.atan2(-dy, -dx)) if d.state != "flyaway" else 0.0
                    loom[side] += max(0.0, min(1.0, (closing - 35) / 40)) * min(1.0, theta * 3)
            d.theta_prev, d.dist_prev = theta, dist
            if d.touching:
                r[f"touch_{side}"] = max(r[f"touch_{side}"], 80.0 * self.gain["touch"])
        # touch receptors are phasic: a new bump is felt fully, sustained contact (walking
        # along a wall, a fly leaning on her) adapts within ~0.6 s
        for s in "LR":
            a = self.touch_adapt[s]
            a = min(1.0, a + (TICK_MS / 600)) if r[f"touch_{s}"] > 0 else max(0.0, a - TICK_MS / 300)
            self.touch_adapt[s] = a
            r[f"touch_{s}"] *= 1.0 - 0.85 * a
        for s in "LR":
            r[f"loom_{s}"] = 150.0 * min(1.0, loom[s]) * self.gain["loom"]
            r[f"fly_seen_{s}"] = 40.0 * min(1.0, seen[s])  # >40 Hz only costs speed
        # obstacles ahead (walls, stones): LPLC1 codes an object *approaching* while she
        # walks, i.e. angular expansion ~ approach speed / distance. Standing still = 0.
        # (Driving it all the time made it leak into the giant fiber: 20 jumps/min.)
        v = max(0.0, f["speed"]) if f["state"] == "walk" else 0.0
        for s, sgn in (("L", -1), ("R", 1)):
            exp = 0.0
            for deg in (15, 40, 65):
                rad = math.radians(deg)
                d = self.ray(x, y, h + sgn * rad, 50)
                exp = max(exp, v * math.cos(rad) / max(d, 3.0))
            self.vision[f"obstacle_{s}"] = min(1.0, exp / 1.5)
            r[f"obstacle_{s}"] = 25.0 * self.vision[f"obstacle_{s}"]
        # light: each eye (+-60 deg from the heading) collects brightness from the lamps
        bright = {"L": 0.0, "R": 0.0}
        for t in self.things:
            if t.kind == "lamp":
                d = math.hypot(t.x - x, t.y - y)
                ang = _ang(math.atan2(t.y - y, t.x - x) - h)
                for s, eye in (("L", -1.0), ("R", 1.0)):
                    bright[s] += max(0.0, math.cos(ang - eye)) / (1 + (d / 70) ** 2)
        for s in "LR":
            self.vision[f"light_{s}"] = min(1.0, bright[s])
            r[f"light_{s}"] = 100.0 * self.vision[f"light_{s}"]
        wind = sum(max(0.0, 1 - math.hypot(t.x - x, t.y - y) / 120) for t in self.things if t.kind == "fan")
        r["wind"] = 60.0 * min(1.0, wind) * self.gain["wind"]
        heat = sum(max(0.0, 1 - math.hypot(t.x - x, t.y - y) / t.r) for t in self.things if t.kind == "heat")
        r["heat"] = 80.0 * min(1.0, heat) * self.gain["heat"]
        # social chemistry and sound from dolls
        r["cva"] = 30.0 * self._doll_field(x, y, lambda d: (2.0 if d.traits["mating"] else 1.0) * d.male)
        r["fly_odor"] = 20.0 * self._doll_field(
            x, y, lambda d: 2.0 if d.traits["mating"] and not d.male else 1.0)
        # song loudness grows with the singers' quality (measured: vpoDN rises 4 -> 13 Hz
        # as vpoEN goes 0 -> 60 Hz, so better singers really are more convincing)
        q = min(1.0, sum(0.15 + 0.85 * d.traits["song"] for d in self.dolls
                         if d.singing and math.hypot(d.x - x, d.y - y) < 30))
        r["song"] = 100.0 * q
        r["song_vpoEN"] = 60.0 * q
        # internal state: mating drive (a mated female stops being receptive)
        r["mating_drive"] = 10.0 if self.fly_in_heat and not self.fly_mated else 0.0
        # internal state: mature eggs push SMP550 -> oviDN (the brain decides when to lay)
        r["egg_drive"] = 40.0 if self.eggs_ready >= 1 and f["state"] not in ("fly", "mate") else 0.0
        # individual odours of nearby flies (10 glomeruli each) -> KCs of the memory centre
        best, self.scent_owner, total = 0.0, None, 0.0
        for k in [k for k in self.brain.stim if k.startswith("scent_")]:
            self.brain.set_stim(k, [], 0)
        # individual odours of flies AND places (each leaf, food patch, hot spot smells of
        # itself). Only the two strongest reach the brain: the rest are masked (and it keeps
        # the brain fast)
        sources = []
        for d in self.dolls:
            sources.append((SCENT_HZ * math.exp(-((d.x - x) ** 2 + (d.y - y) ** 2) / (2 * SCENT_SIGMA ** 2)), d.id, d.scent))
        for t in self.things:
            if t.scent and (t.kind != "food" or t.amount > 0):
                sig = t.r + 15
                sources.append((SCENT_HZ * math.exp(-((t.x - x) ** 2 + (t.y - y) ** 2) / (2 * sig ** 2)), f"p{t.id}", t.scent))
        for rate, key, scent in sorted(sources, key=lambda z: -z[0])[:2]:
            if rate > 3:
                ix = np.concatenate([self.scent_orn[g] for g in scent if g in self.scent_orn])
                self.brain.set_stim(f"scent_{key}", ix, rate)
                total += rate
                if rate > best:
                    best, self.scent_owner = rate, key
        if best < 30 or total - best > 0.3 * best:
            self.scent_owner = None                   # too faint, or a mixture of several flies
        r["scent"] = min(150.0, total)
        # reinforcement (internal signals: nociception and taste reward reach the dopamine
        # neurons through pathways that are not in this brain-only connectome)
        if r["heat"] > 30:                               # heat is noxious: punishment
            self.pain = max(self.pain, TICK_MS / 1000 * 2)
        r["pain"] = 60.0 if self.pain > 0 else 0.0
        r["reward"] = 60.0 if f["state"] == "feed" or self.reward_pulse > 0 else 0.0
        self.brain.set_stim("pain", self.mem.ppl1, r["pain"])
        self.brain.set_stim("reward", self.mem.pam, r["reward"])
        # shade under a leaf dims vision; asleep, every external sense is attenuated
        # (a sleeping fly has a raised arousal threshold); internal states are not
        shade = 0.35 if self.leaf_at(x, y) else 1.0
        arousal = 0.35 if f["state"] == "sleep" else 1.0
        for k in r:
            if k in VISUAL:
                r[k] *= shade * arousal
            elif k not in INTERNAL:
                r[k] *= arousal
        for k in [k for k in self.brain.stim if k.startswith("scent_")]:
            ix, rate = self.brain.stim[k]
            self.brain.set_stim(k, ix, rate * arousal)
        self.sens_rates = r
        for k, rate in r.items():
            if k in self.sens_idx:
                self.brain.set_stim(k, self.sens_idx[k], rate)

    # ---------------------------------------------------------------- tick
    def tick(self):
        self.sense()
        if len(self.silenced_idx):                  # silenced neurons can't fire (optogenetic block)
            self.brain.refr[self.silenced_idx] = 30000
        spk = self.brain.step(self.steps_per_tick)
        self.time_ms += TICK_MS
        if len(spk):
            self.spike_buf.append(spk)
        # smoothed rate (Hz per neuron) of each motor group, tau ~ 100 ms
        a = TICK_MS / 100.0
        counts = np.bincount(spk, minlength=self.brain.n) if len(spk) else None
        for k, ix in self.mot_idx.items():
            n = counts[ix].sum() if counts is not None else 0
            hz = n / max(1, len(ix)) / (TICK_MS / 1000)
            self.mot_rates[k] += a * (hz - self.mot_rates[k])
        dt = TICK_MS / 1000
        if self.daynight:
            self.clock = (self.clock + dt) % (DAY_S + NIGHT_S)
        self.mem.update(counts, dt, self.scent_owner, forget=self.fly["state"] != "sleep")
        own = self.scent_owner
        if own is not None and (self.pain > 0 or self.sens_rates.get("reward", 0) > 0):
            who = self.owner_name(own)
            if self.pain > 0:
                cause = "pasa calor" if self.sens_rates.get("heat", 0) > 30 else "la golpean"
                self.learned("bad", own, f"{cause} oliendo a {who} → lo asocia con dolor (PPL1)")
            elif self.reward_pulse > 0:
                self.learned("good", own, f"durmió tranquila oliendo a {who} → alivio, le gusta más (PAM)")
            else:
                self.learned("good", own, f"come oliendo a {who} → lo asocia con algo bueno (PAM)")
        self.pain = max(0.0, self.pain - dt)
        self.reward_pulse = max(0.0, self.reward_pulse - dt)
        if int(self.time_ms / TICK_MS) % 25 == 0:      # refresh opinions twice a second
            online = self.memory_online()
            self.opinions = {d.id: self.mem.opinion(d.id) if online else None for d in self.dolls}
            self.place_opinions = {t.id: self.mem.opinion(f"p{t.id}") if online else None
                                   for t in self.things if t.scent}
            self._relations(0.5)
        self.body(counts)
        for d in self.dolls:
            d.update(self, TICK_MS / 1000)
        self._separate()
        self.social()
        self.reproduce(TICK_MS / 1000)
        self.update_brood(TICK_MS / 1000)
        self.update_ages()
        self.update_world()

    def body(self, counts):
        f, m, dt = self.fly, self.mot_rates, TICK_MS / 1000
        f["state_t"] += dt
        if f["state"] == "mate":
            if f["state_t"] > self.MATING_S:
                f["state"], f["state_t"] = "walk", 0.0
                self.event("Fin de la cópula: ya está fecundada y rechazará a otros machos")
            return
        # --- sleep (internal state). What wakes her comes from the brain: the giant fiber,
        # the looming alarm DNs, touch; or morning when she has slept enough.
        gf_now = counts[self.mot_idx["escape"]].sum() if counts is not None else 0
        if f["state"] == "sleep":
            self.sleep_p = max(0.0, self.sleep_p - dt / 120.0)
            touched = any(self.sens_rates.get(f"touch_{s}") for s in "LR") or self.pain > 0
            night = self.is_night()
            looming = self.sens_rates.get("loom_L", 0) + self.sens_rates.get("loom_R", 0) > 5
            why = ("fibra gigante" if gf_now >= 3 + min(6.0, self.habituation) and looming else
                   "alarma (DNp02/04/11)" if m["alarm"] > 15 else
                   "la tocan" if touched else
                   "amanece" if self.daynight and not night and self.sleep_p < 0.35 else
                   "descansada" if not night and self.sleep_p <= 0 else None)
            if why is None:
                f["speed"] = 0.0
                return
            f["state"], f["state_t"] = "walk", 0.0
            if why not in ("amanece", "descansada"):
                self.awake_block = 8.0
            elif self.leaf_at(f["x"], f["y"]):              # slept well under a leaf: relief (reward)
                self.reward_pulse = 4.0
            self.event(f"Flippy se despierta: {why}")
        else:
            self.sleep_p = min(1.0, self.sleep_p + dt / 180.0)
            self.awake_block = max(0.0, self.awake_block - dt)
        # graded escape, like real flies: a strong giant-fiber (DNp01) volley -> take off;
        # a weaker looming response (alarm DNs DNp02/04/11) -> walk away. Each take-off
        # habituates the giant fiber (threshold up), recovering in ~20 s of calm.
        self.habituation *= math.exp(-dt / 20.0)
        gf = counts[self.mot_idx["escape"]].sum() if counts is not None else 0
        threshold = 3 + min(6.0, self.habituation)
        # the giant fiber is driven by looming (LC4/LPLC2); a stray volley during an odour
        # burst, with nothing approaching, is not a take-off command
        looming = self.sens_rates.get("loom_L", 0) + self.sens_rates.get("loom_R", 0) > 5
        if gf >= threshold and looming and f["state"] not in ("fly", "mate"):
            f["state"], f["state_t"] = "fly", 0.0
            f["heading"] += np.random.uniform(-2.5, 2.5)
            self.habituation += 1.5
            self.event("¡Fibra gigante (DNp01)! Despega para escapar"
                       + (" (ya un poco habituada)" if self.habituation > 2 else ""))
        elif m["alarm"] > 15 and f["state"] in ("walk", "groom", "feed") and f["state_t"] > 0.3:
            ll, lr = self.sens_rates.get("loom_L", 0), self.sens_rates.get("loom_R", 0)
            if ll + lr > 0:
                f["state"], f["state_t"] = "startle", 0.0
                self.startle_side = 1.0 if ll > lr else -1.0     # threat on the left -> turn right
                self.event("Alarma (DNp02/04/11): se aleja caminando")
        if f["state"] == "startle":                           # brisk walk away from the threat
            f["heading"] += self.startle_side * 4.0 * dt
            f["speed"] += 0.4 * (55 - f["speed"])
            f["x"] += math.cos(f["heading"]) * f["speed"] * dt
            f["y"] += math.sin(f["heading"]) * f["speed"] * dt
            f["energy"] -= 1.0 * dt
            if f["state_t"] > 0.6:
                f["state"], f["state_t"] = "walk", 0.0
        if f["state"] == "fly":
            f["z"] = math.sin(min(1.0, f["state_t"] / 0.8) * math.pi) * 10
            f["x"] += math.cos(f["heading"]) * 90 * dt
            f["y"] += math.sin(f["heading"]) * 90 * dt
            f["energy"] -= 3 * dt
            if f["state_t"] > 0.8:
                f["state"], f["z"] = "walk", 0.0
        elif f["state"] != "startle":
            # hysteresis + minimum dwell so noisy rates don't make the state flicker
            feeding = m["feed"] > (2 if f["state"] == "feed" else 8)
            grooming = m["groom"] > (4 if f["state"] == "groom" else 12) and not feeding
            # grooming bouts are short in real flies: cap at 2 s, then 1 s refractory
            if f["state"] == "groom" and f["state_t"] > 2.0:
                grooming, self.groom_block = False, 1.0
            self.groom_block = max(0.0, self.groom_block - dt)
            grooming = grooming and self.groom_block == 0
            # assisted mode: grooming needs real touch in the last 1.5 s (in this model
            # DNg84 also fires for odours, which would make the fly groom near food)
            if any(self.sens_rates.get(f"touch_{s}") for s in "LR"):
                self.touch_recent = 1.5
            self.touch_recent = max(0.0, self.touch_recent - dt)
            if self.assist and f["state"] != "groom":
                grooming = grooming and self.touch_recent > 0
            new = "feed" if feeding else "groom" if grooming else "walk"
            if new != "feed" and f["state"] in ("walk", "groom") and f["state_t"] < 0.6:
                new = f["state"]
            if new != f["state"]:
                if new == "feed":
                    self.event("MN9 activa: saca la probóscide y come")
                elif new == "groom":
                    self.event("DNg84 activa: se acicala")
                f["state"], f["state_t"] = new, 0.0
            f["proboscis"] += 0.3 * ((1.0 if feeding else 0.0) - f["proboscis"])

            # receptive females slow down and let the male approach; rejecting ones extrude
            # the ovipositor (DNp13), both read straight from the brain
            receptive = m["receptive"] > 6
            f["ovi"] += 0.25 * ((1.0 if m["reject"] > 5 else 0.0) - f["ovi"])
            # --- steering: descending commands + intrinsic wander
            self.wander += np.random.normal(0, 0.6) * dt - self.wander * dt
            turn = 0.08 * (m["turn_R"] - m["turn_L"]) + 0.02 * self.wander * 60
            fwd = self.base_walk + 0.02 * m["forward"] - 0.05 * m["backward"]
            # --- odour navigation. Motivation comes from the brain: attractive odour
            # drives DNg100/DNge053 (smoothed over 1 s); aversive (CO2/heat) drives DNb05.
            self.odor_slow += (m["odor_drive"] - self.odor_slow) * dt / 1.0
            self.avoid_slow += (m["avoid"] - self.avoid_slow) * dt / 0.5
            self.chemo = "—"
            if self.assist:
                (gL, gR), (bL, bR) = self.conc["good"], self.conc["bad"]
                c = (gL + gR) / 2
                rising = c - self.conc_slow
                self.conc_slow += rising * dt / 0.8
                if self.avoid_slow > 3 and bL + bR > 0.05:
                    turn -= 8.0 * (bR - bL) / (bL + bR + 1e-6)   # away from the stronger side
                    fwd += 0.3
                    self.chemo = "se aleja (cerebro: DNb05 · dirección: antenas)"
                elif self.odor_slow > 2 and c > 0.02:
                    side = (gR - gL) / (gL + gR + 1e-6)          # ~0.02: antennae are 4 px apart
                    turn = 0.3 * turn + 60.0 * side               # toward the stronger side
                    if rising < -0.002:                           # overshot: turn back (casting)
                        turn += math.copysign(4.0, side)
                    fwd += 0.3 if rising > 0 else 0.0
                    self.chemo = "se acerca al olor (cerebro: DNg100 · dirección: antenas)"
            # --- light: the brain notices it (ocelli -> DNp18); the body turns toward the
            # brighter eye (walking phototaxis is not wired to DNa01/02 in this model)
            self.light_slow += (m["light_seen"] - self.light_slow) * dt / 0.5
            if self.assist and self.chemo == "—" and self.light_slow > 6:
                bl, br = self.vision.get("light_L", 0), self.vision.get("light_R", 0)
                if bl + br > 0.02:
                    turn = 0.3 * turn + 4.0 * (br - bl) / (bl + br)
                    fwd += 0.2
                    self.chemo = "va hacia la luz (cerebro: ocelos → DNp18 · dirección: ojos)"
            # --- memory of individuals: the opinion lives in the KC->MBON synapses (brain);
            # turning toward / away from that fly is done by the body
            if self.assist:
                near = [(self.opinions.get(d.id), d, math.hypot(d.x - f["x"], d.y - f["y"])) for d in self.dolls]
                near = [(o, d, dist) for o, d, dist in near if o is not None and dist < 70]
                bad = min(near, key=lambda z: z[0], default=None)
                good = max(near, key=lambda z: z[0], default=None)
                if bad and bad[0] < -0.3 and bad[2] < 55:
                    away = math.atan2(f["y"] - bad[1].y, f["x"] - bad[1].x)
                    turn = 5.0 * math.sin(away - f["heading"])
                    fwd += 0.4
                    self.chemo = f"huye de {bad[1].traits['name']} (memoria: la recuerda mal)"
                elif good and good[0] > 0.3 and good[2] > 15 and self.chemo == "—":
                    to = math.atan2(good[1].y - f["y"], good[1].x - f["x"])
                    turn = 3.0 * math.sin(to - f["heading"])
                    self.chemo = f"se acerca a {good[1].traits['name']} (memoria: la recuerda bien)"
            sleepy = ((self.is_night() and self.sleep_p > 0.1) or self.sleep_p > 0.9) and self.awake_block == 0
            if sleepy and f["state"] == "walk" and f["energy"] > 20:
                # her favourite leaf wins over the nearest one (place memory)
                leaf = min((t for t in self.things if t.kind == "leaf"),
                           key=lambda t: math.hypot(t.x - f["x"], t.y - f["y"])
                           - 300 * (self.place_opinions.get(t.id) or 0), default=None)
                self.seek_t += dt
                here = self.leaf_at(f["x"], f["y"])
                if leaf and (not here or (here is not leaf and (self.place_opinions.get(leaf.id) or 0) >
                                          (self.place_opinions.get(here.id) or 0) + 0.3)) \
                        and self.seek_t < 30 and math.hypot(leaf.x - f["x"], leaf.y - f["y"]) < 300:
                    to = math.atan2(leaf.y - f["y"], leaf.x - f["x"])
                    turn = 3.0 * math.sin(to - f["heading"])
                    fav = (self.place_opinions.get(leaf.id) or 0) > 0.3
                    self.chemo = f"tiene sueño: va a {'su hoja favorita' if fav else 'una hoja'} ({leaf.label})"
                else:
                    f["state"], f["state_t"], self.seek_t = "sleep", 0.0, 0.0
                    self.event("Flippy se duerme" + (" bajo una hoja" if self.leaf_at(f["x"], f["y"]) else ""))
                    return
            else:
                self.seek_t = 0.0
            # --- memory of places (same KC->MBON mechanism, each place smells of itself)
            if self.assist:
                places = [(self.place_opinions.get(t.id), t, math.hypot(t.x - f["x"], t.y - f["y"]))
                          for t in self.things if t.scent]
                bad = min(((o, t, d) for o, t, d in places if o is not None and o < -0.3 and d < t.r + 35),
                          key=lambda z: z[0], default=None)
                if bad:
                    away = math.atan2(f["y"] - bad[1].y, f["x"] - bad[1].x)
                    turn = 5.0 * math.sin(away - f["heading"])
                    fwd += 0.3
                    self.chemo = f"evita {bad[1].label} (memoria: allí lo pasó mal)"
                elif f["energy"] < 50 and self.chemo == "—":
                    good = max(((o, t, d) for o, t, d in places if o is not None and o > 0.3 and t.kind == "food"
                                and t.amount > 10), key=lambda z: z[0] - z[2] / 600, default=None)
                    if good:
                        to = math.atan2(good[1].y - f["y"], good[1].x - f["x"])
                        turn = 3.0 * math.sin(to - f["heading"])
                        fwd += 0.2
                        self.chemo = f"tiene hambre: va a su comedero favorito ({good[1].label}, memoria)"
            if receptive:
                fwd *= 0.3
                self.chemo = "se detiene: está receptiva (vpoDN)"
            if f["state"] != "walk":
                fwd, turn = 0.0, 0.0
            if m["backward"] > 5:
                fwd = -0.6
            # body reflex: antenna against a wall -> turn toward the arena centre
            wall_touch = any(self.sens_rates.get(f"touch_{s}") and self._antenna_wall(s) for s in "LR")
            if wall_touch:
                to_c = math.atan2(H / 2 - f["y"], W / 2 - f["x"]) - f["heading"]
                turn += 3.0 * math.copysign(1.0, math.sin(to_c))
            f["heading"] += turn * dt * 3
            f["speed"] += 0.3 * (fwd * 40 - f["speed"])  # px/s
            f["x"] += math.cos(f["heading"]) * f["speed"] * dt
            f["y"] += math.sin(f["heading"]) * f["speed"] * dt
            f["energy"] -= (0.3 + abs(f["speed"]) * 0.01) * dt
            if feeding:
                for t in self.things:
                    if t.kind == "food" and math.hypot(t.x - f["x"], t.y - f["y"]) < t.r + 6:
                        t.amount -= 8 * dt
                        f["energy"] = min(100.0, f["energy"] + 8 * dt)
        # arena walls: bounce only when heading into the wall (avoids heading jitter)
        ch, sh = math.cos(f["heading"]), math.sin(f["heading"])
        if (f["x"] <= 3 and ch < 0) or (f["x"] >= W - 3 and ch > 0):
            f["heading"] = math.pi - f["heading"]
        if (f["y"] <= 3 and sh < 0) or (f["y"] >= H - 3 and sh > 0):
            f["heading"] = -f["heading"]
        f["x"], f["y"] = min(max(f["x"], 3), W - 3), min(max(f["y"], 3), H - 3)
        f["heading"] = _ang(f["heading"])
        f["energy"] = min(100.0, max(0.0, f["energy"]))
        # hunger modulates taste gain (a simple neuromodulation stand-in)
        self.gain["sugar"] = 0.6 + 0.8 * (1 - f["energy"] / 100)

    # ---------------------------------------------------------------- experiment mode
    def _rebuild_silence(self):
        old = self.silenced_idx
        self.silenced_idx = np.unique(np.concatenate([e["idx"] for e in self.silenced])) if self.silenced \
            else np.empty(0, np.int64)
        freed = np.setdiff1d(old, self.silenced_idx)
        self.brain.refr[freed] = 0                    # restored neurons can fire again

    def _region_label(self, idx):
        sc = self.ann.set_index("idx").reindex(idx).super_class.fillna("?")
        names = {"optic": "lóbulo óptico", "central": "cerebro central", "sensory": "neuronas sensoriales",
                 "visual_projection": "proyección visual", "descending": "descendentes", "ascending": "ascendentes",
                 "motor": "motoneuronas", "visual_centrifugal": "centrífugas visuales", "endocrine": "endocrinas",
                 "sensory_ascending": "sensoriales ascendentes"}
        top = sc.value_counts()
        main = names.get(top.index[0], top.index[0]) if len(top) else "?"
        return f"mayoría {main} ({100 * top.iloc[0] / max(1, len(idx)):.0f} %)" if len(top) else ""

    def silence(self, kind, key=None, x=None, y=None, r=None):
        """kind 'function' (catalogue key) or 'region' (frontal-map pixel + radius)."""
        if kind == "function":
            if key not in self.fn_idx or any(e.get("key") == key for e in self.silenced):
                return
            idx, label = self.fn_idx[key], experiments.FUNCTIONS[key][0]
        else:
            idx = experiments.region_indices(self.map_xy, float(x), float(y), float(r))
            side = "izq." if x < 110 else "der." if x > 130 else "central"
            label = f"Zona del cerebro {side} ({round(x)}, {round(y)}) · {self._region_label(idx)}"
        if not len(idx):
            return
        self.silence_id += 1
        self.silenced.append(dict(id=self.silence_id, kind=kind, key=key, x=x, y=y, r=r, label=label, idx=idx))
        self._rebuild_silence()
        self.event(f"🧪 Silenciadas {len(idx):,} neuronas: {label}".replace(",", "."))
        self.record("experiment", f"Silencia {label} ({len(idx)} neuronas)", 0, ids=(0,))

    def unsilence(self, sid=None):
        gone = [e for e in self.silenced if sid is None or e["id"] == sid]
        self.silenced = [e for e in self.silenced if e not in gone]
        self._rebuild_silence()
        for e in gone:
            self.event(f"🧪 Restauradas: {e['label']}")

    def memory_online(self):
        """With most Kenyon cells silenced, odours can't be recognised: no memory readout."""
        kc = self.fn_idx["kenyon"]
        return not len(self.silenced_idx) or np.isin(kc, self.silenced_idx).mean() < 0.5

    # ---------------------------------------------------------------- arena size
    def set_arena(self, key):
        """Resize the arena (S/M/L). Whatever ends up outside is brought to the new edge."""
        global W, H
        from . import dolls as dolls_mod
        W, H = ARENAS[key]
        dolls_mod.W, dolls_mod.H = W, H
        clamp = lambda x, y, m=4: (min(max(x, m), W - m), min(max(y, m), H - m))  # noqa: E731
        for t in self.things:
            t.x, t.y = clamp(t.x, t.y, t.r if t.kind == "stone" else 4)
        for d in self.dolls:
            d.x, d.y = clamp(d.x, d.y)
        for b in self.brood:
            b["x"], b["y"] = clamp(b["x"], b["y"])
        self.fly["x"], self.fly["y"] = clamp(self.fly["x"], self.fly["y"], 3)

    def arena_key(self):
        return next((k for k, v in ARENAS.items() if v == (W, H)), "M")

    # ---------------------------------------------------------------- day / night
    def is_night(self):
        return self.daynight and self.clock >= DAY_S

    def daylight(self):
        """1 at noon .. ~0.2 at night, with dawn and dusk ramps."""
        if not self.daynight:
            return 1.0
        c = self.clock
        if c < TWILIGHT_S:
            return 0.2 + 0.8 * c / TWILIGHT_S
        if c < DAY_S - TWILIGHT_S:
            return 1.0
        if c < DAY_S:
            return 0.2 + 0.8 * (DAY_S - c) / TWILIGHT_S
        return 0.2

    def leaf_at(self, x, y):
        return next((t for t in self.things if t.kind == "leaf" and math.hypot(t.x - x, t.y - y) < t.r - 2), None)

    def ray(self, x, y, a, max_d):
        """Distance along a ray to the first wall or stone (max_d if nothing)."""
        ca, sa = math.cos(a), math.sin(a)
        best = max_d
        for wall, v in ((0.0, ca), (W, ca)):            # vertical walls x=0, x=W
            if v and (t := (wall - x) / v) > 0:
                best = min(best, t)
        for wall, v in ((0.0, sa), (H, sa)):            # horizontal walls
            if v and (t := (wall - y) / v) > 0:
                best = min(best, t)
        for s in self.things:
            if s.kind == "stone":
                dx, dy = s.x - x, s.y - y
                proj = dx * ca + dy * sa
                if proj > 0:
                    perp2 = dx * dx + dy * dy - proj * proj
                    if perp2 < s.r * s.r:
                        best = min(best, proj - math.sqrt(s.r * s.r - perp2))
        return max(0.0, best)

    def _heat_at(self, x, y):
        return sum(max(0.0, 1 - math.hypot(t.x - x, t.y - y) / t.r) for t in self.things if t.kind == "heat")

    def _antenna_wall(self, s):
        f = self.fly
        a = f["heading"] + (-0.7 if s == "L" else 0.7)
        px, py = f["x"] + 4 * math.cos(a), f["y"] + 4 * math.sin(a)
        return px < 2 or py < 2 or px > W - 2 or py > H - 2

    def social(self):
        """Mating decisions come from the real fly's brain: vpoDN (accept) vs DNp13
        (reject by extruding the ovipositor). Male dolls only ask."""
        f, dt, m = self.fly, TICK_MS / 1000, self.mot_rates
        self.mate_block = max(0.0, self.mate_block - dt)
        # male-male competition: two courting males close to each other fight it out
        court = [d for d in self.dolls if d.state == "court" and d.male]
        for a in court:
            for b in court:
                if a.id < b.id and math.hypot(a.x - b.x, a.y - b.y) < 22:
                    for d, o in ((a, b), (b, a)):
                        d._set("fight")
                        d.fight_with = o
                    her = "Flippy" if a.target == 0 else (self.doll(a.target).traits["name"] if self.doll(a.target) else "una hembra")
                    self.event(f"{a.traits['name']} y {b.traits['name']} pelean por {her}")
        for d in self.dolls:
            if d.state == "fight" and d.state_t > 1.5 and d.fight_with is not None:
                o = d.fight_with
                score = lambda z: z.traits["aggression"] * 0.6 + z.traits["size"] * 0.6 + np.random.random() * 0.4
                win, lose = (d, o) if score(d) >= score(o) else (o, d)
                win._set("court"); win._stage(0); win.fight_with = None
                lose._set("retreat"); lose.cooldown = 10.0; lose.fight_with = None
                lose.like(win.id, -40)
                win.like(lose.id, -10)
                lose.heading = math.atan2(lose.y - win.y, lose.x - win.x)
                self.event(f"{win.traits['name']} gana la pelea; {lose.traits['name']} se retira")
        for d in self.dolls:                              # male doll courting a female doll
            if not d.attempting or d.target == 0:
                continue
            her = self.doll(d.target)
            if her is None or her.state in ("copulate", "flyaway"):
                continue
            if not her.mated and np.random.random() < 0.25 + 0.6 * d.traits["song"]:
                for z in (d, her):
                    z._set("copulate"); z.speed = 0.0
                d.x, d.y = her.x - math.cos(her.heading) * 5, her.y - math.sin(her.heading) * 5
                d.heading = her.heading
                her.mated, her.sperm, her.eggs_left = True, dict(id=d.id, traits=dict(d.traits), scent=list(d.scent)), 3
                d.after_mating(GENERATION_S)
                her.after_mating(GENERATION_S)
                d.like(her.id, 40)
                her.like(d.id, 40)
                self.event(f"{her.traits['name']} acepta a {d.traits['name']} → cópula")
                self.record("mating", f"{her.traits['name']} y {d.traits['name']} copulan "
                                      f"({d.traits['name']}: {d.matings}/{MAX_MATINGS}, {her.traits['name']}: {her.matings}/{MAX_MATINGS})",
                            her.gen, ids=(her.id, d.id))
                for z in (d, her):
                    if z.id in self.family:
                        self.family[z.id]["matings"] = z.matings
            else:
                self.event(f"{her.traits['name']} rechaza a {d.traits['name']}")
                d.like(her.id, -15)
                d.rejected(self, "rechazado")
        if f["state"] not in ("walk", "groom") or self.mate_block:
            return
        for d in self.dolls:
            if not d.attempting or d.target != 0:
                continue
            name = d.traits["name"]
            if m["receptive"] > 8 and m["receptive"] > m["reject"]:
                f["state"], f["state_t"], f["speed"] = "mate", 0.0, 0.0
                d._set("copulate")
                d.x, d.y = f["x"] - math.cos(f["heading"]) * 5, f["y"] - math.sin(f["heading"]) * 5
                d.heading = f["heading"]
                self.fly_mated = True
                self.fly_sperm = dict(id=d.id, traits=dict(d.traits), scent=list(d.scent))
                self.fly_matings += 1
                self.eggs_pending = CLUTCH
                d.after_mating(GENERATION_S)
                self.record("mating", f"Flippy acepta a {name} por su vpoDN "
                                      f"({name}: {d.matings}/{MAX_MATINGS} cópulas)", 0, ids=(0, d.id))
                self.family[0]["matings"] = self.fly_matings
                if d.id in self.family:
                    self.family[d.id]["matings"] = d.matings
                self.mate_block = 1.0
                self.event(f"vpoDN activa: acepta a {name} → cópula")
                return
            self.mate_block = 2.0
            if m["reject"] > 5:
                self.event(f"Flippy rechaza a {name} sacando el ovipositor (DNp13)")
            elif not self.fly_in_heat and not self.fly_mated:
                self.event(f"Flippy ignora a {name}: no está en época de apareamiento")
            else:
                self.event(f"Flippy ignora a {name}: su vpoDN no se activó lo suficiente")
            d.rejected(self, "rechazado")
            return

    def reproduce(self, dt):
        """Flippy: eggs mature after mating (internal state); laying happens when her
        integrated oviDN activity crosses a threshold (sugar and food odour lower it)."""
        f = self.fly
        if self.fly_rest > 0:
            self.fly_rest -= dt
            if self.fly_rest <= 0:
                self.fly_rest = 0.0
                self.fly_in_heat, self.fly_mated = True, False
                self.event("Flippy vuelve a estar en época de apareamiento")
                self.record("season", "Flippy vuelve a estar en época de apareamiento", 0, ids=(0,))
        if self.eggs_pending > 0:                      # one egg matures every 12 s
            self.egg_progress += dt / 12.0
            if self.egg_progress >= 1.0:
                self.egg_progress -= 1.0
                self.eggs_pending -= 1
                self.eggs_ready += 1
        if self.eggs_ready >= 1 and f["state"] in ("walk", "feed", "groom"):
            self.ovi_acc += self.mot_rates["lay"] * dt
            if self.ovi_acc > 10.0:
                x, y = f["x"] - math.cos(f["heading"]) * 6, f["y"] - math.sin(f["heading"]) * 6
                if self.lay_egg(x, y, mother=0, father=self.fly_sperm):
                    self.eggs_ready -= 1
                    self.eggs_laid += 1
                    near_food = self._field("food", x, y) > 0.3
                    where = "cerca de la comida" if near_food else "lejos de la comida"
                    self.event(f"oviDN activa: Flippy pone un huevo {where}")
                    self.record("egg", f"Flippy pone un huevo {where} (oviDN)", 0, ids=(0, (self.fly_sperm or {}).get("id")))
                    if self.fly_in_heat:            # after giving birth: rest one generation
                        self.fly_in_heat = False
                        self.fly_rest = GENERATION_S
                        self.event("Flippy deja la época de apareamiento: descansa una generación")
                self.ovi_acc = 0.0

    def update_world(self):
        dt = TICK_MS / 1000
        for sh in self.shadows:
            sh["t"] += dt
            sh["r"] += 30 * dt  # approaching -> expanding
        self.shadows = [s for s in self.shadows if s["t"] < 1.2]
        self.things = [t for t in self.things if t.amount > 0 or t.kind != "food"]

    # ---------------------------------------------------------------- save / load
    def to_save(self):
        """Everything needed to continue later (the brain restarts at rest: its state
        settles in milliseconds, so it is not worth saving 138k membrane potentials)."""
        return dict(
            version=SAVE_VERSION, time_ms=self.time_ms, fly=dict(self.fly),
            fly_in_heat=self.fly_in_heat, fly_mated=self.fly_mated, fly_sperm=self.fly_sperm,
            eggs_ready=self.eggs_ready, eggs_pending=self.eggs_pending, egg_progress=self.egg_progress,
            eggs_laid=self.eggs_laid, fly_rest=self.fly_rest, fly_matings=self.fly_matings,
            things=[t.to_json() for t in self.things], dolls=[d.save() for d in self.dolls],
            brood=self.brood, brood_id=self.brood_id, family=list(self.family.values()),
            chronicle=self.chronicle, log=self.log, assist=self.assist, base_walk=self.base_walk,
            flippy_scent=self.flippy_scent, memory=self.mem.to_save(),
            clock=self.clock, daynight=self.daynight, sleep_p=self.sleep_p, arena=self.arena_key(),
            learn_log=self.learn_log,
            experiments=[{k: e[k] for k in ("kind", "key", "x", "y", "r")} for e in self.silenced])

    def load_save(self, s):
        if s.get("version") != SAVE_VERSION:
            raise ValueError("partida de otra versión")
        self.set_arena(s.get("arena", "S"))        # saves from before arenas were 320x200
        self.reset()
        for k in ("time_ms", "fly_in_heat", "fly_mated", "fly_sperm", "eggs_ready", "eggs_pending",
                  "egg_progress", "eggs_laid", "fly_rest", "fly_matings", "brood", "brood_id",
                  "chronicle", "log", "assist", "base_walk"):
            setattr(self, k, s[k])
        self.fly.update(s["fly"])
        self.things = []
        for t in s["things"]:
            o = Thing(t["kind"], t["x"], t["y"], r=t["r"], amount=t["amount"],
                      label=t.get("label", ""), scent=t.get("scent", []))
            if o.kind in PLACE_KINDS and not o.scent:  # places from older saves get an odour now
                o.scent = [str(g) for g in np.random.choice(self.scent_pool, PLACE_SCENT_SIZE, replace=False)]
                o.label = o.label or f"{PLACE_KINDS[o.kind]} {1 + sum(q.kind == o.kind for q in self.things)}"
            o.id = t["id"]
            Thing._id = max(Thing._id, o.id)
            self.things.append(o)
        self.dolls = [Doll.restore(d) for d in s["dolls"]]
        self.family = {m["id"]: m for m in s["family"]}
        Doll._id = max([Doll._id] + list(self.family))
        self.flippy_scent = s["flippy_scent"]
        self.clock, self.daynight, self.sleep_p = s.get("clock", 30.0), s.get("daynight", True), s.get("sleep_p", 0.2)
        self.learn_log = s.get("learn_log", [])
        self.silenced = []
        for e in s.get("experiments", []):
            self.silence(e["kind"], e.get("key"), e.get("x"), e.get("y"), e.get("r"))
        self._rebuild_silence()
        self.mem.load_save(s["memory"])
        self.opinions = {d.id: self.mem.opinion(d.id) for d in self.dolls}
        self.place_opinions = {t.id: self.mem.opinion(f"p{t.id}") for t in self.things if t.scent}
        self.event("Partida cargada")

    def summary(self):
        return dict(t=round(self.time_ms / 1000), dolls=len(self.dolls), brood=len(self.brood),
                    gen_max=max(m["gen"] for m in self.family.values()),
                    family=len(self.family))

    # ---------------------------------------------------------------- state
    def snapshot(self):
        spk = np.concatenate(self.spike_buf) if self.spike_buf else np.empty(0, np.int64)
        self.spike_buf = []
        f = self.fly
        return dict(
            t=round(self.time_ms / 1000, 2),
            fly={k: (round(float(v), 2) if isinstance(v, (float, np.floating)) else v) for k, v in f.items()},
            in_heat=self.fly_in_heat, mated=self.fly_mated, chemo=self.chemo,
            eggs_ready=int(self.eggs_ready), eggs_laid=self.eggs_laid, eggs_pending=self.eggs_pending,
            fly_rest=round(self.fly_rest, 1), fly_matings=self.fly_matings, n_chronicle=len(self.chronicle),
            brood=[{k: (round(v, 1) if isinstance(v, float) else v) for k, v in b.items() if k not in ("father", "mom")}
                   for b in self.brood],
            family=list(self.family.values()),
            things=[t.to_json() for t in self.things],
            dolls=[d.to_json() for d in self.dolls],
            opinions={str(k): (None if v is None else round(v, 2)) for k, v in self.opinions.items()},
            place_opinions={str(k): (None if v is None else round(v, 2)) for k, v in self.place_opinions.items()},
            learn_log=self.learn_log[-12:],
            silenced=[dict(id=e["id"], kind=e["kind"], key=e["key"], x=e["x"], y=e["y"], r=e["r"],
                           label=e["label"], n=int(len(e["idx"]))) for e in self.silenced],
            n_silenced=int(len(self.silenced_idx)), memory_online=bool(self.memory_online()),
            scent_owner=self.scent_owner,
            vision={k: round(v, 2) for k, v in self.vision.items()},
            arena=dict(w=W, h=H, key=self.arena_key(), max_dolls=max_dolls()),
            clock=round(self.clock, 1), day_len=DAY_S, night_len=NIGHT_S, night=self.is_night(),
            daylight=round(self.daylight(), 2), daynight=self.daynight, sleep_p=round(self.sleep_p, 2),
            under_leaf=bool(self.leaf_at(f["x"], f["y"])), habituation=round(self.habituation, 1),
            shadows=[{k: round(v, 1) for k, v in s.items()} for s in self.shadows],
            sens={k: round(v, 1) for k, v in self.sens_rates.items()},
            motor={k: round(float(v), 1) for k, v in self.mot_rates.items()},
            live=int(len(self.brain.live)),
            spikes=np.unique(spk)[:4000].tolist(),
            n_spikes=int(len(spk)),
            log=self.log,
        )


def _ang(a):
    return math.atan2(math.sin(a), math.cos(a))
