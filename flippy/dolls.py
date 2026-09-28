"""'Doll' flies: customisable companions with scripted (non-neural) behaviour.

Only the real fly has the FlyWire brain. Dolls act through rules driven by their
traits, and reach the real brain exclusively through its senses: they are seen
(LC10a/LC11, LC4 when they lunge), smelled (cVA, fly odours), heard (courtship
song) and felt (touch). Everything the real fly does in response is the connectome.
"""
import math
import random

W, H = 480, 300   # kept in sync with world.W/H by World.set_arena

DEFAULT_TRAITS = dict(
    name="Muñeca", sex="F",   # "F" hembra (rosa) / "M" macho (azul)
    mating=False,             # época de apareamiento
    aggression=0.0,           # 0..1 violenta: embiste a la mosca real
    helper=0.0,               # 0..1 ayudante: trae comida, la defiende
    social=0.5,               # 0..1 sociable (se acerca) / tímida (se aleja)
    speed=0.5,                # 0..1
    size=1.0,                 # 0.7..1.4
    song=0.6,                 # 0..1 calidad del canto (machos): canta más fuerte y avanza antes
    persistence=0.5,          # 0..1 insistencia: probabilidad de volver a intentarlo tras un rechazo
)

COURT_STAGES = ["orientarse", "tocarla", "cantar", "intentar cópula"]


def _ang(a):
    return math.atan2(math.sin(a), math.cos(a))


class Doll:
    _id = 0

    def __init__(self, x, y, **traits):
        Doll._id += 1
        self.id = Doll._id
        self.x, self.y = float(x), float(y)
        self.heading = random.uniform(-math.pi, math.pi)
        self.speed = 0.0
        self.state, self.state_t = "wander", 0.0
        self.cooldown = 0.0
        self.carrying = False
        self.singing = False
        self.touching = False
        self.theta_prev = None  # angular size seen by the real fly (for looming)
        self.dist_prev = None
        self.court_stage, self.court_t = 0, 0.0
        self.attempting = False
        self.fight_with = None
        self.target = 0            # who a male courts: 0 = Flippy, otherwise a female doll id
        # family / reproduction
        self.gen, self.mother, self.father = 0, None, None
        self.mated, self.sperm, self.eggs_left, self.lay_t = False, None, 0, 0.0
        self.age, self.adult_at = 0.0, 0.0   # offspring are born young and mature later
        self.lifespan = 0.0                   # 0 = immortal (dolls you create); offspring age and die
        self.matings = 0                      # dies after MAX_MATINGS copulations
        self.rest = 0.0                       # seconds out of mating season after copulating
        self.scent = []                       # individual odour: 10 ORN glomeruli (set by the world)
        self.affinity = {}                    # other doll id -> -100 (enemy) .. +100 (friend)
        self.traits = dict(DEFAULT_TRAITS)
        self.set_traits(traits)

    def set_traits(self, t):
        for k, v in t.items():
            if k in DEFAULT_TRAITS:
                self.traits[k] = type(DEFAULT_TRAITS[k])(v) if k != "mating" else bool(v)
        self.traits["size"] = min(1.4, max(0.7, self.traits["size"]))

    @property
    def male(self):
        return self.traits["sex"] == "M"

    @property
    def young(self):
        return self.age < self.adult_at

    def after_mating(self, rest_s):
        """Called when a copulation starts: out of season for one generation."""
        self.matings += 1
        self.traits["mating"] = False
        self.rest = rest_s

    def force_season(self, on):
        """Manual override from the editor: back in season now."""
        if on and self.rest > 0:
            self.rest = 0.0
            self.mated = False

    STATE_KEYS = ("id", "x", "y", "heading", "gen", "mother", "father", "mated", "sperm", "eggs_left",
                  "lay_t", "age", "adult_at", "lifespan", "matings", "rest", "scent")

    def save(self):
        return dict({k: getattr(self, k) for k in self.STATE_KEYS}, traits=dict(self.traits),
                    affinity={str(k): v for k, v in self.affinity.items()},
                    sleeping=self.state == "sleep")

    @classmethod
    def restore(cls, d):
        o = cls(d["x"], d["y"], **d["traits"])
        for k in cls.STATE_KEYS:
            if k in d:
                setattr(o, k, d[k])
        o.affinity = {int(k): v for k, v in d.get("affinity", {}).items()}
        if d.get("sleeping"):                 # other states restart cleanly as "wander"
            o.state = "sleep"
        cls._id = max(cls._id, o.id)
        return o

    # ------------------------------------------------------------------ relationships
    def like(self, other_id, delta):
        v = self.affinity.get(other_id, 0.0) + delta
        self.affinity[other_id] = max(-100.0, min(100.0, v))

    def friends_and_enemies(self, world):
        alive = {d.id: d for d in world.dolls if d is not self}
        fr = [(v, alive[i]) for i, v in self.affinity.items() if i in alive and v >= 40]
        en = [(v, alive[i]) for i, v in self.affinity.items() if i in alive and v <= -40]
        return max(fr, key=lambda x: x[0], default=(0, None))[1], min(en, key=lambda x: x[0], default=(0, None))[1]

    def pose(self):
        """Same shape as world.fly, so a male can court a doll exactly like Flippy."""
        st = {"copulate": "mate", "flyaway": "fly"}.get(self.state, "walk")
        return dict(x=self.x, y=self.y, heading=self.heading, speed=self.speed, state=st)

    def _pick_target(self, world):
        """Nearest female in mating season: Flippy (only while she is) or a female doll."""
        cands = ([(0, world.fly)] if world.fly_in_heat else []) + [(d.id, d.pose()) for d in world.dolls
                                    if not d.male and d is not self and d.traits["mating"]
                                    and not d.mated and not d.young]
        if not cands:
            return None, None
        keep = next((p for i, p in cands if i == self.target), None)
        if self.state == "court" and keep is not None:
            return self.target, keep
        # nearest, but a female he likes counts as closer (and one he dislikes as farther)
        i, p = min(cands, key=lambda c: math.hypot(c[1]["x"] - self.x, c[1]["y"] - self.y)
                   - 0.5 * self.affinity.get(c[0], 0.0))
        return i, p

    # ------------------------------------------------------------------ helpers
    def _steer(self, tx, ty, speed, dt, turn_rate=5.0):
        want = math.atan2(ty - self.y, tx - self.x)
        d = _ang(want - self.heading)
        self.heading += max(-turn_rate * dt, min(turn_rate * dt, d))
        self.speed += 0.3 * (speed - self.speed)

    def _stage(self, n):
        self.court_stage, self.court_t = n, 0.0

    def rejected(self, world, how):
        """Flippy said no. Insistent males sing again; the rest give up for a while."""
        name = self.traits["name"]
        self.attempting = False
        if random.random() < self.traits["persistence"]:
            self._stage(2)
            self.cooldown = 1.0
            world.event(f"{name} insiste y vuelve a cantar")
        else:
            self._stage(0)
            self._set("wander")
            self.cooldown = 12.0
            world.event(f"{name} se rinde ({how})")

    def _set(self, state):
        if state != self.state:
            self.state, self.state_t = state, 0.0

    # ------------------------------------------------------------------ update
    def update(self, world, dt):
        t, fly = self.traits, world.fly
        self.state_t += dt
        self.cooldown = max(0.0, self.cooldown - dt)
        self.singing = self.touching = self.attempting = False
        self.age += dt
        base = 18 + 30 * t["speed"]
        if self.rest > 0:                     # resting after copulation: back in season later
            self.rest -= dt
            if self.rest <= 0:
                self.rest = 0.0
                t["mating"] = True
                self.mated = False
                world.event(f"{t['name']} vuelve a estar en época de apareamiento")
        if self.adult_at and abs(self.age - self.adult_at) < dt / 2:
            world.event(f"{t['name']} ya es adulta" if not self.male else f"{t['name']} ya es adulto")
        # a fecundated female doll lays her eggs one by one
        if self.mated and self.eggs_left > 0 and self.state not in ("copulate", "flyaway"):
            self.lay_t += dt
            if self.lay_t > 5.0:
                self.lay_t = 0.0
                self.eggs_left -= 1
                if world.lay_egg(self.x - math.cos(self.heading) * 5, self.y - math.sin(self.heading) * 5,
                                 mother=self.id, father=self.sperm):
                    world.record("egg", f"{t['name']} pone un huevo", self.gen,
                                 ids=(self.id, (self.sperm or {}).get("id")))
        dx, dy = fly["x"] - self.x, fly["y"] - self.y
        dist = math.hypot(dx, dy)

        if self.state == "copulate":
            self.speed = 0.0
            if self.state_t > world.MATING_S:
                self._set("wander")
                self.cooldown = 8.0
            return
        if self.state == "fight":                      # male-male: face off, wings raised
            o = self.fight_with
            if o is not None:
                a = math.atan2(o.y - self.y, o.x - self.x)
                self.heading += max(-0.3, min(0.3, _ang(a - self.heading)))
                self.speed = 25 * math.sin(self.state_t * 18)  # little lunges
                self.x += math.cos(self.heading) * self.speed * dt
                self.y += math.sin(self.heading) * self.speed * dt
            self._walls()
            return
        if self.state == "retreat":
            self.speed = base * 1.5
            self.x += math.cos(self.heading) * self.speed * dt
            self.y += math.sin(self.heading) * self.speed * dt
            if self.state_t > 1.2:
                self._set("wander")
            self._walls()
            return
        if self.state == "flyaway":
            self.x += math.cos(self.heading) * 90 * dt
            self.y += math.sin(self.heading) * 90 * dt
            if self.state_t > 0.7:
                self._set("wander")
            self._walls()
            return

        # 0) night: most dolls go to bed, under a leaf if there is one nearby
        night = world.is_night()
        if self.state == "sleep":
            self.speed = 0.0
            if (not night and random.random() < dt / 6) or any(
                    math.hypot(sh["x"] - self.x, sh["y"] - self.y) < sh["r"] + 25 for sh in world.shadows):
                self._set("wander")
            return
        if self.state == "to_bed":
            leaf = world.leaf_at(self.x, self.y)
            target = min((t for t in world.things if t.kind == "leaf"),
                         key=lambda t: math.hypot(t.x - self.x, t.y - self.y), default=None)
            if leaf or target is None or self.state_t > 20 or not night:
                self._set("sleep" if night else "wander")
                return
            self._steer(target.x + random.uniform(-6, 6), target.y + random.uniform(-6, 6), base * 0.8, dt)
            self.x += math.cos(self.heading) * self.speed * dt
            self.y += math.sin(self.heading) * self.speed * dt
            self._walls()
            return
        if night and self.state in ("wander", "approach", "avoid", "with_friend", "avoid_enemy") \
                and random.random() < dt / 8:
            has_leaf = any(t.kind == "leaf" for t in world.things)
            self._set("to_bed" if has_leaf else "sleep")
            return

        # 1) threats: shadows scare dolls too
        for sh in world.shadows:
            if math.hypot(sh["x"] - self.x, sh["y"] - self.y) < sh["r"] + 25:
                self.heading = math.atan2(self.y - sh["y"], self.x - sh["x"]) + random.uniform(-0.6, 0.6)
                self._set("flyaway")
                return

        aggressor = None
        if t["helper"] > 0.5:
            for d in world.dolls:
                if d is not self and d.traits["aggression"] > 0.5 and \
                        math.hypot(d.x - fly["x"], d.y - fly["y"]) < 40:
                    aggressor = d
                    break

        # 2) helper: defend the real fly, or bring it food when it is hungry
        if aggressor is not None:
            self._set("defend")
            self._steer(aggressor.x, aggressor.y, base * 1.6, dt)
            if math.hypot(aggressor.x - self.x, aggressor.y - self.y) < 7 and self.cooldown == 0:
                a = math.atan2(aggressor.y - fly["y"], aggressor.x - fly["x"])
                aggressor.x += math.cos(a) * 25
                aggressor.y += math.sin(a) * 25
                aggressor.cooldown = 3.0
                aggressor.like(self.id, -30)
                self.like(aggressor.id, -20)
                self.cooldown = 1.5
                world.event(f"{t['name']} defiende a la mosca y aparta a {aggressor.traits['name']}")
        elif t["helper"] > 0.5 and (self.carrying or fly["energy"] < 55) and self.cooldown == 0:
            if not self.carrying:
                food = min((f for f in world.things if f.kind == "food" and f.amount > 15),
                           key=lambda f: math.hypot(f.x - self.x, f.y - self.y), default=None)
                if food is None:
                    self._wander(base, dt)
                else:
                    self._set("fetch")
                    self._steer(food.x, food.y, base * 1.3, dt)
                    if math.hypot(food.x - self.x, food.y - self.y) < food.r + 3:
                        food.amount -= 15
                        self.carrying = True
                        world.event(f"{t['name']} recoge comida para la mosca")
            else:
                self._set("deliver")
                self._steer(fly["x"], fly["y"], base * 1.3, dt)
                if dist < 14:
                    world.add("food", self.x + dx * 0.4, self.y + dy * 0.4, r=3, amount=20)
                    self.carrying = False
                    self.cooldown = 10.0
                    world.event(f"{t['name']} deja una miga junto a la mosca")
        # 3) violent: lunge at the real fly
        elif t["aggression"] > 0.3 and dist < 30 + 40 * t["aggression"] and self.cooldown == 0 and not self.young:
            self._set("lunge")
            self._steer(fly["x"], fly["y"], 60 + 60 * t["aggression"], dt, turn_rate=9)
            if dist < 6:
                self.touching = True
                if fly["state"] not in ("fly", "mate"):
                    fly["x"] += dx / max(dist, 1e-3) * 8
                    fly["y"] += dy / max(dist, 1e-3) * 8
                self.cooldown = 3.5 - 2.5 * t["aggression"]
                world.hurt(self)
        # 4) male in mating season: courtship ritual (orient -> tap -> sing -> attempt)
        elif self.male and t["mating"] and self.cooldown == 0 and not self.young:
            tid, tp = self._pick_target(world)
            if tid is None:                           # no female in season: just wander
                self._wander(base, dt)
            else:
                if tid != self.target:
                    self.target = tid
                    self._stage(0)
                tdx, tdy = tp["x"] - self.x, tp["y"] - self.y
                self._court(world, dt, math.hypot(tdx, tdy), tdx, tdy, base, tp)
        # 5) friends and enemies among dolls
        elif self._social_dolls(world, dt, base):
            pass
        # 6) sociable / shy toward Flippy (sociable ones keep ~25 px: company, not crowding)
        elif t["social"] > 0.55 and dist < 12:
            self._set("approach")
            self._steer(self.x - dx, self.y - dy, base * 0.6, dt)
        elif t["social"] > 0.55 and dist > 25:
            self._set("approach")
            self._steer(fly["x"], fly["y"], base, dt)
        elif t["social"] < 0.35 and dist < 45:
            self._set("avoid")
            self._steer(self.x - dx, self.y - dy, base * 1.2, dt)
        else:
            self._wander(base, dt)
            if dist < 12:                          # personal space: drift away from her
                self.heading = math.atan2(-dy, -dx) + random.uniform(-0.5, 0.5)

        if self.state not in ("copulate", "sleep", "to_bed") and dist < 6 and \
                self.speed * math.cos(self.heading - math.atan2(dy, dx)) > 3:
            self.touching = True                   # only bumping into her counts as touch
        # only violent dolls rush at her: everyone else approaches gently (a fast approach
        # looms on her retina and triggers her giant fiber)
        if self.state != "lunge" and dist < 45:
            closing = self.speed * math.cos(self.heading - math.atan2(dy, dx))
            if closing > 20:
                self.speed *= 20 / closing
        self.x += math.cos(self.heading) * self.speed * dt
        self.y += math.sin(self.heading) * self.speed * dt
        self._walls()

    def _court(self, world, dt, dist, dx, dy, base, f):
        t = self.traits
        if self.state != "court":
            self._stage(0)
        self._set("court")
        self.court_t += dt
        if f["state"] in ("fly", "mate") or dist > 40:
            self._stage(0)                            # lost her: start again
        behind = f["heading"] + math.pi * 0.75
        # approach gently: a fast approach expands on her retina like a looming threat
        # and her giant fiber makes her take off (measured, it happened every time)
        # (what matters is the closing speed, so match her pace plus a small margin)
        gentle = min(base * 1.4, max(0.0, f["speed"]) + 3 + 1.2 * max(0.0, dist - 6))
        st = self.court_stage
        if st == 0:                                   # orient: approach and face her
            self._steer(f["x"], f["y"], gentle, dt)
            if dist < 14 and f["state"] not in ("fly", "mate"):
                self._stage(1)
        elif st == 1:                                 # tap her with a foreleg (tastes her)
            self._steer(f["x"], f["y"], 12, dt)
            if dist < 7:
                self.touching = True
            if self.court_t > 0.8:
                self._stage(2)
        elif st == 2:                                 # sing: beside/behind her, one wing vibrating
            tx, ty = f["x"] + math.cos(behind) * 9, f["y"] + math.sin(behind) * 9
            if dist > 11:
                self._steer(tx, ty, gentle, dt)
            else:
                self.speed *= 0.6
                self.heading += max(-0.2, min(0.2, _ang(math.atan2(dy, dx) - self.heading)))
            self.singing = True
            if self.court_t > 1.0 + 3.0 * (1.0 - t["song"]):   # good singers are ready sooner
                self._stage(3)
        else:                                         # attempt copulation from behind
            tx, ty = f["x"] - math.cos(f["heading"]) * 5, f["y"] - math.sin(f["heading"]) * 5
            self._steer(tx, ty, min(gentle, max(0.0, f["speed"]) + 6), dt, turn_rate=8)
            self.singing = True
            self.attempting = dist < 9
            if self.court_t > 3.0:                    # she neither accepted nor rejected
                self._stage(2)

    def court_progress(self):
        if self.state != "court":
            return 0.0
        dur = [2.0, 0.8, 1.0 + 3.0 * (1.0 - self.traits["song"]), 3.0][self.court_stage]
        return self.court_stage + min(1.0, self.court_t / dur)

    def _social_dolls(self, world, dt, base):
        friend, enemy = self.friends_and_enemies(world)
        if enemy is not None and math.hypot(enemy.x - self.x, enemy.y - self.y) < 45:
            self._set("avoid_enemy")
            self._steer(2 * self.x - enemy.x, 2 * self.y - enemy.y, base * 1.2, dt)
            return True
        if friend is not None and math.hypot(friend.x - self.x, friend.y - self.y) > 20:
            self._set("with_friend")
            self._steer(friend.x, friend.y, base * 0.9, dt)
            return True
        return False

    def _wander(self, base, dt, world=None):
        self._set("wander")
        self.heading += random.gauss(0, 2.0) * dt
        self.speed += 0.2 * (base * 0.6 - self.speed)

    def _walls(self):
        if self.x < 4 or self.x > W - 4:
            self.heading = math.pi - self.heading
        if self.y < 4 or self.y > H - 4:
            self.heading = -self.heading
        self.x, self.y = min(max(self.x, 4), W - 4), min(max(self.y, 4), H - 4)
        self.heading = _ang(self.heading)

    def to_json(self):
        return dict(id=self.id, x=round(self.x, 1), y=round(self.y, 1),
                    heading=round(self.heading, 3), state=self.state,
                    court_stage=self.court_stage if self.state == "court" else None,
                    court_p=round(self.court_progress(), 2),
                    singing=self.singing, carrying=self.carrying, traits=self.traits,
                    gen=self.gen, mother=self.mother, father=self.father, mated=self.mated,
                    young=self.young, target=self.target if self.state == "court" else None,
                    old=bool(self.lifespan and self.age > 0.8 * self.lifespan),
                    matings=self.matings, rest=round(self.rest, 1),
                    affinity={str(k): round(v) for k, v in self.affinity.items() if abs(v) >= 5})
