"""Sensory (input) and motor (output) neuron groups, selected by FlyWire cell types.

Each group is a pandas query over the FlyWire annotation table. Edit freely: this
is the "wiring" between the virtual body and the connectome.
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"

ATTRACTIVE_ORN = ["ORN_DM1", "ORN_DM4", "ORN_DP1m", "ORN_VM7d"]  # vinegar / fruit
AVERSIVE_ORN = ["ORN_V", "ORN_DA2"]  # CO2, geosmin

SENSORY = {  # name -> query ; *_L/*_R are the two sides of the body
    "sugar_L": "cell_sub_class=='sugar/water' and side=='left'",
    "sugar_R": "cell_sub_class=='sugar/water' and side=='right'",
    "bitter_L": "cell_sub_class=='bitter' and side=='left'",
    "bitter_R": "cell_sub_class=='bitter' and side=='right'",
    "odor_good_L": f"cell_type in {ATTRACTIVE_ORN} and side=='left'",
    "odor_good_R": f"cell_type in {ATTRACTIVE_ORN} and side=='right'",
    "odor_bad_L": f"cell_type in {AVERSIVE_ORN} and side=='left'",
    "odor_bad_R": f"cell_type in {AVERSIVE_ORN} and side=='right'",
    "loom_L": "cell_type in ['LC4','LPLC2'] and side=='left'",
    "loom_R": "cell_type in ['LC4','LPLC2'] and side=='right'",
    "wind": "cell_sub_class=='wind_gravity'",
    "touch_L": "cell_sub_class=='grooming' and side=='left'",
    "touch_R": "cell_sub_class=='grooming' and side=='right'",
    "heat": "cell_type in ['TRN_VP2','TRN_VP3a','TRN_VP3b']",
    # --- vision beyond looming / flies ---
    # LPLC1: objects approaching while walking -> the brain turns away (contralateral DNa02)
    "obstacle_L": "cell_type=='LPLC1' and side=='left'",
    "obstacle_R": "cell_type=='LPLC1' and side=='right'",
    # ocelli (OCG projection neurons): light -> DNp18, graded with intensity
    "light_L": "cell_type.str.startswith('OCG') and side=='left'",
    "light_R": "cell_type.str.startswith('OCG') and side=='right'",
    # --- social senses (other flies / "dolls") ---
    "fly_seen_L": "cell_type in ['LC10a','LC11'] and side=='left'",   # small moving object
    "fly_seen_R": "cell_type in ['LC10a','LC11'] and side=='right'",
    "cva": "cell_type=='ORN_DA1'",                    # Or67d: cVA, male pheromone
    "fly_odor": "cell_type in ['ORN_VA1v','ORN_VA1d']",  # Or47b / Or88a: fly body odours
    "song": "cell_type.str.startswith('JO-B')",        # Johnston's organ, courtship song
    # Shortcut: the pulse-song detector between JO-B and vpoEN is a temporal filter
    # that a LIF model with fixed weights does not reproduce (song alone never
    # reaches vpoEN in simulation), so courtship song also drives vpoEN directly.
    "song_vpoEN": "cell_type=='vpoEN'",
    # Internal state, not a sense: mating drive of the female (pC1 is set by
    # hormones / mated status, which the connectome does not contain).
    "mating_drive": "cell_type.str.startswith('pC1')",
    # Internal state: mature eggs after mating. The ovary / sex-peptide signals are not
    # in the brain connectome, so "eggs ready" drives SMP550, the main excitatory input
    # to oviDN (no sensory stimulus reaches oviDN in simulation).
    "egg_drive": "cell_type=='SMP550'",
}

MOTOR = {  # descending / motor neurons read out to drive the body
    "turn_L": "cell_type in ['DNa02','DNa01'] and side=='left'",   # ipsilateral steering
    "turn_R": "cell_type in ['DNa02','DNa01'] and side=='right'",
    "forward": "cell_type in ['DNg100','DNp09']",                  # BDN2 / P9 walking
    "backward": "cell_type=='MDN'",                                 # moonwalker
    "escape": "cell_type=='DNp01'",                                 # giant fiber
    "alarm": "cell_type in ['DNp02','DNp04','DNp11']",             # graded looming response
    "feed": "cell_type in ['CB0701','MN10','CB0700']",             # MN9 proboscis etc.
    "groom": "cell_type=='DNg84'",  # only touch-specific DN (DNg35/29/57 also fire for odor/wind)
    "avoid": "cell_type=='DNb05'",                                # CO2 / heat
    "odor_drive": "cell_type in ['DNg100','DNge053']",             # graded by attractive odour
    "receptive": "cell_type=='DNp37'",                             # vpoDN: female accepts male
    "reject": "cell_type=='DNp13'",                                # ovipositor extrusion: rejects
    "lay": "cell_type.str.startswith('oviDN')",                   # egg laying (sugar lowers it)
    "light_seen": "cell_type=='DNp18'",                           # ocelli -> "there is light"
}


def load_groups(brain):
    ann = pd.read_csv(DATA / "annotations.tsv", sep="\t", low_memory=False,
                      usecols=["root_id", "super_class", "cell_class", "cell_sub_class",
                               "cell_type", "side", "pos_x", "pos_y", "pos_z", "top_nt"])
    ann["idx"] = ann.root_id.map(brain.id2idx)
    ann = ann.dropna(subset=["idx"])
    ann["idx"] = ann.idx.astype(np.int64)
    sens = {k: ann.query(q).idx.to_numpy() for k, q in SENSORY.items()}
    mot = {k: ann.query(q).idx.to_numpy() for k, q in MOTOR.items()}
    return ann, sens, mot


def brain_map(brain, ann, w=240, h=120):
    """Pixel coordinates of every neuron for the frontal 'brain view' (x, y)."""
    xy = np.full((brain.n, 2), -1, np.int16)
    x, y = ann.pos_x.to_numpy(float), ann.pos_y.to_numpy(float)
    lo_x, hi_x = np.percentile(x, [0.5, 99.5])
    lo_y, hi_y = np.percentile(y, [0.5, 99.5])
    px = np.clip((x - lo_x) / (hi_x - lo_x) * (w - 1), 0, w - 1)
    py = np.clip((y - lo_y) / (hi_y - lo_y) * (h - 1), 0, h - 1)
    xy[ann.idx.to_numpy()] = np.stack([px, py], 1).astype(np.int16)
    return xy
