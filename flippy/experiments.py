"""Experiment mode: silence neurons and watch what behaviour disappears.

"Silenced" neurons cannot fire, like optogenetic inhibition (GtACR1) or Kir2.1
expression in real flies: everything downstream loses their input. Two ways:
- a catalogue of well-known groups (FUNCTIONS), each documented for the UI;
- any area of the frontal brain map (like shining light on a spot).
"""
import numpy as np

# key -> (label, pandas query over the annotation table, where, what it is, expected effect, reference)
FUNCTIONS = {
    "giant_fiber": ("Fibra gigante", "cell_type=='DNp01'", "cerebro → médula ventral (2 neuronas)",
                    "La neurona de escape: una descarga suya hace despegar a la mosca en milisegundos.",
                    "Ante una amenaza ya no despega: como mucho se aleja caminando (neuronas de alarma).",
                    "von Reyn et al. 2014"),
    "feeding": ("Comer (MN9)", "cell_type in ['CB0701','MN10','CB0700']", "ganglio subesofágico (7 neuronas)",
                "Motoneuronas de la probóscide: la extienden para comer cuando las patas notan azúcar.",
                "Aunque pise la comida y sus neuronas del azúcar disparen, no come y su energía baja.",
                "Shiu et al. 2024"),
    "steering": ("Girar (DNa01/DNa02)", "cell_type in ['DNa01','DNa02']", "descendentes (4 neuronas)",
                 "Descendentes de giro: cada lado hace girar hacia su propio lado.",
                 "Deja de girar por decisión de su cerebro: no sigue a otras moscas ni esquiva obstáculos.",
                 "Rayshubskiy et al. 2020"),
    "looming": ("Detectar amenazas (LC4/LPLC2)", "cell_type in ['LC4','LPLC2']", "lóbulo óptico → cerebro (~310)",
                "Detectores de objetos que se acercan rápido (la sombra de una mano).",
                "Queda 'ciega' a las amenazas: un manotazo ya no la asusta.", "Ache et al. 2019"),
    "obstacles": ("Esquivar obstáculos (LPLC1)", "cell_type=='LPLC1'", "lóbulo óptico → cerebro (~140)",
                  "Detectan lo que se le acerca mientras camina y la hacen girar al lado contrario.",
                  "Choca más con piedras y paredes.", "Tanaka & Clark 2022"),
    "see_flies": ("Ver otras moscas (LC10a/LC11)", "cell_type in ['LC10a','LC11']", "lóbulo óptico → cerebro (~360)",
                  "Detectores de objetos pequeños en movimiento, como otra mosca.",
                  "Ya no se gira hacia las moscas que ve.", "Ribeiro et al. 2018"),
    "kenyon": ("Memoria (células de Kenyon)", "cell_class=='Kenyon_Cell'", "cuerpo pedunculado (5.177)",
               "Las neuronas del centro de memoria: cada olor activa un grupo distinto.",
               "No aprende nada nuevo y no puede usar lo que recordaba (no reconoce olores).",
               "Heisenberg 2003"),
    "punish": ("Aprender castigos (dopamina PPL1)", "cell_type.str.startswith('PPL1')", "cuerpo pedunculado (16)",
               "Dopamina que enseña 'esto es malo' (golpes, calor).",
               "Deja de aprender de los golpes y del calor; sigue aprendiendo lo bueno.", "Aso et al. 2014"),
    "reward": ("Aprender recompensas (dopamina PAM)", "cell_type.str.startswith('PAM')", "cuerpo pedunculado (~300)",
               "Dopamina que enseña 'esto es bueno' (comer, dormir tranquila).",
               "Deja de aprender lo bueno (comederos, hojas favoritas); sigue aprendiendo castigos.", "Aso et al. 2014"),
    "receptive": ("Aceptar pareja (vpoDN)", "cell_type=='DNp37'", "descendentes (2 neuronas)",
                  "Abren la placa vaginal: la hembra acepta al macho.",
                  "Nunca acepta a un macho, aunque esté en época y él cante muy bien.", "Wang et al. 2021"),
    "reject": ("Rechazar (DNp13)", "cell_type=='DNp13'", "descendentes (2 neuronas)",
               "Sacan el ovipositor para rechazar al macho (típico de hembras fecundadas).",
               "Ya no puede rechazar con el ovipositor.", "Wang et al. 2021"),
    "mating_drive": ("Deseo sexual (pC1)", "cell_type.str.startswith('pC1')", "protocerebro (10 neuronas)",
                     "Codifican el estado de apareamiento de la hembra.",
                     "Aunque la pongas en época, no llega a estar receptiva.", "Zhou et al. 2014"),
    "egg_laying": ("Poner huevos (oviDN)", "cell_type.str.startswith('oviDN')", "descendentes (6 neuronas)",
                   "Ordenan la puesta de huevos.", "Madura huevos pero no los pone.", "Wang et al. 2020"),
    "grooming": ("Acicalarse (DNg84)", "cell_type=='DNg84'", "descendentes (2 neuronas)",
                 "La orden de acicalarse cuando le tocan las antenas.", "Ya no se acicala al tocarla.",
                 "Hampel et al. 2015"),
    "odor_drive": ("Atracción por el olor (DNg100/DNge053)", "cell_type in ['DNg100','DNge053']",
                   "descendentes (4 neuronas)", "Crecen con el olor a comida y la hacen avanzar hacia él.",
                   "El olor a comida deja de atraerla (solo la encuentra por azar).", "este proyecto"),
    "smell_food": ("Oler la fruta (ORN)", "cell_type in ['ORN_DM1','ORN_DM4','ORN_DP1m','ORN_VM7d']",
                   "antenas (~170 neuronas)", "Receptores olfativos del olor a fruta y vinagre.",
                   "Anosmia a la comida: no la huele.", "Semmelhack & Wang 2009"),
    "ocelli": ("Notar la luz (ocelos)", "cell_type.str.startswith('OCG')", "ocelos → cerebro (20)",
               "Los tres ojos simples de la frente: miden la luz.", "La lámpara deja de atraerla.",
               "Hengstenberg 1993"),
    "an_inhibition": ("Freno del olfato (interneuronas GABA)", "cell_class=='ALLN' and top_nt=='gaba'",
                      "lóbulo antenal (~150)", "Interneuronas inhibidoras que evitan que el olfato se desboque.",
                      "Medido en este modelo: las neuronas de proyección del olfato responden un 20-30 % más fuerte "
                      "al mismo olor (menos contraste entre olores). No llega a provocar una convulsión.",
                      "Olsen & Wilson 2008"),
}


def build(brain, ann):
    """key -> neuron indices (needs 'top_nt' and 'cell_class' in ann)."""
    return {k: ann.query(q).idx.to_numpy() for k, (_, q, *_) in FUNCTIONS.items()}


def region_indices(map_xy, x, y, r):
    """Neurons whose frontal-map pixel lies within r of (x, y) (a column through the brain)."""
    d2 = (map_xy[:, 0] - x) ** 2 + (map_xy[:, 1] - y) ** 2
    return np.flatnonzero((map_xy[:, 0] >= 0) & (d2 <= r * r))
