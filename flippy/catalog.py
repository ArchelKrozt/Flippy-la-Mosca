"""Catalogue of fly connectomes FLIPPY can download.

Only FlyWire v783 runs in the game today (the whole model, neuron groups, memory and
tests are built on its cell types). The others can already be downloaded, to make
them compatible later. All URLs were checked to be public, direct downloads.
"""
import sys
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

_SHIU = "https://github.com/philshiu/Drosophila_brain_model/raw/main"
_LEE = "https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data"
_JAN = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
_ANN = ("https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/"
        "Supplemental_file1_neuron_annotations.tsv")

# files: (file name, url, minimum size in bytes, first bytes of a valid file)
MODELS = {
    "flywire783": dict(
        name="FlyWire v783", fly="hembra · cerebro completo", neurons="138.639 neuronas", size="~135 MB",
        cite="Dorkenwald et al. 2024 · Schlegel et al. 2024 · modelo: Shiu et al. 2024", compatible=True,
        folder=DATA,   # historical location: data/ itself
        files=[("Completeness_783.csv", f"{_SHIU}/Completeness_783.csv", 3_000_000, b","),
               ("Connectivity_783.parquet", f"{_SHIU}/Connectivity_783.parquet", 90_000_000, b"PAR1"),
               ("annotations.tsv", _ANN, 30_000_000, b"supervoxel_id")]),
    "flywire630": dict(
        name="FlyWire v630", fly="hembra · cerebro (versión anterior)", neurons="~127.000 neuronas", size="~90 MB",
        cite="Shiu et al. 2024 (primera versión del modelo)", compatible=False,
        files=[("Completeness_630.csv", f"{_SHIU}/2023_03_23_completeness_630_final.csv", 3_000_000, b","),
               ("Connectivity_630.parquet", f"{_SHIU}/2023_03_23_connectivity_630_final.parquet", 80_000_000, b"PAR1")]),
    "hemibrain121": dict(
        name="Hemibrain v1.2.1", fly="hembra · ~1/3 del cerebro central", neurons="~25.000 neuronas", size="~95 MB",
        cite="Scheffer et al. 2020 (Janelia FlyEM) · CC-BY", compatible=False,
        files=[("hemibrain_121_meta.feather", f"{_LEE}/hemibrain_121/hemibrain_121_meta.feather", 1_000_000, b"ARROW1"),
               ("hemibrain_121_simple_edgelist.feather", f"{_LEE}/hemibrain_121/hemibrain_121_simple_edgelist.feather",
                80_000_000, b"ARROW1")]),
    "manc121": dict(
        name="MANC v1.2.1", fly="macho · médula ventral (sin cerebro)", neurons="~23.000 neuronas", size="~90 MB",
        cite="Takemura et al. 2024 · Marin et al. 2024 (Janelia FlyEM) · CC-BY", compatible=False,
        files=[("manc_121_meta.feather", f"{_LEE}/manc_121/manc_121_meta.feather", 1_000_000, b"ARROW1"),
               ("manc_121_simple_edgelist.feather", f"{_LEE}/manc_121/manc_121_simple_edgelist.feather",
                80_000_000, b"ARROW1")]),
    "banc888": dict(
        name="BANC v888", fly="hembra · cerebro + médula ventral", neurons="~160.000 neuronas", size="~420 MB",
        cite="Bates et al. 2025 (Brain And Nerve Cord)", compatible=False,
        files=[("banc_888_meta.feather", f"{_LEE}/banc_888/banc_888_meta.feather", 50_000_000, b"ARROW1"),
               ("banc_888_edgelist_simple_v3.feather", f"{_LEE}/banc_888/banc_888_edgelist_simple_v3.feather",
                300_000_000, b"ARROW1")]),
    "malecns10": dict(
        name="Male CNS v1.0", fly="macho · cerebro + médula ventral completos", neurons="~166.000 neuronas",
        size="~1,1 GB", cite="Berg et al. 2025 (Janelia FlyEM, Cambridge, Google) · CC-BY", compatible=False,
        files=[("body-annotations-male-cns-v1.0-minconf-0.5.feather",
                f"{_JAN}/body-annotations-male-cns-v1.0-minconf-0.5.feather", 10_000_000, b"ARROW1"),
               ("body-neurotransmitters-male-cns-v1.0.feather",
                f"{_JAN}/body-neurotransmitters-male-cns-v1.0.feather", 30_000_000, b"ARROW1"),
               ("connectome-weights-male-cns-v1.0-minconf-0.5.feather",
                f"{_JAN}/connectome-weights-male-cns-v1.0-minconf-0.5.feather", 900_000_000, b"ARROW1")]),
}
DEFAULT = "flywire783"


def folder(key):
    return MODELS[key].get("folder") or DATA / "models" / key


def _valid(path, min_size, magic):
    try:
        return path.stat().st_size >= min_size and path.open("rb").read(len(magic)) == magic
    except OSError:
        return False


def is_downloaded(key):
    m = MODELS[key]
    return all(_valid(folder(key) / n, size, magic) for n, _, size, magic in m["files"])


def download(key, out=sys.stdout):
    """Download (resumable per file: files already valid are skipped)."""
    d = folder(key)
    d.mkdir(parents=True, exist_ok=True)
    for name, url, min_size, magic in MODELS[key]["files"]:
        dest = d / name
        if _valid(dest, min_size, magic):
            print(f"  ✔ {name} ya está", file=out)
            continue
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(url) as r, tmp.open("wb") as fh:
            total, done = int(r.headers.get("Content-Length") or 0), 0
            while chunk := r.read(1 << 20):
                fh.write(chunk)
                done += len(chunk)
                pct = f"{100 * done / total:5.1f}%" if total else ""
                print(f"\r  ↓ {name}: {done / 1e6:7.1f} MB {pct}", end="", flush=True, file=out)
        print(file=out)
        if not _valid(tmp, min_size, magic):
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"{name}: la descarga no es válida (¿sin conexión o el enlace cambió?): {url}")
        tmp.replace(dest)
        print(f"  ✔ {name}", file=out)
