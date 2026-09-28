"""Download fly connectomes into data/.

    python3 download_data.py                 # FlyWire v783, the one the game uses (~135 MB)
    python3 download_data.py --list          # every connectome in the catalogue
    python3 download_data.py malecns10 manc121

`python3 server.py` also shows this catalogue in a menu when it starts.
"""
import sys

from flippy.catalog import MODELS, DEFAULT, download, is_downloaded


def show():
    for k, m in MODELS.items():
        state = "✔ descargado" if is_downloaded(k) else f"descargable ({m['size']})"
        game = "funciona en el juego" if m["compatible"] else "aún no compatible con el juego"
        print(f"  {k:13s} {m['name']:17s} {m['fly']:42s} {state} · {game}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if "--list" in sys.argv:
        show()
        sys.exit()
    for key in args or [DEFAULT]:
        if key not in MODELS:
            sys.exit(f"No conozco '{key}'. Opciones: {', '.join(MODELS)}")
        print(f"{MODELS[key]['name']} ({MODELS[key]['fly']}):")
        download(key)
    print("Listo. Arranca con:  python3 server.py")
