"""FLIPPY LA MOSCA - web server.

    python3 server.py              # connectome menu, then opens http://localhost:8000
    python3 server.py --no-menu    # straight to the game (FlyWire v783)
    python3 server.py --port 8080 --no-browser
"""
import argparse
import json
import re
import sys
import threading
import time
import traceback
import webbrowser

import numpy as np
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parent
from flippy import catalog  # noqa: E402


def startup_menu():
    """Connectome menu in the terminal: pick a downloaded one, or download others.
    Only FlyWire v783 runs in the game for now; the rest can already be downloaded."""
    keys = list(catalog.MODELS)
    while True:
        print("\n=== FLIPPY LA MOSCA · conectomas ===")
        for i, k in enumerate(keys, 1):
            m = catalog.MODELS[k]
            have = catalog.is_downloaded(k)
            state = "✔ descargado" if have else f"↓ descargable ({m['size']})"
            game = "▶ funciona en el juego" if m["compatible"] else "⏳ aún no compatible"
            print(f" [{i}] {m['name']:17s} {m['fly']:42s} {m['neurons']:17s} {state:22s} {game}")
        default = catalog.DEFAULT if catalog.is_downloaded(catalog.DEFAULT) else None
        hint = f"Enter = usar {catalog.MODELS[default]['name']}" if default else "elige 1 para descargar el del juego"
        try:
            ans = input(f"\nNúmero para usar o descargar · {hint} · q = salir: ").strip().lower()
        except EOFError:
            ans = ""
        if ans in ("q", "salir"):
            sys.exit(0)
        if ans == "" and default:
            return default
        if not ans.isdigit() or not 1 <= int(ans) <= len(keys):
            print("  (no entendí la opción)")
            continue
        key = keys[int(ans) - 1]
        m = catalog.MODELS[key]
        if not catalog.is_downloaded(key):
            ok = input(f"  ¿Descargar {m['name']} ({m['size']}) en data/models? [s/N]: ").strip().lower()
            if ok not in ("s", "si", "sí", "y", "yes"):
                continue
            try:
                catalog.download(key)
            except Exception as e:  # network errors etc.: back to the menu
                print(f"  ✖ {e}")
                continue
        if m["compatible"]:
            return key
        print(f"  {m['name']} ya está descargado ({m['cite']}).\n"
              f"  Todavía no es compatible con el juego: el cerebro, los grupos de neuronas y la memoria están\n"
              f"  construidos sobre FlyWire v783. Elige FlyWire v783 para jugar.")


ap = argparse.ArgumentParser(description="FLIPPY LA MOSCA")
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--no-browser", action="store_true", help="no abrir el navegador")
ap.add_argument("--no-menu", action="store_true", help="no mostrar el menú de conectomas al arrancar")
args = ap.parse_args()
if sys.stdin.isatty() and not args.no_menu:
    startup_menu()
if not catalog.is_downloaded(catalog.DEFAULT):
    sys.exit("Faltan los datos de FlyWire v783 en data/. Descárgalos con:  python3 download_data.py")

from flippy.world import World, TICK_MS  # noqa: E402  (after choosing / checking the data)

SAVES = ROOT / "saves"
SAVES.mkdir(exist_ok=True)
AUTOSAVE_EVERY_S = 60  # simulated seconds
sys.stdout.reconfigure(line_buffering=True)  # messages show up even when output goes to a file
print("Cargando el conectoma (138.639 neuronas, 15 M de conexiones)...")
world = World(dt=0.2)
lock = threading.Lock()
ctl = dict(paused=False, speed=1.0, ratio=0.0, last_autosave=0.0, error=None)


def write_save(name, auto=False):
    """Serialise under the lock, write the file outside it."""
    with lock:
        data = dict(world.to_save(), name=name, auto=auto, saved_at=time.time(), summary=world.summary())
    slug = "autosave" if auto else (re.sub(r"[^\w\- ]", "", name, flags=re.U).strip().replace(" ", "_")[:40] or "partida")
    path = SAVES / f"{slug}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, default=float))
    tmp.replace(path)  # atomic: a crash mid-write never leaves a broken save
    return path.name


_META = {}


def brain_meta():
    """For the experiment window: per-neuron class and type, and the documented catalogue."""
    if not _META:
        from flippy import experiments
        ann = world.ann.set_index("idx")
        n = world.brain.n
        classes = ["optic", "central", "sensory", "visual_projection", "descending", "ascending", "motor",
                   "visual_centrifugal", "endocrine", "sensory_ascending"]
        cls = np.full(n, -1, np.int16)
        sc = ann.super_class.reindex(range(n))
        for i, c in enumerate(classes):
            cls[(sc == c).to_numpy()] = i
        ct = ann.cell_type.reindex(range(n)).fillna("").astype(str)
        types, ctype = np.unique(ct.to_numpy(), return_inverse=True)
        _META.update(classes=classes, cls=cls.tolist(), types=types.tolist(), ctype=ctype.tolist(),
                     functions=[dict(key=k, label=v[0], where=v[2], what=v[3], effect=v[4], ref=v[5],
                                     idx=world.fn_idx[k].tolist()) for k, v in experiments.FUNCTIONS.items()])
    return _META


def list_saves():
    out = []
    for f in sorted(SAVES.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            d = json.loads(f.read_text())
            out.append(dict(file=f.name, name=d.get("name", f.stem), auto=d.get("auto", False),
                            saved_at=d.get("saved_at", f.stat().st_mtime), summary=d.get("summary", {})))
        except (OSError, ValueError):
            continue
    return out


def sim_loop():
    while True:
        if ctl["paused"]:
            time.sleep(0.05)
            continue
        t0 = time.perf_counter()
        try:
            with lock:
                world.tick()
                due = world.time_ms / 1000 - ctl["last_autosave"] >= AUTOSAVE_EVERY_S
        except Exception:  # never let the simulation thread die silently
            traceback.print_exc()
            ctl["error"] = traceback.format_exc().strip().splitlines()[-1]
            ctl["paused"] = True
            continue
        if due:
            ctl["last_autosave"] = world.time_ms / 1000
            try:
                write_save("Autoguardado", auto=True)
            except OSError as e:
                print("autosave failed:", e)
        el = time.perf_counter() - t0
        budget = TICK_MS / 1000 / ctl["speed"]
        ctl["ratio"] = 0.9 * ctl["ratio"] + 0.1 * (TICK_MS / 1000 / max(el, budget))
        if el < budget:
            time.sleep(budget - el)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send((ROOT / "web" / "index.html").read_bytes(), "text/html; charset=utf-8")
        if self.path == "/api/map":
            with lock:
                return self._send(dict(
                    n=world.brain.n,
                    xy=world.map_xy.ravel().tolist(),
                    sensory={k: v.tolist() for k, v in world.sens_idx.items()},
                    motor={k: v.tolist() for k, v in world.mot_idx.items()},
                    **brain_meta(),
                ))
        if self.path == "/api/saves":
            return self._send(list_saves())
        if self.path == "/api/history":
            with lock:
                return self._send(dict(chronicle=world.chronicle, family=list(world.family.values())))
        if self.path == "/api/state":
            with lock:
                s = world.snapshot()
            s.update(paused=ctl["paused"], speed=ctl["speed"], ratio=round(ctl["ratio"], 2), error=ctl["error"],
                     assist=world.assist, base_walk=world.base_walk,
                     gain={k: round(v, 2) for k, v in world.gain.items()})
            return self._send(s)
        self.send_error(404)

    def do_POST(self):
        cmd = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        c = cmd.get("cmd")
        if c == "save":
            return self._send({"ok": True, "file": write_save(str(cmd.get("name") or "Partida"))})
        if c == "shutdown":                        # "save and switch off" button in the page
            ctl["paused"] = True
            name = write_save("Autoguardado", auto=True)
            ctl["shutting_down"] = True
            self._send({"ok": True, "file": name})
            threading.Thread(target=httpd.shutdown, daemon=True).start()
            return
        if c == "load":
            f = SAVES / Path(str(cmd.get("file", ""))).name   # no path tricks: only files in saves/
            try:
                data = json.loads(f.read_text())
                with lock:
                    world.load_save(data)
                    ctl["last_autosave"] = world.time_ms / 1000
            except (OSError, ValueError, KeyError) as e:
                return self._send({"ok": False, "error": str(e)})
            return self._send({"ok": True})
        with lock:
            if c == "add":
                world.add(cmd["kind"], cmd["x"], cmd["y"], **cmd.get("traits", {}))
            elif c == "doll":
                d = world.doll(cmd["id"])
                if d:
                    d.set_traits(cmd["traits"])
                    d.force_season(bool(cmd["traits"].get("mating")))
                    if d.id in world.family:
                        world.family[d.id].update(name=d.traits["name"], sex=d.traits["sex"])
            elif c == "doll_remove":
                world.mark_gone([cmd["id"]], cause="retirada de la arena")
                world.dolls = [d for d in world.dolls if d.id != cmd["id"]]
            elif c == "heat":
                world.set_heat(cmd["value"])
            elif c == "assist":
                world.assist = bool(cmd["value"])
            elif c == "silence":
                world.silence(cmd.get("kind"), cmd.get("key"), cmd.get("x"), cmd.get("y"), cmd.get("r"))
            elif c == "unsilence":
                world.unsilence(cmd.get("id"))
            elif c == "daynight":
                world.daynight = bool(cmd["value"])
            elif c == "arena" and cmd.get("value") in ("S", "M", "L"):
                world.set_arena(cmd["value"])
            elif c == "remove":
                world.remove_near(cmd["x"], cmd["y"])
            elif c == "reset":
                world.reset()
                ctl["last_autosave"] = 0.0
                ctl["error"] = None
            elif c == "pause":
                ctl["paused"] = not ctl["paused"]
                ctl["error"] = None
            elif c == "speed":
                ctl["speed"] = float(cmd["value"])
            elif c == "gain":
                world.gain[cmd["key"]] = float(cmd["value"])
            elif c == "walk":
                world.base_walk = float(cmd["value"])
        self._send({"ok": True})


def _stop(*_):
    raise KeyboardInterrupt  # SIGTERM (kill) also autosaves before exiting


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGTERM, _stop)
    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", args.port), H)
    except OSError:
        sys.exit(f"El puerto {args.port} está ocupado (¿ya hay un FLIPPY abierto?). Prueba: python3 server.py --port 8001")
    threading.Thread(target=sim_loop, daemon=True).start()
    url = f"http://localhost:{args.port}"
    print(f"FLIPPY listo en {url}  (Ctrl+C para salir; se autoguarda)")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        httpd.serve_forever()
        if ctl.get("shutting_down"):
            print("Partida guardada y FLIPPY apagado desde la página. Para volver:  python3 server.py")
    except KeyboardInterrupt:
        print("\nGuardando antes de salir...", write_save("Autoguardado", auto=True))
