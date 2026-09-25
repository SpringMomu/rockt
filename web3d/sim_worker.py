"""Browser-side glue: runs the 3D simulator inside Pyodide (Web Worker).

Same physics/guidance/state as ``python main3d.py``; the real-time loop is
main3d.Runner's, driven by the worker's timer instead of a thread.
"""

import json
import sys

sys.path.insert(0, "/home/pyodide/sim")

import main3d
from sim3d import PHYSICS_DT, Sim3D

sim = Sim3D(cfd=True)
_acc = 0.0


def advance(real: float) -> None:
    """Advance by ``real`` wall seconds (x time warp); never fast-forwards to
    catch up after a slow step, like main3d.Runner."""
    global _acc
    _acc += min(0.1, max(0.0, real)) * sim.time_warp
    _acc = min(_acc, 0.1 * max(1.0, sim.time_warp))
    n = 0
    while _acc >= PHYSICS_DT and n < 2400:
        sim.step(PHYSICS_DT)
        _acc -= PHYSICS_DT
        n += 1
    if n >= 2400:
        _acc = 0.0


def state() -> str:
    return json.dumps(main3d._finite(main3d.snapshot(sim)), separators=(",", ":"), allow_nan=False,
                      default=lambda o: main3d._finite(float(o)) if hasattr(o, "__float__") else str(o))


def handle_input(payload: str) -> None:
    p = json.loads(payload)
    with sim.lock:
        sim.keys = {k: bool(v) for k, v in p.get("keys", {}).items()}
        for action in p.get("actions", []):
            main3d.apply_action(sim, action)


def handle_cfd(payload: str) -> None:
    p = json.loads(payload)
    if sim.cfd is not None and isinstance(p.get("c_world"), list):
        sim.cfd.update(p["c_world"])
