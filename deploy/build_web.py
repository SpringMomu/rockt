"""Build the static web version of the 3D simulator (runs entirely in the browser).

    python deploy/build_web.py [out_dir]        (default: dist/web3d)

Output: web3d/ front end + the simulator's Python sources (py/) + a trimmed
Pyodide distribution (pyodide/: runtime, numpy, scipy, clarabel and their
dependencies).  Serve the directory with any static web server; no Python
backend is needed.  Pyodide files are downloaded once into .cache/.
"""

from __future__ import annotations

import json
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYODIDE_VERSION = "0.28.3"
CDN = f"https://cdn.jsdelivr.net/pyodide/v{PYODIDE_VERSION}/full/"
CORE = ["pyodide.js", "pyodide.asm.js", "pyodide.asm.wasm", "python_stdlib.zip", "pyodide-lock.json"]
PACKAGES = ["numpy", "scipy", "clarabel"]
SIM_FILES = ["rocket3d.py", "aero3d.py", "guidance.py", "guidance3d.py", "guidance3d_legacy.py", "sim3d.py", "main3d.py"]


def fetch(name: str, cache: Path) -> Path:
    path = cache / name
    if not path.exists():
        print(f"  download {name}", flush=True)
        tmp = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(CDN + name, timeout=120) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.replace(path)
    return path


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "dist" / "web3d"
    cache = ROOT / ".cache" / f"pyodide-{PYODIDE_VERSION}"
    cache.mkdir(parents=True, exist_ok=True)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(ROOT / "web3d", out)
    (out / "py").mkdir()
    for f in SIM_FILES:
        shutil.copy2(ROOT / f, out / "py" / f)

    pyo = out / "pyodide"
    pyo.mkdir()
    for name in CORE:
        shutil.copy2(fetch(name, cache), pyo / name)
    lock = json.loads((cache / "pyodide-lock.json").read_text(encoding="utf-8"))["packages"]
    need, todo = set(), list(PACKAGES)
    while todo:
        p = todo.pop()
        if p not in need:
            need.add(p)
            todo.extend(lock[p].get("depends", []))
    for p in sorted(need):
        shutil.copy2(fetch(lock[p]["file_name"], cache), pyo / lock[p]["file_name"])
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"built {out}  ({size / 1e6:.1f} MB, packages: {', '.join(sorted(need))})")


if __name__ == "__main__":
    main()
