// Browser-side simulator: loads Pyodide and runs the Python 3D simulator
// (rocket3d / aero3d / guidance3d with Clarabel / sim3d) in this worker.
// Protocol (same data as the /api endpoints of main3d.py):
//   in:  {type: "input", keys, actions} | {type: "cfd", c_world}
//   out: {type: "progress", text} | {type: "state", json} | {type: "error", text}
const PYODIDE = "pyodide/";
const SIM_FILES = ["rocket3d.py", "aero3d.py", "guidance.py", "guidance3d.py", "guidance3d_legacy.py", "sim3d.py", "main3d.py"];
importScripts(PYODIDE + "pyodide.js");

let glue = null;
const queue = [];
self.onmessage = (ev) => { queue.push(ev.data); };

const progress = (text) => self.postMessage({ type: "progress", text });

async function main() {
  progress("正在加载 Python 运行时…");
  const py = await loadPyodide({ indexURL: PYODIDE });
  progress("正在加载 numpy / scipy / Clarabel…");
  await py.loadPackage(["numpy", "scipy", "clarabel"]);
  progress("正在加载仿真代码…");
  py.FS.mkdirTree("/home/pyodide/sim");
  const files = await Promise.all(SIM_FILES.map(async (f) => {
    const r = await fetch("py/" + f, { cache: "no-cache" });
    if (!r.ok) throw new Error(`py/${f}: HTTP ${r.status}`);
    return [f, await r.text()];
  }));
  for (const [f, src] of files) py.FS.writeFile("/home/pyodide/sim/" + f, src);
  const glueSrc = await (await fetch("sim_worker.py", { cache: "no-cache" })).text();
  const ns = py.globals.get("dict")();
  py.runPython(glueSrc, { globals: ns });
  glue = { advance: ns.get("advance"), state: ns.get("state"), input: ns.get("handle_input"), cfd: ns.get("handle_cfd") };
  progress("");
  let last = performance.now();
  let lastPost = 0;
  const tick = () => {
    try {
      while (queue.length) {
        const m = queue.shift();
        if (m.type === "input") glue.input(JSON.stringify({ keys: m.keys, actions: m.actions }));
        else if (m.type === "cfd") glue.cfd(JSON.stringify({ c_world: m.c_world }));
      }
      const now = performance.now();
      glue.advance((now - last) / 1000);
      last = now;
      if (now - lastPost >= 30) {
        lastPost = now;
        self.postMessage({ type: "state", json: glue.state() });
      }
    } catch (e) {
      self.postMessage({ type: "error", text: String(e && e.message || e) });
      return;
    }
    setTimeout(tick, 2);
  };
  tick();
}

main().catch((e) => self.postMessage({ type: "error", text: String(e && e.message || e) }));
