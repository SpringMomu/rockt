// Front-end main loop: state polling + interpolation, keyboard (KSP layout),
// camera, particle effects, the GPU flow solver and the HUD.
// The flow solver runs whenever a state is available, whether or not the
// airflow is shown in the main view: the wind-tunnel window and the CFD
// force coefficients are always live.
import { V, rot, slerp } from "./gl.js";
import { Scene } from "./scene.js";
import { HUD } from "./hud.js";
import { Fluid3D } from "./fluid.js";
import { plumeParams } from "./plume.js";

const canvas = document.getElementById("view");
let scene;
try { scene = new Scene(canvas); } catch (e) {
  const el = document.getElementById("err");
  el.style.display = "flex"; el.textContent = "WebGL2 初始化失败：" + e.message + "（请使用支持 WebGL2 的 Chrome 或 Edge）";
  throw e;
}
const hud = new HUD();
// 3-D GPU flow solver, world-aligned grid.  ?cfd=low for weak GPUs /
// software WebGL, ?cfd=high for a finer grid.
const cfdQ = (location.search.match(/cfd=(low|high)/) || [])[1];
let GRID = cfdQ === "low" ? { nx: 40, ny: 40, nz: 60, cell: 1.6 } : cfdQ === "high" ? { nx: 96, ny: 96, nz: 128, cell: 0.72 } : { nx: 64, ny: 64, nz: 96, cell: 1.0 };
// Custom grid: ?grid=48x48x72&cell=1.3
const gq = location.search.match(/grid=(\d+)x(\d+)x(\d+)/), cq = location.search.match(/cell=([\d.]+)/);
if (gq) GRID = { nx: +gq[1], ny: +gq[2], nz: +gq[3], cell: cq ? +cq[1] : GRID.cell };
let fluid = null;
try {
  fluid = new Fluid3D(scene.gl, GRID);
  if (!fluid.ok) { hud.tunnel.setError("3D CFD 不可用：" + (fluid.reason || "显卡不支持浮点渲染目标")); fluid = null; }
} catch (e) { console.error(e); hud.tunnel.setError("3D CFD 初始化失败"); fluid = null; }
const GRID_TXT = `${GRID.nx}×${GRID.ny}×${GRID.nz} · ${GRID.cell} m`;

// ------------------------------------------------------------------ state
const buf = [];                 // [{t: arrival ms, st}]
let latest = null;
let online = true;
const CLEAN = /[?&]clean/.test(location.search);
const view = { flowMode: CLEAN ? 0 : 1, volField: 0, cam: 0, cfdCoef: null, cfdSps: 0, flowTime: 0 };
const CAM_NAMES = ["环绕", "着陆台", "侧视"];
const VOL_NAMES = ["染料", "涡量", "速度"];
const VOL_TO_TUNNEL = ["dye", "vort", "speed"];

function receive(st) {
  buf.push({ t: performance.now(), st });
  while (buf.length > 12) buf.shift();
  latest = st; online = true;
}

// Where the simulation runs: the Python server (python main3d.py serves
// /api) or, for the static web build, this browser (Pyodide in a worker).
// ?local forces the in-browser simulator.
const IN_BROWSER = /[?&]local/.test(location.search) || !(await fetch("/api/scenarios", { cache: "no-store" })
  .then((r) => r.ok && (r.headers.get("Content-Type") || "").includes("json")).catch(() => false));
let simWorker = null;
if (IN_BROWSER) {
  const note = document.createElement("div");
  note.id = "simLoading";
  note.style.cssText = "position:fixed;left:50%;top:42%;transform:translateX(-50%);z-index:50;padding:14px 22px;background:rgba(4,7,12,.88);color:#d8e6f2;border:1px solid rgba(120,170,210,.4);border-radius:6px;font:14px/1.5 sans-serif;text-align:center;pointer-events:none";
  note.textContent = "正在启动本地仿真…";
  document.body.appendChild(note);
  simWorker = new Worker("sim_worker.js");
  simWorker.onmessage = (ev) => {
    const m = ev.data;
    if (m.type === "state") receive(JSON.parse(m.json));
    else if (m.type === "progress") { note.hidden = !m.text; note.textContent = m.text + "（首次加载约 30 MB，之后走浏览器缓存）"; }
    else if (m.type === "error") { online = false; note.hidden = false; note.style.color = "#ff5a4e"; note.textContent = "本地仿真出错：" + m.text; }
  };
} else {
  const poll = async () => {
    try {
      const r = await fetch("/api/state", { cache: "no-store" });
      receive(await r.json());
    } catch (e) { online = false; }
    setTimeout(poll, 30);
  };
  poll();
}

// Interpolated state ~70 ms behind the newest sample (smooth at any poll jitter).
function interpState(now) {
  if (buf.length === 0) return null;
  const tr = now - 70;
  let a = buf[0], b = buf[buf.length - 1];
  for (let i = 0; i < buf.length - 1; i++) {
    if (buf[i].t <= tr && buf[i + 1].t >= tr) { a = buf[i]; b = buf[i + 1]; break; }
  }
  if (tr >= b.t || a === b) return b.st;
  const f = Math.max(0, Math.min(1, (tr - a.t) / Math.max(1, b.t - a.t)));
  const A = a.st, B = b.st;
  if (A.scenario !== B.scenario || V.len(V.sub(A.pos, B.pos)) > 400) return B;
  return Object.assign({}, B, {
    pos: V.lerp(A.pos, B.pos, f), vel: V.lerp(A.vel, B.vel, f), q: slerp(A.q, B.q, f),
    throttle: A.throttle + (B.throttle - A.throttle) * f, legs: A.legs + (B.legs - A.legs) * f,
    gimbal: [A.gimbal[0] + (B.gimbal[0] - A.gimbal[0]) * f, A.gimbal[1] + (B.gimbal[1] - A.gimbal[1]) * f, 0],
    fins: (A.fins && B.fins) ? { deploy: A.fins.deploy + (B.fins.deploy - A.fins.deploy) * f,
      defl: B.fins.defl.map((d, k) => A.fins.defl[k] + (d - A.fins.defl[k]) * f) } : B.fins,
    alt: A.alt + (B.alt - A.alt) * f, t: A.t + (B.t - A.t) * f,
  });
}

// ------------------------------------------------------------------ input
const held = {};
const KEYMAP = { w: "w", a: "a", s: "s", d: "d", q: "q", e: "e", shift: "shift", control: "ctrl", arrowup: "shift", arrowdown: "ctrl" };
let pending = [];
let dirty = false;
function send() {
  const keys = {};
  for (const k of Object.values(KEYMAP)) keys[k] = false;
  for (const [k, v] of Object.entries(held)) if (v && KEYMAP[k]) keys[KEYMAP[k]] = true;
  const actions = pending; pending = []; dirty = false;
  if (simWorker) simWorker.postMessage({ type: "input", keys, actions });
  else fetch("/api/input", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ keys, actions }) }).catch(() => {});
}
setInterval(() => { if (dirty || Object.values(held).some(Boolean)) send(); }, 100);
function act(a) { pending.push(a); send(); }

window.addEventListener("keydown", (ev) => {
  const k = ev.key.toLowerCase();
  if (ev.ctrlKey && ["s", "d", "w", "q", "e", "a"].includes(k)) ev.preventDefault();
  if (KEYMAP[k]) { if (!held[k]) { held[k] = true; dirty = true; send(); } ev.preventDefault(); return; }
  if (ev.repeat) return;
  if (k >= "1" && k <= "9") act("scenario:" + (Number(k) - 1));
  else if (k === "0") act("scenario:9");
  else if (k === "m") act("autopilot");
  else if (k === "t") act("sas");
  else if (k === "g") act("legs");
  else if (k === "z") act("full");
  else if (k === "x") act("cut");
  else if (k === "p") act("pause");
  else if (k === "backspace") { act("pad"); ev.preventDefault(); }
  else if (k === ",") act("warp-");
  else if (k === ".") act("warp+");
  else if (k === "c") act("cfd_forces");
  else if (k === "y") act("wind");
  else if (k === "f") { view.flowMode = 1 - view.flowMode; toast(view.flowMode ? "主视图气流：" + VOL_NAMES[view.volField] : "主视图气流已关闭，流场计算与风洞截面照常"); }
  else if (k === "b") { view.volField = (view.volField + 1) % 3; hud.tunnel.setMode(VOL_TO_TUNNEL[view.volField]); toast("显示量：" + VOL_NAMES[view.volField]); }
  else if (k === "v") { view.cam = (view.cam + 1) % 3; toast("相机：" + CAM_NAMES[view.cam]); }
  else if (k === "l") { hud.resetLayout(); toast("仪表布局与窗口大小已恢复默认"); }
  else if (k === "h") hud.toggleHelp();
});
window.addEventListener("keyup", (ev) => {
  const k = ev.key.toLowerCase();
  if (KEYMAP[k]) { held[k] = false; dirty = true; send(); }
  if (k === "control" || k === "shift") { held.control = held.shift = held[k] = false; }
});
window.addEventListener("blur", () => { for (const k in held) held[k] = false; dirty = true; send(); });

let toastEl = null, toastUntil = 0;
function toast(msg) {
  if (!toastEl) {
    toastEl = document.createElement("div");
    Object.assign(toastEl.style, { position: "fixed", left: "50%", top: "74px", transform: "translateX(-50%)", padding: "4px 14px", background: "rgba(9,14,22,0.85)", border: "1px solid rgba(140,210,255,0.5)", borderRadius: "4px", fontSize: "12px", color: "#46d6ff", fontFamily: "inherit" });
    document.body.appendChild(toastEl);
  }
  toastEl.textContent = msg; toastEl.style.display = "block"; toastUntil = performance.now() + 1800;
}

// ------------------------------------------------------------------ camera
const cam = { yaw: -2.25, pitch: 0.18, dist: 70, fov: 0.9, eye: [0, 0, 0], at: [0, 0, 0], smoothAt: null, auto: true };
let drag = null;
canvas.addEventListener("mousedown", (e) => { drag = { x: e.clientX, y: e.clientY }; cam.auto = false; });
window.addEventListener("mouseup", () => { drag = null; });
window.addEventListener("mousemove", (e) => {
  if (!drag) return;
  cam.yaw -= (e.clientX - drag.x) * 0.006;
  cam.pitch = Math.max(-0.2, Math.min(1.45, cam.pitch + (e.clientY - drag.y) * 0.005));
  drag = { x: e.clientX, y: e.clientY };
});
canvas.addEventListener("wheel", (e) => { cam.dist = Math.max(18, Math.min(6000, cam.dist * Math.exp(e.deltaY * 0.0012))); cam.auto = false; e.preventDefault(); }, { passive: false });
canvas.addEventListener("dblclick", () => { cam.auto = true; });

function updateCamera(st, dt) {
  const center = V.add(st.pos, rot(st.q, [0, 0, 8.5]));
  cam.smoothAt = center;
  const sp = V.len(st.vel);
  if (cam.auto) {
    // Auto distance only (never auto-rotation): wide when high and fast.
    const want = st.state !== "FLYING" ? 58 : Math.max(48, Math.min(260, 42 + 0.35 * sp + 0.012 * st.alt));
    cam.dist += (want - cam.dist) * (1 - Math.exp(-dt * 1.5));
  }
  let fov = 0.9, eye;
  if (view.cam === 1) {
    // Pad camera: fixed on the ground, zooms onto the booster.
    eye = [-150, -210, 3.5];
    const d = V.len(V.sub(center, eye));
    fov = Math.max(0.035, Math.min(0.9, 2 * Math.atan(38 / d)));
  } else if (view.cam === 2) {
    // Side view: fixed direction (looking north), follows the booster.
    eye = V.add(center, [0, -cam.dist * 1.2, cam.dist * 0.05]);
  } else {
    // Orbit: angles only change when the user drags the mouse.
    const cp = Math.cos(cam.pitch);
    eye = V.add(center, [Math.cos(cam.yaw) * cp * cam.dist, Math.sin(cam.yaw) * cp * cam.dist, Math.sin(cam.pitch) * cam.dist]);
  }
  eye[2] = Math.max(eye[2], 2.2);
  cam.eye = eye; cam.at = center; cam.fov = fov;
}
function angDiff(a, b) { let d = a - b; while (d > Math.PI) d -= 2 * Math.PI; while (d < -Math.PI) d += 2 * Math.PI; return d; }

// ------------------------------------------------------------------ effects
// Particle smoke / dust / RCS / ember system.  The engine jet and its wall
// jet act on every particle (entrainment along the jet, radial outflow along
// the pad, roll-up into a toroidal cloud at the wall-jet front), so the
// cloud responds to throttle, height and gimbal instead of being scripted.
const smoke = [];     // {p, v, age, life, size, kind, col, a0, seed, rot, rotV, heat, grow}
const MAX_SMOKE = 1400;
let lastSimT = null;
let plumeNow = null;
const rnd = (a = 1) => (Math.random() * 2 - 1) * a;

function thrustDirW(st) {
  const gx = st.gimbal[0] * Math.PI / 180, gy = st.gimbal[1] * Math.PI / 180;
  return rot(st.q, [-Math.sin(gy), Math.sin(gx) * Math.cos(gy), -Math.cos(gx) * Math.cos(gy)]); // exhaust direction
}

function addP(o) {
  if (smoke.length >= MAX_SMOKE) smoke.splice(0, smoke.length - MAX_SMOKE + 1);
  o.age = 0; o.seed = Math.random(); o.rot = Math.random() * 6.283; o.rotV = rnd(0.35);
  smoke.push(o);
}

function spawnEffects(st, dt, P) {
  const flying = st.state === "FLYING";
  if (P.on) {
    const ex = P.axis;
    const pAmb = P.pamb;
    // Perpendicular basis around the jet.
    const u1 = V.norm(V.cross(ex, Math.abs(ex[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0])), u2 = V.cross(ex, u1);
    // Pad interaction: dust/smoke clouds removed; only the hot embers remain.
    if (P.hit && P.blast > 0.01) {
      const hit = P.hit;
      // Hot embers / grit ricocheting off the pad while the flame touches it.
      if (P.heat > 0.05) {
        const ne = Math.floor(18 * P.heat * dt * 60 * Math.random());
        for (let i = 0; i < ne; i++) {
          const a = Math.random() * 6.283, spd = 25 + 55 * Math.random();
          addP({ p: [hit[0] + rnd(P.Rj), hit[1] + rnd(P.Rj), 0.3], v: [Math.cos(a) * spd, Math.sin(a) * spd, 4 + 16 * Math.random()],
            life: 0.4 + Math.random() * 0.9, size: 0.07 + 0.05 * Math.random(), kind: 4, col: [1, 1, 1], a0: Math.random(), heat: 0, grow: 0 });
        }
      }
    }
  }
  // RCS: cold nitrogen thrusters on the interstage.  Short, bright white
  // jets that leave the nozzle fast in a narrow cone, then expand and fade
  // within a fraction of a second (moisture condensing in the cold gas);
  // high up, in thin air, the plume spreads much wider.
  if (flying) {
    const r = st.rcs;
    const pAmb = Math.exp(-st.pos[2] / 8400);
    // The jets leave the four pods (45 deg azimuths, r = 1.58 m, z = 17.3 m,
    // above the CoM) OPPOSITE to the force they put on the booster:
    //   +pitch (body +x torque) swings the nose towards -y  -> exhaust +y
    //   +yaw   (body +y torque) swings the nose towards +x  -> exhaust -x
    //   +roll  (body +z torque) spins counter-clockwise     -> exhaust clockwise (tangential)
    // Pitch/yaw fire only the pods on the exhaust side; roll fires all four.
    const thrusters = [];            // [pod index, exhaust dir (body), magnitude]
    const pods = [0, 1, 2, 3].map((k) => { const a = Math.PI / 4 + k * Math.PI / 2; return [Math.cos(a), Math.sin(a), 0]; });
    const addAxis = (cmd, dirOf, weight, podFilter) => {
      if (Math.abs(cmd) <= 0.08) return;
      pods.forEach((u, k) => {
        const e = dirOf(u, Math.sign(cmd));
        if (!podFilter || V.dot(u, e) > 0) thrusters.push([k, e, Math.abs(cmd) * weight]);
      });
    };
    addAxis(r[0], (u, s) => [0, s, 0], 1.0, true);
    addAxis(r[1], (u, s) => [-s, 0, 0], 1.0, true);
    addAxis(r[2], (u, s) => [s * u[1], -s * u[0], 0], 0.45, false);
    const want = new Map();
    for (const [k, side, mag] of thrusters) {
      const key = k + ":" + side.map((x) => Math.round(x * 10)).join(",");
      want.set(key, { k, side, mag: Math.min(1, mag) });
    }
    for (const [key, w] of want) { if (!rcsState.has(key)) rcsState.set(key, { ...w, I: 0, seed: Math.random() }); rcsState.get(key).mag = w.mag; rcsState.get(key).on = true; }
    for (const [key, j] of rcsState) {
      const on = want.has(key);
      j.I += ((on ? 0.4 + 0.6 * j.mag : 0) - j.I) * (1 - Math.exp(-dt / (on ? 0.03 : 0.12)));
      if (!on && j.I < 0.01) rcsState.delete(key);
    }
    rcsSpread = 0.14 + 0.5 * (1 - pAmb);
  } else rcsState.clear();
}
const rcsState = new Map();
let rcsSpread = 0.14;
function rcsJets(st) {
  const out = [];
  if (st.state !== "FLYING") return out;
  for (const j of rcsState.values()) {
    const a = Math.PI / 4 + j.k * Math.PI / 2, u = [Math.cos(a), Math.sin(a)];
    const base = V.add(st.pos, rot(st.q, [u[0] * 1.58 + j.side[0] * 0.2, u[1] * 1.58 + j.side[1] * 0.2, 17.3]));
    const len = 2.5 + 3.5 * j.I;
    out.push({ base, dir: rot(st.q, j.side), len, r1: len * rcsSpread, I: j.I, seed: j.seed });
  }
  return out;
}

function stepEffects(st, dt, P) {
  const wind = st.wind || [0, 0, 0];
  const jet = P && P.on;
  const U0 = jet ? 60 + 160 * P.thr : 0;                     // jet speed scale for forcing [m/s]
  for (let i = smoke.length - 1; i >= 0; i--) {
    const s = smoke[i];
    s.age += dt;
    if (s.age > s.life) { smoke.splice(i, 1); continue; }
    if (s.kind === 4) {                                        // embers: ballistic with light drag
      s.v[2] -= 9.81 * dt;
      for (let c = 0; c < 3; c++) s.v[c] *= Math.exp(-0.6 * dt);
      s.p = V.add(s.p, V.mul(s.v, dt));
      if (s.p[2] < 0.05) { s.p[2] = 0.05; s.v[2] = Math.abs(s.v[2]) * 0.35; s.v[0] *= 0.7; s.v[1] *= 0.7; }
      continue;
    }
    // Relaxation towards the wind (bigger puffs respond more slowly).
    const drag = s.kind === 2 ? 5.0 : 1.1 / (1 + 0.08 * s.size);
    for (let c = 0; c < 3; c++) s.v[c] += (wind[c] - s.v[c]) * (1 - Math.exp(-drag * dt));
    // Buoyancy of warm gas, decaying as it mixes.
    if (s.kind !== 2) s.v[2] += 2.4 * s.heat * Math.exp(-s.age / 4) * dt;
    if (jet && s.kind !== 2) {
      // Entrainment into the free jet.
      const q = V.sub(s.p, P.exit), sa = V.dot(q, P.axis);
      if (sa > 0 && sa < Math.min(P.hdist, P.len * 2.5)) {
        const rr = V.len(V.sub(q, V.mul(P.axis, sa))), Rl = P.jetR(sa) * 2.2 + 1;
        if (rr < Rl) {
          const uj = U0 * Math.exp(-sa / (1.2 * P.len)) * (1 - rr / Rl);
          const va = V.dot(s.v, P.axis);
          s.v = V.add(s.v, V.mul(P.axis, (uj - va) * Math.min(1, 5 * dt)));
        }
      }
      // Radial wall jet along the pad: u ~ 1/rho, confined to a thin layer.
      if (P.hit && P.blast > 0.005) {
        const dx = s.p[0] - P.hit[0], dy = s.p[1] - P.hit[1], rho = Math.hypot(dx, dy) + 1e-3;
        const delta = 0.6 + 0.35 * P.Rj + 0.12 * rho;
        const reach = 8 + 90 * P.blast;
        if (rho < reach && s.p[2] < 4 * delta) {
          const ur = U0 * 0.8 * P.blast * Math.min(1, (P.Rj + 1) / rho) * Math.exp(-s.p[2] / (1.5 * delta));
          const vr = (s.v[0] * dx + s.v[1] * dy) / rho;
          const dv = (ur - vr) * Math.min(1, 3 * dt);
          s.v[0] += dv * dx / rho; s.v[1] += dv * dy / rho;
          // Roll-up: where the radial flow decelerates the gas lifts off (toroidal vortex).
          s.v[2] += Math.max(0, 1 - ur / 15) * 2.0 * P.blast * dt * Math.min(1, rho / (P.Rj * 3 + 2));
        }
      }
    }
    s.p = V.add(s.p, V.mul(s.v, dt));
    const floor = 0.25 * s.size;
    if (s.p[2] < floor) { s.p[2] = floor; s.v[2] = Math.abs(s.v[2]) * 0.2; }
    // Growth by turbulent entrainment: faster puffs grow faster.
    s.size += dt * (s.grow + 0.035 * V.len(s.v));
    s.rot += s.rotV * dt;
  }
}

// Build the instance buffer: back-to-front sorted, split around the plume.
const partData = new Float32Array(2000 * 12);
const occGrid = new Map();
let rocketAxis = null;
function buildParticles(eye, P) {
  const n = smoke.length;
  // Crowding (self-shadowing) estimate from a coarse spatial hash.
  occGrid.clear();
  const cell = 5;
  const key = (p) => ((Math.floor(p[0] / cell) * 73856093) ^ (Math.floor(p[1] / cell) * 19349663) ^ (Math.floor(p[2] / cell) * 83492791));
  for (const s of smoke) if (s.kind !== 4 && s.kind !== 2) { const k = key(s.p); occGrid.set(k, (occGrid.get(k) || 0) + s.a0 * Math.min(4, s.size)); }
  const order = new Array(n);
  for (let i = 0; i < n; i++) { const p = smoke[i].p; order[i] = [i, (p[0] - eye[0]) ** 2 + (p[1] - eye[1]) ** 2 + (p[2] - eye[2]) ** 2]; }
  order.sort((a, b) => b[1] - a[1]);
  let dPl = -1;
  if (P && P.on) { const c = V.add(P.exit, V.mul(P.axis, Math.min(P.len * 0.4, P.hdist * 0.6))); dPl = (c[0] - eye[0]) ** 2 + (c[1] - eye[1]) ** 2 + (c[2] - eye[2]) ** 2; }
  let nFar = 0, m = 0;
  for (const [i, d2] of order) {
    const s = smoke[i];
    const life = s.age / s.life;
    let a;
    if (s.kind === 2) a = s.a0 * Math.pow(1 - life, 1.6);
    else if (s.kind === 4) a = s.a0;
    else a = s.a0 * Math.min(1, s.age * 3) * Math.pow(1 - life, 1.3);
    if (s.kind === 1 && rocketAxis) {
      const q = V.sub(s.p, rocketAxis[0]), t = Math.max(0, Math.min(20, V.dot(q, rocketAxis[1])));
      const dd = V.len(V.sub(q, V.mul(rocketAxis[1], t)));
      a *= Math.min(1, Math.max(0.25, (dd - 1.0) / (s.size + 3)));
    }
    if (a < 0.003) continue;
    if (dPl > 0 && d2 > dPl) nFar = m + 1;
    const o = m * 12;
    partData[o] = s.p[0]; partData[o + 1] = s.p[1]; partData[o + 2] = s.p[2]; partData[o + 3] = s.size;
    partData[o + 4] = s.col[0]; partData[o + 5] = s.col[1]; partData[o + 6] = s.col[2]; partData[o + 7] = a;
    let emis = 0, open = 1;
    if (s.kind === 4) { emis = 6 * Math.pow(1 - life, 2); open = -1; }
    else if (s.kind !== 2) {
      emis = s.kind === 0 ? 0.25 * Math.exp(-s.age * 3) : 0;
      open = Math.exp(-0.05 * ((occGrid.get(key(s.p)) || 0) - s.a0 * Math.min(4, s.size)));
      open *= Math.min(1, 0.45 + s.p[2] / (s.size * 2.5 + 1));
    }
    partData[o + 8] = s.seed; partData[o + 9] = s.rot; partData[o + 10] = emis; partData[o + 11] = open;
    m++;
    if (m >= 2000) break;
  }
  if (dPl < 0) nFar = m;
  return { data: partData, n: m, nFar };
}

// ------------------------------------------------------------------ ribbons
const lineBuf = new Float32Array(60000 * 8);
let lineN = 0;
function ribbon(pts, width, color, eye, fade = false) {
  if (pts.length < 2) return;
  for (let i = 0; i < pts.length - 1; i++) {
    if (lineN + 6 > 60000) return;
    const a = pts[i], b = pts[i + 1];
    const d = V.sub(b, a);
    if (V.len(d) < 1e-4) continue;
    const toEye = V.sub(eye, a);
    let s = V.cross(d, toEye);
    const ls = V.len(s); if (ls < 1e-6) continue;
    const wa = typeof width === "function" ? width(a) : width;
    const wb = typeof width === "function" ? width(b) : width;
    const sa = V.mul(s, wa / ls), sb = V.mul(s, wb / ls);
    const ca = fade ? color.map((c, k) => (k === 3 ? c * (i / (pts.length - 1)) : c)) : color;
    const cb = fade ? color.map((c, k) => (k === 3 ? c * ((i + 1) / (pts.length - 1)) : c)) : color;
    const v = [[V.add(a, sa), ca], [V.sub(a, sa), ca], [V.add(b, sb), cb], [V.sub(a, sa), ca], [V.sub(b, sb), cb], [V.add(b, sb), cb]];
    for (const [p, c] of v) {
      const o = lineN * 8;
      lineBuf[o] = p[0]; lineBuf[o + 1] = p[1]; lineBuf[o + 2] = p[2];
      // Premultiplied alpha.
      lineBuf[o + 3] = c[0] * c[3]; lineBuf[o + 4] = c[1] * c[3]; lineBuf[o + 5] = c[2] * c[3]; lineBuf[o + 6] = c[3]; lineBuf[o + 7] = 1;
      lineN++;
    }
  }
}


// ------------------------------------------------------------------ frame
let lastFrame = performance.now();
let frameNo = 0;
function frame(now) {
  const dt = Math.min(0.05, (now - lastFrame) / 1000);
  lastFrame = now;
  const st = interpState(now);
  if (st) {
    const simDt = lastSimT === null ? 0 : Math.max(0, Math.min(0.1, st.t - lastSimT));
    if (lastSimT !== null && st.t < lastSimT - 0.5) { smoke.length = 0; cam.smoothAt = null; cam.auto = true; if (fluid) fluid.reset(); }
    lastSimT = st.t;
    scene.resize();
    updateCamera(st, dt);
    cam.dist = cam.dist;
    const stP = Object.assign({}, st, { wind: st.wind || [0, 0, 0] });
    plumeNow = plumeParams(stP, now / 1000);
    if (simDt > 0) { spawnEffects(stP, simDt, plumeNow); stepEffects(stP, simDt, plumeNow); }
    // 3-D flow: always computed; the main-view volume is optional (F).
    // The tunnel window's dye / vorticity / speed buttons also set the volume field.
    const tIdx = VOL_TO_TUNNEL.indexOf(hud.tunnel.mode);
    if (tIdx >= 0) view.volField = tIdx;
    frameNo++;
    if (fluid) {
      fluid.planeNormal = hud.tunnel.en;
      fluid.update(Object.assign({}, st, { wind: st.wind || [0, 0, 0] }), dt, latest && !latest.paused ? latest.warp : 0);
      view.cfdSps = fluid.sps;
      view.flowTime = fluid.flowTime;
      if (frameNo % 10 === 0) {
        const fc = fluid.readForce(st);
        if (fc) {
          view.cfdCoef = fc;
          if (simWorker) simWorker.postMessage({ type: "cfd", c_world: fc.world });
          else fetch("/api/cfd3d", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ c_world: fc.world, cn: fc.cn, ca: fc.ca, zcp: fc.zcp, t: st.t }) }).catch(() => {});
        }
      }
      if (frameNo % 3 === 1) {
        const box = fluid.box;
        const region = hud.tunnel.region(box);
        const sl = region && fluid.readSlice(region);
        if (sl) hud.tunnel.setSlice(sl, { pos: st.pos.slice(), q: st.q.slice(), legs: st.legs, box, grid: GRID_TXT });
      }
    }
    const fluidOn = !!fluid;

    lineN = 0;
    const eye = cam.eye;
    const camDist = (p) => V.len(V.sub(p, eye));
    const pxW = (p) => Math.max(0.05, camDist(p) * 0.0022);   // ~constant screen width
    // Actual trajectory (blue) and G-FOLD plan (green).
    if (!CLEAN && view.flowMode === 0 && st.trail && st.trail.length > 1) ribbon(st.trail.concat([st.pos]), pxW, [0.3, 0.65, 1.0, 0.75], eye);
    if (!CLEAN && view.flowMode === 0 && st.plan && st.plan.length > 1) ribbon(st.plan, (p) => pxW(p) * 1.2, [0.35, 1.0, 0.6, 0.85], eye);
    scene.time = now / 1000;
    const stR = Object.assign({}, st, { wind: st.wind || [0, 0, 0] });
    const fx = { rcs: rcsJets(st), lines: lineBuf.subarray(0, lineN * 8), particles: (rocketAxis = [V.add(st.pos, rot(st.q, [0, 0, -2])), rot(st.q, [0, 0, 1])], buildParticles(cam.eye, plumeNow)), plume: plumeNow };
    if (fluidOn && view.flowMode > 0) {
      fx.fluid = fluid; fx.fluidMode = view.volField; fx.fluidVolume = true;
    }
    scene.render({ eye: cam.eye, at: cam.at, fov: cam.fov, dist: camDist(st.pos) }, stR, fx, view);
    hud.update(latest ? Object.assign({}, latest, { pos: st.pos, vel: st.vel, q: st.q, alt: st.alt }) : st, view);
  }
  if (toastEl && now > toastUntil) toastEl.style.display = "none";
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
window.__app = { view, cam, get frameNo() { return frameNo; }, get fluid() { return fluid; }, get state() { return latest; } };
