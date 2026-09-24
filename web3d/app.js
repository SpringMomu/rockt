// Front-end main loop: state polling + interpolation, keyboard (KSP layout),
// camera, particle effects, the GPU flow solver and the HUD.
// The flow solver runs whenever a state is available, whether or not the
// airflow is shown in the main view: the wind-tunnel window and the CFD
// force coefficients are always live.
import { V, rot, slerp } from "./gl.js";
import { Scene } from "./scene.js";
import { HUD } from "./hud.js";
import { Fluid3D } from "./fluid.js";

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
const view = { flowMode: 1, volField: 0, cam: 0, cfdCoef: null, cfdSps: 0, flowTime: 0 };
const CAM_NAMES = ["环绕", "着陆台", "侧视"];
const VOL_NAMES = ["染料", "涡量", "速度"];
const VOL_TO_TUNNEL = ["dye", "vort", "speed"];

async function poll() {
  try {
    const r = await fetch("/api/state", { cache: "no-store" });
    const st = await r.json();
    buf.push({ t: performance.now(), st });
    while (buf.length > 12) buf.shift();
    latest = st; online = true;
  } catch (e) { online = false; }
  setTimeout(poll, 30);
}
poll();

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
  fetch("/api/input", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ keys, actions }) }).catch(() => {});
}
setInterval(() => { if (dirty || Object.values(held).some(Boolean)) send(); }, 100);
function act(a) { pending.push(a); send(); }

window.addEventListener("keydown", (ev) => {
  const k = ev.key.toLowerCase();
  if (ev.ctrlKey && ["s", "d", "w", "q", "e", "a"].includes(k)) ev.preventDefault();
  if (KEYMAP[k]) { if (!held[k]) { held[k] = true; dirty = true; send(); } ev.preventDefault(); return; }
  if (ev.repeat) return;
  if (k >= "1" && k <= "6") act("scenario:" + (Number(k) - 1));
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
const smoke = [];     // {p, v, age, life, size, kind}
let lastSimT = null;
const rnd = (a = 1) => (Math.random() * 2 - 1) * a;

function thrustDirW(st) {
  const gx = st.gimbal[0] * Math.PI / 180, gy = st.gimbal[1] * Math.PI / 180;
  return rot(st.q, [-Math.sin(gy), Math.sin(gx) * Math.cos(gy), -Math.cos(gx) * Math.cos(gy)]); // exhaust direction
}

function spawnEffects(st, dt) {
  const exitW = V.add(st.pos, rot(st.q, [0, 0, -1.6]));
  const flying = st.state === "FLYING";
  if (flying && st.throttle > 0.02) {
    const ex = thrustDirW(st);
    const pAmb = Math.exp(-st.pos[2] / 8400);
    // Exhaust smoke (dense at low altitude, thin high up).
    const n = Math.floor((4 + 20 * st.throttle) * pAmb * dt * 60);
    for (let i = 0; i < n; i++) {
      const p = V.add(exitW, V.mul(ex, 14 + 18 * st.throttle + Math.random() * 12));
      smoke.push({ p: V.add(p, [rnd(1.5), rnd(1.5), rnd(1.5)]), v: V.add(V.add(st.vel, V.mul(ex, 18 + 20 * Math.random())), [rnd(4), rnd(4), rnd(3)]), age: 0, life: 4 + Math.random() * 4, size: 3 + Math.random() * 3, kind: 0 });
    }
    // Ground interaction: dust / steam ring spreading from the impingement point.
    const hGround = st.pos[2];
    if (hGround < 70 && ex[2] < -0.3) {
      const s = hGround / -ex[2];
      const hit = V.add(exitW, V.mul(ex, s));
      const strength = st.throttle * Math.max(0, 1 - hGround / 70);
      const m = Math.floor(40 * strength * dt * 60);
      for (let i = 0; i < m; i++) {
        const a = Math.random() * Math.PI * 2, spd = 18 + 30 * Math.random() * strength;
        smoke.push({ p: [hit[0] + Math.cos(a) * 3, hit[1] + Math.sin(a) * 3, 0.6 + Math.random()], v: [Math.cos(a) * spd, Math.sin(a) * spd, 1 + Math.random() * 4], age: 0, life: 3 + Math.random() * 3, size: 2.5 + Math.random() * 2.5, kind: 1 });
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
    for (const [k, side, mag] of thrusters) {
      const u = pods[k];
      const base = V.add(st.pos, rot(st.q, [u[0] * 1.58 + side[0] * 0.25, u[1] * 1.58 + side[1] * 0.25, 17.3]));
      const dir = rot(st.q, side);
      const n = Math.max(1, Math.round(mag * 70 * dt * 10));
      for (let i = 0; i < n; i++) {
        const spread = 0.12 + 0.35 * (1 - pAmb);
        const d = V.norm(V.add(dir, [rnd(spread), rnd(spread), rnd(spread)]));
        smoke.push({ p: V.add(base, V.mul(d, Math.random() * 1.2)), v: V.add(st.vel, V.mul(d, 45 + 35 * Math.random())),
          age: 0, life: 0.35 + Math.random() * 0.35, size: 0.35 + 0.25 * Math.random(), kind: 2, grow: 5 + 9 * (1 - pAmb) });
      }
    }
  }
  while (smoke.length > 4200) smoke.shift();
}

function stepEffects(st, dt) {
  const wind = st.wind || [0, 0, 0];
  for (let i = smoke.length - 1; i >= 0; i--) {
    const s = smoke[i];
    s.age += dt;
    if (s.age > s.life) { smoke.splice(i, 1); continue; }
    const drag = s.kind === 2 ? 5.0 : 1.3;
    for (let c = 0; c < 3; c++) s.v[c] += (wind[c] - s.v[c]) * (1 - Math.exp(-drag * dt));
    if (s.kind !== 2) s.v[2] += 1.2 * dt;              // warm exhaust rises
    s.p = V.add(s.p, V.mul(s.v, dt));
    if (s.p[2] < 0.3) { s.p[2] = 0.3; s.v[2] = Math.abs(s.v[2]) * 0.2; }
    s.size += dt * (s.kind === 1 ? 3.2 : s.kind === 2 ? s.grow : 2.2);
  }
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
    if (simDt > 0) { spawnEffects(st, simDt); stepEffects(st, simDt); }
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
          fetch("/api/cfd3d", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ c_world: fc.world, cn: fc.cn, ca: fc.ca, zcp: fc.zcp, t: st.t }) }).catch(() => {});
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
    if (view.flowMode === 0 && st.trail && st.trail.length > 1) ribbon(st.trail.concat([st.pos]), pxW, [0.3, 0.65, 1.0, 0.75], eye);
    if (view.flowMode === 0 && st.plan && st.plan.length > 1) ribbon(st.plan, (p) => pxW(p) * 1.2, [0.35, 1.0, 0.6, 0.85], eye);
    // Particles.
    const pts = [];
    for (const s of smoke) {
      const life = s.age / s.life;
      let r, g, b, a;
      if (s.kind === 0) { const hot = Math.max(0, 1 - s.age * 1.2); r = 0.62 + 0.3 * hot; g = 0.6 + 0.12 * hot; b = 0.58; a = 0.32 * (1 - life); }
      else if (s.kind === 1) { r = 0.72; g = 0.66; b = 0.58; a = 0.38 * (1 - life) * Math.min(1, s.age * 4); }
      else { r = 0.97; g = 0.98; b = 1.0; a = 0.75 * Math.pow(1 - life, 1.6); }
      pts.push([s.p[0], s.p[1], s.p[2], r, g, b, a, s.size]);
    }
    const glow = [];
    if (st.state === "FLYING" && st.throttle > 0.02) {
      const exitW = V.add(st.pos, rot(st.q, [0, 0, -1.6]));
      const ex = thrustDirW(st);
      const fl = 0.85 + 0.15 * Math.random();
      glow.push([exitW[0], exitW[1], exitW[2], 1.0 * fl, 0.55 * fl, 0.2 * fl, 1, 9 + 9 * st.throttle]);
      const p2 = V.add(exitW, V.mul(ex, 5));
      glow.push([p2[0], p2[1], p2[2], 0.9 * fl, 0.35 * fl, 0.08 * fl, 1, 14 + 10 * st.throttle]);
      if (st.pos[2] < 60) {
        const s = Math.max(0.01, st.pos[2] / Math.max(0.3, -ex[2]));
        const hit = V.add(exitW, V.mul(ex, s));
        const k = (1 - st.pos[2] / 60) * st.throttle;
        glow.push([hit[0], hit[1], 0.5, 1.0 * k, 0.5 * k, 0.15 * k, k, 30 + 20 * k]);
      }
    }
    scene.time = now / 1000;
    const stR = Object.assign({}, st, { wind: st.wind || [0, 0, 0] });
    const fx = { lines: lineBuf.subarray(0, lineN * 8), points: pts, glow };
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
window.__app = { view, cam, get fluid() { return fluid; }, get state() { return latest; } };
