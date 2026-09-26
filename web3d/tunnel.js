// Wind-tunnel window: a vertical cut-plane through the booster in FIXED
// world axes (horizontal = east or north, vertical = up), so the booster is
// drawn with its real attitude.  The GPU flow solver supplies the float
// cut-plane; this module colours it (colour maps and exposure from
// fluid/src/render.c), advects tracer particles in it and draws the booster,
// the ground, the free stream and the aerodynamic forces on top.
import { V, quatToMat3 } from "./gl.js";

const D2R = Math.PI / 180;
const FONT = '"JetBrains Mono", "Cascadia Mono", Consolas, "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", monospace';
const INFERNO = [[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164]];
const ICEFIRE = [[190, 240, 255], [60, 150, 230], [30, 60, 150], [8, 8, 14], [140, 30, 40], [230, 90, 40], [255, 225, 150]];

function lut(keys) {
  const out = new Uint8ClampedArray(1024 * 3), n = keys.length;
  for (let i = 0; i < 1024; i++) {
    const t = i / 1023 * (n - 1), k = Math.min(n - 2, Math.floor(t)), f = t - k;
    for (let c = 0; c < 3; c++) out[i * 3 + c] = keys[k][c] + (keys[k + 1][c] - keys[k][c]) * f + 0.5;
  }
  return out;
}
const L_INF = lut(INFERNO), L_ICE = lut(ICEFIRE);
const cssRamp = (keys) => `linear-gradient(90deg,${keys.map((c, i) => `rgb(${c.join(",")}) ${(100 * i / (keys.length - 1)).toFixed(1)}%`).join(",")})`;

export const TUNNEL_MODES = ["dye", "speed", "vort", "press", "schlieren"];
const MODE_NAMES = { dye: "染料", speed: "速度", vort: "涡量", press: "压力", schlieren: "纹影" };
const PLANES = { EU: { eh: [1, 0, 0], name: "东—天", h: "东" }, NU: { eh: [0, 1, 0], name: "北—天", h: "北" } };

function niceStep(x) {
  const p = Math.pow(10, Math.floor(Math.log10(x))), m = x / p;
  return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
}
function fmtN(x) {
  const a = Math.abs(x);
  return a >= 100 ? x.toFixed(0) : a >= 10 ? x.toFixed(1) : a >= 1 ? x.toFixed(2) : x.toFixed(2);
}
function kN(n) { return Math.abs(n) >= 1e5 ? (n / 1000).toFixed(0) + " kN" : (n / 1000).toFixed(1) + " kN"; }

export class TunnelView {
  constructor({ canvas, wrap, modeBar, planeBar, partBtn, forceBtn, foot, tag }) {
    this.cv = canvas; this.ctx = canvas.getContext("2d");
    this.wrap = wrap; this.foot = foot; this.tag = tag;
    this.mode = "vort"; this.plane = "EU"; this.particles = true; this.forces = true;
    this.zoom = 1;
    this.vis = null;                 // auto colour scales (smoothed, as in fluid.c)
    this.slice = null; this.img = null; this.off = document.createElement("canvas");
    this.parts = []; this.lastFlowT = null;
    this.error = null;
    try {
      const s = JSON.parse(localStorage.getItem("booster3d.tunnel.v3") || "{}");
      if (TUNNEL_MODES.includes(s.mode)) this.mode = s.mode;
      if (PLANES[s.plane]) this.plane = s.plane;
      if (typeof s.particles === "boolean") this.particles = s.particles;
      if (typeof s.forces === "boolean") this.forces = s.forces;
    } catch (e) { /* no storage */ }

    // Toolbar.
    this.modeBtns = {};
    for (const m of TUNNEL_MODES) {
      const b = document.createElement("button");
      b.textContent = MODE_NAMES[m]; b.dataset.m = m;
      b.addEventListener("click", () => this.setMode(m));
      modeBar.appendChild(b); this.modeBtns[m] = b;
    }
    this.planeBtns = {};
    for (const [k, p] of Object.entries(PLANES)) {
      const b = document.createElement("button");
      b.textContent = p.name;
      b.title = k === "EU" ? "截面含东向与天向，从南向北看" : "截面含北向与天向，从东向西看";
      b.addEventListener("click", () => { this.plane = k; this.parts = []; this._save(); this._sync(); });
      planeBar.appendChild(b); this.planeBtns[k] = b;
    }
    this.partBtn = partBtn; this.forceBtn = forceBtn;
    partBtn.addEventListener("click", () => { this.particles = !this.particles; this._save(); this._sync(); });
    forceBtn.addEventListener("click", () => { this.forces = !this.forces; this._save(); this._sync(); });
    for (const el of [modeBar, planeBar, partBtn, forceBtn]) el.addEventListener("mousedown", (e) => e.stopPropagation());

    // Zoom (wheel, towards the booster), double-click resets.
    canvas.addEventListener("wheel", (e) => {
      e.preventDefault(); e.stopPropagation();
      this.zoom = Math.max(0.6, Math.min(8, this.zoom * Math.exp(-e.deltaY * 0.0015)));
      this.parts = [];
    }, { passive: false });
    canvas.addEventListener("dblclick", (e) => { e.stopPropagation(); this.zoom = 1; this.parts = []; });
    canvas.addEventListener("mousedown", (e) => e.stopPropagation());

    // Canvas follows the (resizable) panel.
    this.dpr = Math.min(2, window.devicePixelRatio || 1);
    const fit = () => {
      const r = wrap.getBoundingClientRect();
      const w = Math.max(80, Math.round(r.width)), h = Math.max(80, Math.round(r.height));
      if (w !== this.cssW || h !== this.cssH) {
        this.cssW = w; this.cssH = h;
        canvas.width = Math.round(w * this.dpr); canvas.height = Math.round(h * this.dpr);
        canvas.style.width = w + "px"; canvas.style.height = h + "px";
        this.parts = [];
      }
    };
    new ResizeObserver(fit).observe(wrap);
    fit();
    this._sync();
  }

  setMode(m) { if (TUNNEL_MODES.includes(m)) { this.mode = m; this._save(); this._sync(); this.imgStale = true; } }
  _save() { try { localStorage.setItem("booster3d.tunnel.v3", JSON.stringify({ mode: this.mode, plane: this.plane, particles: this.particles, forces: this.forces })); } catch (e) { /* ignore */ } }
  _sync() {
    for (const [m, b] of Object.entries(this.modeBtns)) b.classList.toggle("on", m === this.mode);
    for (const [k, b] of Object.entries(this.planeBtns)) b.classList.toggle("on", k === this.plane);
    this.partBtn.classList.toggle("on", this.particles);
    this.forceBtn.classList.toggle("on", this.forces);
    if (this.tag) this.tag.textContent = `${PLANES[this.plane].name} · 世界坐标`;
  }

  get eh() { return PLANES[this.plane].eh; }
  get en() { return V.cross(this.eh, [0, 0, 1]); }

  // Region of the cut-plane to sample, in world metres.  box = fluid.box.
  region(box) {
    if (!box || !this.cssW) return null;
    const eh = this.eh;
    const Eh = Math.abs(V.dot(V.sub(box.max, box.min), eh)), Ez = box.max[2] - box.min[2];
    const mppFill = Math.min(Eh / this.cssW, Ez / this.cssH);
    const mpp = mppFill / this.zoom;
    const spanW = this.cssW * mpp, spanH = this.cssH * mpp;
    // Centre: between the booster and the middle of the grid, kept inside the grid.
    const off = V.sub(box.center, box.mid);
    const k = 0.45 / Math.max(1, this.zoom);
    let ch = V.dot(V.add(box.mid, V.mul(off, k)), eh), cz = box.mid[2] + off[2] * k;
    const hMin = V.dot(box.min, eh), hMax = V.dot(box.max, eh);
    const lo = Math.min(hMin, hMax), hi = Math.max(hMin, hMax);
    if (spanW <= hi - lo) ch = Math.max(lo + spanW / 2, Math.min(hi - spanW / 2, ch));
    if (spanH <= Ez) cz = Math.max(box.min[2] + spanH / 2, Math.min(box.max[2] - spanH / 2, cz));
    // The plane passes through the booster's mid-point.
    const en = this.en;
    const cn = V.dot(box.mid, en);
    const center = V.add(V.add(V.mul(eh, ch), V.mul(en, cn)), [0, 0, cz]);
    const long = Math.max(this.cssW, this.cssH), sc = Math.min(1, 300 / long);
    return { center, eh, en, spanW, spanH, mpp, sw: Math.max(32, Math.round(this.cssW * sc)), sh: Math.max(32, Math.round(this.cssH * sc)) };
  }

  setError(msg) { this.error = msg; }

  // New cut-plane from the solver + the state it was taken at.
  setSlice(slice, meta) {
    if (!slice) return;
    const prev = this.slice;
    this.slice = { ...slice, A: slice.A.slice(), B: slice.B.slice(), C: slice.C.slice(), meta };
    if (prev && (prev.region.mpp !== slice.region.mpp || prev.region.eh !== slice.region.eh)) this.parts = [];
    this._stats();
    this._buildImage();
  }

  // Smoothed colour scales, as fluid.c does for its diagnostic views.
  _stats() {
    const s = this.slice, { A, C, w, h } = s, U = s.uref, uinf = Math.max(0.5, V.len(s.uinfW || [0, 0, 0]));
    const q2 = s.inflow ? Math.max(1e-6, V.dot(s.inflow, s.inflow)) : 1;
    const kw = U / s.cell;                          // curl: cells/step per cell -> 1/s
    let w2 = 0, p2 = 0, n = 0;
    for (let id = 0; id < w * h; id++) {
      if (C[id * 4 + 1] > 0.5) continue;
      const o = C[id * 4] * kw; w2 += o * o;
      const cp = 2 * A[id * 4 + 3] / q2; p2 += cp * cp; n++;
    }
    n = Math.max(1, n);
    const t = { speed: 1.6 * uinf, vort: Math.max(3 * Math.sqrt(w2 / n), 0.05 * uinf / s.cell), press: Math.max(3 * Math.sqrt(p2 / n), 0.2) };
    if (!this.vis) this.vis = t;
    else for (const k of Object.keys(t)) this.vis[k] += 0.08 * (t[k] - this.vis[k]);
    this.vis.speed = t.speed;                       // fixed: fraction of the free stream
  }

  _buildImage() {
    const s = this.slice; if (!s) return;
    const { A, B, C, w, h } = s, U = s.uref, mode = this.mode, vis = this.vis;
    if (!this.img || this.img.width !== w || this.img.height !== h) {
      this.img = new ImageData(w, h); this.off.width = w; this.off.height = h;
    }
    const d = this.img.data;
    const q2 = s.inflow ? Math.max(1e-6, V.dot(s.inflow, s.inflow)) : 1;
    const kw = U / s.cell;
    for (let j = 0; j < h; j++) {
      const src = h - 1 - j;                   // GL rows are bottom-up
      for (let i = 0; i < w; i++) {
        const id = src * w + i, o = (j * w + i) * 4, flag = C[id * 4 + 1];
        let r, g, b;
        if (flag > 0.5) {
          if (flag < 1.5) { const hatch = ((i + j) % 7) === 0; r = hatch ? 30 : 16; g = hatch ? 34 : 19; b = hatch ? 42 : 25; }
          else if (flag < 2.5) { r = 205; g = 210; b = 216; }
          else { r = 44; g = 40; b = 36; }
        } else if (mode === "dye") {
          const ex = Math.max(0, B[id * 4 + 3]) * 0.55;
          r = 255 * (0.03 + 0.97 * (1 - Math.exp(-1.6 * (B[id * 4] + ex))));
          g = 255 * (0.03 + 0.97 * (1 - Math.exp(-1.6 * (B[id * 4 + 1] + ex))));
          b = 255 * (0.05 + 0.95 * (1 - Math.exp(-1.6 * (B[id * 4 + 2] + ex * 1.05))));
        } else if (mode === "speed") {
          const k = Math.min(1023, Math.round(Math.min(1, A[id * 4 + 2] * U / vis.speed) * 1023)) * 3;
          r = L_INF[k]; g = L_INF[k + 1]; b = L_INF[k + 2];
        } else if (mode === "vort") {
          const k = Math.round((0.5 + 0.5 * Math.tanh(C[id * 4] * kw / vis.vort)) * 1023) * 3;
          r = L_ICE[k]; g = L_ICE[k + 1]; b = L_ICE[k + 2];
        } else if (mode === "press") {
          const k = Math.round((0.5 + 0.5 * Math.tanh(2 * A[id * 4 + 3] / q2 / vis.press)) * 1023) * 3;
          r = L_ICE[k]; g = L_ICE[k + 1]; b = L_ICE[k + 2];
        } else {                                  // schlieren of the tracer density (per cell)
          const l = Math.exp(-3 * C[id * 4 + 2]);
          r = 255 * (0.93 * l + 0.02); g = 255 * (0.90 * l + 0.02); b = 255 * (0.82 * l + 0.03);
        }
        d[o] = r; d[o + 1] = g; d[o + 2] = b; d[o + 3] = 255;
      }
    }
    this.off.getContext("2d").putImageData(this.img, 0, 0);
    this.imgStale = false;
  }

  // --------------------------------------------------------------- particles
  _velAt(x, y) {                                // window metres -> m/s (in-plane) or null
    const s = this.slice, { w, h, A, C } = s, R = s.region;
    const fx = (x / R.spanW + 0.5) * w - 0.5, fy = (y / R.spanH + 0.5) * h - 0.5;
    if (fx < 0 || fy < 0 || fx > w - 1 || fy > h - 1) return null;
    const i = Math.floor(fx), j = Math.floor(fy), ax = fx - i, ay = fy - j;
    const i1 = Math.min(w - 1, i + 1), j1 = Math.min(h - 1, j + 1);
    const ids = [j * w + i, j * w + i1, j1 * w + i, j1 * w + i1];
    for (const k of ids) if (C[k * 4 + 1] > 0.5) return null;
    const wts = [(1 - ax) * (1 - ay), ax * (1 - ay), (1 - ax) * ay, ax * ay];
    let u = 0, v = 0;
    for (let k = 0; k < 4; k++) { u += A[ids[k] * 4] * wts[k]; v += A[ids[k] * 4 + 1] * wts[k]; }
    return [u * s.uref, v * s.uref];
  }

  _stepParticles(dtPhys) {
    const s = this.slice; if (!s) return;
    const R = s.region;
    const want = Math.round(Math.max(80, Math.min(500, this.cssW * this.cssH / 560)));
    const spawn = (p) => {
      p.x = (Math.random() - 0.5) * R.spanW; p.y = (Math.random() - 0.5) * R.spanH;
      p.px = p.x; p.py = p.y; p.age = 0; p.life = 1.5 + Math.random() * 2.5; return p;
    };
    while (this.parts.length < want) { const p = spawn({}); p.age = Math.random() * p.life; this.parts.push(p); }
    if (this.parts.length > want) this.parts.length = want;
    const dtp = Math.min(0.2, Math.max(0, dtPhys));
    // Particle life in physical time, scaled so a tracer crosses ~1/3 of the window.
    const uinf = Math.max(1, V.len(s.uinfW || [0, 0, 0]));
    const lifeScale = 0.35 * Math.max(R.spanW, R.spanH) / uinf;
    for (const p of this.parts) {
      p.px = p.x; p.py = p.y;
      if (dtp <= 0) continue;
      const v1 = this._velAt(p.x, p.y);
      if (!v1) { spawn(p); continue; }
      const v2 = this._velAt(p.x + 0.5 * dtp * v1[0], p.y + 0.5 * dtp * v1[1]);
      if (!v2) { spawn(p); continue; }
      p.x += dtp * v2[0]; p.y += dtp * v2[1];
      p.age += dtp / lifeScale;
      if (p.age > p.life || Math.abs(p.x) > R.spanW / 2 || Math.abs(p.y) > R.spanH / 2) spawn(p);
    }
  }

  // ------------------------------------------------------------------- draw
  draw(st, view, flowTime) {
    const ctx = this.ctx, dpr = this.dpr, W = this.cssW, H = this.cssH;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#05080c"; ctx.fillRect(0, 0, W, H);
    const s = this.slice;
    if (!s) {
      ctx.fillStyle = "#6b7885"; ctx.font = "13px " + FONT; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText(this.error || "流场初始化中…", W / 2, H / 2);
      return;
    }
    if (this.imgStale) this._buildImage();
    const R = s.region, mpp = R.mpp;
    // Field image (the region may lag the canvas size by a frame).
    ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
    const iw = R.spanW / mpp, ih = R.spanH / mpp;
    ctx.drawImage(this.off, W / 2 - iw / 2, H / 2 - ih / 2, iw, ih);
    const X = (x) => W / 2 + x / mpp, Y = (y) => H / 2 - y / mpp;
    const toWin = (w) => [V.dot(V.sub(w, R.center), R.eh), w[2] - R.center[2]];

    // Particles.
    if (this.particles) {
      if (this.lastFlowT !== null && flowTime >= this.lastFlowT) this._stepParticles(flowTime - this.lastFlowT);
      else this._stepParticles(0);
      const dark = this.mode === "schlieren";
      ctx.lineCap = "round";
      ctx.strokeStyle = dark ? "rgba(20,24,30,0.45)" : "rgba(255,255,255,0.32)";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      for (const p of this.parts) {
        const x1 = X(p.x), y1 = Y(p.y);
        let dx = X(p.px) - x1, dy = Y(p.py) - y1;
        const dl = Math.hypot(dx, dy);
        if (dl > 60) continue;                       // respawned
        if (dl > 7) { dx *= 7 / dl; dy *= 7 / dl; }  // short tail only
        if (dl < 0.8) { ctx.moveTo(x1 - 0.5, y1); ctx.lineTo(x1 + 0.5, y1); } else { ctx.moveTo(x1 + dx, y1 + dy); ctx.lineTo(x1, y1); }
      }
      ctx.stroke();
    }
    this.lastFlowT = flowTime;

    const m = s.meta || {};
    // Ground and pad.
    const gy = Y(-R.center[2]);
    if (gy < H) {
      const gg = ctx.createLinearGradient(0, gy, 0, H);
      gg.addColorStop(0, "#3a352e"); gg.addColorStop(1, "#1d1a16");
      ctx.fillStyle = gg; ctx.fillRect(0, gy, W, H - gy);
      const dPad = Math.abs(V.dot(V.sub([0, 0, 0], R.center), R.en));
      if (dPad < 26) {
        const c = V.dot(V.sub([0, 0, 0], R.center), R.eh), half = Math.sqrt(26 * 26 - dPad * dPad);
        ctx.fillStyle = "#7d7c77"; ctx.fillRect(X(c - half), gy - Math.max(1.5, 0.4 / mpp), (2 * half) / mpp, Math.max(1.5, 0.4 / mpp));
        if (dPad < 19) {
          const r19 = Math.sqrt(19 * 19 - dPad * dPad);
          ctx.fillStyle = "#f2c21b";
          for (const sx of [-1, 1]) ctx.fillRect(X(c + sx * r19) - Math.max(1, 0.5 / mpp), gy - Math.max(1.5, 0.4 / mpp), Math.max(2, 1 / mpp), Math.max(1.5, 0.4 / mpp));
        }
      }
      ctx.strokeStyle = "rgba(220,200,170,0.5)"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(W, gy); ctx.stroke();
    }

    // Computational domain border.
    if (m.box) {
      const a = toWin(m.box.min), b = toWin(m.box.max);
      const x0 = X(Math.min(a[0], b[0])), x1 = X(Math.max(a[0], b[0])), y0 = Y(Math.max(a[1], b[1])), y1 = Y(Math.min(a[1], b[1]));
      if (x0 > 1 || y0 > 1 || x1 < W - 1 || y1 < H - 1) {
        ctx.setLineDash([4, 4]); ctx.strokeStyle = "rgba(150,180,205,0.45)"; ctx.lineWidth = 1;
        ctx.strokeRect(x0, y0, x1 - x0, y1 - y0); ctx.setLineDash([]);
      }
    }

    // Booster (state at the time of the cut).
    if (m.pos && m.q) this._drawBooster(ctx, m, R, X, Y, toWin);

    // Forces.
    if (this.forces && m.pos && m.q) this._drawForces(ctx, st, view, m, R, X, Y, toWin);

    this._drawFreeStream(ctx, s, st, W, H);
    this._drawScale(ctx, R, W, H);
    this._drawLegend(ctx, s, W, H);
    ctx.fillStyle = "rgba(200,215,228,0.85)"; ctx.font = "11px " + FONT; ctx.textAlign = "right"; ctx.textBaseline = "top";
    ctx.fillText(`缩放 ${this.zoom.toFixed(1)}×`, W - 10, 8);
    if (this.foot) {
      const g = m.grid || "";
      this.foot.textContent = `网格 ${g} · ${(view.cfdSps || 0).toFixed(0)} 步/s · 流动时间 ${(flowTime || 0).toFixed(1)} s`;
    }
  }

  _drawBooster(ctx, m, R, X, Y, toWin) {
    const Rm = quatToMat3(m.q);
    const bw = (pb) => [m.pos[0] + Rm[0][0] * pb[0] + Rm[0][1] * pb[1] + Rm[0][2] * pb[2],
                        m.pos[1] + Rm[1][0] * pb[0] + Rm[1][1] * pb[1] + Rm[1][2] * pb[2],
                        m.pos[2] + Rm[2][0] * pb[0] + Rm[2][1] * pb[1] + Rm[2][2] * pb[2]];
    const P = (pb) => { const w = toWin(bw(pb)); return [X(w[0]), Y(w[1])]; };
    const axis = [Rm[0][2], Rm[1][2], Rm[2][2]];
    const a2 = [V.dot(axis, R.eh), axis[2]], an = Math.abs(V.dot(axis, R.en));
    const ang = Math.atan2(-a2[1], a2[0]);
    const scale = Math.hypot(a2[0], a2[1]) / R.mpp;
    const cyl = (z0, z1, r, cLight, cDark, stroke = true) => {
      const [x, y] = P([0, 0, z0]);
      const L = (z1 - z0) * scale, Rr = r / R.mpp, mn = Math.max(0.5, Rr * an);
      ctx.save(); ctx.translate(x, y); ctx.rotate(ang);
      const g = ctx.createLinearGradient(0, -Rr, 0, Rr);
      g.addColorStop(0, cDark); g.addColorStop(0.35, cLight); g.addColorStop(0.6, cLight); g.addColorStop(1, cDark);
      ctx.beginPath();
      ctx.moveTo(0, -Rr); ctx.lineTo(L, -Rr);
      ctx.ellipse(L, 0, mn, Rr, 0, -Math.PI / 2, Math.PI / 2);
      ctx.lineTo(0, Rr);
      ctx.ellipse(0, 0, mn, Rr, 0, Math.PI / 2, Math.PI * 1.5);
      ctx.closePath();
      ctx.fillStyle = g; ctx.fill();
      if (stroke) { ctx.strokeStyle = "rgba(0,0,0,0.65)"; ctx.lineWidth = 1; ctx.stroke(); }
      ctx.restore();
    };
    // Legs behind, then fins, body, interstage, engine section.
    const legs = m.legs || 0;
    const fr = 1.95 + (5.9 - 1.95) * legs, fz = 9.0 + (-1.9 - 9.0) * legs;
    ctx.lineCap = "round";
    for (let i = 0; i < 4; i++) {
      const a = Math.PI / 4 + i * Math.PI / 2, c = Math.cos(a), s = Math.sin(a);
      const p0 = P([1.55 * c, 1.55 * s, 3.2]), p1 = P([fr * c, fr * s, fz]);
      ctx.strokeStyle = "#1c1f23"; ctx.lineWidth = Math.max(2, 0.7 / R.mpp);
      ctx.beginPath(); ctx.moveTo(...p0); ctx.lineTo(...p1); ctx.stroke();
      ctx.strokeStyle = "#5b636c"; ctx.lineWidth = Math.max(1, 0.4 / R.mpp);
      ctx.beginPath(); ctx.moveTo(...p0); ctx.lineTo(...p1); ctx.stroke();
    }
    cyl(-1.7, 0.0, 0.95, "#6d737a", "#2a2e33");
    cyl(-0.1, 18.0, 1.83, "#f1f3f5", "#8d96a0");
    cyl(15.4, 18.0, 1.84, "#3b4047", "#15181b", false);
    cyl(0.0, 1.4, 1.84, "#6b6158", "#2d2824", false);         // soot band near the base
    for (let i = 0; i < 4; i++) {
      const a = Math.PI / 4 + i * Math.PI / 2, c = Math.cos(a), s = Math.sin(a);
      const pts = [[1.8, -0.55, 16.68], [3.1, -0.55, 16.68], [3.1, 0.55, 16.92], [1.8, 0.55, 16.92]]
        .map(([r, t, z]) => P([r * c - t * s, r * s + t * c, z]));
      const p0 = P([1.8 * c, 1.8 * s, 16.8]), p1 = P([3.1 * c, 3.1 * s, 16.8]);
      ctx.strokeStyle = "#16191c"; ctx.lineWidth = Math.max(2, 0.45 / R.mpp);
      ctx.beginPath(); ctx.moveTo(...p0); ctx.lineTo(...p1); ctx.stroke();
      ctx.fillStyle = "rgba(40,44,50,0.9)"; ctx.beginPath(); pts.forEach((p, k) => (k ? ctx.lineTo(...p) : ctx.moveTo(...p))); ctx.closePath(); ctx.fill();
    }
  }

  _drawForces(ctx, st, view, m, R, X, Y, toWin) {
    const aero = st.aero || {};
    const Rm = quatToMat3(m.q);
    const at = (z) => { const w = [m.pos[0] + Rm[0][2] * z, m.pos[1] + Rm[1][2] * z, m.pos[2] + Rm[2][2] * z]; const q = toWin(w); return [X(q[0]), Y(q[1])]; };
    const legend = [];
    const arrow = (x, y, F, col, dash, label) => {
      const f2 = [V.dot(F, R.eh), F[2]], mag = V.len(F);
      const fl = Math.hypot(f2[0], f2[1]);
      if (mag < 50 || fl < 1e-6) return;
      const L = Math.min(130, 16 + 30 * Math.log10(1 + mag / 2000)) * (fl / mag);
      const dx = f2[0] / fl, dy = -f2[1] / fl;
      const x1 = x + dx * L, y1 = y + dy * L;
      ctx.save();
      ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 2.4; ctx.setLineDash(dash || []);
      ctx.shadowColor = "rgba(0,0,0,0.8)"; ctx.shadowBlur = 3;
      ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x1 - dx * 6, y1 - dy * 6); ctx.stroke();
      ctx.setLineDash([]);
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x1 - dx * 11 - dy * 5, y1 - dy * 11 + dx * 5); ctx.lineTo(x1 - dx * 11 + dy * 5, y1 - dy * 11 - dx * 5); ctx.closePath(); ctx.fill();
      legend.push([col, dash, label]);
      ctx.restore();
    };
    const q = aero.q || 0;
    // Applied aerodynamic force (the one integrated by the 6-DoF model).
    if (aero.force && q > 5) {
      const [x, y] = at(aero.cp_z ?? 9);
      arrow(x, y, aero.force, "#ffb43c", null, `气动合力（作用于箭体）${kN(V.len(aero.force))}`);
    }
    // Live CFD surface-pressure resultant (C · q · A_ref).
    const fc = view.cfdCoef;
    if (fc && q > 5) {
      const F = V.mul(fc.world, q * Math.PI * 1.83 * 1.83);
      const [x, y] = at(fc.zcp ?? aero.cp_z ?? 9);
      arrow(x, y, F, "#46d6ff", [5, 4], `CFD 表面压力合力 ${kN(V.len(F))}`);
    }
    // Centre of mass and centre of pressure symbols.
    const [cx, cy] = at(aero.com_z ?? 6);
    ctx.save();
    ctx.lineWidth = 1.2; ctx.strokeStyle = "#000";
    for (let k = 0; k < 4; k++) { ctx.beginPath(); ctx.moveTo(cx, cy); ctx.arc(cx, cy, 7, k * Math.PI / 2, (k + 1) * Math.PI / 2); ctx.closePath(); ctx.fillStyle = k % 2 ? "#ffffff" : "#1f6fe0"; ctx.fill(); }
    ctx.beginPath(); ctx.arc(cx, cy, 7, 0, Math.PI * 2); ctx.stroke();
    if (q > 5 && aero.cp_z !== undefined) {
      const [px, py] = at(aero.cp_z);
      ctx.fillStyle = "#e0322a"; ctx.beginPath(); ctx.arc(px, py, 4.5, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.arc(px, py, 1.5, 0, Math.PI * 2); ctx.fill();
    }
    ctx.restore();
    // Legend (top right).
    const rows = legend.concat([["com"], ["cp"]]);
    ctx.save();
    ctx.font = "11px " + FONT; ctx.textBaseline = "middle"; ctx.textAlign = "left";
    const bw = Math.max(...legend.map((l) => ctx.measureText(l[2]).width), 90) + 34;
    const x0 = this.cssW - bw - 10, y0 = 26;
    ctx.fillStyle = "rgba(5,8,12,0.62)"; ctx.fillRect(x0, y0, bw, rows.length * 16 + 8);
    rows.forEach((r, i) => {
      const y = y0 + 12 + i * 16;
      if (r[0] === "com") {
        for (let k = 0; k < 4; k++) { ctx.beginPath(); ctx.moveTo(x0 + 14, y); ctx.arc(x0 + 14, y, 5, k * Math.PI / 2, (k + 1) * Math.PI / 2); ctx.closePath(); ctx.fillStyle = k % 2 ? "#fff" : "#1f6fe0"; ctx.fill(); }
        ctx.fillStyle = "#cfdbe6"; ctx.fillText("质心", x0 + 28, y);
      } else if (r[0] === "cp") {
        ctx.fillStyle = "#e0322a"; ctx.beginPath(); ctx.arc(x0 + 14, y, 4.5, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.arc(x0 + 14, y, 1.4, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = "#cfdbe6"; ctx.fillText("压心（工程模型）", x0 + 28, y);
      } else {
        ctx.strokeStyle = r[0]; ctx.lineWidth = 2.2; ctx.setLineDash(r[1] || []);
        ctx.beginPath(); ctx.moveTo(x0 + 6, y); ctx.lineTo(x0 + 22, y); ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = r[0]; ctx.fillText(r[2], x0 + 28, y);
      }
    });
    ctx.restore();
  }

  _drawFreeStream(ctx, s, st, W, H) {
    const u = s.uinfW || [0, 0, 0], R = s.region;
    const u2 = [V.dot(u, R.eh), u[2]], sp = V.len(u), sp2 = Math.hypot(u2[0], u2[1]);
    const x = 14, y = 14;
    ctx.save();
    ctx.fillStyle = "rgba(5,8,12,0.62)"; ctx.fillRect(x - 6, y - 6, 176, 50);
    ctx.strokeStyle = "rgba(140,190,225,0.35)"; ctx.strokeRect(x - 6, y - 6, 176, 50);
    const cx = x + 18, cy = y + 19;
    ctx.strokeStyle = "rgba(160,190,210,0.35)"; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(cx, cy, 16, 0, Math.PI * 2); ctx.stroke();
    if (sp > 0.3) {
      const k = sp2 / sp, dx = sp2 > 1e-6 ? u2[0] / sp2 : 0, dy = sp2 > 1e-6 ? -u2[1] / sp2 : 0;
      const L = 14 * Math.max(0.15, k);
      ctx.strokeStyle = "#dff4ff"; ctx.fillStyle = "#dff4ff"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(cx - dx * L, cy - dy * L); ctx.lineTo(cx + dx * (L - 4), cy + dy * (L - 4)); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(cx + dx * L, cy + dy * L); ctx.lineTo(cx + dx * (L - 7) - dy * 4, cy + dy * (L - 7) + dx * 4); ctx.lineTo(cx + dx * (L - 7) + dy * 4, cy + dy * (L - 7) - dx * 4); ctx.closePath(); ctx.fill();
      if (k < 0.5) { ctx.beginPath(); ctx.arc(cx, cy, 3, 0, Math.PI * 2); ctx.stroke(); }
    }
    ctx.fillStyle = "#dff4ff"; ctx.font = "12px " + FONT; ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
    ctx.fillText(`来流 ${sp.toFixed(1)} m/s`, x + 42, y + 13);
    const a = st.aero || {};
    ctx.fillStyle = "#9fb4c6"; ctx.font = "11px " + FONT;
    ctx.fillText(`Ma ${(a.mach || 0).toFixed(2)}  α ${(a.alpha || 0).toFixed(1)}°`, x + 42, y + 30);
    ctx.restore();
  }

  _drawScale(ctx, R, W, H) {
    const len = niceStep(90 * R.mpp), px = len / R.mpp;
    const x = 14, y = H - 16;
    ctx.save();
    ctx.strokeStyle = "#e6eef5"; ctx.fillStyle = "#e6eef5"; ctx.lineWidth = 1.5;
    ctx.shadowColor = "rgba(0,0,0,0.9)"; ctx.shadowBlur = 2;
    ctx.beginPath(); ctx.moveTo(x, y - 4); ctx.lineTo(x, y); ctx.lineTo(x + px, y); ctx.lineTo(x + px, y - 4); ctx.stroke();
    ctx.font = "11px " + FONT; ctx.textAlign = "center"; ctx.textBaseline = "bottom";
    ctx.fillText(`${len} m`, x + px / 2, y - 3);
    // Axes glyph.
    const ax = x + px + 22, ay = y;
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(ax + 20, ay); ctx.moveTo(ax, ay); ctx.lineTo(ax, ay - 20); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(ax + 22, ay); ctx.lineTo(ax + 16, ay - 3); ctx.lineTo(ax + 16, ay + 3); ctx.closePath(); ctx.fill();
    ctx.beginPath(); ctx.moveTo(ax, ay - 22); ctx.lineTo(ax - 3, ay - 16); ctx.lineTo(ax + 3, ay - 16); ctx.closePath(); ctx.fill();
    ctx.textAlign = "left"; ctx.textBaseline = "middle";
    ctx.fillText(PLANES[this.plane].h, ax + 25, ay);
    ctx.fillText("天", ax + 5, ay - 21);
    ctx.restore();
  }

  _drawLegend(ctx, s, W, H) {
    const bw = Math.min(190, W * 0.42), x = W - bw - 12, y = H - 34;
    ctx.save();
    ctx.fillStyle = "rgba(5,8,12,0.62)"; ctx.fillRect(x - 6, y - 16, bw + 12, 44);
    ctx.font = "10.5px " + FONT; ctx.textBaseline = "alphabetic";
    const bar = (keys) => {
      const g = ctx.createLinearGradient(x, 0, x + bw, 0);
      keys.forEach((c, i) => g.addColorStop(i / (keys.length - 1), `rgb(${c.join(",")})`));
      ctx.fillStyle = g; ctx.fillRect(x, y, bw, 7);
      ctx.strokeStyle = "rgba(255,255,255,0.25)"; ctx.strokeRect(x + 0.5, y + 0.5, bw - 1, 6);
    };
    const ticks = (labels) => {
      ctx.fillStyle = "#b9c8d6";
      labels.forEach(([t, l]) => { ctx.textAlign = t < 0.05 ? "left" : t > 0.95 ? "right" : "center"; ctx.fillText(l, x + t * bw, y + 20); });
    };
    const title = (t) => { ctx.fillStyle = "#e6eef5"; ctx.textAlign = "left"; ctx.fillText(t, x, y - 5); };
    const v = this.vis;
    if (this.mode === "dye") {
      title("示踪流管（颜色 = 来流中的位置）");
      const hue = (h) => { h -= Math.floor(h); const q = h * 6; return [Math.abs(q - 3) - 1, 2 - Math.abs(q - 2), 2 - Math.abs(q - 4)].map((c) => Math.round(255 * Math.min(1, Math.max(0, c)))); };
      bar([0, 0.2, 0.4, 0.6, 0.8, 1].map((t) => hue(0.7 * (1 - t))));
      const period = 14 * s.cell / Math.max(0.5, V.len(s.uinfW || [0, 0, 0]));
      ticks([[0, "灰白 = 燃气"], [1, `时间线 ${period.toFixed(2)} s`]]);
    } else if (this.mode === "speed") {
      title("速度大小（相对箭体）m/s");
      bar(INFERNO);
      ticks([[0, "0"], [1 / 1.6, `来流 ${fmtN(v.speed / 1.6)}`], [1, fmtN(v.speed)]]);
    } else if (this.mode === "vort") {
      title("截面法向涡量 1/s");
      bar(ICEFIRE);
      ticks([[0, `−${fmtN(v.vort)} 顺`], [0.5, "0"], [1, `+${fmtN(v.vort)} 逆`]]);
    } else if (this.mode === "press") {
      title("压力系数 Cp");
      bar(ICEFIRE);
      ticks([[0, `−${fmtN(v.press)}`], [0.5, "0"], [1, `+${fmtN(v.press)}`]]);
    } else {
      title("纹影（示踪剂浓度梯度）");
      bar([[250, 245, 225], [120, 118, 110], [8, 8, 10]]);
      ticks([[0, "均匀"], [1, "强梯度"]]);
    }
    ctx.restore();
  }

  static rampCss() { return { inferno: cssRamp(INFERNO), icefire: cssRamp(ICEFIRE) }; }
}
