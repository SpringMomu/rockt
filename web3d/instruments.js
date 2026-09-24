// Realistic cockpit instruments drawn on 2-D canvases:
//   * FDAI  — Apollo-style flight director attitude indicator ("8-ball"):
//             texture-mapped, lit sphere behind glass, roll bezel, three
//             attitude-error needles, three rate scales, flag window.
//   * ND    — landing-zone navigation display on a multi-function display
//             bezel with working soft keys (range, track-up, overlays).
import { quatToMat3 } from "./gl.js";

const D2R = Math.PI / 180;

// ---------------------------------------------------------------- helpers
function screw(ctx, x, y, r = 7) {
  const g = ctx.createRadialGradient(x - r * 0.3, y - r * 0.3, 1, x, y, r);
  g.addColorStop(0, "#9aa1a8"); g.addColorStop(0.6, "#4d535a"); g.addColorStop(1, "#1d2024");
  ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = "rgba(0,0,0,0.7)"; ctx.lineWidth = 1.6;
  ctx.beginPath(); ctx.moveTo(x - r * 0.6, y - r * 0.2); ctx.lineTo(x + r * 0.6, y + r * 0.2);
  ctx.moveTo(x - r * 0.2, y + r * 0.6); ctx.lineTo(x + r * 0.2, y - r * 0.6); ctx.stroke();
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y); ctx.lineTo(x + w - r, y); ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r); ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h); ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r); ctx.quadraticCurveTo(x, y, x + r, y); ctx.closePath();
}

function faceplate(ctx, W, H, r = 18) {
  const g = ctx.createLinearGradient(0, 0, 0, H);
  g.addColorStop(0, "#30353b"); g.addColorStop(0.5, "#24282d"); g.addColorStop(1, "#1a1d21");
  roundRect(ctx, 1, 1, W - 2, H - 2, r); ctx.fillStyle = g; ctx.fill();
  // brushed-metal streaks
  ctx.save(); ctx.clip();
  ctx.globalAlpha = 0.05;
  for (let y = 0; y < H; y += 3) { ctx.fillStyle = (y / 3) % 2 ? "#ffffff" : "#000000"; ctx.fillRect(0, y, W, 1); }
  ctx.restore();
  ctx.strokeStyle = "#0b0c0e"; ctx.lineWidth = 3; roundRect(ctx, 1.5, 1.5, W - 3, H - 3, r); ctx.stroke();
  ctx.strokeStyle = "rgba(255,255,255,0.10)"; ctx.lineWidth = 1; roundRect(ctx, 4, 4, W - 8, H - 8, r - 3); ctx.stroke();
}

function engrave(ctx, text, x, y, size = 12, align = "center") {
  ctx.font = `600 ${size}px "DejaVu Sans", Arial, "Microsoft YaHei", "PingFang SC", sans-serif`;
  ctx.textAlign = align; ctx.textBaseline = "middle";
  ctx.fillStyle = "rgba(0,0,0,0.8)"; ctx.fillText(text, x + 0.8, y + 0.8);
  ctx.fillStyle = "#d9dde2"; ctx.fillText(text, x, y);
}

// ------------------------------------------------------------------- FDAI
export class FDAI {
  constructor(canvas) {
    this.cv = canvas;
    this.ctx = canvas.getContext("2d");
    this.W = canvas.width;                     // 480
    this.R = 142;                              // ball radius (px)
    this.n = this.R * 2;
    this.ball = document.createElement("canvas");
    this.ball.width = this.ball.height = this.n;
    this.bctx = this.ball.getContext("2d");
    this.img = this.bctx.createImageData(this.n, this.n);
    this._buildTexture();
    this._buildStatic();
    // Per-pixel sphere geometry (normals + lighting) precomputed once.
    const n = this.n, R = this.R;
    this.geo = new Float32Array(n * n * 5);
    const L = [-0.42, 0.55, 0.72], Ln = Math.hypot(...L);
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const px = (i + 0.5 - R) / R, py = (R - j - 0.5) / R, r2 = px * px + py * py, o = (j * n + i) * 5;
      if (r2 > 1) { this.geo[o + 3] = -1; continue; }
      const sz = Math.sqrt(1 - r2);
      const dif = Math.max(0, (px * L[0] + py * L[1] + sz * L[2]) / Ln);
      const h = [L[0] / Ln, L[1] / Ln, L[2] / Ln + 1], hn = Math.hypot(...h);
      const spec = Math.pow(Math.max(0, (px * h[0] + py * h[1] + sz * h[2]) / hn), 60);
      this.geo[o] = px; this.geo[o + 1] = py; this.geo[o + 2] = sz;
      this.geo[o + 3] = (0.30 + 0.78 * dif) * (0.55 + 0.45 * Math.pow(sz, 0.5));
      this.geo[o + 4] = 0.55 * spec;
    }
  }

  // Equirectangular ball texture, 0.5 deg per texel: white sky, black ground,
  // pitch lines every 10 deg, heading meridians every 30 deg, Apollo numerals.
  _buildTexture() {
    const TW = 720, TH = 360;
    const c = document.createElement("canvas"); c.width = TW; c.height = TH;
    const g = c.getContext("2d");
    const X = (az) => ((az + 180) / 360) * TW, Y = (el) => ((90 - el) / 180) * TH;
    g.fillStyle = "#e4e0d4"; g.fillRect(0, 0, TW, TH / 2);
    g.fillStyle = "#141414"; g.fillRect(0, TH / 2, TW, TH / 2);
    // pitch lines
    for (let el = -80; el <= 80; el += 10) {
      if (el === 0) continue;
      const major = el % 30 === 0;
      g.fillStyle = el > 0 ? "#1b1b1b" : "#ececec";
      g.fillRect(0, Y(el) - (major ? 1.1 : 0.5), TW, major ? 2.2 : 1.0);
    }
    // 5-deg pitch ticks on the four cardinal meridians
    for (let el = -85; el <= 85; el += 5) {
      if (el % 10 === 0) continue;
      g.fillStyle = el > 0 ? "#1b1b1b" : "#ececec";
      for (const az of [-180, -90, 0, 90]) g.fillRect(X(az) - 6, Y(el) - 0.5, 12, 1);
    }
    // heading meridians
    for (let az = -180; az < 180; az += 10) {
      const major = az % 30 === 0;
      for (const hemi of [1, -1]) {
        g.fillStyle = hemi > 0 ? "#1b1b1b" : "#ececec";
        if (major) g.fillRect(X(az) - 0.8, hemi > 0 ? Y(84) : Y(0), 1.6, hemi > 0 ? Y(0) - Y(84) : Y(-84) - Y(0));
        else g.fillRect(X(az) - 0.5, hemi > 0 ? Y(4) : Y(0), 1, Y(0) - Y(4));
      }
    }
    // horizon: bold divider
    g.fillStyle = "#7a7a72"; g.fillRect(0, TH / 2 - 1.5, TW, 3);
    // numerals (stretched by 1/cos(el) so they look right on the sphere)
    const text = (s, az, el, col, size) => {
      g.save(); g.translate(X(az), Y(el)); g.scale(1 / Math.cos(el * D2R), 1);
      g.font = `bold ${size}px "DejaVu Sans Mono", "Microsoft YaHei", "PingFang SC", monospace`; g.textAlign = "center"; g.textBaseline = "middle";
      g.fillStyle = col; g.fillText(s, 0, 0); g.restore();
    };
    const hdg = { 0: "北", 90: "东", 180: "南", "-180": "南", "-90": "西" };
    for (let az = -180; az < 180; az += 30) {
      const lab = hdg[az] || String(((az + 360) % 360) / 10).padStart(2, "0");
      text(lab, az, 8, "#151515", 15);
      text(lab, az, -8, "#efefef", 15);
    }
    for (const el of [-60, -30, 30, 60]) {
      const lab = el > 0 ? String(el / 10).padStart(2, "0") : String(36 + el / 10);
      for (const az of [-180, -90, 0, 90]) text(lab, az + 9, el, el > 0 ? "#151515" : "#efefef", 16);
    }
    this.tex = g.getImageData(0, 0, TW, TH).data;
    this.TW = TW; this.TH = TH;
  }

  // Faceplate, bezel and roll scale (static layer).
  _buildStatic() {
    const W = this.W, c = W / 2, R = this.R;
    const s = document.createElement("canvas"); s.width = s.height = W;
    const g = s.getContext("2d");
    faceplate(g, W, W, 26);
    for (const [x, y] of [[26, 26], [W - 26, 26], [26, W - 26], [W - 26, W - 26]]) screw(g, x, y, 9);
    // bezel ring
    const rg = g.createRadialGradient(c, c, R + 4, c, c, R + 56);
    rg.addColorStop(0, "#0a0b0c"); rg.addColorStop(0.75, "#15171a"); rg.addColorStop(1, "#3b4046");
    g.fillStyle = rg; g.beginPath(); g.arc(c, c, R + 56, 0, Math.PI * 2); g.fill();
    g.strokeStyle = "#5a6068"; g.lineWidth = 2; g.stroke();
    // roll scale (fixed on the bezel, 0 at top)
    for (let a = 0; a < 360; a += 5) {
      const t = (a - 90) * D2R, major = a % 30 === 0, mid = a % 10 === 0;
      const r0 = R + 8, r1 = R + (major ? 24 : mid ? 17 : 12);
      g.strokeStyle = "#e8e8e2"; g.lineWidth = major ? 2.6 : 1.2;
      g.beginPath(); g.moveTo(c + Math.cos(t) * r0, c + Math.sin(t) * r0); g.lineTo(c + Math.cos(t) * r1, c + Math.sin(t) * r1); g.stroke();
      if (major) {
        g.save(); g.translate(c + Math.cos(t) * (R + 38), c + Math.sin(t) * (R + 38)); g.rotate(t + Math.PI / 2);
        g.font = "bold 13px 'DejaVu Sans Mono', 'Microsoft YaHei', 'PingFang SC', monospace"; g.fillStyle = "#e8e8e2"; g.textAlign = "center"; g.textBaseline = "middle";
        g.fillText(String(a / 10).padStart(2, "0"), 0, 0); g.restore();
      }
    }
    // fixed roll index (top)
    g.fillStyle = "#ff9a1f"; g.beginPath(); g.moveTo(c, c - R - 6); g.lineTo(c - 9, c - R - 22); g.lineTo(c + 9, c - R - 22); g.closePath(); g.fill();
    // engraved legends
    engrave(g, "滚转速率", c, 14, 11);
    engrave(g, "俯仰", W - 16, c - 92, 10); engrave(g, "速率", W - 16, c - 79, 10);
    engrave(g, "偏航速率", c, W - 13, 11);
    engrave(g, "姿态", 26, c - 8, 12); engrave(g, "指引", 26, c + 8, 12);
    // rate-scale windows
    const win = (x, y, w, h) => { g.fillStyle = "#050606"; roundRect(g, x, y, w, h, 3); g.fill(); g.strokeStyle = "#50565e"; g.lineWidth = 1.2; g.stroke(); };
    win(c - 80, 24, 160, 18); win(W - 30, c - 70, 18, 140); win(c - 80, W - 42, 160, 18);
    for (let i = -4; i <= 4; i++) {
      const o = i * 18, big = i % 2 === 0;
      g.fillStyle = "#dcdcd4";
      g.fillRect(c + o - 0.7, 26, 1.4, big ? 8 : 5);
      g.fillRect(c + o - 0.7, W - 42 + (big ? 8 : 11), 1.4, big ? 8 : 5);
      g.fillRect(W - 30 + (big ? 2 : 2), c + o * 0.78 - 0.7, big ? 8 : 5, 1.4);
    }
    // flag window
    win(W - 98, W - 74, 58, 22);
    this.stat = s;
  }

  draw(st) {
    const ctx = this.ctx, W = this.W, c = W / 2, R = this.R, n = this.n;
    ctx.clearRect(0, 0, W, W);
    ctx.drawImage(this.stat, 0, 0);
    // ---- sphere
    const Q = quatToMat3(st.q);
    const bx = [Q[0][0], Q[1][0], Q[2][0]], by = [Q[0][1], Q[1][1], Q[2][1]], bz = [Q[0][2], Q[1][2], Q[2][2]];
    const d = this.img.data, geo = this.geo, tex = this.tex, TW = this.TW, TH = this.TH;
    for (let k = 0, o = 0, q = 0; k < n * n; k++, o += 4, q += 5) {
      const dif = geo[q + 3];
      if (dif < 0) { d[o + 3] = 0; continue; }
      const px = geo[q], py = geo[q + 1], sz = geo[q + 2], sp = geo[q + 4] * 255;
      const wx = bx[0] * px - by[0] * py + bz[0] * sz;
      const wy = bx[1] * px - by[1] * py + bz[1] * sz;
      const wz = bx[2] * px - by[2] * py + bz[2] * sz;
      const el = Math.asin(wz > 1 ? 1 : wz < -1 ? -1 : wz);
      const az = Math.atan2(wx, wy);
      // bilinear texture fetch
      const u = (az / Math.PI + 1) * 0.5 * TW - 0.5, v = (0.5 - el / Math.PI) * TH - 0.5;
      let x0 = Math.floor(u), y0 = Math.floor(v); const fx = u - x0, fy = v - y0;
      const x1 = (x0 + 1) % TW; x0 = (x0 + TW) % TW;
      const y1 = Math.min(TH - 1, y0 + 1); y0 = Math.max(0, y0);
      const i00 = (y0 * TW + x0) * 4, i10 = (y0 * TW + x1) * 4, i01 = (y1 * TW + x0) * 4, i11 = (y1 * TW + x1) * 4;
      const w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy), w01 = (1 - fx) * fy, w11 = fx * fy;
      let r = tex[i00] * w00 + tex[i10] * w10 + tex[i01] * w01 + tex[i11] * w11;
      let gg = tex[i00 + 1] * w00 + tex[i10 + 1] * w10 + tex[i01 + 1] * w01 + tex[i11 + 1] * w11;
      let b = tex[i00 + 2] * w00 + tex[i10 + 2] * w10 + tex[i01 + 2] * w01 + tex[i11 + 2] * w11;
      // pole markers
      const ael = Math.abs(el);
      if (ael > 1.50 && ael < 1.535) { const v2 = wz > 0 ? 25 : 235; r = gg = b = v2; }
      d[o] = r * dif + sp; d[o + 1] = gg * dif + sp; d[o + 2] = b * dif + sp * 0.97; d[o + 3] = 255;
    }
    this.bctx.putImageData(this.img, 0, 0);
    ctx.drawImage(this.ball, c - R, c - R);
    // inner shadow of the window (the ball sits recessed)
    const sh = ctx.createRadialGradient(c, c, R - 18, c, c, R + 2);
    sh.addColorStop(0, "rgba(0,0,0,0)"); sh.addColorStop(1, "rgba(0,0,0,0.65)");
    ctx.fillStyle = sh; ctx.beginPath(); ctx.arc(c, c, R + 2, 0, Math.PI * 2); ctx.fill();

    // ---- symbols painted over the ball (velocity, retrograde, pad, command)
    const toBall = (w) => {
      const q = [bx[0] * w[0] + bx[1] * w[1] + bx[2] * w[2], by[0] * w[0] + by[1] * w[1] + by[2] * w[2], bz[0] * w[0] + bz[1] * w[1] + bz[2] * w[2]];
      return { x: c + q[0] * R, y: c + q[1] * R, front: q[2] > 0.15 };
    };
    ctx.save(); ctx.beginPath(); ctx.arc(c, c, R, 0, Math.PI * 2); ctx.clip();
    const vv = st.vel, sp = Math.hypot(vv[0], vv[1], vv[2]);
    if (sp > 0.5) {
      const u = [vv[0] / sp, vv[1] / sp, vv[2] / sp];
      const pro = toBall(u), ret = toBall([-u[0], -u[1], -u[2]]);
      if (pro.front) this._vel(ctx, pro.x, pro.y, false);
      if (ret.front) this._vel(ctx, ret.x, ret.y, true);
    }
    const tp = [-st.pos[0], -st.pos[1], -st.alt], dp = Math.hypot(...tp);
    if (dp > 1) {
      const p = toBall([tp[0] / dp, tp[1] / dp, tp[2] / dp]);
      if (p.front) {
        ctx.strokeStyle = "#1ec8ff"; ctx.lineWidth = 3;
        ctx.beginPath(); ctx.moveTo(p.x, p.y - 9); ctx.lineTo(p.x + 9, p.y); ctx.lineTo(p.x, p.y + 9); ctx.lineTo(p.x - 9, p.y); ctx.closePath(); ctx.stroke();
      }
    }
    ctx.restore();

    // ---- fixed reference symbol (orange)
    ctx.save();
    ctx.shadowColor = "rgba(0,0,0,0.6)"; ctx.shadowBlur = 4; ctx.shadowOffsetY = 2;
    ctx.fillStyle = "#ff8f1a";
    ctx.fillRect(c - 62, c - 2.5, 40, 5); ctx.fillRect(c + 22, c - 2.5, 40, 5);
    ctx.fillRect(c - 2.5, c + 20, 5, 22);
    ctx.beginPath(); ctx.arc(c, c, 5, 0, Math.PI * 2); ctx.fill();
    ctx.restore();

    // ---- attitude-error needles (yellow): roll top, pitch right, yaw bottom
    const auto = st.cmd_axis && st.autopilot && st.state === "FLYING";
    let eY = 0, eP = 0;
    if (auto) {
      const w = st.cmd_axis;
      eY = bx[0] * w[0] + bx[1] * w[1] + bx[2] * w[2];
      eP = by[0] * w[0] + by[1] * w[1] + by[2] * w[2];
    }
    const om = st.omega || [0, 0, 0];
    const eR = -om[2] * 2.0;
    const needle = (val, side) => {
      const k = Math.max(-1, Math.min(1, val / 0.26)) * R * 0.6;
      ctx.save();
      ctx.shadowColor = "rgba(0,0,0,0.7)"; ctx.shadowBlur = 5; ctx.shadowOffsetX = 2; ctx.shadowOffsetY = 3;
      ctx.fillStyle = "#ffd21f"; ctx.strokeStyle = "#7a5a00"; ctx.lineWidth = 1;
      ctx.beginPath();
      if (side === "top") { ctx.moveTo(c + k - 4, c - R); ctx.lineTo(c + k + 4, c - R); ctx.lineTo(c + k + 1.5, c - R + 70); ctx.lineTo(c + k - 1.5, c - R + 70); }
      else if (side === "bottom") { ctx.moveTo(c + k - 4, c + R); ctx.lineTo(c + k + 4, c + R); ctx.lineTo(c + k + 1.5, c + R - 70); ctx.lineTo(c + k - 1.5, c + R - 70); }
      else { ctx.moveTo(c + R, c + k - 4); ctx.lineTo(c + R, c + k + 4); ctx.lineTo(c + R - 70, c + k + 1.5); ctx.lineTo(c + R - 70, c + k - 1.5); }
      ctx.closePath(); ctx.fill(); ctx.stroke(); ctx.restore();
    };
    needle(eR, "top"); needle(eP, "right"); needle(eY, "bottom");

    // ---- sky pointer (moving roll index)
    const rr = (st.roll || 0) * D2R - Math.PI / 2;
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.moveTo(c + Math.cos(rr) * (R + 2), c + Math.sin(rr) * (R + 2));
    ctx.lineTo(c + Math.cos(rr + 0.05) * (R + 13), c + Math.sin(rr + 0.05) * (R + 13));
    ctx.lineTo(c + Math.cos(rr - 0.05) * (R + 13), c + Math.sin(rr - 0.05) * (R + 13));
    ctx.closePath(); ctx.fill();

    // ---- glass: highlight + edge reflection
    const gl1 = ctx.createLinearGradient(c - R, c - R, c + R * 0.2, c + R * 0.3);
    gl1.addColorStop(0, "rgba(255,255,255,0.20)"); gl1.addColorStop(0.45, "rgba(255,255,255,0.04)"); gl1.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = gl1; ctx.beginPath(); ctx.ellipse(c - R * 0.25, c - R * 0.35, R * 0.78, R * 0.5, -0.6, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = "rgba(255,255,255,0.12)"; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(c, c, R + 1, Math.PI * 1.05, Math.PI * 1.6); ctx.stroke();

    // ---- rate pointers (deg/s, full scale 10)
    const ptr = (val, horiz, at) => {
      const v = Math.max(-1, Math.min(1, val / 10)) * 72;
      ctx.fillStyle = "#ffffff";
      ctx.beginPath();
      if (horiz) { ctx.moveTo(c + v, at); ctx.lineTo(c + v - 6, at + (at < c ? 12 : -12)); ctx.lineTo(c + v + 6, at + (at < c ? 12 : -12)); }
      else { ctx.moveTo(at, c + v * 0.78); ctx.lineTo(at + 11, c + v * 0.78 - 6); ctx.lineTo(at + 11, c + v * 0.78 + 6); }
      ctx.closePath(); ctx.fill();
    };
    ptr(om[2] / D2R, true, 40);
    ptr(-om[0] / D2R, false, W - 28);
    ptr(om[1] / D2R, true, W - 26);

    // ---- flag window: guidance steering valid / OFF (striped)
    const fx0 = W - 96, fy0 = W - 72;
    if (auto) {
      ctx.fillStyle = "#0c2b16"; ctx.fillRect(fx0, fy0, 54, 18);
      ctx.fillStyle = "#4dff8e"; ctx.font = "bold 12px 'DejaVu Sans Mono', 'Microsoft YaHei', 'PingFang SC', monospace"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText("制导", fx0 + 27, fy0 + 9.5);
    } else {
      ctx.save(); ctx.beginPath(); ctx.rect(fx0, fy0, 54, 18); ctx.clip();
      for (let i = -3; i < 10; i++) { ctx.fillStyle = i % 2 ? "#d8d8d8" : "#c8261e"; ctx.beginPath(); ctx.moveTo(fx0 + i * 8, fy0 + 18); ctx.lineTo(fx0 + i * 8 + 8, fy0 + 18); ctx.lineTo(fx0 + i * 8 + 26, fy0); ctx.lineTo(fx0 + i * 8 + 18, fy0); ctx.closePath(); ctx.fill(); }
      ctx.restore();
      ctx.fillStyle = "#111"; ctx.font = "bold 11px 'DejaVu Sans Mono', 'Microsoft YaHei', 'PingFang SC', monospace"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText("断开", fx0 + 27, fy0 + 9.5);
    }
    // ---- attitude digits (small data plate, lower left)
    ctx.fillStyle = "#050606"; roundRect(ctx, 44, W - 76, 78, 40, 3); ctx.fill();
    ctx.strokeStyle = "#50565e"; ctx.lineWidth = 1.2; ctx.stroke();
    ctx.font = "bold 12px 'DejaVu Sans Mono', 'Microsoft YaHei', 'PingFang SC', monospace"; ctx.textAlign = "left"; ctx.fillStyle = "#7dffb0";
    ctx.fillText(`航向 ${String(Math.round(st.heading) % 360).padStart(3, "0")}`, 50, W - 64);
    ctx.fillText(`俯仰 ${st.pitch.toFixed(1).padStart(5, " ")}`, 50, W - 48);
  }

  _vel(ctx, x, y, retro) {
    ctx.save();
    ctx.shadowColor = "rgba(0,0,0,0.55)"; ctx.shadowBlur = 3;
    ctx.strokeStyle = "#9cff3a"; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.arc(x, y, 9, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath();
    if (retro) { ctx.moveTo(x - 6, y - 6); ctx.lineTo(x + 6, y + 6); ctx.moveTo(x + 6, y - 6); ctx.lineTo(x - 6, y + 6); }
    else { ctx.moveTo(x - 9, y); ctx.lineTo(x - 18, y); ctx.moveTo(x + 9, y); ctx.lineTo(x + 18, y); ctx.moveTo(x, y - 9); ctx.lineTo(x, y - 17); }
    ctx.stroke(); ctx.restore();
  }
}

// ------------------------------------------------------------ Nav display
const RANGES = [12, 25, 50, 100, 250, 500, 1000, 2500, 5000];

export class LandingND {
  constructor(canvas) {
    this.cv = canvas;
    this.ctx = canvas.getContext("2d");
    this.W = canvas.width; this.H = canvas.height;           // 520 x 580
    this.scr = { x: 50, y: 34, w: this.W - 100, h: this.H - 78 };
    this.opt = { auto: true, rangeIdx: 5, trkUp: false, plan: true, hist: true, wind: true, pred: true };
    const bw = 34, bh = 40, sy = this.scr.y + 62, gap = (this.scr.h - 150 - 4 * bh) / 3;
    this.keys = [];
    const L = [["量程", "▲", () => { this.opt.auto = false; this.opt.rangeIdx = Math.min(RANGES.length - 1, this.opt.rangeIdx + 1); }],
               ["量程", "▼", () => { this.opt.auto = false; this.opt.rangeIdx = Math.max(0, this.opt.rangeIdx - 1); }],
               ["量程", "自动", () => { this.opt.auto = !this.opt.auto; }, () => this.opt.auto],
               ["定向", "", () => { this.opt.trkUp = !this.opt.trkUp; }, null, () => (this.opt.trkUp ? "航迹" : "北向")]];
    const Rk = [["规划", "", () => { this.opt.plan = !this.opt.plan; }, () => this.opt.plan],
                ["历史", "", () => { this.opt.hist = !this.opt.hist; }, () => this.opt.hist],
                ["风", "", () => { this.opt.wind = !this.opt.wind; }, () => this.opt.wind],
                ["预测", "", () => { this.opt.pred = !this.opt.pred; }, () => this.opt.pred]];
    L.forEach((k, i) => this.keys.push({ x: 8, y: sy + i * (bh + gap), w: bw, h: bh, side: "L", label: k[0], sub: k[1], fn: k[2], on: k[3], dyn: k[4] }));
    Rk.forEach((k, i) => this.keys.push({ x: this.W - 8 - bw, y: sy + i * (bh + gap), w: bw, h: bh, side: "R", label: k[0], sub: k[1], fn: k[2], on: k[3], dyn: k[4] }));
    this.pressed = null;
    canvas.addEventListener("mousedown", (e) => {
      const r = canvas.getBoundingClientRect();
      const x = (e.clientX - r.left) * this.W / r.width, y = (e.clientY - r.top) * this.H / r.height;
      for (const k of this.keys) if (x >= k.x && x <= k.x + k.w && y >= k.y && y <= k.y + k.h) {
        k.fn(); this.pressed = k; setTimeout(() => { this.pressed = null; }, 140); e.stopPropagation(); e.preventDefault();
      }
    });
    this._buildBezel();
  }

  _buildBezel() {
    const W = this.W, H = this.H, s = this.scr;
    const c = document.createElement("canvas"); c.width = W; c.height = H;
    const g = c.getContext("2d");
    faceplate(g, W, H, 22);
    for (const [x, y] of [[18, 16], [W - 18, 16], [18, H - 16], [W - 18, H - 16]]) screw(g, x, y, 7);
    // recessed screen frame
    g.fillStyle = "#0a0c0e"; roundRect(g, s.x - 6, s.y - 6, s.w + 12, s.h + 12, 8); g.fill();
    g.strokeStyle = "rgba(255,255,255,0.10)"; g.lineWidth = 1; roundRect(g, s.x - 6.5, s.y - 6.5, s.w + 13, s.h + 13, 8); g.stroke();
    engrave(g, "着陆区  ·  导航显示", W / 2, 16, 12);
    engrave(g, "亮度", 40, H - 20, 10); engrave(g, "模式", W - 44, H - 20, 10);
    for (const x of [72, W - 80]) {
      const kg = g.createRadialGradient(x - 3, H - 24, 1, x, H - 20, 11);
      kg.addColorStop(0, "#7c838b"); kg.addColorStop(1, "#1b1e22");
      g.fillStyle = kg; g.beginPath(); g.arc(x, H - 20, 10, 0, Math.PI * 2); g.fill();
      g.strokeStyle = "#0c0d0f"; g.lineWidth = 1.5; g.stroke();
      g.strokeStyle = "#dfe3e8"; g.lineWidth = 2; g.beginPath(); g.moveTo(x, H - 20); g.lineTo(x + 5, H - 27); g.stroke();
    }
    this.bezel = c;
  }

  _key(ctx, k) {
    const down = this.pressed === k;
    const g = ctx.createLinearGradient(0, k.y, 0, k.y + k.h);
    g.addColorStop(0, down ? "#1c1f23" : "#4a5058"); g.addColorStop(1, down ? "#2a2e33" : "#23272c");
    ctx.fillStyle = g; roundRect(ctx, k.x, k.y + (down ? 1 : 0), k.w, k.h, 5); ctx.fill();
    ctx.strokeStyle = "#0b0c0e"; ctx.lineWidth = 1.5; ctx.stroke();
    ctx.strokeStyle = "rgba(255,255,255,0.12)"; ctx.lineWidth = 1; roundRect(ctx, k.x + 1.5, k.y + 1.5, k.w - 3, k.h - 3, 4); ctx.stroke();
  }

  draw(st) {
    const ctx = this.ctx, W = this.W, H = this.H, s = this.scr;
    ctx.clearRect(0, 0, W, H);
    ctx.drawImage(this.bezel, 0, 0);
    for (const k of this.keys) this._key(ctx, k);

    // ---- screen
    ctx.save();
    roundRect(ctx, s.x, s.y, s.w, s.h, 5); ctx.clip();
    const bg = ctx.createRadialGradient(s.x + s.w / 2, s.y + s.h / 2, 20, s.x + s.w / 2, s.y + s.h / 2, s.h * 0.75);
    bg.addColorStop(0, "#07121a"); bg.addColorStop(1, "#02060a");
    ctx.fillStyle = bg; ctx.fillRect(s.x, s.y, s.w, s.h);

    const dist = Math.hypot(st.pos[0], st.pos[1]);
    const final = st.alt < 50 || st.state === "LANDED";
    if (this.opt.auto) {
      const need = final ? 12 : Math.max(25, dist * 1.3);
      this.opt.rangeIdx = RANGES.findIndex((r) => r >= need);
      if (this.opt.rangeIdx < 0) this.opt.rangeIdx = RANGES.length - 1;
    }
    const range = RANGES[this.opt.rangeIdx];
    const cx = s.x + s.w / 2, cy = s.y + s.h / 2 + 8;
    const Rr = Math.min(s.w, s.h) / 2 - 62;
    const k = Rr / range;
    const trk = Math.atan2(st.vel[0], st.vel[1]);               // track from north (rad)
    const rot = this.opt.trkUp && Math.hypot(st.vel[0], st.vel[1]) > 1 ? -trk : 0;
    const P = (e, n) => {                                     // world east/north (m) -> screen
      const x = e * Math.cos(rot) - n * Math.sin(rot), y = e * Math.sin(rot) + n * Math.cos(rot);
      return [cx + x * k, cy - y * k];
    };

    // compass rose
    ctx.lineWidth = 1.2; ctx.strokeStyle = "#cfd8df"; ctx.fillStyle = "#cfd8df";
    ctx.font = "bold 16px 'DejaVu Sans Mono', 'Microsoft YaHei', 'PingFang SC', monospace"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    for (let a = 0; a < 360; a += 5) {
      const t = a * D2R + rot, major = a % 30 === 0;
      const r0 = Rr + 4, r1 = Rr + (major ? 14 : a % 10 === 0 ? 10 : 7);
      const sx = Math.sin(t), sy = -Math.cos(t);
      ctx.beginPath(); ctx.moveTo(cx + sx * r0, cy + sy * r0); ctx.lineTo(cx + sx * r1, cy + sy * r1); ctx.stroke();
      if (major) {
        const lab = { 0: "北", 90: "东", 180: "南", 270: "西" }[a] || String(a / 10);
        ctx.fillStyle = a === 0 ? "#ffffff" : "#aab6c0";
        ctx.fillText(lab, cx + sx * (Rr + 26), cy + sy * (Rr + 26));
      }
    }
    // range rings
    ctx.strokeStyle = "rgba(170,190,205,0.35)"; ctx.lineWidth = 1;
    for (const f of [0.25, 0.5, 0.75]) { ctx.setLineDash(f === 0.5 ? [] : [3, 5]); ctx.beginPath(); ctx.arc(cx, cy, Rr * f, 0, Math.PI * 2); ctx.stroke(); }
    ctx.setLineDash([]); ctx.strokeStyle = "rgba(170,190,205,0.55)"; ctx.beginPath(); ctx.arc(cx, cy, Rr, 0, Math.PI * 2); ctx.stroke();
    ctx.fillStyle = "rgba(170,190,205,0.9)"; ctx.font = "14px 'DejaVu Sans Mono', 'Microsoft YaHei', 'PingFang SC', monospace"; ctx.textAlign = "left";
    const fmtR = (m) => (m >= 1000 ? (m / 1000).toFixed(m % 1000 ? 1 : 0) + "km" : m >= 10 ? m.toFixed(0) : m.toFixed(1));
    ctx.fillText(fmtR(range / 2), cx + Rr * 0.5 * 0.71 + 3, cy - Rr * 0.5 * 0.71 - 3);

    // map layer, clipped to the outer range ring
    ctx.save(); ctx.beginPath(); ctx.arc(cx, cy, Rr, 0, Math.PI * 2); ctx.clip();
    // landing pad (same markings as the 3-D pad)
    const pad = (r, col, w, dash) => { ctx.strokeStyle = col; ctx.lineWidth = w; ctx.setLineDash(dash || []); ctx.beginPath(); ctx.arc(cx, cy, r * k, 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([]); };
    if (26 * k > 4) {
      ctx.fillStyle = "rgba(160,160,150,0.16)"; ctx.beginPath(); ctx.arc(cx, cy, 26 * k, 0, Math.PI * 2); ctx.fill();
      pad(19, "rgba(245,200,30,0.85)", Math.max(1, 1.0 * k)); pad(21.5, "rgba(245,200,30,0.6)", Math.max(1, 0.6 * k));
      ctx.strokeStyle = "rgba(225,228,230,0.35)"; ctx.lineWidth = Math.max(1.2, 1.2 * k);
      for (const [a, b] of [[[-12, 0], [12, 0]], [[0, -12], [0, 12]]]) { const p0 = P(...a), p1 = P(...b); ctx.beginPath(); ctx.moveTo(...p0); ctx.lineTo(...p1); ctx.stroke(); }
    } else {
      ctx.strokeStyle = "#f5c81e"; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(cx, cy, 5, 0, Math.PI * 2); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(cx - 8, cy); ctx.lineTo(cx + 8, cy); ctx.moveTo(cx, cy - 8); ctx.lineTo(cx, cy + 8); ctx.stroke();
    }
    if (final) { pad(1, "#4dff8e", 1.5, [4, 3]); pad(3, "rgba(255,180,60,0.9)", 1.2, [6, 5]); }

    // history breadcrumbs
    if (this.opt.hist && st.trail) {
      const tr = st.trail, n = tr.length;
      for (let i = 0; i < n; i += 1) {
        const p = P(tr[i][0], tr[i][1]);
        ctx.fillStyle = `rgba(80,190,255,${0.15 + 0.6 * i / n})`;
        ctx.fillRect(p[0] - 1.5, p[1] - 1.5, 3, 3);
      }
    }
    // planned path + touchdown point (magenta)
    if (this.opt.plan && st.plan && st.plan.length > 1) {
      ctx.strokeStyle = "#ff4fe0"; ctx.lineWidth = 2.2; ctx.beginPath();
      st.plan.forEach((q, i) => { const p = P(q[0], q[1]); i ? ctx.lineTo(...p) : ctx.moveTo(...p); });
      ctx.stroke();
      const e = st.plan[st.plan.length - 1], p = P(e[0], e[1]);
      ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(p[0] - 7, p[1] - 7); ctx.lineTo(p[0] + 7, p[1] + 7); ctx.moveTo(p[0] + 7, p[1] - 7); ctx.lineTo(p[0] - 7, p[1] + 7); ctx.stroke();
      ctx.fillStyle = "#ff4fe0"; ctx.font = "bold 14px 'DejaVu Sans Mono', 'Microsoft YaHei', 'PingFang SC', monospace"; ctx.textAlign = "left"; ctx.fillText("触地", p[0] + 9, p[1] - 9);
    }

    ctx.restore();
    // own ship
    let [ox, oy] = P(st.pos[0], st.pos[1]);
    const off = Math.hypot(ox - cx, oy - cy) > Rr - 6;
    if (off) {
      const a = Math.atan2(oy - cy, ox - cx);
      ox = cx + Math.cos(a) * (Rr - 10); oy = cy + Math.sin(a) * (Rr - 10);
    }
    if (this.opt.pred && !off && st.state === "FLYING") {
      ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 1.6; ctx.beginPath(); ctx.moveTo(ox, oy);
      const e10 = P(st.pos[0] + st.vel[0] * 10, st.pos[1] + st.vel[1] * 10); ctx.lineTo(...e10); ctx.stroke();
      ctx.fillStyle = "#ffffff";
      for (let t = 2; t <= 10; t += 2) { const q = P(st.pos[0] + st.vel[0] * t, st.pos[1] + st.vel[1] * t); ctx.beginPath(); ctx.arc(q[0], q[1], 2, 0, Math.PI * 2); ctx.fill(); }
    }
    const R3 = quatToMat3(st.q);
    ctx.save(); ctx.translate(ox, oy);
    if (off) {
      ctx.fillStyle = "#ffb43c"; const a = Math.atan2(oy - cy, ox - cx);
      ctx.rotate(a); ctx.beginPath(); ctx.moveTo(10, 0); ctx.lineTo(-6, -7); ctx.lineTo(-6, 7); ctx.closePath(); ctx.fill();
    } else {
      const rb = Math.max(5, 1.83 * k);
      ctx.strokeStyle = "#35d6ff"; ctx.lineWidth = 2.2;
      for (let i = 0; i < 4; i++) {                              // legs (deployed footprint)
        const a = Math.PI / 4 + i * Math.PI / 2, fr = (1.9 + 3.9 * st.legs);
        const fx = R3[0][0] * Math.cos(a) * fr + R3[0][1] * Math.sin(a) * fr, fy = R3[1][0] * Math.cos(a) * fr + R3[1][1] * Math.sin(a) * fr;
        const q = [(fx * Math.cos(rot) - fy * Math.sin(rot)) * k, -(fx * Math.sin(rot) + fy * Math.cos(rot)) * k];
        const L = Math.hypot(...q), m = Math.max(L, rb + 5) / Math.max(L, 1e-6);
        ctx.beginPath(); ctx.moveTo(q[0] / Math.max(L, 1e-6) * rb, q[1] / Math.max(L, 1e-6) * rb); ctx.lineTo(q[0] * m, q[1] * m); ctx.stroke();
        ctx.fillStyle = "#35d6ff"; ctx.fillRect(q[0] * m - 2.5, q[1] * m - 2.5, 5, 5);
      }
      ctx.fillStyle = "rgba(53,214,255,0.25)"; ctx.beginPath(); ctx.arc(0, 0, rb, 0, Math.PI * 2); ctx.fill();
      ctx.beginPath(); ctx.arc(0, 0, rb, 0, Math.PI * 2); ctx.stroke();
      ctx.fillStyle = "#35d6ff"; ctx.beginPath(); ctx.arc(0, 0, 2.5, 0, Math.PI * 2); ctx.fill();
    }
    ctx.restore();

    // wind (top-right, inside the screen)
    const wind = st.wind || [0, 0, 0], ws = Math.hypot(wind[0], wind[1]);
    ctx.font = "bold 14px 'DejaVu Sans Mono', 'Microsoft YaHei', 'PingFang SC', monospace"; ctx.textBaseline = "middle";
    if (this.opt.wind && ws > 0.2) {
      const from = (Math.atan2(-wind[0], -wind[1]) / D2R + 360) % 360;
      const wx = s.x + s.w - 30, wy = s.y + 72;
      ctx.save(); ctx.translate(wx, wy); ctx.rotate(Math.atan2(wind[0], wind[1]) + rot);
      ctx.strokeStyle = "#aee7ff"; ctx.fillStyle = "#aee7ff"; ctx.lineWidth = 2.2;
      ctx.beginPath(); ctx.moveTo(0, -13); ctx.lineTo(0, 13); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, -15); ctx.lineTo(-5, -6); ctx.lineTo(5, -6); ctx.closePath(); ctx.fill();
      ctx.restore();
      ctx.fillStyle = "#aee7ff"; ctx.textAlign = "right";
      ctx.fillText(`${String(Math.round(from)).padStart(3, "0")}°/${ws.toFixed(1)}`, s.x + s.w - 8, s.y + 98);
    }

    // data blocks
    const txt = (t, x, y, col = "#4dff8e", al = "left", size = 16) => { ctx.font = `bold ${size}px 'DejaVu Sans Mono', 'Microsoft YaHei', 'PingFang SC', monospace`; ctx.textAlign = al; ctx.fillStyle = col; ctx.fillText(t, x, y); };
    const gs = Math.hypot(st.vel[0], st.vel[1]);
    const brg = (Math.atan2(-st.pos[0], -st.pos[1]) / D2R + 360) % 360;
    txt(`${this.opt.trkUp ? "航迹朝上" : "北朝上"}`, s.x + 8, s.y + 16, "#e0e6ea");
    txt(`量程 ${fmtR(range)}${this.opt.auto ? " 自动" : ""}`, s.x + 8, s.y + 36, "#e0e6ea");
    txt(`地速 ${gs.toFixed(1)}`, s.x + s.w - 8, s.y + 16, "#4dff8e", "right");
    txt(`航迹 ${String(Math.round((trk / D2R + 360) % 360)).padStart(3, "0")}`, s.x + s.w - 8, s.y + 36, "#4dff8e", "right");
    const warnD = final && dist > 3;
    const errNow = st.touchdown && st.state !== "FLYING" ? st.touchdown.error : dist;
    txt(`距离 ${errNow < 100 ? errNow.toFixed(2) : errNow.toFixed(0)} m`, s.x + 8, s.y + s.h - 36, warnD ? "#ffb43c" : "#4dff8e");
    txt(`方位 ${String(Math.round(brg)).padStart(3, "0")}`, s.x + 8, s.y + s.h - 15, "#4dff8e");
    txt(`高度 ${st.alt < 100 ? st.alt.toFixed(1) : st.alt.toFixed(0)}`, s.x + s.w - 8, s.y + s.h - 36, "#4dff8e", "right");
    const vs = st.vel[2];
    txt(`升降 ${vs.toFixed(1)}`, s.x + s.w - 8, s.y + s.h - 15, vs < -8 && st.alt < 30 ? "#ffb43c" : "#4dff8e", "right");
    if (final) {
      const de = st.pos[0], dn = st.pos[1];
      txt(`Δ东 ${de >= 0 ? "+" : ""}${de.toFixed(2)}  Δ北 ${dn >= 0 ? "+" : ""}${dn.toFixed(2)}`, cx, s.y + 60, "#ffffff", "center", 16);
    }
    if (st.state === "LANDED") txt("已落台 · 发动机已安全", cx, s.y + s.h - 58, "#4dff8e", "center", 15);
    else if (st.state === "CRASHED") txt("箭体损毁", cx, s.y + s.h - 58, "#ff5a4e", "center", 16);
    else if (st.tgo !== null && st.tgo >= 0 && st.autopilot) txt(`剩余 ${st.tgo.toFixed(1)} s`, cx, s.y + s.h - 58, "#ff4fe0", "center", 15);

    // soft-key legends on the screen edge
    for (const kk of this.keys) {
      const on = kk.on ? kk.on() : null;
      const label = kk.dyn ? kk.dyn() : kk.label;
      const x = kk.side === "L" ? s.x + 4 : s.x + s.w - 4;
      ctx.font = "bold 14px 'DejaVu Sans Mono', 'Microsoft YaHei', 'PingFang SC', monospace"; ctx.textAlign = kk.side === "L" ? "left" : "right";
      ctx.fillStyle = on === null ? "#e0e6ea" : on ? "#4dff8e" : "#6b7680";
      ctx.fillText(label, x + (kk.side === "L" ? 3 : -3), kk.y + kk.h / 2 - (kk.sub ? 8 : 0));
      if (kk.sub) { ctx.fillStyle = "#e0e6ea"; ctx.fillText(kk.sub, x + (kk.side === "L" ? 3 : -3), kk.y + kk.h / 2 + 9); }
      ctx.fillStyle = "rgba(224,230,234,0.5)"; ctx.fillRect(kk.side === "L" ? s.x : s.x + s.w - 3, kk.y + kk.h / 2 - 1, 3, 2);
    }
    // glass: faint reflection + vignette
    const gr = ctx.createLinearGradient(s.x, s.y, s.x + s.w, s.y + s.h);
    gr.addColorStop(0, "rgba(255,255,255,0.05)"); gr.addColorStop(0.35, "rgba(255,255,255,0.0)");
    ctx.fillStyle = gr; ctx.fillRect(s.x, s.y, s.w, s.h);
    ctx.restore();
  }
}
