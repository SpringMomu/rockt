// 3D scene renderer (WebGL2).  All geometry in world ENU metres, z up.
// Linear HDR pipeline: MSAA RGBA16F scene target -> resolve -> volumetric
// plume + lit smoke (soft, depth-aware) -> bloom -> ACES tone map -> sRGB.
import { M4, V, rot, createProgram, createMesh, drawMesh, createDynamic, uploadDynamic,
  frustum, box, merge, transformPart, polarGrid } from "./gl.js";
import * as S from "./scene_shaders.js";
import { plumeParams, EXIT_Z, EXIT_R } from "./plume.js";

const FAR = S.FAR, LOG_FC = S.LOG_FC;
const TERRAIN_MARKINGS = `  // ---- Landing zone: access road, apron, drainage trench, pad with markings,
  // hazard band, tie-downs, heading arrow, blast streaks and engine scorch.
  float ang = atan(p.y, p.x);
  const float TAU = 6.2831853;
  // Access road (runs out from the apron towards the south-south-west).
  {
    vec2 rd = vec2(-0.17365, -0.98481);              // bearing -100 deg
    float t = dot(p.xy, rd), s = abs(rd.x * p.y - rd.y * p.x);
    if (t > 40.0 && t < 430.0 && s < 5.6) {
      vec3 asph = mix(vec3(0.17, 0.17, 0.18), vec3(0.23, 0.23, 0.24), nz2) * (0.92 + 0.12 * fbm(p.xy / 9.0));
      vec3 gravel = mix(vec3(0.42, 0.40, 0.36), vec3(0.50, 0.48, 0.43), nz2);
      vec3 road = mix(asph, gravel, smoothstep(4.4, 4.6, s));
      road = mix(road, vec3(0.88, 0.88, 0.85), step(3.85, s) * step(s, 4.05));                         // edge lines
      road = mix(road, vec3(0.90, 0.72, 0.12), step(s, 0.12) * step(fract(t / 9.0), 0.5));             // centre dashes
      float fadeOut = smoothstep(430.0, 400.0, t);
      col = mix(col, road, fadeOut * smoothstep(5.6, 5.0, s));
    }
  }
  if (r < 75.0) {
    vec3 apron = mix(vec3(0.31, 0.32, 0.33), vec3(0.36, 0.37, 0.37), nz2);
    apron *= 0.94 + 0.10 * fbm(p.xy / 6.0);                                   // weathering
    // Slab joints: radial every 15 deg plus two concentric joints.
    float jr = abs(fract(ang / TAU * 24.0 + 0.5) - 0.5) * TAU / 24.0 * r;
    float joint = step(jr, 0.07) + step(abs(r - 38.0), 0.07) + step(abs(r - 52.0), 0.07);
    apron *= 1.0 - 0.18 * clamp(joint, 0.0, 1.0) * step(29.0, r);
    // Oil / water stains.
    apron *= 1.0 - 0.18 * smoothstep(0.62, 0.78, fbm(p.xy / 11.0 + 3.7));
    col = mix(apron, col, smoothstep(68.0, 75.0, r));
    if (abs(r - 66.0) < 0.35) col = vec3(0.85, 0.85, 0.8);
    // Dashed yellow keep-out circle.
    if (abs(r - 45.0) < 0.22 && fract(ang / TAU * 70.0) < 0.55) col = vec3(0.90, 0.72, 0.14);
    // Drainage trench with grating around the pad.
    if (r > 27.2 && r < 28.8) {
      float bar = step(0.55, fract(ang / TAU * 540.0));
      vec3 grate = mix(vec3(0.07, 0.07, 0.075), vec3(0.30, 0.31, 0.32), bar);
      grate = mix(grate, vec3(0.45, 0.45, 0.44), step(abs(r - 28.0), 0.06));   // centre rail
      col = mix(col, grate, smoothstep(27.2, 27.35, r) * smoothstep(28.8, 28.65, r));
    }
  }
  if (r < 27.2) {
    // Poured-concrete pad, 6 m slabs with per-slab tone and saw-cut joints.
    vec2 cell = floor(p.xy / 6.0);
    vec2 tile = abs(fract(p.xy / 6.0) - 0.5);
    vec3 pad = mix(vec3(0.60, 0.60, 0.58), vec3(0.66, 0.66, 0.64), nz2);
    pad *= 0.95 + 0.08 * hash(cell);
    pad *= 1.0 - 0.10 * step(0.488, max(tile.x, tile.y));
    pad *= 1.0 - 0.12 * smoothstep(0.60, 0.80, fbm(p.xy / 5.0 + 11.0));           // stains
    col = mix(pad, col, smoothstep(26.9, 27.2, r));
    // Raised curb at the pad rim.
    if (r > 25.9) col = mix(col, vec3(0.50, 0.50, 0.49) * (0.9 + 0.1 * nz2), smoothstep(25.9, 26.0, r) * smoothstep(27.2, 27.0, r));
    // Yellow / black hazard band just inside the curb.
    if (r > 24.1 && r < 25.7) {
      float stripe = step(0.5, fract(ang / TAU * 71.0 + r * 0.55));
      vec3 hz = mix(vec3(0.93, 0.74, 0.10), vec3(0.06, 0.06, 0.06), stripe);
      col = mix(col, hz, smoothstep(24.1, 24.2, r) * smoothstep(25.7, 25.6, r));
    }
    // Twin yellow capture rings.
    float ring = smoothstep(0.55, 0.35, abs(r - 19.0)) + smoothstep(0.35, 0.2, abs(r - 21.5)) * 0.8;
    col = mix(col, vec3(0.95, 0.78, 0.12), clamp(ring, 0.0, 1.0));
    // Bearing ticks every 10 deg between the outer ring and the hazard band (long at cardinals).
    float tk = abs(fract(ang / TAU * 36.0 + 0.5) - 0.5) * TAU / 36.0 * r;
    float card = abs(fract(ang / TAU * 4.0 + 0.5) - 0.5) * TAU / 4.0 * r;
    float tick = step(tk, 0.13) * step(22.3, r) * step(r, 23.3) + step(card, 0.22) * step(22.1, r) * step(r, 23.9);
    col = mix(col, vec3(0.95, 0.95, 0.92), clamp(tick, 0.0, 1.0));
    // Central cross, aim circle and centre dot.
    float cross_ = (step(abs(p.x), 0.6) + step(abs(p.y), 0.6)) * step(r, 12.0) * step(1.6, r);
    col = mix(col, vec3(0.95, 0.95, 0.92), clamp(cross_, 0.0, 1.0));
    col = mix(col, vec3(0.95, 0.95, 0.92), smoothstep(0.3, 0.18, abs(r - 6.0)));
    col = mix(col, vec3(0.95, 0.78, 0.12), smoothstep(1.1, 0.95, r));
    // North heading arrow.
    vec2 q = p.xy - vec2(0.0, 13.2);
    float arrow = step(0.0, q.y) * step(q.y, 3.6) * step(abs(q.x), (3.6 - q.y) * 0.42);
    col = mix(col, vec3(0.95, 0.95, 0.92), arrow);
    // Eight tie-down anchor plates between the rings.
    float aSeg = TAU / 8.0;
    float aLoc = (fract((ang - aSeg * 0.5) / aSeg + 0.5) - 0.5) * aSeg;
    vec2 ap = vec2(r - 15.6, aLoc * r);
    float plate = step(max(abs(ap.x), abs(ap.y)), 0.42);
    float bolt = step(length(abs(ap) - 0.24), 0.07);
    col = mix(col, vec3(0.22, 0.23, 0.24), plate);
    col = mix(col, vec3(0.55, 0.56, 0.58), plate * bolt);
    // Engine scorch + radial blast streaks (painted markings get burnt too).
    float scorch = smoothstep(14.0, 0.0, r + (nz - 0.5) * 7.0);
    float streak = pow(vnoise(vec2(ang * 34.0, r * 0.05)), 2.6) * smoothstep(20.0, 4.0, r);
    col = mix(col, vec3(0.08, 0.075, 0.07), clamp(scorch * 0.78 + streak * 0.1, 0.0, 0.9));
  }
`;

// Surface of revolution from a [r, z] profile (top -> bottom), smooth normals.
// extra = (azimuth fraction, z) like frustum().
function lathe(profile, seg = 48) {
  const d = [], idx = [];
  const n = profile.length;
  const nrm2 = profile.map((p, i) => {
    const a = profile[Math.max(0, i - 1)], b = profile[Math.min(n - 1, i + 1)];
    const dr = b[0] - a[0], dz = b[1] - a[1], l = Math.hypot(dr, dz) || 1;
    return [-dz / l, dr / l];                       // outward for a top->bottom profile
  });
  for (let k = 0; k < n; k++) {
    for (let i = 0; i <= seg; i++) {
      const a = (i / seg) * Math.PI * 2, c = Math.cos(a), s = Math.sin(a);
      const [r, z] = profile[k], [nr, nz] = nrm2[k];
      d.push(c * r, s * r, z, c * nr, s * nr, nz, i / seg, z);
    }
  }
  const row = seg + 1;
  for (let k = 0; k < n - 1; k++) for (let i = 0; i < seg; i++) {
    const B0 = k * row + i, A0 = (k + 1) * row + i;          // A = lower ring, B = upper ring
    idx.push(A0, A0 + 1, B0, B0, A0 + 1, B0 + 1);
  }
  return { d, idx };
}
// Box tapered along z (w0 x t0 at z=0 to w1 x t1 at z=len), flat shaded.
function taperedBox(w0, t0, w1, t1, len) {
  const c = [];
  for (const [z, w, t] of [[0, w0, t0], [len, w1, t1]]) for (const [sx, sy] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) c.push([sx * t / 2, sy * w / 2, z]);
  const faces = [[0, 1, 2, 3], [7, 6, 5, 4], [0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]];
  const d = [], idx = [];
  for (const f of faces) {
    const [a, b, cc] = f.map((i) => c[i]);
    let nn = V.norm(V.cross(V.sub(b, a), V.sub(cc, a)));
    const ctr = f.reduce((s, i) => V.add(s, c[i]), [0, 0, 0]);
    if (V.dot(nn, V.sub(V.mul(ctr, 0.25), [0, 0, len / 2])) < 0) nn = V.mul(nn, -1);
    const base = d.length / 8;
    f.forEach((i, k) => d.push(...c[i], ...nn, k & 1, k >> 1));
    idx.push(base, base + 1, base + 2, base, base + 2, base + 3);
  }
  return { d, idx };
}
// Beam (box) between two points.
function beamBetween(a, b, w, h = w) {
  const dir = V.sub(b, a), len = V.len(dir), zax = V.norm(dir);
  const axr = V.cross([0, 0, 1], zax), an = Math.acos(Math.max(-1, Math.min(1, zax[2])));
  const m = M4.mul(M4.translate(a), M4.mul(V.len(axr) > 1e-6 ? M4.rotAxis(axr, an) : (zax[2] < 0 ? M4.rotAxis([1, 0, 0], Math.PI) : M4.ident()), M4.scale([1, 1, len])));
  return transformPart(box(w, h, 1, 0, 0, 0.5), m);
}

export class Scene {
  constructor(canvas) {
    const gl = canvas.getContext("webgl2", { antialias: false, alpha: false, preserveDrawingBuffer: true });
    if (!gl) throw new Error("WebGL2 is not available in this browser");
    this.gl = gl;
    this.canvas = canvas;
    this.hdr = !!gl.getExtension("EXT_color_buffer_float");
    gl.getExtension("OES_texture_float_linear");
    this.sun = V.norm([-0.55, -0.62, 0.56]);
    this.exposure = 1.05;
    this.bloomK = 0.12;
    this.progSky = createProgram(gl, S.SKY_VS, S.SKY_FS);
    this.progTerrain = createProgram(gl, S.TERRAIN_VS, S.terrainFS(TERRAIN_MARKINGS));
    this.progLit = createProgram(gl, S.LIT_VS, S.LIT_FS);
    this.progDepth = createProgram(gl, S.DEPTH_VS, S.DEPTH_FS);
    this.progPoint = createProgram(gl, S.POINT_VS, S.POINT_FS);
    this.progLine = createProgram(gl, S.LINE_VS, S.LINE_FS);
    this.progPlume = createProgram(gl, S.FS_VS, S.PLUME_FS);
    this.progPart = createProgram(gl, S.PART_VS, S.PART_FS);
    this.progDown = createProgram(gl, S.FS_VS, S.DOWN_FS);
    this.progUp = createProgram(gl, S.FS_VS, S.UP_FS);
    this.progComp = createProgram(gl, S.FS_VS, S.COMPOSITE_FS);
    this.progCopy = createProgram(gl, S.FS_VS, S.COPY_FS);
    this.progRcs = createProgram(gl, S.RCS_VS, S.RCS_FS);
    const rc = frustum(1, 1, 0, 1, 20, false);
    this.meshRcsJet = createMesh(gl, rc.d, rc.idx);
    const t = polarGrid(230, 200, 3.0, 1.0415);
    this.terrain = createMesh(gl, t.d, t.idx);
    this._buildRocket();
    this._buildPadProps();
    const q = { d: [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1], idx: [0, 1, 2, 0, 2, 3] };
    this.quad = createMesh(gl, q.d, q.idx);
    this.points = createDynamic(gl, 6000);
    this.lines = createDynamic(gl, 60000);
    this.emptyVao = gl.createVertexArray();
    this._initParticles(2000);
    // Shadow map.
    this.shadowSize = 2048;
    this.shadowTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.shadowTex);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.DEPTH_COMPONENT24, this.shadowSize, this.shadowSize);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_COMPARE_MODE, gl.COMPARE_REF_TO_TEXTURE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    this.shadowFbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.shadowFbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, this.shadowTex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.targets = null;
    this.outFbo = null;
    this.time = 0;
  }

  // ------------------------------------------------------------ render targets
  _texture(w, h, ifmt, fmt, type, filter) {
    const gl = this.gl, t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, ifmt, w, h, 0, fmt, type, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  }
  _fbo(color, depth) {
    const gl = this.gl, f = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, f);
    if (color) gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, color, 0);
    if (depth) gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, depth, 0);
    return f;
  }
  _ensureTargets(W, H) {
    if (this.targets && this.targets.W === W && this.targets.H === H) return;
    const gl = this.gl;
    if (this.targets) {
      const o = this.targets;
      [o.colorTex, o.depthTex, o.plumeTex, ...o.bloom.map((b) => b.tex)].forEach((t) => gl.deleteTexture(t));
      [o.msFbo, o.resolveFbo, o.transFbo, o.plumeFbo, ...o.bloom.map((b) => b.fbo)].forEach((f) => gl.deleteFramebuffer(f));
      [o.msColor, o.msDepth].forEach((r) => gl.deleteRenderbuffer(r));
    }
    const ifmt = this.hdr ? gl.RGBA16F : gl.RGBA8, type = this.hdr ? gl.HALF_FLOAT : gl.UNSIGNED_BYTE;
    const samples = Math.min(4, gl.getParameter(gl.MAX_SAMPLES) || 1);
    const msColor = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, msColor);
    gl.renderbufferStorageMultisample(gl.RENDERBUFFER, samples, ifmt, W, H);
    const msDepth = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, msDepth);
    gl.renderbufferStorageMultisample(gl.RENDERBUFFER, samples, gl.DEPTH_COMPONENT24, W, H);
    const msFbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, msFbo);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.RENDERBUFFER, msColor);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, msDepth);
    const colorTex = this._texture(W, H, ifmt, gl.RGBA, type, gl.LINEAR);
    const depthTex = this._texture(W, H, gl.DEPTH_COMPONENT24, gl.DEPTH_COMPONENT, gl.UNSIGNED_INT, gl.NEAREST);
    const resolveFbo = this._fbo(colorTex, depthTex);
    const transFbo = this._fbo(colorTex, null);
    const bloom = [];
    let w = W, h = H;
    for (let i = 0; i < 6; i++) {
      w = Math.max(1, w >> 1); h = Math.max(1, h >> 1);
      const tex = this._texture(w, h, ifmt, gl.RGBA, type, gl.LINEAR);
      bloom.push({ tex, fbo: this._fbo(tex, null), w, h });
      if (w <= 4 || h <= 4) break;
    }
    const PW = Math.max(1, W >> 1), PH = Math.max(1, H >> 1);
    const plumeTex = this._texture(PW, PH, ifmt, gl.RGBA, type, gl.LINEAR);
    const plumeFbo = this._fbo(plumeTex, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.targets = { W, H, PW, PH, msColor, msDepth, msFbo, colorTex, depthTex, resolveFbo, transFbo, plumeTex, plumeFbo, bloom };
  }

  _initParticles(max) {
    const gl = this.gl;
    this.partMax = max;
    this.partVao = gl.createVertexArray();
    gl.bindVertexArray(this.partVao);
    const qb = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, qb);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 8, 0);
    this.partBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.partBuf);
    gl.bufferData(gl.ARRAY_BUFFER, max * 12 * 4, gl.DYNAMIC_DRAW);
    for (let k = 0; k < 3; k++) {
      gl.enableVertexAttribArray(1 + k);
      gl.vertexAttribPointer(1 + k, 4, gl.FLOAT, false, 48, k * 16);
      gl.vertexAttribDivisor(1 + k, 1);
    }
    gl.bindVertexArray(null);
  }

  _buildRocket() {
    const gl = this.gl;
    // Tank barrel + interstage (radius 1.5 m, 18 m), raceway along +x.
    const body = merge([frustum(1.5, 1.5, 0.0, 15.0, 128), frustum(1.5, 1.5, 15.0, 18.0, 128),
      box(0.26, 0.34, 13.7, 1.56, 0, 1.3 + 13.7 / 2),                        // raceway (cable tunnel)
      box(0.2, 0.5, 0.5, 1.52, 0, 14.8)]);
    this.meshBody = createMesh(gl, body.d, body.idx);
    // Merlin bell: throat at z=-0.2, exit (r = EXIT_R) at EXIT_Z, parabolic contour.
    const bellProfile = (re, zt, ze, rt) => {
      const p = [];
      for (let i = 0; i <= 14; i++) { const t = i / 14; p.push([rt + (re - rt) * (1 - Math.pow(1 - t, 2.1)), zt + (ze - zt) * t]); }
      return p;
    };
    const bell = lathe(bellProfile(EXIT_R, -0.2, EXIT_Z, 0.14), 40);
    this.meshBell = createMesh(gl, bell.d, bell.idx);
    // Outer eight engines (fixed during the single-engine landing burn) + hardware.
    const outer = [], hw = [];
    const ringR = 1.02;
    for (let k = 0; k < 8; k++) {
      const a = (k / 8) * Math.PI * 2 + Math.PI / 8;
      const m = M4.translate([Math.cos(a) * ringR, Math.sin(a) * ringR, 0]);
      outer.push(transformPart(lathe(bellProfile(EXIT_R, -0.2, EXIT_Z, 0.14), 32), m));
      hw.push(transformPart(frustum(0.2, 0.16, -0.22, 0.0, 16), m));        // chamber / injector
      hw.push(transformPart(box(0.22, 0.14, 0.2, 0.22, 0, -0.1), M4.mul(m, M4.rotAxis([0, 0, 1], a))));  // turbopump exhaust stub
    }
    hw.push(frustum(0.2, 0.16, -0.22, 0.3, 16));                            // centre engine chamber
    const om = merge(outer); this.meshOuterBells = createMesh(gl, om.d, om.idx);
    const hm = merge(hw); this.meshEngineHW = createMesh(gl, hm.d, hm.idx);
    // Grid fins: titanium lattice, deployed (flow passes through the cells along the body axis).
    const fins = [];
    for (let k = 0; k < 4; k++) {
      const parts = [];
      const x0 = 1.6, x1 = 2.85, y0 = -0.62, y1 = 0.62, zc = 16.8, dep = 0.32;
      parts.push(box(x1 - x0, 0.06, dep, (x0 + x1) / 2, y0, zc), box(x1 - x0, 0.06, dep, (x0 + x1) / 2, y1, zc));
      parts.push(box(0.06, y1 - y0 + 0.06, dep, x1, 0, zc), box(0.08, y1 - y0 + 0.06, dep, x0, 0, zc));
      const sp = 0.155;
      for (const sgn of [1, -1]) for (let c = -3; c <= 3; c += sp) {
        // lines x - sgn*y = c + mid
        const cx = (x0 + x1) / 2;
        let ya = Math.max(y0, sgn > 0 ? x0 - (cx + c) : (cx + c) - x1), yb = Math.min(y1, sgn > 0 ? x1 - (cx + c) : (cx + c) - x0);
        if (yb - ya < 0.05) continue;
        const pa = [cx + c + sgn * ya, ya, zc], pb = [cx + c + sgn * yb, yb, zc];
        const L = V.len(V.sub(pb, pa)), mid = V.mul(V.add(pa, pb), 0.5);
        const ang = Math.atan2(pb[1] - pa[1], pb[0] - pa[0]);
        parts.push(transformPart(box(L, 0.022, dep - 0.02, 0, 0, 0), M4.mul(M4.translate(mid), M4.rotAxis([0, 0, 1], ang))));
      }
      parts.push(box(0.2, 0.3, 0.5, 1.5, 0, zc - 0.05));                  // hinge / actuator fairing
      fins.push(transformPart(merge(parts), M4.rotAxis([0, 0, 1], (k * Math.PI) / 2)));
    }
    const fm = merge(fins);
    this.meshFins = createMesh(gl, fm.d, fm.idx);
    // Landing leg: tapered carbon-fibre beam + foot pad oriented flat on the ground when deployed.
    const pad = transformPart(frustum(0.5, 0.42, -0.08, 0.1, 24), M4.mul(M4.translate([0.08, 0, 6.25]), M4.rotAxis([0, 1, 0], 35 * Math.PI / 180)));
    const leg = merge([taperedBox(0.95, 0.42, 0.42, 0.24, 6.2), transformPart(box(0.12, 0.7, 0.1, 0, 0, 0), M4.translate([0.12, 0, 0.3])), pad]);
    this.meshLeg = createMesh(gl, leg.d, leg.idx);
    const strut = merge([frustum(0.12, 0.12, 0, 0.55, 12), frustum(0.08, 0.08, 0.55, 1.0, 12), frustum(0.14, 0.14, 0.5, 0.58, 12)]);
    this.meshStrut = createMesh(gl, strut.d, strut.idx);
    const rcs = merge([0, 1, 2, 3].map((k) => transformPart(merge([box(0.35, 0.35, 0.5, 1.58, 0, 17.3), frustum(0.06, 0.08, 17.3, 17.42, 8)]), M4.rotAxis([0, 0, 1], k * Math.PI / 2 + Math.PI / 4))));
    this.meshRcs = createMesh(gl, rcs.d, rcs.idx);
    this.meshPole = createMesh(gl, frustum(0.08, 0.08, 0, 6, 8).d, frustum(0.08, 0.08, 0, 6, 8).idx);
    const sock = frustum(0.45, 0.14, 0, 2.6, 16, false);
    this.meshSock = createMesh(gl, sock.d, sock.idx);
  }

  // Static landing-zone furniture: pad-edge lights, floodlight masts, service
  // shed, deluge water tank, containers, pad cameras and road bollards.  All
  // props sit at r >= 27 m (outside the touchdown area) and are purely visual:
  // the physics only knows the flat ground.  Geometry is baked in world space
  // and merged per material, so the whole set costs a handful of draw calls.
  _buildPadProps() {
    const gl = this.gl;
    const groups = {};
    const add = (key, g) => { (groups[key] = groups[key] || []).push(g); };
    const glow = [];      // [x, y, z, kind, phase]
    // Local frame of a prop at bearing `ang` and radius `r`: +x points away
    // from the pad centre, so the side facing the pad is -x.
    const placeM = (ang, r, z = 0) => M4.mul(M4.translate([r * Math.cos(ang), r * Math.sin(ang), z]), M4.rotAxis([0, 0, 1], ang));
    const beam = (a, b, w) => {
      const dir = V.sub(b, a), len = V.len(dir), zax = V.norm(dir);
      const axr = V.cross([0, 0, 1], zax), an = Math.acos(Math.max(-1, Math.min(1, zax[2])));
      const m = M4.mul(M4.translate(a), M4.mul(V.len(axr) > 1e-6 ? M4.rotAxis(axr, an) : M4.ident(), M4.scale([1, 1, len])));
      return transformPart(box(w, w, 1, 0, 0, 0.5), m);
    };
    const deg = Math.PI / 180;
    const T0 = (P, g) => transformPart(g, P);

    // Pad-edge lights on the curb (bases here, lenses drawn per light so they
    // can run a slow chase pattern).
    this.edgeLights = [];
    for (let k = 0; k < 16; k++) {
      const a = (k / 16) * Math.PI * 2 + Math.PI / 16;
      const pos = [Math.cos(a) * 26.55, Math.sin(a) * 26.55, 0];
      add("steel", transformPart(frustum(0.2, 0.17, 0, 0.18, 12), M4.translate(pos)));
      this.edgeLights.push(pos);
    }
    const lens = frustum(0.13, 0.09, 0.18, 0.36, 12);
    this.meshLens = createMesh(gl, lens.d, lens.idx);

    // Floodlight masts (4), lamp heads tilted down towards the pad.
    for (const bearing of [20, 110, 200, 290]) {
      const a = bearing * deg, P = placeM(a, 58);
      const T = (g) => transformPart(g, P);
      add("hazard", T(box(1.9, 1.9, 0.75, 0, 0, 0.375)));
      add("steel", T(frustum(0.34, 0.2, 0.75, 22.0, 16)));
      add("steel", T(box(1.1, 4.4, 0.22, -0.2, 0, 22.1)));                              // service platform
      for (const y of [-2.1, 2.1]) add("steel", T(box(0.06, 0.06, 1.0, -0.65, y, 22.7))); // railing posts
      add("steel", T(box(0.06, 4.3, 0.06, -0.65, 0, 23.15)));
      for (let z = 1.5; z < 21.8; z += 0.45) add("steel", T(box(0.05, 0.5, 0.05, 0.42, 0, z))); // ladder rungs
      add("steel", T(box(0.05, 0.05, 21.2, 0.42, -0.24, 11.4)));
      add("steel", T(box(0.05, 0.05, 21.2, 0.42, 0.24, 11.4)));
      // Tilted lamp head.
      const c = [-0.35, 0, 23.6];
      const tilt = M4.mul(P, M4.mul(M4.translate(c), M4.mul(M4.rotAxis([0, 1, 0], -0.5), M4.translate(V.mul(c, -1)))));
      add("dark", transformPart(box(0.4, 4.0, 2.1, -0.35, 0, 23.6), tilt));
      for (const y of [-1.35, 0, 1.35]) for (const z of [23.1, 24.1]) {
        add("lamp", transformPart(box(0.1, 1.15, 0.8, -0.6, y, z), tilt));
        glow.push([...M4.xform(tilt, [-0.75, y, z]), 1, 0]);
      }
      add("steel", T(frustum(0.08, 0.08, 24.7, 25.3, 8)));
      glow.push([...M4.xform(P, [0, 0, 25.45]), 2, bearing * 0.013]);
      add("red", T(frustum(0.14, 0.1, 25.3, 25.6, 10)));
    }

    // Service shed facing the pad: walls, roof, door, windows, AC units, antenna.
    {
      const P = placeM(150 * deg, 60);
      const T = (g) => transformPart(g, P);
      add("concrete", T(box(7.4, 11.4, 0.3, 0, 0, 0.15)));            // plinth
      add("shed", T(box(6, 10, 3.6, 0, 0, 2.1)));
      add("dark", T(box(6.4, 10.4, 0.28, 0, 0, 4.04)));               // roof slab
      add("dark", T(box(0.08, 1.3, 2.3, -3.02, -2.6, 1.45)));          // door
      add("hazard", T(box(0.1, 1.6, 0.18, -3.04, -2.6, 2.75)));        // door lintel
      for (const y of [0.4, 2.6]) add("glass", T(box(0.06, 1.5, 0.9, -3.02, y, 2.3)));
      add("steel", T(box(1.4, 1.2, 0.9, 1.0, 2.8, 4.63)));
      add("steel", T(box(1.4, 1.2, 0.9, 1.0, 0.8, 4.63)));
      add("steel", T(frustum(0.06, 0.04, 4.18, 10.5, 6)));
      add("steel", T(box(0.04, 1.2, 0.04, 0, 0, 9.4)));
      add("red", T(frustum(0.12, 0.09, 10.5, 10.8, 8)));
      glow.push([...M4.xform(P, [0, 0, 10.75]), 2, 0.5]);
      add("hazard", T(box(0.5, 0.5, 1.0, -3.8, -3.8, 0.5)));          // bollards at the door
      add("hazard", T(box(0.5, 0.5, 1.0, -3.8, -1.4, 0.5)));
    }

    // Deluge / sound-suppression water tank with ladder cage and pipe.
    {
      const P = placeM(62 * deg, 62);
      const T = (g) => transformPart(g, P);
      add("concrete", T(frustum(3.6, 3.6, 0, 0.35, 32)));
      add("white", T(frustum(3.1, 3.1, 0.35, 10.0, 40)));
      add("white", T(frustum(3.1, 0.7, 10.0, 11.0, 40)));
      add("steel", T(frustum(3.13, 3.13, 3.4, 3.55, 40, false)));      // weld bands
      add("steel", T(frustum(3.13, 3.13, 6.8, 6.95, 40, false)));
      for (let z = 0.8; z < 10.2; z += 0.4) add("steel", T(box(0.05, 0.5, 0.05, -3.25, 0, z)));
      add("steel", T(box(0.05, 0.05, 10, -3.25, -0.24, 5.3)));
      add("steel", T(box(0.05, 0.05, 10, -3.25, 0.24, 5.3)));
      add("steel", T(frustum(0.28, 0.28, 0.25, 0.8, 12)));
      add("steel", beam(M4.xform(P, [-3.0, 1.6, 0.5]), M4.xform(P, [-24.0, 1.6, 0.5]), 0.5));    // feed pipe
      add("steel", T(box(0.6, 0.6, 1.1, -24.0, 1.6, 0.55)));
      add("red", T(frustum(0.12, 0.09, 11.0, 11.3, 8)));
      glow.push([...M4.xform(P, [0, 0, 11.25]), 2, 1.0]);
    }

    // Two shipping containers (ground support equipment storage).
    {
      const P = placeM(-40 * deg, 63);
      const T = (g) => transformPart(g, P);
      add("boxBlue", T(box(2.44, 6.1, 2.6, 0, -1.8, 1.3)));
      add("boxRust", T(box(2.44, 6.1, 2.6, 0.4, 4.6, 1.3)));
      add("boxBlue", transformPart(box(2.44, 6.1, 2.6, 0.2, 1.4, 3.9), P));
    }

    // Pad cameras on tripods (small housings aimed at the pad centre).
    for (const [bearing, r] of [[75, 36], [-15, 35], [165, 37], [245, 34]]) {
      const P = placeM(bearing * deg, r);
      const top = M4.xform(P, [0, 0, 1.35]);
      for (let k = 0; k < 3; k++) {
        const a = k * 2 * Math.PI / 3;
        add("steel", beam(M4.xform(P, [Math.cos(a) * 0.55, Math.sin(a) * 0.55, 0]), top, 0.05));
      }
      add("white", T0(P, box(0.55, 0.3, 0.34, 0.0, 0, 1.55)));
      add("dark", transformPart(transformPart(frustum(0.1, 0.12, 0, 0.18, 12), M4.rotAxis([0, 1, 0], -Math.PI / 2)), M4.mul(P, M4.translate([-0.27, 0, 1.55]))));
      add("dark", T0(P, box(0.6, 0.4, 0.04, 0.02, 0, 1.74)));           // sun hood
    }

    // Bollards with reflector bands along the access road.
    const rd = [Math.cos(-100 * deg), Math.sin(-100 * deg)];
    const nrm = [-rd[1], rd[0]];
    for (let t = 80; t <= 400; t += 20) for (const side of [-1, 1]) {
      const x = rd[0] * t + nrm[0] * side * 6.2, y = rd[1] * t + nrm[1] * side * 6.2;
      add("white", transformPart(frustum(0.09, 0.08, 0, 1.0, 8), M4.translate([x, y, 0])));
      add("reflector", transformPart(frustum(0.095, 0.095, 0.78, 0.9, 8, false), M4.translate([x, y, 0])));
    }
    // Jersey barriers flanking the road entrance to the apron.
    for (const side of [-1, 1]) for (let t = 72; t <= 90; t += 3.4) {
      const x = rd[0] * t + nrm[0] * side * 7.0, y = rd[1] * t + nrm[1] * side * 7.0;
      const m = M4.mul(M4.translate([x, y, 0]), M4.rotAxis([0, 0, 1], -100 * deg));
      add("concrete", transformPart(merge([box(3.2, 0.62, 0.3, 0, 0, 0.15), box(3.2, 0.3, 0.55, 0, 0, 0.57)]), m));
    }

    const mats = {
      steel: [2, [0.55, 0.57, 0.60]], dark: [1, [0.13, 0.13, 0.14]], white: [1, [0.86, 0.87, 0.88]],
      concrete: [5, [0.62, 0.62, 0.60]], hazard: [6, [1, 1, 1]], shed: [8, [0.78, 0.79, 0.77]],
      boxBlue: [8, [0.14, 0.30, 0.52]], boxRust: [8, [0.62, 0.30, 0.12]], glass: [2, [0.10, 0.16, 0.22]],
      lamp: [9, [1.0, 0.97, 0.86]], red: [1, [0.55, 0.06, 0.05]], reflector: [2, [0.85, 0.10, 0.08]],
    };
    this.padProps = Object.entries(groups).map(([key, list]) => {
      const g = merge(list);
      return { mesh: createMesh(gl, g.d, g.idx), mat: mats[key][0], color: mats[key][1], key };
    });
    this.padGlow = glow;
  }

  _drawPadProps(pl, t) {
    const gl = this.gl;
    gl.uniformMatrix4fv(pl.u.u_model, false, M4.ident());
    for (const p of this.padProps) {
      let c = p.color;
      const lit = p.key === "red" && this._obstructionOn(t);
      if (lit) c = [1.0, 0.12, 0.08];
      gl.uniform1i(pl.u.u_mat, lit ? 9 : p.mat);
      gl.uniform3fv(pl.u.u_color, c);
      drawMesh(gl, p.mesh);
    }
    // Pad-edge light lenses: steady amber with a slow clockwise chase.
    gl.uniform1i(pl.u.u_mat, 9);
    this.edgeLights.forEach((pos, k) => {
      const b = this._edgeLevel(t, k);
      gl.uniform3fv(pl.u.u_color, [0.35 + 0.65 * b, 0.22 + 0.5 * b, 0.04 + 0.1 * b]);
      gl.uniformMatrix4fv(pl.u.u_model, false, M4.translate(pos));
      drawMesh(gl, this.meshLens);
    });
  }

  _obstructionOn(t) { return (t % 1.6) < 0.55; }
  _edgeLevel(t, k) { const n = this.edgeLights.length; return Math.pow(0.5 + 0.5 * Math.cos(2 * Math.PI * (t * 0.45 - k / n)), 6); }

  _padGlowPoints(t) {
    const out = [];
    this.edgeLights.forEach((p, k) => {
      const b = this._edgeLevel(t, k);
      out.push([p[0], p[1], p[2] + 0.3, 1.0, 0.62, 0.12, 0.25 + 0.6 * b, 1.1 + 0.9 * b]);
    });
    const red = this._obstructionOn(t);
    for (const [x, y, z, kind] of this.padGlow) {
      if (kind === 1) out.push([x, y, z, 1.0, 0.93, 0.75, 0.22, 2.4]);
      else if (red) out.push([x, y, z, 1.0, 0.15, 0.08, 0.9, 1.6]);
    }
    return out;
  }


  // Pieces of the booster as (mesh, material, model matrix, colour, glow).
  rocketParts(st) {
    const base = M4.fromQuatPos(st.q, st.pos);
    const engineOn = st.throttle > 0.01 && st.state === "FLYING";
    const parts = [
      [this.meshBody, 0, base], [this.meshFins, 3, base], [this.meshRcs, 1, base, [0.08, 0.08, 0.085]],
      [this.meshOuterBells, 10, base, null, 0], [this.meshEngineHW, 13, base, [0.14, 0.13, 0.12]],
    ];
    const [gx, gy] = [st.gimbal[0] * Math.PI / 180, st.gimbal[1] * Math.PI / 180];
    const pivot = M4.translate([0, 0, 0.3]), unpivot = M4.translate([0, 0, -0.3]);
    const g = M4.mul(M4.rotAxis([0, 1, 0], gy), M4.rotAxis([1, 0, 0], gx));
    parts.push([this.meshBell, 10, M4.mul(base, M4.mul(pivot, M4.mul(g, unpivot))), null, engineOn ? 0.4 + 0.6 * st.throttle : 0]);
    const deploy = st.legs;
    const theta = (145 * Math.PI / 180) * deploy;
    for (let k = 0; k < 4; k++) {
      const az = Math.PI / 4 + (k * Math.PI) / 2;
      const hinge = [Math.cos(az) * 1.55, Math.sin(az) * 1.55, 3.2];
      const tangent = [-Math.sin(az), Math.cos(az), 0];
      const m = M4.mul(M4.translate(hinge), M4.mul(M4.rotAxis(tangent, theta), M4.rotAxis([0, 0, 1], az)));
      parts.push([this.meshLeg, 4, M4.mul(base, m)]);
      if (deploy > 0.05) {
        const foot = M4.xform(m, [0, 0, 3.4]);
        const anchor = [Math.cos(az) * 1.5, Math.sin(az) * 1.5, 6.6];
        const dir = V.sub(foot, anchor), len = V.len(dir);
        const zaxis = V.norm(dir), ang = Math.acos(Math.max(-1, Math.min(1, zaxis[2])));
        const ax = V.cross([0, 0, 1], zaxis);
        const sm = M4.mul(M4.translate(anchor), M4.mul(V.len(ax) > 1e-6 ? M4.rotAxis(ax, ang) : M4.ident(), M4.scale([1, 1, len])));
        parts.push([this.meshStrut, 2, M4.mul(base, sm), [0.62, 0.63, 0.65]]);
      }
    }
    return parts;
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 1);    // render at CSS resolution (MSAA covers edges)
    const w = Math.floor(this.canvas.clientWidth * dpr), h = Math.floor(this.canvas.clientHeight * dpr);
    if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; }
  }

  // Scene lights from the flame: l0 = luminous jet, l1 = impingement / wall jet.
  _lights(P) {
    if (!P.on) return { l0p: [0, 0, -1e4], l0c: [0, 0, 0], l1p: [0, 0, -1e4], l1c: [0, 0, 0] };
    const I = P.intensity * (0.35 + 0.65 * P.pamb);
    const s = Math.min(P.len * 0.22, Math.max(0.5, P.hdist * 0.5));
    const l0p = V.add(P.exit, V.mul(P.axis, s));
    const k0 = 110 * I;
    const l0c = [1.0 * k0, 0.50 * k0, 0.17 * k0];
    let l1p = [0, 0, -1e4], l1c = [0, 0, 0];
    if (P.hit && P.heat > 0.003) {
      l1p = [P.hit[0], P.hit[1], 1.2 + 0.3 * P.Rj];
      const k1 = 110 * P.heat;
      l1c = [1.0 * k1, 0.46 * k1, 0.14 * k1];
    }
    return { l0p, l0c, l1p, l1c };
  }

  // ---------------------------------------------------------------- render
  render(cam, st, fx, opts) {
    const gl = this.gl;
    this.resize();
    const W = this.canvas.width, H = this.canvas.height;
    this._ensureTargets(W, H);
    const T = this.targets;
    const proj = M4.perspective(cam.fov, W / H, 0.3, FAR);
    const view = M4.lookAt(cam.eye, cam.at, [0, 0, 1]);
    const vp = M4.mul(proj, view);
    this.vp = vp; this.proj = proj; this.viewM = view; this.W = W; this.H = H;
    this.invVP = M4.invert(vp); this.logFC = LOG_FC;
    const camR = [view[0], view[4], view[8]], camU = [view[1], view[5], view[9]], camF = [-view[2], -view[6], -view[10]];
    const parts = this.rocketParts(st);
    const P = fx.plume || plumeParams(st, this.time);
    const Lt = this._lights(P);

    // ---- shadow pass (sun ortho around the booster)
    const focus = [st.pos[0], st.pos[1], Math.max(0, Math.min(st.pos[2], 60))];
    const sunEye = V.add(focus, V.mul(this.sun, 400));
    const sview = M4.lookAt(sunEye, focus, [0, 0, 1]);
    const sproj = M4.ortho(-45, 45, -45, 45, 250, 560);
    const svp = M4.mul(sproj, sview);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.shadowFbo);
    gl.viewport(0, 0, this.shadowSize, this.shadowSize);
    gl.clear(gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    gl.useProgram(this.progDepth.program);
    gl.uniformMatrix4fv(this.progDepth.u.u_vp, false, svp);
    if (st.pos[2] < 400) for (const [mesh, , m] of parts) { gl.uniformMatrix4fv(this.progDepth.u.u_model, false, m); drawMesh(gl, mesh); }
    gl.uniformMatrix4fv(this.progDepth.u.u_model, false, M4.ident());
    for (const p of this.padProps) drawMesh(gl, p.mesh);

    // ---- opaque HDR pass (MSAA)
    gl.bindFramebuffer(gl.FRAMEBUFFER, T.msFbo);
    gl.viewport(0, 0, W, H);
    gl.depthMask(true);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.disable(gl.DEPTH_TEST);
    gl.useProgram(this.progSky.program);
    gl.uniformMatrix4fv(this.progSky.u.u_invVP, false, this.invVP);
    gl.uniform3fv(this.progSky.u.u_cam, cam.eye);
    gl.uniform3fv(this.progSky.u.u_sun, this.sun);
    gl.uniform1f(this.progSky.u.u_alt, cam.eye[2]);
    gl.bindVertexArray(this.emptyVao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.enable(gl.DEPTH_TEST);

    const setLights = (u) => {
      gl.uniform3fv(u.u_l0Pos, Lt.l0p); gl.uniform3fv(u.u_l0Col, Lt.l0c);
      gl.uniform3fv(u.u_l1Pos, Lt.l1p); gl.uniform3fv(u.u_l1Col, Lt.l1c);
    };
    // terrain
    const pt = this.progTerrain;
    gl.useProgram(pt.program);
    gl.uniformMatrix4fv(pt.u.u_vp, false, vp);
    gl.uniform1f(pt.u.u_logFC, LOG_FC);
    gl.uniform3fv(pt.u.u_cam, cam.eye);
    gl.uniform3fv(pt.u.u_sun, this.sun);
    gl.uniform1f(pt.u.u_alt, cam.eye[2]);
    gl.uniformMatrix4fv(pt.u.u_shadowVP, false, svp);
    setLights(pt.u);
    gl.uniform3fv(pt.u.u_hit, P.hit || [0, 0, -1e4]);
    gl.uniform1f(pt.u.u_Rw, P.Rw || 1);
    gl.uniform1f(pt.u.u_heat, P.on && P.hit ? P.heat * P.intensity : 0);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.shadowTex);
    gl.uniform1i(pt.u.u_shadow, 0);
    drawMesh(gl, this.terrain);

    // booster + props
    const pl = this.progLit;
    gl.useProgram(pl.program);
    gl.uniformMatrix4fv(pl.u.u_vp, false, vp);
    gl.uniform1f(pl.u.u_logFC, LOG_FC);
    gl.uniform3fv(pl.u.u_cam, cam.eye);
    gl.uniform3fv(pl.u.u_sun, this.sun);
    gl.uniform1f(pl.u.u_alt, cam.eye[2]);
    gl.uniformMatrix4fv(pl.u.u_shadowVP, false, svp);
    gl.uniform1i(pl.u.u_shadow, 0);
    setLights(pl.u);
    const vrel = V.sub(st.vel, st.wind);
    const sp = V.len(vrel);
    const flowW = sp > 0.5 ? V.mul(vrel, -1 / sp) : [0, 0, 1];
    const Rm = M4.fromQuatPos(st.q, [0, 0, 0]);
    const flowB = [Rm[0] * flowW[0] + Rm[1] * flowW[1] + Rm[2] * flowW[2], Rm[4] * flowW[0] + Rm[5] * flowW[1] + Rm[6] * flowW[2], Rm[8] * flowW[0] + Rm[9] * flowW[1] + Rm[10] * flowW[2]];
    gl.uniform1i(pl.u.u_cpMode, 0);
    gl.uniform3fv(pl.u.u_flowB, flowB);
    gl.uniform1f(pl.u.u_sinA, Math.sin(((st.aero && st.aero.alpha) || 0) * Math.PI / 180));
    for (const [mesh, mat, m, color, glow] of parts) {
      gl.uniform1i(pl.u.u_mat, mat);
      gl.uniform3fv(pl.u.u_color, color || [0.9, 0.9, 0.9]);
      gl.uniform1f(pl.u.u_glow, glow || 0);
      gl.uniformMatrix4fv(pl.u.u_model, false, m);
      drawMesh(gl, mesh);
    }
    gl.uniform1f(pl.u.u_glow, 0);
    // Windsock at the apron edge.
    const wind = st.wind, ws = Math.hypot(wind[0], wind[1]);
    const sockBase = [62, -18, 0];
    gl.uniform1i(pl.u.u_mat, 2); gl.uniform3fv(pl.u.u_color, [0.7, 0.7, 0.72]);
    gl.uniformMatrix4fv(pl.u.u_model, false, M4.translate(sockBase)); drawMesh(gl, this.meshPole);
    const droop = Math.max(0.05, Math.min(1, ws / 10));
    const wdir = ws > 0.1 ? V.norm([wind[0], wind[1], 0]) : [1, 0, 0];
    const sockDir = V.norm(V.add(V.mul(wdir, droop), [0, 0, -(1 - droop)]));
    const sax = V.cross([0, 0, 1], sockDir), sang = Math.acos(Math.max(-1, Math.min(1, sockDir[2])));
    gl.uniform1i(pl.u.u_mat, 1); gl.uniform3fv(pl.u.u_color, [0.95, 0.38, 0.08]);
    gl.uniformMatrix4fv(pl.u.u_model, false, M4.mul(M4.translate([sockBase[0], sockBase[1], 5.8]), V.len(sax) > 1e-6 ? M4.rotAxis(sax, sang) : M4.ident()));
    drawMesh(gl, this.meshSock);
    this._drawPadProps(pl, this.time);

    // lines + lamp glows (depth tested against the MSAA depth)
    gl.enable(gl.BLEND);
    gl.depthMask(false);
    this._drawLines(fx.lines);
    this._drawPoints(this._padGlowPoints(this.time), true);
    gl.depthMask(true);
    gl.disable(gl.BLEND);

    // ---- resolve colour + depth
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, T.msFbo);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, T.resolveFbo);
    gl.blitFramebuffer(0, 0, W, H, 0, 0, W, H, gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT, gl.NEAREST);
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null);

    // ---- transparent volumetrics into the resolved colour (manual depth tests)
    gl.bindFramebuffer(gl.FRAMEBUFFER, T.transFbo);
    this.outFbo = T.transFbo;
    gl.viewport(0, 0, W, H);
    gl.disable(gl.DEPTH_TEST);
    gl.depthMask(false);
    gl.enable(gl.BLEND);
    if (fx.fluid) {
      fx.fluid.draw(this, cam, fx.fluidMode, fx.fluidVolume);
      gl.bindFramebuffer(gl.FRAMEBUFFER, T.transFbo);
      gl.viewport(0, 0, W, H);
      gl.disable(gl.DEPTH_TEST);
      gl.depthMask(false);
      gl.enable(gl.BLEND);
    }
    const partCtx = { camR, camU, camF, cam: cam.eye, Lt };
    const pa = fx.particles;
    if (pa && pa.n > 0) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.partBuf);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, pa.data, 0, Math.min(pa.n, this.partMax) * 12);
    }
    if (pa && pa.nFar > 0) this._drawParticles(0, pa.nFar, partCtx);
    if (P.on) this._drawPlume(P, cam, camF);
    if (pa && pa.n > pa.nFar) this._drawParticles(pa.nFar, Math.min(pa.n, this.partMax) - pa.nFar, partCtx);
    if (fx.rcs && fx.rcs.length) this._drawRcs(fx.rcs, cam);

    // ---- bloom + tone map
    gl.disable(gl.BLEND);
    this._post();
    gl.depthMask(true);
    gl.enable(gl.DEPTH_TEST);
  }

  _fullscreen() { this.gl.bindVertexArray(this.emptyVao); this.gl.drawArrays(this.gl.TRIANGLES, 0, 3); }

  _drawPlume(P, cam, camF) {
    const gl = this.gl, T = this.targets, pp = this.progPlume;
    // Scissor to the projected bounds of the jet and the wall jet.
    const pts = [];
    const Rb = P.jetR(Math.min(P.len * 1.25, P.hdist)) * 2.2 + 1.5;
    for (let i = 0; i <= 4; i++) {
      const s = Math.min(P.len * 1.25, P.hdist + 1) * i / 4;
      const c = V.add(P.exit, V.mul(P.axis, s));
      for (const o of [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]) pts.push(V.add(c, V.mul(o, Rb + V.len(P.bend) * s * s)));
    }
    if (P.hit && P.heat > 0.002) {
      const R = P.Rw * 2.6 + 2, Hh = 0.3 * P.Rj + 0.26 * P.Rw + 3;
      for (let k = 0; k < 8; k++) { const a = k * Math.PI / 4; for (const z of [0, Hh]) pts.push([P.hit[0] + Math.cos(a) * R, P.hit[1] + Math.sin(a) * R, z]); }
    }
    let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9, behind = false;
    for (const p of pts) {
      const v = this.vp;
      const x = v[0] * p[0] + v[4] * p[1] + v[8] * p[2] + v[12], y = v[1] * p[0] + v[5] * p[1] + v[9] * p[2] + v[13], w = v[3] * p[0] + v[7] * p[1] + v[11] * p[2] + v[15];
      if (w <= 0.1) { behind = true; break; }
      const sx = (x / w * 0.5 + 0.5) * T.W, sy = (y / w * 0.5 + 0.5) * T.H;
      x0 = Math.min(x0, sx); x1 = Math.max(x1, sx); y0 = Math.min(y0, sy); y1 = Math.max(y1, sy);
    }
    if (behind) { x0 = 0; y0 = 0; x1 = T.W; y1 = T.H; }
    x0 = Math.max(0, Math.floor(x0 / 2) - 2); y0 = Math.max(0, Math.floor(y0 / 2) - 2);
    x1 = Math.min(T.PW, Math.ceil(x1 / 2) + 2); y1 = Math.min(T.PH, Math.ceil(y1 / 2) + 2);
    if (x1 <= x0 || y1 <= y0) return;
    // Ray-march at half resolution, then composite.
    gl.bindFramebuffer(gl.FRAMEBUFFER, T.plumeFbo);
    gl.viewport(0, 0, T.PW, T.PH);
    gl.disable(gl.BLEND);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT); gl.clearColor(0, 0, 0, 1);
    gl.enable(gl.SCISSOR_TEST);
    gl.scissor(x0, y0, x1 - x0, y1 - y0);
    gl.useProgram(pp.program);
    gl.uniformMatrix4fv(pp.u.u_invVP, false, this.invVP);
    gl.uniform3fv(pp.u.u_cam, cam.eye);
    gl.uniform3fv(pp.u.u_fwd, camF);
    gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, T.depthTex); gl.uniform1i(pp.u.u_depth, 3);
    gl.uniform1f(pp.u.u_logFar, Math.log2(FAR + 1));
    gl.uniform3fv(pp.u.u_exit, P.exit);
    gl.uniform3fv(pp.u.u_axis, P.axis);
    gl.uniform3fv(pp.u.u_bend, P.bend);
    gl.uniform1f(pp.u.u_len, P.len);
    gl.uniform1f(pp.u.u_re, EXIT_R);
    gl.uniform1f(pp.u.u_spread, P.spread);
    gl.uniform1f(pp.u.u_bulge, P.bulge);
    gl.uniform1f(pp.u.u_time, this.time % 1000);
    gl.uniform1f(pp.u.u_I, P.intensity);
    gl.uniform1f(pp.u.u_dia, P.diamonds);
    gl.uniform1f(pp.u.u_lambda, P.lambda);
    gl.uniform1f(pp.u.u_pamb, P.pamb);
    gl.uniform1f(pp.u.u_hdist, P.hdist);
    gl.uniform1f(pp.u.u_Rj, P.Rj || 1);
    gl.uniform1f(pp.u.u_Rw, P.Rw || 1);
    gl.uniform1f(pp.u.u_wall, P.hit ? P.heat * P.intensity : 0);
    gl.uniform1f(pp.u.u_blast, P.blast);
    gl.uniform3fv(pp.u.u_hit, P.hit || [0, 0, -1e4]);
    gl.uniform1f(pp.u.u_dscale, T.W / T.PW);
    this._fullscreen();
    gl.disable(gl.SCISSOR_TEST);
    gl.bindFramebuffer(gl.FRAMEBUFFER, T.transFbo);
    gl.viewport(0, 0, T.W, T.H);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.useProgram(this.progCopy.program);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, T.plumeTex);
    gl.uniform1i(this.progCopy.u.u_src, 0);
    this._fullscreen();
    gl.activeTexture(gl.TEXTURE0);
  }

  _drawParticles(first, count, c) {
    const gl = this.gl, T = this.targets, pp = this.progPart;
    gl.useProgram(pp.program);
    gl.uniformMatrix4fv(pp.u.u_vp, false, this.vp);
    gl.uniform1f(pp.u.u_logFC, LOG_FC);
    gl.uniform3fv(pp.u.u_camR, c.camR); gl.uniform3fv(pp.u.u_camU, c.camU); gl.uniform3fv(pp.u.u_camF, c.camF);
    gl.uniform3fv(pp.u.u_cam, c.cam);
    gl.uniform3fv(pp.u.u_sun, this.sun);
    gl.uniform1f(pp.u.u_time, this.time % 1000);
    gl.uniform3fv(pp.u.u_l0Pos, c.Lt.l0p); gl.uniform3fv(pp.u.u_l0Col, c.Lt.l0c);
    gl.uniform3fv(pp.u.u_l1Pos, c.Lt.l1p); gl.uniform3fv(pp.u.u_l1Col, c.Lt.l1c);
    gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, T.depthTex); gl.uniform1i(pp.u.u_depth, 3);
    gl.uniform1f(pp.u.u_logFar, Math.log2(FAR + 1));
    gl.bindVertexArray(this.partVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.partBuf);
    for (let k = 0; k < 3; k++) gl.vertexAttribPointer(1 + k, 4, gl.FLOAT, false, 48, first * 48 + k * 16);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, count);
    gl.bindVertexArray(null);
    gl.activeTexture(gl.TEXTURE0);
  }

  _drawRcs(jets, cam) {
    const gl = this.gl, T = this.targets, pr = this.progRcs;
    gl.useProgram(pr.program);
    gl.uniformMatrix4fv(pr.u.u_vp, false, this.vp);
    gl.uniform3fv(pr.u.u_cam, cam.eye);
    gl.uniform3fv(pr.u.u_sun, this.sun);
    gl.uniform1f(pr.u.u_time, this.time % 1000);
    gl.uniform1f(pr.u.u_logFar, Math.log2(FAR + 1));
    gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, T.depthTex); gl.uniform1i(pr.u.u_depth, 3);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.enable(gl.CULL_FACE); gl.cullFace(gl.BACK);
    for (const j of jets) {
      const z = V.norm(j.dir), ang = Math.acos(Math.max(-1, Math.min(1, z[2]))), ax = V.cross([0, 0, 1], z);
      const m = M4.mul(M4.translate(j.base), V.len(ax) > 1e-6 ? M4.rotAxis(ax, ang) : (z[2] < 0 ? M4.rotAxis([1, 0, 0], Math.PI) : M4.ident()));
      gl.uniformMatrix4fv(pr.u.u_model, false, m);
      gl.uniform1f(pr.u.u_len, j.len); gl.uniform1f(pr.u.u_r0, 0.05); gl.uniform1f(pr.u.u_r1, j.r1);
      gl.uniform1f(pr.u.u_I, j.I); gl.uniform1f(pr.u.u_seed, j.seed);
      drawMesh(gl, this.meshRcsJet);
    }
    gl.disable(gl.CULL_FACE);
    gl.activeTexture(gl.TEXTURE0);
  }

  _post() {
    const gl = this.gl, T = this.targets;
    const B = T.bloom;
    // downsample chain
    gl.useProgram(this.progDown.program);
    let src = T.colorTex, sw = T.W, sh = T.H;
    for (let i = 0; i < B.length; i++) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, B[i].fbo);
      gl.viewport(0, 0, B[i].w, B[i].h);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, src);
      gl.uniform1i(this.progDown.u.u_src, 0);
      gl.uniform2f(this.progDown.u.u_texel, 1 / sw, 1 / sh);
      gl.uniform1i(this.progDown.u.u_first, i === 0 ? 1 : 0);
      this._fullscreen();
      src = B[i].tex; sw = B[i].w; sh = B[i].h;
    }
    // upsample + accumulate
    gl.useProgram(this.progUp.program);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    for (let i = B.length - 1; i > 0; i--) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, B[i - 1].fbo);
      gl.viewport(0, 0, B[i - 1].w, B[i - 1].h);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, B[i].tex);
      gl.uniform1i(this.progUp.u.u_src, 0);
      gl.uniform2f(this.progUp.u.u_texel, 1 / B[i].w, 1 / B[i].h);
      this._fullscreen();
    }
    gl.disable(gl.BLEND);
    // composite
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, T.W, T.H);
    const pc = this.progComp;
    gl.useProgram(pc.program);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, T.colorTex); gl.uniform1i(pc.u.u_hdr, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, B[0].tex); gl.uniform1i(pc.u.u_bloom, 1);
    gl.uniform1f(pc.u.u_exposure, this.exposure);
    gl.uniform1f(pc.u.u_bloomK, this.bloomK / Math.max(1, B.length) * 1.0);
    gl.uniform1f(pc.u.u_time, this.time % 1000);
    this._fullscreen();
    gl.activeTexture(gl.TEXTURE0);
  }

  _drawPoints(list, additive) {
    if (!list || list.length === 0) return;
    const gl = this.gl, d = this.points;
    const n = Math.min(list.length, d.max);
    for (let i = 0; i < n; i++) {
      const p = list[i], o = i * 8;
      for (let k = 0; k < 8; k++) d.data[o + k] = p[k];
    }
    d.n = n;
    uploadDynamic(gl, d);
    const pp = this.progPoint;
    gl.useProgram(pp.program);
    gl.uniformMatrix4fv(pp.u.u_vp, false, this.vp);
    gl.uniform1f(pp.u.u_logFC, LOG_FC);
    gl.uniform1f(pp.u.u_scale, this.proj[5] * this.H * 0.5);
    if (additive) gl.blendFunc(gl.ONE, gl.ONE);
    else gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.bindVertexArray(d.vao);
    gl.drawArrays(gl.POINTS, 0, n);
  }

  _drawLines(tris) {
    if (!tris || tris.length === 0) return;
    const gl = this.gl, d = this.lines;
    const n = Math.min(tris.length / 8, d.max);
    d.data.set(tris.subarray ? tris.subarray(0, n * 8) : tris.slice(0, n * 8));
    d.n = n;
    uploadDynamic(gl, d);
    const pl = this.progLine;
    gl.useProgram(pl.program);
    gl.uniformMatrix4fv(pl.u.u_vp, false, this.vp);
    gl.uniform1f(pl.u.u_logFC, LOG_FC);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.bindVertexArray(d.vao);
    gl.drawArrays(gl.TRIANGLES, 0, n);
  }

  project(p) {
    const vp = this.vp;
    const x = vp[0] * p[0] + vp[4] * p[1] + vp[8] * p[2] + vp[12];
    const y = vp[1] * p[0] + vp[5] * p[1] + vp[9] * p[2] + vp[13];
    const w = vp[3] * p[0] + vp[7] * p[1] + vp[11] * p[2] + vp[15];
    if (w <= 0) return null;
    return [(x / w * 0.5 + 0.5) * this.canvas.clientWidth, (1 - (y / w * 0.5 + 0.5)) * this.canvas.clientHeight, w];
  }
}
