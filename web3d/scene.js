// 3D scene renderer (WebGL2).  All geometry in world ENU metres, z up.
import { M4, V, rot, createProgram, createMesh, drawMesh, createDynamic, uploadDynamic,
  frustum, box, merge, transformPart, polarGrid } from "./gl.js";

const FAR = 160000.0;
const LOG_FC = 2.0 / Math.log2(FAR + 1.0);

const LOGDEPTH_VS = `
out float v_logz;
void logDepth() { v_logz = 1.0 + gl_Position.w; }`;
const LOGDEPTH_FS = `
in float v_logz;
uniform float u_logFC;
void logDepth() { gl_FragDepth = log2(v_logz) * u_logFC * 0.5; }`;

const NOISE = `
float hash(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
float vnoise(vec2 p) {
  vec2 i = floor(p), f = fract(p); vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1, 0)), u.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), u.x), u.y);
}
float fbm(vec2 p) { float s = 0.0, a = 0.5; for (int i = 0; i < 6; i++) { s += a * vnoise(p); p = p * 2.03 + vec2(17.1, 9.2); a *= 0.5; } return s; }
float terrainHeight(vec2 p) {
  float r = length(p);
  float flat_ = smoothstep(420.0, 1100.0, r);
  float hills = (fbm(p / 2600.0) - 0.45) * 380.0;
  float ridge = 1.0 - abs(vnoise(p / 9000.0) * 2.0 - 1.0);
  float mountains = pow(ridge, 3.0) * 2300.0 * smoothstep(9000.0, 26000.0, r);
  return flat_ * (max(hills, -30.0) + mountains) - 0.02;
}`;

const SKY_FS = `#version 300 es
precision highp float;
in vec2 v_uv; out vec4 o;
uniform mat4 u_invVP; uniform vec3 u_cam; uniform vec3 u_sun; uniform float u_alt;
vec3 skyColor(vec3 d) {
  float h = clamp(u_alt / 22000.0, 0.0, 1.0);
  vec3 zen = mix(vec3(0.20, 0.42, 0.82), vec3(0.01, 0.03, 0.10), h);
  vec3 hor = mix(vec3(0.72, 0.82, 0.93), vec3(0.28, 0.40, 0.62), h);
  float e = max(d.z, 0.0);
  vec3 c = mix(hor, zen, pow(e, 0.45));
  float s = max(dot(d, u_sun), 0.0);
  c += vec3(1.0, 0.85, 0.6) * pow(s, 8.0) * 0.25 + vec3(1.0, 0.95, 0.85) * pow(s, 900.0) * 4.0;
  if (d.z < 0.0) c = mix(hor * 0.92, vec3(0.45, 0.50, 0.52), clamp(-d.z * 4.0, 0.0, 1.0));
  return c;
}
void main() {
  vec4 p = u_invVP * vec4(v_uv * 2.0 - 1.0, 1.0, 1.0);
  vec3 d = normalize(p.xyz / p.w - u_cam);
  o = vec4(skyColor(d), 1.0);
}`;
const SKY_VS = `#version 300 es
out vec2 v_uv;
void main() { vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2); v_uv = p; gl_Position = vec4(p * 2.0 - 1.0, 0.9999999, 1.0); }`;

const TERRAIN_VS = `#version 300 es
precision highp float;
layout(location=0) in vec3 a_pos;
uniform mat4 u_vp; out vec3 v_world; out vec3 v_n;
${NOISE}
${LOGDEPTH_VS}
void main() {
  vec3 p = vec3(a_pos.xy, 0.0); p.z = terrainHeight(p.xy);
  // Smooth vertex normal (finite differences at the local mesh scale).
  float e = max(1.5, 0.03 * length(p.xy));
  float hx = terrainHeight(p.xy + vec2(e, 0.0)) - terrainHeight(p.xy - vec2(e, 0.0));
  float hy = terrainHeight(p.xy + vec2(0.0, e)) - terrainHeight(p.xy - vec2(0.0, e));
  v_n = normalize(vec3(-hx, -hy, 2.0 * e));
  v_world = p; gl_Position = u_vp * vec4(p, 1.0); logDepth();
}`;
const TERRAIN_FS = `#version 300 es
precision highp float;
precision highp sampler2DShadow;
in vec3 v_world; in vec3 v_n; out vec4 o;
uniform vec3 u_cam; uniform vec3 u_sun; uniform float u_alt; uniform float u_time;
uniform mat4 u_shadowVP; uniform sampler2DShadow u_shadow;
uniform vec3 u_plumePos; uniform float u_plumeI;
${NOISE}
${LOGDEPTH_FS}
vec3 fogColor(vec3 d) {
  float h = clamp(u_alt / 22000.0, 0.0, 1.0);
  return mix(vec3(0.70, 0.80, 0.91), vec3(0.30, 0.42, 0.62), h);
}
float shadowAt(vec3 p) {
  vec4 s = u_shadowVP * vec4(p, 1.0); vec3 c = s.xyz / s.w * 0.5 + 0.5;
  if (c.x <= 0.0 || c.x >= 1.0 || c.y <= 0.0 || c.y >= 1.0 || c.z >= 1.0) return 1.0;
  float sum = 0.0; vec2 texel = vec2(1.0 / 2048.0);
  for (int i = -1; i <= 1; i++) for (int j = -1; j <= 1; j++) sum += texture(u_shadow, vec3(c.xy + vec2(i, j) * texel, c.z - 0.0015));
  return sum / 9.0;
}
void main() {
  logDepth();
  vec3 p = v_world;
  vec3 n = normalize(v_n);
  float r = length(p.xy);
  float slope = 1.0 - n.z;
  float nz = fbm(p.xy / 38.0), nz2 = vnoise(p.xy / 4.0), big = fbm(p.xy / 700.0);
  vec3 grass = mix(vec3(0.20, 0.33, 0.12), vec3(0.34, 0.42, 0.18), nz);
  grass = mix(grass, vec3(0.45, 0.43, 0.26), smoothstep(0.55, 0.8, big) * 0.6);
  vec3 rock = mix(vec3(0.36, 0.34, 0.31), vec3(0.50, 0.48, 0.45), nz2);
  vec3 col = mix(grass, rock, smoothstep(0.18, 0.42, slope));
  col = mix(col, vec3(0.93, 0.95, 0.98), smoothstep(1300.0, 1700.0, p.z + nz * 200.0) * smoothstep(0.55, 0.2, slope));
  // ---- Landing zone: access road, apron, drainage trench, pad with markings,
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
    col = mix(col, vec3(0.08, 0.075, 0.07), clamp(scorch * 0.78 + streak * 0.28, 0.0, 0.9));
  }
  float sh = shadowAt(p + n * 0.05);
  float diff = max(dot(n, u_sun), 0.0) * sh;
  vec3 light = vec3(1.0, 0.96, 0.88) * diff * 1.05 + vec3(0.42, 0.50, 0.62) * (0.45 + 0.35 * n.z);
  vec3 lp = u_plumePos - p; float d2 = dot(lp, lp);
  light += vec3(1.0, 0.55, 0.18) * u_plumeI * max(dot(n, normalize(lp)), 0.0) * 900.0 / (d2 + 60.0);
  col *= light;
  float dist = length(p - u_cam);
  float fog = 1.0 - exp(-dist / mix(26000.0, 70000.0, clamp(u_alt / 12000.0, 0.0, 1.0)));
  o = vec4(mix(col, fogColor(normalize(p - u_cam)), fog), 1.0);
}`;

const LIT_VS = `#version 300 es
precision highp float;
layout(location=0) in vec3 a_pos; layout(location=1) in vec3 a_nrm; layout(location=2) in vec2 a_ex;
uniform mat4 u_vp; uniform mat4 u_model;
out vec3 v_world; out vec3 v_n; out vec3 v_nb; out vec2 v_ex; out vec3 v_pb;
${LOGDEPTH_VS}
void main() {
  vec4 w = u_model * vec4(a_pos, 1.0);
  v_world = w.xyz; v_n = normalize(mat3(u_model) * a_nrm); v_nb = a_nrm; v_ex = a_ex; v_pb = a_pos;
  gl_Position = u_vp * w; logDepth();
}`;
const LIT_FS = `#version 300 es
precision highp float;
in vec3 v_world; in vec3 v_n; in vec3 v_nb; in vec2 v_ex; in vec3 v_pb; out vec4 o;
uniform vec3 u_cam; uniform vec3 u_sun; uniform int u_mat; uniform vec3 u_color;
uniform vec3 u_plumePos; uniform float u_plumeI; uniform int u_cpMode; uniform vec3 u_flowB; uniform float u_sinA;
${NOISE}
${LOGDEPTH_FS}
vec3 cpColor(float cp) {
  vec3 neg = vec3(0.10, 0.35, 1.0), pos = vec3(1.0, 0.25, 0.10);
  return cp < 0.0 ? mix(vec3(1.0), neg, clamp(-cp / 2.5, 0.0, 1.0)) : mix(vec3(1.0), pos, clamp(cp, 0.0, 1.0));
}
void main() {
  logDepth();
  vec3 n = normalize(v_n); if (!gl_FrontFacing) n = -n;
  vec3 vd = normalize(u_cam - v_world);
  float z = v_pb.z; float az = v_ex.x;
  vec3 base = u_color; float shin = 30.0; float spec = 0.25;
  if (u_mat == 0) {
    base = vec3(0.90, 0.91, 0.92);
    float streak = vnoise(vec2(az * 60.0, z * 0.25)) * 0.6 + vnoise(vec2(az * 180.0, z * 1.3)) * 0.4;
    float soot = smoothstep(8.5, 0.8, z + streak * 3.0) * 0.85 + smoothstep(14.5, 3.0, z) * streak * 0.25;
    base = mix(base, vec3(0.10, 0.09, 0.085), clamp(soot, 0.0, 0.95));
    if (z > 14.95) { base = vec3(0.06, 0.065, 0.07); shin = 60.0; spec = 0.35; }
    if (z < 1.05) { base = vec3(0.12, 0.12, 0.13); }
  } else if (u_mat == 2) { shin = 80.0; spec = 0.6; }
  else if (u_mat == 3) {
    vec2 g = abs(fract(v_ex * vec2(7.0, 6.0)) - 0.5);
    base = mix(vec3(0.10, 0.10, 0.11), vec3(0.45, 0.46, 0.48), step(0.38, max(g.x, g.y)));
    spec = 0.5; shin = 50.0;
  }
  else if (u_mat == 5) { // weathered concrete (ground props)
    base = u_color * (0.82 + 0.26 * vnoise(v_world.xy * 1.7 + v_world.z * 1.3)) * (1.0 - 0.25 * smoothstep(0.6, 0.0, v_world.z));
    spec = 0.05; shin = 8.0;
  } else if (u_mat == 6) { // yellow / black hazard paint
    float st = step(0.5, fract((v_world.x + v_world.y + v_world.z) / 0.7));
    base = mix(vec3(0.92, 0.72, 0.10), vec3(0.05, 0.05, 0.05), st);
    spec = 0.2; shin = 20.0;
  } else if (u_mat == 8) { // corrugated steel (containers, shed walls)
    float rib = smoothstep(0.25, 0.75, abs(fract(v_ex.x * 22.0) - 0.5) * 2.0);
    base = u_color * (0.80 + 0.20 * rib) * (0.9 + 0.2 * vnoise(v_world.xy * 0.8 + v_world.z));
    spec = 0.3; shin = 35.0;
  }
  if (u_cpMode == 1 && (u_mat == 0)) {
    // Surface pressure coefficient: crossflow potential flow around the
    // cylinder (scaled by sin^2 alpha) + stagnation on the leading end.
    vec3 nb = normalize(v_nb);
    vec3 fperp = u_flowB - dot(u_flowB, vec3(0, 0, 1)) * vec3(0, 0, 1);
    float cp = 0.0;
    if (length(fperp) > 1e-3 && abs(nb.z) < 0.5) {
      float c = -dot(normalize(nb.xy), normalize(fperp.xy));
      float s2 = 1.0 - c * c;
      cp = u_sinA * u_sinA * (1.0 - 4.0 * s2);
      if (c < -0.2) cp = u_sinA * u_sinA * -0.6; // separated wake
    }
    float axial = dot(u_flowB, vec3(0, 0, 1));
    float leadZ = axial > 0.0 ? 0.0 : 18.0;
    cp += (1.0 - u_sinA * u_sinA) * smoothstep(3.0, 0.0, abs(z - leadZ)) * 0.9;
    base = cpColor(cp);
  }
  float diff = max(dot(n, u_sun), 0.0);
  vec3 h = normalize(u_sun + vd);
  float sp = pow(max(dot(n, h), 0.0), shin) * spec;
  float fres = pow(1.0 - max(dot(n, vd), 0.0), 4.0);
  vec3 amb = vec3(0.40, 0.47, 0.58) * (0.55 + 0.45 * n.z) + vec3(0.16, 0.14, 0.12) * (0.5 - 0.5 * n.z);
  vec3 col = base * (vec3(1.0, 0.96, 0.9) * diff * 1.1 + amb) + vec3(1.0, 0.97, 0.9) * sp * step(0.0, diff) + amb * fres * 0.35;
  vec3 lp = u_plumePos - v_world; float d2 = dot(lp, lp);
  col += base * vec3(1.0, 0.55, 0.18) * u_plumeI * max(dot(n, normalize(lp)), 0.0) * 260.0 / (d2 + 12.0);
  if (u_mat == 9) col = u_color; // unlit (markers)
  o = vec4(col, 1.0);
}`;

const DEPTH_FS = `#version 300 es
precision highp float; void main() {}`;
const DEPTH_VS = `#version 300 es
layout(location=0) in vec3 a_pos; uniform mat4 u_vp; uniform mat4 u_model;
void main() { gl_Position = u_vp * u_model * vec4(a_pos, 1.0); }`;

const PLUME_VS = `#version 300 es
precision highp float;
layout(location=0) in vec3 a_pos; layout(location=1) in vec3 a_nrm; layout(location=2) in vec2 a_ex;
uniform mat4 u_vp; uniform mat4 u_model; uniform float u_len; uniform float u_r0; uniform float u_expand;
out vec3 v_world; out vec3 v_n; out float v_s;
${LOGDEPTH_VS}
void main() {
  float s = clamp(a_pos.z, 0.0, 1.0);
  float r = u_r0 * (1.0 + u_expand * s) * (1.0 - 0.55 * pow(s, 3.0));
  vec3 p = vec3(a_pos.xy * r, s * u_len);
  vec4 w = u_model * vec4(p, 1.0);
  v_world = w.xyz; v_n = normalize(mat3(u_model) * vec3(a_nrm.xy, 0.0)); v_s = s;
  gl_Position = u_vp * w; logDepth();
}`;
const PLUME_FS = `#version 300 es
precision highp float;
in vec3 v_world; in vec3 v_n; in float v_s; out vec4 o;
uniform vec3 u_cam; uniform float u_time; uniform float u_int; uniform vec3 u_c0; uniform vec3 u_c1; uniform float u_len; uniform float u_diamonds;
${NOISE}
${LOGDEPTH_FS}
void main() {
  if (v_world.z < 0.05) discard;
  logDepth();
  vec3 vd = normalize(u_cam - v_world);
  float edge = pow(abs(dot(normalize(v_n), vd)), 1.3);
  float fade = pow(1.0 - v_s, 1.25) * smoothstep(0.0, 0.04, v_s);
  float flick = 0.85 + 0.3 * vnoise(vec2(v_s * 9.0 - u_time * 18.0, u_time * 3.0));
  float dia = 1.0 + u_diamonds * 0.8 * pow(max(0.0, cos(v_s * u_len / 2.3 * 6.2832)), 6.0) * (1.0 - v_s);
  vec3 c = mix(u_c0, u_c1, smoothstep(0.0, 0.8, v_s));
  o = vec4(c * edge * fade * flick * dia * u_int, 1.0);
}`;

const POINT_VS = `#version 300 es
precision highp float;
layout(location=0) in vec3 a_pos; layout(location=1) in vec4 a_col; layout(location=2) in float a_size;
uniform mat4 u_vp; uniform float u_scale;
out vec4 v_col;
${LOGDEPTH_VS}
void main() { gl_Position = u_vp * vec4(a_pos, 1.0); gl_PointSize = clamp(a_size * u_scale / gl_Position.w, 1.5, 512.0); v_col = a_col; logDepth(); }`;
const POINT_FS = `#version 300 es
precision highp float;
in vec4 v_col; out vec4 o;
${LOGDEPTH_FS}
void main() {
  vec2 d = gl_PointCoord * 2.0 - 1.0; float r2 = dot(d, d); if (r2 > 1.0) discard;
  logDepth();
  float a = v_col.a * pow(1.0 - r2, 1.6);
  o = vec4(v_col.rgb * a, a);
}`;

const LINE_VS = `#version 300 es
precision highp float;
layout(location=0) in vec3 a_pos; layout(location=1) in vec4 a_col;
uniform mat4 u_vp; out vec4 v_col;
${LOGDEPTH_VS}
void main() { gl_Position = u_vp * vec4(a_pos, 1.0); v_col = a_col; logDepth(); }`;
const LINE_FS = `#version 300 es
precision highp float;
in vec4 v_col; out vec4 o;
${LOGDEPTH_FS}
void main() { logDepth(); o = vec4(v_col.rgb * v_col.a, v_col.a); }`;

const SLICE_VS = `#version 300 es
precision highp float;
layout(location=0) in vec3 a_pos; layout(location=1) in vec3 a_nrm; layout(location=2) in vec2 a_ex;
uniform mat4 u_vp; uniform vec3 u_origin; uniform vec3 u_ax; uniform vec3 u_ay; uniform vec2 u_size;
out vec2 v_uv; out vec3 v_world;
${LOGDEPTH_VS}
void main() {
  v_uv = a_ex; vec3 p = u_origin + u_ax * (a_ex.x - 0.36) * u_size.x + u_ay * (a_ex.y - 0.5) * u_size.y;
  v_world = p; gl_Position = u_vp * vec4(p, 1.0); logDepth();
}`;
const SLICE_FS = `#version 300 es
precision highp float;
in vec2 v_uv; in vec3 v_world; out vec4 o;
uniform sampler2D u_tex; uniform sampler2D u_solid; uniform int u_mode; uniform float u_alpha;
${LOGDEPTH_FS}
vec3 diverge(float t) { // t in [-1,1]: blue - white - red
  return t < 0.0 ? mix(vec3(0.95), vec3(0.05, 0.35, 1.0), -t) : mix(vec3(0.95), vec3(1.0, 0.2, 0.05), t);
}
vec3 turbo(float t) {
  return clamp(vec3(0.13572138 + t * (4.61539260 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943)))),
                    0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604)))),
                    0.10667330 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973))))), 0.0, 1.0);
}
void main() {
  if (v_world.z < 0.05) discard;
  if (texture(u_solid, v_uv).r > 0.5) discard;
  logDepth();
  float v = texture(u_tex, v_uv).r;
  vec3 c; float a;
  if (u_mode == 2) { c = turbo(v); a = 0.55; }
  else { float t = (v - 0.5) * 2.0; c = diverge(t); a = smoothstep(0.06, 0.7, abs(t)) * 0.8; }
  float edge = smoothstep(0.0, 0.06, min(min(v_uv.x, 1.0 - v_uv.x), min(v_uv.y, 1.0 - v_uv.y)));
  vec2 q = (v_uv - vec2(0.42, 0.5)) * vec2(1.6, 2.2);
  edge *= 1.0 - smoothstep(0.55, 1.0, length(q));      // soft vignette around the booster
  a *= edge * u_alpha;
  o = vec4(c * a, a);
}`;

// ---------------------------------------------------------------------------
export class Scene {
  constructor(canvas) {
    const gl = canvas.getContext("webgl2", { antialias: true, alpha: false, preserveDrawingBuffer: true });
    if (!gl) throw new Error("WebGL2 is not available in this browser");
    this.gl = gl;
    this.canvas = canvas;
    this.sun = V.norm([-0.55, -0.62, 0.56]);
    this.progSky = createProgram(gl, SKY_VS, SKY_FS);
    this.progTerrain = createProgram(gl, TERRAIN_VS, TERRAIN_FS);
    this.progLit = createProgram(gl, LIT_VS, LIT_FS);
    this.progDepth = createProgram(gl, DEPTH_VS, DEPTH_FS);
    this.progPlume = createProgram(gl, PLUME_VS, PLUME_FS);
    this.progPoint = createProgram(gl, POINT_VS, POINT_FS);
    this.progLine = createProgram(gl, LINE_VS, LINE_FS);
    this.progSlice = createProgram(gl, SLICE_VS, SLICE_FS);
    const t = polarGrid(230, 200, 3.0, 1.0415);
    this.terrain = createMesh(gl, t.d, t.idx);
    this._buildRocket();
    this._buildPadProps();
    const pl = frustum(1, 1, 0, 1, 36, false);
    this.plumeMesh = createMesh(gl, pl.d, pl.idx);
    const q = { d: [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1], idx: [0, 1, 2, 0, 2, 3] };
    this.quad = createMesh(gl, q.d, q.idx);
    this.points = createDynamic(gl, 6000);
    this.lines = createDynamic(gl, 60000);
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
    // CFD textures.
    this.cfdTex = this._tex(); this.solidTex = this._tex();
    this.cfdMeta = null;
    this.time = 0;
  }

  _tex() {
    const gl = this.gl, t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, 1, 1, 0, gl.RED, gl.UNSIGNED_BYTE, new Uint8Array([128]));
    return t;
  }

  setCFD(meta, field, solid) {
    const gl = this.gl;
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.bindTexture(gl.TEXTURE_2D, this.cfdTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, meta.nx, meta.ny, 0, gl.RED, gl.UNSIGNED_BYTE, field);
    gl.bindTexture(gl.TEXTURE_2D, this.solidTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, meta.nx, meta.ny, 0, gl.RED, gl.UNSIGNED_BYTE, solid);
    this.cfdMeta = meta;
  }

  _buildRocket() {
    const gl = this.gl;
    const body = merge([frustum(1.5, 1.5, 0.0, 15.0, 48), frustum(1.5, 1.5, 15.0, 18.0, 48)]);
    this.meshBody = createMesh(gl, body.d, body.idx);
    const bell = frustum(0.86, 0.34, -1.6, 0.3, 32, false);
    this.meshBell = createMesh(gl, bell.d, bell.idx);
    const fins = [];
    for (let k = 0; k < 4; k++) {
      const a = (k * Math.PI) / 2;
      const f = box(1.35, 0.14, 1.2, 1.5 + 0.67, 0, 16.8);
      fins.push(transformPart(f, M4.rotAxis([0, 0, 1], a)));
    }
    const fm = merge(fins);
    this.meshFins = createMesh(gl, fm.d, fm.idx);
    const leg = merge([box(0.34, 0.26, 6.2, 0, 0, 3.1), box(1.0, 0.8, 0.12, 0, 0, 6.2)]);
    this.meshLeg = createMesh(gl, leg.d, leg.idx);
    const strut = box(0.16, 0.16, 1.0, 0, 0, 0.5);
    this.meshStrut = createMesh(gl, strut.d, strut.idx);
    const rcs = merge([0, 1, 2, 3].map((k) => transformPart(box(0.35, 0.35, 0.5, 1.58, 0, 17.3), M4.rotAxis([0, 0, 1], k * Math.PI / 2 + Math.PI / 4))));
    this.meshRcs = createMesh(gl, rcs.d, rcs.idx);
    const pole = merge([frustum(0.08, 0.08, 0, 6, 8), frustum(0.5, 0.15, 0, 2.6, 16, false)]);
    this.meshPole = createMesh(gl, frustum(0.08, 0.08, 0, 6, 8).d, frustum(0.08, 0.08, 0, 6, 8).idx);
    const sock = frustum(0.45, 0.14, 0, 2.6, 16, false);
    this.meshSock = createMesh(gl, sock.d, sock.idx);
    const sph = frustum(0.5, 0.5, -0.5, 0.5, 12);
    this.meshMarker = createMesh(gl, sph.d, sph.idx);
    void pole;
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

  // Pieces of the booster as (mesh, material, model matrix).
  rocketParts(st) {
    const base = M4.fromQuatPos(st.q, st.pos);
    const parts = [
      [this.meshBody, 0, base], [this.meshFins, 3, base], [this.meshRcs, 1, base],
    ];
    const [gx, gy] = [st.gimbal[0] * Math.PI / 180, st.gimbal[1] * Math.PI / 180];
    const pivot = M4.translate([0, 0, 0.3]), unpivot = M4.translate([0, 0, -0.3]);
    const g = M4.mul(M4.rotAxis([0, 1, 0], gy), M4.rotAxis([1, 0, 0], gx));
    parts.push([this.meshBell, 2, M4.mul(base, M4.mul(pivot, M4.mul(g, unpivot)))]);
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
        parts.push([this.meshStrut, 2, M4.mul(base, sm)]);
      }
    }
    return parts;
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.floor(this.canvas.clientWidth * dpr), h = Math.floor(this.canvas.clientHeight * dpr);
    if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; }
  }

  // ---------------------------------------------------------------- render
  render(cam, st, fx, opts) {
    const gl = this.gl;
    this.resize();
    this.time += 1 / 60;
    const W = this.canvas.width, H = this.canvas.height;
    const proj = M4.perspective(cam.fov, W / H, 0.3, FAR);
    const view = M4.lookAt(cam.eye, cam.at, [0, 0, 1]);
    const vp = M4.mul(proj, view);
    this.vp = vp; this.proj = proj; this.viewM = view; this.W = W; this.H = H;
    this.invVP = M4.invert(vp); this.logFC = LOG_FC;
    const parts = this.rocketParts(st);
    const engineOn = st.throttle > 0.01 && st.state === "FLYING";
    const R = (v) => rot(st.q, v);
    const exitPos = V.add(st.pos, R(V.add([0, 0, 0.3], this._thrustDirBody(st, -1.9))));
    const plumeI = engineOn ? st.throttle : 0.0;

    // ---- shadow pass (sun ortho around the booster)
    const focus = [st.pos[0], st.pos[1], Math.max(0, Math.min(st.pos[2], 60))];
    const sunEye = V.add(focus, V.mul(this.sun, 400));
    const sview = M4.lookAt(sunEye, focus, [0, 0, 1]);
    const sproj = M4.ortho(-45, 45, -45, 45, 1, 1200);
    const svp = M4.mul(sproj, sview);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.shadowFbo);
    gl.viewport(0, 0, this.shadowSize, this.shadowSize);
    gl.clear(gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.useProgram(this.progDepth.program);
    gl.uniformMatrix4fv(this.progDepth.u.u_vp, false, svp);
    if (st.pos[2] < 400) for (const [mesh, , m] of parts) { gl.uniformMatrix4fv(this.progDepth.u.u_model, false, m); drawMesh(gl, mesh); }
    gl.uniformMatrix4fv(this.progDepth.u.u_model, false, M4.ident());
    for (const p of this.padProps) drawMesh(gl, p.mesh);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, W, H);

    // ---- sky
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.disable(gl.DEPTH_TEST);
    gl.useProgram(this.progSky.program);
    gl.uniformMatrix4fv(this.progSky.u.u_invVP, false, M4.invert(vp));
    gl.uniform3fv(this.progSky.u.u_cam, cam.eye);
    gl.uniform3fv(this.progSky.u.u_sun, this.sun);
    gl.uniform1f(this.progSky.u.u_alt, cam.eye[2]);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.enable(gl.DEPTH_TEST);

    // ---- terrain
    const pt = this.progTerrain;
    gl.useProgram(pt.program);
    gl.uniformMatrix4fv(pt.u.u_vp, false, vp);
    gl.uniform1f(pt.u.u_logFC, LOG_FC);
    gl.uniform3fv(pt.u.u_cam, cam.eye);
    gl.uniform3fv(pt.u.u_sun, this.sun);
    gl.uniform1f(pt.u.u_alt, cam.eye[2]);
    gl.uniformMatrix4fv(pt.u.u_shadowVP, false, svp);
    gl.uniform3fv(pt.u.u_plumePos, V.add(exitPos, [0, 0, -2]));
    gl.uniform1f(pt.u.u_plumeI, plumeI * Math.max(0, 1 - exitPos[2] / 90));
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.shadowTex);
    gl.uniform1i(pt.u.u_shadow, 0);
    drawMesh(gl, this.terrain);

    // ---- booster + windsock + markers
    const pl = this.progLit;
    gl.useProgram(pl.program);
    gl.uniformMatrix4fv(pl.u.u_vp, false, vp);
    gl.uniform1f(pl.u.u_logFC, LOG_FC);
    gl.uniform3fv(pl.u.u_cam, cam.eye);
    gl.uniform3fv(pl.u.u_sun, this.sun);
    gl.uniform3fv(pl.u.u_plumePos, exitPos);
    gl.uniform1f(pl.u.u_plumeI, plumeI);
    // Air-flow direction in the body frame for surface-pressure colouring.
    const vrel = V.sub(st.vel, st.wind);
    const sp = V.len(vrel);
    const flowW = sp > 0.5 ? V.mul(vrel, -1 / sp) : [0, 0, 1];
    const Rm = M4.fromQuatPos(st.q, [0, 0, 0]);
    const flowB = [Rm[0] * flowW[0] + Rm[1] * flowW[1] + Rm[2] * flowW[2], Rm[4] * flowW[0] + Rm[5] * flowW[1] + Rm[6] * flowW[2], Rm[8] * flowW[0] + Rm[9] * flowW[1] + Rm[10] * flowW[2]];
    gl.uniform1i(pl.u.u_cpMode, 0);
    gl.uniform3fv(pl.u.u_flowB, flowB);
    gl.uniform1f(pl.u.u_sinA, Math.sin((st.aero.alpha || 0) * Math.PI / 180));
    const colors = { 1: [0.07, 0.075, 0.08], 2: [0.30, 0.31, 0.33], 3: [0.2, 0.2, 0.2], 4: [0.09, 0.09, 0.10] };
    for (const [mesh, mat, m] of parts) {
      gl.uniform1i(pl.u.u_mat, mat);
      gl.uniform3fv(pl.u.u_color, colors[mat] || [0.9, 0.9, 0.9]);
      gl.uniformMatrix4fv(pl.u.u_model, false, m);
      drawMesh(gl, mesh);
    }
    // Windsock at the apron edge.
    const wind = st.wind, ws = Math.hypot(wind[0], wind[1]);
    const sockBase = [62, -18, 0];
    gl.uniform1i(pl.u.u_mat, 2); gl.uniform3fv(pl.u.u_color, [0.7, 0.7, 0.72]);
    gl.uniformMatrix4fv(pl.u.u_model, false, M4.translate(sockBase)); drawMesh(gl, this.meshPole);
    const droop = Math.max(0.05, Math.min(1, ws / 10));
    const wdir = ws > 0.1 ? V.norm([wind[0], wind[1], 0]) : [1, 0, 0];
    const sockDir = V.norm(V.add(V.mul(wdir, droop), [0, 0, -(1 - droop)]));
    const sax = V.cross([0, 0, 1], sockDir), sang = Math.acos(Math.max(-1, Math.min(1, sockDir[2])));
    gl.uniform1i(pl.u.u_mat, 9); gl.uniform3fv(pl.u.u_color, [1.0, 0.45, 0.1]);
    gl.uniformMatrix4fv(pl.u.u_model, false, M4.mul(M4.translate([sockBase[0], sockBase[1], 5.8]), V.len(sax) > 1e-6 ? M4.rotAxis(sax, sang) : M4.ident()));
    drawMesh(gl, this.meshSock);
    // Landing-zone furniture (lights, masts, shed, tank, containers, cameras).
    this._drawPadProps(pl, this.time);

    // ---- transparent: lines/ribbons, CFD slice, particles, plume
    gl.enable(gl.BLEND);
    gl.depthMask(false);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    this._drawLines(fx.lines);
    this._drawPoints(fx.points, false);
    this._drawPoints(this._padGlowPoints(this.time), true);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    if (fx.fluid) {
      fx.fluid.draw(this, cam, fx.fluidMode, fx.fluidVolume);
      gl.depthMask(false);
      gl.enable(gl.BLEND);
    }
    // Plume (additive).
    if (engineOn) {
      gl.blendFunc(gl.ONE, gl.ONE);
      const pp = this.progPlume;
      gl.useProgram(pp.program);
      gl.uniformMatrix4fv(pp.u.u_vp, false, vp);
      gl.uniform1f(pp.u.u_logFC, LOG_FC);
      gl.uniform3fv(pp.u.u_cam, cam.eye);
      gl.uniform1f(pp.u.u_time, this.time);
      const dirW = R(this._thrustDirBody(st, -1));
      const zaxis = V.norm(dirW), ang = Math.acos(Math.max(-1, Math.min(1, zaxis[2])));
      const ax = V.cross([0, 0, 1], zaxis);
      const model = M4.mul(M4.translate(exitPos), V.len(ax) > 1e-6 ? M4.rotAxis(ax, ang) : (zaxis[2] < 0 ? M4.rotAxis([1, 0, 0], Math.PI) : M4.ident()));
      gl.uniformMatrix4fv(pp.u.u_model, false, model);
      const pAmb = Math.exp(-st.pos[2] / 8400);
      const L = (7 + 24 * st.throttle) * (1 + 1.6 * (1 - pAmb));
      const expand = 0.25 + 3.2 * (1 - pAmb);
      const layers = [
        [0.95, 2.6 + expand, [1.0, 0.36, 0.08], [0.55, 0.12, 0.03], 0.55, 0],
        [0.80, 1.4 + expand * 0.6, [1.0, 0.62, 0.22], [0.9, 0.3, 0.06], 0.8, 0],
        [0.55, 0.55 + expand * 0.3, [1.0, 0.95, 0.75], [1.0, 0.6, 0.2], 1.0, 1],
        [0.42, 0.2, [1.0, 1.0, 0.95], [1.0, 0.85, 0.5], 1.1, 1],
      ];
      for (const [r0, ex, c0, c1, it, dia] of layers) {
        gl.uniform1f(pp.u.u_len, L * (dia ? 0.55 : 1.0));
        gl.uniform1f(pp.u.u_r0, r0);
        gl.uniform1f(pp.u.u_expand, ex);
        gl.uniform3fv(pp.u.u_c0, c0);
        gl.uniform3fv(pp.u.u_c1, c1);
        gl.uniform1f(pp.u.u_int, it * (0.35 + 0.65 * st.throttle));
        gl.uniform1f(pp.u.u_diamonds, dia * pAmb);
        drawMesh(gl, this.plumeMesh);
      }
      this._drawPoints(fx.glow, true);
    }
    gl.depthMask(true);
    gl.disable(gl.BLEND);
  }

  _thrustDirBody(st, sign) {
    const gx = st.gimbal[0] * Math.PI / 180, gy = st.gimbal[1] * Math.PI / 180;
    const d = [Math.sin(gy), -Math.sin(gx) * Math.cos(gy), Math.cos(gx) * Math.cos(gy)];
    return V.mul(d, sign);
  }

  _drawPoints(list, additive) {
    if (!list || list.length === 0) return;
    const gl = this.gl, d = this.points;
    const n = Math.min(list.length, d.max);
    for (let i = 0; i < n; i++) {
      const p = list[i], o = i * 8;
      d.data[o] = p[0]; d.data[o + 1] = p[1]; d.data[o + 2] = p[2];
      d.data[o + 3] = p[3]; d.data[o + 4] = p[4]; d.data[o + 5] = p[5]; d.data[o + 6] = p[6]; d.data[o + 7] = p[7];
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

  _drawSlice(st, flowW, opts) {
    const gl = this.gl, m = this.cfdMeta;
    // Plane through the booster containing the airflow and the body axis.
    const axis = rot(st.q, [0, 0, 1]);
    const e = m.tail_first ? axis : V.mul(axis, -1);
    let ay = V.sub(e, V.mul(flowW, V.dot(e, flowW)));
    if (V.len(ay) < 1e-3) ay = V.norm(V.cross(flowW, Math.abs(flowW[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0]));
    ay = V.norm(ay);
    const cell = 18.0 / (m.nx * 0.30);
    const origin = V.add(st.pos, rot(st.q, [0, 0, 9.0]));
    const ps = this.progSlice;
    gl.useProgram(ps.program);
    gl.uniformMatrix4fv(ps.u.u_vp, false, this.vp);
    gl.uniform1f(ps.u.u_logFC, LOG_FC);
    gl.uniform3fv(ps.u.u_origin, origin);
    gl.uniform3fv(ps.u.u_ax, flowW);
    gl.uniform3fv(ps.u.u_ay, ay);
    gl.uniform2f(ps.u.u_size, m.nx * cell, m.ny * cell);
    gl.uniform1i(ps.u.u_mode, opts.cfdField === "speed" ? 2 : 1);
    gl.uniform1f(ps.u.u_alpha, 0.9);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.cfdTex); gl.uniform1i(ps.u.u_tex, 1);
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, this.solidTex); gl.uniform1i(ps.u.u_solid, 2);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.disable(gl.CULL_FACE);
    drawMesh(gl, this.quad);
    gl.activeTexture(gl.TEXTURE0);
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
