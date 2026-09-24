// Real-time 3-D flow around the booster on the GPU (WebGL2).
//
// Incompressible "stable fluids" on a 3-D grid stored as a 2-D atlas of
// z-slices.  The grid axes are FIXED to the world (x = east, y = north,
// z = up); the grid only translates with the booster, so the picture never
// rotates when the relative wind turns.  The far-field air velocity (wind
// minus grid velocity) enters through whichever faces are upstream, and a
// change of the grid's velocity is applied to the whole field as the
// fictitious acceleration of the moving frame.
//
//   velocity : semi-Lagrangian, RK2 back-trace, vorticity confinement,
//              Jacobi pressure projection (warm-started)
//   dye      : MacCormack advection with min/max clamping (as in fluid/),
//              injected by a "smoke rake" of coloured streak tubes upstream
//              of the booster, with periodic time-line gaps
//   exhaust  : density source in the plume, advected with the dye
// Outputs: dye / vorticity / speed volumes for the main view, a float
// cut-plane for the wind-tunnel window, and the surface-pressure force.
import { V, quatToMat3, createProgram } from "./gl.js";

const TRI_VS = `#version 300 es
layout(location=0) in vec2 a_p;
out vec2 v_ndc;
void main() { v_ndc = a_p; gl_Position = vec4(a_p, 0.0, 1.0); }`;
const COMPOSITE_FS = `#version 300 es
precision highp float;
in vec2 v_ndc; uniform sampler2D u_image; out vec4 o;
void main(){ o=texture(u_image,v_ndc*0.5+0.5); }`;

// Atlas helpers shared by all grid passes.
const COMMON = `
precision highp float;
precision highp int;
uniform ivec3 N; uniform int TX; uniform vec2 AS;
ivec3 cellOf(ivec2 px) { int tx = px.x / N.x; int ty = px.y / N.y; return ivec3(px.x - tx * N.x, px.y - ty * N.y, ty * TX + tx); }
ivec2 pxOf(ivec3 c) { return ivec2((c.z % TX) * N.x + c.x, (c.z / TX) * N.y + c.y); }
bool inGrid(ivec3 c) { return c.x >= 0 && c.y >= 0 && c.z >= 0 && c.x < N.x && c.y < N.y && c.z < N.z; }
bool inside(vec3 p) { return all(greaterThanEqual(p, vec3(0.5))) && all(lessThanEqual(p, vec3(N) - 0.5)); }
vec4 fetchC(sampler2D t, ivec3 c) { return texelFetch(t, pxOf(clamp(c, ivec3(0), N - 1)), 0); }
vec4 samp(sampler2D t, vec3 p) {
  p = clamp(p, vec3(0.5), vec3(N) - 0.5);
  float z = p.z - 0.5; float k0 = floor(z); float f = z - k0;
  int i0 = int(k0); int i1 = min(i0 + 1, N.z - 1);
  vec2 o0 = vec2(float((i0 % TX) * N.x), float((i0 / TX) * N.y));
  vec2 o1 = vec2(float((i1 % TX) * N.x), float((i1 / TX) * N.y));
  return mix(texture(t, (o0 + p.xy) / AS), texture(t, (o1 + p.xy) / AS), f);
}
`;
const CELL = `ivec3 thisCell() { return cellOf(ivec2(gl_FragCoord.xy)); }
`;

// Booster geometry (metres, body frame: z along the axis, base at 0).
const SDF = `
uniform mat3 u_c2bM; uniform vec3 u_c2bO;     // cell -> body
uniform vec4 u_c2wZ;                          // cell -> world z (dot(xyz, cell) + w)
uniform float u_legs;
float sdCyl(vec3 p, float r, float z0, float z1) {
  vec2 d = vec2(length(p.xy) - r, max(z0 - p.z, p.z - z1));
  return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}
float sdSeg(vec3 p, vec3 a, vec3 b, float r) { vec3 pa = p - a, ba = b - a; float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0); return length(pa - ba * h) - r; }
float sdBox(vec3 p, vec3 b) { vec3 q = abs(p) - b; return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0); }
float rocketSDF(vec3 p) {
  float d = sdCyl(p, 1.83, -0.1, 18.0);
  d = min(d, sdCyl(p, 0.95, -1.7, 0.0));
  for (int i = 0; i < 4; i++) {
    float a = 0.7853982 + float(i) * 1.5707963;
    vec2 cs = vec2(cos(a), sin(a));
    vec3 q = vec3(cs.x * p.x + cs.y * p.y, -cs.y * p.x + cs.x * p.y, p.z);
    d = min(d, sdBox(q - vec3(2.45, 0.0, 16.8), vec3(0.65, 0.55, 0.12)));
    float fr = mix(1.95, 5.9, u_legs), fz = mix(9.0, -1.9, u_legs);
    d = min(d, sdSeg(q, vec3(1.55, 0.0, 3.2), vec3(fr, 0.0, fz), 0.28));
  }
  return d;
}
vec3 bodyOf(vec3 cellPos) { return u_c2bM * cellPos + u_c2bO; }
float worldZ(vec3 cellPos) { return dot(u_c2wZ.xyz, cellPos) + u_c2wZ.w; }
`;

// ---- pass: obstacle / sources (RGBA8: r solid, g exhaust source, b ground)
const OBST_FS = `#version 300 es
${COMMON}
${CELL}
${SDF}
uniform vec3 u_jetBody0; uniform float u_jetOn;
out vec4 o;
void main() {
  ivec3 c = thisCell(); if (c.z >= N.z) { o = vec4(0); return; }
  vec3 cp = vec3(c) + 0.5;
  vec3 pb = bodyOf(cp);
  float solid = rocketSDF(pb) < 0.0 ? 1.0 : 0.0;
  float ground = worldZ(cp) < 0.0 ? 1.0 : 0.0;
  // Exhaust source: short cone below the nozzle (gimballed jet direction).
  vec3 rel = pb - vec3(0.0, 0.0, -1.7);
  vec3 jd = normalize(u_jetBody0);
  float along = dot(rel, jd);
  float rad = length(rel - jd * along);
  float jet = (along > 0.0 && along < 4.0 && rad < 1.0 + 0.25 * along) ? u_jetOn : 0.0;
  o = vec4(max(solid, ground), jet * (1.0 - solid) * (1.0 - ground), ground, 1.0);
}`;

// ---- pass: advect velocity + frame acceleration + sources + boundaries
const ADV_VEL_FS = `#version 300 es
${COMMON}
${CELL}
uniform sampler2D u_vel; uniform sampler2D u_obst;
uniform vec3 u_inflowOld; uniform vec3 u_inflow; uniform float u_scale; uniform vec3 u_shift; uniform vec3 u_jetVel;
out vec4 o;
vec3 velAt(vec3 p) { return inside(p) ? samp(u_vel, p).xyz : u_inflowOld; }
void main() {
  ivec3 c = thisCell(); if (c.z >= N.z) { o = vec4(0); return; }
  vec4 ob = fetchC(u_obst, c);
  if (ob.r > 0.5) { o = vec4(0.0); return; }
  vec3 p = vec3(c) + 0.5;
  vec3 v = fetchC(u_vel, c).xyz;
  vec3 back = p - velAt(p - 0.5 * v);
  vec3 nv = velAt(back) * u_scale + u_shift;
  // Upstream faces: prescribed free stream.
  bvec3 lo = equal(c, ivec3(0)), hi = equal(c, N - 1);
  if ((lo.x && u_inflow.x > 0.0) || (hi.x && u_inflow.x < 0.0) ||
      (lo.y && u_inflow.y > 0.0) || (hi.y && u_inflow.y < 0.0) ||
      (lo.z && u_inflow.z > 0.0) || (hi.z && u_inflow.z < 0.0)) nv = u_inflow;
  nv = mix(nv, u_jetVel, clamp(ob.g, 0.0, 1.0));
  o = vec4(nv, 0.0);
}`;

// ---- pass: curl (xyz) and |curl| (w)
const CURL_FS = `#version 300 es
${COMMON}
${CELL}
uniform sampler2D u_vel;
out vec4 o;
void main() {
  ivec3 c = thisCell(); if (c.z >= N.z) { o = vec4(0); return; }
  vec3 vxp = fetchC(u_vel, c + ivec3(1,0,0)).xyz, vxm = fetchC(u_vel, c - ivec3(1,0,0)).xyz;
  vec3 vyp = fetchC(u_vel, c + ivec3(0,1,0)).xyz, vym = fetchC(u_vel, c - ivec3(0,1,0)).xyz;
  vec3 vzp = fetchC(u_vel, c + ivec3(0,0,1)).xyz, vzm = fetchC(u_vel, c - ivec3(0,0,1)).xyz;
  vec3 w = 0.5 * vec3((vyp.z - vym.z) - (vzp.y - vzm.y), (vzp.x - vzm.x) - (vxp.z - vxm.z), (vxp.y - vxm.y) - (vyp.x - vym.x));
  o = vec4(w, length(w));
}`;

// ---- pass: vorticity confinement
const CONF_FS = `#version 300 es
${COMMON}
${CELL}
uniform sampler2D u_vel; uniform sampler2D u_curl; uniform sampler2D u_obst; uniform float u_eps; uniform float u_dt;
out vec4 o;
void main() {
  ivec3 c = thisCell(); if (c.z >= N.z) { o = vec4(0); return; }
  vec4 v = fetchC(u_vel, c);
  if (fetchC(u_obst, c).r > 0.5) { o = vec4(0.0); return; }
  vec3 g = 0.5 * vec3(fetchC(u_curl, c + ivec3(1,0,0)).w - fetchC(u_curl, c - ivec3(1,0,0)).w,
                      fetchC(u_curl, c + ivec3(0,1,0)).w - fetchC(u_curl, c - ivec3(0,1,0)).w,
                      fetchC(u_curl, c + ivec3(0,0,1)).w - fetchC(u_curl, c - ivec3(0,0,1)).w);
  float glen = length(g);
  vec3 f = glen > 1e-5 ? u_eps * cross(g / glen, fetchC(u_curl, c).xyz) : vec3(0.0);
  o = vec4(v.xyz + f * u_dt, 0.0);
}`;

// ---- pass: divergence
const DIV_FS = `#version 300 es
${COMMON}
${CELL}
uniform sampler2D u_vel; uniform sampler2D u_obst; uniform vec3 u_inflow;
out vec4 o;
vec3 vAt(ivec3 c) {
  if (!inGrid(c)) return u_inflow;
  if (fetchC(u_obst, c).r > 0.5) return vec3(0.0);
  return fetchC(u_vel, c).xyz;
}
void main() {
  ivec3 c = thisCell(); if (c.z >= N.z) { o = vec4(0); return; }
  float d = 0.5 * ((vAt(c + ivec3(1,0,0)).x - vAt(c - ivec3(1,0,0)).x) + (vAt(c + ivec3(0,1,0)).y - vAt(c - ivec3(0,1,0)).y) + (vAt(c + ivec3(0,0,1)).z - vAt(c - ivec3(0,0,1)).z));
  o = vec4(d, 0.0, 0.0, 0.0);
}`;

// Pressure at a neighbour: solid walls and upstream faces are Neumann,
// the remaining (outflow / side) faces are open, p = 0.
const P_AT = `
float pAt(ivec3 c, ivec3 d, float pc) {
  ivec3 n = c + d;
  if (!inGrid(n)) return dot(vec3(d), u_inflow) < -0.05 * length(u_inflow) ? pc : 0.0;
  if (fetchC(u_obst, n).r > 0.5) return pc;
  return fetchC(u_p, n).r;
}`;

// ---- pass: Jacobi pressure iteration
const JAC_FS = `#version 300 es
${COMMON}
${CELL}
uniform sampler2D u_p; uniform sampler2D u_div; uniform sampler2D u_obst; uniform vec3 u_inflow;
out vec4 o;
${P_AT}
void main() {
  ivec3 c = thisCell(); if (c.z >= N.z) { o = vec4(0); return; }
  if (fetchC(u_obst, c).r > 0.5) { o = vec4(0.0); return; }
  float pc = fetchC(u_p, c).r;
  float s = pAt(c, ivec3(1,0,0), pc) + pAt(c, ivec3(-1,0,0), pc) + pAt(c, ivec3(0,1,0), pc) + pAt(c, ivec3(0,-1,0), pc) + pAt(c, ivec3(0,0,1), pc) + pAt(c, ivec3(0,0,-1), pc);
  o = vec4((s - fetchC(u_div, c).r) / 6.0, 0.0, 0.0, 0.0);
}`;

// ---- pass: subtract pressure gradient
const PROJ_FS = `#version 300 es
${COMMON}
${CELL}
uniform sampler2D u_vel; uniform sampler2D u_p; uniform sampler2D u_obst; uniform vec3 u_inflow;
out vec4 o;
${P_AT}
void main() {
  ivec3 c = thisCell(); if (c.z >= N.z) { o = vec4(0); return; }
  if (fetchC(u_obst, c).r > 0.5) { o = vec4(0.0); return; }
  float pc = fetchC(u_p, c).r;
  vec3 g = 0.5 * vec3(pAt(c, ivec3(1,0,0), pc) - pAt(c, ivec3(-1,0,0), pc), pAt(c, ivec3(0,1,0), pc) - pAt(c, ivec3(0,-1,0), pc), pAt(c, ivec3(0,0,1), pc) - pAt(c, ivec3(0,0,-1), pc));
  o = vec4(fetchC(u_vel, c).xyz - g, 0.0);
}`;

// ---- dye, MacCormack step 1: plain semi-Lagrangian prediction
const DYE_PRED_FS = `#version 300 es
${COMMON}
${CELL}
uniform sampler2D u_dye; uniform sampler2D u_vel;
out vec4 o;
void main() {
  ivec3 c = thisCell(); if (c.z >= N.z) { o = vec4(0); return; }
  vec3 p = vec3(c) + 0.5;
  vec3 v = fetchC(u_vel, c).xyz;
  vec3 back = p - samp(u_vel, p - 0.5 * v).xyz;
  o = inside(back) ? samp(u_dye, back) : vec4(0.0);
}`;

// ---- dye, MacCormack step 2: error correction, clamping, rake sources
const DYE_CORR_FS = `#version 300 es
${COMMON}
${CELL}
uniform sampler2D u_dye; uniform sampler2D u_hat; uniform sampler2D u_vel; uniform sampler2D u_obst;
uniform vec3 u_rakeC; uniform vec3 u_e0; uniform vec3 u_e1; uniform vec3 u_e2;
uniform vec4 u_rake;          // band pitch, row pitch, sheet thickness, half width (cells)
uniform float u_gate; uniform float u_air;
out vec4 o;
vec3 hue(float h) {           // fluid/src/fluid.c fl_hue
  h -= floor(h); float x = h * 6.0;
  return clamp(vec3(abs(x - 3.0) - 1.0, 2.0 - abs(x - 2.0), 2.0 - abs(x - 4.0)), 0.0, 1.0);
}
void main() {
  ivec3 c = thisCell(); if (c.z >= N.z) { o = vec4(0); return; }
  vec4 ob = fetchC(u_obst, c);
  if (ob.r > 0.5) { o = vec4(0.0); return; }
  vec3 p = vec3(c) + 0.5;
  vec3 v = fetchC(u_vel, c).xyz;
  vec3 back = p - samp(u_vel, p - 0.5 * v).xyz;
  vec4 hat = fetchC(u_hat, c);
  vec4 d = hat;
  if (inside(back)) {
    vec3 fwd = p + v;
    vec4 tilde = inside(fwd) ? samp(u_hat, fwd) : hat;
    d = hat + 0.5 * (fetchC(u_dye, c) - tilde);
    ivec3 b0 = ivec3(floor(back - 0.5));
    vec4 mn = vec4(1e9), mx = vec4(-1e9);
    for (int k = 0; k < 8; k++) {
      vec4 s = fetchC(u_dye, b0 + ivec3(k & 1, (k >> 1) & 1, (k >> 2) & 1));
      mn = min(mn, s); mx = max(mx, s);
    }
    d = clamp(d, mn, mx);
  }
  d *= vec4(0.9994, 0.9994, 0.9994, 0.985);
  // Smoke rake: rows of streak tubes in a plane across the flow.  Row 0
  // lies in the cut-plane of the wind-tunnel window.
  vec3 r = p - u_rakeC;
  float al = abs(dot(r, u_e0));
  if (al < 1.0) {
    float s1 = dot(r, u_e1), s2 = dot(r, u_e2);
    float k = floor(s2 / u_rake.y + 0.5);
    float ds = abs(s2 - k * u_rake.y);
    if (abs(k) <= 1.0 && ds < u_rake.z && abs(s1) < u_rake.w) {
      float bi = floor(s1 / u_rake.x + 0.5);
      float fr = abs(s1 / u_rake.x - bi);                 // 0 at the tube axis, 0.5 between tubes
      float w = (1.0 - smoothstep(0.16, 0.27, fr)) * (1.0 - smoothstep(0.45, 1.0, ds / u_rake.z)) * (1.0 - smoothstep(0.4, 1.0, al));
      float nb = max(1.0, floor(u_rake.w / u_rake.x));
      float t = clamp((bi + nb) / (2.0 * nb), 0.0, 1.0);
      vec3 col = hue(0.70 * (1.0 - t)) * (k == 0.0 ? 1.0 : 0.62);
      d.rgb = mix(d.rgb, col * u_gate, w * u_air);
    }
  }
  d.a = max(d.a, ob.g);
  o = clamp(d, 0.0, 2.0);
}`;

// ---- surface pressure integration (one texel per surface sample)
const FORCE_FS = `#version 300 es
${COMMON}
${CELL}
uniform sampler2D u_p; uniform mat3 u_b2cM; uniform vec3 u_b2cO; uniform ivec2 u_n;
out vec4 o;
void main() {
  ivec2 id = ivec2(gl_FragCoord.xy);
  int ns = u_n.x - 2;                        // side stations; last two columns = end caps
  float a = 6.2831853 * (float(id.y) + 0.5) / float(u_n.y);
  vec3 radial = vec3(cos(a), sin(a), 0.0);
  vec3 nrm, pb; float area; float zs = -99.0;
  if (id.x < ns) {
    float z = -0.5 + 18.3 * (float(id.x) + 0.5) / float(ns);
    float r = z < 0.0 ? 1.0 : 1.83;
    nrm = radial; pb = vec3(radial.xy * (r + 0.9), z);          // just outside the wall
    area = 6.2831853 * r / float(u_n.y) * 18.3 / float(ns);
    zs = z;
  } else if (id.x == ns) {                                        // top (interstage) cap
    nrm = vec3(0.0, 0.0, 1.0); pb = vec3(radial.xy * 1.1, 18.9);
    area = 3.14159265 * 1.83 * 1.83 / float(u_n.y);
  } else {                                                        // base (engine) end
    nrm = vec3(0.0, 0.0, -1.0); pb = vec3(radial.xy * 1.1, -2.6);
    area = 3.14159265 * 1.83 * 1.83 / float(u_n.y);
  }
  float p = samp(u_p, u_b2cM * pb + u_b2cO).r;
  o = vec4(-p * nrm * area, zs);             // body-frame force (pressure units x m^2), station z
}`;

// Shared inferno ramp (fluid/src/render.c).
const INFERNO = `
vec3 inferno(float t) {
  t = clamp(t, 0.0, 1.0) * 7.0;
  vec3 k[8] = vec3[8](vec3(0,0,4), vec3(40,11,84), vec3(101,21,110), vec3(159,42,99), vec3(212,72,66), vec3(245,125,21), vec3(250,193,39), vec3(252,255,164));
  int i = min(int(t), 6);
  return mix(k[i], k[i + 1], t - float(i)) / 255.0;
}`;

// ---- volume rendering (main 3-D view)
const VOL_FS = `#version 300 es
${COMMON}
${CELL}
in vec2 v_ndc;
${INFERNO}
uniform sampler2D u_dye; uniform sampler2D u_curl; uniform sampler2D u_vel; uniform sampler2D u_obst;
uniform mat4 u_invVP; uniform vec3 u_cam;
uniform mat3 u_w2cM; uniform vec3 u_w2cO; uniform vec4 u_c2wZ;
uniform int u_mode; uniform float u_alpha; uniform float u_inflowMag;
out vec4 o;

void main() {
  vec4 far = u_invVP * vec4(v_ndc, 1.0, 1.0); far /= far.w;
  vec3 dw = normalize(far.xyz - u_cam);
  vec3 ro = u_w2cM * u_cam + u_w2cO;
  vec3 rd = u_w2cM * dw;                      // cell units per metre along the ray
  vec3 inv = 1.0 / rd;
  vec3 t0 = (vec3(0.0) - ro) * inv, t1 = (vec3(N) - ro) * inv;
  vec3 tmin = min(t0, t1), tmax = max(t0, t1);
  float ta = max(max(tmin.x, tmin.y), max(tmin.z, 0.0));
  float tb = min(min(tmax.x, tmax.y), tmax.z);
  if (tb <= ta) discard;
  const int STEPS = 96;
  float dt = (tb - ta) / float(STEPS);
  float stepCells = dt * length(rd);
  vec3 acc = vec3(0.0); float alpha = 0.0;
  float jitter = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
  for (int i = 0; i < STEPS; i++) {
    float t = ta + (float(i) + jitter) * dt;
    vec3 p = ro + rd * t;
    if (dot(u_c2wZ.xyz, p) + u_c2wZ.w < 0.0) break;               // ground
    vec4 ob = samp(u_obst, p);
    if (ob.r > 0.5 && ob.b < 0.5) break;                          // booster
    vec3 edge = min(p, vec3(N) - p);
    float fade = clamp(min(min(edge.x, edge.y), edge.z) / 4.0, 0.0, 1.0);
    vec3 col; float a;
    if (u_mode == 0) {           // coloured streak tubes + exhaust
      vec4 d = samp(u_dye, p);
      float dens = max(d.r, max(d.g, d.b));
      vec3 hueC = d.rgb / max(dens, 1e-3);
      col = mix(hueC, vec3(1.0), 0.18) * (0.55 + 0.6 * dens);
      col = mix(col, vec3(0.80, 0.80, 0.82), d.a / max(dens + d.a, 1e-3));
      a = 0.30 * smoothstep(0.04, 0.6, dens) + 0.10 * d.a;
    } else if (u_mode == 1) {    // vorticity magnitude
      float w = samp(u_curl, p).w / max(u_inflowMag, 0.05);
      float s = smoothstep(0.10, 0.9, w);
      col = mix(vec3(0.24, 0.59, 0.90), vec3(0.90, 0.35, 0.16), smoothstep(0.3, 1.3, w));
      a = 0.10 * s;
    } else {                     // speed relative to the free stream
      float sp = length(samp(u_vel, p).xyz) / max(u_inflowMag, 0.05);
      float dev = abs(sp - 1.0);
      col = inferno(sqrt(clamp(sp / 1.8, 0.0, 1.0)));
      a = 0.08 * smoothstep(0.10, 0.6, dev);
    }
    a *= fade * u_alpha * stepCells;
    a = clamp(a, 0.0, 1.0);
    acc += (1.0 - alpha) * col * a;
    alpha += (1.0 - alpha) * a;
    if (alpha > 0.97) break;
  }
  o = vec4(acc, alpha);
}`;

// ---- cut-plane for the wind-tunnel window (three float targets)
//   A = (u along the window's horizontal axis, u_z, |u|, p)          [cells/step]
//   B = (dye r, g, b, exhaust)
//   C = (vorticity normal to the window, flag, |grad tracer|, 0)
//   flag: 0 fluid, 1 outside the grid, 2 booster, 3 ground
// Derivatives come from the curl texture / +-1 cell differences of the
// trilinear field, so the picture stays smooth inside each cell.
const SLICE_FS = `#version 300 es
${COMMON}
${CELL}
uniform sampler2D u_vel; uniform sampler2D u_obst; uniform sampler2D u_p; uniform sampler2D u_dye; uniform sampler2D u_curl;
uniform vec2 u_size; uniform vec3 u_ax; uniform vec3 u_ay; uniform vec3 u_c0;
uniform vec3 u_eh; uniform vec3 u_en; uniform vec3 u_inflow;
layout(location=0) out vec4 oA;
layout(location=1) out vec4 oB;
layout(location=2) out vec4 oC;
float rho(vec3 p) { vec4 d = samp(u_dye, p); return d.r + d.g + d.b + d.a; }
void main() {
  vec2 uv = gl_FragCoord.xy / u_size;
  vec3 p = u_c0 + (uv.x - 0.5) * u_ax + (uv.y - 0.5) * u_ay;
  if (any(lessThan(p, vec3(0.0))) || any(greaterThan(p, vec3(N)))) {
    oA = vec4(dot(u_inflow, u_eh), u_inflow.z, length(u_inflow), 0.0);
    oB = vec4(0.0);
    oC = vec4(0.0, 1.0, 0.0, 0.0);
    return;
  }
  vec4 ob = samp(u_obst, p);
  vec3 v = samp(u_vel, p).xyz;
  oA = vec4(dot(v, u_eh), v.z, length(v), samp(u_p, p).r);
  oB = samp(u_dye, p);
  float flag = ob.b > 0.5 ? 3.0 : ob.r > 0.5 ? 2.0 : 0.0;
  vec3 ez = vec3(0.0, 0.0, 1.0);
  vec2 g = 0.5 * vec2(rho(p + u_eh) - rho(p - u_eh), rho(p + ez) - rho(p - ez));
  oC = vec4(dot(samp(u_curl, p).xyz, u_en), flag, length(g), 0.0);
}`;

function mat3cols(a, b, c) { return new Float32Array([a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]]); }
function mulM3(m, v) { return [m[0] * v[0] + m[3] * v[1] + m[6] * v[2], m[1] * v[0] + m[4] * v[1] + m[7] * v[2], m[2] * v[0] + m[5] * v[1] + m[8] * v[2]]; }

export class Fluid3D {
  constructor(gl, { nx = 64, ny = 64, nz = 96, cell = 1.0 } = {}) {
    this.gl = gl;
    this.ok = !!gl.getExtension("EXT_color_buffer_float");
    gl.getExtension("OES_texture_float_linear");
    if (!this.ok) { this.reason = "不支持 EXT_color_buffer_float"; return; }
    this.N = [nx, ny, nz];
    this.h = cell;
    this.E = [nx * cell, ny * cell, nz * cell];          // grid extent (m), world-aligned
    this.TX = Math.max(1, Math.floor(4096 / nx));
    this.TX = Math.min(this.TX, nz, Math.ceil(Math.sqrt(nz * ny / nx)) + 1);
    this.W = this.TX * nx;
    this.H = Math.ceil(nz / this.TX) * ny;
    const mk = (fs) => createProgram(gl, TRI_VS, `#version 300 es\n` + fs.replace(/^#version 300 es\n/, ""));
    this.pObst = mk(OBST_FS); this.pAdvV = mk(ADV_VEL_FS); this.pCurl = mk(CURL_FS); this.pConf = mk(CONF_FS);
    this.pDiv = mk(DIV_FS); this.pJac = mk(JAC_FS); this.pProj = mk(PROJ_FS);
    this.pDyeP = mk(DYE_PRED_FS); this.pDyeC = mk(DYE_CORR_FS);
    this.pForce = mk(FORCE_FS); this.pVol = mk(VOL_FS); this.pSlice = mk(SLICE_FS);
    this.pComposite = mk(COMPOSITE_FS);
    this.tri = gl.createVertexArray();
    gl.bindVertexArray(this.tri);
    const b = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
    const T = (w, h, fmt = gl.RGBA16F, type = gl.HALF_FLOAT, lin = true) => this._target(w, h, fmt, type, lin);
    this.vel = [T(this.W, this.H), T(this.W, this.H)];
    this.pr = [T(this.W, this.H), T(this.W, this.H)];
    this.dye = [T(this.W, this.H), T(this.W, this.H)];
    this.dyeHat = T(this.W, this.H);
    this.curl = T(this.W, this.H);
    this.div = T(this.W, this.H);
    this.obst = T(this.W, this.H, gl.RGBA8, gl.UNSIGNED_BYTE);
    this.FN = [48, 16];
    this.force = T(this.FN[0], this.FN[1], gl.RGBA32F, gl.FLOAT, false);
    this.forceBuf = new Float32Array(this.FN[0] * this.FN[1] * 4);
    this.sliceT = null;
    this.acc = 0; this.steps = 0; this.sps = 0; this._spsT = performance.now(); this._spsN = 0;
    this.flowTime = 0;            // physical seconds of flow simulated
    this.phase = 0;
    this.forceCoef = null;
    this.inflowMag = 1;
    this.uref = 20;
    this.planeNormal = [0, 1, 0]; // normal of the wind-tunnel window's cut-plane (world)
    this.reset();
  }

  _target(w, h, fmt, type, linear) {
    const gl = this.gl;
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texStorage2D(gl.TEXTURE_2D, 1, fmt, w, h);
    const f = linear ? gl.LINEAR : gl.NEAREST;
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, f);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, f);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) this.ok = false;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return { tex, fbo, w, h };
  }

  reset() {
    const gl = this.gl;
    for (const t of [...this.vel, ...this.pr, ...this.dye, this.dyeHat, this.curl, this.div]) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, t.fbo);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.clearColor(0, 0, 0, 1);
    this.uinfPrev = null;         // free stream at the last step (cells/step), null = fill on next step
    this.off = null; this.dir = null; this.lastMid = null;
  }

  _uniformsGrid(p) {
    const gl = this.gl;
    gl.uniform3i(p.u.N, this.N[0], this.N[1], this.N[2]);
    gl.uniform1i(p.u.TX, this.TX);
    gl.uniform2f(p.u.AS, this.W, this.H);
  }

  _bindTex(p, name, tex, unit) {
    const gl = this.gl;
    if (p.u[name] === undefined) return;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.uniform1i(p.u[name], unit);
  }

  _run(p, target, texs, set) {
    const gl = this.gl;
    gl.useProgram(p.program);
    gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
    gl.viewport(0, 0, target.w, target.h);
    this._uniformsGrid(p);
    let unit = 0;
    for (const [name, t] of Object.entries(texs)) this._bindTex(p, name, t.tex, unit++);
    if (set) set(p.u);
    gl.bindVertexArray(this.tri);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  // Fill the whole field with the free stream (first step after a reset).
  _fillFreeStream(u) {
    const gl = this.gl;
    for (const t of this.vel) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, t.fbo);
      gl.clearColor(u[0], u[1], u[2], 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
    }
    gl.clearColor(0, 0, 0, 1);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  // Advance the flow: st = interpolated state, dtReal = frame time (s).
  update(st, dtReal, warp = 1) {
    if (!this.ok || !st) return;
    const gl = this.gl;
    const h = this.h, E = this.E;
    const dtSim = dtReal * warp;
    const R = quatToMat3(st.q);
    const wind = st.wind || [0, 0, 0];
    const mid = V.add(st.pos, [R[0][2] * 8.5, R[1][2] * 8.5, R[2][2] * 8.5]);
    if (this.lastMid && V.len(V.sub(mid, this.lastMid)) > 80) this.reset();   // scenario change / teleport
    this.lastMid = mid;

    // Air motion relative to the booster; the grid sits so that the booster
    // is in the upstream part of the box (smoothed in simulation time).
    const vrel = V.sub(wind, st.vel);
    const sp = V.len(vrel);
    const dirNow = sp > 1.5 ? V.mul(vrel, 1 / sp) : (this.dir || [0, 0, 1]);
    const kd = this.dir ? 1 - Math.exp(-dtSim / 0.6) : 1;
    this.dir = this.dir ? V.norm(V.lerp(this.dir, dirNow, kd)) : dirNow;
    const target = [this.dir[0] * 0.2 * E[0], this.dir[1] * 0.2 * E[1], this.dir[2] * 0.2 * E[2]];
    const prevOff = this.off || target;
    const ko = this.off ? 1 - Math.exp(-dtSim / 1.5) : 1;
    this.off = V.lerp(prevOff, target, ko);
    const dOff = dtSim > 1e-6 ? V.mul(V.sub(this.off, prevOff), 1 / dtSim) : [0, 0, 0];

    const vGrid = V.add(st.vel, dOff);
    const uinfW = V.sub(wind, vGrid);                         // far-field air velocity in the grid frame (m/s)
    const uref = Math.max(V.len(uinfW), 20.0);
    const uinf = V.mul(uinfW, 1 / uref);                      // cells per step
    this.inflowMag = Math.max(V.len(uinf), 0.05);
    this.uinfW = uinfW;
    this.airSpeed = sp;

    // Geometry transforms (grid axes = world axes).
    const center = V.add(mid, this.off);
    const W0 = V.sub(center, V.mul(E, 0.5));
    this.W0 = W0; this.center = center; this.mid = mid;
    const Rm = new Float32Array([R[0][0], R[1][0], R[2][0], R[0][1], R[1][1], R[2][1], R[0][2], R[1][2], R[2][2]]); // body->world (col-major)
    const Rt = new Float32Array([R[0][0], R[0][1], R[0][2], R[1][0], R[1][1], R[1][2], R[2][0], R[2][1], R[2][2]]); // world->body
    this.c2bM = Rt.map((x) => x * h);
    this.c2bO = mulM3(Rt, V.sub(W0, st.pos));
    this.b2cM = Rm.map((x) => x / h);
    this.b2cO = V.mul(V.sub(st.pos, W0), 1 / h);
    this.w2cM = mat3cols([1 / h, 0, 0], [0, 1 / h, 0], [0, 0, 1 / h]);
    this.w2cO = V.mul(W0, -1 / h);
    this.c2wZ = [0, 0, h, W0[2]];

    // Exhaust: momentum source (cells/step).
    const gx = st.gimbal[0] * Math.PI / 180, gy = st.gimbal[1] * Math.PI / 180;
    const jetB = [-Math.sin(gy), Math.sin(gx) * Math.cos(gy), -Math.cos(gx) * Math.cos(gy)];
    const on = st.state === "FLYING" && st.throttle > 0.02 ? 1 : 0;
    const jetW = mulM3(Rm, jetB);
    const jetC = V.mul(jetW, on ? Math.min(3.0, 1.2 + 2.2 * st.throttle * (60 / (sp + 30))) : 0);

    this._run(this.pObst, this.obst, {}, (u) => {
      gl.uniformMatrix3fv(u.u_c2bM, false, this.c2bM); gl.uniform3fv(u.u_c2bO, this.c2bO);
      gl.uniform4fv(u.u_c2wZ, this.c2wZ); gl.uniform1f(u.u_legs, st.legs || 0);
      gl.uniform3fv(u.u_jetBody0, jetB); gl.uniform1f(u.u_jetOn, on);
    });

    // Smoke rake: a plane across the flow, upstream of the booster, inside the box.
    const e0 = this.dir;
    let e2 = V.sub(this.planeNormal, V.mul(e0, V.dot(this.planeNormal, e0)));
    if (V.len(e2) < 0.25) { e2 = V.sub([0, 0, 1], V.mul(e0, e0[2])); if (V.len(e2) < 0.25) e2 = V.sub([1, 0, 0], V.mul(e0, e0[0])); }
    e2 = V.norm(e2);
    const e1 = V.norm(V.cross(e2, e0));
    let tUp = 1e9;                                           // distance from mid to the upstream face
    for (let i = 0; i < 3; i++) {
      const rel = mid[i] - W0[i];
      if (e0[i] > 1e-4) tUp = Math.min(tUp, rel / e0[i]);
      else if (e0[i] < -1e-4) tUp = Math.min(tUp, (E[i] - rel) / -e0[i]);
    }
    const dRake = Math.max(4, Math.min(16, tUp - 3 * h));
    const rakeC = V.mul(V.sub(V.sub(mid, V.mul(e0, dRake)), W0), 1 / h);
    const band = Math.max(4.0, 4.2 / h), row = Math.max(6.0, 7.0 / h), thick = Math.max(1.5, 1.6 / h);
    const halfW = Math.min(17 / h, 0.42 * Math.min(this.N[0], this.N[1], this.N[2]));

    // Steps: physical time per step = h / uref.
    this.acc += dtSim * uref / h;
    let n = Math.floor(this.acc);
    this.acc -= n;
    n = Math.min(n, 3);
    if (n > 0 && this.uinfPrev === null) { this._fillFreeStream(uinf); this.uinfPrev = uinf; this.urefPrev = uref; }
    const dt = 1.0;
    let stepPhys = 0;
    for (let s = 0; s < n; s++) {
      const scale = s === 0 ? this.urefPrev / uref : 1;
      const inflowOld = s === 0 ? this.uinfPrev : uinf;
      const shift = V.sub(uinf, V.mul(inflowOld, scale));
      this._run(this.pAdvV, this.vel[1], { u_vel: this.vel[0], u_obst: this.obst }, (u) => {
        gl.uniform3fv(u.u_inflowOld, inflowOld); gl.uniform3fv(u.u_inflow, uinf);
        gl.uniform1f(u.u_scale, scale); gl.uniform3fv(u.u_shift, shift); gl.uniform3fv(u.u_jetVel, jetC);
      });
      this.vel.reverse();
      this._run(this.pCurl, this.curl, { u_vel: this.vel[0] });
      this._run(this.pConf, this.vel[1], { u_vel: this.vel[0], u_curl: this.curl, u_obst: this.obst }, (u) => {
        gl.uniform1f(u.u_eps, 0.30 * this.inflowMag); gl.uniform1f(u.u_dt, dt);
      });
      this.vel.reverse();
      this._run(this.pDiv, this.div, { u_vel: this.vel[0], u_obst: this.obst }, (u) => gl.uniform3fv(u.u_inflow, uinf));
      for (let k = 0; k < 24; k++) {
        this._run(this.pJac, this.pr[1], { u_p: this.pr[0], u_div: this.div, u_obst: this.obst }, (u) => gl.uniform3fv(u.u_inflow, uinf));
        this.pr.reverse();
      }
      this._run(this.pProj, this.vel[1], { u_vel: this.vel[0], u_p: this.pr[0], u_obst: this.obst }, (u) => gl.uniform3fv(u.u_inflow, uinf));
      this.vel.reverse();
      // Dye: MacCormack.  Time lines: short dark gaps every ~14 cells of travel.
      this.phase += this.inflowMag / 14;
      const gate = (this.phase % 1) < 0.11 ? 0.10 : 1.0;
      this._run(this.pDyeP, this.dyeHat, { u_dye: this.dye[0], u_vel: this.vel[0] });
      this._run(this.pDyeC, this.dye[1], { u_dye: this.dye[0], u_hat: this.dyeHat, u_vel: this.vel[0], u_obst: this.obst }, (u) => {
        gl.uniform3fv(u.u_rakeC, rakeC); gl.uniform3fv(u.u_e0, e0); gl.uniform3fv(u.u_e1, e1); gl.uniform3fv(u.u_e2, e2);
        gl.uniform4f(u.u_rake, band, row, thick, halfW);
        gl.uniform1f(u.u_gate, gate); gl.uniform1f(u.u_air, Math.min(1, sp / 2));
      });
      this.dye.reverse();
      this.uinfPrev = uinf; this.urefPrev = uref;
      stepPhys += h / uref;
      this.steps++; this._spsN++;
    }
    if (n > 0) this._run(this.pCurl, this.curl, { u_vel: this.vel[0] });   // curl of the projected field (display)
    this.uref = uref;
    this.flowTime += stepPhys;
    const now = performance.now();
    if (now - this._spsT > 1000) { this.sps = this._spsN * 1000 / (now - this._spsT); this._spsN = 0; this._spsT = now; }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  // Surface-pressure force on the booster, as a coefficient vector
  // C = F / (q * A_ref), A_ref = frontal area; zcp = centre of pressure of
  // the side (normal) force, metres from the base.
  readForce(st) {
    if (!this.ok || !this.b2cM || this.steps < 5) return null;
    const gl = this.gl;
    this._run(this.pForce, this.force, { u_p: this.pr[0] }, (u) => {
      gl.uniformMatrix3fv(u.u_b2cM, false, this.b2cM); gl.uniform3fv(u.u_b2cO, this.b2cO); gl.uniform2i(u.u_n, this.FN[0], this.FN[1]);
    });
    gl.readPixels(0, 0, this.FN[0], this.FN[1], gl.RGBA, gl.FLOAT, this.forceBuf);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    const B = this.forceBuf;
    let fx = 0, fy = 0, fz = 0, sx = 0, sy = 0;
    for (let i = 0; i < this.FN[0] * this.FN[1]; i++) {
      fx += B[i * 4]; fy += B[i * 4 + 1]; fz += B[i * 4 + 2];
      if (B[i * 4 + 3] > -50) { sx += B[i * 4]; sy += B[i * 4 + 1]; }
    }
    // Centre of pressure of the side force along the axis.
    let zcp = null;
    const sn = Math.hypot(sx, sy);
    if (sn > 1e-9) {
      const nx = sx / sn, ny = sy / sn;
      let m = 0, f = 0;
      for (let i = 0; i < this.FN[0] * this.FN[1]; i++) {
        if (B[i * 4 + 3] <= -50) continue;
        const fn = B[i * 4] * nx + B[i * 4 + 1] * ny;
        m += fn * B[i * 4 + 3]; f += fn;
      }
      if (Math.abs(f) > 1e-9) zcp = Math.max(-2, Math.min(20, m / f));
    }
    const scale = 2.0 / (this.inflowMag * this.inflowMag) / (Math.PI * 1.83 * 1.83);
    const cb = [fx * scale, fy * scale, fz * scale];
    if (!cb.every(Number.isFinite)) return null;
    const R = quatToMat3(st.q);
    const cw = [R[0][0] * cb[0] + R[0][1] * cb[1] + R[0][2] * cb[2], R[1][0] * cb[0] + R[1][1] * cb[1] + R[1][2] * cb[2], R[2][0] * cb[0] + R[2][1] * cb[1] + R[2][2] * cb[2]];
    this.forceCoef = { body: cb, world: cw, cn: Math.hypot(cb[0], cb[1]), ca: -cb[2], zcp };
    return this.forceCoef;
  }

  // Coloured fluid volume, drawn after opaque geometry.
  draw(scene, cam, mode, showVolume) {
    if (!this.ok || !this.w2cM || !showVolume) return;
    const gl = scene.gl;
    // Ray march at half resolution and composite.
    const width = Math.max(1, Math.ceil(gl.drawingBufferWidth / 2));
    const height = Math.max(1, Math.ceil(gl.drawingBufferHeight / 2));
    if (!this.volume || this.volume.w !== width || this.volume.h !== height) {
      if (this.volume) { gl.deleteTexture(this.volume.tex); gl.deleteFramebuffer(this.volume.fbo); }
      this.volume = this._target(width, height, gl.RGBA8, gl.UNSIGNED_BYTE, true);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.volume.fbo);
    gl.viewport(0, 0, width, height);
    gl.disable(gl.BLEND); gl.disable(gl.DEPTH_TEST);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    const p = this.pVol;
    gl.useProgram(p.program);
    this._uniformsGrid(p);
    this._bindTex(p, "u_dye", this.dye[0].tex, 0);
    this._bindTex(p, "u_curl", this.curl.tex, 1);
    this._bindTex(p, "u_vel", this.vel[0].tex, 2);
    this._bindTex(p, "u_obst", this.obst.tex, 3);
    gl.uniformMatrix4fv(p.u.u_invVP, false, scene.invVP);
    gl.uniform3fv(p.u.u_cam, cam.eye);
    gl.uniformMatrix3fv(p.u.u_w2cM, false, this.w2cM); gl.uniform3fv(p.u.u_w2cO, this.w2cO);
    gl.uniform4fv(p.u.u_c2wZ, this.c2wZ);
    gl.uniform1i(p.u.u_mode, mode);
    gl.uniform1f(p.u.u_alpha, 1.0);
    gl.uniform1f(p.u.u_inflowMag, this.inflowMag);
    gl.bindVertexArray(this.tri);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.clearColor(0, 0, 0, 1);
    gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
    gl.useProgram(this.pComposite.program);
    this._bindTex(this.pComposite, "u_image", this.volume.tex, 0);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.enable(gl.DEPTH_TEST);
    gl.activeTexture(gl.TEXTURE0);
  }

  _sliceTarget(w, h) {
    const gl = this.gl;
    if (this.sliceT && this.sliceT.w === w && this.sliceT.h === h) return this.sliceT;
    if (this.sliceT) { for (const t of this.sliceT.tex) gl.deleteTexture(t); gl.deleteFramebuffer(this.sliceT.fbo); }
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    const tex = [0, 1, 2].map((i) => {
      const t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, w, h);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, t, 0);
      return t;
    });
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2]);
    const ok = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.sliceT = { fbo, tex, w, h, ok, A: new Float32Array(w * h * 4), B: new Float32Array(w * h * 4), C: new Float32Array(w * h * 4) };
    return this.sliceT;
  }

  // Vertical cut-plane in world axes.  region = {center (world), eh (unit,
  // horizontal axis of the window), spanW, spanH (m), sw, sh (samples)}.
  readSlice(region) {
    if (!this.ok || !this.W0 || this.uinfPrev === null) return null;
    const gl = this.gl, h = this.h;
    const T = this._sliceTarget(region.sw, region.sh);
    if (!T.ok) return null;
    const eh = region.eh, en = V.cross(eh, [0, 0, 1]);
    const c0 = V.mul(V.sub(region.center, this.W0), 1 / h);
    const ax = V.mul(eh, region.spanW / h), ay = [0, 0, region.spanH / h];
    const p = this.pSlice;
    gl.useProgram(p.program);
    gl.bindFramebuffer(gl.FRAMEBUFFER, T.fbo);
    gl.viewport(0, 0, T.w, T.h);
    this._uniformsGrid(p);
    this._bindTex(p, "u_vel", this.vel[0].tex, 0);
    this._bindTex(p, "u_obst", this.obst.tex, 1);
    this._bindTex(p, "u_p", this.pr[0].tex, 2);
    this._bindTex(p, "u_dye", this.dye[0].tex, 3);
    this._bindTex(p, "u_curl", this.curl.tex, 4);
    gl.uniform2f(p.u.u_size, T.w, T.h); gl.uniform3fv(p.u.u_ax, ax); gl.uniform3fv(p.u.u_ay, ay); gl.uniform3fv(p.u.u_c0, c0);
    gl.uniform3fv(p.u.u_eh, eh); gl.uniform3fv(p.u.u_en, en); gl.uniform3fv(p.u.u_inflow, this.uinfPrev);
    gl.bindVertexArray(this.tri);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    gl.readPixels(0, 0, T.w, T.h, gl.RGBA, gl.FLOAT, T.A);
    gl.readBuffer(gl.COLOR_ATTACHMENT1);
    gl.readPixels(0, 0, T.w, T.h, gl.RGBA, gl.FLOAT, T.B);
    gl.readBuffer(gl.COLOR_ATTACHMENT2);
    gl.readPixels(0, 0, T.w, T.h, gl.RGBA, gl.FLOAT, T.C);
    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return {
      w: T.w, h: T.h, A: T.A, B: T.B, C: T.C, region,
      uref: this.urefPrev || this.uref, cell: h, uinfW: this.uinfW, inflow: this.uinfPrev,
      flowTime: this.flowTime,
    };
  }

  // Grid box and offset for the window (world metres).
  get box() { return this.W0 ? { min: this.W0, max: V.add(this.W0, this.E), center: this.center, mid: this.mid } : null; }
  get extent() { return this.E; }
}
