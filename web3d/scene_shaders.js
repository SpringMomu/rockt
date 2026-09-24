// GLSL sources for the 3-D renderer.  Linear-light HDR pipeline:
// every shader outputs scene-referred linear radiance into an RGBA16F MSAA
// target; bloom, ACES filmic tone mapping and sRGB encoding happen in the
// final composite.  Albedos authored in sRGB are decoded with pow(c, 2.2).

export const FAR = 160000.0;
export const LOG_FC = 2.0 / Math.log2(FAR + 1.0);

export const LOGDEPTH_VS = `
out float v_logz;
void logDepth() { v_logz = 1.0 + gl_Position.w; }`;
export const LOGDEPTH_FS = `
in float v_logz;
uniform float u_logFC;
void logDepth() { gl_FragDepth = log2(v_logz) * u_logFC * 0.5; }`;

export const NOISE = `
float hash(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
float vnoise(vec2 p) {
  vec2 i = floor(p), f = fract(p); vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1, 0)), u.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), u.x), u.y);
}
float fbm(vec2 p) { float s = 0.0, a = 0.5; for (int i = 0; i < 6; i++) { s += a * vnoise(p); p = p * 2.03 + vec2(17.1, 9.2); a *= 0.5; } return s; }
float fbm3o(vec2 p) { float s = 0.0, a = 0.5; for (int i = 0; i < 3; i++) { s += a * vnoise(p); p = p * 2.03 + vec2(17.1, 9.2); a *= 0.5; } return s / 0.875; }
float terrainHeight(vec2 p) {
  float r = length(p);
  float flat_ = smoothstep(420.0, 1100.0, r);
  float hills = (fbm(p / 2600.0) - 0.45) * 380.0;
  float ridge = 1.0 - abs(vnoise(p / 9000.0) * 2.0 - 1.0);
  float mountains = pow(ridge, 3.0) * 2300.0 * smoothstep(9000.0, 26000.0, r);
  return flat_ * (max(hills, -30.0) + mountains) - 0.02;
}`;

export const NOISE3 = `
float hash13(vec3 p3) { p3 = fract(p3 * 0.1031); p3 += dot(p3, p3.zyx + 31.32); return fract((p3.x + p3.y) * p3.z); }
float noise3(vec3 x) {
  vec3 i = floor(x), f = fract(x); f = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix(hash13(i), hash13(i + vec3(1, 0, 0)), f.x), mix(hash13(i + vec3(0, 1, 0)), hash13(i + vec3(1, 1, 0)), f.x), f.y),
             mix(mix(hash13(i + vec3(0, 0, 1)), hash13(i + vec3(1, 0, 1)), f.x), mix(hash13(i + vec3(0, 1, 1)), hash13(i + vec3(1, 1, 1)), f.x), f.y), f.z);
}
float fbm3(vec3 p) { float s = 0.0, a = 0.5; for (int i = 0; i < 4; i++) { s += a * noise3(p); p = p * 2.03 + vec3(3.1, 1.7, 4.3); a *= 0.5; } return s / 0.9375; }
float fbm3l(vec3 p) { float s = 0.0, a = 0.5; for (int i = 0; i < 2; i++) { s += a * noise3(p); p = p * 2.07 + vec3(3.1, 1.7, 4.3); a *= 0.5; } return s / 0.75; }
// Planck-locus colour (linear RGB, max component 1) for a temperature in K.
vec3 blackbody(float T) {
  float t = clamp(T, 900.0, 12000.0) / 100.0;
  float r = t <= 66.0 ? 1.0 : clamp(1.292936186 * pow(t - 60.0, -0.1332047592), 0.0, 1.0);
  float g = t <= 66.0 ? clamp(0.3900815788 * log(t) - 0.6318414438, 0.0, 1.0) : clamp(1.129890861 * pow(t - 60.0, -0.0755148492), 0.0, 1.0);
  float b = t >= 66.0 ? 1.0 : (t <= 19.0 ? 0.0 : clamp(0.5432067891 * log(t - 10.0) - 1.19625408914, 0.0, 1.0));
  return pow(vec3(r, g, b), vec3(2.2));
}`;

export const SKY = `
const vec3 SUN_E = vec3(3.05, 2.92, 2.70);
vec3 skyRad(vec3 d, vec3 sun, float alt) {
  float h = clamp(alt / 22000.0, 0.0, 1.0);
  float mu = d.z, m = max(mu, 0.0);
  vec3 zen = mix(vec3(0.050, 0.135, 0.42), vec3(0.003, 0.009, 0.04), h);
  vec3 hor = mix(vec3(0.44, 0.55, 0.70), vec3(0.10, 0.17, 0.32), h);
  float t = pow(1.0 - m, 5.0);
  vec3 c = mix(zen, hor, t);
  c = mix(c, hor * 1.08, pow(1.0 - m, 24.0) * 0.7);
  float cs = dot(d, sun);
  c *= 0.82 + 0.18 * (1.0 + cs * cs);
  float g = 0.78, mie = (1.0 - g * g) / pow(1.0 + g * g - 2.0 * g * cs, 1.5);
  c += vec3(1.0, 0.88, 0.72) * mie * 0.010 * (0.35 + t) * (1.0 - 0.8 * h);
  if (mu < 0.0) c = mix(hor, vec3(0.16, 0.18, 0.16), clamp(-mu * 6.0, 0.0, 1.0));
  return c;
}
// Hemispherical ambient irradiance (already divided by pi): sky above, ground bounce below.
vec3 hemiAmb(float nz) { return mix(vec3(0.085, 0.085, 0.070), vec3(0.17, 0.26, 0.42), 0.5 + 0.5 * nz); }`;

export const PBR = `
float D_GGX(float NdH, float a) { float a2 = a * a; float d = NdH * NdH * (a2 - 1.0) + 1.0; return a2 / (3.14159 * d * d); }
float V_Smith(float NdV, float NdL, float a) { float k = a * 0.5; return 0.25 / ((NdV * (1.0 - k) + k) * (NdL * (1.0 - k) + k)); }
vec3 F_Schlick(vec3 F0, float c) { return F0 + (1.0 - F0) * pow(1.0 - c, 5.0); }
// Radiance leaving the surface for irradiance E along l (diffuse albedo is not divided by pi; spec scaled accordingly).
vec3 brdfLight(vec3 n, vec3 v, vec3 l, vec3 E, vec3 dif, vec3 F0, float rough) {
  float NdL = dot(n, l); if (NdL <= 0.0) return vec3(0.0);
  vec3 h = normalize(l + v);
  float NdV = max(dot(n, v), 1e-3), NdH = max(dot(n, h), 0.0), VdH = max(dot(v, h), 0.0);
  float a = max(rough * rough, 0.003);
  vec3 F = F_Schlick(F0, VdH);
  vec3 spec = D_GGX(NdH, a) * V_Smith(NdV, NdL, a) * F;
  return (dif * (1.0 - F) + spec * 3.14159) * E * NdL;
}`;

// ------------------------------------------------------------------ sky
export const SKY_VS = `#version 300 es
out vec2 v_uv;
void main() { vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2); v_uv = p; gl_Position = vec4(p * 2.0 - 1.0, 0.9999999, 1.0); }`;
export const SKY_FS = `#version 300 es
precision highp float;
in vec2 v_uv; out vec4 o;
uniform mat4 u_invVP; uniform vec3 u_cam; uniform vec3 u_sun; uniform float u_alt;
${SKY}
void main() {
  vec4 p = u_invVP * vec4(v_uv * 2.0 - 1.0, 1.0, 1.0);
  vec3 d = normalize(p.xyz / p.w - u_cam);
  vec3 c = skyRad(d, u_sun, u_alt);
  float cs = dot(d, u_sun);
  c += vec3(1.0, 0.96, 0.9) * smoothstep(0.999955, 0.99998, cs) * 400.0;     // solar disc (0.53 deg)
  c += vec3(1.0, 0.93, 0.82) * pow(max(cs, 0.0), 600.0) * 6.0;               // aureole
  o = vec4(c, 1.0);
}`;

// ------------------------------------------------------------------ terrain
export const TERRAIN_VS = `#version 300 es
precision highp float;
layout(location=0) in vec3 a_pos;
uniform mat4 u_vp; out vec3 v_world; out vec3 v_n;
${NOISE}
${LOGDEPTH_VS}
void main() {
  vec3 p = vec3(a_pos.xy, 0.0); p.z = terrainHeight(p.xy);
  float e = max(1.5, 0.03 * length(p.xy));
  float hx = terrainHeight(p.xy + vec2(e, 0.0)) - terrainHeight(p.xy - vec2(e, 0.0));
  float hy = terrainHeight(p.xy + vec2(0.0, e)) - terrainHeight(p.xy - vec2(0.0, e));
  v_n = normalize(vec3(-hx, -hy, 2.0 * e));
  v_world = p; gl_Position = u_vp * vec4(p, 1.0); logDepth();
}`;

export function terrainFS(markings) {
  return `#version 300 es
precision highp float;
precision highp sampler2DShadow;
in vec3 v_world; in vec3 v_n; out vec4 o;
uniform vec3 u_cam; uniform vec3 u_sun; uniform float u_alt; uniform float u_time;
uniform mat4 u_shadowVP; uniform sampler2DShadow u_shadow;
uniform vec3 u_l0Pos, u_l0Col, u_l1Pos, u_l1Col;
uniform vec3 u_hit; uniform float u_Rw; uniform float u_heat;
${NOISE}
${SKY}
${LOGDEPTH_FS}
float shadowAt(vec3 p) {
  vec4 s = u_shadowVP * vec4(p, 1.0); vec3 c = s.xyz / s.w * 0.5 + 0.5;
  if (c.x <= 0.0 || c.x >= 1.0 || c.y <= 0.0 || c.y >= 1.0 || c.z >= 1.0) return 1.0;
  float sum = 0.0; vec2 texel = vec2(1.0 / 2048.0);
  for (int i = -1; i <= 1; i++) for (int j = -1; j <= 1; j++) sum += texture(u_shadow, vec3(c.xy + vec2(i, j) * texel * 1.3, c.z - 0.0012));
  return sum / 9.0;
}
vec3 pLight(vec3 lp, vec3 lc, vec3 p, vec3 n) {
  vec3 L = lp - p; float d2 = dot(L, L);
  return lc * max(dot(n, L * inversesqrt(d2)), 0.0) / (d2 + 2.0);
}
void main() {
  logDepth();
  vec3 p = v_world;
  vec3 n = normalize(v_n);
  float r = length(p.xy);
  float slope = 1.0 - n.z;
  float nz = fbm(p.xy / 38.0), nz2 = vnoise(p.xy / 4.0), big = fbm(p.xy / 700.0);
  float dist = length(p - u_cam);
  // Coastal scrub / grass with dry sandy patches, detail fading with distance.
  float det = fbm3o(p.xy / 1.3) * smoothstep(400.0, 40.0, dist);
  vec3 grass = mix(vec3(0.27, 0.31, 0.17), vec3(0.37, 0.39, 0.23), nz);
  grass *= 0.86 + 0.28 * det;
  grass = mix(grass, vec3(0.46, 0.44, 0.32), smoothstep(0.5, 0.85, big) * 0.35);
  grass = mix(grass, vec3(0.56, 0.52, 0.42), smoothstep(0.62, 0.75, fbm(p.xy / 90.0 + 4.0)) * 0.22 * smoothstep(1500.0, 300.0, r));
  vec3 rock = mix(vec3(0.40, 0.38, 0.35), vec3(0.52, 0.50, 0.47), nz2);
  vec3 col = mix(grass, rock, smoothstep(0.18, 0.42, slope));
  col = mix(col, vec3(0.93, 0.95, 0.98), smoothstep(1300.0, 1700.0, p.z + nz * 200.0) * smoothstep(0.55, 0.2, slope));
${markings}
  vec3 alb = pow(col, vec3(2.2));
  float sh = shadowAt(p + n * 0.05);
  float ndl = max(dot(n, u_sun), 0.0);
  vec3 light = SUN_E * ndl * sh + hemiAmb(n.z);
  light += pLight(u_l0Pos, u_l0Col, p, n) + pLight(u_l1Pos, u_l1Col, p, n);
  // Radiant heating of the pad by the wall jet (area light hugging the ground).
  float rho = length(p.xy - u_hit.xy);
  light += u_l1Col * 0.004 * exp(-rho * rho / max(1.0, u_Rw * u_Rw));
  vec3 c = alb * light;
  // Incandescent stagnation spot while the flame is on the concrete.
  c += vec3(1.0, 0.42, 0.12) * u_heat * 0.25 * exp(-rho * rho / max(0.5, 0.08 * u_Rw * u_Rw));
  // Aerial perspective.
  vec3 vd = normalize(p - u_cam);
  float k = mix(22000.0, 70000.0, clamp(u_alt / 12000.0, 0.0, 1.0));
  float fog = 1.0 - exp(-dist / k);
  vec3 fogC = skyRad(normalize(vec3(vd.xy, max(vd.z, 0.015))), u_sun, u_alt);
  o = vec4(mix(c, fogC, fog), 1.0);
}`;
}

// ------------------------------------------------------------------ lit geometry (booster + props)
export const LIT_VS = `#version 300 es
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
export const LIT_FS = `#version 300 es
precision highp float;
precision highp sampler2DShadow;
in vec3 v_world; in vec3 v_n; in vec3 v_nb; in vec2 v_ex; in vec3 v_pb; out vec4 o;
uniform vec3 u_cam; uniform vec3 u_sun; uniform int u_mat; uniform vec3 u_color; uniform float u_alt;
uniform vec3 u_l0Pos, u_l0Col, u_l1Pos, u_l1Col; uniform float u_glow;
uniform int u_cpMode; uniform vec3 u_flowB; uniform float u_sinA;
uniform mat4 u_shadowVP; uniform sampler2DShadow u_shadow;
${NOISE}
${NOISE3}
${SKY}
${PBR}
${LOGDEPTH_FS}
const float TAU = 6.2831853;
vec3 cpColor(float cp) {
  vec3 neg = vec3(0.10, 0.35, 1.0), pos = vec3(1.0, 0.25, 0.10);
  return cp < 0.0 ? mix(vec3(1.0), neg, clamp(-cp / 2.5, 0.0, 1.0)) : mix(vec3(1.0), pos, clamp(cp, 0.0, 1.0));
}
float shadowAt(vec3 p) {
  vec4 s = u_shadowVP * vec4(p, 1.0); vec3 c = s.xyz / s.w * 0.5 + 0.5;
  if (c.x <= 0.0 || c.x >= 1.0 || c.y <= 0.0 || c.y >= 1.0 || c.z >= 1.0) return 1.0;
  float sum = 0.0; vec2 texel = vec2(1.0 / 2048.0);
  for (int i = -1; i <= 1; i++) for (int j = -1; j <= 1; j++) sum += texture(u_shadow, vec3(c.xy + vec2(i, j) * texel, c.z - 0.0005));
  return sum / 9.0;
}
vec3 envColor(vec3 R, float rough) {
  vec3 sky = mix(skyRad(normalize(vec3(R.xy, max(R.z, 0.0))), u_sun, u_alt), vec3(0.20, 0.30, 0.48), clamp(rough * 1.2, 0.0, 1.0));
  vec3 gnd = vec3(0.13, 0.13, 0.10);
  return mix(gnd, sky, smoothstep(-0.12, 0.08 + 0.2 * rough, R.z));
}
void main() {
  logDepth();
  vec3 n = normalize(v_n); if (!gl_FrontFacing) n = -n;
  vec3 V = normalize(u_cam - v_world);
  vec3 alb = u_color; float rough = 0.5, metal = 0.0, ao = 1.0; vec3 emis = vec3(0.0);
  vec3 pb = v_pb; float z = pb.z;
  float azf = fract(atan(pb.y, pb.x) / TAU);
  if (u_mat == 0) {
    // ---- booster skin: white paint over Al-Li tanks, soot from the entry/landing burns
    vec2 cyl = vec2(azf * TAU * 1.5, z);
    if (v_nb.z < -0.9 && z < 0.05) {                      // heat shield under the octaweb
      alb = vec3(0.11, 0.105, 0.10) * (0.75 + 0.5 * fbm(pb.xy * 3.0)); rough = 0.92;
    } else if (z > 14.95) {                               // carbon-composite interstage
      vec2 wv = floor(vec2(cyl.x, z) * 22.0);
      float weave = mod(wv.x + wv.y, 2.0);
      alb = vec3(0.038, 0.040, 0.043) * (0.9 + 0.2 * weave);
      rough = 0.33 + 0.08 * weave + 0.15 * fbm(cyl * 3.0);
      alb = mix(alb, vec3(0.30, 0.30, 0.30), smoothstep(0.78, 0.9, fbm(cyl * 4.0 + 9.0)) * 0.15);   // scuffs
      if (v_nb.z > 0.9) { alb = vec3(0.05); rough = 0.7; }
    } else if (z < 1.05) {                                // engine-section skirt / octaweb closeouts
      alb = vec3(0.085, 0.08, 0.075) * (0.7 + 0.6 * fbm(cyl * 2.5)); rough = 0.8; metal = 0.2;
      alb *= 1.0 - 0.3 * step(abs(fract(azf * 16.0) - 0.5), 0.02);                          // panel seams
    } else {
      float stkA = vnoise(vec2(cyl.x * 5.5, z * 0.2));            // fine vertical streaks
      float stkB = fbm(vec2(cyl.x * 1.3, z * 0.06));               // broad flame tongues
      float blot = fbm(cyl * 0.4 + 7.0);
      float side = 0.5 + 0.5 * cos(azf * TAU - 0.7);               // entry windward side soots higher
      float front = 6.6 + 3.6 * side + (stkB - 0.5) * 5.5 + (stkA - 0.5) * 2.6 + (blot - 0.5) * 2.0;
      float soot = pow(smoothstep(front + 2.6, front - 3.8, z), 1.25);
      float dens = mix(0.45, 1.0, smoothstep(front - 1.0, 1.5, z)) * (0.68 + 0.32 * stkA);
      // thin grime streaks dragged upward during the entry burn
      float grime = smoothstep(0.6, 0.92, vnoise(vec2(cyl.x * 4.0, z * 0.1))) * smoothstep(14.8, front, z) * 0.22 * stkB;
      // clean strips where the stowed legs shielded the paint
      float d = abs(fract((azf - 0.125) * 4.0 + 0.5) - 0.5) / 4.0 * TAU * 1.5;
      float strip = smoothstep(0.5, 0.36, d + (stkA - 0.5) * 0.1) * smoothstep(3.1, 3.5, z) * smoothstep(9.8, 9.3, z);
      soot *= 1.0 - 0.8 * strip;
      float s = clamp(max(soot * dens, grime), 0.0, 0.96);
      vec3 paint = vec3(0.80, 0.805, 0.81) * (0.97 + 0.03 * vnoise(cyl * 20.0));
      vec3 sootC = mix(vec3(0.36, 0.31, 0.26), vec3(0.055, 0.05, 0.045), smoothstep(0.25, 0.85, s));
      alb = mix(paint, sootC, smoothstep(0.0, 0.6, s));
      rough = mix(0.38, 0.88, smoothstep(0.1, 0.7, s));
      // tank dome welds / barrel joints
      float seam = 0.0;
      for (int i = 0; i < 4; i++) { float zs = i == 0 ? 1.2 : i == 1 ? 4.3 : i == 2 ? 11.6 : 14.9; seam += smoothstep(0.03, 0.0, abs(z - zs)); }
      alb *= 1.0 - 0.25 * clamp(seam, 0.0, 1.0);
      rough += 0.1 * seam;
      ao *= mix(1.0, 0.75, smoothstep(2.5, 1.05, z));
    }
  } else if (u_mat == 2) { alb = u_color; rough = 0.35; metal = 1.0; }
  else if (u_mat == 3) {                                   // titanium grid fins (heat tinted)
    float h = fbm(v_pb.xy * 2.0 + v_pb.z);
    alb = mix(vec3(0.52, 0.50, 0.47), vec3(0.42, 0.36, 0.30), h);
    alb = mix(alb, vec3(0.30, 0.32, 0.40), smoothstep(0.55, 0.8, fbm(v_pb.xy * 5.0)) * 0.6);
    alb = mix(alb, vec3(0.12, 0.11, 0.10), smoothstep(0.5, 0.75, h) * 0.5);
    metal = 1.0; rough = 0.42 + 0.2 * h;
  } else if (u_mat == 4) {                                 // carbon-fibre landing legs, metal foot pads
    alb = vec3(0.05, 0.05, 0.055) * (0.8 + 0.4 * fbm(v_pb.xz * 3.0)); rough = 0.62;
    if (v_pb.z > 6.05) { alb = vec3(0.45, 0.45, 0.46); metal = 1.0; rough = 0.5; }
  } else if (u_mat == 10) {                                // Merlin nozzle extension / bell
    float t = clamp((-z - 0.2) / 1.65, 0.0, 1.0);          // 0 throat .. 1 exit
    float h = fbm(vec2(azf * 40.0, z * 3.0));
    vec3 bronze = vec3(0.50, 0.36, 0.22), blue = vec3(0.26, 0.26, 0.36), dark = vec3(0.16, 0.15, 0.15);
    alb = mix(mix(bronze, blue, smoothstep(0.1, 0.45, t + (h - 0.5) * 0.3)), dark, smoothstep(0.4, 0.9, t));
    metal = 0.9; rough = 0.35 + 0.25 * h;
    // regenerative-cooling tubes near the throat
    alb *= 0.85 + 0.15 * step(0.5, fract(azf * 180.0)) * (1.0 - t);
    if (!gl_FrontFacing) {                                  // inside of the bell
      alb = vec3(0.05, 0.045, 0.04); metal = 0.3; rough = 0.8;
      float T = 2900.0 - 900.0 * t;
      emis = blackbody(T) * u_glow * (4.0 + 26.0 * pow(1.0 - t, 2.0));
    }
    // exit lip glows dull red after a long burn
    emis += blackbody(1100.0) * u_glow * 0.6 * smoothstep(0.93, 1.0, t);
  } else if (u_mat == 13) { alb = u_color; metal = 0.6; rough = 0.55; }
  else if (u_mat == 5) {
    alb = u_color * (0.82 + 0.26 * vnoise(v_world.xy * 1.7 + v_world.z * 1.3)) * (1.0 - 0.25 * smoothstep(0.6, 0.0, v_world.z));
    rough = 0.9;
  } else if (u_mat == 6) {
    float st = step(0.5, fract((v_world.x + v_world.y + v_world.z) / 0.7));
    alb = mix(vec3(0.92, 0.72, 0.10), vec3(0.05, 0.05, 0.05), st); rough = 0.6;
  } else if (u_mat == 8) {
    float rib = smoothstep(0.25, 0.75, abs(fract(v_ex.x * 22.0) - 0.5) * 2.0);
    alb = u_color * (0.80 + 0.20 * rib) * (0.9 + 0.2 * vnoise(v_world.xy * 0.8 + v_world.z));
    rough = 0.45; metal = 0.5;
  } else if (u_mat == 1) { rough = 0.6; }
  if (u_cpMode == 1 && u_mat == 0) {
    vec3 nb = normalize(v_nb);
    vec3 fperp = u_flowB - dot(u_flowB, vec3(0, 0, 1)) * vec3(0, 0, 1);
    float cp = 0.0;
    if (length(fperp) > 1e-3 && abs(nb.z) < 0.5) {
      float c = -dot(normalize(nb.xy), normalize(fperp.xy));
      float s2 = 1.0 - c * c;
      cp = u_sinA * u_sinA * (1.0 - 4.0 * s2);
      if (c < -0.2) cp = u_sinA * u_sinA * -0.6;
    }
    float axial = dot(u_flowB, vec3(0, 0, 1));
    float leadZ = axial > 0.0 ? 0.0 : 18.0;
    cp += (1.0 - u_sinA * u_sinA) * smoothstep(3.0, 0.0, abs(z - leadZ)) * 0.9;
    alb = cpColor(cp); rough = 0.6; metal = 0.0;
  }
  if (u_mat == 9) { o = vec4(pow(u_color, vec3(2.2)) * 6.0, 1.0); return; }
  alb = pow(alb, vec3(2.2));
  vec3 F0 = mix(vec3(0.04), alb, metal), dif = alb * (1.0 - metal);
  float sh = shadowAt(v_world + n * 0.06);
  vec3 col = brdfLight(n, V, u_sun, SUN_E * sh, dif, F0, rough);
  col += dif * hemiAmb(n.z) * ao;
  float NdV = max(dot(n, V), 0.0);
  vec3 Fr = F0 + (max(vec3(1.0 - rough), F0) - F0) * pow(1.0 - NdV, 5.0);
  col += envColor(reflect(-V, n), rough) * Fr * ao * mix(0.35, 1.0, sh);
  // plume / impingement lights
  for (int i = 0; i < 2; i++) {
    vec3 lp = i == 0 ? u_l0Pos : u_l1Pos, lc = i == 0 ? u_l0Col : u_l1Col;
    vec3 L = lp - v_world; float d2 = dot(L, L);
    col += brdfLight(n, V, L * inversesqrt(d2), lc / (d2 + 1.5), dif, F0, max(rough, 0.3));
  }
  col += emis;
  o = vec4(col, 1.0);
}`;

export const DEPTH_FS = `#version 300 es
precision highp float; void main() {}`;
export const DEPTH_VS = `#version 300 es
layout(location=0) in vec3 a_pos; uniform mat4 u_vp; uniform mat4 u_model;
void main() { gl_Position = u_vp * u_model * vec4(a_pos, 1.0); }`;

// ------------------------------------------------------------------ fullscreen helpers
export const FS_VS = `#version 300 es
out vec2 v_uv;
void main() { vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2); v_uv = p; gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }`;

// ------------------------------------------------------------------ volumetric plume
export const PLUME_FS = `#version 300 es
precision highp float;
in vec2 v_uv; out vec4 o;
uniform mat4 u_invVP; uniform vec3 u_cam; uniform vec3 u_fwd;
uniform highp sampler2D u_depth; uniform float u_logFar;
uniform vec3 u_exit, u_axis, u_bend;
uniform float u_len, u_re, u_spread, u_bulge, u_time, u_I, u_dia, u_lambda, u_pamb;
uniform float u_hdist, u_Rj, u_Rw, u_wall, u_blast;
uniform vec3 u_hit; uniform float u_dscale;
${NOISE3}
float jetR(float s) { return u_re * (1.0 + u_bulge * (1.0 - exp(-s / (4.0 * u_re)))) + max(s, 0.0) * u_spread; }
// Emission (rgb) and extinction (a) of the free jet at p.
// Luminous RP-1/LOX flame colour: incandescent soot, white-yellow when hottest.
vec3 flameColor(float h) {
  vec3 c = mix(vec3(0.42, 0.07, 0.01), vec3(1.0, 0.36, 0.05), smoothstep(0.0, 0.3, h));
  c = mix(c, vec3(1.0, 0.66, 0.22), smoothstep(0.25, 0.6, h));
  c = mix(c, vec3(1.0, 0.90, 0.66), smoothstep(0.55, 0.95, h));
  return c;
}
vec4 freeJet(vec3 p) {
  vec3 q = p - u_exit;
  float s0 = dot(q, u_axis);
  if (s0 < -0.3 || s0 > u_len * 1.3) return vec4(0.0);
  vec3 qb = q - u_bend * s0 * s0;
  float s = dot(qb, u_axis);
  float r = length(qb - u_axis * s);
  float R = jetR(s);
  if (r > R * 1.7 + 0.3) return vec4(0.0);
  float sn = max(s, 0.0) / u_len;
  float tn = fbm3l(qb * 0.6 - u_axis * u_time * 28.0);
  float tn2 = noise3(qb * 2.2 - u_axis * u_time * 60.0);
  float wob = (tn - 0.5) * (0.2 + 0.9 * sn) + (tn2 - 0.5) * 0.22;
  float rn = r / max(0.05, R * 1.2 * (1.0 + wob));
  float body = smoothstep(1.0, 0.4, rn);                       // dense flame, ragged turbulent edge
  float start = smoothstep(-0.15, 0.1, s);
  float tip = smoothstep(1.05, 0.55, sn + (tn - 0.5) * 0.5 + (tn2 - 0.5) * 0.15);
  float heat = body * (1.0 - 0.5 * sn) * tip * start;
  float Lc = u_len * 0.25 + 1.2;
  float coreR = u_re * 0.9 * clamp(1.0 - s / Lc, 0.0, 1.0);
  float core = smoothstep(coreR + 0.06, coreR * 0.3, r) * start;
  float ph = fract(s / u_lambda);
  float cellDecay = exp(-s / (Lc * 0.7)) * start;
  float rd = u_re * 0.8 * (0.15 + 0.85 * abs(ph - 0.5) * 2.0) * clamp(1.0 - s / (Lc * 1.3), 0.0, 1.0);
  float shock = u_dia * (smoothstep(rd + 0.04, rd * 0.3, r) * 0.7 + exp(-pow((ph - 0.58) / 0.06, 2.0)) * smoothstep(u_re * 0.6, 0.0, r) * 1.4) * cellDecay;
  float gcut = smoothstep(0.0, 0.3 * u_Rj + 0.3, p.z);
  float after = 0.3 + 0.7 * u_pamb;
  float h = clamp(heat * 0.72 + core * 0.7 + shock * 0.5, 0.0, 1.0);
  vec3 em = (flameColor(h) * heat * after * 13.0 + flameColor(0.8 + 0.2 * core) * (core * 24.0 + shock * 30.0)) * u_I * gcut;
  float ext = (heat * 1.8 + core * 1.5) * u_I * gcut;
  return vec4(em, ext);
}
// Flame turned by the pad: a flat turbulent fire sheet spreading radially.
vec4 wallJet(vec3 p) {
  vec2 dxy = p.xy - u_hit.xy; float rho = length(dxy); float zg = p.z;
  if (zg < 0.0 || rho > u_Rw * 1.8 + 1.0) return vec4(0.0);
  vec2 dir = rho > 1e-3 ? dxy / rho : vec2(0.0);
  vec3 wp = vec3((dxy - dir * u_time * 24.0) * 0.5, zg * 1.3 - u_time * 2.0);
  float wn = fbm3l(wp), wn2 = noise3(wp * 2.4 + 11.0);
  float delta = 0.35 * u_Rj + 0.07 * rho + 0.35;
  float prof = smoothstep(1.0, 0.25, zg / (delta * (0.6 + 0.9 * wn)));
  float rr = rho / max(0.5, u_Rw * (0.65 + 0.55 * wn + 0.2 * wn2));
  float radial = smoothstep(1.0, 0.35, rr);
  float stag = exp(-pow(rho / (u_Rj * 1.2 + 0.2), 2.0)) * smoothstep(1.0, 0.2, zg / (0.7 * u_Rj + 0.5));
  float w = u_wall * (prof * radial * (1.0 - 0.45 * rr) + stag * 1.2);
  float h = clamp(w * 0.85 + stag * 0.3, 0.0, 1.0);
  vec3 em = flameColor(h) * w * 11.0 + flameColor(0.9) * stag * u_wall * 14.0;
  return vec4(em, w * 1.6);
}
vec2 rayCyl(vec3 ro, vec3 rd, vec3 a, vec3 ax, float R, float L) {
  // intersection of ray with finite cylinder [a, a+ax*L], radius R
  vec3 oc = ro - a;
  float card = dot(ax, rd), caoc = dot(ax, oc);
  float A = 1.0 - card * card, B = dot(oc, rd) - caoc * card, C = dot(oc, oc) - caoc * caoc - R * R;
  float t0 = -1e9, t1 = 1e9;
  if (A > 1e-6) {
    float h = B * B - A * C; if (h < 0.0) return vec2(1.0, -1.0);
    h = sqrt(h); t0 = (-B - h) / A; t1 = (-B + h) / A;
  } else if (C > 0.0) return vec2(1.0, -1.0);
  // slab along the axis
  if (abs(card) > 1e-6) {
    float s0 = (0.0 - caoc) / card, s1 = (L - caoc) / card;
    t0 = max(t0, min(s0, s1)); t1 = min(t1, max(s0, s1));
  } else if (caoc < 0.0 || caoc > L) return vec2(1.0, -1.0);
  return vec2(t0, t1);
}
float ign(vec2 p) { return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715)))); }
void main() {
  vec4 pw = u_invVP * vec4(v_uv * 2.0 - 1.0, 1.0, 1.0);
  vec3 rd = normalize(pw.xyz / pw.w - u_cam);
  ivec2 dp = ivec2(gl_FragCoord.xy * u_dscale);
  float dS = max(max(texelFetch(u_depth, dp, 0).r, texelFetch(u_depth, dp + ivec2(1, 0), 0).r), max(texelFetch(u_depth, dp + ivec2(0, 1), 0).r, texelFetch(u_depth, dp + ivec2(1, 1), 0).r));
  float wS = exp2(dS * u_logFar) - 1.0;
  float tScene = wS / max(1e-4, dot(rd, u_fwd));
  float jit = ign(gl_FragCoord.xy);
  vec3 acc = vec3(0.0); float trans = 1.0;
  // free jet
  float Lb = min(u_len * 1.25, u_hdist + u_Rj + 1.0);
  float bendMax = length(u_bend) * Lb * Lb;
  vec2 t = rayCyl(u_cam, rd, u_exit, u_axis, jetR(Lb) * 2.2 + 0.5 + bendMax, Lb + 0.3);
  t.x = max(t.x, 0.0); t.y = min(t.y, tScene);
  if (t.y > t.x) {
    const int N = 36;
    float dt = (t.y - t.x) / float(N);
    for (int i = 0; i < N; i++) {
      vec3 p = u_cam + rd * (t.x + (float(i) + jit) * dt);
      vec4 e = freeJet(p);
      acc += e.rgb * trans * dt;
      trans *= exp(-e.a * dt);
    }
  }
  // wall jet
  if (u_wall > 0.002) {
    float H = 0.3 * u_Rj + 0.1 * u_Rw * 2.6 + 3.0;
    vec2 tw = rayCyl(u_cam, rd, vec3(u_hit.xy, 0.0), vec3(0.0, 0.0, 1.0), u_Rw * 2.6 + 2.0, H);
    tw.x = max(tw.x, 0.0); tw.y = min(tw.y, tScene);
    if (tw.y > tw.x) {
      const int M = 22;
      float dt = (tw.y - tw.x) / float(M);
      for (int i = 0; i < M; i++) {
        vec3 p = u_cam + rd * (tw.x + (float(i) + jit) * dt);
        vec4 e = wallJet(p);
        acc += e.rgb * trans * dt;
        trans *= exp(-e.a * dt);
      }
    }
  }
  float a = 1.0 - trans;
  o = vec4(acc, a);
}`;

// ------------------------------------------------------------------ smoke / dust / sparks (instanced billboards)
export const PART_VS = `#version 300 es
precision highp float;
layout(location=0) in vec2 a_corner;
layout(location=1) in vec4 a_ps;   // centre, radius
layout(location=2) in vec4 a_col;  // linear albedo, opacity
layout(location=3) in vec4 a_ex;   // seed, rotation, emissive, openness (-1 = spark)
uniform mat4 u_vp; uniform vec3 u_camR, u_camU;
uniform vec3 u_l0Pos, u_l0Col, u_l1Pos, u_l1Col;
out vec2 v_uv; out vec2 v_k; out vec4 v_col; out vec4 v_ex; out float v_w; out vec3 v_plume; out float v_size; out vec3 v_world;
${LOGDEPTH_VS}
void main() {
  float c = cos(a_ex.y), s = sin(a_ex.y);
  vec2 k = vec2(c * a_corner.x - s * a_corner.y, s * a_corner.x + c * a_corner.y);
  vec3 p = a_ps.xyz + (u_camR * k.x + u_camU * k.y) * a_ps.w;
  gl_Position = u_vp * vec4(p, 1.0); logDepth();
  v_uv = a_corner; v_k = k; v_col = a_col; v_ex = a_ex; v_w = gl_Position.w; v_size = a_ps.w; v_world = a_ps.xyz;
  vec3 d0 = u_l0Pos - a_ps.xyz, d1 = u_l1Pos - a_ps.xyz;
  float r2 = a_ps.w * a_ps.w;
  v_plume = u_l0Col / (dot(d0, d0) + r2 + 2.0) + u_l1Col / (dot(d1, d1) + r2 + 2.0);
}`;
export const PART_FS = `#version 300 es
precision highp float;
in vec2 v_uv; in vec2 v_k; in vec4 v_col; in vec4 v_ex; in float v_w; in vec3 v_plume; in float v_size; in vec3 v_world;
out vec4 o;
uniform highp sampler2D u_depth; uniform float u_logFar; uniform vec3 u_sun, u_cam, u_camR, u_camU, u_camF; uniform float u_time;
${NOISE3}
${SKY}
void main() {
  float r2 = dot(v_uv, v_uv); if (r2 > 1.0) discard;
  float dS = texelFetch(u_depth, ivec2(gl_FragCoord.xy), 0).r;
  float wS = exp2(dS * u_logFar) - 1.0;
  if (v_ex.w < -0.5) {                     // spark / ember: hot point, additive
    float g = exp(-r2 * 5.0);
    float vis = step(v_w, wS + 0.2);
    o = vec4(blackbody(1500.0 + 600.0 * v_col.a) * v_ex.z * g * vis, 0.0);
    return;
  }
  float r = sqrt(r2);
  vec3 np = vec3(v_uv * 1.35, v_ex.x * 17.3 + u_time * 0.07);
  float n = fbm3l(np), n2 = noise3(np * 3.3 + 5.0);
  float dens = smoothstep(1.0, 0.22, r + (n - 0.5) * 1.0 + (n2 - 0.5) * 0.3);
  float a = v_col.a * dens;
  a *= clamp((wS - v_w) / (v_size * 0.3 + 0.1), 0.0, 1.0);        // soft intersection with geometry
  a *= clamp((v_w - 0.8) / 5.0, 0.0, 1.0);                           // camera inside the cloud
  if (a < 0.002) discard;
  // pseudo normal: sphere + noise relief (screen-space derivative of the density noise)
  vec2 g = vec2(dFdx(n), dFdy(n)) * 40.0;
  vec3 nv = normalize(vec3(v_k * 0.9 - g * 0.25, sqrt(max(0.08, 1.0 - r2))));
  vec3 N = normalize(u_camR * nv.x + u_camU * nv.y - u_camF * nv.z);
  float open = clamp(v_ex.w, 0.0, 1.0);
  float ndl = dot(N, u_sun);
  float wrap = pow(clamp(ndl * 0.55 + 0.45, 0.0, 1.0), 1.6);
  vec3 vdir = normalize(v_world - u_cam);
  float cs = dot(vdir, u_sun), gg = 0.6;
  float hg = (1.0 - gg * gg) / pow(1.0 + gg * gg - 2.0 * gg * cs, 1.5) * 0.08;
  float selfSh = mix(0.35, 1.0, open) * mix(0.55, 1.0, smoothstep(-0.6, 0.6, N.z));
  vec3 lit = SUN_E * (wrap * 0.85 + hg * (1.2 - dens)) * selfSh;
  vec3 amb = hemiAmb(N.z) * (0.55 + 0.45 * open) * 1.3;
  vec3 col = v_col.rgb * (lit + amb + v_plume * (0.35 + 0.3 * (1.0 - dens))) + blackbody(1300.0) * v_ex.z * dens;
  o = vec4(col * a, a);
}`;

// ------------------------------------------------------------------ misc (lines, pad lights)
export const POINT_VS = `#version 300 es
precision highp float;
layout(location=0) in vec3 a_pos; layout(location=1) in vec4 a_col; layout(location=2) in float a_size;
uniform mat4 u_vp; uniform float u_scale;
out vec4 v_col;
${LOGDEPTH_VS}
void main() { gl_Position = u_vp * vec4(a_pos, 1.0); gl_PointSize = clamp(a_size * u_scale / gl_Position.w, 1.5, 256.0); v_col = a_col; logDepth(); }`;
export const POINT_FS = `#version 300 es
precision highp float;
in vec4 v_col; out vec4 o;
${LOGDEPTH_FS}
void main() {
  vec2 d = gl_PointCoord * 2.0 - 1.0; float r2 = dot(d, d); if (r2 > 1.0) discard;
  logDepth();
  float a = v_col.a * (exp(-r2 * 9.0) * 1.6 + pow(1.0 - r2, 2.0) * 0.25);
  o = vec4(pow(v_col.rgb, vec3(2.2)) * a * 3.0, 0.0);
}`;
export const LINE_VS = `#version 300 es
precision highp float;
layout(location=0) in vec3 a_pos; layout(location=1) in vec4 a_col;
uniform mat4 u_vp; out vec4 v_col;
${LOGDEPTH_VS}
void main() { gl_Position = u_vp * vec4(a_pos, 1.0); v_col = a_col; logDepth(); }`;
export const LINE_FS = `#version 300 es
precision highp float;
in vec4 v_col; out vec4 o;
${LOGDEPTH_FS}
void main() { logDepth(); vec3 c = v_col.a > 0.0 ? v_col.rgb / v_col.a : vec3(0.0); o = vec4(pow(c, vec3(2.2)) * v_col.a, v_col.a); }`;

// ------------------------------------------------------------------ post: bloom + tone map
export const DOWN_FS = `#version 300 es
precision highp float;
in vec2 v_uv; out vec4 o;
uniform sampler2D u_src; uniform vec2 u_texel; uniform int u_first;
vec3 s(vec2 uv) { return texture(u_src, uv).rgb; }
float lumw(vec3 c) { return 1.0 / (1.0 + dot(c, vec3(0.2126, 0.7152, 0.0722)) * 0.25); }
void main() {
  vec2 t = u_texel;
  vec3 a = s(v_uv + t * vec2(-2, 2)), b = s(v_uv + t * vec2(0, 2)), c = s(v_uv + t * vec2(2, 2));
  vec3 d = s(v_uv + t * vec2(-2, 0)), e = s(v_uv), f = s(v_uv + t * vec2(2, 0));
  vec3 g = s(v_uv + t * vec2(-2, -2)), h = s(v_uv + t * vec2(0, -2)), i = s(v_uv + t * vec2(2, -2));
  vec3 j = s(v_uv + t * vec2(-1, 1)), k = s(v_uv + t * vec2(1, 1)), l = s(v_uv + t * vec2(-1, -1)), m = s(v_uv + t * vec2(1, -1));
  vec3 r;
  if (u_first == 1) {
    // Karis average against fireflies
    vec3 g0 = (a + b + d + e) * 0.25, g1 = (b + c + e + f) * 0.25, g2 = (d + e + g + h) * 0.25, g3 = (e + f + h + i) * 0.25, g4 = (j + k + l + m) * 0.25;
    float w0 = lumw(g0), w1 = lumw(g1), w2 = lumw(g2), w3 = lumw(g3), w4 = lumw(g4);
    r = (g0 * w0 * 0.125 + g1 * w1 * 0.125 + g2 * w2 * 0.125 + g3 * w3 * 0.125 + g4 * w4 * 0.5) / (w0 * 0.125 + w1 * 0.125 + w2 * 0.125 + w3 * 0.125 + w4 * 0.5);
  } else {
    r = e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
  }
  o = vec4(min(r, vec3(3.0e4)), 1.0);
}`;
export const UP_FS = `#version 300 es
precision highp float;
in vec2 v_uv; out vec4 o;
uniform sampler2D u_src; uniform vec2 u_texel;
void main() {
  vec2 t = u_texel;
  vec3 r = texture(u_src, v_uv).rgb * 4.0;
  r += (texture(u_src, v_uv + t * vec2(-1, 0)).rgb + texture(u_src, v_uv + t * vec2(1, 0)).rgb + texture(u_src, v_uv + t * vec2(0, -1)).rgb + texture(u_src, v_uv + t * vec2(0, 1)).rgb) * 2.0;
  r += texture(u_src, v_uv + t * vec2(-1, -1)).rgb + texture(u_src, v_uv + t * vec2(1, -1)).rgb + texture(u_src, v_uv + t * vec2(-1, 1)).rgb + texture(u_src, v_uv + t * vec2(1, 1)).rgb;
  o = vec4(r / 16.0, 1.0);
}`;
export const COMPOSITE_FS = `#version 300 es
precision highp float;
in vec2 v_uv; out vec4 o;
uniform sampler2D u_hdr, u_bloom; uniform float u_exposure, u_bloomK, u_time;
vec3 RRTAndODTFit(vec3 v) { vec3 a = v * (v + 0.0245786) - 0.000090537; vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081; return a / b; }
vec3 ACES(vec3 c) {
  const mat3 I = mat3(0.59719, 0.07600, 0.02840, 0.35458, 0.90834, 0.13383, 0.04823, 0.01566, 0.83777);
  const mat3 O = mat3(1.60475, -0.10208, -0.00327, -0.53108, 1.10813, -0.07276, -0.07367, -0.00605, 1.07602);
  return clamp(O * RRTAndODTFit(I * c), 0.0, 1.0);
}
vec3 toSRGB(vec3 c) { return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c)); }
float h12(vec2 p) { vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
void main() {
  vec3 c = texture(u_hdr, v_uv).rgb;
  vec3 b = texture(u_bloom, v_uv).rgb;
  c = mix(c, b, u_bloomK);
  c = ACES(c * u_exposure);
  vec2 d = v_uv - 0.5; c *= 1.0 - 0.22 * dot(d, d);
  c = toSRGB(c);
  c += (h12(gl_FragCoord.xy + fract(u_time) * 97.0) - 0.5) / 255.0;
  o = vec4(c, 1.0);
}`;

export const COPY_FS = `#version 300 es
precision highp float;
in vec2 v_uv; out vec4 o;
uniform sampler2D u_src;
void main() { o = texture(u_src, v_uv); }`;

// Cold-gas (N2) RCS jet: narrow white cone, condensation-lit, fading in a few metres.
export const RCS_VS = `#version 300 es
precision highp float;
layout(location=0) in vec3 a_pos; layout(location=1) in vec3 a_nrm;
uniform mat4 u_vp, u_model; uniform float u_len, u_r0, u_r1;
out vec3 v_world; out vec3 v_n; out float v_s; out vec2 v_az; out float v_w;
void main() {
  float s = a_pos.z;
  float r = mix(u_r0, u_r1, pow(s, 0.75));
  vec4 w = u_model * vec4(a_pos.xy * r, s * u_len, 1.0);
  v_world = w.xyz; v_n = normalize(mat3(u_model) * vec3(a_nrm.xy, 0.0)); v_s = s; v_az = a_pos.xy;
  gl_Position = u_vp * w; v_w = gl_Position.w;
}`;
export const RCS_FS = `#version 300 es
precision highp float;
in vec3 v_world; in vec3 v_n; in float v_s; in vec2 v_az; in float v_w; out vec4 o;
uniform highp sampler2D u_depth; uniform float u_logFar; uniform vec3 u_cam, u_sun; uniform float u_I, u_time, u_seed;
${NOISE3}
${SKY}
void main() {
  float wS = exp2(texelFetch(u_depth, ivec2(gl_FragCoord.xy), 0).r * u_logFar) - 1.0;
  if (v_w > wS) discard;
  vec3 V = normalize(u_cam - v_world);
  float facing = abs(dot(normalize(v_n), V));
  float n = noise3(vec3(v_az * 2.5, v_s * 7.0 - u_time * 30.0 + u_seed * 13.0));
  float dens = pow(facing, 1.4) * pow(1.0 - v_s, 1.8) * smoothstep(0.0, 0.04, v_s) * (0.55 + 0.9 * n);
  float a = clamp(dens * u_I * 0.85, 0.0, 0.85);
  vec3 col = vec3(0.93, 0.95, 0.98) * (SUN_E * 0.5 + hemiAmb(1.0) * 1.3);
  o = vec4(col * a, a);
}`;
