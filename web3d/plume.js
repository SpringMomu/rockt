// Engine plume model shared by the renderer (volumetric ray-marched flame,
// lights) and the particle system (jet / wall-jet forcing of smoke and dust).
//
// Physical picture (Merlin-class RP-1/LOX engine, single centre engine for
// the landing burn):
//  * Sea level: nearly ideally expanded, narrow luminous jet with a bright
//    supersonic potential core and Mach diamonds, a turbulent orange shear
//    layer where the fuel-rich exhaust after-burns with air; spreading
//    half-angle ~5 deg.
//  * Altitude: under-expanded, the jet balloons right at the exit ("plume
//    bloom"), diamonds stretch and fade, after-burning (and thus
//    luminosity) drops with the air density.
//  * Retro-propulsion: the oncoming freestream (rocket falling into its own
//    exhaust) shortens and fattens the jet; crossflow bends the far jet.
//  * Ground: below a few jet lengths the jet stagnates on the pad, forms a
//    plate shock and turns into a radial wall jet that sweeps dust outwards
//    and rolls up into a toroidal cloud.
import { V, rot } from "./gl.js";

export const EXIT_Z = -1.85;       // centre-engine nozzle exit plane (body frame)
export const EXIT_R = 0.37;        // nozzle exit radius [m]

export function exhaustDirBody(st) {
  const gx = st.gimbal[0] * Math.PI / 180, gy = st.gimbal[1] * Math.PI / 180;
  return [-Math.sin(gy), Math.sin(gx) * Math.cos(gy), -Math.cos(gx) * Math.cos(gy)];
}

export function plumeParams(st, time = 0) {
  const on = st.state === "FLYING" && st.throttle > 0.01;
  const q = st.q;
  const axis = V.norm(rot(q, exhaustDirBody(st)));
  // Gimbal pivot at body z = +0.3; exit sits (0.3 - EXIT_Z) behind it along the thrust line.
  const pivot = V.add(st.pos, rot(q, [0, 0, 0.3]));
  const exit = V.add(pivot, V.mul(axis, 0.3 - EXIT_Z));
  const alt = Math.max(0, exit[2]);
  const pamb = Math.exp(-alt / 8400);
  const thr = on ? st.throttle : 0;
  // Freestream seen by the vehicle.
  const wind = st.wind || [0, 0, 0];
  const free = V.sub(wind, st.vel);
  const vAx = V.dot(free, axis);                       // <0: counter-flow (falling into the plume)
  const cperp = V.sub(free, V.mul(axis, vAx));
  const counter = Math.max(0, -vAx);
  let len = (9 + 22 * thr) * (1 + 1.4 * (1 - pamb));
  len /= 1 + counter / 320;
  const spread = (0.13 + 0.33 * (1 - pamb)) * (1 + counter / 280);
  const bulge = 2.8 * Math.pow(1 - pamb, 1.5) + counter / 400;
  const Ueff = 420;
  const bend = V.mul(cperp, 1 / (2 * Ueff * Math.max(4, len)));
  const jetR = (s) => EXIT_R * (1 + bulge * (1 - Math.exp(-s / (4 * EXIT_R)))) + s * spread;
  // Impingement on flat ground (z = 0).
  let hdist = 1e5, hit = null, Rj = 0, heat = 0, blast = 0;
  if (axis[2] < -0.08) {
    hdist = exit[2] / -axis[2];
    hit = V.add(exit, V.mul(axis, hdist));
    Rj = jetR(hdist);
    heat = thr * Math.exp(-hdist / (0.55 * len));                 // flame still luminous on the pad
    // Dynamic pressure of the jet reaching the ground (drives dust), falls ~1/h^2 past the core.
    const hRef = 10 + 40 * thr;
    blast = thr * Math.min(1, Math.pow(hRef / Math.max(hRef, hdist), 2)) * Math.max(0, 1 - hdist / (6 * hRef));
  }
  const Rw = Rj * 1.6 + 14 * heat + 3 * blast;
  const flick = 0.94 + 0.06 * Math.sin(time * 61.0) * Math.sin(time * 23.7 + 1.3);
  // Merlin gas-generator (turbopump) exhaust: a separate fuel-rich, sooty
  // stream dumped beside the engine -- the dark smoke trail seen next to a
  // Falcon 9 plume.
  const gg = V.add(st.pos, rot(q, [0.62, 0.0, -1.3]));
  return {
    on, thr, axis, exit, gg, pamb, len, spread, bulge, bend, jetR, hdist, hit, Rj, Rw, heat, blast,
    intensity: on ? (0.35 + 0.65 * thr) * flick : 0,
    diamonds: Math.pow(pamb, 2.0) * (0.55 + 0.45 * thr),
    lambda: 1.3 * 2 * EXIT_R * Math.pow(1 / Math.max(0.05, pamb), 0.35),
    counter,
  };
}
