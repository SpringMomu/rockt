"""Robustness harness for the boost-back and unpowered-glide phases (3-D sim).

Truth model perturbations (guidance keeps its nominal model):
  aero_scale   : true aerodynamic force/moment = scale x engineering model
  lift_scale   : extra scale on the normal (lift) part only
  thrust_scale : true thrust = scale x nominal (guidance believes nominal)
  wind         : speed, heading, gust
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np

import guidance3d
from rocket3d import Vehicle, G0, atmosphere, quat_from_axis_angle
from sim3d import Sim3D, PHYSICS_DT


class TruthAero:
    def __init__(self, base, scale=1.0, lift_scale=1.0):
        self.base, self.scale, self.lift_scale = base, scale, lift_scale

    def __call__(self, veh, v_rel, rho, a, com):
        f, m, info = self.base(veh, v_rel, rho, a, com)
        if self.lift_scale != 1.0:
            ax = veh.axis
            fa = np.dot(f, ax) * ax
            f = fa + (f - fa) * self.lift_scale
        return f * self.scale, m * self.scale, info


def truth_impact(r, v, m, aero, s_aero, s_lift, wind, t0):
    """Point-mass engines-first coast with TRUE aero + TRUE (height-varying) wind."""
    p, u, t = r.copy(), v.copy(), t0
    dt = 0.1
    while p[2] > 0 and t - t0 < 300:
        w = wind.at(np.array([p[0], p[1], p[2]]), t)
        rho, _, _, a = atmosphere(p[2])
        f = aero.quick_force(u - w, None, rho, a, 0.0) * s_aero
        u = u + (f / m + np.array([0, 0, -G0])) * dt
        p = p + u * dt
        t += dt
    return p[:2]


def run(case: dict) -> dict:
    sim = Sim3D(cfd=False)
    if case.get("patched"):
        import guidance3d_patched
        sim.autopilot = guidance3d_patched.Autopilot3D(sim.aero)
    sim.load_scenario(case.get("base", 3))
    veh = sim.vehicle
    ap = sim.autopilot
    # --- initial state
    pos = np.array(case["pos"], float)
    vel = np.array(case["vel"], float)
    tilt = math.radians(case.get("tilt", 10.0))
    horiz = np.array([vel[0], vel[1], 0.0])
    axis = np.cross([0, 0, 1.0], horiz / np.linalg.norm(horiz)) if np.linalg.norm(horiz) > 1e-6 else np.array([0, 1.0, 0])
    veh.reset(position=tuple(pos), velocity=tuple(vel), attitude=quat_from_axis_angle(axis, -tilt),
              propellant=case.get("prop", 5500), on_pad=False)
    ap.reset()
    sim.time = 0.0
    # --- truth perturbations
    sa, sl, st = case.get("aero_scale", 1.0), case.get("lift_scale", 1.0), case.get("thrust_scale", 1.0)
    sim.wind.speed_10m = case.get("wind", 6.0)
    sim.wind.heading_deg = case.get("wind_hdg", 90.0)
    sim.wind.gust = case.get("gust", 1.8)
    sim.wind.enabled = sim.wind.speed_10m > 0 or sim.wind.gust > 0
    truth = TruthAero(sim.aero, sa, sl)
    if st != 1.0:
        orig = veh.engine_performance
        veh.engine_performance = lambda p, _o=orig: (lambda r: (r[0] * st, r[1], r[2]))(_o(p))
    if case.get("no_glide"):
        ap._aero_steer = lambda veh_, r, v, m, w: -(v - w) / np.linalg.norm(v - w)
    if case.get("no_boostback"):
        ap._boostback_done = True

    # --- fly
    rec = dict(bb=False, bb_end=None, coast_start=None, ign=None)
    prop0 = veh.propellant
    prev_phase = None
    max_throttle_burn = 0.0
    sat_time = 0.0
    last_impact = None
    ap_step = ap.command
    while sim.time < 300 and veh.state == "FLYING":
        wind_now = sim.wind.at(veh.pos, veh.time)
        if sim.autopilot_on:
            ap_step(veh, wind_now, PHYSICS_DT)
        veh.step(PHYSICS_DT, truth, sim.wind)
        sim.time += PHYSICS_DT
        ph = ap.phase
        if ph == "BOOSTBACK":
            rec["bb"] = True
        if ph in ("COAST",) and ap.impact is not None:
            last_impact = (sim.time, float(np.linalg.norm(ap.impact)), veh.pos.copy(), veh.vel.copy())
        if ph != prev_phase:
            if prev_phase == "BOOSTBACK":
                rec["bb_end"] = dict(t=sim.time, alt=float(veh.altitude), pred_miss=float(np.linalg.norm(ap.impact)),
                                     prop_used=float(prop0 - veh.propellant),
                                     true_miss=float(np.linalg.norm(truth_impact(np.r_[veh.pos[:2], veh.altitude], veh.vel, veh.mass, sim.aero, sa, sl, sim.wind, veh.time))))
            if ph == "COAST" and rec["coast_start"] is None:
                rec["coast_start"] = dict(t=sim.time, alt=float(veh.altitude), pred_miss=float(np.linalg.norm(ap.impact)),
                                          true_miss=float(np.linalg.norm(truth_impact(np.r_[veh.pos[:2], veh.altitude], veh.vel, veh.mass, sim.aero, sa, sl, sim.wind, veh.time))))
            if ph == "LANDING BURN" and rec["ign"] is None:
                r = np.r_[veh.pos[:2], veh.altitude]
                rec["ign"] = dict(t=sim.time, alt=float(veh.altitude), speed=float(np.linalg.norm(veh.vel)),
                                  vh=float(np.linalg.norm(veh.vel[:2])), dist_xy=float(np.linalg.norm(veh.pos[:2])),
                                  pred_miss=float(last_impact[1]) if last_impact else None,
                                  true_miss=float(np.linalg.norm(truth_impact(r, veh.vel, veh.mass, sim.aero, sa, sl, sim.wind, veh.time))),
                                  prop=float(veh.propellant))
            prev_phase = ph
        if ph == "LANDING BURN":
            max_throttle_burn = max(max_throttle_burn, veh.throttle)
            if veh.throttle > 0.97:
                sat_time += PHYSICS_DT
    td = veh.touchdown or {}
    ok = (veh.state == "LANDED" and td.get("error", 99) <= 3.5 and abs(td.get("vz", 9)) <= 2.5
          and td.get("vh", 9) <= 1.5 and td.get("tilt", 90) <= 5.0)
    return dict(case=case, state=veh.state, reason=veh.crash_reason, success=ok, touchdown=td,
                prop_left=float(veh.propellant), t=round(sim.time, 1), burn_fraction=ap.burn_fraction,
                max_throttle_burn=max_throttle_burn, sat_time=sat_time, **rec)


if __name__ == "__main__":
    cases = json.load(open(sys.argv[1]))
    out = sys.argv[2]
    from multiprocessing import Pool
    with Pool(2, maxtasksperchild=1) as pool, open(out, "w") as fh:
        for res in pool.imap_unordered(run, cases):
            fh.write(json.dumps(res, default=float) + "\n")
            fh.flush()
