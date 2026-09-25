"""Quantitative acceptance metrics for the 3-D recovery (boost-back, glide, landing burn).

run(sc, over={}, pert={}) -> dict
  over : class attribute overrides for guidance3d.Autopilot3D / Planner3D  ({"A.x": v, "P.y": v})
  pert : truth perturbations {aero, lift, thrust, wind, hdg, gust, seed}
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np


def run(sc, over=None, pert=None, t_max=400.0):
    over = over or {}
    pert = pert or {}
    import guidance3d
    for k, v in over.items():
        cls, name = k.split(".", 1)
        target = guidance3d.Autopilot3D if cls == "A" else guidance3d.Planner3D
        setattr(target, name, tuple(v) if isinstance(v, list) else v)
    import mc_harness as H
    from sim3d import Sim3D, PHYSICS_DT
    sim = Sim3D(cfd=False)
    sim.load_scenario(sc)
    veh, ap = sim.vehicle, sim.autopilot
    sa, sl, st = pert.get("aero", 1.0), pert.get("lift", 1.0), pert.get("thrust", 1.0)
    if "wind" in pert:
        sim.wind.speed_10m = pert["wind"]
    if "hdg" in pert:
        sim.wind.heading_deg = pert["hdg"]
    if "gust" in pert:
        sim.wind.gust = pert["gust"]
    if "seed" in pert:
        rng = np.random.default_rng(pert["seed"])
        veh.pos[:2] += rng.normal(0, 20, 2)
        veh.vel += rng.normal(0, 2, 3)
    truth = H.TruthAero(sim.aero, sa, sl)
    if st != 1.0:
        orig = veh.engine_performance
        veh.engine_performance = lambda p, _o=orig: (lambda r: (r[0] * st, r[1], r[2]))(_o(p))

    def true_miss():
        r = np.r_[veh.pos[:2], veh.altitude]
        return float(np.linalg.norm(H.truth_impact(r, veh.vel, veh.mass, sim.aero, sa, sl, sim.wind, veh.time)))

    M = dict(bb=False)
    prev_phase = None
    t_ign = None
    axis_prev = None
    tilt_rate = []
    max_tilt = max_tilt_low = 0.0
    climb = 0.0
    lowslow = 0.0
    plans = []            # (created_at, plan) published after ignition
    last_plan = None
    td_prev = None
    retimes = 0
    while sim.time < t_max and veh.state == "FLYING":
        w = sim.wind.at(veh.pos, veh.time)
        ap.command(veh, w, PHYSICS_DT)
        veh.step(PHYSICS_DT, truth, sim.wind)
        sim.time += PHYSICS_DT
        ph = ap.phase
        if ph == "BOOSTBACK":
            M["bb"] = True
        if ph != prev_phase:
            if prev_phase == "BOOSTBACK":
                M["bb_end_miss"] = true_miss()
                M["bb_end_alt"] = veh.altitude
            if ph == "LANDING BURN" and t_ign is None:
                t_ign = sim.time
                M["ign_alt"] = veh.altitude
                M["ign_d"] = float(np.hypot(*veh.pos[:2]))
                M["ign_vh"] = float(np.hypot(*veh.vel[:2]))
                M["ign_vz"] = float(veh.vel[2])
                M["ign_miss"] = true_miss()
                M["ign_q"] = float(veh.last.get("q", 0.0))
                M["ign_line"] = float(np.linalg.norm(veh.pos[:2] + veh.vel[:2] * veh.altitude / max(1.0, -float(veh.vel[2]))))
                M["mode"] = "vertical" if getattr(ap, "_vertical_mode", False) else "divert"
            prev_phase = ph
        if t_ign is not None and veh.feet_contact == 0 and veh.touchdown is None:
            tl = veh.tilt_deg()
            max_tilt = max(max_tilt, tl)
            if veh.altitude < 200:
                max_tilt_low = max(max_tilt_low, tl)
            if axis_prev is not None:
                ang = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(axis_prev, veh.axis)))))) / PHYSICS_DT
                tilt_rate.append(ang)
            axis_prev = veh.axis.copy()
            if veh.vel[2] > 0.5:
                climb = max(climb, float(veh.vel[2]))
            M["burn_vh_max"] = max(M.get("burn_vh_max", 0.0), float(np.hypot(*veh.vel[:2])))
            if "h200_d" not in M and veh.altitude <= 200.0:
                M["h200_d"] = float(np.hypot(*veh.pos[:2]))
                M["h200_vh"] = float(np.hypot(*veh.vel[:2]))
                M["h200_vz"] = float(veh.vel[2])
            if veh.altitude < 150 and abs(veh.vel[2]) < 3.0 and veh.altitude > 2.0:
                lowslow += PHYSICS_DT
            p = ap.plan
            if p is not None and p is not last_plan:
                if last_plan is not None:
                    dev = 0.0
                    t_rel = ap.elapsed - last_plan.created_at
                    for tt in np.linspace(0, min(p.t_f, max(0.0, last_plan.t_f - t_rel)), 8):
                        a = p.sample(tt)[0]
                        b = last_plan.sample(t_rel + tt)[0]
                        dev = max(dev, float(np.linalg.norm(a - b)))
                    plans.append(dev)
                last_plan = p
            if td_prev is not None and ap.touchdown_time is not None and abs(ap.touchdown_time - td_prev) > 0.5:
                retimes += 1
            td_prev = ap.touchdown_time
    td = veh.touchdown or {}
    tr = np.array(tilt_rate) if tilt_rate else np.zeros(1)
    M.update(dict(
        state=veh.state, reason=veh.crash_reason,
        err=td.get("error", float("nan")), vz=td.get("vz", float("nan")), vh=td.get("vh", float("nan")),
        tilt_td=td.get("tilt", float("nan")), prop=float(veh.propellant),
        burn_s=(sim.time - t_ign) if t_ign else float("nan"),
        max_tilt=max_tilt, max_tilt_low=max_tilt_low,
        tilt_rate_p95=float(np.percentile(tr, 95)), tilt_rate_max=float(tr.max()),
        jump_max=float(max(plans)) if plans else 0.0, jump_mean=float(np.mean(plans)) if plans else 0.0,
        retimes=retimes, climb=climb, lowslow=lowslow,
    ))
    return M


def fmt(sc, M):
    f = lambda k, n=1: ("-" if k not in M or M[k] is None or (isinstance(M[k], float) and math.isnan(M[k])) else f"{M[k]:.{n}f}")
    return (f"{sc} {M['state'][:6]:6s} bbEnd={f('bb_end_miss',0):>5} ignMiss={f('ign_miss',0):>4} ignLine={f('ign_line',0):>4} ignD={f('ign_d',0):>4} "
            f"ignVh={f('ign_vh',0):>3} ignH={f('ign_alt',0):>5} {M.get('mode','-')[:4]} | burn={f('burn_s')}s tilt={f('max_tilt',0)} "
            f"tilt<200={f('max_tilt_low',0)} rate95={f('tilt_rate_p95',0)} jump={f('jump_max',0)}/{f('jump_mean',0)} "
            f"ret={M['retimes']} climb={f('climb')} lowslow={f('lowslow')} | @200m d={f('h200_d',0)} vh={f('h200_vh',1)} vz={f('h200_vz',0)} burnVhMax={f('burn_vh_max',0)} | err={f('err')} vh={f('vh',2)} vz={f('vz',2)} prop={f('prop',0)}")


if __name__ == "__main__":
    from multiprocessing import Pool
    scen = [int(x) for x in sys.argv[1].split(",")]
    over = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    perts = json.loads(sys.argv[3]) if len(sys.argv) > 3 else [{}]
    jobs = [(sc, over, p) for p in perts for sc in scen]

    def one(j):
        sc, o, p = j
        try:
            return sc, p, run(sc, o, p)
        except Exception as e:  # pragma: no cover
            return sc, p, dict(state="ERROR", reason=repr(e), retimes=0)

    with Pool(2, maxtasksperchild=1) as pool:
        for sc, p, M in pool.imap_unordered(one, jobs):
            tag = ",".join(f"{k}{v}" for k, v in p.items()) or "nominal"
            print(f"[{tag}] " + (fmt(sc, M) if M["state"] != "ERROR" else f"{sc} ERROR {M['reason']}"), flush=True)
