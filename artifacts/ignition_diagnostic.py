"""Read-only closed-loop ignition diagnostics; does not change flight control."""
import json
import math
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim3d import Sim3D, SCENARIOS, PHYSICS_DT

def angle(a, b):
    return math.degrees(math.acos(float(np.clip(np.dot(a,b) / max(1e-12, np.linalg.norm(a)*np.linalg.norm(b)), -1, 1))))

results = []
for idx in [int(s) for s in sys.argv[1:]] or range(6):
    sim = Sim3D(cfd=False)
    sim.load_scenario(idx)
    ap, v = sim.autopilot, sim.vehicle
    rows = []
    ignition = None
    initial_plan = None
    old_plan = None
    prev_cmd = ap.cmd_axis.copy()
    n = 0
    while sim.time < 260 and v.state == 'FLYING':
        old_vel = v.vel.copy()
        sim.step(PHYSICS_DT)
        if ap.burn_locked and ignition is None:
            ignition = sim.time
            initial_plan = ap.plan
        changed = ap.plan is not old_plan
        if n % 24 == 0 or changed:
            row = dict(t=sim.time, phase=ap.phase, h=v.altitude, r=v.pos[:2].tolist(), vel=v.vel.tolist(),
                       axis=v.axis.tolist(), cmd_axis=ap.cmd_axis.tolist(),
                       cmd_error=angle(v.axis, ap.cmd_axis), cmd_jump=angle(prev_cmd, ap.cmd_axis),
                       omega=np.degrees(v.omega).tolist(), throttle=v.throttle, throttle_cmd=v.controls.throttle,
                       gimbal=np.degrees(v.gimbal).tolist(), gimbal_cmd=np.degrees(v.controls.gimbal).tolist(),
                       thrust=(v.last['thrust']/v.mass).tolist(), aero=(v.last['aero']/v.mass).tolist(),
                       actual_acc=((v.vel-old_vel)/PHYSICS_DT).tolist(), ignition_in=ap.ignition_in,
                       dir_rate=np.degrees(ap._dir_rate).tolist(), replan=changed)
            if ap.plan is not None:
                p, vv, u = ap.plan.sample(ap.elapsed-ap.plan.created_at)
                row.update(plan_u=u.tolist(), plan_angle=angle(v.axis,u),
                           cmd_plan_angle=angle(ap.cmd_axis,u), pos_error=(p-np.array([*v.pos[:2],v.altitude])).tolist(),
                           vel_error=(vv-v.vel).tolist(), plan_status=ap.plan.status, reaches=ap.plan.reaches_pad,
                           tf=ap.plan.t_f, end_time=ap.touchdown_time,
                           plan_drag=ap.plan.drag[0].tolist())
            if initial_plan is not None:
                p, vv, u = initial_plan.sample(sim.time-ignition)
                row['initial_plan_pos_error']=(p-np.array([*v.pos[:2],v.altitude])).tolist()
                row['initial_plan_vel_error']=(vv-v.vel).tolist()
            rows.append(row)
        prev_cmd = ap.cmd_axis.copy()
        old_plan = ap.plan
        n += 1
        if ignition is not None and sim.time > ignition+6:
            break
    rows = [r for r in rows if ignition is None or r['t'] > ignition-5]
    result=dict(scenario=SCENARIOS[idx]['name'], index=idx, ignition=ignition, backend=ap.planner.backend, rows=rows)
    results.append(result)
    burn=[r for r in rows if r['phase']=='LANDING BURN']
    summary=dict(scenario=result['scenario'], ignition=ignition, backend=result['backend'])
    if burn:
        summary.update(max_cmd_error=max(r['cmd_error'] for r in burn), max_plan_angle=max(r.get('plan_angle',0) for r in burn),
                       max_initial_pos_error=max(float(np.linalg.norm(r['initial_plan_pos_error'])) for r in burn),
                       max_initial_vel_error=max(float(np.linalg.norm(r['initial_plan_vel_error'])) for r in burn))
    print(json.dumps(summary), flush=True)
Path(__file__).with_suffix('.json').write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf-8')
