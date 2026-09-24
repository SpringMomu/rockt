"""Record the AGENTS.md terminal-flight constraints for all six scenarios."""
import sys
import json
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim3d import Sim3D, SCENARIOS, PHYSICS_DT

results=[]
for i in range(len(SCENARIOS)):
    sim=Sim3D(cfd=False)
    sim.load_scenario(i)
    v=sim.vehicle
    min_sink=math.inf
    max_rate=0.0
    while sim.time<260 and v.state=='FLYING':
        sim.step(PHYSICS_DT)
        if v.touchdown is None and sim.autopilot.phase in ('LANDING BURN','TERMINAL'):
            if v.altitude>1.5:
                min_sink=min(min_sink,-float(v.vel[2]))
            if v.altitude<8:
                max_rate=max(max_rate,math.degrees(max(abs(float(v.omega[0])),abs(float(v.omega[1])))))
    td=v.touchdown or {}
    ok=(v.state=='LANDED' and td.get('error',99)<=3.5 and abs(td.get('vz',99))<=2.5
        and td.get('vh',99)<=1.5 and td.get('tilt',99)<=1.5 and min_sink>=0.8 and max_rate<=12)
    row={'scenario':sim.scenario_name,'pass':ok,'minimum_sink_above_1_5m':min_sink,
         'maximum_pitch_yaw_rate_below_8m':max_rate,'touchdown':td}
    results.append(row)
    print(json.dumps(row),flush=True)
Path(__file__).with_name('landing-envelope.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
raise SystemExit(0 if all(r['pass'] for r in results) else 1)
