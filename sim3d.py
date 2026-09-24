"""Simulation core for the 3D game: vehicle + aero + wind + autopilot + CFD.

Used by the browser front end (``main3d.py``) and by the headless tests.
"""

from __future__ import annotations

import math
import threading

import numpy as np

from aero3d import AeroModel, ExternalCFD
from guidance3d import Autopilot3D, attitude_control
from rocket3d import Vehicle, Wind, atmosphere, quat_between, quat_from_axis_angle, quat_mul

PHYSICS_DT = 1.0 / 240.0

SCENARIOS = [
    {"name": "HOVER-SLAM 100 m", "pos": (0, 0, 100), "vel": (0, 0, 0), "tilt": 0.0, "prop": 2500},
    {"name": "2000 m / +30 m/s UP", "pos": (0, 0, 2000), "vel": (0, 0, 30), "tilt": 0.0, "prop": 4000},
    {"name": "2000 m / -30 m/s DOWN", "pos": (0, 0, 2000), "vel": (0, 0, -30), "tilt": 0.0, "prop": 4000},
    {"name": "BOOSTBACK +1000 m / 100 m/s", "pos": (1000, 300, 2000), "vel": (100, -20, 0), "tilt": 10.0, "prop": 5500},
    {"name": "DIVERT -1000 m / 3-D", "pos": (-1000, 600, 2000), "vel": (100, -40, 0), "tilt": 10.0, "prop": 5500},
    {"name": "ENTRY 12 km / 280 m/s", "pos": (1800, -1100, 12000), "vel": (-88, 54, -260), "tilt": 8.0, "prop": 6000},
    # ---- return to launch site from stage separation (keys 7, 8, 9, 0) ------
    # Full Falcon-9 style sequence: the booster starts nose-first, still
    # climbing and flying AWAY from the pad; it flips, boosts back (in thin
    # air), coasts over the apogee and re-enters engines-first, steers the
    # impact point with body lift, then lights one landing burn.
    # "axis": initial body axis (world), here along the velocity (prograde).
    # Checked once each (engineering aero, no CFD): boost-back 33-38 s, apogee
    # 37-51 km, re-entry max-q 19-40 kPa; landed 0.07-0.9 m from the centre.
    # LOW FUEL lands with ~0.8 t left (about one landing burn of margin).
    {"name": "RTLS 30 km / 450 m/s", "pos": (6000, 0, 30000), "vel": (300, 0, 330), "axis": (300, 0, 330),
     "prop": 7000},
    {"name": "RTLS 35 km / CROSSWIND", "pos": (8000, -2500, 35000), "vel": (340, -90, 360), "axis": (340, -90, 360),
     "prop": 7500, "wind": (12.0, 45.0, 3.5)},
    {"name": "RTLS 30 km / LOW FUEL", "pos": (6000, 1500, 30000), "vel": (300, 60, 330), "axis": (300, 60, 330),
     "prop": 3000},
    {"name": "RTLS 40 km / GUST + TUMBLE", "pos": (9000, 3000, 40000), "vel": (380, 120, 380), "axis": (380, 120, 330),
     "prop": 8000, "wind": (15.0, 250.0, 5.0), "omega": (0.08, -0.05, 0.2)},
]
DEFAULT_WIND = (6.0, 90.0, 1.8)   # speed at 10 m (m/s), heading blown toward (deg), gust (m/s)


class Sim3D:
    def __init__(self, cfd: bool = True) -> None:
        self.vehicle = Vehicle()
        self.aero = AeroModel(self.vehicle.spec)
        self.wind = Wind()
        self.autopilot = Autopilot3D(self.aero)
        self.cfd = ExternalCFD() if cfd else None
        self.aero.cfd = self.cfd
        self.lock = threading.RLock()
        self.autopilot_on = False
        self.sas = "OFF"          # OFF | STABILITY | RETROGRADE
        self.keys: dict = {}
        self.throttle_manual = 0.0
        self.time = 0.0
        self.time_warp = 1.0
        self.paused = False
        self.trail: list = []
        self.scenario_name = ""
        self.touchdown = None
        self.events: list = []
        self.load_scenario(3)

    # --------------------------------------------------------------- setup
    def load_scenario(self, index: int, autopilot: bool = True) -> None:
        with self.lock:
            sc = SCENARIOS[index % len(SCENARIOS)]
            tilt = math.radians(sc.get("tilt", 0.0))
            vel = np.array(sc["vel"], float)
            horiz = np.array([vel[0], vel[1], 0.0])
            axis = np.array([0.0, 1.0, 0.0]) if np.linalg.norm(horiz) < 1e-6 else np.cross([0.0, 0.0, 1.0], horiz / np.linalg.norm(horiz))
            q = quat_from_axis_angle(axis, -tilt)
            if "axis" in sc:
                q = quat_between(np.array([0.0, 0.0, 1.0]), np.array(sc["axis"], float))
            self.vehicle.reset(position=sc["pos"], velocity=sc["vel"], attitude=q, propellant=sc["prop"], on_pad=False)
            self.vehicle.pos[2] += self.vehicle._rest_height()
            if "omega" in sc:
                self.vehicle.omega = np.array(sc["omega"], float)
            ws, wh, wg = sc.get("wind", DEFAULT_WIND)
            self.wind.speed_10m, self.wind.heading_deg, self.wind.gust = float(ws), float(wh), float(wg)
            self.autopilot.reset()
            self.autopilot_on = autopilot
            self.sas = "OFF"
            self.throttle_manual = 0.0
            self.time = 0.0
            self.trail = [self.vehicle.pos.copy()]
            self.scenario_name = sc["name"]
            self.events = [(0.0, f"SCENARIO: {sc['name']}")]

    def reset_to_pad(self) -> None:
        with self.lock:
            self.vehicle.reset(on_pad=True)
            self.autopilot.reset()
            self.autopilot_on = False
            self.throttle_manual = 0.0
            self.time = 0.0
            self.trail = [self.vehicle.pos.copy()]
            self.scenario_name = "ON PAD - MANUAL"

    # ------------------------------------------------------------- control
    def _manual(self, dt: float) -> None:
        v = self.vehicle
        k = self.keys
        c = v.controls
        # Throttle: Shift up, Ctrl down, Z full, X cut (KSP layout).
        if k.get("shift"):
            self.throttle_manual = min(1.0, self.throttle_manual + 0.8 * dt)
        if k.get("ctrl"):
            self.throttle_manual = max(0.0, self.throttle_manual - 0.8 * dt)
        c.throttle = self.throttle_manual
        # Pitch W/S, yaw A/D, roll Q/E: body-rate commands.
        rate = math.radians(18.0)
        w_cmd = np.array([
            (1.0 if k.get("s") else 0.0) - (1.0 if k.get("w") else 0.0),
            (1.0 if k.get("d") else 0.0) - (1.0 if k.get("a") else 0.0),
            (1.0 if k.get("q") else 0.0) - (1.0 if k.get("e") else 0.0),
        ]) * rate
        steering = bool(np.any(w_cmd != 0.0))
        if self.sas == "RETROGRADE" and not steering:
            vr = v.vel - self.wind.at(v.pos, v.time)
            sp = float(np.linalg.norm(vr))
            target = -vr / sp if sp > 3.0 else np.array([0.0, 0.0, 1.0])
            attitude_control(v, target, 1.2, 2.5, math.radians(20.0), use_rcs=True)
            return
        if self.sas == "STABILITY" and not steering:
            if not hasattr(self, "_hold_axis") or self._hold_axis is None:
                self._hold_axis = v.axis.copy()
            attitude_control(v, self._hold_axis, 1.2, 2.5, math.radians(20.0), use_rcs=True)
            return
        self._hold_axis = None
        m, zc, ixx, izz = v.mass_properties()
        alpha = 2.5 * (w_cmd - v.omega) if (steering or self.sas != "OFF") else np.zeros(3)
        if not steering and self.sas == "OFF":
            alpha = np.zeros(3)
        tau = np.array([ixx, ixx, izz]) * alpha
        s = v.spec
        thrust = v.throttle * s.thrust_sl
        gx = gy = 0.0
        if thrust > 0.04 * s.thrust_sl:
            lever = zc - s.gimbal_z
            gy = math.asin(float(np.clip(-tau[1] / lever / thrust, -0.99, 0.99)))
            gx = math.asin(float(np.clip(-(tau[0] / lever) / thrust, -0.99, 0.99)))
        c.gimbal = (gx, gy)
        c.fins = (0.0, 0.0, 0.0)
        c.rcs = tuple(float(x) for x in np.clip(tau / np.array(s.rcs_torque), -1.0, 1.0))

    def toggle_legs(self) -> None:
        self.vehicle.controls.legs_down = not self.vehicle.controls.legs_down

    # ---------------------------------------------------------------- step
    def step(self, dt: float) -> None:
        with self.lock:
            if self.paused:
                return
            v = self.vehicle
            prev_state = v.state
            wind_now = self.wind.at(v.pos, v.time)
            if self.autopilot_on:
                self.autopilot.command(v, wind_now, dt)
                self.throttle_manual = v.controls.throttle
            else:
                self._manual(dt)
            v.step(dt, self.aero, self.wind)
            self.time += dt
            if len(self.trail) == 0 or np.linalg.norm(v.pos - self.trail[-1]) > 4.0:
                self.trail.append(v.pos.copy())
                if len(self.trail) > 1500:
                    self.trail.pop(0)
            if v.state != prev_state:
                self.events.append((self.time, f"{v.state}{': ' + v.crash_reason if v.crash_reason else ''}"))
            ap = self.autopilot
            if ap.phase and (not self.events or not self.events[-1][1].endswith(ap.phase)) and self.autopilot_on:
                if ap.phase in ("COAST", "BOOSTBACK", "LANDING BURN"):
                    if not any(e[1] == f"PHASE {ap.phase}" for e in self.events[-6:]):
                        self.events.append((self.time, f"PHASE {ap.phase}"))
            if len(self.events) > 40:
                self.events = self.events[-40:]

    def advance(self, seconds: float) -> None:
        n = max(1, int(round(seconds / PHYSICS_DT)))
        for _ in range(n):
            self.step(PHYSICS_DT)
