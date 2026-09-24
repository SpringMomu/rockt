"""6-DOF booster model for the 3D simulator.

Frames
------
* World: local East-North-Up (x east, y north, z up); pad centre at the
  origin, ground plane z = 0 (the pad area is flat).
* Body: +z along the vehicle axis (engine -> nose), origin at the bottom of
  the engine section.  ``q`` rotates body vectors into the world frame.

What is modelled
----------------
* Rigid-body translation and rotation (quaternion attitude, Euler's equation
  with the current inertia tensor).
* Propellant: LOX and RP-1 drain at the engine mass flow (mixture ratio
  2.56).  Mass, centre of mass and inertia follow the remaining propellant.
* Engine: throttle 0-100 %, finite throttle and gimbal slew rates, 2-axis
  gimbal.  Thrust and Isp depend on ambient pressure (sea level -> vacuum);
  the mass flow is set by the throttle, so a lighter vehicle accelerates
  harder for the same throttle -- exactly what the ln(m) GFOLD models.
* Cold-gas RCS in pitch, yaw and roll.
* ISA atmosphere (troposphere + lower stratosphere) and a wind field with a
  shear profile and gusts.
* Four landing legs that deploy (2 s), spring-damper feet with friction,
  and crash rules for hull impact, leg overload and tip-over.

Aerodynamic forces come from ``aero3d.AeroModel`` (engineering model, or the
live CFD when enabled) and are passed in by the caller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

G0 = 9.80665
EARTH_RADIUS = 6_371_000.0


# ---------------------------------------------------------------- math
def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_to_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_from_axis_angle(axis, angle):
    axis = np.asarray(axis, float)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = axis / n
    s = math.sin(angle / 2.0)
    return np.array([math.cos(angle / 2.0), axis[0] * s, axis[1] * s, axis[2] * s])


def quat_between(u, v):
    """Shortest rotation taking unit vector u to unit vector v."""
    u = np.asarray(u, float) / np.linalg.norm(u)
    v = np.asarray(v, float) / np.linalg.norm(v)
    d = float(np.dot(u, v))
    if d < -0.999999:
        axis = np.cross(u, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(u, [0.0, 1.0, 0.0])
        return quat_from_axis_angle(axis, math.pi)
    c = np.cross(u, v)
    q = np.array([1.0 + d, c[0], c[1], c[2]])
    return q / np.linalg.norm(q)


# --------------------------------------------------------------- atmosphere
def atmosphere(h: float):
    """ISA: returns (density kg/m3, pressure Pa, temperature K, speed of sound m/s)."""
    h = max(0.0, h)
    if h < 11_000.0:
        T = 288.15 - 0.0065 * h
        p = 101_325.0 * (T / 288.15) ** 5.25588
    elif h < 25_000.0:
        T = 216.65
        p = 22_632.1 * math.exp(-0.000157688 * (h - 11_000.0))
    else:
        T = 216.65 + 0.003 * (h - 25_000.0)
        p = 2_488.6 * (T / 216.65) ** -11.388 if h < 47_000 else 2_488.6 * math.exp(-(h - 25_000.0) / 7_000.0)
    rho = p / (287.05 * T)
    return rho, p, T, math.sqrt(1.4 * 287.05 * T)


@dataclass
class Wind:
    """Wind toward +x (from the west) with a log shear profile and gusts."""

    speed_10m: float = 6.0
    heading_deg: float = 90.0  # direction the wind blows TOWARD, from north
    gust: float = 1.8
    enabled: bool = True

    def at(self, pos, t: float) -> np.ndarray:
        if not self.enabled:
            return np.zeros(3)
        z = max(0.5, float(pos[2]))
        profile = self.speed_10m * min(2.6, max(0.25, math.log(z / 0.05) / math.log(10.0 / 0.05)))
        g = self.gust * (0.55 * math.sin(0.31 * t + 0.002 * pos[0]) + 0.3 * math.sin(0.97 * t + 1.7) + 0.15 * math.sin(2.3 * t + 0.4))
        s = profile + g
        a = math.radians(self.heading_deg)
        gy = self.gust * 0.35 * math.sin(0.53 * t + 0.8)
        return np.array([math.sin(a) * s, math.cos(a) * s + gy, 0.25 * g])


# ------------------------------------------------------------------ vehicle
@dataclass
class VehicleSpec:
    length: float = 18.0
    diameter: float = 3.0
    dry_mass: float = 8_000.0
    lox_capacity: float = 8_630.0
    rp1_capacity: float = 3_370.0
    thrust_sl: float = 360_000.0
    isp_sl: float = 282.0
    isp_vac: float = 311.0
    throttle_rate: float = 1.2       # full range per second
    gimbal_limit: float = math.radians(8.0)
    gimbal_rate: float = math.radians(30.0)
    gimbal_z: float = 0.3            # pivot height above the base
    nozzle_exit_z: float = -1.6
    rcs_torque: tuple = (60_000.0, 60_000.0, 22_000.0)  # pitch(x), yaw(y), roll(z) N*m
    rcs_z: float = 17.2
    # Grid-fin control moment per unit dynamic pressure at full deflection
    # (4 fins x 0.8 m2 x CL 0.8 x ~10.5 m lever; roll ~2.5 m arm).
    fin_moment: tuple = (27.0, 27.0, 6.4)
    fin_rate: float = 3.0            # full-scale deflections per second
    tank_bottom: float = 2.0
    tank_top: float = 15.5
    dry_com_z: float = 7.0
    leg_hinge_r: float = 1.55
    leg_hinge_z: float = 3.2
    leg_length: float = 6.2
    leg_deploy_time: float = 2.0
    grid_fin_z: float = 16.8

    @property
    def propellant_capacity(self) -> float:
        return self.lox_capacity + self.rp1_capacity

    @property
    def thrust_vac(self) -> float:
        return self.thrust_sl * self.isp_vac / self.isp_sl

    @property
    def max_mass_flow(self) -> float:
        return self.thrust_vac / (self.isp_vac * G0)


@dataclass
class Controls:
    throttle: float = 0.0            # commanded 0..1
    gimbal: tuple = (0.0, 0.0)       # commanded (about body x, about body y), rad
    rcs: tuple = (0.0, 0.0, 0.0)     # -1..1 per axis (pitch x, yaw y, roll z)
    fins: tuple = (0.0, 0.0, 0.0)    # grid-fin deflection -1..1 per axis (pitch, yaw, roll)
    legs_down: bool = False


class Vehicle:
    def __init__(self, spec: VehicleSpec | None = None) -> None:
        self.spec = spec or VehicleSpec()
        self.controls = Controls()
        self.reset()

    # ----------------------------------------------------------- state
    def reset(self, *, position=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0), attitude=None,
              propellant: float | None = None, on_pad: bool = True) -> None:
        s = self.spec
        self.pos = np.array(position, float)
        self.vel = np.array(velocity, float)
        self.q = np.array([1.0, 0.0, 0.0, 0.0]) if attitude is None else np.array(attitude, float)
        self.omega = np.zeros(3)  # body rates (rad/s)
        prop = s.propellant_capacity if propellant is None else max(0.0, min(s.propellant_capacity, propellant))
        ratio = s.lox_capacity / s.propellant_capacity
        self.lox = prop * ratio
        self.rp1 = prop * (1.0 - ratio)
        self.throttle = 0.0
        self.gimbal = np.zeros(2)
        self.rcs = np.zeros(3)
        self.fins = np.zeros(3)
        self.legs = 1.0 if on_pad else 0.0
        self.controls = Controls(legs_down=on_pad)
        self.state = "ON PAD" if on_pad else "FLYING"
        self.crash_reason = ""
        self.touchdown: dict | None = None
        self.time = 0.0
        self.settle_timer = 0.0
        self.feet_contact = 0
        self.last = {"thrust": np.zeros(3), "aero": np.zeros(3), "aero_moment": np.zeros(3), "mass_flow": 0.0, "isp": s.isp_sl,
                     "thrust_mag": 0.0, "g": G0}
        if on_pad:
            self.pos[2] = self._rest_height()

    # --------------------------------------------------------- properties
    @property
    def propellant(self) -> float:
        return self.lox + self.rp1

    @property
    def mass(self) -> float:
        return self.spec.dry_mass + self.propellant

    @property
    def R(self) -> np.ndarray:
        return quat_to_matrix(self.q)

    @property
    def axis(self) -> np.ndarray:
        return self.R[:, 2]

    @property
    def altitude(self) -> float:
        """Height of the engine section base above the ground (0 when landed)."""
        return float(self.pos[2] - self._rest_height())

    def _rest_height(self) -> float:
        # Base height above ground when standing upright on deployed legs.
        s = self.spec
        return -(s.leg_hinge_z + s.leg_length * math.cos(math.radians(145.0)))

    def mass_properties(self):
        """(mass, com_z, Ixx=Iyy, Izz) about the current centre of mass."""
        s = self.spec
        md, mf = s.dry_mass, self.propellant
        tank_h = s.tank_top - s.tank_bottom
        fill = tank_h * (mf / s.propellant_capacity)
        zf = s.tank_bottom + fill * 0.5
        m = md + mf
        zc = (md * s.dry_com_z + mf * zf) / m
        r = s.diameter * 0.5
        i_dry = md * (s.length ** 2 / 12.0) * 0.85 + md * (s.dry_com_z - zc) ** 2
        i_f = mf * (fill ** 2 / 12.0 + r * r / 4.0) + mf * (zf - zc) ** 2
        izz = 0.5 * md * r * r * 0.7 + 0.5 * mf * r * r
        return m, zc, i_dry + i_f, izz

    def com_world(self) -> np.ndarray:
        _, zc, _, _ = self.mass_properties()
        return self.pos + self.R @ np.array([0.0, 0.0, zc])

    def engine_performance(self, pressure: float):
        s = self.spec
        frac = min(1.0, pressure / 101_325.0)
        isp = s.isp_vac - (s.isp_vac - s.isp_sl) * frac
        mdot = self.throttle * s.max_mass_flow if self.propellant > 0.0 else 0.0
        return mdot * isp * G0, mdot, isp

    def thrust_direction_body(self, gimbal=None) -> np.ndarray:
        gx, gy = self.gimbal if gimbal is None else gimbal
        d = np.array([math.sin(gy), -math.sin(gx) * math.cos(gy), math.cos(gx) * math.cos(gy)])
        return d / np.linalg.norm(d)

    def leg_foot_body(self, deploy: float):
        s = self.spec
        theta = math.radians(145.0) * deploy
        r = s.leg_hinge_r + s.leg_length * math.sin(theta)
        z = s.leg_hinge_z + s.leg_length * math.cos(theta)
        feet = []
        for k in range(4):
            a = math.radians(45.0 + 90.0 * k)
            feet.append(np.array([math.cos(a) * r, math.sin(a) * r, z]))
        return feet

    def hull_points_body(self):
        s = self.spec
        r = s.diameter * 0.5
        pts = [np.array([0.0, 0.0, s.nozzle_exit_z])]
        for k in range(4):
            a = math.radians(90.0 * k)
            pts.append(np.array([math.cos(a) * r, math.sin(a) * r, 0.0]))
            pts.append(np.array([math.cos(a) * r, math.sin(a) * r, s.length]))
        return pts

    # ------------------------------------------------------------ stepping
    def step(self, dt: float, aero=None, wind: Wind | None = None) -> None:
        """Advance the vehicle by ``dt``.

        ``aero`` is a callable ``aero(vehicle, v_rel, rho, a, com_world)``
        returning ``(force_world, moment_world_about_com, info)``.
        """
        if self.state in ("CRASHED", "LANDED"):
            return
        s = self.spec
        c = self.controls
        self.time += dt

        # Actuators with finite rates.
        self.throttle += max(-s.throttle_rate * dt, min(s.throttle_rate * dt, c.throttle - self.throttle))
        self.throttle = min(1.0, max(0.0, self.throttle))
        if self.propellant <= 0.0:
            self.throttle = 0.0
        g_cmd = np.clip(np.asarray(c.gimbal, float), -s.gimbal_limit, s.gimbal_limit)
        self.gimbal += np.clip(g_cmd - self.gimbal, -s.gimbal_rate * dt, s.gimbal_rate * dt)
        self.rcs = np.clip(np.asarray(c.rcs, float), -1.0, 1.0)
        f_cmd = np.clip(np.asarray(c.fins, float), -1.0, 1.0)
        self.fins += np.clip(f_cmd - self.fins, -s.fin_rate * dt, s.fin_rate * dt)
        target_legs = 1.0 if c.legs_down else 0.0
        self.legs += max(-dt / s.leg_deploy_time, min(dt / s.leg_deploy_time, target_legs - self.legs))

        m, zc, ixx, izz = self.mass_properties()
        R = self.R
        com = self.pos + R @ np.array([0.0, 0.0, zc])
        h = float(com[2])
        rho, p, T, a_snd = atmosphere(h)
        g = G0 * (EARTH_RADIUS / (EARTH_RADIUS + max(0.0, h))) ** 2

        # Engine.
        thrust_mag, mdot, isp = self.engine_performance(p)
        d_body = self.thrust_direction_body()
        f_thrust_b = d_body * thrust_mag
        r_gimbal = np.array([0.0, 0.0, s.gimbal_z - zc])
        torque_b = np.cross(r_gimbal, f_thrust_b)
        force_w = R @ f_thrust_b + np.array([0.0, 0.0, -m * g])

        # RCS.
        torque_b = torque_b + self.rcs * np.array(s.rcs_torque)

        # Aerodynamics.
        wind_v = wind.at(com, self.time) if wind is not None else np.zeros(3)
        v_rel = self.vel - wind_v
        aero_info = {}
        if aero is not None and rho > 1e-6:
            f_aero, m_aero, aero_info = aero(self, v_rel, rho, a_snd, com)
            force_w = force_w + f_aero
            torque_b = torque_b + R.T @ m_aero
            self.last["aero"] = f_aero
            self.last["aero_moment"] = m_aero
            # Grid-fin steering moment (body frame), proportional to dynamic pressure.
            q_dyn = 0.5 * rho * float(np.dot(v_rel, v_rel))
            torque_b = torque_b + self.fins * q_dyn * np.array(s.fin_moment)
            self.last["q"] = q_dyn
        else:
            self.last["aero"] = np.zeros(3)
            self.last["aero_moment"] = np.zeros(3)

        # Ground contact.
        f_contact, t_contact_b = self._contact(R, com, dt, m)
        force_w = force_w + f_contact
        torque_b = torque_b + t_contact_b

        if self.state == "ON PAD":
            if (R @ f_thrust_b)[2] <= m * g * 1.001:
                self.vel[:] = 0.0
                self.omega[:] = 0.0
                self._consume(mdot, dt)
                self._store(R @ f_thrust_b, thrust_mag, mdot, isp, g, wind_v, v_rel, rho, a_snd, aero_info, m, zc)
                return
            self.state = "FLYING"

        if not (np.all(np.isfinite(force_w)) and np.all(np.isfinite(torque_b))):
            self._crash("NUMERICAL FAULT (NON-FINITE FORCE)")
            return

        # Integrate (semi-implicit Euler).
        acc = force_w / m
        self.vel += acc * dt
        self.pos += self.vel * dt
        inertia = np.array([ixx, ixx, izz])
        omega_dot = (torque_b - np.cross(self.omega, inertia * self.omega)) / inertia
        self.omega += omega_dot * dt
        dq = quat_mul(self.q, np.array([0.0, *self.omega])) * 0.5
        self.q = self.q + dq * dt
        self.q /= np.linalg.norm(self.q)
        self._consume(mdot, dt)
        self._store(R @ f_thrust_b, thrust_mag, mdot, isp, g, wind_v, v_rel, rho, a_snd, aero_info, m, zc)
        self._check_landed(dt)

    def _consume(self, mdot, dt):
        if mdot <= 0.0:
            return
        use = min(self.propellant, mdot * dt)
        ratio = self.spec.lox_capacity / self.spec.propellant_capacity
        self.lox = max(0.0, self.lox - use * ratio)
        self.rp1 = max(0.0, self.rp1 - use * (1.0 - ratio))

    def _store(self, thrust_w, thrust_mag, mdot, isp, g, wind_v, v_rel, rho, a_snd, aero_info, m, zc):
        self.last.update({"thrust": thrust_w, "thrust_mag": thrust_mag, "mass_flow": mdot, "isp": isp, "g": g,
                          "wind": wind_v, "v_rel": v_rel, "rho": rho, "a": a_snd, "aero_info": aero_info,
                          "mass": m, "com_z": zc})

    def _contact(self, R, com, dt, m):
        force = np.zeros(3)
        torque_b = np.zeros(3)
        self.feet_contact = 0
        if self.pos[2] > 40.0:
            return force, torque_b
        k, c_d, mu = 3.0e6, 1.6e5, 0.7
        omega_w = R @ self.omega
        feet = self.leg_foot_body(self.legs) if self.legs > 0.02 else []
        points = [(p, True) for p in feet] + [(p, False) for p in self.hull_points_body()]
        for pb, is_foot in points:
            pw = self.pos + R @ pb
            depth = -pw[2]
            if depth <= 0.0:
                continue
            r = pw - com
            v = self.vel + np.cross(omega_w, r)
            if is_foot:
                self.feet_contact += 1
                if self.touchdown is None and self.state == "FLYING":
                    self._record_touchdown()
                if self.state == "FLYING" and v[2] < -7.0:
                    self._crash("LEG FAILURE (HARD LANDING)")
                    return np.zeros(3), np.zeros(3)
            elif self.state == "FLYING" and (np.linalg.norm(v) > 2.5 or not is_foot and self.legs < 0.95):
                self._crash("HULL IMPACT" if self.legs >= 0.95 else "IMPACT - LEGS NOT DEPLOYED")
                return np.zeros(3), np.zeros(3)
            fn = max(0.0, k * depth - c_d * v[2])
            vt = np.array([v[0], v[1], 0.0])
            vt_n = np.linalg.norm(vt)
            ft = np.zeros(3)
            if vt_n > 1e-6:
                ft = -vt / vt_n * min(mu * fn, 1.5e4 * vt_n)  # gain bounded for dt stability (yaw eff. mass)
            f = np.array([0.0, 0.0, fn]) + ft
            force += f
            torque_b += R.T @ np.cross(r, f)
        if self.feet_contact and self.state == "FLYING":
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, self.axis[2]))))
            if tilt > 30.0:
                self._crash("TIPPED OVER")
        return force, torque_b

    def _record_touchdown(self):
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, self.axis[2]))))
        self.touchdown = {
            "x": float(self.pos[0]), "y": float(self.pos[1]),
            "error": float(math.hypot(self.pos[0], self.pos[1])),
            "vz": float(self.vel[2]), "vh": float(math.hypot(self.vel[0], self.vel[1])),
            "tilt": tilt, "propellant": float(self.propellant), "time": float(self.time),
        }

    def _check_landed(self, dt):
        if self.state != "FLYING" or self.feet_contact < 3:
            self.settle_timer = 0.0
            return
        if np.linalg.norm(self.vel) < 0.25 and np.linalg.norm(self.omega) < 0.06 and self.throttle < 0.05:
            self.settle_timer += dt
            if self.settle_timer > 0.8:
                self.state = "LANDED"
                self.vel[:] = 0.0
                self.omega[:] = 0.0
        else:
            self.settle_timer = 0.0

    def _crash(self, reason: str):
        self.state = "CRASHED"
        self.crash_reason = reason
        self.throttle = 0.0
        self.vel[:] = 0.0
        self.omega[:] = 0.0

    # ----------------------------------------------------------- helpers
    def delta_v_remaining(self, pressure: float = 101_325.0) -> float:
        s = self.spec
        frac = min(1.0, pressure / 101_325.0)
        isp = s.isp_vac - (s.isp_vac - s.isp_sl) * frac
        return isp * G0 * math.log(self.mass / s.dry_mass)

    def tilt_deg(self) -> float:
        return math.degrees(math.acos(max(-1.0, min(1.0, self.axis[2]))))

    def euler_deg(self):
        """(heading, pitch, roll) in degrees, aviation-style for the navball.

        Pitch is the elevation of the vehicle axis above the horizon (90 when
        vertical); heading is the compass direction the axis leans toward.
        """
        ax = self.axis
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, ax[2]))))
        heading = math.degrees(math.atan2(ax[0], ax[1])) % 360.0
        # Roll: rotation of body x about the axis relative to world north/east.
        bx = self.R[:, 0]
        ref = np.cross(ax, [0.0, 0.0, 1.0]) if abs(ax[2]) < 0.999 else np.array([1.0, 0.0, 0.0])
        ref = ref / (np.linalg.norm(ref) + 1e-12)
        roll = math.degrees(math.atan2(np.dot(np.cross(ref, bx), ax), np.dot(ref, bx)))
        return heading, pitch, roll
