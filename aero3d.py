"""Aerodynamics for the 3D booster: engineering model + live 3-D CFD coupling.

``AeroModel`` (always on)
    A semi-empirical engineering model of the kind used in flight
    simulators and preliminary design (slender-body theory + Jorgensen
    crossflow + Mach-dependent drag + grid fins).  It is valid at the real
    flight Reynolds and Mach numbers and produces the forces and moments
    that act on the vehicle every physics step.

``ExternalCFD`` (live 3-D CFD, optional)
    The browser runs an incompressible 3-D flow solver on the GPU
    (web3d/fluid.js: by default a 64x64x96 grid of 1 m cells whose axes are
    fixed to east/north/up and which translates with the booster) and
    continuously posts the integrated
    surface-pressure force coefficient to /api/cfd3d.  With ``cfd_forces``
    enabled those coefficients replace the model's normal force and blend
    into the axial force, bounded to 0.5-2x of the model (the real-time grid
    is coarse and inviscid, far below flight Reynolds number).  Stale data
    (> 1 s) falls back to the model automatically.
"""

from __future__ import annotations

import math
import threading
import time

import numpy as np

from rocket3d import cross3, vnorm


# ------------------------------------------------------------ engineering
def _transonic(mach: float, low: float, peak: float, high: float) -> float:
    """Smooth drag rise: low below M0.8, peak near M1.1, decaying supersonic."""
    if mach < 0.8:
        return low
    if mach < 1.1:
        t = (mach - 0.8) / 0.3
        return low + (peak - low) * t * t * (3 - 2 * t)
    return high + (peak - high) * math.exp(-(mach - 1.1) / 0.6)


class AeroModel:
    """Forces and moments on the booster from the relative wind."""

    def __init__(self, spec) -> None:
        self.spec = spec
        self.cfd: "LBMWindTunnel | None" = None
        self.cfd_forces = False
        self.last: dict = {}

    def coefficients(self, alpha: float, mach: float, tail_first: bool):
        """Axial and normal force coefficients (reference: frontal area)."""
        s = self.spec
        a_ref = math.pi * (s.diameter / 2) ** 2
        a_plan = s.length * s.diameter
        if tail_first:
            ca0 = _transonic(mach, 0.95, 1.45, 1.10)  # engine section / base leads
        else:
            ca0 = _transonic(mach, 0.62, 1.05, 0.85)  # flat interstage leads
        sa, cabs = math.sin(alpha), abs(math.cos(alpha))
        pg = 1.0 / math.sqrt(max(0.2, 1.0 - min(mach, 0.85) ** 2))
        cn_pot = 2.0 * sa * cabs * pg * 0.5  # blunt ends carry about half the ideal slope
        cdc = _transonic(mach, 1.2, 1.75, 1.45)
        cn_visc = 0.62 * cdc * (a_plan / a_ref) * sa * sa
        ca = ca0 * cabs * cabs
        return ca, cn_pot, cn_visc

    def quick_force(self, v_rel, axis, rho, a_snd, legs=0.0):
        """Force only (no moments) for trajectory prediction in guidance."""
        s = self.spec
        speed = float(vnorm(v_rel))
        if speed < 0.3 or rho <= 0.0:
            return np.zeros(3)
        u = v_rel / speed
        if axis is None:
            axis = -u  # engines-first, aligned with the flow
        c = float(np.dot(u, axis))
        alpha = math.acos(min(1.0, abs(c)))
        mach = speed / a_snd
        q = 0.5 * rho * speed * speed
        a_ref = math.pi * (s.diameter / 2) ** 2
        ca, cn_pot, cn_visc = self.coefficients(alpha, mach, c < 0.0)
        perp = u - c * axis
        pn = float(vnorm(perp))
        f = -math.copysign(1.0, c) * q * a_ref * ca * axis
        fin_area = 3.2
        f += -math.copysign(1.0, c) * q * fin_area * 0.35 * abs(c) * axis
        if pn > 1e-6:
            f += -(perp / pn) * q * (a_ref * (cn_pot + cn_visc) + fin_area * 1.6 * math.sin(alpha))
        f += -u * q * 4.0 * legs * 0.9
        return f

    def __call__(self, vehicle, v_rel, rho, a_snd, com_world):
        s = self.spec
        R = vehicle.R
        axis = R[:, 2]
        speed = float(vnorm(v_rel))
        _, zc, _, _ = vehicle.mass_properties()
        if speed < 0.3:
            self.last = {"mach": 0.0, "q": 0.0, "alpha": 0.0, "cp_z": zc, "drag": 0.0, "normal": 0.0,
                         "ca": 0.0, "cn": 0.0, "source": "MODEL"}
            return np.zeros(3), np.zeros(3), self.last
        u = v_rel / speed
        c = float(np.dot(u, axis))            # >0 nose-first, <0 engine-first
        tail_first = c < 0.0
        alpha = math.acos(min(1.0, abs(c)))    # total angle of attack w.r.t. the leading end
        mach = speed / a_snd
        q = 0.5 * rho * speed * speed
        a_ref = math.pi * (s.diameter / 2) ** 2

        ca, cn_pot, cn_visc = self.coefficients(alpha, mach, tail_first)
        source = "MODEL"
        if self.cfd_forces and self.cfd is not None:
            cfd = self.cfd.coefficients(axis)
            if cfd is not None:
                # 3-D surface-pressure coefficients from the browser's GPU
                # solver (reference: frontal area).  The coarse inviscid grid
                # is trusted within a factor of two of the engineering model;
                # skin friction/base drag stay in the model part.
                ca_c, cn_c = cfd
                comp = _transonic(mach, 1.0, 1.5, 1.2)
                cn_m = cn_pot + cn_visc
                if math.isfinite(ca_c) and math.isfinite(cn_c):
                    ca = float(np.clip(0.5 * ca + 0.5 * abs(ca_c) * comp, 0.5 * ca, 2.0 * ca + 0.05))
                    cn_visc = float(np.clip(cn_c * comp, 0.5 * cn_m, 2.0 * cn_m + 0.05))
                    cn_pot = 0.0
                    source = "CFD"

        # Directions: axial force opposes the axial flow, normal force opposes
        # the crossflow.
        perp = u - c * axis
        perp_n = float(vnorm(perp))
        f_axial = -math.copysign(1.0, c) * q * a_ref * ca * axis
        f_normal = np.zeros(3)
        n_dir = np.zeros(3)
        if perp_n > 1e-6:
            n_dir = perp / perp_n
            f_normal_pot = -n_dir * q * a_ref * cn_pot
            f_normal_visc = -n_dir * q * a_ref * cn_visc
        else:
            f_normal_pot = f_normal_visc = np.zeros(3)
        z_lead = 0.0 if tail_first else s.length  # potential lift sits at the leading end
        z_mid = s.length * 0.5

        # Grid fins (at the top): lift against the local crossflow; they add
        # drag and a strong restoring moment when falling engine-first.
        fin_area = 3.2
        fin_cn = 1.6 * math.sin(alpha) * (1.0 if tail_first else 0.6)
        f_fins = -n_dir * q * fin_area * fin_cn - math.copysign(1.0, c) * q * fin_area * 0.35 * abs(c) * axis
        # Deployed legs add drag near the base.
        leg_area = 4.0 * vehicle.legs
        f_legs = -u * q * leg_area * 0.9

        forces = [(f_axial, z_mid), (f_normal_pot, z_lead), (f_normal_visc, z_mid), (f_fins, s.grid_fin_z), (f_legs, 2.0)]
        total = np.zeros(3)
        moment = np.zeros(3)
        for f, zb in forces:
            r = R @ np.array([0.0, 0.0, zb - zc])
            total += f
            moment += cross3(r, f)
        # Aerodynamic damping (pitch/yaw from the long body + fins, roll from fins).
        omega_w = R @ vehicle.omega
        w_axial = float(np.dot(omega_w, axis))
        w_perp = omega_w - w_axial * axis
        moment += -q * a_ref * s.length ** 2 * 3.0 * w_perp / max(speed, 5.0)
        moment += -q * fin_area * (s.diameter * 0.8) ** 2 * 1.2 * w_axial * axis / max(speed, 5.0)

        normal_mag = float(vnorm(f_normal_pot + f_normal_visc + (f_fins - np.dot(f_fins, axis) * axis)))
        cp_z = zc
        if normal_mag > 1e-3:
            # Centre of pressure: where the normal force would act to give the
            # same moment about the CoM.
            m_perp = vnorm(cross3(axis, cross3(moment, axis)))
            sign = 1.0 if np.dot(cross3(axis, -n_dir), moment) >= 0 else -1.0
            cp_z = zc + sign * m_perp / normal_mag
        drag = float(-np.dot(total, u))
        self.last = {"mach": mach, "q": q, "alpha": math.degrees(alpha), "tail_first": tail_first, "cp_z": cp_z,
                     "com_z": zc, "drag": drag, "normal": normal_mag, "ca": ca, "cn": cn_pot + cn_visc,
                     "source": source, "force": total, "n_dir": n_dir}
        return total, moment, self.last


# ------------------------------------------------------------ 3-D GPU CFD
class ExternalCFD:
    """Latest force coefficients from the browser's 3-D GPU flow solver.

    ``update`` receives the pressure-force coefficient vector in the world
    frame (F / (q * A_frontal)); ``coefficients(axis)`` splits it into axial
    and normal parts for the current attitude, or returns None when the data
    is older than ``max_age`` seconds (then the engineering model is used).
    """

    def __init__(self, max_age: float = 1.0) -> None:
        self.c_world = None
        self.stamp = 0.0
        self.max_age = max_age
        self.lock = threading.Lock()

    def update(self, c_world) -> None:
        c = np.asarray(c_world, float).reshape(3)
        if not np.all(np.isfinite(c)):
            return
        with self.lock:
            self.c_world = np.clip(c, -20.0, 20.0)
            self.stamp = time.monotonic()

    @property
    def fresh(self) -> bool:
        return self.c_world is not None and time.monotonic() - self.stamp < self.max_age

    def coefficients(self, axis):
        with self.lock:
            if self.c_world is None or time.monotonic() - self.stamp > self.max_age:
                return None
            c = self.c_world.copy()
        ca = float(-np.dot(c, axis))
        cn = float(vnorm(c - np.dot(c, axis) * np.asarray(axis)))
        return ca, cn
