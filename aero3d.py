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


def fin_normal_coeff(alpha: float, cn_alpha: float) -> float:
    """Normal-force coefficient of one set of grid-fin cell walls.

    Linear (cn_alpha) at small angles, peaking at 45 deg -- grid fins keep
    working to large angles instead of stalling like a planar fin."""
    return cn_alpha * math.sin(alpha) * math.cos(alpha)


def fin_mach_factor(mach: float) -> float:
    """Grid fins lose part of their effectiveness around Mach 1 (the cells
    choke) and recover supersonically."""
    return 1.0 - 0.35 * math.exp(-((mach - 1.05) / 0.22) ** 2)


def grid_fin_forces(spec, R, zc, omega_b, v_rel_w, rho, a_snd, defl, deploy):
    """Force and moment of the four grid fins.

    Each panel sits at radius fin_r_cp and height grid_fin_z and is hinged
    about its radial axis e_r.  The cells point along c = cos(d) e_z -
    sin(d) e_t (d: deflection, e_t the tangential direction).  The local
    air velocity relative to the panel (including the body rotation, which
    gives the fins their damping) is split into components along the cells
    and across them; each set of cell walls turns the crossflow back into
    the cells and receives a force along the crossflow (lift), and the panel
    has a drag along the air velocity.

    Returns (force_world, moment_world_about_com, force_body_per_fin)."""
    F_w = np.zeros(3)
    M_w = np.zeros(3)
    if deploy <= 0.01 or rho <= 0.0:
        return F_w, M_w, None
    Rt = R.T
    v_b = Rt @ v_rel_w
    zf = spec.grid_fin_z - zc
    S = spec.fin_area * deploy
    F_b = np.zeros(3)
    M_b = np.zeros(3)
    speed = float(np.linalg.norm(v_rel_w))
    fM = fin_mach_factor(speed / max(a_snd, 1.0))
    for i, az in enumerate(spec.fin_azimuth_deg):
        ph = math.radians(az)
        cph, sph = math.cos(ph), math.sin(ph)
        e_r = np.array([cph, sph, 0.0])
        e_t = np.array([-sph, cph, 0.0])
        p_b = spec.fin_r_cp * e_r + np.array([0.0, 0.0, zf])
        v_loc = v_b + cross3(omega_b, p_b)
        w = -v_loc                                   # air velocity relative to the panel
        w2 = float(np.dot(w, w))
        if w2 < 1e-6:
            continue
        d = float(defl[i])
        cd, sd = math.cos(d), math.sin(d)
        c = np.array([0.0, 0.0, cd]) - sd * e_t      # cell axis
        n_t = cd * e_t + np.array([0.0, 0.0, sd])    # across the cells, tangential walls
        wc, wt, wr = float(np.dot(w, c)), float(np.dot(w, n_t)), float(np.dot(w, e_r))
        q_loc = 0.5 * rho * w2
        a_t = math.atan2(wt, abs(wc))
        a_r = math.atan2(wr, abs(wc))
        f = q_loc * S * (fM * (fin_normal_coeff(a_t, spec.fin_cn_alpha) * n_t
                               + fin_normal_coeff(a_r, spec.fin_cn_alpha) * e_r)
                         + spec.fin_cd0 * w / math.sqrt(w2))
        F_b += f
        M_b += cross3(p_b, f)
    return R @ F_b, R @ M_b, F_b


def fin_static_force(spec, u, axis, alpha, q, mach, deploy):
    """Sum of the four grid fins at zero deflection (for trajectory
    prediction, where the roll angle is unknown): identical to the per-fin
    model for any roll angle at small angle of attack."""
    if deploy <= 0.01:
        return np.zeros(3)
    c = float(np.dot(u, axis))
    perp = u - c * axis
    pn = float(np.linalg.norm(perp))
    S = spec.fin_area * deploy
    f = -u * q * 4.0 * S * spec.fin_cd0
    if pn > 1e-9:
        f += -(perp / pn) * q * 4.0 * S * fin_mach_factor(mach) * fin_normal_coeff(alpha, spec.fin_cn_alpha)
    return f


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

    def quick_force(self, v_rel, axis, rho, a_snd, legs=0.0, fins=1.0):
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
        if pn > 1e-6:
            f += -(perp / pn) * q * a_ref * (cn_pot + cn_visc)
        f += fin_static_force(s, u, axis, alpha, q, mach, fins)
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

        # Deployed legs add drag near the base.
        leg_area = 4.0 * vehicle.legs
        f_legs = -u * q * leg_area * 0.9

        forces = [(f_axial, z_mid), (f_normal_pot, z_lead), (f_normal_visc, z_mid), (f_legs, 2.0)]
        total = np.zeros(3)
        moment = np.zeros(3)
        for f, zb in forces:
            r = R @ np.array([0.0, 0.0, zb - zc])
            total += f
            moment += cross3(r, f)
        # Body aerodynamic damping (pitch/yaw of the long body).  The grid
        # fins' damping comes from their own model (local flow includes the
        # rotation).
        omega_w = R @ vehicle.omega
        w_axial = float(np.dot(omega_w, axis))
        w_perp = omega_w - w_axial * axis
        moment += -q * a_ref * s.length ** 2 * 2.2 * w_perp / max(speed, 5.0)
        moment += -q * 1.0 * (s.diameter * 0.8) ** 2 * 0.4 * w_axial * axis / max(speed, 5.0)
        moment_body = moment.copy()
        # Grid fins: four hinged lattice panels (deflection = control).
        f_fins, m_fins, _ = grid_fin_forces(s, R, zc, vehicle.omega, v_rel, rho, a_snd,
                                            getattr(vehicle, "fin_defl", np.zeros(4)),
                                            float(getattr(vehicle, "fins_deploy", 1.0)))
        total += f_fins
        moment += m_fins

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
                     "source": source, "force": total, "n_dir": n_dir,
                     "force_fins": f_fins, "moment_fins": m_fins, "moment_body": moment_body}
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
