"""3-D powered-descent guidance with propellant (G-FOLD structure, ln m).

Planner (``Planner3D``)
-----------------------
Second-order-cone program over N+1 knots with first-order-hold thrust:

    states   r_k (3), v_k (3), z_k = ln m_k
    controls u_k = T_k / m_k (3), sigma_k >= ||u_k||   (lossless relaxation)

    r' = v,  v' = u + g + d(t),  z' = -alpha * sigma,  alpha = 1/(Isp g0)

    ||u_k|| <= sigma_k                                    (SOC, dim 4)
    sigma_k <= T_max e^{-z0_k} (1 - (z_k - z0_k))         (thrust upper bound,
    sigma_k >= T_min e^{-z0_k} (1 - (z_k - z0_k))          first-order in z around
                                                           the max-burn mass z0_k)
    z_N >= ln(m_dry) - s_fuel                             (propellant limit)
    ||(x_k, y_k)|| <= tan(gamma) (h_k + s_k)              (glide-slope cone, soft)
    u_z,k >= cos(theta(t_go)) sigma_k                     (pointing: vertical at the end)
    |sigma_{k+1} - sigma_k|, |u_{x/y,k+1} - u_{x/y,k}| <= rate * dt
    r_N = 0, v_N = (0, 0, -v_td)                           (exact L1 penalty)

Objective: fuel (sum sigma dt ~ -z_N) before ignition; after ignition a smooth
thrust profile.  Drag d(t) is predicted along the previous plan.

Autopilot (``Autopilot3D``)
---------------------------
Coast engines-first -> (boost-back) -> single landing burn with a locked
touchdown time -> terminal vertical descent -> engine cutoff on leg contact.
Attitude: SO(3) error -> capped body-rate command -> angular acceleration ->
2-axis TVC (pitch/yaw) + RCS (roll always, pitch/yaw at low thrust).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from guidance import clarabel, sparse, _NativeClarabelBackend  # shared Clarabel back-ends
from rocket3d import G0, atmosphere

from pathlib import Path
import os


@dataclass
class PlanConfig3D:
    mode: str
    thrust_min: float          # N
    thrust_max: float          # N
    isp: float                 # s
    dry_mass: float            # kg
    sigma_now: float           # current thrust acceleration magnitude
    u_now: tuple               # current thrust acceleration vector
    engine_on: bool
    max_tilt_deg: float = 60.0
    touchdown_speed: float = 1.0
    drag_model: Callable | None = None
    target_alt: float = 0.0          # plan ends at the terminal gate above the pad
    u_cmd: tuple | None = None       # previous plan's command now (continuity anchor)
    target_xy: tuple = (0.0, 0.0)    # gate position (normally the pad centre)


@dataclass
class Plan3D:
    t_f: float
    dt: float
    r: np.ndarray
    v: np.ndarray
    z: np.ndarray
    u: np.ndarray
    sigma: np.ndarray
    drag: np.ndarray
    status: str
    objective: float
    terminal_position_error: float
    terminal_velocity_error: float
    glide_violation: float
    fuel_violation: float
    created_at: float = 0.0

    @property
    def knots(self) -> int:
        return len(self.sigma) - 1

    @property
    def reaches_pad(self) -> bool:
        return (self.terminal_position_error < 1.0 and self.terminal_velocity_error < 0.8
                and self.glide_violation < 1.0 and self.fuel_violation < 1e-3)

    @property
    def final_mass(self) -> float:
        return float(math.exp(self.z[-1]))

    def sample(self, tau: float):
        g = np.array([0.0, 0.0, -G0])
        if tau >= self.t_f:
            return self.r[-1].copy(), self.v[-1].copy(), self.u[-1].copy()
        tau = max(0.0, tau)
        k = min(int(tau / self.dt), self.knots - 1)
        t = tau - k * self.dt
        u0, u1 = self.u[k], self.u[k + 1]
        du = (u1 - u0) / self.dt
        d = self.drag[k]
        pos = self.r[k] + self.v[k] * t + 0.5 * u0 * t * t + du * t ** 3 / 6.0 + 0.5 * (d + g) * t * t
        vel = self.v[k] + u0 * t + 0.5 * du * t * t + (d + g) * t
        return pos, vel, u0 + du * t

    def sigma_at(self, tau: float) -> float:
        if tau >= self.t_f:
            return float(self.sigma[-1])
        tau = max(0.0, tau)
        k = min(int(tau / self.dt), self.knots - 1)
        f = (tau - k * self.dt) / self.dt
        return float((1 - f) * self.sigma[k] + f * self.sigma[k + 1])


class Planner3D:
    knots = 40
    terminal_position_weight = 400.0
    terminal_velocity_weight = 250.0
    glide_slope_weight = 30.0
    glide_slope_tan = math.tan(math.radians(65.0))
    fuel_slack_weight = 2.0e4
    lateral_position_weight = 0.008
    lateral_velocity_weight = 0.03
    lateral_jerk_factor = 8.0

    def __init__(self) -> None:
        self.solve_count = 0
        self.last_status = "idle"
        self.native_error = ""
        self._native = None
        if os.environ.get("ROCKET_NATIVE_SOLVER", "1").lower() not in {"0", "false", "no"} and sparse is not None:
            try:
                self._native = _NativeClarabelBackend(Path(__file__).with_name("clarabel_hover.dll"))
            except (AttributeError, OSError, RuntimeError) as error:
                self.native_error = str(error)
        if self._native is not None:
            self.backend = self._native.name
        elif clarabel is not None and sparse is not None:
            self.backend = "clarabel-python"
        else:
            self.backend = "feedback-fallback"
        self.lateral_rate = 4.0

    @property
    def available(self) -> bool:
        return self.backend != "feedback-fallback"

    @staticmethod
    def _tilt_limit(t_go: np.ndarray, far_deg: float) -> np.ndarray:
        far = max(8.0, far_deg)
        return np.radians(np.interp(t_go, [0.0, 1.0, 3.0, 8.0], [min(6.0, far), min(8.0, far), min(22.0, far), far]))

    def build(self, r0, v0, m0, t_f, cfg: PlanConfig3D):
        N = int(np.clip(round(t_f / 1.0), self.knots, 70))
        dt = t_f / N
        g = G0
        alpha = 1.0 / (cfg.isp * G0)
        NX = 7
        nS = NX * (N + 1)
        iu = nS
        nU = 4 * (N + 1)
        iep = iu + nU
        iem = iep + 6
        isg = iem + 6
        ifs = isg + N
        n = ifs + 1

        def S(k, c):
            return NX * k + c

        def U(k, c):
            return iu + 4 * k + c

        mid = (np.arange(N) + 0.5) * dt
        Gm = None
        if cfg.drag_model is not None:
            res = cfg.drag_model(mid)
            if isinstance(res, tuple):
                drag, Gm = np.asarray(res[0], float).reshape(N, 3), np.asarray(res[1], float).reshape(N, 3, 3)
            else:
                drag = np.asarray(res, float).reshape(N, 3)
        else:
            drag = np.zeros((N, 3))
        t_k = np.arange(N + 1) * dt
        m_ref = np.maximum(cfg.dry_mass, m0 - alpha * cfg.thrust_max * t_k)
        z0 = np.log(m_ref)
        mu2 = cfg.thrust_max * np.exp(-z0)
        mu1 = cfg.thrust_min * np.exp(-z0)

        rows, cols, vals, b = [], [], [], []

        def row(entries, rhs):
            r_ = len(b)
            for c_, v_ in entries:
                rows.append(r_)
                cols.append(c_)
                vals.append(v_)
            b.append(float(rhs))

        # ---- zero cone
        for c in range(3):
            row([(S(0, c), 1.0)], r0[c])
            row([(S(0, 3 + c), 1.0)], v0[c])
        row([(S(0, 6), 1.0)], math.log(m0))
        h2a, h2b = dt * dt / 3.0, dt * dt / 6.0
        for k in range(N):
            dk = drag[k]
            for c in range(3):
                gc = -g if c == 2 else 0.0
                lift_p, lift_v = [], []
                if Gm is not None:
                    # Body lift from tilting the thrust (and therefore the
                    # vehicle) against the flow: linear in u around the
                    # reference thrust magnitude.
                    for j in range(3):
                        if Gm[k, c, j] != 0.0:
                            lift_p.append((U(k, j), -0.5 * dt * dt * Gm[k, c, j]))
                            lift_v.append((U(k, j), -dt * Gm[k, c, j]))
                row([(S(k + 1, c), 1.0), (S(k, c), -1.0), (S(k, 3 + c), -dt), (U(k, c), -h2a), (U(k + 1, c), -h2b)] + lift_p,
                    0.5 * dt * dt * (dk[c] + gc))
                row([(S(k + 1, 3 + c), 1.0), (S(k, 3 + c), -1.0), (U(k, c), -0.5 * dt), (U(k + 1, c), -0.5 * dt)] + lift_v,
                    dt * (dk[c] + gc))
            row([(S(k + 1, 6), 1.0), (S(k, 6), -1.0), (U(k, 3), 0.5 * dt * alpha), (U(k + 1, 3), 0.5 * dt * alpha)], 0.0)
        target = (float(cfg.target_xy[0]), float(cfg.target_xy[1]), float(cfg.target_alt), 0.0, 0.0, -abs(cfg.touchdown_speed))
        for c in range(6):
            row([(S(N, c), 1.0), (iep + c, -1.0), (iem + c, 1.0)], target[c])
        n_zero = len(b)

        # ---- nonnegative cone (A x <= b)
        for c in range(6):
            row([(iep + c, -1.0)], 0.0)
            row([(iem + c, -1.0)], 0.0)
        for k in range(N):
            row([(isg + k, -1.0)], 0.0)
        row([(ifs, -1.0)], 0.0)
        for k in range(N + 1):
            # sigma <= mu2 (1 - z + z0)  ->  sigma + mu2 z <= mu2 (1 + z0)
            row([(U(k, 3), 1.0), (S(k, 6), mu2[k])], mu2[k] * (1.0 + z0[k]))
            if cfg.thrust_min > 0.0 and t_k[k] >= 0.6:
                row([(U(k, 3), -1.0), (S(k, 6), -mu1[k])], -mu1[k] * (1.0 + z0[k]))
        # propellant: z_N >= ln(m_dry) - s_fuel
        row([(S(N, 6), -1.0), (ifs, -1.0)], -math.log(cfg.dry_mass))
        t_go = t_f - t_k
        cos_tilt = np.cos(self._tilt_limit(t_go, cfg.max_tilt_deg))
        cos_tilt[0] = min(cos_tilt[0], math.cos(math.radians(80.0)))
        for k in range(N + 1):
            row([(U(k, 3), float(cos_tilt[k])), (U(k, 2), -1.0)], 0.0)
        rate_s = 0.8 * 1.2 * cfg.thrust_max / m0 * dt
        lat = self.lateral_rate * dt
        for k in range(N):
            row([(U(k + 1, 3), 1.0), (U(k, 3), -1.0)], rate_s)
            row([(U(k, 3), 1.0), (U(k + 1, 3), -1.0)], rate_s)
            for c in (0, 1):
                row([(U(k + 1, c), 1.0), (U(k, c), -1.0)], lat)
                row([(U(k, c), 1.0), (U(k + 1, c), -1.0)], lat)
        spool = 0.8 * 1.2 * cfg.thrust_max / m0 * 0.15
        row([(U(0, 3), 1.0)], cfg.sigma_now + spool)
        row([(U(0, 3), -1.0)], -(cfg.sigma_now - spool))
        if cfg.engine_on:
            # Vertical thrust is pinned near the current value (engine spool);
            # the lateral components are only pulled toward it by the cost,
            # so a lagging attitude never makes the problem infeasible.
            row([(U(0, 2), 1.0)], cfg.u_now[2] + spool + 1.0)
            row([(U(0, 2), -1.0)], -(cfg.u_now[2] - spool - 1.0))
        n_nonneg = len(b) - n_zero

        # ---- second-order cones
        soc_dims = []
        for k in range(N + 1):
            row([(U(k, 3), -1.0)], 0.0)
            row([(U(k, 0), -1.0)], 0.0)
            row([(U(k, 1), -1.0)], 0.0)
            row([(U(k, 2), -1.0)], 0.0)
            soc_dims.append(4)
        tg = self.glide_slope_tan
        for k in range(1, N + 1):
            row([(S(k, 2), -tg), (isg + k - 1, -tg)], 0.0)
            row([(S(k, 0), -1.0)], 0.0)
            row([(S(k, 1), -1.0)], 0.0)
            soc_dims.append(3)

        A = sparse.csc_matrix((vals, (rows, cols)), shape=(len(b), n))
        b_vec = np.asarray(b, float)

        # ---- objective
        w = np.full(N + 1, dt)
        w[0] *= 0.5
        w[-1] *= 0.5
        pr, pc, pv = [], [], []
        q = np.zeros(n)
        const = 0.0

        def add_p(i, j, v_):
            if i > j:
                i, j = j, i
            pr.append(i)
            pc.append(j)
            pv.append(v_)

        if cfg.mode == "fuel":
            fuel_w, accel_w, jerk_w = 1.0, 0.0, 0.004
        else:
            fuel_w, accel_w, jerk_w = 0.02, 0.05, 0.25
        for k in range(N + 1):
            q[U(k, 3)] += fuel_w * w[k]
            if accel_w > 0:
                for c in range(3):
                    add_p(U(k, c), U(k, c), 2 * accel_w * w[k])
                q[U(k, 2)] += -2 * accel_w * w[k] * g
                const += accel_w * w[k] * g * g
        # Horizontal distance from the pad axis: pulls lateral corrections
        # early, while the vehicle is high and tilting is cheap.
        lat_w = 0.0 if cfg.mode == "fuel" else self.lateral_position_weight
        if lat_w > 0.0:
            for k in range(1, N + 1):
                add_p(S(k, 0), S(k, 0), 2 * lat_w * w[k])
                add_p(S(k, 1), S(k, 1), 2 * lat_w * w[k])
                # Lateral speed, weighted up near touchdown: no fast sideways
                # sweep across the pad at the end.
                wv = self.lateral_velocity_weight * w[k] * (1.0 + 4.0 * k / N)
                add_p(S(k, 3), S(k, 3), 2 * wv)
                add_p(S(k, 4), S(k, 4), 2 * wv)
        jw = jerk_w / dt
        for k in range(N):
            for c in range(3):
                # Lateral jerk costs more: lateral thrust means tilting the
                # whole vehicle, and fast tilt reversals are what look like rocking.
                jc = jw * (self.lateral_jerk_factor if c < 2 and cfg.mode != "fuel" else 1.0)
                add_p(U(k, c), U(k, c), 2 * jc)
                add_p(U(k + 1, c), U(k + 1, c), 2 * jc)
                add_p(U(k, c), U(k + 1, c), -2 * jc)
        if cfg.engine_on:
            # Continuity with the command being flown (not with the lagging
            # measured thrust: anchoring to that makes every new plan postpone
            # the lateral correction a little more).
            # Blend: mostly what the engine is really doing now (attitude lag
            # is real), a little of the previous command (no jumps).
            anchor = cfg.u_now if cfg.u_cmd is None else tuple(0.7 * a + 0.3 * b for a, b in zip(cfg.u_now, cfg.u_cmd))
            for c in range(2):
                add_p(U(0, c), U(0, c), 1.5)
                q[U(0, c)] += -1.5 * anchor[c]
                const += 0.75 * anchor[c] ** 2
        for c in range(6):
            wgt = self.terminal_position_weight if c < 3 else self.terminal_velocity_weight
            q[iep + c] = wgt
            q[iem + c] = wgt
        q[isg:isg + N] = self.glide_slope_weight
        q[ifs] = self.fuel_slack_weight
        for i in range(n):
            add_p(i, i, 1e-7)
        P = sparse.csc_matrix((pv, (pr, pc)), shape=(n, n))
        P.sum_duplicates()
        layout = dict(N=N, dt=dt, nS=nS, iu=iu, iep=iep, iem=iem, isg=isg, ifs=ifs, NX=NX, G=Gm)
        return P, q, A, b_vec, n_zero, n_nonneg, soc_dims, layout, drag, const

    def solve(self, r0, v0, m0, t_f, cfg: PlanConfig3D) -> Plan3D | None:
        if not self.available or not (t_f > 0.3):
            return None
        P, q, A, b, nz, nn, socs, L, drag, const = self.build(r0, v0, m0, t_f, cfg)
        x, status = self._solve_conic(P, q, A, b, nz, nn, socs)
        self.last_status = status
        if x is None:
            return None
        self.solve_count += 1
        N = L["N"]
        X = x[: L["nS"]].reshape(N + 1, L["NX"])
        Uc = x[L["iu"]: L["iep"]].reshape(N + 1, 4)
        if L["G"] is not None:
            drag = drag + np.einsum("kij,kj->ki", L["G"], Uc[:N, 0:3])
        e = x[L["iep"]: L["iep"] + 6] + x[L["iem"]: L["iem"] + 6]
        full = P + P.T - sparse.diags(P.diagonal())
        obj = float(0.5 * x @ (full @ x) + q @ x + const)
        return Plan3D(
            t_f=float(t_f), dt=L["dt"], r=X[:, 0:3].copy(), v=X[:, 3:6].copy(), z=X[:, 6].copy(),
            u=Uc[:, 0:3].copy(), sigma=Uc[:, 3].copy(), drag=drag, status=status, objective=obj,
            terminal_position_error=float(np.linalg.norm(e[0:3])), terminal_velocity_error=float(np.linalg.norm(e[3:6])),
            glide_violation=float(np.max(x[L["isg"]: L["isg"] + N], initial=0.0)), fuel_violation=float(max(0.0, x[L["ifs"]])),
        )

    def _solve_conic(self, P, q, A, b, nz, nn, socs):
        if self._native is not None:
            try:
                return self._native.solve(P, q, A, b, nz, nn, socs, 100)
            except (OSError, RuntimeError, ValueError) as error:  # pragma: no cover
                self.native_error = str(error)
                self._native = None
                self.backend = "clarabel-python" if clarabel is not None else "feedback-fallback"
        if clarabel is None:
            return None, "no-solver"
        cones = [clarabel.ZeroConeT(nz), clarabel.NonnegativeConeT(nn)] + [clarabel.SecondOrderConeT(d) for d in socs]
        st = clarabel.DefaultSettings()
        st.verbose = False
        st.max_iter = 100
        try:
            res = clarabel.DefaultSolver(P, q, A, b, cones, st).solve()
        except Exception as error:  # pragma: no cover
            return None, f"python:{type(error).__name__}"
        status = str(res.status)
        if status not in ("Solved", "AlmostSolved") or res.x is None:
            return None, status
        x = np.asarray(res.x, float)
        return (x, status) if np.all(np.isfinite(x)) else (None, "nonfinite")

    def search(self, r0, v0, m0, lo, hi, cfg, evaluations=10):
        lo = max(0.6, float(lo))
        hi = max(lo + 0.5, float(hi))
        ratio = (math.sqrt(5.0) - 1.0) / 2.0
        best = None

        def f(t):
            nonlocal best
            p = self.solve(r0, v0, m0, t, cfg)
            val = p.objective if p is not None else math.inf
            if p is not None and (best is None or val < best.objective):
                best = p
            return val

        a, bb = lo, hi
        c = bb - ratio * (bb - a)
        d = a + ratio * (bb - a)
        fc, fd = f(c), f(d)
        for _ in range(max(0, evaluations - 2)):
            if fc <= fd:
                bb, d, fd = d, c, fc
                c = bb - ratio * (bb - a)
                fc = f(c)
            else:
                a, c, fc = c, d, fd
                d = a + ratio * (bb - a)
                fd = f(d)
        return best


# ---------------------------------------------------------------- autopilot
class Autopilot3D:
    """Falcon-9 style recovery: coast -> (boost-back) -> one landing burn -> terminal.

    * COAST: engine off, engines-first and aligned with the relative wind (the
      grid fins make that attitude aerodynamically stable).  A drag-aware
      ballistic prediction gives the impact point, and an ignition predictor
      decides the latest safe moment to light the engine.
    * BOOSTBACK: only when the ballistic impact point is far from the pad; the
      burn is steered so that the predicted impact moves onto the pad.
    * LANDING BURN: at ignition the touchdown time is locked and the 3-D
      G-FOLD SOCP (Planner3D) is re-solved every 0.2 s with a smooth
      objective; if the pad becomes unreachable the time moves to the
      earliest reachable one (never simply later).
    * TERMINAL: constant-deceleration vertical descent to 1.2 m/s, engine
      cut-off on leg contact.
    """

    burn_replan_period = 0.2
    coast_predict_period = 0.25
    ignition_fraction = 0.62
    burn_fraction_nominal = 0.85
    minimum_throttle = 0.20
    divert_tilt_deg = 65.0
    touchdown_speed = 1.2
    gate_height = 4.0             # G-FOLD plan ends here, vertical, at gate_speed ...
    gate_speed = 2.6              # ... then a short constant-deceleration final descent
    terminal_height = 12.0
    terminal_time_to_go = 2.0
    terminal_max_tilt_deg = 10.0
    contact_margin = 1.0          # first foot touches before the base reaches 0 m when tilted
    legs_deploy_height = 450.0
    boostback_miss = 350.0
    aero_steer_max_deg = 15.0
    lift_gradient_cap = 3.0       # body lift per radian of tilt, in units of thrust accel
    boostback_done_miss = 25.0
    # attitude loop
    k_angle = 2.0
    k_rate = 4.0
    max_rate = math.radians(30.0)

    def __init__(self, aero_model=None) -> None:
        self.planner = Planner3D()
        self.aero = aero_model
        self.reset()

    def reset(self) -> None:
        self.elapsed = 0.0
        self.plan: Plan3D | None = None
        self.next_replan = 0.0
        self.next_predict = 0.0
        self.burn_locked = False
        self.touchdown_time: float | None = None
        self.terminal_decel: float | None = None
        self.burn_fraction = self.burn_fraction_nominal
        self._failed = 0
        self._boostback = False
        self._boostback_done = False
        self._touched = False
        self._term_lat = None
        self._dir_prev = None
        self._gate_xy = np.zeros(2)
        self._dir_rate = np.zeros(3)
        self._dt = 1.0 / 240.0
        self._best_miss = math.inf
        self.impact = None
        self.ignition_in = math.inf
        self.phase = "STANDBY"
        self.last_status = "idle"
        self.time_to_go = math.nan
        self.cmd_accel = np.zeros(3)
        self.cmd_throttle = 0.0
        self.cmd_axis = np.array([0.0, 0.0, 1.0])
        self.predicted_points: list = []
        self.visited: list = []

    # ------------------------------------------------------------ helpers
    def _state(self, veh):
        r = np.array([veh.pos[0], veh.pos[1], veh.altitude])
        return r, veh.vel.copy(), veh.mass

    def _drag_model(self, veh, wind_now, t_guess):
        """Aerodynamics along the previous plan (or a decelerating guess).

        Returns ``(d0, G)``: d0 is the drag acceleration with the booster
        aligned with the flow; ``G @ u`` adds the body/grid-fin normal force
        produced by tilting the thrust vector (and the vehicle) by
        ``u_perp / sigma_ref`` against the flow.  For an engines-first
        booster this force points opposite to the tilt, which is why a plan
        that ignored it would overshoot laterally at high dynamic pressure.
        """
        prev = self.plan if self.burn_locked else None
        now = self.elapsed
        r0, v0, m0 = self._state(veh)
        legs = max(veh.legs, 1.0 if veh.controls.legs_down else 0.0)
        s = veh.spec
        sig_now = max(3.0, veh.throttle * s.thrust_sl / m0)

        def model(times):
            n = len(times)
            out = np.zeros((n, 3))
            G = np.zeros((n, 3, 3))
            if self.aero is None:
                return out, G
            for i, t in enumerate(times):
                u_ref = None
                if prev is None or t < 0.3:
                    f = max(0.0, 1.0 - t / max(t_guess, 0.5))
                    h, v, m, sig = r0[2] + v0[2] * t * 0.5, v0 * f, m0, max(sig_now, 0.6 * s.thrust_sl / m0)
                    if t < 0.3 and veh.throttle > 0.05:
                        u_ref = veh.last["thrust"] / m0
                else:
                    tau = now + t - prev.created_at
                    pos, vel, acc = prev.sample(tau)
                    h, v = pos[2], vel
                    k = min(int(max(tau, 0.0) / prev.dt), prev.knots)
                    m = float(math.exp(prev.z[k]))
                    sig = max(3.0, prev.sigma_at(tau))
                    u_ref = acc
                rho, _, _, a = atmosphere(max(0.0, h))
                vr = v - wind_now
                sp = float(np.linalg.norm(vr))
                if sp < 3.0:
                    continue
                u = vr / sp
                base = -u
                # Linearise around the reference attitude (successive
                # linearisation: each replan uses the previous plan's tilt).
                tilt = np.zeros(3)
                if u_ref is not None and float(np.linalg.norm(u_ref)) > 2.0:
                    d_ref = u_ref / np.linalg.norm(u_ref)
                    tilt = d_ref - base * float(np.dot(d_ref, base))
                    if float(np.dot(d_ref, base)) < 0.2:
                        tilt = np.zeros(3)
                axis0 = base + tilt
                axis0 /= np.linalg.norm(axis0)
                f0 = self.aero.quick_force(vr, axis0, rho, a, legs)
                e1 = np.cross(u, [0.0, 0.0, 1.0] if abs(u[2]) < 0.9 else [1.0, 0.0, 0.0])
                e1 /= np.linalg.norm(e1)
                e2 = np.cross(u, e1)
                d = math.radians(2.0)
                Gi = np.zeros((3, 3))
                for e in (e1, e2):
                    ax1 = axis0 + d * e
                    f1 = self.aero.quick_force(vr, ax1 / np.linalg.norm(ax1), rho, a, legs)
                    grad = (f1 - f0) / d / m                  # accel per radian of tilt along e
                    # Keep the net effect of a tilt in the thrust direction
                    # (limits over-reliance on a local linearisation).
                    gn = float(np.linalg.norm(grad))
                    if gn > self.lift_gradient_cap * sig:
                        grad *= self.lift_gradient_cap * sig / gn
                    Gi += np.outer(grad, e) / sig             # tilt along e ~ (e . u) / sigma
                out[i] = f0 / m - (Gi @ u_ref if u_ref is not None and np.any(tilt) else 0.0)
                G[i] = Gi
            return out, G

        return model

    def _config(self, veh, frac):
        s = veh.spec
        return PlanConfig3D(
            mode="smooth", thrust_min=self.minimum_throttle * s.thrust_sl, thrust_max=frac * s.thrust_sl, isp=s.isp_sl,
            dry_mass=s.dry_mass + 150.0, sigma_now=veh.throttle * s.thrust_sl / veh.mass,
            u_now=tuple(veh.last["thrust"] / veh.mass), engine_on=True,
            u_cmd=tuple(self.plan.sample(self.elapsed - self.plan.created_at)[2]) if self.plan is not None else None,
            # Low, short burns get a small tilt budget: a big lateral swing
            # close to the ground is exactly the "rocking" to avoid.
            # ... unless there is real sideways speed to kill (divert).
            max_tilt_deg=float(max(np.interp(veh.altitude, [20.0, 120.0, 500.0], [6.0, 14.0, self.divert_tilt_deg]),
                                   min(self.divert_tilt_deg, 2.0 * float(np.linalg.norm(veh.vel[:2]))))),
            touchdown_speed=self.gate_speed, target_alt=self.gate_height, target_xy=tuple(self._gate_xy),
        )

    @staticmethod
    def _thrust_available(veh, h):
        s = veh.spec
        _, p, _, _ = atmosphere(max(0.0, h))
        return s.thrust_vac - (s.thrust_vac - s.thrust_sl) * min(1.0, p / 101_325.0)

    # ------------------------------------------------------- predictions
    def _burn_need(self, veh, r, v, m, drag_up):
        """(thrust fraction needed for a landing burn started now, burn time, thrust direction)."""
        g = G0
        vz = float(v[2])
        h_eff = r[2] - self.gate_height - 0.45 * max(0.0, -vz)   # engine spool-up margin
        if vz > -(self.touchdown_speed + 2.0) and h_eff > 5.0:
            return 0.0, math.inf, np.array([0.0, 0.0, 1.0])
        if h_eff <= 0.5:
            return 9.0, 1.0, np.array([0.0, 0.0, 1.0])
        a_v = max(0.1, (vz * vz - self.gate_speed ** 2) / (2.0 * h_eff))
        t_b = max(0.5, (-vz - self.gate_speed) / a_v)
        a_lat = -(6.0 * r[:2] / t_b ** 2 + 4.0 * v[:2] / t_b) * 1.0
        need = np.array([a_lat[0], a_lat[1], a_v + g - drag_up / 3.0])
        frac = m * float(np.linalg.norm(need)) / self._thrust_available(veh, r[2])
        return frac, t_b, need / float(np.linalg.norm(need))

    def _ballistic(self, veh, r, v, m, wind_now, dt=0.25, t_max=240.0, stop_on_ignition=True):
        """Drag-aware coast prediction. Returns (points, impact_xy, t_impact, t_ignition, burn_time)."""
        pts = [r.copy()]
        p, u = r.copy(), v.copy()
        t = 0.0
        t_ign, t_burn = math.inf, math.inf
        while t < t_max:
            rho, _, _, a = atmosphere(max(0.0, p[2]))
            f = self.aero.quick_force(u - wind_now, None, rho, a, 0.0) if self.aero is not None else np.zeros(3)
            if stop_on_ignition and not math.isfinite(t_ign):
                frac, tb, _ = self._burn_need(veh, p, u, m, max(0.0, f[2] / m))
                if frac >= self.ignition_fraction:
                    t_ign, t_burn = t, tb
                    ign_p, ign_v = p.copy(), u.copy()
            acc = f / m + np.array([0.0, 0.0, -G0])
            u = u + acc * dt
            p = p + u * dt
            t += dt
            if len(pts) < 400:
                pts.append(p.copy())
            if p[2] <= 0.0:
                break
        impact = p[:2].copy()
        if math.isfinite(t_ign):
            # Visual prediction: coast until ignition, then a smooth arc to the pad.
            k = int(t_ign / dt)
            pts = pts[:k + 1]
            p0, v0 = ign_p, ign_v
            T = max(t_burn, 0.5)
            for s in np.linspace(0.0, 1.0, 30)[1:]:
                tt = s * T
                # Cubic Hermite from (p0, v0) to (pad, touchdown velocity).
                h00, h10, h01, h11 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s, -2 * s ** 3 + 3 * s ** 2, s ** 3 - s ** 2
                pts.append(h00 * p0 + h10 * T * v0 + h01 * np.zeros(3) + h11 * T * np.array([0, 0, -self.touchdown_speed]))
        return pts, impact, t, t_ign, t_burn

    # ------------------------------------------------------------- replan
    def _replan(self, veh, wind_now):
        r, v, m = self._state(veh)
        P = self.planner
        t_go = self.touchdown_time - self.elapsed
        cfg = self._config(veh, self.burn_fraction)
        cfg.drag_model = self._drag_model(veh, wind_now, t_go)
        plan = P.solve(r, v, m, t_go, cfg)
        if plan is None:
            self._failed += 1
            if self._failed < 3 and self.plan is not None:
                return
        else:
            self._failed = 0
        if (plan is None or not plan.reaches_pad) and (t_go > 5.0 or plan is None):
            found = self._earliest(r, v, m, cfg, t_go, veh)
            if found is not None:
                t_new, frac = found
                # Earliest reachable time at this thrust level plus a small
                # tracking margin (the smooth plan then is not saturated).
                t_new = t_new + min(1.0, 0.08 * t_new)
                self.burn_fraction = max(self.burn_fraction, frac)
                cfg.thrust_max = self.burn_fraction * veh.spec.thrust_sl
                cfg.drag_model = self._drag_model(veh, wind_now, t_new)
                smooth = P.solve(r, v, m, t_new, cfg)
                if smooth is not None:
                    plan = smooth
        self.last_status = P.last_status
        if plan is not None:
            plan.created_at = self.elapsed
            self.plan = plan
            self.touchdown_time = self.elapsed + plan.t_f
            self.last_status = plan.status

    def _earliest(self, r, v, m, cfg, t_go, veh):
        for frac in (self.burn_fraction, 0.92, 0.97):
            chk = PlanConfig3D(**{**cfg.__dict__, "mode": "fuel", "thrust_max": frac * veh.spec.thrust_sl})

            def ok(t):
                p = self.planner.solve(r, v, m, t, chk)
                return p is not None and p.reaches_pad

            lo = max(0.8, 0.6 * t_go)
            hi = max(t_go + 1.5, 1.5 * t_go)
            if not ok(hi):
                continue
            if ok(lo):
                return lo, frac
            for _ in range(6):
                mid = 0.5 * (lo + hi)
                if ok(mid):
                    hi = mid
                else:
                    lo = mid
            return hi, frac
        return None

    def _publish(self):
        p = self.plan
        if p is None:
            return
        start = max(0.0, self.elapsed - p.created_at)
        n = max(2, min(200, int(max(0.0, p.t_f - start) / 0.1) + 2))
        pts = []
        for t in np.linspace(start, p.t_f, n):
            pos, _, _ = p.sample(float(t))
            pts.append((float(pos[0]), float(pos[1]), float(max(0.0, pos[2]))))
        self.predicted_points = pts

    # ------------------------------------------------------------ command
    def command(self, veh, wind_now, dt):
        c = veh.controls
        if veh.state in ("CRASHED", "LANDED"):
            c.throttle = 0.0
            c.rcs = (0.0, 0.0, 0.0)
            c.gimbal = (0.0, 0.0)
            self.phase = "LANDED" if veh.state == "LANDED" else "STANDBY"
            self._mark()
            return
        if veh.state == "ON PAD":
            c.throttle = 0.0
            self.phase = "STANDBY"
            return
        self.elapsed += dt
        self._dt = dt
        s = veh.spec
        r, v, m = self._state(veh)
        g = G0
        h = r[2]

        # Touchdown: cut the engine as soon as a foot is on the ground.
        if veh.feet_contact > 0 or self._touched:
            self._touched = True
            c.throttle = 0.0
            self.phase = "TERMINAL"
            self._attitude(veh, np.array([0.0, 0.0, 1.0]), use_rcs=True)
            self._mark()
            return

        if h < self.legs_deploy_height:
            c.legs_down = True

        if not self.burn_locked:
            self._pre_ignition(veh, r, v, m, wind_now)
            return

        t_go = self.touchdown_time - self.elapsed
        # Hand-over to the final descent at the gate: no waiting, no hover.
        if self.terminal_decel is None and (t_go <= 0.25 or h <= self.gate_height + 0.3):
            self._start_terminal(veh, r, v, m)
        terminal = self.terminal_decel is not None
        if not terminal and self.planner.available and self.elapsed >= self.next_replan and t_go > 0.6:
            self._replan(veh, wind_now)
            self._publish()
            self.next_replan = self.elapsed + self.burn_replan_period
            t_go = self.touchdown_time - self.elapsed
        self.time_to_go = t_go + (self._terminal_time if not terminal else 0.0)
        if t_go < 8.0:
            c.legs_down = True

        drag_now = veh.last["aero"] / m
        if terminal:
            self.phase = "TERMINAL"
            accel = self._terminal_accel(veh, r, v, m, drag_now)
            tilt_lim = float(np.interp(h, [0.3, 1.5, 3.0], [0.6, 2.0, 5.0]))   # upright at contact
            max_tilt = math.radians(tilt_lim)
            self.predicted_points = [(float(r[0]), float(r[1]), float(h)), (0.0, 0.0, 0.0)]
        elif self.plan is not None:
            self.phase = "LANDING BURN"
            tau = self.elapsed - self.plan.created_at
            p_ref, v_ref, u_ff = self.plan.sample(tau)
            fb = 0.5 * (p_ref - r) + 1.6 * (v_ref - v)
            nfb = float(np.linalg.norm(fb))
            if nfb > 5.0:
                fb *= 5.0 / nfb
            accel = u_ff + fb
            max_tilt = math.radians(75.0)
        else:
            self.phase = "FALLBACK"
            a_max_now = s.thrust_sl / m
            vz_ref = -math.sqrt(2 * 0.5 * (a_max_now - g) * max(h, 0.0)) - self.touchdown_speed
            accel = np.array([-0.08 * r[0] - 0.9 * v[0], -0.08 * r[1] - 0.9 * v[1], g + 1.5 * (vz_ref - v[2])])
            max_tilt = math.radians(35.0)
        self._mark()
        self._fly_accel(veh, accel, max_tilt, floor=None if terminal else 0.9 * self.minimum_throttle)

    @property
    def _terminal_time(self):
        return 2.0 * (self.gate_height - self.contact_margin) / (self.gate_speed + self.touchdown_speed)

    def _start_terminal(self, veh, r, v, m):
        s = veh.spec
        h = max(r[2] - self.contact_margin, 0.3)
        sink = max(0.0, -v[2])
        need = (sink * sink - self.touchdown_speed ** 2) / (2.0 * h)
        self.terminal_decel = float(np.clip(need, 0.0, 0.6 * (s.thrust_sl / m - G0)))

    def _terminal_accel(self, veh, r, v, m, drag_now):
        """Constant-deceleration vertical descent from the gate to leg contact.

        The sink rate only ever decreases toward the touchdown speed (no
        hover, no climbing); lateral velocity is damped to zero.
        """
        g = G0
        h = r[2]
        a_ref = self.terminal_decel
        hh = max(0.0, h - self.contact_margin)
        vz_ref = -math.sqrt(self.touchdown_speed ** 2 + 2.0 * a_ref * hh)
        a_ff = a_ref if hh > 0.05 else 0.0
        az = g - drag_now[2] + a_ff + 2.5 * (vz_ref - v[2])
        az = float(np.clip(az, 0.4 * g, g + 2.0 * a_ref + 3.0))
        if v[2] > -0.6 * self.touchdown_speed:
            az = min(az, 0.9 * g)            # never stall above the pad
        lat = -drag_now[:2] - 1.3 * v[:2] - 0.18 * r[:2]
        # First-order filter (tau 0.5 s) so the thrust direction never jumps
        # at the hand-over or when the lateral error changes sign.
        if self._term_lat is None:
            self._term_lat = np.asarray(self.cmd_accel[:2], float).copy()
        a = min(1.0, self._dt / 0.5)
        self._term_lat = self._term_lat + a * (lat - self._term_lat)
        lat = self._term_lat
        return np.array([lat[0], lat[1], az])

    def _fly_accel(self, veh, accel, max_tilt, floor=None, align_gate=0.35):
        s = veh.spec
        c = veh.controls
        m = veh.mass
        horiz = accel[:2].copy()
        az = max(float(accel[2]), 1e-3)
        hn = float(np.linalg.norm(horiz))
        lim = math.tan(max_tilt) * az
        if hn > lim:
            horiz *= lim / hn
        accel = np.array([horiz[0], horiz[1], az])
        self.cmd_accel = accel
        mag = float(np.linalg.norm(accel))
        d = accel / mag
        # Slew-rate limit on the commanded thrust direction: a vehicle this
        # size cannot follow direction jumps between replans, and chasing
        # them is what makes it rock.  Slower near the ground.
        if self._dir_prev is not None:
            slew = math.radians(float(np.interp(veh.altitude, [2.0, 30.0, 300.0], [8.0, 15.0, 25.0]))) * self._dt
            ang = math.acos(max(-1.0, min(1.0, float(np.dot(self._dir_prev, d)))))
            if ang > slew:
                axis_r = np.cross(self._dir_prev, d)
                an = float(np.linalg.norm(axis_r))
                if an > 1e-9:
                    axis_r /= an
                    d = self._dir_prev * math.cos(slew) + np.cross(axis_r, self._dir_prev) * math.sin(slew)
                    d /= np.linalg.norm(d)
        # Angular-rate feed-forward of the (slew-limited) commanded direction.
        self._dir_rate = np.zeros(3)
        if self._dir_prev is not None and self._dt > 0:
            wv = np.cross(self._dir_prev, d) / self._dt
            self._dir_rate = veh.R.T @ wv
            self._dir_rate[2] = 0.0
        self._dir_prev = d.copy()
        err = math.acos(max(-1.0, min(1.0, float(np.dot(veh.axis, d)))))
        t_av = self._thrust_available(veh, veh.pos[2])
        if err < math.radians(25.0):
            # Small attitude lag: deliver the commanded component along d.
            thr = float(np.clip(m * mag / (t_av * math.cos(err)), 0.0, 1.0))
        else:
            thr = float(np.clip(m * mag / t_av, 0.0, 1.0)) * float(np.clip(math.cos(err), align_gate, 1.0))
        if floor is not None:
            thr = max(thr, floor)
        self.cmd_throttle = thr
        self.cmd_axis = d
        c.throttle = thr
        attitude_control(veh, d, self.k_angle, self.k_rate, self.max_rate,
                         veh.throttle < 0.25 or self.terminal_decel is not None, hold_roll=True)

    # ----------------------------------------------------- before ignition
    def _pre_ignition(self, veh, r, v, m, wind_now):
        c = veh.controls
        s = veh.spec
        if self.elapsed >= self.next_predict or self.impact is None:
            pts, impact, t_imp, t_ign, t_burn = self._ballistic(veh, r, v, m, wind_now)
            self.impact = impact
            self.ignition_in = t_ign
            self.predicted_points = [(float(p[0]), float(p[1]), float(max(0.0, p[2]))) for p in pts[:: max(1, len(pts) // 120)]]
            self.time_to_go = t_ign + t_burn if math.isfinite(t_ign) else t_imp
            self.next_predict = self.elapsed + self.coast_predict_period
        miss = float(np.linalg.norm(self.impact))

        # ---- boost-back: move the ballistic impact point onto the pad.
        airspeed = float(np.linalg.norm(v - wind_now))
        if not self._boostback_done and (self._boostback or (miss > self.boostback_miss and r[2] > 1200.0 and airspeed < 180.0)):
            self._boostback = True
            self.phase = "BOOSTBACK"
            self._mark()
            t_fall = max(5.0, self.time_to_go)
            # Aim slightly short of the pad; the landing burn removes the rest.
            dv_h = -self.impact / t_fall
            dvn = float(np.linalg.norm(dv_h))
            self._best_miss = min(self._best_miss, miss)
            if miss < self.boostback_done_miss or dvn < 0.4 or r[2] < 700.0 or miss > self._best_miss + 30.0:
                self._boostback = False
                self._boostback_done = True
            else:
                d = np.array([dv_h[0], dv_h[1], 0.3 * dvn]) / dvn
                d /= np.linalg.norm(d)
                err = math.acos(max(-1.0, min(1.0, float(np.dot(veh.axis, d)))))
                thr = float(np.clip(dvn / 8.0, 0.4, 0.9)) if err < math.radians(25.0) else 0.0
                c.throttle = thr
                self.cmd_throttle = thr
                self.cmd_axis = d
                self.cmd_accel = d * thr * s.thrust_sl / m
                self.next_predict = min(self.next_predict, self.elapsed + 0.1)
                self._attitude(veh, d, use_rcs=True)
                return

        # ---- coast, and the ignition decision.
        self.phase = "COAST"
        self._mark()
        drag_up = max(0.0, float(veh.last["aero"][2]) / m)
        frac, t_burn, burn_dir = self._burn_need(veh, r, v, m, drag_up)
        # Short, low burns (hover-slam) light a little earlier so the burn
        # lasts long enough to trim drift without a late sideways swing.
        ign = self.ignition_fraction if t_burn > 5.0 else float(np.interp(t_burn, [2.0, 5.0], [0.48, self.ignition_fraction]))
        if frac >= ign:
            self._ignite(veh, r, v, m, wind_now, t_burn)
            return
        c.throttle = 0.0
        self.cmd_throttle = 0.0
        self.cmd_accel = np.zeros(3)
        vr = v - wind_now
        sp = float(np.linalg.norm(vr))
        if self.ignition_in < 3.0:
            # Line up for the landing burn, but only pre-tilt as much as the
            # altitude warrants (a low, short burn should start nearly upright).
            cap = math.radians(float(np.clip(r[2] / 40.0, 1.0, 20.0)))
            if float(np.linalg.norm(v[:2])) > 8.0:
                cap = math.pi            # real divert: line up fully
            tl = math.acos(max(-1.0, min(1.0, float(burn_dir[2]))))
            target = burn_dir
            if tl > cap:
                hz = np.array([burn_dir[0], burn_dir[1], 0.0]); hz /= max(1e-9, float(np.linalg.norm(hz)))
                target = hz * math.sin(cap) + np.array([0.0, 0.0, math.cos(cap)])
        elif sp > 25.0:
            target = self._aero_steer(veh, r, v, m, wind_now)   # engines-first, steered by body lift
        else:
            target = np.array([0.0, 0.0, 1.0])
        self.cmd_axis = target
        self._attitude(veh, target, use_rcs=True)

    def _aero_steer(self, veh, r, v, m, wind_now):
        """Engines-first attitude with a small angle of attack whose body/grid-fin
        lift moves the ballistic impact point toward the pad (Falcon-9 style)."""
        vr = v - wind_now
        sp = float(np.linalg.norm(vr))
        u = vr / sp
        base = -u
        if self.impact is None or self.aero is None:
            return base
        t_imp = max(3.0, self.time_to_go)
        t_ign = min(max(1.5, self.ignition_in - 3.0), t_imp)   # finish before lining up for the burn
        lever = t_ign * max(1.0, t_imp - 0.5 * t_ign)
        a_des = np.array([-self.impact[0], -self.impact[1], 0.0]) / lever
        a_des -= u * float(np.dot(a_des, u))
        an = float(np.linalg.norm(a_des))
        if an < 0.02:
            return base
        e_a = a_des / an
        rho, _, _, a_snd = atmosphere(max(0.0, r[2]))
        probe = math.radians(5.0)
        best = None
        for e in (-e_a, e_a):
            axis = math.cos(probe) * base + math.sin(probe) * e
            lift = float(np.dot(self.aero.quick_force(vr, axis, rho, a_snd, veh.legs), e_a))
            if lift > 0 and (best is None or lift > best[1]):
                best = (e, lift)
        if best is None:
            return base
        e, lift = best
        alpha = float(np.clip(an * m / (lift / probe), 0.0, math.radians(self.aero_steer_max_deg)))
        return math.cos(alpha) * base + math.sin(alpha) * e

    def _ignite(self, veh, r, v, m, wind_now, t_burn):
        self._dir_prev = veh.axis.copy()
        # Very short burns (hover-slam from low altitude) cannot chase a small
        # drift without a visible sideways swing: aim part-way and let the
        # final descent take out the rest gently.
        if t_burn < 5.0:
            keep = float(np.clip((5.0 - t_burn) / 3.0, 0.0, 1.0)) * 0.6
            proj = r[:2] + v[:2] * min(t_burn, 3.0) * 0.5
            self._gate_xy = proj * keep if float(np.linalg.norm(proj)) < 6.0 else np.zeros(2)
        self.burn_locked = True
        self._boostback = False
        self.touchdown_time = self.elapsed + max(1.5, 1.05 * t_burn)
        self.next_replan = self.elapsed
        self.plan = None
        self._replan(veh, wind_now)
        self._publish()
        self.next_replan = self.elapsed + self.burn_replan_period
        self.phase = "LANDING BURN"
        self._mark()
        veh.controls.throttle = self.minimum_throttle

    def _mark(self):
        if self.phase in ("COAST", "BOOSTBACK", "LANDING BURN", "TERMINAL", "LANDED") and self.phase not in self.visited:
            self.visited.append(self.phase)

    # ------------------------------------------------------------ attitude
    def _attitude(self, veh, desired_axis, use_rcs):
        """SO(3) attitude control: TVC for pitch/yaw, RCS for roll (+ low thrust)."""
        attitude_control(veh, desired_axis, self.k_angle, self.k_rate, self.max_rate, use_rcs, hold_roll=True)


def attitude_control(veh, desired_axis, k_angle, k_rate, max_rate, use_rcs, hold_roll=True, roll_rate_cmd=0.0,
                     rate_bias=None):
    """Drive the vehicle axis toward ``desired_axis`` (world); writes controls."""
    s = veh.spec
    c = veh.controls
    R = veh.R
    zd = np.asarray(desired_axis, float)
    zd = zd / (np.linalg.norm(zd) + 1e-12)
    bx = R[:, 0]
    xd = bx - np.dot(bx, zd) * zd
    if np.linalg.norm(xd) < 1e-6:
        xd = np.cross([0.0, 1.0, 0.0], zd)
    xd /= np.linalg.norm(xd)
    yd = np.cross(zd, xd)
    Rd = np.column_stack([xd, yd, zd])
    E = Rd.T @ R - R.T @ Rd
    e = 0.5 * np.array([E[2, 1], E[0, 2], E[1, 0]])  # body-frame attitude error
    w_cmd = -k_angle * e
    m, zc, ixx, izz = veh.mass_properties()
    # Braking-distance limit: never command a rate that the actuators
    # (TVC at the current thrust + grid fins + RCS) cannot stop before the
    # target attitude -> no overshoot / wobble at low throttle.
    t_now = veh.throttle * s.thrust_sl
    tq = t_now * max(zc - s.gimbal_z, 0.5) * math.sin(s.gimbal_limit) * 0.7
    tq += 0.8 * s.rcs_torque[0] if (use_rcs or t_now <= 0.04 * s.thrust_sl) else 0.0
    tq += 0.5 * float(veh.last.get("q", 0.0)) * s.fin_moment[0]
    a_avail = max(0.02, tq / ixx)
    e_mag = float(np.linalg.norm(e[:2]))
    rate_cap = min(max_rate, 0.85 * math.sqrt(2.0 * a_avail * e_mag) + 0.004)
    n = float(np.linalg.norm(w_cmd[:2]))
    if n > rate_cap:
        w_cmd[:2] *= rate_cap / n
    w_cmd[2] = 0.0 if hold_roll else roll_rate_cmd
    if rate_bias is not None:
        w_cmd = w_cmd + rate_bias
    alpha = k_rate * (w_cmd - veh.omega)
    alpha[2] = 2.0 * (w_cmd[2] - veh.omega[2])
    tau = np.array([ixx, ixx, izz]) * alpha
    # Cancel the known aerodynamic moment (the stable booster weathervanes
    # back into the wind; without this the commanded tilt is never reached).
    m_aero = veh.last.get("aero_moment")
    if m_aero is not None:
        tau = tau - R.T @ np.asarray(m_aero, float)
    # Grid fins first (free, strong at high dynamic pressure), then TVC, then RCS.
    q_dyn = float(veh.last.get("q", 0.0))
    fins = np.zeros(3)
    if q_dyn > 50.0:
        cap = q_dyn * np.array(s.fin_moment)
        fins = np.clip(tau / cap, -1.0, 1.0)
        tau = tau - fins * cap
    c.fins = tuple(float(x) for x in fins)
    thrust = veh.throttle * s.thrust_sl
    gx = gy = 0.0
    rcs = np.zeros(3)
    if thrust > 0.04 * s.thrust_sl:
        lever = zc - s.gimbal_z
        fy = tau[0] / lever
        fx = -tau[1] / lever
        sy = float(np.clip(fx / thrust, -0.99, 0.99))
        gy = math.asin(sy)
        sx = float(np.clip(-fy / (thrust * max(0.2, math.cos(gy))), -0.99, 0.99))
        gx = math.asin(sx)
        lim = s.gimbal_limit
        # Whatever TVC cannot deliver, RCS adds.
        gx_c, gy_c = float(np.clip(gx, -lim, lim)), float(np.clip(gy, -lim, lim))
        if abs(gx_c - gx) > 1e-6 or abs(gy_c - gy) > 1e-6:
            rcs[0] = (tau[0] * (1 - gx_c / gx if gx else 0)) / s.rcs_torque[0]
            rcs[1] = (tau[1] * (1 - gy_c / gy if gy else 0)) / s.rcs_torque[1]
        gx, gy = gx_c, gy_c
    if use_rcs or thrust <= 0.04 * s.thrust_sl:
        rcs[0] = tau[0] / s.rcs_torque[0]
        rcs[1] = tau[1] / s.rcs_torque[1]
    rcs[2] = tau[2] / s.rcs_torque[2]
    c.gimbal = (gx, gy)
    c.rcs = tuple(float(v) for v in np.clip(rcs, -1.0, 1.0))
