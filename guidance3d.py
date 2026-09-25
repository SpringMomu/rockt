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
touchdown time, planned by G-FOLD all the way to leg contact -> engine cutoff
on leg contact.  There is no separate terminal-descent phase.
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
    target_alt: float = 0.0          # plan end height above the pad (0 = touchdown)
    u_cmd: tuple | None = None       # previous plan's command now (continuity anchor)
    target_xy: tuple = (0.0, 0.0)    # gate position (normally the pad centre)
    # Final-approach sink envelope (soft): sink <= touchdown_speed + sink_gain * h.
    # Arrive slowly with thrust ~ weight instead of braking hard into contact.
    sink_gain: float = 0.0
    # Predicted altitude of each knot (from the previous plan: successive
    # approximation, like the drag).  callable(times) -> heights, or None.
    alt_ref: object = None
    approach: bool = True             # SpaceX-style approach weights (steep arrivals)
    q_now: float = 0.0                # Pa: dynamic pressure now (per-knot tilt cap), 0 = off
    q_cap_q: tuple = ()
    q_cap_deg: tuple = ()
    thrust_min_early: float = 0.0     # N: higher minimum thrust while t_go > early_tgo (0 = off)
    early_tgo: float = 0.0


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
        # (The glide-slope cone is a soft preference; a vehicle already outside
        # it must still count as able to reach the pad, otherwise every plan
        # is "unreachable" and the touchdown time drifts later and later.)
        return (self.terminal_position_error < 1.0 and self.terminal_velocity_error < 0.8
                and self.fuel_violation < 1e-3)

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
    sink_envelope_weight = 80.0
    no_climb_weight = 200.0
    # Vertical final approach (SpaceX style): lateral offset and lateral
    # speed get more expensive as touchdown approaches, so the divert is
    # done early/high and the last seconds are flown (almost) vertically.
    # Weights (per s) vs time-to-go (s).  Quadratic costs, not constraints:
    # a hard corridor makes the burn fight the engines-first aerodynamics.
    approach_t = (0.0, 6.0, 10.0, 15.0)
    approach_pos_w = (3.0, 2.0, 0.2, 0.008)
    approach_vel_w = (4.0, 3.0, 0.3, 0.03)
    # Altitude-based vertical final approach (heights from the previous plan):
    # thrust tilt limit vs predicted height, and lateral weights vs height.
    vertical_h = (0.0, 200.0, 350.0)
    vertical_thrust_w = (2.0, 2.0, 0.0)     # weight on sideways thrust vs predicted height
    vertical_h_w = (0.0, 200.0, 300.0, 500.0)
    vertical_pos_w = (3.0, 3.0, 0.5, 0.008)
    vertical_vel_w = (4.0, 4.0, 0.6, 0.03)
    # ... and in the last seconds, sideways thrust (= tilt) itself is costly:
    # the final part of the descent is flown upright.
    vertical_final_t = 5.0
    vertical_final_w = 0.4
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
        isk = ifs + 1                  # sink-envelope slacks, knots 1..N
        icl = isk + N                  # no-climb slacks, knots 1..N
        n = icl + N

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
        for k in range(N):
            row([(isk + k, -1.0)], 0.0)
            row([(icl + k, -1.0)], 0.0)
        # Never plan a climb: a landing burn that overshoots the pad must not
        # loop back up and around (vz_k <= s_k, heavily penalised).
        for k in range(1, N + 1):
            row([(S(k, 5), 1.0), (icl + k - 1, -1.0)], 0.0)
        if cfg.sink_gain > 0.0:
            # vz_k >= -(v_td + c (h_k - h_target)) - s_k   (linear in the state)
            c_s = cfg.sink_gain
            v_td = abs(cfg.touchdown_speed)
            for k in range(1, N + 1):
                row([(S(k, 5), -1.0), (S(k, 2), -c_s), (isk + k - 1, -1.0)], v_td - c_s * cfg.target_alt)
        for k in range(N + 1):
            # sigma <= mu2 (1 - z + z0)  ->  sigma + mu2 z <= mu2 (1 + z0)
            row([(U(k, 3), 1.0), (S(k, 6), mu2[k])], mu2[k] * (1.0 + z0[k]))
            if cfg.thrust_min > 0.0 and t_k[k] >= 0.6:
                mu1k = mu1[k]
                if cfg.thrust_min_early > cfg.thrust_min and t_f - t_k[k] > cfg.early_tgo:
                    mu1k = cfg.thrust_min_early * math.exp(-z0[k])
                row([(U(k, 3), -1.0), (S(k, 6), -mu1k)], -mu1k * (1.0 + z0[k]))
        # propellant: z_N >= ln(m_dry) - s_fuel
        row([(S(N, 6), -1.0), (ifs, -1.0)], -math.log(cfg.dry_mass))
        t_go = t_f - t_k
        tilt = self._tilt_limit(t_go, cfg.max_tilt_deg)
        if cfg.q_now > 0.0 and cfg.q_cap_q:
            # Dynamic pressure falls as the burn slows the vehicle (speed ~
            # linear in time): tight tilt early, looser later.
            q_k = cfg.q_now * (1.0 - t_k / max(t_f, 1e-3)) ** 2
            tilt = np.minimum(tilt, np.radians(np.interp(q_k, cfg.q_cap_q, cfg.q_cap_deg)))
        h_ref = None
        if cfg.alt_ref is not None:
            h_ref = np.maximum(0.0, np.asarray(cfg.alt_ref(t_k), float))
            # SpaceX-style vertical final approach: knots predicted below
            # ~200 m fly (almost) vertical thrust.
            pass   # (vertical final approach is soft: see the lateral-thrust weight below)
        cos_tilt = np.cos(tilt)
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
        if cfg.mode != "fuel":
            tgo_k = t_f - t_k
            if cfg.approach:
                wr = np.maximum(self.lateral_position_weight, np.interp(tgo_k, self.approach_t, self.approach_pos_w))
                wvv = np.maximum(self.lateral_velocity_weight, np.interp(tgo_k, self.approach_t, self.approach_vel_w))
            else:
                # original weights (slanted arrivals)
                wr = np.full(N + 1, self.lateral_position_weight)
                wvv = self.lateral_velocity_weight * (1.0 + 4.0 * np.arange(N + 1) / N)
            if h_ref is not None:
                # Be over the pad, with no sideways speed, by ~200 m.
                wr = np.maximum(wr, np.interp(h_ref, self.vertical_h_w, self.vertical_pos_w))
                wvv = np.maximum(wvv, np.interp(h_ref, self.vertical_h_w, self.vertical_vel_w))
            tx, ty = float(cfg.target_xy[0]), float(cfg.target_xy[1])
            for k in range(1, N + 1):
                # (x - tx)^2 + (y - ty)^2, weighted up as touchdown approaches
                for c_, t_ in ((0, tx), (1, ty)):
                    add_p(S(k, c_), S(k, c_), 2 * wr[k] * w[k])
                    q[S(k, c_)] += -2 * wr[k] * w[k] * t_
                    const += wr[k] * w[k] * t_ * t_
                # Lateral speed: no sideways sweep across the pad at the end.
                add_p(S(k, 3), S(k, 3), 2 * wvv[k] * w[k])
                add_p(S(k, 4), S(k, 4), 2 * wvv[k] * w[k])
            for k in range(N + 1):
                wu = 0.0
                if cfg.approach and tgo_k[k] < self.vertical_final_t:
                    wu = self.vertical_final_w * (1.0 - tgo_k[k] / self.vertical_final_t)
                if h_ref is not None:
                    # knots predicted below ~200 m: sideways thrust (= tilt) is expensive
                    wu = max(wu, float(np.interp(h_ref[k], self.vertical_h, self.vertical_thrust_w)))
                if wu > 0.0:
                    for c_ in (0, 1):
                        add_p(U(k, c_), U(k, c_), 2 * wu * w[k])
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
        q[isk:isk + N] = self.sink_envelope_weight
        q[icl:icl + N] = self.no_climb_weight
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
    """Falcon-9 style recovery: coast -> (boost-back) -> one landing burn to touchdown.

    * COAST: engine off, engines-first and aligned with the relative wind (the
      grid fins make that attitude aerodynamically stable).  A drag-aware
      ballistic prediction gives the impact point, and an ignition predictor
      decides the latest safe moment to light the engine.
    * BOOSTBACK: only when the ballistic impact point is far from the pad; the
      burn is steered so that the predicted impact moves onto the pad.
    * LANDING BURN: at ignition the touchdown time is locked and the 3-D
      G-FOLD SOCP (Planner3D) is re-solved every 0.2 s with a smooth
      objective; if the pad becomes unreachable the time moves to the
      earliest reachable one (never simply later).  The plan ends on the pad
      itself (base height 0, vertical, 1.2 m/s sink); the burn continues
      until a foot touches and the engine is cut.  The last ~0.6 s are flown
      on the final plan (no re-planning) with a tilt limit that tightens
      with height, so the vehicle is upright at contact.
    """

    burn_replan_period = 0.2
    coast_predict_period = 0.25
    ignition_fraction = 0.62
    burn_fraction_nominal = 0.85
    minimum_throttle = 0.20
    divert_tilt_deg = 65.0
    touchdown_speed = 1.2
    gate_height = 0.0             # steep arrivals: G-FOLD plan ends on the pad (base height 0) ...
    gate_speed = 1.2              # ... vertical, at the touchdown sink rate
    # Slanted arrivals (a real divert in the burn): the original, proven
    # scheme -- plan to a vertical gate 4 m above the pad at 2.6 m/s, then a
    # short constant-deceleration vertical descent that also damps the
    # remaining sideways speed.  Still one continuous burn (no separate phase
    # is shown); the engine cuts on leg contact.
    slew_deg = (8.0, 15.0, 25.0)  # thrust-direction slew limit (deg/s) at 2 / 30 / 300 m
    use_burn_aim = False          # (experimental, off) glide steers the predicted landing-burn touchdown point onto the pad
    use_line_aim = False          # (experimental, off) steer the retrograde-burn landing point (not the ballistic impact) onto the pad
    aim_line_factor = 0.7         # fraction of the straight-line travel after ignition (gravity bends the path down)
    burn_law = "egd"              # steep arrivals: "egd" (explicit SpaceX-style law) or "plan" (G-FOLD re-planning)
    egd_all = True                # fly every arrival with the explicit law (no legacy hand-off)
    egd_switch_h = 30.0           # below this: gentle lateral damping only
    egd_line_gain = 2.0           # gain on the straight-line landing-point error
    egd_t_frac = 0.85             # lateral horizon as a fraction of the time to touchdown
    egd_kv = 0.8                  # vertical segment lateral damping (1/s)
    egd_kp = 0.12                 # vertical segment lateral position gain (1/s^2)
    stop_guard_margin = 1.08      # stopping guard: command >= 108 % of the deceleration needed to stop
    use_legacy_divert = False     # (fallback) slanted arrivals fly the burn with guidance3d_legacy
    divert_gate_height = 4.0
    divert_gate_speed = 2.6
    contact_margin = 1.0
    divert_lift_gradient_cap = 3.0
    divert_lift_sigma_floor = 0.0
    low_rcs_height = 15.0         # RCS assists TVC close to the ground
    sink_gain = 0.5               # final approach: sink <= 1.2 + 0.5 h (m/s): 6.2 at 10 m, 2.2 at 2 m
    sink_track_height = 15.0      # the tracker enforces the envelope below this height
    legs_deploy_height = 450.0
    boostback_miss = 350.0
    aero_steer_max_deg = 15.0
    glide_law = "zem"             # "zem": glide to the pad-overhead ignition state; "impact": null the ballistic impact
    glide_vel_weight = 3.0        # m per (m/s): weight of the sideways speed at 200 m vs the offset
    glide_effort_weight = 5.0     # m per (m/s^2): keeps the glide commands gentle
    glide_ign_vel_weight = 6.0    # m per (m/s): sideways speed at ignition (steep arrival)
    approach_pos_weight = 4.0     # weight of the offset at 200 m (the landing accuracy comes first)
    glide_min_tgo = 4.0           # s: below this time to ignition only a constant push is solved for
    glide_max_accel = 12.0        # m/s^2: cap on the commanded sideways acceleration
    glide_fade_s = 4.0            # s: the steering fades out over this, ending glide_min_tgo before ignition
    glide_kv = 0.5                # last seconds before ignition: sideways damping (1/s)
    glide_kp = 0.06               # ... and position gain (1/s^2)
    egd_aero = True               # landing burn: model body lift (engines-first) in the sideways control
    egd_q_cap_deg = (35.0, 20.0, 15.0)   # angle-of-attack cap of that law at q_cap_q
    final_sink_gain = 0.4         # final approach: sink = 1.2 + 0.4 h (m/s) below the profile knee
    profile_kp = 0.8              # 1/s: sink-rate profile tracking gain
    final_aim_h = 4.0             # m: the final sideways correction completes this high
    lat_target_heights = (200.0, 150.0, 100.0, 60.0, 30.0)   # candidate heights to be over the pad
    lat_accel_gentle = 3.0        # m/s^2: (ZEM law) sideways acceleration regarded as gentle
    lat_law = "los"               # landing-burn sideways law above the target height: "los" or "zem"
    los_kv = 0.6                  # 1/s: line-of-sight velocity feedback
    los_slope_margin = 0.1        # target line may be this much steeper than the flight path (dx/dh)
    egd_path_period = 0.5         # s: refresh of the displayed path during the landing burn
    glide_period = 0.5            # s: glide / boost-back re-solve period
    path_period = 1.0             # s: refresh of the displayed path before ignition
    ballistic_period = 1.0        # s: the (legacy) ballistic prediction
    pred_coast_dt = 0.5           # s: predictor step in the coast (x4 above 12 km)
    divert_look_s = 1.0           # s: predictive ignition compares lighting now with this much later
    divert_miss_ok = math.inf     # m: acceptable predicted miss (inf: predictive ignition off -- targeting upstream makes it unnecessary)
    divert_peak_ok = 0.92         # peak thrust / available acceptable in the predicted burn
    egd_tilt_h = (0.3, 1.5, 3.0, 8.0, 20.0, 60.0)      # absolute tilt envelope near the ground ...
    egd_tilt_deg = (1.0, 3.0, 5.0, 8.0, 15.0, 60.0)    # ... (upright at contact, can lean into wind above)
    final_tgo_min = 4.0           # s: floor on its time to go (gentle near the ground)
    lat_speed_env = (1.0, 0.25)   # final approach: sideways speed <= 1.0 + 0.25 h (m/s)
    lat_speed_gain = 1.5          # 1/s: braking when above that envelope
    lift_gradient_cap = 1.0       # body lift per radian of tilt, in units of thrust accel
    boostback_done_miss = 25.0
    boostback_law = "approach"    # "approach": aim the predicted approach (coast + landing burn); "impact": ballistic impact
    boostback_done_dv = 1.0       # m/s: remaining velocity change at which the boost-back ends
    boostback_fd_dv = 5.0         # m/s: finite-difference step for its sensitivities
    boostback_J_period = 2.0      # s: refresh of those sensitivities
    boostback_effort_weight = 0.5 # m per (m/s): mild preference for a smaller velocity change
    # landing-burn robustness at high dynamic pressure (tuning switches)
    retime_max_later = 3.0        # s: largest single postponement of the touchdown time
    retime_max_earlier = 2.0      # s: largest single advance
    ignition_iterations = 2       # plans solved at ignition before flying the first one
    burn_tilt_margin_deg = 5.0    # deg above the q-based tilt cap the tracker may use (None = off)
    lift_sigma_floor = 0.5        # linearise body lift around >= this fraction of max thrust
    early_min_throttle = 0.0      # minimum throttle while t_go > early_min_tgo (0 = off)
    early_min_tgo = 8.0
    boostback_max_q = 3_000.0     # Pa: only in thin air (after stage separation); entry at 10+ km is ~10 kPa
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
        self._ref_prev: Plan3D | None = None
        self._alt_profile = None
        self.terminal_decel = None
        self._term_lat = None
        self._delegate = None
        self._wind_now = np.zeros(3)
        self.next_replan = 0.0
        self.next_predict = 0.0
        self.burn_locked = False
        self.touchdown_time: float | None = None
        self.burn_fraction = self.burn_fraction_nominal
        self._failed = 0
        self._unreach = 0
        self._boostback = False
        self._boostback_done = False
        self._bb_fired = False
        self._bb_dv = None
        self._bb_J = None
        self._bb_J_next = 0.0
        self._bb_cost = math.inf
        self._bb_next = 0.0
        self._vertical_mode = False
        self._touched = False
        self._dir_prev = None
        self._gate_xy = np.zeros(2)
        self._dir_rate = np.zeros(3)
        self._dt = 1.0 / 240.0
        self._best_miss = math.inf
        self.impact = None
        self.impact_ballistic = None
        self.aim = None
        self.burn_aim = None
        self._glide_acc = None
        self._prof_a = None
        self._lat_h = [None]
        self._next_div_check = 0.0
        self._next_glide = 0.0
        self._next_path = 0.0
        self._div_fire = False
        self.glide_info = None
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
                    pos, vel, acc, m, sig_ref = self._ref_sample(now + t)
                    h, v = pos[2], vel
                    floor_f = self.lift_sigma_floor if self._vertical_mode else self.divert_lift_sigma_floor
                    sig = max(3.0, sig_ref, floor_f * s.thrust_sl / m)
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
                    cap = self.lift_gradient_cap if self._vertical_mode else self.divert_lift_gradient_cap
                    if gn > cap * sig:
                        grad *= cap * sig / gn
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
            # At high dynamic pressure a tilted engines-first booster makes
            # body lift that nearly cancels the lateral thrust: big tilts there
            # buy almost nothing and overshoot once q drops (per-knot cap).
            q_now=float(veh.last.get("q", 0.0)), q_cap_q=tuple(self.q_cap_q), q_cap_deg=tuple(self._q_cap_table()),
            thrust_min_early=self.early_min_throttle * s.thrust_sl, early_tgo=self.early_min_tgo,
            touchdown_speed=self._gs, target_alt=self._gh, target_xy=tuple(self._gate_xy),
            sink_gain=self.sink_gain if self._vertical_mode else 0.0,
            approach=self._vertical_mode,
        )

    q_cap_q = (1500.0, 4000.0, 9000.0)      # Pa
    # Largest thrust tilt at those dynamic pressures.  A steep arrival (high
    # entry, vertical final approach) stays near the flow direction while q is
    # high -- tilting then mostly fights the body lift and makes consecutive
    # plans disagree.  A low, slanted arrival still has a real divert to fly
    # early in the burn and needs more.
    q_cap_deg = (35.0, 15.0, 8.0)           # steep arrival (vertical mode)
    q_cap_deg_divert = (75.0, 75.0, 75.0)   # slanted arrival: no cap (it has a real divert to fly)
    retime_limit_divert = False             # slanted arrivals may need one big re-timing (original behaviour)

    def _q_cap_table(self):
        return self.q_cap_deg if self._vertical_mode else self.q_cap_deg_divert

    def _q_tilt_cap(self, veh):
        """Largest useful thrust tilt (deg) at the current dynamic pressure."""
        return float(np.interp(float(veh.last.get("q", 0.0)), self.q_cap_q, self._q_cap_table()))

    @property
    def _gh(self):
        return self.gate_height if (self._vertical_mode or not self.burn_locked) else self.divert_gate_height

    @property
    def _gs(self):
        return self.gate_speed if (self._vertical_mode or not self.burn_locked) else self.divert_gate_speed

    @staticmethod
    def _thrust_available(veh, h):
        s = veh.spec
        _, p, _, _ = atmosphere(max(0.0, h))
        return s.thrust_vac - (s.thrust_vac - s.thrust_sl) * min(1.0, p / 101_325.0)

    # ------------------------------------------------------- predictions
    def _vertical_need(self, veh, r, v, m, drag_up):
        """Thrust fraction for the vertical part of a landing burn started now."""
        vz = float(v[2])
        if vz > -(self.touchdown_speed + 2.0):
            return 0.0
        h_e = r[2] - self.gate_height - 0.45 * max(0.0, -vz)
        if h_e <= 1.0:
            return 9.0
        a_v = max(0.1, (vz * vz - self.gate_speed ** 2) / (2.0 * h_e))
        return m * (a_v + G0 - drag_up / 3.0) / self._thrust_available(veh, r[2])

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
        # Lateral need is spread over the whole time left to the gate (coast +
        # burn), not over the vertical-only burn time, which collapses to ~0
        # near the apex of a boost-back arc and makes the need blow up.
        t_fall = (vz + math.sqrt(vz * vz + 2.0 * g * h_eff)) / g
        t_lat = max(t_b, t_fall)
        a_lat = -(6.0 * r[:2] / t_lat ** 2 + 4.0 * v[:2] / t_lat) * 1.0
        need = np.array([a_lat[0], a_lat[1], a_v + g - drag_up / 3.0])
        frac = m * float(np.linalg.norm(need)) / self._thrust_available(veh, r[2])
        return frac, t_b, need / float(np.linalg.norm(need))

    def _predict_burn_landing(self, veh, p, u, m, wind_now, dt=0.25):
        """Where a gravity-turn landing burn started at (p, u) would land,
        with drag, wind and the angle-of-attack cap but no steering
        correction.  The glide steers THIS point onto the pad, so the burn
        itself only has to thrust (almost) against the velocity."""
        p = np.asarray(p, float).copy()
        u = np.asarray(u, float).copy()
        vt = self.touchdown_speed
        t_max_acc = self._thrust_available(veh, p[2]) / m
        for _ in range(400):
            h = p[2]
            spd = float(np.linalg.norm(u))
            if h <= self.egd_switch_h or spd < 10.0 or u[2] > -5.0:
                break
            rho, _, _, a_snd = atmosphere(max(0.0, h))
            v_rel = u - wind_now
            sp = float(np.linalg.norm(v_rel))
            drag = self.aero.quick_force(v_rel, None, rho, a_snd, 1.0) / m if (self.aero is not None and sp > 1.0) else np.zeros(3)
            sink = -u[2]
            a_v = (sink * sink - vt * vt) / (2.0 * max(h, 0.3))
            ret = -u / spd
            a_d = a_v / max(0.05, ret[2])
            thr = np.array([*(a_d * ret[:2]), G0 + a_v]) - drag
            if sp > 15.0:
                ret_a = -v_rel / sp
                mag = float(np.linalg.norm(thr))
                d = thr / max(mag, 1e-6)
                q = 0.5 * rho * sp * sp
                cap = math.radians(float(np.interp(q, self.q_cap_q, self._q_cap_table_for(True))))
                ang = math.acos(max(-1.0, min(1.0, float(np.dot(d, ret_a)))))
                if ang > cap:
                    ax_r = np.cross(ret_a, d)
                    an = float(np.linalg.norm(ax_r))
                    if an > 1e-9:
                        ax_r /= an
                        d = ret_a * math.cos(cap) + np.cross(ax_r, ret_a) * math.sin(cap)
                        thr = d * mag
            mag = float(np.linalg.norm(thr))
            if mag > t_max_acc:
                thr *= t_max_acc / mag
            acc = thr + drag + np.array([0.0, 0.0, -G0])
            u = u + acc * dt
            p = p + u * dt
        # the last few tens of metres are flown vertically
        return p[:2] + u[:2] * max(0.0, p[2]) / max(1.0, -u[2]) * 0.5

    def _q_cap_table_for(self, vertical):
        return self.q_cap_deg if vertical else self.q_cap_deg_divert

    def _ballistic(self, veh, r, v, m, wind_now, dt=0.25, t_max=900.0, stop_on_ignition=True):
        """Drag-aware coast prediction. Returns (points, impact_xy, t_impact, t_ignition, burn_time).

        Long enough for a coast from stage separation (apogee tens of km);
        coarser steps high up where the air is thin."""
        pts = [r.copy()]
        p, u = r.copy(), v.copy()
        t = 0.0
        t_ign, t_burn = math.inf, math.inf
        k_ign = 0
        aim_p = aim_v = None      # predicted ignition state for the aim point (vertical criterion only)
        t_av_sl = veh.spec.thrust_sl
        dt0 = dt
        while t < t_max:
            dt = dt0 if p[2] < 12_000.0 else 4.0 * dt0
            rho, _, _, a = atmosphere(max(0.0, p[2]))
            f = self.aero.quick_force(u - wind_now, None, rho, a, 0.0) if self.aero is not None else np.zeros(3)
            if aim_p is None and u[2] < -(self.touchdown_speed + 2.0):
                # Vertical-only ignition criterion (the lateral term of
                # _burn_need is singular near the apex of an arc).
                h_e = p[2] - self.gate_height - 0.45 * max(0.0, -u[2])
                if h_e > 1.0:
                    a_v = (u[2] * u[2] - self.gate_speed ** 2) / (2.0 * h_e)
                    if m * (a_v + G0) / t_av_sl >= self.ignition_fraction:
                        aim_p, aim_v = p.copy(), u.copy()
                else:
                    aim_p, aim_v = p.copy(), u.copy()
            if stop_on_ignition and not math.isfinite(t_ign):
                frac, tb, _ = self._burn_need(veh, p, u, m, max(0.0, f[2] / m))
                if frac >= self.ignition_fraction:
                    t_ign, t_burn = t, tb
                    ign_p, ign_v = p.copy(), u.copy()
                    k_ign = len(pts) - 1
            acc = f / m + np.array([0.0, 0.0, -G0])
            u = u + acc * dt
            p = p + u * dt
            t += dt
            if len(pts) < 1500:
                pts.append(p.copy())
            if p[2] <= 0.0:
                break
        impact = p[:2].copy()
        # Aim point for boost-back and glide steering.  Not the ballistic
        # impact: the landing burn thrusts (roughly) against the velocity, so
        # from the predicted ignition point the vehicle continues along its
        # flight path and lands where that path meets the ground.  Putting
        # THAT point on the pad makes the landing burn a nearly pure
        # retrograde burn (small, smooth tilt) instead of a sideways divert.
        self.aim = impact.copy()
        self.burn_aim = None
        if aim_p is not None and self.use_burn_aim:
            self.burn_aim = self._predict_burn_landing(veh, aim_p, aim_v, m, wind_now)
        if aim_p is not None and self.aim_line_factor > 0.0:
            vz_i = min(-1.0, float(aim_v[2]))
            self.aim = aim_p[:2] + aim_v[:2] * (max(0.0, aim_p[2]) / -vz_i) * self.aim_line_factor
        if math.isfinite(t_ign):
            # Visual prediction: coast until ignition, then a smooth arc to the pad.
            pts = pts[:k_ign + 1]
            p0, v0 = ign_p, ign_v
            T = max(t_burn, 0.5)
            for s in np.linspace(0.0, 1.0, 30)[1:]:
                tt = s * T
                # Cubic Hermite from (p0, v0) to (pad, touchdown velocity).
                h00, h10, h01, h11 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s, -2 * s ** 3 + 3 * s ** 2, s ** 3 - s ** 2
                pts.append(h00 * p0 + h10 * T * v0 + h01 * np.zeros(3) + h11 * T * np.array([0, 0, -self.touchdown_speed]))
        return pts, impact, t, t_ign, t_burn


    @staticmethod
    def _wind_scale(z):
        """Log wind-shear profile shape (the same law the range wind follows)."""
        return min(2.6, max(0.25, math.log(max(z, 0.06) / 0.05) / math.log(10.0 / 0.05)))

    def _predict_approach(self, veh, r, v, m, wind_now, a0=None, a1=None, dt=0.5, t_max=600.0, t_ign_est=None):
        """Coast (with an optional sideways acceleration a0 + a1*t) to the
        landing-burn ignition point, then fly a modelled landing burn
        (thrust against the air-relative velocity, constant deceleration to
        the touchdown speed at the pad) down to the start of the vertical
        final approach (vertical_h[1], 200 m).

        Returns (p, u) at 200 m, the ignition state (p_i, u_i) and the time to
        ignition.  Wind: the measured wind scaled with the log shear profile."""
        p, u = r.copy(), v.copy()
        t = 0.0
        a0 = np.zeros(2) if a0 is None else a0
        a1 = np.zeros(2) if a1 is None else a1
        z_now = max(1.0, float(r[2]))
        w_ref = wind_now / self._wind_scale(z_now)
        h1 = float(self.planner.vertical_h[1])
        t_ign = None
        p_i = u_i = None
        pa_ = None
        lh_ = [None]
        g = np.array([0.0, 0.0, -G0])
        t_av_sl = veh.spec.thrust_sl
        mm = m
        while t < t_max:
            if t_ign is None:
                dtt = self.pred_coast_dt if p[2] < 12_000.0 else 4.0 * self.pred_coast_dt
            else:
                dtt = dt
            rho, _, _, a = atmosphere(max(0.0, p[2]))
            w = w_ref * self._wind_scale(p[2])
            fade = 1.0 if t_ign_est is None else float(np.clip((t_ign_est - t - self.glide_min_tgo) / self.glide_fade_s, 0.0, 1.0))
            vr = u - w
            if t_ign is None:
                f = self.aero.quick_force(vr, None, rho, a, 0.0) if self.aero is not None else np.zeros(3)
                if u[2] < -(self.touchdown_speed + 2.0):
                    h_e = p[2] - self.gate_height - 0.45 * max(0.0, -u[2])
                    av = max(0.1, (u[2] * u[2] - self.gate_speed ** 2) / (2.0 * max(h_e, 0.5)))
                    need = av + G0 - max(0.0, f[2] / m) / 3.0
                    if h_e <= 1.0 or m * need / self._thrust_available(veh, p[2]) >= self.ignition_fraction:
                        t_ign, p_i, u_i = t, p.copy(), u.copy()
                        pa_ = self._profile_a(max(0.0, float(p[2])), max(0.0, -float(u[2])))
                if t_ign is None:
                    acc = f / m + g
                    ag = np.array([*((a0 + a1 * t) * fade), 0.0])
                    spd = float(np.linalg.norm(vr))
                    if spd > 1.0:
                        # aerodynamic steering acts perpendicular to the flow
                        # (in a shallow path most of it is vertical)
                        uu = vr / spd
                        ag = ag - uu * float(np.dot(ag, uu))
                    acc += ag
                    u = u + acc * dtt
                    p = p + u * dtt
                    t += dtt
                    if p[2] <= 0.0:
                        return p, u, p, u, t
                    continue
            # ---- the landing burn, flown by the same law as the flight code
            if p[2] <= h1:
                return p, u, p_i, u_i, t_ign
            f_max = self._thrust_available(veh, p[2]) / mm
            rho, _, _, a = atmosphere(max(0.0, p[2]))
            sp = float(np.linalg.norm(vr))
            ax_guess = -vr / sp if sp > 1.0 else np.array([0.0, 0.0, 1.0])
            fz = (self.aero.quick_force(vr, ax_guess, rho, a, 1.0)[2] / mm) if self.aero is not None else 0.0
            thr, _ = self._egd_core(p, u, mm, w, fz, 1.0, f_max, pa_, lh_)
            thr = self._clip_tilt(thr, p[2])
            tn = float(np.linalg.norm(thr))
            axis = thr / tn if tn > 1e-6 else np.array([0.0, 0.0, 1.0])
            f = self.aero.quick_force(vr, axis, rho, a, 1.0) if self.aero is not None else np.zeros(3)
            acc = thr + f / mm + g
            u = u + acc * dt
            p = p + u * dt
            mm -= tn * mm / (veh.spec.isp_sl * G0) * dt
            t += dt
        return p, u, (p_i if p_i is not None else p), (u_i if u_i is not None else u), (t_ign if t_ign is not None else t)

    def _clip_tilt(self, thr, h):
        """The absolute tilt envelope near the ground, as _fly_accel applies it."""
        lim = math.tan(math.radians(float(np.interp(max(0.0, h), self.egd_tilt_h, self.egd_tilt_deg))))
        az = max(float(thr[2]), 1e-3)
        hn = float(np.linalg.norm(thr[:2]))
        if hn > lim * az:
            thr = np.array([thr[0] * lim * az / hn, thr[1] * lim * az / hn, az])
        return thr

    def _sim_burn(self, veh, r, v, m, wind_now, coast_s=0.0, dt=0.25):
        """Coast coast_s seconds, then fly the landing-burn law to the ground.
        Returns (miss, sideways speed at touchdown, peak thrust / available)."""
        p, u = r.copy(), v.copy()
        z_now = max(1.0, float(r[2]))
        w_ref = wind_now / self._wind_scale(z_now)
        g = np.array([0.0, 0.0, -G0])
        t = 0.0
        while t < coast_s and p[2] > 0.0:
            rho, _, _, a = atmosphere(max(0.0, p[2]))
            w = w_ref * self._wind_scale(p[2])
            f = self.aero.quick_force(u - w, None, rho, a, 0.0) if self.aero is not None else np.zeros(3)
            u = u + (f / m + g) * dt
            p = p + u * dt
            t += dt
        if p[2] <= 1.0:
            return math.inf, math.inf, math.inf
        pa = self._profile_a(float(p[2]), max(0.0, -float(u[2])))
        lh = [None]
        mm = m
        peak = 0.0
        t = 0.0
        while p[2] > 0.3 and t < 120.0:
            rho, _, _, a = atmosphere(max(0.0, p[2]))
            w = w_ref * self._wind_scale(p[2])
            vr = u - w
            f_max = self._thrust_available(veh, p[2]) / mm
            sp = float(np.linalg.norm(vr))
            ax_g = -vr / sp if sp > 1.0 else np.array([0.0, 0.0, 1.0])
            fz = (self.aero.quick_force(vr, ax_g, rho, a, 1.0)[2] / mm) if self.aero is not None else 0.0
            thr, _ = self._egd_core(p, u, mm, w, fz, 1.0, f_max, pa, lh)
            thr = self._clip_tilt(thr, p[2])
            tn = float(np.linalg.norm(thr))
            peak = max(peak, tn / f_max)
            axis = thr / tn if tn > 1e-6 else np.array([0.0, 0.0, 1.0])
            f = self.aero.quick_force(vr, axis, rho, a, 1.0) if self.aero is not None else np.zeros(3)
            u = u + (thr + f / mm + g) * dt
            p = p + u * dt
            mm -= tn * mm / (veh.spec.isp_sl * G0) * dt
            t += dt
            if u[2] > 1.0 and p[2] > 5.0:
                break                              # climbing: not a landing
        return float(np.linalg.norm(p[:2])), float(np.linalg.norm(u[:2])), peak

    def _boostback_dv(self, veh, r, v, m, wind_now):
        """Horizontal velocity change for the boost-back (Newton step on the
        approach predictor, 2x2 sensitivities by finite differences).
        Returns (dv, cost) where cost is the weighted predicted error (m)."""
        wv, wi, wp = self.glide_vel_weight, self.glide_ign_vel_weight, self.approach_pos_weight
        p0, u0, _, ui0, _ = self._predict_approach(veh, r, v, m, wind_now)
        res0 = np.r_[wp * p0[:2], wv * u0[:2], wi * ui0[:2]]
        d = self.boostback_fd_dv
        if self._bb_J is None or self.elapsed >= self._bb_J_next:
            # sensitivities are refreshed every boostback_J_period only
            # (quasi-Newton in between): they change slowly
            J = np.zeros((6, 2))
            for j in range(2):
                e = np.zeros(3)
                e[j] = d
                p1, u1, _, ui1, _ = self._predict_approach(veh, r, v + e, m, wind_now)
                J[:, j] = (np.r_[wp * p1[:2], wv * u1[:2], wi * ui1[:2]] - res0) / d
            self._bb_J = J
            self._bb_J_next = self.elapsed + self.boostback_J_period
        J = self._bb_J
        lam = self.boostback_effort_weight
        A = np.vstack([J, lam * np.eye(2)])
        b = np.r_[-res0, 0.0, 0.0]
        dv, *_ = np.linalg.lstsq(A, b, rcond=None)
        return dv, float(np.linalg.norm(res0))

    def _path_points(self, veh, r, v, m, wind_now, burning, glide_acc=None, dt=0.25, n_max=160):
        """Predicted path for display (the green line): the glide (current
        steering, faded before ignition like the real one) to the ignition
        point, then the landing burn flown by its own law, to the pad --
        exactly what the vehicle is going to do, so it stays put at ignition."""
        p, u = r.copy(), v.copy()
        pts = [p.copy()]
        z_now = max(1.0, float(r[2]))
        w_ref = wind_now / self._wind_scale(z_now)
        g = np.array([0.0, 0.0, -G0])
        t = 0.0
        mm = m
        t_av_sl = veh.spec.thrust_sl
        T_ign = self.glide_info[2] if (self.glide_info is not None and not burning) else None
        if not burning:
            while t < 600.0:
                dtt = self.pred_coast_dt if p[2] < 12_000.0 else 4.0 * self.pred_coast_dt
                rho, _, _, a = atmosphere(max(0.0, p[2]))
                w = w_ref * self._wind_scale(p[2])
                vr = u - w
                f = self.aero.quick_force(vr, None, rho, a, 0.0) if self.aero is not None else np.zeros(3)
                if u[2] < -(self.touchdown_speed + 2.0):
                    h_e = p[2] - self.gate_height - 0.45 * max(0.0, -u[2])
                    av = max(0.1, (u[2] * u[2] - self.gate_speed ** 2) / (2.0 * max(h_e, 0.5)))
                    need = av + G0 - max(0.0, f[2] / m) / 3.0
                    if h_e <= 1.0 or m * need / self._thrust_available(veh, p[2]) >= self.ignition_fraction:
                        break
                acc = f / m + g
                if glide_acc is not None:
                    fade = 1.0 if T_ign is None else float(np.clip((T_ign - t - self.glide_min_tgo) / self.glide_fade_s, 0.0, 1.0))
                    ag = np.array([glide_acc[0] * fade, glide_acc[1] * fade, 0.0])
                    spd = float(np.linalg.norm(vr))
                    if spd > 1.0:
                        uu = vr / spd
                        ag = ag - uu * float(np.dot(ag, uu))
                    acc += ag
                u = u + acc * dtt
                p = p + u * dtt
                t += dtt
                pts.append(p.copy())
                if p[2] <= 0.0:
                    break
            pa = self._profile_a(max(0.0, float(p[2])), max(0.0, -float(u[2])))
            lh = [None]
        else:
            pa = self._prof_a
            lh = [self._lat_h[0]]
        t = 0.0
        while p[2] > 0.05 and t < 120.0:
            rho, _, _, a = atmosphere(max(0.0, p[2]))
            w = w_ref * self._wind_scale(p[2])
            vr = u - w
            f_max = self._thrust_available(veh, p[2]) / mm
            sp = float(np.linalg.norm(vr))
            ax_g = -vr / sp if sp > 1.0 else np.array([0.0, 0.0, 1.0])
            fz = (self.aero.quick_force(vr, ax_g, rho, a, 1.0)[2] / mm) if self.aero is not None else 0.0
            thr, _ = self._egd_core(p, u, mm, w, fz, 1.0, f_max, pa, lh)
            thr = self._clip_tilt(thr, p[2])
            tn = float(np.linalg.norm(thr))
            axis = thr / tn if tn > 1e-6 else np.array([0.0, 0.0, 1.0])
            f = self.aero.quick_force(vr, axis, rho, a, 1.0) if self.aero is not None else np.zeros(3)
            u = u + (thr + f / mm + g) * dt
            p = p + u * dt
            mm -= tn * mm / (veh.spec.isp_sl * G0) * dt
            t += dt
            pts.append(p.copy())
            if u[2] > 0.5 and p[2] > 3.0:
                break
        step = max(1, len(pts) // n_max)
        out = pts[::step]
        if out[-1] is not pts[-1]:
            out.append(pts[-1])
        return [(float(q[0]), float(q[1]), float(max(0.0, q[2]))) for q in out]

    def _glide_command(self, veh, r, v, m, wind_now):
        """Sideways acceleration (x, y) for the unpowered glide.

        Target: when the landing burn (modelled, thrusting against the air
        flow, with the wind) reaches 200 m, the booster is right above the
        pad with no sideways speed -- the final 200 m are then vertical
        (SpaceX-like), and the burn itself needs (almost) no divert.  At
        ignition the booster is therefore slightly UPWIND: in a crosswind the
        drag and the air-relative braking thrust carry it downwind during the
        burn, and at the dynamic pressure of ignition an engines-first
        booster cannot fight that (tilting the thrust also tilts the body,
        whose lift pushes the other way).

        The glide acceleration profile a(t) = c0 + c1 t/T (T: time to
        ignition) is solved from numerical sensitivities of that predictor
        (drag partly cancels any sideways push, so textbook gains would be
        too weak); only c0 is flown and the solution is refreshed every
        prediction period."""
        p0, u0, pi, ui, T = self._predict_approach(veh, r, v, m, wind_now)
        self.glide_info = (p0.copy(), u0.copy(), T, pi.copy())
        if not (T > 1.0):
            return None, T
        e0 = np.array([1.0, 0.0])
        pa, ua, _, uia, _ = self._predict_approach(veh, r, v, m, wind_now, a0=e0, t_ign_est=T)
        wv, wi, lam = self.glide_vel_weight, self.glide_ign_vel_weight, self.glide_effort_weight
        cols = [(pa[0] - p0[0], ua[0] - u0[0], uia[0] - ui[0])]
        if T > self.glide_min_tgo:
            pb, ub, _, uib, _ = self._predict_approach(veh, r, v, m, wind_now, a1=e0 / T, t_ign_est=T)
            cols.append((pb[0] - p0[0], ub[0] - u0[0], uib[0] - ui[0]))
        # weighted, regularised least squares per axis:
        #   min |p200|^2 + wv^2 |u200|^2 + wi^2 |u_ign|^2 + lam^2 |c|^2
        # (u_ign: sideways speed at ignition -> a steep, SpaceX-like arrival)
        n_c = len(cols)
        wp = self.approach_pos_weight
        A = np.zeros((3 + n_c, n_c))
        for j, (jp, ju, jui) in enumerate(cols):
            A[0, j] = wp * jp
            A[1, j] = wv * ju
            A[2, j] = wi * jui
            A[3 + j, j] = lam
        acc = np.zeros(2)
        for i in range(2):
            b = np.zeros(3 + n_c)
            b[0] = -wp * p0[i]
            b[1] = -wv * u0[i]
            b[2] = -wi * ui[i]
            c, *_ = np.linalg.lstsq(A, b, rcond=None)
            acc[i] = float(c[0])
        n = float(np.linalg.norm(acc))
        if n > self.glide_max_accel:
            acc *= self.glide_max_accel / n
        # Fade the steering out before ignition: the landing burn then starts
        # from a clean attitude aligned with the flow.  (Swinging from a
        # steering angle of attack onto the burn direction at 20 kPa gives a
        # lift transient that the burn cannot take back.)
        acc *= float(np.clip((T - self.glide_min_tgo) / self.glide_fade_s, 0.0, 1.0))
        return acc, T

    # ------------------------------------------------------------- replan
    def _replan(self, veh, wind_now):
        r, v, m = self._state(veh)
        P = self.planner
        t_go = self.touchdown_time - self.elapsed
        cfg = self._config(veh, self.burn_fraction)
        cfg.drag_model = self._drag_model(veh, wind_now, t_go)
        if self._vertical_mode and self.burn_law != "egd" and r[2] < self.planner.vertical_h[1] and (
                float(np.linalg.norm(r[:2])) > 5.0 or float(np.linalg.norm(v[:2])) > 3.0):
            self._vertical_mode = False       # not over the pad by 200 m: finish the divert normally
        cfg.alt_ref = self._alt_ref(r[2], t_go) if self._vertical_mode else None
        plan = P.solve(r, v, m, t_go, cfg)
        if plan is None:
            self._failed += 1
            if self._failed < 3 and self.plan is not None:
                return
        else:
            self._failed = 0
        unreachable = plan is None or not plan.reaches_pad
        self._unreach = self._unreach + 1 if unreachable else 0
        # Re-time only if the pad stays unreachable on two consecutive plans,
        # and only in small steps: big jumps of the touchdown time are what
        # made the plan (green line) swing around right after ignition.  A
        # plan that misses a little is still the best one (exact L1 penalty),
        # and the misses shrink as the dynamic pressure (and the tilt limit)
        # relaxes.
        if unreachable and (t_go > 5.0 or plan is None) and (self._unreach >= 2 or plan is None or not self._vertical_mode):
            found = self._earliest(r, v, m, cfg, t_go, veh)
            if found is not None:
                t_new, frac = found
                # Earliest reachable time at this thrust level plus a small
                # tracking margin (the smooth plan then is not saturated).
                t_new = t_new + min(1.0, 0.08 * t_new)
                if self._vertical_mode or self.retime_limit_divert:
                    t_new = float(np.clip(t_new, t_go - self.retime_max_earlier, t_go + self.retime_max_later))
                self.burn_fraction = max(self.burn_fraction, frac)
                cfg.thrust_max = self.burn_fraction * veh.spec.thrust_sl
                cfg.drag_model = self._drag_model(veh, wind_now, t_new)
                cfg.alt_ref = self._alt_ref(r[2], t_new) if self._vertical_mode else None
                smooth = P.solve(r, v, m, t_new, cfg)
                if smooth is not None:
                    plan = smooth
        self.last_status = P.last_status
        if plan is not None:
            plan.created_at = self.elapsed
            if self.plan is not plan:
                self._ref_prev = self.plan
            self.plan = plan
            self.touchdown_time = self.elapsed + plan.t_f
            self.last_status = plan.status

    def _ref_sample(self, t_abs):
        """Reference state at absolute time t_abs for the successive
        approximations (aerodynamics, knot heights): the average of the last
        two plans.  Using only the latest plan makes consecutive plans
        alternate between two solutions (a period-2 oscillation of the
        fixed-point iteration); averaging damps it."""
        out = []
        for p in (self.plan, self._ref_prev):
            if p is None:
                continue
            tau = t_abs - p.created_at
            pos, vel, acc = p.sample(tau)
            k = min(int(max(tau, 0.0) / p.dt), p.knots)
            out.append((pos, vel, acc, float(math.exp(p.z[k])), p.sigma_at(tau)))
        if len(out) == 1:
            return out[0]
        a, b = out
        return tuple((x + y) * 0.5 for x, y in zip(a, b))

    def _alt_ref(self, h0, t_f):
        """Predicted height along the burn: the previous plan, else a
        constant-deceleration guess (h falls fast early, slowly late)."""
        now = self.elapsed
        prof = self._alt_profile

        def f(times):
            times = np.asarray(times, float)
            # Fixed profile set at ignition (height vs fraction of the burn,
            # constant-deceleration shape), independent of previous plans:
            # using the previous plan's heights here made consecutive plans
            # alternate between two solutions.
            if prof is not None:
                t0, hi = prof
                span = max(0.5, self.touchdown_time - t0)
                s_ = np.clip((now + times - t0) / span, 0.0, 1.0)
                return hi * (1.0 - s_) ** 2
            s_ = np.clip(times / max(t_f, 0.5), 0.0, 1.0)
            return h0 * (1.0 - s_) ** 2

        return f

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
            self.phase = "LANDING BURN"
            self._attitude(veh, np.array([0.0, 0.0, 1.0]), use_rcs=True)
            self._mark()
            return

        if h < self.legs_deploy_height:
            c.legs_down = True

        if not self.burn_locked:
            self._pre_ignition(veh, r, v, m, wind_now)
            return

        if self._delegate is not None:
            d = self._delegate
            d.command(veh, wind_now, dt)
            self.plan, self.touchdown_time = d.plan, d.touchdown_time
            self.predicted_points, self.time_to_go = d.predicted_points, d.time_to_go
            self.cmd_accel, self.cmd_throttle, self.cmd_axis = d.cmd_accel, d.cmd_throttle, d.cmd_axis
            self.last_status, self.burn_fraction = d.last_status, d.burn_fraction
            self.phase = "LANDING BURN" if d.phase in ("LANDING BURN", "TERMINAL") else d.phase
            self._mark()
            return

        t_go = self.touchdown_time - self.elapsed
        # Slanted arrival: at the 4 m gate hand over to the final vertical
        # descent (same burn, no hover, no separate phase).
        if (not self._vertical_mode and self.terminal_decel is None
                and (t_go <= 0.25 or h <= self.divert_gate_height + 0.3)):
            self._start_terminal(veh, r, v, m)
        terminal = self.terminal_decel is not None
        # One burn all the way to leg contact: re-plan until the last ~0.6 s,
        # then fly the final plan.  Past its end the plan's reference is
        # "on the pad, sinking at touchdown speed", so a late vehicle keeps
        # descending instead of hovering.
        egd = (self._vertical_mode or self.egd_all) and self.burn_law == "egd"
        if egd:
            if self.elapsed >= self.next_replan and self.aero is not None:
                self._wind_now = wind_now
                self.predicted_points = self._path_points(veh, r, v, m, wind_now, True)
                self.next_replan = self.elapsed + self.egd_path_period
        elif not terminal and self.planner.available and self.elapsed >= self.next_replan and t_go > 0.6:
            self._replan(veh, wind_now)
            self._publish()
            self.next_replan = self.elapsed + self.burn_replan_period
            t_go = self.touchdown_time - self.elapsed
        self.time_to_go = max(0.0, t_go)
        if t_go < 8.0:
            c.legs_down = True

        self._wind_now = wind_now
        if (self._vertical_mode or self.egd_all) and self.burn_law == "egd":
            self.phase = "LANDING BURN"
            if self.egd_aero:
                accel, max_tilt = self._egd_aero_accel(veh, r, v, m)
            else:
                accel, max_tilt = self._egd_accel(veh, r, v, m)
            self._mark()
            self._fly_accel(veh, accel, max_tilt, floor=0.9 * self.minimum_throttle if h > 1.0 else None)
            return
        if terminal:
            self.phase = "LANDING BURN"
            drag_now = veh.last["aero"] / m
            accel = self._terminal_accel(veh, r, v, m, drag_now)
            max_tilt = math.radians(float(np.interp(h, [0.3, 1.5, 3.0], [0.6, 2.0, 5.0])))   # upright at contact
            self._mark()
            self._fly_accel(veh, accel, max_tilt, floor=None)
            return
        if self.plan is not None:
            self.phase = "LANDING BURN"
            tau = self.elapsed - self.plan.created_at
            p_ref, v_ref, u_ff = self.plan.sample(tau)
            fb = 0.5 * (p_ref - r) + 1.6 * (v_ref - v)
            nfb = float(np.linalg.norm(fb))
            if nfb > 5.0:
                fb *= 5.0 / nfb
            accel = u_ff + fb
            if h < 10.0 and v[2] > -0.6 * self.touchdown_speed:
                accel[2] = min(accel[2], 0.9 * g)     # never stall or climb above the pad
            # Stopping guard (independent of the plan): never command less
            # vertical deceleration than needed to stop above the pad.  A
            # plan built on a wrong prediction must not fly the vehicle into
            # the ground.
            if v[2] < -5.0:
                a_stop = (v[2] * v[2] - self.touchdown_speed ** 2) / (2.0 * max(0.5, h - self._gh))
                az_need = (a_stop + g - max(0.0, float(veh.last["aero"][2]) / m)) * self.stop_guard_margin
                if az_need > accel[2]:
                    accel[2] = az_need
            if self._vertical_mode and h < self.sink_track_height:
                # Final approach envelope (same as the planner's): if sinking
                # faster than 1.2 + 0.5 h m/s, brake the excess.  Near the pad
                # the sink rate settles at touchdown speed with thrust ~ weight,
                # so cutting the engine at contact cannot bounce the vehicle.
                v_lim = -(self.gate_speed + self.sink_gain * max(0.0, h))
                if v[2] < v_lim:
                    accel[2] += 2.0 * (v_lim - v[2])
            # Tilt budget tightens with height: upright at contact.
            if self._vertical_mode:
                max_tilt = math.radians(float(np.interp(h, [0.3, 1.5, 3.0, 12.0, 200.0, 320.0], [0.6, 2.0, 4.0, 5.0, 6.0, 75.0])))
            else:
                max_tilt = math.radians(75.0)          # slanted arrival: original tilt budget
            if self.burn_tilt_margin_deg is not None and self._vertical_mode:
                # Feedback must not tilt the vehicle far beyond what the plan
                # is allowed to use at this dynamic pressure.
                max_tilt = min(max_tilt, math.radians(self._q_tilt_cap(veh) + self.burn_tilt_margin_deg))
        else:
            self.phase = "FALLBACK"
            a_max_now = s.thrust_sl / m
            vz_ref = -math.sqrt(2 * 0.5 * (a_max_now - g) * max(h, 0.0)) - self.touchdown_speed
            accel = np.array([-0.08 * r[0] - 0.9 * v[0], -0.08 * r[1] - 0.9 * v[1], g + 1.5 * (vz_ref - v[2])])
            max_tilt = math.radians(35.0)
        self._mark()
        self._fly_accel(veh, accel, max_tilt, floor=0.9 * self.minimum_throttle if h > 1.0 else None)

    # ------------------------------------------------ vertical burn profile
    def _profile_a(self, h0, s0):
        """Deceleration a of the upper (constant-deceleration) segment of the
        sink-rate profile that passes through (h0, s0).

        Profile: s(h) = vt + k h below h_c (gentle final approach, the
        deceleration fades to vt k), s^2 = s_c^2 + 2 a (h - h_c) above, joined
        with continuous speed and deceleration at s_c = a / k.
        Returns None when the vehicle is already slower than the gentlest
        profile, -1.0 when it is too fast for any profile (low and fast:
        plain constant deceleration to the pad instead)."""
        k, vt = self.final_sink_gain, self.touchdown_speed
        h0 = max(0.0, h0)
        a_lo = vt * k * 1.0001
        if s0 * s0 <= vt * vt + 2.0 * vt * k * h0:
            return None
        a_hi = k * (vt + k * h0)            # knot exactly at h0

        def s_at(a):
            s_c = a / k
            h_c = (s_c - vt) / k
            return math.sqrt(s_c * s_c + 2.0 * a * max(0.0, h0 - h_c))

        if s0 >= s_at(a_hi):
            return -1.0
        lo, hi = a_lo, a_hi
        for _ in range(60):
            a = 0.5 * (lo + hi)
            if s_at(a) > s0:
                hi = a
            else:
                lo = a
        return 0.5 * (lo + hi)

    def _profile(self, h, a):
        """(reference sink rate, its kinematic deceleration) at height h."""
        k, vt = self.final_sink_gain, self.touchdown_speed
        h = max(0.0, h)
        s_c = a / k
        h_c = max(0.0, (s_c - vt) / k)
        if h <= h_c:
            s = vt + k * h
            return s, s * k
        return math.sqrt(s_c * s_c + 2.0 * a * (h - h_c)), a

    def _profile_tgo(self, h, a):
        """Time to descend from h to the pad along the sink-rate profile."""
        k, vt = self.final_sink_gain, self.touchdown_speed
        h = max(0.0, h)
        a = max(a, vt * k * 1.0001)
        s_c = a / k
        h_c = max(0.0, (s_c - vt) / k)
        if h <= h_c:
            return math.log((vt + k * h) / vt) / k
        s, _ = self._profile(h, a)
        return (s - s_c) / a + math.log((vt + k * h_c) / vt) / k

    def _lat_target_h(self, r, v, h, prof_a, lat_h):
        """Height of the point above the pad that the line of sight aims at.

        The highest candidate whose line is not (much) shallower than the
        current flight path -- following it then needs no sideways push
        against the drag; steep arrivals get 200 m (vertical final approach).
        lat_h is a one-element list holding the current choice (it only moves
        down, so the target never jumps up)."""
        cur = lat_h[0] if lat_h is not None and lat_h[0] is not None else self.lat_target_heights[0]
        sink = max(1.0, -float(v[2]))
        rn = float(np.linalg.norm(r[:2]))
        slope_v = float(np.linalg.norm(v[:2])) / sink
        pick = self.lat_target_heights[-1]
        for hc in self.lat_target_heights:
            if hc > cur:
                continue
            if h <= hc + 20.0:
                pick = hc
                break
            if rn / (h - hc) <= slope_v + self.los_slope_margin:
                pick = hc
                break
        if lat_h is not None:
            lat_h[0] = pick
        return pick

    def _egd_aero_accel(self, veh, r, v, m):
        """Landing burn for steep arrivals (flight code): see _egd_core."""
        if self._prof_a is None or self._prof_a < 0.0:
            self._prof_a = self._profile_a(max(0.0, float(r[2])), max(0.0, -float(v[2])))
        acc, T = self._egd_core(r, v, m, self._wind_now, float(veh.last["aero"][2]) / m, veh.legs,
                                self._thrust_available(veh, r[2]) / m, self._prof_a, self._lat_h)
        tilt_h = float(np.interp(max(0.0, r[2]), self.egd_tilt_h, self.egd_tilt_deg))
        self.time_to_go = T
        return acc, math.radians(tilt_h)

    def _egd_core(self, r, v, m, wind, aero_z, legs, f_max, prof_a=None, lat_h=None):
        """Landing burn law for steep arrivals, with an explicit body-lift model.

        Returns (THRUST acceleration vector, time to touchdown).  Pure
        function of the state, so the glide predictor flies the very same law.

        Vertical: constant deceleration to the touchdown speed at the pad
        (plus the final sink envelope).  Sideways: ZEM/ZEV to the
        pad-overhead state (r = 0, v = 0) at vertical_h[1] (~200 m), then
        gentle damping.  At high dynamic pressure tilting the thrust of an
        engines-first booster also tilts the body, whose lift pushes the
        OTHER way; the net sideways gain per radian of tilt is
        k = F - L_alpha (thrust accel minus lift gradient) from the aero
        model.  The tilt is solved with that gain (it reverses sign when lift
        dominates and does nothing without authority), and the measured
        sideways aero force is NOT fed back (that loop cancels the thrust)."""
        g = G0
        h = max(0.0, float(r[2]))
        sink = max(0.0, -float(v[2]))
        vt = self.touchdown_speed
        if prof_a is None:
            prof_a = self._profile_a(h, sink)
        if prof_a is None:
            # already slower than the gentlest profile: follow the final approach
            prof_a = self.final_sink_gain * vt * 1.0001
            s_ref, a_ff = self._profile(h, prof_a)
        elif prof_a < 0.0 or h < 0.5:
            # low and fast: constant deceleration to the touchdown speed a
            # metre above the pad (margin: the plain law is only neutrally
            # stable and would arrive late and fast)
            s_ref = sink
            a_ff = max(0.0, (sink * sink - vt * vt)) / (2.0 * max(h - self.contact_margin, 0.3))
            prof_a = max(0.3, a_ff)
        else:
            s_ref, a_ff = self._profile(h, prof_a)
        # sink-rate profile tracking: feed-forward + proportional feedback
        # (the pure v^2/2h law is only neutrally stable and ends in a hard stop)
        a_v = a_ff + self.profile_kp * (sink - s_ref)
        az_T = g + a_v - aero_z
        if sink < vt * 0.8 and h > 1.0:
            az_T = min(az_T, 0.9 * g)
        az_T = max(0.3 * g, az_T)
        T = 2.0 * h / max(1.0, sink + vt)
        # --- desired total sideways acceleration: ZEM/ZEV onto the pad at a
        # target height -- 200 m (vertical final approach) when that is
        # reachable with a gentle sideways acceleration, otherwise the highest
        # of a few lower heights that is (a slanted arrival: a smooth arc into
        # the pad instead of translating and braking late).  The target height
        # only ever moves down (no chattering).
        h1 = self._lat_target_h(r, v, h, prof_a, lat_h)
        if h > h1 + 20.0 and sink > 5.0:
            if self.lat_law == "los":
                # Line of sight onto the point h1 above the pad: sideways speed
                # proportional to the sink rate (a straight, decelerating path
                # -- a gravity turn onto the pad).  Feed-forward keeps it on the
                # line, feedback absorbs drag and wind as they act.
                H = h - h1
                s_ref_now, a_prof = self._profile(h, prof_a)
                v_des = -r[:2] * sink / H
                a_ff = r[:2] * a_prof / H
                a_des = a_ff + self.los_kv * (v_des - v[:2])
            else:
                t1 = max(4.0, self._profile_tgo(h, prof_a) - self._profile_tgo(h1, prof_a))
                a_des = -(6.0 * r[:2] / t1 ** 2 + 4.0 * v[:2] / t1)
        else:
            # final vertical segment: ZEM/ZEV onto the pad over the profile's
            # time to go (to a point a few metres up, so the last metres are
            # only damping), gains limited so it stays gentle
            if h > self.final_aim_h:
                T_go = max(self.final_tgo_min, self._profile_tgo(h, prof_a) - self._profile_tgo(self.final_aim_h, prof_a))
                a_des = -(6.0 * r[:2] / T_go ** 2 + 4.0 * v[:2] / T_go)
            else:
                a_des = -(self.egd_kv * v[:2] + self.egd_kp * r[:2])
            # Sideways speed envelope near the ground: never chase the pad at a
            # speed the legs cannot take -- a small miss beats tipping over.
            vh = float(np.linalg.norm(v[:2]))
            v_max = self.lat_speed_env[0] + self.lat_speed_env[1] * h
            if vh > 0.5:
                e_v = v[:2] / vh
                along = float(np.dot(a_des, e_v))
                if vh > 0.7 * v_max and along > 0.0:
                    a_des = a_des - e_v * along * float(np.clip((vh - 0.7 * v_max) / (0.3 * v_max), 0.0, 1.0))
                if vh > v_max:
                    a_des = a_des - e_v * self.lat_speed_gain * (vh - v_max)
        # --- baseline direction and aero model at zero angle of attack
        v_rel = v - wind
        sp = float(np.linalg.norm(v_rel))
        up = np.array([0.0, 0.0, 1.0])
        ret = up
        f0 = np.zeros(3)
        L_a = 0.0
        q = 0.0
        if self.aero is not None and sp > 15.0:
            rho, _, _, a_snd = atmosphere(h)
            # air-relative retrograde at speed, blending to vertical when slow
            wgt = float(np.interp(sp, [20.0, 60.0], [0.0, 1.0]))
            ret = wgt * (-v_rel / sp) + (1.0 - wgt) * up
            ret /= float(np.linalg.norm(ret))
            f0 = self.aero.quick_force(v_rel, ret, rho, a_snd, legs) / m
            e = np.array([-ret[2], 0.0, ret[0]]) if abs(ret[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
            e -= ret * float(np.dot(e, ret))
            e /= float(np.linalg.norm(e))
            probe = math.radians(6.0)
            f1 = self.aero.quick_force(v_rel, math.cos(probe) * ret + math.sin(probe) * e, rho, a_snd, legs) / m
            L_a = -float(np.dot(f1 - f0, e)) / probe          # > 0: lift opposes the tilt
            q = 0.5 * rho * sp * sp
        F = min(f_max, az_T / max(0.2, float(ret[2])))       # thrust accel if flown along ret
        k = F - L_a
        base_lat = F * ret[:2] + f0[:2]
        delta = np.zeros(2)
        if abs(k) > 3.0:
            delta = (a_des - base_lat) / k
        cap = math.tan(math.radians(float(np.interp(q, self.q_cap_q, self.egd_q_cap_deg))))
        dn = float(np.linalg.norm(delta))
        if dn > cap:
            delta *= cap / dn
        d = np.array([ret[0] + delta[0], ret[1] + delta[1], ret[2]])
        d /= float(np.linalg.norm(d))
        acc = d * min(f_max, az_T / max(0.2, float(d[2])))
        return acc, T

    def _egd_accel(self, veh, r, v, m):
        """SpaceX-style landing burn law (steep arrivals).

        Vertical: constant deceleration that brings the sink rate to the
        touchdown speed exactly at the pad (the same model the ignition
        predictor uses, so the burn starts where it should).
        Lateral: zero-effort-miss / zero-effort-velocity guidance that nulls
        the horizontal offset and speed by the time the vehicle is ~200 m up;
        below that the thrust stays (nearly) vertical and only damps what is
        left.  The command is a continuous function of the state: no
        re-planning, nothing to jump.
        """
        g = G0
        h = max(0.0, r[2])
        drag = veh.last["aero"] / m
        sink = max(0.0, -float(v[2]))
        vt = self.touchdown_speed
        # --- vertical: constant deceleration to vt at h = 0 (plus the final
        # sink envelope: never faster than vt + 0.5 h near the pad)
        a_v = (sink * sink - vt * vt) / (2.0 * max(h, 0.3))
        v_env = vt + self.sink_gain * h
        az = g + a_v - drag[2]
        if sink > v_env and h < self.sink_track_height:
            az += 2.0 * (sink - v_env)
        if sink < vt * 0.8 and h > 1.0:
            az = min(az, 0.9 * g)            # never stall or climb above the pad
        az = max(0.3 * g, az)
        # --- lateral: finish by ~vertical_h[1] (200 m), time from the
        # constant-deceleration profile
        T = 2.0 * h / max(1.0, sink + vt)                     # time to touchdown
        spd = float(np.linalg.norm(v))
        if h > self.egd_switch_h and spd > 10.0 and v[2] < -5.0:
            # Gravity turn: the kinematic deceleration points against the
            # velocity, so horizontal and vertical speed shrink in the same
            # ratio and the path is a straight line to where it meets the
            # ground (the glide put that point near the pad).  Its size is set
            # by the vertical requirement a_v.
            ret = -v / spd
            a_d = a_v / max(0.05, ret[2])
            a_lat = a_d * ret[:2]
            # Small correction that moves the straight-line landing point
            # (r + v T / 2 for a constant deceleration) onto the pad.
            L = r[:2] + v[:2] * T * 0.5
            a_lat = a_lat - self.egd_line_gain * 2.0 * L / max(T * T, 1.0)
        else:
            # final vertical segment: damp what is left, gently
            a_lat = -(self.egd_kv * v[:2] + self.egd_kp * r[:2])
        a_lat = a_lat - drag[:2]
        acc = np.array([a_lat[0], a_lat[1], az])
        # Angle-of-attack budget: body lift comes from the angle between the
        # thrust axis and the (reversed) airflow, not from the tilt from
        # vertical.  Keep the thrust within the dynamic-pressure cap of
        # retrograde; retrograde itself is always allowed.
        v_rel = v - self._wind_now
        sp = float(np.linalg.norm(v_rel))
        if sp > 15.0:
            ret_a = -v_rel / sp
            mag = float(np.linalg.norm(acc))
            d = acc / mag
            cap = math.radians(float(np.interp(float(veh.last.get("q", 0.0)), self.q_cap_q, self.q_cap_deg)))
            ang = math.acos(max(-1.0, min(1.0, float(np.dot(d, ret_a)))))
            if ang > cap:
                axis_r = np.cross(ret_a, d)
                an = float(np.linalg.norm(axis_r))
                if an > 1e-9:
                    axis_r /= an
                    d = ret_a * math.cos(cap) + np.cross(axis_r, ret_a) * math.sin(cap)
                    acc = d * mag
        # absolute tilt budget near the ground (upright at contact)
        tilt_h = float(np.interp(h, [0.3, 1.5, 3.0, 12.0, 60.0], [0.6, 2.0, 4.0, 6.0, 60.0]))
        self.time_to_go = T
        return acc, math.radians(tilt_h)

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
            slew = math.radians(float(np.interp(veh.altitude, [2.0, 30.0, 300.0], self.slew_deg))) * self._dt
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
                         veh.throttle < 0.25 or veh.altitude < self.low_rcs_height or self.terminal_decel is not None,
                         hold_roll=True)

    # ----------------------------------------------------- before ignition
    def _pre_ignition(self, veh, r, v, m, wind_now):
        c = veh.controls
        s = veh.spec
        new_mode = self.egd_all and self.burn_law == "egd" and self.aero is not None
        if self.elapsed >= self.next_predict or self.impact is None:
            pts, impact, t_imp, t_ign, t_burn = self._ballistic(veh, r, v, m, wind_now)
            self.impact = impact
            self.impact_ballistic = impact
            self.ignition_in = t_ign
            if not new_mode:
                self.predicted_points = [(float(p[0]), float(p[1]), float(max(0.0, p[2]))) for p in pts[:: max(1, len(pts) // 120)]]
            self.time_to_go = t_ign + t_burn if math.isfinite(t_ign) else t_imp
            self.next_predict = self.elapsed + (self.ballistic_period if new_mode else self.coast_predict_period)
            if not new_mode:
                self._glide_acc = None
                if self.glide_law == "zem" and self.aero is not None and not self._boostback:
                    self._glide_acc, _ = self._glide_command(veh, r, v, m, wind_now)
        if new_mode:
            # staggered: glide solve and display path never on the same step
            if self.elapsed >= self._next_glide:
                self._next_glide = self.elapsed + self.glide_period
                if self.glide_law == "zem" and not self._boostback:
                    self._glide_acc, T_i = self._glide_command(veh, r, v, m, wind_now)
                    if T_i is not None and math.isfinite(T_i):
                        self.ignition_in = T_i
                else:
                    self._glide_acc = None
                self._next_path = max(self._next_path, self.elapsed + 0.5 * self.glide_period)
            elif self.elapsed >= self._next_path:
                self._next_path = self.elapsed + self.path_period
                self.predicted_points = self._path_points(veh, r, v, m, wind_now, False, self._glide_acc)
        miss = float(np.linalg.norm(self.impact))

        # ---- boost-back: move the ballistic impact point onto the pad.
        airspeed = float(np.linalg.norm(v - wind_now))
        rho_now = atmosphere(max(0.0, r[2]))[0]
        q_now = 0.5 * rho_now * airspeed * airspeed
        # Boost-back needs the vehicle to be able to turn around: low airspeed
        # (low scenarios) or thin air (after stage separation, tens of km up).
        can_turn = airspeed < 180.0 or q_now < self.boostback_max_q
        if not self._boostback_done and (self._boostback or (miss > self.boostback_miss and r[2] > 1200.0 and can_turn)):
            self._boostback = True
            self.phase = "BOOSTBACK"
            self._mark()
            if self.boostback_law == "approach" and self.aero is not None:
                # Velocity change that puts the whole approach -- coast,
                # glide-free, and the landing burn flown by its own law --
                # over the pad at 200 m with a steep arrival (the same target
                # the glide uses afterwards).
                if self._bb_dv is None or self.elapsed >= self._bb_next:
                    self._bb_dv, self._bb_cost = self._boostback_dv(veh, r, v, m, wind_now)
                    self._bb_next = self.elapsed + self.glide_period
                    self._next_path = max(self._next_path, self.elapsed + 0.5 * self.glide_period)
                dv_h = self._bb_dv
                dvn = float(np.linalg.norm(dv_h))
                cost = self._bb_cost
                self._best_miss = min(self._best_miss, cost)
                diverging = self._bb_fired and cost > 1.3 * self._best_miss + 200.0
                done = dvn < self.boostback_done_dv
            else:
                t_fall = max(5.0, self.time_to_go)
                # Aim slightly short of the pad; the landing burn removes the rest.
                dv_h = -self.impact / t_fall
                dvn = float(np.linalg.norm(dv_h))
                self._best_miss = min(self._best_miss, miss)
                diverging = self._bb_fired and miss > self._best_miss + 30.0   # only after thrust has acted
                done = miss < self.boostback_done_miss or dvn < 0.4
            if done or r[2] < 700.0 or diverging:
                self._boostback = False
                self._boostback_done = True
            else:
                d = np.array([dv_h[0], dv_h[1], 0.3 * dvn]) / dvn
                d /= np.linalg.norm(d)
                err = math.acos(max(-1.0, min(1.0, float(np.dot(veh.axis, d)))))
                t_av = self._thrust_available(veh, r[2])
                # Remove the remaining velocity error in ~1 s (fine control at
                # the very end, but no long tail: a slow boost-back ends up
                # flying nose-first into a rising dynamic pressure, which
                # weathervanes the booster off the burn direction).
                # In denser air keep enough thrust for TVC authority: flying
                # back toward the pad the booster moves nose-first, which is
                # aerodynamically unstable, and at low throttle the gimbal
                # cannot hold it (it weathervanes off and the burn stops).
                floor = 0.3 if q_now > 1000.0 else 0.08
                if err < math.radians(25.0):
                    thr = float(np.clip(m * dvn / (1.0 * t_av), floor, 0.9))
                else:
                    thr = 0.2 if (q_now > 1000.0 and self._bb_fired) else 0.0
                c.throttle = thr
                if thr > 0.0 and not self._bb_fired:
                    self._bb_fired = True
                    self._best_miss = self._bb_cost if self.boostback_law == "approach" else miss
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
        if self.egd_all and self.burn_law == "egd":
            # the explicit law and the glide handle the sideways part: light
            # on the vertical need only (the criterion the predictors use)
            frac = self._vertical_need(veh, r, v, m, drag_up)
        # Short, low burns (hover-slam) light a little earlier so the burn
        # lasts long enough to trim drift without a late sideways swing.
        ign = self.ignition_fraction if t_burn > 5.0 else float(np.interp(t_burn, [2.0, 5.0], [0.48, self.ignition_fraction]))
        if frac >= ign:
            self._ignite(veh, r, v, m, wind_now, t_burn)
            return
        if self._divert_ignition(veh, r, v, m, wind_now):
            self._ignite(veh, r, v, m, wind_now, t_burn)
            return
        c.throttle = 0.0
        self.cmd_throttle = 0.0
        self.cmd_accel = np.zeros(3)
        vr = v - wind_now
        sp = float(np.linalg.norm(vr))
        steep = (r[2] > 400.0 and float(np.linalg.norm(r[:2])) < 0.3 * max(1.0, r[2] - self.planner.vertical_h[1])
                 and float(np.linalg.norm(v[:2])) < 0.3 * max(1.0, -float(v[2])))
        if self.glide_law == "zem" and (steep or self.egd_all) and sp > 25.0 and self.aero is not None:
            # Steep arrival (the glide did its job): no pre-tilt of the engine
            # -- at this dynamic pressure the body lift of a tilted,
            # engines-first booster outweighs the sideways thrust and would
            # push it the wrong way.  Keep steering aerodynamically.
            target = self._aero_steer(veh, r, v, m, wind_now)
        elif self.ignition_in < 3.0:
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
        tgt = self.aim if (self.use_line_aim and self.aim is not None) else self.impact
        if self.use_burn_aim and self.burn_aim is not None:
            tgt = self.burn_aim
        if self.glide_law == "zem":
            if self._glide_acc is None:
                return base
            a_des = np.array([self._glide_acc[0], self._glide_acc[1], 0.0])
        else:
            t_imp = max(3.0, self.time_to_go)
            t_ign = min(max(1.5, self.ignition_in - 3.0), t_imp)   # finish before lining up for the burn
            lever = t_ign * max(1.0, t_imp - 0.5 * t_ign)
            a_des = np.array([-tgt[0], -tgt[1], 0.0]) / lever
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

    def _divert_ignition(self, veh, r, v, m, wind_now):
        """Predictive ignition for arrivals that still need a sideways
        correction: light now if waiting one more second would make the
        landing burn (flown by the same law) miss the pad or run out of
        thrust margin.  Checked at the prediction rate, only when the burn is
        near and the sideways offset/speed is significant."""
        if not self.egd_all or self.aero is None or v[2] > -10.0:
            return False
        if self.elapsed < self._next_div_check:
            return self._div_fire
        self._next_div_check = self.elapsed + self.coast_predict_period
        self._div_fire = False
        lat = float(np.linalg.norm(r[:2])) + 4.0 * float(np.linalg.norm(v[:2]))
        if lat < 40.0 or not (self.ignition_in < 12.0):
            return False
        miss0, vh0, pk0 = self._sim_burn(veh, r, v, m, wind_now, 0.0)
        miss1, vh1, pk1 = self._sim_burn(veh, r, v, m, wind_now, self.divert_look_s)
        bad1 = miss1 > self.divert_miss_ok or pk1 > self.divert_peak_ok
        worse = (miss1 > miss0 + 0.5) or (pk1 > pk0 + 0.02)
        self._div_fire = bool(bad1 and worse)
        return self._div_fire

    def _ignite(self, veh, r, v, m, wind_now, t_burn):
        self._dir_prev = veh.axis.copy()
        self._gate_xy = np.zeros(2)          # the burn itself lands on the pad centre
        # SpaceX-style vertical final approach (last ~200 m vertical) when the
        # geometry allows it: ignition high enough, and the descent already
        # steep (small sideways offset/speed relative to height/sink).  A low,
        # slanted arrival (e.g. 1 km off from only 2 km up) cannot finish the
        # divert above 200 m and uses the time-based approach instead.
        above = max(1.0, r[2] - self.planner.vertical_h[1])
        self._vertical_mode = bool(r[2] > 400.0
                                   and float(np.linalg.norm(r[:2])) < 0.3 * above
                                   and float(np.linalg.norm(v[:2])) < 0.3 * max(1.0, -float(v[2])))
        self.burn_locked = True
        self._boostback = False
        # + 2 s for the gentle final approach of a steep arrival; a slanted
        # arrival plans to the 4 m gate (original timing)
        self.touchdown_time = self.elapsed + max(1.5, 1.05 * t_burn) + (2.0 if self._vertical_mode else 0.0)
        self.next_replan = self.elapsed
        self.plan = None
        self._ref_prev = None
        # Converge the successive approximations (aerodynamics along the plan,
        # knot heights) before flying: the first plan only has a crude guess
        # to linearise around, and flying it would make the second plan jump.
        self._alt_profile = (self.elapsed, float(r[2]))
        self._prof_a = None
        self._lat_h = [None]
        self.terminal_decel = None
        self._term_lat = None
        self._delegate = None
        if not self._vertical_mode and self.use_legacy_divert and not self.egd_all:
            # Slanted arrival (a real divert in the burn): fly the burn with the
            # original, proven landing guidance (guidance3d_legacy): G-FOLD to a
            # vertical gate 4 m above the pad, then a short vertical descent.
            import guidance3d_legacy
            d = guidance3d_legacy.Autopilot3D(self.aero)
            d.elapsed = self.elapsed
            d._ignite(veh, r, v, m, wind_now, t_burn)
            self._delegate = d
            self.burn_locked = True
            self.plan = d.plan
            self.touchdown_time = d.touchdown_time
            self.predicted_points = d.predicted_points
            self.phase = "LANDING BURN"
            self._mark()
            return
        if self.egd_all and self.burn_law == "egd":
            # the explicit law needs no plan; the display path is refreshed in flight
            self.next_replan = self.elapsed
        else:
            for _ in range(self.ignition_iterations if self._vertical_mode else 1):
                self._replan(veh, wind_now)
            self._publish()
            self.next_replan = self.elapsed + self.burn_replan_period
        self.phase = "LANDING BURN"
        self._mark()
        veh.controls.throttle = self.minimum_throttle

    def _mark(self):
        if self.phase in ("COAST", "BOOSTBACK", "LANDING BURN", "LANDED") and self.phase not in self.visited:
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
