"""Successive convexification (SCvx / penalized trust region) landing-burn planner.

The landing burn is a nonconvex optimal-control problem because of the
aerodynamics: the drag and the body/grid-fin lift depend on the velocity
relative to the air AND on the direction of the thrust (an engines-first
booster tilts its whole body with the thrust), the density on the height,
and the thrust bounds on the mass.  G-FOLD makes the problem convex by
approximating all of that once; when the approximation is poor the next
re-plan lands somewhere else and the plans jump.

SCvx instead iterates: linearise the TRUE (nonlinear) dynamics around the
current reference trajectory, solve the convex sub-problem (a second-order
cone program, Clarabel), use the solution as the next reference, and repeat
until the solution stops changing.  A converged plan is self-consistent with
the nonlinear model, so the next re-plan (warm-started from it) agrees with
it.  Standard ingredients (Mao, Szmuk & Acikmese 2016-2018; Reynolds &
Mesbahi "PTR"):

* virtual control nu on the dynamics (L1 penalty) -> every sub-problem is
  feasible, the penalty drives nu to zero at convergence;
* quadratic trust-region penalty on the change of states / controls / time;
* free final time (the time step is a decision variable, linearised);
* lossless convexification of the thrust bounds with z = ln m.

Variables per knot k = 0..N: position r (3), velocity v (3), z = ln m,
thrust acceleration a = T/m (3) and its bound s >= |a|.  Controls are first-
order hold; the discretisation is exact for an acceleration that varies
linearly over the interval.

Constraints / objective that make it look like a Falcon 9 landing:
* terminal: on the pad, sinking at the touchdown speed, upright;
* tilt of the thrust from vertical limited by the (reference) height --
  nearly upright in the last 100-200 m;
* angle between the thrust and the air-relative retrograde direction limited
  by the dynamic pressure (body lift at high q);
* sink-rate envelope near the ground (vt + k h), no climbing;
* soft "vertical final approach" corridor: small sideways offset and speed
  below ~250 m;
* cost: propellant, plus smoothness of the thrust (no bang-bang divert) and
  sideways thrust.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

try:
    from scipy import sparse
except ImportError:  # pragma: no cover
    sparse = None

from rocket3d import G0, atmosphere


@dataclass
class SCvxPlan:
    t0: float                 # absolute time of knot 0
    t_f: float                # duration
    r: np.ndarray             # (N+1, 3)
    v: np.ndarray
    z: np.ndarray             # (N+1,)
    a: np.ndarray             # thrust acceleration (N+1, 3)
    nu: float = 0.0           # total |virtual control| of the last iteration
    status: str = ""
    iters: int = 0

    @property
    def N(self) -> int:
        return self.r.shape[0] - 1

    def sample(self, t_abs: float):
        """(r, v, a) at absolute time (FOH on a, cubic-consistent on r, v)."""
        N = self.N
        dt = self.t_f / N
        tau = float(np.clip(t_abs - self.t0, 0.0, self.t_f))
        k = min(N - 1, int(tau / dt))
        s = (tau - k * dt)
        a0, a1 = self.a[k], self.a[k + 1]
        # total acceleration is not stored; interpolate r, v with Hermite
        # from the knot states (consistent to second order)
        h = dt
        x = s / h
        h00, h10, h01, h11 = 2 * x ** 3 - 3 * x ** 2 + 1, x ** 3 - 2 * x ** 2 + x, -2 * x ** 3 + 3 * x ** 2, x ** 3 - x ** 2
        r = h00 * self.r[k] + h10 * h * self.v[k] + h01 * self.r[k + 1] + h11 * h * self.v[k + 1]
        v = self.v[k] + (self.v[k + 1] - self.v[k]) * x
        a = a0 + (a1 - a0) * x
        return r, v, a

    def points(self):
        return [(float(p[0]), float(p[1]), float(max(0.0, p[2]))) for p in self.r]


class _Builder:
    """Assemble a Clarabel problem  min 1/2 x'Px + q'x  s.t.  Ax + s = b,
    s in (zero cone, nonnegative cone, second-order cones)."""

    def __init__(self, n):
        self.n = n
        self.P = np.zeros(n)
        self.q = np.zeros(n)
        self.eq = []
        self.le = []
        self.soc = []

    def add_eq(self, entries, rhs):
        self.eq.append((entries, rhs))

    def add_le(self, entries, rhs):
        self.le.append((entries, rhs))

    def add_soc(self, rows):
        """rows: list of (entries, const) -> vector y_i = const_i + sum(entries_i) with y in the SOC."""
        self.soc.append(rows)

    def assemble(self):
        R, C, V, b = [], [], [], []
        r = 0
        for entries, rhs in self.eq:
            for c, v in entries:
                R.append(r); C.append(c); V.append(v)
            b.append(rhs)
            r += 1
        nz = r
        for entries, rhs in self.le:
            for c, v in entries:
                R.append(r); C.append(c); V.append(v)
            b.append(rhs)
            r += 1
        nn = r - nz
        dims = []
        for rows in self.soc:
            for entries, const in rows:
                for c, v in entries:
                    R.append(r); C.append(c); V.append(-v)
                b.append(const)
                r += 1
            dims.append(len(rows))
        A = sparse.csc_matrix((V, (R, C)), shape=(r, self.n))
        P = sparse.diags(self.P).tocsc()
        return P, self.q.copy(), A, np.asarray(b, float), nz, nn, dims


class SCvxLander:
    """Landing-burn planner.  ``backend`` is an object with the
    ``_solve_conic(P, q, A, b, nz, nn, socs)`` method (guidance3d.Planner3D:
    native Clarabel DLL on Windows, Python clarabel otherwise)."""

    N = 20
    rk_substeps = 2              # RK4 steps per interval for the defect correction
    touchdown_speed = 1.2
    gate_h = 15.0                # the plan ends at this gate above the pad ...
    # ... sinking at vt + final_sink_gain * gate_h, upright; a short fixed
    # final-descent law (same sink profile) flies the last metres
    thrust_margin = 0.88         # plan with 88 % of the available thrust (tracking headroom)
    min_throttle = 0.22
    final_sink_gain = 0.4        # sink <= vt + k h (soft)
    # tilt of the thrust from vertical vs (reference) height
    tilt_h = (0.0, 2.0, 8.0, 25.0, 80.0, 200.0, 500.0, 1500.0)
    tilt_deg = (1.0, 3.0, 6.0, 10.0, 14.0, 20.0, 30.0, 40.0)
    # angle of attack (thrust vs air-relative retrograde) vs dynamic pressure
    aoa_q = (1500.0, 4000.0, 9000.0, 20000.0)
    aoa_deg = (30.0, 14.0, 8.0, 6.0)
    # soft vertical-final-approach corridor below ~250 m
    corr_h = (0.0, 30.0, 100.0, 250.0)
    corr_r = (0.5, 3.0, 10.0, 40.0)       # sideways offset (m)
    corr_v = (0.4, 1.5, 3.0, 8.0)         # sideways speed (m/s)
    # weights
    w_fuel = 1000.0             # on -z_N  (1 % of mass ~ 10; hovering 1 s ~ 3.5)
    w_smooth = 0.5               # on |a_xy,k+1 - a_xy,k|^2 (thrust direction)
    w_smooth_z = 0.005           # on the vertical component (tiny: throttle profile free)
    w_lat = 0.002                # on |a_xy|^2
    w_aoa = 0.0                  # (optional) cost on |thrust across the air-relative retrograde|^2 at 10 kPa;
                                 # off: it depends on the reference's q and rewards slowing down early
    w_nu = 1.0e4                 # virtual control (L1) -- dominates everything
    w_term_r = 60.0              # terminal position error (L1, m)
    w_term_v = 60.0              # terminal velocity error (L1, m/s)
    w_soft = 30.0                # sink-envelope / no-climb slacks (L1)
    w_corr_r = 0.1               # vertical-final corridor: sideways offset slack (per m) -- a preference,
    w_corr_v = 0.2               # ... never worth hovering for (per m/s)
    w_tr_a = 0.1                 # trust region: controls
    w_tr_r = 5.0e-5              # trust region: position (1/m^2)
    w_tr_v = 5.0e-3              # trust region: velocity
    w_tr_dt = 500.0              # trust region: time step (tight: dt enters every interval bilinearly)
    w_tf_hold = 0.0              # keep the touchdown time (set by caller when re-planning)
    w_gate_a = 2.0               # thrust at the gate close to the final-descent thrust

    def __init__(self, aero, spec, backend):
        self.aero = aero
        self.spec = spec
        self.backend = backend
        self.solves = 0
        self.last_status = "idle"
        self.last_iters = 0
        self.throttle_now = 1.0          # current engine throttle (spool-up limit on the first knots)
        self.throttle_rate = 10.0        # throttle change per second

    # ----------------------------------------------------------- model
    def _wind(self, h, w_ref):
        z = max(h, 0.06)
        return w_ref * min(2.6, max(0.25, math.log(z / 0.05) / math.log(10.0 / 0.05)))

    def aero_acc(self, r, v, z, a, w_ref, legs_h=450.0):
        """Aerodynamic acceleration with the vehicle axis along the thrust."""
        if self.aero is None:
            return np.zeros(3)
        h = max(0.0, float(r[2]))
        rho, _, _, a_snd = atmosphere(h)
        vr = v - self._wind(h, w_ref)
        an = float(np.linalg.norm(a))
        axis = a / an if an > 1e-6 else np.array([0.0, 0.0, 1.0])
        legs = float(np.clip((legs_h - h) / 120.0, 0.0, 1.0))    # deploying over ~2 s
        return self.aero.quick_force(vr, axis, rho, a_snd, legs) * math.exp(-z)

    def _deriv(self, x, a, w_ref):
        r, v, z = x[:3], x[3:6], x[6]
        acc = a + np.array([0.0, 0.0, -G0]) + self.aero_acc(r, v, z, a, w_ref)
        _, isp = self.thrust_avail(r[2])
        return np.r_[v, acc, -float(np.linalg.norm(a)) / (isp * G0)]

    def propagate(self, x0, a0, a1, dt, w_ref, n=4):
        """RK4 across one interval with the FOH thrust acceleration a0 -> a1."""
        x = np.array(x0, float)
        h = dt / n
        for i in range(n):
            s0, s1 = i / n, (i + 1) / n
            sm = 0.5 * (s0 + s1)
            u0 = a0 + (a1 - a0) * s0
            um = a0 + (a1 - a0) * sm
            u1 = a0 + (a1 - a0) * s1
            k1 = self._deriv(x, u0, w_ref)
            k2 = self._deriv(x + 0.5 * h * k1, um, w_ref)
            k3 = self._deriv(x + 0.5 * h * k2, um, w_ref)
            k4 = self._deriv(x + h * k3, u1, w_ref)
            x = x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return x

    def thrust_avail(self, h):
        s = self.spec
        _, p, _, _ = atmosphere(max(0.0, h))
        frac = min(1.0, p / 101_325.0)
        isp = s.isp_vac - (s.isp_vac - s.isp_sl) * frac
        return s.thrust_vac * isp / s.isp_vac, isp

    def simulate(self, r0, v0, m0, a_knots, t_f, w_ref, substeps=8):
        """Nonlinear propagation of a FOH thrust-acceleration profile (for
        checking a plan against the true model)."""
        N = a_knots.shape[0] - 1
        x = np.r_[r0, v0, math.log(m0)]
        out = [x.copy()]
        for k in range(N):
            x = self.propagate(x, a_knots[k], a_knots[k + 1], t_f / N, w_ref, n=substeps)
            out.append(x.copy())
        return np.array(out)

    # -------------------------------------------------- one convex step
    def _subproblem(self, r0, v0, z0, ref, w_ref, tf_hold=None):
        N = self.N
        rb, vb, zb, ab, tfb = ref["r"], ref["v"], ref["z"], ref["a"], ref["t_f"]
        dtb = tfb / N
        g = np.array([0.0, 0.0, -G0])
        vt = self.touchdown_speed
        # ---- linearisation of the aero acceleration at every knot
        F = np.zeros((N + 1, 3))
        Jv = np.zeros((N + 1, 3, 3))
        Ja = np.zeros((N + 1, 3, 3))
        Jz = np.zeros((N + 1, 3))
        Jh = np.zeros((N + 1, 3))
        for k in range(N + 1):
            f0 = self.aero_acc(rb[k], vb[k], zb[k], ab[k], w_ref)
            F[k] = f0
            for j in range(3):
                e = np.zeros(3)
                e[j] = 0.5
                Jv[k][:, j] = (self.aero_acc(rb[k], vb[k] + e, zb[k], ab[k], w_ref) - f0) / 0.5
                ea = np.zeros(3)
                ea[j] = max(0.05, 0.02 * float(np.linalg.norm(ab[k])))
                Ja[k][:, j] = (self.aero_acc(rb[k], vb[k], zb[k], ab[k] + ea, w_ref) - f0) / ea[j]
            eh = np.array([0.0, 0.0, 5.0])
            Jh[k] = (self.aero_acc(rb[k] + eh, vb[k], zb[k], ab[k], w_ref) - f0) / 5.0
            Jz[k] = -f0
        # ---- variable layout
        NX = 11                               # r3 v3 z1 a3 s1
        nK = NX * (N + 1)
        i_dt = nK
        i_nu = i_dt + 1                       # 7 per interval, + and -
        n_nu = 14 * N
        i_te = i_nu + n_nu                    # terminal slacks: r(3)+,- v(3)+,-  -> 12
        i_soft = i_te + 12                    # per knot: corr_r, corr_v, sink, climb
        i_d = i_soft + 4 * (N + 1)            # smoothness diffs, 3 per interval
        i_p = i_d + 3 * N                     # thrust component across the air-relative retrograde, 3 per knot
        n = i_p + 3 * (N + 1)

        def R_(k, j): return NX * k + j
        def V_(k, j): return NX * k + 3 + j
        def Z_(k): return NX * k + 6
        def A_(k, j): return NX * k + 7 + j
        def S_(k): return NX * k + 10

        B = _Builder(n)
        # ---- objective
        B.q[Z_(N)] += -self.w_fuel
        for k in range(N + 1):
            for j in range(3):
                B.P[A_(k, j)] += 2.0 * self.w_tr_a
                B.q[A_(k, j)] += -2.0 * self.w_tr_a * ab[k][j]
                B.P[R_(k, j)] += 2.0 * self.w_tr_r
                B.q[R_(k, j)] += -2.0 * self.w_tr_r * rb[k][j]
                B.P[V_(k, j)] += 2.0 * self.w_tr_v
                B.q[V_(k, j)] += -2.0 * self.w_tr_v * vb[k][j]
            for j in range(2):
                B.P[A_(k, j)] += 2.0 * self.w_lat
        B.P[i_dt] += 2.0 * self.w_tr_dt
        B.q[i_dt] += -2.0 * self.w_tr_dt * dtb
        if tf_hold is not None and self.w_tf_hold > 0.0:
            dth = tf_hold / N
            B.P[i_dt] += 2.0 * self.w_tf_hold
            B.q[i_dt] += -2.0 * self.w_tf_hold * dth
        for i in range(n_nu):
            B.q[i_nu + i] += self.w_nu
        for i in range(6):
            B.q[i_te + i] += self.w_term_r
            B.q[i_te + 6 + i] += self.w_term_v
        for k in range(N + 1):
            B.q[i_soft + 4 * k] += self.w_corr_r          # corridor: sideways offset (m)
            B.q[i_soft + 4 * k + 1] += self.w_corr_v      # corridor: sideways speed (m/s)
            B.q[i_soft + 4 * k + 2] += self.w_soft        # sink envelope
            B.q[i_soft + 4 * k + 3] += self.w_soft        # no climb
        for k in range(N):
            # smoothness of the thrust DIRECTION (sideways components); the
            # throttle profile of a hover-slam is free
            B.P[i_d + 3 * k] += 2.0 * self.w_smooth
            B.P[i_d + 3 * k + 1] += 2.0 * self.w_smooth
            B.P[i_d + 3 * k + 2] += 2.0 * self.w_smooth_z
        # ---- nonnegativity of slack variables
        for i in range(n_nu + 12 + 4 * (N + 1)):
            B.add_le([(i_nu + i, -1.0)], 0.0)
        # ---- initial state
        for j in range(3):
            B.add_eq([(R_(0, j), 1.0)], float(r0[j]))
            B.add_eq([(V_(0, j), 1.0)], float(v0[j]))
        B.add_eq([(Z_(0), 1.0)], float(z0))
        # ---- time step bounds
        B.add_le([(i_dt, -1.0)], -0.02)
        B.add_le([(i_dt, 1.0)], 6.0)

        # total acceleration A_k = a_k + g + F_k + Jv(v-vb) + Ja(a-ab) + Jz(z-zb) + Jh(h-hb)
        def acc_terms(k):
            """(list of (col, 3-vector coeff), constant 3-vector)"""
            terms = []
            const = g + F[k] - Jv[k] @ vb[k] - Ja[k] @ ab[k] - Jz[k] * zb[k] - Jh[k] * rb[k][2]
            for j in range(3):
                col_a = np.zeros(3)
                col_a[j] = 1.0
                terms.append((A_(k, j), col_a + Ja[k][:, j]))
                terms.append((V_(k, j), Jv[k][:, j].copy()))
            terms.append((Z_(k), Jz[k].copy()))
            terms.append((R_(k, 2), Jh[k].copy()))
            return terms, const

        acc_ref = [ab[k] + g + F[k] for k in range(N + 1)]
        isp_k = [self.thrust_avail(rb[k][2])[1] for k in range(N + 1)]
        # Defect correction: the FOH formulas below are exact only for an
        # acceleration linear in time.  Integrate the TRUE dynamics across
        # each interval along the reference and add the difference, so a
        # converged plan satisfies the nonlinear model (the Jacobians then
        # only affect the speed of convergence, not the answer).
        dfx = np.zeros((N, 7))
        for k in range(N):
            xk = np.r_[rb[k], vb[k], zb[k]]
            x1 = self.propagate(xk, ab[k], ab[k + 1], dtb, w_ref, n=self.rk_substeps)
            v_f = vb[k] + 0.5 * dtb * (acc_ref[k] + acc_ref[k + 1])
            r_f = rb[k] + dtb * vb[k] + dtb * dtb / 6.0 * (2.0 * acc_ref[k] + acc_ref[k + 1])
            al0, al1 = 1.0 / (isp_k[k] * G0), 1.0 / (isp_k[k + 1] * G0)
            z_f = zb[k] - 0.5 * dtb * (al0 * np.linalg.norm(ab[k]) + al1 * np.linalg.norm(ab[k + 1]))
            dfx[k] = np.r_[x1[:3] - r_f, x1[3:6] - v_f, x1[6] - z_f]
        for k in range(N):
            tA0, cA0 = acc_terms(k)
            tA1, cA1 = acc_terms(k + 1)
            # velocity: v1 - v0 - dtb/2 (A0 + A1) - (Ab0 + Ab1)/2 (dt - dtb) - nu = 0
            for j in range(3):
                ent = [(V_(k + 1, j), 1.0), (V_(k, j), -1.0)]
                cst = 0.0
                for col, vec in tA0:
                    if vec[j] != 0.0:
                        ent.append((col, -0.5 * dtb * vec[j]))
                for col, vec in tA1:
                    if vec[j] != 0.0:
                        ent.append((col, -0.5 * dtb * vec[j]))
                cst += 0.5 * dtb * (cA0[j] + cA1[j])
                slope = 0.5 * (acc_ref[k][j] + acc_ref[k + 1][j])
                ent.append((i_dt, -slope))
                cst += -slope * dtb
                iv = i_nu + 14 * k + 3 + j
                ent.append((iv, -1.0))
                ent.append((iv + 7, 1.0))
                B.add_eq(ent, cst + dfx[k][3 + j])
            # position: r1 - r0 - dt v0 - dt^2/6 (2 A0 + A1) - nu = 0
            for j in range(3):
                ent = [(R_(k + 1, j), 1.0), (R_(k, j), -1.0), (V_(k, j), -dtb)]
                cst = 0.0
                c2 = dtb * dtb / 6.0
                for col, vec in tA0:
                    if vec[j] != 0.0:
                        ent.append((col, -2.0 * c2 * vec[j]))
                for col, vec in tA1:
                    if vec[j] != 0.0:
                        ent.append((col, -c2 * vec[j]))
                cst += c2 * (2.0 * cA0[j] + cA1[j])
                slope = vb[k][j] + dtb / 3.0 * (2.0 * acc_ref[k][j] + acc_ref[k + 1][j])
                ent.append((i_dt, -slope))
                cst += -slope * dtb                      # dt*v0 ~ dtb*v0 (in ent) + vb*(dt - dtb) (in slope)
                iv = i_nu + 14 * k + j
                ent.append((iv, -1.0))
                ent.append((iv + 7, 1.0))
                B.add_eq(ent, cst + dfx[k][j])
            # mass: z1 - z0 + dtb/2 (al0 s0 + al1 s1) + (al0 sb0 + al1 sb1)/2 (dt - dtb) - nu = 0
            al0, al1 = 1.0 / (isp_k[k] * G0), 1.0 / (isp_k[k + 1] * G0)
            sb0, sb1 = float(np.linalg.norm(ab[k])), float(np.linalg.norm(ab[k + 1]))
            slope = 0.5 * (al0 * sb0 + al1 * sb1)
            ent = [(Z_(k + 1), 1.0), (Z_(k), -1.0), (S_(k), 0.5 * dtb * al0), (S_(k + 1), 0.5 * dtb * al1),
                   (i_dt, slope)]
            iv = i_nu + 14 * k + 6
            ent.append((iv, -1.0))
            ent.append((iv + 7, 1.0))
            B.add_eq(ent, slope * dtb + dfx[k][6])
            # mass can only go down (closes a loophole of the linearised time term)
            B.add_le([(Z_(k + 1), 1.0), (Z_(k), -1.0)], 0.0)
            # smoothness helper d_k = a_{k+1} - a_k
            for j in range(3):
                B.add_eq([(i_d + 3 * k + j, 1.0), (A_(k + 1, j), -1.0), (A_(k, j), 1.0)], 0.0)
        # ---- terminal (soft, L1): the gate above the pad
        s_gate = vt + self.final_sink_gain * self.gate_h
        tgt_r = np.array([0.0, 0.0, self.gate_h])
        tgt_v = np.array([0.0, 0.0, -s_gate])
        for j in range(3):
            # r_N - e+ + e- = target
            B.add_eq([(R_(N, j), 1.0), (i_te + j, -1.0), (i_te + 3 + j, 1.0)], float(tgt_r[j]))
            B.add_eq([(V_(N, j), 1.0), (i_te + 6 + j, -1.0), (i_te + 9 + j, 1.0)], float(tgt_v[j]))
        # thrust at the gate: vertical, continuing into the final descent
        # (deceleration final_sink_gain * s_gate); strong quadratic pull
        a_gate = G0 + self.final_sink_gain * s_gate
        for j, tv in enumerate((0.0, 0.0, a_gate)):
            B.P[A_(N, j)] += 2.0 * self.w_gate_a
            B.q[A_(N, j)] += -2.0 * self.w_gate_a * tv
        # ---- per-knot constraints
        for k in range(N + 1):
            hb = max(0.0, float(rb[k][2]))
            T_av, _ = self.thrust_avail(hb)
            ez = math.exp(-zb[k])
            rho2 = self.thrust_margin * T_av
            rho1 = self.min_throttle * T_av
            # |a| <= s
            B.add_soc([([(S_(k), 1.0)], 0.0)] + [([(A_(k, j), 1.0)], 0.0) for j in range(3)])
            # engine spool-up: the first knots cannot exceed the throttle the
            # engine can reach by then (nor need to reach the minimum yet)
            t_k = k * dtb
            reach = self.throttle_now + self.throttle_rate * t_k
            if reach < self.thrust_margin:
                rho2 = min(rho2, max(0.02, reach) * T_av)
            if reach < self.min_throttle:
                rho1 = max(0.0, reach) * T_av * 0.9
            # s <= rho2 e^-zb (1 - (z - zb));  s >= rho1 e^-zb (1 - (z - zb))
            B.add_le([(S_(k), 1.0), (Z_(k), rho2 * ez)], rho2 * ez * (1.0 + zb[k]))
            B.add_le([(S_(k), -1.0), (Z_(k), -rho1 * ez)], -rho1 * ez * (1.0 + zb[k]))
            # tilt from vertical: |a_xy| <= tan(th) a_z
            th = math.radians(float(np.interp(hb, self.tilt_h, self.tilt_deg)))
            tn = math.tan(th)
            B.add_soc([([(A_(k, 2), tn)], 0.0), ([(A_(k, 0), 1.0)], 0.0), ([(A_(k, 1), 1.0)], 0.0)])
            # angle of attack vs air-relative retrograde (high dynamic pressure)
            rho_air, _, _, _ = atmosphere(hb)
            vr = vb[k] - self._wind(hb, w_ref)
            spd = float(np.linalg.norm(vr))
            qd = 0.5 * rho_air * spd * spd
            if spd > 20.0 and self.w_aoa > 0.0:
                # angle-of-attack cost (grows with the dynamic pressure): at
                # high q the burn flies (air-relative) retrograde -- a gravity
                # turn -- instead of using body lift as a brake
                e = -vr / spd
                P = np.eye(3) - np.outer(e, e)
                wq = self.w_aoa * float(np.clip(qd / 10_000.0, 0.0, 3.0)) * float(np.clip((spd - 20.0) / 40.0, 0.0, 1.0))
                for i in range(3):
                    B.add_eq([(i_p + 3 * k + i, 1.0)] + [(A_(k, j), -P[i, j]) for j in range(3) if abs(P[i, j]) > 1e-12], 0.0)
                    B.P[i_p + 3 * k + i] += 2.0 * wq
            if spd > 30.0 and qd > 800.0:
                e = -vr / spd
                cap = math.tan(math.radians(float(np.interp(qd, self.aoa_q, self.aoa_deg))))
                # component along e and the perpendicular part
                P = np.eye(3) - np.outer(e, e)
                rows = [([(A_(k, j), cap * e[j]) for j in range(3)], 0.0)]
                for i in range(3):
                    rows.append(([(A_(k, j), P[i, j]) for j in range(3) if abs(P[i, j]) > 1e-12], 0.0))
                B.add_soc(rows)
            # altitude non-negative (except the last knot, which is the pad)
            if k < N:
                B.add_le([(R_(k, 2), -1.0)], 0.0)
            ic, iv_, isk, icl = i_soft + 4 * k, i_soft + 4 * k + 1, i_soft + 4 * k + 2, i_soft + 4 * k + 3
            if k > 0:
                # vertical-final corridor (soft): |r_xy| <= cr(h) + slack, |v_xy| <= cv(h) + slack
                if hb < self.corr_h[-1]:
                    cr = float(np.interp(hb, self.corr_h, self.corr_r))
                    cv = float(np.interp(hb, self.corr_h, self.corr_v))
                    B.add_soc([([(ic, 1.0)], cr), ([(R_(k, 0), 1.0)], 0.0), ([(R_(k, 1), 1.0)], 0.0)])
                    B.add_soc([([(iv_, 1.0)], cv), ([(V_(k, 0), 1.0)], 0.0), ([(V_(k, 1), 1.0)], 0.0)])
                # sink envelope (soft), only near the gate: -v_z <= vt + k_f h
                # (higher up a hover-slam follows sink ~ sqrt(h), which a
                # linear envelope would turn into early braking and hovering)
                if hb < self.gate_h + 25.0:
                    B.add_le([(V_(k, 2), -1.0), (R_(k, 2), -self.final_sink_gain), (isk, -1.0)], vt)
                # no climbing (soft): v_z <= -0.3 + slack
                if k < N:
                    B.add_le([(V_(k, 2), 1.0), (icl, -1.0)], -0.3)
        P, q, A, b, nz, nn, dims = B.assemble()
        x, status = self.backend._solve_conic(P, q, A, b, nz, nn, dims)
        self.solves += 1
        if x is None:
            return None, status
        sol = {
            "r": np.array([[x[R_(k, j)] for j in range(3)] for k in range(N + 1)]),
            "v": np.array([[x[V_(k, j)] for j in range(3)] for k in range(N + 1)]),
            "z": np.array([x[Z_(k)] for k in range(N + 1)]),
            "a": np.array([[x[A_(k, j)] for j in range(3)] for k in range(N + 1)]),
            "t_f": float(x[i_dt]) * N,
            "nu": float(np.sum(np.abs(x[i_nu:i_nu + n_nu]))),
            "term": float(np.sum(x[i_te:i_te + 12])),
            "soft": float(np.sum(x[i_soft:i_soft + 4 * (N + 1)])),
            "soft_by": np.array([float(np.sum(x[i_soft + j:i_soft + 4 * (N + 1):4])) for j in range(4)]),
        }
        return sol, status

    # --------------------------------------------------------- iterate
    def solve(self, r0, v0, m0, ref, w_ref, iters=6, tol_a=0.05, tf_hold=None, t_abs=0.0):
        """Run up to ``iters`` SCvx iterations from reference ``ref``
        (dict r, v, z, a, t_f).  Returns an SCvxPlan or None."""
        z0 = math.log(m0)
        cur = {k: (np.array(v, float) if k != "t_f" else float(v)) for k, v in ref.items()}
        # the first knot of the reference must be the current state
        cur["r"][0], cur["v"][0], cur["z"][0] = r0, v0, z0
        status = "no-solve"
        last = None
        for it in range(iters):
            sol, status = self._subproblem(r0, v0, z0, cur, w_ref, tf_hold)
            if sol is None:
                break
            da = float(np.max(np.abs(sol["a"] - cur["a"])))
            dtf = abs(sol["t_f"] - cur["t_f"])
            cur = {"r": sol["r"], "v": sol["v"], "z": sol["z"], "a": sol["a"], "t_f": sol["t_f"]}
            last = sol
            if da < tol_a and dtf < 0.05 and sol["nu"] < 0.5:
                it += 1
                break
        self.last_status = status
        if last is None:
            return None
        self.last_iters = it + 1 if last is not None else 0
        return SCvxPlan(t0=t_abs, t_f=last["t_f"], r=last["r"], v=last["v"], z=last["z"], a=last["a"],
                        nu=last["nu"], status=status, iters=self.last_iters)
