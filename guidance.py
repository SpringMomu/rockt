"""Convex powered-descent guidance for a Falcon-9-style precision landing.

Why the old controller looked nothing like a SpaceX landing
-----------------------------------------------------------
The previous version solved a 600 s / 5 s-step MPC but then executed only
8-20 % of its answer; the rest of the command came from a hand-tuned
descent-rate table (25 -> 18 -> 5 m/s).  The optimizer could not express the
ignition time, the vehicle hovered its way down, and every table boundary was
a new transient.

What this module does instead
-----------------------------
It follows the structure of flight-proven powered-descent guidance
(Acikmese & Ploen lossless convexification, "G-FOLD"):

* **Convex planner.**  A second-order-cone program over N = 40 knots with a
  *first-order-hold* thrust profile (thrust is piecewise linear, never
  stepped).  Constraints: point-mass dynamics with gravity and a predicted
  aerodynamic-drag term, the relaxed thrust cone ``||u|| <= Gamma``,
  ``rho_min <= Gamma <= rho_max``, a thrust-tilt limit that tightens to a few
  degrees in the last seconds (vertical final approach), a glide-slope cone
  around the pad, and throttle-rate limits that match the engine.  The
  terminal state (pad centre, zero horizontal speed, ~1 m/s sink) is enforced
  with an exact L1 penalty so the problem is always feasible.
* **Free final time -> optimal ignition.**  While coasting, the planner
  searches the time of flight that minimises fuel.  The fuel-optimal answer
  is "coast, then one hard burn" (hover-slam): the engine lights exactly when
  the burn has to begin.  Planning uses 80 % of the maximum thrust so the
  closed loop always keeps a control margin.
* **Locked landing burn.**  After ignition the touchdown time is fixed and
  the plan is re-solved at 5 Hz with a smooth objective (acceleration and
  jerk energy) and a minimum throttle, so the engine never shuts down and the
  thrust profile stays continuous between re-plans.  The burn is planned at
  85 % thrust; the remaining authority is reserved for feedback.  If the
  locked time becomes unreachable it is moved to the *earliest* reachable
  time, never simply later.
* **One burn to touchdown.**  There is no separate terminal-descent phase:
  the planned landing burn runs all the way to contact.  Re-planning stops
  only in the last ~1.2 s, and the tilt limit tightens with height so the
  vehicle is upright when it touches down.
* **120 Hz tracking.**  Between re-plans the controller follows the planned
  thrust (feed-forward) plus PD feedback on the planned position/velocity.  A
  cascaded attitude loop (angle -> capped body rate -> angular acceleration
  -> gimbal angle, with turn-rate feed-forward) points the engine; cold-gas
  RCS holds attitude while the engine is off.

The optimisation is solved by Clarabel: through the native Rust DLL when it
exposes ABI 2, otherwise through the Python ``clarabel`` package with the
same matrices.
"""

from __future__ import annotations

import ctypes
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

try:
    import clarabel
    from scipy import sparse
except Exception:  # pragma: no cover - fallback is used on minimal installs.
    clarabel = None
    sparse = None


NATIVE_ABI_VERSION = 2
NATIVE_BACKEND_PREFIX = "clarabel-rust-conic-"


# ---------------------------------------------------------------------------
# Planner data structures
# ---------------------------------------------------------------------------


@dataclass
class PlanConfig:
    """Per-solve settings for :class:`ConvexLandingPlanner`."""

    mode: str  # "fuel" (free-final-time search) or "smooth" (locked burn)
    rho_min: float
    rho_max: float
    gamma_now: float
    u_now: tuple[float, float]
    engine_on: bool
    max_tilt_deg: float = 60.0
    touchdown_speed: float = 1.0
    drag_model: Callable[[np.ndarray], np.ndarray] | None = None


@dataclass
class LandingPlan:
    """A solved trajectory.  Arrays use the local pad frame.

    ``states`` rows are ``(x, h, vx, vz)`` at the N+1 knots, ``controls`` are
    thrust accelerations ``(ax, az)`` at the knots (linearly interpolated
    between them) and ``gammas`` the thrust-magnitude slack of the lossless
    relaxation.
    """

    t_f: float
    dt: float
    states: np.ndarray
    controls: np.ndarray
    gammas: np.ndarray
    drag: np.ndarray
    status: str
    objective: float
    terminal_position_error: float
    terminal_velocity_error: float
    glide_slope_violation: float = 0.0
    created_at: float = 0.0
    config: PlanConfig | None = field(default=None, repr=False)

    @property
    def knots(self) -> int:
        return len(self.gammas) - 1

    def sample(self, tau: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Return (position, velocity, thrust-accel, drag) at ``tau`` s after creation."""
        n = self.knots
        g = np.array([0.0, -ConvexLandingPlanner.gravity])
        if tau >= self.t_f:
            last = self.states[-1]
            return last[:2].copy(), last[2:].copy(), self.controls[-1].copy(), self.drag[-1].copy()
        tau = max(0.0, tau)
        k = min(int(tau / self.dt), n - 1)
        t = tau - k * self.dt
        u0 = self.controls[k]
        u1 = self.controls[k + 1]
        du = (u1 - u0) / self.dt
        d = self.drag[k]
        p = self.states[k, :2]
        v = self.states[k, 2:]
        accel = u0 + du * t
        vel = v + u0 * t + 0.5 * du * t * t + (d + g) * t
        pos = p + v * t + 0.5 * u0 * t * t + du * t**3 / 6.0 + 0.5 * (d + g) * t * t
        return pos, vel, accel, d.copy()

    @property
    def reaches_pad(self) -> bool:
        """True when the plan lands on the pad without bending any soft limit."""
        return (
            self.terminal_position_error < 0.3
            and self.terminal_velocity_error < 0.2
            and self.glide_slope_violation < 1.0
        )

    def gamma_at(self, tau: float) -> float:
        if tau >= self.t_f:
            return float(self.gammas[-1])
        tau = max(0.0, tau)
        k = min(int(tau / self.dt), self.knots - 1)
        f = (tau - k * self.dt) / self.dt
        return float((1.0 - f) * self.gammas[k] + f * self.gammas[k + 1])


class _NativeDiagnostics(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_int32),
        ("iterations", ctypes.c_uint32),
        ("objective", ctypes.c_double),
        ("solve_time_seconds", ctypes.c_double),
    ]


class _NativeClarabelBackend:
    """ctypes adapter for the Rust Clarabel DLL (generic conic ABI 2)."""

    _double_p = ctypes.POINTER(ctypes.c_double)
    _size_p = ctypes.POINTER(ctypes.c_size_t)
    _status_names = {1: "Solved", 4: "AlmostSolved"}

    def __init__(self, path: Path) -> None:
        library = ctypes.CDLL(str(path))
        library.clarabel_hover_abi_version.argtypes = []
        library.clarabel_hover_abi_version.restype = ctypes.c_uint32
        version = library.clarabel_hover_abi_version()
        if version != NATIVE_ABI_VERSION:
            raise RuntimeError(
                f"native solver ABI {version} is outdated (need {NATIVE_ABI_VERSION}); "
                "run native\\build_solver.ps1"
            )
        library.clarabel_hover_backend_name.argtypes = []
        library.clarabel_hover_backend_name.restype = ctypes.c_char_p
        d, s = self._double_p, self._size_p
        library.clarabel_conic_solve.argtypes = [
            ctypes.c_size_t, ctypes.c_size_t,       # n, m
            s, s, d, ctypes.c_size_t,               # P colptr/rowval/nzval/nnz
            d,                                      # q
            s, s, d, ctypes.c_size_t,               # A colptr/rowval/nzval/nnz
            d,                                      # b
            ctypes.c_size_t, ctypes.c_size_t,       # zero rows, nonneg rows
            s, ctypes.c_size_t,                     # soc dims, soc count
            ctypes.c_uint32,                        # max iterations
            d, ctypes.c_size_t,                     # x_out, len
            ctypes.POINTER(_NativeDiagnostics),
        ]
        library.clarabel_conic_solve.restype = ctypes.c_int32
        name = library.clarabel_hover_backend_name()
        if not name:
            raise RuntimeError("native backend has no identity")
        self.library = library
        self.name = name.decode("ascii", errors="replace")

    def solve(self, P, q, A, b, n_zero, n_nonneg, soc_dims, max_iter):
        n = q.size
        m = b.size

        def csc_parts(M):
            M = M.tocsc()
            M.sort_indices()
            return (
                np.ascontiguousarray(M.indptr, dtype=np.uintp),
                np.ascontiguousarray(M.indices, dtype=np.uintp),
                np.ascontiguousarray(M.data, dtype=np.float64),
            )

        pc, pr, pv = csc_parts(P)
        ac, ar, av = csc_parts(A)
        q = np.ascontiguousarray(q, dtype=np.float64)
        b = np.ascontiguousarray(b, dtype=np.float64)
        socs = np.ascontiguousarray(soc_dims, dtype=np.uintp)
        x = np.empty(n, dtype=np.float64)
        diag = _NativeDiagnostics()
        dp = lambda a: a.ctypes.data_as(self._double_p)  # noqa: E731
        sp = lambda a: a.ctypes.data_as(self._size_p)  # noqa: E731
        code = self.library.clarabel_conic_solve(
            n, m,
            sp(pc), sp(pr), dp(pv), pv.size,
            dp(q),
            sp(ac), sp(ar), dp(av), av.size,
            dp(b),
            n_zero, n_nonneg,
            sp(socs), socs.size,
            max_iter,
            dp(x), x.size,
            ctypes.byref(diag),
        )
        status = self._status_names.get(code)
        if status is None:
            return None, f"native:{code}"
        if not np.all(np.isfinite(x)):
            return None, "native:nonfinite"
        return x, status


# ---------------------------------------------------------------------------
# Convex planner
# ---------------------------------------------------------------------------


class ConvexLandingPlanner:
    """Lossless-convexification powered-descent SOCP (Clarabel)."""

    knots = 40
    gravity = 9.80665
    # Objective weights.
    terminal_position_weight = 400.0
    terminal_velocity_weight = 250.0
    glide_slope_weight = 30.0

    # Rate at which the planned lateral thrust may change (m/s^3).  It keeps
    # the requested thrust direction within what the TVC can rotate the body
    # through, so the plan is flyable rather than merely optimal.
    lateral_rate = 4.0

    def __init__(self, max_acceleration: float = 18.0, throttle_rate: float = 1.2) -> None:
        self.max_acceleration = float(max_acceleration)
        # Thrust-magnitude rate the plan may use (80 % of the engine limit).
        self.gamma_rate = 0.8 * throttle_rate * self.max_acceleration
        self.solve_count = 0
        self.native_error = ""
        self.last_status = "idle"
        self._native: _NativeClarabelBackend | None = None
        native_enabled = os.environ.get("ROCKET_NATIVE_SOLVER", "1").lower() not in {"0", "false", "no"}
        if native_enabled and sparse is not None:
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

    @property
    def available(self) -> bool:
        return self.backend != "feedback-fallback"

    # Glide slope: the vehicle must stay inside a cone of half-angle 65 deg
    # (measured from vertical) above the pad.  Soft, so it never makes the
    # problem infeasible; it keeps the approach from skimming the ground.
    glide_slope_tan = math.tan(math.radians(65.0))

    @staticmethod
    def _tilt_limit(time_to_go: np.ndarray, far_deg: float) -> np.ndarray:
        """Maximum thrust tilt (rad) versus time-to-go.

        Large diverts are allowed early; the last ~6 s are flown almost
        vertically (20 deg at 6 s, 3 deg at 2 s), like a real landing burn.
        """
        far = max(20.0, far_deg)
        deg = np.interp(time_to_go, [0.0, 2.0, 6.0, 15.0], [3.0, 3.0, 20.0, far])
        return np.radians(deg)

    def build(self, state, t_f: float, cfg: PlanConfig):
        """Assemble the Clarabel data ``(P, q, A, b, cone sizes, layout)``."""
        # More knots on long flights keeps each step near 1.2 s (drag and the
        # ignition time are resolved finely), capped for solve time.
        N = int(np.clip(round(t_f / 1.2), self.knots, 70))
        dt = t_f / N
        g = self.gravity
        nS = 4 * (N + 1)
        iu = nS
        ig = iu + 2 * (N + 1)
        iep = ig + (N + 1)
        iem = iep + 4
        isg = iem + 4
        n = isg + N

        def S(k, c):
            return 4 * k + c

        def U(k, c):
            return iu + 2 * k + c

        def G(k):
            return ig + k

        mid_times = (np.arange(N) + 0.5) * dt
        if cfg.drag_model is not None:
            drag = np.asarray(cfg.drag_model(mid_times), dtype=float).reshape(N, 2)
        else:
            drag = np.zeros((N, 2))

        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        b: list[float] = []

        def row(entries, rhs):
            r = len(b)
            for c, v in entries:
                rows.append(r)
                cols.append(c)
                vals.append(v)
            b.append(float(rhs))

        # ---- zero cone: initial state, FOH dynamics, terminal target -----
        for c in range(4):
            row([(S(0, c), 1.0)], state[c])
        h2a = dt * dt / 3.0
        h2b = dt * dt / 6.0
        for k in range(N):
            dx, dz = drag[k]
            row([(S(k + 1, 0), 1.0), (S(k, 0), -1.0), (S(k, 2), -dt), (U(k, 0), -h2a), (U(k + 1, 0), -h2b)],
                0.5 * dt * dt * dx)
            row([(S(k + 1, 1), 1.0), (S(k, 1), -1.0), (S(k, 3), -dt), (U(k, 1), -h2a), (U(k + 1, 1), -h2b)],
                0.5 * dt * dt * (dz - g))
            row([(S(k + 1, 2), 1.0), (S(k, 2), -1.0), (U(k, 0), -0.5 * dt), (U(k + 1, 0), -0.5 * dt)],
                dt * dx)
            row([(S(k + 1, 3), 1.0), (S(k, 3), -1.0), (U(k, 1), -0.5 * dt), (U(k + 1, 1), -0.5 * dt)],
                dt * (dz - g))
        target = (0.0, 0.0, 0.0, -abs(cfg.touchdown_speed))
        for c in range(4):
            row([(S(N, c), 1.0), (iep + c, -1.0), (iem + c, 1.0)], target[c])
        n_zero = len(b)

        # ---- nonnegative cone (rows mean  A x <= b) ----------------------
        for c in range(4):
            row([(iep + c, -1.0)], 0.0)
            row([(iem + c, -1.0)], 0.0)
        for k in range(N):
            row([(isg + k, -1.0)], 0.0)
        rho_min = max(0.0, min(cfg.rho_min, cfg.rho_max))
        for k in range(N + 1):
            row([(G(k), 1.0)], cfg.rho_max)
            # Knot 0 is "now" and is pinned to the running engine instead.
            if rho_min > 0.0 and k > 0:
                row([(G(k), -1.0)], -rho_min)
        time_to_go = t_f - np.arange(N + 1) * dt
        cos_tilt = np.cos(self._tilt_limit(time_to_go, cfg.max_tilt_deg))
        # Knot 0 is the thrust being produced right now; it only has to point
        # upward-ish (the engine cannot pull), later knots follow the schedule.
        cos_tilt[0] = min(cos_tilt[0], math.cos(math.radians(80.0)))
        for k in range(N + 1):
            # Thrust tilt:  ||u|| cos(theta) <= u_z   (with Gamma = ||u||).
            row([(G(k), float(cos_tilt[k])), (U(k, 1), -1.0)], 0.0)
        tg = self.glide_slope_tan
        for k in range(1, N + 1):
            row([(S(k, 0), 1.0), (S(k, 1), -tg), (isg + k - 1, -1.0)], 0.0)
            row([(S(k, 0), -1.0), (S(k, 1), -tg), (isg + k - 1, -1.0)], 0.0)
        rate = self.gamma_rate * dt
        lateral = self.lateral_rate * dt
        for k in range(N):
            row([(G(k + 1), 1.0), (G(k), -1.0)], rate)
            row([(G(k), 1.0), (G(k + 1), -1.0)], rate)
            row([(U(k + 1, 0), 1.0), (U(k, 0), -1.0)], lateral)
            row([(U(k, 0), 1.0), (U(k + 1, 0), -1.0)], lateral)
        # The first knot is "now": it must agree with the running engine.
        spool = self.gamma_rate * 0.15
        row([(G(0), 1.0)], cfg.gamma_now + spool)
        row([(G(0), -1.0)], -(cfg.gamma_now - spool))
        if cfg.engine_on:
            slew = max(1.0, 0.3 * self.lateral_rate)
            for c in range(2):
                row([(U(0, c), 1.0)], cfg.u_now[c] + slew + (spool if c else 0.0))
                row([(U(0, c), -1.0)], -(cfg.u_now[c] - slew - (spool if c else 0.0)))
        n_nonneg = len(b) - n_zero

        # ---- second-order cones: ||(ux, uz)|| <= Gamma ------------------
        for k in range(N + 1):
            row([(G(k), -1.0)], 0.0)
            row([(U(k, 0), -1.0)], 0.0)
            row([(U(k, 1), -1.0)], 0.0)
        soc_dims = [3] * (N + 1)

        A = sparse.csc_matrix((vals, (rows, cols)), shape=(len(b), n))
        b_vec = np.asarray(b, dtype=float)

        # ---- objective ---------------------------------------------------
        w = np.full(N + 1, dt)
        w[0] *= 0.5
        w[-1] *= 0.5  # trapezoidal quadrature weights
        p_rows: list[int] = []
        p_cols: list[int] = []
        p_vals: list[float] = []
        q = np.zeros(n)
        constant = 0.0

        def add_p(i, j, v):
            if i > j:
                i, j = j, i
            p_rows.append(i)
            p_cols.append(j)
            p_vals.append(v)

        if cfg.mode == "fuel":
            fuel_w, accel_w, jerk_w = 1.0, 0.0, 0.004
        else:
            # Smooth landing burn: even thrust (acceleration energy) and low
            # jerk dominate; a small fuel term would push braking late.
            fuel_w, accel_w, jerk_w = 0.02, 0.05, 0.25
        for k in range(N + 1):
            q[G(k)] += fuel_w * w[k]
            if accel_w > 0.0:
                # accel_w * w_k * ||u_k - (0, g)||^2
                add_p(U(k, 0), U(k, 0), 2.0 * accel_w * w[k])
                add_p(U(k, 1), U(k, 1), 2.0 * accel_w * w[k])
                q[U(k, 1)] += -2.0 * accel_w * w[k] * g
                constant += accel_w * w[k] * g * g
        jw = jerk_w / dt
        for k in range(N):
            for c in range(2):
                add_p(U(k, c), U(k, c), 2.0 * jw)
                add_p(U(k + 1, c), U(k + 1, c), 2.0 * jw)
                add_p(U(k, c), U(k + 1, c), -2.0 * jw)
        if cfg.engine_on:
            # Continuity with the thrust vector the engine is producing now.
            w0 = 0.5
            for c in range(2):
                add_p(U(0, c), U(0, c), 2.0 * w0)
                q[U(0, c)] += -2.0 * w0 * cfg.u_now[c]
                constant += w0 * cfg.u_now[c] ** 2
        for c in range(4):
            weight = self.terminal_position_weight if c < 2 else self.terminal_velocity_weight
            q[iep + c] = weight
            q[iem + c] = weight
        q[isg : isg + N] = self.glide_slope_weight
        # Tiny regularisation keeps the KKT system well conditioned.
        for i in range(n):
            add_p(i, i, 1e-7)
        P = sparse.csc_matrix((p_vals, (p_rows, p_cols)), shape=(n, n))
        P.sum_duplicates()
        layout = dict(N=N, dt=dt, nS=nS, iu=iu, ig=ig, iep=iep, iem=iem, isg=isg, n=n)
        return P, q, A, b_vec, n_zero, n_nonneg, soc_dims, layout, drag, constant

    def solve(self, state, t_f: float, cfg: PlanConfig) -> LandingPlan | None:
        if not self.available or not (t_f > 0.2) or not np.all(np.isfinite(state)):
            return None
        P, q, A, b, n_zero, n_nonneg, soc_dims, L, drag, constant = self.build(state, t_f, cfg)
        x, status = self._solve_conic(P, q, A, b, n_zero, n_nonneg, soc_dims)
        self.last_status = status
        if x is None:
            return None
        self.solve_count += 1
        N = L["N"]
        states = x[: L["nS"]].reshape(N + 1, 4)
        controls = x[L["iu"] : L["ig"]].reshape(N + 1, 2)
        gammas = x[L["ig"] : L["iep"]]
        e = x[L["iep"] : L["iep"] + 4] + x[L["iem"] : L["iem"] + 4]
        # Objective of the upper-triangular P representation.
        full = P + P.T - sparse.diags(P.diagonal())
        objective = float(0.5 * x @ (full @ x) + q @ x + constant)
        return LandingPlan(
            t_f=float(t_f),
            dt=L["dt"],
            states=states.copy(),
            controls=controls.copy(),
            gammas=gammas.copy(),
            drag=drag,
            status=status,
            objective=objective,
            terminal_position_error=float(math.hypot(e[0], e[1])),
            terminal_velocity_error=float(math.hypot(e[2], e[3])),
            glide_slope_violation=float(np.max(x[L["isg"] : L["isg"] + N], initial=0.0)),
            config=cfg,
        )

    def _solve_conic(self, P, q, A, b, n_zero, n_nonneg, soc_dims):
        if self._native is not None:
            try:
                x, status = self._native.solve(P, q, A, b, n_zero, n_nonneg, soc_dims, 100)
                return x, status
            except (OSError, RuntimeError, ValueError) as error:  # pragma: no cover
                self.native_error = str(error)
                self._native = None
                self.backend = "clarabel-python" if clarabel is not None else "feedback-fallback"
        if clarabel is None:
            return None, "no-solver"
        cones = [clarabel.ZeroConeT(n_zero), clarabel.NonnegativeConeT(n_nonneg)]
        cones.extend(clarabel.SecondOrderConeT(d) for d in soc_dims)
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.max_iter = 100
        try:
            solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
            result = solver.solve()
        except Exception as error:  # pragma: no cover - keeps the loop alive
            return None, f"python:{type(error).__name__}"
        status = str(result.status)
        if status not in {"Solved", "AlmostSolved"} or result.x is None:
            return None, status
        x = np.asarray(result.x, dtype=float)
        if not np.all(np.isfinite(x)):
            return None, "nonfinite"
        return x, status

    def search_final_time(self, state, lo: float, hi: float, cfg: PlanConfig, evaluations: int = 10) -> LandingPlan | None:
        """Golden-section search of the time of flight minimising the objective."""
        lo = max(0.5, float(lo))
        hi = max(lo + 0.5, float(hi))
        ratio = (math.sqrt(5.0) - 1.0) / 2.0
        best: LandingPlan | None = None
        cache: dict[float, float] = {}

        def f(t):
            nonlocal best
            plan = self.solve(state, t, cfg)
            value = plan.objective if plan is not None else math.inf
            cache[t] = value
            if plan is not None and (best is None or value < best.objective):
                best = plan
            return value

        a, b = lo, hi
        c = b - ratio * (b - a)
        d = a + ratio * (b - a)
        fc, fd = f(c), f(d)
        for _ in range(max(0, evaluations - 2)):
            if fc <= fd:
                b, d, fd = d, c, fc
                c = b - ratio * (b - a)
                fc = f(c)
            else:
                a, c, fc = c, d, fd
                d = a + ratio * (b - a)
                fd = f(d)
        return best


# Backwards-compatible name used by older scripts.
ClarabelMPC = ConvexLandingPlanner


# ---------------------------------------------------------------------------
# Closed-loop guidance
# ---------------------------------------------------------------------------


class AutonomousGuidance:
    """Coast -> (boost-back) -> single landing burn -> touchdown."""

    # Re-planning cadence.
    coast_replan_period = 0.5
    burn_replan_period = 0.2
    # Thrust budget: plan the ignition with margin, fly the burn with more.
    ignition_thrust_fraction = 0.80
    burn_thrust_fraction = 0.85
    minimum_throttle_fraction = 0.30
    # Largest thrust tilt from vertical the planner may use for a divert.
    divert_tilt_deg = 70.0
    touchdown_speed = 1.0
    # No separate terminal phase: the landing burn is flown to contact.
    # Below ``late_height`` a short time-to-go is normal (the plan is about to
    # end on the pad); above it, it means the vehicle is running late.
    late_height = 20.0
    default_target_altitude = 0.0
    landing_altitude = 0.0

    def __init__(self, target_altitude: float | None = None) -> None:
        del target_altitude  # the pad surface is always the target
        self.configured_target = self.landing_altitude
        self.target_altitude: float | None = self.landing_altitude
        self.mpc = ConvexLandingPlanner()
        self.reset()

    # -- public API used by main.py ------------------------------------------
    def reset(self) -> None:
        self.target_altitude = self.configured_target
        self.elapsed = 0.0
        self.plan: LandingPlan | None = None
        self.next_replan = 0.0
        self.burn_locked = False
        self.touchdown_time: float | None = None
        self._failed_solves = 0
        self._previous_desired_angle: float | None = None
        self._desired_rate = 0.0
        self.burn_fraction = self.burn_thrust_fraction
        self._last_search = -math.inf
        self.predicted_points: list[tuple[float, float]] = []
        self.phase = "STANDBY"
        self.last_status = "idle"
        self.time_to_go = float("nan")
        self.last_command = (0.0, 0.0)
        self.touchdown_velocity = (0.0, 0.0)

    def set_target_altitude(self, altitude: float) -> None:
        del altitude
        self.configured_target = self.landing_altitude
        self.target_altitude = self.landing_altitude

    @property
    def solution(self):  # compatibility with older UI code
        return self.plan

    # -- helpers ---------------------------------------------------------------
    @staticmethod
    def _wrap(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _drag_accel(rocket, h: float, vx: float, vz: float, body=None) -> np.ndarray:
        """Drag acceleration in the local frame, same model as ``Rocket.step``.

        ``body`` is the unit vector of the vehicle axis (engine -> nose).  If
        omitted the vehicle is assumed to fly engines-first (axial flow).
        """
        speed = math.hypot(vx, vz)
        if speed < 1e-6:
            return np.zeros(2)
        altitude = max(0.0, h + rocket.pad_center_height)
        if altitude >= rocket.ATMOSPHERE_TOP:
            return np.zeros(2)
        density = rocket.SEA_LEVEL_DENSITY * math.exp(-altitude / rocket.ATMOSPHERE_SCALE_HEIGHT)
        area = rocket.FRONTAL_AREA
        if body is not None:
            ux, uz = vx / speed, vz / speed
            axial = abs(ux * body[0] + uz * body[1])
            broadside = abs(ux * body[1] - uz * body[0])
            area = rocket.FRONTAL_AREA * axial + rocket.SIDE_AREA * broadside
        k = 0.5 * density * rocket.DRAG_COEFFICIENT * area / rocket.mass
        return -k * speed * np.array([vx, vz])

    def _drag_model(self, rocket, state):
        """Drag along the previous plan (a successive-convexification step).

        The vehicle axis is taken from the previous plan's thrust direction
        (the TVC keeps the body aligned with it), so a tilted divert burn is
        charged the much larger broadside drag it really produces.
        """
        previous = self.plan
        now = self.elapsed
        axis = rocket.body_up()
        body_now = (float(axis.dot(rocket.local_right())), float(axis.dot(rocket.local_up())))
        current = self._drag_accel(rocket, state[1], state[2], state[3], body_now)

        def model(times: np.ndarray) -> np.ndarray:
            out = np.empty((len(times), 2))
            for i, t in enumerate(times):
                if t < 0.6:
                    # The next fraction of a second is flown in the present attitude.
                    out[i] = self._drag_accel(rocket, state[1], state[2], state[3], body_now)
                elif previous is not None:
                    pos, vel, accel, _ = previous.sample(now + t - previous.created_at)
                    norm = float(np.linalg.norm(accel))
                    body = accel / norm if norm > 2.0 else None
                    out[i] = self._drag_accel(rocket, pos[1], vel[0], vel[1], body)
                else:
                    out[i] = current
            return out

        return model

    def _initial_time_bracket(self, state, a_max: float) -> tuple[float, float]:
        x, h, vx, vz = state
        g = self.mpc.gravity
        a_net = max(1.0, self.ignition_thrust_fraction * a_max - g)
        v_ign = math.sqrt(max(0.0, (vz * vz + 2.0 * g * max(h, 0.0)) / (1.0 + g / a_net)))
        t_vertical = max(0.0, (vz + v_ign) / g) + v_ign / a_net
        lateral = 0.5 * a_max
        t_lateral = abs(vx) / lateral + 2.0 * math.sqrt(abs(x) / lateral)
        t0 = max(2.0, t_vertical, t_lateral)
        return max(1.0, 0.35 * t0), 2.5 * t0 + 6.0

    def _publish_prediction(self, rocket) -> None:
        plan = self.plan
        if plan is None:
            self.predicted_points = []
            return
        pad_y = rocket.world_y_for_altitude(rocket.launch_x, 0.0)
        start = max(0.0, self.elapsed - plan.created_at)
        remaining = max(0.0, plan.t_f - start)
        samples = max(2, min(240, int(remaining / 0.1) + 2))
        points = []
        for t in np.linspace(start, plan.t_f, samples):
            pos, _, _, _ = plan.sample(float(t))
            points.append((rocket.launch_x + float(pos[0]), pad_y + max(0.0, float(pos[1]))))
        self.predicted_points = points

    # -- planning ----------------------------------------------------------------
    def _replan(self, rocket, state, a_max: float) -> None:
        throttle_accel = rocket.throttle * a_max
        up = rocket.local_up()
        right = rocket.local_right()
        thrust = rocket.last_thrust / rocket.mass
        u_now = (float(thrust.dot(right)), float(thrust.dot(up)))
        engine_on = rocket.throttle > 0.02
        drag_model = self._drag_model(rocket, state)
        plan: LandingPlan | None = None

        if not self.burn_locked:
            cfg = PlanConfig(
                mode="fuel",
                rho_min=0.0,
                rho_max=self.ignition_thrust_fraction * a_max,
                gamma_now=throttle_accel,
                u_now=u_now,
                engine_on=engine_on,
                max_tilt_deg=self.divert_tilt_deg,
                touchdown_speed=self.touchdown_speed,
                drag_model=drag_model,
            )
            if self.plan is None or self.touchdown_time is None:
                lo, hi = self._initial_time_bracket(state, a_max)
                plan = self.mpc.search_final_time(state, lo, hi, cfg, evaluations=16)
            else:
                guess = max(1.0, self.touchdown_time - self.elapsed)
                kept = self.mpc.solve(state, guess, cfg)
                kept_ok = kept is not None and kept.reaches_pad
                recently_searched = self.elapsed - self._last_search < self.coast_replan_period - 1e-6
                if kept_ok and (engine_on or recently_searched):
                    # A burn is in progress (boost-back), or the time was just
                    # optimised: keep the touchdown time fixed so consecutive
                    # plans agree (and the fast pre-ignition cadence costs one
                    # solve instead of a whole search).
                    plan = kept
                else:
                    self._last_search = self.elapsed
                    lo, hi = 0.85 * guess - 0.5, 1.15 * guess + 0.5
                    plan = self.mpc.search_final_time(state, lo, hi, cfg, evaluations=8)
                    if plan is not None and min(plan.t_f - lo, hi - plan.t_f) < 0.03 * (hi - lo):
                        lo2, hi2 = self._initial_time_bracket(state, a_max)
                        wider = self.mpc.search_final_time(state, lo2, hi2, cfg, evaluations=16)
                        if wider is not None and wider.objective < plan.objective:
                            plan = wider
                    # Hysteresis: only move the touchdown time for a clear gain.
                    if kept_ok and (plan is None or plan.objective > kept.objective - max(0.5, 0.005 * abs(kept.objective))):
                        plan = kept
            if plan is not None:
                reference = self.ignition_thrust_fraction * a_max
                later = plan.gammas[1:]
                if later.min() >= 0.25 * reference and plan.gammas[1] >= 0.5 * reference:
                    # One continuous burn from now until touchdown: this is
                    # the landing-burn ignition point.  Lock the touchdown time.
                    self.burn_locked = True
        else:
            t_go = self.touchdown_time - self.elapsed
            cfg = PlanConfig(
                mode="smooth",
                rho_min=self.minimum_throttle_fraction * a_max,
                rho_max=self.burn_fraction * a_max,
                gamma_now=throttle_accel,
                u_now=u_now,
                engine_on=True,
                max_tilt_deg=self.divert_tilt_deg,
                touchdown_speed=self.touchdown_speed,
                drag_model=drag_model,
            )
            plan = self.mpc.solve(state, t_go, cfg)
            if plan is None:
                # A failed solve is not evidence that the target moved: keep
                # flying the previous plan and try again next cycle.
                self._failed_solves += 1
                if self._failed_solves < 3:
                    self.last_status = self.mpc.last_status
                    return
            else:
                self._failed_solves = 0
            if plan is None or not plan.reaches_pad:
                # The locked touchdown time is no longer reachable.  Move it
                # to the *earliest* time that is reachable (never simply
                # "later": a smooth plan with spare time would descend lazily
                # and hover above the pad), raising the thrust budget if needed.
                found = self._earliest_reachable_time(state, cfg, t_go, a_max)
                if found is not None:
                    t_new, fraction = found
                    self.burn_fraction = max(self.burn_fraction, min(0.97, fraction + 0.05))
                    cfg.rho_max = self.burn_fraction * a_max
                    smooth = self.mpc.solve(state, t_new, cfg)
                    if smooth is not None:
                        plan = smooth

        self.last_status = self.mpc.last_status
        if plan is not None:
            plan.created_at = self.elapsed
            self.plan = plan
            self.touchdown_time = self.elapsed + plan.t_f
            self.last_status = plan.status

    def _earliest_reachable_time(self, state, cfg: PlanConfig, t_go: float, a_max: float) -> tuple[float, float] | None:
        """Shortest time of flight that reaches the pad, and the thrust used.

        The thrust budget is escalated (80 % -> 90 % -> 97 %) before the time
        of flight is stretched: stretching time on a falling vehicle only
        helps by flying low and sideways, which is exactly what must not
        happen near the ground.
        """
        for fraction in (self.ignition_thrust_fraction, 0.90, 0.97):
            check = PlanConfig(**{**cfg.__dict__, "mode": "fuel", "rho_max": fraction * a_max})

            def reachable(t: float) -> bool:
                plan = self.mpc.solve(state, t, check)
                return plan is not None and plan.reaches_pad

            lo = max(0.8, 0.6 * t_go)
            hi = max(t_go + 1.0, 1.25 * t_go)
            if not reachable(hi):
                continue
            if reachable(lo):
                return lo, fraction
            for _ in range(6):
                mid = 0.5 * (lo + hi)
                if reachable(mid):
                    hi = mid
                else:
                    lo = mid
            return hi, fraction
        return None

    # -- main entry point ----------------------------------------------------
    def command(self, rocket, dt: float) -> None:
        """Run one 120 Hz guidance/control step on the nonlinear rocket."""
        if rocket.state in {"CRASHED", "LANDED"}:
            rocket.throttle = 0.0
            rocket.gimbal = move_angle(rocket.gimbal, 0.0, rocket.GIMBAL_RATE * dt)
            rocket.rcs_command = 0.0
            if rocket.state == "LANDED":
                self.phase = "LANDED"
            self.predicted_points = []
            return
        if rocket.state == "ON PAD":
            rocket.throttle = 0.0
            rocket.gimbal = move_angle(rocket.gimbal, 0.0, rocket.GIMBAL_RATE * dt)
            rocket.rcs_command = 0.0
            self.phase = "STANDBY"
            return

        dt = float(dt)
        self.elapsed += dt
        up = rocket.local_up()
        right = rocket.local_right()
        x = float(rocket.position.x - rocket.launch_x)
        h = float(rocket.altitude)
        vx = float(rocket.velocity.dot(right))
        vz = float(rocket.vertical_speed)
        state = np.array([x, h, vx, vz])
        self.touchdown_velocity = (vx, vz)
        a_max = rocket.MAX_THRUST / rocket.mass
        g = self.mpc.gravity
        drag_now = np.array([rocket.last_drag.dot(right), rocket.last_drag.dot(up)]) / rocket.mass

        t_go = (self.touchdown_time - self.elapsed) if self.touchdown_time is not None else math.inf
        if self.burn_locked and t_go < 1.5 and h > self.late_height:
            # Running late (e.g. heavy disturbance): re-open the touchdown time.
            self.touchdown_time = self.elapsed + 3.0
            self.next_replan = self.elapsed
            t_go = 3.0

        if self.mpc.available and self.elapsed >= self.next_replan:
            if not self.burn_locked or t_go > 1.2:
                self._replan(rocket, state, a_max)
                self._publish_prediction(rocket)
            period = self.burn_replan_period if self.burn_locked else self.coast_replan_period
            if not self.burn_locked and self.plan is not None:
                # Tighten the cadence as the planned ignition approaches.
                hot = np.nonzero(self.plan.gammas > 0.25 * self.ignition_thrust_fraction * a_max)[0]
                if hot.size:
                    until_ignition = hot[0] * self.plan.dt
                    period = float(np.clip(until_ignition - 0.05, 0.1, period))
            self.next_replan = self.elapsed + period
            t_go = (self.touchdown_time - self.elapsed) if self.touchdown_time is not None else math.inf
        self.time_to_go = t_go

        coasting = False
        if self.plan is not None:
            tau = self.elapsed - self.plan.created_at
            pos_ref, vel_ref, u_ff, _ = self.plan.sample(tau)
            gamma_ff = self.plan.gamma_at(tau)
            coasting = (not self.burn_locked) and gamma_ff < 0.06 * a_max
            fb = 0.45 * (pos_ref - state[:2]) + 1.5 * (vel_ref - state[2:])
            norm = float(np.linalg.norm(fb))
            if norm > 4.0:
                fb *= 4.0 / norm
            # Drag is already in the plan (re-predicted every re-plan); the PD
            # term absorbs the residual, so no separate drag cancellation.
            accel = u_ff + fb
            if self.burn_locked:
                self.phase = "LANDING BURN"
            else:
                if coasting:
                    self.phase = "COAST"
                else:
                    # A burn counts as boost-back only if the plan coasts
                    # again afterwards; otherwise it is the landing-burn
                    # ignition ramp just before the burn is locked.
                    k0 = min(int(max(tau, 0.0) / self.plan.dt) + 1, self.plan.knots)
                    later = self.plan.gammas[k0:]
                    gap = later.size > 0 and later.min() < 0.06 * a_max
                    self.phase = "BOOSTBACK" if gap else "LANDING BURN"
            max_tilt = math.radians(75.0)
            if self.burn_locked:
                # Past the plan's end the reference is "on the pad, sinking at
                # touchdown speed", so a late vehicle keeps descending.  Never
                # stall or climb just above the pad.
                if h < 10.0 and vz > -0.6 * self.touchdown_speed:
                    accel[1] = min(accel[1], 0.9 * g)
                # Tilt budget tightens with height: upright at contact.
                max_tilt = math.radians(float(np.interp(h, [0.3, 1.5, 3.0, 12.0, 40.0], [0.6, 2.0, 4.0, 6.0, 75.0])))
        else:
            # No solver available: bounded suicide-burn feedback law.
            self.phase = "FALLBACK"
            v_ref = -math.sqrt(2.0 * 0.5 * (a_max - g) * max(h, 0.0)) - self.touchdown_speed
            az = g - drag_now[1] + 1.5 * (v_ref - vz)
            ax = -drag_now[0] + 0.08 * (-x) - 0.9 * vx
            accel = np.array([ax, max(0.0, az)])
            max_tilt = math.radians(35.0)

        # -- actuators -----------------------------------------------------
        if coasting:
            # Engine off.  Like Falcon 9, coast engines-first (lowest drag,
            # and exactly what the planner's drag model assumes); only in
            # the last few seconds before ignition does RCS turn the vehicle
            # to the direction of the upcoming burn.
            rocket.throttle = move_towards(rocket.throttle, 0.0, rocket.THROTTLE_RATE * dt)
            burn_dir = self._upcoming_burn_direction(a_max, within=4.0)
            if burn_dir is None:
                speed = math.hypot(vx, vz)
                burn_dir = np.array([-vx, -vz]) / speed if speed > 5.0 else np.array([0.0, 1.0])
                if burn_dir[1] < 0.2:
                    burn_dir = np.array([np.sign(burn_dir[0]) * 0.98, 0.2])
            desired_world = right * float(burn_dir[0]) + up * float(burn_dir[1])
            self._attitude(rocket, math.atan2(desired_world.x, desired_world.y), dt, use_rcs=True)
            self.last_command = (0.0, 0.0)
            return

        # Limit the thrust tilt from the local vertical.
        ax, az = float(accel[0]), float(accel[1])
        az = max(az, 0.0)
        limit = math.tan(max_tilt) * az if max_tilt < math.radians(89.0) else math.inf
        if az < 1e-6:
            az = 1e-6
            limit = math.tan(max_tilt) * max(abs(ax), 1.0)
        ax = float(np.clip(ax, -limit, limit))
        self.last_command = (ax, az)
        thrust_world = right * ax + up * az
        magnitude = thrust_world.length()
        desired_angle = math.atan2(thrust_world.x, thrust_world.y)
        error = self._wrap(rocket.angle - desired_angle)
        # Do not push hard in a wrong direction while the body is still turning.
        alignment = float(np.clip(math.cos(error), 0.35, 1.0))
        throttle = float(np.clip(magnitude / a_max, 0.0, 1.0)) * alignment
        if self.burn_locked:
            throttle = max(throttle, 0.9 * self.minimum_throttle_fraction)
        rocket.throttle = move_towards(rocket.throttle, throttle, rocket.THROTTLE_RATE * dt)
        self._attitude(rocket, desired_angle, dt, use_rcs=rocket.throttle < 0.25)

    def _upcoming_burn_direction(self, a_max: float, within: float = math.inf) -> np.ndarray | None:
        plan = self.plan
        if plan is None:
            return None
        tau = self.elapsed - plan.created_at
        start = min(int(max(tau, 0.0) / plan.dt), plan.knots)
        for k in range(start, plan.knots + 1):
            if k * plan.dt - tau > within:
                return None
            if plan.gammas[k] > 0.25 * self.ignition_thrust_fraction * a_max:
                u = plan.controls[k]
                norm = float(np.linalg.norm(u))
                if norm > 1e-6:
                    return u / norm
        return None

    # Attitude loop: angle error -> rate command (capped) -> angular accel.
    attitude_angle_gain = 1.0      # 1/s
    attitude_rate_gain = 2.0       # 1/s   (-> wn 1.4 rad/s, zeta 0.7)
    attitude_max_rate = math.radians(20.0)
    # The gimbal slews at only 42 deg/s; commanding small deflections keeps
    # its reversal time short, which is what prevents attitude overshoot.
    attitude_max_gimbal = math.radians(12.0)

    def _attitude(self, rocket, desired_angle: float, dt: float, *, use_rcs: bool) -> None:
        """Cascaded attitude loop driving TVC (and RCS at low thrust).

        The outer loop turns the pointing error into a body-rate command capped
        at 30 deg/s; the inner loop converts the rate error into an angular
        acceleration and, through the thrust-dependent lever arm, into a
        gimbal angle.  The rate cap is what removes the whip-like overshoot of
        a plain PD law on large re-pointing manoeuvres.
        """
        error = self._wrap(rocket.angle - desired_angle)
        omega = rocket.angular_velocity
        inertia = rocket.moment_of_inertia
        # Feed-forward of the commanded turn rate (filtered derivative of the
        # desired angle) removes the steady lag while following a turning
        # thrust vector, e.g. the final "S" correction before touchdown.
        if self._previous_desired_angle is not None:
            raw_rate = self._wrap(desired_angle - self._previous_desired_angle) / max(dt, 1e-6)
            raw_rate = float(np.clip(raw_rate, -self.attitude_max_rate, self.attitude_max_rate))
            blend = dt / (0.25 + dt)
            self._desired_rate += blend * (raw_rate - self._desired_rate)
        self._previous_desired_angle = desired_angle
        rate_cmd = self._desired_rate - float(np.clip(self.attitude_angle_gain * error, -self.attitude_max_rate, self.attitude_max_rate))
        rate_cmd = float(np.clip(rate_cmd, -self.attitude_max_rate, self.attitude_max_rate))
        alpha = self.attitude_rate_gain * (rate_cmd - omega)
        thrust = rocket.throttle * rocket.MAX_THRUST
        if thrust > 0.05 * rocket.MAX_THRUST:
            # alpha = -L * T * sin(gimbal) / I
            ratio = float(np.clip(-alpha * inertia / (rocket.THRUST_LEVER_ARM * thrust), -1.0, 1.0))
            limit = min(rocket.GIMBAL_LIMIT, self.attitude_max_gimbal)
            target = float(np.clip(math.asin(ratio), -limit, limit))
        else:
            target = 0.0
        rocket.gimbal = move_angle(rocket.gimbal, target, rocket.GIMBAL_RATE * dt)
        if use_rcs:
            authority = rocket.RCS_MAX_TORQUE / inertia
            rocket.rcs_command = float(np.clip(alpha / authority, -1.0, 1.0))
        else:
            rocket.rcs_command = 0.0


def move_angle(value: float, target: float, maximum_delta: float) -> float:
    delta = (target - value + math.pi) % (2.0 * math.pi) - math.pi
    if abs(delta) <= maximum_delta:
        return target
    return value + math.copysign(maximum_delta, delta)


def move_towards(value: float, target: float, maximum_delta: float) -> float:
    """Move a scalar toward a target by at most ``maximum_delta``."""
    if value < target:
        return min(value + maximum_delta, target)
    return max(value - maximum_delta, target)
