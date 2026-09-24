"""Glass-cockpit instrument panels for the landing simulator.

Layout (1280x720 reference, adapts to the window size):

* top bar            - mission clock, autopilot mode, solver status
* NAV (top-left)     - trajectory overview with the pad and the plan
* PFD (bottom-left)  - attitude ball with tilt scale, speed tape, altitude
                       tape, vertical-speed scale, velocity vector, command bug
* AUTOPILOT (right)  - engagement, guidance phase sequence, solver data,
                       caution / advisory lights
* ENGINE (right)     - throttle lever slider (actual + commanded), gimbal
                       gauge with limits, thrust and T/W
* LANDING ZONE       - bottom-centre precision monitor below 50 m: pad
  (bottom-centre)      schematic, lateral error with tolerance bands,
                       predicted touchdown point and pass/fail gates

Colours follow airliner conventions: green = active/normal, cyan = selected
data, magenta = autopilot targets, amber = caution, red = warning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pygame

# ---------------------------------------------------------------- palette
BG = (6, 12, 20)
FRAME = (58, 86, 106)
FRAME_HI = (98, 150, 182)
TEXT = (226, 236, 242)
DIM = (120, 142, 158)
FAINT = (70, 88, 102)
CYAN = (40, 214, 255)
GREEN = (60, 240, 130)
AMBER = (255, 184, 40)
RED = (255, 72, 60)
MAGENTA = (255, 96, 220)
WHITE = (250, 252, 255)
SKY_TOP = (22, 88, 176)
SKY_LOW = (70, 148, 226)
EARTH_HI = (132, 84, 40)
EARTH_LOW = (84, 52, 24)

PHASES = ("COAST", "BOOSTBACK", "LANDING BURN", "LANDED")


@dataclass
class Telemetry:
    mission_time: float = 0.0
    autopilot: bool = False
    phase: str = "STANDBY"
    state: str = "ON PAD"
    crash_reason: str = ""
    altitude: float = 0.0
    x: float = 0.0
    vx: float = 0.0
    vz: float = 0.0
    tilt_deg: float = 0.0
    tilt_rate_deg: float = 0.0
    gimbal_deg: float = 0.0
    gimbal_limit_deg: float = 30.0
    ap_gimbal_limit_deg: float = 12.0
    throttle: float = 0.0
    throttle_cmd: float | None = None
    cmd_tilt_deg: float | None = None
    thrust_kn: float = 0.0
    twr: float = 0.0
    rcs: float = 0.0
    t_go: float = math.nan
    solver_status: str = "idle"
    backend: str = ""
    solves: int = 0
    burn_locked: bool = False
    thrust_margin: float = 0.15
    fps: float = 0.0
    scenario: str = ""
    legs: float = 0.0
    nav_pad: tuple = (0.0, 0.0)
    nav_rocket: tuple = (0.0, 0.0)
    nav_trajectory: list = field(default_factory=list)
    nav_plan: list = field(default_factory=list)
    touchdown: dict | None = None
    paused: bool = False


class Cockpit:
    def __init__(self) -> None:
        sans = "bahnschrift,segoeui,arial"
        mono = "consolas,cascadiamono,dejavusansmono,couriernew"
        self.f_label = pygame.font.SysFont(sans, 12)
        self.f_small = pygame.font.SysFont(sans, 13)
        self.f_title = pygame.font.SysFont(sans, 13, bold=True)
        self.f_big = pygame.font.SysFont(sans, 22, bold=True)
        self.f_huge = pygame.font.SysFont(sans, 34, bold=True)
        self.f_mono_s = pygame.font.SysFont(mono, 12)
        self.f_mono = pygame.font.SysFont(mono, 14)
        self.f_mono_b = pygame.font.SysFont(mono, 17, bold=True)
        self.f_mono_xl = pygame.font.SysFont(mono, 24, bold=True)
        self._text_cache: dict = {}
        self._panel_cache: dict = {}
        self.reset()

    def reset(self) -> None:
        self.visited: list[str] = []
        self.callout: tuple[str, float, tuple] | None = None
        self.last_phase = ""
        self.landing_alpha = 0.0
        self._last_time = None

    # ------------------------------------------------------------ helpers
    def text(self, surf, s, font, colour, pos, anchor="topleft"):
        key = (id(font), s, colour)
        img = self._text_cache.get(key)
        if img is None:
            if len(self._text_cache) > 3000:
                self._text_cache.clear()
            img = font.render(s, True, colour)
            self._text_cache[key] = img
        r = img.get_rect()
        setattr(r, anchor, (int(pos[0]), int(pos[1])))
        surf.blit(img, r)
        return r

    def panel(self, surf, rect, title=None, accent=CYAN, alpha=205):
        rect = pygame.Rect(rect)
        key = (rect.size, title, accent, alpha)
        bg = self._panel_cache.get(key)
        if bg is None:
            bg = pygame.Surface(rect.size, pygame.SRCALPHA)
            h = rect.height
            for i in range(0, h, 4):  # subtle vertical gradient
                k = i / max(1, h)
                col = (int(BG[0] + 8 * (1 - k)), int(BG[1] + 12 * (1 - k)), int(BG[2] + 18 * (1 - k)), alpha)
                pygame.draw.rect(bg, col, (0, i, rect.width, 4))
            pygame.draw.rect(bg, (*FRAME, 255), bg.get_rect(), 1, border_radius=6)
            c = 12
            w, hh = rect.width - 1, rect.height - 1
            for (x, y, dx, dy) in ((0, 0, 1, 1), (w, 0, -1, 1), (0, hh, 1, -1), (w, hh, -1, -1)):
                pygame.draw.line(bg, (*accent, 255), (x, y), (x + dx * c, y), 2)
                pygame.draw.line(bg, (*accent, 255), (x, y), (x, y + dy * c), 2)
            if title:
                pygame.draw.line(bg, (*FRAME, 255), (8, 22), (rect.width - 8, 22), 1)
                img = self.f_title.render(title, True, (*accent,))
                bg.blit(img, (10, 4))
            self._panel_cache[key] = bg
        surf.blit(bg, rect.topleft)
        top = 26 if title else 6
        return pygame.Rect(rect.x + 8, rect.y + top, rect.width - 16, rect.height - top - 6)

    def box(self, surf, rect, colour, fill=None, width=1, radius=3):
        if fill is not None:
            pygame.draw.rect(surf, fill, rect, 0, border_radius=radius)
        pygame.draw.rect(surf, colour, rect, width, border_radius=radius)

    # --------------------------------------------------------------- main
    def draw(self, surf, t: Telemetry) -> None:
        w, h = surf.get_size()
        self._track_phase(t)
        self.top_bar(surf, t, w)
        margin = 14
        right_w = 300
        nav = pygame.Rect(margin, 46, 300, 168)
        pfd = pygame.Rect(margin, h - margin - 330, 330, 330)
        ap = pygame.Rect(w - margin - right_w, 46, right_w, 306)
        eng = pygame.Rect(w - margin - right_w, h - margin - 236, right_w, 236)
        if eng.top < ap.bottom + 8:
            eng.top = ap.bottom + 8
        if h >= 640:
            self.nav_panel(surf, nav, t)
        self.pfd(surf, pfd, t)
        self.autopilot_panel(surf, ap, t)
        self.engine_panel(surf, eng, t)
        # Landing-zone monitor fades in during the last 50 m.
        show = t.state == "LANDED" or (t.state == "FLYING" and t.altitude < 50.0 and t.autopilot)
        target_alpha = 1.0 if show else 0.0
        now = t.mission_time
        step = 1.0 / 60.0 if self._last_time is None else max(0.0, min(1.0, now - self._last_time))
        self._last_time = now
        if t.state != "FLYING":
            step = max(step, 1.0 / 60.0)
        self.landing_alpha += (target_alpha - self.landing_alpha) * min(1.0, step * 5.0)
        if self.landing_alpha > 0.03:
            lw = min(560, eng.left - pfd.right - 24)
            if lw > 360:
                lz = pygame.Rect(0, 0, lw, 176)
                lz.midbottom = ((pfd.right + eng.left) // 2, h - margin)
                self.landing_monitor(surf, lz, t, self.landing_alpha)
        self.callouts(surf, t, w, h)

    def _track_phase(self, t: Telemetry) -> None:
        phase = "LANDED" if t.state == "LANDED" else t.phase
        if phase != self.last_phase:
            if phase in PHASES and phase not in self.visited:
                self.visited.append(phase)
            messages = {
                "BOOSTBACK": ("BOOSTBACK BURN", CYAN),
                "LANDING BURN": ("LANDING BURN START", AMBER),
                "COAST": ("ENGINE CUTOFF - COAST", DIM),
                "LANDED": ("TOUCHDOWN", GREEN),
            }
            if phase in messages and self.last_phase:
                text, colour = messages[phase]
                self.callout = (text, t.mission_time, colour)
            self.last_phase = phase

    # ------------------------------------------------------------- top bar
    def top_bar(self, surf, t, w):
        bar = pygame.Surface((w, 34), pygame.SRCALPHA)
        bar.fill((4, 9, 16, 215))
        pygame.draw.line(bar, (*FRAME, 255), (0, 33), (w, 33), 1)
        surf.blit(bar, (0, 0))
        pygame.draw.polygon(surf, CYAN, [(18, 17), (24, 9), (30, 17), (24, 25)])
        self.text(surf, "VECTOR THRUST", self.f_title, TEXT, (38, 9))
        mode = "AUTOLAND" if t.autopilot else "MANUAL"
        col = GREEN if t.autopilot else AMBER
        r = pygame.Rect(150, 7, 84, 20)
        self.box(surf, r, col, fill=(10, 30, 20) if t.autopilot else (40, 28, 6))
        self.text(surf, mode, self.f_title, col, r.center, "center")
        if t.scenario:
            self.text(surf, t.scenario, self.f_small, DIM, (246, 10))
        mm = int(t.mission_time // 60)
        ss = t.mission_time - 60 * mm
        self.text(surf, f"T+ {mm:02d}:{ss:04.1f}", self.f_mono_xl, WHITE, (w // 2, 17), "center")
        backend = "RUST" if t.backend.startswith("clarabel-rust") else ("PY" if "python" in t.backend else "FB")
        status_col = GREEN if t.solver_status in ("Solved", "AlmostSolved") else (DIM if t.solver_status == "idle" else AMBER)
        right = f"CLARABEL {backend}   {t.solver_status.upper()[:12]}   #{t.solves}   {t.fps:5.0f} FPS   F1 HELP"
        self.text(surf, right, self.f_mono_s, status_col, (w - 16, 17), "midright")

    # ----------------------------------------------------------------- NAV
    def nav_panel(self, surf, rect, t):
        c = self.panel(surf, rect, "NAV  -  TRAJECTORY", CYAN)
        pts = [t.nav_pad, t.nav_rocket] + list(t.nav_trajectory[-400:]) + list(t.nav_plan)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        span_x = max(200.0, (max(xs) - min(xs)) * 1.25)
        span_y = max(120.0, (max(ys) - min(ys)) * 1.25)
        scale = min(c.width / span_x, c.height / span_y)

        def M(p):
            return (c.centerx + (p[0] - cx) * scale, c.centery - (p[1] - cy) * scale)

        old = surf.get_clip()
        surf.set_clip(c)
        ground = M(t.nav_pad)[1]
        pygame.draw.rect(surf, (26, 22, 16), (c.left, ground, c.width, c.bottom - ground))
        pygame.draw.line(surf, (110, 84, 48), (c.left, ground), (c.right, ground), 1)
        # range rings every "nice" distance around the pad
        step = _nice(span_x / 4)
        px, py = M(t.nav_pad)
        for k in range(1, 5):
            r = int(step * k * scale)
            if r > 4:
                pygame.draw.circle(surf, (24, 44, 58), (px, py), r, 1)
        if len(t.nav_trajectory) > 1:
            pygame.draw.lines(surf, (60, 150, 220), False, [M(p) for p in t.nav_trajectory[-400:]], 2)
        if len(t.nav_plan) > 1:
            pygame.draw.lines(surf, GREEN, False, [M(p) for p in t.nav_plan], 1)
        pygame.draw.polygon(surf, GREEN, [(px, py - 5), (px + 5, py), (px, py + 5), (px - 5, py)], 2)
        rx, ry = M(t.nav_rocket)
        pygame.draw.circle(surf, WHITE, (rx, ry), 4)
        pygame.draw.circle(surf, CYAN, (rx, ry), 7, 1)
        surf.set_clip(old)
        dist = math.hypot(t.nav_rocket[0] - t.nav_pad[0], t.altitude)
        self.text(surf, f"RNG {dist:7.0f} m", self.f_mono_s, CYAN, (c.left + 2, c.top + 2))
        self.text(surf, f"RING {step:.0f} m", self.f_mono_s, DIM, (c.right - 2, c.top + 2), "topright")

    # ----------------------------------------------------------------- PFD
    def pfd(self, surf, rect, t):
        c = self.panel(surf, rect, "PFD  -  PRIMARY FLIGHT DISPLAY", CYAN)
        top, bottom = c.top + 20, c.bottom - 50
        spd = pygame.Rect(c.left, top, 52, bottom - top)
        alt = pygame.Rect(c.right - 78, top, 56, bottom - top)
        vsi = pygame.Rect(c.right - 18, top + 10, 18, bottom - top - 20)
        att = pygame.Rect(spd.right + 6, top, alt.left - spd.right - 12, bottom - top)
        self._attitude(surf, att, t)
        speed = math.hypot(t.vx, t.vz)
        self._tape(surf, spd, speed, "SPD", "m/s", left=True, floor=None)
        self._tape(surf, alt, t.altitude, "ALT", "m", left=False, floor=0.0)
        self._vsi(surf, vsi, t.vz)
        # bottom strip readouts
        y = bottom + 8
        vs_col = GREEN if abs(t.vz) < 2.5 or t.vz > 0 else (AMBER if t.vz > -60 else RED)
        cells = (("TILT", f"{t.tilt_deg:+5.1f}°", TEXT), ("RATE", f"{t.tilt_rate_deg:+5.1f}", TEXT),
                 ("GMBL", f"{t.gimbal_deg:+5.1f}°", AMBER if abs(t.gimbal_deg) > t.ap_gimbal_limit_deg - 0.5 else TEXT),
                 ("V/S", f"{t.vz:+6.1f}", vs_col), ("H-SPD", f"{t.vx:+6.1f}", CYAN))
        cw = c.width // 5
        for i, (label, value, col) in enumerate(cells):
            x = c.left + i * cw
            self.text(surf, label, self.f_label, DIM, (x + 2, y))
            self.text(surf, value, self.f_mono, col, (x + 2, y + 15))

    def _attitude(self, surf, r, t):
        old = surf.get_clip()
        surf.set_clip(r)
        cx, cy = r.centerx, r.centery + 6
        ang = math.radians(t.tilt_deg)
        # Horizon rotates opposite to the vehicle tilt (bank-style).
        dx, dy = math.cos(-ang), math.sin(-ang)
        nx, ny = -dy, dx
        big = 900
        sky = [(cx - dx * big, cy - dy * big), (cx + dx * big, cy + dy * big),
               (cx + dx * big - nx * big, cy + dy * big - ny * big), (cx - dx * big - nx * big, cy - dy * big - ny * big)]
        earth = [(cx - dx * big, cy - dy * big), (cx + dx * big, cy + dy * big),
                 (cx + dx * big + nx * big, cy + dy * big + ny * big), (cx - dx * big + nx * big, cy - dy * big + ny * big)]
        pygame.draw.rect(surf, SKY_TOP, r)
        pygame.draw.polygon(surf, SKY_LOW, [(cx - dx * big - nx * 40, cy - dy * big - ny * 40), (cx + dx * big - nx * 40, cy + dy * big - ny * 40),
                                            (cx + dx * big, cy + dy * big), (cx - dx * big, cy - dy * big)])
        del sky
        pygame.draw.polygon(surf, EARTH_HI, earth)
        pygame.draw.polygon(surf, EARTH_LOW, [(cx - dx * big + nx * 45, cy - dy * big + ny * 45), (cx + dx * big + nx * 45, cy + dy * big + ny * 45),
                                              (cx + dx * big + nx * big, cy + dy * big + ny * big), (cx - dx * big + nx * big, cy - dy * big + ny * big)])
        pygame.draw.line(surf, WHITE, (cx - dx * big, cy - dy * big), (cx + dx * big, cy + dy * big), 2)
        # Tilt reference lines (every 10 deg of tilt) parallel to horizon.
        for k in (-2, -1, 1, 2):
            off = k * 22
            half = 22 if abs(k) == 1 else 34
            a = (cx + nx * off - dx * half, cy + ny * off - dy * half)
            b = (cx + nx * off + dx * half, cy + ny * off + dy * half)
            pygame.draw.line(surf, (235, 240, 245), a, b, 1)
        # Tilt scale arc with ticks at the top.
        radius = min(r.width, r.height) * 0.42
        for deg in (-60, -45, -30, -20, -10, 0, 10, 20, 30, 45, 60):
            a = math.radians(deg) - math.pi / 2
            inner = radius - (10 if deg % 30 == 0 else 6)
            p1 = (cx + math.cos(a) * radius, cy + math.sin(a) * radius)
            p2 = (cx + math.cos(a) * inner, cy + math.sin(a) * inner)
            pygame.draw.line(surf, WHITE, p1, p2, 2 if deg % 30 == 0 else 1)
        # Pointer (vehicle tilt) and magenta command bug (planned thrust tilt).
        a = math.radians(t.tilt_deg) - math.pi / 2
        tip = (cx + math.cos(a) * (radius - 11), cy + math.sin(a) * (radius - 11))
        side = (math.cos(a + math.pi / 2) * 6, math.sin(a + math.pi / 2) * 6)
        back = (cx + math.cos(a) * (radius - 22), cy + math.sin(a) * (radius - 22))
        pygame.draw.polygon(surf, AMBER, [tip, (back[0] + side[0], back[1] + side[1]), (back[0] - side[0], back[1] - side[1])])
        if t.cmd_tilt_deg is not None and t.autopilot:
            a = math.radians(max(-70, min(70, t.cmd_tilt_deg))) - math.pi / 2
            p = (cx + math.cos(a) * (radius + 2), cy + math.sin(a) * (radius + 2))
            q = (cx + math.cos(a) * (radius + 11), cy + math.sin(a) * (radius + 11))
            sd = (math.cos(a + math.pi / 2) * 5, math.sin(a + math.pi / 2) * 5)
            pygame.draw.polygon(surf, MAGENTA, [p, (q[0] + sd[0], q[1] + sd[1]), (q[0] - sd[0], q[1] - sd[1])])
        # Velocity vector (where the vehicle is going).
        speed = math.hypot(t.vx, t.vz)
        if speed > 0.5:
            k = min(1.0, speed / 30.0) * radius * 0.72
            vxp, vyp = cx + t.vx / speed * k, cy - t.vz / speed * k
            pygame.draw.circle(surf, GREEN, (vxp, vyp), 6, 2)
            pygame.draw.line(surf, GREEN, (vxp - 13, vyp), (vxp - 6, vyp), 2)
            pygame.draw.line(surf, GREEN, (vxp + 6, vyp), (vxp + 13, vyp), 2)
            pygame.draw.line(surf, GREEN, (vxp, vyp - 6), (vxp, vyp - 11), 2)
        # Fixed vehicle symbol with thrust vector.
        pygame.draw.polygon(surf, (20, 20, 20), [(cx - 4, cy - 16), (cx + 4, cy - 16), (cx + 5, cy + 14), (cx - 5, cy + 14)])
        pygame.draw.polygon(surf, AMBER, [(cx - 3, cy - 15), (cx + 3, cy - 15), (cx + 4, cy + 13), (cx - 4, cy + 13)])
        pygame.draw.line(surf, AMBER, (cx - 26, cy), (cx - 9, cy), 3)
        pygame.draw.line(surf, AMBER, (cx + 9, cy), (cx + 26, cy), 3)
        if t.throttle > 0.02:
            g = math.radians(t.gimbal_deg)
            L = 8 + 22 * t.throttle
            pygame.draw.line(surf, (255, 140, 50), (cx, cy + 14), (cx + math.sin(g) * L, cy + 14 + math.cos(g) * L), 3)
        surf.set_clip(old)
        pygame.draw.rect(surf, FRAME_HI, r, 1)

    def _tape(self, surf, r, value, label, unit, left, floor):
        pygame.draw.rect(surf, (12, 18, 26), r)
        old = surf.get_clip()
        surf.set_clip(r)
        # Adaptive range so the tape is readable from 20 km down to the pad.
        mag = abs(value)
        if label == "ALT":
            step = 1000 if mag > 6000 else 100 if mag > 800 else 20 if mag > 150 else 5
        else:
            step = 50 if mag > 300 else 10 if mag > 40 else 2
        px_per_unit = 22.0 / step
        cy = r.centery
        first = math.floor((value - r.height / 2 / px_per_unit) / step) * step
        v = first
        while v <= value + r.height / 2 / px_per_unit + step:
            y = cy - (v - value) * px_per_unit
            if floor is not None and v < floor:
                v += step
                continue
            x0 = r.right - 8 if left else r.left
            x1 = r.right if left else r.left + 8
            pygame.draw.line(surf, TEXT, (x0, y), (x1, y), 1)
            half_y = y - px_per_unit * step / 2
            pygame.draw.line(surf, FAINT, (r.right - 4 if left else r.left, half_y), (r.right if left else r.left + 4, half_y), 1)
            txt = f"{v:.0f}"
            if left:
                self.text(surf, txt, self.f_mono_s, TEXT, (r.right - 11, y), "midright")
            else:
                self.text(surf, txt, self.f_mono_s, TEXT, (r.left + 11, y), "midleft")
            v += step
        if floor is not None:
            gy = cy + (value - floor) * px_per_unit
            if gy < r.bottom:
                pygame.draw.rect(surf, (90, 58, 26), (r.left, gy, r.width, r.bottom - gy))
                for k in range(-r.height, r.width + r.height, 7):
                    pygame.draw.line(surf, (140, 92, 44), (r.left + k, gy), (r.left + k + 30, gy + 30), 1)
                pygame.draw.line(surf, GREEN, (r.left, gy), (r.right, gy), 2)
        surf.set_clip(old)
        pygame.draw.rect(surf, FRAME_HI, r, 1)
        # Readout box (rolling-drum style window).
        box = pygame.Rect(0, 0, r.width + 6, 26)
        box.center = (r.centerx + (-3 if left else 3), cy)
        pygame.draw.rect(surf, (0, 0, 0), box)
        pygame.draw.rect(surf, WHITE, box, 1)
        shown = f"{value:.0f}" if abs(value) >= 100 else f"{value:.1f}"
        self.text(surf, shown, self.f_mono_b, WHITE, box.center, "center")
        self.text(surf, f"{label} {unit}", self.f_label, CYAN, (r.centerx, r.top - 1), "midbottom")

    def _vsi(self, surf, r, vz):
        pygame.draw.rect(surf, (12, 18, 26), r, border_radius=3)
        cy = r.centery
        half = r.height / 2 - 4

        def ypos(v):
            s = 1 if v > 0 else -1
            return cy - s * math.sqrt(min(abs(v), 120.0) / 120.0) * half

        for v, lab in ((5, "5"), (20, "20"), (50, "50"), (100, "")):
            for s in (1, -1):
                y = ypos(s * v)
                pygame.draw.line(surf, TEXT, (r.left, y), (r.left + 5, y), 1)
                if lab and r.width >= 16:
                    self.text(surf, lab, self.f_label, DIM, (r.left + 7, y), "midleft")
        pygame.draw.line(surf, TEXT, (r.left, cy), (r.left + 8, cy), 2)
        y = ypos(vz)
        col = GREEN if abs(vz) < 2.5 or vz > 0 else (AMBER if vz > -60 else RED)
        bar = pygame.Rect(r.right - 7, min(cy, y), 5, max(2, abs(y - cy)))
        pygame.draw.rect(surf, col, bar)
        pygame.draw.polygon(surf, col, [(r.right - 8, y), (r.right - 14, y - 4), (r.right - 14, y + 4)])
        pygame.draw.rect(surf, FRAME_HI, r, 1, border_radius=3)

    # ------------------------------------------------------------ autopilot
    def autopilot_panel(self, surf, rect, t):
        c = self.panel(surf, rect, "AUTOPILOT  -  CONVEX GUIDANCE", GREEN if t.autopilot else AMBER)
        eng = pygame.Rect(c.left, c.top + 2, c.width, 28)
        if t.autopilot:
            self.box(surf, eng, GREEN, fill=(8, 36, 20), width=2)
            self.text(surf, "AP ENGAGED   AUTOLAND", self.f_title, GREEN, eng.center, "center")
        else:
            self.box(surf, eng, AMBER, fill=(44, 30, 4), width=2)
            self.text(surf, "AP DISENGAGED   MANUAL", self.f_title, AMBER, eng.center, "center")
        # Phase sequence.
        y = eng.bottom + 8
        current = "LANDED" if t.state == "LANDED" else t.phase
        for name in PHASES:
            row = pygame.Rect(c.left, y, c.width, 21)
            active = name == current and t.autopilot
            done = name in self.visited and not active
            lamp = GREEN if active else ((40, 110, 70) if done else (40, 52, 62))
            if active:
                pygame.draw.rect(surf, (14, 46, 28), row, border_radius=3)
                pygame.draw.rect(surf, GREEN, row, 1, border_radius=3)
            pygame.draw.circle(surf, lamp, (row.left + 11, row.centery), 5)
            if active:
                pygame.draw.circle(surf, WHITE, (row.left + 11, row.centery), 2)
            label_col = WHITE if active else (GREEN if done else DIM)
            self.text(surf, name, self.f_small, label_col, (row.left + 24, row.centery), "midleft")
            status = "ACTIVE" if active else ("DONE" if done else "ARMED" if t.autopilot else "")
            self.text(surf, status, self.f_label, label_col if active else FAINT if not done else (60, 170, 110), (row.right - 6, row.centery), "midright")
            y += 23
        # Data grid.
        y += 4
        tgo = f"{t.t_go:5.1f} s" if math.isfinite(t.t_go) and t.t_go > 0 and t.state == "FLYING" else "  --"
        items = (("T-GO", tgo, MAGENTA), ("PAD DX", f"{t.x:+7.1f} m", CYAN),
                 ("SOLVER", t.solver_status[:12], GREEN if t.solver_status in ("Solved", "AlmostSolved") else AMBER),
                 ("MARGIN", f"{t.thrust_margin * 100:4.0f} %", CYAN))
        for i, (label, value, col) in enumerate(items):
            x = c.left + (i % 2) * (c.width // 2)
            yy = y + (i // 2) * 30
            self.text(surf, label, self.f_label, DIM, (x, yy))
            self.text(surf, value, self.f_mono, col, (x, yy + 13))
        # Caution / advisory lights.
        y += 64
        lights = (
            ("ENG", t.throttle > 0.02, GREEN),
            ("RCS", abs(t.rcs) > 0.08, CYAN),
            ("GMBL LIM", abs(t.gimbal_deg) > t.ap_gimbal_limit_deg - 0.5, AMBER),
            ("THR MAX", t.throttle > 0.97, AMBER),
            ("LEGS DN", t.legs > 0.98, GREEN),
            ("LOW ALT", t.state == "FLYING" and t.altitude < 50.0, AMBER),
        )
        lw = (c.width - 10) // 3
        for i, (label, on, col) in enumerate(lights):
            lr = pygame.Rect(c.left + (i % 3) * (lw + 5), y + (i // 3) * 24, lw, 20)
            if lr.bottom > c.bottom + 4:
                break
            if on:
                pygame.draw.rect(surf, tuple(v // 5 for v in col), lr, border_radius=3)
                pygame.draw.rect(surf, col, lr, 1, border_radius=3)
                self.text(surf, label, self.f_label, col, lr.center, "center")
            else:
                pygame.draw.rect(surf, (20, 26, 32), lr, border_radius=3)
                pygame.draw.rect(surf, (40, 50, 60), lr, 1, border_radius=3)
                self.text(surf, label, self.f_label, FAINT, lr.center, "center")

    # --------------------------------------------------------------- engine
    def engine_panel(self, surf, rect, t):
        c = self.panel(surf, rect, "ENGINE  -  THROTTLE / TVC", AMBER)
        # Throttle lever slider.
        track = pygame.Rect(c.left + 30, c.top + 12, 26, c.height - 30)
        pygame.draw.rect(surf, (14, 18, 24), track, border_radius=6)
        level_y = track.bottom - track.height * t.throttle
        fill = pygame.Rect(track.left + 3, level_y, track.width - 6, track.bottom - level_y - 2)
        if fill.height > 0:
            for i in range(0, fill.height, 3):
                k = 1.0 - (i / max(1, track.height))
                col = (255, int(120 + 90 * k), int(30 + 40 * k))
                pygame.draw.rect(surf, col, (fill.left, fill.bottom - i - 3, fill.width, 3))
        pygame.draw.rect(surf, FRAME_HI, track, 1, border_radius=6)
        for pct in range(0, 101, 10):
            y = track.bottom - track.height * pct / 100
            major = pct % 25 == 0 or pct == 100
            pygame.draw.line(surf, TEXT if major else FAINT, (track.left - (9 if major else 5), y), (track.left - 2, y), 1)
            if pct in (0, 25, 50, 75, 100):
                self.text(surf, f"{pct}", self.f_label, DIM, (track.left - 12, y), "midright")
        # Lever handle (grip) at the actual throttle.
        grip = pygame.Rect(0, 0, 44, 14)
        grip.center = (track.centerx, level_y)
        pygame.draw.rect(surf, (190, 196, 204), grip, border_radius=3)
        pygame.draw.rect(surf, (90, 96, 104), grip, 1, border_radius=3)
        for k in (-8, -4, 0, 4, 8):
            pygame.draw.line(surf, (110, 116, 124), (grip.centerx + k, grip.top + 3), (grip.centerx + k, grip.bottom - 4), 1)
        # Commanded throttle (autopilot) as a magenta chevron.
        if t.throttle_cmd is not None and t.autopilot:
            cy = track.bottom - track.height * max(0.0, min(1.0, t.throttle_cmd))
            x = track.right + 6
            pygame.draw.polygon(surf, MAGENTA, [(x, cy), (x + 9, cy - 6), (x + 9, cy + 6)])
        self.text(surf, "THR", self.f_label, DIM, (track.centerx, track.bottom + 9), "midtop")
        # Right column: readouts + gimbal gauge.
        x0 = track.right + 26
        self.text(surf, "THROTTLE", self.f_label, DIM, (x0, c.top + 4))
        self.text(surf, f"{t.throttle * 100:5.1f} %", self.f_mono_xl, AMBER if t.throttle > 0.02 else DIM, (x0, c.top + 17))
        cmd = "--" if t.throttle_cmd is None or not t.autopilot else f"{t.throttle_cmd * 100:4.0f} %"
        self.text(surf, f"CMD {cmd}", self.f_mono_s, MAGENTA, (x0, c.top + 46))
        self.text(surf, f"THRUST {t.thrust_kn:6.0f} kN", self.f_mono_s, TEXT, (x0, c.top + 64))
        self.text(surf, f"T/W    {t.twr:6.2f}", self.f_mono_s, TEXT, (x0, c.top + 80))
        # Gimbal semicircle gauge.
        gc = (x0 + (c.right - x0) // 2, c.bottom - 22)
        gr = min(62, (c.right - x0) // 2 - 6)
        lim = t.gimbal_limit_deg
        steps = 24
        pts = []
        for i in range(steps + 1):
            a = math.radians(-lim + 2 * lim * i / steps) * (90.0 / lim) - math.pi / 2
            pts.append((gc[0] + math.cos(a) * gr, gc[1] + math.sin(a) * gr))
        pygame.draw.lines(surf, FRAME_HI, False, pts, 2)
        for deg in (-30, -20, -12, -10, 0, 10, 12, 20, 30):
            if abs(deg) > lim:
                continue
            a = math.radians(deg * 90.0 / lim) - math.pi / 2
            col = AMBER if abs(deg) == 12 else TEXT
            inner = gr - (10 if deg in (0, -30, 30) or abs(deg) == 12 else 6)
            pygame.draw.line(surf, col, (gc[0] + math.cos(a) * gr, gc[1] + math.sin(a) * gr),
                             (gc[0] + math.cos(a) * inner, gc[1] + math.sin(a) * inner), 2 if abs(deg) == 12 else 1)
        a = math.radians(max(-lim, min(lim, t.gimbal_deg)) * 90.0 / lim) - math.pi / 2
        pygame.draw.line(surf, WHITE, gc, (gc[0] + math.cos(a) * (gr - 4), gc[1] + math.sin(a) * (gr - 4)), 3)
        pygame.draw.circle(surf, WHITE, gc, 4)
        self.text(surf, f"GIMBAL {t.gimbal_deg:+5.1f}°", self.f_mono_s, TEXT, (gc[0], gc[1] + 6), "midtop")
        # RCS thruster indicators.
        for side in (-1, 1):
            on = (t.rcs * side) < -0.08
            col = CYAN if on else (40, 52, 62)
            px = gc[0] + side * (gr + 14)
            pygame.draw.polygon(surf, col, [(px, gc[1] - 30), (px - 6 * side, gc[1] - 36), (px - 6 * side, gc[1] - 24)])
        self.text(surf, "RCS", self.f_label, DIM, (gc[0] + gr + 14, gc[1] - 52), "midtop")

    # ------------------------------------------------------- landing monitor
    def landing_monitor(self, surf, rect, t, alpha):
        layer = pygame.Surface(rect.size, pygame.SRCALPHA)
        local = pygame.Rect(0, 0, rect.width, rect.height)
        ok_col = GREEN
        c = self.panel(layer, local, "LANDING ZONE  -  PRECISION MONITOR", ok_col)
        landed = t.state == "LANDED" and t.touchdown is not None
        err = t.touchdown["x"] if landed else t.x
        # Header readouts.
        self.text(layer, f"ALT {max(0.0, t.altitude):5.1f} m", self.f_mono_b, WHITE, (c.right, c.top - 20), "topright")
        # Scale: +/-10 m (widens if the error is larger).
        half_span = 10.0 if abs(err) < 9 else math.ceil(abs(err) * 1.2 / 5) * 5
        sx0, sx1 = c.left + 16, c.right - 16
        ppm = (sx1 - sx0) / (2 * half_span)
        mid = (sx0 + sx1) / 2
        ground_y = c.top + 72
        top_y = c.top + 4

        def X(m):
            return mid + m * ppm

        # Tolerance bands on the pad.
        for lim, col in ((half_span, (70, 22, 20)), (3.0, (80, 60, 10)), (1.0, (16, 80, 40))):
            pygame.draw.rect(layer, col, (X(-lim), ground_y, X(lim) - X(-lim), 7))
        pygame.draw.rect(layer, (150, 152, 158), (X(-9), ground_y - 3, X(9) - X(-9), 4))
        for m in range(-int(half_span), int(half_span) + 1):
            major = m % 5 == 0
            pygame.draw.line(layer, TEXT if major else FAINT, (X(m), ground_y + 8), (X(m), ground_y + (16 if major else 12)), 1)
            if major:
                self.text(layer, f"{m:+d}" if m else "0", self.f_label, DIM, (X(m), ground_y + 17), "midtop")
        pygame.draw.line(layer, (255, 204, 40), (X(0), ground_y - 6), (X(0), ground_y + 7), 2)
        # Vehicle schematic descending onto the pad (height mapped 50..0 m).
        h_frac = max(0.0, min(1.0, t.altitude / 50.0)) if not landed else 0.0
        body_h = 30
        vy = ground_y - 4 - h_frac * (ground_y - 4 - body_h - top_y)
        vx = X(max(-half_span, min(half_span, err)))
        tilt = math.radians(t.tilt_deg)
        colour = GREEN if abs(err) < 1.0 else (AMBER if abs(err) < 3.0 else RED)

        def P(dx, dy):
            return (vx + dx * math.cos(tilt) + dy * math.sin(tilt), vy + dx * math.sin(tilt) - dy * math.cos(tilt))

        body = [P(-5, 0), P(5, 0), P(5, body_h), P(-5, body_h)]
        pygame.draw.polygon(layer, (235, 238, 242), body)
        pygame.draw.polygon(layer, (30, 32, 36), [P(-5, body_h - 9), P(5, body_h - 9), P(5, body_h), P(-5, body_h)])
        pygame.draw.polygon(layer, colour, body, 2)
        legs = max(0.3, t.legs)
        for s in (-1, 1):
            pygame.draw.line(layer, (200, 204, 210), P(s * 5, 6), P(s * (5 + 9 * legs), -1 * legs), 2)
        if t.throttle > 0.02 and not landed:
            fl = 6 + 18 * t.throttle
            pygame.draw.polygon(layer, (255, 170, 60), [P(-3, 0), P(3, 0), P(0, -fl)])
        # Predicted touchdown point from current drift (magenta).
        if not landed and t.vz < -0.3:
            t_touch = t.altitude / max(0.5, -t.vz)
            pred = err + t.vx * t_touch
            px = X(max(-half_span, min(half_span, pred)))
            pygame.draw.line(layer, MAGENTA, (px - 5, ground_y - 13), (px + 5, ground_y - 3), 2)
            pygame.draw.line(layer, MAGENTA, (px + 5, ground_y - 13), (px - 5, ground_y - 3), 2)
            pygame.draw.line(layer, (150, 60, 130), (vx, vy + 2), (px, ground_y - 8), 1)
        # Lateral velocity arrow.
        if abs(t.vx) > 0.05 and not landed:
            L = max(-60, min(60, t.vx * 18))
            ay = vy - body_h - 8
            pygame.draw.line(layer, CYAN, (vx, ay), (vx + L, ay), 2)
            s = 1 if L > 0 else -1
            pygame.draw.polygon(layer, CYAN, [(vx + L, ay), (vx + L - 7 * s, ay - 4), (vx + L - 7 * s, ay + 4)])
        # Readout cells and gates.
        cells_y = ground_y + 30
        if landed:
            td = t.touchdown
            values = (("PAD ERROR", f"{abs(td['x']):5.2f} m", colour), ("SINK", f"{abs(td['vz']):4.2f} m/s", GREEN if abs(td['vz']) < 2 else AMBER),
                      ("DRIFT", f"{td['vx']:+4.2f} m/s", GREEN if abs(td['vx']) < 1 else AMBER), ("TILT", f"{td['tilt']:+4.1f}°", GREEN if abs(td['tilt']) < 5 else AMBER))
        else:
            values = (("DX", f"{err:+6.2f} m", colour), ("V DOWN", f"{-t.vz:5.1f} m/s", GREEN if -t.vz < 4 or t.altitude > 15 else AMBER),
                      ("V SIDE", f"{t.vx:+5.2f} m/s", GREEN if abs(t.vx) < 1 else AMBER), ("TILT", f"{t.tilt_deg:+5.1f}°", GREEN if abs(t.tilt_deg) < 5 else AMBER))
        cw = (c.width - 150) // 4
        for i, (label, value, col) in enumerate(values):
            x = c.left + i * cw
            self.text(layer, label, self.f_label, DIM, (x, cells_y))
            self.text(layer, value, self.f_mono_b, col, (x, cells_y + 13))
        gates = (("POS", abs(err) < 1.0), ("VEL", abs(t.vx) < 1.0 and (landed or -t.vz < 4.0 or t.altitude > 15)),
                 ("ATT", abs(t.tilt_deg) < 5.0))
        if landed:
            gates = (("POS", abs(t.touchdown["x"]) < 1.0), ("VEL", abs(t.touchdown["vz"]) < 2.0 and abs(t.touchdown["vx"]) < 1.0),
                     ("ATT", abs(t.touchdown["tilt"]) < 5.0))
        gx = c.right - 140
        for i, (label, ok) in enumerate(gates):
            gr = pygame.Rect(gx + i * 47, cells_y + 2, 43, 26)
            col = GREEN if ok else AMBER
            pygame.draw.rect(layer, tuple(v // 5 for v in col), gr, border_radius=4)
            pygame.draw.rect(layer, col, gr, 1, border_radius=4)
            self.text(layer, label, self.f_title, col, gr.center, "center")
        if landed:
            banner = pygame.Rect(0, 0, 210, 24)
            banner.midtop = (c.centerx, top_y)
            pygame.draw.rect(layer, (10, 50, 26), banner, border_radius=4)
            pygame.draw.rect(layer, GREEN, banner, 1, border_radius=4)
            self.text(layer, "TOUCHDOWN CONFIRMED", self.f_title, GREEN, banner.center, "center")
        layer.set_alpha(int(255 * alpha))
        surf.blit(layer, rect.topleft)

    # --------------------------------------------------------------- callouts
    def callouts(self, surf, t, w, h):
        if self.callout is not None:
            text, start, colour = self.callout
            age = t.mission_time - start
            if 0.0 <= age < 2.6:
                a = min(1.0, age / 0.25) * min(1.0, (2.6 - age) / 0.6)
                img = self.f_big.render(text, True, colour)
                pad = 18
                box = pygame.Surface((img.get_width() + pad * 2, img.get_height() + 12), pygame.SRCALPHA)
                box.fill((4, 10, 18, 190))
                pygame.draw.line(box, (*colour, 255), (0, 0), (box.get_width(), 0), 2)
                pygame.draw.line(box, (*colour, 255), (0, box.get_height() - 1), (box.get_width(), box.get_height() - 1), 2)
                box.blit(img, (pad, 6))
                box.set_alpha(int(255 * a))
                surf.blit(box, (w // 2 - box.get_width() // 2, 58))
            elif age >= 2.6:
                self.callout = None
        if t.state == "CRASHED":
            self._banner(surf, w, h, "VEHICLE LOST", f"{t.crash_reason or 'GROUND IMPACT'}   -   R: RESET   1-5: DEMO", RED)
        if t.paused:
            self._banner(surf, w, h, "SIMULATION PAUSED", "PRESS P TO RESUME", CYAN)

    def _banner(self, surf, w, h, title, sub, colour):
        img = self.f_huge.render(title, True, colour)
        sub_img = self.f_small.render(sub, True, TEXT)
        bw = max(img.get_width(), sub_img.get_width()) + 60
        box = pygame.Surface((bw, 84), pygame.SRCALPHA)
        box.fill((4, 8, 14, 215))
        pygame.draw.rect(box, (*colour, 255), box.get_rect(), 2, border_radius=6)
        box.blit(img, ((bw - img.get_width()) // 2, 8))
        box.blit(sub_img, ((bw - sub_img.get_width()) // 2, 54))
        surf.blit(box, (w // 2 - bw // 2, int(h * 0.30)))

    # ------------------------------------------------------------------ help
    def help_overlay(self, surf):
        w, h = surf.get_size()
        rows = (("1 - 5", "Demo landings (autopilot)"), ("M", "Autopilot on / off"), ("T", "Set flight state"),
                ("W / S", "Manual throttle"), ("A / D", "Manual gimbal"), ("Z / SPACE", "Full throttle / cutoff"),
                ("WHEEL", "Zoom (manual camera)"), ("TAB", "Force vectors"), ("P / R", "Pause / reset to pad"), ("F1", "Close help"))
        rect = pygame.Rect(0, 0, 380, 44 + 24 * len(rows))
        rect.center = (w // 2, h // 2)
        c = self.panel(surf, rect, "CONTROLS", CYAN, alpha=235)
        for i, (k, d) in enumerate(rows):
            y = c.top + 6 + i * 24
            kr = pygame.Rect(c.left + 4, y, 92, 20)
            pygame.draw.rect(surf, (20, 30, 40), kr, border_radius=3)
            pygame.draw.rect(surf, FRAME_HI, kr, 1, border_radius=3)
            self.text(surf, k, self.f_mono_s, CYAN, kr.center, "center")
            self.text(surf, d, self.f_small, TEXT, (kr.right + 12, kr.centery), "midleft")


def _nice(value: float) -> float:
    value = max(value, 1.0)
    e = 10 ** math.floor(math.log10(value))
    f = value / e
    return (1 if f <= 1 else 2 if f <= 2 else 5 if f <= 5 else 10) * e
