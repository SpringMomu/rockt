"""World-space visuals: booster, engine plume, smoke/dust and the landing pad.

Everything here only *draws*; physics lives in ``main.Rocket``.  Coordinates
are the simulator's world metres, converted with the simulator's
``world_to_screen``.  Glow effects use additive blending
(``pygame.BLEND_RGB_ADD``) on small off-screen surfaces, and smoke uses a
cache of soft round sprites, so the per-frame cost stays low.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import pygame

# ----------------------------------------------------------------- palette
BODY = (232, 235, 238)
BODY_SHADE = (188, 194, 202)
BODY_DEEP = (150, 157, 168)
BODY_LIGHT = (250, 251, 252)
SOOT = (58, 54, 52)
INTERSTAGE = (26, 28, 32)
INTERSTAGE_LIGHT = (58, 62, 70)
CARBON = (34, 36, 40)
ENGINE_BAY = (44, 46, 50)
NOZZLE = (96, 99, 106)
NOZZLE_DARK = (58, 60, 66)
OUTLINE = (30, 34, 40)


def _mix(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _smooth(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


@dataclass
class _Puff:
    x: float
    y: float
    vx: float
    vy: float
    r: float
    growth: float
    life: float
    max_life: float
    alpha: float
    color: tuple
    kind: int  # 0 smoke, 1 dust, 2 rcs


class Visuals:
    """Stateful renderer for the vehicle, its plume and particle effects."""

    MAX_PUFFS = 650

    def __init__(self, seed: int = 11) -> None:
        self.random = random.Random(seed)
        self.puffs: list[_Puff] = []
        self.legs = 0.0  # 0 stowed .. 1 deployed
        self.time = 0.0
        self._sprites: dict[tuple, pygame.Surface] = {}
        self._smoke_budget = 0.0
        self._dust_budget = 0.0
        self._rcs_budget = 0.0

    def reset(self) -> None:
        self.puffs.clear()
        self.legs = 0.0
        self._smoke_budget = self._dust_budget = self._rcs_budget = 0.0

    # ------------------------------------------------------------ geometry
    @staticmethod
    def _plume_direction(rocket) -> pygame.Vector2:
        a = rocket.angle - rocket.gimbal
        return pygame.Vector2(-math.sin(a), -math.cos(a))

    @staticmethod
    def _nozzle_exit(rocket) -> pygame.Vector2:
        half = rocket.LENGTH * 0.5
        pivot = rocket.local_to_world(0.0, -half)
        return pivot + Visuals._plume_direction(rocket) * 1.7

    @staticmethod
    def plume_length_m(throttle: float) -> float:
        return 6.0 + 24.0 * throttle

    # --------------------------------------------------------------- update
    def update(self, dt: float, rocket) -> None:
        self.time += dt
        flying = rocket.state == "FLYING"
        deploy = rocket.state == "LANDED" or (flying and rocket.altitude < 160.0)
        target = 1.0 if deploy else 0.0
        self.legs += max(-dt / 1.4, min(dt / 1.4, target - self.legs))

        for p in self.puffs:
            drag = math.exp(-(1.1 if p.kind == 0 else 1.6 if p.kind == 1 else 4.0) * dt)
            p.vx *= drag
            p.vy *= drag
            p.vy += (1.2 if p.kind == 0 else -0.6 if p.kind == 1 else 0.0) * dt
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.r += p.growth * dt
            p.life -= dt
            # Particles cannot enter the ground: they spread along it instead.
            floor = rocket.surface_y(p.x) + p.r * 0.35
            if p.y < floor:
                p.y = floor
                if p.vy < 0.0:
                    spread = -p.vy * 0.8
                    p.vx += spread if (p.vx >= 0.0) else -spread
                    p.vy = 0.0
        self.puffs = [p for p in self.puffs if p.life > 0.0]

        if not rocket.engine_running or not flying:
            self._spawn_rcs(dt, rocket)
            return
        thr = rocket.throttle
        direction = self._plume_direction(rocket)
        exit_point = self._nozzle_exit(rocket)
        up = rocket.local_up(exit_point)
        plume = self.plume_length_m(thr)
        nozzle_alt = rocket.altitude_at(exit_point)
        down = -direction.dot(up)  # >0 when the plume points at the ground

        # Exhaust smoke leaves the end of the plume (a contrail at speed).
        self._smoke_budget += (14.0 + 34.0 * thr) * dt
        tip = exit_point + direction * plume
        if down > 0.2 and nozzle_alt < plume:
            tip = exit_point + direction * max(0.0, nozzle_alt / max(down, 0.2))
        while self._smoke_budget >= 1.0 and len(self.puffs) < self.MAX_PUFFS:
            self._smoke_budget -= 1.0
            spread = self.random.uniform(-0.35, 0.35)
            speed = self.random.uniform(18.0, 38.0) * (0.5 + thr)
            dvec = direction.rotate(math.degrees(spread) * 0.4)
            g = self.random.randint(196, 226)
            self.puffs.append(_Puff(
                tip.x + self.random.uniform(-1, 1), tip.y + self.random.uniform(-1, 1),
                dvec.x * speed + rocket.velocity.x * 0.25, dvec.y * speed + rocket.velocity.y * 0.25,
                1.4 + 1.6 * thr, self.random.uniform(2.2, 4.2), self.random.uniform(1.6, 3.0), 3.0,
                self.random.uniform(55, 95), (g, g - 3, g - 8), 0))
            self.puffs[-1].max_life = self.puffs[-1].life

        # Ground dust / steam when the plume impinges on the surface.
        reach = plume * 1.25
        if down > 0.3 and nozzle_alt < reach:
            k = (1.0 - nozzle_alt / reach) * (0.35 + 0.65 * thr)
            hit = exit_point + direction * (nozzle_alt / max(down, 0.3))
            ground_up = rocket.local_up(hit)
            ground_right = pygame.Vector2(ground_up.y, -ground_up.x)
            self._dust_budget += 90.0 * k * dt
            while self._dust_budget >= 1.0 and len(self.puffs) < self.MAX_PUFFS:
                self._dust_budget -= 1.0
                side = -1.0 if self.random.random() < 0.5 else 1.0
                speed = self.random.uniform(16.0, 55.0) * (0.4 + k)
                lift = self.random.uniform(1.5, 7.0)
                vel = ground_right * (side * speed) + ground_up * lift
                c = self.random.randint(150, 190)
                self.puffs.append(_Puff(
                    hit.x + ground_right.x * side * 1.5, hit.y + 0.6,
                    vel.x, vel.y, 1.6, self.random.uniform(5.0, 10.0),
                    self.random.uniform(1.2, 2.8), 2.8, self.random.uniform(90, 150),
                    (c + 12, c, c - 22), 1))
                self.puffs[-1].max_life = self.puffs[-1].life
        self._spawn_rcs(dt, rocket)

    def _spawn_rcs(self, dt: float, rocket) -> None:
        cmd = getattr(rocket, "rcs_command", 0.0)
        if abs(cmd) < 0.08 or rocket.state != "FLYING":
            return
        self._rcs_budget += 40.0 * abs(cmd) * dt
        half = rocket.LENGTH * 0.5
        # A positive command spins the nose clockwise: fire from the left side.
        side = -1.0 if cmd > 0 else 1.0
        base = rocket.local_to_world(side * rocket.WIDTH * 0.5, half - 0.6)
        right = pygame.Vector2(math.cos(rocket.angle), -math.sin(rocket.angle))
        while self._rcs_budget >= 1.0 and len(self.puffs) < self.MAX_PUFFS:
            self._rcs_budget -= 1.0
            v = right * (side * self.random.uniform(10, 18)) + rocket.velocity
            self.puffs.append(_Puff(base.x, base.y, v.x, v.y, 0.25, 3.5, 0.35, 0.35, 170, (240, 245, 250), 2))

    # --------------------------------------------------------------- sprites
    def _sprite(self, radius_px: int, color: tuple) -> pygame.Surface:
        key = (radius_px, color)
        sprite = self._sprites.get(key)
        if sprite is None:
            if len(self._sprites) > 900:
                self._sprites.clear()
            size = radius_px * 2 + 2
            sprite = pygame.Surface((size, size), pygame.SRCALPHA)
            steps = max(3, min(12, radius_px // 2 + 2))
            for i in range(steps):
                f = 1.0 - i / steps
                a = int(255 * (1.0 - f) ** 1.6 * 0.9 + 18)
                pygame.draw.circle(sprite, (*color, min(255, a)), (radius_px + 1, radius_px + 1), max(1, int(radius_px * f)))
            self._sprites[key] = sprite
        return sprite

    def draw_particles(self, surface, world_to_screen, scale: float) -> None:
        for p in self.puffs:
            frac = p.life / p.max_life
            alpha = p.alpha * (frac ** 1.25 if p.kind != 2 else frac)
            if alpha < 3:
                continue
            r = p.r * scale
            if r < 1.2:
                r = 1.2
                alpha *= 0.6
            rq = int(min(160, max(1, round(r / 2.0) * 2)))
            quant = tuple(v // 8 * 8 for v in p.color)
            sprite = self._sprite(rq, quant)
            sprite.set_alpha(int(alpha))
            c = world_to_screen(pygame.Vector2(p.x, p.y))
            surface.blit(sprite, (c.x - rq - 1, c.y - rq - 1))

    # ----------------------------------------------------------------- pad
    def draw_pad(self, surface, rocket, world_to_screen, scale: float, mission_time: float) -> None:
        pad_x = rocket.launch_x
        surf_y = rocket.surface_y(pad_x)

        def P(x, dy):
            return world_to_screen(pygame.Vector2(pad_x + x, surf_y + dy))

        half = 9.0
        apron = 38.0
        # Apron and concrete slab (side view).
        pygame.draw.polygon(surface, (70, 74, 72), [P(-apron, 0.02), P(apron, 0.02), P(apron, -0.6), P(-apron, -0.6)])
        pygame.draw.polygon(surface, (156, 158, 160), [P(-half, 0.06), P(half, 0.06), P(half + 0.8, -0.9), P(-half - 0.8, -0.9)])
        pygame.draw.polygon(surface, (118, 120, 124), [P(-half - 0.8, -0.9), P(half + 0.8, -0.9), P(half + 0.8, -1.4), P(-half - 0.8, -1.4)])
        if scale > 1.2:
            # Yellow touchdown mark and hash lines on the slab edge.
            a, b = P(-0.6, 0.07), P(0.6, 0.07)
            pygame.draw.line(surface, (255, 204, 40), a, b, max(2, int(scale * 0.25)))
            for i in range(-4, 5):
                if i == 0:
                    continue
                x = i * 2.0
                pygame.draw.line(surface, (200, 202, 206), P(x - 0.3, -0.35), P(x + 0.3, -0.35), 1)
        # Edge lights: slow red strobe, steady green centre-line beacons.
        blink = (math.sin(mission_time * 3.2) > 0.2)
        for x, col in ((-half - 0.4, (255, 60, 50) if blink else (90, 30, 30)), (half + 0.4, (255, 60, 50) if blink else (90, 30, 30)),
                       (-apron + 1, (60, 255, 120)), (apron - 1, (60, 255, 120))):
            c = P(x, 0.4)
            radius = max(2, min(5, int(scale * 0.35)))
            pygame.draw.circle(surface, col, c, radius)
            if col[0] > 200 or col[1] > 200:
                glow = self._glow(radius * 4, col)
                surface.blit(glow, (c.x - radius * 4, c.y - radius * 4), special_flags=pygame.BLEND_RGB_ADD)

    def _glow(self, radius: int, color: tuple) -> pygame.Surface:
        key = ("glow", radius, color)
        g = self._sprites.get(key)
        if g is None:
            g = pygame.Surface((radius * 2, radius * 2))
            g.fill((0, 0, 0))
            steps = 8
            for i in range(steps):
                f = 1.0 - i / steps
                k = (1.0 - f) ** 2 * 0.45
                pygame.draw.circle(g, tuple(int(v * k) for v in color), (radius, radius), max(1, int(radius * f)))
            self._sprites[key] = g
        return g

    # --------------------------------------------------------------- plume
    def draw_plume(self, surface, rocket, world_to_screen, scale: float, icon_scale: float) -> None:
        if not rocket.engine_running or rocket.state != "FLYING":
            return
        thr = rocket.throttle
        rs = icon_scale
        direction = self._plume_direction(rocket)
        exit_w = self._nozzle_exit_icon(rocket, world_to_screen, rs)
        d = pygame.Vector2(direction.x, -direction.y)  # screen direction
        n = pygame.Vector2(-d.y, d.x)
        flicker = 1.0 + 0.10 * math.sin(self.time * 47.0) + 0.06 * math.sin(self.time * 83.0 + 1.3)
        length = max(10.0, self.plume_length_m(thr) * 0.8 * rs) * flicker
        base_w = max(4.0, 2.3 * rs)

        # Distance (px) along the plume before it meets the ground line.
        ground_screen_y = world_to_screen(pygame.Vector2(rocket.position.x, rocket.surface_y(rocket.position.x))).y
        ground_hit = math.inf
        if d.y > 0.05:
            ground_hit = max(0.0, (ground_screen_y - exit_w.y) / d.y)

        pad = int(base_w * 2.5 + 8)
        span = length + pad
        xs = [exit_w.x, exit_w.x + d.x * span]
        ys = [exit_w.y, exit_w.y + d.y * span]
        left = int(min(xs) - base_w * 2.5 - pad)
        top = int(min(ys) - base_w * 2.5 - pad)
        width = int(max(xs) - min(xs) + base_w * 5 + pad * 2)
        height = int(max(ys) - min(ys) + base_w * 5 + pad * 2)
        if width <= 0 or height <= 0 or width > 4000 or height > 4000:
            return
        layer = pygame.Surface((width, height))
        layer.fill((0, 0, 0))
        origin = pygame.Vector2(exit_w.x - left, exit_w.y - top)

        def shape(lk, wmax, steps=14):
            left_side, right_side = [], []
            for i in range(steps + 1):
                s = i / steps
                w = base_w * 0.5 * (1.0 - s) + wmax * (s ** 0.5) * ((1.0 - s) ** 0.7)
                p = origin + d * (lk * s)
                left_side.append(p + n * w)
                right_side.append(p - n * w)
            return left_side + right_side[::-1]

        layers = (
            (1.00, base_w * 2.6, (92, 30, 6)),
            (0.78, base_w * 1.9, (128, 60, 12)),
            (0.52, base_w * 1.2, (100, 70, 28)),
            (0.30, base_w * 0.6, (55, 48, 34)),
        )
        for lk, wmax, colour in layers:
            pygame.draw.polygon(layer, colour, shape(length * lk, wmax))
        # Shock diamonds in the core.
        for i in range(1, 4):
            c = origin + d * (length * 0.09 * i)
            r = base_w * (0.30 - 0.05 * i)
            if r >= 1.0:
                pts = [c + d * r * 1.8, c + n * r, c - d * r * 1.8, c - n * r]
                pygame.draw.polygon(layer, (70, 70, 55), pts)
        # Hot glow around the nozzle.
        glow = self._glow(int(base_w * 2.2 + 4), (255, 150, 60))
        layer.blit(glow, (origin.x - glow.get_width() / 2, origin.y - glow.get_height() / 2), special_flags=pygame.BLEND_RGB_ADD)
        # The ground stops the plume: cut it and splash sideways.
        if ground_hit < length:
            cut = int(origin.y + d.y * ground_hit) + 1
            if cut < height:
                layer.fill((0, 0, 0), pygame.Rect(0, max(0, cut), width, height - max(0, cut)))
            remaining = (length - ground_hit) / length
            splash_w = base_w * (2.0 + 7.0 * remaining)
            splash_h = max(2.0, base_w * 0.7)
            sx = origin.x + d.x * ground_hit
            sy = min(cut, height) - splash_h
            ell = pygame.Surface((int(splash_w * 2) + 2, int(splash_h * 2) + 2))
            ell.fill((0, 0, 0))
            for i, col in enumerate(((120, 50, 12), (160, 100, 35), (140, 125, 80))):
                f = 1.0 - i * 0.3
                r = pygame.Rect(0, 0, int(splash_w * 2 * f), int(splash_h * 2 * f))
                r.center = (ell.get_width() // 2, ell.get_height() // 2)
                pygame.draw.ellipse(ell, col, r)
            layer.blit(ell, (sx - ell.get_width() / 2, sy - ell.get_height() / 2 + splash_h * 0.6), special_flags=pygame.BLEND_RGB_ADD)
        surface.blit(layer, (left, top), special_flags=pygame.BLEND_RGB_ADD)

        # Warm light pool on the ground below the vehicle.
        nozzle_alt = rocket.altitude_at(self._nozzle_exit(rocket))
        if nozzle_alt < 70.0 and d.y > 0.2:
            k = (1.0 - nozzle_alt / 70.0) * thr
            gx = exit_w.x + d.x * min(ground_hit, 5000)
            radius = int(max(12, min(420, (26.0 + 30 * k) * scale)))
            pool = self._glow(radius, tuple(int(v * min(1.0, 0.35 + k)) for v in (255, 140, 50)))
            flat = pygame.transform.smoothscale(pool, (radius * 2, max(4, radius // 2)))
            surface.blit(flat, (gx - radius, ground_screen_y - radius // 4), special_flags=pygame.BLEND_RGB_ADD)

    def _nozzle_exit_icon(self, rocket, world_to_screen, rs):
        centre = world_to_screen(rocket.position)
        half = rocket.LENGTH * 0.5
        a = rocket.angle
        up = pygame.Vector2(math.sin(a), -math.cos(a))
        pivot = centre - up * (half * rs)
        pd = self._plume_direction(rocket)
        return pivot + pygame.Vector2(pd.x, -pd.y) * (1.7 * rs)

    # --------------------------------------------------------------- rocket
    def draw_rocket(self, surface, rocket, world_to_screen, rs: float) -> None:
        """Draw the booster with ``rs`` px/m (may exceed the map scale when far)."""
        centre = world_to_screen(rocket.position)
        a = rocket.angle
        up = pygame.Vector2(math.sin(a), -math.cos(a))
        right = pygame.Vector2(math.cos(a), math.sin(a))
        hl = rocket.LENGTH * 0.5
        hw = rocket.WIDTH * 0.5

        def B(x, y):
            return centre + right * (x * rs) + up * (y * rs)

        def quad(x0, x1, y0, y1, colour):
            pygame.draw.polygon(surface, colour, [B(x0, y0), B(x1, y0), B(x1, y1), B(x0, y1)])

        detailed = rocket.LENGTH * rs >= 40.0
        legs = _smooth(self.legs)

        # Landing legs behind the body.
        self._draw_legs(surface, B, hl, hw, rs, legs, behind=True)

        # Nozzle (follows the gimbal).
        pd = self._plume_direction(rocket)
        nd = pygame.Vector2(pd.x, -pd.y)
        nn = pygame.Vector2(-nd.y, nd.x)
        pivot = B(0.0, -hl)
        throat = pivot + nd * (0.2 * rs)
        exit_c = pivot + nd * (1.7 * rs)
        bell = [throat + nn * (0.34 * rs), throat - nn * (0.34 * rs), exit_c - nn * (0.8 * rs), exit_c + nn * (0.8 * rs)]
        pygame.draw.polygon(surface, NOZZLE, bell)
        if detailed:
            pygame.draw.polygon(surface, NOZZLE_DARK, [throat + nn * (0.34 * rs), throat, exit_c, exit_c + nn * (0.8 * rs)])
            if rocket.engine_running:
                pygame.draw.line(surface, (255, 170, 80), exit_c - nn * (0.8 * rs), exit_c + nn * (0.8 * rs), max(1, int(rs * 0.18)))

        # Tank with cylindrical shading and a soot gradient at the bottom.
        strips = ((-hw, -hw + 0.28, BODY_DEEP), (-hw + 0.28, -hw + 0.85, BODY_SHADE), (-hw + 0.85, hw - 0.8, BODY),
                  (hw - 0.8, hw - 0.35, BODY_LIGHT), (hw - 0.35, hw, BODY))
        tank_bottom = -hl + 1.1
        tank_top = hl - 3.0
        if detailed:
            soot_top = -hl + 8.5
            bands = 7
            for x0, x1, col in strips:
                quad(x0, x1, soot_top, tank_top, col)
                for i in range(bands):
                    y0 = tank_bottom + (soot_top - tank_bottom) * i / bands
                    y1 = tank_bottom + (soot_top - tank_bottom) * (i + 1) / bands + 0.02
                    quad(x0, x1, y0, y1, _mix(col, SOOT, 0.72 * (1.0 - i / bands) ** 1.3))
        else:
            quad(-hw, hw, tank_bottom, tank_top, BODY)
            quad(-hw, -hw + 0.8, tank_bottom, tank_top, BODY_SHADE)
        # Engine bay (octaweb) and black interstage with grid fins.
        quad(-hw, hw, -hl, tank_bottom, ENGINE_BAY)
        quad(-hw, hw, tank_top, hl, INTERSTAGE)
        if detailed:
            quad(hw - 0.55, hw - 0.2, tank_top, hl, INTERSTAGE_LIGHT)
            pygame.draw.line(surface, (80, 84, 92), B(-hw, tank_top), B(hw, tank_top), max(1, int(rs * 0.08)))
        for side in (-1.0, 1.0):
            x_in = side * hw
            x_out = side * (hw + 1.35)
            fin = [B(x_in, hl - 1.55), B(x_out, hl - 1.45), B(x_out, hl - 0.35), B(x_in, hl - 0.3)]
            pygame.draw.polygon(surface, (72, 76, 84), fin)
            if detailed and rs > 6:
                for k in range(1, 4):
                    t = k / 4
                    pygame.draw.line(surface, (40, 43, 48), B(x_in + (x_out - x_in) * t, hl - 1.52), B(x_in + (x_out - x_in) * t, hl - 0.32), 1)
                pygame.draw.line(surface, (40, 43, 48), B(x_in, hl - 0.92), B(x_out, hl - 0.9), 1)
        if detailed:
            pygame.draw.polygon(surface, OUTLINE, [B(-hw, -hl), B(hw, -hl), B(hw, hl), B(-hw, hl)], 1)

        self._draw_legs(surface, B, hl, hw, rs, legs, behind=False)

        # Keep a far-away vehicle findable: amber target brackets.
        if rocket.LENGTH * rs < 22.0:
            c = centre
            s = 13
            for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                corner = c + pygame.Vector2(sx * s, sy * s)
                pygame.draw.line(surface, (255, 200, 80), corner, corner - pygame.Vector2(sx * 5, 0), 2)
                pygame.draw.line(surface, (255, 200, 80), corner, corner - pygame.Vector2(0, sy * 5), 2)

    @staticmethod
    def _draw_legs(surface, B, hl, hw, rs, deploy, behind):
        # Legs hinge near the base; stowed they lie along the tank pointing
        # up, deployed they swing down and out to the footpads.
        length = 5.6
        stowed = math.radians(90.0)
        deployed = math.atan2(-3.3, 4.5)
        theta = stowed + (deployed - stowed) * deploy
        width = max(2, int(rs * 0.38))
        for side in (-1.0, 1.0):
            hinge_x, hinge_y = side * (hw + 0.05), -hl + 3.2
            foot_x = hinge_x + side * math.cos(theta) * length
            foot_y = hinge_y + math.sin(theta) * length
            if behind:
                continue
            strut_x, strut_y = side * hw, -hl + 6.4
            mid_x = hinge_x + (foot_x - hinge_x) * 0.55
            mid_y = hinge_y + (foot_y - hinge_y) * 0.55
            if deploy > 0.05:
                pygame.draw.line(surface, (70, 74, 80), B(strut_x, strut_y), B(mid_x, mid_y), max(1, width // 2))
            pygame.draw.line(surface, CARBON, B(hinge_x, hinge_y), B(foot_x, foot_y), width)
            if deploy > 0.6 and rs > 2.5:
                pygame.draw.line(surface, (150, 154, 160), B(foot_x - 0.45, foot_y), B(foot_x + 0.45, foot_y), max(2, int(rs * 0.22)))
