"""2D vector-thrust rocket simulator with convex powered-descent landing.

The simulation uses SI units internally (metres, seconds, kilograms, newtons)
and a fixed physics time step.  The screen is only a view of that world;
``guidance.AutonomousGuidance`` is an optional closed-loop landing controller.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from dataclasses import dataclass

import pygame
import numpy as np

from cockpit import Cockpit, Telemetry
from guidance import NATIVE_BACKEND_PREFIX, AutonomousGuidance
from visuals import Visuals


WIDTH = 1280
HEIGHT = 720
FPS = 120
PHYSICS_HZ = 120
FIXED_DT = 1.0 / PHYSICS_HZ


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def move_towards(value: float, target: float, max_delta: float) -> float:
    if value < target:
        return min(value + max_delta, target)
    return max(value - max_delta, target)


@dataclass
class ExhaustParticle:
    position: pygame.Vector2
    velocity: pygame.Vector2
    life: float
    max_life: float
    radius: float


class Rocket:
    """A simplified planar rigid-body rocket model."""

    # Large near-Earth flight region. Coordinates use the launch pad tangent
    # plane, with Earth's centre at (0, -EARTH_RADIUS).
    EARTH_RADIUS = 6_371_000.0
    MAP_LEFT = -2_000_000.0
    MAP_RIGHT = 2_000_000.0
    MAP_BOTTOM = -400_000.0
    MAP_TOP = 500_000.0
    ATMOSPHERE_TOP = 120_000.0
    SPACE_ALTITUDE = 100_000.0
    INITIAL_ALTITUDE = 25_000.0

    # Vehicle geometry and mass.
    LENGTH = 18.0
    WIDTH = 3.0
    DRY_MASS = 8_000.0
    FUEL_CAPACITY = 12_000.0

    # Engine model.
    MAX_THRUST = 360_000.0
    GIMBAL_LIMIT = math.radians(30.0)
    GIMBAL_RATE = math.radians(42.0)
    # A finite throttle response keeps manual starts and cutoffs readable.
    THROTTLE_RATE = 1.20
    THRUST_LEVER_ARM = 7.4
    # Optional cold-gas RCS actuator retained for manual/diagnostic use.  The
    # automatic landing controller deliberately commands zero RCS so TVC is
    # the only attitude actuator during an autonomous descent.
    # Sized so the nominal 20 t vehicle can rotate at roughly 40 deg/s^2;
    # unlike the engine gimbal this actuator remains available at zero thrust.
    # The independent attitude actuator remains available at zero thrust, so
    # a pilot can rotate during an engine-off coast.
    RCS_MAX_TORQUE = 1_300_000.0
    RCS_ANGULAR_ACCELERATION = math.radians(140.0)
    RCS_RATE = math.radians(90.0)

    # Environment and aerodynamic model.
    G0 = 9.80665
    SEA_LEVEL_DENSITY = 1.225
    ATMOSPHERE_SCALE_HEIGHT = 8_500.0
    DRAG_COEFFICIENT = 0.38
    FRONTAL_AREA = math.pi * (WIDTH * 0.5) ** 2
    SIDE_AREA = LENGTH * WIDTH

    def __init__(self, random_source: random.Random | None = None) -> None:
        self.launch_x = 0.0
        self.pad_center_height = self.LENGTH * 0.5 + 0.04
        self.random = random_source or random.Random()
        self.reset()

    def reset(self) -> None:
        # Start every new flight safely on the launch pad.  The optional
        # random source keeps deterministic simulations possible.
        self.spawn_x = self.launch_x
        self.position = pygame.Vector2(
            self.launch_x,
            self.world_y_for_altitude(self.launch_x, 0.0),
        )
        self.velocity = pygame.Vector2()
        local_up = self.local_up()
        self.angle = math.atan2(local_up.x, local_up.y)
        self.angular_velocity = 0.0
        self.throttle = 0.0
        self.gimbal = 0.0  # Positive moves the nozzle to the right.
        self.rcs_command = 0.0
        self.rcs_enabled = True
        self.fuel = self.FUEL_CAPACITY
        self.state = "ON PAD"
        self.crash_reason = ""
        self.last_acceleration = pygame.Vector2()
        self.last_drag = pygame.Vector2()
        self.last_thrust = pygame.Vector2()
        self.max_altitude = 0.0

    @property
    def mass(self) -> float:
        # Infinite fuel is a gameplay rule, not infinite carried mass. Keep a
        # finite nominal wet mass so thrust and inertia remain well defined.
        return self.DRY_MASS + self.FUEL_CAPACITY

    @property
    def altitude(self) -> float:
        # Read zero when the upright vehicle rests on the local surface.
        return max(0.0, self.geometric_altitude - self.pad_center_height)

    @property
    def geometric_altitude(self) -> float:
        return (self.position - self.earth_center()).length() - self.EARTH_RADIUS

    @property
    def vertical_speed(self) -> float:
        return self.velocity.dot(self.local_up())

    @property
    def horizontal_speed(self) -> float:
        return self.velocity.dot(self.local_right())

    @property
    def local_pitch(self) -> float:
        local_vertical_angle = math.atan2(self.local_up().x, self.local_up().y)
        return (self.angle - local_vertical_angle + math.pi) % (2.0 * math.pi) - math.pi

    @property
    def moment_of_inertia(self) -> float:
        # Uniform-body approximation around the centre of mass.
        return self.mass * (self.LENGTH**2 + self.WIDTH**2) / 12.0

    @property
    def engine_running(self) -> bool:
        return self.throttle > 0.001 and self.state not in {"CRASHED", "LANDED"}

    @classmethod
    def earth_center(cls) -> pygame.Vector2:
        return pygame.Vector2(0.0, -cls.EARTH_RADIUS)

    @classmethod
    def surface_y(cls, x: float, altitude: float = 0.0) -> float:
        radius = cls.EARTH_RADIUS + altitude
        inside = max(0.0, radius * radius - x * x)
        return -cls.EARTH_RADIUS + math.sqrt(inside)

    def world_y_for_altitude(self, x: float, altitude: float) -> float:
        return self.surface_y(x, self.pad_center_height + altitude)

    def local_up(self, point: pygame.Vector2 | None = None) -> pygame.Vector2:
        radial = (point or self.position) - self.earth_center()
        if radial.length_squared() <= 1e-12:
            return pygame.Vector2(0.0, 1.0)
        return radial.normalize()

    def local_right(self, point: pygame.Vector2 | None = None) -> pygame.Vector2:
        up = self.local_up(point)
        return pygame.Vector2(up.y, -up.x)

    def altitude_at(self, point: pygame.Vector2) -> float:
        return (point - self.earth_center()).length() - self.EARTH_RADIUS

    def body_up(self) -> pygame.Vector2:
        return pygame.Vector2(math.sin(self.angle), math.cos(self.angle))

    def local_to_world(self, local_x: float, local_y: float) -> pygame.Vector2:
        # Local +y runs from engine to nose; local +x is the vehicle's right.
        right = pygame.Vector2(math.cos(self.angle), -math.sin(self.angle))
        return self.position + right * local_x + self.body_up() * local_y

    def outline(self) -> list[pygame.Vector2]:
        half_width = self.WIDTH * 0.5
        half_length = self.LENGTH * 0.5
        return [
            self.local_to_world(0.0, half_length),
            self.local_to_world(half_width, half_length - 3.0),
            self.local_to_world(half_width, -half_length),
            self.local_to_world(-half_width, -half_length),
            self.local_to_world(-half_width, half_length - 3.0),
        ]

    def collision_points(self) -> list[pygame.Vector2]:
        """Include the hull and fins in ground/boundary checks."""
        half_width = self.WIDTH * 0.5
        half_length = self.LENGTH * 0.5
        points = self.outline()
        points.extend(
            [
                self.local_to_world(-half_width - 1.45, -half_length - 0.1),
                self.local_to_world(half_width + 1.45, -half_length - 0.1),
            ]
        )
        return points

    def apply_input(self, keys: pygame.key.ScancodeWrapper, dt: float) -> None:
        if self.state in {"CRASHED", "LANDED"}:
            return

        throttle_axis = float(keys[pygame.K_w] or keys[pygame.K_UP]) - float(
            keys[pygame.K_s] or keys[pygame.K_DOWN]
        )
        self.throttle = clamp(
            self.throttle + throttle_axis * self.THROTTLE_RATE * dt, 0.0, 1.0
        )

        gimbal_axis = float(keys[pygame.K_d] or keys[pygame.K_RIGHT]) - float(
            keys[pygame.K_a] or keys[pygame.K_LEFT]
        )
        target = gimbal_axis * self.GIMBAL_LIMIT
        self.gimbal = move_towards(self.gimbal, target, self.GIMBAL_RATE * dt)

    def set_flight_state(
        self,
        altitude: float,
        horizontal_position: float,
        horizontal_speed: float,
    ) -> None:
        """Immediately place the vehicle in a local, upright flight state."""
        world_x = self.launch_x + horizontal_position
        self.position = pygame.Vector2(
            world_x,
            self.world_y_for_altitude(world_x, altitude),
        )
        local_up = self.local_up()
        self.velocity = self.local_right() * horizontal_speed
        self.angle = math.atan2(local_up.x, local_up.y)
        self.angular_velocity = 0.0
        self.throttle = 0.0
        self.gimbal = 0.0
        self.rcs_command = 0.0
        self.state = (
            "ON PAD"
            if (
                altitude <= 1e-9
                and abs(horizontal_position) <= 1e-9
                and abs(horizontal_speed) <= 1e-9
            )
            else "FLYING"
        )
        self.crash_reason = ""
        self.last_acceleration = pygame.Vector2()
        self.last_drag = pygame.Vector2()
        self.last_thrust = pygame.Vector2()
        self.max_altitude = max(0.0, altitude)

    def step(self, dt: float) -> None:
        if self.state in {"CRASHED", "LANDED"}:
            return

        mass = self.mass
        available_thrust = self.MAX_THRUST * self.throttle

        # The nozzle deflects with gimbal; thrust acts in the opposite direction.
        thrust_angle = self.angle - self.gimbal
        thrust_direction = pygame.Vector2(math.sin(thrust_angle), math.cos(thrust_angle))
        thrust_force = thrust_direction * available_thrust
        local_up = self.local_up()
        gravity_acceleration = self.G0 * (
            self.EARTH_RADIUS / max(self.EARTH_RADIUS, self.EARTH_RADIUS + self.geometric_altitude)
        ) ** 2
        gravity_force = -local_up * mass * gravity_acceleration

        atmospheric_altitude = max(0.0, self.geometric_altitude)
        density = (
            self.SEA_LEVEL_DENSITY
            * math.exp(-atmospheric_altitude / self.ATMOSPHERE_SCALE_HEIGHT)
            if atmospheric_altitude < self.ATMOSPHERE_TOP
            else 0.0
        )
        drag_force = pygame.Vector2()
        speed = self.velocity.length()

        if speed > 0.01:
            velocity_direction = self.velocity / speed
            body_up = self.body_up()
            axial_amount = abs(velocity_direction.dot(body_up))
            broadside_amount = abs(velocity_direction.cross(body_up))
            reference_area = (
                self.FRONTAL_AREA * axial_amount + self.SIDE_AREA * broadside_amount
            )
            drag_magnitude = (
                0.5
                * density
                * self.DRAG_COEFFICIENT
                * reference_area
                * speed
                * speed
            )
            drag_force = -velocity_direction * drag_magnitude

        # The engine is below the centre of mass.  A positive gimbal commands
        # thrust to the vehicle's left and therefore produces a negative
        # (left-rotating) body torque.  Keeping this sign consistent with the
        # thrust vector is essential for TVC-only attitude stability.
        gimbal_torque = -self.THRUST_LEVER_ARM * available_thrust * math.sin(self.gimbal)
        rcs_torque = (
            self.rcs_command * self.RCS_MAX_TORQUE
            if self.rcs_enabled and self.state == "FLYING"
            else 0.0
        )

        net_force = thrust_force + gravity_force + drag_force
        acceleration = net_force / mass
        # Translational drag remains, but the atmosphere exerts no pitch
        # torque or angular damping. Steering is controlled only by gimbal.
        angular_acceleration = (
            gimbal_torque + rcs_torque
        ) / self.moment_of_inertia

        self.last_acceleration = acceleration
        self.last_drag = drag_force
        self.last_thrust = thrust_force

        # A vehicle on the launch pad is held until thrust overcomes weight.
        if self.state == "ON PAD":
            if thrust_force.dot(local_up) <= mass * gravity_acceleration * 1.001:
                self.velocity.update(0.0, 0.0)
                self.angular_velocity = 0.0
                self.position = self.earth_center() + local_up * (
                    self.EARTH_RADIUS + self.pad_center_height
                )
                self.angle = math.atan2(local_up.x, local_up.y)
                self._consume_fuel(available_thrust, dt)
                return
            self.state = "FLYING"

        # Semi-implicit Euler is stable enough at the fixed 120 Hz time step.
        self.velocity += acceleration * dt
        self.position += self.velocity * dt
        self.angular_velocity += angular_acceleration * dt
        self.angle += self.angular_velocity * dt
        self.angle = (self.angle + math.pi) % (2.0 * math.pi) - math.pi

        self._consume_fuel(available_thrust, dt)
        self.max_altitude = max(self.max_altitude, self.altitude)
        self._resolve_ground_contact()
        if self.state != "CRASHED":
            self._resolve_map_boundaries()

    def _consume_fuel(self, thrust: float, dt: float) -> None:
        # Unlimited-fuel mode: preserve the nominal tank value indefinitely.
        # Keep the integration flow compatible with the finite-fuel model.
        del thrust, dt

    def _resolve_ground_contact(self) -> None:
        points = self.collision_points()
        contact_point = min(points, key=self.altitude_at)
        lowest_altitude = self.altitude_at(contact_point)
        local_up = self.local_up(contact_point)
        vertical_velocity = self.velocity.dot(local_up)
        if lowest_altitude > 0.0 or vertical_velocity > 0.0:
            return

        self.position += local_up * -lowest_altitude
        # A bounded pad-capture envelope turns ordinary ground contact into a
        # successful landing only when the vehicle is close to the pad,
        # upright, and moving slowly.  Every other terrain contact remains a
        # normal crash.
        pad_error = abs(self.position.x - self.launch_x)
        pitch_error = abs(self.local_pitch)
        if (
            pad_error <= 8.0
            and abs(self.horizontal_speed) <= 4.0
            and abs(self.vertical_speed) <= 4.0
            and pitch_error <= math.radians(12.0)
            and abs(self.angular_velocity) <= math.radians(20.0)
        ):
            self.state = "LANDED"
            self.crash_reason = ""
            self.velocity.update(0.0, 0.0)
            self.angular_velocity = 0.0
            self.throttle = 0.0
            self.gimbal = 0.0
            self.rcs_command = 0.0
            return

        self.state = "CRASHED"
        self.crash_reason = "GROUND IMPACT"
        self.velocity.update(0.0, 0.0)
        self.angular_velocity = 0.0
        self.throttle = 0.0
        self.gimbal = 0.0
        self.rcs_command = 0.0

    def _resolve_map_boundaries(self) -> None:
        points = self.collision_points()
        min_x = min(point.x for point in points)
        max_x = max(point.x for point in points)
        highest_altitude = max(self.altitude_at(point) for point in points)
        hit_boundary = False

        if min_x < self.MAP_LEFT:
            self.position.x += self.MAP_LEFT - min_x
            hit_boundary = True
        if max_x > self.MAP_RIGHT:
            self.position.x -= max_x - self.MAP_RIGHT
            hit_boundary = True
        if highest_altitude > self.MAP_TOP:
            self.position -= self.local_up() * (highest_altitude - self.MAP_TOP)
            hit_boundary = True

        if hit_boundary:
            self.state = "CRASHED"
            self.crash_reason = "MAP BOUNDARY IMPACT"
            self.velocity.update(0.0, 0.0)
            self.angular_velocity = 0.0
            self.throttle = 0.0


class RocketSimulator:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Vector Thrust - 2D Rocket Simulator")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.random = random.Random(17)
        self.rocket = Rocket(self.random)
        self.guidance = AutonomousGuidance()
        # Manual mode remains available, while an environment flag makes
        # deterministic headless/autonomous trials convenient.
        self.autopilot_enabled = os.environ.get("ROCKET_AUTOPILOT", "0").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.running = True
        self.paused = False
        self.show_help = False
        self.show_vectors = False
        self.accumulator = 0.0
        self.camera_center = pygame.Vector2()
        self.camera_screen_center = pygame.Vector2(WIDTH * 0.5, HEIGHT * 0.5)
        self.zoom_factor = 1.0
        self.metres_to_pixels = self.zoom_factor
        # While the autopilot flies, the camera frames rocket + pad and zooms
        # in smoothly as they converge.  Mouse-wheel zoom hands control back
        # to the user; toggling M re-enables it.
        self.auto_camera = True
        self._auto_zoom: float | None = None
        self.max_camera_scale = 7.0
        self.visuals = Visuals()
        self.mission_time = 0.0
        self.touchdown: dict | None = None
        self.particle_budget = 0.0
        self.trajectory: list[pygame.Vector2] = [self.rocket.position.copy()]
        self.predicted_trajectory: list[pygame.Vector2] = []
        self.trajectory_timer = 0.0
        self.atmosphere_cache: pygame.Surface | None = None
        self.atmosphere_cache_center = pygame.Vector2(float("inf"), float("inf"))
        self.atmosphere_cache_scale = -1.0
        # Modal state editor opened with ``T``.  Keeping this in the
        # simulator (rather than using a platform dialog) makes it work in
        # headless/test environments as well as in the Pygame window.
        self.state_dialog_active = False
        self.state_dialog_fields = ["", "", ""]
        self.state_dialog_field_index = 0
        self.state_dialog_error = ""
        self.stars = [
            (self.random.random(), self.random.random(), self.random.choice((1, 1, 1, 2)))
            for _ in range(110)
        ]
        self._load_fonts()
        self.cockpit = Cockpit()
        self.scenario_title = ""
        self._update_camera_view()
        # Start straight into a demo landing so the guidance is visible at
        # once (set ROCKET_DEMO=0 to start parked on the pad as before).
        if os.environ.get("ROCKET_DEMO", "1").lower() not in {"0", "false", "no", "off"}:
            self.launch_scenario(3)

    def launch_scenario(self, index: int) -> None:
        """Put the rocket in preset scenario ``index`` (0-4) and engage autopilot."""
        name, scenario = RECORD_SCENARIOS[index]
        rocket = self.rocket
        rocket.set_flight_state(
            scenario["initial_altitude"],
            scenario.get("horizontal_position", 0.0),
            scenario.get("horizontal_speed", 0.0),
        )
        rocket.velocity += rocket.local_up() * scenario.get("vertical_speed", 0.0)
        self.guidance.reset()
        self.autopilot_enabled = True
        self.auto_camera = True
        self._auto_zoom = None
        self._reset_effects()
        self.trajectory = [rocket.position.copy()]
        self.predicted_trajectory = []
        self.trajectory_timer = 0.0
        self.accumulator = 0.0
        self.scenario_title = f"DEMO {name.replace('_', ' ')}"
        self._update_camera_view()

    def _load_fonts(self) -> None:
        self.font_small = pygame.font.SysFont("consolas", 15)
        self.font_medium = pygame.font.SysFont("consolas", 18)
        self.font_large = pygame.font.SysFont("consolas", 28, bold=True)
        self.font_title = pygame.font.SysFont("arial", 22, bold=True)

    def world_to_screen(self, point: pygame.Vector2) -> pygame.Vector2:
        return pygame.Vector2(
            self.camera_screen_center.x
            + (point.x - self.camera_center.x) * self.metres_to_pixels,
            self.camera_screen_center.y
            - (point.y - self.camera_center.y) * self.metres_to_pixels,
        )

    def screen_to_world(self, point: pygame.Vector2) -> pygame.Vector2:
        return pygame.Vector2(
            self.camera_center.x
            + (point.x - self.camera_screen_center.x) / self.metres_to_pixels,
            self.camera_center.y
            - (point.y - self.camera_screen_center.y) / self.metres_to_pixels,
        )

    def _pad_position(self) -> pygame.Vector2:
        return pygame.Vector2(
            self.rocket.launch_x,
            self.rocket.world_y_for_altitude(self.rocket.launch_x, 0.0),
        )

    def _update_camera_view(self) -> None:
        """Keep the main view locked to the rocket; the minimap tracks the pad."""
        width, height = self.screen.get_size()
        self.camera_center = self.rocket.position.copy()
        self.camera_screen_center = pygame.Vector2(width * 0.5, height * 0.5)
        full_map_scale = min(
            width / (self.rocket.MAP_RIGHT - self.rocket.MAP_LEFT),
            height / (self.rocket.MAP_TOP - self.rocket.MAP_BOTTOM),
        )
        self.metres_to_pixels = clamp(
            self.zoom_factor,
            full_map_scale,
            self.max_camera_scale,
        )
        if self.autopilot_enabled and self.auto_camera and self.rocket.state != "ON PAD":
            # Frame the vehicle and the pad together (log-smoothed zoom).
            pad = self._pad_position()
            span_x = abs(self.rocket.position.x - pad.x) + 70.0
            span_y = abs(self.rocket.position.y - pad.y) + 70.0
            target = clamp(min(width * 0.62 / span_x, height * 0.70 / span_y), full_map_scale, 5.0)
            if self._auto_zoom is None:
                self._auto_zoom = target
            else:
                self._auto_zoom = math.exp(0.92 * math.log(self._auto_zoom) + 0.08 * math.log(target))
            self.metres_to_pixels = self._auto_zoom
            self.camera_center = (self.rocket.position + pad) * 0.5
            self.camera_screen_center = pygame.Vector2(width * 0.56, height * 0.5)

    def _map_rect(self) -> pygame.Rect:
        return self.screen.get_rect()

    def run(self, max_frames: int | None = None) -> None:
        frames = 0
        while self.running and (max_frames is None or frames < max_frames):
            frame_dt = min(self.clock.tick(FPS) / 1000.0, 0.05)
            self._handle_events()

            if not self.paused and not self.state_dialog_active:
                # Manual input and autonomous guidance are two independent
                # command sources.  Applying the manual actuator update while
                # the autopilot is enabled creates a read/modify/write race:
                # ``apply_input`` slews the gimbal toward its neutral target
                # every render frame, immediately before MPC writes the next
                # command.  Both slew rates are comparable, so the gimbal
                # remains almost at zero and the vehicle cannot build the
                # lateral thrust/attitude required by the planned trajectory.
                # Feed the manual controls only in manual mode; MPC then owns
                # throttle, gimbal and RCS for the whole fixed-step update.
                if not self.autopilot_enabled:
                    keys = pygame.key.get_pressed()
                    self.rocket.apply_input(keys, frame_dt)
                self.accumulator += frame_dt
                while self.accumulator >= FIXED_DT:
                    self._physics_step()
                    self._sync_predicted_trajectory()
                    self.trajectory_timer += FIXED_DT
                    if self.trajectory_timer >= 0.20:
                        self.trajectory_timer = 0.0
                        self.trajectory.append(self.rocket.position.copy())
                        if len(self.trajectory) > 2400:
                            self.trajectory.pop(0)
                    self.accumulator -= FIXED_DT
                self._update_particles(frame_dt)

            self._update_camera_view()
            self._draw()
            pygame.display.flip()
            frames += 1

        pygame.quit()

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif self.state_dialog_active:
                self._handle_state_dialog_event(event)
                continue
            elif event.type == pygame.MOUSEWHEEL:
                if self.auto_camera and self.autopilot_enabled:
                    self.zoom_factor = self.metres_to_pixels
                self.auto_camera = False
                self.zoom_factor = clamp(
                    self.zoom_factor * (1.18 ** event.y),
                    0.0001,
                    self.max_camera_scale,
                )
                self._update_camera_view()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_r:
                    self.scenario_title = ""
                    self.rocket.reset()
                    self.guidance.reset()
                    self._reset_effects()
                    self.trajectory = [self.rocket.position.copy()]
                    self.predicted_trajectory = []
                    self.trajectory_timer = 0.0
                elif event.key == pygame.K_SPACE:
                    self.rocket.throttle = 0.0
                elif event.key == pygame.K_z:
                    self.rocket.throttle = 1.0
                elif event.key == pygame.K_t:
                    self._open_state_dialog()
                elif event.key == pygame.K_m:
                    self.autopilot_enabled = not self.autopilot_enabled
                    if self.autopilot_enabled:
                        self.guidance.reset()
                        self.auto_camera = True
                        self._auto_zoom = None
                    else:
                        self.predicted_trajectory = []
                    self._sync_predicted_trajectory()
                elif event.key == pygame.K_p:
                    self.paused = not self.paused
                    self.accumulator = 0.0
                elif pygame.K_1 <= event.key <= pygame.K_5:
                    self.launch_scenario(event.key - pygame.K_1)
                elif event.key == pygame.K_F1:
                    self.show_help = not self.show_help
                elif event.key == pygame.K_TAB:
                    self.show_vectors = not self.show_vectors

    def _open_state_dialog(self) -> None:
        """Open the modal altitude/position/velocity editor."""
        self.state_dialog_active = True
        self.accumulator = 0.0
        self.state_dialog_fields = ["", "", ""]
        self.state_dialog_field_index = 0
        self.state_dialog_error = ""

    def _close_state_dialog(self) -> None:
        self.state_dialog_active = False
        self.state_dialog_error = ""

    def _handle_state_dialog_event(self, event: pygame.event.Event) -> None:
        if event.type != pygame.KEYDOWN:
            return
        if event.key == pygame.K_ESCAPE:
            self._close_state_dialog()
            return
        if event.key in (pygame.K_TAB, pygame.K_DOWN):
            self.state_dialog_field_index = (self.state_dialog_field_index + 1) % 3
            return
        if event.key == pygame.K_UP:
            self.state_dialog_field_index = (self.state_dialog_field_index - 1) % 3
            return
        if event.key == pygame.K_BACKSPACE:
            self.state_dialog_fields[self.state_dialog_field_index] = self.state_dialog_fields[
                self.state_dialog_field_index
            ][:-1]
            self.state_dialog_error = ""
            return
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self._apply_state_dialog()
            return

        # KEYDOWN.unicode handles digits, decimal points and +/- signs on
        # every keyboard layout without requiring a separate text-input loop.
        character = getattr(event, "unicode", "")
        if character and (character.isdigit() or character in ".+-eE"):
            self.state_dialog_fields[self.state_dialog_field_index] += character
            self.state_dialog_error = ""

    def _apply_state_dialog(self) -> None:
        try:
            altitude, horizontal_position, horizontal_speed = (
                float(value.strip()) for value in self.state_dialog_fields
            )
        except (TypeError, ValueError):
            self.state_dialog_error = "Enter three valid numbers (use +/- for signed values)."
            return

        if not all(
            math.isfinite(value)
            for value in (altitude, horizontal_position, horizontal_speed)
        ):
            self.state_dialog_error = "Values must be finite numbers."
            return
        if altitude < 0.0:
            self.state_dialog_error = "Altitude must be zero or greater."
            return
        maximum_center_altitude = max(
            0.0,
            self.rocket.MAP_TOP - self.rocket.LENGTH * 0.5 - 1.0,
        )
        if altitude > maximum_center_altitude:
            self.state_dialog_error = (
                f"Altitude must stay within 0..{maximum_center_altitude:.0f} m."
            )
            return

        world_x = self.rocket.launch_x + horizontal_position
        if not (self.rocket.MAP_LEFT <= world_x <= self.rocket.MAP_RIGHT):
            self.state_dialog_error = (
                f"Horizontal position must stay within {self.rocket.MAP_LEFT:.0f}.."
                f"{self.rocket.MAP_RIGHT:.0f} m."
            )
            return

        self.rocket.set_flight_state(
            altitude,
            horizontal_position,
            horizontal_speed,
        )
        self.guidance.set_target_altitude(0.0)
        self.guidance.reset()

        self._reset_effects()
        self.trajectory = [self.rocket.position.copy()]
        self.predicted_trajectory = []
        self.trajectory_timer = 0.0
        self.accumulator = 0.0
        self._update_camera_view()
        self._close_state_dialog()

    def _sync_predicted_trajectory(self) -> None:
        """Copy the planner's remaining trajectory (green line) for drawing.

        ``guidance.predicted_points`` already samples the exact piecewise-
        polynomial plan every 0.1 s from "now" to touchdown, so no smoothing
        is needed; the line is only anchored at the measured vehicle position.
        """
        points = self.guidance.predicted_points
        if not self.autopilot_enabled or not points:
            self.predicted_trajectory = []
            return
        raw = [pygame.Vector2(x, y) for x, y in points]
        raw[0] = self.rocket.position.copy()
        for point in raw:
            point.y = max(point.y, self.rocket.world_y_for_altitude(point.x, 0.0))
        self.predicted_trajectory = raw

    def _reset_effects(self) -> None:
        self.visuals.reset()
        self.cockpit.reset()
        self.particle_budget = 0.0
        self.mission_time = 0.0
        self.touchdown = None

    def _physics_step(self) -> None:
        """One fixed physics step (+ autopilot), with touchdown bookkeeping."""
        rocket = self.rocket
        if self.autopilot_enabled:
            self.guidance.command(rocket, FIXED_DT)
        before = (rocket.position.x - rocket.launch_x, rocket.vertical_speed, rocket.horizontal_speed, math.degrees(rocket.local_pitch))
        was_flying = rocket.state == "FLYING"
        rocket.step(FIXED_DT)
        if rocket.state == "FLYING":
            self.mission_time += FIXED_DT
        if was_flying and rocket.state == "LANDED" and self.touchdown is None:
            self.touchdown = {"x": before[0], "vz": before[1], "vx": before[2], "tilt": before[3]}

    def _update_particles(self, dt: float) -> None:
        self.visuals.update(dt, self.rocket)

    def _draw(self) -> None:
        self.screen.fill((3, 7, 15))
        self.screen.set_clip(self._map_rect())
        self._draw_sky()
        self._draw_ground()
        scale = self.metres_to_pixels
        self.visuals.draw_pad(self.screen, self.rocket, self.world_to_screen, scale, self.mission_time)
        self._draw_trajectory()
        self.visuals.draw_particles(self.screen, self.world_to_screen, scale)
        icon_scale = max(scale, 26.0 / self.rocket.LENGTH)
        self.visuals.draw_plume(self.screen, self.rocket, self.world_to_screen, scale, icon_scale)
        self.visuals.draw_rocket(self.screen, self.rocket, self.world_to_screen, icon_scale)
        if self.show_vectors:
            self._draw_force_vectors()
        self.screen.set_clip(None)
        self.cockpit.draw(self.screen, self._telemetry())
        if self.show_help:
            self.cockpit.help_overlay(self.screen)
        if self.state_dialog_active:
            self._draw_state_dialog()

    def _telemetry(self) -> Telemetry:
        rocket = self.rocket
        guidance = self.guidance
        a_max = rocket.MAX_THRUST / rocket.mass
        up = rocket.local_up()
        local_rate = rocket.angular_velocity - rocket.horizontal_speed / max(1.0, rocket.EARTH_RADIUS + rocket.geometric_altitude)
        cmd_x, cmd_z = getattr(guidance, "last_command", (0.0, 0.0))
        commanding = self.autopilot_enabled and rocket.state == "FLYING"
        pad = self._pad_position()
        return Telemetry(
            mission_time=self.mission_time,
            autopilot=self.autopilot_enabled,
            phase=guidance.phase if self.autopilot_enabled else "MANUAL",
            state=rocket.state,
            crash_reason=rocket.crash_reason,
            altitude=rocket.altitude,
            x=rocket.position.x - rocket.launch_x,
            vx=rocket.horizontal_speed,
            vz=rocket.vertical_speed,
            tilt_deg=math.degrees(rocket.local_pitch),
            tilt_rate_deg=math.degrees(local_rate),
            gimbal_deg=math.degrees(rocket.gimbal),
            gimbal_limit_deg=math.degrees(rocket.GIMBAL_LIMIT),
            ap_gimbal_limit_deg=math.degrees(getattr(guidance, "attitude_max_gimbal", rocket.GIMBAL_LIMIT)),
            throttle=rocket.throttle if rocket.state == "FLYING" else 0.0,
            throttle_cmd=min(1.0, math.hypot(cmd_x, cmd_z) / a_max) if commanding else None,
            cmd_tilt_deg=math.degrees(math.atan2(cmd_x, cmd_z)) if commanding and cmd_z > 0.5 else None,
            thrust_kn=rocket.last_thrust.length() / 1000.0 if rocket.state == "FLYING" else 0.0,
            twr=rocket.last_thrust.length() / (rocket.mass * rocket.G0) if rocket.state == "FLYING" else 0.0,
            rcs=rocket.rcs_command,
            t_go=getattr(guidance, "time_to_go", math.nan),
            solver_status=guidance.last_status,
            backend=guidance.mpc.backend,
            solves=guidance.mpc.solve_count,
            burn_locked=getattr(guidance, "burn_locked", False),
            thrust_margin=1.0 - getattr(guidance, "burn_fraction", 0.85),
            fps=self.clock.get_fps(),
            scenario=self.scenario_title if self.autopilot_enabled else "",
            legs=self.visuals.legs,
            nav_pad=(pad.x, pad.y),
            nav_rocket=(rocket.position.x, rocket.position.y),
            nav_trajectory=[(p.x, p.y) for p in self.trajectory[-400:]],
            nav_plan=[(p.x, p.y) for p in self.predicted_trajectory[::4]],
            touchdown=self.touchdown,
            paused=self.paused,
        )

    def _draw_sky(self) -> None:
        width, height = self.screen.get_size()
        centre_shift_pixels = float("inf")
        if math.isfinite(self.atmosphere_cache_center.x):
            centre_shift_pixels = (
                self.camera_center - self.atmosphere_cache_center
            ).length() * self.metres_to_pixels
        scale_change = (
            abs(math.log(self.metres_to_pixels / self.atmosphere_cache_scale))
            if self.atmosphere_cache_scale > 0.0
            else float("inf")
        )
        needs_rebuild = (
            self.atmosphere_cache is None
            or self.atmosphere_cache.get_size() != (width, height)
            or centre_shift_pixels >= 3.0
            or scale_change >= 0.018
        )
        if needs_rebuild:
            self.atmosphere_cache = self._build_radial_sky(width, height)
            self.atmosphere_cache_center = self.camera_center.copy()
            self.atmosphere_cache_scale = self.metres_to_pixels
        self.screen.blit(self.atmosphere_cache, (0, 0))

        # Stars fade in progressively through the upper atmosphere. Their
        # screen-space placement keeps the backdrop stable during zooming.
        for x_ratio, y_ratio, radius in self.stars:
            position = pygame.Vector2(x_ratio * width, y_ratio * height)
            world_point = self.screen_to_world(position)
            altitude = self.rocket.altitude_at(world_point)
            visibility = clamp((altitude - 18_000.0) / 65_000.0, 0.0, 1.0)
            if visibility <= 0.02:
                continue
            brightness = round(90 + 165 * visibility)
            pygame.draw.circle(
                self.screen,
                (brightness, brightness, min(255, brightness + 18)),
                position,
                radius,
            )

    def _build_radial_sky(self, width: int, height: int) -> pygame.Surface:
        """Build a smooth atmosphere using radial altitude at every sample."""
        reduction = 3
        low_width = max(1, math.ceil(width / reduction))
        low_height = max(1, math.ceil(height / reduction))
        scale_x = width / low_width
        scale_y = height / low_height

        screen_x = (np.arange(low_width, dtype=np.float64) + 0.5) * scale_x
        screen_y = (np.arange(low_height, dtype=np.float64) + 0.5) * scale_y
        world_x = self.camera_center.x + (
            screen_x - self.camera_screen_center.x
        ) / self.metres_to_pixels
        world_y = self.camera_center.y - (
            screen_y - self.camera_screen_center.y
        ) / self.metres_to_pixels
        radial_y = world_y + self.rocket.EARTH_RADIUS
        altitude = np.sqrt(
            world_x[:, np.newaxis] ** 2 + radial_y[np.newaxis, :] ** 2
        ) - self.rocket.EARTH_RADIUS

        altitude_stops = np.array(
            [0.0, 600.0, 2_500.0, 8_000.0, 20_000.0, 45_000.0, 85_000.0, 120_000.0],
            dtype=np.float64,
        )
        color_stops = np.array(
            [
                (176, 208, 228),
                (128, 184, 226),
                (88, 150, 212),
                (50, 106, 184),
                (22, 52, 112),
                (8, 20, 54),
                (3, 8, 24),
                (1, 3, 11),
            ],
            dtype=np.float64,
        )

        clipped_altitude = np.clip(altitude, altitude_stops[0], altitude_stops[-1])
        pixels = np.empty((low_width, low_height, 3), dtype=np.uint8)
        for channel in range(3):
            pixels[:, :, channel] = np.interp(
                clipped_altitude,
                altitude_stops,
                color_stops[:, channel],
            ).astype(np.uint8)

        low_surface = pygame.surfarray.make_surface(pixels)
        return pygame.transform.smoothscale(low_surface, (width, height)).convert()

    def _draw_ground(self) -> None:
        width, height = self.screen.get_size()
        sample_step = 3
        surface_points: list[pygame.Vector2] = []
        for screen_x in range(-sample_step, width + sample_step * 2, sample_step):
            world_x = self.screen_to_world(pygame.Vector2(screen_x, 0.0)).x
            world_x = clamp(world_x, -self.rocket.EARTH_RADIUS, self.rocket.EARTH_RADIUS)
            surface = pygame.Vector2(world_x, self.rocket.surface_y(world_x))
            surface_points.append(self.world_to_screen(surface))

        earth_polygon = surface_points + [
            pygame.Vector2(width + sample_step, height + sample_step),
            pygame.Vector2(-sample_step, height + sample_step),
        ]
        pygame.draw.polygon(self.screen, (40, 44, 38), earth_polygon)
        pygame.draw.lines(self.screen, (84, 98, 74), False, surface_points, 3)
        pygame.draw.aalines(self.screen, (128, 146, 108), False, surface_points)
        if self.metres_to_pixels < 0.02:
            pygame.draw.aalines(self.screen, (92, 164, 196), False, surface_points)


        # Adaptive surface markers make scale and planetary curvature readable.
        spacing = self._nice_world_spacing(95.0 / self.metres_to_pixels)
        left_world = self.screen_to_world(pygame.Vector2(0.0, height * 0.5)).x
        right_world = self.screen_to_world(pygame.Vector2(width, height * 0.5)).x
        left_world = max(left_world, self.rocket.MAP_LEFT)
        right_world = min(right_world, self.rocket.MAP_RIGHT)
        first = math.floor(left_world / spacing) * spacing
        marker_x = first
        while marker_x <= right_world + spacing:
            if abs(marker_x) < self.rocket.EARTH_RADIUS:
                surface = pygame.Vector2(marker_x, self.rocket.surface_y(marker_x))
                up = self.rocket.local_up(surface)
                base = self.world_to_screen(surface)
                tip = self.world_to_screen(surface + up * (8.0 / self.metres_to_pixels))
                pygame.draw.line(self.screen, (96, 108, 86), base, tip, 1)
            marker_x += spacing

    @staticmethod
    def _nice_world_spacing(value: float) -> float:
        value = max(value, 1.0)
        exponent = 10.0 ** math.floor(math.log10(value))
        fraction = value / exponent
        if fraction <= 1.0:
            nice = 1.0
        elif fraction <= 2.0:
            nice = 2.0
        elif fraction <= 5.0:
            nice = 5.0
        else:
            nice = 10.0
        return nice * exponent

    def _draw_trajectory(self) -> None:
        if len(self.predicted_trajectory) >= 2:
            forecast_points = [self.world_to_screen(point) for point in self.predicted_trajectory]
            # Green is the live MPC forecast; blue remains the measured
            # flight trace so tracking error is visible at a glance.
            pygame.draw.lines(self.screen, (18, 70, 40), False, forecast_points, 4)
            pygame.draw.aalines(self.screen, (90, 255, 150), False, forecast_points)
        if len(self.trajectory) >= 2:
            stride = max(1, math.ceil(len(self.trajectory) / 600))
            sampled = self.trajectory[::stride]
            if sampled[-1] is not self.trajectory[-1]:
                sampled = sampled + [self.trajectory[-1]]
            points = [self.world_to_screen(point) for point in sampled]
            # Use a darker trace for the actual flight path.
            pygame.draw.lines(self.screen, (60, 150, 230), False, points, 2)

    def _draw_force_vectors(self) -> None:
        origin = self.world_to_screen(self.rocket.position)

        def arrow(vector: pygame.Vector2, color: tuple[int, int, int], scale: float) -> None:
            if vector.length_squared() < 1e-8:
                return
            screen_vector = pygame.Vector2(vector.x, -vector.y) * scale
            if screen_vector.length() > 150.0:
                screen_vector.scale_to_length(150.0)
            end = origin + screen_vector
            pygame.draw.line(self.screen, color, origin, end, 3)
            direction = origin - end
            if direction.length_squared() > 0.0:
                direction.scale_to_length(9.0)
                side = pygame.Vector2(-direction.y, direction.x) * 0.45
                pygame.draw.polygon(self.screen, color, [end, end + direction + side, end + direction - side])

        arrow(self.rocket.last_thrust, (105, 255, 132), 0.00012)
        arrow(self.rocket.last_drag, (255, 202, 79), 0.00035)
        velocity_scale = 1.2
        velocity_screen = pygame.Vector2(
            self.rocket.velocity.x, -self.rocket.velocity.y
        ) * velocity_scale
        if velocity_screen.length_squared() > 1.0:
            max_length = 130.0
            if velocity_screen.length() > max_length:
                velocity_screen.scale_to_length(max_length)
            end = origin + velocity_screen
            pygame.draw.line(self.screen, (104, 205, 255), origin, end, 2)

    def _draw_state_dialog(self) -> None:
        width, height = self.screen.get_size()
        dimmer = pygame.Surface((width, height), pygame.SRCALPHA)
        dimmer.fill((0, 0, 0, 150))
        self.screen.blit(dimmer, (0, 0))

        panel_width = min(570, max(360, width - 32))
        panel_height = 314
        panel_rect = pygame.Rect(
            (width - panel_width) // 2,
            max(16, (height - panel_height) // 2),
            panel_width,
            panel_height,
        )
        self.cockpit.panel(self.screen, panel_rect, "SET FLIGHT STATE  -  INITIAL CONDITIONS", (40, 214, 255), alpha=240)

        field_labels = (
            "ALTITUDE (m, >= 0)",
            "HORIZONTAL POSITION (m, +/-)",
            "HORIZONTAL SPEED (m/s, +/-)",
        )
        field_left = panel_rect.left + 22
        field_width = panel_rect.width - 44
        row_top = panel_rect.top + 62
        for index, (label, value) in enumerate(
            zip(field_labels, self.state_dialog_fields)
        ):
            y = row_top + index * 62
            active = index == self.state_dialog_field_index
            self._blit(
                label,
                self.font_small,
                (215, 231, 240) if active else (155, 181, 196),
                (field_left, y),
            )
            input_rect = pygame.Rect(field_left, y + 20, field_width, 32)
            pygame.draw.rect(self.screen, (15, 29, 44), input_rect, border_radius=3)
            pygame.draw.rect(
                self.screen,
                (111, 218, 255) if active else (74, 105, 124),
                input_rect,
                2 if active else 1,
                border_radius=3,
            )
            shown_value = value
            self._blit(
                shown_value,
                self.font_medium,
                (240, 246, 249),
                (input_rect.left + 9, input_rect.top + 5),
            )
            if active and pygame.time.get_ticks() % 1000 < 550:
                caret_x = input_rect.left + 9 + self.font_medium.size(shown_value)[0]
                pygame.draw.line(
                    self.screen,
                    (240, 246, 249),
                    (caret_x, input_rect.top + 6),
                    (caret_x, input_rect.bottom - 6),
                    1,
                )

        footer_y = panel_rect.bottom - 38
        if self.state_dialog_error:
            self._blit(
                self.state_dialog_error,
                self.font_small,
                (255, 124, 108),
                (field_left, footer_y),
            )
        else:
            self._blit(
                "TAB / UP / DOWN: FIELD     ENTER: APPLY     ESC: CANCEL",
                self.font_small,
                (171, 199, 213),
                (field_left, footer_y),
            )

    def _blit(
        self,
        text: str,
        font: pygame.font.Font,
        color: tuple[int, int, int],
        position: tuple[float, float],
    ) -> None:
        self.screen.blit(font.render(text, True, color), position)

    def _center_message(self, text: str, color: tuple[int, int, int], y: float) -> None:
        rendered = self.font_large.render(text, True, color)
        shadow = self.font_large.render(text, True, (12, 17, 22))
        x = (self.screen.get_width() - rendered.get_width()) * 0.5
        self.screen.blit(shadow, (x + 2, y + 2))
        self.screen.blit(rendered, (x, y))


RECORD_SCENARIOS = [
    ("1_hover_slam_100m", {"initial_altitude": 100.0}),
    ("2_2000m_up_30mps", {"initial_altitude": 2_000.0, "vertical_speed": 30.0}),
    ("3_2000m_down_30mps", {"initial_altitude": 2_000.0, "vertical_speed": -30.0}),
    ("4_boostback_x+1000_vx+100", {"initial_altitude": 2_000.0, "horizontal_position": 1_000.0, "horizontal_speed": 100.0}),
    ("5_divert_x-1000_vx+100", {"initial_altitude": 2_000.0, "horizontal_position": -1_000.0, "horizontal_speed": 100.0}),
]


def record_landing_video(
    path: str,
    title: str,
    initial_altitude: float,
    horizontal_position: float = 0.0,
    horizontal_speed: float = 0.0,
    vertical_speed: float = 0.0,
    *,
    fps: int = 30,
    max_seconds: float = 150.0,
) -> dict[str, float | str]:
    """Render one autonomous landing with the game's own renderer to a video.

    The simulation runs at the normal fixed 120 Hz step; every ``120 / fps``
    steps one frame is drawn off-screen.  The camera frames both the vehicle
    and the pad so the whole approach is visible.  Frames go to an MP4 via
    ``imageio-ffmpeg`` when it is installed, otherwise to numbered JPEGs.
    """
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    simulator = RocketSimulator()
    simulator.show_help = False
    simulator.scenario_title = ""
    simulator.autopilot_enabled = True
    rocket = simulator.rocket
    rocket.set_flight_state(initial_altitude, horizontal_position, horizontal_speed)
    rocket.velocity += rocket.local_up() * vertical_speed
    simulator.guidance.reset()
    simulator._reset_effects()
    simulator.trajectory = [rocket.position.copy()]
    width, height = simulator.screen.get_size()

    writer = None
    frame_dir = None
    try:
        import imageio_ffmpeg

        writer = imageio_ffmpeg.write_frames(
            path, (width, height), fps=fps, codec="libx264", quality=7,
            pix_fmt_out="yuv420p", macro_block_size=8,
        )
        writer.send(None)
    except Exception:  # no encoder: fall back to JPEG frames
        writer = None
        frame_dir = os.path.splitext(path)[0] + "_frames"
        os.makedirs(frame_dir, exist_ok=True)

    steps_per_frame = max(1, PHYSICS_HZ // fps)
    frame = 0
    settle_frames = int(2.5 * fps)
    telemetry = []
    total_steps = int(max_seconds * PHYSICS_HZ)
    step = 0
    while step < total_steps and settle_frames > 0:
        for _ in range(steps_per_frame):
            if rocket.state not in {"LANDED", "CRASHED"}:
                simulator._physics_step()
                simulator._sync_predicted_trajectory()
                simulator.trajectory_timer += FIXED_DT
                if simulator.trajectory_timer >= 0.20:
                    simulator.trajectory_timer = 0.0
                    simulator.trajectory.append(rocket.position.copy())
            step += 1
        if rocket.state in {"LANDED", "CRASHED"}:
            settle_frames -= 1
            simulator.predicted_trajectory = []
        simulator._update_particles(1.0 / fps)

        simulator._update_camera_view()  # auto-framing camera (rocket + pad)
        simulator._draw()
        simulator.scenario_title = title
        if writer is not None:
            to_bytes = getattr(pygame.image, "tobytes", None) or pygame.image.tostring
            writer.send(to_bytes(simulator.screen, "RGB"))
        else:
            pygame.image.save(simulator.screen, os.path.join(frame_dir, f"f{frame:05d}.jpg"))
        frame += 1
        telemetry.append(
            f"{step * FIXED_DT:.3f},{simulator.guidance.phase},{rocket.position.x - rocket.launch_x:.3f},"
            f"{rocket.altitude:.3f},{rocket.horizontal_speed:.3f},{rocket.vertical_speed:.3f},"
            f"{rocket.throttle:.3f},{math.degrees(rocket.local_pitch):.2f},{math.degrees(rocket.gimbal):.2f}"
        )
    if writer is not None:
        writer.close()
    with open(os.path.splitext(path)[0] + "_telemetry.csv", "w", encoding="utf-8") as handle:
        handle.write("t,phase,x,h,vx,vz,throttle,pitch_deg,gimbal_deg\n" + "\n".join(telemetry) + "\n")
    pygame.quit()
    return {
        "video": path if frame_dir is None else frame_dir,
        "frames": frame,
        "state": rocket.state,
        "pad_error_m": round(abs(rocket.position.x - rocket.launch_x), 3),
        "backend": simulator.guidance.mpc.backend,
    }


def run_landing_scenario(
    initial_altitude: float = 100.0,
    target_altitude: float | None = None,
    horizontal_position: float = 0.0,
    horizontal_speed: float = 0.0,
    vertical_speed: float = 0.0,
    *,
    max_seconds: float = 180.0,
) -> dict[str, float | str | bool]:
    """Run a deterministic, no-window precision-landing trial.

    Uses exactly the interactive physics/guidance path.  Success means a
    SpaceX-style landing: on the pad centre within 1 m, touchdown sink rate
    at most 2 m/s, sideways drift at most 1 m/s and the body within 5 deg of
    vertical, with every re-plan solved by Clarabel.
    """
    pygame.init()
    rocket = Rocket(random.Random(17))
    rocket.set_flight_state(initial_altitude, horizontal_position, horizontal_speed)
    rocket.velocity += rocket.local_up() * vertical_speed
    del target_altitude  # the pad surface is always the target
    guidance = AutonomousGuidance()
    steps = max(1, math.ceil(max_seconds / FIXED_DT))
    completed_steps = 0
    ignition_time = math.nan
    ignition_altitude = math.nan
    max_tilt = 0.0
    touchdown_pitch = 0.0
    for completed_steps in range(1, steps + 1):
        guidance.command(rocket, FIXED_DT)
        touchdown_pitch = rocket.local_pitch
        rocket.step(FIXED_DT)
        max_tilt = max(max_tilt, abs(rocket.local_pitch))
        if math.isnan(ignition_time) and guidance.burn_locked:
            ignition_time = completed_steps * FIXED_DT
            ignition_altitude = rocket.altitude
        if rocket.state in {"CRASHED", "LANDED"}:
            break
    touchdown_vx, touchdown_vz = guidance.touchdown_velocity
    pad_error = abs(rocket.position.x - rocket.launch_x)
    backend = guidance.mpc.backend
    solved = guidance.last_status in {"Solved", "AlmostSolved"} or guidance.phase in {"TERMINAL", "LANDED"}
    success = (
        rocket.state == "LANDED"
        and solved
        and backend != "feedback-fallback"
        and pad_error <= 1.0
        and abs(touchdown_vz) <= 2.0
        and abs(touchdown_vx) <= 1.0
        and abs(math.degrees(touchdown_pitch)) <= 5.0
    )
    result: dict[str, float | str | bool] = {
        "state": rocket.state,
        "success": success,
        "flight_time_s": round(completed_steps * FIXED_DT, 2),
        "ignition_time_s": round(ignition_time, 2),
        "ignition_altitude_m": round(ignition_altitude, 1),
        "landing_pad_error_m": round(pad_error, 3),
        "touchdown_vertical_speed_mps": round(touchdown_vz, 3),
        "touchdown_horizontal_speed_mps": round(touchdown_vx, 3),
        "touchdown_pitch_deg": round(math.degrees(touchdown_pitch), 2),
        "max_pitch_deg": round(math.degrees(max_tilt), 1),
        "solver": guidance.last_status,
        "solver_backend": backend,
        "native_solver": backend.startswith(NATIVE_BACKEND_PREFIX),
        "solve_count": guidance.mpc.solve_count,
    }
    pygame.quit()
    return result


def run_hover_scenario(*args, **kwargs):
    """Backward-compatible name for callers that used the old helper."""
    return run_landing_scenario(*args, **kwargs)


def main() -> None:
    if "--landing-test" in sys.argv or "--hover-test" in sys.argv or os.environ.get("ROCKET_LANDING_TEST", "").lower() in {"1", "true", "yes"} or os.environ.get("ROCKET_HOVER_TEST", "").lower() in {"1", "true", "yes"}:
        scenarios = [
            {"initial_altitude": 100.0},
            {"initial_altitude": 2_000.0, "vertical_speed": 30.0},
            {"initial_altitude": 2_000.0, "vertical_speed": -30.0},
            {"initial_altitude": 2_000.0, "horizontal_position": 1_000.0, "horizontal_speed": 100.0},
            {"initial_altitude": 2_000.0, "horizontal_position": -1_000.0, "horizontal_speed": 100.0},
        ]
        results = []
        for scenario in scenarios:
            result = run_landing_scenario(**scenario, max_seconds=180.0)
            results.append({"scenario": scenario, **result})
            print(json.dumps(results[-1]), flush=True)
        if results and not results[0]["native_solver"]:
            print("note: native clarabel_hover.dll (ABI 2) not in use; ran on the Python clarabel package. "
                  "Run native\\build_solver.ps1 to rebuild it.", file=sys.stderr)
        if not all(bool(result["success"]) for result in results):
            raise SystemExit(1)
        return
    if "--record" in sys.argv:
        index = sys.argv.index("--record")
        out_dir = sys.argv[index + 1] if len(sys.argv) > index + 1 else "videos"
        os.makedirs(out_dir, exist_ok=True)
        for name, scenario in RECORD_SCENARIOS:
            info = record_landing_video(os.path.join(out_dir, name + ".mp4"), name.replace("_", " "), **scenario)
            print(json.dumps({"scenario": name, **info}), flush=True)
        return
    simulator = RocketSimulator()
    test_frames_text = os.environ.get("ROCKET_SIM_TEST_FRAMES")
    max_frames = int(test_frames_text) if test_frames_text else None
    simulator.run(max_frames=max_frames)


if __name__ == "__main__":
    main()
