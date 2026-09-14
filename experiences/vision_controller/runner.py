import math
import random
import time
from enum import Enum
from typing import List, Tuple, Optional
import pygame
import numpy as np

from core.config import palette, display_config
from core.gestures import GestureType
from experiences.base import BaseExperience

class ObstacleType(Enum):
    HURDLE = "HURDLE"       # Low laser beam (Jump over or dodge)
    OVERHEAD = "OVERHEAD"   # Hanging cyber arch (Slide under or dodge)
    BARRIER = "BARRIER"     # Solid energy monolith (Dodge left or right)
    CORE = "CORE"           # Collectible Aether energy core

class Obstacle:
    def __init__(self, lane: int, z: float, obs_type: ObstacleType):
        self.lane = lane        # 0 (Left), 1 (Center), 2 (Right)
        self.z = z              # Distance along track (1200 down to 0)
        self.obs_type = obs_type
        self.passed = False
        self.collected = False

class Particle:
    def __init__(self, x: float, y: float, vx: float, vy: float, color: Tuple[int, int, int], life: float):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.color = color
        self.life = life
        self.max_life = life

class SoundSynth:
    """Procedurally synthesizes 8-bit / cyber audio using numpy without external files."""
    def __init__(self):
        self.enabled = False
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            self.sample_rate = 44100
            self.sound_jump = self._synth_freq_sweep(320, 780, 0.16)
            self.sound_slide = self._synth_noise_whoosh(0.20)
            self.sound_collect = self._synth_chime(580, 880, 0.18)
            self.sound_hit = self._synth_crash(0.35)
            self.sound_lane = self._synth_freq_sweep(440, 520, 0.08)
            self.enabled = True
        except Exception:
            self.enabled = False

    def _make_stereo(self, mono: np.ndarray) -> pygame.mixer.Sound:
        mono = np.clip(mono, -1.0, 1.0)
        stereo = np.column_stack((mono, mono))
        int16_arr = (stereo * 30000).astype(np.int16)
        return pygame.sndarray.make_sound(int16_arr)

    def _synth_freq_sweep(self, start_f: float, end_f: float, dur: float) -> pygame.mixer.Sound:
        t = np.linspace(0, dur, int(self.sample_rate * dur), endpoint=False)
        freqs = np.linspace(start_f, end_f, len(t))
        phase = 2 * np.pi * np.cumsum(freqs) / self.sample_rate
        env = np.linspace(1.0, 0.0, len(t)) ** 1.5
        wave = np.sin(phase) * env
        return self._make_stereo(wave)

    def _synth_noise_whoosh(self, dur: float) -> pygame.mixer.Sound:
        t = np.linspace(0, dur, int(self.sample_rate * dur), endpoint=False)
        noise = np.random.uniform(-1, 1, len(t))
        env = np.sin(np.pi * np.linspace(0, 1, len(t))) ** 2
        wave = noise * env * 0.45
        return self._make_stereo(wave)

    def _synth_chime(self, f1: float, f2: float, dur: float) -> pygame.mixer.Sound:
        t = np.linspace(0, dur, int(self.sample_rate * dur), endpoint=False)
        env = np.exp(-10 * t)
        wave = (np.sin(2 * np.pi * f1 * t) + 0.6 * np.sin(2 * np.pi * f2 * t)) * env * 0.5
        return self._make_stereo(wave)

    def _synth_crash(self, dur: float) -> pygame.mixer.Sound:
        t = np.linspace(0, dur, int(self.sample_rate * dur), endpoint=False)
        noise = np.random.uniform(-1, 1, len(t))
        low_f = np.sin(2 * np.pi * 90 * np.linspace(1, 0.1, len(t)) * t)
        env = np.linspace(1.0, 0.0, len(t)) ** 0.8
        wave = (noise * 0.6 + low_f * 0.4) * env
        return self._make_stereo(wave)

    def play(self, sound_name: str):
        if not self.enabled:
            return
        try:
            snd = getattr(self, f"sound_{sound_name}", None)
            if snd:
                snd.play()
        except Exception:
            pass

class NeonRunnerExperience(BaseExperience):
    """
    Exhibition Game: AETHER // NEON RUNNER.
    A high-octane, perspective 3D endless arcade runner controlled entirely
    via real-time hand gestures.
    """
    def __init__(self, width: int = None, height: int = None):
        self.width = width or display_config.width
        self.height = height or display_config.height
        
        # Audio Synthesizer
        self.synth = SoundSynth()
        
        # World & Perspective settings
        self.vp_x = self.width // 2
        self.vp_y = int(self.height * 0.38)     # Horizon line
        self.ground_y = self.height - 80        # Ground at player position
        self.track_length = 1200.0              # Z depth of track
        self.lane_offsets = [-260.0, 0.0, 260.0]# World X for Lanes 0, 1, 2
        
        # Grid visual settings
        self.grid_scroll = 0.0
        self.speed = 460.0                      # Pixels/sec along track
        self.base_speed = 460.0
        self.max_speed = 780.0
        
        # Game State
        self.state = "READY"                    # READY, PLAYING, GAMEOVER
        self.score = 0
        self.high_score = 0
        self.distance = 0.0
        self.multiplier = 1
        self.last_state_change = time.time()
        
        # Player Mechanics
        self.target_lane = 1                    # 0, 1, 2
        self.current_lane_pos = 1.0             # Smooth interpolated lane float
        self.jump_t = 0.0                       # 0.0 to 1.0
        self.is_jumping = False
        self.jump_duration = 0.65               # Seconds in air
        self.slide_t = 0.0                      # 0.0 to 1.0
        self.is_sliding = False
        self.slide_duration = 0.55              # Seconds in slide
        
        # Entities & Particles
        self.obstacles: List[Obstacle] = []
        self.particles: List[Particle] = []
        self.spawn_timer = 0.0
        self.spawn_interval = 1.45
        
        # Fonts
        self.font_large = pygame.font.SysFont("Consolas", 42, bold=True)
        self.font_med = pygame.font.SysFont("Consolas", 22, bold=True)
        self.font_stat = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_banner = pygame.font.SysFont("Consolas", 52, bold=True)

    def enter(self):
        self.reset_game()
        self.state = "READY"
        self.last_state_change = time.time()

    def reset_game(self):
        self.target_lane = 1
        self.current_lane_pos = 1.0
        self.jump_t = 0.0
        self.is_jumping = False
        self.slide_t = 0.0
        self.is_sliding = False
        self.score = 0
        self.multiplier = 1
        self.distance = 0.0
        self.speed = self.base_speed
        self.obstacles.clear()
        self.particles.clear()
        self.spawn_timer = 0.0
        self.grid_scroll = 0.0

    def handle_gesture(self, gesture: GestureType, action: GestureType, hand_pos: Tuple[float, float]):
        now = time.time()
        
        # 1. State: READY or GAMEOVER -> Any sustained hand starts / restarts game
        if self.state in ("READY", "GAMEOVER"):
            if now - self.last_state_change > 1.2:
                # Require neutral hand or any clear gesture to launch
                if action != GestureType.NEUTRAL or gesture != GestureType.NEUTRAL:
                    self.reset_game()
                    self.state = "PLAYING"
                    self.last_state_change = now
                    self.synth.play("collect")
            return

        # 2. State: PLAYING -> Respond to action impulses
        if action == GestureType.LEFT:
            if self.target_lane > 0:
                self.target_lane -= 1
                self.synth.play("lane")
                self._spawn_lane_particles(-1)
        elif action == GestureType.RIGHT:
            if self.target_lane < 2:
                self.target_lane += 1
                self.synth.play("lane")
                self._spawn_lane_particles(1)
        elif action == GestureType.JUMP:
            if not self.is_jumping and not self.is_sliding:
                self.is_jumping = True
                self.jump_t = 0.0
                self.synth.play("jump")
                self._spawn_jump_particles()
        elif action == GestureType.SLIDE:
            if not self.is_sliding and not self.is_jumping:
                self.is_sliding = True
                self.slide_t = 0.0
                self.synth.play("slide")
                self._spawn_slide_particles()

    def handle_key(self, event: pygame.event.Event) -> bool:
        now = time.time()
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_SPACE, pygame.K_r):
                if self.state in ("READY", "GAMEOVER"):
                    self.reset_game()
                    self.state = "PLAYING"
                    self.last_state_change = now
                    self.synth.play("collect")
                    return True
                elif self.state == "PLAYING" and event.key == pygame.K_SPACE:
                    if not self.is_jumping and not self.is_sliding:
                        self.is_jumping = True
                        self.jump_t = 0.0
                        self.synth.play("jump")
                        self._spawn_jump_particles()
                        return True
            elif self.state == "PLAYING":
                if event.key in (pygame.K_LEFT, pygame.K_a):
                    if self.target_lane > 0:
                        self.target_lane -= 1
                        self.synth.play("lane")
                    return True
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    if self.target_lane < 2:
                        self.target_lane += 1
                        self.synth.play("lane")
                    return True
                elif event.key in (pygame.K_UP, pygame.K_w):
                    if not self.is_jumping and not self.is_sliding:
                        self.is_jumping = True
                        self.jump_t = 0.0
                        self.synth.play("jump")
                    return True
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    if not self.is_sliding and not self.is_jumping:
                        self.is_sliding = True
                        self.slide_t = 0.0
                        self.synth.play("slide")
                    return True
        return False

    def update(self, dt: float):
        # Update particles regardless of state for fluid visual persistence
        self._update_particles(dt)

        if self.state != "PLAYING":
            return

        # Advance Track Distance & Score
        self.distance += self.speed * dt
        self.score += int(self.speed * dt * 0.1 * self.multiplier)
        if self.score > self.high_score:
            self.high_score = self.score

        # Progressively increase speed gently
        self.speed = min(self.max_speed, self.base_speed + (self.distance * 0.025))

        # Grid scroll animation
        self.grid_scroll = (self.grid_scroll + self.speed * dt) % 100.0

        # Smooth Player Lane Transition (Lerp for responsive yet silky movement)
        lane_diff = self.target_lane - self.current_lane_pos
        self.current_lane_pos += lane_diff * min(1.0, dt * 14.0)

        # Jump Animation Cycle
        if self.is_jumping:
            self.jump_t += dt / self.jump_duration
            if self.jump_t >= 1.0:
                self.is_jumping = False
                self.jump_t = 0.0

        # Slide Animation Cycle
        if self.is_sliding:
            self.slide_t += dt / self.slide_duration
            if self.slide_t >= 1.0:
                self.is_sliding = False
                self.slide_t = 0.0

        # Spawn obstacles and power cores
        self.spawn_timer += dt
        dynamic_interval = max(0.9, self.spawn_interval - (self.distance * 0.00008))
        if self.spawn_timer >= dynamic_interval:
            self.spawn_timer = 0.0
            self._spawn_wave()

        # Update and collide obstacles
        player_z = 80.0
        collision_depth = 45.0
        
        for obs in self.obstacles:
            obs.z -= self.speed * dt

            # Collision Detection Window
            if not obs.passed and abs(obs.z - player_z) < collision_depth:
                # Check lane match (within 0.45 lane tolerance)
                if abs(self.current_lane_pos - obs.lane) < 0.48:
                    if obs.obs_type == ObstacleType.CORE:
                        if not obs.collected:
                            obs.collected = True
                            self.score += 250 * self.multiplier
                            self.multiplier = min(4, self.multiplier + 1)
                            self.synth.play("collect")
                            self._spawn_collect_burst(obs.lane)
                    elif obs.obs_type == ObstacleType.HURDLE:
                        # Must be JUMPING high enough
                        jump_height_ratio = math.sin(self.jump_t * math.pi) if self.is_jumping else 0.0
                        if jump_height_ratio < 0.52:
                            self._trigger_game_over()
                    elif obs.obs_type == ObstacleType.OVERHEAD:
                        # Must be SLIDING low enough
                        if not self.is_sliding:
                            self._trigger_game_over()
                    elif obs.obs_type == ObstacleType.BARRIER:
                        # Cannot jump or slide through solid barrier!
                        self._trigger_game_over()

            # Mark as passed once beyond player
            if obs.z < player_z - collision_depth:
                obs.passed = True

        # Cull offscreen obstacles
        self.obstacles = [o for o in self.obstacles if o.z > -100.0]

        # Regular thruster particle emission
        self._emit_thruster_particles()

    def _spawn_wave(self):
        """Generates engaging obstacle patterns (always leaving at least one valid path)."""
        pattern_choice = random.random()
        
        if pattern_choice < 0.40:
            # Low Hurdle in one lane, Core in another
            lane_h = random.choice([0, 1, 2])
            self.obstacles.append(Obstacle(lane_h, self.track_length, ObstacleType.HURDLE))
            free_lanes = [l for l in [0, 1, 2] if l != lane_h]
            if random.random() < 0.65:
                self.obstacles.append(Obstacle(random.choice(free_lanes), self.track_length, ObstacleType.CORE))
        elif pattern_choice < 0.70:
            # Overhead Plasma Arch
            lane_o = random.choice([0, 1, 2])
            self.obstacles.append(Obstacle(lane_o, self.track_length, ObstacleType.OVERHEAD))
            free_lanes = [l for l in [0, 1, 2] if l != lane_o]
            if random.random() < 0.6:
                self.obstacles.append(Obstacle(random.choice(free_lanes), self.track_length, ObstacleType.CORE))
        elif pattern_choice < 0.90:
            # Solid Barrier in one lane
            lane_b = random.choice([0, 1, 2])
            self.obstacles.append(Obstacle(lane_b, self.track_length, ObstacleType.BARRIER))
        else:
            # High-reward Core lane
            lane_c = random.choice([0, 1, 2])
            self.obstacles.append(Obstacle(lane_c, self.track_length, ObstacleType.CORE))

    def _trigger_game_over(self):
        self.state = "GAMEOVER"
        self.last_state_change = time.time()
        self.multiplier = 1
        self.synth.play("hit")
        self._spawn_crash_burst()

    def _project(self, world_x: float, world_y: float, z: float) -> Tuple[int, int, float]:
        """Projects 3D world coordinate (X, Y, Z) to 2D screen coordinates."""
        scale = 320.0 / (320.0 + max(0.1, z))
        screen_x = self.vp_x + int(world_x * scale)
        screen_y = self.vp_y + int((world_y - self.vp_y) * scale)
        return screen_x, screen_y, scale

    def _emit_thruster_particles(self):
        player_x = self.lane_offsets[1] + (self.current_lane_pos - 1.0) * (self.lane_offsets[2] - self.lane_offsets[1])
        jump_h = math.sin(self.jump_t * math.pi) * 110.0 if self.is_jumping else 0.0
        py_world = self.ground_y - jump_h
        
        px, py, _ = self._project(player_x, py_world, 80.0)
        for _ in range(2):
            self.particles.append(Particle(
                x=px + random.uniform(-6, 6),
                y=py + 10,
                vx=random.uniform(-15, 15),
                vy=random.uniform(40, 120),
                color=palette.CYAN_NEON if random.random() < 0.7 else palette.MAGENTA_LASER,
                life=random.uniform(0.2, 0.45)
            ))

    def _spawn_lane_particles(self, dir_sign: int):
        px, py, _ = self._get_player_screen_pos()
        for _ in range(12):
            self.particles.append(Particle(
                x=px,
                y=py,
                vx=-dir_sign * random.uniform(80, 220),
                vy=random.uniform(-40, 40),
                color=palette.CYAN_NEON,
                life=random.uniform(0.25, 0.5)
            ))

    def _spawn_jump_particles(self):
        px, py, _ = self._get_player_screen_pos()
        for _ in range(16):
            self.particles.append(Particle(
                x=px + random.uniform(-20, 20),
                y=py + 15,
                vx=random.uniform(-60, 60),
                vy=random.uniform(30, 90),
                color=palette.GOLD_ACCENT,
                life=random.uniform(0.3, 0.6)
            ))

    def _spawn_slide_particles(self):
        px, py, _ = self._get_player_screen_pos()
        for _ in range(14):
            self.particles.append(Particle(
                x=px + random.uniform(-25, 25),
                y=py + 8,
                vx=random.uniform(-110, 110),
                vy=random.uniform(-20, -70),
                color=palette.MAGENTA_LASER,
                life=random.uniform(0.25, 0.5)
            ))

    def _spawn_collect_burst(self, lane: int):
        wx = self.lane_offsets[lane]
        px, py, _ = self._project(wx, self.ground_y - 25, 80.0)
        for _ in range(22):
            ang = random.uniform(0, 2 * math.pi)
            spd = random.uniform(80, 260)
            self.particles.append(Particle(
                x=px,
                y=py,
                vx=math.cos(ang) * spd,
                vy=math.sin(ang) * spd,
                color=palette.GOLD_ACCENT if random.random() < 0.6 else palette.GREEN_MATRIX,
                life=random.uniform(0.35, 0.7)
            ))

    def _spawn_crash_burst(self):
        px, py, _ = self._get_player_screen_pos()
        for _ in range(45):
            ang = random.uniform(0, 2 * math.pi)
            spd = random.uniform(90, 380)
            self.particles.append(Particle(
                x=px,
                y=py,
                vx=math.cos(ang) * spd,
                vy=math.sin(ang) * spd,
                color=palette.MAGENTA_LASER if random.random() < 0.5 else palette.GOLD_ACCENT,
                life=random.uniform(0.4, 0.9)
            ))

    def _update_particles(self, dt: float):
        survivors = []
        for p in self.particles:
            p.life -= dt
            p.x += p.vx * dt
            p.y += p.vy * dt
            if p.life > 0:
                survivors.append(p)
        self.particles = survivors

    def _get_player_screen_pos(self) -> Tuple[int, int, float]:
        player_x = self.lane_offsets[1] + (self.current_lane_pos - 1.0) * (self.lane_offsets[2] - self.lane_offsets[1])
        jump_h = math.sin(self.jump_t * math.pi) * 110.0 if self.is_jumping else 0.0
        py_world = self.ground_y - jump_h
        return self._project(player_x, py_world, 80.0)

    # -------------------------------------------------------------
    # RENDERING PIPELINE
    # -------------------------------------------------------------
    def render(self, surface: pygame.Surface):
        # 1. Background Neon Horizon & Grid
        self._render_background_grid(surface)

        # 2. Obstacles and Cores
        self._render_obstacles(surface)

        # 3. Player Hovercraft
        self._render_player(surface)

        # 4. Active Particle Systems
        self._render_particles(surface)

        # 5. In-Game Exhibition Score & Combo HUD
        self._render_hud_overlay(surface)

        # 6. State Overlays (READY / GAMEOVER)
        if self.state == "READY":
            self._render_ready_overlay(surface)
        elif self.state == "GAMEOVER":
            self._render_gameover_overlay(surface)

    def _render_background_grid(self, surface: pygame.Surface):
        # Deep space gradient above horizon
        horizon_y = self.vp_y
        sky_rect = pygame.Rect(0, 0, self.width, horizon_y)
        pygame.draw.rect(surface, (8, 12, 20), sky_rect)

        # Distant Cyberpunk Sun / Horizon Glow
        sun_surf = pygame.Surface((320, 160), pygame.SRCALPHA)
        pygame.draw.circle(sun_surf, (255, 0, 85, 45), (160, 160), 150)
        pygame.draw.circle(sun_surf, (255, 215, 0, 75), (160, 160), 100)
        pygame.draw.circle(sun_surf, (255, 255, 255, 120), (160, 160), 50)
        surface.blit(sun_surf, (self.vp_x - 160, horizon_y - 150))

        # Ground Track Floor
        floor_rect = pygame.Rect(0, horizon_y, self.width, self.height - horizon_y)
        pygame.draw.rect(surface, (12, 16, 26), floor_rect)

        # Draw Perspective Track Borders and Lane Separators
        lane_boundaries = [-390.0, -130.0, 130.0, 390.0]
        for i, bx in enumerate(lane_boundaries):
            p_far_x, p_far_y, _ = self._project(bx, self.ground_y, self.track_length)
            p_near_x, p_near_y, _ = self._project(bx, self.ground_y, 0.0)
            color = palette.CYAN_NEON if i in (0, 3) else (40, 60, 90)
            thickness = 2 if i in (0, 3) else 1
            pygame.draw.line(surface, color, (p_far_x, p_far_y), (p_near_x, p_near_y), thickness)

        # Draw Moving Horizontal Perspective Grid Lines
        num_grid_lines = 16
        for i in range(num_grid_lines):
            # Compute distance with scroll offset
            z_pos = ((i * (self.track_length / num_grid_lines)) - self.grid_scroll) % self.track_length
            p_left_x, p_left_y, _ = self._project(lane_boundaries[0], self.ground_y, z_pos)
            p_right_x, p_right_y, _ = self._project(lane_boundaries[-1], self.ground_y, z_pos)
            
            # Alpha/Intensity fades with distance
            alpha = int(255 * (1.0 - (z_pos / self.track_length)))
            line_color = (0, min(240, int(alpha * 0.9)), min(255, alpha))
            pygame.draw.line(surface, line_color, (p_left_x, p_left_y), (p_right_x, p_right_y), 1)

    def _render_obstacles(self, surface: pygame.Surface):
        # Sort back-to-front for proper depth occlusion
        sorted_obs = sorted(self.obstacles, key=lambda o: o.z, reverse=True)

        for obs in sorted_obs:
            if obs.passed or obs.collected:
                continue

            wx = self.lane_offsets[obs.lane]
            sx, sy, scale = self._project(wx, self.ground_y, obs.z)
            if scale <= 0:
                continue

            width = int(170 * scale)
            height = int(95 * scale)

            if obs.obs_type == ObstacleType.CORE:
                # Floating glowing diamond crystal
                core_y = sy - int(45 * scale)
                r = max(4, int(22 * scale))
                pygame.draw.circle(surface, palette.GOLD_ACCENT, (sx, core_y), r)
                pygame.draw.circle(surface, palette.TEXT_WHITE, (sx, core_y), max(2, r // 2))
                # Diamond outline
                pts = [(sx, core_y - r - 4), (sx + r + 4, core_y), (sx, core_y + r + 4), (sx - r - 4, core_y)]
                pygame.draw.polygon(surface, palette.GREEN_MATRIX, pts, 2)

            elif obs.obs_type == ObstacleType.HURDLE:
                # Low ground laser hurdle (must jump over)
                h_top = sy - int(38 * scale)
                # Left/Right pylons
                half_w = width // 2
                pygame.draw.rect(surface, (180, 20, 50), (sx - half_w, h_top, int(10 * scale), sy - h_top))
                pygame.draw.rect(surface, (180, 20, 50), (sx + half_w - int(10 * scale), h_top, int(10 * scale), sy - h_top))
                # Glowing laser beam
                pygame.draw.line(surface, palette.MAGENTA_LASER, (sx - half_w, h_top + 4), (sx + half_w, h_top + 4), max(2, int(6 * scale)))
                pygame.draw.line(surface, palette.TEXT_WHITE, (sx - half_w, h_top + 4), (sx + half_w, h_top + 4), max(1, int(2 * scale)))

            elif obs.obs_type == ObstacleType.OVERHEAD:
                # High hanging arch (must slide under)
                arch_top = sy - int(130 * scale)
                arch_bottom = sy - int(55 * scale)
                half_w = width // 2
                # High gate bar
                pygame.draw.rect(surface, palette.CYAN_NEON, (sx - half_w, arch_top, width, int(18 * scale)))
                # Energy hazard stripes
                pygame.draw.line(surface, palette.GOLD_ACCENT, (sx - half_w, arch_bottom), (sx + half_w, arch_bottom), max(2, int(4 * scale)))
                # Arch legs extending up into sky
                pygame.draw.line(surface, palette.BORDER_DIM, (sx - half_w, arch_top), (sx - half_w, sy), max(1, int(3 * scale)))
                pygame.draw.line(surface, palette.BORDER_DIM, (sx + half_w, arch_top), (sx + half_w, sy), max(1, int(3 * scale)))

            elif obs.obs_type == ObstacleType.BARRIER:
                # Solid monolithic barrier (must steer away)
                half_w = width // 2
                top_y = sy - int(120 * scale)
                poly = [
                    (sx - half_w, sy),
                    (sx - half_w, top_y),
                    (sx + half_w, top_y),
                    (sx + half_w, sy)
                ]
                pygame.draw.polygon(surface, (25, 35, 55), poly)
                pygame.draw.polygon(surface, palette.MAGENTA_LASER, poly, 2)
                # Neon core in center
                pygame.draw.line(surface, palette.MAGENTA_LASER, (sx, sy), (sx, top_y), max(1, int(3 * scale)))

    def _render_player(self, surface: pygame.Surface):
        px, py, scale = self._get_player_screen_pos()
        
        # Ground Shadow (projects to ground_y when jumping)
        player_x = self.lane_offsets[1] + (self.current_lane_pos - 1.0) * (self.lane_offsets[2] - self.lane_offsets[1])
        gx, gy, gscale = self._project(player_x, self.ground_y, 80.0)
        
        shadow_w = int(75 * gscale)
        shadow_h = int(20 * gscale)
        if self.is_jumping:
            # Shadow shrinks and fades when high
            h_ratio = math.sin(self.jump_t * math.pi)
            shadow_w = int(shadow_w * (1.0 - h_ratio * 0.45))
            shadow_h = int(shadow_h * (1.0 - h_ratio * 0.45))
            
        shadow_surf = pygame.Surface((shadow_w * 2, shadow_h * 2), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surf, (0, 0, 0, 140), (0, 0, shadow_w * 2, shadow_h * 2))
        surface.blit(shadow_surf, (gx - shadow_w, gy - shadow_h // 2))

        # Player Hovercraft Shape
        pw = int(72 * scale)
        ph = int(36 * scale)

        # Slide transformation: crouches and widens
        if self.is_sliding:
            ph = int(ph * 0.45)
            pw = int(pw * 1.25)

        # Hovercraft Sci-Fi Polygon Chassis
        nose = (px, py - ph)
        wing_l = (px - pw, py + ph // 2)
        wing_r = (px + pw, py + ph // 2)
        tail_l = (px - pw // 2, py + ph)
        tail_r = (px + pw // 2, py + ph)
        core_pos = (px, py + ph // 4)

        chassis_poly = [nose, wing_r, tail_r, tail_l, wing_l]
        pygame.draw.polygon(surface, palette.PANEL_BG, chassis_poly)
        pygame.draw.polygon(surface, palette.CYAN_NEON, chassis_poly, 2)

        # Glowing Cockpit & Core
        pygame.draw.circle(surface, palette.CYAN_NEON, core_pos, max(3, int(6 * scale)))
        pygame.draw.circle(surface, palette.TEXT_WHITE, core_pos, max(1, int(3 * scale)))

        # Wingtip Beacons
        pygame.draw.circle(surface, palette.MAGENTA_LASER, wing_l, max(2, int(4 * scale)))
        pygame.draw.circle(surface, palette.MAGENTA_LASER, wing_r, max(2, int(4 * scale)))

    def _render_particles(self, surface: pygame.Surface):
        for p in self.particles:
            alpha_ratio = p.life / p.max_life
            r = max(1, int(3.5 * alpha_ratio))
            pygame.draw.circle(surface, p.color, (int(p.x), int(p.y)), r)

    def _render_hud_overlay(self, surface: pygame.Surface):
        # Score & Distance Telemetry in Upper Right
        score_surf = self.font_large.render(f"{self.score:06d}", True, palette.TEXT_WHITE)
        score_lbl = self.font_stat.render("DISTANCE SCORE", True, palette.TEXT_MUTED)
        
        surface.blit(score_lbl, (self.width - 240, 68))
        surface.blit(score_surf, (self.width - 240, 88))

        # Multiplier Badge
        mult_color = palette.GOLD_ACCENT if self.multiplier > 1 else palette.TEXT_MUTED
        mult_surf = self.font_med.render(f"x{self.multiplier} COMBO", True, mult_color)
        surface.blit(mult_surf, (self.width - 240, 138))

        # Speed Gauge
        spd_kmh = int(self.speed * 0.4)
        spd_surf = self.font_stat.render(f"VELOCITY: {spd_kmh} KM/H", True, palette.CYAN_NEON)
        surface.blit(spd_surf, (24, 68))

    def _render_ready_overlay(self, surface: pygame.Surface):
        dim_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        dim_surf.fill((10, 14, 23, 175))
        surface.blit(dim_surf, (0, 0))

        title = self.font_banner.render("VISION CONTROLLER", True, palette.CYAN_NEON)
        sub = self.font_med.render("AUTONOMOUS COMPUTER VISION RUNNER", True, palette.TEXT_WHITE)
        prompt = self.font_large.render("RAISE HAND TO ENGAGE", True, palette.GOLD_ACCENT)
        hint = self.font_stat.render("[ Hand Left / Right to Steer  |  Swipe Up to Jump  |  Swipe Down to Slide ]", True, palette.TEXT_MUTED)

        cx = self.width // 2
        cy = self.height // 2

        surface.blit(title, (cx - title.get_width() // 2, cy - 120))
        surface.blit(sub, (cx - sub.get_width() // 2, cy - 65))
        surface.blit(prompt, (cx - prompt.get_width() // 2, cy + 20))
        surface.blit(hint, (cx - hint.get_width() // 2, cy + 90))

    def _render_gameover_overlay(self, surface: pygame.Surface):
        dim_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        dim_surf.fill((18, 10, 14, 195))
        surface.blit(dim_surf, (0, 0))

        title = self.font_banner.render("SYSTEM IMPACT DETECTED", True, palette.MAGENTA_LASER)
        score_txt = self.font_large.render(f"FINAL SCORE: {self.score}", True, palette.TEXT_WHITE)
        high_txt = self.font_med.render(f"HIGH SCORE: {self.high_score}", True, palette.GOLD_ACCENT)
        prompt = self.font_large.render("HOLD NEUTRAL TO RESTART", True, palette.CYAN_NEON)
        menu_hint = self.font_stat.render("Press ESC to return to AETHER Master Menu", True, palette.TEXT_MUTED)

        cx = self.width // 2
        cy = self.height // 2

        surface.blit(title, (cx - title.get_width() // 2, cy - 130))
        surface.blit(score_txt, (cx - score_txt.get_width() // 2, cy - 60))
        surface.blit(high_txt, (cx - high_txt.get_width() // 2, cy - 15))
        surface.blit(prompt, (cx - prompt.get_width() // 2, cy + 50))
        surface.blit(menu_hint, (cx - menu_hint.get_width() // 2, cy + 120))

    def exit(self):
        self.obstacles.clear()
        self.particles.clear()
