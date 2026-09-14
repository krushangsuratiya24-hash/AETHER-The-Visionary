import sys
import time
import threading
import os
import pygame
import cv2
import numpy as np

from core.config import display_config, palette, camera_config
from core.camera import ThreadedCamera
from core.tracker import HandTracker
from core.gestures import GestureProcessor, GestureType
from core.ui import HUD
from experiences.vision_controller.runner import NeonRunnerExperience
from experiences.elemental_cultivation import ElementalCultivationExperience
from experiences.phase_shift import PhaseShiftExperience


class AetherApp:
    """
    AETHER — THE VISIONARY: Master Exhibition Application.
    Orchestrates camera acquisition, MediaPipe landmark tracking, gesture recognition,
    and modular experience switching.

    Startup order (critical for Windows window visibility):
      1. Pygame display init → window appears immediately
      2. Splash screen rendered & flipped → OS registers visible window
      3. Camera + MediaPipe init runs in background thread
      4. Main loop pumps events throughout → window stays responsive
      5. When background init finishes → hub is shown
    """

    def __init__(self):
        # ── Step 1: SDL / Pygame Display — MUST happen first ─────────────────
        #
        # Remove any leftover headless/dummy SDL overrides that tests may have set.
        # On a real desktop run this is usually a no-op, but guards against
        # pytest pollution leaking into a direct `python main.py` invocation.
        for _k in ('SDL_VIDEODRIVER', 'SDL_WINDOWID'):
            if os.environ.get(_k) in (None, ''):
                continue  # not set — fine
            if os.environ[_k] == 'dummy':
                del os.environ[_k]
                print(f'[AETHER-DISPLAY] Removed headless env override: {_k}=dummy')

        # Center window on primary monitor
        os.environ['SDL_VIDEO_CENTERED'] = '1'

        print('[AETHER-DISPLAY] Initializing pygame...')
        pygame.init()

        if not pygame.display.get_init():
            pygame.display.init()

        print(f'[AETHER-DISPLAY] Display initialized: {pygame.display.get_init()}')

        try:
            driver = pygame.display.get_driver()
            print(f'[AETHER-DISPLAY] SDL video driver: {driver}')
        except Exception:
            driver = 'unknown'

        try:
            di = pygame.display.Info()
            print(f'[AETHER-DISPLAY] Desktop size: {di.current_w}x{di.current_h}')
        except Exception:
            pass

        # ── Step 2: Create the window ─────────────────────────────────────────
        #
        # Use plain windowed mode — no FULLSCREEN, no DOUBLEBUF, no OPENGL.
        # These flags can cause the window to be invisible on some Windows/SDL
        # configurations.  1280×720 windowed is maximally compatible.
        print(f'[AETHER-DISPLAY] Creating {display_config.width}x{display_config.height} window...')

        flags = 0
        if display_config.fullscreen:
            flags |= pygame.FULLSCREEN

        self.screen = pygame.display.set_mode(
            (display_config.width, display_config.height), flags
        )

        if self.screen is None:
            raise RuntimeError('[AETHER-DISPLAY] FATAL: pygame.display.set_mode() returned None.')

        print(f'[AETHER-DISPLAY] Surface created: {self.screen.get_size()}')

        pygame.display.set_caption(display_config.window_title)

        # ── Step 3: Paint the FIRST frame immediately ─────────────────────────
        #
        # This is the critical fix: we must call display.flip() *and* pump
        # events before doing ANY blocking I/O (camera, model download).
        # If the main thread blocks for >5 s without pumping events the OS
        # marks the window "Not Responding" and may never paint it.
        self.screen.fill(palette.VOID_DARK)
        pygame.display.flip()
        pygame.event.pump()

        print('[AETHER-DISPLAY] First frame presented (window should be visible now).')

        # ── Step 4: Win32 — bring window to foreground ────────────────────────
        self._ensure_window_foreground()

        # ── Step 5: Render proper splash screen ───────────────────────────────
        self._render_splash_screen('INITIALIZING VISION SYSTEM...')

        # ── Step 6: Pygame clock & application state ──────────────────────────
        self.clock = pygame.time.Clock()
        self.running = True

        # App state: 'LOADING' | 'MENU' | 'VISION_CONTROLLER' | 'ELEMENTAL_CULTIVATION' | 'PHASE_SHIFT'
        self.mode = 'LOADING'
        self.debug_mode = False

        # Performance telemetry
        self.fps = 0.0
        self.latency_ms = 0.0

        # UI Assets & Fonts (lightweight — must load before loop starts)
        self.font_hero       = pygame.font.SysFont('Consolas', 54, bold=True)
        self.font_hero_sub   = pygame.font.SysFont('Consolas', 20)
        self.font_tagline    = pygame.font.SysFont('Consolas', 17, italic=True)
        self.font_card_title = pygame.font.SysFont('Consolas', 22, bold=True)
        self.font_card_desc  = pygame.font.SysFont('Consolas', 14)
        self.font_badge      = pygame.font.SysFont('Consolas', 12, bold=True)
        self.font_instr      = pygame.font.SysFont('Consolas', 15)

        # Subsystem references (populated by background init thread)
        self.camera: 'ThreadedCamera | None' = None
        self.tracker: 'HandTracker | None' = None
        self.gestures: 'GestureProcessor | None' = None
        self.hud: 'HUD | None' = None
        self.runner_experience: 'NeonRunnerExperience | None' = None
        self.elemental_experience: 'ElementalCultivationExperience | None' = None
        self.phase_shift_experience: 'PhaseShiftExperience | None' = None

        self._init_error: str | None = None
        self._init_status: str = 'Starting...'
        self._init_done = threading.Event()

        # ── Step 7: Start background initialisation ───────────────────────────
        print('[AETHER] Spawning background subsystem initialisation thread...')
        self._init_thread = threading.Thread(
            target=self._background_init,
            daemon=True,
            name='AetherSubsystemInit'
        )
        self._init_thread.start()

    # ─────────────────────────────────────────────────────────────────────────
    # Background Initialisation  (runs in a worker thread, NOT the main thread)
    # ─────────────────────────────────────────────────────────────────────────

    def _background_init(self):
        """
        Initialises camera + MediaPipe in a background thread so the pygame
        event loop can run continuously on the main thread, keeping the window
        alive and responsive throughout.
        """
        try:
            self._init_status = 'Opening camera...'
            print('[AETHER] Initializing Camera Pipeline...')
            cam = ThreadedCamera().start()
            self.camera = cam

            self._init_status = 'Loading MediaPipe model...'
            print('[AETHER] Initializing MediaPipe Hand Tracker...')
            tracker = HandTracker()
            gestures = GestureProcessor()
            self.tracker = tracker
            self.gestures = gestures

            self._init_status = 'Building experiences...'
            runner = NeonRunnerExperience(display_config.width, display_config.height)
            self.runner_experience = runner

            elemental = ElementalCultivationExperience(display_config.width, display_config.height)
            self.elemental_experience = elemental

            self._init_status = 'Loading Phase Shift...'
            phase_shift = PhaseShiftExperience(display_config.width, display_config.height)
            self.phase_shift_experience = phase_shift

            # HUD must be created here but uses existing screen — that is fine
            # because we only blit to the screen from the main thread.
            self.hud = HUD(self.screen)

            self._init_status = 'Ready.'
            print('[AETHER] Subsystem initialisation complete.')

        except Exception as exc:
            self._init_error = str(exc)
            self._init_status = f'ERROR: {exc}'
            print(f'[AETHER] Subsystem initialisation error: {exc}')
        finally:
            self._init_done.set()

    # ─────────────────────────────────────────────────────────────────────────
    # Window Utilities
    # ─────────────────────────────────────────────────────────────────────────

    def _ensure_window_foreground(self):
        """Ensures the Pygame window is visible, restored, and foregrounded (Windows)."""
        if sys.platform == 'win32':
            try:
                import ctypes
                wm_info = pygame.display.get_wm_info()
                hwnd = wm_info.get('window') if wm_info else None
                if hwnd:
                    user32 = ctypes.windll.user32
                    user32.ShowWindow(hwnd, 9)   # SW_RESTORE
                    user32.ShowWindow(hwnd, 5)   # SW_SHOW
                    user32.BringWindowToTop(hwnd)
                    user32.SetForegroundWindow(hwnd)
                    print(f'[AETHER-DISPLAY] Win32 HWND: {hwnd:#010x} — window foregrounded.')
            except Exception as e:
                print(f'[AETHER-DISPLAY] Window foreground notice: {e}')

    def _render_splash_screen(self, message: str):
        """Renders a loading splash and immediately flips + pumps events."""
        self.screen.fill(palette.VOID_DARK)
        try:
            splash_font = pygame.font.SysFont('Consolas', 42, bold=True)
            title_font  = pygame.font.SysFont('Consolas', 20)
            t1 = splash_font.render('AETHER', True, palette.CYAN_NEON)
            t2 = title_font.render('THE VISIONARY', True, palette.TEXT_WHITE)
            t3 = pygame.font.SysFont('Consolas', 18).render(message, True, palette.GOLD_ACCENT)

            cx = display_config.width  // 2
            cy = display_config.height // 2
            self.screen.blit(t1, (cx - t1.get_width() // 2, cy - 80))
            self.screen.blit(t2, (cx - t2.get_width() // 2, cy - 20))
            self.screen.blit(t3, (cx - t3.get_width() // 2, cy + 30))
        except Exception:
            pass
        pygame.display.flip()
        pygame.event.pump()
        print('[AETHER-DISPLAY] Splash screen rendered.')

    # ─────────────────────────────────────────────────────────────────────────
    # Main Loop
    # ─────────────────────────────────────────────────────────────────────────

    def run(self):
        print('[AETHER] Entering Main Exhibition Loop.')
        last_time = time.time()

        try:
            while self.running:
                now = time.time()
                dt = min(0.1, now - last_time)
                last_time = now

                # ── 1. Handle Window Events (ALWAYS — keeps OS happy) ─────────
                self._handle_events()

                # ── 2. Loading phase ──────────────────────────────────────────
                if self.mode == 'LOADING':
                    self._render_loading_screen()
                    # Transition to MENU when background init finishes
                    if self._init_done.is_set():
                        if self._init_error:
                            # Render error and keep running (camera may be mock)
                            print(f'[AETHER] Init error (non-fatal): {self._init_error}')
                        self.mode = 'MENU'
                        print('[AETHER-DISPLAY] Init complete. Showing Master Hub.')
                    pygame.display.flip()
                    self.clock.tick(display_config.target_fps)
                    self.fps = self.clock.get_fps()
                    continue

                # ── 3. Camera Frame Acquisition & Gesture Tracking ────────────
                t_start = time.time()
                ret, frame = (False, None)
                hands = []
                current_gesture = GestureType.NEUTRAL
                active_action   = GestureType.NEUTRAL
                norm_pos        = (0.5, 0.5)

                if self.camera is not None:
                    ret, frame = self.camera.read()

                if ret and frame is not None and self.tracker is not None:
                    hands = self.tracker.process_frame(frame)
                    if hands:
                        self.tracker.draw_landmarks(frame, hands)
                        current_gesture, active_action, norm_pos = self.gestures.process(hands[0])
                    else:
                        current_gesture, active_action, norm_pos = self.gestures.process(None)
                elif self.gestures is not None:
                    current_gesture, active_action, norm_pos = self.gestures.process(None)

                self.latency_ms = (time.time() - t_start) * 1000.0

                # ── 4. Update Active Experience ───────────────────────────────
                if self.mode == 'VISION_CONTROLLER' and self.runner_experience is not None:
                    self.runner_experience.handle_gesture(current_gesture, active_action, norm_pos)
                    self.runner_experience.update(dt)
                elif self.mode == 'ELEMENTAL_CULTIVATION' and self.elemental_experience is not None:
                    # Pass full landmark data to elemental experience for rich gesture detection
                    self.elemental_experience.handle_hands(hands)
                    self.elemental_experience.handle_gesture(current_gesture, active_action, norm_pos)
                    self.elemental_experience.update(dt)
                elif self.mode == 'PHASE_SHIFT' and self.phase_shift_experience is not None:
                    # Phase Shift: push hand landmarks + camera frame, then update
                    self.phase_shift_experience.push_hands(hands)
                    if ret and frame is not None:
                        self.phase_shift_experience.push_camera_frame(frame)
                    self.phase_shift_experience.update(dt)

                # ── 5. Render Pipeline ────────────────────────────────────────
                # Phase Shift renders the full composited frame itself — no fill needed
                if self.mode != 'PHASE_SHIFT':
                    self.screen.fill(palette.VOID_DARK)

                if self.mode == 'MENU':
                    self._render_master_menu(norm_pos, hands)
                elif self.mode == 'VISION_CONTROLLER' and self.runner_experience is not None:
                    self.runner_experience.render(self.screen)
                elif self.mode == 'ELEMENTAL_CULTIVATION' and self.elemental_experience is not None:
                    self.elemental_experience.render(self.screen)
                elif self.mode == 'PHASE_SHIFT' and self.phase_shift_experience is not None:
                    self.phase_shift_experience.render(self.screen)

                # ── 6. Composite HUD Overlays ─────────────────────────────────
                # Phase Shift draws its own HUD; skip global HUD for that mode
                if self.hud is not None and self.mode != 'PHASE_SHIFT':
                    is_mock = self.camera.is_mock if self.camera else True
                    mode_labels = {
                        'MENU': 'MAIN HUB',
                        'VISION_CONTROLLER': 'PHASE 1 // VISION CONTROLLER',
                        'ELEMENTAL_CULTIVATION': 'PHASE 2 // ELEMENTAL CULTIVATION',
                    }
                    self.hud.draw_top_bar(mode_labels.get(self.mode, self.mode), self.fps, is_mock)

                    if self.mode == 'VISION_CONTROLLER':
                        self.hud.draw_control_guides()
                        self.hud.draw_hand_reticle(norm_pos, len(hands) > 0)
                        self.hud.draw_gesture_card(current_gesture, active_action)

                # ── 7. Camera PIP ─────────────────────────────────────────────
                # Phase Shift uses the full-frame composited view — no PIP needed
                if frame is not None and self.mode not in ('PHASE_SHIFT',):
                    self._render_camera_pip(frame)

                # ── 8. Debug Telemetry ────────────────────────────────────────
                if self.debug_mode and self.hud is not None:
                    debug_info = {
                        'Mode':              self.mode,
                        'FPS':               f'{int(self.fps)}',
                        'Inference Latency': f'{self.latency_ms:.1f} ms',
                        'Camera Source':     'MOCK' if (self.camera and self.camera.is_mock) else 'USB WEBCAM',
                        'Hands Tracked':     len(hands),
                        'Gesture State':     current_gesture.value,
                        'Triggered Action':  active_action.value,
                        'Hand Pos (X, Y)':   f'({norm_pos[0]:.2f}, {norm_pos[1]:.2f})',
                        'Confidence':        f'{self.gestures.hand_confidence:.2f}' if self.gestures else 'N/A',
                    }
                    self.hud.draw_debug_panel(debug_info)

                # ── 9. Present Frame ──────────────────────────────────────────
                pygame.display.flip()
                self.clock.tick(display_config.target_fps)
                self.fps = self.clock.get_fps()

        finally:
            self._cleanup()

    # ─────────────────────────────────────────────────────────────────────────
    # Event Handling
    # ─────────────────────────────────────────────────────────────────────────

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

            if event.type == pygame.KEYDOWN:
                # Global Debug Toggle
                if event.key == pygame.K_d:
                    self.debug_mode = not self.debug_mode
                    continue

                # ESC Navigation
                if event.key == pygame.K_ESCAPE:
                    if self.mode == 'LOADING':
                        self.running = False
                    elif self.mode == 'VISION_CONTROLLER':
                        if self.runner_experience:
                            self.runner_experience.exit()
                        self.mode = 'MENU'
                    elif self.mode == 'ELEMENTAL_CULTIVATION':
                        if self.elemental_experience:
                            self.elemental_experience.exit()
                        self.mode = 'MENU'
                    elif self.mode == 'PHASE_SHIFT':
                        if self.phase_shift_experience:
                            self.phase_shift_experience.exit()
                        self.mode = 'MENU'
                    elif self.mode == 'MENU':
                        self.running = False
                    continue

                # Mode Hotkeys (only when not loading)
                if self.mode == 'MENU':
                    if event.key in (pygame.K_1, pygame.K_SPACE, pygame.K_RETURN):
                        self._enter_vision_controller()
                    elif event.key == pygame.K_2:
                        self._enter_elemental_cultivation()
                    elif event.key == pygame.K_3:
                        self._enter_phase_shift()
                elif self.mode == 'VISION_CONTROLLER' and self.runner_experience:
                    self.runner_experience.handle_key(event)
                elif self.mode == 'ELEMENTAL_CULTIVATION' and self.elemental_experience:
                    self.elemental_experience.handle_key(event)
                elif self.mode == 'PHASE_SHIFT' and self.phase_shift_experience:
                    self.phase_shift_experience.handle_key(event)

    def _enter_vision_controller(self):
        if self.runner_experience is None:
            return
        self.mode = 'VISION_CONTROLLER'
        self.runner_experience.enter()

    def _enter_elemental_cultivation(self):
        if self.elemental_experience is None:
            return
        self.mode = 'ELEMENTAL_CULTIVATION'
        self.elemental_experience.enter()

    def _enter_phase_shift(self):
        if self.phase_shift_experience is None:
            return
        self.mode = 'PHASE_SHIFT'
        self.phase_shift_experience.enter()

    # ─────────────────────────────────────────────────────────────────────────
    # Rendering Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _render_loading_screen(self):
        """Animated loading screen shown while background init is running."""
        self.screen.fill(palette.VOID_DARK)
        try:
            cx = display_config.width  // 2
            cy = display_config.height // 2

            font_title = pygame.font.SysFont('Consolas', 52, bold=True)
            font_sub   = pygame.font.SysFont('Consolas', 22)
            font_msg   = pygame.font.SysFont('Consolas', 16)

            t1 = font_title.render('A E T H E R', True, palette.CYAN_NEON)
            t2 = font_sub.render('THE VISIONARY', True, palette.TEXT_WHITE)

            # Animated dots
            dots = '.' * (1 + int(time.time() * 2) % 4)
            t3 = font_msg.render(f'{self._init_status}{dots}', True, palette.GOLD_ACCENT)
            t4 = font_msg.render('Vision system coming online...', True, palette.TEXT_MUTED)

            self.screen.blit(t1, (cx - t1.get_width() // 2, cy - 100))
            self.screen.blit(t2, (cx - t2.get_width() // 2, cy - 40))
            self.screen.blit(t3, (cx - t3.get_width() // 2, cy + 20))
            self.screen.blit(t4, (cx - t4.get_width() // 2, cy + 50))

            # Simple animated spinner line
            elapsed = time.time() % 2.0
            bar_w   = int(300 * (elapsed / 2.0))
            pygame.draw.rect(self.screen, palette.BORDER_DIM, (cx - 150, cy + 90, 300, 6), border_radius=3)
            pygame.draw.rect(self.screen, palette.CYAN_NEON,  (cx - 150, cy + 90, bar_w, 6), border_radius=3)
        except Exception:
            pass

    def _render_master_menu(self, norm_pos: tuple, hands: list):
        w, h = display_config.width, display_config.height

        # Subtle background grid lines
        for gx in range(0, w, 80):
            pygame.draw.line(self.screen, (15, 22, 34), (gx, 50), (gx, h), 1)
        for gy in range(50, h, 80):
            pygame.draw.line(self.screen, (15, 22, 34), (0, gy), (w, gy), 1)

        # Header Hero Identity
        hero_title = self.font_hero.render('A E T H E R', True, palette.CYAN_NEON)
        hero_sub   = self.font_hero_sub.render('THE VISIONARY', True, palette.TEXT_WHITE)
        tagline    = self.font_tagline.render('"Your hands are the controller."', True, palette.GOLD_ACCENT)

        cx = w // 2
        self.screen.blit(hero_title, (cx - hero_title.get_width() // 2, 75))
        self.screen.blit(hero_sub,   (cx - hero_sub.get_width()   // 2, 140))
        self.screen.blit(tagline,    (cx - tagline.get_width()    // 2, 172))

        instr_txt = self.font_instr.render(
            'Select an exhibition experience below [ Press 1 to Launch ]',
            True, palette.TEXT_MUTED
        )
        self.screen.blit(instr_txt, (cx - instr_txt.get_width() // 2, 210))

        # 4 Experience Cards
        cards = [
            {
                'key': '1',
                'title': 'VISION CONTROLLER',
                'subtitle': 'Full Computer Vision Endless Arcade Runner',
                'status': 'READY // UNLOCKED',
                'active': True,
                'color': palette.CYAN_NEON,
                'border': palette.CYAN_NEON,
                'action': '1',
            },
            {
                'key': '2',
                'title': 'ELEMENTAL CULTIVATION',
                'subtitle': '7-Element Particle VFX Gesture Mudras',
                'status': 'READY // UNLOCKED',
                'active': True,
                'color': palette.GOLD_ACCENT,
                'border': palette.GOLD_ACCENT,
                'action': '2',
            },
            {
                'key': '3',
                'title': 'PHASE SHIFT',
                'subtitle': 'Hand-Gesture Real-time Invisibility Segmentation',
                'status': 'READY // UNLOCKED',
                'active': True,
                'color': (0, 220, 255),
                'border': (0, 220, 255),
                'action': '3',
            },
            {
                'key': '4',
                'title': 'REALITY SCULPTOR',
                'subtitle': 'Spatial 3D Mesh Manipulation Holography',
                'status': 'LOCKED — PHASE 4',
                'active': False,
                'color': palette.LOCKED_GRAY,
                'border': palette.BORDER_DIM,
                'action': None,
            },
        ]

        card_w  = 265
        card_h  = 240
        total_w = 4 * card_w + 3 * 22
        start_x = cx - total_w // 2
        card_y  = 250

        hx = int(norm_pos[0] * w)
        hy = int(norm_pos[1] * h)

        for i, card in enumerate(cards):
            x = start_x + i * (card_w + 22)
            card_rect = pygame.Rect(x, card_y, card_w, card_h)

            is_hovered = card['active'] and card_rect.collidepoint(hx, hy) and len(hands) > 0

            bg_color = (20, 30, 48, 220) if is_hovered else (14, 20, 32, 200)
            card_surf = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
            card_surf.fill(bg_color)

            border_c = palette.GOLD_ACCENT if is_hovered else card['border']
            border_w = 3 if is_hovered else (2 if card['active'] else 1)
            pygame.draw.rect(card_surf, border_c, (0, 0, card_w, card_h), border_w, border_radius=10)

            k_txt = self.font_badge.render(f"[{card['key']}]", True, border_c)
            card_surf.blit(k_txt, (14, 14))

            badge_c    = palette.GREEN_MATRIX if card['active'] else palette.LOCKED_GRAY
            status_txt = self.font_badge.render(card['status'], True, badge_c)
            card_surf.blit(status_txt, (card_w - status_txt.get_width() - 14, 14))

            t_txt = self.font_card_title.render(card['title'], True, card['color'])
            card_surf.blit(t_txt, (14, 55))

            div_c = palette.CYAN_GLOW if card['active'] else (35, 45, 60)
            pygame.draw.line(card_surf, div_c, (14, 90), (card_w - 14, 90), 1)

            sub_words = card['subtitle'].split(' ')
            line1 = ' '.join(sub_words[:3])
            line2 = ' '.join(sub_words[3:])
            text_c = palette.TEXT_WHITE if card['active'] else palette.LOCKED_GRAY
            card_surf.blit(self.font_card_desc.render(line1, True, text_c), (14, 105))
            card_surf.blit(self.font_card_desc.render(line2, True, text_c), (14, 128))

            if card['active']:
                key_hint = card['key']
                btn_txt = f'► PRESS {key_hint} TO LAUNCH' if not is_hovered else f'► ACTIVE (PRESS {key_hint})'
                btn_surf = self.font_badge.render(btn_txt, True, card['color'])
                card_surf.blit(btn_surf, (14, card_h - 32))
            else:
                btn_surf = self.font_badge.render('EXHIBITION LOCKED', True, palette.LOCKED_GRAY)
                card_surf.blit(btn_surf, (14, card_h - 32))

            self.screen.blit(card_surf, (x, card_y))

        footer_txt = self.font_instr.render(
            'Hotkeys: [1] Vision Controller  [2] Elemental Cultivation  [3] Phase Shift  |  [D] Debug  |  [ESC] Exit',
            True, palette.TEXT_MUTED
        )
        self.screen.blit(footer_txt, (cx - footer_txt.get_width() // 2, h - 38))

    def _render_camera_pip(self, frame_bgr: np.ndarray):
        """Renders a picture-in-picture webcam display in the bottom-right corner."""
        pip_w, pip_h = 220, 124
        px = display_config.width  - pip_w - 20
        py = display_config.height - pip_h - 20

        small_bgr = cv2.resize(frame_bgr, (pip_w, pip_h))
        small_rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)

        pip_surface = pygame.surfarray.make_surface(np.transpose(small_rgb, (1, 0, 2)))
        self.screen.blit(pip_surface, (px, py))
        pygame.draw.rect(self.screen, palette.CYAN_NEON, (px - 1, py - 1, pip_w + 2, pip_h + 2), 1)

        lbl = self.font_badge.render('OPTICAL SENSOR', True, palette.CYAN_NEON)
        self.screen.blit(lbl, (px + 6, py + 6))

    # ─────────────────────────────────────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────────────────────────────────────

    def _cleanup(self):
        print('[AETHER] Performing clean shutdown...')
        # Signal init thread to stop waiting if it hasn't finished
        self._init_done.set()
        if self.runner_experience:
            self.runner_experience.exit()
        if self.elemental_experience:
            self.elemental_experience.exit()
        if self.phase_shift_experience:
            self.phase_shift_experience.exit()
        if self.camera:
            self.camera.release()
        pygame.quit()
        print('[AETHER] Shutdown complete. Resources released.')


if __name__ == '__main__':
    app = AetherApp()
    app.run()
