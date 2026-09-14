import sys
import time
import pygame
import cv2
import numpy as np

from core.config import display_config, palette, camera_config
from core.camera import ThreadedCamera
from core.tracker import HandTracker
from core.gestures import GestureProcessor, GestureType
from core.ui import HUD
from experiences.vision_controller.runner import NeonRunnerExperience

class AetherApp:
    """
    AETHER — THE VISIONARY: Master Exhibition Application.
    Orchestrates camera acquisition, MediaPipe landmark tracking, gesture recognition,
    and modular experience switching.
    """
    def __init__(self):
        # 0. Ensure SDL2 centers the window on monitor
        import os
        os.environ['SDL_VIDEO_CENTERED'] = '1'

        # 1. Initialize Pygame Display Subsystem
        pygame.init()
        if not pygame.display.get_init():
            pygame.display.init()

        pygame.display.set_caption(display_config.window_title)
        
        # Clean display flags: Use standard window flags (avoid DOUBLEBUF without OPENGL)
        flags = 0
        if display_config.fullscreen:
            flags |= pygame.FULLSCREEN

        self.screen = pygame.display.set_mode((display_config.width, display_config.height), flags)
        if self.screen is None:
            raise RuntimeError("[AETHER] Failed to create a valid Pygame display surface.")

        # Immediate paint & event pump so the OS window manager registers and paints the window immediately
        self.screen.fill(palette.VOID_DARK)
        pygame.display.flip()
        pygame.event.pump()

        # Bring window to foreground and ensure visibility on Windows
        self._ensure_window_foreground()

        # Show immediate startup feedback so user sees the window is active while loading
        self._render_splash_screen("AETHER // INITIALIZING OPTICAL SENSORS...")

        self.clock = pygame.time.Clock()
        self.running = True
        
        # 2. Initialize Core Subsystems
        print('[AETHER] Initializing Camera Pipeline...')
        self.camera = ThreadedCamera().start()
        
        print('[AETHER] Initializing MediaPipe Hand Tracker...')
        self.tracker = HandTracker()
        self.gestures = GestureProcessor()
        self.hud = HUD(self.screen)
        
        # 3. Initialize Experiences
        self.runner_experience = NeonRunnerExperience(display_config.width, display_config.height)
        
        # App State: 'MENU' or 'VISION_CONTROLLER'
        self.mode = 'MENU'
        self.debug_mode = False
        
        # Performance Telemetry
        self.fps = 0.0
        self.latency_ms = 0.0
        
        # UI Assets & Fonts
        self.font_hero = pygame.font.SysFont('Consolas', 54, bold=True)
        self.font_hero_sub = pygame.font.SysFont('Consolas', 20)
        self.font_tagline = pygame.font.SysFont('Consolas', 17, italic=True)
        self.font_card_title = pygame.font.SysFont('Consolas', 22, bold=True)
        self.font_card_desc = pygame.font.SysFont('Consolas', 14)
        self.font_badge = pygame.font.SysFont('Consolas', 12, bold=True)
        self.font_instr = pygame.font.SysFont('Consolas', 15)

    def _ensure_window_foreground(self):
        """Ensures the Pygame window is visible, restored, and brought to the foreground on Windows."""
        if sys.platform == 'win32':
            try:
                import ctypes
                wm_info = pygame.display.get_wm_info()
                hwnd = wm_info.get('window') if wm_info else None
                if hwnd:
                    user32 = ctypes.windll.user32
                    user32.ShowWindow(hwnd, 9)  # SW_RESTORE (9)
                    user32.ShowWindow(hwnd, 5)  # SW_SHOW (5)
                    user32.BringWindowToTop(hwnd)
                    user32.SetForegroundWindow(hwnd)
            except Exception as e:
                print(f'[AETHER] Window foreground notice: {e}')

    def _render_splash_screen(self, message: str):
        """Renders an immediate visual feedback screen during initialization."""
        self.screen.fill(palette.VOID_DARK)
        try:
            splash_font = pygame.font.SysFont('Consolas', 24, bold=True)
            txt = splash_font.render(message, True, palette.CYAN_NEON)
            self.screen.blit(txt, (display_config.width // 2 - txt.get_width() // 2,
                                   display_config.height // 2 - txt.get_height() // 2))
        except Exception:
            pass
        pygame.display.flip()
        pygame.event.pump()

    def run(self):
        print('[AETHER] Entering Main Exhibition Loop.')
        last_time = time.time()
        
        try:
            while self.running:
                now = time.time()
                dt = min(0.1, now - last_time)
                last_time = now

                # 1. Handle Window Events
                self._handle_events()

                # 2. Camera Frame Acquisition & Gesture Tracking
                t_start = time.time()
                ret, frame = self.camera.read()
                hands = []
                current_gesture = GestureType.NEUTRAL
                active_action = GestureType.NEUTRAL
                norm_pos = (0.5, 0.5)

                if ret and frame is not None:
                    # Run landmark detection
                    hands = self.tracker.process_frame(frame)
                    # Overlay skeleton drawing onto frame for PIP feed
                    if hands:
                        self.tracker.draw_landmarks(frame, hands)
                        current_gesture, active_action, norm_pos = self.gestures.process(hands[0])
                    else:
                        current_gesture, active_action, norm_pos = self.gestures.process(None)
                
                self.latency_ms = (time.time() - t_start) * 1000.0

                # 3. Update Active Experience
                if self.mode == 'VISION_CONTROLLER':
                    self.runner_experience.handle_gesture(current_gesture, active_action, norm_pos)
                    self.runner_experience.update(dt)

                # 4. Render Pipeline
                self.screen.fill(palette.VOID_DARK)
                
                if self.mode == 'MENU':
                    self._render_master_menu(norm_pos, hands)
                elif self.mode == 'VISION_CONTROLLER':
                    self.runner_experience.render(self.screen)

                # 5. Composite HUD Overlays
                mode_labels = {
                    'MENU': 'MAIN HUB',
                    'VISION_CONTROLLER': 'PHASE 1 // VISION CONTROLLER'
                }
                self.hud.draw_top_bar(mode_labels.get(self.mode, self.mode), self.fps, self.camera.is_mock)
                
                if self.mode == 'VISION_CONTROLLER':
                    self.hud.draw_control_guides()
                    self.hud.draw_hand_reticle(norm_pos, len(hands) > 0)
                    self.hud.draw_gesture_card(current_gesture, active_action)

                # 6. Render Camera PIP (Picture-In-Picture) in corner
                if frame is not None:
                    self._render_camera_pip(frame)

                # 7. Debug Telemetry Panel
                if self.debug_mode:
                    debug_info = {
                        'Mode': self.mode,
                        'FPS': f'{int(self.fps)}',
                        'Inference Latency': f'{self.latency_ms:.1f} ms',
                        'Camera Source': 'MOCK' if self.camera.is_mock else 'USB WEBCAM',
                        'Hands Tracked': len(hands),
                        'Gesture State': current_gesture.value,
                        'Triggered Action': active_action.value,
                        'Hand Pos (X, Y)': f'({norm_pos[0]:.2f}, {norm_pos[1]:.2f})',
                        'Confidence': f'{self.gestures.hand_confidence:.2f}'
                    }
                    self.hud.draw_debug_panel(debug_info)

                # Flip Frame & Update Clock
                pygame.display.flip()
                self.clock.tick(display_config.target_fps)
                self.fps = self.clock.get_fps()

        finally:
            self._cleanup()

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
                    if self.mode == 'VISION_CONTROLLER':
                        self.runner_experience.exit()
                        self.mode = 'MENU'
                    elif self.mode == 'MENU':
                        self.running = False
                    continue

                # Mode Hotkeys
                if self.mode == 'MENU':
                    if event.key in (pygame.K_1, pygame.K_SPACE, pygame.K_RETURN):
                        self._enter_vision_controller()
                elif self.mode == 'VISION_CONTROLLER':
                    self.runner_experience.handle_key(event)

    def _enter_vision_controller(self):
        self.mode = 'VISION_CONTROLLER'
        self.runner_experience.enter()

    def _render_master_menu(self, norm_pos: tuple, hands: list):
        # Background cyber aura
        w, h = display_config.width, display_config.height
        
        # Subtle glowing background grid lines
        for gx in range(0, w, 80):
            pygame.draw.line(self.screen, (15, 22, 34), (gx, 50), (gx, h), 1)
        for gy in range(50, h, 80):
            pygame.draw.line(self.screen, (15, 22, 34), (0, gy), (w, gy), 1)

        # Header Hero Identity
        hero_title = self.font_hero.render("A E T H E R", True, palette.CYAN_NEON)
        hero_sub = self.font_hero_sub.render("THE VISIONARY", True, palette.TEXT_WHITE)
        tagline = self.font_tagline.render('"Your hands are the controller."', True, palette.GOLD_ACCENT)

        cx = w // 2
        self.screen.blit(hero_title, (cx - hero_title.get_width() // 2, 75))
        self.screen.blit(hero_sub, (cx - hero_sub.get_width() // 2, 140))
        self.screen.blit(tagline, (cx - tagline.get_width() // 2, 172))

        # Instructions banner
        instr_txt = self.font_instr.render("Select an exhibition experience below [ Press 1 to Launch ]", True, palette.TEXT_MUTED)
        self.screen.blit(instr_txt, (cx - instr_txt.get_width() // 2, 210))

        # 4 Experience Cards
        cards = [
            {
                'key': '1',
                'title': '🎮 VISION CONTROLLER',
                'subtitle': 'Full Computer Vision Endless Arcade Runner',
                'status': 'READY // UNLOCKED',
                'active': True,
                'color': palette.CYAN_NEON,
                'border': palette.CYAN_NEON
            },
            {
                'key': '2',
                'title': '🔥 ELEMENTAL CULTIVATION',
                'subtitle': '7-Element Particle VFX & Gesture Mudras',
                'status': 'LOCKED — PHASE 2',
                'active': False,
                'color': palette.LOCKED_GRAY,
                'border': palette.BORDER_DIM
            },
            {
                'key': '3',
                'title': '👻 PHASE SHIFT',
                'subtitle': 'Clap Acoustic Trigger & Predator Cloaking',
                'status': 'LOCKED — PHASE 3',
                'active': False,
                'color': palette.LOCKED_GRAY,
                'border': palette.BORDER_DIM
            },
            {
                'key': '4',
                'title': '🌀 REALITY SCULPTOR',
                'subtitle': 'Spatial 3D Mesh Manipulation & Holography',
                'status': 'LOCKED — PHASE 4',
                'active': False,
                'color': palette.LOCKED_GRAY,
                'border': palette.BORDER_DIM
            }
        ]

        card_w = 265
        card_h = 240
        total_w = 4 * card_w + 3 * 22
        start_x = cx - total_w // 2
        card_y = 250

        # Check if hand hover or click interacts with Phase 1 card
        hx = int(norm_pos[0] * w)
        hy = int(norm_pos[1] * h)

        for i, card in enumerate(cards):
            x = start_x + i * (card_w + 22)
            card_rect = pygame.Rect(x, card_y, card_w, card_h)
            
            # Hover check on Phase 1 card
            is_hovered = card['active'] and card_rect.collidepoint(hx, hy) and len(hands) > 0

            # Glass background
            bg_color = (20, 30, 48, 220) if is_hovered else (14, 20, 32, 200)
            card_surf = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
            card_surf.fill(bg_color)
            
            border_c = palette.GOLD_ACCENT if is_hovered else card['border']
            border_w = 3 if is_hovered else (2 if card['active'] else 1)
            pygame.draw.rect(card_surf, border_c, (0, 0, card_w, card_h), border_w, border_radius=10)

            # Key Badge in Corner
            k_txt = self.font_badge.render(f"[{card['key']}]", True, border_c)
            card_surf.blit(k_txt, (14, 14))

            # Status Badge
            badge_c = palette.GREEN_MATRIX if card['active'] else palette.LOCKED_GRAY
            status_txt = self.font_badge.render(card['status'], True, badge_c)
            card_surf.blit(status_txt, (card_w - status_txt.get_width() - 14, 14))

            # Title
            t_txt = self.font_card_title.render(card['title'], True, card['color'])
            card_surf.blit(t_txt, (14, 55))

            # Divider line
            div_c = palette.CYAN_GLOW if card['active'] else (35, 45, 60)
            pygame.draw.line(card_surf, div_c, (14, 90), (card_w - 14, 90), 1)

            # Subtitle / Description
            sub_words = card['subtitle'].split(' ')
            line1 = ' '.join(sub_words[:3])
            line2 = ' '.join(sub_words[3:])
            card_surf.blit(self.font_card_desc.render(line1, True, palette.TEXT_WHITE if card['active'] else palette.LOCKED_GRAY), (14, 105))
            card_surf.blit(self.font_card_desc.render(line2, True, palette.TEXT_WHITE if card['active'] else palette.LOCKED_GRAY), (14, 128))

            # Action Callout
            if card['active']:
                btn_txt = "► PRESS 1 OR HOVER TO PLAY" if not is_hovered else "► ACTIVE (ENTER TO PLAY)"
                btn_surf = self.font_badge.render(btn_txt, True, palette.CYAN_NEON)
                card_surf.blit(btn_surf, (14, card_h - 32))
            else:
                btn_txt = "🔒 EXHIBITION LOCKED"
                btn_surf = self.font_badge.render(btn_txt, True, palette.LOCKED_GRAY)
                card_surf.blit(btn_surf, (14, card_h - 32))

            self.screen.blit(card_surf, (x, card_y))

        # Bottom navigation footer
        footer_txt = self.font_instr.render("Hotkeys: [1] Play Vision Controller  |  [D] Toggle Debug Telemetry  |  [ESC] Exit", True, palette.TEXT_MUTED)
        self.screen.blit(footer_txt, (cx - footer_txt.get_width() // 2, h - 38))

    def _render_camera_pip(self, frame_bgr: np.ndarray):
        """
        Renders a picture-in-picture webcam display in the bottom-right corner.
        """
        pip_w, pip_h = 220, 124
        px = display_config.width - pip_w - 20
        py = display_config.height - pip_h - 20

        # Resize camera feed
        small_bgr = cv2.resize(frame_bgr, (pip_w, pip_h))
        small_rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
        
        # Convert to Pygame surface (note transposed shape in pygame surfarray)
        pip_surface = pygame.surfarray.make_surface(np.transpose(small_rgb, (1, 0, 2)))
        
        self.screen.blit(pip_surface, (px, py))
        pygame.draw.rect(self.screen, palette.CYAN_NEON, (px - 1, py - 1, pip_w + 2, pip_h + 2), 1)

        # Pip Label
        lbl = self.font_badge.render("OPTICAL SENSOR", True, palette.CYAN_NEON)
        self.screen.blit(lbl, (px + 6, py + 6))

    def _cleanup(self):
        print('[AETHER] Performing clean shutdown...')
        self.runner_experience.exit()
        self.camera.release()
        pygame.quit()
        print('[AETHER] Shutdown complete. Resources released.')

if __name__ == '__main__':
    app = AetherApp()
    app.run()
