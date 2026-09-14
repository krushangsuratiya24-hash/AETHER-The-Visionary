import pygame
from typing import Tuple, Optional
from core.config import palette, display_config, gesture_config
from core.gestures import GestureType

class HUD:
    """
    Futuristic Exhibition HUD for AETHER.
    Renders clean cybernetic telemetry, gesture status, reticles, and debug panels.
    """
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.width = screen.get_width()
        self.height = screen.get_height()
        
        # Fonts
        pygame.font.init()
        self.font_title = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_sub = pygame.font.SysFont("Consolas", 14)
        self.font_bold = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 13)
        self.font_gesture = pygame.font.SysFont("Consolas", 28, bold=True)
        
        # Reticle smoothing
        self.reticle_pos = (self.width // 2, self.height // 2)

    def draw_top_bar(self, mode_name: str, fps: float, is_mock: bool):
        """
        Renders the sleek top navigation bar.
        """
        # Glass panel header
        header_surf = pygame.Surface((self.width, 50), pygame.SRCALPHA)
        header_surf.fill((10, 14, 23, 210))
        self.screen.blit(header_surf, (0, 0))
        pygame.draw.line(self.screen, palette.BORDER_DIM, (0, 50), (self.width, 50), 1)

        # Title
        title_txt = self.font_title.render("AETHER", True, palette.CYAN_NEON)
        sub_txt = self.font_sub.render("THE VISIONARY", True, palette.TEXT_MUTED)
        self.screen.blit(title_txt, (24, 8))
        self.screen.blit(sub_txt, (120, 15))

        # Mode Badge
        mode_surf = self.font_bold.render(f"[ {mode_name} ]", True, palette.GOLD_ACCENT)
        self.screen.blit(mode_surf, (self.width // 2 - mode_surf.get_width() // 2, 14))

        # Status & FPS
        fps_color = palette.GREEN_MATRIX if fps >= 45 else (palette.GOLD_ACCENT if fps >= 25 else palette.MAGENTA_LASER)
        fps_txt = self.font_bold.render(f"{int(fps)} FPS", True, fps_color)
        
        cam_color = palette.MAGENTA_LASER if is_mock else palette.GREEN_MATRIX
        cam_label = "MOCK CAM" if is_mock else "CAM ONLINE"
        cam_txt = self.font_small.render(cam_label, True, cam_color)

        self.screen.blit(fps_txt, (self.width - 110, 14))
        pygame.draw.circle(self.screen, cam_color, (self.width - 130, 25), 5)
        self.screen.blit(cam_txt, (self.width - 230, 17))

    def draw_control_guides(self):
        """
        Draws subtle, semi-transparent threshold lines for gesture guidance.
        """
        guide_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        # Vertical boundaries (Left / Neutral / Right)
        lx = int((gesture_config.left_threshold - gesture_config.hysteresis_x) * self.width)
        rx = int((gesture_config.right_threshold + gesture_config.hysteresis_x) * self.width)
        pygame.draw.line(guide_surf, (0, 240, 255, 35), (lx, 50), (lx, self.height), 1)
        pygame.draw.line(guide_surf, (0, 240, 255, 35), (rx, 50), (rx, self.height), 1)
        
        # Horizontal boundaries (Jump / Slide)
        jy = int(gesture_config.jump_threshold * self.height)
        sy = int(gesture_config.slide_threshold * self.height)
        pygame.draw.line(guide_surf, (255, 215, 0, 35), (0, jy), (self.width, jy), 1)
        pygame.draw.line(guide_surf, (255, 0, 85, 35), (0, sy), (self.width, sy), 1)

        # Label hints
        txt_l = self.font_small.render("◄ LEFT ZONE", True, (0, 240, 255, 60))
        txt_r = self.font_small.render("RIGHT ZONE ►", True, (0, 240, 255, 60))
        txt_j = self.font_small.render("▲ JUMP THRESHOLD", True, (255, 215, 0, 60))
        txt_s = self.font_small.render("▼ SLIDE THRESHOLD", True, (255, 0, 85, 60))
        
        guide_surf.blit(txt_l, (15, self.height - 35))
        guide_surf.blit(txt_r, (self.width - txt_r.get_width() - 15, self.height - 35))
        guide_surf.blit(txt_j, (self.width // 2 - txt_j.get_width() // 2, jy + 6))
        guide_surf.blit(txt_s, (self.width // 2 - txt_s.get_width() // 2, sy - 20))
        
        self.screen.blit(guide_surf, (0, 0))

    def draw_hand_reticle(self, norm_pos: Tuple[float, float], is_hand_present: bool):
        """
        Draws a sci-fi tracking reticle over the active hand position.
        """
        if not is_hand_present:
            return

        tx = int(norm_pos[0] * self.width)
        ty = int(norm_pos[1] * self.height)
        
        # Smooth reticle
        rx = int(self.reticle_pos[0] * 0.7 + tx * 0.3)
        ry = int(self.reticle_pos[1] * 0.7 + ty * 0.3)
        self.reticle_pos = (rx, ry)

        # Glowing rings
        pygame.draw.circle(self.screen, palette.CYAN_NEON, (rx, ry), 18, 1)
        pygame.draw.circle(self.screen, palette.CYAN_GLOW, (rx, ry), 24, 1)
        pygame.draw.circle(self.screen, palette.MAGENTA_LASER, (rx, ry), 4)

        # Crosshair lines
        pygame.draw.line(self.screen, palette.CYAN_NEON, (rx - 28, ry), (rx - 12, ry), 1)
        pygame.draw.line(self.screen, palette.CYAN_NEON, (rx + 12, ry), (rx + 28, ry), 1)
        pygame.draw.line(self.screen, palette.CYAN_NEON, (rx, ry - 28), (rx, ry - 12), 1)
        pygame.draw.line(self.screen, palette.CYAN_NEON, (rx, ry + 12), (rx, ry + 28), 1)

    def draw_gesture_card(self, current_gesture: GestureType, active_action: GestureType):
        """
        Renders the active gesture telemetry card at bottom-center.
        """
        card_w, card_h = 320, 68
        cx = self.width // 2 - card_w // 2
        cy = self.height - card_h - 18
        
        card_surf = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
        card_surf.fill((14, 20, 32, 220))
        
        # Border color depending on gesture
        border_color = palette.BORDER_DIM
        if current_gesture == GestureType.LEFT:
            border_color = palette.CYAN_NEON
        elif current_gesture == GestureType.RIGHT:
            border_color = palette.CYAN_NEON
        elif current_gesture == GestureType.JUMP:
            border_color = palette.GOLD_ACCENT
        elif current_gesture == GestureType.SLIDE:
            border_color = palette.MAGENTA_LASER
            
        pygame.draw.rect(card_surf, border_color, (0, 0, card_w, card_h), 2, border_radius=8)
        
        # Label
        lbl = self.font_small.render("ACTIVE GESTURE", True, palette.TEXT_MUTED)
        card_surf.blit(lbl, (card_w // 2 - lbl.get_width() // 2, 8))
        
        # Main gesture name
        sym_map = {
            GestureType.NEUTRAL: "[ NEUTRAL ]",
            GestureType.LEFT: "[ ◄ LEFT ]",
            GestureType.RIGHT: "[ RIGHT ► ]",
            GestureType.JUMP: "[ ▲ JUMP ]",
            GestureType.SLIDE: "[ ▼ SLIDE ]"
        }
        g_txt = self.font_gesture.render(sym_map.get(current_gesture, "[ NEUTRAL ]"), True, border_color)
        card_surf.blit(g_txt, (card_w // 2 - g_txt.get_width() // 2, 26))
        
        self.screen.blit(card_surf, (cx, cy))

    def draw_debug_panel(self, info: dict):
        """
        Renders the collapsible developer/debug diagnostics panel.
        """
        panel_w, panel_h = 280, 190
        px, py = 20, 65
        
        debug_surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        debug_surf.fill((10, 14, 23, 230))
        pygame.draw.rect(debug_surf, palette.CYAN_NEON, (0, 0, panel_w, panel_h), 1, border_radius=6)
        
        title = self.font_bold.render("DEBUG TELEMETRY [D]", True, palette.CYAN_NEON)
        debug_surf.blit(title, (12, 10))
        
        y_off = 38
        for k, v in info.items():
            line_str = f"{k}: {v}"
            line_txt = self.font_small.render(line_str, True, palette.TEXT_WHITE)
            debug_surf.blit(line_txt, (12, y_off))
            y_off += 24
            
        self.screen.blit(debug_surf, (px, py))
