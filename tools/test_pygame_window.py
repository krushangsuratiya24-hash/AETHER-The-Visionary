"""
AETHER — WINDOWS DISPLAY TEST
Completely isolated Pygame desktop window test.
No OpenCV, no MediaPipe, no AETHER code.
Purpose: verify Pygame can create a real desktop window on this machine.

Run:
    .venv\\Scripts\\python.exe tools\\test_pygame_window.py
"""
import sys
import time
import os

# ── Explicitly ensure NO headless/dummy overrides ────────────────────────────
# Remove any SDL_VIDEODRIVER override that would prevent a real window
for _env_key in ('SDL_VIDEODRIVER', 'SDL_WINDOWID', 'SDL_VIDEO_X11_WMCLASS'):
    if _env_key in os.environ:
        print(f'[DISPLAY-TEST] WARNING: removing environment override {_env_key}={os.environ[_env_key]}')
        del os.environ[_env_key]

# Center window on primary monitor
os.environ['SDL_VIDEO_CENTERED'] = '1'

import pygame

def main():
    print("=" * 60)
    print("AETHER — WINDOWS DISPLAY TEST")
    print("=" * 60)

    # ── Step 1: Init Pygame ───────────────────────────────────────────────────
    print(f"[DISPLAY-TEST] pygame version   : {pygame.version.ver}")
    print(f"[DISPLAY-TEST] SDL version      : {pygame.version.SDL}")

    result = pygame.init()
    print(f"[DISPLAY-TEST] pygame.init()    : {result[0]} succeeded, {result[1]} failed")

    if not pygame.display.get_init():
        pygame.display.init()

    print(f"[DISPLAY-TEST] display init     : {pygame.display.get_init()}")

    # ── Step 2: Read display driver & desktop info ───────────────────────────
    try:
        driver = pygame.display.get_driver()
        print(f"[DISPLAY-TEST] SDL video driver : {driver}")
        if driver == 'dummy':
            print("[DISPLAY-TEST] FATAL: SDL video driver is 'dummy' — no real window will appear!")
            print("               Check SDL_VIDEODRIVER environment variable.")
            sys.exit(1)
    except Exception as e:
        print(f"[DISPLAY-TEST] Could not get driver: {e}")

    try:
        desktop_info = pygame.display.Info()
        print(f"[DISPLAY-TEST] desktop size     : {desktop_info.current_w} x {desktop_info.current_h}")
        print(f"[DISPLAY-TEST] display depth    : {desktop_info.bitsize} bpp")
    except Exception as e:
        print(f"[DISPLAY-TEST] Could not get desktop info: {e}")

    # ── Step 3: Create window ────────────────────────────────────────────────
    WIN_W, WIN_H = 960, 540
    print(f"[DISPLAY-TEST] creating {WIN_W}x{WIN_H} window (standard windowed mode)...")

    screen = pygame.display.set_mode((WIN_W, WIN_H), 0)
    pygame.display.set_caption("AETHER \u2014 WINDOWS DISPLAY TEST")

    if screen is None:
        print("[DISPLAY-TEST] FATAL: pygame.display.set_mode() returned None!")
        sys.exit(1)

    print(f"[DISPLAY-TEST] surface created  : {screen.get_size()}")

    # On Windows, bring window to foreground via Win32 API
    if sys.platform == 'win32':
        try:
            import ctypes
            wm = pygame.display.get_wm_info()
            hwnd = wm.get('window') if wm else None
            if hwnd:
                user32 = ctypes.windll.user32
                user32.ShowWindow(hwnd, 9)   # SW_RESTORE
                user32.ShowWindow(hwnd, 5)   # SW_SHOW
                user32.BringWindowToTop(hwnd)
                user32.SetForegroundWindow(hwnd)
                print(f"[DISPLAY-TEST] Win32 HWND       : {hwnd:#010x}")
                print("[DISPLAY-TEST] window foreground: OK")
        except Exception as e:
            print(f"[DISPLAY-TEST] Win32 foreground note: {e}")

    # ── Step 4: Draw obvious test frame ─────────────────────────────────────
    pygame.font.init()
    font_big  = pygame.font.SysFont('Consolas', 52, bold=True)
    font_med  = pygame.font.SysFont('Consolas', 28)
    font_info = pygame.font.SysFont('Consolas', 18)

    BG_COLOR       = (10, 14, 23)
    CYAN           = (0, 240, 255)
    GOLD           = (255, 215, 0)
    GREEN          = (0, 255, 102)
    WHITE          = (245, 247, 250)

    screen.fill(BG_COLOR)
    pygame.draw.rect(screen, CYAN, (0, 0, WIN_W, WIN_H), 4)

    t1 = font_big.render("AETHER DISPLAY TEST", True, CYAN)
    t2 = font_med.render("WINDOWS GUI IS WORKING", True, GREEN)
    t3 = font_info.render(f"pygame-ce {pygame.version.ver}  |  SDL {pygame.version.SDL}", True, GOLD)
    t4 = font_info.render("This window will close in 15 seconds.", True, WHITE)

    screen.blit(t1, (WIN_W//2 - t1.get_width()//2, 160))
    screen.blit(t2, (WIN_W//2 - t2.get_width()//2, 240))
    screen.blit(t3, (WIN_W//2 - t3.get_width()//2, 310))
    screen.blit(t4, (WIN_W//2 - t4.get_width()//2, 360))

    pygame.display.flip()
    pygame.event.pump()

    print("[DISPLAY-TEST] first frame presented — window should now be VISIBLE on desktop")
    print("[DISPLAY-TEST] keeping window alive for 15 seconds (close manually or wait)...")

    # ── Step 5: Event loop for 15 seconds ───────────────────────────────────
    clock = pygame.time.Clock()
    start = time.time()
    while time.time() - start < 15.0:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("[DISPLAY-TEST] Window closed by user.")
                pygame.quit()
                sys.exit(0)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                print("[DISPLAY-TEST] ESC pressed — closing.")
                pygame.quit()
                sys.exit(0)

        # Update countdown timer on screen each second
        elapsed = time.time() - start
        remaining = max(0, 15 - int(elapsed))

        screen.fill(BG_COLOR)
        pygame.draw.rect(screen, CYAN, (0, 0, WIN_W, WIN_H), 4)
        screen.blit(t1, (WIN_W//2 - t1.get_width()//2, 160))
        screen.blit(t2, (WIN_W//2 - t2.get_width()//2, 240))
        screen.blit(t3, (WIN_W//2 - t3.get_width()//2, 310))
        countdown = font_med.render(f"Closing in {remaining}s  (press ESC to quit early)", True, GOLD)
        screen.blit(countdown, (WIN_W//2 - countdown.get_width()//2, 370))
        pygame.display.flip()
        clock.tick(30)

    print("[DISPLAY-TEST] 15-second test complete.")
    pygame.quit()
    print("[DISPLAY-TEST] pygame shut down cleanly.")
    print("=" * 60)
    print("RESULT: If you saw the window above, Pygame window creation is WORKING.")
    print("=" * 60)

if __name__ == '__main__':
    main()
