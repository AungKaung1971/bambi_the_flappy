import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class FlappyBirdRGBEnv(gym.Env):
    """A polished, visual-only Flappy Bird environment.
    This environment is NOT for training — only for rendering videos.
    Physics and game logic mirror the simple environment, but visuals
    are much more polished (circle bird, smoother motion, better pipes).
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(self, simple_env):
        """
        We pass in an ALREADY-STEPPED simple environment.
        This renderer simply displays the same state with better visuals.

        Args:
            simple_env: the FlappyBirdSimpleEnv instance providing state.
        """
        super().__init__()
        self.simple = simple_env

        self.screen_width = self.simple.screen_width
        self.screen_height = self.simple.screen_height

        self.screen = None

    # ------------------------------------------------------------
    # RENDER AS RGB ARRAY (NO HUMAN MODE)
    # ------------------------------------------------------------
    def render(self):
        """Render a polished RGB frame."""
        if self.screen is None:
            pygame.init()
            self.screen = pygame.Surface(
                (self.screen_width, self.screen_height))

        # Extract state from simple environment
        bird_x = self.simple.bird_x
        bird_y = self.simple.bird_y
        pipe_x = self.simple.pipe_x
        pipe_width = self.simple.pipe_width
        gap_top = self.simple.pipe_gap_top
        gap_bottom = self.simple.pipe_gap_bottom

        # ----------------------------------------
        # Draw background
        # ----------------------------------------
        sky_top = (135, 206, 250)
        sky_bottom = (100, 180, 230)

        # Gradient sky
        for y in range(self.screen_height):
            t = y / self.screen_height
            r = int(sky_top[0] * (1 - t) + sky_bottom[0] * t)
            g = int(sky_top[1] * (1 - t) + sky_bottom[1] * t)
            b = int(sky_top[2] * (1 - t) + sky_bottom[2] * t)
            pygame.draw.line(self.screen, (r, g, b),
                             (0, y), (self.screen_width, y))

        # ----------------------------------------
        # Draw pipes (smoother, nicer looking)
        # ----------------------------------------
        pipe_color = (80, 200, 80)

        # Top pipe
        pygame.draw.rect(
            self.screen,
            pipe_color,
            pygame.Rect(pipe_x, 0, pipe_width, gap_top)
        )

        # Bottom pipe
        pygame.draw.rect(
            self.screen,
            pipe_color,
            pygame.Rect(pipe_x, gap_bottom, pipe_width,
                        self.screen_height - gap_bottom)
        )

        # Add pipe shading
        shade_color = (50, 150, 50)
        pygame.draw.rect(
            self.screen,
            shade_color,
            pygame.Rect(pipe_x + pipe_width - 8, 0, 8, self.screen_height)
        )

        # ----------------------------------------
        # Draw bird (circle)
        # ----------------------------------------
        bird_color = (255, 255, 0)
        bird_radius = 12

        pygame.draw.circle(
            self.screen,
            bird_color,
            (int(bird_x), int(bird_y)),
            bird_radius
        )

        # Eye
        pygame.draw.circle(
            self.screen,
            (0, 0, 0),
            (int(bird_x + 5), int(bird_y - 4)),
            3
        )

        # ----------------------------------------
        # Return RGB array
        # ----------------------------------------
        frame = pygame.surfarray.array3d(self.screen).swapaxes(0, 1)
        return frame

    def close(self):
        pygame.quit()
