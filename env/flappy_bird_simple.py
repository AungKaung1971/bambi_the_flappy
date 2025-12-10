import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random


class FlappyBirdSimpleEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()

        self.WIDTH = 640
        self.HEIGHT = 360
        self.FPS = 60

        self.GRAVITY = 0.25
        self.FLAP_STRENGTH = -5.0

        self.BIRD_RADIUS = 12
        self.BIRD_X = 50.0

        self.PIPE_WIDTH = 50
        self.PIPE_GAP_HEIGHT = 140
        self.PIPE_SPEED = 2.0
        self.PIPE_INTERVAL = 100

        # Rendering
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None

        # Game state
        self.bird_y = None
        self.bird_vel = None
        self.pipes = []
        self.pipe_timer = 0
        self.score = 0

        low = np.array([0.0, -10.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array(
            [float(self.HEIGHT), 10.0, float(self.WIDTH),
             float(self.HEIGHT), float(self.HEIGHT)],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    # ------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.bird_y = self.HEIGHT / 2.0
        self.bird_vel = 0.0
        self.score = 0
        self.pipe_timer = 0
        self.pipes = []

        gap_y = random.randint(100, self.HEIGHT - 100)
        self.pipes.append(
            {
                "x": self.WIDTH,
                "gap_y": float(gap_y),
                "gap_height": float(self.PIPE_GAP_HEIGHT),
                "speed": float(self.PIPE_SPEED),
                "width": self.PIPE_WIDTH,
                "scored": False,
            }
        )

        if self.render_mode == "human":
            self._init_pygame_display()
        elif self.render_mode == "rgb_array":
            self._init_pygame_surface()

        return self._get_obs(), {}

    # ------------------------------------------------------------
    def step(self, action):
        assert self.action_space.contains(action)

        # --------------------------------------------------------
        # ACTION
        # --------------------------------------------------------
        if action == 1:
            self.bird_vel = self.FLAP_STRENGTH

        # BIRD PHYSICS
        self.bird_vel += self.GRAVITY
        self.bird_y += self.bird_vel

        # --------------------------------------------------------
        # UPDATE PIPES + REWARDS
        # --------------------------------------------------------
        survival_reward = 1.0
        pipe_bonus = 0.0

        for pipe in self.pipes:
            pipe["x"] -= pipe["speed"]

            if pipe["x"] + pipe["width"] < self.BIRD_X and not pipe["scored"]:
                pipe["scored"] = True
                self.score += 1
                pipe_bonus = 10.0

        reward = survival_reward + pipe_bonus
        terminated = False

        # Collision detection
        for pipe in self.pipes:
            if self._check_collision(pipe):
                terminated = True
                break

        # Out of bounds
        if self.bird_y < 0 or self.bird_y > self.HEIGHT:
            terminated = True

        if terminated:
            reward = -100.0

        # Clean pipes
        self.pipes = [p for p in self.pipes if p["x"] + p["width"] > 0]

        # Spawn new pipes
        self.pipe_timer += 1
        if self.pipe_timer >= self.PIPE_INTERVAL:
            gap_y = random.randint(100, self.HEIGHT - 100)
            self.pipes.append(
                {
                    "x": float(self.WIDTH),
                    "gap_y": float(gap_y),
                    "gap_height": float(self.PIPE_GAP_HEIGHT),
                    "speed": float(self.PIPE_SPEED),
                    "width": self.PIPE_WIDTH,
                    "scored": False,
                }
            )
            self.pipe_timer = 0

        # --------------------------------------------------------
        # FIND NEAREST PIPE (for logging)
        # --------------------------------------------------------
        nearest_pipe = None
        min_dx = float("inf")
        for pipe in self.pipes:
            dx = pipe["x"] + pipe["width"] - self.BIRD_X
            if dx >= 0 and dx < min_dx:
                min_dx = dx
                nearest_pipe = pipe

        if nearest_pipe:
            gap_center_y = nearest_pipe["gap_y"]
            dist_to_gap_center = float(abs(self.bird_y - gap_center_y))
        else:
            dist_to_gap_center = 0.0

        # --------------------------------------------------------
        # INFO DICTIONARY FOR TENSORBOARD  <<< NEW! >>>
        # --------------------------------------------------------
        info = {
            "survival_reward": survival_reward,
            "pipe_bonus": pipe_bonus,
            "death": terminated,
            "dist_to_gap_center": dist_to_gap_center,
        }

        # --------------------------------------------------------
        # RENDER
        # --------------------------------------------------------
        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "rgb_array":
            frame = self._render_rgb_array()

        return self._get_obs(), reward, terminated, False, info

    # ------------------------------------------------------------
    def _get_obs(self):
        nearest_pipe = None
        min_dx = float("inf")

        for pipe in self.pipes:
            dx = pipe["x"] + pipe["width"] - self.BIRD_X
            if dx >= 0 and dx < min_dx:
                min_dx = dx
                nearest_pipe = pipe

        if nearest_pipe is None:
            nearest_pipe = {
                "x": self.BIRD_X + 200,
                "gap_y": self.HEIGHT / 2,
                "gap_height": self.PIPE_GAP_HEIGHT,
            }

        gap_top = nearest_pipe["gap_y"] - nearest_pipe["gap_height"] / 2
        gap_bottom = nearest_pipe["gap_y"] + nearest_pipe["gap_height"] / 2

        return np.array(
            [
                float(self.bird_y),
                float(self.bird_vel),
                float(nearest_pipe["x"] - self.BIRD_X),
                float(gap_top),
                float(gap_bottom),
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------
    def _check_collision(self, pipe):
        if (
            self.BIRD_X + self.BIRD_RADIUS > pipe["x"]
            and self.BIRD_X - self.BIRD_RADIUS < pipe["x"] + pipe["width"]
        ):
            top_pipe_bottom = pipe["gap_y"] - pipe["gap_height"] / 2
            bottom_pipe_top = pipe["gap_y"] + pipe["gap_height"] / 2

            if self.bird_y - self.BIRD_RADIUS < top_pipe_bottom:
                return True
            if self.bird_y + self.BIRD_RADIUS > bottom_pipe_top:
                return True

        return False

    # ------------------------------------------------------------
    # PYGAME RENDERING
    # ------------------------------------------------------------
    def _init_pygame_display(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Flappy Bird - RL Env")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 32)

    def _init_pygame_surface(self):
        pygame.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 32)

    def _render_human(self):
        pygame.event.pump()
        self._draw_scene()
        pygame.display.flip()
        self.clock.tick(self.FPS)

    def _render_rgb_array(self):
        self._draw_scene()
        return pygame.surfarray.array3d(self.screen).swapaxes(0, 1)

    # ------------------------------------------------------------
    # DRAW SCENE (unchanged)
    # ------------------------------------------------------------
    def _draw_scene(self):
        self.screen.fill((0, 150, 255))

        for pipe in self.pipes:
            gap_y = pipe["gap_y"]
            gap_h = pipe["gap_height"]
            x = pipe["x"]
            w = pipe["width"]

            top_rect = pygame.Rect(x, 0, w, gap_y - gap_h // 2)
            bottom_rect = pygame.Rect(
                x,
                gap_y + gap_h // 2,
                w,
                self.HEIGHT - (gap_y + gap_h // 2),
            )

            pygame.draw.rect(self.screen, (0, 255, 0), top_rect)
            pygame.draw.rect(self.screen, (0, 255, 0), bottom_rect)

        # Raycast + HUD remain unchanged (your existing code)
        # ----------------------------------------------------

        nearest_pipe = None
        min_dx = float("inf")
        for pipe in self.pipes:
            dx = pipe["x"] + pipe["width"] - self.BIRD_X
            if dx >= 0 and dx < min_dx:
                min_dx = dx
                nearest_pipe = pipe

        if nearest_pipe is not None:
            gap_center_y = nearest_pipe["gap_y"]
            gap_top = gap_center_y - nearest_pipe["gap_height"] / 2
            gap_bottom = gap_center_y + nearest_pipe["gap_height"] / 2
            pipe_x = nearest_pipe["x"]

            bird_pos = (int(self.BIRD_X), int(self.bird_y))

            # Center ray
            pygame.draw.line(self.screen, (255, 255, 0), bird_pos,
                             (int(pipe_x), int(gap_center_y)), 2)
            pygame.draw.circle(self.screen, (255, 120, 0),
                               (int(pipe_x), int(gap_center_y)), 5)

            # Top ray
            pygame.draw.line(self.screen, (255, 0, 0), bird_pos,
                             (int(pipe_x), int(gap_top)), 2)
            pygame.draw.circle(self.screen, (255, 0, 0),
                               (int(pipe_x), int(gap_top)), 4)

            # Bottom ray
            pygame.draw.line(self.screen, (0, 200, 255), bird_pos,
                             (int(pipe_x), int(gap_bottom)), 2)
            pygame.draw.circle(self.screen, (0, 200, 255),
                               (int(pipe_x), int(gap_bottom)), 4)

            small_font = pygame.font.SysFont(None, 16)
            dist_to_top = gap_top - self.bird_y
            dist_to_bottom = gap_bottom - self.bird_y
            dist_to_center = gap_center_y - self.bird_y

            self.screen.blit(small_font.render(f"T:{dist_to_top:.0f}", True, (255, 0, 0)),
                             (pipe_x + 6, gap_top - 10))
            self.screen.blit(small_font.render(f"C:{dist_to_center:.0f}", True, (255, 255, 0)),
                             (pipe_x + 6, gap_center_y - 5))
            self.screen.blit(small_font.render(f"B:{dist_to_bottom:.0f}", True, (0, 200, 255)),
                             (pipe_x + 6, gap_bottom))

        pygame.draw.circle(
            self.screen,
            (255, 255, 0),
            (int(self.BIRD_X), int(self.bird_y)),
            self.BIRD_RADIUS,
        )

        if nearest_pipe:
            distance_to_pipe = nearest_pipe["x"] - self.BIRD_X
            gap_offset = gap_center_y - self.bird_y

            hud_font = pygame.font.SysFont(None, 18)
            lines = [
                f"S: {self.score}",
                f"V: {self.bird_vel:.1f}",
                f"D: {distance_to_pipe:.0f}",
                f"Off: {gap_offset:.0f}",
            ]
            y = 6
            for line in lines:
                self.screen.blit(hud_font.render(
                    line, True, (255, 255, 255)), (8, y))
                y += 14

    # ------------------------------------------------------------
    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None


# ------------------------------------------------------------
# MANUAL TEST
# ------------------------------------------------------------
if __name__ == "__main__":
    env = FlappyBirdSimpleEnv(render_mode="human")
    obs, info = env.reset()
    while True:
        obs, reward, terminated, truncated, info = env.step(0)
        if terminated:
            env.reset()
