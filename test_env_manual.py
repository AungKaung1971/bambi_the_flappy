from env.flappy_bird_simple import FlappyBirdSimpleEnv
import pygame

env = FlappyBirdSimpleEnv(render_mode="human")
obs, info = env.reset()

# ---- IMPORTANT FIX ----
# Render one frame so pygame initializes the display BEFORE we read events
env.step(0)

running = True
action = 0

while running:
    # Handle input
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                action = 1
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_SPACE:
                action = 0

    # Step environment with current action
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated:
        obs, info = env.reset()
        env.step(0)  # initialize render again after reset

env.close()
