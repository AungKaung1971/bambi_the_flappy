from env.flappy_bird_simple import FlappyBirdSimpleEnv
import pygame

env = FlappyBirdSimpleEnv(render_mode="human")
obs, info = env.reset()

running = True
while running:
    # Take no action
    obs, reward, terminated, truncated, info = env.step(0)

    # If episode ends, reset
    if terminated:
        obs, info = env.reset()

    # Keep pygame window alive
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break

env.close()
