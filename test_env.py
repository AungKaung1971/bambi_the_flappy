from env.flappy_bird_simple import FlappyBirdSimpleEnv
import pygame

env = FlappyBirdSimpleEnv(render_mode="human")
obs, info = env.reset()

running = True
while running:
    obs, reward, terminated, truncated, info = env.step(0)

    if terminated:
        obs, info = env.reset()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break

env.close()
