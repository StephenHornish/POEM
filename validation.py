import gymnasium as gym
import pygame
import numpy as np

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption("Car Racing Manual Control")

# Define control keys
KEY_MAP = {
    pygame.K_UP: np.array([0.0, 1.0, 0.0]),   # Accelerate
    pygame.K_DOWN: np.array([0.0, 0.0, 0.8]), # Brake
    pygame.K_LEFT: np.array([-1.0, 0.0, 0.0]), # Left
    pygame.K_RIGHT: np.array([1.0, 0.0, 0.0])  # Right
}

# Create the environment
env = gym.make("CarRacing-v3", render_mode="human")
observation, info = env.reset()

running = True
action = np.array([0.0, 0.0, 0.0])  # [steer, gas, brake]

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False  # Close the game
        elif event.type == pygame.KEYDOWN:
            if event.key in KEY_MAP:
                action += KEY_MAP[event.key]  # Apply movement
        elif event.type == pygame.KEYUP:
            if event.key in KEY_MAP:
                action -= KEY_MAP[event.key]  # Stop movement

    # Step the environment with the current action
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
pygame.quit()
