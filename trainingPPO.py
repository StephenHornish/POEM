import gymnasium as gym
import numpy as np
import pygame
import cv2
import matplotlib.pyplot as plt
import os
from stable_baselines3 import PPO
import time

start_time = time.time() 

# Create environment
env = gym.make("CarRacing-v3")
# Create PPO model with TensorBoard logging
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="logs/ppo_car_racing")
model.learn(total_timesteps=100000)
print("Total time taken for training: ", time.time() - start_time)
rewards_per_episode = []


# Create test environment with human rendering
test_env = gym.make("CarRacing-v3", render_mode="human")
num_episodes = 5

for ep in range(num_episodes):
    obs, _ = test_env.reset()
    # print(test_env.observation_space)
    total_reward = 0
    done = False

    while not done:
        action, _ = model.predict(obs)
        cv2.imwrite("frame.png", cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
        obs, reward, done, _, info = test_env.step(action)
        # print(obs)
        total_reward += reward

    rewards_per_episode.append(total_reward)
    print(f"Episode {ep+1}: Reward = {total_reward}")

# Close test environment when finished
test_env.close()


print("Total time taken for test+train: ", time.time() - start_time)
# Save rewards with unique filename
filename = "ppo_rewards.csv"
if os.path.exists(filename):
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = f"{base}_{counter}{ext}"
    
    while os.path.exists(new_filename):
        counter += 1
        new_filename = f"{base}_{counter}{ext}"
    
    filename = new_filename

np.savetxt(filename, rewards_per_episode, delimiter=",")
print(f"Saved rewards to {filename}")

# Plot Training Performance
plt.plot(rewards_per_episode)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("PPO CarRacing Training Performance")
plt.show()

# Save Model
model.save("ppo_car_racing")
env.close()



