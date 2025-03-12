import gymnasium as gym
import numpy as np
import pygame
import cv2
import matplotlib.pyplot as plt
import os
from stable_baselines3 import PPO



# Create environment
env = gym.make("CarRacing-v3",render_mode = "human")
# Create PPO model with TensorBoard logging
model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="logs/ppo_car_racing")
rewards_per_episode = []
num_episodes = 100
for ep in range(num_episodes):
    obs, _ = env.reset() #resets the enviornment to its initial state and states the simulation
    #general nature of observtion
    print(env.observation_space)
     #Box(0, 255, (96, 96, 3), uint8)
     #each observation is a single 96×96 RGB frame with values ranging from 0 to 255 (standard for 8-bit images)
    total_reward = 0
    done = False

    while not done:
        action, _ = model.predict(obs) #model is given an observation it preicts based on/Get the policy action from an observation 
        cv2.imwrite("frame.png", cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
        obs, reward, done, _, _ = env.step(action) #return self.state, step_reward, terminated/bool, truncated, info
        total_reward += reward #tracking the total reward for this episode

    rewards_per_episode.append(total_reward) #totals teh rewards 
    print(f"Episode {ep+1}: Reward = {total_reward}")

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

