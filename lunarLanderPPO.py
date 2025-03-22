import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
from stable_baselines3 import PPO
import time

s = time.time()

# Create environment - note: render_mode can be set to "human" if you want to see a visual display.
env = gym.make("LunarLander-v3")

# Create PPO model with TensorBoard logging (update the log directory as needed)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="logs/PPO_lunar_lander")
model.learn(total_timesteps=50000)
rewards_per_episode = []
num_episodes = 10

# Create test environment with human rendering
test_env = gym.make("LunarLander-v3", render_mode="human")

for ep in range(num_episodes):
    ep_start = time.time()
    obs, _ = test_env.reset()  # Reset environment to initial state
    print("Observation space:", test_env.observation_space)
    # Expected observation space: Box(-array([...]), array([...]), (8,), float32)
    
    total_reward = 0
    done = False

    while not done:
        # Get the action from the model's policy
        action, _ = model.predict(obs)
        
        # Step the environment using the chosen action
        obs, reward, done, _, info = test_env.step(action)
        total_reward += reward

    rewards_per_episode.append(total_reward)
    print(f"Episode {ep+1}: Reward = {total_reward}")
    ep_end = time.time()
    print(f"Episode {ep+1} Time: {ep_end - ep_start:.2f} seconds")

# Close test environment
test_env.close()

# Save rewards with a unique filename
filename = "Luna_PPO_rewards.csv"
if os.path.exists(filename):
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = f"{base}_{counter}{ext}"
    while os.path.exists(new_filename):
        counter += 1
        new_filename = f"{base}_{counter}{ext}"
    filename = new_filename

print("Total time taken for training:", time.time() - s)
np.savetxt(filename, rewards_per_episode, delimiter=",")
print(f"Saved rewards to {filename}")

# Plot Training Performance
plt.plot(rewards_per_episode)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("PPO LunarLander Training Performance")
plt.show()

# Save Model
model.save("ppo_lunar_lander")
env.close()
