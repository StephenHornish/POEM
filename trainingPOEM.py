import gymnasium as gym
import numpy as np
import pygame
import cv2
import matplotlib.pyplot as plt
import os
from poem_model import POEM
import time

start_time = time.time() 

TEST_MODEL = True


if not TEST_MODEL:
    # Create environment
    env = gym.make("CarRacing-v3")
    # Create PPO model with TensorBoard logging
    model = POEM("MlpPolicy", env, verbose=1, tensorboard_log="logs/POEM_car_racing")
    model.learn(total_timesteps=100000)
    # Save Model
    model.save("POEM_car_racing")
    print("Total time taken for training: ", time.time() - start_time)
    


    # Create test environment with human rendering
    test_env = gym.make("CarRacing-v3", render_mode="human")
    num_episodes = 30
    rewards_per_episode = []
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
    filename = "POEM_rewards.csv"
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
    plt.title("POEM CarRacing Training Performance")
    plt.show()

    # Save Model
    model.save("POEM_car_racing")
    env.close()

else:
    # Load the trained model
    model = POEM.load("POEM_car_racing")
    
    # Create test environment with human rendering
    test_env = gym.make("CarRacing-v3", render_mode="human")
    num_episodes = 30

    rewards_per_episode = []
    for ep in range(num_episodes):
        ep_start = time.time()
        obs, _ = test_env.reset()
        # print(test_env.observation_space)
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
    filename = "POEM_test_rewards.csv"
    if os.path.exists(filename):
        base, ext = os.path.splitext(filename)
        counter = 1
        new_filename = f"{base}_{counter}{ext}"
        
        while os.path.exists(new_filename):
            counter += 1
            new_filename = f"{base}_{counter}{ext}"
            counter += 1
            new_filename = f"{base}_{counter}{ext}"
        
        filename = new_filename
    np.savetxt(filename, rewards_per_episode, delimiter=",")
    print(f"Saved rewards to {filename}")
    # Plot Training Performance
    plt.plot(rewards_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("POEM CarRacing Training Performance")
    plt.show()

