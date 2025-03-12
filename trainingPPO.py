import gymnasium as gym
import numpy as np
import pygame
import cv2
import matplotlib.pyplot as plt
import os
from stable_baselines3 import PPO
import sys
sys.path.insert(0, r"C:\Users\horni\Projects\POEM")

from car_racing import CarRacing  # Now it will load from POEM, not gymnasium
from gymnasium.envs.box2d import car_dynamics
Car = car_dynamics.Car



# Create environment
env = gym.make("CarRacing-v3",render_mode = "human")
# Create PPO model with TensorBoard logging
model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="logs/ppo_car_racing")

# Store rewards
rewards_per_episode = []
def detect_color_change(image_observation, sensor_location, direction='horizontal', max_distance=78):
    # Validate if the image is loaded
    if image_observation is None:
        return "Error: No Observation image"

    start_x, start_y = sensor_location

    # Convert to grayscale for easier processing
    gray_image = cv2.cvtColor(image_observation, cv2.COLOR_BGR2GRAY)

    # Get initial pixel color (convert to int to avoid uint8 overflow issues)
    initial_color = int(gray_image[start_y, start_x])  

    # Define movement (horizontal or vertical)
    if direction == 'left_horizontal':
        delta_x, delta_y = -1, 0  # Move left along x-axis
    elif direction == 'right_horizontal':
        delta_x, delta_y = 1, 0  # Move right along x-axis
    elif direction == 'right_diagonal':
        delta_x, delta_y = 1, -1  # Move diagonally right-up
    elif direction == 'left_diagonal':
        delta_x, delta_y = -1, -1  # Move diagonally left-up
    elif direction == 'vertical':
        delta_x, delta_y = 0, -1  # Move up along y-axis
    else:
        return "Error: Direction given is invalid."

    # Iterate to find the color change
    threshold = 30  # Define a color change threshold
    for distance in range(1, max_distance):
        new_x = start_x + distance * delta_x
        new_y = start_y + distance * delta_y

        # Check if we are still within the image bounds
        if new_x < 0 or new_y < 0 or new_x >= gray_image.shape[1] or new_y >= gray_image.shape[0]:
            return max_distance

        # Get new pixel color and convert to int
        new_color = int(gray_image[new_y, new_x])

        # If color has changed, mark the location
        if abs(new_color - initial_color) > threshold:
            """
            print(f"New Color: {new_color}, Initial Color: {initial_color}")

            # Draw a blue dot at the detected point
            cv2.circle(image_observation, (new_x, new_y), radius=5, color=(255, 0, 0), thickness=-1)  # Blue dot

            # Save the modified image
            cv2.imwrite(output_path, image_observation)

            # Display the modified image in Colab
            cv2_imshow(image_observation)
            print(f"Color change detected at {distance} pixels away at ({new_x}, {new_y}). Image saved as {output_path}.")
            """
            return distance

    return max_distance

def read_sensors(image):
    if image is None:
        return "Error: No image provided to read_sensors"

    sensor_location = (49, 78)
    sensor_reading = []
    sensor_directions = ["left_horizontal", "right_horizontal", "right_diagonal", "left_diagonal", "vertical"]

    for direction in sensor_directions: 
        result = detect_color_change(image, sensor_location, direction)
        sensor_reading.append(result)

    return sensor_reading
# Train agent and log rewards
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
        print(read_sensors(obs))
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

