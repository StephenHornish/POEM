import os
import time
import numpy as np
import gymnasium as gym
import torch
import csv
import matplotlib.pyplot as plt
import random
import pandas as pd
from stable_baselines3 import PPO
from poem_model import POEM
import argparse

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
#if you want to only evalute one remove the other 
MODEL_DIRS = {
    "PPO": "ppo_tuned_run_car",
    "POEM": "poem_tuned_run_car"
}
#RESULTS_BASE_DIR = "lander_results"
RESULTS_BASE_DIR = "car_results"
#gym_environment = "LunarLander-v3"
gym_environment = "CarRacing-v3"
LONG_TRAINING_EVAL_EPISODES = 10
r_m= None


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def total_action(action, action_tracker):
    if gym_environment == "CarRacing-v3":
        direction, brake, gas = action
        if direction < 0:
            action_tracker[0] += direction
        else:
            action_tracker[1] += direction
        action_tracker[2] += brake
        action_tracker[3] += gas
    elif gym_environment == "LunarLander-v3":
        main_engine, direction = action
        # Track lateral boosters (left/right)
        if direction < -0.5:
            # Left booster fires
            throttle = (abs(direction) - 0.5) * 2  # Scale from 0 to 1
            action_tracker[0] += throttle
        elif direction > 0.5:
            # Right booster fires
            throttle = (direction - 0.5) * 2
            action_tracker[1] += throttle

        # Track main engine
        if main_engine >= 0:
            # Main engine throttle scales from 0.5 to 1.0 => 0 to 1 scaled
            throttle = 2 * (main_engine - 0.5)
            throttle = max(throttle, 0.0)  # Avoid negative values
            action_tracker[2] += throttle
    return action_tracker

def plot_trained_model(rewards, average_action_space, ep_step_rewards, model_type, save_dir):
    plt.plot(rewards)
    plt.title(f"{model_type} - Training Performance {gym_environment}")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig(os.path.join(save_dir, "training_performance.png"))
    plt.close()

    plt.figure()
    if gym_environment == "CarRacing-v3":
        labels = ["LEFT", "RIGHT", "GAS", "BRAKE"]
    elif gym_environment == "LunarLander-v3":
        labels = ["DO NOTHING", "FIRE LEFT", "FIRE MAIN", "FIRE RIGHT"]
    else:
        labels = [f"Action {i}" for i in range(len(average_action_space))]

    plt.bar(labels, average_action_space)
    plt.title(f"{model_type} - Average Action Usage {gym_environment}")
    plt.ylabel("Total Action Count")
    plt.xlabel("Action")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "average_action_space.png"))
    plt.close()

    

    for episode_idx, step_list in enumerate(ep_step_rewards):
        steps, step_rewards = zip(*step_list)
        filtered_steps = [steps[0]]
        filtered_rewards = [step_rewards[0]]

        for i in range(1, len(steps)):
            diff = step_rewards[i] - step_rewards[i - 1]
            if diff >= -50:
                filtered_steps.append(steps[i])
                filtered_rewards.append(step_rewards[i])
            else:
                # Optional: skip or interpolate based on previous value
                pass

        plt.plot(filtered_steps, filtered_rewards, label=f"Episode {episode_idx + 1}")

    plt.title(f"{model_type} Step-wise Reward per Episode {gym_environment}")
    plt.xlabel("Step")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "stepwise_rewards.png"))
    plt.close()

def load_and_evaluate(model_path, eval_episodes, model_type, save_dir):
    eval_env = gym.make(gym_environment,continuous=True, render_mode = r_m)

    if model_type == "POEM":
        model = POEM.load(model_path, env=eval_env)
    else:
        model = PPO.load(model_path, env=eval_env)

    seed = 1
    eval_env.action_space.seed(seed)
    eval_env.observation_space.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    rewards = []
    ep_step_rewards = []
    ep_action_space = []
    ep_start = time.time()

    for ep in range(eval_episodes):
        obs, _ = eval_env.reset(seed=seed)
        seed += 1
        done = False
        ep_reward = 0.0
        step = 0
        step_rewards = []
        action_space = [0, 0, 0, 0]

        while not done:
            if(gym_environment == "LunarLander-v3"):
                if step >= 600:
                    break
            with torch.no_grad():
                action, _ = model.predict(obs)
            obs, reward, done, _, _ = eval_env.step(action)
            action_space = total_action(action, action_space)
            
            ep_reward += reward
            step_rewards.append((step, ep_reward))
            step += 1

        ep_action_space.append(action_space)
        ep_step_rewards.append(step_rewards)
        rewards.append(ep_reward)

    eval_env.close()

    ep_action_space = np.array(ep_action_space)
    average_action_space = np.mean(ep_action_space, axis=0)
    if gym_environment == "CarRacing-v3":
        average_action_space[0] *= -1

    np.savetxt(os.path.join(save_dir, "average_action_space.csv"),
               average_action_space[None], delimiter=",", header="LEFT,RIGHT,MAIN,RIGHT", comments='')
    np.savetxt(os.path.join(save_dir, "eval_rewards.csv"),
               rewards, delimiter=",")

    with open(os.path.join(save_dir, "step_rewards.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Step", "EpisodeReward"])
        for episode_idx, step_list in enumerate(ep_step_rewards):
            for step, reward in step_list:
                writer.writerow([episode_idx + 1, step, reward])

    plot_trained_model(rewards, average_action_space, ep_step_rewards, model_type, save_dir)
    print(f"Avg eval reward for {model_type}: {np.mean(rewards):.2f}")
    return rewards, average_action_space

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PPO/POEM models")
    parser.add_argument("--env", choices=["lander", "car"], required=True,
                        help="Which environment to evaluate: 'lander' or 'car'")
    parser.add_argument("-r", "--human", action="store_true",
                        help="Use human render mode for visualization")
    return parser.parse_args()
# -----------------------------------------------------------------------------
# Main: Evaluate both models and compare
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    print(f"Running eval on {args.env} with render_mode={r_m}")
    r_m = "human" if args.human else None
    if args.env == "lander": 
        RESULTS_BASE_DIR = "lander_results"
        gym_environment = "LunarLander-v3"
        MODEL_DIRS = {
            "PPO": "ppo_tuned_run_lander",
            "POEM": "poem_tuned_run_lander"
        }
    elif args.env == "car":
        RESULTS_BASE_DIR = "car_results"
        gym_environment = "CarRacing-v3"
        MODEL_DIRS = {
            "PPO": "ppo_tuned_run_car",
            "POEM": "poem_tuned_run_car"
        }

    os.makedirs(RESULTS_BASE_DIR, exist_ok=True)
    all_rewards = {}
    all_actions = {}


    for model_type, model_dir in MODEL_DIRS.items():
        print(f"\n--- Evaluating {model_type} ---")
        save_dir = os.path.join(RESULTS_BASE_DIR, model_type.lower())
        os.makedirs(save_dir, exist_ok=True)

        model_path = os.path.join(model_dir, "model.zip")
        rewards,actions = load_and_evaluate(model_path, LONG_TRAINING_EVAL_EPISODES, model_type, save_dir)

        all_actions[model_type] = actions
        all_rewards[model_type] = rewards

    # -------------------------------------------------------------------------
    # Comparison Plots
    # -------------------------------------------------------------------------
    # Total reward per episode
    plt.figure()

    # Determine number of episodes (assuming all models have same number of episodes)
    num_episodes = len(next(iter(all_rewards.values())))
    num_models = len(all_rewards)
    bar_width = 0.8 / num_models  # space bars evenly within each episode group

    # X locations for episodes
    episodes = np.arange(num_episodes)

    # Plot each model's bars
    for i, (model_type, rewards) in enumerate(all_rewards.items()):
        offset = (i - num_models / 2) * bar_width + bar_width / 2
        plt.bar(episodes + offset, rewards, width=bar_width, label=model_type)

    plt.title(f"{model_type} - Total Reward per Episode {gym_environment}")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True, axis='y')
    plt.xticks(episodes, [str(i + 1) for i in episodes])
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_BASE_DIR, "comparison_total_reward_bar.png"))
    plt.close()

    # Average action usage comparison
    if(gym_environment == "CarRacing-v3"):
        labels = ["LEFT", "RIGHT", "GAS", "BRAKE"]
    else:
        labels = ["FIRE LEFT", "FIRE RIGHT","FIRE MAIN", ]
        for key in all_actions:
            all_actions[key] = all_actions[key][:3]  
    x = np.arange(len(labels))  # [0, 1, 2, 3]
    width = 0.35

    plt.figure()
    plt.bar(x - width/2, all_actions["PPO"], width, label="PPO")
    plt.bar(x + width/2, all_actions["POEM"], width, label="POEM")

    plt.title(f"{model_type} - Action Usage {gym_environment}")
    plt.xlabel("Action")
    plt.ylabel("Frequency of Action")
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_BASE_DIR, "comparison_action_bar.png"))
    plt.close()