import os
import time
import numpy as np
import gymnasium as gym
import torch
import matplotlib.pyplot as plt
from poem_model import POEM

# ---------------------------------------------------------------------
# POEM Configuration
# ---------------------------------------------------------------------
LEARNING_RATE = 0.0003
SIGMA_MIN = 0.01
SIGMA_MAX = 0.1
LAMBDA_DIVERSITY = 0.1

TRAIN = True  # Set to False to only evaluate a saved model
LOG_DIR = "poem_tuned_run_lander"
os.makedirs(LOG_DIR, exist_ok=True)

TIMESTEPS = 100000
EVAL_EPISODES = 5

# ---------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------
def train_and_evaluate(timesteps, eval_episodes, run_dir):
    env = gym.make("LunarLander-v3",continuous=True)

    model = POEM(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=LEARNING_RATE,
        kl_threshold=0.1,
        sigma_min=SIGMA_MIN,
        sigma_max=SIGMA_MAX,
        lambda_diversity=LAMBDA_DIVERSITY,
        tensorboard_log=os.path.join(run_dir, "tensorboard"),
    )

    start_time = time.time()
    model.learn(total_timesteps=timesteps)
    train_time = time.time() - start_time

    model_path = os.path.join(run_dir, "model.zip")
    model.save(model_path)

    #avg_reward = evaluate_model(model, eval_episodes, run_dir)
    #print(f"Trained {timesteps} steps in {train_time:.2f}s, avg eval reward={avg_reward:.2f}")
   # return avg_reward

# ---------------------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------------------
def evaluate_model(model, eval_episodes, save_dir):
    eval_env = gym.make("LunarLander-v3",continuous=True, render_mode="human")
    rewards = []

    for ep in range(eval_episodes):
        ep_start = time.time()
        obs, _ = eval_env.reset()
        total_reward = 0
        done = False
        

        while not done:
            if time.time() - ep_start >= 5.0:
                break
            with torch.no_grad():
                action, _ = model.predict(obs)
            obs, reward, done, _, _ = eval_env.step(action)
            total_reward += reward

        rewards.append(total_reward)
        print(f"Episode {ep+1}: Reward = {total_reward:.2f}")
        print(f"Episode {ep+1} Time: {time.time() - ep_start:.2f} seconds")

    eval_env.close()

    # Save rewards
    reward_csv = os.path.join(save_dir, "eval_rewards.csv")
    np.savetxt(reward_csv, rewards, delimiter=",")
    print(f"Saved episode rewards to {reward_csv}")

    # Plot and save performance
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("POEM LunarLander Evaluation")
    plt.savefig(os.path.join(save_dir, "training_performance.png"))
    plt.close()
    print(f"Saved performance plot to {save_dir}/training_performance.png")

    return np.mean(rewards)

# ---------------------------------------------------------------------
# Load and evaluate a saved model
# ---------------------------------------------------------------------
def load_and_evaluate_model(model_path, eval_episodes, run_dir):
    env = gym.make("LunarLander-v3",continuous=True, render_mode="human")
    model = POEM.load(model_path, env=env)
    return evaluate_model(model, eval_episodes, run_dir)

# ---------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------
if __name__ == "__main__":
    if TRAIN:
        avg = train_and_evaluate(TIMESTEPS, EVAL_EPISODES, LOG_DIR)
    else:
        model_path = os.path.join(LOG_DIR, "model.zip")
        avg = load_and_evaluate_model(model_path, EVAL_EPISODES, LOG_DIR)

    print(f"Final avg reward = {avg:.2f} (see {LOG_DIR}/eval_rewards.csv)")
