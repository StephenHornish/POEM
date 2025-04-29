import os
import time
import numpy as np
import gymnasium as gym
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# ---------------------------------------------------------------------
# PPO Configuration
# ---------------------------------------------------------------------


TRAIN = True  # Set to False to skip training and evaluate a saved model
LOG_DIR = os.path.join("trained_models", "ppo_tuned_run_bipedal")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

TIMESTEPS = 1500000
EVAL_EPISODES = 10
#decorator
def count_forward_passes(method):
        def wrapper(*args, **kwargs):
            wrapper.calls += 1
            return method(*args, **kwargs)
        wrapper.calls = 0
        return wrapper
# ---------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------
def train_and_evaluate(timesteps, eval_episodes, run_dir):
    env = gym.make("BipedalWalker-v3")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.003,
        clip_range=0.1,
        gae_lambda=0.9,
        batch_size=128,
        n_epochs=9,
        n_steps=2048,
        vf_coef=0.7,
        tensorboard_log=os.path.join(run_dir, "tensorboard"),
        device = "cpu",
    )

    model.policy.forward = count_forward_passes(model.policy.forward)

    start_time = time.time()
    model.learn(total_timesteps=timesteps)
    train_time = time.time() - start_time

    # Retrieve forward pass count
    forward_pass_count = model.policy.forward.calls
    print(f"Total forward passes during training: {forward_pass_count}")

    # Save forward pass count and traintime to txt file
    with open(os.path.join(run_dir, "forward_pass_count.txt"), "w") as f:
        f.write(str(forward_pass_count))
        f.write(f"Training Time (seconds): {train_time:.2f}\n")

    # Save model
    model_path = os.path.join(run_dir, "model.zip")
    model.save(model_path)

    avg_reward = evaluate_model(model, eval_episodes, run_dir)
    print(f"Trained {timesteps} steps in {train_time:.2f}s, avg eval reward={avg_reward:.2f}")
    return avg_reward

# ---------------------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------------------
def evaluate_model(model, eval_episodes, save_dir):
    eval_env = gym.make("BipedalWalker-v3", render_mode="human")
    rewards = []

    for ep in range(eval_episodes):
        ep_start = time.time()
        obs, _ = eval_env.reset()
        total_reward = 0
        done = False
        while not done:
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
    plt.title("PPO LunarLander Evaluation")
    plt.savefig(os.path.join(save_dir, "training_performance.png"))
    plt.close()
    print(f"Saved performance plot to {save_dir}/training_performance.png")

    return np.mean(rewards)

# ---------------------------------------------------------------------
# Load and evaluate a saved model
# ---------------------------------------------------------------------
def load_and_evaluate_model(model_path, eval_episodes, run_dir):
    env = gym.make("BipedalWalker-v3", render_mode="human")
    model = PPO.load(model_path, env=env)
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

