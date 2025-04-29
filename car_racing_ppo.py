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


TRAIN = True  # Set to True to train a new model
LOG_DIR = os.path.join("trained_models", "ppo_tuned_run_car")
os.makedirs(LOG_DIR, exist_ok=True)

TIMESTEPS = 500000
EVAL_EPISODES = 1

#Decorator 
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
    os.makedirs(run_dir, exist_ok=True)
    env = gym.make("CarRacing-v3",continuous=True)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.003,
        clip_range=0.1,
        ent_coef=0.01,
        gae_lambda=0.9,
        batch_size=128,
        n_epochs=18,
        n_steps=512,
        vf_coef=1,
        tensorboard_log=os.path.join(run_dir, "tensorboard"),
        device="cpu",
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
    eval_env = gym.make("CarRacing-v3",continuous=True, render_mode="human")
    rewards = []

    for _ in range(eval_episodes):
        obs, _ = eval_env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            with torch.no_grad():
                action, _ = model.predict(obs)
            obs, reward, done, _, _ = eval_env.step(action)
            ep_reward += reward
        rewards.append(ep_reward)

    eval_env.close()

    avg_reward = np.mean(rewards)
    np.savetxt(os.path.join(save_dir, "eval_rewards.csv"), rewards, delimiter=",")

    plt.plot(rewards)
    plt.title("PPO Training Performance")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig(os.path.join(save_dir, "training_performance.png"))
    plt.close()

    print(f"Evaluation complete. Plot saved to {save_dir}/training_performance.png")
    return avg_reward

# ---------------------------------------------------------------------
# Load and evaluate saved model
# ---------------------------------------------------------------------
def load_and_evaluate_model(model_path, eval_episodes, run_dir):
    env = gym.make("CarRacing-v3",continuous=True, render_mode="human")
    model = PPO.load(model_path, env=env)
    avg_reward = evaluate_model(model, eval_episodes, run_dir)
    print(f"Final average reward after loading model: {avg_reward:.2f}")
    return avg_reward

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
