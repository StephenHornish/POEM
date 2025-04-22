import os
import time
import numpy as np
import gymnasium as gym
import torch
import matplotlib.pyplot as plt
from poem_model import POEM

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

TRAIN = True
LOG_DIR = os.path.join("trained_models", "poem_tuned_run_mountaincar")
os.makedirs(LOG_DIR, exist_ok=True)

TIMESTEPS = 200000
EVAL_EPISODES = 10

# ---------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------
def train_and_evaluate(timesteps, eval_episodes, run_dir):
    os.makedirs(run_dir, exist_ok=True)
    env = gym.make("MountainCarContinuous-v0")

    model = POEM(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-06,
        clip_range=0.1,
        gae_lambda=0.9,
        batch_size=256,
        n_epochs=3,
        n_steps=512,
        ent_coef=0.0,
        vf_coef=1,
        #POEM ONLY
        kl_threshold=0.1,
        sigma_min=0.08,
        sigma_max=0.15,
        beta=0.7,
        lambda_diversity=0.05,
        device="cpu",
    )

    start_time = time.time()
    model.learn(total_timesteps=timesteps)
    train_time = time.time() - start_time

    model_path = os.path.join(run_dir, "model.zip")
    model.save(model_path)

    avg_reward = evaluate_model(model, eval_episodes, run_dir)
    print(f"Trained {timesteps} steps in {train_time:.2f}s, avg eval reward={avg_reward:.2f}")
    return avg_reward

# ---------------------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------------------
def evaluate_model(model, eval_episodes, save_dir):
    
    eval_env = gym.make("MountainCarContinuous-v0", render_mode="human")
    max_steps = eval_env.spec.max_episode_steps
    rewards = []
    steps = 0
    for _ in range(eval_episodes):
        obs, _ = eval_env.reset()
        done = False
        ep_reward = 0.0
        while not done and steps < max_steps:
            with torch.no_grad():
                action, _ = model.predict(obs)
            obs, reward, done, _, _ = eval_env.step(action)
            ep_reward += reward
            steps +=1 
        rewards.append(ep_reward)

    eval_env.close()

    avg_reward = np.mean(rewards)
    np.savetxt(os.path.join(save_dir, "eval_rewards.csv"), rewards, delimiter=",")

    plt.plot(rewards)
    plt.title("POEM Training Performance")
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
    env = gym.make("MountainCarContinuous-v0", render_mode="human")
    model = POEM.load(model_path, env=env)
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
