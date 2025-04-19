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

TRAIN = True  # Set to False to only evaluate a saved model
LOG_DIR = os.path.join("trained_models", "poem_tuned_run_lander")
os.makedirs(LOG_DIR, exist_ok=True)

TIMESTEPS = 250000
EVAL_EPISODES = 1

# ---------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------
def train_and_evaluate(timesteps, eval_episodes, run_dir):
    env = gym.make("LunarLander-v3",continuous=True)

    model = POEM(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.003,
        clip_range=0.2,
        gae_lambda=1.0,
        batch_size=256,
        n_epochs=30,
        n_steps=512,
        normalize_advantage=True,
        target_kl=0.003,
        ent_coef=0.01,
        vf_coef=0.5,
        kl_threshold=0.25,
        sigma_min=0.01,
        sigma_max=0.15,
        beta=0.9,
        lambda_diversity=0.05,
        policy_kwargs=dict(log_std_init=1),
        device = "cpu",
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
    eval_env = gym.make("LunarLander-v3",continuous=True, render_mode="human")
    rewards = []

    for ep in range(eval_episodes):
        ep_start = time.time()
        obs, _ = eval_env.reset()
        total_reward = 0
        done = False
        step = 0 
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
