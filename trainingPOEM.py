import os
import time
import numpy as np
import gymnasium as gym
import torch

from poem_model import POEM

# Best hyperparams from short runs:
best_params = {
    "learning_rate": 0.0003,
    "sigma_min": 0.01,
    "sigma_max": 0.1,
    "lambda_diversity": 0.1
}

LONG_TRAINING_TIMESTEPS = 500000
LONG_TRAINING_EVAL_EPISODES = 10
FINAL_LOG_DIR = "poem_tuned_run"
os.makedirs(FINAL_LOG_DIR, exist_ok=True)

def train_and_evaluate(params, timesteps, eval_episodes, run_dir):
    os.makedirs(run_dir, exist_ok=True)
    env = gym.make("CarRacing-v3")
    model = POEM(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=params["learning_rate"],
        kl_threshold=0.1,  # or adjust if you like
        sigma_min=params["sigma_min"],
        sigma_max=params["sigma_max"],
        lambda_diversity=params["lambda_diversity"],
        tensorboard_log=os.path.join(run_dir, "tensorboard"),
    )
    start_time = time.time()
    model.learn(total_timesteps=timesteps)
    train_time = time.time() - start_time

    # Save the model
    model_path = os.path.join(run_dir, "model.zip")
    model.save(model_path)

    # Evaluate
    eval_env = gym.make("CarRacing-v3", render_mode="human")
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
    reward_csv = os.path.join(run_dir, "eval_rewards.csv")
    np.savetxt(reward_csv, rewards, delimiter=",")
    
    print(f"Trained {timesteps} steps in {train_time:.2f} s, avg eval reward={avg_reward:.2f}")
    return avg_reward

def load_and_evaluate(model_path, eval_episodes):
    env = gym.make("CarRacing-v3", render_mode="human")
    model = POEM.load(model_path, env=env)
    rewards = []
    for _ in range(eval_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            with torch.no_grad():
                action, _ = model.predict(obs)
            obs, reward, done, _, _ = env.step(action)
            ep_reward += reward
        rewards.append(ep_reward)
    env.close()
    avg_reward = np.mean(rewards)
    reward_csv = os.path.join(FINAL_LOG_DIR, "eval_rewards.csv")
    np.savetxt(reward_csv, rewards, delimiter=",")          
    print(f"Avg eval reward after loading model: {avg_reward:.2f}")
    # Plot Training Performance
    import matplotlib.pyplot as plt
    plt.plot(rewards)
    plt.title("POEM Training Performance")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig(os.path.join(FINAL_LOG_DIR, "training_performance.png"))
    plt.close()
    print(f"Training performance plot saved to {FINAL_LOG_DIR}/training_performance.png")

    return avg_reward


if __name__ == "__main__":

    TRAIN = False
    if TRAIN:
        avg_final = train_and_evaluate(
            best_params,
            LONG_TRAINING_TIMESTEPS,
            LONG_TRAINING_EVAL_EPISODES,
            FINAL_LOG_DIR
        )
        print(f"Final average reward = {avg_final:.2f} (see {FINAL_LOG_DIR}/eval_rewards.csv)")
    else:
        # Load the model and evaluate
        model_path = os.path.join(FINAL_LOG_DIR, "model.zip")
        avg_final = load_and_evaluate(model_path, LONG_TRAINING_EVAL_EPISODES)
        print(f"Final average reward after loading model = {avg_final:.2f} (see {FINAL_LOG_DIR}/eval_rewards.csv)")
