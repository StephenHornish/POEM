import os
import time
import numpy as np
import gymnasium as gym
import torch
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Specify which model to use: "POEM" or "PPO"
# -----------------------------------------------------------------------------
MODEL_TYPE = "POEM"  # Change to "PPO" to use PPO

if MODEL_TYPE == "POEM":
    from poem_model import POEM
elif MODEL_TYPE == "PPO":
    from stable_baselines3 import PPO
else:
    raise ValueError("Unsupported MODEL_TYPE. Please choose 'POEM' or 'PPO'.")

# These will be used for training
best_params = {
    "POEM": {
        "learning_rate": 0.0003,
        "sigma_min": 0.01,
        "sigma_max": 0.1,
        "lambda_diversity": 0.1
    },
    "PPO": {
        "learning_rate": 0.0001,
        "lambda_diversity": 0.1  # This is mapped to PPO's entropy coefficient (ent_coef)
    }
}

params = best_params[MODEL_TYPE]

log_dirs = {
    "POEM": "poem_tuned_run",
    "PPO": "PPO_final_best_run"
}
FINAL_LOG_DIR = log_dirs[MODEL_TYPE]
os.makedirs(FINAL_LOG_DIR, exist_ok=True)

# Training parameters
LONG_TRAINING_TIMESTEPS = 500000 #500,000 seems to be drivable
LONG_TRAINING_EVAL_EPISODES = 10

# -----------------------------------------------------------------------------
# Training and evaluation function
# -----------------------------------------------------------------------------
def train_and_evaluate(params, timesteps, eval_episodes, run_dir):
    os.makedirs(run_dir, exist_ok=True)
    env = gym.make("CarRacing-v3")
    
    # Instantiate model with model-specific parameters
    if MODEL_TYPE == "POEM":
        model = POEM(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=params["learning_rate"],
            kl_threshold=0.1,  # fixed value for POEM
            sigma_min=params["sigma_min"],
            sigma_max=params["sigma_max"],
            lambda_diversity=params["lambda_diversity"],
            tensorboard_log=os.path.join(run_dir, "tensorboard"),
        )
    elif MODEL_TYPE == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=params["learning_rate"],
            ent_coef=params["lambda_diversity"],  # mapping to PPO's entropy coefficient
            # gamma=0.99,
            # gae_lambda=0.95,
            # clip_range=0.2,
        )
    
    start_time = time.time()
    model.learn(total_timesteps=timesteps)
    train_time = time.time() - start_time

    # Save the model
    model_path = os.path.join(run_dir, "model.zip")
    model.save(model_path)
    
    # Evaluate the trained model
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

# -----------------------------------------------------------------------------
# Load and evaluate function
# -----------------------------------------------------------------------------
def load_and_evaluate(model_path, eval_episodes):
    eval_env = gym.make("CarRacing-v3", render_mode="human")
    
    if MODEL_TYPE == "POEM":
        model = POEM.load(model_path, env=eval_env)
    elif MODEL_TYPE == "PPO":
        model = PPO.load(model_path, env=eval_env)
        
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
    reward_csv = os.path.join(FINAL_LOG_DIR, "eval_rewards.csv")
    np.savetxt(reward_csv, rewards, delimiter=",")
    
    print(f"Avg eval reward after loading model: {avg_reward:.2f}")
    
    # Plot and save training performance
    plt.plot(rewards)
    plt.title(f"{MODEL_TYPE} Training Performance")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig(os.path.join(FINAL_LOG_DIR, "training_performance.png"))
    plt.close()
    print(f"Training performance plot saved to {FINAL_LOG_DIR}/training_performance.png")
    
    return avg_reward



if __name__ == "__main__":
    TRAIN = False  # Set to True to train; otherwise the script will load and evaluate the saved model
    if TRAIN:
        avg_final = train_and_evaluate(
            params,
            LONG_TRAINING_TIMESTEPS,
            LONG_TRAINING_EVAL_EPISODES,
            FINAL_LOG_DIR
        )
        print(f"Final average reward = {avg_final:.2f} (see {FINAL_LOG_DIR}/eval_rewards.csv)")
    else:
        model_path = os.path.join(FINAL_LOG_DIR, "model.zip")
        avg_final = load_and_evaluate(model_path, LONG_TRAINING_EVAL_EPISODES)
        print(f"Final average reward after loading model = {avg_final:.2f} (see {FINAL_LOG_DIR}/eval_rewards.csv)")
