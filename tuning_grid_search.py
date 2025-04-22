import os
import time
import itertools
import numpy as np
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from poem_model import POEM  

MODEL_NAME = "PPO"  
ENV = "LunarLander-v3"

param_grid = {
    "learning_rate": [0.000003, 0.00003, 0.0003, 0.003],
    "clip_range": [0.1, 0.2, 0.3],
    "ent_coef": [0, 0.001, 0.001, 0.01],          
    "gae_lambda": [0.9, 1.0],
    "batch_size": [32, 64, 128, 256],
    "n_epochs": [3,9,15, 21, 27],
    "n_steps": [512, 1023, 2048],
    "target_kl": [0.03, 0.003],
    "vf_coef": [0.5, 1],                         
}


if MODEL_NAME == "PPO":
    param_grid = {
        "learning_rate": [0.000003, 0.00003, 0.0003, 0.003],
        "clip_range": [0.1, 0.2, 0.3],
        "ent_coef": [0, 0.001, 0.001, 0.01],          
        "gae_lambda": [0.9, 1.0],
        "batch_size": [32, 64, 128, 256],
        "n_epochs": [3,9,15, 21, 27],
        "n_steps": [512, 1023, 2048],
        "target_kl": [0.03, 0.003],
        "vf_coef": [0.5, 1],                          
}


# Create a list of (param_dict) for each combination
keys = list(param_grid.keys())
param_combos = []
for vals in itertools.product(*(param_grid[k] for k in keys)):
    combo = dict(zip(keys, vals))
    param_combos.append(combo)


GRIDSEARCH_TIMESTEPS = 80000   # short training run for each combo
GRIDSEARCH_EVAL_EPISODES = 3    # quick evaluation after short run

# Final training once best combo is found
LONG_TRAINING_TIMESTEPS = 250000
LONG_TRAINING_EVAL_EPISODES = 10

# Logging directories
GRIDSEARCH_LOG_DIR = os.path.join("tuning_grid_search", MODEL_NAME + "_gridsearch_short_runs_" + ENV)
FINAL_LOG_DIR = os.path.join("tuning_grid_search", MODEL_NAME + "_final_best_run_" + ENV)

def train_and_evaluate(params, timesteps, eval_episodes, run_dir):
    """
    Trains a model on 'env' for 'timesteps' steps
    using hyperparams in 'params'.
    Then runs 'eval_episodes' episodes to measure average reward.
    Saves the model and returns the average reward.
    """
    os.makedirs(run_dir, exist_ok=True)
    tensorboard_dir = os.path.join(run_dir, "tensorboard")
    
    # 1) Create environment (headless/no-render)
    env = gym.make(ENV,continuous=True)

    # Instantiate model based on MODEL_NAME
    if MODEL_NAME == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=params["learning_rate"],
            clip_range=params["clip_range"],
            gae_lambda=params["gae_lambda"],
            batch_size=params["batch_size"],
            n_epochs=params["n_epochs"],
            n_steps=params["n_steps"],
            normalize_advantage=params["normalize_advantage"],
            target_kl=params["target_kl"],
            ent_coef=params["ent_coef"],     
            vf_coef=params["vf_coef"],           
            policy_kwargs=dict(log_std_init=params["log_std_init"]),
            tensorboard_log=tensorboard_dir,
        )
    else:
        model = POEM(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=params["learning_rate"],
            clip_range=params["clip_range"],
            gae_lambda=params["gae_lambda"],
            batch_size=params["batch_size"],
            n_epochs=params["n_epochs"],
            n_steps=params["n_steps"],
            normalize_advantage=params["normalize_advantage"],
            target_kl=params["target_kl"],
            ent_coef=params["ent_coef"],         
            vf_coef=params["vf_coef"],           
            policy_kwargs=dict(log_std_init=params["log_std_init"]),
            kl_threshold=0.1, 
        )
            

    # 3) Train short-run or long-run
    start_time = time.time()
    model.learn(total_timesteps=timesteps)
    train_time = time.time() - start_time
    
    # 4) Save the short-run model
    model_path = os.path.join(run_dir, "model.zip")
    model.save(model_path)

    # 5) Evaluate
    eval_env = gym.make(ENV,continuous=True)
    rewards = []
    for _ in range(eval_episodes):
        obs, _ = eval_env.reset()
        done = False
        ep_reward = 0.0
        step = 0 
        while not done:
            if step >= 1000:
                break
            with torch.no_grad():
                action, _ = model.predict(obs)
            obs, reward, done, _, _ = eval_env.step(action)
            ep_reward += reward
            step+=1
        rewards.append(ep_reward)
    eval_env.close()
    
    avg_reward = np.mean(rewards)
    
    # 6) Save the rewards and log
    reward_csv = os.path.join(run_dir, "eval_rewards.csv")
    np.savetxt(reward_csv, rewards, delimiter=",")
    
    print(f"Trained {timesteps} steps in {train_time:.2f} s, avg eval reward={avg_reward:.2f}")
    return avg_reward


if __name__ == "__main__":
    overall_start = time.time()


    best_score = -float("inf")
    best_params = None
    
    # Directory for the short-run gridsearch logs
    os.makedirs(GRIDSEARCH_LOG_DIR, exist_ok=True)

    for i, combo in enumerate(param_combos, start=1):
        # Build a name for this run
        run_name_parts = []
        for k, v in combo.items():
            run_name_parts.append(f"{k}={v}")
        run_name = f"grid_{i}_{'_'.join(run_name_parts)}"
        run_dir = os.path.join(GRIDSEARCH_LOG_DIR, run_name)

        print(f"\n=== Grid Search: Starting short run for {run_name} ===")
        avg_reward = train_and_evaluate(
            combo,
            timesteps=GRIDSEARCH_TIMESTEPS,
            eval_episodes=GRIDSEARCH_EVAL_EPISODES,
            run_dir=run_dir
        )

        # Track the best average reward so far
        if avg_reward > best_score:
            best_score = avg_reward
            best_params = combo

    print("\n=== Grid Search Complete ===")
    print(f"Best average reward = {best_score:.2f} with params = {best_params}")
    
    print("\n=== Starting Final Long Training with Best Params ===")
    os.makedirs(FINAL_LOG_DIR, exist_ok=True)
    
    final_avg_reward = train_and_evaluate(
        best_params,
        timesteps=LONG_TRAINING_TIMESTEPS,
        eval_episodes=LONG_TRAINING_EVAL_EPISODES,
        run_dir=FINAL_LOG_DIR
    )
    
    overall_time = time.time() - overall_start
    print("\n=== ALL DONE ===")
    print(f"Final run average reward = {final_avg_reward:.2f}")
    print(f"Total time: {overall_time/3600:.2f} hours")
