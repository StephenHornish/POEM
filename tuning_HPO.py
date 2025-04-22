import os
import gymnasium as gym
import torch
import optuna
from stable_baselines3 import PPO
from poem_model import POEM  
import optuna.visualization.matplotlib as vis
import joblib
import optuna.visualization as vis
import argparse

# Command-line argument parsing
parser = argparse.ArgumentParser(description="Optuna Hyperparameter Tuning")
#
#python tuning_HPO.py --model poem --env lander --trials 150 --timestep 100000
#
parser.add_argument('--model', type=str, required=True, help="Model type: 'ppo' or 'poem'")
parser.add_argument('--trials', type=int, default=200, help="Number of Optuna trials")
parser.add_argument('--env', type=str, required=True, help="Environment: 'car' , 'lander','bipedal', or 'cart' ")
parser.add_argument('--timestep', type=int, default=100000, help="Total training timesteps (default: 100000)")
device = "cpu"
print(f"Using device: {device}")



args = parser.parse_args()

# ----------------------------------------
# Validate and map input values
# ----------------------------------------

# Available model and environment mappings
valid_models = {
    "ppo": "PPO",
    "poem": "POEM"
}

valid_envs = {
    "car": "CarRacing-v3",
    "lander": "LunarLander-v3",
    "bipedal":"BipedalWalker-v3",
    "cart":"MountainCarContinuous-v0"
}

# Normalize to lowercase
model_input = args.model.lower()
env_input = args.env.lower()

# Map model
if model_input in valid_models:
    MODEL_NAME = valid_models[model_input]
else:
    raise ValueError(f"Unknown model type '{args.model}'. Supported options: {list(valid_models.keys())}")

# Map environment
if env_input in valid_envs:
    ENV = valid_envs[env_input]
else:
    raise ValueError(f"Unknown environment '{args.env}'. Supported options: {list(valid_envs.keys())}")

# ----------------------------------------
# Config values
# ----------------------------------------

TRIALS = args.trials
TIMESTEP = args.timestep
SAVE_DIR = "optuna_"



param_grid = {
    "learning_rate": [0.000003, 0.00003, 0.0003, 0.003],
    "clip_range": [0.1, 0.2, 0.3],
    "ent_coef": [0, 0.001, 0.01],          
    "gae_lambda": [0.9, 1.0],
    "batch_size": [32, 64, 128, 256],
    "n_epochs": [3,6,9,12,15,18],
    "n_steps": [256,512,1024,2048],
    "vf_coef": [0.5,0.7, 1],

    #ONLY FOR POEM
    "kl_threshold": [0.025,0.05,0.075,0.1,0.15,0.2,0.25],    
    "sigma_min": [0.01,0.03,0.06,0.08],           
    "sigma_max": [0.1,0.15,0.2,0.25] ,             
    "beta": [0.7,0.8,0.9,1.0],                       
    "lambda_diversity":[0.025,0.05,0.075,0.1,0.15,0.2,0.25],                           
}

def optimize(trial):
    #  creating sample parameter
    
    params = {
        key: trial.suggest_categorical(key, param_grid[key])
        for key in [
            "learning_rate", "clip_range", "ent_coef", "gae_lambda", "batch_size",
             "n_epochs", "n_steps",  "vf_coef"
        ]
    }
    # If POEM, add extra parameters
    if MODEL_NAME == "POEM":
        params.update({
            key: trial.suggest_categorical(key, param_grid[key])
            for key in [
                "kl_threshold", "sigma_min", "sigma_max",
                "beta", "lambda_diversity"
            ]
        })

    if(ENV == "BipedalWalker-v3" or ENV == "MountainCarContinuous-v0" ):
        env = gym.make(ENV)
    else:
        env = gym.make(ENV, continuous=True)

    if MODEL_NAME == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            learning_rate=params["learning_rate"],
            clip_range=params["clip_range"],
            gae_lambda=params["gae_lambda"],
            batch_size=params["batch_size"],
            n_epochs=params["n_epochs"],
            n_steps=params["n_steps"],
            ent_coef=params["ent_coef"],     
            vf_coef=params["vf_coef"],           
            device = device,
        )
    else:
        model = POEM(
            "MlpPolicy",
            env,
            verbose=0,
            learning_rate=params["learning_rate"],
            clip_range=params["clip_range"],
            gae_lambda=params["gae_lambda"],
            batch_size=params["batch_size"],
            n_epochs=params["n_epochs"],
            n_steps=params["n_steps"],
            ent_coef=params["ent_coef"],         
            vf_coef=params["vf_coef"],
            kl_threshold = params["kl_threshold"],        
            sigma_min = params["sigma_min"],            
            sigma_max = params["sigma_max"],              
            beta = params["beta"],                     
            lambda_diversity = params["lambda_diversity"],          
            device = device,
        )

    try:
        model.learn(total_timesteps=TIMESTEP)
    except ValueError as e:
        if "nan" in str(e).lower():
            raise optuna.TrialPruned()
        else:
            raise e

    final_mean_reward = evaluate(model, env)
    return final_mean_reward  #optuna set to maximize reward 


def evaluate(model, env, n_episodes=4):
    max_steps = env.spec.max_episode_steps
    rewards = []
    for _ in range(n_episodes):
        done = False
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            steps += 1 
        rewards.append(total_reward)
    return sum(rewards) / len(rewards)

if __name__ == "__main__":
    SAVE_PATH = os.path.join("tuning_hpo", SAVE_DIR + MODEL_NAME + "_" + ENV)
    os.makedirs(SAVE_PATH, exist_ok=True)

    study = optuna.create_study(direction='maximize')
    study.optimize(optimize, n_trials=TRIALS)

    # Save best params to file
    with open(os.path.join(SAVE_PATH, "optuna_tuning.txt"), "w") as f:
        f.write("Best Hyperparameters:\n")
        for key, value in study.best_params.items():
            f.write(f"{key}: {value}\n")
        f.write(f"\nBest Value (mean reward): {study.best_value}\n")

    joblib.dump(study, os.path.join(SAVE_PATH, "optuna_tuning.pkl"))

    print('Best ' + model_input + ' Hyperparameters for ' + env_input + ": ", study.best_params)

    # Plot 1: Optimization history
    fig1 = vis.plot_optimization_history(study)
    fig1.update_layout(
        title=MODEL_NAME + " Optuna Optimization Progress " + ENV,
        xaxis_title="Trial Number",
        yaxis_title="Mean Reward",
        font=dict(size=16)
    )
    fig1.write_html(os.path.join(SAVE_PATH, "optimization_history.html"))

    # Plot 2: Parameter importances
    fig2 = vis.plot_param_importances(study)
    fig2.update_layout(
        title=MODEL_NAME + " Hyperparameter Importance " + ENV,
        xaxis_title="Importance Score",
        font=dict(size=16)
    )
    fig2.write_html(os.path.join(SAVE_PATH, "param_importances.html"))

    # Plot 3: Parallel coordinate
    fig3 = vis.plot_parallel_coordinate(study)
    fig3.update_layout(
        title=MODEL_NAME + " Parallel Coordinate " + ENV,
        font=dict(size=16)
    )
    fig3.write_html(os.path.join(SAVE_PATH, "parallel_coordinate.html"))


