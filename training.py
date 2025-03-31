import os
import time
import numpy as np
import gymnasium as gym
import torch
import csv
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Specify which model to use: "POEM" or "PPO"
# Specify which gym enviornmetn to use currently support CarRacing-v3 and LunarLander-v3
# -----------------------------------------------------------------------------
MODEL_TYPE = "PPO"  # Change to "PPO" to use PPO
gym_environment = "CarRacing-v3"
if gym_environment == "CarRacing-v3":
    log_dirs = {
        "POEM": "POEM_tuned_run_car",
        "PPO": "PPO_tuned_run_car",
    }
elif gym_environment == "LunarLander-v3":
    log_dirs = {
        "POEM": "POEM_tuned_run_lander",
        "PPO": "PPO_tuned_run_lander",
    }
else:
    raise ValueError(f"Unknown environment: {gym_environment}")

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
        "learning_rate": 0.0003,
        #"lambda_diversity": 0.1,  # This is mapped to PPO's entropy coefficient (ent_coef)
        #"gamma": 0.99,
        #"gae_lambda":0.97,
        "clip_range": 0.4
    }
}

params = best_params[MODEL_TYPE]


FINAL_LOG_DIR = log_dirs[MODEL_TYPE]
os.makedirs(FINAL_LOG_DIR, exist_ok=True)

# Training parameters
if(gym_environment == "CarRacing-v3"):
    LONG_TRAINING_TIMESTEPS = 500000 #500,000 seems to be drivable
elif(gym_environment == "LunarLander-v3"):
    LONG_TRAINING_TIMESTEPS = 70000 
else:
    raise ValueError("Gym Enviornment not valid need to set up'.")

LONG_TRAINING_EVAL_EPISODES = 10

class ActionFilter:
    def __init__(self,alpha=0.01):
        self.prev_action = None
        self.alpha = alpha
    def smooth(self,action):
        if self.prev_action is None:
            self.prev_action = action 
        else: 
            self.prev_action = self.alpha * action+(1-self.alpha)*self.prev_action
        return self.prev_action

# -----------------------------------------------------------------------------
# Training and evaluation function
# -----------------------------------------------------------------------------
def train_and_evaluate(params, timesteps, eval_episodes, run_dir):
    os.makedirs(run_dir, exist_ok=True)
    env = gym.make(gym_environment,render_mode = "human")
    
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
            #ent_coef=params["lambda_diversity"],  # mapping to PPO's entropy coefficient
            # gamma=0.99,
            # gae_lambda=0.95,
             clip_range=params["clip_range"],
             tensorboard_log=os.path.join(run_dir, "tensorboard"),
        )
    
    start_time = time.time()
    model.learn(total_timesteps=timesteps)
    train_time = time.time() - start_time

    # Save the model
    model_path = os.path.join(run_dir, "model.zip")
    model.save(model_path)
    
    # Evaluate the trained model
    eval_env = gym.make(gym_environment, render_mode="human")
    rewards = []
    #smoother = ActionFilter( alpha=0.2)
    
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

def total_action(action, action_tracker):

    if(gym_environment == "CarRacing-v3"):
        direction = action [0]
        brake = action[1]
        gas = action[2]
        if direction < 0:    
            action_tracker[0] = action_tracker[0]  + direction
        else:
            action_tracker[1] = action_tracker[1]  + direction
        action_tracker[2] = action_tracker[2]+ brake
        action_tracker[3] =  action_tracker[3]+ gas
    elif(gym_environment == "LunarLander-v3"):
        action_tracker[action] = action_tracker[action] + 1
    return action_tracker
    
# -----------------------------------------------------------------------------
# Creates a reward per episode, average action and step reward graphs
# -----------------------------------------------------------------------------
def plot_trained_model(rewards,average_action_space,ep_step_rewards):
    # Plot and save training performance
    plt.plot(rewards)
    plt.title(f"{MODEL_TYPE} Training Performance")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig(os.path.join(FINAL_LOG_DIR, "training_performance.png"))
    plt.close()
    print(f"Training performance plot saved to {FINAL_LOG_DIR}/training_performance.png")
    
    plt.figure()
    # Plot and save the average action space usage
    if gym_environment == "CarRacing-v3":
        labels = ["LEFT", "RIGHT", "GAS", "BRAKE"]
    elif gym_environment == "LunarLander-v3":
        labels = ["DO NOTHING", "FIRE LEFT", "FIRE MAIN", "FIRE RIGHT"]
    else:
        labels = [f"Action {i}" for i in range(len(average_action_space))]  # fallback

    plt.bar(labels, average_action_space)
    plt.title(f"{gym_environment} - Average Action Usage")
    plt.ylabel("Total Action Magnitude")
    plt.xlabel("Action")
    plt.tight_layout()
    action_plot = os.path.join(FINAL_LOG_DIR, "average_action_space.png")
    plt.savefig(action_plot)
    plt.close()

    # Plot reward per step for each episode
# Plot reward per step for each episode
    plt.figure()
    for episode_idx, step_list in enumerate(ep_step_rewards):
        steps, rewards = zip(*step_list)

        if len(rewards) > 1:
            reward_change = rewards[-1] - rewards[-2]

            if reward_change < -50:
                # Sudden drop — mark red 'x'
                plt.plot(steps[:-1], rewards[:-1], label=f"Episode {episode_idx + 1}")
                plt.plot(steps[-2], rewards[-2], 'rx', markersize=6, markeredgewidth=1.5)

            elif reward_change > 50:
                # Sudden jump — mark green 'x'
                plt.plot(steps[:-1], rewards[:-1], label=f"Episode {episode_idx + 1}")
                plt.plot(steps[-2], rewards[-2], 'gx', markersize=6, markeredgewidth=1.5)

            else:
                # Normal episode
                plt.plot(steps, rewards, label=f"Episode {episode_idx + 1}")

        else:
            # Not enough points to compare, just plot it
            plt.plot(steps, rewards, label=f"Episode {episode_idx + 1}")


    plt.title("Step-wise Reward per Episode")
    plt.xlabel("Step")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    step_plot_path = os.path.join(FINAL_LOG_DIR, "stepwise_rewards.png")
    plt.savefig(step_plot_path)
    plt.close()


# -----------------------------------------------------------------------------
# Load and evaluate function
# -----------------------------------------------------------------------------
def load_and_evaluate(model_path, eval_episodes):
    eval_env = gym.make(gym_environment, render_mode="human")
    
    if MODEL_TYPE == "POEM":
        model = POEM.load(model_path, env=eval_env)
    elif MODEL_TYPE == "PPO":
        model = PPO.load(model_path, env=eval_env)
        
    rewards = []
    ep_step_rewards = [] 
    ep_action_space = []
    #smoother = ActionFilter( alpha=0.2)
    for _ in range(eval_episodes):
        obs, _ = eval_env.reset()
        done = False
        ep_reward = 0.0
        step = 0
        step_rewards = [] 
        action_space = [0,0,0,0]
        while not done:
            with torch.no_grad():
                action, _ = model.predict(obs)
            obs, reward, done, _, _ = eval_env.step(action)
            action_space = total_action(action,action_space)
            ep_reward += reward
            step_rewards.append((step, ep_reward)) 
            step = step + 1
        ep_action_space.append(action_space)
        ep_step_rewards.append(step_rewards)
        rewards.append(ep_reward)
            
    eval_env.close()
    ep_action_space = np.array(ep_action_space)
    average_action_space  = np.mean(ep_action_space, axis=0)
    if(gym_environment == "CarRacing-v3"): average_action_space[0] *= -1
    avg_reward = np.mean(rewards)
    action_csv = os.path.join(FINAL_LOG_DIR, "average_action_space.csv")
    np.savetxt(action_csv, average_action_space[None], delimiter=",", header="LEFT,RIGHT,GAS,BRAKE", comments='')
    reward_csv = os.path.join(FINAL_LOG_DIR, "eval_rewards.csv")
    np.savetxt(reward_csv, rewards, delimiter=",")

    step_csv = os.path.join(FINAL_LOG_DIR, "step_rewards.csv")
    with open(step_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Step", "EpisodeReward"])
        for episode_idx, step_list in enumerate(ep_step_rewards):
            for step, reward in step_list:
                writer.writerow([episode_idx + 1, step, reward])
    
    plot_trained_model(rewards,average_action_space,ep_step_rewards)  
    print(f"Avg eval reward after loading model: {avg_reward:.2f}")
    return avg_reward
    





if __name__ == "__main__":
    TRAIN = False # Set to True to train; otherwise the script will load and evaluate the saved model
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
