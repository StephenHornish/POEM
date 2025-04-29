import os
import pandas as pd
import scipy.stats as stats

# Root results folder
ROOT_FOLDER = "results"

# List of environment result directories
ENV_FOLDERS = ["bipedal_results", "car_results", "cart_results", "lander_results"]

# Where the eval_rewards.csv should be
CSV_FILENAME = "eval_rewards.csv"

# Storage for T-test results
t_test_results = {}

# Loop through each environment folder
for env in ENV_FOLDERS:
    ppo_path = os.path.join(ROOT_FOLDER, env, "PPO", CSV_FILENAME)
    poem_path = os.path.join(ROOT_FOLDER, env, "POEM", CSV_FILENAME)

    if os.path.exists(ppo_path) and os.path.exists(poem_path):
        # Load rewards
        ppo_rewards = pd.read_csv(ppo_path).squeeze()  # Squeeze to Series if single column
        poem_rewards = pd.read_csv(poem_path).squeeze()

        # Perform independent two-sample T-test
        t_stat, p_value = stats.ttest_ind(ppo_rewards, poem_rewards, equal_var=False)

        # Store t-statistic and p-value for saving
        t_test_results[env] = (t_stat, p_value)

        print(f"{env}: T-statistic={t_stat:.8f}, P-value={p_value:.8f}")
    else:
        print(f"Missing PPO or POEM eval_rewards.csv for {env}")

# --- Saving to TXT file ---

# Ensure root folder exists
os.makedirs(ROOT_FOLDER, exist_ok=True)

# Save results in a TXT file
summary_path = os.path.join(ROOT_FOLDER, "t_test_summary.txt")
with open(summary_path, 'w') as f:
    for env, (t_stat, p_value) in t_test_results.items():
        f.write(f"{env}:\n")
        f.write(f"  T-statistic: {t_stat:.8f}\n")
        f.write(f"  P-value    : {p_value:.8f}\n")
        f.write("\n")

# Also save into each individual environment folder
for env in ENV_FOLDERS:
    if env in t_test_results:
        env_result_path = os.path.join(ROOT_FOLDER, env, "t_test_result.txt")
        os.makedirs(os.path.join(ROOT_FOLDER, env), exist_ok=True)  # In case folder missing
        with open(env_result_path, 'w') as f:
            t_stat, p_value = t_test_results[env]
            f.write(f"{env}:\n")
            f.write(f"  T-statistic: {t_stat:.8f}\n")
            f.write(f"  P-value    : {p_value:.8f}\n")

print("T-test results saved into each environment folder and summary saved in results/.")
