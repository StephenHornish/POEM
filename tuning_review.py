# This is for viewing the results of an earlier tuning run.

import os
import numpy as np

GRIDSEARCH_LOG_DIR = "POEM_gridsearch_short_runs"  # or whatever you used
FINAL_LOG_DIR = "POEM_final_best_run"

def summarize_gridsearch_results():
    results = []
    if not os.path.isdir(GRIDSEARCH_LOG_DIR):
        print(f"No directory named '{GRIDSEARCH_LOG_DIR}' found.")
        return
    
    # Loop over subdirectories in the gridsearch dir
    for item in os.listdir(GRIDSEARCH_LOG_DIR):
        run_path = os.path.join(GRIDSEARCH_LOG_DIR, item)
        if not os.path.isdir(run_path):
            continue  # skip if it's not a directory

        # Look for 'eval_rewards.csv'
        eval_csv = os.path.join(run_path, "eval_rewards.csv")
        if os.path.isfile(eval_csv):
            # Load the rewards and compute average
            rewards = np.loadtxt(eval_csv, delimiter=",")
            if rewards.size == 0:
                avg_reward = float('nan')
            else:
                avg_reward = float(np.mean(rewards))

            results.append((item, avg_reward))
        else:
            # If there's no file, skip
            continue

    # Sort by average reward descending
    results.sort(key=lambda x: x[1], reverse=True)

    # Print them
    print("=== Grid Search Results (short runs) ===")
    if results:
        for (run_name, avg_r) in results:
            print(f"{run_name}: avg_reward={avg_r:.2f}")
    else:
        print("No runs found.")

def summarize_final_run():
    if not os.path.isdir(FINAL_LOG_DIR):
        print(f"No directory named '{FINAL_LOG_DIR}' found for final run.")
        return
    
    eval_csv = os.path.join(FINAL_LOG_DIR, "eval_rewards.csv")
    if os.path.isfile(eval_csv):
        rewards = np.loadtxt(eval_csv, delimiter=",")
        avg_reward = np.mean(rewards)
        print("\n=== Final Best Run ===")
        print(f"Directory: {FINAL_LOG_DIR}")
        print(f"Avg reward across {rewards.size} episodes: {avg_reward:.2f}")
    else:
        print(f"No 'eval_rewards.csv' file found in {FINAL_LOG_DIR}.")

def main():
    summarize_gridsearch_results()
    summarize_final_run()

if __name__ == "__main__":
    main()
