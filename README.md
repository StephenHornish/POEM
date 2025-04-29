# Reinforcement Learning with Car Racing Environment

This repository contains code for training a reinforcement learning agent using the Car Racing environment from Gymnasium (formerly OpenAI Gym). It uses Proximal Policy Optimization (PPO) from Stable Baselines3 for training the agent. PPO is compared to POEM, our adaptation to PPO.

## Installation Guide

### 1. Install Anaconda and Create Environment

1. Download and install [Anaconda](https://www.anaconda.com/download/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. Open Anaconda Prompt (Windows) or Terminal (Mac/Linux)
3. Create a new conda environment:
   ```bash
   conda create -n car-racing python=3.10
   conda activate car-racing
   ```

### 2. Set Up Environment Dependencies

1. Install Microsoft Build Tools (Windows users only):
   - Download and install from [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/?q=build+tools)
   - During installation, select "C++ build tools" or "Desktop development with C++"

2. Install required packages:
   ```bash
   pip install gymnasium[classic-control]
   pip install swig
   pip install gymnasium[box2d]
   pip install stable-baselines3
   ```

### 3. Clone and Set Up Repository

1. Clone this repository:
   ```bash
   git clone [your-repository-url]
   cd [repository-name]
   ```

2. Ensure your conda environment is active:
   ```bash
   conda activate car-racing
   ```

### 4. Running the Code

- For manual control of the vehicle:
  ```bash
  python car_racing.py
  ```

- To train the RL agents specify the model and set TRAIN to True. To evealuate an existing model, set the path and TRAIN to False:
  ```bash
  python training.py
  ```

### 5. Important: Environment File Replacement

The training algorithm relies on a modified version of the Car Racing environment. You need to:

1. Locate the original environment file in your conda environment:
   ```
   [conda-path]/envs/car-racing/Lib/site-packages/gymnasium/envs/box2d/car_racing.py
   ```

2. Replace it with the custom `car_racing.py` file from this repository.

   **Note:** This manual replacement is required for the training algorithm to work correctly. We're working on a more elegant solution for future updates.


### 6. Training a Model

Each environment has an associated training script named using the format: <environment>_<algorithm>.py For example, to run PPO on the CarRacing environment:python car_racing_ppo.py

1. Model Creation 
A model is initialized using either PPO or POEM. POEM is an extension of Stable Baselines3's PPO implementation, with five additional hyperparameters based on the research paper. These hyperparameters have been tuned, but users can modify them as
2. Training 
The model is trained for a set number of timesteps, defined by the TIMESTEPS variable
3. Initial Evaluation 
After training, the model is evaluated on 10 randomly generated episodes. Visual feedback is provided to help debug and analyze agent behavior.
**Note:** Deterministic evaluation and reward data collection are handled by a separate script.
4. Running Pretrained Models
If a model has already been trained, you can view it by setting TRAIN = False.
This will load the saved model and run it in random environments for visual inspection and debugging 
5. Model Storage
Trained models are saved in the trained_models/ directory. These saved models are later used by the evaluate_model script for deterministic evaluation and performance comparison. This folder also stores the Tensorboard logs showing the trianing progress. 


### 7. Evaluating Model 
To evaluate a model you need both a PPO and POEM model trained for the same environment script is ran using format: evaluate_model.py --env <environment>  with an optional --human flag if the user wants to visualize what is happening.
1. POEM/PPO folders 
Containg respective Training performance, Stepwise reward and average action space graphs along with CSV files for the models indiviual performance. 
2. Comparison Graphs
In the result directory there will be two graphs showing how the models compared to one another in the frequency specific actions and a comprison of total rewards. 
3. T test results    
Contains a text file with the T Test and P values of the two sets of rewards during training. 
4. Methodology 
Each algorithm is evaluated using a preseeded environment the number of enviorments a model is tested on is determined by the valriable LONG_TRAINING_EVAL_EPISODES 

### 8. Hyper Parameter Tuning 
There are two ways to hyper parameter tune we have provided a grid search and HPO tuning
1. Grid Search
Done by a script called tuning_grid_search which will iterate through the hyper parameters and find the best combination. GRIDSEARCH_TIMESTEP determines how long a model is trianed before evaluated and GRIDSEARCH_EVAL_EPISODES determines on how many environments a model is ran on to determine its average reward. Once ran the data will be saved in the folder tuning_grid_search and the script tuning_review_grid_search will display which model had the best hyper parameters. 
- To tune hyperparameters for POEM and PPO using Grid Search
```bash
  python tuning.py
  ```

- To review results of tuning hyperparameters using Grid Search
```bash
  python tuning_review.py
  ```
2. HPO 
Done through optuna this utalizes a HPO algorithm to approximate the optimal hyper parameters. To use this script use the following syntax python tuning_HPO.py --model <model_type>--env <environment> --trials <integer> --timestep <integer>. This will gerneate a folder in tuning_hpo directory containg graphs showing which hyper parameters where most critical and graphing which hyperparameters resulted in the best rewards. In addition a text file will be provided listing the optimal set of hyper parameters. 



## Tutorial

For a tutorial on setting up project (https://www.youtube.com/watch?v=gMgj4pSHLww&ab_channel=JohnnyCode)




## Acknowledgements

- Gymnasium (formerly OpenAI Gym) for the Car Racing environment
- Stable Baselines3 for the PPO implementation
- Optuna for Hyperparameter tuning 
