# Reinforcement Learning with Car Racing Environment

This repository contains code for training a reinforcement learning agent using the Car Racing environment from Gymnasium (formerly OpenAI Gym). It uses Proximal Policy Optimization (PPO) from Stable Baselines3 for training the agent.

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

- To train the RL agent using PPO:
  ```bash
  python trainingPPO.py
  ```

### 5. Important: Environment File Replacement

The training algorithm relies on a modified version of the Car Racing environment. You need to:

1. Locate the original environment file in your conda environment:
   ```
   [conda-path]/envs/car-racing/Lib/site-packages/gymnasium/envs/box2d/car_racing.py
   ```

2. Replace it with the custom `car_racing.py` file from this repository.

   **Note:** This manual replacement is required for the training algorithm to work correctly. We're working on a more elegant solution for future updates.

## Tutorial

For a tutorial on setting up project (https://www.youtube.com/watch?v=gMgj4pSHLww&ab_channel=JohnnyCode)

## License

[Your license information here]

## Acknowledgements

- Gymnasium (formerly OpenAI Gym) for the Car Racing environment
- Stable Baselines3 for the PPO implementation
