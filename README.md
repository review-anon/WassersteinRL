# Code for "Distributional Off-Policy Evaluation in Reinforcement Learning"

## Package environments

Please refer to `environment.yml` for the package depandence. If use use conda, run 

```
conda env create -f environment.yml
```

in this repository to install all dependencies for this project. 


## Run Experiments for Maze

Use the following scripts to train an agent in `MultiRewardMaze-v0` task.

```python
python -um dopamine.discrete_domains.train \
    --base_dir ./dopamine_runs/MultiRewardMaze_GANND_V0_P01N5 \
    --gin_files dopamine/agents/gan_nd/configs/gan_nd.gin \
    --gin_bindings atari_lib.create_atari_environment.game_name=\"MultiRewardMaze-v0\" \
    --gin_bindings gan_nd_agent.GANAgent.grad_penalty_factor=0.1 \
    --gin_bindings gan_nd_agent.GANAgent.num_discriminator=5 \
    --gin_bindings Runner.reward_logdir=\"./reward-compose/MultiRewardMaze-v0-reward.txt\"
```

Change the `game_name` to `MultiRewardMaze-v1` or `MultiRewardMaze-v0` to train our agent in other tasks. 

The results will be saved in `./dopamine_runs/MultiRewardMaze_GANND_V0_P01N5`, it contains `evaluation_plots`, `logs`, and `plots`. You can run `plot-maze.py` to obtain the return distribution from the logged data.

## Run Experiments for Atari Games

Use the following scripts to train an agent in `Pong` task from Atari suite with multi-dimensional reward.

```python
python -um dopamine.discrete_domains.train \
   --base_dir ./dopamine_runs/GANND_Pong \
   --gin_files dopamine/agents/gan_nd/configs/gan_nd.gin \
   --gin_bindings atari_lib.create_atari_environment.game_name=\"$1\" \
   --gin_bindings Runner.reward_logdir=\"./reward-compose/Pong-reward.txt\" \
   --gin_bindings gan_nd_agent.GANAgent.maze_env=False \
   --gin_bindings gan_nd_agent.GANAgent.evaluation_setting=False \
   --gin_bindings Runner.clip_rewards=True \
   --gin_bindings Runner.num_iterations=200 
```

Change the `game_name` to other games in `atari.txt` to train our agents. 

The results will be saved in `./dopamine_runs/GANND_Pong`, it contains `evaluation_plots`, `logs`, and `plots`. You can run `plot-atari.py` to obtain the return distribution from the logged data.

## Repo Components

The implementation of three maze environments is in `dopamine/environment/maze.py`,
and the implementation of MD3QN is in `dopamine/agents/gan_nd/gan_nd_agent.py`. 
