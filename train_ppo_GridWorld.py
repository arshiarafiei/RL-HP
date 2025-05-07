from gym_grid.envs.grid_env import GridEnv
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.env_checker import check_env

import os
from multiprocessing import Process




def train_original():

    models_dir = "models/PPO_Suny_orginal"
    logdir = "logs"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    env = GridEnv(map_name='SUNY', nagents=2, norender=False, padding=True, method="baseline")


    model = PPO(
        "MlpPolicy",
        env,
        gamma=0.995,
        n_steps=64,
        gae_lambda=0.95,
        ent_coef=0.001,
        learning_rate=0.0003,
        vf_coef=0.5,
        batch_size=64,
        clip_range=0.2,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=logdir,
    )
    TIMESTEPS = 500000
    for i in range(1):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO_sunny_orginal", log_interval=10)
        # model.save(f"{models_dir}/{TIMESTEPS*i}_steps")
        model.save(f"{models_dir}/PPO_{i}")





def train_hypRL(map = 'SUNY'):

    models_dir = f"models/PPO_{map}_hyprl"
    logdir = f"logs_{map}"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    env = GridEnv(map_name=map, nagents=2, norender=False, padding=True, method="hypRL")



    model = PPO(
        "MlpPolicy",
        env,
        gamma=0.995,
        n_steps=128,
        gae_lambda=0.95,
        ent_coef=0.001,
        learning_rate=0.0003,
        vf_coef=0.5,
        batch_size=128,
        clip_range=0.2,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=logdir,
    )
    TIMESTEPS = 500000
    for i in range(1):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO_hyprl", log_interval=1)
        # model.save(f"{models_dir}/{TIMESTEPS*i}_steps")
        model.save(f"{models_dir}/PPO_{i}")


if __name__ == "__main__":

    maps = ["ISR","MIT","Pentagon","SUNY"]

    for map in maps:
        train_hypRL(map = map)

    
    # p1 = Process(target=train_hypRL)
    # p2 = Process(target=train_original)

    # p1.start()
    # p2.start()

    # p1.join()
    # p2.join()


