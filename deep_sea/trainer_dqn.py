import gymnasium as gym
from DeapSea import DeepSeaTreasureEnv
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np

import os


models_dir = "models/DQN"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)


env = DeepSeaTreasureEnv()
env.reset()

# Hyperparameters based on RL zoo

model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    train_freq=16,
    gradient_steps=8,
    gamma=0.99,
    exploration_fraction=0.2,
    exploration_final_eps=0.07,
    target_update_interval=600,
    learning_starts=1000,
    buffer_size=10000,
    batch_size=128,
    learning_rate=4e-3,
    policy_kwargs=dict(net_arch=[256, 256]),
    tensorboard_log=logdir,
    seed=2,
)

TIMESTEPS = int(100000)
for i in range(30):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="DQN", log_interval=10)
    model.save(f"{models_dir}/{TIMESTEPS*i}_steps")


    mean_reward, std_reward = evaluate_policy(
    model,
    model.get_env(),
    deterministic=True,
    n_eval_episodes=20)

    print(f"Time Step {TIMESTEPS*i} -- mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")