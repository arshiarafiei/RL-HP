import gymnasium as gym
from stable_baselines3 import DQN
from DeapSea import DeepSeaTreasureEnv
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import os
from multiprocessing import Process



def train_hypRL():
    models_dir = "models/DQN_hypRL"
    logdir     = "logs_hypRL"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logdir,     exist_ok=True)


    env = DeepSeaTreasureEnv(hypRL=True)
    env.reset()




# Hyperparameters based on RL zoo

    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        gradient_steps=8,
        gamma=0.99,
        exploration_fraction=0.2,
        exploration_final_eps=0.07,
        target_update_interval=600,
        learning_starts=25,
        batch_size=16,
        learning_rate=4e-3,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=logdir,
    )
    TIMESTEPS = 500
    
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="DQN_hyprl", log_interval=10)
    model.save(f"{models_dir}/{TIMESTEPS}_steps")

    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        deterministic=True,
        n_eval_episodes=1
    )
    print(f"hypRL Time Step {TIMESTEPS} -- mean_reward: {mean_reward:.2f} ± {std_reward:.2f}")



def train_original():
    models_dir = "models/DQN_original"
    logdir     = "logs_original"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logdir,     exist_ok=True)




    env = DeepSeaTreasureEnv()
    env.reset()




# Hyperparameters based on RL zoo

    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        gradient_steps=8,
        gamma=0.99,
        exploration_fraction=0.2,
        exploration_final_eps=0.07,
        target_update_interval=600,
        learning_starts=25,
        batch_size=16,
        learning_rate=4e-3,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=logdir,
    )
    TIMESTEPS = 500
    
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="DQN_original", log_interval=10)
    model.save(f"{models_dir}/{TIMESTEPS}_steps")

    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        deterministic=True,
        n_eval_episodes=1
    )
    print(f"Original Time Step {TIMESTEPS} -- mean_reward: {mean_reward:.2f} ± {std_reward:.2f}")

if __name__ == "__main__":
    p1 = Process(target=train_hypRL)
    p2 = Process(target=train_original)

    p1.start()
    p2.start()

    p1.join()
    p2.join()
