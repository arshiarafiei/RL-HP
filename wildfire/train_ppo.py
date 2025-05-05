import gymnasium as gym
from WildFire import WildFireEnv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

import os



models_dir = "models/PPO_original"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = WildFireEnv()
env.reset()


model = PPO(
    "MlpPolicy",
    env, n_steps=50, tensorboard_log=logdir)

TIMESTEPS = 100000
for i in range(10):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO", log_interval=100)
    # model.save(f"{models_dir}/{TIMESTEPS*i}_steps")
    model.save(f"{models_dir}/PPO_{i*100000}")


    mean_reward, std_reward = evaluate_policy(
    model,
    model.get_env(),
    deterministic=True,
    n_eval_episodes=20)

    print(f"Time Step {TIMESTEPS*i} -- mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")


# # env = WildFireEnv()

# # state, info = env.reset()

# # model = PPO.load("models/PPO_original/PPO_0")

# # print("##############################\n \n############################## \n \n##############################")
# # done = False
# # step = 0

# # while not done:

    
# #     action = model.predict(state)

# #     print("action: ", action)

 
# #     obs, reward, done, goals, info = env.step(action)
# #     # step += 1
# #     # if info['sub_goals'][1] == True:
# #     #     print("done in saving victims with: ",step , info['sub_goals'])
# #     #     break
# #     # print(info)
# #     # env.render()



