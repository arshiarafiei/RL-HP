import gymnasium as gym
from gym_grid.envs.grid_env import GridEnv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.wrappers import FlattenObservation

import os


hyprl = list()
env = GridEnv(map_name="ISR", nagents=2, norender=False, padding=True, method="baseline", mode='inference')
for i in range(1):
    state, info = env.reset()


    model = PPO.load("models/PPO_ISR_hyprl/PPO_49")
    done = False
    stepi = 0
    flag1 = 0
    coll = 0
    while not done:
        action = model.predict(state)

        # action  = env.action_space.sample()
        print("action: ", action)
        stepi += 1

        obs, reward, done, goals, info = env.step(action)
        print("obs: ", obs)
        print("reward: ", reward)
        if info['done'] == True and flag1 ==0:
            terminate = stepi
        if info['collisions'] == True: 
            coll+= 1
        import time 
        # time.sleep(0)
        print("#################\n################\n")

    if flag1 == 0:
        terminate = stepi
    
    hyprl.append([terminate, coll])

hyprl_1 = sum(item[0] for item in hyprl)/len(hyprl)
hyprl_2 = sum(item[1] for item in hyprl)/len(hyprl)
print("terminate: ", hyprl_1)
print("coll: ", hyprl_2)
