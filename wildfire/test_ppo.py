import gymnasium as gym
from WildFire import WildFireEnv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.wrappers import FlattenObservation

import os


hyprl = list()
env   = WildFireEnv(method="hypRL",n_grid=5, mode='inference')
for i in range(10):
    state, info = env.reset()

    model = PPO.load("models/PPO_5_hyprl/PPO_0")
    done = False
    stepi = 0
    flag1 = 0
    flag2 = 0
    dist = 0
    while not done:
        action = model.predict(state)
        stepi += 1

        obs, reward, done, goals, info = env.step(action)
        if info['sub_goals'][0] == True and flag1 ==0:
            fire = stepi
            flag1 = 1
        if info['sub_goals'][1] == True and flag2 ==0:
            victim = stepi
            flag2 = 1
        dist += env.trajectory[-1][2]
    if flag1 == 0:
        fire = stepi
    if flag2 == 0:
        victim = stepi
    
    hyprl.append([fire, victim, stepi, dist/stepi])

        
baseline = list()
for i in range(10):
    state, info = env.reset()

    model = PPO.load("models/PPO_5_orginal/PPO_0")
    done = False
    stepi1 = 0
    flag1 = 0
    flag2 = 0
    dist = 0
    while not done:

        stepi1 += 1
        action = model.predict(state)

        obs, reward, done, goals, info = env.step(action)

        if info['sub_goals'][0] == True and flag1 ==0:
            fire = stepi1
            flag1 = 1
        if info['sub_goals'][1] == True and flag2 ==0:
            victim = stepi1
            flag2 = 1
        dist += env.trajectory[-1][2]
    
    if flag1 == 0:
        fire = stepi1
    if flag2 == 0:
        victim = stepi1

    
    baseline.append([fire, victim, stepi1, dist/stepi1])

baseline_1 = sum(item[0] for item in baseline)/len(baseline)
baseline_2 = sum(item[1] for item in baseline)/len(baseline)
baseline_3 = sum(item[2] for item in baseline)/len(baseline)
baseline_4 = sum(item[3] for item in baseline)/len(baseline)

hyprl_1 = sum(item[0] for item in hyprl)/len(hyprl)
hyprl_2 = sum(item[1] for item in hyprl)/len(hyprl) 
hyprl_3 = sum(item[2] for item in hyprl)/len(hyprl)
hyprl_4 = sum(item[3] for item in hyprl)/len(hyprl)

print("baseline fire victime step dist: ", baseline_1, baseline_2, baseline_3, baseline_4)
print("hyprl fire victim step dist: ", hyprl_1, hyprl_2, hyprl_3, hyprl_4)