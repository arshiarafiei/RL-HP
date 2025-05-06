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
env   = WildFireEnv(method="hypRL")
for i in range(100):
    state, info = env.reset()

    model = PPO.load("models/PPO_hyprl/PPO_1")
    done = False
    stepi = 0
    flag1 = 0
    flag2 = 0
    dist = 0
    while not done:
        action = model.predict(state)
        stepi += 1

        obs, reward, done, goals, info = env.step(action, mode='inference')
        if info['sub_goals'][0] == True and flag1 ==0:
            fire = stepi
            flag1 = 1
        if info['sub_goals'][1] == True and flag2 ==0:
            victim = stepi
            flag2 = 1
        dist += env.trajectory[-1][2]
    
    hyprl.append([fire, victim, stepi, dist/stepi])

        
baseline = list()
for i in range(100):
    state, info = env.reset()

    model = PPO.load("models/PPO_orginal/PPO_1")
    done = False
    stepi = 0
    flag1 = 0
    flag2 = 0
    dist = 0
    while not done:

        stepi += 1
        action = model.predict(state)

        obs, reward, done, goals, info = env.step(action, mode='inference')

        if info['sub_goals'][0] == True and flag1 ==0:
            fire = stepi
            flag1 = 1
        if info['sub_goals'][1] == True and flag2 ==0:
            victim = stepi
            flag2 = 1
        dist += env.trajectory[-1][2]
    
    baseline.append([fire, victim, stepi, dist/stepi])

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