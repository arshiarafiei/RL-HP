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
env   = WildFireEnv(method="baseline",n_grid=10, mode='inference')
for i in range(10):
    state, info = env.reset()

    model = PPO.load("models/PPO_10_hyprl/PPO_0")
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

    model = PPO.load("models/PPO_10_orginal/PPO_0")
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



baseline = np.array(baseline)
hyprl = np.array(hyprl)

# Compute mean and standard error for each column
baseline_mean = np.mean(baseline, axis=0)
baseline_se = np.std(baseline, axis=0, ddof=1) / np.sqrt(len(baseline))

hyprl_mean = np.mean(hyprl, axis=0)
hyprl_se = np.std(hyprl, axis=0, ddof=1) / np.sqrt(len(hyprl))

# Print results
print("baseline fire victim step dist (mean ± SE):")
for i in range(4):
    print(f"  Metric {i+1}: {baseline_mean[i]:.3f} ± {baseline_se[i]:.3f}")

print("hyprl fire victim step dist (mean ± SE):")
for i in range(4):
    print(f"  Metric {i+1}: {hyprl_mean[i]:.3f} ± {hyprl_se[i]:.3f}")