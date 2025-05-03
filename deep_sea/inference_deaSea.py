from stable_baselines3 import PPO
from stable_baselines3 import DQN
from DeapSea import DeepSeaTreasureEnv
import numpy as np


def inference_ppo():

    env = DeepSeaTreasureEnv()

    state, info = env.reset()

    model = PPO.load("models/PPO_hypRL/1000_steps")


    hyprl_f = list()
    for i in range (100):
        state, info = env.reset()
        for j in range(25):
            # print(env.normalize_state(env.get_state()))
            # print(env.get_state())
            action, _states = model.predict(state)
            next_state, _, done, _, info = env.step(action)
            state = next_state
            print(info['treasure'])
        hyprl_f.append(info['treasure'])
        



    env = DeepSeaTreasureEnv()

    state, info = env.reset()

    model = PPO.load("models/PPO_original/500_steps")

    print("##############################\n \n############################## \n \n##############################")
    org = list()
    for i in range (100):
        state, info = env.reset()
        org  = list()
        for j in range(25):
            # print(env.normalize_state(env.get_state()))
            # print(env.get_state())
            action, _states = model.predict(state)
            next_state, _, done, _, info = env.step(action)
            state = next_state
        org.append(info['treasure'])

    print("PPO hypRL: ", sum(hyprl_f)/len(hyprl_f))
    print("PPO org: ", sum(org)/len(org))


def inference_dqn():


    env = DeepSeaTreasureEnv()

    state, info = env.reset()

    model = DQN.load("models/DQN_hypRL/500_steps")


    hyprl_f = list()
    for i in range (100):
        state, info = env.reset()
        for j in range(25):
            # print(env.normalize_state(env.get_state()))
            # print(env.get_state())
            action, _states = model.predict(state)
            next_state, _, done, _, info = env.step(action)
            state = next_state
            print(info['treasure'])
        hyprl_f.append(info['treasure'])
        



    env = DeepSeaTreasureEnv()

    state, info = env.reset()

    model = DQN.load("models/DQN_original/500_steps")

    print("##############################\n \n############################## \n \n##############################")
    org = list()
    for i in range (100):
        state, info = env.reset()
        org  = list()
        for j in range(25):
            # print(env.normalize_state(env.get_state()))
            # print(env.get_state())
            action, _states = model.predict(state)
            next_state, _, done, _, info = env.step(action)
            state = next_state
        org.append(info['treasure'])



    print("DQN hypRL: ", sum(hyprl_f)/len(hyprl_f))
    print("DQN org: ", sum(org)/len(org))


if __name__ == "__main__":
    # inference_ppo()
    inference_dqn()