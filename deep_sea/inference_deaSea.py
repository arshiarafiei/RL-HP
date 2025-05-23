from stable_baselines3 import PPO
from stable_baselines3 import DQN
from DeapSea import DeepSeaTreasureEnv
import numpy as np





def ppo_1():


    trs = [1,5,10,15,20,30]


    env = DeepSeaTreasureEnv()

    state, info = env.reset()

    model = PPO.load("models/PPO_hypRL/1000_steps")


    hyprl_f = list()

    for i in range (1000):
            state, info = env.reset()
            temp  = trs.copy()
            li = []
            step = 0
            while True:
                if len(temp)<1:
                    break
                step = step + 1 

                action, _states = model.predict(state)
                next_state, _, done, _, info = env.step(action)
                state = next_state

                if info['treasure']>=temp[0]:
                    li.append(step)
                    temp.pop(0)
            hyprl_f.append(li)


    data = np.array(hyprl_f)


    means = data.mean(axis=0)
    ses   = data.std(axis=0, ddof=1) / np.sqrt(data.shape[0])

    for i, (m, se) in enumerate(zip(means, ses), start=1):
        print(f"Position {i}: mean = {m:.2f}, std error = {se:.2f}")

        



    env = DeepSeaTreasureEnv()

    state, info = env.reset()

    model = PPO.load("models/PPO_original/1000_steps")

    org = list()

    for i in range (1000):
            state, info = env.reset()
            temp  = trs.copy()
            li = []
            step = 0
            while True:
                if len(temp)<1:
                    break
                step = step + 1 

                action, _states = model.predict(state)
                next_state, _, done, _, info = env.step(action)
                state = next_state

                # print('tr', info['treasure'])

                if info['treasure']>=temp[0]:
                    # print(step, info['treasure'])
                    li.append(step)
                    temp.pop(0)
            org.append(li)


    data1 = np.array(org)


    means = data1.mean(axis=0)
    ses   = data1.std(axis=0, ddof=1) / np.sqrt(data1.shape[0])

    for i, (m, se) in enumerate(zip(means, ses), start=1):
        print(f"Position {i}: mean = {m:.2f}, std error = {se:.2f}")







def dqn_1():


    trs = [1,5,10,15,20,30]


    env = DeepSeaTreasureEnv()

    state, info = env.reset()

    model = DQN.load("models/DQN_hypRL/500_steps")


    hyprl_f = list()

    for i in range (1000):
            state, info = env.reset()
            temp  = trs.copy()
            li = []
            step = 0
            while True:
                if len(temp)<1:
                    break
                if step>1000:
                    x = 6-len(li)
                    for j in range(x):
                        li.append(1000)
                    break
                step = step + 1 

                action, _states = model.predict(state)
                next_state, _, done, _, info = env.step(action)
                state = next_state

                if info['treasure']>=temp[0]:
                    li.append(step)
                    temp.pop(0)
            hyprl_f.append(li)


    data = np.array(hyprl_f)


    means = data.mean(axis=0)
    ses   = data.std(axis=0, ddof=1) / np.sqrt(data.shape[0])

    for i, (m, se) in enumerate(zip(means, ses), start=1):
        print(f"Position {i}: mean = {m:.2f}, std error = {se:.2f}")

        



    # env = DeepSeaTreasureEnv()

    # state, info = env.reset()

    # model = DQN.load("models/DQN_original/1000_steps")

    # org = list()

    # for i in range (1000):
    #         state, info = env.reset()
    #         temp  = trs.copy()
    #         li = []
    #         step = 0
    #         while True:
    #             if len(temp)<1:
    #                 break
    #             if step>1000:
    #                 x = 6-len(li)
    #                 for j in range(x):
    #                     li.append(1000)
    #                 break

    #             step = step + 1 

    #             action, _states = model.predict(state)
    #             next_state, _, done, _, info = env.step(action)
    #             state = next_state

    #             # print('tr', info['treasure'])

    #             if info['treasure']>=temp[0]:
    #                 # print(step, info['treasure'])
    #                 li.append(step)
    #                 temp.pop(0)
    #         org.append(li)


    # data1 = np.array(org)


    # means = data1.mean(axis=0)
    # ses   = data1.std(axis=0, ddof=1) / np.sqrt(data1.shape[0])

    # for i, (m, se) in enumerate(zip(means, ses), start=1):
    #     print(f"Position {i}: mean = {m:.2f}, std error = {se:.2f}")













def inference_ppo():


    env = DeepSeaTreasureEnv()

    state, info = env.reset()

    model = PPO.load("models/PPO_hypRL/500_steps")


    hyprl_f = list()
    for i in range (100):
        state, info = env.reset()
        for j in range(25):
            # print(env.normalize_state(env.get_state()))
            # print(env.get_state())
            action, _states = model.predict(state)
            next_state, _, done, _, info = env.step(action)
            state = next_state
        hyprl_f.append(info['treasure'])
        



    env = DeepSeaTreasureEnv()

    state, info = env.reset()

    model = PPO.load("models/PPO_original/500_steps")

    org = list()
    for i1 in range (100):
        state, info = env.reset()
        for j1 in range(25):
            # print(env.normalize_state(env.get_state()))
            # print(env.get_state())
            action, _states = model.predict(state)
            next_state, _, done, _, info = env.step(action)
            state = next_state
        org.append(info['treasure'])

    arr = np.array(hyprl_f, dtype=np.float64)

    mean = np.mean(arr)
    std_error = np.std(arr, ddof=1) / np.sqrt(len(arr))


    print("PPO hypRL mean: ", mean)
    print("PPO hypRL std error: ", std_error)

    arr1 = np.array(org, dtype=np.float64)


    mean = np.mean(arr1)
    std_error = np.std(arr1, ddof=1) / np.sqrt(len(arr1))

    print("org mean: ", mean)
    print("org std error: ", std_error)


def inference_dqn():


    env = DeepSeaTreasureEnv()

    state, info = env.reset()

    model = DQN.load("models/DQN_hypRL/1000_steps")


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

    model = DQN.load("models/DQN_original/1000_steps")

    print("##############################\n \n############################## \n \n##############################")
    org = list()
    for i in range (100):
        state, info = env.reset()
        for j in range(25):
            # print(env.normalize_state(env.get_state()))
            # print(env.get_state())
            action, _states = model.predict(state)
            next_state, _, done, _, info = env.step(action)
            state = next_state
        org.append(info['treasure'])

    arr = np.array(hyprl_f, dtype=np.float64)

    mean = np.mean(arr)
    std_error = np.std(arr, ddof=1) / np.sqrt(len(arr))


    print("PPO hypRL mean: ", mean)
    print("PPO hypRL std error: ", std_error)

    arr1 = np.array(org, dtype=np.float64)



    mean = np.mean(arr1)
    std_error = np.std(arr1, ddof=1) / np.sqrt(len(arr1))

    print("org mean: ", mean)
    print("org std error: ", std_error)


if __name__ == "__main__":
    #inference_ppo()
    # inference_dqn()
    #ppo_1()
    dqn_1()