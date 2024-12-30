import numpy as np
from env0 import SimpleMDPEnv


    
    
    
if __name__ == "__main__":
   
    env = SimpleMDPEnv()
    env1 = SimpleMDPEnv()
    
    state = env.reset()
    state1 = env1.reset()
    print(f"Agent 0: Initial State: {state}, AP: {env.atomic_propositions[state]}")
    print(f"Agent 1: Initial State: {state1}, AP: {env1.atomic_propositions[state1]}")


    done = False
    done1 = False 


    total_reward = 0
    total_reward1 = 0

    step = 0

    while (done == False or done1 == False):
        # action = np.random.choice(env.action_space)
        # action1 = np.random.choice(env1.action_space)

        action = 0
        action1 = 1

        if done == False: 
            next_state, reward, done = env.step(action,"b")
            print("agent 0")
            env.render()
            total_reward += reward


        if done1 == False:
            next_state1, reward1, done1 = env1.step(action1,"c")
            print("agent 1")
            env1.render()
            total_reward1 += reward1

        step += 1

    print(f"Reward agent 0: {total_reward}")
    print(f"Reward agent 1: {total_reward1}")
