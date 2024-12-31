import numpy as np
from env0 import SimpleMDPEnv
from reward import reward_env0


    
    
    
if __name__ == "__main__":


    env = SimpleMDPEnv()
    env1 = SimpleMDPEnv()

    trajectory  = list()

    trajectory1 = list()

    
    
    state = env.reset()
    state1 = env1.reset()


    trajectory.append(state)
    trajectory1.append(state1)

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
            
            next_state, done = env.step(action,"b")
            trajectory.append(next_state)

            # print("agent 0")
            # env.render()

        if done1 == False:
            next_state1, done1 = env1.step(action1,"c")

            trajectory1.append(next_state1)
            # print("agent 1")
            # env1.render()

        
        print(reward_env0(trajectory, trajectory1, env, env1, step))
            

        step += 1

    # print(trajectory)
    # print(trajectory1)

    # print(f"Reward agent 0: {total_reward}")
    # print(f"Reward agent 1: {total_reward1}")