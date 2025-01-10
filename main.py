import numpy as np
from env0 import SimpleMDPEnv
from reward import reward_env0, reward_pcp, reward_pcp_new
from model import MultiAgentDQN
from pcpenv import PCPMDPEnv
from model import MultiAgentRQN, MultiAgentRQN_NEW
import time 



def simple_env():
    env1 = SimpleMDPEnv()
    env2 = SimpleMDPEnv()

    

    state_size = 1  
    action_size = len(env1.action_space)

    agent = MultiAgentDQN(state_size, action_size)

    episodes = 100
    batch_size = 32


    for e in range(episodes):



        state1 = env1.reset()
        state2 = env2.reset()

        trajectory1  = list()

        trajectory2 = list()

        combined_state = np.reshape([state1, state2], [1, state_size * 2])

        total_reward = 0

        done1 = False
        done2 = False

        for time in range(50):
            
            actions = agent.act(combined_state)

            action1, action2 = actions[0], actions[1]


            if done1 == False:
            
                next_state1, done1 = env1.step(action1,"b")
                trajectory1.append(next_state1)

            
            if done2 == False:

                next_state2, done2 = env2.step(action2,"c")
                trajectory2.append(next_state2)


            total_reward += reward_env0(trajectory1, trajectory2, env1, env2, time)
            

            combined_next_state = np.reshape([next_state1, next_state2], [1, state_size * 2])

            agent.remember(combined_state, actions, total_reward, combined_next_state, done1 and done2)
            combined_state = combined_next_state

            print("done1:", done1)
            print("done2:", done2)

            if done1 and done2:
                print(f"Episode {e+1}/{episodes} - Reward {total_reward}, Epsilon: {agent.epsilon:.2f}")
                break

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)



def pcp_env():
    env1 = PCPMDPEnv()
    env2 = PCPMDPEnv()
    agent = MultiAgentRQN(state_embedding_size=10, action_space=env1.action_space)

    episodes = 1000
    batch_size = 32

    for e in range(episodes):
        state1 = env1.reset()
        state2 = env2.reset()
        combined_state = state1 + state2
        total_reward = 0
        done = False
        step = 0

        while not done:
            # env1.render()
            # env2.render()


            actions = agent.act(combined_state)

            next_state1, domino1 = env1.step(actions[0])


            next_state2, domino2 = env2.step(actions[1])

            total_reward += reward_pcp(next_state1, next_state2, domino1, domino2, step)

            print(total_reward)

            

            next_combined_state = next_state1 + next_state2

            


            done = domino2[0] == domino2[1]


            # print(combined_state)
            # print(actions)
            time.sleep(1)

            
            agent.remember(combined_state, actions, 1, next_combined_state, done)
            combined_state = next_combined_state
            step += 1

            # if done1 and done2:
            #     print(f"Episode {e+1}/{episodes} - Total Reward: {total_reward}")

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

    
def pcp_env_new():
    env = PCPMDPEnv()  # Single environment
    agent = MultiAgentRQN_NEW(state_embedding_size=100, action_space=env.action_space)

    episodes = 1000
    step_size = 10
    batch_size = 32

    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        step = 0
        done = False

        while step < step_size:
            action = agent.act(state)  # Get action for the current state
            next_state, domino = env.step(action)  # Take action in the environment
            total_reward = reward_pcp_new(domino)  # Compute reward
            print("step:", step, "  reward:",reward_pcp_new(domino))
            
            
            done = domino[0] == domino[1]  # Check if the episode is done
            
            agent.remember(state, action, total_reward, next_state, done)  # Store in memory
            state = next_state  # Update state
            step += 1

            if done or step == step_size:
                print(domino)
                print(f"Episode {e + 1}/{episodes} - Total Reward: {total_reward}")

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)




if __name__ == "__main__":

    # simple_env()
    #pcp_env()
    pcp_env_new()

    

    