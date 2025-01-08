import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.optimizers import Adam
import random
import time 

class PCPMDPEnv:
    def __init__(self):
        
        self.dominos_context = {
            1: ("ab", "a"),
            2: ("aba", "ba"),
            3: ("c", "ba"),
            4: ("bb", "cb"),
            5: ("c", "bc")
        }  

        self.action_space = [1, 2, 3, 4, 5]   # Actions correspond to the dominos
        self.state_space = [1, 2, 3, 4, 5]  # States represent the labels of dominos picked
        self.state = None  
        self.domino_strings = ("", "")  

    def reset(self):
        
        self.state = []  
        self.domino_strings = ("", "")  
        return self.state

    def transition_function(self, current_state, action):
       
        next_state = current_state + [action]  # Append the domino label to the state sequence
        return next_state

    def update_domino_strings(self, action):
        
        top_string, bottom_string = self.domino_strings
        domino_top, domino_bottom = self.dominos_context[action]
        self.domino_strings = (top_string + domino_top, bottom_string + domino_bottom)

    def step(self, action):
       
        if action not in self.action_space:
            raise ValueError(f"Invalid action: {action}. Valid actions: {self.action_space}")

        
        next_state = self.transition_function(self.state, action)

        
        self.update_domino_strings(action)

        
        # done = self.domino_strings[0] == self.domino_strings[1]


        next_domino = self.domino_strings
        

        self.state = next_state


        return next_state, next_domino

    def render(self):
        
        print(f"Current State (Domino Labels): {self.state}")
        # print(f"Domino Strings: Top: {self.domino_strings[0]}, Bottom: {self.domino_strings[1]}")


def reward_pcp(trajectory, trajectory1, domino, domino1, step):

    ##############Semimatch############## 

    ### phi1

    phi_1_list = list()

    domino_1_top = domino[0] + "#"

    domino_1_bottom = domino[1] + "#"


    for index in range(min(len(domino_1_top), len(domino_1_bottom))-1):

        if domino_1_top[index] ==  domino_1_bottom[index]:
            phi_1_list.append(1) ## 1 - dist (a,a) = 1
        else:
            phi_1_list.append(0)  ## 1 - dist (a,b) = 0

    phi_1 = min(phi_1_list)

    #######phi_2

    index = min(len(domino_1_top), len(domino_1_bottom)) - 1

    phi_2_list = [min(-1*int(domino_1_top[index]== "#"), int(domino_1_bottom[index]== "#")), min(int(domino_1_top[index]== "#"), -1 * int(domino_1_bottom[index]== "#"))]

    phi_2 = max(phi_2_list)

    semimatch = max(phi_1,phi_2)

    #################Match##################

    domino_2_top = domino1[0] + "#"

    domino_2_bottom = domino1[1] + "#"

    max_length = max(len(domino_2_top), len(domino_2_bottom))

    # Padding 

    # print(domino_2_top, domino_2_bottom)

    domino_2_top = domino_2_top.ljust(max_length, '#')
    domino_2_bottom = domino_2_bottom.ljust(max_length, '#')

    # print(domino_2_top,domino_2_bottom)

    match_list = list()

    for index in range(min(len(domino_2_top), len(domino_2_bottom))-1):

        if domino_2_top[index] ==  domino_2_bottom[index]:
            match_list.append(1) ## 1 - dist (a,a) = 1
        else:
            match_list.append(0)  ## 1 - dist (a,b) = 0

    match = min(match_list)

    ##################Extend##################

    # print(trajectory,trajectory1)
    extend_list= list()

    for index in range(min(len(trajectory), len(trajectory1))):

        if trajectory[index] ==  trajectory1[index]:
            extend_list.append(1) ## 1 - dist (a,a) = 1
        else:
            extend_list.append(0)  ## 1 - dist (a,b) = 0
    
    extend = min(extend_list)

   

    reward = max (-1*semimatch , min(extend, match))

    return reward



class MultiAgentRQN:
    def __init__(self, state_embedding_size, action_space):
        self.state_embedding_size = state_embedding_size  # Embedding size for the variable-length state sequences
        self.action_space = action_space  # Pass the valid actions from the environment
        self.memory = []
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Embedding(input_dim=6, output_dim=self.state_embedding_size, input_length=None))  # Embedding for state sequences
        model.add(LSTM(32, return_sequences=False))  # LSTM for processing variable-length sequences
        model.add(Dense(24, activation='relu'))
        model.add(Dense(len(self.action_space) * 2, activation='linear'))  # Output size for two agents
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, combined_state, actions, rewards, next_combined_state, done):
        self.memory.append((combined_state, actions, rewards, next_combined_state, done))

    def act(self, combined_state):
        if np.random.rand() <= self.epsilon:
            return [random.choice(self.action_space), random.choice(self.action_space)]  # Random valid actions
        combined_state = np.array(combined_state).reshape(1, -1)  # Reshape the state for prediction
        act_values = self.model.predict(combined_state, verbose=0)
        action1_index = np.argmax(act_values[0][:len(self.action_space)])  # First agent
        action2_index = np.argmax(act_values[0][len(self.action_space):])  # Second agent
        return [self.action_space[action1_index], self.action_space[action2_index]]

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for combined_state, actions, rewards, next_combined_state, done in minibatch:
            combined_state = np.array(combined_state).reshape(1, -1)
            next_combined_state = np.array(next_combined_state).reshape(1, -1)
            targets = rewards
            if not done:
                targets += self.gamma * np.amax(self.model.predict(next_combined_state, verbose=0)[0])
            target_f = self.model.predict(combined_state, verbose=0)
            target_f[0][self.action_space.index(actions[0])] = targets
            target_f[0][len(self.action_space) + self.action_space.index(actions[1])] = targets
            self.model.fit(combined_state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# Example Usage
if __name__ == "__main__":
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
