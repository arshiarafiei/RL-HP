import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.optimizers import Adam
import random
from tensorflow.keras.preprocessing.sequence import pad_sequences

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


def reward_pcp_old(trajectory, trajectory1, domino, domino1, step):

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

def reward_pcp(domino, step):

    ##############Semimatch############## 

    ### phi1

    phi_1_list = list()

    domino_1_top = domino[0] + "#"

    domino_1_bottom = domino[1] + "#"

    phi_1 = 1

    flag = 0


    for index in range(min(len(domino_1_top), len(domino_1_bottom))-1):

        if domino_1_top[index] ==  domino_1_bottom[index]:
            continue
        else:
            flag = 1
            phi_1 = phi_1 - 1 
    
    if flag ==0:
        phi_1 = 10


    # phi_1 = min(phi_1_list)

    #######phi_2

    index = min(len(domino_1_top), len(domino_1_bottom)) - 1

    phi_2_list = [min(-1*int(domino_1_top[index]== "#"), int(domino_1_bottom[index]== "#")), min(int(domino_1_top[index]== "#"), -1 * int(domino_1_bottom[index]== "#"))]

    phi_2 = max(phi_2_list)

    # semimatch = max(phi_1,phi_2)

    semimatch = phi_1

    #################Match##################

    domino_2_top = domino[0] + "#"

    domino_2_bottom = domino[1] + "#"

    max_length = max(len(domino_2_top), len(domino_2_bottom))

    # Padding 

    # print(domino_2_top, domino_2_bottom)

    domino_2_top = domino_2_top.ljust(max_length, '#')
    domino_2_bottom = domino_2_bottom.ljust(max_length, '#')

    # print(domino_2_top,domino_2_bottom)

    match_list = list()

    match = 1
    flag1 = 0

    for index in range(max_length):

        if domino_2_top[index] ==  domino_2_bottom[index]:
            continue
        else:
            flag1 = 1
            match = match -1 

    if flag1 == 0:
        match = 50

    reward = max(semimatch , match)


    return reward












def reward_pcp(domino, step):

    ##############Semimatch############## 

    ### phi1

    phi_1_list = list()

    domino_1_top = domino[0] + "#"

    domino_1_bottom = domino[1] + "#"

    phi_1 = 1

    flag = 0


    for index in range(min(len(domino_1_top), len(domino_1_bottom))-1):

        if domino_1_top[index] ==  domino_1_bottom[index]:
            continue
        else:
            flag = 1
            phi_1 = phi_1 - 1 
    
    if flag ==0:
        phi_1 = 10


    # phi_1 = min(phi_1_list)

    #######phi_2

    index = min(len(domino_1_top), len(domino_1_bottom)) - 1

    phi_2_list = [min(-1*int(domino_1_top[index]== "#"), int(domino_1_bottom[index]== "#")), min(int(domino_1_top[index]== "#"), -1 * int(domino_1_bottom[index]== "#"))]

    phi_2 = max(phi_2_list)

    # semimatch = max(phi_1,phi_2)

    semimatch = phi_1

    #################Match##################

    domino_2_top = domino[0] + "#"

    domino_2_bottom = domino[1] + "#"

    max_length = max(len(domino_2_top), len(domino_2_bottom))

    # Padding 

    # print(domino_2_top, domino_2_bottom)

    domino_2_top = domino_2_top.ljust(max_length, '#')
    domino_2_bottom = domino_2_bottom.ljust(max_length, '#')

    # print(domino_2_top,domino_2_bottom)

    match_list = list()

    match = 1
    flag1 = 0

    for index in range(max_length):

        if domino_2_top[index] ==  domino_2_bottom[index]:
            continue
        else:
            flag1 = 1
            match = match -1 

    if flag1 == 0:
        match = 50

    reward = max(semimatch , match)


    return reward












def reward_pcp_new(state):

    def domino_cont(state):
        dominos_context = {
                1: ("ab", "a"),
                2: ("aba", "ba"),
                3: ("c", "ba"),
                4: ("bb", "cb"),
                5: ("c", "bc")
            } 
        final_domino = ("", "")
        for key in state:
            start, end = dominos_context[key]
            final_domino = (final_domino[0] + start, final_domino[1] + end)
        return final_domino
    
    def count_diff(str1, str2):

        if len(str1) != len(str2):
            raise ValueError("Strings must be of the same length.")
        
        differences = sum(1 for a, b in zip(str1, str2) if a != b)
        return differences

    match_list = list()
    semimatch_list = list()


    for t in range( len(state)):
        temp = state[:t+1].copy()
        domino = domino_cont(temp)

        domino_top = domino[0] + "#"

        domino_bottom = domino[1] + "#"

        ###############Match################################

        max_length = max(len(domino_top), len(domino_bottom))

        domino_top = domino_top.ljust(max_length, '#')
        domino_bottom = domino_bottom.ljust(max_length, '#')

        # print(domino_bottom, domino_top)

        diff = count_diff(domino_bottom, domino_top)

        match_list.append(1-diff)

        ###############SemiMatch################################

        domino_top = domino[0] + "#"

        domino_bottom = domino[1] + "#"

        diff = 0

        for index in range(min(len(domino_top), len(domino_bottom))-1):

            if domino_top[index] ==  domino_bottom[index]:
                continue
            else:
                diff = diff + 1
        
        semimatch_list.append(1-diff)

    ###############Formula################################

    semimatch = min(semimatch_list)


    matched = min(match_list)

    if semimatch == 1:
        semimatch = 5
    if match_list[-1] == 1:
        matched = 10

    # print(semimatch_list)
    # print(match_list)

    reward = max(semimatch, matched)

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
        self.learning_rate = 0.005
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Embedding(input_dim=6, output_dim=self.state_embedding_size, input_length=None))  # Embedding for state sequences
        model.add(LSTM(32, return_sequences=False))  # LSTM for processing variable-length sequences
        model.add(Dense(24, activation='relu'))
        model.add(Dense(len(self.action_space) , activation='linear'))  # Output size for two agents
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(self.action_space)  # Random valid action

        # Handle empty state
        if len(state) == 0:
            state = [0]  # Provide a placeholder for empty sequences

        # Ensure state is a NumPy array, padded, and reshaped
        state = pad_sequences([state], maxlen=10, padding='post')  # Pad state to ensure consistent input length
        state = np.array(state)  # Convert to NumPy array if not already

        # Make prediction
        act_values = self.model.predict(state, verbose=0)
        action_index = np.argmax(act_values[0])  # Get the best action based on predicted Q-values
        if action_index >= len(self.action_space):
            raise ValueError(f"Invalid action index: {action_index}. Model output: {act_values[0]}")
        
        return self.action_space[action_index]






    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = pad_sequences([state], maxlen=10, padding='post')
            next_state = pad_sequences([next_state], maxlen=10, padding='post')
            
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            
            target_f = self.model.predict(state, verbose=0)
            target_f[0][self.action_space.index(action)] = target  # Update the specific action's Q-value
            
            self.model.fit(state, target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay



if __name__ == "__main__":
    env = PCPMDPEnv()  # Single environment
    agent = MultiAgentRQN(state_embedding_size=100, action_space=env.action_space)

    episodes = 10000
    step_size = 10
    batch_size = 32

    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        step = 0
        done = False
        domino = ("", "")

        while step < step_size:
            
            action = agent.act(state)  # Get action for the current state
            next_state, next_domino = env.step(action)  # Take action in the environment

            print(next_state)
            print(next_domino)
            
            
            # total_reward = reward_pcp(next_domino, step)  # Compute reward
            total_reward = reward_pcp_new(next_state) 
            print("step:", step, "  reward:",total_reward)
            
            
            done = next_domino[0] == next_domino[1]  # Check if the episode is done
            
            agent.remember(state, action, total_reward, next_state, done)  # Store in memory
            state = next_state  # Update state
            domino = next_domino
            step += 1

            time.sleep(1)

            if done or step == step_size:
                print(state)
                print(domino)
                print(f"Episode {e + 1}/{episodes} - Total Reward: {total_reward}")
                break

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
