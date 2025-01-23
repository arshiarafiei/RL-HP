import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, LSTM, Embedding, Input, Concatenate
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.preprocessing.sequence import pad_sequences
import random
import time


class PCPMDPEnv:
    def __init__(self):
        self.dominos_context = {
            1: ("a", "ab"),
            2: ("bab", "a"),
            3: ("ac", "ba"),
            4: ("b", "c"),
            5: ("bc", "bbc")
        }
        self.action_space = [1, 2, 3, 4, 5]
        self.state = None
        self.domino_strings = ("", "")

    def reset(self):
        self.state = []
        self.domino_strings = ("", "")
        return self.domino_strings

    def update_domino_strings(self, action):
        top_string, bottom_string = self.domino_strings
        domino_top, domino_bottom = self.dominos_context[action]
        self.domino_strings = (top_string + domino_top, bottom_string + domino_bottom)

    def transition_function(self, current_state, action):

        next_state = current_state + [action] 
        return next_state


    def step(self, action):
        if action not in self.action_space:
            raise ValueError(f"Invalid action: {action}. Valid actions: {self.action_space}")
        self.update_domino_strings(action)

        next_state = self.transition_function(self.state, action)
        self.state = next_state

        return self.domino_strings, next_state

    def render(self):
        print(f"Domino Strings: Top: {self.domino_strings[0]}, Bottom: {self.domino_strings[1]}")

    def domino_cont(self, state):
        dominos_context = self.dominos_context 
        final_domino = ("", "")
        for key in state:
            start, end = dominos_context[key]
            final_domino = (final_domino[0] + start, final_domino[1] + end)
        return final_domino



def reward_until(env, state, lab):

    
    def count_diff(str1, str2):

        if len(str1) != len(str2):
            raise ValueError("Strings must be of the same length.")
        
        differences = sum(1 for a, b in zip(str1, str2) if a != b)
        return differences

    if len(state) <=1:
        
        domino = env.domino_cont(state)
        domino_top = domino[0] + "#"
        domino_bottom = domino[1] + "#"

        diff = 0

        for indexj in range(min(len(domino_top), len(domino_bottom))-1):
            if domino_top[indexj] ==  domino_bottom[indexj]:
                continue
            else:
                diff = diff + 1    
        num = min(len(domino_top), len(domino_bottom))


        return int(num/2)*10 - (diff*10)
    
    else:
        formula = list()

        for t in range(1,len(state)):
            # print("t",t)

            
            ###############SemiMatch################################
            temp_semimatch = state[:t+1]
            
            domino = env.domino_cont(temp_semimatch)
            domino_top = domino[0] + "#"
            domino_bottom = domino[1] + "#"

            diff = 0

            for indexj in range(min(len(domino_top)-1, len(domino_bottom)-1)):
                if domino_top[indexj] ==  domino_bottom[indexj]:
                    continue
                else:
                    diff = diff + 1   
            # print("semi",domino_top, domino_bottom, diff)   
            num = min(len(domino_top)-1, len(domino_bottom)-1)
            semimatch_list = (int(num/2))*10 - (diff*10)
                
            
            # print("semimatch",semimatch_list)

            match_list = list()

            match_list.append(semimatch_list)
            for i in range(t+1,len(state)):
                # print("index",i)
                temp = state[:i+1].copy()
                domino = env.domino_cont(temp)

                domino_top = domino[0]

                domino_bottom = domino[1]


                max_length = max(len(domino_top), len(domino_bottom))

                # print("match", domino_top, domino_bottom)

                ###############Match################################

                domino_top = domino_top.ljust(max_length, '#')
                domino_bottom = domino_bottom.ljust(max_length, '#')

                # print("match222", domino_top, domino_bottom)

                diff = count_diff(domino_bottom, domino_top)

                match_list.append((int((max_length)/2))*10 - (diff*10))
            
            # print("match",match_list)
            
            formula.append(min(match_list))

        # print("formula",formula)
    
        
        reward = max(formula)

        # print(reward)

        return reward
                


        



if __name__ == "__main__":
    env = PCPMDPEnv()
    env.reset()
    

    episodes = 1000
    batch_size = 32

    step = 0
    done = False

    a = [1,2,5]

    while step < 3:

        action =a[step]

        print("action:", action)


        next_domino, labeles = env.step(action)
        print("nex dom:", next_domino)
        # print("lab:", labeles)

        # print(reward_until(env, labeles, next_domino))
        print(reward_until(env, labeles, next_domino))
        # print("reward:", reward)

        time.sleep(1)




        done = next_domino[0] == next_domino[1]

        
        domino = next_domino
        step += 1
