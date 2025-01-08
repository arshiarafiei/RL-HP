import numpy as np
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


# Example Usage
if __name__ == "__main__":
    env = PCPMDPEnv()

    done = False
    total_reward = 0

    state = env.reset()
    while not done:
        env.render()
        print(env.action_space)
        action = np.random.choice(env.action_space)  # Randomly pick a domino
        next_state, reward, done = env.step(action)
        total_reward += reward
        time.sleep(1)


    print(f"Final State (Domino Labels): {next_state}")
    print(f"Final Domino Strings: Top: {env.domino_strings[0]}, Bottom: {env.domino_strings[1]}")
    print(f"Total Reward: {total_reward}")
