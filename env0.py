import numpy as np

class SimpleMDPEnv:
    def __init__(self):
        # Define the state space and action space
        self.state_space = [0, 1, 2, 3, 4, 5]  #  states
        self.action_space = [0, 1]  # actions
        self.initial_state = self.state_space[0]
        self.state = self.state_space[0] 
        self.transitions = [
            (0, 0, 2),
            (2, 0, 3),
            (3, 0, 4),
            (0, 1, 1),
            (1, 1, 5)
            
        ]
        self.atomic_propositions = {0: "a", 1: "a", 2: "a", 3: "b", 4: "b", 5: "c"}  

    def reset(self):
        return self.initial_state

    def transition_function(self, current_state, action):
        
        for (cur_state, act, next_state) in self.transitions:
            if cur_state == current_state and act == action:
                return next_state
        raise ValueError(f"Invalid transition for state {current_state} and action {action}.")

    def step(self, action, target):
        
        if action not in self.action_space:
            raise ValueError(f"Invalid action: {action}. Valid actions: {self.action_space}")

        # Transition
        next_state = self.transition_function(self.state, action)

        # Reward function
        reward = 1 if next_state == len(self.state_space) - 1 else -0.1

        # terminal condition
        done = self.atomic_propositions[next_state] == target

        self.state = next_state
        return next_state, reward, done

    def render(self):
        
        print(f"Current State: {self.state}, Atomic Proposition: {self.atomic_propositions[self.state]}")

