import numpy as np
import math

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
            (2, 1, 2),
            (3, 0, 4),
            (3, 1, 3),
            (4,1,4),
            (4,0,4),
            (0, 1, 1),
            (1, 1, 5),
            (1, 0, 1),
            (5, 0, 5),
            (5, 1, 5)            
        ]
        self.atomic_propositions = {0: "a", 1: "a", 2: "a", 3: "a", 4: "b", 5: "c"}  

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

        

        # terminal condition
        done = self.atomic_propositions[next_state] == target

        self.state = next_state
        return next_state , done


    def render(self):
        
        print(f"Current State: {self.state}, Atomic Proposition: {self.atomic_propositions[self.state]}")




    def distance_to_target(self, start_state, target_ap):
        
        visited = set()
        queue = [(start_state, 0)]  # (current_state, distance)

        while queue:
            current_state, distance = queue.pop(0)

            if self.atomic_propositions[current_state] == target_ap:
                return distance

            visited.add(current_state)

            for (cur_state, act, next_state) in self.transitions:
                if cur_state == current_state and next_state not in visited:
                    queue.append((next_state, distance + 1))

        return math.inf