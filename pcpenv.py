import numpy as np
import time 


class PCPMDPEnv:
    def __init__(self):
        # Define the dominos context (action -> top and bottom strings)
        self.dominos_context = {
            1: ("ab", "ab"),
            2: ("aba", "a"),
            3: ("c", "ba"),
            4: ("bb", "cb"),
            5: ("a", "ba")
        }  # Domino format: action -> (top string, bottom string)

        self.action_space = [1, 2, 3, 4, 5]   # Actions correspond to the dominos
        self.state_space = [1, 2, 3, 4, 5]  # States represent the labels of dominos picked
        self.state = None  # Current state represented as the sequence of picked dominos (labels)
        self.domino_strings = ("", "")  # Current top and bottom strings

    def reset(self):
        """Resets the environment to the initial state."""
        self.state = []  # Start with no dominos picked (empty state sequence)
        self.domino_strings = ("", "")  # Empty top and bottom strings
        return self.state

    def transition_function(self, current_state, action):
        """Encodes the transition logic by appending the chosen domino to the current state.

        Args:
            current_state (list): The current state (sequence of picked dominos).
            action (int): The action to take (pick a domino).

        Returns:
            next_state (list): The next state after taking the action.
        """
        next_state = current_state + [action]  # Append the domino label to the state sequence
        return next_state

    def update_domino_strings(self, action):
        """Updates the top and bottom strings based on the chosen domino.

        Args:
            action (int): The action to take (pick a domino).
        """
        top_string, bottom_string = self.domino_strings
        domino_top, domino_bottom = self.dominos_context[action]
        self.domino_strings = (top_string + domino_top, bottom_string + domino_bottom)

    def step(self, action):
        """Takes an action and returns the next state, reward, and done flag.

        Args:
            action (int): The action to take (must be in the action space).

        Returns:
            next_state (list): The next state after taking the action.
            reward (float): The reward for taking the action.
            done (bool): Whether the episode is finished.
        """
        if action not in self.action_space:
            raise ValueError(f"Invalid action: {action}. Valid actions: {self.action_space}")

        # Update the state with the domino label
        next_state = self.transition_function(self.state, action)

        # Update the domino strings (top and bottom)
        self.update_domino_strings(action)

        # Check if the top and bottom strings are equal
        done = self.domino_strings[0] == self.domino_strings[1]

        # Reward logic
        reward = 1 if done else -0.1

        self.state = next_state
        return next_state, reward, done

    def render(self):
        """Renders the current state."""
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
