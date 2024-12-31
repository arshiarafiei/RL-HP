import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import random

class SimpleMDPEnv:
    def __init__(self):
        # Define the state space and action space
        self.state_space = [0, 1, 2, 3, 4]  # Example states
        self.action_space = [0, 1]  # Example actions: 0 and 1
        self.state = None  # Current state
        self.atomic_propositions = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e"}  # Atomic propositions for each state

        # Transition array: (current_state, action, next_state)
        self.transitions = [
            (0, 0, 4), (0, 1, 1),
            (1, 0, 0), (1, 1, 2),
            (2, 0, 1), (2, 1, 3),
            (3, 0, 2), (3, 1, 4),
            (4, 0, 3), (4, 1, 0)
        ]

    def reset(self):
        """Resets the environment to an initial state."""
        self.state = np.random.choice(self.state_space)  # Random initial state
        return self.state

    def transition_function(self, current_state, action):
        """Encodes the transition logic between states using a transition array.

        Args:
            current_state (int): The current state.
            action (int): The action to take.

        Returns:
            next_state (int): The next state after taking the action.
        """
        for (cur_state, act, next_state) in self.transitions:
            if cur_state == current_state and act == action:
                return next_state
        raise ValueError(f"Invalid transition for state {current_state} and action {action}.")

    def step(self, action, target_proposition):
        """Takes an action and returns the next state, reward, and done flag.

        Args:
            action (int): The action to take (must be in the action space).
            target_proposition (str): The atomic proposition that marks the terminal state.

        Returns:
            next_state (int): The next state after taking the action.
            reward (float): The reward for taking the action.
            done (bool): Whether the episode is finished.
        """
        if action not in self.action_space:
            raise ValueError(f"Invalid action: {action}. Valid actions: {self.action_space}")

        # Use the transition function to determine the next state
        next_state = self.transition_function(self.state, action)

        # Example reward logic (can be customized)
        reward = 1 if self.atomic_propositions[next_state] == target_proposition else -0.1

        # Terminal condition based on atomic proposition
        done = self.atomic_propositions[next_state] == target_proposition

        self.state = next_state
        return next_state, reward, done

    def render(self):
        """Renders the current state."""
        print(f"Current State: {self.state}, Atomic Proposition: {self.atomic_propositions[self.state]}")

# DQN Agent
class MultiAgentDQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size * 2  # Combined state size for two agents
        self.action_size = action_size * 2  # Combined action size for two agents
        self.memory = []
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return [random.randrange(2), random.randrange(2)]  # Two random actions
        act_values = self.model.predict(state, verbose=0)
        return [np.argmax(act_values[0][:2]), np.argmax(act_values[0][2:])]  # Actions for both agents

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action[0]] = target
            target_f[0][action[1] + 2] = target  # Offset for second agent actions
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Main
if __name__ == "__main__":
    env1 = SimpleMDPEnv()
    env2 = SimpleMDPEnv()

    state_size = 1  # Single integer representing the state
    action_size = len(env1.action_space)

    agent = MultiAgentDQN(state_size, action_size)

    episodes = 1000
    batch_size = 32

    target_proposition1 = input("Enter the target atomic proposition for Environment 1 (e.g., 'e'): ")
    target_proposition2 = input("Enter the target atomic proposition for Environment 2 (e.g., 'e'): ")

    for e in range(episodes):
        state1 = env1.reset()
        state2 = env2.reset()

        combined_state = np.reshape([state1, state2], [1, state_size * 2])

        total_reward1 = 0
        total_reward2 = 0

        for time in range(500):
            # Multi-agent action
            actions = agent.act(combined_state)
            action1, action2 = actions[0], actions[1]

            # Agent 1 environment step
            next_state1, reward1, done1 = env1.step(action1, target_proposition1)
            total_reward1 += reward1

            # Agent 2 environment step
            next_state2, reward2, done2 = env2.step(action2, target_proposition2)
            total_reward2 += reward2

            combined_next_state = np.reshape([next_state1, next_state2], [1, state_size * 2])

            agent.remember(combined_state, actions, reward1 + reward2, combined_next_state, done1 and done2)
            combined_state = combined_next_state

            if done1 and done2:
                print(f"Episode {e+1}/{episodes} - Reward1: {total_reward1}, Reward2: {total_reward2}, Epsilon: {agent.epsilon:.2f}")
                break

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
