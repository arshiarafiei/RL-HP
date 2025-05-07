import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from collections import deque
import pandas as pd
import random

import os



class PCPMDPEnv:
    def __init__(self):
        self.dominos_context = {
            1: ("a", "ab"),
            2: ("bc", "cd"),
            3: ("de", "ef"),
            4: ("f", "gh"),
            5: ("gh", "i"),
            6: ("ij","j"),
        }
        self.action_space = [1, 2, 3, 4, 5, 6]
        # self.dominos_context = {
        #     1: ("a", "ab"),
        #     2: ("bcd", "c"),
        #     3: ("e", "d"),
        #     4: ("f", "efg"),
        #     5: ("gh", "h")
        # }
        # self.action_space = [1, 2, 3, 4, 5]
        self.state = None
        self.domino_strings = ("", "")

    def reset(self):
        """Reset the environment to its initial state."""
        self.state = []
        self.domino_strings = ("", "")
        return self.state

    def step(self, action):
        """Execute an action and update the environment."""
        if action not in self.action_space:
            raise ValueError(f"Invalid action: {action}. Valid actions: {self.action_space}")

        domino_top, domino_bottom = self.dominos_context[action]
        self.domino_strings = (
            self.domino_strings[0] + domino_top,
            self.domino_strings[1] + domino_bottom
        )
        self.state.append(action)
        return self.state, self.domino_strings

    def domino_cont(self, state):
        """Reconstruct domino strings based on state."""
        top_string, bottom_string = "", ""
        for action in state:
            top, bottom = self.dominos_context[action]
            top_string += top
            bottom_string += bottom
        return top_string, bottom_string


def reward_until(env, state, lab):
    def count_diff(str1, str2):
        if len(str1) != len(str2):
            raise ValueError("Strings must be of the same length.")
        differences = sum(1 for a, b in zip(str1, str2) if a != b)
        return differences

    if len(state) <= 1:
        domino = env.domino_cont(state)
        domino_top = domino[0] + "#"
        domino_bottom = domino[1] + "#"

        diff = sum(1 for i in range(min(len(domino_top), len(domino_bottom)) - 1)
                   if domino_top[i] != domino_bottom[i])
        return 1 - diff

    else:
        formula = []
        for t in range(1, len(state)):
            temp_semimatch = state[:t + 1]
            domino = env.domino_cont(temp_semimatch)
            domino_top = domino[0] + "#"
            domino_bottom = domino[1] + "#"

            diff = sum(1 for i in range(min(len(domino_top) - 1, len(domino_bottom) - 1))
                       if domino_top[i] != domino_bottom[i])
            semimatch_list = 1 - diff

            match_list = [semimatch_list]
            for i in range(t + 1, len(state)):
                temp = state[:i + 1]
                domino = env.domino_cont(temp)

                domino_top = domino[0].ljust(len(domino[1]), '#')
                domino_bottom = domino[1].ljust(len(domino[0]), '#')

                diff = count_diff(domino_bottom, domino_top)
                match_list.append(1 - diff )

            formula.append(min(match_list))

        return max(formula)

def count_diff(str1, str2):
    if len(str1) != len(str2):
        raise ValueError("Strings must be of the same length.")
    differences = sum(1 for a, b in zip(str1, str2) if a != b)
    return differences
# Define the Q-network
def create_q_network(state_size, action_size):
    model = Sequential([
        Dense(512, input_dim=state_size, activation='relu'),
        Dense(512, activation='relu'),
        Dense(512, activation='relu'),
        Dense(action_size, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

# Replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self):
        return len(self.buffer)

# Training function
def train_dqn(env, max_steps, num_episodes, batch_size, gamma, epsilon_decay, tr):
    column = ["episode", "tot_done", "reward"]

    df = pd.DataFrame(columns=column)
    folder_path = "data/dom_base6_3"
    state_size = max_steps
    action_size = len(env.action_space)
    q_network = create_q_network(state_size, action_size)
    target_network = create_q_network(state_size, action_size)
    target_network.set_weights(q_network.get_weights())
    replay_buffer = ReplayBuffer(10000)
    epsilon = 1.0
    epsilon_min = 0.01
    total_dones = 0


    for episode in range(num_episodes):
        state = env.reset()
        state = np.zeros(max_steps)  # Initial state is padded
        total_reward = 0
        done = False

        for t in range(max_steps):
            # Epsilon-greedy policy
            if random.random() < epsilon:
                action = random.choice(env.action_space)
            else:
                q_values = q_network.predict(state.reshape(1, -1), verbose=0)
                action = np.argmax(q_values[0]) + 1  # Convert zero-based to one-based indexing


            # Step the environment
            next_state, domino_strings = env.step(action)
            len(domino_strings[0]), len(domino_strings[1])

            domino_top = domino_strings[0].ljust(len(domino_strings[1]), '#')
            domino_bottom = domino_strings[1].ljust(len(domino_strings[0]), '#')

            diff = count_diff(domino_bottom, domino_top)

            reward = max(len(domino_top),len(domino_bottom)) -  5 * diff 
            

            done = domino_strings[0] == domino_strings[1]

            if done:
                total_dones += 1

            # Pad next_state to fixed size
            next_state_padded = np.zeros(max_steps)
            next_state_padded[:len(next_state)] = next_state

            # Store transition in replay buffer
            replay_buffer.push(state, action, reward, next_state_padded, done)

            state = next_state_padded
            total_reward += reward

            # Train the network
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                # Compute Q-values
                target_qs = q_network.predict(states, verbose=0)
                next_qs = target_network.predict(next_states, verbose=0)

                for i in range(batch_size):
                  target = rewards[i]
                  if not dones[i]:
                      target += gamma * np.max(next_qs[i])
                  target_qs[i, actions[i] - 1] = target  # Adjust action indexing


                q_network.fit(states, target_qs, epochs=1, verbose=0, batch_size=batch_size)

            if done:
                break

        # Update epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Update target network
        if episode % 50 == 0:
            target_network.set_weights(q_network.get_weights())

        print(f"{tr} Episode {episode}, Total Reward: {total_reward}, Total Dones: {total_dones}, Total domino: {domino_strings}, Epsilon: {epsilon}")
        arr = [episode, total_dones , total_reward ]
        df.loc[len(df)] = arr
        df.to_csv(f"{folder_path}/data_{tr}.csv", index=False)

    return q_network
if __name__ == "__main__":

    # Environment and training
    env = PCPMDPEnv()
    max_steps = 10 # Maximum state length = max steps
    num_episodes = 1000
    batch_size = 16
    gamma = 0.99
    epsilon_decay = 0.995

    for i in range(1,10):
      folder_path = "/Users/tartmsu/Desktop/RL-HP/data/dom_base6_3"
      os.makedirs(folder_path, exist_ok=True)

      trained_model = train_dqn(env, max_steps, num_episodes, batch_size, gamma, epsilon_decay,i)

