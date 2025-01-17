import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from copy import deepcopy
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import random
from collections import deque

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon
        self.clipob = 10.
        self.epsilon = 1e-8

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

    def obs_filter(self, obs):
        self.update(obs)
        obs = np.clip((obs - self.mean) / np.sqrt(self.var + self.epsilon), -self.clipob, self.clipob)
        return obs


class Env():

    def __init__(self, normalize):
        #shared parameters
        self.neighbors_size = 8
        self.T = 25
        self.max_steps = 1000
        self.n_signal = 4
        
        self.n_agent = 4
        self.n_actions = 5
        self.n_episode = 1000
        self.max_u = 0.25
        self.n_neighbors = 3
    
        self.input_size = 13
        self.nD = self.n_agent
        self.GAMMA = 0.98

        self.fileresults = open('learning.data', "w")
        self.normalize = normalize
        self.compute_neighbors = False
        if normalize:
            self.obs_rms = [RunningMeanStd(shape=self.input_size) for _ in range(self.n_agent)]

    def __del__(self):
        self.fileresults.close()

    def toggle_compute_neighbors(self):
        self.compute_neighbors = True

    def neighbors(self):
        assert self.compute_neighbors
        return self.compute_neighbors_last, self.compute_neighbors_last_index

    def reset(self):
        self.env = np.zeros((8, 8))
        self.target = [4, 4]  # Fixed target position
        self.ant = [
            [0, 0],  # Predefined position for agent 1
            [0, 6],  # Predefined position for agent 2
            [6, 0],  # Predefined position for agent 3
            [6, 6],  # Predefined position for agent 4
        ]
        for i in range(self.n_agent):
            self.env[self.ant[i][0]][self.ant[i][1]] = 1  # Place agents on the grid
        self.rinfo = np.array([0.] * self.n_agent)

        # Ensure state consistency by flattening
        flattened_state = np.array(self.ant).flatten()
        return flattened_state.reshape(1, -1)



    def _get_obs(self):
        if self.compute_neighbors:
            distances = distance_matrix(self.ant, self.ant, p=float('+inf'))
            distances = np.array(distances).astype(np.float)
            for i in range(len(self.ant)):
                distances[i,i]=float('+inf')
            distances = np.argsort(distances)[:,:self.n_neighbors]
            self.compute_neighbors_last = distances

            self.compute_neighbors_last_index=[[] for _ in range(self.n_agent)]
            for k in range(len(self.ant)):
                index = 0
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if i != 0 or j != 0:
                            if self.env[self.ant[k][0] + i][self.ant[k][1] + j] == 1:
                                self.compute_neighbors_last_index[k].append(index)
                            index += 1

        h = []
        for k in range(self.n_agent):
            state = []
            state.append(self.ant[k][0])
            state.append(self.ant[k][1])
            state.append(self.target[0] - self.ant[k][0])
            state.append(self.target[1] - self.ant[k][1])
            for i in range(-1, 2):
                for j in range(-1, 2):
                    state.append(self.env[self.ant[k][0] + i][self.ant[k][1] + j])
            h.append(state)

        if self.normalize:
            for i in range(self.n_agent):
                h[i] = list(self.obs_rms[i].obs_filter(np.array(h[i])))

        return h

    def step(self, action):
        next_ant = []

        # 1: move left, 2: move right, 3: move down, 4: move up
        for i in range(self.n_agent):
            x, y = self.ant[i][0], self.ant[i][1]
            if action[i] == 0:
                next_ant.append([x, y])
            elif action[i] == 1:  # Move left
                x = max(0, x - 1)
            elif action[i] == 2:  # Move right
                x = min(7, x + 1)
            elif action[i] == 3:  # Move down
                y = max(0, y - 1)
            elif action[i] == 4:  # Move up
                y = min(7, y + 1)

            if self.env[x][y] != 1:  # Check if the position is already occupied
                self.env[x][y] = 1
                next_ant.append([x, y])
            else:
                next_ant.append(self.ant[i])  # Stay in the same position if blocked

        self.ant = next_ant
        self.env *= 0  # Reset the grid
        rewards = [0.] * self.n_agent

        for i in range(self.n_agent):
            self.env[self.ant[i][0]][self.ant[i][1]] = 1
            if (self.ant[i][0] == self.target[0]) and (self.ant[i][1] == self.target[1]):
                rewards[i] = 1  # Reward if the agent reaches the target

        self.rinfo += rewards

        # Ensure state consistency by truncating or padding
        flattened_state = np.array(self.ant).flatten()
        max_length = self.n_agent * 2
        if len(flattened_state) > max_length:
            flattened_state = flattened_state[:max_length]
        elif len(flattened_state) < max_length:
            flattened_state = np.pad(flattened_state, (0, max_length - len(flattened_state)), mode='constant')

        return flattened_state.reshape(1, -1), rewards, False




    def end_episode(self):
        self.fileresults.write(','.join(self.rinfo.flatten().astype('str')) + '\n')
        self.fileresults.flush()

    def render(self, mode="console"):
        if mode == "visual":
            for i in range(self.n_agent):
                theta = np.arange(0, 2 * np.pi, 0.01)
                x = self.ant[i][0] + 0.05 * np.cos(theta)
                y = self.ant[i][1] + 0.05 * np.sin(theta)
                plt.plot(x, y)

            plt.scatter(self.target[0], self.target[1], color='green')
            plt.axis("equal")
            plt.xlim(-1, 7)
            plt.ylim(-1, 7)
            plt.pause(0.1)
            plt.cla()
        elif mode == "console":
            grid = [[" . " for _ in range(8)] for _ in range(8)]
            grid[self.target[0]][self.target[1]] = " T "  
            for idx, ant in enumerate(self.ant):
                grid[ant[0]][ant[1]] = f" A{idx}"  
            print("\n".join("".join(row) for row in grid))
            print("-" * 16)

# def test_environment():
#     import time

#     env = Env(normalize=False, resource_type='all')

    
#     num_episodes = 20
#     max_steps_per_episode = 5

#     for episode in range(num_episodes):
#         print(f"Starting Episode {episode + 1}")
        
        
#         obs = env.reset()
#         print(f"Initial Observations: {obs}")
        
#         for step in range(max_steps_per_episode):
#             print(f"\nStep {step + 1}:")
            
            
#             actions = [4 for _ in range(env.n_agent)]
#             print(f"Actions taken: {actions}")

            
#             obs, rewards, done = env.step(actions)

#             # Log the results of this step
#             print(f"Observations: {obs}")
#             print(f"Rewards: {rewards}")
#             print(f"Done: {done}")

            
#             env.render(mode="console")

            
#             time.sleep(0.5)

#         env.end_episode()

# if __name__ == "__main__":
#     test_environment()



class DQNAgent:
    def __init__(self, state_size, action_size, n_agents):
        self.state_size = state_size  # Flattened state size
        self.action_size = action_size  # Number of actions per agent
        self.n_agents = n_agents  # Number of agents
        self.memory = deque(maxlen=2000)
        self.gamma = 0.98  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(tf.keras.Input(shape=(self.state_size,)))  # Input for flattened state
        model.add(Dense(64, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size * self.n_agents, activation='linear'))  # Actions for all agents
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, actions, rewards, next_state, done):
        self.memory.append((state, actions, rewards, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # Random actions for each agent
            return [random.randint(0, self.action_size - 1) for _ in range(self.n_agents)]
        state = np.array(state).flatten().reshape(1, -1)  # Flatten the state
        act_values = self.model.predict(state, verbose=0)
        # Extract actions for each agent
        actions = [np.argmax(act_values[0][i * self.action_size:(i + 1) * self.action_size]) for i in range(self.n_agents)]
        return actions

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, actions, rewards, next_state, done in minibatch:
            # Ensure consistent state size
            state = np.array(state).reshape(1, -1)
            next_state = np.array(next_state).reshape(1, -1)

            target = self.model.predict(state, verbose=0)
            for i in range(self.n_agents):
                action_index = actions[i]
                reward = rewards[i]
                if not done:
                    reward += self.gamma * np.amax(
                        self.model.predict(next_state, verbose=0)[0][i * self.action_size:(i + 1) * self.action_size]
                    )
                target[0][i * self.action_size + action_index] = reward

            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay






# Initialize environment and agent
env = Env(normalize=False)
n_agents = env.n_agent
state_size = n_agents * 2  # 2 features (x, y) per agent
action_size = 5  # Number of possible actions
agent = DQNAgent(state_size=n_agents * 2 , action_size=action_size, n_agents=n_agents)
batch_size = 32
episodes = 20
print(agent.model.summary())


for e in range(episodes):
    state = env.reset()  # State: [[x1, y1], [x2, y2], ...]
    state = np.array(state).flatten().reshape(1, -1)  # Flatten the state
    print(f"Reset state shape: {state.shape}")
    env.render()

    

    for time in range(100):
        actions = agent.act(state)  # Get actions for all agents
        print(actions)
        next_state, rewards, done = env.step(actions)
        
        next_state = np.array(next_state).flatten().reshape(1, -1)  # Flatten the next state
        env.render()

        print(rewards)
        agent.remember(state, actions, rewards, next_state, done)
        state = next_state

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        if done:
            break

    print(f"Episode {e}/{episodes}, epsilon: {agent.epsilon:.2f}")


print("Training complete.")




