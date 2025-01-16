import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from copy import deepcopy

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

    def __init__(self, normalize, resource_type):
        #shared parameters
        self.neighbors_size = 8
        self.T = 25
        self.max_steps = 1000
        self.n_signal = 4
        self.resource_type = resource_type
        if self.resource_type != 'all':
            self.n_agent = 3
            self.n_actions = 2
            self.n_episode = 4000
            self.max_u = 1/3
            self.n_neighbors = 2
        else:
            self.n_agent = 4
            self.n_actions = 5
            self.n_episode = 10000
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
        self.target = np.random.randint(2, 5, 2)
        self.ant = []
        for i in range(self.n_agent):
            candidate = list(np.random.randint(1, 6, 2))
            while candidate in self.ant:
                candidate = list(np.random.randint(1, 6, 2))
            self.ant.append(candidate)
            self.env[self.ant[i][0]][self.ant[i][1]] = 1
        self.rinfo = np.array([0.] * self.n_agent)

        return self._get_obs()

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
        if self.resource_type != 'all':
            action = list(deepcopy(action))
            for i in range(self.n_agent):
                if action[i] != 0:
                    if self.target[0] < self.ant[i][0]:
                        action[i]=1
                    elif self.target[0] > self.ant[i][0]:
                        action[i]=2
                    elif self.target[1] < self.ant[i][1]:
                        action[i]=3
                    elif self.target[1] > self.ant[i][1]:
                        action[i]=4
                    else:
                        action[i]=0
                else:
                    action[i] = np.random.randint(1, 5)

        next_ant = []


        #1: move left, 2: move right, 3: move down, 4: Move up

        for i in range(self.n_agent):
            x = self.ant[i][0]
            y = self.ant[i][1]
            if action[i] == 0:
                next_ant.append([x, y])
            if action[i] == 1:
                x = x - 1
                if x == 0:
                    next_ant.append([x + 1, y])
                    continue
                if self.env[x][y] != 1:
                    self.env[x][y] = 1
                    next_ant.append([x, y])
                else:
                    next_ant.append([x + 1, y])
            if action[i] == 2:
                x = x + 1
                if x == 6:
                    next_ant.append([x - 1, y])
                    continue
                if self.env[x][y] != 1:
                    self.env[x][y] = 1
                    next_ant.append([x, y])
                else:
                    next_ant.append([x - 1, y])
            if action[i] == 3:
                y = y - 1
                if y == 0:
                    next_ant.append([x, y + 1])
                    continue
                if self.env[x][y] != 1:
                    self.env[x][y] = 1
                    next_ant.append([x, y])
                else:
                    next_ant.append([x, y + 1])
            if action[i] == 4:
                y = y + 1
                if y == 6:
                    next_ant.append([x, y - 1])
                    continue
                if self.env[x][y] != 1:
                    self.env[x][y] = 1
                    next_ant.append([x, y])
                else:
                    next_ant.append([x, y - 1])
        self.ant = next_ant
        self.env *= 0
        re = [0.] * self.n_agent
        for i in range(self.n_agent):
            self.env[self.ant[i][0]][self.ant[i][1]] = 1
            if (self.ant[i][0] == self.target[0]) & (self.ant[i][1] == self.target[1]):
                re[i] = 1

        self.rinfo += re
        return self._get_obs(), re, False

    def end_episode(self):
        self.fileresults.write(','.join(self.rinfo.flatten().astype('str')) + '\n')
        self.fileresults.flush()

    def render(self, mode="visual"):
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
            grid[self.target[0]][self.target[1]] = " T "  # Target location
            for idx, ant in enumerate(self.ant):
                grid[ant[0]][ant[1]] = f" A{idx}"  # Agent location with number
            print("\n".join("".join(row) for row in grid))
            print("-" * 16)

def test_environment():
    import time

    # Initialize the environment with normalization and resource type 'all'
    env = Env(normalize=False, resource_type='all')

    # Number of episodes and steps per episode for testing
    num_episodes = 20
    max_steps_per_episode = 5

    for episode in range(num_episodes):
        print(f"Starting Episode {episode + 1}")
        
        # Reset the environment at the start of each episode
        obs = env.reset()
        print(f"Initial Observations: {obs}")
        
        for step in range(max_steps_per_episode):
            print(f"\nStep {step + 1}:")
            
            # Take random actions for each agent
            actions = [4 for _ in range(env.n_agent)]
            print(f"Actions taken: {actions}")

            # Perform the action and get the next observation, reward, and done flag
            obs, rewards, done = env.step(actions)

            # Log the results of this step
            print(f"Observations: {obs}")
            print(f"Rewards: {rewards}")
            print(f"Done: {done}")

            # Render the environment to visualize it
            env.render(mode="console")

            # Pause briefly to observe the rendering
            time.sleep(0.5)

        # End the episode and log results
        env.end_episode()
        print(f"Episode {episode + 1} ended.")

if __name__ == "__main__":
    test_environment()

