import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import random
import time 

class Job:
    def __init__(self, n_agent=4, grid_size=5, starting_positions=None, resource_position=None):
        self.n_agent = n_agent
        self.grid_size = grid_size
        self.env = np.zeros((grid_size, grid_size))
        self.resource = resource_position if resource_position else (grid_size - 1, grid_size - 1)  # Fixed or default resource location
        self.starting_positions = starting_positions if starting_positions else [(0, 0), (0, 2), (0, 4), (3, 3)]  # Fixed or default starting positions
        self.state = self.starting_positions[:n_agent]

        self.update_env()

    def reset(self):
        self.env = np.zeros((self.grid_size, self.grid_size))
        self.state = self.starting_positions[:self.n_agent]  # Reset to fixed starting positions
        self.update_env()
        return self.env, self.state, self.resource

    def update_env(self):
        self.env.fill(0)
        self.env[self.resource[0], self.resource[1]] = 9  # Mark the resource with 9
        for i, agent in enumerate(self.state):
            self.env[agent[0], agent[1]] = i + 1  # Mark agents with unique IDs

    def step(self, action):
        next_positions = []

        for i, (x, y) in enumerate(self.state):
            if action[i] == 1:  # Move up
                x = max(0, x - 1)
            elif action[i] == 2:  # Move down
                x = min(self.grid_size - 1, x + 1)
            elif action[i] == 3:  # Move left
                y = max(0, y - 1)
            elif action[i] == 4:  # Move right
                y = min(self.grid_size - 1, y + 1)

            next_positions.append((x, y))

        # Resolve conflicts
        unique_positions = []
        for pos in next_positions:
            if pos not in unique_positions:
                unique_positions.append(pos)
            else:
                unique_positions.append(self.state[len(unique_positions)])  # Stay in place

        self.state = unique_positions  
        self.update_env()  

        return self.env, self.state, self.resource
    def calculate_distance(self, state):
        return abs(state[0] - self.resource[0]) + abs(state[1] - self.resource[1])

class DQNAgent:
    def __init__(self, state_size, action_size, n_agent, learning_rate=0.001, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.n_agent = n_agent
        self.memory = []
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.n_agent * self.action_size, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return [random.randint(0, self.action_size - 1) for _ in range(self.n_agent)]
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
        return np.argmax(q_values.reshape(self.n_agent, self.action_size), axis=1)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_q_values = self.model.predict(np.expand_dims(next_state, axis=0), verbose=0)
                target = reward + self.gamma * np.max(next_q_values)

            q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
            for i, a in enumerate(action):
                q_values[0, i * self.action_size + a] = target

            self.model.fit(np.expand_dims(state, axis=0), q_values, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def reward_fairness(env,trajectory1, trajectory2):

    agent1_dist = list()
    agent2_dist = list()
    
    for i in range(len(trajectory1)):
        agent1_dist.append(50 - 10*env.calculate_distance(trajectory1[i]))
        agent2_dist.append(50 - 10*env.calculate_distance(trajectory2[i]))
    
    temp_ag1= list()
    temp_ag2= list()

    for index in range(len(agent1_dist)):
        
        ag1_num = max(agent1_dist[len(agent1_dist)-index-1:])
        temp_ag1.append(ag1_num)
        
        ag2_num = max(agent2_dist[len(agent2_dist)-index-1:])
        temp_ag2.append(ag2_num)

    
    phi_1 = min(temp_ag1)
    phi_2 = min(temp_ag2)

    reward = min(phi_1, phi_2)

    return reward

def log(step,actions,grid,state, next_state, next_grid,reward):
    print("step:", step)
    print("actions",actions)
    print(grid)
    print("current state: ",state)
    print("after action state: ",next_state)
    print(next_grid)
    print("Reward:", reward)

    print("-"*20)
    print("-"*20)
    




def train_dqn(env, agent, episodes=1000, batch_size=32,step_size = 10000):
    
    grid, state, resource = env.reset()
    state_flat = np.array([coord for agent in state for coord in agent])

    total_reward = 0
    done = False
    trajectory1 = list()
    trajectory2 = list()
    trajectory1.append(state[0])
    trajectory2.append(state[1])


    for step in range(step_size):
        actions = agent.act(state_flat)


        next_grid, next_state, resource = env.step(actions)


        trajectory1.append(next_state[0])
        trajectory2.append(next_state[1])







        next_state_flat = np.array([coord for agent in next_state for coord in agent])

        reward = reward_fairness(env, trajectory1, trajectory2)

        log(step,actions,grid,state, next_state, next_grid,reward)

        

        agent.remember(state_flat, actions, reward, next_state_flat, done)
        state_flat = next_state_flat
        total_reward += reward
        grid = next_grid
        state = next_state
        time.sleep(0.1)
        if agent.epsilon > agent.epsilon_min and reward >30:
                agent.epsilon *= agent.epsilon_decay
        if step % 50 == 0:  # Adjust epsilon every 100 steps
            print(f"Step {step}/{step_size} - Total Reward: {reward} - Epsilon: {agent.epsilon:.2f}")

    # print(f"Episode {e + 1}/{episodes} - Total Reward: {total_reward} - Epsilon: {agent.epsilon:.2f}")

    # if len(agent.memory) > batch_size:
    #     agent.replay(batch_size)


if __name__ == "__main__":

    # Main parameters
    n_agent = 2
    grid_size = 5
    starting_positions = [(0, 0), (3, 3)]  # Custom starting positions
    resource_position = (1, 4)  # Custom resource position

    action_size = 5  # [Stay, Up, Down, Left, Right]
    state_size = n_agent * 2  

    
    env = Job(n_agent=n_agent, grid_size=grid_size, starting_positions=starting_positions, resource_position=resource_position)
    agent = DQNAgent(state_size, action_size, n_agent)

    train_dqn(env, agent, 500, step_size=100000)
