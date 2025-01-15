import numpy as np

class Job:
    def __init__(self, n_agent=4, grid_size=5):
        self.n_agent = n_agent
        self.grid_size = grid_size
        self.env = np.zeros((grid_size, grid_size))
        self.resource = np.random.randint(0, grid_size, 2)  # Random resource location
        self.state = [np.random.randint(0, grid_size, 2) for _ in range(n_agent)]

        
        while len({tuple(agent) for agent in self.state}) < n_agent or any(np.array_equal(agent, self.resource) for agent in self.state):
            self.state = [np.random.randint(0, grid_size, 2) for _ in range(n_agent)]

        self.update_env()

    def reset(self):
        self.env = np.zeros((self.grid_size, self.grid_size))
        self.resource = np.random.randint(0, self.grid_size, 2)  
        self.state = [np.random.randint(0, self.grid_size, 2) for _ in range(self.n_agent)]

        
        while len({tuple(agent) for agent in self.state}) < self.n_agent or any(np.array_equal(agent, self.resource) for agent in self.state):
            self.state = [np.random.randint(0, self.grid_size, 2) for _ in range(self.n_agent)]

        self.update_env()
        return self.env, self.state, self.resource

    def update_env(self):
        self.env *= 0
        self.env[self.resource[0], self.resource[1]] = 9  # Mark the resource with 9
        for i, agent in enumerate(self.state):
            self.env[agent[0], agent[1]] = i + 1  

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

            next_positions.append([x, y])

        # conflicts
        unique_positions = []
        for pos in next_positions:
            if pos not in unique_positions:
                unique_positions.append(pos)
            else:
                unique_positions.append(self.state[len(unique_positions)])  # Stay in place

        self.state = unique_positions  
        self.update_env()  

        return self.env, self.state, self.resource  

if __name__ == "__main__":
    n_agents = 4
    env = Job(n_agent=n_agents, grid_size=5)

    # Print initial environment state
    print("Initial Environment:")
    print(env.env)

   
    # Actions: 0 = stay, 1 = move up, 2 = move down, 3 = move left, 4 = move right
    actions = [2, 2, 2, 2]  # Example actions for all agents

    new_env, new_state, resource = env.step(actions)

    
    print("Updated Environment:")
    print(new_env)
    
