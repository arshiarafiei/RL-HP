import numpy as np
import random

class ManufacturingEnv:
    def __init__(self, n_agents=5, n_resources=5, grid_size=8, max_gems=5, gem_limits={'y': 20, 'g': 20, 'b': 20}):
        self.n_agents = n_agents
        self.n_resources = n_resources
        self.grid_size = grid_size
        self.max_gems = max_gems  
        self.gem_limits = gem_limits  
        self.env = np.zeros((grid_size, grid_size))
        self.agents = [np.random.randint(0, grid_size, 2) for _ in range(n_agents)]
        self.resources = [np.random.randint(0, grid_size, 2) for _ in range(n_resources)]
        self.resource_types = [random.choice(['y', 'g', 'b']) for _ in range(n_resources)]
        self.possession = [[0, 0, 0] for _ in range(n_agents)]  
        self.requirements = [random.choices(range(1, 3), k=3) for _ in range(n_agents)]
        self.products = 0  

        self.gem_counts = {'y': 0, 'g': 0, 'b': 0}  
        for gem_type in self.resource_types:
            self.gem_counts[gem_type] += 1

        self.update_env()

    def reset(self):
        self.env = np.zeros((self.grid_size, self.grid_size))
        self.agents = [np.random.randint(0, self.grid_size, 2) for _ in range(self.n_agents)]
        self.resources = [np.random.randint(0, self.grid_size, 2) for _ in range(self.n_resources)]
        self.resource_types = [random.choice(['y', 'g', 'b']) for _ in range(self.n_resources)]
        self.possession = [[0, 0, 0] for _ in range(self.n_agents)]
        self.products = 0

        self.gem_counts = {'y': 0, 'g': 0, 'b': 0}
        for gem_type in self.resource_types:
            self.gem_counts[gem_type] += 1

        self.update_env()

    def update_env(self):
        self.env *= 0
        for i, agent in enumerate(self.agents):
            self.env[agent[0], agent[1]] = i + 1  # Mark agents
        for i, resource in enumerate(self.resources):
            if self.resource_types[i] == 'y':
                self.env[resource[0], resource[1]] = 9  # Yellow gem
            elif self.resource_types[i] == 'g':
                self.env[resource[0], resource[1]] = 8  # Green gem
            elif self.resource_types[i] == 'b':
                self.env[resource[0], resource[1]] = 7  # Blue gem

    def step(self, actions):
        new_agents = []

        # Move agents
        for i, (x, y) in enumerate(self.agents):
            action = actions[i]
            if action == 1:  # Move up
                x = max(0, x - 1)
            elif action == 2:  # Move down
                x = min(self.grid_size - 1, x + 1)
            elif action == 3:  # Move left
                y = max(0, y - 1)
            elif action == 4:  # Move right
                y = min(self.grid_size - 1, y + 1)
            new_agents.append([x, y])

        self.agents = new_agents
        rewards = [0] * self.n_agents

        
        for i, agent in enumerate(self.agents):
            for j, resource in enumerate(self.resources):
                if list(agent) == list(resource):  # Agent collects the gem
                    gem_type = self.resource_types[j]
                    if gem_type == 'y':
                        self.possession[i][0] += 1
                    elif gem_type == 'g':
                        self.possession[i][1] += 1
                    elif gem_type == 'b':
                        self.possession[i][2] += 1
                    rewards[i] += 0.01

                    
                    if len(self.resources) < self.max_gems and self.gem_counts[gem_type] < self.gem_limits[gem_type]:
                        self.resources[j] = np.random.randint(0, self.grid_size, 2)
                        new_gem_type = random.choice(['y', 'g', 'b'])
                        self.resource_types[j] = new_gem_type
                        self.gem_counts[new_gem_type] += 1

        # Calculate products manufactured
        min_parts = float('inf')
        for i in range(self.n_agents):
            parts = min(
                self.possession[i][j] // self.requirements[i][j]
                if self.requirements[i][j] > 0 else float('inf')
                for j in range(3)
            )
            min_parts = min(min_parts, parts)

        self.products += min_parts
        for i in range(self.n_agents):
            for j in range(3):
                self.possession[i][j] -= self.requirements[i][j] * min_parts

        self.update_env()
        return self.env, self.agents, self.resources, self.resource_types, rewards, self.products

    def render(self):
        print("Environment:")
        for i in range(self.grid_size):
            row = []
            for j in range(self.grid_size):
                if self.env[i, j] == 0:
                    row.append(".")  # Empty grid
                elif self.env[i, j] in range(1, self.n_agents + 1):
                    row.append(f"A{int(self.env[i, j])}")  # Agents
                elif self.env[i, j] == 9:
                    row.append("Y")  # Yellow gem
                elif self.env[i, j] == 8:
                    row.append("G")  # Green gem
                elif self.env[i, j] == 7:
                    row.append("B")  # Blue gem
            print(" ".join(row))
        print("Agents' Positions:", self.agents)
        print("Resources' Positions and Types:", list(zip(self.resources, self.resource_types)))
        print("Agents' Possessions:", self.possession)
        print("Products Manufactured:", self.products)
        print("Gem Counts:", self.gem_counts)



if __name__ == "__main__":
    env = ManufacturingEnv(gem_limits={'y': 10, 'g': 15, 'b': 20})
    env.render()

    actions = [1,1,1,1,1]  
    env.step(actions)
    env.render()
