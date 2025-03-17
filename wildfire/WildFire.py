import gym
from gym import spaces
import numpy as np

class WildFireEnv(gym.Env):
    def __init__(self):
        super(WildFireEnv, self).__init__()

        self.grid_size = (3, 3) 
        self.FF = [2, 0]  
        self.med = [2, 0]  
        self.fire = [[0, 2], [1, 2], [2, 2]]  
        self.victims = [[0, 0], [1, 2]]
        self.victim_saved = 0
        self.fire_ex = 0

        self.action_space = spaces.MultiDiscrete([5, 5])  
        self.observation_space = spaces.Box(low=0, high=5, shape=(9,), dtype=np.int32)  

    def get_observation(self):
        grid = np.zeros((3, 3), dtype=np.int32)

        if self.FF == self.med:
            grid[tuple(self.FF)] = 3  
        else:
            grid[tuple(self.FF)] = 1  
            grid[tuple(self.med)] = 2

        for f in self.fire:
            grid[tuple(f)] = 4  

        for v in self.victims:
            if v in self.fire:
                grid[tuple(v)] = 8 # Victim in Fire
            else:
                grid[tuple(v)] = 5
    


        if self.FF in self.fire:
            grid[tuple(self.FF)] = 6 #FF and Fire
        if self.FF in self.victims:
            grid[tuple(self.FF)] =  7 # FF and victim
        

        if self.med in self.fire:
            grid[tuple(self.FF)] = 9 #med in Fire
        if self.med in self.victims:
            grid[tuple(self.FF)] =  10 # med and victim


        if self.FF == self.med and self.FF in self.victims:
            grid[tuple(self.FF)] = 11 
        if self.FF == self.med and self.FF in self.fire:
            grid[tuple(self.FF)] = 12
        if self.FF == self.med and self.FF in self.fire and self.FF in self.victims:
            grid[tuple(self.FF)] = 13
        

        
        return grid.flatten()  

    def step(self, action):
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]  # (dy, dx) - Up, Down, Left, Right, Stay

        # Move agents
        new_FF = [self.FF[0] + moves[action[0]][0], self.FF[1] + moves[action[0]][1]]
        new_med = [self.med[0] + moves[action[1]][0], self.med[1] + moves[action[1]][1]]

        # Keep agents within grid boundaries
        self.FF = np.clip(new_FF, 0, 2).tolist()
        self.med = np.clip(new_med, 0, 2).tolist()


        if self.med in self.victims:
            self.victims.remove(self.med)  # Remove rescued victim



        if self.FF in self.fire:
            self.fire.remove(self.FF)  # Extinguish fire

        terminated = len(self.fire) == 0 and len(self.victims) == 0
        sub_goals = [len(self.fire) == 0 , len(self.victims) == 0]
        reward = -1  
        info = [len(self.fire) == 0 , len(self.victims) == 0]

        return self.get_observation(), reward, terminated, sub_goals, info

    def reset(self, seed=None, options=None):
        self.FF = [2, 0]  
        self.med = [2, 0] 
        self.fire = [[0, 2], [1, 2], [2, 2]]  
        self.victims = [[0, 0], [1, 2]]
        self.victim_saved = 0
        self.fire_ex = 0 
        return self.get_observation()
    


    def render(self):
        grid = np.full((3, 3), ' . ', dtype=object)  

        temp_victim = self.victims.copy()


        for f in self.fire:
            grid[tuple(f)] = '🔥'
            if f in  temp_victim:
                grid[tuple(f)] = 'V🔥'
                temp_victim.remove(f)


 
        for v in temp_victim:
            grid[tuple(v)] = 'V'  


        if self.FF == self.med:
            grid[tuple(self.FF)] = "FM"  
        else:
            grid[tuple(self.FF)] = "FF"  # Firefighter
            grid[tuple(self.med)] = 'MD'  # Medic


        formatted_grid = "\n".join(["  ".join(f"{cell:3}" for cell in row) for row in grid])

        print(formatted_grid)

    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    

    def reward(self):



env = WildFireEnv()
print(env.observation_space)
env.render()

obs, _ = env.reset()
print(obs)
for i in range(50):
    
    action = env.action_space.sample()
    
    print(obs)
    env.render()
    obs, reward, terminated, truncated, info = env.step(action)




