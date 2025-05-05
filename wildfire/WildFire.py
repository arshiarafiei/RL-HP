import gymnasium as gym
from gymnasium import spaces 
import numpy as np

class WildFireEnv(gym.Env):
    def __init__(self, n_grid = 3, method = "baseline"):
        super(WildFireEnv, self).__init__()

        self.n_grid = n_grid
        self.method = method

        self.grid_size = (self.n_grid, self.n_grid) 
        self.FF = [2, 0]  
        self.med = [2, 0]  
        self.fire = [[0, 2], [1, 2], [2, 2]]  
        self.victims = [[0, 0], [1, 2]]
        self.victim_saved = 0
        self.fire_ex = 0
        self.trajectory = list()
        self.max_step = 100
        self.trunct = False

        self.action_space = spaces.MultiDiscrete([5, 5]) 
        # self.observation_space = spaces.Box(low=0, high=13, shape=(self.n_grid*self.n_grid,), dtype=np.int32)  

        # 14 possible values (0‒13) for each grid cell
        self.observation_space = spaces.MultiDiscrete(np.full(self.n_grid * self.n_grid, 14, dtype=np.int32))


    def get_observation(self):
        grid = np.zeros((self.n_grid, self.n_grid), dtype=np.int32)

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

    def step(self, action, model = 'train'):

        # print('action',action)

        moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]  # (dy, dx) - Up, Down, Left, Right, Stay

        # print("FF", self.FF)
        # print("med", self.med)

        if model == 'inference':
            act = action[0]
            new_FF = [self.FF[0] + moves[int(act[0])][0], self.FF[1] + moves[int(act[0])][1]]
            new_med = [self.med[0] + moves[int(act[1])][0], self.med[1] + moves[int(act[1])][1]]


        # Move agents
        new_FF = [self.FF[0] + moves[int(action[0])][0], self.FF[1] + moves[int(action[0])][1]]
        new_med = [self.med[0] + moves[int(action[1])][0], self.med[1] + moves[int(action[1])][1]]

        # print("new_FF", new_FF)
        # print("new_med", new_med)

        
        # Keep agents within grid boundaries
        self.FF = np.clip(new_FF, 0, self.n_grid-1).tolist()
        self.med = np.clip(new_med, 0, self.n_grid-1).tolist()

        # print("new_FF1", self.FF)
        # print("new_med1", self.med)



        if self.med in self.victims:
            self.victims.remove(self.med)  # Remove rescued victim
            self.victim_saved += 1



        if self.FF in self.fire:
            self.fire.remove(self.FF)  # Extinguish fire
            self.fire_ex += 1

        terminated = len(self.fire) == 0 and len(self.victims) == 0
        sub_goals = [len(self.fire) == 0 , len(self.victims) == 0]

        self.trajectory.append((self.FF, self.med, self.calculate_distance_med_FF()))

        reward = self.reward() 
        info = {
        "fires_extinguished": self.fire_ex,
        "victims_saved": self.victim_saved,
        "sub_goals": sub_goals}

        
        if len(self.trajectory) > self.max_step:
            terminated = True
            self.trunct =True


        return self.get_observation(), reward, terminated, self.trunct, info
    
    def calculate_distance_med_FF(self):
        return abs(self.FF[0] - self.med[0]) + abs(self.FF[1] - self.med[1])

    def calculate_distance(self, start, target):
        return abs(start[0] - target[0]) + abs(start[1] - target[1])
    
    def reward(self):
        if self.method == "baseline":
            reward = -0.5
            if self.FF in self.fire:
                reward += 50  
            if self.med in self.fire:
                reward += -100 
            if self.med in self.victims:
                reward += 10
            if self.calculate_distance_med_FF() > 2:
                reward += -100
            if self.calculate_distance_med_FF() <=2:
                reward += 1
            return reward
        
    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        self.FF = [self.n_grid -1, 0]  
        self.med = [self.n_grid -1, 0] 
        self.fire = [[0, self.n_grid -1], [1, self.n_grid -1], [2, self.n_grid -1]]  
        self.victims = [[0, 0], [1, self.n_grid -1]]
        self.victim_saved = 0
        self.fire_ex = 0 
        return self.get_observation(), {}
    


    def render(self):
        grid = np.full((self.n_grid, self.n_grid), ' . ', dtype=object)  

        temp_victim = self.victims.copy()

        for v in self.victims:
            grid[tuple(v)] = 'V'

        for f in self.fire:
            grid[tuple(f)] = '🔥'
            if f in  temp_victim:
                grid[tuple(f)] = 'V🔥'
                temp_victim.remove(f)
   

        if self.FF == self.med:
            grid[tuple(self.FF)] = "FM" 
        elif self.med in temp_victim:
            grid[tuple(self.med)] = 'MDV'
        elif self.FF in temp_victim:
            grid[tuple(self.FF)] = 'FFV'
        elif self.FF in self.fire:
            grid[tuple(self.FF)] = 'FF🔥'
        elif self.med in self.fire:
            grid[tuple(self.med)] = 'MD🔥' 
        else:
            grid[tuple(self.FF)] = "FF"  # Firefighter
            grid[tuple(self.med)] = 'MD'  # Medic
            
        formatted_grid = "\n".join(["  ".join(f"{cell:3}" for cell in row) for row in grid])

        print(formatted_grid)

    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])



if __name__ == "__main__":

    env = WildFireEnv()
    env.reset()
    print(env.observation_space)
    env.render()

    done = False
    step = 0

    print(env.observation_space.sample())
    print(env.observation_space)
    for i in range(20):
        
        action = env.action_space.sample()
        
        obs, reward, done, trunct, info = env.step(action)
        step += 1 
        # env.render()
        # print("step", step)
        # print("reward", reward)
        print("obs ", obs)
    #     print("terminate", done)
    #     print("info", info)
    #     print("trunct", trunct)
    #     print("############################## \n \n##############################")





