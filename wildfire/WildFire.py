import gymnasium as gym
from gymnasium import spaces 
import numpy as np
import math

class WildFireEnv(gym.Env):
    def __init__(self, n_grid = 3, method = "baseline", mode = 'train'):
        super(WildFireEnv, self).__init__()

        self.n_grid = n_grid
        self.method = method

        self.grid_size = (self.n_grid, self.n_grid) 
        self.FF = [2, 0]  
        self.med = [2, 0]  
        self.fire = [[0, 2], [1, 2], [2, 1]]  
        self.victims = [[0, 0], [1, 2]]
        self.victim_saved = 0
        self.fire_ex = 0
        self.trajectory = list()
        if mode == 'train':
            self.max_step = 1000
        else:
            self.max_step = 10000

        self.mode = mode
        self.trunct = False

        self.action_space = spaces.MultiDiscrete([5, 5]) 
        # self.observation_space = spaces.Box(low=0, high=13, shape=(self.n_grid*self.n_grid,), dtype=np.int32)  

        # 14 possible values (0‒13) for each grid cell
        self.observation_space = spaces.MultiDiscrete(np.full(self.n_grid * self.n_grid, 14, dtype=np.int32))


    # def get_observation(self):
    #     grid = np.zeros((self.n_grid, self.n_grid), dtype=np.int32)

    #     if self.FF == self.med:
    #         grid[tuple(self.FF)] = 3  
    #     else:
    #         grid[tuple(self.FF)] = 1  
    #         grid[tuple(self.med)] = 2

    #     for f in self.fire:
    #         grid[tuple(f)] = 4  

    #     for v in self.victims:
    #         if v in self.fire:
    #             grid[tuple(v)] = 8 # Victim in Fire
    #         else:
    #             grid[tuple(v)] = 5
    


    #     if self.FF in self.fire:
    #         grid[tuple(self.FF)] = 6 #FF and Fire
    #     if self.FF in self.victims:
    #         grid[tuple(self.FF)] =  7 # FF and victim
        

    #     if self.med in self.fire:
    #         grid[tuple(self.FF)] = 9 #med in Fire
    #     if self.med in self.victims:
    #         grid[tuple(self.FF)] =  10 # med and victim


    #     if self.FF == self.med and self.FF in self.victims:
    #         grid[tuple(self.FF)] = 11 
    #     if self.FF == self.med and self.FF in self.fire:
    #         grid[tuple(self.FF)] = 12
    #     if self.FF == self.med and self.FF in self.fire and self.FF in self.victims:
    #         grid[tuple(self.FF)] = 13
        

        
    #     return grid.flatten() 
    # 
    # 
    #  


    def get_observation(self):
        grid = np.zeros((self.n_grid, self.n_grid), dtype=np.int8)

        # Base positions
        if self.FF == self.med:
            grid[tuple(self.FF)] = 3
        else:
            grid[tuple(self.FF)] = 1
            grid[tuple(self.med)] = 2

        # Fires
        for f in self.fire:
            grid[tuple(f)] = 4

        # Victims
        for v in self.victims:
            grid[tuple(v)] = 8 if v in self.fire else 5

        # Pairwise overlaps
        if self.FF in self.fire:
            grid[tuple(self.FF)] = 6
        if self.FF in self.victims:
            grid[tuple(self.FF)] = 7
        if self.med in self.fire:
            grid[tuple(self.med)] = 9            # fixed
        if self.med in self.victims:
            grid[tuple(self.med)] = 10           # fixed

        # Triple overlaps (FF and med share a cell)
        if self.FF == self.med:
            if self.FF in self.fire:
                grid[tuple(self.FF)] = 12
            elif self.FF in self.victims:
                grid[tuple(self.FF)] = 11

        return grid.flatten()

    

    def step(self, action):

        # print('action',action)

        moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]  # (dy, dx) - Up, Down, Left, Right, Stay

        # print("FF", self.FF)
        # print("med", self.med)

        if self.mode == 'inference':
            act = action[0]
            new_FF = [self.FF[0] + moves[int(act[0])][0], self.FF[1] + moves[int(act[0])][1]]
            new_med = [self.med[0] + moves[int(act[1])][0], self.med[1] + moves[int(act[1])][1]]
        else:

            new_FF = [self.FF[0] + moves[int(action[0])][0], self.FF[1] + moves[int(action[0])][1]]
            new_med = [self.med[0] + moves[int(action[1])][0], self.med[1] + moves[int(action[1])][1]]


        # Move agents
        

        # print("new_FF", new_FF)
        # print("new_med", new_med)

        
        self.FF = np.clip(new_FF, 0, self.n_grid-1).tolist()
        self.med = np.clip(new_med, 0, self.n_grid-1).tolist()

    

        # print("new_FF1", self.FF)
        # print("new_med1", self.med)
        self.trajectory.append((self.FF, self.med, self.calculate_distance_med_FF()))

        reward = self.reward() 

        state = self.get_observation()


        vistm_copy = self.victims.copy()

        if self.med in self.victims:
            self.victim_saved += 1
            vistm_copy.remove(self.med)
        self.victims = vistm_copy.copy()

        fire_copy = self.fire.copy()
        if self.FF in self.fire:
            self.fire_ex += 1
            fire_copy.remove(self.FF)  # Extinguish fire
        self.fire = fire_copy.copy()
            

        terminated = len(self.fire) == 0 and len(self.victims) == 0
        sub_goals = [len(self.fire) == 0 , len(self.victims) == 0]



        
        info = {
        "fires_extinguished": self.fire_ex,
        "victims_saved": self.victim_saved,
        "sub_goals": sub_goals}

        
        if len(self.trajectory) > self.max_step:
            terminated = True
            self.trunct =True

        # print(self.get_observation())


        return state, reward, terminated, self.trunct, info
    
    def calculate_distance_med_FF(self):
        return abs(self.FF[0] - self.med[0]) + abs(self.FF[1] - self.med[1])

    def calculate_distance(self, start, target):
        return abs(start[0] - target[0]) + abs(start[1] - target[1])
    
    def reward(self):
        if self.method == "baseline":
            reward = 0
            if self.FF in self.fire:
                # reward += 10
                reward += 50
            if self.med in self.fire:
                reward += -100 
            if self.med in self.victims:
                # reward += 50
                reward += 10
            # if self.calculate_distance_med_FF() > 2:
            #     reward += -100
            # if self.calculate_distance_med_FF() <= 2:
            #     reward += 10
            return reward/10
        
        if self.method == 'hypRL':


            dist = list()
            # for tr in self.trajectory:
            #     dist.append(3 - tr[2])
            fire_list = list()
            victim_list = list()

            # dist_term = min(dist)

            if len(self.fire) > 0:
                fire_list = list()
                for fire in self.fire:
                    temp = list()
                    temp1 = list()
                    temp2 = list()
                    for index in range(1,len(self.trajectory)):

                        for tr in self.trajectory[:index]:
                            temp1.append(-1 * (1 - self.calculate_distance(fire, tr[0])))
                        for tr in self.trajectory[index:]:
                            temp2.append(1 - self.calculate_distance(fire, tr[1]))
                        temp2.append(min(temp1))
                        temp.append(min(temp2))
                    fire_list.append(max(temp))
                fire_term = min(fire_list)
            else:
                fire_term = math.inf

            # print('victime len',len(self.victims))
            # print('fire len',len(self.fire))

            if len(self.victims) > 0:

                for victim in self.victims:
                    Victim_temp = list()
                    for tr in self.trajectory:
                        Victim_temp.append(1 - self.calculate_distance(victim, tr[0]))
                    victim_list.append(max(Victim_temp))

                victim_term = min(victim_list)
            else:
                victim_term = math.inf
            
            # reward = min(dist_term, fire_term, victim_term)
            reward = min(fire_term, victim_term)

            # print("reward",reward)
            # print("dist",dist_term)
            # print("fire",fire_term)
            # print("vict",victim_term)

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
        self.trunct = False
        self.trajectory = list()
        self.trajectory.append((self.FF, self.med, self.calculate_distance_med_FF()))
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

        grid[tuple(self.FF)] = "FF"  # Firefighter
        grid[tuple(self.med)] = 'MD'  # Medic
   
   
        
        if self.med in temp_victim:
            grid[tuple(self.med)] = 'MDV'
        elif self.FF in temp_victim:
            grid[tuple(self.FF)] = 'FFV'
        elif self.FF in self.fire:
            grid[tuple(self.FF)] = 'FF🔥'
        elif self.med in self.fire:
            grid[tuple(self.med)] = 'MD🔥'
        elif self.FF == self.med:
            grid[tuple(self.FF)] = "FM"
            if self.FF in self.victims:
                grid[tuple(self.FF)] = 'FMV'
            elif self.FF in self.fire:
                grid[tuple(self.FF)] = 'FM🔥'
            elif self.FF in temp_victim and self.FF in self.fire:
                grid[tuple(self.FF)] = 'FMV🔥'
            
            
        formatted_grid = "\n".join(["  ".join(f"{cell:3}" for cell in row) for row in grid])
        print('###################\n\n\n###################')
        print(formatted_grid)

    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])



if __name__ == "__main__":

    env = WildFireEnv(method="hypRL", n_grid=5)
    env.reset()
    print(env.observation_space)
    env.render()

    done = False
    step = 0

    print(env.observation_space.sample())
    print(env.observation_space)
    for i in range(10):
        
        action = env.action_space.sample()
        
        obs, reward, done, trunct, info = env.step(action)
        step += 1 
        env.render()
        print("reward", reward)



