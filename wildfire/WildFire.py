import gymnasium as gym
from gymnasium import spaces 
import numpy as np
import math

class WildFireEnv(gym.Env):
    def __init__(self, n_grid = 3, method = "baseline", mode = 'train', FF_coords = [2, 0], med_coords = [2, 0]):
        super(WildFireEnv, self).__init__()

        self.n_grid = n_grid
        self.method = method

        self.grid_size = (self.n_grid, self.n_grid) 
        self.FF = FF_coords
        self.med = med_coords
        self.number_of_FF = len(self.FF)
        self.number_of_med = len(self.med)
        self.agents = [self.FF, self.med]
        self.fire = [[0, 1], [1, 2], [2, 1]]
        self.victims = [[0, 0], [1, 2]]
        self.victim_saved = 0
        self.fire_ex = 0
        self.trajectory = list()
        if mode == 'train':
            self.max_step = 1000
        else:
            self.max_step = 30000

        self.mode = mode
        self.trunct = False

        self.action_space = spaces.MultiDiscrete([5, 5]) 
        # self.observation_space = spaces.Box(low=0, high=13, shape=(self.n_grid*self.n_grid,), dtype=np.int32)  

        # 14 possible values (0‒13) for each grid cell
        self.observation_space = spaces.Dict({"FF" : spaces.MultiDiscrete(np.full(self.n_grid * self.n_grid, 14, dtype=np.int32)),
                                               "MD" : spaces.MultiDiscrete(np.full(self.n_grid * self.n_grid, 14, dtype=np.int32))})



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

    def update_beliefs(self):
        return 0

    def crop_observation(self, agent_pos, obs):
        px, py = agent_pos

        if (px == 0):
            obs = np.delete(obs, 0, axis = 0)
        
        if (px == self.n_grid - 1):
            obs = np.delete(obs, 2, axis = 0)
        
        if (py == 0):
            obs = np.delete(obs, 0, axis = 1)

        if (py == self.n_grid - 1):
            obs = np.delete(obs, 2, axis = 1)

        return obs

    def get_observation(self):
        FFgrid = np.zeros((3, 3), dtype=np.int8)
        MDgrid = np.zeros((3, 3), dtype=np.int8)


        full_grid = np.zeros((self.n_grid, self.n_grid), dtype = np.int8)

        # how local coords work
        # the first coordinate is the grid spot being observed in terms of the second coord.
        # EX: x1 - x2 : x2 is the observer and x1 is what is being obsreved

       
        # Base positions
        if self.FF == self.med:
            full_grid[tuple(self.FF)] = 3
            MDgrid[(1, 1)] = 3
            FFgrid[(1, 1)] = 3

        else:
            full_grid[tuple(self.FF)] = 1
            full_grid[tuple(self.med)] = 2

            if (self._chebyshev_distance(self.FF, self.med) <= 1):
                fx, fy = self.FF
                mx, my = self.med

                local_x_med = (fx - mx) + 1
                local_y_med = (fy - my) + 1

                local_x_FF = (mx - fx) + 1
                local_y_FF = (my - fy) + 1

                FFgrid[(local_x_FF, local_y_FF)] = 2
                MDgrid[(local_x_med, local_y_med)] = 1

            MDgrid[(1, 1)] = 2
            FFgrid[(1, 1)] = 1

        # Fires
        for f in self.fire:
            full_grid[tuple(f)] = 4

            if (self._chebyshev_distance(f, self.FF) <= 1):
                fx, fy = f
                FFx, FFy = self.FF

                local_x = (fx - FFx) + 1
                local_y = (fy - FFy) + 1

                FFgrid[(local_x, local_y)] = 4

            if (self._chebyshev_distance(f, self.med) <= 1):
                fx, fy = f
                medx, medy = self.med

                local_x = (fx - medx) + 1
                local_y = (fy - medy) + 1

                MDgrid[(local_x, local_y)] = 4

        # Victims
        for v in self.victims:
            full_grid[tuple(v)] = 8 if v in self.fire else 5

            if (self._chebyshev_distance(v, self.FF) <= 1):
                vx, vy = f
                FFx, FFy = self.FF

                local_x = (vx - FFx) + 1
                local_y = (vy - FFy) + 1

                FFgrid[(local_x, local_y)] = 8 if v in self.fire else 5

            if (self._chebyshev_distance(v, self.med) <= 1):
                vx, vy = f
                medx, medy = self.med

                local_x = (vx - medx) + 1
                local_y = (vy - medy) + 1

                MDgrid[(local_x, local_y)] = 8 if v in self.fire else 5

            #grid[tuple(v)] = 8 if v in self.fire else 5

        # Pairwise overlaps
        if self.FF in self.fire:
            full_grid[tuple(self.FF)] = 6
            FFgrid[(1, 1)] = 6

        if self.FF in self.victims:
            full_grid[tuple(self.FF)] = 7
            FFgrid[(1, 1)] = 7

        if self.med in self.fire:
            full_grid[tuple(self.med)] = 9            # fixed
            MDgrid[(1, 1)] = 9

        if self.med in self.victims:
            full_grid[tuple(self.med)] = 10           # fixed
            MDgrid[(1, 1)] = 10

        # Triple overlaps (FF and med share a cell)
        if self.FF == self.med:
            if self.FF in self.fire:
                full_grid[tuple(self.FF)] = 12
                FFgrid[(1, 1)] = 12
                MDgrid[(1, 1)] = 12

            elif self.FF in self.victims:
                full_grid[tuple(self.FF)] = 11
                FFgrid[(1, 1)] = 11
                MDgrid[(1, 1)] = 11
        
        FFgrid = self.crop_observation(self.FF, FFgrid)
        MDgrid = self.crop_observation(self.med, MDgrid)



        return [FFgrid, MDgrid], full_grid

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

        local_states, states = self.get_observation()


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


        return local_states, states, reward, terminated, self.trunct, info
    
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
        self.fire = [[0, self.n_grid -1], [3, self.n_grid -1], [4, self.n_grid -1]]  
        self.victims = [[0, 0], [0, self.n_grid -1]]
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

        grid[tuple(self.FF)] = "FF"
        #grid[tuple(self.FF)] = "FF"  # Firefighter
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
    
    def _chebyshev_distance(self, p1, p2):
        return max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))

    def render_obs(self, obs):
        grid = np.full(obs.shape, ' . ', dtype=object)
        temp_victim = self.victims.copy()

        for i in range (obs.shape[0]):
            for j in range (obs.shape[1]):
                if obs[i][j] == 4:
                    grid[tuple([i, j])] = '🔥'

                if obs[i][j] == 5:
                    grid[tuple([i, j])] = 'V'

                if obs[i][j] == 1:
                    grid[tuple([i, j])] = 'FF'

                if obs[i][j] == 2:
                    grid[tuple([i, j])] = 'MD'

                if obs[i][j] == 10:
                    grid[tuple([i, j])] = 'MDV'

                if obs[i][j] == 7:
                    grid[tuple([i, j])] = 'FFV'

                if obs[i][j] == 6:
                    grid[tuple([i, j])] = 'FF🔥'

                if obs[i][j] == 8:
                    grid[tuple([i, j])] = 'V🔥'

                if obs[i][j] == 9:
                    grid[tuple([i, j])] = 'MD🔥'

                if obs[i][j] == 3:
                    grid[tuple([i, j])] = 'FM'
                
                if obs[i][j] == 11:
                    grid[tuple([i, j])] = 'FMV'
                
                if obs[i][j] == 12:
                    grid[tuple([i, j])] = 'FM🔥'
            
        formatted_grid = "\n".join(["  ".join(f"{cell:3}" for cell in row) for row in grid])
        print('###################\n\n\n###################')
        print(formatted_grid)

if __name__ == "__main__":

    number_of_FF = 4
    number_of_MD = 4
    FF_coords = [[2, 0], [2, 1], [3, 1], [3, 2]]
    MD_coords = [[0, 0], [1, 0], [0, 1], [1, 1]]

    env = WildFireEnv(method="hypRL", n_grid=5)
    env.reset()
    print("observation space ", env.observation_space)
    env.render()


    done = False
    step = 0

    print(env.observation_space.sample())
    print(env.observation_space)
    for i in range(10):        
        action = env.action_space.sample()
        
        obs, state, reward, done, trunct, info = env.step(action)
        env.render()

        print("firefighter observation")
        env.render_obs(obs[0])

        print("medic observation")
        env.render_obs(obs[1])

        step += 1 
        print("reward", reward)



