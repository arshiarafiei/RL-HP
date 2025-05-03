import gym
from gym import spaces
import numpy as np

class DeepSeaTreasureEnv(gym.Env):
    

    metadata = {'render.modes': ['human']}

    def __init__(self, hypRL = False):
        # the map of the deep sea treasure (convex version)
        self.sea_map = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0.7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-10, 8.2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-10, -10, 11.5, 0, 0, 0, 0, 0, 0, 0, 0],
            [-10, -10, -10, 14.0, 15.1, 16.1, 0, 0, 0, 0, 0],
            [-10, -10, -10, -10, -10, -10, 0, 0, 0, 0, 0],
            [-10, -10, -10, -10, -10, -10, 0, 0, 0, 0, 0],
            [-10, -10, -10, -10, -10, -10, 19.6, 20.3, 0, 0, 0],
            [-10, -10, -10, -10, -10, -10, -10, -10, 0, 0, 0],
            [-10, -10, -10, -10, -10, -10, -10, -10, 22.4, 0, 0],
            [-10, -10, -10, -10, -10, -10, -10, -10, -10, 23.7, 0]
        ], dtype=np.float32)

        self.max_reward = 1.0
        self.hypRL = hypRL
        self.step_num = 0

        # state is a pair of discrete coords in [0..10]
        self.observation_space = spaces.MultiDiscrete([11, 11])

        # 4 actions: 0=up,1=down,2=left,3=right
        self.action_space = spaces.Discrete(4)

        # start in top-left
        self.current_state = np.array([0, 0], dtype=np.int32)
        self.terminal = False

        self.trajctory = list()

        self.treasure_cells = self.get_treasure_cells()

        self.treasure_achieve = list()



    def get_treasure_cells(self, not_treasure =  0):

        coords = np.argwhere(self.sea_map > not_treasure)
        return [(int(r), int(c)) for r, c in coords]
    
    def calculate_distance(self, state, traget):
        return abs(state[0] - traget[0]) + abs(state[1] - traget[1])


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_state[:] = 0
        self.trajctory.append(self.current_state.tolist())
        self.terminal = False
        self.step_num = 0
        self.treasure_achieve = list()
        return self.current_state, {}  # second return is "info"
    
    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]
    
    def reward_hypRL(self, step_traget = 0):


        # print("trajectory",self.trajctory)
        # print("treasures",self.get_treasure_cells())

        reach = list()

        t = 0
        for tr in self.treasure_cells:
            temp = list()
            for index, state in enumerate(self.trajctory):
                if index < t:
                    continue
                temp.append([float(self.sea_map[tuple(tr)]) * (1 -  self.calculate_distance(state, tr)), index])
            # print("temp",temp)
            reach.append(max(temp, key=lambda x: x[0]))
            t = max(temp, key=lambda x: x[0])[1] +1
            t = min(t, len(self.trajctory)-1)
        
        # print("reach",reach)
        # print(min(reach, key=lambda x: x[0])[0])

        reach_term = min(reach, key=lambda x: x[0])[0]

        # step_term = step_traget - self.step_num

        # print("step_term",step_term)

        # reward = min(reach_term, step_term)
        # print("reward",reward)

        return reach_term



    def step(self, action):
        # 4‐way motion
        dirs = {
            0: np.array([-1,  0]),  # up
            1: np.array([ 1,  0]),  # down
            2: np.array([ 0, -1]),  # left
            3: np.array([ 0,  1]),  # right
        }
        move = dirs[int(action)]
        candidate = self.current_state + move

        # original valid check:
        valid = lambda x, ind: (
            x[ind] >= 0 and x[ind] <= self.observation_space.nvec[ind] - 1
        )

        # only accept if both coords in [0..10] and map value ≠ -1
        if valid(candidate, 0) and valid(candidate, 1):
            if self.sea_map[tuple(candidate)] != -1:
                self.current_state = candidate


        # compute multi‐objective reward
        val = float(self.sea_map[tuple(self.current_state)])
        if val <= 0:
            treasure = 0.0
        else:
            self.treasure_achieve.append(val)
            # print("treasure_achieve",self.treasure_achieve)
            # print(val)
            # print(self.current_state)
            # print("step",self.step_num)
            treasure = val / self.max_reward
            self.terminal = True

        

        time_penalty = -1.0 / self.max_reward



        self.trajctory.append(self.current_state.tolist())

        reward_vector = np.array([treasure, time_penalty], dtype=np.float32)

        if self.hypRL:
            reward = self.reward_hypRL()
        else:
            reward = reward_vector[0] + reward_vector[1]

        self.step_num = self.step_num + 1

        if self.step_num >= 25:
            self.terminal = True


        return self.current_state, reward, self.terminal, False, {'treasure': sum(self.treasure_achieve), 'steps' :self.step_num}


    def render(self, mode='human'):
        print(f"Position: {tuple(self.current_state)}, Cell value: {self.sea_map[tuple(self.current_state)]}")

    def close(self):
        pass

    
if __name__ == "__main__":
    env = DeepSeaTreasureEnv(hypRL=True)
    env.reset()

    # print(env.get_treasure_cells())

    # print(env.action_space.sample())
    # print(env.observation_space.sample())
    #env.render()

    action = [1,3,1,3,1,3,1,3,3,3,1,1,3,3,1,1,3,1]

    done = False
    for i in range(16):
        state , reward, term, _, _ = env.step(action[i])
        print(reward)

    
