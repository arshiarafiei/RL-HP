import gym
from gym import spaces
import numpy as np

class DeepSeaTreasureEnv(gym.Env):
    

    metadata = {'render.modes': ['human']}

    def __init__(self):
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

        # state is a pair of discrete coords in [0..10]
        self.observation_space = spaces.MultiDiscrete([11, 11])

        # 4 actions: 0=up,1=down,2=left,3=right
        self.action_space = spaces.Discrete(4)

        # start in top-left
        self.current_state = np.array([0, 0], dtype=np.int32)
        self.terminal = False

        self.trajctory = list()

    import numpy as np

    def get_treasure_cells(self, not_treasure =  0):

        coords = np.argwhere(self.sea_map > not_treasure)
        return [(int(r), int(c)) for r, c in coords]
    
    def calculate_distance(self, state, traget):
        return abs(state[0] - traget[0]) + abs(state[1] - traget[1])


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_state[:] = 0
        self.trajctory.append(self.current_state)
        self.terminal = False
        return self.current_state, {}  # second return is "info"
    
    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]
    
    def reward_hypRL(self):
        
        pass

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
            treasure = val / self.max_reward
            self.terminal = True

        

        time_penalty = -1.0 / self.max_reward



        self.trajctory.append(self.current_state)

        reward_vector = np.array([treasure, time_penalty], dtype=np.float32)

        reward = reward_vector[0] +  reward_vector[1]


        return self.current_state, reward, self.terminal, False, {}


    def render(self, mode='human'):
        print(f"Position: {tuple(self.current_state)}, Cell value: {self.sea_map[tuple(self.current_state)]}")

    def close(self):
        pass

    
if __name__ == "__main__":
    env = DeepSeaTreasureEnv()
    env.reset()

    print(env.get_treasure_cells())

    # print(env.action_space.sample())
    # print(env.observation_space.sample())
    # env.render()

    # done = False
    # for i in range(10):
    #     state , reward, term, _ = env.step(1)
    #     print(state, reward, term)

    
