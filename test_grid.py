from gym_grid.envs.grid_env import GridEnv
import numpy as np


if __name__ == "__main__":


    env = GridEnv(map_name='example', nagents=2, norender=False, padding=True)
    # env.render()
    # a = input('next:\n')
    env.pos = np.array([[0, 1], [1, 0]])
    obs, rew, x, y = env.step([1, 3])
    # env.render()
    print("Obs: ", obs, "  rew: ", rew, x , y)
    print(type(obs))
    # a = input('next:\n')
    # obs, rew, _, _ = env.step([3, 1])
    # # env.render()
    # print("Obs: ", obs, "  rew: ", rew)
    # # a = input('next:\n')
    # obs, rew, _, _ = env.step([2, 1])
    # # env.render()
    # print("Obs: ", obs, "  rew: ", rew)
    # a = input('next:\n')

    # env.final_render()
