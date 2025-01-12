from gym_grid.envs.grid_env import GridEnv
import numpy as np


if __name__ == "__main__":


    # env = GridEnv(map_name='SUNY', nagents=2, norender=False, padding=True)
    # # env.render()
    # # a = input('next:\n')
    # env.pos = np.array([[1, 1], [1, 1]])
    # obs, rew, x, y, w, oob = env.step([0, 0])
    # # env.render()
    # print("Obs: ", obs, "  rew: ", rew, x , y, w, oob)
    # obs, rew, x, y , w, oob = env.step([4, 4])
    # # env.render()
    # print("Obs: ", obs, "  rew: ", rew, x , y, w, oob)
    # env.reset(debug=True)
    # print(type(obs))
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


    env = GridEnv(map_name='SUNY', nagents=2, norender=True, padding=True)
    env.pos = np.array([[8, 22], [1, 1]])  # Example current positions

    # Calculate distance for agent 0 to a specific target
    agent_id = 0
    target_position = (9, 20)
    print(tuple(env.targets[0]))



    distance = env.calculate_closest_distance(agent_id, tuple(env.targets[0]))
    print(f"Agent {agent_id} closest distance to {tuple(env.targets[0])}: {distance}")



