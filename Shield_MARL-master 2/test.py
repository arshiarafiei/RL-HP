
from gym_grid.envs.grid_env import GridEnv

env = GridEnv(map_name='ISR', nagents=2, norender=False, padding=True)
pos = env.reset()
print('Initial position:', pos)
print('target:', env.targets)

print(env.calculate_distance(tuple(pos[0]), tuple(pos[1])))