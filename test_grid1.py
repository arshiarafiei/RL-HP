from gym_grid.envs.grid_env import GridEnv
import numpy as np
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import pandas as pd

# Hyperparameters
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
LR = 0.001
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 10000
NUM_EPISODES = 200
MAX_STEPS = 500

NUM_AGENTS = 2
ACTION_SPACE = 5  # Number of actions per agent
STATE_DIM = 2 * NUM_AGENTS  # Concatenated (x, y) for all agents

# Replay buffer
class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def remember(self, state, actions, rewards, next_state, done):
        self.buffer.append((state, actions, rewards, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)

# Build the Q-network
def build_model():
    model = Sequential([
        Input(shape=(STATE_DIM,)),
        Dense(512, activation='relu'),
        Dense(512, activation='relu'),
        Dense(NUM_AGENTS * ACTION_SPACE)  # Output Q-values for all actions of all agents
    ])
    model.compile(optimizer=Adam(learning_rate=LR), loss='mse')
    return model

# Epsilon-greedy policy
def select_actions(state, epsilon, model):
    if np.random.rand() < epsilon:
        return [np.random.randint(0, ACTION_SPACE) for _ in range(NUM_AGENTS)]
    state = np.expand_dims(state, axis=0)
    q_values = model.predict(state, verbose=0)
    actions = []
    for i in range(NUM_AGENTS):
        agent_q_values = q_values[0, i * ACTION_SPACE:(i + 1) * ACTION_SPACE]
        actions.append(np.argmax(agent_q_values))
    return actions

# Training function
def train_network(replay_buffer, main_model, target_model):
    if len(replay_buffer) < BATCH_SIZE:
        return  # Not enough samples to train

    # Sample a batch from the replay buffer
    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

    # Predict Q-values for current and next states
    q_values = main_model.predict(states, verbose=0)
    next_q_values = target_model.predict(next_states, verbose=0)

    # Update Q-values
    for i in range(BATCH_SIZE):
        for agent in range(NUM_AGENTS):
            action_idx = agent * ACTION_SPACE + actions[i][agent]
            if dones[i]:
                q_values[i][action_idx] = rewards[i]  # Scalar reward
            else:
                max_next_q = np.max(next_q_values[i, agent * ACTION_SPACE:(agent + 1) * ACTION_SPACE])
                q_values[i][action_idx] = rewards[i] + GAMMA * max_next_q  # Scalar reward

    # Train the main network
    main_model.fit(states, q_values, batch_size=BATCH_SIZE, verbose=0)






def reward_grid_env(env, trajectory1, trajectory2, step, episode):

    target1 = tuple(env.targets[0])
    target2 = tuple(env.targets[1])


    phi3_list = list()

    phi1_list = list()

    phi2_list = list()

    value = env.calculate_distance(target1, target2)



    for index in range(len(trajectory1)):

        phi3_list.append(-1 + env.calculate_distance(tuple(trajectory1[index]), tuple(trajectory2[index])))

        # phi3_list.append([-1 + env.calculate_distance(tuple(trajectory1[index]), tuple(trajectory2[index])), tuple(trajectory1[index]), tuple(trajectory2[index])])

        phi1_list.append(1 - env.calculate_distance(tuple(trajectory1[index]), target1))


        phi2_list.append(1 - env.calculate_distance(tuple(trajectory2[index]), target2))

    # print(phi3_list)
    # print(phi2_list)
    # print(phi1_list)

    reward = min(min(phi3_list), max(phi1_list), max(phi2_list))

    # f = open("log.txt", "a")
    # f.write(f"Episode {episode}, step: {step}\n")
    # f.write("Phi_1:")
    # f.writelines([f"{line}  " for line in phi1_list])
    # f.write("\nPhi_2:")
    # f.writelines([f"{line}  " for line in phi2_list])
    # f.write("\nPhi_3:")
    # f.writelines([f"{line}  " for line in phi3_list])
    # f.write("\n#######################################\n\n")


    # print("Reward", reward)


    return reward


    
    



        





# Main loop
def main(tr):
    # Initialize environment and models

    column = ["episode", "total_done", "total_col", "step"]

    df = pd.DataFrame(columns=column)


    env = GridEnv(map_name='Pentagon', nagents=NUM_AGENTS, norender=True, padding=True)
    main_model = build_model()
    target_model = build_model()
    target_model.set_weights(main_model.get_weights())

    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    epsilon = EPSILON

    total_done = 0

    for episode in range(NUM_EPISODES):


        states = env.reset(debug=True)  # Reset environment
        state_flat = np.concatenate(states)  # Flatten state [[x1, y1], [x2, y2]] -> [x1, y1, x2, y2]
        total_reward = 0

        trajectory1 = list()
        trajectory2 = list()
        reward_list = list()
        collision = False
        done = False
        
        total_collision = 0
        s= 0

        for step in range(MAX_STEPS):
            # Select actions for all agents
            actions = select_actions(state_flat, epsilon, main_model)

            

            next_pos, _, coll, goal_flags, _, _ = env.step(actions)


            trajectory1.append(next_pos[0].tolist())
            trajectory2.append(next_pos[1].tolist())

            reward = reward_grid_env(env, trajectory1, trajectory2, step, episode + 1)

            next_state_flat = np.concatenate(next_pos)  


            done = all(goal_flags)  

            s +=1


            # Store transition 
            replay_buffer.remember(state_flat, actions, reward, next_state_flat, done)

            # Update current state
            state_flat = next_state_flat
            total_reward += reward
            reward_list.append(reward)

            if done:
                total_done +=1
                break
            if coll:
                collision = True
                total_collision +=1
        

            # Train the network
            train_network(replay_buffer, main_model, target_model)

        # Decay epsilon
        epsilon = max(epsilon * EPSILON_DECAY, EPSILON_MIN)
        

        # Update target model periodically
        if episode % 10 == 0:
            target_model.set_weights(main_model.get_weights())
        f = open("/Users/tartmsu/Desktop/result_run.txt", "a")
        f.write(f"Episode {episode + 1}/{NUM_EPISODES}, Total Reward: {total_reward}, Done: {done}, Collision: {collision} , Epsilon: {epsilon:.2f}\n")
        f.writelines([f"{line}  " for line in reward_list])
        f.write("\n#######################################\n\n\n\n#######################################\n")



        print(f"Zero Episode {episode + 1}/{NUM_EPISODES}, Total Reward: {total_reward}, Done: {done}, Done: {total_done}, Collision: {total_collision} ,Epsilon: {epsilon:.2f}")
        arr = [episode, total_done, total_collision, s]
        df.loc[len(df)] = arr
        st = "data/pent/"+str(tr)+".csv"
        df.to_csv(st, index=False)

        # print(reward_list)

# Run the main loop
if __name__ == "__main__":
    for i in range(30,40):
        main(i)





# if __name__ == "__main__":


#     # env = GridEnv(map_name='SUNY', nagents=2, norender=False, padding=True)
#     # # env.render()
#     # # a = input('next:\n')
#     # env.pos = np.array([[1, 1], [1, 1]])
#     # obs, rew, x, y, w, oob = env.step([0, 0])
#     # # env.render()
#     # print("Obs: ", obs, "  rew: ", rew, x , y, w, oob)
#     # obs, rew, x, y , w, oob = env.step([4, 4])
#     # # env.render()
#     # print("Obs: ", obs, "  rew: ", rew, x , y, w, oob)
#     # env.reset(debug=True)
#     # print(type(obs))
#     # a = input('next:\n')
#     # obs, rew, _, _ = env.step([3, 1])
#     # # env.render()
#     # print("Obs: ", obs, "  rew: ", rew)
#     # # a = input('next:\n')
#     # obs, rew, _, _ = env.step([2, 1])
#     # # env.render()
#     # print("Obs: ", obs, "  rew: ", rew)
#     # a = input('next:\n')

#     # env.final_render()


    # env = GridEnv(map_name='SUNY', nagents=2, norender=True, padding=True)


#     print(env.action_space[0])

    # env.pos = np.array([[8, 22], [1, 1]])  # Example current positions

    # # Calculate distance for agent 0 to a specific target
    # agent_id = 0
    # target_position = tuple((8, 22))
    # print(tuple(env.targets[0]))

    



    # distance = env.calculate_closest_distance(target_position,  tuple(env.targets[0]))
    # print(f"Agent {agent_id} closest distance to {tuple(env.targets[0])}: {distance}")



