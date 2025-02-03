import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import random
import pandas as pd
import time



class Job:
    def __init__(self, n_agent=4, grid_size=5, starting_positions=None, resource_position=None):
        self.n_agent = n_agent
        self.grid_size = grid_size
        self.env = np.zeros((grid_size, grid_size))
        self.resource = resource_position if resource_position else (grid_size - 1, grid_size - 1)  
        self.starting_positions = starting_positions if starting_positions else [(0, 0), (0, 2), (0, 4), (3, 3)]  
        self.state = [(pos[0], pos[1], 0) for pos in self.starting_positions[:n_agent]]  # (x, y, resource_alloc)
        self.action_space = [0, 1, 2, 3]

        self.update_env()

    def reset(self, random=False):
        if random:
            self.env = np.zeros((self.grid_size, self.grid_size))
            self.state = []
            for _ in range(self.n_agent):
                while True:
                    random_position = (
                        np.random.randint(0, self.grid_size),
                        np.random.randint(0, self.grid_size)
                    )
                    if random_position not in [(s[0], s[1]) for s in self.state] and random_position != self.resource:
                        self.state.append((random_position[0], random_position[1], 0))  # (x, y, resource_alloc)
                        break
        else:
            self.env = np.zeros((self.grid_size, self.grid_size))
            self.state = [(pos[0], pos[1], 0) for pos in self.starting_positions[:self.n_agent]]  

        self.update_env()
        return self.env, self.state, self.resource, self.action_space

    def update_env(self):
        self.env.fill(0)
        self.env[self.resource[0], self.resource[1]] = 9  # Mark the resource with 9
        for i, (x, y, _) in enumerate(self.state):
            self.env[x, y] = i + 1  # Mark agents with unique IDs

    def step(self, action):
        next_positions = []

        for i, (x, y, alloc) in enumerate(self.state):
            if action[i] == 0:  # Move up
                x = max(0, x - 1)
            elif action[i] == 1:  # Move down
                x = min(self.grid_size - 1, x + 1)
            elif action[i] == 2:  # Move left
                y = max(0, y - 1)
            elif action[i] == 3:  # Move right
                y = min(self.grid_size - 1, y + 1)
            
            next_positions.append((x, y, alloc))

        # Resolve conflicts
        unique_positions = []
        for i, (x, y, alloc) in enumerate(next_positions):
            if (x, y) not in [(p[0], p[1]) for p in unique_positions]:
                unique_positions.append((x, y, alloc))
            else:
                unique_positions.append(self.state[len(unique_positions)])  # Stay in place if conflict

        # Check if any agent reached the resource
        updated_state = []
        for x, y, alloc in unique_positions:
            if (x, y) == self.resource:
                alloc += 1  # Increment agent's resource count
            updated_state.append((x, y, alloc))

        self.state = updated_state
        self.update_env()

        return self.env, self.state, self.resource

    def calculate_distance(self, state):
        return abs(state[0] - self.resource[0]) + abs(state[1] - self.resource[1])


class PPOAgent:
    def __init__(self, state_size, action_size, n_agent, learning_rate=0.001, gamma=0.95, epsilon_clip=0.2, entropy_beta=0.1):
        self.state_size = state_size  # Now includes (x, y, resource_alloc) for each agent
        self.action_size = action_size  # 5 possible actions (Up, Down, Left, Right, Stay)
        self.n_agent = n_agent
        self.gamma = gamma
        self.epsilon_clip = epsilon_clip
        self.entropy_beta = entropy_beta

        # Policy and Value networks
        self.policy_model = self._build_policy_model()
        self.value_model = self._build_value_model()

        # Optimizers
        self.policy_optimizer = Adam(learning_rate=learning_rate)
        self.value_optimizer = Adam(learning_rate=learning_rate)

    def _build_policy_model(self):
        model = Sequential()
        model.add(Dense(256, input_dim=self.state_size, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.n_agent * self.action_size, activation='softmax'))  # Output shape = n_agent * action_size
        return model

    def _build_value_model(self):
        model = Sequential()
        model.add(Dense(512, input_dim=self.state_size, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(1, activation='linear'))
        return model

    def act(self, state):
        policy = self.policy_model.predict(np.expand_dims(state, axis=0), verbose=0)
        policy = policy.reshape(self.n_agent, self.action_size)

        # Normalize probabilities to handle numerical instability
        for i in range(self.n_agent):
            policy_sum = np.sum(policy[i])
            if policy_sum == 0 or np.isnan(policy_sum):  
                policy[i] = np.ones(self.action_size) / self.action_size  
            else:
                policy[i] /= policy_sum

        actions = [np.random.choice(self.action_size, p=policy[i]) for i in range(self.n_agent)]
        return actions, policy

    def train(self, states, actions, rewards, old_policies, advantages):
        with tf.GradientTape() as tape:
            policies = self.policy_model(states, training=True)
            policies = tf.reshape(policies, (-1, self.n_agent, self.action_size))
            old_policies = tf.reshape(old_policies, (-1, self.n_agent, self.action_size))

            action_masks = tf.one_hot(actions, self.action_size)
            selected_policies = tf.reduce_sum(action_masks * policies, axis=-1)  
            old_selected_policies = tf.reduce_sum(action_masks * old_policies, axis=-1)  

            ratios = tf.exp(tf.math.log(selected_policies + 1e-10) - tf.math.log(old_selected_policies + 1e-10))
            clipped_ratios = tf.clip_by_value(ratios, 1 - self.epsilon_clip, 1 + self.epsilon_clip)

            surrogate = tf.minimum(ratios * advantages, clipped_ratios * advantages)
            entropy = -tf.reduce_sum(policies * tf.math.log(policies + 1e-10), axis=-1)
            loss = -tf.reduce_mean(surrogate + self.entropy_beta * entropy)

        grads = tape.gradient(loss, self.policy_model.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(grads, self.policy_model.trainable_variables))

        # Update Value
        with tf.GradientTape() as tape:
            values = self.value_model(states, training=True)
            value_loss = tf.reduce_mean(tf.square(rewards - values))

        grads = tape.gradient(value_loss, self.value_model.trainable_variables)
        self.value_optimizer.apply_gradients(zip(grads, self.value_model.trainable_variables))



def reward_fairness(env,tarjectories,step_size):

    agent_dist = [[] for _ in range(env.n_agent)]

    for ag in range(env.n_agent):
        temp = list()
        for tr in tarjectories[ag]:
            temp.append((int(env.grid_size/2))*10 - (10* env.calculate_distance(tr)))
        agent_dist[ag] = temp.copy()
    
    temp_ag = [[] for _ in range(env.n_agent)]

    for ag in range(env.n_agent):
        for index in range(len(agent_dist[ag])):
            num = max(agent_dist[ag][len(agent_dist[ag])-index-1:])
            temp = temp_ag[ag].copy()
            temp.append(num)
            temp_ag[ag]= temp.copy()
    

    phi_list = list()

    for ag in range(env.n_agent):
        phi_list.append(min(temp_ag[ag]))

    
    
    partone = min(phi_list)

    phi_dist = list()

    # print(tarjectories)

    for index in range(len(tarjectories[0])):

        phi_dist.append(int((step_size * 0.01)) - abs(tarjectories[0][index][2]-tarjectories[1][index][2]))

    parttwo = min(phi_dist)

    reward = min(partone, parttwo)
          

    return reward



def output(e, episodes, epsilon, allocation_total, done):
    f = open("Fair_ppo_new.txt", "a")
    print(f"Episode {e + 1}/{episodes} - Done: {done} - Epsilon: {epsilon:.2f}")
    f.write(f"Episode {e + 1}/{episodes} - Done: {done} - Epsilon: {epsilon:.2f} \n")
    for i, item in enumerate(allocation_total):
        print(f"Agent:{i} -- Total Allocation:{item}")
        f.write(f"Agent:{i} -- Total Allocation:{item} \n")
    print("-"*40)
    print("-"*40)
    f.write("-"*40)
    f.write("\n")
    f.write("-"*40)
    f.write("\n")
    f.write("\n")
    f.close()


def log(e,step,state, next_state, action ,reward):
    f = open("log_fair.txt", "a")
    f.write(f"Episode {e}, step: {step}, reward: {reward}\n")
    f.write("State:")
    f.writelines([f"{line}  " for line in state])
    f.write("\nAction:")
    f.writelines([f"{line}  " for line in action])
    f.write("\nNext State:")
    f.writelines([f"{line}  " for line in next_state])
    f.write("\n#######################################\n\n")










def train_ppo(env, agent, tries, episodes=1000, step_size=1000, batch_size=64, update_epochs=50):
    df = pd.DataFrame(columns=[f"Agent {i}" for i in range(env.n_agent)])

    for e in range(episodes):
        states, actions, rewards, old_policies = [], [], [], []
        _, state, resource, action_space = env.reset()
        print("Starting From State: ", state)
        state_flat = np.array([coord for agent in state for coord in agent])

        trajectories = [[] for _ in range(env.n_agent)]
        for st in range(env.n_agent):
            trajectories[st].append(state[st])

        allocation_total = [0] * env.n_agent
        done = False

        for step in range(step_size):
            st = env.state
            action, policy = agent.act(state_flat)
            next_grid, next_state, resource = env.step(action)



            for i in range(env.n_agent):
                
                if (next_state[i][0], next_state[i][1]) == env.resource:
                    allocation_total[i] += 1
                

            for ag in range(env.n_agent):
                trajectories[ag].append(next_state[ag])

            reward = reward_fairness(env, trajectories,step_size)
            # reward = +1

            next_state_flat = np.array([coord for agent in next_state for coord in agent])

            states.append(state_flat)
            actions.append(action)
            rewards.append(reward)
            old_policies.append(policy)

            state_flat = next_state_flat

            fairness_metric = [abs(max(allocation_total)-i)/(i+1) if i != 0 else 1 for i in allocation_total]

            if max(fairness_metric) < 0.1 and sum(allocation_total) > (step_size * 0.5):
                done = True

        rewards = np.array(rewards, dtype=np.float32)

        states = np.array(states).reshape(-1, agent.state_size)  
        actions = np.array(actions).reshape(-1, agent.n_agent)
        old_policies = np.array(old_policies).reshape(-1, agent.n_agent, agent.action_size)

        values = agent.value_model.predict(states)
        next_values = np.roll(values, -1, axis=0)
        next_values[-1] = 0  
        discounted_rewards = rewards + agent.gamma * next_values.flatten()
        advantages = discounted_rewards - values.flatten()
        advantages = tf.expand_dims(advantages, axis=-1)  

        for _ in range(update_epochs):
            agent.train(states, actions, rewards, old_policies, advantages)

        output(e, episodes, agent.epsilon_clip, allocation_total, done)

        df.loc[len(df)] = allocation_total
        st = "data/grid"+str(env.grid_size)+"new/fair_"+str(env.grid_size)+"_"+str(tries)+".csv"
        
        df.to_csv(st, index=False)





if __name__ == "__main__":
    if __name__ == "__main__":
        n_ag = [2]
        gr = [5]
        for n in n_ag:
            for g in gr:

                n_agent = n
                grid_size = g
                starting_positions = [(0, 0), (4, 4)]
                if g%2 ==0:
                    resource_position = (int(g/2)-1, int(g/2)-1)  

                else:
                    resource_position = (int(g/2), int(g/2))  

                action_size = 5
                state_size = n_agent * 3

                env = Job(n_agent=n_agent, grid_size=grid_size, starting_positions=starting_positions, resource_position=resource_position)
                agent = PPOAgent(state_size, action_size, n_agent)

                for i in range(5,10):

                    # f = open("log_fair.txt", "w")
                    # f = open("Fair_ppo.txt", "w")

                    train_ppo(env, agent, i,episodes=1000, step_size=1000, batch_size=64, update_epochs=50)
