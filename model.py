import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random




class MultiAgentDQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size * 2  
        self.action_size = action_size * 2  
        self.memory = []
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return [random.randrange(2), random.randrange(2)]  
        act_values = self.model.predict(state, verbose=0)
        return [np.argmax(act_values[0][:2]), np.argmax(act_values[0][2:])]  # act_values = [[ag0 act 0,  ag0 act1, ag1 act 0,  ag1 act1]]

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action[0]] = target
            target_f[0][action[1] + 2] = target  
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class MultiAgentRQN:
    def __init__(self, state_embedding_size, action_space):
        self.state_embedding_size = state_embedding_size  
        self.action_space = action_space  
        self.memory = []
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Embedding(input_dim=6, output_dim=self.state_embedding_size, input_length=None))  
        model.add(LSTM(32, return_sequences=False))  
        model.add(Dense(24, activation='relu'))
        model.add(Dense(len(self.action_space) * 2, activation='linear'))  # Output size for two agents
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, combined_state, actions, rewards, next_combined_state, done):
        self.memory.append((combined_state, actions, rewards, next_combined_state, done))

    def act(self, combined_state):
        if np.random.rand() <= self.epsilon:
            return [random.choice(self.action_space), random.choice(self.action_space)] 
        combined_state = np.array(combined_state).reshape(1, -1)  
        act_values = self.model.predict(combined_state, verbose=0)
        action1_index = np.argmax(act_values[0][:len(self.action_space)])  
        action2_index = np.argmax(act_values[0][len(self.action_space):])  
        return [self.action_space[action1_index], self.action_space[action2_index]]

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for combined_state, actions, rewards, next_combined_state, done in minibatch:
            combined_state = np.array(combined_state).reshape(1, -1)
            next_combined_state = np.array(next_combined_state).reshape(1, -1)
            targets = rewards
            if not done:
                targets += self.gamma * np.amax(self.model.predict(next_combined_state, verbose=0)[0])
            target_f = self.model.predict(combined_state, verbose=0)
            target_f[0][self.action_space.index(actions[0])] = targets
            target_f[0][len(self.action_space) + self.action_space.index(actions[1])] = targets
            self.model.fit(combined_state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class MultiAgentRQN_NEW:
    def __init__(self, state_embedding_size, action_space):
        self.state_embedding_size = state_embedding_size 
        self.action_space = action_space  
        self.memory = []
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Embedding(input_dim=6, output_dim=self.state_embedding_size, input_length=None))  
        model.add(LSTM(32, return_sequences=False))  
        model.add(Dense(24, activation='relu'))
        model.add(Dense(len(self.action_space) , activation='linear'))  
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(self.action_space)  

        # Handle empty state
        if len(state) == 0:
            state = [0]  

        
        state = pad_sequences([state], maxlen=10, padding='post')  
        state = np.array(state)  

        
        act_values = self.model.predict(state, verbose=0)
        action_index = np.argmax(act_values[0])  # Get the best action based on predicted Q-values
        if action_index >= len(self.action_space):
            raise ValueError(f"Invalid action index: {action_index}. Model output: {act_values[0]}")
        
        return self.action_space[action_index]






    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = pad_sequences([state], maxlen=10, padding='post')
            next_state = pad_sequences([next_state], maxlen=10, padding='post')
            
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            
            target_f = self.model.predict(state, verbose=0)
            target_f[0][self.action_space.index(action)] = target  # Update the specific action's Q-value
            
            self.model.fit(state, target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay