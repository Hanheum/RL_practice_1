import tensorflow as tf
import numpy as np
import gymnasium as gym
from collections import deque
import random

env_name = 'CartPole-v1'

class DQN(tf.keras.Model):
    def __init__(self, action_size, state_size):
        super(DQN, self).__init__()
        self.action_size = action_size
        self.state_size = state_size

        self.fc1 = tf.keras.layers.Dense(128, activation='relu', input_shape=state_size)
        self.fc2 = tf.keras.layers.Dense(64, activation='relu')
        self.fc3 = tf.keras.layers.Dense(32, activation='relu')
        self.q_layer = tf.keras.layers.Dense(action_size)

    def __call__(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        q = self.q_layer(x)
        return q
    
class DQNAgent:
    def __init__(self, action_size, state_size=(4, )):
        self.action_size = action_size
        self.state_size = state_size

        self.model = DQN(action_size, state_size)
        self.target_model = DQN(action_size, state_size)

        self.epsilon = 0.6
        self.epsilon_start = 1.
        self.epsilon_end = 0.02
        self.epsilon_steps = 1000000
        self.epsilon_step = self.epsilon_start-self.epsilon_end
        self.epsilon_step /= self.epsilon_steps

        self.train_start = 50000
        
        self.memory = deque(maxlen=100000)
        
        self.learning_rate = 1e-4
        self.discount_rate = 0.9
        self.batch_size = 32
        self.update_target_rate = 10000

        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, clipnorm=10.)

        self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def policy(self, observation):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        else:
            prediction = self.model(np.reshape(observation, [1, 4]))
            return np.argmax(prediction[0])
        
    def add_sample(self, observation, action, reward, next_observation, done):
        self.memory.append((observation, action, reward, next_observation, done))

    def train_model(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_step
        
        batch = random.sample(self.memory, self.batch_size)
        
        observations = np.array([sample[0] for sample in batch], np.float32)
        actions = np.array([sample[1] for sample in batch])
        rewards = np.array([sample[2] for sample in batch])
        next_observations = np.array([sample[3] for sample in batch], np.float32)
        dones = np.array([sample[4] for sample in batch])

        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            predictions = self.model(observations)
            one_hot_action = tf.one_hot(actions, self.action_size)
            predictions = tf.reduce_sum(predictions*one_hot_action, axis=1)

            target_predictions = self.target_model(next_observations)

            max_q = np.amax(target_predictions, axis=1)
            targets = rewards + (1-dones) * self.discount_rate * max_q

            loss = tf.reduce_mean((targets-predictions)**2)

        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))

env = gym.make(env_name)
#cartpole: observation, reward, done, truncated, info

agent = DQNAgent(action_size=2, state_size=(4, ))

agent.model.load_weights('./save_model_cartpole/cartpole_model')
agent.update_target_model()

num_episodes = 50000
global_steps = 0

for e in range(num_episodes):
    done = False
    score = 0

    observation, _ = env.reset()
    
    while not done:
        global_steps += 1

        action = agent.policy(observation)
        next_observation, reward, done, _, _ = env.step(action)

        score += reward

        agent.add_sample(observation, action, reward, next_observation, done)

        if len(agent.memory) >= agent.train_start:
            agent.train_model()

            if global_steps % agent.update_target_rate:
                agent.update_target_model()

        if done:
            log = 'episode: {} | '.format(e)
            log += 'score: {} | '.format(score)
            log += 'memory length: {} | '.format(len(agent.memory))
            log += 'epsilon: {}'.format(agent.epsilon)
            print(log)
        else:
            observation = next_observation

    if (e+1) % 1000 == 0:
        agent.model.save_weights("./save_model_cartpole/cartpole_model", save_format='tf')