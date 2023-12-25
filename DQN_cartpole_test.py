import tensorflow as tf
import numpy as np
import gymnasium as gym

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
    def __init__(self, action_size, state_size):
        self.action_size = action_size
        self.state_size = state_size

        self.model = DQN(action_size, state_size)

    def policy(self, observation):
        prediction = self.model(np.reshape(observation, [1, 4]))
        return np.argmax(prediction)
    
env = gym.make(env_name, render_mode='human')

agent = DQNAgent(2, (4, ))
agent.model.load_weights('./save_model_cartpole/cartpole_model')

for e in range(10):
    observation, _ = env.reset()
    done = False
    score = 0
    while not done:
        action = agent.policy(observation)
        observation, reward, done, _, _ = env.step(action)
        score += reward

    print(score)
