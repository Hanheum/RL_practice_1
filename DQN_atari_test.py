import time
import gymnasium as gym
import random
import numpy as np
import tensorflow as tf

from skimage.color import rgb2gray
from skimage.transform import resize

class DQN(tf.keras.Model):
    def __init__(self, action_size, state_size):
        super(DQN, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=state_size)
        self.conv2 = tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(512, activation='relu')
        self.fc_out = tf.keras.layers.Dense(action_size)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc(x)
        q = self.fc_out(x)
        return q
    
class DQNAgent:
    def __init__(self, action_size, state_size, model_path):
        self.render = False

        self.action_size = action_size
        self.state_size = state_size

        self.epsilon = 0.02
        
        self.model = DQN(action_size, state_size)
        self.model.load_weights(model_path)

    def get_action(self, history):
        history = np.float32(history/255.)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model(history)
            return np.argmax(q_value[0])
        
def pre_processing(observe):
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe

if __name__ == '__main__':
    env = gym.make('BreakoutDeterministic-v4', render_mode='human')
    render = True

    state_size = (84, 84, 4)
    action_size = 3
    model_path = './save_model/model'
    agent = DQNAgent(action_size, state_size, model_path)

    action_dict = {0:1, 1:2, 2:3, 3:3}

    num_episodes = 3
    for e in range(num_episodes):
        done = False
        dead = False

        score, start_life = 0, 5
        observe, info = env.reset()

        state = pre_processing(observe)
        history = np.stack([state, state, state, state], axis=2)
        history = np.reshape(history, (1, 84, 84, 4))

        while not done:
            if render:
                env.render()
                time.sleep(0.05)

            action = agent.get_action(history)
            real_action = action_dict[action]

            if dead:
                action, real_action, dead = 0, 1, False

            observe, reward, done, _, info = env.step(real_action)
            next_state = pre_processing(observe)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)

            if start_life > info['lives']:
                dead, start_life = True, info['lives']

            score += reward

            if dead:
                history = np.stack((next_state, next_state, next_state, next_state), axis=2)
                history = np.reshape(history, (1, 84, 84, 4))
            else:
                history = next_history

            if done:
                print('episode: {} | score: {}'.format(e, score))