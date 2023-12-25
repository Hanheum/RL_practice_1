import tensorflow as tf
import numpy as np
import gymnasium as gym
import random
from collections import deque
from PIL import Image

env_name = 'ALE/Breakout-v5'

class DQN(tf.keras.Model):
    def __init__(self, action_size, state_size):
        super(DQN, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(128, (5, 5), activation='relu', input_shape=state_size)
        self.conv2 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(32, (5, 5), activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(64, activation='relu')
        self.out = tf.keras.layers.Dense(action_size)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        out = self.out(x)
        return out
    
class Agent:
    def __init__(self, action_size, state_size=(84, 84, 4)):
        self.action_size = action_size
        self.state_size = state_size

        self.model = DQN(action_size, state_size)
        self.target_model = DQN(action_size, state_size)
        self.optimizer = tf.keras.optimizers.Adam()

        self.memory = deque(maxlen=100000)
        
        self.epsilon = 0.02
        self.epsilon_end = 0.02
        self.epsilon_steps = 1000000
        self.epsilon_step = self.epsilon-self.epsilon_end
        self.epsilon_step /= self.epsilon_steps

        self.update_target_rate = 10000
        self.start_train = 50000

        self.batch_size = 32
        self.discount_rate = 0.99

        self.update_target()

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())

    def add_memory(self, history, action, reward, next_history, done):
        self.memory.append((history, action, reward, next_history, done))

    def policy(self, history):
        if self.epsilon >= np.random.rand():
            return random.randrange(self.action_size)
        else:
            q_values = self.model(np.reshape(history, [1, 84, 84, 4]))
            return np.argmax(q_values[0])
        
    def train(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_step

        batch = random.sample(self.memory, self.batch_size)

        histories, actions, rewards, next_histories, dones = [], [], [], [], []
        for sample in batch:
            histories.append(sample[0])
            actions.append(sample[1])
            rewards.append(sample[2])
            next_histories.append(sample[3])
            dones.append(sample[4])

        histories = np.array(histories, dtype=np.float32)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_histories = np.array(next_histories, dtype=np.float32)
        dones = np.array(dones)

        with tf.GradientTape() as tape:
            q_values = self.model(histories)
            target_q = self.target_model(next_histories)

            q_values = q_values * tf.one_hot(actions, self.action_size)
            q_values = tf.reduce_sum(q_values, axis=1)

            loss = (rewards + self.discount_rate * np.amax(target_q, axis=1) - q_values)**2

        gradient = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))

def preprocessing(img):
    img = Image.fromarray(img)
    img = img.resize((84, 84))
    img = np.array(img, dtype=np.float32)/255.
    return img

env = gym.make(env_name, obs_type='grayscale', render_mode='human')
agent = Agent(4)
agent.model.load_weights('./breakout/breakout')

episodes = 3
global_steps = 0
for episode in range(episodes):
    done = False

    observation, info = env.reset()
    observation = preprocessing(observation)
    history = np.stack([observation, observation, observation, observation], axis=2)

    starting_lives = info['lives']

    summed_reward = 0
    while not done:
        global_steps += 1
        action = agent.policy(history)
        observation, reward, done, _, info = env.step(action)
        observation = preprocessing(observation)
        summed_reward += reward

        next_history = np.append(np.reshape(observation, [84, 84, 1]), history[:, :, :3], axis=2)

        agent.add_memory(history, action, reward, next_history, done)

        lives = info['lives']
        if lives < starting_lives:
            history = np.stack([observation, observation, observation, observation], axis=2)
        else:
            history = next_history

        if len(agent.memory) >= agent.start_train:
            agent.train()

        if global_steps % agent.update_target_rate == 0:
            agent.update_target()