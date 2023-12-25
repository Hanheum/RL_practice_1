import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import gymnasium as gym
import random

env_name = 'CartPole-v1'

class Policy(tf.keras.Model):
    def __init__(self, action_size, state_size):
        super(Policy, self).__init__()

        self.action_size = action_size
        self.state_size = state_size

        self.fc1 = tf.keras.layers.Dense(30, activation='relu', input_shape=state_size)
        self.fc2 = tf.keras.layers.Dense(30, activation='relu')
        self.out = tf.keras.layers.Dense(action_size, activation='softmax')

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        out = self.out(x)
        return out
    
class Agent:
    def __init__(self, action_size, state_size):
        self.action_size = action_size
        self.state_size = state_size

        self.policy = Policy(action_size, state_size)
        self.discount_factor = 1

        self.optimizer = tf.keras.optimizers.Adam()

        self.memory = []

        self.random_episodes = 30

    def act(self, x, episode):
        if episode <= self.random_episodes:
            return random.randrange(self.action_size)
        else:
            prob = self.policy(x)
            return np.argmax(prob, axis=1)[0]
    
    def add_memory(self, state, action, reward):
        self.memory.append((state, action, reward))

    def reset_memory(self):
        self.memory = []

    def loss_fn(self, prob, action, reward):
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        log_prob = dist.log_prob(action)
        loss = -log_prob * reward
        return loss

    def train(self):
        sum_reward = 0
        discnt_rewards = []
        
        states = [sample[0] for sample in self.memory]
        actions = [sample[1] for sample in self.memory]
        rewards = [sample[2] for sample in self.memory]

        rewards.reverse()
        for r in rewards:
            sum_reward = r + sum_reward*self.discount_factor
            discnt_rewards.append(sum_reward)

        discnt_rewards.reverse()

        for state, action, reward in zip(states, actions, discnt_rewards):
            with tf.GradientTape() as tape:
                prob = self.policy(np.array([state]), training=True)
                loss = self.loss_fn(prob, action, reward)
            grad = tape.gradient(loss, self.policy.trainable_variables)
            self.optimizer.apply_gradients(zip(grad, self.policy.trainable_variables))

env = gym.make(env_name)
agent = Agent(2, (4, ))

episodes = 500

for episode in range(episodes):
    done = False
    agent.reset_memory()
    
    state, _ = env.reset()
    action = 0
    reward = 0
    summed_reward = 0

    while not done:
        action = agent.act(np.reshape(state, [1, 4]), episode)
        next_state, reward, done, _, _ = env.step(action)
        summed_reward += reward

        agent.add_memory(state, action, reward)
        state = next_state
        
        if done:
            agent.train()
            print('episode: {} | reward: {}'.format(episode, summed_reward))

    if (episode+1) % 100 == 0:
        agent.policy.save_weights('./saved_model_PG_cartpole/PG_cartpole')