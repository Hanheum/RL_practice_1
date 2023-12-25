import gymnasium as gym

env_name = 'ALE/Pong-v5'
env = gym.make(env_name, render_mode='human')

obs, info = env.reset()

for i in range(1000):
    action = env.action_space.sample()
    obs, reward, done, _, info = env.step(action)
    print(reward, info, done)