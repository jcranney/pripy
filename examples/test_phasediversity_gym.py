import gymnasium as gym
import gym_phasediversity
import numpy as np

env = gym.make("phasediversity-v0")
observation, info = env.reset(seed=42, return_info=True)

for i in range(100):
    action = np.random.randn(env.env.nactu).astype(dtype=np.float32)/10
    observation, reward, done, info = env.step(action)
    if done:
        break
    env.render()
    #print(reward)

env.close()