import gym
import gym_gmtphasing
import numpy as np

env = gym.make("gmtphasing-v0")
observation, info = env.reset(seed=42, return_info=True)

for i in range(1000):
    action = np.zeros(env.env.nactu,dtype=np.float32)
    observation, reward, done, info = env.step(action)
    if done:
        break
    env.render()
    #print(reward)

env.close()