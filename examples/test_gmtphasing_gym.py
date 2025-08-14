import gymnasium as gym
import segment_phasing_fp_env  # noqa: F401
import numpy as np

env = gym.make("SegmentPhasingFP-v0")
observation, info = env.reset(seed=42)

if env.action_space.shape is None:
    raise ValueError("action space is null")

for i in range(1000):
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    observation, reward, terminated, trunacated, info = env.step(action)
    if terminated:
        break
    print(info["se_strehl"])
