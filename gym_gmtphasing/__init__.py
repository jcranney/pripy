from gymnasium.envs.registration import register

register(
    id='gmtphasing-v0',
    entry_point='gym_gmtphasing.envs:GMTPhasingEnv',
)