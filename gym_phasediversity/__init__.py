from gymnasium.envs.registration import register

register(
    id='phasediversity-v0',
    entry_point='gym_phasediversity.envs:PhaseDiversityEnv',
)