from gym.envs.registration import register

register(
    id='meda-v0',
    entry_point='envs:MEDAEnv',
)
