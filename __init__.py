from gym.envs.registration import register

register(
    id='dmfb-v0',
    entry_point='dmfb_env.envs:DMFBEnv',
)
