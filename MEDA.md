PPO2:

[ppo2]:

Contexts:
    Train model:

[__init__]:
    [super().__init__][base_class.ActorCriticRLModel]:
        initializes properties
    [setup_model]:
        create tensorflow graph [graph]
        make tf session [sess]
        create actor model [act_model] by calling [MyCnnPolicy.__init__]
            n_envs=n_env, reuse=False
        with train_model scope and reuse:
            create train model [tain_model] by calling [MyCnnPolicy.__init__]
                n_envs = n_envs//nminibatches (which is 0!)

    


