{
    "experiment": {
        "seed": 0,
        "meta_train": False,
        "observation_sampling_mode": "largest_uncertainty",
        "num_total_queries": 20,
        "num_queries_per_iteration": 2,
        "should_vectorize_environment": True,
        "env_config": {
            "name": "point_env_goal",
            "rng_seed": 0,
            "arena_size": 5.0,
            "arena_dim": 2,
            "max_episode_length": 200,
            "use_upper_conf_bound": True,
            "num_tasks": 1,
            # "target_pos": {0: [3, -2]},
        },
        "reward_model_config": {
            "name": "learned_gp",
            "rng_seed": 100,
            "model_name": "vanilla_gp",
            "config": {
                "input_dim": 2,
                "kernel_variance": 0.9,  # 0.5,
                "kernel_lengthscale": 0.95,  # 0.4,
                "likelihood_std": 0.0001,  # 0.01,
                "normalize_data": True,
            },
        },
        "agent_config": {
            ###################
            # SAC CONFIG
            ###################
            "name": "SAC",
            "rng_seed": 200,
            "policy": "MlpPolicy",
            "verbose": 1,
            "optimize_memory_usage": False,
            "buffer_size": 110001,  # Needs to be >= than total_timesteps below
            "learning_rate": 0.001,
            "tau": 0.01,
            "gamma": 0.99,
            "train_freq": 64,
            "gradient_steps": 64,
            "ent_coef": "auto",
            "learning_starts": 1000,
            "policy_kwargs": {"net_arch": [256, 256]},
            "device": "cpu",
            "action_noise": {
                "name": "OrnsteinUhlenbeckActionNoise",
                "mean": [0, 0],
                "sigma": [0.05, 0.05],
            }
            ###################
            # PPO CONFIG
            ###################
            # "name": "PPO",
            # "policy": "MlpPolicy",
            # "verbose": 1,
            # "use_sde": True,
            # "policy_kwargs": {"ortho_init": False},
            # "n_steps": 256,
            # "learning_rate": 1e-4,
            # "ent_coef": 0.01,
            # "gae_lambda": 0.9,
            # "vf_coef": 0.5,
            # "device": "cpu",
        },
        "agent_train_config": {"total_timesteps": 100000, "log_interval": 50},  # SAC
        # "agent_train_config": {"total_timesteps": 2000, "log_interval": 10},  # SAC SMALL
        # "agent_train_config": {"total_timesteps": 2000000, "log_interval": 500},  # PPO
        "callback_configs": [
            {
                "name": "rollout_buffer",
                "agent": True,
                "size": 10000,
                "dim": 2,
                "expected_num_elements": 2000000,
                "verbose": 0,
            },
            # {"name": "time_logging"},
        ],
    },
    "prefix": "simple_gp",
    "name": "goal_env_simple_GP_SAC",
    "num_runs": 50,
    "num_env_seeds": 10,
    "num_reward_learner_seeds": 2,
    "num_rl_seeds": 5,
}
