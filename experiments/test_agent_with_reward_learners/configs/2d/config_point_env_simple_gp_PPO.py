{
    "experiment": {
        "seed": 0,
        "observation_sampling_mode": "largest_uncertainty",
        "num_total_queries": 50,
        "num_queries_per_iteration": 5,
        "env_config": {
            "name": "point_env_gp_reward",  # "point_env_goal",
            "arena_size": 5.0,
            "arena_dim": 2,
            "num_tasks": 1,
            "max_episode_length": 200,
            "use_upper_conf_bound": True,
            # "target_pos": {0: [-4, 4]},
        },
        "reward_model_config": {
            "name": "learned_gp",
            "model_name": "vanilla_gp",
            "config": {
                "input_dim": 2,
                "kernel_variance": 0.5,
                "kernel_lengthscale": 0.4,
                "likelihood_std": 0.01,
                "normalize_data": True,
            },
        },
        "agent_config": {
            ###################
            # SAC CONFIG
            ###################
            # "name": "SAC",
            # "policy": "MlpPolicy",
            # "verbose": 1,
            # "optimize_memory_usage": False,
            # "buffer_size": 50000,  # Needs to be >= than total_timesteps below
            # "learning_rate": 0.01,
            # "tau": 0.02,
            # "gamma": 0.99,
            # "train_freq": 200,
            # "gradient_steps": 200,
            # "ent_coef": "auto",
            # "learning_starts": 1000,
            # "policy_kwargs": {"net_arch": [128, 128]},
            ###################
            # PPO CONFIG
            ###################
            "name": "PPO",
            "policy": "MlpPolicy",
            "verbose": 1,
            "n_steps": 128,
            "batch_size": 32,
            "learning_rate": 1e-3,
            "ent_coef": 0.01,
            "gae_lambda": 0.9,
            "vf_coef": 0.1,
        },
        # "agent_train_config": {"total_timesteps": 40000, "log_interval": 10},  # SAC
        "agent_train_config": {"total_timesteps": 500000, "log_interval": 500},  # PPO
        "callback_configs": [
            {
                "name": "rollout_buffer",
                "agent": True,
                "size": 10000,
                "dim": 2,
                "expected_num_elements": 500000,
                "verbose": 0,
            },
            {"name": "time_logging"},
        ],
    },
    "prefix": "simple",
    "name": "GP_mean:zero_kernel:SE_PPO",
    "num_runs": 50,
}
