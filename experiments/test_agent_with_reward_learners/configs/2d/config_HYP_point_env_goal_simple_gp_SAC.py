{
    "experiment": {
        "seed": 0,
        "observation_sampling_mode": "largest_uncertainty",
        "num_total_queries": 50,
        "num_queries_per_iteration": 5,
        "should_vectorize_environment": True,
        "env_config": {
            "name": "point_env_goal",  # "point_env_goal",
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
            "name": "SAC",
            "policy": "MlpPolicy",
            "verbose": 1,
            "optimize_memory_usage": False,
            "buffer_size": 110001,  # Needs to be > than total_timesteps below
            "learning_rate": [1e-4, 1e-3, 1e-2],
            "tau": [0.01, 0.02],
            "gamma": [0.99, 0.999, 1],
            "train_freq": [1, 32, 64, 200],
            # "gradient_steps": 200,
            "ent_coef": "auto",
            "learning_starts": [1000, 3000],
            "policy_kwargs": [
                {"net_arch": [64, 64]},
                {"net_arch": [128, 128]},
                {"net_arch": [256, 256]},
            ],
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
            # "n_steps": [256, 512, 1024, 2048],
            # "learning_rate": [1e-2, 1e-3, 1e-4],
            # "ent_coef": [0, 0.005, 0.01],
            # "gae_lambda": [0.8, 0.9, 0.92],
            # "vf_coef": [0.1, 0.3, 0.5],
            # "device": "cpu",
        },
        "agent_train_config": {"total_timesteps": 100000, "log_interval": 50},
        # "agent_train_config": {"total_timesteps": 2000000, "log_interval": 100}, # PPO
        "callback_configs": [
            {
                "name": "rollout_buffer",
                "agent": True,
                "size": 10000,
                "dim": 2,
                "expected_num_elements": 100000,
                "verbose": 0,
            },
            {"name": "time_logging"},
        ],
    },
    "prefix": "simple",
    "name": "HYP_GP_mean:zero_kernel:SE_PPO",
    "num_runs": 50,
}
