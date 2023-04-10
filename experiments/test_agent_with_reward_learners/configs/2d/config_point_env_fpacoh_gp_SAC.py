{
    "experiment": {
        "seed": 0,
        "meta_train": True,
        "meta_train_size": 20,
        "num_meta_tasks": 20,
        "observation_sampling_mode": "largest_uncertainty",
        "num_total_queries": 50,
        "num_queries_per_iteration": 5,
        "env_config": {
            "name": "point_env_gp_reward",
            "arena_size": 5.0,
            "arena_dim": 2,
            "max_episode_length": 200,
            "use_upper_conf_bound": True,
            # "target_pos": {0: [3, -2]},
        },
        "reward_model_config": {
            "name": "fpacoh_learned_gp",
            "model_name": "fpacoh",
            "config": {
                "learning_mode": "both",  # "vanilla",
                "mean_module": "NN",  # "zero"
                "covar_module": "NN",  # "SE"
                "weight_decay": 1e-4,  # 1e-3,
                "feature_dim": 2,
                "num_iter_fit": 5000,
                "lr": 1e-3,  # 0.01,  # 1e-3,
                "lr_decay": 1.0,
                "prior_lengthscale": 0.5,  # 0.2,
                "prior_outputscale": 1,
                "num_samples_kl": 20,
                "prior_factor": 0.2,  # 0.2,
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
            "buffer_size": 50000,  # Needs to be >= than total_timesteps below
            "learning_rate": 0.01,
            "tau": 0.02,
            "gamma": 0.99,
            "train_freq": 200,
            "gradient_steps": 200,
            "ent_coef": "auto",
            "learning_starts": 1000,
            "policy_kwargs": {"net_arch": [128, 128]},
            "device": "cpu",
            ###################
            # PPO CONFIG
            ###################
            # "name": "PPO",
            # "policy": "MlpPolicy",
            # "verbose": 1,
            # "n_steps": 64,
            # "batch_size": 32,
            # "learning_rate": 1e-3,
            # "ent_coef": 0.01,
            # "gae_lambda": 0.9,
            # "vf_coef": 0.1,
            # "device": "cpu",
        },
        "agent_train_config": {"total_timesteps": 40000, "log_interval": 10},  # SAC
        # "agent_train_config": {"total_timesteps": 2000, "log_interval": 10},  # SAC SMALL
        # "agent_train_config": {"total_timesteps": 500000, "log_interval": 500}, # PPO
        "callback_configs": [
            {
                "name": "rollout_buffer",
                "agent": True,
                "size": 10000,
                "dim": 2,
                "expected_num_elements": 40000,
                "verbose": 0,
            },
            {"name": "time_logging"},
        ],
    },
    "prefix": "fpacoh_learnt",
    "name": "fpacoh_GP_mean:NN_kernel:NN_SAC_logging",
    "num_runs": 50,
}
