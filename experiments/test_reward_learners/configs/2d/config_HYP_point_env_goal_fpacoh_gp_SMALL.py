{
    "experiment": {
        "seed": 0,
        "meta_train": True,
        "meta_train_size": 20,
        "num_meta_tasks": 20,
        "train_size": 100,  # 100,
        "test_size": 200,
        "env_config": {
            "name": "point_env_goal",
            "arena_size": 5.0,
            "arena_dim": 2,
            "max_episode_length": 200,
            "use_upper_conf_bound": False,
            # "target_pos": {0: [3, -2]},
        },
        "reward_model_config": {
            "name": "fpacoh_learned_gp",
            "model_name": "fpacoh",
            "config": {
                "learning_mode": "both",  # "vanilla",
                "mean_module": "NN",  # "zero"
                "covar_module": "NN",  # "SE"
                "weight_decay": [1e-2],
                "feature_dim": 2,
                "num_iter_fit": [5000],
                "lr": [1e-2],
                "lr_decay": 1.0,
                "prior_lengthscale": [0.5],
                "prior_outputscale": [1],
                "num_samples_kl": 20,
                "prior_factor": [0.05],
            },
        },
    },
    "prefix": "fpacoh_hyp",
    "name": "goal_env_fpacoh_GP",
    "num_runs": 30,
}
