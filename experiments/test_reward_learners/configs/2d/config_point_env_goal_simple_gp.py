{
    "experiment": {
        "seed": 0,
        "train_size": 100,  # 100,
        "test_size": 200,
        "env_config": {
            "name": "point_env_goal",  # "point_env_goal",
            "arena_size": 5.0,
            "arena_dim": 2,
            "num_tasks": 1,
            # "target_pos": {0: [3, -2]},
        },
        "reward_model_config": {
            "name": "learned_gp",
            "model_name": "vanilla_gp",
            "config": {
                "input_dim": 2,
                "kernel_variance": 0.9,  # 0.5,
                "kernel_lengthscale": 0.95,  # 0.4,
                "likelihood_std": 0.0001,  # 0.01,
                "normalize_data": True,
            },
        },
    },
    "prefix": "simple",
    "name": "GP_mean:zero_kernel:SE",
    "num_runs": 50,
}
