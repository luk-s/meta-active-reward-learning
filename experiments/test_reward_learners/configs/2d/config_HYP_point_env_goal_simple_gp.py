{
    "experiment": {
        "seed": 0,
        "train_size": 100,  # 100,
        "test_size": 200,
        "env_config": {
            "name": "point_env_goal",
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
                "kernel_variance": [0.8, 0.85, 0.9, 0.95, 1, 1.25, 1.5],  # [0.1, 0.3, 0.5, 1, 2],
                "kernel_lengthscale": [
                    0.1,
                    0.3,
                    0.4,
                    0.5,
                    0.7,
                    0.8,
                    0.9,
                    0.95,
                    1,
                    1.05,
                ],  # [0.1, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 1],
                "likelihood_std": [
                    0.0001,
                ],  # [0.001, 0.005, 0.01, 0.03, 0.05, 0.1],
                "normalize_data": True,
            },
        },
    },
    "prefix": "simple",
    "name": "GP_mean:zero_kernel:SE",
    "num_runs": 50,
}