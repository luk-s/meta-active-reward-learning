import multiprocessing
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# from .abstract import EnvironmentBase
from .point_env_goal import PointEnv
from .point_env_goal_3d import PointEnv3d
from .point_env_gp_reward import PointEnvGPReward
from .point_env_tile import PointEnvTileReward

if TYPE_CHECKING:
    from meta_arl.environments.abstract import EnvironmentBase


def get_environment(config: dict) -> "EnvironmentBase":
    """Given a configuration dictionary, initialize and return the corresponding environment.

    Args:
        config (dict): A dictionary containing the configuration for the environment.

    Raises:
        KeyError: If the configuration dictionary does not contain a key "name".
        Exception: If the environment name is not recognized.

    Returns:
        EnvironmentBase: _description_
    """
    # Check if the config contains a 'name' entry
    if "name" not in config:
        raise KeyError("The environment config requires an entry 'name'!")

    # Create a copy of the config dictionary
    parameter_config = dict(config)

    # Extract the name
    name_orig = config["name"]
    name = name_orig.lower()

    # Extract the seed for the random state
    if "rng_seed" in config:
        seed = parameter_config["rng_seed"]
        del parameter_config["rng_seed"]
    else:
        # raise ValueError("You need to specify 'rng_seed' for the environment!")
        seed = None

    # Create the random state
    rng = np.random.default_rng(seed)

    # Delete the name from the dictionary. This allows to pass the remaining
    # parameters directly to the initialization function of the environment
    del parameter_config["name"]

    if name == "point_env_goal":
        environment = PointEnv(**parameter_config, rng=rng)
    elif name == "point_env_goal_3d":
        environment = PointEnv3d(**parameter_config, rng=rng)
    elif name == "point_env_gp_reward":
        environment = PointEnvGPReward(**parameter_config, rng=rng)
    elif name == "point_env_tile":
        environment = PointEnvTileReward(**parameter_config, rng=rng)
    else:
        raise Exception(f"No implementation of the environment {name_orig} exists!")

    return environment


def vectorize_environment(
    base_environment: "EnvironmentBase",
    env_config: dict,
    num_copies: int = 4,
    start_method: str = "forkserver",
) -> SubprocVecEnv:
    """Vectorize the given environment.

    Args:
        base_environment (EnvironmentBase): The environment to vectorize.
        env_config (dict): The configuration for the environment.
        num_copies (int, optional): The number of copies of the environment to
            create. Defaults to 4.
        start_method (str, optional): The start method to use for the
            multiprocessing. Defaults to "forkserver".

    Raises:
        ValueError: If the start method is not recognized.

    Returns:
        SubprocVecEnv: The vectorized environment.
    """
    if start_method not in multiprocessing.get_all_start_methods():
        raise ValueError(
            f"'start_method' must be one of {multiprocessing.get_all_start_methods()}"
            f" but got {start_method} instead."
        )

    def make_env() -> "EnvironmentBase":

        # Create the new environment
        my_env = get_environment(env_config)

        # Copy the tasks of the base environment
        my_env.set_tasks(base_environment.tasks)

        # Copy the reward model of the base environment
        my_env.set_reward_model(deepcopy(base_environment.reward_model))

        # Make the environment use the reward model for reward prediction
        my_env.set_use_true_reward(base_environment.use_true_reward)

        # Make sure that the current task is task 0
        my_env.set_current_task(0)

        return my_env

    # Create the vectorized environment
    vec_env = SubprocVecEnv([make_env for env_id in range(num_copies)], start_method=start_method)

    return vec_env
