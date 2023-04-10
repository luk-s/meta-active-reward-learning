from typing import TYPE_CHECKING, Optional

import numpy as np

from meta_arl.reward_models.abstract import RewardModelGPBase
from meta_arl.reward_models.reward_model_gp import RewardModelGP
from meta_arl.reward_models.reward_model_pacoh_gp import RewardModelPacohGP

if TYPE_CHECKING:
    from meta_arl.environments.abstract import EnvironmentBase


def get_reward_model(
    config: dict, environment: Optional["EnvironmentBase"] = None
) -> RewardModelGPBase:
    """Takes a config dictionary and returns an initialized reward model

    Args:
        config (dict): The config dictionary
        environment (Optional[EnvironmentBase], optional): An optional environment
        whose reward function the reward model should learn. Defaults to None.

    Raises:
        KeyError: If the reward model config misses an entry 'name'
        Exception: If the specified reward model config is not supported.

    Returns:
        RewardModelGPBase: _description_
    """

    # Check if the config contains a 'name' entry
    if "name" not in config:
        raise KeyError("The reward model config requires an entry 'name'!")

    # Create a copy of the config dictionary
    parameter_config = dict(config)

    # Extract the name
    name_orig = config["name"]
    name = name_orig.lower()

    # Extract the seed for the random state
    if "rng_seed" in config:
        seed = config["rng_seed"]
        del parameter_config["rng_seed"]
    else:
        # raise ValueError("You need to specify 'rng_seed' for the reward model!")
        seed = None

    # Create the random state
    rng = np.random.default_rng(seed)

    # Delete the name from the dictionary. This allows to pass the remaining
    # parameters directly to the initialization function of the reward model
    del parameter_config["name"]

    reward_model: RewardModelGPBase

    if name == "vanilla_gp":
        pass
    elif name == "learned_gp":
        reward_model = RewardModelGP(**parameter_config, environment=environment, rng=rng)
    elif name == "pacoh_learned_gp" or name == "fpacoh_learned_gp":
        reward_model = RewardModelPacohGP(**parameter_config, environment=environment, rng=rng)
    else:
        raise Exception(f"No implementation of the reward model {name_orig} exists!")

    return reward_model
