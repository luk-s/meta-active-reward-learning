from typing import Any, Optional, Union

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback

from .rollout_buffer import RolloutBufferCallback
from .time_logging import TimeLoggingCallback


def get_callbacks(
    configs: Union[list[dict[str, Any]], dict[str, Any]], agent: Optional[BaseAlgorithm] = None
) -> list[BaseCallback]:
    """Given a (list of) callback configurations, return a (list of) initialized callbacks.

    Args:
        configs (Union[list[dict[str, Any]], dict[str, Any]]): The (list of) callback configurations.
        agent (Optional[BaseAlgorithm], optional): The agent to use for the callback. Defaults to None.

    Raises:
        KeyError: If one of the configurations misses the 'name' key.
        ValueError: If the specified callback requires an agent but no agent has been provided.
        ValueError: If the specified callback is not supported.

    Returns:
        list[BaseCallback]: A (list of) initialized callbacks.
    """
    # Wrap the configs into a list if necessary
    if isinstance(configs, dict):
        configs = [configs]

    callbacks: list[BaseCallback] = []

    for config in configs:
        # Check if the config contains a 'name' entry
        if "name" not in config:
            raise KeyError("The callback config requires an entry 'name'!")

        # Extract the name
        name_orig = config["name"]
        name = name_orig.lower()

        # Delete the name from the dictionary. This allows to pass the remaining
        # parameters directly to the initialization function of the callback
        parameter_config = dict(config)
        del parameter_config["name"]

        # Check if this config requires an agent and if so, add it to the config
        needs_agent = ("agent" in parameter_config) and (parameter_config["agent"])
        if needs_agent:
            if agent is None:
                raise ValueError("The config requires an agent but no agent has been provided!")
            parameter_config["agent"] = agent

        callback: Union[TimeLoggingCallback, RolloutBufferCallback]

        if name == "time_logging":
            callback = TimeLoggingCallback(**parameter_config)
        elif name == "rollout_buffer":
            callback = RolloutBufferCallback(**parameter_config)
        else:
            raise ValueError(f"No implementation of the callback {name_orig} exists!")

        callbacks.append(callback)

    return callbacks
