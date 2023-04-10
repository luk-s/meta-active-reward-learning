import inspect
from argparse import Action
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Optional, Protocol, Union

import numpy as np
import stable_baselines3
from stable_baselines3.common.vec_env import SubprocVecEnv

if TYPE_CHECKING:
    from meta_arl.environments.abstract import EnvironmentBase


RLAlgorithm = stable_baselines3.common.base_class.BaseAlgorithm
ActionNoise = stable_baselines3.common.noise.ActionNoise


class RLAlgorithmConstructor(Protocol):
    def __call__(
        self,
        env: Union["EnvironmentBase", SubprocVecEnv],
        seed: int,
        action_noise: Optional[ActionNoise],
    ) -> Any:
        """This class is only used for python typing. It describes
        the signature of the constructor of a stable-baselines3 RL algorithm.

        Args:
            env (Union[&quot;EnvironmentBase&quot;, SubprocVecEnv]): The environment to train the
                RL algorithm on.
            seed (int): The seed to use for the RL algorithm.
            action_noise (Optional[ActionNoise]): The optional action noise to use for the RL
                algorithm.

        Returns:
            Any: The stable-baselines3 RL algorithm instance.
        """
        ...


class ActionNoiseConstructor(Protocol):
    def __call__(self, **kwds: Any) -> Any:
        """This class is only used for python typing. It describes
        the signature of the constructor of a stable-baselines3 action noise class.

        Args:
            **kwds (Any): The keyword arguments to use for the action noise constructor.

        Returns:
            Any: The stable-baselines3 action noise instance.
        """
        ...


agent_map: dict[str, RLAlgorithmConstructor] = {
    a[0].lower(): a[1] for a in inspect.getmembers(stable_baselines3, inspect.isclass)
}

action_noise_map: dict[str, ActionNoiseConstructor] = {
    n[0].lower(): n[1] for n in inspect.getmembers(stable_baselines3.common.noise, inspect.isclass)
}


class CustomRolloutBuffer:
    supported_classes: ClassVar[dict[str, dict[str, str]]] = {
        "PPO": {"buffer": "rollout_buffer", "next_observations": "observations"},
        "SAC": {"buffer": "replay_buffer"},
    }

    def __init__(self, agent: RLAlgorithm) -> None:
        """Initialize the custom rollout buffer.

        Args:
            agent (RLAlgorithm): The RL algorithm whose rollout buffer
                should be wrapped with this class.

        Raises:
            TypeError: If the specified agent is not supported
        """
        self.agent = agent
        self.class_name = type(self.agent).__name__

        # Check that the class of the agent is supported
        if self.class_name not in self.supported_classes:
            raise TypeError(
                f"Agents of type '{self.class_name}' are currently "
                f"not supported! Supported types: {self.supported_classes.keys()}"
            )

    @property
    def buffer(self) -> Any:
        """Return the buffer of the agent.

        Returns:
            Any: The buffer of the agent.
        """
        # Return the buffer of our agent (e.g. 'ReplayBuffer' or 'RolloutBuffer')
        return getattr(self.agent, self.supported_classes[self.class_name]["buffer"])

    def __getattr__(self, name: str) -> Any:
        """Return the attribute with the specified name and replace it with
        the corresponding attribute of the 'support_classes' dictionary.

        Args:
            name (str): The name of the attribute to return.

        Returns:
            Any: The attribute with the specified name or a corresponding
                attribute from the 'supported_classes' dictionary.
        """
        # Replace the attribute name with a class-specific name if necessary
        name = self.supported_classes[self.class_name].get(name, name)

        # Return the corrected attribute
        return getattr(self.buffer, name)


# def get_agent(config: dict[str, Any], environment: EnvironmentBase) -> RLAlgorithm:
def get_agent(
    config: dict[str, Any], environment: Union["EnvironmentBase", SubprocVecEnv]
) -> RLAlgorithm:
    """Given an environment and an RL agent configuration, return the corresponding
    RL agent instance.

    Args:
        config (dict[str, Any]): The configuration of the RL agent.
        environment (Union[&quot;EnvironmentBase&quot;, SubprocVecEnv]): The environment to train
            the RL agent on.

    Raises:
        KeyError: If the agent config does not contain a 'name' key.
        ValueError: If the specified agent is not supported.
        ValueError: If the specified action noise is not supported.

    Returns:
        RLAlgorithm: An initialized RL agent instance.
    """

    # Check if the config contains a 'name' entry
    if "name" not in config:
        raise KeyError("The reward model config requires an entry 'name'!")

    # Create a copy of the config dictionary
    param_config = dict(config)

    # Extract the name
    name_orig = config["name"]
    name = name_orig.lower()

    # Extract the seed for the random state
    if "rng_seed" in config:
        seed = config["rng_seed"]
        del param_config["rng_seed"]
        if "seed" in config:
            del param_config["seed"]
    else:
        # raise ValueError("You need to specify 'rng_seed' for the agent!")
        seed = None

    # Create the random state
    rng = np.random.default_rng(seed)

    # Delete the name from the dictionary. This allows to pass the remaining
    # parameters directly to the initialization function of the reward model
    del param_config["name"]

    agent_class = agent_map.get(name, None)
    if agent_class is None:
        raise ValueError(
            f"No implementation of this agent {name_orig} exists! "
            f"Use one of {list(agent_map.keys())} instead."
        )

    if "action_noise" in param_config:
        action_noise_config = dict(param_config["action_noise"])
        action_noise_name = action_noise_config["name"].lower()
        action_noise_class = action_noise_map.get(action_noise_name, None)
        if action_noise_class is None:
            raise ValueError(
                f"No implementation of this action noise {action_noise_name} exists! "
                f"Use one of {list(action_noise_map.keys())} instead."
            )
        for key in action_noise_config:
            if isinstance(action_noise_config[key], list):
                action_noise_config[key] = np.array(action_noise_config[key])

        del action_noise_config["name"]
        action_noise = action_noise_class(**action_noise_config)
        del param_config["action_noise"]

        agent = agent_class(
            env=environment,
            action_noise=action_noise,
            seed=int(rng.integers(1 << 32) - 1),
            **param_config,
        )

    else:
        agent = agent_class(env=environment, seed=int(rng.integers(1 << 32) - 1), **param_config)

    # Add a custom rollout buffer
    agent.custom_buffer = CustomRolloutBuffer(agent)

    return agent
