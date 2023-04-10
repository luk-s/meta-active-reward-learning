from typing import Any, ClassVar, Union

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from meta_arl.agents import CustomRolloutBuffer


class RolloutBufferCallback(BaseCallback):
    """A Callback which acts as a wrapper and storage of the rl agent's
    rollout buffer. This is necessary because, depending on the rl agent instance,
    the rollout buffer is deleted at the end of each rollout.

    Args:
        BaseCallback (_type_): The base class for callbacks.

    Raises:
        TypeError: If the provided rl agent type is not supported.
    """

    supported_classes: ClassVar[dict[str, dict[str, str]]] = {
        "PPO": {"type": "on_policy"},
        "SAC": {"type": "off_policy"},
    }

    def __init__(
        self,
        agent: Any,  # This is an object of type stable_baselines3.common.base_class.BaseAlgorithm
        size: int = 10000,
        dim: int = 2,
        expected_num_elements: int = 10000,
        verbose: int = 0,
    ) -> None:
        """Initialize the callback with the necessary parameters.

        Args:
            agent (Any): An instance of a stable_baselines3 rl agent.
            dim (int, optional): The dimension of the state space in which the
                agent operates. Defaults to 2.
            expected_num_elements (int, optional): The expected number of elements
                which will be added to the buffer. Defaults to 10000.
            verbose (int, optional): The verbosity level. Defaults to 0.

        Raises:
            TypeError: If the provided rl agent type is not supported.
        """
        super(RolloutBufferCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm

        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]

        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int

        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]

        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger

        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.class_name = type(agent).__name__

        # Check that the class of the agent is supported
        if self.class_name not in self.supported_classes:
            raise TypeError(
                f"Agents of type '{self.class_name}' are currently "
                f"not supported! Supported types: {self.supported_classes.keys()}"
            )

        self.is_on_policy = self.supported_classes[self.class_name]["type"] == "on_policy"

        agent.custom_buffer = self

        self.replay_buffer: Union[CustomRolloutBuffer, Container]
        if not self.is_on_policy:
            self.replay_buffer = agent.replay_buffer

        self.prob_to_add = size / max(expected_num_elements, size)
        self.size = size
        self._pos = 0
        self._observations = np.zeros((size, 1, dim))
        self._actions = np.zeros((size, 1, dim))

        self._flattened = False

    @property
    def pos(self) -> int:
        """Returns an index which indicates the position in the pre-allocated
        rollout buffer until which data has been stored.

        Returns:
            int: The position in the pre-allocated rollout buffer.
        """
        if self.is_on_policy:
            return self._pos
        else:
            # WRONG! assert isinstance(self.replay_buffer, CustomRolloutBuffer)
            is_full = self.replay_buffer.full  # type: ignore

            if not self._flattened:
                self.replay_buffer = flatten_buffer(self.replay_buffer)
                assert isinstance(self.replay_buffer, Container) or isinstance(
                    self.replay_buffer, CustomRolloutBuffer
                )

                self._flattened = True

            if is_full:
                print("REPLAY BUFFER OFERFLOW")
                return self.observations.shape[0]

            return self.replay_buffer.pos

    @property
    def observations(self) -> np.ndarray:
        """Returns the observations of each step stored in the rollout buffer.

        Returns:
            np.ndarray: The observations
        """
        if self.is_on_policy:
            return self._observations
        else:
            assert self.replay_buffer is not None
            if not self._flattened:
                self.replay_buffer = flatten_buffer(self.replay_buffer)
                self._flattened = True
            return self.replay_buffer.observations

    @property
    def actions(self) -> np.ndarray:
        """Returns the actions of each step stored in the rollout buffer.

        Returns:
            np.ndarray: The actions
        """
        if self.is_on_policy:
            return self._actions
        else:
            if not self._flattened:
                self.replay_buffer = flatten_buffer(self.replay_buffer)
                self._flattened = True
            return self.replay_buffer.actions

    @property
    def next_observations(self) -> np.ndarray:
        """Returns the next observations of each step stored in the rollout buffer.

        Returns:
            np.ndarray: The next observations
        """
        if self.is_on_policy:
            return self._observations
        else:
            if not self._flattened:
                self.replay_buffer = flatten_buffer(self.replay_buffer)
                self._flattened = True
            return self.replay_buffer.next_observations

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        # This is only necessary for on-policy algorithms which delete their
        # rollout buffer at the start of each rollout collection round
        if self.is_on_policy:
            rollout_buffer = flatten_buffer(self.locals["rollout_buffer"], flatten_pos=False)

            # The number of elements which can be filled directly
            num_to_add = min(self.size - self.pos, rollout_buffer.pos)

            # If the buffer still has available space, fill it
            if self.pos < self.size:
                self._observations[self.pos : self.pos + num_to_add] = rollout_buffer.observations[
                    :num_to_add
                ]
                self._actions[self.pos : self.pos + num_to_add] = rollout_buffer.actions[
                    :num_to_add
                ]
                self._pos += num_to_add

            # Add every additional element with a certain probability
            if num_to_add < rollout_buffer.pos:
                # Compute the remaining elements which could potentially be added
                num_to_add_new = rollout_buffer.pos - num_to_add

                # Select each of the remaining elements with a certain probability
                random_numbers = np.random.random(size=num_to_add_new)
                indices_to_add = np.where(random_numbers <= self.prob_to_add)[0]

                # If some elements have been selected
                if len(indices_to_add) > 0:
                    # Create the real indices
                    indices_to_add = indices_to_add + num_to_add

                    # Select the positions into which the elements shall be stored
                    positions = np.random.choice(range(self.size), len(indices_to_add))

                    # Add the elements to the buffer
                    self._observations[positions] = rollout_buffer.observations[indices_to_add]
                    self._actions[positions] = rollout_buffer.actions[indices_to_add]

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """


class Container:
    def __init__(self) -> None:
        """A container class which is used to wrap a flattened rollout buffer."""
        self.pos: int
        self.observations: np.ndarray
        self.actions: np.ndarray
        self.next_observations: np.ndarray


def flatten_buffer(
    rollout_buffer: Union[CustomRolloutBuffer, Container], flatten_pos: bool = True
) -> Union[Container, CustomRolloutBuffer]:
    """Flattens the observation- and state-arrays stored in a provided rollout buffer.

    Args:
        rollout_buffer (Union[CustomRolloutBuffer, Container]): The rollout buffer which shall be
            flattened.
        flatten_pos (bool, optional): Whether the position in the rollout buffer shall be
            flattened.

    Returns:
        Union[Container, CustomRolloutBuffer]: The flattened rollout buffer.
    """
    # Get the dimensions of the observations
    size, num_env, dim = rollout_buffer.observations.shape

    # If the dimensions are already flattened, don't do anything
    if num_env == 1:
        return rollout_buffer

    # Define an empty object. We only use it to assign attributes to it.
    empty = Container()

    # Assign the flattened attributes to it
    empty.observations = rollout_buffer.observations.reshape(size * num_env, 1, dim)
    empty.actions = rollout_buffer.actions.reshape(size * num_env, 1, dim)

    if flatten_pos:
        empty.pos = rollout_buffer.pos * num_env
    else:
        empty.pos = rollout_buffer.pos
    if hasattr(rollout_buffer, "next_observations"):
        empty.next_observations = rollout_buffer.next_observations.reshape(size * num_env, 1, dim)
    return empty
