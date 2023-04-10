from typing import Any, Callable, Optional, Union, cast

import gym
import numpy as np

from meta_arl.meta_bo.domain import ContinuousDomain, DiscreteDomain
from meta_arl.reward_models.abstract import RewardModelGPBase
from meta_arl.util.myrandom import set_all_seeds
from meta_arl.util.noise import Noise2D

Domain = Union[ContinuousDomain, DiscreteDomain]


class PointTasksBase:
    """A class to handle multiple point tasks."""

    _current_task: int
    _current_pos: np.ndarray
    _step_cnt: np.ndarray
    _is_done: np.ndarray
    num_tasks: int

    @property
    def current_task(self) -> int:
        """Returns the index of the currently active task."""
        return self._current_task

    @property
    def current_pos(self) -> np.ndarray:
        """Returns the position of the currently active task."""
        return self._current_pos[self.current_task]

    @property
    def step_cnt(self) -> int:
        """Returns the number of steps in the environment, which have
        been performed with the current task
        """
        return self._step_cnt[self.current_task]

    @property
    def is_done(self) -> bool:
        """Returns whether the current task is done."""
        return self._is_done[self.current_task]

    def set_current_pos(self, pos: np.ndarray) -> None:
        """Set the position of the currently active task

        Args:
            pos (np.ndarray): The new position

        Raises:
            ValueError: If the position is not of the correct shape
        """
        pos = np.array(pos)
        if pos.shape != self.current_pos.shape:
            raise ValueError(
                "The provided position must have the same shape as the current "
                f" position! Provided pos shape: {pos.shape}, current pos shape: "
                f"{self.current_pos.shape}"
            )
        self._current_pos[self.current_task] = pos

    def set_current_task(self, task_id: int) -> None:
        """Changes the current task to the task with the given id.

        Args:
            task_id (int): The id of the task to change to

        Raises:
            TypeError: If the task id is not a non-negative integer
        """
        if type(task_id) is not int or (type(task_id) is int and task_id < 0):
            raise TypeError("'task_id' needs to be a non-negative 'int'!")
        self._current_task = task_id

    def set_step_cnt(self, count: int) -> None:
        """Changes the step count of the currently active task.

        Args:
            count (int): The new step count

        Raises:
            TypeError: If the step count is not an integer
        """
        if type(count) in [float, np.float32, np.float64]:
            count = int(count)
        if type(count) is not int:
            raise TypeError("'count' needs to be of type 'int'!")
        self._step_cnt[self.current_task] = count

    def set_is_done(self, done: bool) -> None:
        """Sets the done flag of the currently active task.

        Args:
            done (bool): _description_

        Raises:
            TypeError: _description_
        """
        if type(done) is not bool:
            raise TypeError("'done' needs to be of type 'bool'!")
        self._is_done[self.current_task] = done

    def set_task(self, task_index: int, task: Any) -> None:
        """Replace the task at index 'task_index' with the given task.

        Args:
            task_index (int): The index of the task to replace
            task (Any): The new task

        Raises:
            NotImplementedError: This method needs to be implemented by the
                subclass
        """
        raise NotImplementedError()

    def print_task(self, task_index: int) -> None:
        """Print information about the task at index 'task_index'.

        Args:
            task_index (int): The index of the task to print

        Raises:
            NotImplementedError: This method needs to be implemented by the
                subclass
        """
        raise NotImplementedError()


class EnvironmentBase(gym.Env):
    """A base class for all environments."""

    def __init__(self, rng: np.random.Generator) -> None:
        """Initialize the environment.

        Args:
            rng (np.random.Generator): The random number generator to use
        """
        self._action_space: gym.spaces.Space
        self._observation_space: gym.spaces.Space
        self._domain: Domain
        self._task: PointTasksBase
        self._reward_model: Optional[RewardModelGPBase]
        self._num_tasks: int
        self._use_true_reward: bool = True
        self._use_upper_conf_bound: bool = False
        self._normalization_stats: Optional[dict[str, np.ndarray]] = None
        self._max_episode_length: int
        self._noise: Optional[Noise2D] = None
        assert rng is not None, "You need to provide a random state!"
        self._rng = rng

    @property
    def action_space(self) -> gym.spaces.Space:
        """Return the action space of the environment."""
        return self._action_space

    @property
    def domain(self) -> Domain:
        """Return the domain of the environment."""
        return self._domain

    @property
    def normalization_stats(self) -> dict[str, np.ndarray]:
        """Provides statistics to normalize observations, actions, and rewards of
        the environment.

        Returns:
            dict[str, np.ndarray]: A dictionary containing the mean and
                standard deviation of   observations, actions, and rewards
        """
        if self._normalization_stats is None:
            """
            if isinstance(self.domain, ContinuousDomain):
                x_points = np.random.uniform(
                    self.domain.l,
                    self.domain.u,
                    size=(1000 * self.domain.d**2, self.domain.d),
                )
            elif isinstance(self.domain, DiscreteDomain):
                x_points = self.domain.points
            else:
                raise NotImplementedError
            """
            # Compute mean and std of observation space
            if isinstance(self.observation_space, gym.spaces.Box):
                obs_mean = (self.observation_space.high + self.observation_space.low) / 2
                obs_std = np.sqrt(1 / 12) * (
                    self.observation_space.high - self.observation_space.low
                )
            else:
                raise ValueError("Only 'gym.spaces.Box' spaces are supported!")

            # Compute mean and std of action space
            if isinstance(self.action_space, gym.spaces.Box):
                action_mean = (self.action_space.high + self.action_space.low) / 2
                action_std = np.sqrt(1 / 12) * (self.action_space.high - self.action_space.low)
            else:
                raise ValueError("Only 'gym.spaces.Box' spaces are supported!")

            # Compute min and max of reward space
            if isinstance(self.observation_space, gym.spaces.Box):
                obs = self.observation_space
                dim = len(obs.low)
                resolution = 100 if dim == 3 else 500
                x_points_first = np.meshgrid(
                    *[np.linspace(low, high, resolution) for low, high in zip(obs.low, obs.high)]
                )
                x_points = np.array(
                    [
                        np.concatenate(x_points_dim, axis=None)  # The axis=None is important!
                        for x_points_dim in x_points_first
                    ]
                ).T

            else:
                raise ValueError("Only 'gym.spaces.Box' spaces are supported!")

            ys = np.array(self._reward(x_points), dtype=np.float32)
            y_min, y_max = np.min(ys), np.max(ys)
            self._normalization_stats = {
                "x_mean": obs_mean,
                "x_std": obs_std,
                "y_mean": (y_max + y_min) / 2.0,
                "y_std": (y_max - y_min) / 5.0,
                "a_mean": action_mean,
                "a_std": action_std,
            }
            for key, value in self._normalization_stats.items():
                self._normalization_stats[key] = np.array(value, dtype=np.float64)

        return self._normalization_stats

    @property
    def num_tasks(self) -> int:
        """Return the number of tasks which this environment stores."""
        return self._num_tasks

    @property
    def observation_space(self) -> gym.spaces.Space:
        """Return the observation space of the environment."""
        return self._observation_space

    @property
    def render_modes(self) -> list[str]:
        """list: A list of string representing the supported render modes."""
        return [
            "ascii",
        ]

    @property
    def reward_model(self) -> Optional[RewardModelGPBase]:
        """Return the reward model of the environment."""
        return self._reward_model

    @property
    def use_true_reward(self) -> bool:
        """Return whether the true reward or the learned reward should be used."""
        return self._use_true_reward

    @property
    def use_upper_conf_bound(self) -> bool:
        """Return whether the upper confidence bound of the learnt reward or
        the true reward should be used."""
        return self._use_upper_conf_bound

    @property
    def tasks(self) -> PointTasksBase:
        """Return the tasks of the environment."""
        return self._task

    def set_reward_noise(self, noise: Noise2D) -> None:
        """Set the noise which should be added to the reward function

        Args:
            noise (Noise2D): The noise which should be added to the reward

        Raises:
            TypeError: If the noise is not of type 'Noise2D'
        """
        if not isinstance(noise, Noise2D):
            raise TypeError(
                "'noise' must be of type or a subtype of 'Noise2D! " f"Got {type(noise)} instead."
            )
        self._noise = noise

    def normalize_observation(self, observation: np.ndarray) -> np.ndarray:
        """Normalize a given observation.

        Args:
            observation (np.ndarray): The observation which should be normalized

        Returns:
            np.ndarray: The normalized observation
        """
        norm_stats = self.normalization_stats
        obs_mean, obs_std = norm_stats["x_mean"], norm_stats["x_std"]
        return (observation - obs_mean) / obs_std

    def set_use_true_reward(self, value: bool) -> None:
        """Set whether the true reward or the learned reward should be used.

        Args:
            value (bool): The flag which indicates whether the true reward or
                the learned reward should be used

        Raises:
            TypeError: If the value is not of type 'bool'
        """
        if not isinstance(value, bool):
            raise TypeError("'value' needs to be of type 'bool'!")
        self._use_true_reward = value

    def _reward(self, state: np.ndarray) -> Union[float, np.ndarray]:
        """Compute the reward of the environment based only on the current state

        Args:
            state (np.ndarray): The current state of the environment

        Raises:
            NotImplementedError: This method needs to be implemented by the
                subclass

        Returns:
            Union[float, np.ndarray]: The reward of the current state
        """
        raise NotImplementedError()

    def reward(
        self, state: Optional[np.ndarray], action: Optional[np.ndarray], next_state: np.ndarray
    ) -> np.ndarray:
        """Compute the reward of the environment based on the current state, action and
        the next state

        Args:
            state (Optional[np.ndarray]): The current state of the environment
            action (Optional[np.ndarray]): The action which was taken in the current state
            next_state (np.ndarray): The next state of the environment

        Raises:
            NotImplementedError: This method needs to be implemented by the
                subclass

        Returns:
            np.ndarray: The reward of the current transition
        """
        raise NotImplementedError()

    def reset(self, seed: Optional[int] = None, normalize_obs: bool = True) -> np.ndarray:
        """Reset the current task of the environment to the initial state

        Args:
            seed (Optional[int], optional): The seed which should be used to
                reset the environment. Defaults to None.
            normalize_obs (bool, optional): Whether the observation should be
                normalized. Defaults to True.

        Returns:
            np.ndarray: The initial state of the current task of the environment
        """
        # Reset the current position
        self._task.set_current_pos(np.zeros_like(self._task.current_pos))

        # Reset the step count
        self._task.set_step_cnt(0)

        # Reset the done flag
        self._task.set_is_done(False)

        first_obs = self._task.current_pos.copy()

        if normalize_obs:
            first_obs = self.normalize_observation(first_obs)

        return first_obs

    def set_task(self, task_index: int, task: PointTasksBase) -> None:
        """Replace the task at the given index with the given task

        Args:
            task_index (int): The index of the task which should be replaced
            task (PointTasksBase): The task which should be used to replace the
                current task at the given index

        Raises:
            TypeError:
            ValueError: _description_
            ValueError: _description_
        """
        if not isinstance(task, type(self._task)):
            raise TypeError(f"Type of 'task' must be '{type(self._task)}'!")
        if task_index < 0 or task_index >= self.num_tasks:
            raise ValueError(f"The 'task_index' must be in the interval [0,{self.num_tasks - 1}]")
        if task.num_tasks != 1:
            raise ValueError(
                f"This function only sets a single task! Please make sure that "
                f"'num_tasks'==1. Got f{task.num_tasks}"
            )
        self._task.set_task(task_index, task)

    def set_tasks(self, task: PointTasksBase) -> None:
        """Replace all tasks with the given task

        Args:
            task (PointTasksBase): The task which should be used to replace all
                tasks

        Raises:
            TypeError: If the task is not an instance of type 'PointTasksBase'
        """
        if not isinstance(task, type(self._task)):
            raise TypeError(
                "Type of 'task' must be 'PointTasks(GP)'! " f"Got {type(self)} instead"
            )
        self._task = task

    def set_current_task(self, task_id: int) -> None:
        """Change the current task to the task with the given id

        Args:
            task_id (int): The id of the task which should be used as the
                current task

        Raises:
            TypeError: If the task id is not an appropriate integer
        """
        if type(task_id) is not int or task_id < 0 or task_id >= self._num_tasks:
            raise TypeError(
                f"'task_id' needs to be an integer in the interval " f"[0,{self._num_tasks - 1}]"
            )

        self._task.set_current_task(task_id)

    def set_reward_model(self, reward_model: Optional[RewardModelGPBase]) -> None:
        """Set the reward model which should be used to learn the reward of the
        environment

        Args:
            reward_model (Optional[RewardModelGPBase]): The reward model
        """
        self._reward_model = reward_model

    def set_seed(self, seed: int) -> list[int]:
        """Set the seed of the environment

        Args:
            seed (int): The seed

        Raises:
            TypeError: If the seed is not an integer

        Returns:
            list[int]: The seed of the environment wrapped in a list
        """
        if type(seed) is not int:
            raise TypeError("'seed' needs to be of type 'int'!")
        self._observation_space.seed(seed)
        self._action_space.seed(seed)
        # set_all_seeds(seed)
        self._rng = np.random.default_rng(seed)

        return [seed]

    def seed(self, seed: int) -> list[int]:
        """Set the seed of the environment

        Args:
            seed (int): The seed

        Returns:
            list[int]: The seed of the environment wrapped in a list
        """
        return self.set_seed(seed)

    def sample_rewards(
        self,
        num_samples: int,
        tasks: Union[int, list[int]] = 0,
        use_reward_model: bool = False,
        seed: Optional[int] = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Sample a set of observations and rewards from the environment

        Args:
            num_samples (int): The number of samples which should be drawn
            tasks (Union[int, list[int]], optional): The task id(s) which indicate
                the task(s) from which the samples should be drawn. Defaults to 0.
            use_reward_model (bool, optional): Whether the reward model should be
                used to compute the reward. Defaults to False.
            seed (Optional[int], optional): The seed which should be used to
                sample the observations and rewards. Defaults to None.

        Raises:
            ValueError: If the 'tasks' argument is not an integer or a list of
                integers

        Returns:
            list[tuple[np.ndarray, np.ndarray]]: A list of tuples which contain
                the observations and the rewards sampled from the environment
        """
        task_samples: list[tuple[np.ndarray, np.ndarray]] = []

        if type(tasks) is int:
            tasks = [tasks]
        elif tasks == "all":
            tasks = list(range(self._task.num_tasks))
        elif type(tasks) is not list:
            raise ValueError(
                "'tasks' needs to be either the string 'all' or " "be of type 'int', 'list(int)'"
            )

        # Set seed if desired
        if seed is not None:
            self.set_seed(seed)

        # Define which reward function to use
        get_reward: Callable
        if use_reward_model:
            assert self._reward_model is not None
            get_reward = self._reward_model.predict
        else:
            get_reward = self._reward

        for task_id in tasks:
            self._task.set_current_task(task_id)
            observations = []
            rewards = []

            for i in range(num_samples):
                observations.append(self.observation_space.sample())
                rewards.append(get_reward(observations[-1]))

            task_samples.append((np.array(observations), np.array(rewards)))

        return task_samples

    def step(
        self, action: np.ndarray, normalize_obs: bool = True
    ) -> tuple[np.ndarray, Union[float, np.ndarray], bool, dict]:
        """Perform a step in the environment

        Args:
            action (np.ndarray): The action which should be taken in the current state
            normalize_obs (bool, optional): Whether the observations should be
                normalized. Defaults to True.

        Raises:
            NotImplementedError: This function needs to be implemented by the
                subclass

        Returns:
            tuple[np.ndarray, Union[float, np.ndarray], bool, dict]: The next
                observation, the reward, whether the episode is done and a
                dictionary containing additional information
        """
        raise NotImplementedError()

    def unnormalize_observation(self, observation: np.ndarray) -> np.ndarray:
        """Unnormalize the given observation

        Args:
            observation (np.ndarray): The observation which should be unnormalized

        Returns:
            np.ndarray: The unnormalized observation
        """
        norm_stats = self.normalization_stats
        obs_mean, obs_std = norm_stats["x_mean"], norm_stats["x_std"]
        return observation * obs_std + obs_mean

    def print_task(self, task_index: int) -> None:
        """Print information about the task with the given index

        Args:
            task_index (int): The index of the task which should be printed

        Raises:
            ValueError: If the task index is not an appropriate integer
        """
        if task_index < 0 or task_index >= self.num_tasks:
            raise ValueError(f"The 'task_index' must be in the interval [0,{self.num_tasks - 1}]")
        self._task.print_task(task_index)
