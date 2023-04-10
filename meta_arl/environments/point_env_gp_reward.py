from typing import Any, Iterable, Optional, Tuple, Union

import gym
import numpy as np
from gym import spaces

from meta_arl.environments.abstract import EnvironmentBase, PointTasksBase
from meta_arl.meta_bo.domain import ContinuousDomain
from meta_arl.reward_models.abstract import RewardModelGPBase


class PointTasksGP(PointTasksBase):
    def __init__(
        self,
        space: gym.spaces.Space,
        rng: np.random.Generator,
        num_tasks: int = 1,
        lengthscale: float = 2.0,
        l: int = 20,
        d: int = 2,
    ):
        """Initialize 'num_tasks' tasks for the PointEnvGP environment.

        Args:
            space (gym.spaces.Space): The observation space of the environment.
            rng (np.random.Generator): The random number generator.
            num_tasks (int, optional): The number of tasks. Defaults to 1.
            lengthscale (float, optional): The lengthscale of the GP. Defaults to 2.0.
            l (int, optional): A helper variable for the reward function. Defaults to 20.
            d (int, optional): The dimension of the observation space. Defaults to 2.
        """
        self._current_task = 0
        self.num_tasks = num_tasks
        samples = np.array([space.sample() for _ in range(num_tasks)])
        self._current_pos = np.zeros_like(samples)
        self._step_cnt = np.zeros(shape=num_tasks)
        self._is_done = np.zeros(shape=num_tasks)

        self._omega = rng.normal(scale=1 / lengthscale, size=(num_tasks, l, d))
        self._b = rng.uniform(0, 2 * np.pi, size=(num_tasks, l))
        self._w = rng.normal(size=(num_tasks, l))
        self._l = np.array([l])

    def reward(self, state: Iterable[Union[np.ndarray, float]]) -> np.ndarray:
        """Returns the reward of the current state in the current task."""

        def phi(x: Iterable[Union[np.ndarray, float]]) -> np.ndarray:
            return np.sqrt(2 / self._l) * np.cos(
                x @ self._omega[self.current_task].T + self._b[self.current_task]
            )

        return phi(state).dot(self._w[self.current_task])

    def print_task(self, task_index: int) -> None:
        """Print information about the task at index 'task_index'.

        Args:
            task_index (int): The index of the task to print
        """
        print(
            f"Task {task_index} has the following parameters:\n"
            f"\tomega = {self._omega[task_index]}\n"
            f"\tb = {self._b[task_index]}\n"
            f"\tw = {self._w[task_index]}\n"
            f"\tcurrent pos: {self._current_pos[task_index]}\n"
            f"\tstep count: {self._step_cnt[task_index]}\n"
            f"\tis done: {self._is_done[task_index]}\n"
        )

    def set_task(self, task_index: int, task: "PointTasksGP") -> None:
        """Replace the task at index 'task_index' with the given task.

        Args:
            task_index (int): The index of the task to replace
            task (Any): The new task
        """
        if task._l != self._l:
            raise ValueError(
                "Can't set new task because the '_l' variables differ!"
                f"Got self._l: {self._l}, task._l: {task._l}"
            )
        self._current_pos[task_index] = task.current_pos
        self._step_cnt[task_index] = task.step_cnt
        self._is_done[task_index] = task.is_done
        self._omega[task_index] = task._omega[0]
        self._b[task_index] = task._b[0]
        self._w[task_index] = task._w[0]


class PointEnvGPReward(EnvironmentBase):
    """A simple 2D point environment."""

    def __init__(
        self,
        rng: np.random.Generator,
        arena_size: float = 5.0,
        arena_dim: int = 2,
        num_tasks: int = 1,
        max_episode_length: int = 200,
        action_dist: float = 0.1,
        reward_model: Optional[RewardModelGPBase] = None,
        use_upper_conf_bound: bool = False,
        visualize: bool = False,
    ):
        """Initialize the environment.

        Args:
            rng (np.random.Generator): The random number generator.
            arena_size (float, optional): The length of the observation space. Defaults to 5.0.
            arena_dim (int, optional): The dimension of the observation space. Defaults to 2.
            num_tasks (int, optional): The number of tasks to initialize for the environment.
                Defaults to 1.
            max_episode_length (int, optional): The maximum number of steps in an episode.
            action_dist (float, optional): The maximum step size an agent can take in one step.
                Defaults to 0.1.
            reward_model (Optional[RewardModelGPBase], optional): The reward model to learn
                the reward function of the environment. Defaults to None.
            use_upper_conf_bound (bool, optional): Whether to use the upper confidence bound
                for the reward model. Defaults to False.
            visualize (bool, optional): Whether to visualize the environment. Defaults to False.
        """
        super(PointEnvGPReward, self).__init__(rng)

        self._arena_size = arena_size
        self._done_bonus = 0
        self._num_tasks = num_tasks
        self._max_episode_length = max_episode_length
        self._use_upper_conf_bound = use_upper_conf_bound
        self._visualize = visualize

        self._observation_space = spaces.Box(
            -arena_size,
            arena_size,
            shape=[
                arena_dim,
            ],
            seed=int(rng.integers(1 << 32 - 1)),
        )
        self._action_space = spaces.Box(
            -action_dist,
            action_dist,
            shape=[
                arena_dim,
            ],
            seed=int(rng.integers(1 << 32 - 1)),
        )

        # Used for compatibility with pacoh
        self._domain = ContinuousDomain(
            l=np.repeat(-arena_size, arena_dim), u=np.repeat(arena_size, arena_dim)
        )

        self._task: PointTasksGP = PointTasksGP(
            self._observation_space, self._rng, num_tasks=num_tasks
        )

        self._reward_model = reward_model

        # sample random gp reward function
        self.mean_fn_loc = np.array([-2, -2])
        self.mean_fn_std = 3.0
        self.d = 2

        # self._observation_space_old = akro.Box(
        #    low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        # )

    def _mean_fn(self, x: Iterable[Union[float, np.ndarray]]) -> Union[float, np.ndarray]:
        """A helpef function for the reward function. Was previously a lambda function,
        but these are not pickable."""
        return (
            100
            * (2 * np.pi * self.mean_fn_std**2) ** (-self.d / 2)
            * np.exp(-np.linalg.norm((x - self.mean_fn_loc) / self.mean_fn_std, axis=-1) ** 2 / 2)
        )

    def _reward(self, state: Iterable[Union[float, np.ndarray]]) -> Union[float, np.ndarray]:
        """Compute the reward of the environment based only on the current state

        Args:
            state (np.ndarray): The current state of the environment

        Returns:
            Union[float, np.ndarray]: The reward of the current state
        """
        return self._mean_fn(state) + self._task.reward(state)

    def reward(
        self,
        state: Optional[np.ndarray],
        action: Optional[np.ndarray],
        next_state: Iterable[Union[float, np.ndarray]],
    ) -> np.ndarray:
        """Compute the reward of the environment based on the current state, action and
        the next state

        Args:
            state (Optional[np.ndarray]): The current state of the environment
            action (Optional[np.ndarray]): The action which was taken in the current state
            next_state (np.ndarray): The next state of the environment

        Returns:
            np.ndarray: The reward of the current transition
        """
        if not self.use_true_reward:
            if self._reward_model is None:
                raise ValueError(
                    "You need to set the environment's reward model in order "
                    "to use it for generating reward!"
                )
            reward, _ = self._reward_model.predict(
                np.array(next_state), upper_conf_bound=self.use_upper_conf_bound
            )
        else:
            reward = np.array(self._reward(next_state))

        if self._noise is None:
            return reward

        return reward + self._noise(next_state)

    def render(self, mode: str) -> str:
        """Renders the environment.

        Args:
            mode (str): the mode to render with. The string must be present in
                `self.render_modes`.

        Returns:
            str: the point and goal of environment.
        """
        if mode not in self.render_modes:
            raise ValueError(f"'mode' must be in {self.render_modes}")
        if mode == "ascii":
            return f"Current position: {self._task.current_pos}"
        return ""

    def sample_tasks(
        self,
        num_tasks: int,
        lengthscale: float = 2.0,
        l: int = 20,
        d: int = 2,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> PointTasksGP:
        """Sample a set of tasks for the environment.

        Args:
            num_tasks (int): The number of tasks to sample.
            lengthscale (float, optional): The lengthscale of the GP. Defaults to 2.0.
            l (int, optional): Helper parameter for the GP. Defaults to 20.
            d (int, optional): Dimension of the GP. Defaults to 2.
            seed (Optional[int], optional): The seed to use for sampling the tasks.

        Returns:
            PointTasksGP: An object of type 'PointTasksGP' containing the sampled tasks.
        """
        if seed is not None:
            self.set_seed(seed)
        tasks = PointTasksGP(
            self._observation_space,
            self._rng,
            num_tasks=num_tasks,
            lengthscale=lengthscale,
            l=l,
            d=d,
        )
        return tasks

    def step(
        self, action: np.ndarray, normalize_obs: bool = True
    ) -> Tuple[np.ndarray, Union[float, np.ndarray], bool, dict]:
        """Perform a step in the environment

        Args:
            action (np.ndarray): The action which should be taken in the current state
            normalize_obs (bool, optional): Whether the observations should be
                normalized. Defaults to True.

        Returns:
            tuple[np.ndarray, Union[float, np.ndarray], bool, dict]: The next
                observation, the reward, whether the episode is done and a
                dictionary containing additional information
        """

        if self._task.is_done:
            raise RuntimeError("reset() must be called before step()!")

        # enforce action space
        a = action.copy()  # NOTE: we MUST copy the action before modifying it
        a = np.clip(a, self.action_space.low, self.action_space.high)

        self._task.set_current_pos(
            np.clip(
                self._task.current_pos + a,
                self.observation_space.low,
                self.observation_space.high,
            )
        )

        if self._visualize:
            print(self.render("ascii"))

        observation = self._task.current_pos.copy()

        # Normalize if necessary
        if normalize_obs:
            observation = self.normalize_observation(observation)

        # Get the reward for the current action
        reward = self.reward(state=None, action=None, next_state=self._task.current_pos)
        if isinstance(reward, np.ndarray) and len(reward.shape) == 1 and reward.shape[0] == 1:
            reward = reward[0]

        # Increase the step count of this task
        self._task.set_step_cnt(self._task.step_cnt + 1)

        # Determine if the task has timed out or is completed
        if self._task.step_cnt >= self._max_episode_length:
            self._task.set_is_done(done=True)
            done = True
        else:
            done = False

        return (observation, reward, done, {})


if __name__ == "__main__":
    resolution = 100
    env = PointEnvGPReward(rng=np.random.default_rng(42))

    x = np.meshgrid(np.linspace(-5, 5, resolution), np.linspace(-5, 5, resolution))
    states = np.stack([x[0].flatten(), x[1].flatten()], axis=-1)
    rewards = env._reward(states)
    assert isinstance(rewards, np.ndarray)
    rewards_map = rewards.reshape(resolution, resolution)

    # plot rollouts
    from matplotlib import pyplot as plt

    c = plt.pcolormesh(x[0], x[1], rewards_map, shading="auto", alpha=0.3)
    plt.colorbar(
        c,
    )
    plt.show()
