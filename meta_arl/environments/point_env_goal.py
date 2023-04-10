import time
from typing import Any, Iterable, Optional, Union

import numpy as np
from gym import spaces

from meta_arl.environments.abstract import EnvironmentBase, PointTasksBase
from meta_arl.meta_bo.domain import ContinuousDomain
from meta_arl.reward_models.abstract import RewardModelGPBase


class PointTasks(PointTasksBase):
    def __init__(
        self, space: spaces.Space, target_pos: Optional[np.ndarray] = None, num_tasks: int = 1
    ):
        """Initialize 'num_tasks' tasks for the PointEnv environment.

        Args:
            space (spaces.Space): The observation space of the environment.
            target_pos (Optional[np.ndarray], optional): The target position of the tasks.
                Defaults to None.
            num_tasks (int, optional): The number of tasks. Defaults to 1.

        Raises:
            ValueError: If the target position has a wrong shape.
        """
        self._current_task = 0
        self.num_tasks = num_tasks
        samples = np.array([space.sample() for _ in range(num_tasks)])
        self._current_pos = np.zeros_like(samples)
        self._step_cnt = np.zeros(shape=num_tasks)
        self._is_done = np.zeros(shape=num_tasks)
        self._target_pos: np.ndarray[np.ndarray, np.dtype[np.float64]]

        if target_pos is None:
            self._target_pos = np.array([space.sample() for _ in range(num_tasks)])
        else:
            sample = space.sample()
            if target_pos.shape == sample.shape:
                self._target_pos = np.array([target_pos for _ in range(num_tasks)])
            elif (
                len(target_pos.shape) == 2
                and target_pos.shape[1:] == sample.shape
                and len(target_pos) == num_tasks
            ):
                self._target_pos = target_pos
            else:
                raise ValueError("Could not assign target position!")

    @property
    def target_pos(self) -> np.ndarray:
        """Returns the target position of the current task."""
        return self._target_pos[self.current_task]

    def reward(self, state: Iterable[Union[np.ndarray, float]]) -> np.ndarray:
        """Returns the reward of the current state in the current task."""
        return -np.linalg.norm(state - self.target_pos, axis=-1)

    def print_task(self, task_index: int) -> None:
        """Print information about the task at index 'task_index'.

        Args:
            task_index (int): The index of the task to print
        """
        print(
            f"Task {task_index} has the following parameters:\n"
            f"\ttarget pos = {self._target_pos[task_index]}\n"
            f"\tcurrent pos: {self._current_pos[task_index]}\n"
            f"\tstep count: {self._step_cnt[task_index]}\n"
            f"\tis done: {self._is_done[task_index]}\n"
        )

    def set_task(self, task_index: int, task: "PointTasks") -> None:
        """Replace the task at index 'task_index' with the given task.

        Args:
            task_index (int): The index of the task to replace
            task (Any): The new task
        """
        self._current_pos[task_index] = task.current_pos
        self._step_cnt[task_index] = task.step_cnt
        self._is_done[task_index] = task.is_done
        self._target_pos[task_index] = task.target_pos


class PointEnv(EnvironmentBase):
    """A simple 2D point environment."""

    def __init__(
        self,
        rng: np.random.Generator,
        arena_size: float = 5.0,
        arena_dim: int = 2,
        done_bonus: int = 0,
        num_tasks: int = 1,
        max_episode_length: int = 200,
        action_dist: float = 0.1,
        target_pos: Optional[np.ndarray] = None,
        reward_model: Optional[RewardModelGPBase] = None,
        use_upper_conf_bound: bool = False,
        visualize: bool = False,
    ):
        """Initialize the PointEnv environment.

        Args:
            rng (np.random.Generator): The random number generator to use.
            arena_size (float, optional): The length of the observation space.
                Defaults to 5.0.
            arena_dim (int, optional): The dimension of the observation space.
                Defaults to 2.
            done_bonus (int, optional): The bonus for reaching the goal. Defaults to 0.
            num_tasks (int, optional): The number of tasks. Defaults to 1.
            max_episode_length (int, optional): The maximum length of an episode.
                Defaults to 200.
            action_dist (float, optional): The maximum stepsize. Defaults to 0.1.
            target_pos (Optional[np.ndarray], optional): The target position of the tasks.
                Defaults to None.
            reward_model (Optional[RewardModelGPBase], optional): The reward model to learn
                the reward function of the environment. Defaults to None.
            use_upper_conf_bound (bool, optional): Whether to use the upper confidence bound
                for the reward model. Defaults to False.
            visualize (bool, optional): Whether to visualize the environment. Defaults to False.
        """
        super(PointEnv, self).__init__(rng)

        # DEBUG START
        self._reward_times: list[float] = []
        # DEBUG END

        self._arena_size = arena_size
        self._arena_dim = arena_dim
        self._done_bonus = done_bonus
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

        self._task: PointTasks = self.sample_tasks(num_tasks=num_tasks, target_pos=target_pos)

        self._reward_model = reward_model

    def _reward(self, state: Iterable[Union[float, np.ndarray]]) -> np.ndarray:
        """Compute the reward of the environment based only on the current state

        Args:
            state (np.ndarray): The current state of the environment

        Returns:
            Union[float, np.ndarray]: The reward of the current state
        """
        return self._task.reward(state)

    def reward(
        self,
        state: Optional[np.ndarray],
        action: Optional[np.ndarray],
        next_state: Iterable[np.ndarray],
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
            return (
                f"Current position: {self._task.current_pos}, "
                f"Target position: {self._task.target_pos}, "
                f"Distance: {-self._reward(self._task.current_pos)}"
            )
        return ""

    def sample_tasks(
        self,
        num_tasks: int,
        target_pos: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> PointTasks:
        """Sample a set of tasks for the environment.

        Args:
            num_tasks (int): The number of tasks to sample.
            target_pos (Optional[np.ndarray], optional): The target position of the tasks.
            seed (Optional[int], optional): The seed to use for sampling the tasks.

        Returns:
            PointTasks: An object of type 'PointTasks' containing the sampled tasks.
        """
        if seed is not None:
            self.set_seed(seed)
        tasks = PointTasks(
            self._observation_space,
            num_tasks=num_tasks,
            target_pos=target_pos,
        )
        return tasks

    def step(
        self, action: np.ndarray, normalize_obs: bool = True
    ) -> tuple[np.ndarray, Union[float, np.ndarray], bool, dict]:
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

        # DEBUG START
        reward_start = time.time()
        # DEBUG END

        # Get the reward for the current action
        reward = self.reward(state=None, action=None, next_state=self._task.current_pos)
        if isinstance(reward, np.ndarray) and len(reward.shape) == 1 and reward.shape[0] == 1:
            reward = reward[0]

        # DEBUG START
        reward_end = time.time()
        self._reward_times.append(reward_end - reward_start)
        # DEBUG END

        # Increase the step count of this task
        self._task.set_step_cnt(self._task.step_cnt + 1)

        # Determine if the task has timed out or is completed
        if self._task.step_cnt >= self._max_episode_length:
            self._task.set_is_done(done=True)
            done = True
        else:
            done = False

        return (observation, reward, done, {})

    def get_average_times(self) -> tuple[float, list[float]]:
        """Get the average time it takes to compute the reward of the environment

        Returns:
            tuple[float, list[float]]: The average time it takes to compute the reward
                of the environment and a list of times for each reward computation
        """
        reward_time_avg = np.mean(self._reward_times)

        return reward_time_avg, self._reward_times

    def reset_step_counters(self) -> None:
        """Reset the list of times it takes to compute the reward of the environment"""
        del self._reward_times
        self._reward_times = []


if __name__ == "__main__":
    resolution = 100
    rng = np.random.default_rng()
    env = PointEnv(rng=rng)

    x = np.meshgrid(np.linspace(-5, 5, resolution), np.linspace(-5, 5, resolution))
    states = np.stack([x[0].flatten(), x[1].flatten()], axis=-1)
    rewards = env._reward(states)
    assert isinstance(rewards, np.ndarray)
    rewards_map = rewards.reshape(resolution, resolution)

    # plot rollouts
    from matplotlib import pyplot as plt

    c = plt.pcolormesh(x[0], x[1], rewards_map, shading="auto", alpha=0.6)
    plt.colorbar(
        c,
    )
    plt.show()
