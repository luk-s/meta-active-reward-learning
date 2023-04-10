from typing import Any, Iterable, Optional, Tuple, Union, cast

import gym
import numpy as np
from gym import spaces

from meta_arl.environments.abstract import EnvironmentBase, PointTasksBase
from meta_arl.meta_bo.domain import ContinuousDomain
from meta_arl.reward_models.abstract import RewardModelGPBase


class PointTasksTile(PointTasksBase):
    def __init__(
        self,
        space: gym.spaces.Space,
        rng: np.random.Generator,
        num_tasks: int = 1,
        num_rows: int = 5,
        num_cols: int = 5,
        min_reward: float = -3,
        max_reward: float = 3,
    ) -> None:
        """Initialize 'num_tasks' tasks for the PointEnvTile environment.

        Args:
            space (gym.spaces.Space): The observation space of the environment.
            rng (np.random.Generator): The random number generator.
            num_tasks (int, optional): The number of tasks. Defaults to 1.
            num_rows (int, optional): The number of rows in the tiled grid. Defaults to 5.
            num_cols (int, optional): The number of columns in the tiled grid. Defaults to 5.
            min_reward (float, optional): The minimum reward in the tiled environment.
                Defaults to -3.
            max_reward (float, optional): The maximum reward in the tiled environment.
                Defaults to 3.
        """
        self._current_task = 0
        self.num_tasks = num_tasks
        samples = np.array([space.sample() for _ in range(num_tasks)])
        self._current_pos = np.zeros_like(samples)
        self._step_cnt = np.zeros(shape=num_tasks)
        self._is_done = np.zeros(shape=num_tasks)

        self.num_rows = num_rows
        self.num_cols = num_cols
        self._reward_tiles = rng.uniform(
            min_reward, max_reward, size=(num_tasks, num_rows, num_cols)
        )
        self.box_min = space.low
        self.box_max = space.high
        self.row_length, self.col_length = (self.box_max - self.box_min) / (
            self.num_rows,
            self.num_cols,
        )

    def reward(self, state: Iterable[Union[np.ndarray, float]]) -> Union[float, np.ndarray]:
        """Returns the reward of the current state in the current task."""
        # Identify into which tile the input 'state' falls
        position = np.array(state) - self.box_min
        tile_dims = np.array([[self.row_length, self.col_length]])
        indices = (position - (position % tile_dims)) / tile_dims
        row_indices, col_indices = indices[:, 0], indices[:, 1]

        # Fix a small inaccuracy. If 'state' is actually at the upper limit of the
        # box, the above calculation returns 'row_index' and 'col_index' which are
        # too large by 1
        row_indices[row_indices == self.num_rows] = self.num_rows - 1
        col_indices[col_indices == self.num_cols] = self.num_cols - 1

        # Return the reward of the specified file
        indices = np.vstack([row_indices, col_indices]).astype(int).T
        rewards = self._reward_tiles[self._current_task, indices[:, 0], indices[:, 1]]
        if len(rewards) == 1:
            return rewards[0]
        return rewards

    def print_task(self, task_index: int) -> None:
        """Print information about the task at index 'task_index'.

        Args:
            task_index (int): The index of the task to print
        """
        print(
            f"Task {task_index} has the following parameters:\n"
            f"\t(num_rows, num_cols) = {self.num_rows, self.num_cols}\n"
            f"\tReward tiles = {self._reward_tiles[task_index]}\n"
            f"\tcurrent pos: {self._current_pos[task_index]}\n"
            f"\tstep count: {self._step_cnt[task_index]}\n"
            f"\tis done: {self._is_done[task_index]}\n"
        )

    def set_task(self, task_index: int, task: "PointTasksTile") -> None:
        """Replace the task at index 'task_index' with the given task.

        Args:
            task_index (int): The index of the task to replace
            task (Any): The new task
        """
        same_attributes = ["num_rows", "num_cols", "box_min", "box_max"]
        is_array = [False, False, True, True]
        for check_all, attr in zip(is_array, same_attributes):
            if (check_all and (getattr(self, attr) != getattr(task, attr)).all()) or (
                not check_all and getattr(self, attr) != getattr(task, attr)
            ):
                raise ValueError(f"The new task must have the same '{attr}' as the current one!")
        self.num_rows = task.num_rows
        self.num_cols = task.num_cols
        self.box_min = task.box_min
        self.box_max = task.box_max
        self.row_length = task.row_length
        self.col_length = task.col_length
        self._current_pos[task_index] = task.current_pos
        self._step_cnt[task_index] = task.step_cnt
        self._is_done[task_index] = task.is_done
        self._reward_tiles[task_index] = task._reward_tiles


class PointEnvTileReward(EnvironmentBase):
    """A 2d environment where the reward function has the form of
    a tiled grid.
    ."""

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
        """Initialize the PointEnvTileReward environment.

        Args:
            rng (np.random.Generator): The random number generator.
            arena_size (float, optional): The size of the observation space. Defaults to 5.0.
            arena_dim (int, optional): The dimension of the observation space. Defaults to 2.
            num_tasks (int, optional): The number of tasks. Defaults to 1.
            max_episode_length (int, optional): The maximum number of steps in an episode.
                Defaults to 200.
            action_dist (float, optional): The maximum distance the agent can move in one step.
            reward_model (Optional[RewardModelGPBase], optional): The reward model to use.
            use_upper_conf_bound (bool, optional): Whether to use the upper confidence bound
                instead of the mean when using a reward model. Defaults to False.
            visualize (bool, optional): Whether to visualize the environment. Defaults to False.
        """
        super(PointEnvTileReward, self).__init__(rng=rng)

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

        self._task: PointTasksTile = PointTasksTile(
            self._observation_space, self._rng, num_tasks=num_tasks
        )

        self._reward_model = reward_model

    def _reward(self, state: Iterable[Union[float, np.ndarray]]) -> Union[float, np.ndarray]:
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
            assert next_state is not None
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
        num_rows: int = 5,
        num_cols: int = 5,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> PointTasksTile:
        """Sample a set of tasks for the environment.

        Args:
            num_tasks (int): The number of tasks to sample.
            num_rows (int, optional): The number of rows in the tiled grid. Defaults to 5.
            num_cols (int, optional): The number of columns in the tiled grid. Defaults to 5.
            seed (Optional[int], optional): The seed to use for sampling the tasks.

        Returns:
            PointTasksTile: An object of type 'PointTasksTile' containing the sampled tasks.
        """

        if seed is not None:
            self.set_seed(seed)
        tasks = PointTasksTile(
            self._observation_space,
            self._rng,
            num_tasks=num_tasks,
            num_rows=num_rows,
            num_cols=num_cols,
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
    seed = 3
    env = PointEnvTileReward(rng=np.random.default_rng(seed))
    env.set_task(0, env.sample_tasks(num_tasks=1, seed=seed))

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
