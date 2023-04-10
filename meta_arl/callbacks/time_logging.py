import time
from typing import Optional

from stable_baselines3.common.callbacks import BaseCallback


class TimeLoggingCallback(BaseCallback):
    """
    A custom callback which measures the time of various different parts of
    the rl training process.
    """

    def __init__(self, verbose: int = 0) -> None:
        """Initialize the callback.

        Args:
            verbose (int, optional): Verbosity level. Defaults to 0.
        """
        super(TimeLoggingCallback, self).__init__(verbose)

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

        self.training_start: Optional[str] = None
        self.training_end: Optional[str] = None

        self.collect_rollouts_sum: float = 0
        self.collect_rollouts_start: Optional[float] = None
        self.collect_rollouts_end: Optional[float] = None
        self.num_rollout_collections: float = 0

        self.old_step_time: Optional[float] = None
        self.step_time_sum: float = 0
        self.num_steps: float = 0

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self.training_start = time.ctime()

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        self.collect_rollouts_start = time.time()

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        t = time.time()
        if self.old_step_time is None:
            self.old_step_time = t
        else:
            assert self.old_step_time is not None
            self.step_time_sum += t - self.old_step_time
            self.old_step_time = t
            self.num_steps += 1

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        assert self.collect_rollouts_start is not None
        self.collect_rollouts_end = time.time()
        self.collect_rollouts_sum += self.collect_rollouts_end - self.collect_rollouts_start
        self.num_rollout_collections += 1

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        self.training_end = time.ctime()

    def print_results(self) -> None:
        """Print the results of the time logging."""
        print("[TIME] training start: ", self.training_start)
        print("[TIME] training ends: ", self.training_end)
        print(
            "[TIME++] average rollout collection time: ",
            self.collect_rollouts_sum / self.num_rollout_collections,
        )
        print(
            "[TIME++] average environment step time: ",
            self.step_time_sum / self.num_steps,
        )
