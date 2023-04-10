import logging
import os
import warnings
from typing import Any, Callable, Iterable, Optional, Union

import numpy as np
import torch
from absl import flags
from numpy.typing import NDArray


def find_root_by_bounding(
    fun: Callable,
    left: torch.Tensor,
    right: torch.Tensor,
    eps: float = 1e-6,
    max_iter: float = 1e4,
) -> torch.Tensor:
    """
    Root finding method that uses selective shrinking of a target interval bounded by left and right
    --> other than the newton method, this method only works for for vectorized univariate functions
    Args:
        fun (callable): function f for which f(x) = 0 shall be solved
        left: (torch.Tensor): initial left bound
        right (torch.Tensor): initial right bound
        eps (float): tolerance
        max_iter (int): maximum iterations
    """

    assert callable(fun)

    n_iter = 0
    approx_error: Union[float, torch.Tensor] = 1e12
    while approx_error > eps:
        middle = (right + left) / 2
        f = fun(middle)

        left_of_zero = (f < 0).flatten()
        left[left_of_zero] = middle[left_of_zero]
        right[~left_of_zero] = middle[~left_of_zero]

        assert torch.all(left <= right).item()

        approx_error = torch.max(torch.abs(right - left)) / 2
        n_iter += 1

        if n_iter > max_iter:
            warnings.warn(
                "Max_iter has been reached - stopping newton method for determining quantiles"
            )
            return torch.Tensor([np.nan for _ in range(len(left))])

    return middle


def _handle_input_dimensionality(
    x: np.ndarray, y: Optional[np.ndarray] = None
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    if x.ndim == 1:
        # LUKAS FIX
        # x = np.expand_dims(x, -1)
        x = np.expand_dims(x, axis=0)

    assert x.ndim == 2

    if y is not None:
        if y.ndim == 1:
            y = np.expand_dims(y, -1)
        assert x.shape[0] == y.shape[0]
        assert y.ndim == 2

        return x, y
    else:
        return x


def get_logger(
    log_dir: Optional[str] = None, log_file: str = "output.log", expname: str = ""
) -> logging.Logger:

    if log_dir is None and flags.FLAGS.is_parsed() and hasattr(flags.FLAGS, "log_dir"):
        log_dir = flags.FLAGS.log_dir

    logger = logging.getLogger("gp-priors")
    logger.setLevel(logging.INFO)

    if len(logger.handlers) == 0:

        # formatting
        if len(expname) > 0:
            expname = " %s - " % expname
        formatter = logging.Formatter(
            "[%(asctime)s -" + "%s" % expname + "%(levelname)s]  %(message)s"
        )

        # Stream Handler
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        sh.setLevel(logging.INFO)
        logger.addHandler(sh)

        logger.propagate = False

        # File Handler
        if log_dir is not None and len(log_dir) > 0:
            fh = logging.FileHandler(os.path.join(log_dir, log_file))
            fh.setFormatter(formatter)
            fh.setLevel(logging.INFO)
            logger.addHandler(fh)
            logger.log_dir = log_dir  # type: ignore
        else:
            logger.log_dir = None  # type: ignore
    return logger


class DummyLRScheduler:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def step(self, *args: Any, **kwargs: Any) -> None:
        pass


""" ------ Lightweight mltiprocessing utilities ------ """

import multiprocessing
from multiprocessing import Process

import numpy as np


class AsyncExecutor:
    def __init__(self, n_jobs: int = 1):
        self.num_workers = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()
        self._pool: list[Process] = []
        self._populate_pool()

    def run(self, target: Callable, *args_iter: Iterable, verbose: bool = False) -> None:
        workers_idle = [False] * self.num_workers
        tasks = list(zip(*args_iter))
        n_tasks = len(tasks)

        while not all(workers_idle):
            for i in range(self.num_workers):
                if not self._pool[i].is_alive():
                    self._pool[i].terminate()
                    if len(tasks) > 0:
                        if verbose:
                            print("task %i of %i" % (n_tasks - len(tasks), n_tasks))
                        next_task = tasks.pop(0)
                        self._pool[i] = _start_process(target, next_task)
                    else:
                        workers_idle[i] = True

    def _populate_pool(self) -> None:
        self._pool = [_start_process(_dummy_fun) for _ in range(self.num_workers)]


class LoopExecutor:
    def run(self, target: Callable, *args_iter: Iterable, verbose: bool = False) -> None:
        tasks = list(zip(*args_iter))
        n_tasks = len(tasks)

        for i, task in enumerate(tasks):
            target(*task)
            if verbose:
                print("task %i of %i" % (n_tasks - len(tasks), n_tasks))


def _start_process(target: Callable, args: Optional[tuple] = None) -> Process:
    if args:
        p = Process(target=target, args=args)
    else:
        p = Process(target=target)
    p.start()
    return p


def _dummy_fun() -> None:
    pass
