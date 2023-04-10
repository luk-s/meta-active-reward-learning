import random
import sys
import warnings

import numpy as np


def set_all_seeds(seed: int = 0) -> None:
    """
    Set the process-wide random seed. This method was taken from the garage rl library.
    See: https://github.com/rlworkgroup/garage for the library and
    https://garage.readthedocs.io/en/latest/_modules/garage/experiment/deterministic.html?
    for the original source of this method

    Args:
        seed (int): A positive integer
    """
    seed %= 4294967294
    # pylint: disable=global-statement
    random.seed(seed)
    np.random.seed(seed)
    if "torch" in sys.modules:
        warnings.warn(
            "Enabeling deterministic mode in PyTorch can have a performance "
            "impact when using GPU."
        )
        import torch  # pylint: disable=import-outside-toplevel

        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
