from typing import Optional, get_type_hints

import numpy as np
import torch

from meta_arl.meta_bo.models.f_pacoh_map import FPACOH_MAP_GP
from meta_arl.meta_bo.models.models import AffineTransformedDistribution
from meta_arl.reward_models.abstract import RewardModelGPBase


class RewardModelPacohGP(RewardModelGPBase):
    """GP reward model that iteratively selects samples and allows meta training."""

    def _vectorize_pred_dist(
        self, pred_dist: AffineTransformedDistribution
    ) -> torch.distributions.Normal:
        """Convert the affine distribution into a normal distribution

        Args:
            pred_dist (AffineTransformedDistribution): The affine distribution to convert

        Returns:
            torch.distributions.Normal: The converted distribution
        """
        return torch.distributions.Normal(pred_dist.mean, pred_dist.stddev)

    def eval(self, test_x: np.ndarray, test_y: np.ndarray) -> tuple[float, float, float, float]:
        """Evaluate the reward model on the provided test data

        Args:
            test_x (np.ndarray): The test observations
            test_y (np.ndarray): The test rewards

        Raises:
            Exception: If the reward model has not yet been initialized with meta data

        Returns:
            tuple[float, float, float, float]: The log likelihood, the residual mean squared
                error, the calibration error and the calibration error chi squared.
        """
        if not hasattr(self, "model") or self.model is None:
            raise Exception("The reward model was not yet initialized with meta data!")
        test_x, test_y = self._handle_input_dim(test_x, test_y)
        ll, rmse, calib_err, calib_err_chi2 = self.model.eval(test_x, test_y)
        return ll, rmse, calib_err, calib_err_chi2

    def initialize(
        self, context_x: np.ndarray, context_y: np.ndarray, num_init_samples: int = 2
    ) -> None:
        """Initialize the reward model with a set of gathered observations and reward values

        Args:
            context_x (np.ndarray): A set of gathered observations from the environment
            context_y (np.ndarray): A set of gathered (true) reward values.
            num_init_samples (int, optional): The number of initial samples of the reward model.
                Defaults to 2.

        Raises:
            ValueError: If the number of initial samples is < 2
        """
        # Check that 'num_init_samples' >= 2
        if num_init_samples < 2:
            raise ValueError("'num_init_samples' must be larger or equal than 2!")
        self.initialized = True
        self.context_x = np.array(context_x)
        self.context_y = np.array(context_y)
        self.selected_samples_idx = []

        # initialize selected_samples with 'num_init_samples' random samples
        if num_init_samples > 0:
            idx = self._rng.choice(
                list(range(self.context_size)), size=num_init_samples, replace=False
            )
            self.selected_samples_idx = list(idx)

        # Initialize the model
        if self.model is None:
            self.model = self.initialize_model()

        # Add the initial data
        x, y = (
            self.context_x[self.selected_samples_idx],
            self.context_y[self.selected_samples_idx],
        )
        x, y = self._handle_input_dim(x, y)
        self.model.add_data(x, y)

        # Clip the context to the maximum context size
        self._clip_context()

    def initialize_meta_data(
        self,
        meta_train_data: list[tuple[np.ndarray, np.ndarray]],
        meta_valid_data: Optional[
            list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
        ] = None,
    ) -> None:
        """Meta train the reward model with the given training and valudation tuples

        Args:
            meta_train_data (list[tuple[np.ndarray, np.ndarray]]): A list of meta-training tuples
            meta_valid_data (Optional[
                list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
            ], optional): A list of meta-validation tuples. Defaults to None.
        """
        # Initialize the model
        self.model = self.initialize_model()
        assert isinstance(self.model, FPACOH_MAP_GP)

        self.model.meta_fit(meta_train_data, meta_valid_data, log_period=100)

        if self.selected_samples_idx is not None:
            assert self.context_x is not None
            assert self.context_y is not None
            # Add the initial data
            x, y = (
                self.context_x[self.selected_samples_idx],
                self.context_y[self.selected_samples_idx],
            )
            x, y = self._handle_input_dim(x, y)
            self.model.add_data(x, y)

    def meta_predict(
        self, pred_x: np.ndarray, max_size: int = 10000, upper_conf_bound: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Meta-train the reward model and predict the reward for the given observations

        Args:
            pred_x (np.ndarray): The observations to predict the reward for
            max_size (int, optional): The batch size to use for the meta prediction.
                Defaults to 10000.
            upper_conf_bound (bool, optional): Whether to return the upper confidence bound
                instead of the mean. Defaults to False.

        Returns:
            tuple[np.ndarray, np.ndarray]: The predicted mean reward and the predicted
                standard deviation of the reward
        """
        assert self.initialized, "Model is not initialized."
        assert isinstance(self.model, FPACOH_MAP_GP)
        assert self.context_x is not None
        assert self.context_y is not None

        pred_x = np.array(self._handle_input_dim(pred_x))

        # Split pred_x up into individual pieces to save memory if necessary
        pred_means, pred_stds = [], []

        # Split pred_x up into pieces of equal size
        pred_x_array = np.split(pred_x, np.arange(max_size, len(pred_x), max_size))

        # Make predictions for each individual piece
        for pred_x_part in pred_x_array:
            result = self.model.meta_predict(
                self.context_x[self.selected_samples_idx],
                self.context_y[self.selected_samples_idx],
                pred_x_part,
            )
            assert isinstance(result, tuple)
            pred_mean, pred_std = result
            pred_means.append(pred_mean)
            pred_stds.append(pred_std)

        if len(pred_means) > 1:
            pred_mean = np.concatenate(pred_means)
            pred_std = np.concatenate(pred_stds)
        else:
            pred_mean = pred_means[0]
            pred_std = pred_stds[0]

        if upper_conf_bound:
            return pred_mean + 2 * pred_std, pred_std
        else:
            return pred_mean, pred_std

    def observe(self, idx: int, train_model: bool = False) -> None:
        """Add a new observation to the reward model

        Args:
            idx (int): The index of the observation to add to the reward model
            train_model (bool, optional): Whether to train the model after adding the new
                observation. Defaults to False.
        """
        assert 0 <= idx < self.context_size, f"Index {idx} not in dataset."
        assert self.selected_samples_idx is not None
        assert self.context_x is not None
        assert self.context_y is not None
        # assert idx not in self.selected_samples_idx, f"Already observed {idx}."
        self.selected_samples_idx.append(idx)

        if train_model:
            assert self.model is not None
            x, y = (
                self.context_x[self.selected_samples_idx],
                self.context_y[self.selected_samples_idx],
            )
            x, y = self._handle_input_dim(x, y)
            self.model.add_data(x, y)

    def predict(
        self, pred_x: np.ndarray, max_size: int = 10000, upper_conf_bound: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict the reward for the given observations

        Args:
            pred_x (np.ndarray): The observations to predict the reward for
            max_size (int, optional): The batch size to use for the prediction.
            upper_conf_bound (bool, optional): Whether to return the upper confidence bound
                instead of the mean. Defaults to False.

        Returns:
            tuple[np.ndarray, np.ndarray]: The predicted mean reward and the predicted
                standard deviation of the reward
        """
        assert self.initialized, "Model is not initialized."
        assert self.model is not None

        pred_x = np.array(self._handle_input_dim(pred_x))

        # Split pred_x up into individual pieces to save memory if necessary
        pred_means, pred_stds = [], []

        # Split pred_x up into pieces of equal size
        pred_x_array = np.split(pred_x, np.arange(max_size, len(pred_x), max_size))

        # Make predictions for each individual piece
        for pred_x_part in pred_x_array:
            result = self.model.predict(
                pred_x_part,
            )
            assert isinstance(result, tuple)
            pred_mean, pred_std = result
            pred_means.append(pred_mean)
            pred_stds.append(pred_std)

        if len(pred_means) > 1:
            pred_mean = np.concatenate(pred_means)
            pred_std = np.concatenate(pred_stds)
        else:
            pred_mean = pred_means[0]
            pred_std = pred_stds[0]

        if upper_conf_bound:
            return pred_mean + 2 * pred_std, pred_std
        else:
            return pred_mean, pred_std
