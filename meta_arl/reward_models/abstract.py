from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

import numpy as np
import torch

from meta_arl.meta_bo.domain import ContinuousDomain

# from meta_arl.environments.abstract import EnvironmentBase
from meta_arl.meta_bo.models import FPACOH_MAP_GP, GPRegressionVanilla
from meta_arl.meta_bo.models.models import AffineTransformedDistribution

if TYPE_CHECKING:
    from meta_arl.environments.abstract import EnvironmentBase


class RewardModelGPBase:
    """A base class for all reward models which use a GP reward model"""

    observation_sampling_modes = ["random", "largest_uncertainty"]

    def __init__(
        self,
        model_name: str,
        config: dict,
        rng: np.random.Generator,
        environment: Optional["EnvironmentBase"] = None,
        max_context_size: int = 10000,
    ):
        """Initialize the reward model

        Args:
            model_name (str): The name of the reward model class to use
            config (dict): The configuration for the reward model
            rng (np.random.Generator): The random number generator to use
            environment (Optional[EnvironmentBase], optional): The environment whose
                reward should be modeled. Defaults to None.
            max_context_size (int, optional): The maximum number of observations to
                store. Defaults to 10000.
        """
        self.model_name: str = model_name
        self.model: Optional[Union[GPRegressionVanilla, FPACOH_MAP_GP]] = None
        self.config: dict = config
        self.context_x: Optional[np.ndarray] = None
        self.context_y: Optional[np.ndarray] = None
        self.selected_samples_idx: Optional[list[int]] = None
        self.initialized: bool = False
        # self.environment: Optional[EnvironmentBase] = environment
        self.environment = environment
        self.max_context_size: int = max_context_size
        assert rng is not None, "You need to provide a random state!"
        self._rng: np.random.Generator = rng
        self.config["random_state"] = self._rng

    @property
    def context_size(self) -> int:
        """Returns the number of observations in the context"""
        if self.context_x is None:
            raise TypeError("The variable 'self.context_x' has not yet been set!")
        return self.context_x.shape[0]

    def _clip_context(self) -> None:
        """
        This method makes sure that the stored contexts don't grow too large. If the
        current context size is larger than the maximum conxt size, keep a mix of
        the observations with the highest uncertainty and randomly sampled observations
        """
        if self.context_size <= self.max_context_size:
            return

        new_indices: list[int]

        # Add all examples which have already been observed to the new context
        if self.selected_samples_idx is not None:
            new_indices = list(self.selected_samples_idx)
        else:
            new_indices = []

        assert self.context_x is not None
        assert self.context_y is not None
        assert self.selected_samples_idx is not None

        # Adding only the elements for which the model is most uncertain does only
        # work well if the context size is not too much larger than the maximum size
        if self.context_size <= 2 * self.max_context_size:

            # Compute the uncertainty of each sample in the current context
            _, pred_std = self.predict(self.context_x)

            # We are interested in the largest uncertainty. Because the partition below
            # uses a min-sort we will negate the uncertainties here to make partitioning
            # easier
            neg_pred_std = np.negative(pred_std)

            # Get the indices of the 'self.max_context_size' smallest elements
            # We use np.argpartition instead of np.argsort because its faster
            indices = np.argpartition(neg_pred_std, self.max_context_size - 1)[
                : self.max_context_size
            ]

            # Filter out potential elements from 'self.selected_samples_idx'
            indices = [i for i in indices if i not in new_indices]

            # Fill the context with the most uncertain examples which haven't
            # already been observed
            new_indices += indices[: self.max_context_size - len(new_indices)]

        # If the context size is much larger than the maximum possible size,
        # just add elements randomly, to ensure a more even spread of the
        # newly added elements.
        else:
            print("DEBUG: Adding elements randomly to new context!")
            # Select randomly the maximum possible amount of elements from the context
            indices = self._rng.choice(
                list(range(self.context_size)), self.max_context_size, replace=False
            )

            # Filter out all elements which are already in 'new_indices'
            indices = [element for element in indices if element not in new_indices]

            # Add as many elements as allowed to 'new_indices'
            new_indices += indices[: self.max_context_size - len(new_indices)]

        # Create the new contexts
        self.context_x = self.context_x[new_indices]
        self.context_y = self.context_y[new_indices]
        self.selected_samples_idx = list(range(len(self.selected_samples_idx)))

    def _handle_input_dim(
        self, x: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Makes sure that the observations and rewards have the correct shape

        Args:
            x (np.ndarray): The observations
            y (Optional[np.ndarray], optional): Optional rewards. Defaults to None.

        Raises:
            AssertionError: If the rewards are not 1-dimensional

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: The corrected observations
                and rewards
        """
        if x.ndim == 1:
            assert self.context_x is not None
            assert x.shape[-1] == len(self.context_x[0])
            x = x.reshape((-1, x.shape[-1]))

        if y is not None:
            if isinstance(y, float) or y.ndim == 0:
                y = np.array(y)
                y = y.reshape((1,))
            elif y.ndim == 1:
                pass
            else:
                raise AssertionError("y must not have more than 1 dim")
            return x, y

        return x

    def _vectorize_pred_dist(
        self, pred_dist: AffineTransformedDistribution
    ) -> torch.distributions.Normal:
        """Converts a distribution to a torch.distributions.Normal distribution

        Args:
            pred_dist (AffineTransformedDistribution): The distribution to convert

        Raises:
            NotImplementedError: This function must be implemented by a subclass

        Returns:
            torch.distributions.Normal: The converted distribution
        """
        raise NotImplementedError

    def confidence_intervals(
        self, test_x: np.ndarray, confidence: float = 0.9, **kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return lower and upper confidence bounds for the predictions
        of the observations in 'test_x'

        Args:
            test_x (np.ndarray): The observations for which the confidence intervals
                should be computed
            confidence (float, optional): The confidence level. Defaults to 0.9.

        Returns:
            tuple[np.ndarray, np.ndarray]: The lower and upper confidence bounds
        """
        assert self.model is not None
        handled = self._handle_input_dim(test_x)
        assert isinstance(handled, np.ndarray)
        test_x = handled
        confidence_np = np.array(confidence)
        pred_dist = self.model.predict(test_x, return_density=True, **kwargs)
        assert isinstance(pred_dist, AffineTransformedDistribution)
        pred_dist_vec = self._vectorize_pred_dist(pred_dist)

        alpha = (1 - confidence_np) / 2

        if len(alpha) == 1:
            ucb = pred_dist_vec.icdf(torch.ones(test_x.shape[0]) * (1 - alpha))
            lcb = pred_dist_vec.icdf(torch.ones(test_x.shape[0]) * alpha)
        elif len(alpha) > 1:
            lcbs, ucbs = [], []
            for alpha_val in alpha:
                ucbs.append(pred_dist_vec.icdf(torch.ones(test_x.shape[0]) * (1 - alpha_val)))
                lcbs.append(pred_dist_vec.icdf(torch.ones(test_x.shape[0]) * alpha_val))
            ucb = torch.vstack(ucbs)
            lcb = torch.vstack(lcbs)

        return lcb, ucb

    def eval(self, test_x: np.ndarray, test_y: np.ndarray) -> tuple[float, float, float, float]:
        """Evaluate posterior model conditioned on samples observed so far.

        Args:
            test_x (np.ndarray): The test observations
            test_y (np.ndarray): The test rewards

        Raises:
            NotImplementedError: This function must be implemented by a subclass

        Returns:
            tuple[float, float, float, float]: The log likelihood, the residual mean
                squared error, the calibration error and the calibration chi squared error
        """
        raise NotImplementedError()

    def initialize(
        self, context_x: np.ndarray, context_y: np.ndarray, num_init_samples: int = 2
    ) -> None:
        """Initialize the model with the given observations and rewards

        Args:
            context_x (np.ndarray): The observations
            context_y (np.ndarray): The rewards
            num_init_samples (int, optional): The number of samples to use for the
                initialization. Defaults to 2.

        Raises:
            NotImplementedError: This function must be implemented by a subclass
        """
        raise NotImplementedError()

    def initialize_model(self) -> Union[GPRegressionVanilla, FPACOH_MAP_GP]:
        """Initialize the reward model

        Raises:
            ValueError: If an environmet is required but not set
            ValueError: If the model type is not supported

        Returns:
            Union[GPRegressionVanilla, FPACOH_MAP_GP]: The initialized reward model
        """
        assert self.environment is not None
        model: Union[GPRegressionVanilla, FPACOH_MAP_GP]
        # if self.model_name == "vanilla_gp":
        #     model = GPRegressionVanilla(
        #         normalization_stats=self.environment.normalization_stats,
        #         **self.config,
        #     )
        if self.model_name == "vanilla_gp":
            model = GPRegressionVanilla(
                normalization_stats=self.environment.normalization_stats,
                **self.config,
            )

        # elif self.model_name == "gp_learnt":
        #     model = GPRegressionLearned(
        #         self.context_x[self.selected_samples_idx],
        #         self.context_y[self.selected_samples_idx],
        #         **self.config,
        #     )

        # elif self.model_name == "pacoh":
        #     if self.environment is None:
        #         raise ValueError(
        #             "You first need to set the environment "
        #             "before initializing the model!"
        #         )
        #     model = PACOH_MAP_GP(
        #         input_dim=self.environment.domain.d,
        #         normalization_stats=self.environment.normalization_stats,
        #         **self.config,
        #     )

        # elif self.model_name == "fpacoh":
        #     if self.environment is None:
        #         raise ValueError(
        #             "You first need to set the environment "
        #             "before initializing the model!"
        #         )
        #     model = FPACOH_MAP_GP(
        #         domain=self.environment.domain,
        #         normalization_stats=self.environment.normalization_stats,
        #         **self.config,
        #     )
        elif self.model_name == "fpacoh":
            if self.environment is None:
                raise ValueError(
                    "You first need to set the environment " "before initializing the model!"
                )
            domain = self.environment.domain
            assert isinstance(domain, ContinuousDomain)
            model = FPACOH_MAP_GP(
                domain=domain,
                normalization_stats=self.environment.normalization_stats,
                **self.config,
            )
        else:
            raise ValueError(f"The specified model name '{self.model_name}' is unknown!")

        return model

    def make_query(
        self, observation_sampling_mode: str, allow_replacement: bool = False
    ) -> Optional[int]:
        """Choose an observation from the context for which the true reward function
        should be queried. The observation to be queried is either chosen randomly
        or by choosing the obsercation with the largest uncertainty.

        Args:
            observation_sampling_mode (str): The mode to use for sampling the
                observation. Can be either "random" or "largest_uncertainty"
            allow_replacement (bool, optional): Whether the same observation can be
                sampled multiple times. Defaults to False.

        Raises:
            ValueError: If the observation sampling mode is not supported

        Returns:
            Optional[int]: The index of the observation to be queried or None if all
                observations have already been queried
        """
        assert self.selected_samples_idx is not None
        # Check input arguments
        if observation_sampling_mode not in RewardModelGPBase.observation_sampling_modes:
            raise ValueError(
                f"'observation_sampling_mode' must be one of "
                f"{RewardModelGPBase.observation_sampling_modes}. "
                f"Got '{observation_sampling_mode}' instead."
            )

        # Return None if all samples have already been observed
        if self.observed_all_samples():
            return None

        # Get a list of all samples which already have been observed and all
        # unobserved samples
        observed_samples_idx = self.selected_samples_idx
        unobserved_samples_idx = [
            i for i in range(self.context_size) if i not in observed_samples_idx
        ]

        # Return a random sample id
        if observation_sampling_mode == "random":
            if allow_replacement:
                return self._rng.integers(self.context_size)
            elif len(unobserved_samples_idx) > 0:
                return unobserved_samples_idx[self._rng.integers(len(unobserved_samples_idx))]
            else:
                return None

        # Select the sample about which the reward model is most uncertain
        elif observation_sampling_mode == "largest_uncertainty":
            assert self.context_x is not None

            # Compute the uncertainty for each existing sample
            _, pred_std = self.predict(self.context_x)

            if allow_replacement:
                return int(np.argmax(pred_std))
            else:
                pred_std = np.negative(pred_std)

                # Get a few of the largest values
                indices = np.argpartition(pred_std, len(observed_samples_idx))[
                    : len(observed_samples_idx) + 1
                ]

                # Filter out all values that have already been observed
                indices = [i for i in indices if i not in observed_samples_idx]

                return indices[np.argmin(pred_std[indices])]
        else:
            raise ValueError(
                f"'observation_sampling_mode' has to be one of {self.observation_sampling_modes}"
            )

    def observe(self, idx: int, train_model: bool = False) -> None:
        """Observe the true reward of the observation with the given index.

        Args:
            idx (int): The index of the observation whose true reward should be observed
            train_model (bool, optional): Whether to retrain the reward model after
                observing the new data point. Defaults to False.

        Raises:
            NotImplementedError: This method has to be implemented by the subclass
        """
        raise NotImplementedError()

    def predict(
        self, pred_x: np.ndarray, max_size: int = 10000, upper_conf_bound: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Given a set of observations 'pred_x', predic the reward using the reward model

        Args:
            pred_x (np.ndarray): A set of observations for which the reward should be
                predicted
            max_size (int, optional): If the set of observations is too large, it is
                split into smaller chunks of size 'max_size' to avoid memory and compute
                issues. Defaults to 10000.
            upper_conf_bound (bool, optional): Whether to return the upper confidence
                bound of the predicted reward or the mean. Defaults to False.

        Raises:
            NotImplementedError: This method has to be implemented by the subclass

        Returns:
            Tuple[np.ndarray, np.ndarray]: The predicted reward and the uncertainty
                of the prediction
        """
        raise NotImplementedError()

    def update_context(self, context_x: np.ndarray, context_y: np.ndarray) -> None:
        """Add new observations to the context of the reward model

        Args:
            context_x (np.ndarray): The new observations
            context_y (np.ndarray): The true reward of the new observations
        """
        assert self.context_x is not None
        assert self.context_y is not None
        # Extend the context
        self.context_x = np.concatenate([self.context_x, context_x])
        self.context_y = np.concatenate([self.context_y, context_y])

        # Clip it to the maximum context size if necessary
        self._clip_context()

    # def set_environment(self, env: EnvironmentBase) -> None:
    def set_environment(self, env: "EnvironmentBase") -> None:
        """Set the environment whose reward function should be learned

        Args:
            env (EnvironmentBase): The environment
        """
        self.environment = env

    def observed_all_samples(self) -> bool:
        """Check whether all samples have already been observed

        Returns:
            bool: True if all samples have already been observed, False otherwise
        """
        assert self.selected_samples_idx is not None
        return len(self.selected_samples_idx) >= self.context_size
