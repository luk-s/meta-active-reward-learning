import inspect
import time
from typing import Any, Callable, Optional, OrderedDict, Union, cast

import gpytorch
import numpy as np
import torch
from meta_arl.config.global_config import device
from meta_arl.meta_bo.models.abstract import RegressionModel
from meta_arl.meta_bo.models.models import (
    AffineTransformedDistribution,
    LearnedGPRegressionModel,
    NeuralNetwork,
)


class GPRegressionVanilla(RegressionModel):
    def __init__(
        self,
        input_dim: int,
        kernel_variance: float = 2.0,
        kernel_lengthscale: float = 0.2,
        likelihood_std: float = 0.05,
        normalize_data: bool = True,
        normalization_stats: Optional[dict[str, np.ndarray]] = None,
        random_state: Optional[np.random.Generator] = None,
    ):

        super().__init__(normalize_data=normalize_data, random_state=random_state)
        # save init args for serialization purposes
        self._init_args = {
            k: v
            for k, v in locals().items()
            if k in inspect.signature(self.__init__).parameters.keys()  # type: ignore
        }

        """  ------ Setup model ------ """
        self.input_dim = input_dim

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=self.input_dim)
        ).to(device)
        self.covar_module.outputscale = kernel_variance
        self.covar_module.base_kernel.lengthscale = kernel_lengthscale

        self.mean_module = gpytorch.means.ZeroMean().to(device)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        self.likelihood.noise = likelihood_std

        """ ------- normalization stats & data setup ------- """
        self._set_normalization_stats(normalization_stats)
        self.reset_to_prior()

    def _reset_posterior(self) -> None:
        x_context = torch.from_numpy(self.X_data)
        y_context = torch.from_numpy(self.y_data)
        self.gp: Union[LearnedGPRegressionModel, Callable] = LearnedGPRegressionModel(
            x_context,
            y_context,
            self.likelihood,
            learned_kernel=None,
            learned_mean=None,
            covar_module=self.covar_module,
            mean_module=self.mean_module,
        )
        self.gp.eval()
        self.likelihood.eval()

    def _prior(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x).squeeze()
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def reset_to_prior(self) -> None:
        self._reset_data()
        self.gp = lambda x: self._prior(x)

    def predict(
        self, test_x: np.ndarray, return_density: bool = False, **kwargs: Any
    ) -> Union[AffineTransformedDistribution, tuple[np.ndarray, np.ndarray]]:
        """
        computes the predictive distribution of the targets p(t|test_x, train_x, train_y)

        Args:
            test_x: (ndarray) query input data of shape (n_samples, ndim_x)
            return_density (bool) whether to return a density object or

        Returns:
            (pred_mean, pred_std) predicted mean and standard deviation corresponding to p(y_test|X_test, X_train, y_train)
        """
        if test_x.ndim == 1:
            # LUKAS FIX
            # test_x = np.expand_dims(test_x, axis=-1)
            test_x = np.expand_dims(test_x, axis=0)

        with torch.no_grad():
            test_x_normalized = self._normalize_data(test_x)
            test_x_tensor = torch.from_numpy(test_x_normalized).to(device)

            # get predictive posterior dist
            post_f = self.gp(test_x_tensor)
            if post_f.mean.ndim == 0:
                # There is a bug in gpytorch so that it can't handle only 1 data-point.
                # The following line fixes this issue.
                post_f = gpytorch.distributions.MultivariateNormal(
                    mean=post_f.mean.unsqueeze(0),
                    covariance_matrix=post_f.covariance_matrix,
                )
            pred_dist = self.likelihood(post_f)

            pred_dist_transformed = AffineTransformedDistribution(
                pred_dist, normalization_mean=self.y_mean, normalization_std=self.y_std
            )
            if return_density:
                return pred_dist_transformed
            else:
                pred_mean = pred_dist_transformed.mean.cpu().numpy()
                pred_std = pred_dist_transformed.stddev.cpu().numpy()
                return pred_mean, pred_std

    def predict_mean_std(self, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pred = self.predict(test_x, return_density=False)
        pred = cast(tuple[np.ndarray, np.ndarray], pred)
        return pred

    def state_dict(self) -> dict[str, OrderedDict[str, torch.Tensor]]:
        assert isinstance(self.gp, LearnedGPRegressionModel)
        state_dict = {
            "model": self.gp.state_dict(),
        }
        return state_dict

    def load_state_dict(self, state_dict: dict[str, OrderedDict[str, torch.Tensor]]) -> None:
        assert isinstance(self.gp, LearnedGPRegressionModel)
        self.gp.load_state_dict(state_dict["model"])

    def _vectorize_pred_dist(
        self, pred_dist: AffineTransformedDistribution
    ) -> torch.distributions.Normal:
        return torch.distributions.Normal(pred_dist.mean, pred_dist.stddev)


if __name__ == "__main__":
    import numpy as np
    import torch
    from matplotlib import pyplot as plt

    n_train_samples = 20
    n_test_samples = 200

    torch.manual_seed(25)
    x_data = torch.normal(mean=-1, std=2.0, size=(n_train_samples + n_test_samples, 1))
    W = torch.tensor([[0.6]])
    b = torch.tensor([-1])
    y_data = (
        x_data.matmul(W.T)
        + torch.sin((0.6 * x_data) ** 2)
        + b
        + torch.normal(mean=0.0, std=0.1, size=(n_train_samples + n_test_samples, 1))
    )
    y_data = torch.reshape(y_data, (-1,))

    x_data_train, x_data_test = (
        x_data[:n_train_samples].numpy(),
        x_data[n_train_samples:].numpy(),
    )
    y_data_train, y_data_test = (
        y_data[:n_train_samples].numpy(),
        y_data[n_train_samples:].numpy(),
    )

    gp_mll = GPRegressionVanilla(input_dim=x_data.shape[-1], kernel_lengthscale=1.0)
    gp_mll.add_data(x_data_train, y_data_train)

    x_plot = np.linspace(6, -6, num=200)
    gp_mll.confidence_intervals(x_plot)

    result = gp_mll.predict(x_plot)
    result = cast(tuple[np.ndarray, np.ndarray], result)
    pred_mean, pred_std = result
    pred_mean, pred_std = pred_mean.flatten(), pred_std.flatten()

    plt.scatter(x_data_test, y_data_test)
    plt.plot(x_plot, pred_mean)

    # lcb, ucb = pred_mean - pred_std, pred_mean + pred_std
    lcb, ucb = gp_mll.confidence_intervals(x_plot)
    plt.fill_between(x_plot, lcb, ucb, alpha=0.4)
    plt.show()
