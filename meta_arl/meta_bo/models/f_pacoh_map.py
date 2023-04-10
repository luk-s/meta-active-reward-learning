import inspect
import math
import time
import warnings
from typing import Any, List, Optional, Sequence, Union, cast

import gpytorch
import numpy as np
import torch
from absl import logging
from meta_arl.config.global_config import device
from meta_arl.meta_bo.domain import ContinuousDomain, DiscreteDomain
from meta_arl.meta_bo.models.abstract import RegressionModelMetaLearned
from meta_arl.meta_bo.models.models import (
    AffineTransformedDistribution,
    LearnedGPRegressionModel,
    NeuralNetwork,
    SEKernelLight,
)
from meta_arl.meta_bo.models.util import DummyLRScheduler, _handle_input_dimensionality
from torch.distributions import MultivariateNormal, Uniform, kl_divergence
from typing_extensions import reveal_type


class FPACOH_MAP_GP(RegressionModelMetaLearned):
    def __init__(
        self,
        domain: ContinuousDomain,
        learning_mode: str = "both",
        weight_decay: float = 0.0,
        feature_dim: int = 2,
        num_iter_fit: int = 10000,
        covar_module: str = "NN",
        mean_module: str = "NN",
        mean_nn_layers: Sequence[int] = (32, 32, 32),
        kernel_nn_layers: Sequence[int] = (32, 32, 32),
        prior_lengthscale: float = 0.2,
        prior_outputscale: float = 2.0,
        prior_kernel_noise: float = 1e-3,
        train_data_in_kl: bool = True,
        num_samples_kl: int = 20,
        task_batch_size: int = 5,
        lr: float = 1e-3,
        lr_decay: float = 1.0,
        normalize_data: bool = True,
        prior_factor: float = 0.1,
        normalization_stats: Optional[dict[str, np.ndarray]] = None,
        random_state: Optional[np.random.Generator] = None,
    ):

        super().__init__(normalize_data, random_state)

        # save init args for serialization
        self._init_args = {
            k: v
            for k, v in locals().items()
            if k in inspect.signature(self.__init__).parameters.keys()  # type: ignore
        }

        assert isinstance(domain, ContinuousDomain) or isinstance(domain, DiscreteDomain)
        assert learning_mode in ["learn_mean", "learn_kernel", "both", "vanilla"]
        assert mean_module in ["NN", "constant", "zero"] or isinstance(
            mean_module, gpytorch.means.Mean
        )
        assert covar_module in ["NN", "SE"] or isinstance(covar_module, gpytorch.kernels.Kernel)

        self.domain = domain
        self.input_dim: int = domain.d
        self.lr, self.weight_decay, self.feature_dim = lr, weight_decay, feature_dim
        self.num_iter_fit, self.task_batch_size, self.normalize_data = (
            num_iter_fit,
            task_batch_size,
            normalize_data,
        )

        """ Setup prior and likelihood """
        self._setup_gp_prior(
            mean_module,
            covar_module,
            learning_mode,
            feature_dim,
            mean_nn_layers,
            kernel_nn_layers,
        )
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.likelihoods.noise_models.GreaterThan(1e-3)
        ).to(device)
        self.shared_parameters.append({"params": self.likelihood.parameters(), "lr": self.lr})
        self._setup_optimizer(lr, lr_decay)

        """ domain support dist & prior kernel """
        self.prior_factor = prior_factor
        self.domain_dist = Uniform(
            low=torch.from_numpy(domain.l).float(),
            high=torch.from_numpy(domain.u).float(),
        )
        lengthscale = prior_lengthscale * torch.ones((1, self.input_dim))
        self.prior_covar_module = SEKernelLight(
            lengthscale, output_scale=torch.tensor(prior_outputscale)
        )
        self.prior_kernel_noise = prior_kernel_noise
        self.train_data_in_kl = train_data_in_kl
        self.num_samples_kl = num_samples_kl

        """ ------- normalization stats & data setup  ------- """
        self._normalization_stats = normalization_stats
        self.reset_to_prior()

        self.fitted = False
        self.task_dicts: list[dict[str, Any]] = []

    def meta_fit(
        self,
        meta_train_tuples: list[tuple[np.ndarray, np.ndarray]],
        meta_valid_tuples: Optional[
            list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
        ] = None,
        verbose: bool = True,
        log_period: int = 500,
        n_iter: Optional[int] = None,
    ) -> float:

        assert (meta_valid_tuples is None) or (
            all([len(valid_tuple) == 4 for valid_tuple in meta_valid_tuples])
        )
        self.likelihood.train()

        task_dicts: list[dict[str, Any]] = self._prepare_meta_train_tasks(meta_train_tuples)

        t = time.time()
        cum_loss = 0.0
        n_iter = self.num_iter_fit if n_iter is None else n_iter

        # DEBUG
        mll_list, kl_list = [], []

        for itr in range(1, n_iter + 1):
            # actual meta-training step
            task_dict_batch: list[dict[str, Any]] = self._rds.choice(task_dicts, size=self.task_batch_size)  # type: ignore

            # DEBUG
            loss, mlls, kls = self._step(task_dict_batch, n_tasks=len(task_dicts))
            cum_loss += loss

            mll_list.append(mlls)
            kl_list.append(kls)

            # print training stats stats
            if itr == 1 or itr % log_period == 0:
                duration = time.time() - t
                avg_loss = cum_loss / (log_period if itr > 1 else 1.0)
                cum_loss = 0.0
                t = time.time()

                message = "Iter %d/%d - Loss: %.6f - Time %.2f sec" % (
                    itr,
                    self.num_iter_fit,
                    avg_loss,
                    duration,
                )

                # if validation data is provided  -> compute the valid log-likelihood
                if meta_valid_tuples is not None:
                    self.likelihood.eval()
                    (
                        valid_ll,
                        valid_rmse,
                        calibr_err,
                        calibr_err_chi2,
                    ) = self.eval_datasets(meta_valid_tuples)
                    self.likelihood.train()
                    message += " - Valid-LL: %.3f - Valid-RMSE: %.3f - Calib-Err %.3f" % (
                        valid_ll,
                        valid_rmse,
                        calibr_err,
                    )
                print(message)
                if verbose:
                    logging.info(message)

        self.fitted = True

        # DEBUG START

        # print("\nMLLs\n")
        # for mll in mll_list:
        #    print("\t", mll)
        # print("\nKLs\n")
        # for kll in kl_list:
        #    print("\t", kll)

        for index, task_dict in enumerate(task_dicts):
            print(
                f"task dict {index}: {task_dict['model'].not_positive_definite_counter}"
                " not positive definite matrices"
            )

        # DEBUG END

        # set gpytorch modules to eval mode and set gp to meta-learned gp prior
        assert self.X_data.shape[0] == 0 and self.y_data.shape[0] == 0, (
            "Data for posterior inference can be passed " "only after the meta-training"
        )
        for task_dict in task_dicts:
            task_dict["model"].eval()
        self.likelihood.eval()
        self.reset_to_prior()
        self.task_dicts = task_dicts
        return loss

    def predict(
        self, test_x: np.ndarray, return_density: bool = False, **kwargs: Any
    ) -> Union[AffineTransformedDistribution, tuple[np.ndarray, np.ndarray]]:
        if test_x.ndim == 1:
            # LUKAS FIX
            # test_x = np.expand_dims(test_x, axis=-1)
            test_x = np.expand_dims(test_x, axis=0)

        with torch.no_grad():
            test_x_normalized = self._normalize_data(test_x)
            test_x_tensor = torch.from_numpy(test_x_normalized).float().to(device)

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

    def meta_predict(
        self,
        context_x: np.ndarray,
        context_y: np.ndarray,
        test_x: np.ndarray,
        return_density: bool = False,
    ) -> Union[AffineTransformedDistribution, tuple[np.ndarray, np.ndarray]]:
        """
        Performs posterior inference (target training) with (context_x, context_y) as training data and then
        computes the predictive distribution of the targets p(y|test_x, test_context_x, context_y) in the test points

        Args:
            context_x: (ndarray) context input data for which to compute the posterior
            context_y: (ndarray) context targets for which to compute the posterior
            test_x: (ndarray) query input data of shape (n_samples, ndim_x)
            return_density: (bool) whether to return result as mean and std ndarray or as MultivariateNormal pytorch object

        Returns:
            (pred_mean, pred_std) predicted mean and standard deviation corresponding to p(t|test_x, test_context_x, context_y)
        """

        context_x, context_y = _handle_input_dimensionality(context_x, context_y)
        result = _handle_input_dimensionality(test_x)
        assert isinstance(result, np.ndarray)
        test_x = result
        assert test_x.shape[1] == context_x.shape[1]

        # normalize data and convert to tensor
        context_x_tensor, context_y_tensor = self._prepare_data_per_task(context_x, context_y)

        result = self._normalize_data(X=test_x, Y=None)
        assert isinstance(result, np.ndarray)
        test_x = result
        test_x_tensor = torch.from_numpy(test_x).float().to(device)

        with torch.no_grad():
            # compute posterior given the context data
            gp_model = LearnedGPRegressionModel(
                context_x_tensor,
                context_y_tensor,
                self.likelihood,
                learned_kernel=self.nn_kernel_map,
                learned_mean=self.nn_mean_fn,
                covar_module=self.covar_module,
                mean_module=self.mean_module,
            )
            gp_model.eval()
            self.likelihood.eval()
            pred_dist = self.likelihood(gp_model(test_x_tensor))
            pred_dist_transformed = AffineTransformedDistribution(
                pred_dist, normalization_mean=self.y_mean, normalization_std=self.y_std
            )

        if return_density:
            return pred_dist_transformed
        else:
            pred_mean = pred_dist_transformed.mean
            pred_std = pred_dist_transformed.stddev
            return pred_mean.detach().cpu().numpy(), pred_std.detach().cpu().numpy()

    def reset_to_prior(self) -> None:
        self._reset_data()
        self.gp = lambda x: self._prior(x)

    def _reset_posterior(self) -> None:
        x_context = torch.from_numpy(self.X_data).float().to(device)
        y_context = torch.from_numpy(self.y_data).float().to(device)
        self.gp = LearnedGPRegressionModel(
            x_context,
            y_context,
            self.likelihood,
            learned_kernel=self.nn_kernel_map,
            learned_mean=self.nn_mean_fn,
            covar_module=self.covar_module,
            mean_module=self.mean_module,
        )
        self.gp.eval()
        self.likelihood.eval()

    def state_dict(self) -> dict[str, Any]:
        state_dict = {
            "optimizer": self.optimizer.state_dict(),
            "model": self.task_dicts[0]["model"].state_dict(),
        }
        for task_dict in self.task_dicts:
            for key, tensor in task_dict["model"].state_dict().items():
                assert torch.all(state_dict["model"][key] == tensor).item()
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for task_dict in self.task_dicts:
            task_dict["model"].load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def _prior(self, x: np.ndarray) -> gpytorch.distributions.MultivariateNormal:
        if self.nn_kernel_map is not None:
            projected_x = self.nn_kernel_map(x)
        else:
            projected_x = x

            # feed through mean module
        if self.nn_mean_fn is not None:
            mean_x = self.nn_mean_fn(x).squeeze()
        else:
            assert self.mean_module is not None
            mean_x = self.mean_module(projected_x).squeeze()

        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def _prepare_meta_train_tasks(
        self, meta_train_tuples: list[tuple[np.ndarray, np.ndarray]]
    ) -> list[dict[str, Any]]:
        self._check_meta_data_shapes(meta_train_tuples)

        if self._normalization_stats is None:
            self._compute_meta_normalization_stats(meta_train_tuples)
        else:
            self._set_normalization_stats(self._normalization_stats)

        task_dicts = [self._dataset_to_task_dict(x, y) for x, y in meta_train_tuples]
        return task_dicts

    def _dataset_to_task_dict(self, x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        # a) prepare data
        x_tensor, y_tensor = self._prepare_data_per_task(x, y)
        task_dict = {"x_train": x_tensor, "y_train": y_tensor}

        # b) prepare model
        task_dict["model"] = LearnedGPRegressionModel(
            task_dict["x_train"],
            task_dict["y_train"],
            self.likelihood,
            learned_kernel=self.nn_kernel_map,
            learned_mean=self.nn_mean_fn,
            covar_module=self.covar_module,
            mean_module=self.mean_module,
        )
        task_dict["mll_fn"] = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.likelihood, task_dict["model"]
        )
        return task_dict

    def _step(
        self, task_dict_batch: list[dict[str, Any]], n_tasks: int
    ) -> tuple[float, np.ndarray, np.ndarray]:
        assert len(task_dict_batch) > 0
        loss = torch.tensor(0.0)
        self.optimizer.zero_grad()

        # DEBUG
        kls = []
        mlls = []

        for task_dict in task_dict_batch:
            # mll term
            output = task_dict["model"](task_dict["x_train"])
            mll = task_dict["mll_fn"](output, task_dict["y_train"])
            mlls.append(mll.item())

            # kl term
            kl = self._f_kl(task_dict)
            kls.append(kl.item())

            #  terms for pre-factors
            n = n_tasks
            m = task_dict["x_train"].shape[0]

            # loss for this batch
            loss += (
                -mll / (self.task_batch_size * m)
                + self.prior_factor * (1 / math.sqrt(n) + 1 / (n * m)) * kl / self.task_batch_size
            )

        # DEBUG
        mlls_np = np.array(mlls)
        kls_np = np.array(kls)

        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        return loss.item(), mlls_np, kls_np

    def _sample_measurement_set(self, x_train: np.ndarray) -> torch.Tensor:
        if self.train_data_in_kl:
            n_train_x = min(x_train.shape[0], self.num_samples_kl // 2)
            n_rand_x = self.num_samples_kl - n_train_x
            idx_rand = np.random.choice(x_train.shape[0], n_train_x)
            x_kl = torch.cat([x_train[idx_rand], self.domain_dist.sample((n_rand_x,))], dim=0)
        else:
            print("I'm using the fixed code!!!")
            x_kl = self.domain_dist.sample((self.num_samples_kl,))
            result = self._normalize_data(x_kl.numpy())
            assert not isinstance(result, tuple)
            x_kl = torch.from_numpy(result)
        assert x_kl.shape == (self.num_samples_kl, self.input_dim)
        return x_kl

    def _f_kl(self, task_dict: dict[str, Any]) -> torch.Tensor:
        with gpytorch.settings.debug(False):

            # sample / construc measurement set
            x_kl = self._sample_measurement_set(task_dict["x_train"])

            # functional KL
            dist_f_posterior = task_dict["model"](x_kl)
            K_prior = torch.reshape(
                self.prior_covar_module(x_kl).evaluate(), (x_kl.shape[0], x_kl.shape[0])
            )

            #####################################
            #   Fixing inaccuracy of gpytorch   #
            #####################################
            K_prior = (K_prior + K_prior.T) / 2

            inject_noise_prior_std = self.prior_kernel_noise
            inject_noise_posterior_std = self.prior_kernel_noise

            def is_positive_definite(matrix: torch.Tensor) -> bool:
                """Adapted from .../torch/distributions/constraints.py"""

                # Check if the matrix is symmetric
                is_symmetric = torch.isclose(matrix, matrix.mT, atol=1e-6).all()
                if not is_symmetric:
                    raise ValueError("Matrix is not symmetric!")

                # Try to do a cholesky decomposition
                return torch.linalg.cholesky_ex(matrix).info.eq(0)

            # Make sure that the prior covariance matrix is positive definite
            for error_counter_prior in range(5):
                covar_prior = K_prior + inject_noise_prior_std * torch.eye(x_kl.shape[0])

                if is_positive_definite(covar_prior):
                    break

                inject_noise_prior_std = 2 * inject_noise_prior_std

                warnings.warn(
                    "Provided prior covariance matrix was not positive definite! "
                    "Doubling inject_noise_prior_std to %.4f and retrying"
                    % (inject_noise_prior_std)
                )
            # This else belongs to the for- statement not the if- statement!
            else:
                raise RuntimeError("Not able to create positive definite prior covariance matrix!")

            # Create the prior distribution
            dist_f_prior = MultivariateNormal(
                loc=torch.zeros(x_kl.shape[0]),
                covariance_matrix=covar_prior,
            )

            # Make sure that the posterior covariance matrix is positive definite
            base_covariance = dist_f_posterior.covariance_matrix
            covar_posterior = base_covariance  # we start with 0 jitter for this matrix
            for error_counter_posterior in range(9):
                if is_positive_definite(covar_posterior):
                    break

                inject_noise_posterior_std = 2 * inject_noise_posterior_std

                covar_posterior = base_covariance + inject_noise_posterior_std * torch.eye(
                    x_kl.shape[0]
                )

                warnings.warn(
                    "Provided posterior covariance matrix was not positive definite! "
                    "Doubling inject_noise_posterior_std to %.4f and retrying"
                    % (inject_noise_posterior_std)
                )

            # This else belongs to the for- statement not the if- statement!
            else:
                raise RuntimeError(
                    "Not able to create positive definite posterior covariance matrix!"
                )

            # Create the posterior distribution
            dist_f_posterior = MultivariateNormal(
                loc=dist_f_posterior.loc,
                covariance_matrix=covar_posterior,
            )

            #####################################
            #            End of fix             #
            #####################################

            return kl_divergence(dist_f_posterior, dist_f_prior)

    def _setup_gp_prior(
        self,
        mean_module: Union[str, gpytorch.means.Mean],
        covar_module: Union[str, gpytorch.kernels.Kernel],
        learning_mode: str,
        feature_dim: int,
        mean_nn_layers: Sequence[int],
        kernel_nn_layers: Sequence[int],
    ) -> None:

        self.shared_parameters = []

        # a) determine kernel map & module
        if covar_module == "NN":
            assert learning_mode in [
                "learn_kernel",
                "both",
            ], "neural network parameters must be learned"
            self.nn_kernel_map: Optional[NeuralNetwork] = NeuralNetwork(
                input_dim=self.input_dim,
                output_dim=feature_dim,
                layer_sizes=kernel_nn_layers,
            ).to(device)
            self.shared_parameters.append(
                {
                    "params": self.nn_kernel_map.parameters(),
                    "lr": self.lr,
                    "weight_decay": self.weight_decay,
                }
            )
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=feature_dim)
            ).to(device)
        else:
            self.nn_kernel_map = None

        if covar_module == "SE":
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=self.input_dim)
            ).to(device)
        elif isinstance(covar_module, gpytorch.kernels.Kernel):
            self.covar_module = covar_module.to(device)

        # b) determine mean map & module

        if mean_module == "NN":
            assert learning_mode in [
                "learn_mean",
                "both",
            ], "neural network parameters must be learned"
            self.nn_mean_fn: Optional[NeuralNetwork] = NeuralNetwork(
                input_dim=self.input_dim, output_dim=1, layer_sizes=mean_nn_layers
            ).to(device)
            self.shared_parameters.append(
                {
                    "params": self.nn_mean_fn.parameters(),
                    "lr": self.lr,
                    "weight_decay": self.weight_decay,
                }
            )
            self.mean_module = None
        else:
            self.nn_mean_fn = None

        if mean_module == "constant":
            self.mean_module = gpytorch.means.ConstantMean().to(device)
        elif mean_module == "zero":
            self.mean_module = gpytorch.means.ZeroMean().to(device)
        elif isinstance(mean_module, gpytorch.means.Mean):
            self.mean_module = mean_module.to(device)

        # c) add parameters of covar and mean module if desired

        if learning_mode in ["learn_kernel", "both"]:
            self.shared_parameters.append(
                {"params": self.covar_module.hyperparameters(), "lr": self.lr}
            )

        if learning_mode in ["learn_mean", "both"] and self.mean_module is not None:
            self.shared_parameters.append(
                {"params": self.mean_module.hyperparameters(), "lr": self.lr}
            )

    def _setup_optimizer(self, lr: float, lr_decay: float) -> None:
        self.optimizer = torch.optim.AdamW(
            self.shared_parameters, lr=lr, weight_decay=self.weight_decay
        )
        if lr_decay < 1.0:
            self.lr_scheduler: Union[
                torch.optim.lr_scheduler.StepLR, DummyLRScheduler
            ] = torch.optim.lr_scheduler.StepLR(self.optimizer, 1000, gamma=lr_decay)
        else:
            self.lr_scheduler = DummyLRScheduler()

    def _vectorize_pred_dist(
        self, pred_dist: AffineTransformedDistribution
    ) -> torch.distributions.Normal:
        return torch.distributions.Normal(pred_dist.mean, pred_dist.stddev)
