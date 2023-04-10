import sys
from typing import Any, Optional, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import mlflow
import numpy as np

from meta_arl.environments import get_environment
from meta_arl.environments.abstract import EnvironmentBase
from meta_arl.meta_bo.models.f_pacoh_map import FPACOH_MAP_GP
from meta_arl.reward_models import get_reward_model
from meta_arl.reward_models.reward_model_pacoh_gp import RewardModelPacohGP
from meta_arl.util.experiments import calibration_values
from meta_arl.util.input import ConfigCreator
from meta_arl.util.myrandom import set_all_seeds
from meta_arl.util.plotting import plot_reward_landscapes

DEBUG = True
NO_SIMULTANEOUS_MEASUREMENTS = len(list(mcolors.TABLEAU_COLORS))
CONFIDENCES = np.linspace(0.05, 1, 20)


def plot_calibration_values(confidences: np.ndarray, calibration_vals: list[np.ndarray]) -> None:
    """Get a set of confidence values and corresponding calibration values and plot them
    in a scatter plot

    Args:
        confidences (np.ndarray): The numpy array containing the confidence values (See
            the 'CONFIDENCES' array)
        calibration_vals (list[np.ndarray]): A list of numpy arrays containing the corresponding
            calibration values for each confidence interval
    """
    confidences_list = list(confidences)
    calibration_vals = list(calibration_vals)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    num_y_vals_per_x = len(calibration_vals[0])

    # Build the x-values
    x_vals_prelim = [[i] * num_y_vals_per_x for i in range(len(calibration_vals))]
    x_vals = []
    [x_vals.extend(vals) for vals in x_vals_prelim]  # type: ignore

    # Build the y-values
    y_vals = confidences_list * len(calibration_vals)

    # Build the color values
    color_prelim: list[np.ndarray] = []
    [color_prelim.extend(vals) for vals in calibration_vals]  # type: ignore
    color_vals = np.array(color_prelim) - np.array(y_vals)

    ax.scatter(
        x_vals,
        y_vals,
        c=color_vals,
        s=200,
        cmap=plt.get_cmap("PiYG"),
        vmin=-1,
        vmax=1,
        edgecolors="black",
        linewidths=1,
    )

    plt.show()


def get_reward_landscape(
    environment: EnvironmentBase,
    true_reward: bool = True,
    return_grid: bool = False,
    resolution: int = 100,
) -> Union[tuple[list[np.ndarray], np.ndarray], np.ndarray]:
    """Computes the following properties of the reward landscape of the environment:
    - The reward landscape itself (can be either the true reward or the learnt reward)
    - The coordinates of the reward landscape

    Args:
        environment (EnvironmentBase): The environment for which the reward landscape should be
            computed.
        true_reward (bool, optional): Whether the true reward or the estimated reward shall be
            computed. Defaults to True.
        return_grid (bool, optional): Whether the coordinates where the reward landscape
            is computed shall be returned. Defaults to False.
        resolution (int, optional): The granularity with which the reward landsacpe should be
            built. Defaults to 100.

    Returns:
        Union[tuple[list[np.ndarray], np.ndarray], np.ndarray]: A tuple containing the desired
            properties of the reward landscape.
    """
    # Store the reward model flag
    reward_flag = environment.use_true_reward

    # Abbreviation
    obs = environment.observation_space

    # Assuming that the environment is 2d, create a 2d grid over
    # the environment dimensions
    lows, highs = obs.low, obs.high
    dim = lows.shape[-1]
    grid = np.meshgrid(*[np.linspace(low, high, resolution) for low, high in zip(lows, highs)])

    # Build the reward landscape of the grid above
    states = np.stack([g.flatten() for g in grid], axis=-1)

    # Get the estimated reward map
    environment.set_use_true_reward(true_reward)
    rewards_map = environment.reward(state=None, action=None, next_state=states).reshape(
        dim * [resolution]
    )

    # Restore the reward flag
    environment.set_use_true_reward(reward_flag)

    if not return_grid:
        return rewards_map

    return grid, rewards_map


def log_metrics(
    suffix: str,
    lls: list[float],
    rmses: list[float],
    calib_errs: list[float],
    calib_errs_chi2: list[float],
    calibration_vals: Optional[list[np.ndarray]] = None,
) -> None:
    """Log the metrics of the experiment

    Args:
        suffix (str): A string that is appended to each metric name
        lls (list[float]): A list of log-likelihoods
        rmses (list[float]): A list of root mean squared errors
        calib_errs (list[float]): A list of calibration errors
        calib_errs_chi2 (list[float]): A list of calibration errors (chi2)
        calibration_vals (Optional[list[np.ndarray]], optional): A list of calibration values.
            Defaults to None.
    """
    for step in range(len(lls)):
        metrics = {
            "log likelihood" + suffix: lls[step],
            "residual mean squared error" + suffix: rmses[step],
            "calibration error" + suffix: calib_errs[step],
            "calibration error chi squared" + suffix: calib_errs_chi2[step],
        }
        if calibration_vals is not None and calibration_vals:
            for index, conf in enumerate(CONFIDENCES):
                metrics[f"calibration at {conf} confidence" + suffix] = calibration_vals[step][
                    index
                ]
        mlflow.log_metrics(metrics=metrics, step=step)


def log_figures(
    suffix: str,
    train_environment: EnvironmentBase,
    test_reward_landscapes_pred: list[np.ndarray],
) -> None:
    """Create and log the figures of the experiment.

    Args:
        suffix (str): A string that is appended to each figure name
        train_environment (EnvironmentBase): The environment which was
            used for training
        test_reward_landscapes_pred (list[np.ndarray]): A list of the reward landscapes
            predicted during the individual iterations of the experiment.
    """
    # Determine the dimension of the observation space
    dim = len(train_environment.observation_space.sample())
    assert dim in [2, 3], "Only 2- and 3-dimensional environments are supported!"
    resolution = 20 if dim == 3 else 100

    if dim == 2:
        # Compute the true reward landscape
        result = get_reward_landscape(
            train_environment, true_reward=True, return_grid=True, resolution=resolution
        )
        assert isinstance(result, list)
        grid_reward, reward_landscape_true = result

        # Plot the true reward landscape
        assert isinstance(reward_landscape_true, np.ndarray)
        assert isinstance(grid_reward, list)
        for index, reward_landscape_pred in enumerate(test_reward_landscapes_pred):
            fig = plot_reward_landscapes(
                grid_reward,
                reward_landscape_true,
                reward_landscape_pred,
                show=False,
            )
            fig_name = f"figures/reward_landscape{suffix}_{index}.png"
            mlflow.log_figure(fig, fig_name)
    elif dim == 3:
        # Compute the true reward landscape
        result = get_reward_landscape(
            train_environment, true_reward=True, return_grid=True, resolution=resolution
        )
        assert isinstance(result, tuple)
        grid_reward, reward_landscape_true = result

        # Plot the true reward landscape
        assert isinstance(reward_landscape_true, np.ndarray)
        assert isinstance(grid_reward, list)
        fig = plot_reward_landscapes(
            grid_reward, reward_landscape_true, None, show=False, slices=True
        )
        fig_name = f"figures/reward_landscape_true{suffix}.png"
        mlflow.log_figure(fig, fig_name)

        # Plot the predicted reward landscapes
        for index, reward_landscape_pred in enumerate(test_reward_landscapes_pred):
            fig = plot_reward_landscapes(
                grid_reward,
                None,
                reward_landscape_pred,
                show=False,
                slices=True,
            )
            fig_name = f"figures/reward_landscape_pred_{suffix}_{index}.png"
            mlflow.log_figure(fig, fig_name)


def test_reward_learners(
    train_size: int,
    test_size: int,
    env_config: dict[str, Any],
    reward_model_config: dict[str, Any],
    seed: Optional[int] = None,
    meta_train: bool = False,
    meta_train_size: Optional[int] = None,
    num_meta_tasks: Optional[int] = None,
    num_meta_valid_tasks: Optional[int] = None,
    leaked_data_fraction: float = 0,
    run_id: Optional[str] = None,
    suffix: str = "",
) -> tuple[
    EnvironmentBase,
    RewardModelPacohGP,
    list[float],
    list[float],
    list[float],
    list[float],
    list[np.ndarray],
    list[np.ndarray],
]:
    """The main function of the experiment. Trains a (potentially meta-trained) reward learner
    using active learning. In regular intervals, the reward learner's performance is tested.

    Args:
        train_size (int): The training set size
        test_size (int): The test set size
        env_config (dict[str, Any]): The configuration of the environment
        reward_model_config (dict[str, Any]): The configuration of the reward learner
        seed (Optional[int], optional): The random seed. Defaults to None.
        meta_train (bool, optional): Whether to meta-train the reward learner. Defaults to False.
        meta_train_size (Optional[int], optional): The size of each meta-training
            dataset. Defaults to None.
        num_meta_tasks (Optional[int], optional): The number of meta-training tasks.
            Defaults to None.
        num_meta_valid_tasks (Optional[int], optional): The number of meta-validation tasks.
            Defaults to None.
        leaked_data_fraction (float, optional): The fraction of the test set that is
            leaked into the training set. Defaults to 0.
        run_id (Optional[str], optional): The run id of the main-training experiment.
            Defaults to None.
        suffix (str, optional): A string that is appended to each figure name. Defaults to "".

    Returns:
        tuple[
            EnvironmentBase,
            RewardModelPacohGP,
            list[float],
            list[float],
            list[float],
            list[float],
            list[np.ndarray],
            list[np.ndarray]
        ]: The environment, the reward learner, and per-iteration information about:
            - the log-likelihood of the reward learner
            - the residual mean-squared errors of the reward learner
            - the calibration errors of the reward learner
            - the calibration error chi-squared values of the reward learner
            - the calibraion values of the reward learner
            - the reward landscapes predicted by the reward learner
    """

    if seed is not None:
        set_all_seeds(seed)
    else:
        seed = 0

    # Get the specified reward model
    reward_model = get_reward_model(reward_model_config)
    assert isinstance(reward_model, RewardModelPacohGP)

    if num_meta_tasks is None:
        num_meta_tasks = 20

    if num_meta_valid_tasks is None:
        num_meta_valid_tasks = 0

    # We need 'num_meta_tasks' for meta training and one task
    # for adaptation
    if meta_train:
        env_config["num_tasks"] = 1 + num_meta_tasks + num_meta_valid_tasks

    # Generate the specified environment
    environment = get_environment(env_config)
    environment.set_tasks(environment.sample_tasks(seed=seed, **env_config))

    # Make sure that all environments have exactly the same test environment
    environment.set_task(0, environment.sample_tasks(num_tasks=1, seed=seed))

    environment.print_task(0)

    # Get the dimension of the observation space
    dim = len(environment.observation_space.sample())

    # Add the environment to the reward model
    reward_model.set_environment(environment)

    if meta_train:
        assert meta_train_size is not None
        # If true, about 'leaked_data_fraction' percent of meta tasks
        # are actually the adaption task. This is useful to check if
        # meta-learning is actually helpful
        if leaked_data_fraction > 0 and leaked_data_fraction <= 1:
            # Compute how many meta-learning tasks should be overwritten
            num_leaked = int(np.ceil(leaked_data_fraction * num_meta_tasks))
            leaked_indices = np.random.choice(
                list(range(1, num_meta_tasks + 1)), num_leaked, replace=False
            )

            # replace 'num_leaked' meta-learning tasks with the training task
            for index in leaked_indices:
                environment.set_task(index, environment.sample_tasks(num_tasks=1, seed=seed))

        # Create the meta-training sets (reserve the first task for testing)
        meta_training_task_ids = list(range(1, num_meta_tasks + 1))
        meta_training_data = environment.sample_rewards(
            num_samples=meta_train_size,
            tasks=meta_training_task_ids,
            use_reward_model=False,
            seed=seed + 100,
        )

        # Create meta-validation sets
        if num_meta_valid_tasks > 0:
            meta_validation_task_ids = list(
                range(num_meta_tasks + 1, num_meta_tasks + num_meta_valid_tasks + 1)
            )
            meta_validation_data_temp = environment.sample_rewards(
                num_samples=2 * meta_train_size,
                tasks=meta_validation_task_ids,
                use_reward_model=False,
                seed=seed + 200,
            )
            meta_validation_data = []
            for x, y in meta_validation_data_temp:
                meta_validation_data.append(
                    (
                        x[:meta_train_size],
                        y[:meta_train_size],
                        x[meta_train_size:],
                        y[meta_train_size:],
                    )
                )
        else:
            meta_validation_data = None

        # Perform meta training
        reward_model.initialize_meta_data(meta_training_data, meta_validation_data)
        inner_model = reward_model.model
        assert isinstance(inner_model, FPACOH_MAP_GP)

        if meta_validation_data is not None:
            (
                valid_ll,
                valid_rmse,
                valid_calib,
                valid_calib_chi_squared,
            ) = inner_model.eval_datasets(meta_validation_data)
            print(
                f"Validation results: {valid_ll = }, {valid_rmse = }, "
                f"{valid_calib = }, {valid_calib_chi_squared = }"
            )

    if seed is not None:
        set_all_seeds(seed)

    # Sample a dataset from the environment
    size = train_size + test_size
    observations, rewards = environment.sample_rewards(
        num_samples=size, tasks=0, use_reward_model=False, seed=seed + 101
    )[0]

    # Perform a train-test split
    train, test = (
        list(zip(observations[:train_size], rewards[:train_size])),
        list(zip(observations[train_size:], rewards[train_size:])),
    )

    # Make sure that no sample is in both, the train and test set
    # Hash str can only be used for short numpy arrays! In this case,
    # each array has length 2 which allows to use hash(str(.))
    hashed_train = [hash(str(t)) for t in train]
    test = [t for t in test if hash(str(t)) not in hashed_train]

    # Finish train test split
    train_x, train_y = np.array([t[0] for t in train]), np.array([t[1] for t in train])
    test_x, test_y = np.array([t[0] for t in test]), np.array([t[1] for t in test])

    # Initialize the reward model with the training data
    reward_model.initialize(train_x, train_y, num_init_samples=2)

    # Set the environments reward model
    environment.set_reward_model(reward_model)

    # Initialize lists for the log-likelihoods, residual sum of squares and
    # calibration errors
    lls, rmses, calib_errs, calib_errs_chi2 = [], [], [], []
    calibration_vals: list[np.ndarray] = []
    reward_landscapes: list[np.ndarray] = []

    # Test whether the reward model is able to learn the reward function
    context_x = reward_model.context_x
    assert context_x is not None
    for sample_idx in range(train_size):
        print(
            "Observing sample ",
            sample_idx,
            ": ",
            context_x[sample_idx],
        )

        # Let the model observe an additional point
        reward_model.observe(sample_idx, train_model=True)

        # Evaluate the model on the test set
        try:
            ll, rmse, calib_err, calib_err_chi2 = reward_model.eval(test_x, test_y)

            lls.append(ll)
            rmses.append(rmse)
            calib_errs.append(calib_err)
            calib_errs_chi2.append(calib_err_chi2)

            # low_conf_bounds, upp_conf_bounds = reward_model.confidence_intervals(
            #     test_x, confidence=CONFIDENCES
            # )
            # calibration_vals.append(
            #    calibration_values(low_conf_bounds, upp_conf_bounds, test_y)
            # )

            if DEBUG and (sample_idx + 1) % 10 == 0:
                resolution = 20 if dim == 3 else 100
                result = get_reward_landscape(
                    environment, true_reward=False, resolution=resolution
                )
                assert isinstance(result, np.ndarray)
                reward_landscapes.append(result)
        except RuntimeError:
            print(f"Sample {sample_idx} caused an error! Skipping...")

    # Log the metrics
    if run_id is not None:
        # print("Step ", sample_idx, ", logging the following metrics: ", metrics)
        with mlflow.start_run(run_id=run_id, nested=True):
            log_metrics(suffix, lls, rmses, calib_errs, calib_errs_chi2, calibration_vals)
            log_figures(suffix, environment, reward_landscapes)
    else:
        log_metrics(suffix, lls, rmses, calib_errs, calib_errs_chi2, calibration_vals)
        log_figures(suffix, environment, reward_landscapes)

    return (
        environment,
        reward_model,
        lls,
        rmses,
        calib_errs,
        calib_errs_chi2,
        calibration_vals,
        reward_landscapes,
    )


if __name__ == "__main__":
    # Track the experiments in the same folder this file is stored
    # mlflow.set_tracking_uri(
    #     "file://" + path.join(path.dirname(path.abspath(__file__)), "mlruns")
    # )

    # If no command-line arguments have been provided, use the following config
    if len(sys.argv) == 1:
        args: Optional[list[str]] = (
            "--configs "
            "3d/config_point_env_goal_3d_fpacoh_gp.py "
            # "config_point_env_learnt_gp.py "
            # "config_point_env_pacoh_gp.py "
            # "config_point_env_fpacoh_gp_leaked_0.6.py "
            # "config_tile_env_simple_gp_sizes.py "
            # "config_tile_env_fpacoh_gp_sizes.py "
            # "config_point_env_goal_fpacoh_gp.py "
            "--params experiment.seed=6::int "
            # "--globalparams num_repeats=10::int"
        ).split()
    else:
        args = None

    global_config, config_list = ConfigCreator.create_from_args(
        args_parsed_list=args, configs_path=None
    )

    # Extract the global variables if they exist
    num_repeats = 1
    if global_config is not None:
        if "num_repeats" in global_config:
            num_repeats = global_config["num_repeats"]

    # Check that we didn't create too many configs
    if len(config_list) > NO_SIMULTANEOUS_MEASUREMENTS:
        raise OverflowError(
            f"This experiment currently only supports at most "
            f"{NO_SIMULTANEOUS_MEASUREMENTS} configurations at the same time."
        )

    # Convert to a list if necessary
    if not isinstance(config_list, list):
        config_list = [config_list]

    # For each config, test its reward learner
    for config in config_list:
        # Run it once if 'num_repeats' == 1
        if num_repeats == 1:
            (
                environment,
                reward_model,
                lls,
                rmses,
                calib_errs,
                calib_errs_chi2,
                calibration_vals,
                reward_landscapes,
            ) = test_reward_learners(**config["experiment"])

            # Compute the true reward landscape
            if DEBUG:
                # plot_calibration_values(CONFIDENCES, calibration_vals)

                print("Log likelihood = ", lls)

        # Otherwise run multiple repeats sequentially
        else:
            # Make sure that each run gets a unique seed and suffix
            config_exp = config["experiment"]
            base_suffix = config_exp.get("suffix", "")
            base_seed = config_exp.get("seed", 0)

            error_counter = 0

            # Run all repeats
            for index in range(num_repeats):
                seed = base_seed + index
                config_exp["seed"] = seed
                config_exp["suffix"] = base_suffix + f"_seed_{seed}"

                try:
                    # Run a single run
                    (
                        environment,
                        reward_model,
                        lls,
                        rmses,
                        calib_errs,
                        calib_errs_chi2,
                        calibration_vals,
                        reward_landscapes,
                    ) = test_reward_learners(**config_exp)

                    print("Log likelihood = ", lls)

                except RuntimeError as e:
                    print(f"Run {index} was aborted due to runtime error!")
                    print("Error message: ", str(e))
                    error_counter += 1

            print(f"{num_repeats - error_counter}/{num_repeats} runs have been successful")
