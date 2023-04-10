import multiprocessing
import sys
from typing import Any, Optional, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import stable_baselines3.common.base_class
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from meta_arl.agents import get_agent
from meta_arl.callbacks import get_callbacks
from meta_arl.environments import get_environment, vectorize_environment
from meta_arl.environments.abstract import EnvironmentBase
from meta_arl.util.input import ConfigCreator
from meta_arl.util.myrandom import set_all_seeds
from meta_arl.util.plotting import (
    plot_agent_policy,
    plot_pointmass_rollouts,
    plot_reward_landscapes,
)

RLAlgorithm = stable_baselines3.common.base_class.BaseAlgorithm
NO_SIMULTANEOUS_MEASUREMENTS = len(list(mcolors.TABLEAU_COLORS))

OBSERVATION_SAMPLING_MODES = ["random", "largest_uncertainty"]

DEBUG = True


def rollout(
    model: RLAlgorithm, env: EnvironmentBase, num_episodes: int = 1, deterministic: bool = True
) -> list[dict[str, list[Union[np.ndarray, float]]]]:
    """Perform 'num_episodes' episode rollouts in the environment and return the results.

    Args:
        model (RLAlgorithm): An RL algorithm from the stable-baselines3 package.
        env (EnvironmentBase): An environment from the meta-arl package.
        num_episodes (int, optional): The number of episodes the rollout should contain.
            Defaults to 1.
        deterministic (bool, optional): Whether to use deterministic model predictions
            or not. Defaults to True.

    Returns:
        list[dict[str, list[Union[np.ndarray, float]]]]: A list of dictionaries. Each dictionary
            contains information about a single episode.
    """
    # Initialize buffer
    replay_buffer = []
    temp_buffer: dict[str, list[Union[np.ndarray, float]]] = {
        "prev_state": [],
        "action": [],
        "next_state": [],
        "reward": [],
        "done": [],
    }

    # Reset the environment
    observation = env.reset(normalize_obs=False)
    state = None
    episode_start = None

    # Perform rollouts of 'num_episodes' episodes
    episode_count = 0
    while episode_count < num_episodes:
        temp_buffer["prev_state"].append(observation)

        # Perform a step
        action, state = model.predict(
            env.normalize_observation(observation),
            state=state,
            episode_start=episode_start,
            deterministic=deterministic,
        )

        # Log the step information
        observation, reward, done, info = env.step(action, normalize_obs=False)
        temp_buffer["action"].append(action)
        temp_buffer["reward"].append(reward)
        temp_buffer["done"].append(done)
        temp_buffer["next_state"].append(observation)

        # Check if current episode has finished (or timed out)
        if done:
            # Increase episode counter and reset buffers and environments
            episode_count += 1
            observation = env.reset(normalize_obs=False)
            replay_buffer.append(temp_buffer)
            temp_buffer = {
                "prev_state": [],
                "action": [],
                "next_state": [],
                "reward": [],
                "done": [],
            }

    return replay_buffer


def get_reward_landscape(
    environment: EnvironmentBase,
    true_reward: bool = True,
    return_rmse: bool = False,
    return_grid: bool = False,
    return_minimum: bool = False,
    return_maximum: bool = False,
    x_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    resolution: int = 100,
) -> Union[
    list[Union[np.ndarray, list[np.ndarray], float]], Union[np.ndarray, list[np.ndarray], float]
]:
    """Computes the following properties of the reward landscape of the environment:
    - The reward landscape itself (can be either the true reward or the learnt reward)
    - The coordinates of the reward landscape
    - The RMSE between the true reward and the learnt reward
    - The smallest achievable reward in the given environment
    - The largest achievable reward in the given environment

    Args:
        environment (EnvironmentBase): The environment for which the reward landscape should be
            computed.
        true_reward (bool, optional): Whether the true reward or the estimated reward shall be
            computed. Defaults to True.
        return_rmse (bool, optional): Whether the RMSE shall be computed. Defaults to False.
        return_grid (bool, optional): Whether the coordinates where the reward landscape
            is computed shall be returned. Defaults to False.
        return_minimum (bool, optional): Whether the minimum possible reward shall be returned.
            Defaults to False.
        return_maximum (bool, optional): Whether the maximum possible reward shall be returned.
            Defaults to False.
        x_test (Optional[np.ndarray], optional): The states on which the environment should make
            predictions. Defaults to None.
        y_test (Optional[np.ndarray], optional): The true reward of the states on which the
            environment should make predictiosn. Defaults to None.
        resolution (int, optional): The granularity with which the reward landsacpe should be
            built. Defaults to 100.

    Raises:
        TypeError: if not enough arguments are provided to compute the rmse

    Returns:
        Union[
            list[Union[np.ndarray, list[np.ndarray], float]], Union[np.ndarray,
            list[np.ndarray], float]
        ]: A list containing the desired properties of the reward landscape.
    """

    if return_rmse:
        is_none = []
        if x_test is None:
            is_none.append("x_test")
        if y_test is None:
            is_none.append("y_test")
        if len(is_none) > 0:
            raise TypeError("The following arguments are missing to compute the rmse: ", is_none)

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

    reward_estimate_max = None
    reward_estimate_min = None
    rmse = None

    # Compute the rmse
    if return_rmse:
        assert x_test is not None
        y_pred = np.array(environment.reward(state=None, action=None, next_state=x_test))
        rmse = np.mean((y_test - y_pred) ** 2)

    if return_maximum:
        # Get the maximum reward and its position in the reward map
        max_reward = rewards_map.max()

        # Convert x,y into the coordiantes of the environment and
        # Compute the distance to the starting point
        max_reward_pos = states[rewards_map.argmax()]
        max_reward_dist = np.sqrt(np.dot(max_reward_pos, max_reward_pos))
        max_action_dist = np.sqrt(
            np.dot(environment.action_space.high, environment.action_space.high)
        )

        # Compute how many steps it would take the agent to reach the maximum reward
        num_steps = int(np.ceil(max_reward_dist / max_action_dist))

        # Compute the path to the maximum reward
        reward_path = np.linspace(dim * [0], max_reward_pos, num_steps)

        # Get the reward of this path
        path_reward = environment.reward(state=None, action=None, next_state=reward_path).sum()

        # Compute the reward estimate
        reward_estimate_max = (
            path_reward + (environment._max_episode_length - num_steps) * max_reward
        )

    if return_minimum:
        # Get the minimum reward and its position in the reward map
        min_reward = rewards_map.min()

        # Convert x,y into the coordiantes of the environment and
        # Compute the distance to the starting point
        min_reward_pos = states[rewards_map.argmin()]
        min_reward_dist = np.sqrt(np.dot(min_reward_pos, min_reward_pos))
        max_action_dist = np.sqrt(
            np.dot(environment.action_space.high, environment.action_space.high)
        )

        # Compute how many steps it would take the agent to reach the minimum reward
        num_steps = int(np.ceil(min_reward_dist / max_action_dist))

        # Compute the path to the minimum reward
        reward_path = np.linspace(dim * [0], min_reward_pos, num_steps)

        # Get the reward of this path
        path_reward = environment.reward(state=None, action=None, next_state=reward_path).sum()

        # Compute the reward estimate
        reward_estimate_min = (
            path_reward + (environment._max_episode_length - num_steps) * min_reward
        )

    # Restore the reward flag
    environment.set_use_true_reward(reward_flag)

    # Return the results
    results: list[Union[np.ndarray, list[np.ndarray], float]] = [rewards_map]

    if return_grid:
        results.insert(0, grid)

    if return_rmse:
        assert rmse is not None
        results.append(float(rmse))

    if return_minimum:
        assert reward_estimate_min is not None
        results.append(float(reward_estimate_min))

    if return_maximum:
        assert reward_estimate_max is not None
        results.append(float(reward_estimate_max))

    if len(results) == 1:
        return results[0]
    return results


def get_agent_policy(
    environment: EnvironmentBase,
    agent: RLAlgorithm,
    return_grid: bool = False,
    resolution: int = 20,
) -> Union[np.ndarray, tuple[list[np.ndarray], np.ndarray]]:
    """Get the policy of an agent acting in a provided environment in the form of
    a vector field on the environment.

    Args:
        environment (EnvironmentBase): The environment.
        agent (RLAlgorithm): An RL algorithm from the stable-baselines3 library.
        return_grid (bool, optional): Whether to return the coordinates on which the policy
            is computed. Defaults to False.
        resolution (int, optional): The resolution of the grid on which the policy is computed.
            Defaults to 20.

    Returns:
        Union[np.ndarray, tuple[list[np.ndarray], np.ndarray]]: The policy of the agent on the
            environmentin in the form of a vector field .
    """

    # Abbreviation
    obs = environment.observation_space

    # Assuming that the environment is 2d, create a 2d grid over
    # the environment dimensions
    lows, highs = obs.low, obs.high
    grid = np.meshgrid(*[np.linspace(low, high, resolution) for low, high in zip(lows, highs)])

    # Build the reward landscape of the grid above
    states = np.stack([g.flatten() for g in grid], axis=-1)

    # Get the estimated reward map
    policy_map, _ = agent.predict(environment.normalize_observation(states))

    if return_grid:
        return grid, policy_map

    return policy_map


def log_metrics(
    suffix: str,
    test_reward_means: list[float],
    test_reward_stds: list[float],
) -> None:
    """Log the metrics of the experiment.

    Args:
        suffix (str): A string to append to the metric names.
        test_reward_means (list[float]): A list of the means of the test rewards.
        test_reward_stds (list[float]): A list of the standard deviations of the test rewards.
    """
    for step in range(len(test_reward_means)):
        metrics = {
            "test reward means" + suffix: test_reward_means[step],
            "test reward stds" + suffix: test_reward_stds[step],
        }
        mlflow.log_metrics(metrics=metrics, step=step)
        if len(test_reward_means) > 1:
            mlflow.log_metric(
                key="mean normalized reward" + suffix, value=np.mean(test_reward_means)
            )


def log_figures(
    suffix: str,
    train_environment: EnvironmentBase,
    agent: RLAlgorithm,
    test_reward_landscapes_pred: list[np.ndarray],
    test_rollouts: list[list[dict[str, list[Union[np.ndarray, float]]]]],
    test_agent_policies: list[np.ndarray],
) -> None:
    """Create and log the figures of the experiment.

    Args:
        suffix (str): A string to append to the figure names.
        train_environment (EnvironmentBase): The training environment.
        agent (RLAlgorithm): A stable-baselines3 RL algorithm.
        test_reward_landscapes_pred (list[np.ndarray]): A list of the reward landscapes
            predicted during the individual iterations of the experiment.
        test_rollouts (list[list[dict[str, list[Union[np.ndarray, float]]]]]): A list of the
            test rollouts conducted during the individual iterations of the experiment.
        test_agent_policies (list[np.ndarray]): A list of the policy vector fields of the
            agent, computed during the individual iterations of the experiment.
    """

    # Compute the true reward landscape
    dim = len(train_environment.observation_space.sample())
    resolution = 20 if dim == 3 else 100
    result = get_reward_landscape(
        train_environment, true_reward=True, return_grid=True, resolution=resolution
    )
    assert isinstance(result, list)
    grid_reward, reward_landscape_true = result
    assert isinstance(grid_reward, list)

    # Get the policy grid
    grid_policy, _ = get_agent_policy(train_environment, agent, return_grid=True)
    assert isinstance(grid_policy, list)

    if len(grid_policy) == 3:
        # Compute the true reward landscape
        result = get_reward_landscape(
            train_environment, true_reward=True, return_grid=True, resolution=resolution
        )
        assert isinstance(result, list)
        grid_reward, reward_landscape_true = result

        # Plot the true reward landscape
        assert isinstance(reward_landscape_true, np.ndarray)
        assert isinstance(grid_reward, list)
        fig = plot_reward_landscapes(
            grid_reward, reward_landscape_true, None, show=False, slices=True
        )
        fig_name = "figures/reward_landscape_true.png"
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
            fig_name = f"figures/reward_landscape_pred_{index}.png"
            mlflow.log_figure(fig, fig_name)

    # Plot the pointmass rollout
    for index, rollout_data in enumerate(test_rollouts):
        assert isinstance(reward_landscape_true, np.ndarray)
        fig = plot_pointmass_rollouts(
            rollout_data,
            grid_reward,
            reward_landscape_true,
            test_reward_landscapes_pred[index],
            show=False,
        )
        fig_name = f"figures/rollout{suffix}_{index}.png"
        mlflow.log_figure(fig, fig_name)

    if len(grid_policy) == 2:

        # Plot the agent policies
        assert isinstance(reward_landscape_true, np.ndarray)
        for index, policy_map in enumerate(test_agent_policies):
            fig = plot_agent_policy(
                grid_reward,
                grid_policy,
                reward_landscape_true,
                test_reward_landscapes_pred[index],
                policy_map,
                show=False,
            )
            fig_name = f"figures/policy_map{suffix}_{index}.png"
            mlflow.log_figure(fig, fig_name)

        # Close all figures
        plt.close("all")


def test_agent_with_reward_learners(
    observation_sampling_mode: str,
    env_config: dict[str, Any],
    reward_model_config: dict[str, Any],  # Unused
    agent_config: dict[str, Any],
    agent_train_config: dict[str, Any],
    callback_configs: Optional[Union[list[dict[str, Any]], dict[str, Any]]] = None,
    seed: Optional[int] = None,
    num_runs: int = 1,
    num_total_queries: int = 50,  # Unused
    num_queries_per_iteration: int = 5,  # Unused
    meta_train: bool = False,  # Unused
    meta_train_size: Optional[int] = None,  # Unused
    num_meta_tasks: Optional[int] = None,  # Unused
    num_meta_valid_tasks: Optional[int] = None,  # Unused
    should_vectorize_environment: bool = False,
    run_id: Optional[str] = None,
    suffix: str = "",
) -> tuple[
    EnvironmentBase,
    list[np.ndarray],
    RLAlgorithm,
    list[float],
    list[float],
    list[list[dict[str, list[Union[np.ndarray, float]]]]],
    list[np.ndarray],
]:
    """The main function of the experiment. Trains an rl agent on a given environment and
    evaluates its performance.

    Args:
        observation_sampling_mode (str): The sampling mode for the observations. Must be
            chosen from 'OBSERVATION_SAMPLING_MODES'.
        env_config (dict[str, Any]): The configuration of the environment.
        reward_model_config (dict[str, Any]): The configuration of the reward model. Unused
            argument. Included for compatibility with the other experiments.
        agent_config (dict[str, Any]): The configuration of the rl agent.
        agent_train_config (dict[str, Any]): The training configuration of the rl agent.
        callback_configs (Optional[Union[list[dict[str, Any]], dict[str, Any]]], optional): The
            configuration of the callbacks. Defaults to None.
        seed (Optional[int], optional): The seed for the experiment. Defaults to None.
        num_runs (int, optional): The number of times to train the rl agent . Defaults to 1.
        num_total_queries (int, optional): The total number of queries to perform during
            the whole experiment. Defaults to 50. Unused argument. Included for compatibility
            with other experiments.
        num_queries_per_iteration (int, optional): The number of queries to perform during
            each iteration. Defaults to 5. Unused argument. Included for compatibility with
            other experiments.
        meta_train (bool, optional): Whether to perform meta-training. Defaults to False.
            Unused argument. Included for compatibility with other experiments.
        meta_train_size (Optional[int], optional): The size of each meta-training
            dataset. Defaults to None. Unused argument. Included for compatibility with
            other experiments.
        num_meta_tasks (Optional[int], optional): The number of meta-training tasks.
            Defaults to None. Unused argument. Included for compatibility with other experiments.
        num_meta_valid_tasks (Optional[int], optional): The number of meta-validation
            tasks. Defaults to None. Unused argument. Included for compatibility with
            other experiments.
        should_vectorize_environment (bool, optional): Whether to vectorize the
            environment. Defaults to False.
        run_id (Optional[str], optional): The run id of the experiment. Defaults to None.
        suffix (str, optional): The suffix string of the experiment. Defaults to "".

    Raises:
        ValueError: If the observation sampling mode is not valid.

    Returns:
        tuple[
            EnvironmentBase,
            list[np.ndarray],
            RLAlgorithm,
            list[float],
            list[float],
            list[list[dict[str, list[Union[np.ndarray, float]]]]],
            list[np.ndarray],
        ]: The environment, the trained rl agent, and per
            iteration information about:
                - the mean rewards,
                - the reward standard deviations,
                - the rl agent rollouts,
                - the learnt reward landscapes,
                - the agent policies.
    """

    # Check input arguments
    if observation_sampling_mode not in OBSERVATION_SAMPLING_MODES:
        raise ValueError(
            f"'observation_sampling_mode' must be one of {OBSERVATION_SAMPLING_MODES}. "
            f"Got '{observation_sampling_mode}' instead."
        )

    # Set the seed
    if seed is not None:
        set_all_seeds(seed)

    # Set the appropriate amount of environments
    env_config["num_tasks"] = num_runs

    # Generate the specified environment
    train_environment = get_environment(env_config)
    train_environment.set_tasks(train_environment.sample_tasks(seed=seed, **env_config))

    # Create the test environment
    test_environment = get_environment(env_config)
    test_environment.set_tasks(test_environment.sample_tasks(seed=seed, **env_config))

    # Let the environments use the true reward
    train_environment.set_use_true_reward(True)
    test_environment.set_use_true_reward(True)

    # Print the environment parameters
    print("TRAIN ENVIRONMENT:\n")
    train_environment.print_task(0)
    print("TEST ENVIRONMENT:\n")
    test_environment.print_task(0)

    # Initialize lists for evaluation
    test_reward_means: list[float] = []
    test_reward_stds: list[float] = []
    test_rollouts: list[list[dict[str, list[Union[np.ndarray, float]]]]] = []
    test_reward_landscapes: list[np.ndarray] = []
    test_agent_policies: list[np.ndarray] = []
    agent = None
    agent_train_environment = None
    callbacks = None

    agent_config["tensorboard_log"] = "./tensorboard/"

    # Compute the dimension of the train environment
    dim = len(train_environment.observation_space.sample())
    resolution = 20 if dim == 3 else 100

    # Perform the experiment 'run_seed' amount of times
    for index in range(num_runs):
        # Explicitly delete unused objects to save memory
        del agent
        del callbacks
        del agent_train_environment

        # Set the correct environment instance
        train_environment.set_current_task(index)
        test_environment.set_current_task(index)

        result = get_reward_landscape(
            train_environment,
            return_grid=False,
            return_maximum=True,
            resolution=resolution,
        )
        assert isinstance(result, list)
        reward_landscape, reward_estimate = result
        assert isinstance(reward_landscape, np.ndarray)

        result = get_reward_landscape(
            train_environment,
            true_reward=True,
            return_grid=False,
            return_minimum=True,
            return_maximum=True,
            resolution=resolution,
        )
        assert isinstance(result, list)
        _, reward_est_min, reward_est_max = result
        assert isinstance(reward_est_min, float)
        assert isinstance(reward_est_max, float)

        test_reward_landscapes.append(reward_landscape)

        # Vectorize the environment if necessary
        if should_vectorize_environment:
            agent_train_environment = vectorize_environment(
                train_environment,
                env_config,
                num_copies=min(multiprocessing.cpu_count(), 16),
                start_method="forkserver",
            )
        else:
            agent_train_environment = train_environment

        # Get the specified agent
        # action_noise = NormalActionNoise(0, 0.1)
        agent = get_agent(
            agent_config,
            environment=agent_train_environment,
        )

        # Get the specified callbacks
        callbacks = None
        if callback_configs is not None:
            callbacks = get_callbacks(callback_configs, agent=agent)

        # Train the policy using the reward model
        agent.learn(
            **agent_train_config,
            tb_log_name=f"{run_id[:6] + '_' if run_id is not None else ''}run{suffix}",
            callback=callbacks,
        )

        # Terminate the subprocesses of the vectorized environment to save memory
        if should_vectorize_environment:
            agent_train_environment.close()

        # Print the callback summaries
        if callbacks is not None:
            for callback in callbacks:
                if hasattr(callback, "print_results"):
                    callback.print_results()  # type: ignore

        # Evaluate the policy
        reward_mean, reward_std = evaluate_policy(
            agent, Monitor(test_environment), deterministic=True
        )
        assert isinstance(reward_mean, float)
        assert isinstance(reward_std, float)

        # Approximately normalize the reward mean
        reward_mean = (reward_mean - reward_est_min) / (reward_est_max - reward_est_min)
        reward_mean_old = reward_mean / reward_est_max

        print(f"reward_mean: {reward_mean};\t reward_std: {reward_std}")
        print(f"Old reward mean: {reward_mean_old}")

        test_reward_means.append(reward_mean)
        test_reward_stds.append(reward_std)

        if DEBUG:
            test_rollouts.append(
                rollout(agent, test_environment, num_episodes=5, deterministic=False)
            )
            policy_map = get_agent_policy(train_environment, agent)
            assert isinstance(policy_map, np.ndarray)
            test_agent_policies.append(policy_map)

    # Log the metrics
    assert isinstance(agent, stable_baselines3.common.base_class.BaseAlgorithm)
    if run_id is not None:
        # print("Step ", sample_idx, ", logging the following metrics: ", metrics)
        with mlflow.start_run(run_id=run_id, nested=True):
            log_metrics(suffix, test_reward_means, test_reward_stds)
            log_figures(
                suffix,
                train_environment,
                agent,
                test_reward_landscapes,
                test_rollouts,
                test_agent_policies,
            )
    else:
        log_metrics(suffix, test_reward_means, test_reward_stds)
        log_figures(
            suffix,
            train_environment,
            agent,
            test_reward_landscapes,
            test_rollouts,
            test_agent_policies,
        )

    return (
        train_environment,
        test_reward_landscapes,
        agent,
        test_reward_means,
        test_reward_stds,
        test_rollouts,
        test_agent_policies,
    )


if __name__ == "__main__":

    # If no command-line arguments have been provided, use the following config
    if len(sys.argv) == 1:
        args: Optional[list[str]] = (
            "--configs "
            # "config_point_env_fpacoh_gp_PPO_small.py "
            # "config_point_env_fpacoh_gp_SAC_small.py "
            # "3d/config_point_env_goal_3d_fpacoh_gp_SAC.py "
            "3d/config_point_env_goal_3d_fpacoh_gp_SAC_small.py "
            # "config_point_env_learnt_gp.py "
            # "config_point_env_pacoh_gp.py "
            # "config_point_env_fpacoh_gp.py "
            "--params experiment.seed=3 experiment.num_runs=1 "
            "--globalparams num_repeats=1"
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

    # For each config, test its agent
    for config in config_list:
        # Run it once if 'num_repeats' == 1
        if num_repeats == 1:
            (
                train_environment,
                test_reward_landscapes,
                agent,
                test_reward_means,
                test_reward_stds,
                test_rollouts,
                test_agent_policies,
            ) = test_agent_with_reward_learners(**config["experiment"])

            print("Reward means =\n", test_reward_means)

        # Otherwise run multiple repeats sequentially
        else:
            # Make sure that each run gets a unique seed and suffix
            config_exp = config["experiment"]
            base_suffix = config_exp.get("suffix", "")
            base_seed = config_exp.get("seed", 0)

            # Run all repeats
            for index in range(num_repeats):
                seed = base_seed + index
                config_exp["seed"] = seed
                config_exp["suffix"] = base_suffix + f"_seed_{seed}"

                # Run a single run
                test_agent_with_reward_learners(**config_exp)
