import os
from typing import Union

import mlflow
import numpy as np

from meta_arl.util.experiments import (
    correct_mlflow_artifact_location,
    gather_metrics_distributed,
    select_mlflow_run,
)
from meta_arl.util.plotting import line_plots

# Build path to mlruns folder
# Folder in same file
# MLFLOW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlruns")
# Folder in main directory
MLFLOW_DIR = os.path.abspath(__file__)
for _ in range(3):
    MLFLOW_DIR = os.path.dirname(MLFLOW_DIR)
MLFLOW_DIR = os.path.join(MLFLOW_DIR, "mlruns")


def compute_mean_and_std(
    metric_dic: dict[str, Union[str, list[dict[str, list[float]]]]]
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Compute mean and standard deviation of all metrics in the metric dictionary.

    Args:
        metric_dic (dict[str, Union[str, list[dict[str, list[float]]]]]): A dictionary
            containing the metrics of all runs.

    Returns:
        dict[str, dict[str, dict[str, np.ndarray]]]: A dictionary containing the
            mean and standard deviation of all metrics.
    """
    results: dict[str, dict[str, dict[str, np.ndarray]]] = {}

    # Iterate over the different models
    for key in metric_dic.keys():
        # Skip the parent run
        if key == "MAIN":
            continue

        results[key] = {}

        # Get all metric names
        for index in range(len(metric_dic[key])):
            metric_sub_dic = metric_dic[key][index]
            assert isinstance(metric_sub_dic, dict)
            if len(metric_sub_dic.keys()) > 0:
                metric_names = metric_sub_dic.keys()
                break
        else:
            continue

        # For each metric, compute avg and std
        metric_sub_dic2 = metric_dic[key]
        assert isinstance(metric_sub_dic2, list)
        for metric_name in metric_names:
            extracted: np.ndarray = np.array(
                [r[metric_name] for r in metric_sub_dic2 if metric_name in r]
            )
            results[key][metric_name] = {}
            results[key][metric_name]["mean"] = np.mean(extracted, axis=0)
            results[key][metric_name]["std"] = np.std(extracted, axis=0)

    return results


if __name__ == "__main__":
    experiment_name = "Test rl agent with reward learners"

    plot_order = [
        "test reward means",
        # "test reward stds",
        "test rmses",
    ]

    # Select the runs to plot
    main_run_id = select_mlflow_run(experiment_name=experiment_name)

    # First fix the artifact location (necessary if you ran the experiments on
    # Euler and then downloaded the results)
    correct_mlflow_artifact_location(
        mlflow_dir=MLFLOW_DIR, experiment_name=experiment_name, run_id=main_run_id
    )

    # Gather all metrics
    metrics = gather_metrics_distributed(main_run_id=main_run_id, mlflow_dir=MLFLOW_DIR)

    # Compute the mean and std of all data series
    results = compute_mean_and_std(metric_dic=metrics)

    # Change the order of the dictionary
    temp = {}
    for model_name in results:
        for key in plot_order:
            temp[key] = results[model_name][key]
        results[model_name] = dict(temp)

    # Prepare the data for plotting
    model_names = list(results.keys())
    metric_names = list(results[model_names[0]].keys())
    num_plots = len(metric_names)

    num_steps = len(results[model_names[0]][metric_names[0]]["mean"])
    x = np.linspace(2, 20, 11)
    # x = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])
    y_arrays: list[list[np.ndarray]] = []
    y_error_bars: list[list[np.ndarray]] = []
    for metric_name in metric_names:
        y_arrays.append([results[model_name][metric_name]["mean"] for model_name in model_names])
        y_error_bars.append(
            [results[model_name][metric_name]["std"] for model_name in model_names]
        )

    # Create better readable legend names
    name_map = {
        "GP_mean:zero_kernel:SE": "GP",
        "fpacoh_GP_mean:NN_kernel:NN": "F-PACOH",
        "fpacoh_GP_mean:NN_kernel:NN_meta_0010": "F-PACOH 10 meta tasks",
        "fpacoh_GP_mean:NN_kernel:NN_meta_0020": "F-PACOH 20 meta tasks",
        "fpacoh_GP_mean:NN_kernel:NN_meta_0050": "F-PACOH 50 meta tasks",
        "fpacoh_GP_mean:NN_kernel:NN_meta_0100": "F-PACOH 100 meta tasks",
        "fpacoh_GP_mean:NN_kernel:NN_meta_0500": "F-PACOH 500 meta tasks",
        "fpacoh_GP_mean:NN_kernel:NN_meta_1000": "F-PACOH 1000 meta tasks",
        "GP_mean:zero_kernel:SE_noise_PPO": "PPO",
        "GP_mean:zero_kernel:SE_noise_SAC": "SAC",
        "fpacoh_GP_mean:NN_kernel:NN_PPO": "F-PACOH + PPO",
        "fpacoh_GP_mean:NN_kernel:NN_SAC": "F-PACOH + SAC",
        "GP_mean:zero_kernel:SE_PPO": "GP + PPO",
        "GP_mean:zero_kernel:SE_SAC": "GP + SAC",
        "goal_env_fpacoh_GP_mean:NN_kernel:NN_PPO": "F-PACOH + SAC",
        "goal_env_3d_simple_GP_SAC": "GP + SAC",
        "goal_env_simple_GP_SAC": "GP + SAC",
        "goal_env_fpacoh_GP_SAC": "F-PACOH + SAC",
        "goal_env_3d_fpacoh_GP_SAC": "F-PACOH + SAC",
        "goal_env_3d_fpacoh_GP_SAC_double": "F-PACOH + SAC",
    }

    model_names = [name_map[name] for name in model_names]

    # Plot the results
    fig = line_plots(
        num_plots=num_plots,
        x_arrays=x,
        y_arrays=y_arrays,
        y_error_bars=y_error_bars,
        y_error_bars_layout="continuous",
        sup_title="Combining Reinforcement Learning with Reward Learning",
        titles=metric_names,
        x_axis_names="Number of observed data points",
        # x_axis_names="Standard deviation of added noise",
        y_axis_names=metric_names,
        # y_axis_names=["Mean approximately normalized reward", "something else"],
        labels=[model_names for _ in range(num_plots)],
        y_limits=(None, None),
        plot_arrangement_shape="auto",
        save_path=None,
        plot_legends=True,
        save=False,
        plot=True,
    )

    # Store the figure
    with mlflow.start_run(run_id=main_run_id):

        # Log the figure
        mlflow.log_figure(figure=fig, artifact_file="metrics.png")
