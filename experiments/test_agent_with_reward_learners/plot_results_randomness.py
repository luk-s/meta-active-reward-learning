import json
import os
from itertools import groupby
from tokenize import group
from typing import Optional, Union

import mlflow
import numpy as np

from meta_arl.util.experiments import (
    correct_mlflow_artifact_location,
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


def group_by_and_compute_mean_and_std(
    metric_dic: dict[str, Union[str, list[dict[str, int]]]],
    group_by: str,
    group_names: list[str],
    plot_type: str = "mean",
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """
    Groups the results stored in the metric_dic by the group_by key and computes the mean and std
    of the metrics for that group.

    Args:
        metric_dic (dict[str, Union[str, list[dict[str, int]]]]): Dictionary containing the results
        group_by (str): Key to group the results by
        group_names (list[str]): Names of the groups which exist in the metric_dic
        plot_type (str, optional): Type of plot to create. Defaults to "mean".

    Returns:
        dict[str, dict[str, dict[str, np.ndarray]]]: Dictionary containing the grouped results
    """
    assert isinstance(group_names, list), "'group_names' must be a list!"
    assert group_by in group_names, "'group_by' key must be in 'group_names'"
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
                metric_names_keys = metric_sub_dic.keys()
                metric_names = [m for m in metric_names_keys if m not in group_names]
                break
        else:
            continue

        # Group the results
        grouped: dict[int, list[dict[str, int]]] = {}
        for dic in metric_dic[key]:
            assert isinstance(dic, dict)
            group_by_value = dic[group_by]
            if group_by_value not in grouped:
                grouped[group_by_value] = []
            grouped[group_by_value].append(dic)

        # For each metric compute avg and std
        for metric_name in metric_names:
            grouped_list = []
            for group_key, group_value in grouped.items():
                print(f"{group_key = }, {len(group_value)}")
                extracted = np.array([r[metric_name] for r in group_value if metric_name in r])
                grouped_list.append(np.mean(extracted, axis=0))

            # Average over the grouped list as well
            grouped_array = np.array(grouped_list)
            results[key][metric_name] = {}

            print("last ones = ", grouped_array.T[-1, :])

            if plot_type == "mean":
                results[key][metric_name]["mean"] = np.mean(grouped_array, axis=0)
                results[key][metric_name]["std"] = np.std(grouped_array, axis=0)
            else:
                results[key][metric_name]["all"] = grouped_array.T

    return results


def gather_metrics(run_id: str) -> dict[str, int]:
    """Extract the metrics of a single run.

    Args:
        run_id (str): ID of the run to extract the metrics from

    Returns:
        dict[str, int]: Dictionary containing the metrics
    """
    # Get the specified run
    run = mlflow.get_run(run_id=run_id)

    metric_names = run.data.metrics.keys()

    # Create a client to access the metrics
    client = mlflow.tracking.MlflowClient()

    # Extract the seeds which were used for this run
    run_name = run.data.tags["mlflow.runName"]
    seeds = list(filter(lambda x: x.isnumeric(), run_name.split("_")))
    metrics = {"env": int(seeds[0]), "rm": int(seeds[1]), "rl": int(seeds[2])}

    # Fetch all metrics
    for metric_name in metric_names:
        metric_list = client.get_metric_history(run_id, metric_name)
        metric_list = [m.value for m in metric_list]

        metrics[metric_name] = metric_list

    # Return the results
    return metrics


def gather_metrics_distributed(
    main_run_id: str,
    rel_file_path: str = "",
    run_id_file_name: str = "run_ids.json",
    mlflow_dir: Optional[str] = None,
) -> dict[str, Union[str, list[dict[str, int]]]]:
    """Extract the metrics of all runs belonging to a single experiment.

    Args:
        main_run_id (str): ID of the main run
        rel_file_path (str, optional): Relative path to the file containing the run ids.
            Defaults to "".
        run_id_file_name (str, optional): Name of the file containing the run ids. Defaults to
            "run_ids.json".
        mlflow_dir (Optional[str], optional): Path to the mlflow directory. This path is used
            to correct the paths inside the 'meta.yaml' files. This becomes necessary if the
            experiment was run on a different machine. Defaults to None.

    Raises:
        TypeError: If the file containing the run ids contains faulty data.

    Returns:
        dict[str, Union[str, list[dict[str, int]]]]: A dictionary containing
            the extracted metrics of all runs belonging to the specified
            experiment.
    """
    # Get the specified run and experiment
    run = mlflow.get_run(run_id=main_run_id)
    experiment = mlflow.get_experiment(run.info.experiment_id)

    # Get the artifact uri
    artifact_path = run.info.artifact_uri
    if "///" in artifact_path:
        artifact_path = artifact_path[artifact_path.index("///") + 2 :]

    # Build the path to the run_ids.json file
    id_path = os.path.join(artifact_path, rel_file_path, run_id_file_name)

    # Load the run ids
    with open(id_path, "r") as id_file:
        content = id_file.read()
        run_ids = json.loads(content)

    # Create result dictionary
    results: dict[str, Union[str, list[dict[str, int]]]] = {}

    # Iterate over all tasks
    for key in run_ids:
        if isinstance(run_ids[key], str):
            # Keep the main run id as is
            if run_ids[key] == main_run_id:
                results[key] = main_run_id
                continue

            # Otherwise just wrap the run id to a list
            run_ids[key] = [run_ids[key]]

        if isinstance(run_ids[key], list):
            results[key] = []
            # Iterate over all run_ids of this task
            for run_id in run_ids[key]:
                # First fix the artifact path
                if mlflow_dir is not None:
                    correct_mlflow_artifact_location(
                        mlflow_dir=mlflow_dir,
                        experiment_name=experiment.name,
                        run_id=run_id,
                    )
                metrics_dic = gather_metrics(run_id)
                results[key].append(metrics_dic)  # type: ignore
        else:
            raise TypeError(f"No implementation available for object of type {type(run_ids[key])}")

    return results


if __name__ == "__main__":
    experiment_name = "Test rl agent with reward learners"

    plot_order = [
        "test reward means",
        "test reward stds",
    ]

    plot_type = "mean"  # one of ["mean", "all"]

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
    result_list = [
        group_by_and_compute_mean_and_std(
            metric_dic=metrics,
            group_by=group_by_key,
            group_names=["env", "rm", "rl"],
            plot_type=plot_type,
        )
        for group_by_key in ["env", "rm", "rl"]
    ]

    result_name_suffixes = ["environment", "reward model", "reinforcement learning"]

    # Change the order of the dictionary
    for index, (suffix, results) in enumerate(zip(result_name_suffixes, result_list)):
        temp = {}
        for model_name in results:
            for key in plot_order:
                temp[key] = results[model_name][key]
            result_list[index][model_name] = dict(temp)

        # Prepare the data for plotting
        model_names = list(results.keys())
        metric_names = list(results[model_names[0]].keys())
        num_plots = len(metric_names)

        num_steps = len(list(results[model_names[0]][metric_names[0]].items())[0][1])
        x = np.linspace(2, 20, 11)
        # x = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])
        y_arrays = []
        y_error_bars = []
        for metric_name in metric_names:
            if plot_type == "mean":
                y_arrays.append(
                    [results[model_name][metric_name]["mean"] for model_name in model_names]
                )
                y_error_bars.append(
                    [results[model_name][metric_name]["std"] for model_name in model_names]
                )
            elif plot_type == "all":
                y_arrays.append(
                    [
                        results[model_name][metric_name]["all"].tolist()
                        for model_name in model_names
                    ]
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
            "goal_env_3d_fpacoh_GP_SAC": "F-PACOH + SAC",
            "goal_env_simple_GP_SAC": "GP + SAC",
            "goal_env_fpacoh_GP_SAC": "F-PACOH + SAC",
        }

        model_names = [name_map[name] for name in model_names]

        # Plot the results
        fig = line_plots(
            num_plots=num_plots,
            x_arrays=x,
            y_arrays=y_arrays,
            y_error_bars=y_error_bars if len(y_error_bars) > 0 else None,
            y_error_bars_layout="continuous",
            sup_title=f"Impact of {suffix} noise",
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
            mlflow.log_figure(figure=fig, artifact_file=f"metrics_{suffix}_noise_{plot_type}.png")
