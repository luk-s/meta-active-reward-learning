import json
import os
from typing import Optional, Union

import mlflow
import numpy as np

from meta_arl.util.experiments import correct_mlflow_artifact_location
from meta_arl.util.plotting import line_plots

# Build path to mlruns folder
# Folder in same file
# MLFLOW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlruns")
# Folder in main directory
MLFLOW_DIR = os.path.abspath(__file__)
for _ in range(3):
    MLFLOW_DIR = os.path.dirname(MLFLOW_DIR)
MLFLOW_DIR = os.path.join(MLFLOW_DIR, "mlruns")
RESULT_DIR = os.path.dirname(os.path.abspath(__file__))


def process_metrics(
    run_id: str, min_runs_completed: Optional[int] = None
) -> dict[str, np.ndarray]:
    """Extract the metrics of a single run.

    Args:
        run_id (str): The id of the run to extract the metrics from.
        min_runs_completed (Optional[int], optional): The minimum number of
            times each metric has to be computed in order to be included
            in the results. Defaults to None.

    Returns:
        dict[str, np.ndarray]: A dictionary containing the extracted metrics
            of the specified run.
    """
    # Get the specified run
    run = mlflow.get_run(run_id=run_id)

    metric_names = run.data.metrics.keys()

    metric_base_names = [m[: m.index("_seed")] for m in metric_names]
    metric_base_names = list(set(metric_base_names))

    # Create a client to access the metrics
    client = mlflow.tracking.MlflowClient()

    metrics = {}

    # Iterate over all metric base names
    for metric_base_name in metric_base_names:
        # Temp list which will be used to average
        metrics_temp = []

        # Iterate over all metrics which have the same basename
        for metric_name in [m for m in metric_names if m[: m.index("_seed")] == metric_base_name]:
            # Extract the metric and append it to the temp list
            metric_list = client.get_metric_history(run_id, metric_name)
            metrics_temp.append([m.value for m in metric_list])

        # Average the metrics
        metrics_temp_np = np.array(metrics_temp)
        if min_runs_completed is not None and metrics_temp_np.shape[0] < min_runs_completed:
            pass  # Do nothing
        else:
            metrics[metric_base_name] = np.mean(metrics_temp_np, axis=0)

    # Return the results
    return metrics


def process_metrics_distributed(
    main_run_id: str,
    rel_file_path: str = "",
    run_id_file_name: str = "run_ids.json",
    min_runs_completed: Optional[int] = None,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Extract the metrics of all runs belonging to a single experiment.

    Args:
        main_run_id (str): The id of the main run of the experiment.
        rel_file_path (str, optional): The relative path to the file containing
            the run ids. Defaults to "".
        run_id_file_name (str, optional): The name of the file containing the
            run ids. Defaults to "run_ids.json".
        min_runs_completed (Optional[int], optional): The minimum number of
            times each metric has to be computed in order to be included
            in the results. Defaults to None.

    Raises:
        TypeError: If the file containing the run ids contains faulty data.

    Returns:
        dict[str, dict[str, dict[str, np.ndarray]]]: A dictionary containing
            the extracted metrics of all runs belonging to the specified
            experiment.
    """
    # Get the specified run
    run = mlflow.get_run(run_id=main_run_id)

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
    results: dict[str, dict[str, dict[str, np.ndarray]]] = {}

    # Iterate over all different models
    for key in run_ids:
        results[key] = {}

        if isinstance(run_ids[key], str):
            # Ignore the main run id
            if run_ids[key] == main_run_id:
                del results[key]
                continue

        elif isinstance(run_ids[key], dict):
            results[key] = {}
            # Iterate over all run_ids of this task
            for run_name in run_ids[key]:
                print(f"Processing {run_name}")
                metrics_dic = process_metrics(run_ids[key][run_name], min_runs_completed)
                results[key][run_name] = metrics_dic
        else:
            raise TypeError(f"No implementation available for object of type {type(run_ids[key])}")

    return results


def find_best_runs(
    metrics: dict[str, dict[str, dict[str, np.ndarray]]],
    metric_name: str,
    direction: str = "largest",
    mode: str = "last",
) -> tuple[dict[str, dict[str, Union[str, float]]], dict[str, list[tuple[float, str]]]]:
    """Find the best run of an experiment w.r.t. a specified metric and return a dictionary
    or lists of the runs sorted by their performance on the specified metric.

    Args:
        metrics (dict[str, dict[str, dict[str, np.ndarray]]]): The extracted metrics of
            the experiment.
        metric_name (str): The name of the metric for which the best runs should be found.
        direction (str, optional): The direction in which the metric should be optimized.
            Can be either "largest" or "smallest". Defaults to "largest".
        mode (str, optional): Whether the last value of each metric list should be used
            or the minimum/maximum value. Can be either "last" or "max". Defaults to "last".

    Raises:
        ValueError: If the specified direction is not "largest" or "smallest".
        ValueError: If the specified mode is not "last" or "max".

    Returns:
        tuple[
            dict[str, dict[str, Union[str, float]]],
            dict[str, list[tuple[float, str]]]
        ]: A tuple containing a dictionary containing the best run of the experiment
            and a dictionary containing a dictionary of list of all runs sorted by
            their performance on the specified metric.
    """
    direction_values = ["smallest", "largest"]
    mode_values = ["best", "last"]

    if direction not in direction_values:
        raise ValueError(
            f"The direction {direction} is not supported! " f"Choose one from {direction_values}"
        )

    if mode not in mode_values:
        raise ValueError(f"The mode {mode} is not supported! " f"Choose one from  {mode_values}")

    results: dict[str, dict[str, Union[str, float]]] = {}
    result_lists: dict[str, list[tuple[float, str]]] = {}

    # Iterate over different models
    for key in metrics:
        best_run = None
        best_run_value = 0

        result_list: list[tuple[float, str]] = []

        # Iterate over all runs of this model
        for run_name in metrics[key]:
            # Get value to compare against
            metric_list = np.array(metrics[key][run_name][metric_name])
            if mode == "last":
                metric_value = metric_list[-1]
            elif mode == "best":
                if direction == "largest":
                    metric_value = metric_list.max()
                elif direction == "smallest":
                    metric_value = metric_list.min()
                else:
                    raise ValueError(f"The direction '{direction}' is unknown!")
            else:
                raise ValueError(f"The mode '{mode}' is not known!")

            # Add the metric value to the result list
            result_list.append((metric_value, run_name))

            # Check if the metric value is better than the previously best value
            metric_is_better = False
            if (direction == "largest" and metric_value > best_run_value) or (
                direction == "smallest" and metric_value < best_run_value
            ):
                metric_is_better = True

            # Update the best run if necessary
            if best_run is None or metric_is_better:
                best_run = run_name
                best_run_value = metric_value
        assert best_run is not None
        results[key] = {"run_id": best_run, metric_name: best_run_value}
        result_lists[key] = sorted(result_list, reverse=True)  # type: ignore

    return results, result_lists


if __name__ == "__main__":
    ################################################
    #                    CONFIG                    #
    ################################################
    experiment_name = "Hyperparameter tuning"
    main_run_id = "585ef9721b024af8a25942a774e9f506"
    run_ids_file_name = "run_ids_0-end.json"
    metric_names = [
        "test reward means",
        "test rmses",
        "test reward stds",
    ]
    min_runs_completed = 1

    metric_name = "test reward means"
    direction = "largest"
    mode = "last"

    ################################################
    #                   PROGRAM                    #
    ################################################

    # First fix the artifact location (necessary if you ran the experiments on
    # Euler and then downloaded the results)
    correct_mlflow_artifact_location(
        mlflow_dir=MLFLOW_DIR, experiment_name=experiment_name, run_id=main_run_id
    )

    # Gather all metrics
    metrics = process_metrics_distributed(
        main_run_id=main_run_id,
        run_id_file_name=run_ids_file_name,
        min_runs_completed=min_runs_completed,
    )

    # Sort out all runs which have missing values
    for key in metrics:
        to_delete = []

        # Iterate over all runs of this model
        for run_name in metrics[key]:
            # Check if this run contains all required metrics
            if set(metrics[key][run_name].keys()) != set(metric_names):
                to_delete.append(run_name)

        # Delete all runs with missing values
        for run_name in to_delete:
            del metrics[key][run_name]

        # Indicate how many successful runs this model has
        print(f"Model {key}: {len(metrics[key])} runs were successful.")

    best_runs, all_runs = find_best_runs(
        metrics, metric_name=metric_name, direction=direction, mode=mode
    )

    print("Best runs = ", best_runs)

    # Write the result of all runs into a file
    with open(f"{RESULT_DIR}/hyp_results/results_{main_run_id[:5]}.txt", "w") as f:
        for key in all_runs:
            f.write(f"{key}:\n{len(key) * '='}\n")
            for (value, name) in all_runs[key]:
                f.write(f"\t{value:.3}\t{name}\n")
