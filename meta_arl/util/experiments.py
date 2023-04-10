import json
import multiprocessing
import os
import re
from datetime import datetime
from typing import Any, Callable, Iterable, Optional, Union, cast
from unittest import result

import mlflow
import numpy as np
import torch
import yaml


class AsyncExecutor:
    """A helper class which allows to run experiments in parallel."""

    def __init__(self, n_jobs: int = 1):
        """Initialize the AsyncExecutor.

        Args:
            n_jobs (int, optional): The number of workers to use. Defaults to 1.
        """
        self.num_workers = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()
        self._pool: list[multiprocessing.Process] = []
        self._populate_pool()

    def run(self, target: Callable, *args_iter: Any, verbose: bool = False) -> None:
        """Run the target function in parallel.

        Args:
            target (Callable): The function to run in parallel.
            verbose (bool, optional): Whether to print debug messages. Defaults to False.
        """
        workers_idle = [False] * self.num_workers
        tasks = list(zip(*args_iter))
        n_tasks = len(tasks)

        while not all(workers_idle):
            for i in range(self.num_workers):
                if not self._pool[i].is_alive():
                    self._pool[i].terminate()
                    if len(tasks) > 0:
                        if verbose:
                            print(n_tasks - len(tasks))
                        next_task = tasks.pop(0)
                        self._pool[i] = _start_process(target, next_task)
                    else:
                        workers_idle[i] = True

    def _populate_pool(self) -> None:
        """Populate the pool with processes."""
        self._pool = [_start_process(_dummy_fun) for _ in range(self.num_workers)]


def _start_process(target: Callable, args: Optional[tuple] = None) -> multiprocessing.Process:
    """Start the command 'target' in a new process.

    Args:
        target (Callable): The function to run in a new process
        args (Optional[tuple], optional): Optional arguments for the function. Defaults to None.

    Returns:
        multiprocessing.Process: The process which is running the function
    """
    if args:
        p = multiprocessing.Process(target=target, args=args)
    else:
        p = multiprocessing.Process(target=target)
    p.start()
    return p


def _dummy_fun() -> None:
    pass


def build_run_command(
    path: str, command: str, params: dict[str, Any] = {}, global_params: dict[str, Any] = {}
) -> str:
    """Given a command and a set of (global) parameters, build the command to run the experiment.

    Args:
        path (str): The path to the config file of the experiment
        command (str): The command to run
        params (dict[str, Any], optional): The parameters to use for the experiment. Defaults to {}.
        global_params (dict[str, Any], optional): The global parameters to use for the experiment. Defaults to {}.

    Returns:
        str: The command to run the experiment
    """
    # Create the base path
    command += f" --config {path} "

    def add_params(params_dict: dict[str, Any]) -> str:
        cmd = ""
        for key, value in params_dict.items():
            # Check if value also contains an accompanying type
            if isinstance(value, tuple):
                value, val_type = value
            else:
                val_type = None

            # Convert 'value' into a string without any spaces
            value = str(value)
            value = re.sub(r"[\n\t\s]*", "", value)

            # Build the parameter string for this parameter
            if val_type is not None:
                cmd += f"{key}={value}::{val_type} "
            else:
                cmd += f"{key}={value} "
        return cmd

    # If custom parameters have been specified, add them
    if len(params) > 0:
        command += "--params "
        command += add_params(params)

    # If custom global parameters have been specified, add them
    if len(global_params) > 0:
        command += "--globalparams "
        command += add_params(global_params)

    return command


def generate_run_commands(
    command_list: list[str],
    num_cpus_per_job: int = 1,
    num_gpus_per_job: int = 0,
    dry: bool = False,
    n_host_per_job: int = 1,
    mem_per_job: int = 6000,
    long_execution_time: bool = False,
    custom_execution_time: Optional[int] = None,
    mode: str = "local",
    promt: bool = True,
    log_files: list[str] = [],
) -> None:
    """Given a set of commands, group them and run them all on a given system. This can be either
    a local machine or a cluster (Euler).

    Args:
        command_list (list[str]): The list of commands to run
        num_cpus_per_job (int, optional): The number of CPUs to use per job. Defaults to 1.
        num_gpus_per_job (int, optional): The number of GPUs to use per job. Defaults to 0.
        dry (bool, optional): Whether to print the commands instead of running them.
            Defaults to False.
        n_host_per_job (int, optional): The number of host machines to use per job. Defaults to 1.
        mem_per_job (int, optional): The amount of memory to use per job in MB. Defaults to 6000.
        long_execution_time (bool, optional): Whether to use a long execution time (23 hours) per
            job. Defaults to False.
        custom_execution_time (Optional[int], optional): A custom execution time in hours.
            Defaults to None.
        mode (str, optional): The mode to use. Can be either 'local', 'local_async' or 'euler'.
            Defaults to "local".
        promt (bool, optional): Whether to promt the user before running the commands.
            Defaults to True.
        log_files (list[str], optional): A list of log files to use for the output of the commands.
            Defaults to [].

    Raises:
        TypeError: If 'log_files' is not a list of strings
        ValueError: If the 'log_files' list does not have a length of {1,2, len(command_list)}
        NotImplementedError: If the 'mode' is not supported
    """
    if not isinstance(log_files, list):
        raise TypeError("'log_files' must be iterable!")
    if len(log_files) > 1 and len(log_files) != len(command_list):
        raise ValueError(
            "The 'log_files' parameter must be a list whose length is one of "
            f"[0, 1, {len(command_list)} == len(command_list)]. Got {len(log_files)}"
            " instead."
        )

    if mode == "euler":
        if custom_execution_time is None:
            execution_time = 23 if long_execution_time else 3
        else:
            execution_time = custom_execution_time

        cluster_cmds = []
        sbatch_cmd = (
            "sbatch "
            + f"--time={execution_time}:59:00 "
            + f"--mem-per-cpu={mem_per_job} "
            + (f"--gpus={num_gpus_per_job} " if num_gpus_per_job > 0 else "")
            + f"--cpus-per-task={num_cpus_per_job} "
            + "-n 1 "
        )

        for index, python_cmd in enumerate(command_list):
            # Check if a log_file path has been specified
            if log_files != []:
                if len(log_files) == 1:
                    log_path = log_files[0]
                else:
                    log_path = log_files[index]

                # Check if the log directory already exists.
                # If not, create it.
                dirname = os.path.dirname(log_path)
                if not os.path.isdir(dirname):
                    os.makedirs(dirname)

                log_cmd = f'--output="{log_path}" --error="{log_path}" '
            else:
                log_cmd = ""

            # Escape some special characters
            python_cmd = re.sub(r"([',])", r"\\\1", python_cmd)

            # Wrap python command into double quotes
            if python_cmd[0] != '"':
                python_cmd = f'"{python_cmd}"'
            cluster_cmds.append(sbatch_cmd + log_cmd + "--wrap=" + python_cmd)

        if promt:
            [print(cmd) for cmd in cluster_cmds]
            answer = input(
                f"About to submit {len(cluster_cmds)} compute jobs to the cluster. "
                "Proceed? [yes/no]"
            )
        else:
            answer = "yes"
        if answer == "yes":
            for cmd in cluster_cmds:
                if dry:
                    print(cmd)
                else:
                    os.system(cmd)

    elif mode == "local":
        # Extend commands by log paths
        command_list_ext = []
        for index, python_cmd in enumerate(command_list):
            if log_files != []:
                if len(log_files) == 1:
                    log_path = log_files[0]
                else:
                    log_path = log_files[index]

                # Check if the log directory already exists.
                # If not, create it.
                dirname = os.path.dirname(log_path)
                if not os.path.isdir(dirname):
                    os.makedirs(dirname)

                log_cmd = f" 2>&1 | tee {log_path}"
                command_list_ext.append("nohup " + python_cmd + log_cmd + " &")
            else:
                command_list_ext.append("nohup " + python_cmd + " & ")
        if promt:
            [print(cmd) for cmd in command_list_ext]
            answer = input(
                f"About to run {len(command_list_ext)} jobs in a loop. Proceed? [yes/no]"
            )
        else:
            answer = "yes"

        if answer == "yes":
            for cmd in command_list_ext:
                if dry:
                    print(cmd)
                else:
                    os.system(cmd)

    elif mode == "local_async":
        # Extend commands by log paths
        if log_files != []:
            command_list_ext = []
            for index, python_cmd in enumerate(command_list):
                if len(log_files) == 1:
                    log_path = log_files[0]
                else:
                    log_path = log_files[index]

                # Check if the log directory already exists.
                # If not, create it.
                dirname = os.path.dirname(log_path)
                if not os.path.isdir(dirname):
                    os.makedirs(dirname)

                log_cmd = f" 2>&1 | tee {log_path}"
                command_list_ext.append(python_cmd + log_cmd)
        else:
            command_list_ext = command_list
        if promt:
            [print(cmd) for cmd in command_list_ext]
            answer = input(
                f"About to launch {len(command_list_ext)} commands in {num_cpus_per_job}"
                " local processes. Proceed? [yes/no]"
            )
        else:
            answer = "yes"

        if answer == "yes":
            if dry:
                for cmd in command_list:
                    print(cmd)
            else:
                exec = AsyncExecutor(n_jobs=num_cpus_per_job)
                cmd_exec_fun = os.system
                exec.run(cmd_exec_fun, command_list_ext)
    else:
        raise NotImplementedError


def correct_mlflow_artifact_location(mlflow_dir: str, experiment_name: str, run_id: str) -> None:
    """Correct the file paths of the artifacts in a given mlrun experiment folder. Correcting
    means to replace the paths stored in the 'meta.yaml' files of the artifacts with the correct
    paths pointing to these folders.

    Args:
        mlflow_dir (str): Path to the mlflow directory.
        experiment_name (str): Name of the mlflow experiment.
        run_id (str): ID of the run.
    """
    # Get the experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id

    # Build the path to the config file of the specified run
    base_path = os.path.join(mlflow_dir, experiment_id, run_id)
    config_path = os.path.join(base_path, "meta.yaml")
    artifact_uri = "file://" + os.path.join(base_path, "artifacts")

    # Read the config file
    with open(config_path, "r") as config_file:
        content = yaml.safe_load(config_file)

    # Only change the file if necessary
    if content["artifact_uri"] != artifact_uri:
        # Change the artifact path
        content["artifact_uri"] = artifact_uri

        # Write the config file
        with open(config_path, "w") as config_file:
            # Write the changed content back into the file
            yaml.dump(content, config_file)


def gather_metrics(run_id: str) -> dict[str, list[float]]:
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

    metrics = {}

    # Fetch all metrics
    for metric_name in metric_names:
        metric_list = client.get_metric_history(run_id, metric_name)
        metric_list = [m.value for m in metric_list]

        metrics[metric_name] = metric_list

    # Return the results
    return metrics


def select_mlflow_run(experiment_name: str) -> str:
    """Lists all main runs of a given experiment and lets the user select one of them.

    Args:
        experiment_name (str): Name of the experiment to select a run from.

    Raises:
        ValueError: If the user entered an invalid run ID.

    Returns:
        str: The mlflow run id of the selected run.
    """
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        output_format="list",
        filter_string="tags.mlflow.runName LIKE '%MAIN%'",
    )
    info_tuples = [(run.info.run_id, run.info.start_time) for run in runs]
    info_tuples = sorted(info_tuples, key=lambda x: x[1])

    info_tuples = [
        (name, datetime.fromtimestamp(timestamp / 1000.0)) for name, timestamp in info_tuples
    ]

    print("Found the following runs:")
    for index, (run_id, start_time) in enumerate(info_tuples):
        print(f"  [{index}]: {run_id} (started at {start_time.strftime('%Y-%m-%d %H:%M:%S')})")

    while True:
        selected_id_str = input(
            f"Please select the id of the run you want to use (0 - {len(info_tuples) - 1}):"
        )

        try:
            selected_id = int(selected_id_str)
            if selected_id < 0 or selected_id >= len(info_tuples):
                raise ValueError("Invalid id")
            break
        except ValueError:
            print("Invalid id")

    return info_tuples[selected_id][0]


def gather_metrics_distributed(
    main_run_id: str,
    rel_file_path: str = "",
    run_id_file_name: str = "run_ids.json",
    mlflow_dir: Optional[str] = None,
) -> dict[str, Union[str, list[dict[str, list[float]]]]]:
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
        dict[str, Union[str, list[dict[str, list[float]]]]]: A dictionary containing
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
    results: dict[str, Union[str, list[dict[str, list[float]]]]] = {}

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
            result_list: list[dict[str, list[float]]] = []
            results[key] = result_list
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
                result_list.append(metrics_dic)
        else:
            raise TypeError(f"No implementation available for object of type {type(run_ids[key])}")

    return results


def calibration_values(
    lower_conf_bounds: Union[list, np.ndarray, torch.Tensor],
    upper_conf_bounds: Union[list, np.ndarray, torch.Tensor],
    true_values: Union[list, np.ndarray, torch.Tensor],
) -> np.ndarray:
    """Calculate the calibration values for a given set of confidence bounds and true values.

    Args:
        lower_conf_bounds (Union[list, np.ndarray, torch.Tensor]): Lower confidence bounds.
        upper_conf_bounds (Union[list, np.ndarray, torch.Tensor]): Upper confidence bounds.
        true_values (Union[list, np.ndarray, torch.Tensor]): True values.

    Returns:
        np.ndarray: _description_
    """
    arrays = [lower_conf_bounds, upper_conf_bounds, true_values]

    # Convert all inputs to numpy arrays
    for index, array in enumerate(arrays):
        if isinstance(array, torch.Tensor):
            arrays[index] = array.numpy()
        else:
            arrays[index] = np.array(array)
        a = arrays[index]
        assert isinstance(a, np.ndarray)
        assert len(a.shape) <= 2, "Arrays need to be at most two-dimensional!"

    lower_conf_bounds, upper_conf_bounds, true_values = arrays
    assert isinstance(lower_conf_bounds, np.ndarray)

    # Count how many elements are inside the confidence bounds
    inside = np.logical_and(true_values >= lower_conf_bounds, true_values <= upper_conf_bounds)
    freqs = inside.sum(axis=1) / lower_conf_bounds.shape[-1]

    return freqs


if __name__ == "__main__":
    lower_conf_bounds = [[5] * 6, [7] * 6, [9] * 6]

    upper_conf_bounds = [[15] * 6, [13] * 6, [11] * 6]

    true_values = [4, 5, 6, 7, 8, 9]

    results = calibration_values(lower_conf_bounds, upper_conf_bounds, true_values)
