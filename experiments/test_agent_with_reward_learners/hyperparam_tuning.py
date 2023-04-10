# import multiprocessing as mp
import itertools
import sys
from datetime import datetime
from typing import Optional

import matplotlib.colors as mcolors
import mlflow

from meta_arl.config.global_config import (
    ARTIFACT_PATH,
    BASE_DIR,
    IGNORE_DIRECTORY_PATTERNS,
    IGNORE_FILE_PATTERNS,
)
from meta_arl.util.experiments import build_run_command, generate_run_commands
from meta_arl.util.helpers import zip_directory
from meta_arl.util.input import ConfigCreator

NO_SIMULTANEOUS_MEASUREMENTS = len(list(mcolors.TABLEAU_COLORS))


if __name__ == "__main__":

    # Set the experiment
    experiment = mlflow.set_experiment("Hyperparameter tuning")
    num_cpus_per_job = 2

    # If no command-line arguments have been provided, use the following config
    if len(sys.argv) == 1:
        args: Optional[list[str]] = (
            "--configs "
            # "config_point_env_pacoh_gp_hyperparam.py "
            # "config_point_env_fpacoh_gp_hyperparam.py "
            # "config_point_env_fpacoh_gp_hyperparam_based_on_learnt.py"
            "config_point_env_simple_gp_hyperparam.py "
            # "config_point_env_learnt_gp_hyperparam.py"
            "--globalparams params_to_optimize=rl_agent::str"
        ).split()
    else:
        args = None

    global_config, config_list = ConfigCreator.create_from_args(
        args_parsed_list=args, configs_path=None
    )

    # Extract global variables if they exist
    main_run_id = None
    start_run_index = 0
    num_param_comb = None
    params_to_optimize = "reward_model"
    if global_config is not None:
        main_run_id = global_config.get("main_run:id", main_run_id)
        start_run_index = global_config.get("start_run_index", start_run_index)
        num_param_comb = global_config.get("num_param_comb", num_param_comb)
        params_to_optimize = global_config.get("params_to_optimize", params_to_optimize)
        num_cpus_per_job = global_config.get("num_cpus_per_job", num_cpus_per_job)

    if num_param_comb is None:
        run_index_range = f"{start_run_index}-end"
    else:
        run_index_range = f"{start_run_index}-{start_run_index + num_param_comb - 1}"

    # Check that we didn't create too many configs
    if len(config_list) > NO_SIMULTANEOUS_MEASUREMENTS:
        raise OverflowError(
            f"This experiment currently only supports at most "
            f"{NO_SIMULTANEOUS_MEASUREMENTS} configurations at the same time."
        )

    # Convert to a list if necessary
    if not isinstance(config_list, list):
        config_list = [config_list]

    # Get current time
    current_time = datetime.now().strftime("%Y-%m-%d_%H:%M")

    # Start parent run
    with mlflow.start_run(
        run_id=main_run_id,
        experiment_id=experiment.experiment_id,
        run_name="MAIN_" + current_time,
    ) as run:

        run_ids = {"MAIN": run.info.run_id}

        # Zip all code files to make it easier to log them
        zip_directory(
            directory_to_zip=BASE_DIR,
            zip_file_path=ARTIFACT_PATH,
            ignore_directory_patterns=IGNORE_DIRECTORY_PATTERNS,
            ignore_file_patterns=IGNORE_FILE_PATTERNS,
        )

        # Log the code files used to run these experiments
        mlflow.log_artifact(local_path=ARTIFACT_PATH)

        run_commands = []
        log_paths = []
        run_index = 0

        # For each config, perform hyperparameter tuning
        for config in config_list:

            run_ids[config["name"]] = {}

            # Log the config used
            mlflow.log_dict(config, artifact_file="configs/config_" + config["name"] + ".json")

            # Get the path to this config file
            path = config["source_path"]
            config_dir_path = ConfigCreator.config_dir_path
            assert config_dir_path is not None
            path = path[len(config_dir_path) + 1 :]

            # Extract parameters over which a search shall be performed
            param_names, param_lists = [], []
            if params_to_optimize == "reward_model":
                param_dict = config["experiment"]["reward_model_config"]["config"]
            elif params_to_optimize == "rl_agent":
                param_dict = config["experiment"]["agent_config"]
            for key, value in param_dict.items():
                if isinstance(value, list):
                    param_names.append(key)
                    param_lists.append(value)

            # Get all parameter combinations
            combinations = itertools.product(*param_lists)

            # Iterate over all parameter combinations
            for combination in combinations:
                if run_index < start_run_index:
                    run_index += 1
                    continue
                elif num_param_comb is not None and run_index >= start_run_index + num_param_comb:
                    break
                run_index += 1

                tuples = list(zip(param_names, combination))
                suffix = "_" + "_".join([f"{t[0]}_{t[1]}" for t in tuples])
                suffix = suffix.replace(" ", "")
                suffix = suffix.replace("'", "")
                suffix = suffix.replace('"', "")

                # Create a child run for every single model
                with mlflow.start_run(run_name=config["name"] + suffix, nested=True) as child_run:

                    # Store the run id of this run
                    child_run_id = child_run.info.run_id
                    run_ids[config["name"]][suffix[1:]] = child_run_id

                    # Initialize a dictionary with parameters which should be changed
                    # in the default config

                    params = {
                        "experiment.run_id": (child_run_id, "str"),
                    }
                    config_params = {}
                    if params_to_optimize == "reward_model":
                        base_name = "experiment.reward_model_config.config."
                    elif params_to_optimize == "rl_agent":
                        base_name = "experiment.agent_config."
                    for name, value in tuples:
                        params[base_name + name] = (
                            value,
                            str(type(value).__name__),
                        )
                        config_params[name] = value

                        # Add the gradient steps parameter if necessary
                        if name == "train_freq":
                            params[base_name + "gradient_steps"] = (
                                value,
                                str(type(value).__name__),
                            )
                            config_params["gradient_steps"] = value

                    # Log the parameters
                    mlflow.log_params(config_params)

                    # Set a few global parameters
                    global_params = {"num_repeats": 30}

                    # Create the run commands
                    run_commands.append(
                        build_run_command(
                            path=path,
                            command="python test_agent.py",
                            params=params,
                            global_params=global_params,
                        )
                    )
                    log_paths.append(f"logs/output_{current_time}/{config['name']}" + suffix)

        # Log all the run ids, so we can extract the results later on
        mlflow.log_dict(run_ids, artifact_file=f"run_ids_{run_index_range}.json")

        # Run all commands
        generate_run_commands(
            run_commands,
            num_cpus_per_job=num_cpus_per_job,
            num_gpus_per_job=0,
            mem_per_job=8000,
            mode="euler",
            long_execution_time=True,
            log_files=log_paths,
            dry=False,
        )
