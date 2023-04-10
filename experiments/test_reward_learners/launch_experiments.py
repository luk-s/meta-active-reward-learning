# import multiprocessing as mp
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
    # Track the experiments in the same folder this file is stored
    # mlflow.set_tracking_uri(
    #     "file://" + path.join(path.dirname(prun_and_resultsath.abspath(__file__)), "mlruns")
    # )

    # Set the experiment
    experiment = mlflow.set_experiment("Test reward learner performance")

    # If no command-line arguments have been provided, use the following config
    if len(sys.argv) == 1:
        args: Optional[list[str]] = (
            "--configs "
            "config_point_env_simple_gp.py "
            # "config_point_env_learnt_gp.py "
            # "config_point_env_pacoh_gp.py "
            "config_point_env_fpacoh_gp.py"
        ).split()
    else:
        args = None

    _, config_list = ConfigCreator.create_from_args(args_parsed_list=args, configs_path=None)

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
        experiment_id=experiment.experiment_id, run_name="MAIN_" + current_time
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

        # For each config, test its reward learner
        for config in config_list:

            run_ids[config["name"]] = []

            # Log the config used
            mlflow.log_dict(config, artifact_file="configs/config_" + config["name"] + ".json")

            # Get the number of individual runs that shall be done for this config
            num_runs = config["num_runs"]
            start_seed = config["experiment"]["seed"]

            # Get the path to this config file
            path = config["source_path"]
            config_dir_path = ConfigCreator.config_dir_path
            assert config_dir_path is not None
            path = path[len(config_dir_path) + 1 :]

            # Create the individual run commands
            for index in range(num_runs):
                suffix = f"_seed_{start_seed + index}"

                # Create a child run for every single model
                with mlflow.start_run(run_name=config["name"] + suffix, nested=True) as child_run:

                    # Store the run id of this run
                    child_run_id = child_run.info.run_id
                    run_ids[config["name"]].append(child_run_id)

                    # Initialize a dictionary with values which should be changed
                    # in the default config
                    values_to_change = {
                        "experiment.run_id": (child_run_id, "str"),
                        "experiment.seed": (start_seed + index, "int"),
                    }

                    # Create the run commands
                    run_commands.append(
                        build_run_command(
                            path=path,
                            command="python test_reward_learners.py",
                            params=values_to_change,
                        )
                    )
                    log_paths.append(f"logs/output_{current_time}/{config['name']}" + suffix)

        # Log all the run ids, so we can extract the results later on
        mlflow.log_dict(run_ids, artifact_file="run_ids.json")

        # Run all commands
        generate_run_commands(
            run_commands,
            num_cpus_per_job=2,
            num_gpus_per_job=0,
            mem_per_job=4000,
            mode="euler",
            long_execution_time=False,
            log_files=log_paths,
            dry=False,
        )
