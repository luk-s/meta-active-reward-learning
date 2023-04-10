# import multiprocessing as mp
import sys
from datetime import datetime
from itertools import product
from typing import Iterable, Optional, Union

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
    # manual configs
    command = "python test_agent_with_reward_learners.py"
    exec_time = None
    experiment_mode = "euler"
    separate_randomness = False
    num_cpus_per_job = 2

    # Set the experiment
    experiment = mlflow.set_experiment("Test rl agent with reward learners")

    # If no command-line arguments have been provided, use the following config
    if len(sys.argv) == 1:
        args: Optional[list[str]] = (
            "--configs "
            # "config_point_env_simple_gp.py "
            # "config_point_env_learnt_gp.py "
            # "config_point_env_pacoh_gp.py "
            # "config_point_env_fpacoh_gp.py "
            "2d/config_point_env_fpacoh_gp_PPO.py "
            "2d/config_point_env_fpacoh_gp_SAC.py "
            # "--params experiment.seed=3 "
            # "--globalparams num_repeats=1"
        ).split()
    else:
        args = None

    global_config, config_list = ConfigCreator.create_from_args(
        args_parsed_list=args, configs_path=None
    )

    # Extract global options
    if global_config is not None:
        command = global_config.get("command", command)
        exec_time = global_config.get("exec_time", exec_time)
        experiment_mode = global_config.get("experiment_mode", experiment_mode)
        num_cpus_per_job = global_config.get("num_cpus_per_job", num_cpus_per_job)
        separate_randomness = global_config.get("separate_randomness", separate_randomness)

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
            if separate_randomness:
                num_env_seeds = config["num_env_seeds"]
                num_reward_learner_seeds = config["num_reward_learner_seeds"]
                num_rl_seeds = config["num_rl_seeds"]

                start_env_seed = config["experiment"]["env_config"]["rng_seed"]
                start_reward_learner_seed = config["experiment"]["reward_model_config"]["rng_seed"]
                start_rl_seed = config["experiment"]["agent_config"]["rng_seed"]

                seed_combinations = product(
                    list(range(num_env_seeds)),
                    list(range(num_reward_learner_seeds)),
                    list(range(num_rl_seeds)),
                )

            # Get the path to this config file
            path = config["source_path"]
            config_dir_path = ConfigCreator.config_dir_path
            assert config_dir_path is not None
            path = path[len(config_dir_path) + 1 :]

            if separate_randomness:
                to_enumerate: Iterable[Union[tuple[int, int, int], int]] = seed_combinations
            else:
                to_enumerate = list(range(num_runs))

            # Create the individual run commands
            for index, element in enumerate(to_enumerate):
                if separate_randomness:
                    assert isinstance(element, tuple)
                    env_seed, rm_seed, rl_seed = element
                    print(f"{env_seed = }, {rm_seed = }, {rl_seed = }")
                    suffix = (
                        f"_env_seed_{start_env_seed + env_seed}"
                        f"_rm_seed_{start_reward_learner_seed + rm_seed}"
                        f"_rl_seed_{start_rl_seed + rl_seed}"
                    )
                else:
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
                    }
                    if separate_randomness:
                        values_to_change["experiment.env_config.rng_seed"] = (
                            start_env_seed + env_seed
                        ), "int"
                        values_to_change["experiment.reward_model_config.rng_seed"] = (
                            start_reward_learner_seed + rm_seed
                        ), "int"
                        values_to_change["experiment.agent_config.rng_seed"] = (
                            start_rl_seed + rl_seed
                        ), "int"
                    else:
                        values_to_change["experiment.seed"] = (
                            start_seed + index,
                            "int",
                        )

                    # Create the run commands
                    run_commands.append(
                        build_run_command(
                            path=path,
                            command=command,
                            params=values_to_change,
                        )
                    )
                    log_paths.append(f"logs/output_{current_time}/{config['name']}" + suffix)

        # Log all the run ids, so we can extract the results later on
        mlflow.log_dict(run_ids, artifact_file="run_ids.json")

        # Run all commands
        generate_run_commands(
            run_commands,
            num_cpus_per_job=num_cpus_per_job,
            num_gpus_per_job=0,
            mem_per_job=8000,
            mode=experiment_mode,
            long_execution_time=True,
            custom_execution_time=exec_time,
            log_files=log_paths,
            dry=False,
        )
