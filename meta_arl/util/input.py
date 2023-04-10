import argparse
import ast
import json
import os
import sys
from typing import Any, Callable, ClassVar, Optional, Union


class ConfigCreator:
    """Reads and parses config file, additionally combines it with commandline arguments
    and stores the config in a dictionary."""

    config_dir_path = None

    # TODO: It may be useful to add support for .json and .yaml
    supported_file_types: ClassVar[list[str]] = [".py", ".txt"]

    supported_variable_types: ClassVar[dict[str, Callable]] = {
        "int": int,
        "float": float,
        "complex": complex,
        "str": str,
        "list": list,
        "tuple": tuple,
        "dict": ast.literal_eval,
        "set": set,
        "bool": bool,
        "bytes": bytes,
        "bytearray": bytearray,
    }

    @classmethod
    def interpret_parameter(cls, param_string: str) -> tuple[str, Any]:
        """Given a parameter string of the form 'a=3::int', extracts the parameter
        name (a) and interprets the parameter value (3) as the specified type (int)
        and then returns it

        Args:
            param_string (str): The parameter string to be interpreted

        Raises:
            TypeError: If the parameter type is not supported

        Returns:
            tuple[str, Any]: A tuple containing the parameter name and the interpreted
                parameter value
        """
        # Split the parameter into name and value
        var_name, value = param_string.split("=")

        # Evaluate the value
        # '::' is used to denote the type of the value
        if "::" in value:
            value, value_type = value.split("::")
            if value_type not in cls.supported_variable_types:
                raise TypeError(
                    f"The variable type '{value_type}' is unknown! "
                    f"Pleas use one of {cls.supported_variable_types}"
                )
            value = cls.supported_variable_types[value_type](value)
        else:
            # Try to automatically infer the type of the value
            value = ast.literal_eval(value)

        return var_name, value

    @classmethod
    def file2dict(cls, path: str) -> dict:
        """Reads a config file and returns its content as a dictionary

        Args:
            path (str): The path to the config file

        Raises:
            FileNotFoundError: If the config file does not exist
            TypeError: If the config file has an unsupported file type

        Returns:
            dict: The content of the config file as a dictionary
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"The file {path} seems to be missing.")

        # Get the file extension
        name, extension = os.path.splitext(path)

        # Check if files of the given extension are supported
        if extension not in cls.supported_file_types:
            raise TypeError(
                f"Files of type {extension} are not (yet) supported! Consider "
                f"using one of the following formats: {cls.supported_file_types}"
            )

        # Extract the dictionary from the file
        if extension in [".py", ".txt"]:
            with open(path, "r") as f:
                content_list = f.readlines()

                # Remove all potential comments
                for index, line in enumerate(content_list):
                    if "#" not in line:
                        continue
                    content_list[index] = line[: line.index("#")] + "\n"
                content = "".join(content_list)
                config = ast.literal_eval(content)

        # Add the file path of this config to the dictionary
        config["source_path"] = path

        return config

    @classmethod
    def create_from_args(
        cls, args_parsed_list: Optional[list[str]] = None, configs_path: Optional[str] = None
    ) -> tuple[dict, Union[dict, list[dict]]]:
        """Creates a config dictionary from the commandline arguments and the config files
        provided by the commandline arguments.

        Args:
            args_parsed_list (Optional[list[str]], optional): A list of commandline arguments
            configs_path (Optional[str], optional): A path to a directory which stores the
                config files

        Raises:
            FileNotFoundError: If the config directory does not exist

        Returns:
            tuple[dict, Union[dict, list[dict]]]: A tuple containing a global config which should
                be used in addition to every individual config and a (list of) config dictionaries.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-c", "--config", "--configs", action="extend", nargs="+", required=True
        )
        parser.add_argument(
            "-p", "--param", "--params", action="extend", nargs="+", required=False
        )
        parser.add_argument(
            "-g",
            "--globalparam",
            "--globalparams",
            action="extend",
            nargs="+",
            required=False,
        )
        args_parsed = parser.parse_args(args_parsed_list)

        # If no 'configs_path' has been provided, take the path to the folder of the
        # executing file
        if configs_path is None:
            run_path = sys.argv[0]
            configs_path = os.path.dirname(run_path)
            configs_path = os.path.join(os.path.abspath(configs_path), "configs")

            if not os.path.isdir(configs_path):
                raise FileNotFoundError(
                    f"The configs directory '{configs_path}' does not exist! "
                    f"Please create it or specify a different path via the "
                    f"'configs_path' variable"
                )

        # Store the configs path for later reuse
        ConfigCreator.config_dir_path = configs_path

        configs: list[dict] = []

        # Read all the specified configs into a file
        for config_file_path in args_parsed.config:
            path = os.path.join(configs_path, config_file_path)
            configs.append(ConfigCreator.file2dict(path))

        # If no config files have been provided, initialize an empty config
        if configs == []:
            configs = [{}]

        # Parse the extra parameters
        if args_parsed.param is not None:

            for param_string in args_parsed.param:
                # Convert the string into a variable name and a corresponding value
                var_name, value = ConfigCreator.interpret_parameter(param_string)

                # Split the name according to the '.' separator
                # This enables an easy way to set dictionary values
                var_name_split = var_name.split(".")

                # Apply the parameter to each individual config
                for config in configs:
                    sub_config = config

                    # If necessary enter a nested dictionary structure
                    for var in var_name_split[:-1]:
                        sub_config = sub_config[var]

                    # Apply the parameter
                    sub_config[var_name_split[-1]] = value

        global_config = None

        # Parse the global parameters
        if args_parsed.globalparam is not None:
            global_config = {}

            for param_string in args_parsed.globalparam:
                # Convert the string into a variable name and a corresponding value
                var_name, value = ConfigCreator.interpret_parameter(param_string)

                # Apply the parameter to each individual config
                global_config[var_name] = value

        if len(configs) == 1:
            return global_config, configs[0]

        # Return the extracted configs
        return global_config, configs


if __name__ == "__main__":
    config_path = (
        "/home/lukas/Polybox_ETH/Research-in-Data-Science/meta-active-reward"
        "-learning/experiments/test_reward_learners/configs"
    )

    args = (
        "--configs point_env_simple_gp.py point_env_learnt_gp.py "
        "--params seed=1 env_config.arena_size=6"
    ).split()

    configs = ConfigCreator.create_from_args(args_parsed_list=args, configs_path=config_path)

    for config in configs:
        print("Loaded config = ", config, "\n")
