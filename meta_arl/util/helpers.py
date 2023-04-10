import os
from zipfile import ZipFile

import numpy as np


def get_file_paths(
    directory_path: str,
    ignore_directory_patterns: list[str] = [],
    ignore_file_patterns: list[str] = [],
) -> list[str]:
    """Creates a list of paths to all files in the directory specified by directory_path
    whose path doesn't contain any ignore_pattern.

    Args:
        directory_path (str): The path to the directory to be listed
        ignore_directory_patterns (list[str], optional): A list of directory names to ignore.
            Defaults to [].
        ignore_file_patterns (list[str], optional): A list of file names to ignore. Defaults to [].

    Returns:
        list[str]: A list of paths to all files in the directory specified by directory_path
    """
    file_paths = []

    # Iterate through the whole directory and its subdirectories
    for root, _, file_list in os.walk(directory_path):
        # Check that the current root doesn't contain any of the ignore patterns:
        if any([pattern in root for pattern in ignore_directory_patterns]):
            continue
        for file_name in file_list:
            # Check that the current file doesn't contain any of the ignore patterns
            if any([pattern in file_name for pattern in ignore_file_patterns]):
                continue
            file_path = os.path.join(root, file_name)
            file_paths.append(file_path)

    # returning all file paths
    return file_paths


def zip_directory(
    directory_to_zip: str,
    zip_file_path: str,
    ignore_directory_patterns: list[str] = [],
    ignore_file_patterns: list[str] = [],
) -> None:
    """Zips the directory specified by directory_to_zip and saves it to the path specified by
    zip_file_path.

    Args:
        directory_to_zip (str): The path to the directory to be zipped
        zip_file_path (str): The path to the zip file to be created
        ignore_directory_patterns (list[str], optional): Directory names which should not be
            zipped
        ignore_file_patterns (list[str], optional): File names which should not be zipped
    """
    # Get a list of all files whose path doesn't contain any ignore_pattern
    file_paths = get_file_paths(
        directory_to_zip,
        ignore_directory_patterns,
        ignore_file_patterns,
    )

    # Zip all files specified in 'file_paths'
    with ZipFile(zip_file_path, "w") as zip_file:
        for path in file_paths:
            zip_file.write(path)


if __name__ == "__main__":
    zip_directory(
        ".",
        zip_file_path="./zipped.zip",
        ignore_directory_patterns=[
            "__pycache__",
            ".venv",
            ".vscode",
            ".trash",
            "mlruns",
            "old",
            "pacoh_old",
            "results",
        ],
    )
