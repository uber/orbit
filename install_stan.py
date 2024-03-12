import json
import os
import platform
import shutil
from multiprocessing import cpu_count

import cmdstanpy

with open("orbit/config.json") as f:
    config = json.load(f)
CMDSTAN_VERSION = config["CMDSTAN_VERSION"]
IS_WINDOWS = platform.platform().startswith("Win")


def remove_older_cmdstan(cmdstan_version):
    """
    Checks the default CmdStan path (~/.cmdstan) for folders with older versions
    and removes them.

    Args:
        cmdstan_version (str): The current CmdStan version.
    """
    default_path = os.path.expanduser("~/.cmdstan")

    if not os.path.exists(default_path):
        # Path doesn't exist, nothing to remove
        return

    for folder in os.listdir(default_path):
        if folder.startswith("cmdstan-"):
            # Extract version number from folder name
            folder_version = folder.split("-")[1]
            if folder_version < cmdstan_version:
                # Remove folder if version is older
                full_path = os.path.join(default_path, folder)
                try:
                    print(f"Removing older CmdStan version: {folder} : {full_path}")
                    shutil.rmtree(full_path)
                    print(f"Done.")
                except OSError as e:
                    print(f"Error removing {folder}: {e}")


def install_stan():
    """
    Compile and install stan backend
    Reference from prophet
    """

    remove_older_cmdstan(CMDSTAN_VERSION)

    if not cmdstanpy.install_cmdstan(
        version=CMDSTAN_VERSION,
        overwrite=True,
        verbose=True,
        cores=cpu_count(),
        progress=True,
        compiler=IS_WINDOWS,
    ):
        raise RuntimeError("CmdStan failed to install.")

    print(f"Installed cmdstanpy (cmdstan v.{CMDSTAN_VERSION}) and compiler.")
