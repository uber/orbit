import json
import os
import platform
import shutil
import sys
from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py

MODEL_SOURCE_DIR = "orbit/stan"
MODELS = ["dlt", "ets", "lgt", "ktrlite"]

DESCRIPTION = "Orbit is a package for Bayesian time series modeling and inference."
BINARIES = ["diagnose", "print", "stanc", "stansummary"]

with open("orbit/config.json") as f:
    config = json.load(f)
CMDSTAN_VERSION = config["CMDSTAN_VERSION"]
IS_WINDOWS = platform.platform().startswith("Win")


def read_long_description(filename="README.md"):
    # with open(filename) as f:
    with open("README.md", "r", encoding="utf-8") as f:
        return f.read().strip()


def requirements(filename="requirements.txt"):
    with open(filename) as f:
        return f.readlines()


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
    from multiprocessing import cpu_count

    import cmdstanpy

    remove_older_cmdstan(CMDSTAN_VERSION)

    if not cmdstanpy.install_cmdstan(
        version=CMDSTAN_VERSION,
        overwrite=True,
        verbose=True,
        cores=cpu_count(),
        progress=True,
        compiler=IS_WINDOWS,
    ):
        raise RuntimeError("CmdStan failed to install in repackaged directory")

    print("Installed cmdstanpy package.")


def build_model(model: str, model_dir: str):
    import cmdstanpy

    # compile stan file in place
    model_name = f"{model}.stan"
    model_path = os.path.join(model_dir, model_name)
    print(f"Compiling stan file in place: {model_path}")
    _ = cmdstanpy.CmdStanModel(stan_file=model_path, force_compile=True)


def build_stan_models():
    if "conda" not in sys.prefix.lower():
        for model in MODELS:
            build_model(
                model=model,
                model_dir=MODEL_SOURCE_DIR,
            )
    else:
        print("Conda env is detected in the package path. Skip compilation.")


class BuildPyCommand(build_py):
    """Custom build command to make sure install cmdstanpy properly."""

    def run(self):
        print("Running build py command.")
        if not self.dry_run:
            print("Not a dry run, run with build")

            # install cmdstan and compilers, in the default directory for simplicity
            install_stan()

            # build all stan models
            build_stan_models()

        build_py.run(self)


about = {}
here = Path(__file__).parent.resolve()
with open(here / "orbit" / "__version__.py", "r") as f:
    exec(f.read(), about)


setup(
    version=about["__version__"],
    packages=find_packages(),
    name="orbit-ml",
    description=DESCRIPTION,
    include_package_data=True,
    install_requires=requirements("requirements.txt"),
    tests_require=requirements("requirements-test.txt"),
    cmdclass={
        "build_py": BuildPyCommand,
    },
    test_suite="orbit.tests",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://orbit-ml.readthedocs.io/en/stable/",
    zip_safe=False,
)
