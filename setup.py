import json
import os
import platform
import sys
import tempfile
from pathlib import Path
from shutil import copy, copytree, rmtree

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.editable_wheel import editable_wheel
from setuptools.command.test import test as test_command
from wheel.bdist_wheel import bdist_wheel

# from setuptools.command.install import install as install_command

# the installation process of stan is taking reference from prophet package:
# https://github.com/facebook/prophet/

MODEL_SOURCE_DIR = "orbit/stan"
MODELS = ["dlt", "ets", "lgt", "ktrlite"]
MODEL_TARGET_DIR = os.path.join("orbit", "stan_compiled")

DESCRIPTION = "Orbit is a package for Bayesian time series modeling and inference."
BINARIES_DIR = "bin"
BINARIES = ["diagnose", "print", "stanc", "stansummary"]
TBB_PARENT = "stan/lib/stan_math/lib"
TBB_DIRS = ["tbb", "tbb_2020.3"]

with open("orbit/cmdstan_version.json") as f:
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


def install_stan():
    """
    Compile and install stan backend
    Reference from prophet
    """
    from multiprocessing import cpu_count
    import cmdstanpy

    import cmdstanpy

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


def build_model(model: str, model_dir: str, target_dir: str):
    import cmdstanpy

    # Copy model.stan file to the target dir
    model_name = f"{model}.stan"
    model_path = os.path.join(model_dir, model_name)
    print(f"Copying source file from {model_path} to {target_dir}")
    temp_stan_file = copy(model_path, target_dir)

    # compile stan file in place
    print(f"Compiling stan file: {temp_stan_file}")
    sm = cmdstanpy.CmdStanModel(stan_file=temp_stan_file)


def build_stan_models(target_dir: str):
    for model in MODELS:
        # note: ensure copy target is a directory not a file.
        build_model(
            model=model,
            model_dir=MODEL_SOURCE_DIR,
            target_dir=target_dir,
        )


class BuildPyCommand(build_py):
    """Custom build command to make sure install cmdstanpy properly."""

    def run(self):
        print("Running build py command.")
        if not self.dry_run:
            target_dir = os.path.join(self.build_lib, MODEL_TARGET_DIR)
            self.mkpath(target_dir)

            print("Not a dry run, run with build, target_dir: {}".format(target_dir))
            # build_stan_model(target_dir)

            # install cmdstan and compilers, in the default directory for simplicity
            install_stan()

            # build all stan models
            build_stan_models(target_dir)

        print("Dry run.")
        build_py.run(self)


class BuildExtCommand(build_ext):
    """Ensure built extensions are added to the correct path in the wheel."""

    def run(self):
        pass


class DevelopCommand(develop):
    """Custom build command to make sure install cmdstanpy properly."""

    def run(self):
        print("Running develop command.")
        if not self.dry_run:
            target_dir = os.path.join(self.build_lib, MODEL_TARGET_DIR)
            self.mkpath(target_dir)

            print("Not a dry run, run with build, target_dir: {}".format(target_dir))

            install_stan()
        else:
            print("Dry run.")
        develop.run(self)


class EditableWheel(editable_wheel):
    """Custom develop command to pre-compile Stan models in-place."""

    def run(self):
        print("Running editable wheel.")
        if not self.dry_run:
            target_dir = os.path.join(self.build_lib, MODEL_TARGET_DIR)
            self.mkpath(target_dir)

            print("Not a dry run, run with editable, target_dir: {}".format(target_dir))

            build_stan_models(target_dir)

        print("Dry run.")
        editable_wheel.run(self)


# class BDistWheelABINone(bdist_wheel):
#     def finalize_options(self):
#         bdist_wheel.finalize_options(self)
#         self.root_is_pure = False

#     def get_tag(self):
#         _, _, plat = bdist_wheel.get_tag(self)
#         return "py3", "none", plat


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
    # extras_require=extras_require,
    cmdclass={
        "build_py": BuildPyCommand,
        # "build_ext": BuildExtCommand,
        "editable_wheel": EditableWheel,
        # "bdist_wheel": BDistWheelABINone,
        "develop": DevelopCommand,
        # "test": PyTestCommand,
    },
    test_suite="orbit.tests",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://orbit-ml.readthedocs.io/en/stable/",
    # version=VERSION, # being maintained by source module
    ext_modules=[Extension("orbit.stan_compiled", [])],
    zip_safe=False,
)
