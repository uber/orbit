import sys
import os
import platform
from pathlib import Path
import tempfile
from shutil import copy, copytree, rmtree

from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.test import test as test_command
from setuptools.command.build_ext import build_ext

# from setuptools.command.install import install as install_command

# install style is taking reference from prophet package installation:
# https://github.com/facebook/prophet/

MODEL_SOURCE_DIR = "orbit/stan"
MODELS = ["dlt", "ets", "lgt", "ktrlite"]
MODEL_TARGET_DIR = os.path.join("orbit", "stan_compiled")

DESCRIPTION = "Orbit is a package for Bayesian time series modeling and inference."
CMDSTAN_VERSION = "2.33.1"
BINARIES_DIR = "bin"
BINARIES = ["diagnose", "print", "stanc", "stansummary"]
TBB_PARENT = "stan/lib/stan_math/lib"
TBB_DIRS = ["tbb", "tbb_2020.3"]
IS_WINDOWS = platform.platform().startswith("Win")


def read_long_description(filename="README.md"):
    # with open(filename) as f:
    with open("README.md", "r", encoding="utf-8") as f:
        return f.read().strip()


def requirements(filename="requirements.txt"):
    with open(filename) as f:
        return f.readlines()


class PyTestCommand(test_command):
    def finalize_options(self):
        test_command.finalize_options(self)
        self.test_args = ["-v"]  # test args
        self.test_suite = True

    def run_tests(self):
        import pytest

        errcode = pytest.main(self.test_args)
        sys.exit(errcode)


def build_stan_model(target_dir):
    print("Importing cmdstanpy...")
    import cmdstanpy
    from multiprocessing import cpu_count

    target_cmdstan_dir = (Path(target_dir) / f"cmdstan-{CMDSTAN_VERSION}").resolve()
    print("target_cmdstan_dir: {}".format(target_cmdstan_dir))

    with tempfile.TemporaryDirectory() as tmp_dir:
        # long paths on windows can cause problems during build
        if IS_WINDOWS:
            print("Windows detected. Use tmp_dir: {}".format(tmp_dir))
            cmdstan_dir = (Path(tmp_dir) / f"cmdstan-{CMDSTAN_VERSION}").resolve()
        else:
            cmdstan_dir = target_cmdstan_dir

        if os.path.isdir(cmdstan_dir):
            rmtree(cmdstan_dir)

        if not cmdstanpy.install_cmdstan(
            version=CMDSTAN_VERSION,
            # if we want to do it inside the repo dir, we need to include the folder in
            # MANIFEST.in
            dir=cmdstan_dir,
            overwrite=True,
            verbose=True,
            cores=cpu_count(),
            progress=True,
        ):
            raise RuntimeError("CmdStan failed to install in repackaged directory")
        else:
            print("Installed cmdstanpy package.")

        print("cmdstan_dir: {}".format(cmdstan_dir))
        for model in MODELS:
            model_name = "{}.stan".format(model)
            # note: ensure copy target is a directory not a file.

            temp_source_file_path = os.path.join(MODEL_SOURCE_DIR, model_name)
            print("Loading source file: {}".format(temp_source_file_path))
            temp_stan_file = copy(temp_source_file_path, cmdstan_dir.parent.resolve())

            # temp_stan_file = os.path.join(MODEL_SOURCE_DIR, model_name)

            print("Compiling stan file: {}".format(temp_stan_file))
            sm = cmdstanpy.CmdStanModel(stan_file=temp_stan_file)
            target_name = "{}.bin".format(model)
            target_file_path = os.path.join(target_dir, target_name)
            print("Copying file to {}".format(target_file_path))
            copy(sm.exe_file, target_file_path)

        # TODO: some clean up needs to be done
        # 1. with the stan/ folder since it duplicates the .stan files
        # 2. the stanlib packages if it is installed with repackaged directory
        # for f in Path(MODEL_SOURCE_DIR).iterdir():
        #     if f.is_file() and f.name != model_name:
        #         os.remove(f)

    # prune_cmdstan(target_cmdstan_dir)


# def prune_cmdstan(cmdstan_dir: str) -> None:
#     """
#     Keep only the cmdstan executables and tbb files (minimum required to run a cmdstanpy commands on a pre-compiled model).
#     """
#     original_dir = Path(cmdstan_dir).resolve()
#     parent_dir = original_dir.parent
#     temp_dir = parent_dir / "temp"
#     if temp_dir.is_dir():
#         rmtree(temp_dir)
#     temp_dir.mkdir()

#     print("Copying ", original_dir, " to ", temp_dir, " for pruning")
#     copytree(original_dir / BINARIES_DIR, temp_dir / BINARIES_DIR)
#     for f in (temp_dir / BINARIES_DIR).iterdir():
#         if f.is_dir():
#             rmtree(f)
#         elif f.is_file() and f.stem not in BINARIES:
#             os.remove(f)
#     for tbb_dir in TBB_DIRS:
#         copytree(original_dir / TBB_PARENT / tbb_dir, temp_dir / TBB_PARENT / tbb_dir)

#     rmtree(original_dir)
#     temp_dir.rename(original_dir)


class BuildPyCommand(build_py):
    """Custom build command to make sure install cmdstanpy properly."""

    def run(self):
        if not self.dry_run:
            target_dir = os.path.join(self.build_lib, MODEL_TARGET_DIR)
            self.mkpath(target_dir)

            print("Not a dry run, target_dir:")
            print(target_dir)
            print(self.build_lib)

            build_stan_model(target_dir)

        print("Dry run.")
        build_py.run(self)


class BuildExtCommand(build_ext):
    """Ensure built extensions are added to the correct path in the wheel."""

    def run(self):
        pass


# class DevelopCommand(develop):
#     """Custom build command to make sure install cmdstanpy properly."""

#     def run(self):
#         if not self.dry_run:
#             install_cmdstanpy()

#         develop.run(self)


setup(
    author="Edwin Ng, Zhishi Wang, Steve Yang, Yifeng Wu, Jing Pan",
    author_email="edwinnglabs@gmail.com",
    description=DESCRIPTION,
    include_package_data=True,
    install_requires=requirements("requirements.txt"),
    tests_require=requirements("requirements-test.txt"),
    # extras_require=extras_require,
    cmdclass={
        "build_ext": BuildExtCommand,
        "build_py": BuildPyCommand,
        # "develop": DevelopCommand,
        "test": PyTestCommand,
    },
    test_suite="orbit.tests",
    license="Apache License 2.0",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    name="orbit-ml",
    packages=find_packages(),
    url="https://orbit-ml.readthedocs.io/en/stable/",
    # version=VERSION, # being maintained by source module
    zip_safe=False,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
