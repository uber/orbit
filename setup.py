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


def repackage_cmdstan():
    return os.environ.get("ORBIT_REPACKAGE_CMDSTAN", "").lower() not in ["false", "0"]


def install_rtools() -> bool:
    """
    Install C++ compilers required to build stan models on Windows machines.
    Reference from prophet
    """
    import cmdstanpy

    print("Windows detected, install C++ Compliers required to build stan models.")
    try:
        cmdstanpy.utils.cxx_toolchain_path()
        return False
    except Exception:
        try:
            from cmdstanpy.install_cxx_toolchain import run_rtools_install
        except ImportError:
            # older versions
            from cmdstanpy.install_cxx_toolchain import main as run_rtools_install

        run_rtools_install({"version": None, "dir": None, "verbose": True})
        compiler, tool = cmdstanpy.utils.cxx_toolchain_path()
        print("Toolchain installed. Compiler:", compiler, ", Tools:", tool)
        return True


def prune_cmdstan(cmdstan_dir: str) -> None:
    """
    Keep only the cmdstan executables and tbb files (minimum required to run a cmdstanpy commands on a pre-compiled model).
    Reference from prophet
    """
    print("Prune stan directory: {}".format(cmdstan_dir))
    original_dir = Path(cmdstan_dir).resolve()
    parent_dir = original_dir.parent
    temp_dir = parent_dir / "temp"
    if temp_dir.is_dir():
        rmtree(temp_dir)
    temp_dir.mkdir()

    print("Copying ", original_dir, " to ", temp_dir, " for pruning")
    copytree(original_dir / BINARIES_DIR, temp_dir / BINARIES_DIR)
    for f in (temp_dir / BINARIES_DIR).iterdir():
        if f.is_dir():
            rmtree(f)
        elif f.is_file() and f.stem not in BINARIES:
            os.remove(f)
    for tbb_dir in TBB_DIRS:
        copytree(original_dir / TBB_PARENT / tbb_dir, temp_dir / TBB_PARENT / tbb_dir)

    rmtree(original_dir)
    temp_dir.rename(original_dir)


def install_stan(cmdstan_dir: Path):
    """
    Compile and install stan backend
    Reference from prophet
    """
    from multiprocessing import cpu_count

    import cmdstanpy

    if repackage_cmdstan():
        if IS_WINDOWS:
            install_rtools()
        print("Installing cmdstan to", cmdstan_dir)
        if os.path.isdir(cmdstan_dir):
            rmtree(cmdstan_dir)

        # if not cmdstanpy.install_cmdstan(
        #     version=CMDSTAN_VERSION,
        #     dir=cmdstan_dir.parent,
        #     overwrite=True,
        #     verbose=True,
        #     cores=cpu_count(),
        #     progress=True,
        # ):
        #     raise RuntimeError("CmdStan failed to install in repackaged directory")

        # TODO: testing installing on default folder
        if not cmdstanpy.install_cmdstan(
            version=CMDSTAN_VERSION, 
            progress=True
        ):
              raise RuntimeError("CmdStan failed to install in repackaged directory")

        print("Installed cmdstanpy package.")


def build_model(model: str, model_dir, cmdstan_dir, target_dir):
    import cmdstanpy

    model_name = f"{model}.stan"

    temp_source_file_path = os.path.join(model_dir, model_name)
    print(
        f"Copying source file from {temp_source_file_path} to {cmdstan_dir.parent.resolve()}"
    )
    temp_stan_file = copy(
        os.path.join(model_dir, model_name), cmdstan_dir.parent.resolve()
    )
    print(f"Compiling stan file: {temp_stan_file}")
    sm = cmdstanpy.CmdStanModel(stan_file=temp_stan_file)
    target_name = f"{model}.bin"
    target_file_path = os.path.join(target_dir, target_name)
    print(f"Copying file from {sm.exe_file} to {target_file_path}")
    copy(sm.exe_file, target_file_path)


def build_stan_model(target_dir):
    print("Importing cmdstanpy...")
    import cmdstanpy

    target_cmdstan_dir = (Path(target_dir) / f"cmdstan-{CMDSTAN_VERSION}").resolve()
    print("target_cmdstan_dir: {}".format(target_cmdstan_dir))

    with tempfile.TemporaryDirectory() as tmp_dir:
        # long paths on windows can cause problems during build
        if IS_WINDOWS:
            print("Windows detected. Use tmp_dir: {}".format(tmp_dir))
            cmdstan_dir = (Path(tmp_dir) / f"cmdstan-{CMDSTAN_VERSION}").resolve()
            install_rtools()
        else:
            cmdstan_dir = target_cmdstan_dir
        print("cmdstan_dir: {}".format(cmdstan_dir))
        install_stan(cmdstan_dir)

        # for model in MODELS:
        #     # note: ensure copy target is a directory not a file.
        #     build_model(
        #         model=model,
        #         model_dir=MODEL_SOURCE_DIR,
        #         cmdstan_dir=cmdstan_dir,
        #         target_dir=target_dir,
        #     )

        if IS_WINDOWS and repackage_cmdstan():
            copytree(cmdstan_dir, target_cmdstan_dir)

        # TODO: some clean up needs to be done
        # 1. with the stan/ folder since it duplicates the .stan files
        # 2. the stanlib packages if it is installed with repackaged directory
        # for f in Path(MODEL_SOURCE_DIR).iterdir():
        #     if f.is_file() and f.name != model_name:
        #         os.remove(f)

    # if repackage_cmdstan():
    #     prune_cmdstan(target_cmdstan_dir)
    repackage_cmdstan()


class BuildPyCommand(build_py):
    """Custom build command to make sure install cmdstanpy properly."""

    def run(self):
        if not self.dry_run:
            target_dir = os.path.join(self.build_lib, MODEL_TARGET_DIR)
            self.mkpath(target_dir)

            print("Not a dry run, run with build, target_dir: {}".format(target_dir))

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


class EditableWheel(editable_wheel):
    """Custom develop command to pre-compile Stan models in-place."""

    def run(self):
        if not self.dry_run:
            target_dir = os.path.join(self.project_dir, MODEL_TARGET_DIR)
            self.mkpath(target_dir)

            print("Not a dry run, run with editable, target_dir: {}".format(target_dir))

            build_stan_model(target_dir)

        editable_wheel.run(self)


class BDistWheelABINone(bdist_wheel):
    def finalize_options(self):
        bdist_wheel.finalize_options(self)
        self.root_is_pure = False

    def get_tag(self):
        _, _, plat = bdist_wheel.get_tag(self)
        return "py3", "none", plat


about = {}
here = Path(__file__).parent.resolve()
with open(here / "orbit" / "__version__.py", "r") as f:
    exec(f.read(), about)


setup(
    version=about["__version__"],
    packages=find_packages(),
    name="orbit-ml",
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
        "editable_wheel": EditableWheel,
        "bdist_wheel": BDistWheelABINone,
        # "develop": DevelopCommand,
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
