import sys
import os
from pathlib import Path

from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.test import test as test_command
from setuptools.command.build_ext import build_ext

# from setuptools.command.install import install as install_command

# # force cython to use setuptools dist
# # see also:
# #   https://bugs.python.org/issue23114
# #   https://bugs.python.org/issue23102
# from setuptools import dist
# dist.Distribution().fetch_build_eggs(['cython'])

DESCRIPTION = "Orbit is a package for Bayesian time series modeling and inference."
CMDSTAN_VERSION = "2.31.0"


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


def install_cmdstanpy():
    print("Importing cmdstanpy...")
    import cmdstanpy
    from multiprocessing import cpu_count

    print("Installing cmdstan...")
    # target_dir = os.path.join(self.setup_path, "stan_compiled")
    # self.mkpath(target_dir)

    if not cmdstanpy.install_cmdstan(
        version=CMDSTAN_VERSION,
        # if we want to do it inside the repo dir, we need to include the folder in
        # MANIFEST.in
        # dir=target_dir,
        overwrite=True,
        verbose=True,
        cores=cpu_count(),
        progress=True,
    ):
        raise RuntimeError("CmdStan failed to install in repackaged directory")
    else:
        print("Installed cmdstanpy package.")


class BuildPyCommand(build_py):
    """Custom build command to make sure install cmdstanpy properly."""

    def run(self):
        if not self.dry_run:
            install_cmdstanpy()

        build_py.run(self)


class BuildExtCommand(build_ext):
    """Ensure built extensions are added to the correct path in the wheel."""

    def run(self):
        pass


class DevelopCommand(develop):
    """Custom build command to make sure install cmdstanpy properly."""

    def run(self):
        if not self.dry_run:
            install_cmdstanpy()

        develop.run(self)


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
        "develop": DevelopCommand,
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
