import sys

from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.test import test as test_command

# from setuptools.command.install import install as install_command

# # force cython to use setuptools dist
# # see also:
# #   https://bugs.python.org/issue23114
# #   https://bugs.python.org/issue23102
# from setuptools import dist
# dist.Distribution().fetch_build_eggs(['cython'])

DESCRIPTION = "Orbit is a package for Bayesian time series modeling and inference."


def read_long_description(filename="README.md"):
    # with open(filename) as f:
    with open("README.md", "r", encoding="utf-8") as f:
        return f.read().strip()


def requirements(filename="requirements.txt"):
    with open(filename) as f:
        return f.readlines()


class PyTest(test_command):
    def finalize_options(self):
        test_command.finalize_options(self)
        self.test_args = ["-v"]  # test args
        self.test_suite = True

    def run_tests(self):
        import pytest

        errcode = pytest.main(self.test_args)
        sys.exit(errcode)


# pystan is calling numpy.distutils which requires this
# numpy>=1.18.2, <=1.22.2
# pystan==2.19.1.1
extras_require = dict()
extras_require["pystan"] = ["numpy<=<=1.22.2", "pystan==2.19.1.1"]
extras_require["cmdstanpy"] = ["cmdstanpy>=1.0.8"]

setup(
    author="Edwin Ng, Zhishi Wang, Steve Yang, Yifeng Wu, Jing Pan",
    author_email="zhishiw@uber.com",
    description=DESCRIPTION,
    include_package_data=True,
    install_requires=requirements("requirements.txt"),
    tests_require=requirements("requirements-test.txt"),
    extras_require=extras_require,
    cmdclass={
        "build_py": build_py,
        "develop": develop,
        "test": PyTest,
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
