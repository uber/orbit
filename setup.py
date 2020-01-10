from __future__ import print_function

import sys

from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand

VERSION = '0.4.0'
DESCRIPTION = "Orbit is a package for bayesian time series modeling and inference."
AUTHOR = "Edwin Ng <edwinng@uber.com>, Steve Yang <steve.yang@uber.com>, Huigang Chen <huigang@uber.com>"


def read_long_description(filename="README.md"):
    with open(filename) as f:
        return f.read().strip()


def requirements(filename="requirements.txt"):
    with open(filename) as f:
        return f.readlines()


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errcode = pytest.main(self.test_args)
        sys.exit(errcode)


setup(
    author=AUTHOR,
    author_email='edwinng@uber.com',
    description=DESCRIPTION,
    include_package_data=True,
    setup_requires=[
        # Setuptools 18.0 properly handles Cython extensions.
        'setuptools>=18.0',
        'cython'
    ],
    install_requires=requirements('requirements.txt'),
    tests_require=requirements('requirements-test.txt'),
    cmdclass={
        'test': PyTest
    },
    license='closed',
    long_description=read_long_description(),
    name='orbit',
    packages=find_packages(),
    url='git@github.com:uber/orbit.git',
    version=VERSION,
    zip_safe=False,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ]
)
