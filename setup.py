from __future__ import print_function

import sys

from setuptools import dist, setup, find_packages
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.test import test as test_command

# # force cython to use setuptools dist
# # see also:
# #   https://bugs.python.org/issue23114
# #   https://bugs.python.org/issue23102
# dist.Distribution().fetch_build_eggs(['cython'])

DESCRIPTION = "Orbit is a package for bayesian time series modeling and inference."
AUTHOR = '''
    Edwin Ng <edwinng@uber.com>, Steve Yang <steve.yang@uber.com>,
    Zhishi Wang <zhishiw@uber.com>, Yifeng Wu <yifeng.wu@uber.com>,  
    Jing Pan <jing.pan@uber.com>
    '''


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
        self.test_args = ['-v']  # test args
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
    install_requires=requirements('requirements.txt'),
    tests_require=requirements('requirements-test.txt'),
    cmdclass={
        'build_py': build_py,
        'develop': develop,
        'test': PyTest,
    },
    test_suite='orbit.tests',
    license='Apache License 2.0',
    long_description=read_long_description(),
    long_description_content_type='text/markdown',
    name='orbit-ml',
    packages=find_packages(),
    url='https://orbit-ml.readthedocs.io/en/stable/',
    # version=VERSION, # being maintained by source module
    zip_safe=False,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ]
)
