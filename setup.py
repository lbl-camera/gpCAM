#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages
from os import path
import sys

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

#requirements = ["requirements.txt"]

#setup_requirements = ["requirements.txt"]

#test_requirements = ["requirements.txt"]


here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'requirements.txt')) as requirements_file:
    # Parse requirements.txt, ignoring any commented-out lines.
    requirements = [line for line in requirements_file.read().splitlines()
                    if not line.startswith('#')]


setup(
    author="Marcus Michael Noack",
    author_email='MarcusNoack@lbl.gov',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        ####License here:
        ####e.g. 'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="gpCAM is a code for autonomous data acquisition",
    install_requires=requirements,
    #license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='gpcam',
    name='gpcam',
    packages=find_packages(include=['gpcam', 'gpcam.*']),
    test_suite='tests',
    url='https://github.com/Marcus Noack/gpcam',
    version='6.0.2',
    zip_safe=False,
)
