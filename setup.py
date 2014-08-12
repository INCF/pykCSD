#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import multiprocessing

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


readme = open('README.rst').read()
history = open('HISTORY.rst').read().replace('.. :changelog:', '')

requirements = [
    'numpy',
    'scipy',
    'scikit-learn',
    # TODO: put package requirements here
]

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='pykCSD',
    version='0.1.0',
    description='Object oriented Python kernel current source density toolbox',
    long_description=readme + '\n\n' + history,
    author='Grzegorz Parka',
    author_email='grzegorz.parka@gmail.com',
    url='https://github.com/INCF/pykCSD',
    packages=[
        'pykCSD',
    ],
    package_dir={'pykCSD':
                 'pykCSD'},
    include_package_data=True,
    install_requires=requirements,
    license="BSD",
    zip_safe=False,
    keywords=[
    	'pykCSD',
    	'kernel current source denisty',
    ],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
