#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    "numpy>=1.17",
    "xarray",
    "dask[array]",  # for `to_netcdf`
]

test_requirements = [
    "pytest",
]

setup(
    name='result_caching',
    version='0.1.0',
    description="Cache results for re-use",
    long_description=readme,
    author="Martin Schrimpf",
    author_email='martin.schrimpf@outlook.com',
    url='https://github.com/brain-score/result_caching',

    packages=find_packages(exclude=['tests']),
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='caching',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
