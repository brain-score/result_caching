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

# boto3 is only needed for the S3-backed storage; every other backend is
# stdlib + xarray. Kept out of install_requires so importing result_caching
# never drags in an AWS SDK.
extras_require = {
    "s3": ["boto3"],
}

test_requirements = [
    "pytest",
]

setup(
    name='result_caching',
    version='0.5',
    description="Cache results for re-use",
    long_description=readme,
    long_description_content_type='text/markdown',
    author="Martin Schrimpf",
    author_email='martin.schrimpf@outlook.com',
    url='https://github.com/brain-score/result_caching',
    packages=find_packages(exclude=['tests']),
    install_requires=requirements,
    extras_require=extras_require,
    license="MIT license",
    zip_safe=False,
    keywords='caching',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.11',
    ],
    test_suite='tests',
    tests_require=test_requirements,
)
