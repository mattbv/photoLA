# -*- coding: utf-8 -*-
"""
Setup file for the photoLA package.

@author: Matheus Boni Vicari (matheus.boni.vicari@gmail.com)
"""

from setuptools import setup, find_packages


def readme():
    with open('README.rst') as f:
        return f.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="photoLA",
    version="0.9.0",
    author='Matheus Boni Vicari',
    author_email='matheus.boni.vicari@gmail.com',
    packages=find_packages(),
    entry_points={
        'console_scripts': ['photoLA_single=photoLA.single_LA:run_cmd',
                            'photoLA_batch=photoLA.batch_LA:run_cmd']},
    url='https://github.com/mattbv/photoLA',
    license='LICENSE.txt',
    description='Estimates Leaf Area from photographs using a direct\
 relationship between pixel count of leaves and of a reference with\
 known area.',
    long_description=readme(),
    classifiers=['Programming Language :: Python',
                 'Topic :: Scientific/Engineering'],
    keywords='Leaf Area photograph',
    install_requires=required,
    # ...
)
