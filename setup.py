#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name='ibvs',
    version='0.1.0',
    description='Image-Based Visual Servoing for UAV control',
    author='User',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.19.0',
        'opencv-python>=4.5.0',
    ],
    python_requires='>=3.6',
)
