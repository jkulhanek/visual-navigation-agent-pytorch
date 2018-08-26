#!/usr/bin/env python

from setuptools import setup

setup(name='deep-reactive-agent-pytorch',
      version='0.1',
      description='Deep RL agent implemented in pytorch',
      url='https://github.com/jkulhanek/deep-reactive-agent-pytorch',
      author='Jonas Kulhanek',
      author_email='jonas.kulhanek@live.com',
      license='MIT',
      packages=['agent'],
      scripts=['bin/train.py'],
      install_requires=["matplotlib", "ai2thor", "Cython", "abc"],
      zip_safe=True)
