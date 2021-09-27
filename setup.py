#!/usr/bin/env python

from distutils.core import setup
import os

thelibFolder = os.path.dirname(os.path.realpath(__file__))
requirementPath = thelibFolder + '/requirements.txt'
install_requires = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

setup(name='speek',
      version='0.0',
      description='Segment Piston Estimation using the Extended Kalman Filter',
      author='Jesse Cranney',
      author_email='jesse.cranney@anu.edu.au',
      url='https://www.github.com/jcranney/gmt-oiwfs.git',
      install_requires=install_requires,
      packages=['speek'],
     )