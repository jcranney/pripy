#!/usr/bin/env python

from distutils.core import setup
import os

thelibFolder = os.path.dirname(os.path.realpath(__file__))
requirementPath = thelibFolder + '/requirements.txt'
install_requires = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

setup(name='pripy',
      version='0.0',
      description='Phase Retrieval In PYthon',
      author='Jesse Cranney',
      author_email='jesse.cranney@anu.edu.au',
      url='https://www.github.com/jcranney/pripy.git',
      install_requires=install_requires,
      packages=['pripy'],
     )
