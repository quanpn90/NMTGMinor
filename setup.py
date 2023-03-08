#!/usr/bin/env python
from setuptools import setup, find_packages

setup(name='NMTGMinor',
      version='0.1',
      author='quanpn90',
      author_email='ngoc.pham@kit.edu',
      url='https://github.com/quanpn90/NMTGMinor',
      license='MIT',
      scripts=[
          'flask_online.py',
          'online.py',
          'preprocess.py',
          'train.py',
          'translate_distributed.py',
          'translate.py',
      ],
      packages=find_packages(),
      install_requires=['torch', 'torchaudio', 'soundfile'])
