#!/usr/bin/env python

from setuptools import find_packages, setup

import os
import subprocess
import time

version_file = 'lbasicsr/version.py'


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


