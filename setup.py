#import filecmp
#import glob
#import importlib
#import json
#import os
#import shutil
#import subprocess
#import sysconfig
#import time
#import sys
#from collections import defaultdict

#import setuptools.command.build_ext
#import setuptools.command.install
import setuptools.command.sdist
from setuptools import Extension, find_packages, setup
#from setuptools.dist import Distribution


#cwd = os.path.dirname(os.path.abspath(__file__))

def main():

    #with open(os.path.join(cwd, "README.md"), encoding="utf-8") as f:
    #    long_description = f.read()
    
    # the list of runtime dependencies required by this built package
    setup(
        name='originality',
        version='0.0.1',
        description=(
            'Calculating text originality in Python'
        ),
        long_description='long_description',
        long_description_content_type="text/markdown",
        packages=find_packages(
            include=['originality'],
            exclude=['cuda_code']),
        install_requires=[
            'numpy >= 1.20.0'
        ],
        include_package_data= True,
        url="https://github.com/MihkelLepson/originality",
        download_url="https://github.com/MihkelLepson/originality",
        author="Mihkel Lepson",
        author_email="mihkel.lepson@gmail.com",
        python_requires=f">={3.9}",
        # PyPI package information.
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Environment :: GPU :: NVIDIA CUDA"
        ],
        license="MIT",
        keywords="NLP",
    )

if __name__ == "__main__":
    main()