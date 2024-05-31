from setuptools import setup, find_packages
import codecs
import os


VERSION = "1.4"


DESCRIPTION = "Genetic algorithm for consensus building"

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name = "consensusGen",
    version = VERSION,
    author = "RomainCoulon (Romain Coulon)",
    author_email = "<romain.coulon@bipm.org>",
    description = DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://pypi.org/project/consensusGen/",
    py_modules=['consensusGen'],
    project_urls={'Documentation': 'https://github.com/RomainCoulon/consensusGen/',},
    packages = find_packages(),
    install_requires = ["numpy","tqdm","matplotlib"],
    keywords = ["genetic algorithm","inter-laboratory comparison","consenus building"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Natural Language :: French",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    include_package_data = True,
    package_data = {'': []},
)
