#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

requirements = [
    "Click>=7.0",
]

test_requirements = []

setup(
    author="Marvin Alberts",
    author_email="marvin.alberts@ibm.com",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Package containing data structures to hold experimental and simulated NMR spectra, validate the spectra and build training data.",
    entry_points={
        "console_scripts": [
            "train_set_nmr=nmr_to_structure.create_nmr_set:main",
            "train_set_nmr_rxn=nmr_to_structure.create_nmr_rxn_set:main",
        ],
    },
    install_requires=requirements,
    license="MIT license",
    include_package_data=True,
    keywords="nmr_to_structure",
    name="nmr_to_structure",
    packages=find_packages(include=["nmr_to_structure", "nmr_to_structure.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/Marvin Alberts/nmr_to_structure",
    version="0.1.0",
    zip_safe=False,
)
