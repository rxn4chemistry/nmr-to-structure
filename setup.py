#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

requirements = [
    "Click>=7.0",
    "pandas>=1.4.2",
    "numpy>=1.23.0",
    "scipy>=1.7.3",
    "scikit-learn>=1.1.3",
    "regex>=2022.3.15",
    "tqdm>= 4.64.0",
    "rdkit>=2022.9.1",
    "tqdm>=4.6.0",
    "opennmt-py>=3.0.3",
    "PyAutoGUI>=0.9.53",
    "rxn-chem-utils>=1.1.4",
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
        "Programming Language :: Python :: 3.8",
    ],
    description="Package containing data structures to hold experimental and simulated NMR spectra, validate the spectra and build training data.",
    entry_points={
        "console_scripts": [
            "run_simulation=nmr_to_structure.nmr_generation.run_mestrenova_simulation:main",
            "gather_data=nmr_to_structure.nmr_generation.gather_data:main",
            "prepare_nmr_input=nmr_to_structure.prepare_input.prepare_nmr_input:main",
            "prepare_nmr_rxn_input=nmr_to_structure.prepare_input.prepare_nmr_rxn_input:main",
            "train_model=nmr_to_structure.training.run_training:main",
            "score_model=nmr_to_structure.training.score:main",
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
