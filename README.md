# NMR to Structure




## Installation guide

Create a conda environment for the scripts:

```
conda create -n nmr python=3.9
conda activate nmr
```

Clone the repository and instal via: 

```
pip install -e .
```

## Simulating NMR spectra

This section explains how to simulate NMR spectra using MestreNova. A working version of MestreNova is required. 

### Running the simulations

Running the MestreNova script requires a .csv file containing SMILES of the molecules for which NMR spectra will be generated. This .csv file requires two columns, 'Smiles' and ... An 'example.csv' is provided. To run the simulations run the following script:

```
run_simulation --smiles_csv example/example.csv --out_folder example/simulation_out/1H --sim_type 1H --mnova_path <Absolute Path to your MestreNova executable> --script_path <Absolute path to the folder containing the MestreNova scripts>
```

The MestreNova simulation scripts can be found here [link to folder]. This script uses the MestreNova scripting tool to run the simulations. As a result all MestreNova will be opened and you will be able to see how spectra are simulated. A file is saved for each molecule.


### Compiling the data

After running the script the files are compiled into one dataframe making handling easier. The 


## Preparing the input data

### Task 1: NMR to Structure


### Task 2: Reaction data + NMR to Structure

## Training a model



## Assessing a model
