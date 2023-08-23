from pathlib import Path

import click
import pandas as pd
from rxn.chemutils.tokenization import tokenize_smiles
from tqdm.auto import tqdm
import random

from .nmr_utils import build_cnmr_string, build_hnmr_string, save_set, split_data

NON_MATCHING_TOKEN = "<no_match>"

def make_nmr(mode: str, component: str, hnmr: dict = None, cnmr: dict = None, hnmr_mode: str = 'range', token_space: str = 'shared'):

    if mode == "combined":
        hnmr_string = build_hnmr_string(
            smiles=component,
            peak_dict=hnmr,
            mode=hnmr_mode,
            header=False,
            token_space=token_space,
        )[0]

        cnmr_string = build_cnmr_string(
            cnmr,
            header=False,
            token_space=token_space,
        )[0]

        nmr_string = " {} {}".format(hnmr_string.strip(), cnmr_string)

    elif mode == "hnmr":
        hnmr_string = build_hnmr_string(
            smiles=component,
            peak_dict=hnmr,
            mode=hnmr_mode,
            header=False,
            token_space=token_space,
        )[0]

        nmr_string = " " + hnmr_string

    elif mode == "cnmr":
        cnmr_string = build_cnmr_string(
            cnmr,
            header=False,
            token_space=token_space,
        )[0]
        nmr_string = " " + cnmr_string
    
    return nmr_string


def make_nmr_rxn_set(
    nmr_set: pd.DataFrame,
    rxn_set: pd.DataFrame,
    mode: str,
    hnmr_mode: str = "range",
    token_space: str ="shared",
    non_matching: bool = True

) -> pd.DataFrame:
    rxns = list(rxn_set.index)
    mols = set(nmr_set.index)
    mols_sample = tuple(mols)

    src_tgt_pairs = list()

    for rxn in tqdm(rxns):
        rxn = rxn.replace(">", ".").replace("..", "")
        components = rxn.split(".")
        components = list(filter(None, components))

        components_tokenized = tokenize_smiles(".".join(components))

        for component in components:
            if component not in mols or len(component) < 5:
                continue
            
            # Make matching set
            nmr_input = components_tokenized
            nmr = make_nmr(mode, component, 
                           hnmr=nmr_set.loc[component]["1H_NMR_sim"]["peaks"], 
                           cnmr=nmr_set.loc[component]["13C_NMR_sim"], 
                           hnmr_mode=hnmr_mode,
                           token_space=token_space)
            nmr_input = components_tokenized + ' ' + nmr

            src_tgt_pairs.append({"nmr_input": nmr_input, "smiles": component, 'actual_smiles': component})

            if not non_matching:
                continue

            # Create non matching input
            random_mol = component
            while random_mol == component:
                random_mol = random.choice(mols_sample)
                
            nmr = make_nmr(mode, component, 
                           hnmr=nmr_set.loc[random_mol]["1H_NMR_sim"]["peaks"], 
                           cnmr=nmr_set.loc[random_mol]["13C_NMR_sim"], 
                           hnmr_mode=hnmr_mode,
                           token_space=token_space)
            nmr_input_non_matching = components_tokenized + ' ' + nmr
            src_tgt_pairs.append({'nmr_input': nmr_input_non_matching, 'smiles': NON_MATCHING_TOKEN, 'actual_smiles': random_mol})

    return pd.DataFrame(src_tgt_pairs)


def make_nmr_mol_set(nmr_df: pd.DataFrame,
                     mode: str, 
                     hnmr_mode: str,
                     token_space: str,
                     non_matching=True, 
                     n_mol_max: int = 7, 
                     n_samples: int = 1000):

    mols = tuple(set(nmr_df.index))
    src_tgt_pairs = list()

    # Default mol_distribution
    mol_distribution = [[i, n_samples] for i in range(2, n_mol_max+1)]

    for n_mols, n_examples in tqdm.tqdm(mol_distribution):
        for _ in range(n_examples):
            # Create set of molecules
            mol_set = random.sample(mols, k=n_mols)

            # Matching set
            nmr_mol = random.choice(mol_set)
            nmr = make_nmr(mode, nmr_mol, 
                           hnmr=nmr_df.loc[nmr_mol]["1H_NMR_sim"]["peaks"], 
                           cnmr=nmr_df.loc[nmr_mol]["13C_NMR_sim"], 
                           hnmr_mode=hnmr_mode,
                           token_space=token_space)
            nmr_input_matching = tokenize_smiles('.'.join(mol_set)) + ' ' + nmr
            src_tgt_pairs.append({'nmr_input': nmr_input_matching, 'smiles': nmr_mol})

            if not non_matching:
                continue

            # Create not matching set
            random_mol = nmr_mol 
            while random_mol in mol_set:
                random_mol = random.choice(mols)

            nmr = make_nmr(mode, nmr_mol, 
                           hnmr=nmr_df.loc[random_mol]["1H_NMR_sim"]["peaks"], 
                           cnmr=nmr_df.loc[random_mol]["13C_NMR_sim"], 
                           hnmr_mode=hnmr_mode,
                           token_space=token_space)
            nmr_input_non_matching = tokenize_smiles('.'.join(mol_set)) + ' ' + nmr
            src_tgt_pairs.append({'nmr_input': nmr_input_non_matching, 'smiles': NON_MATCHING_TOKEN})

    return pd.DataFrame.from_dict(src_tgt_pairs)

@click.command()
@click.option(
    "--nmr_data",
    "-n",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the NMR dataframe",
)
@click.option(
    "--rxn_data",
    "-r",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the reaction dataframe",
)
@click.option(
    "--out_path",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output path",
)
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["hnmr", "cnmr", "combined"]),
    required=True,
    help="Which mode to use. Choose from either solely 1H, 13C NMR or both combined.",
)
@click.option(
    "--hnmr_mode",
    type=click.Choice(["range", "adaptive", "center"]),
    default="range",
    help="How to format the 1H NMR peaks.",
)
@click.option(
    "--token_space",
    type=click.Choice(["shared", "separate"]),
    default="shared",
    help="Wether the token space between the 1H and 13C NMR is shared or not",
)
@click.option(
    "--non_matching",
    is_flag=True,
    help="Wether or not to include non matching examples.",
)

def main(
    nmr_data: Path,
    rxn_data: Path,
    out_path: Path,
    mode: str,
    hnmr_mode: str = "range",
    token_space: str = "shared",
    non_matching: bool = True
):
    nmr_df = pd.read_pickle(nmr_data)
    nmr_df.drop(columns=['1H_NMR_exp', '13C_NMR_exp'], inplace=True)
    nmr_df.dropna(inplace=True)

    rxn_df = pd.read_pickle(rxn_data)

    # Make the training data
    input_rxn = make_nmr_rxn_set(
        nmr_df,
        rxn_df,
        mode,
        hnmr_mode=hnmr_mode,
        token_space=token_space,
        non_matching=non_matching
    )

    # Split into train, test, val on the molecules
    all_smiles = pd.unique(input_rxn.actual_smiles)
    train_smiles, test_smiles, val_smiles = split_data(all_smiles)

    train_set_rxn = input_rxn[input_rxn.actual_smiles.isin(train_smiles)]
    test_set_rxn = input_rxn[input_rxn.actual_smiles.isin(test_smiles)]
    val_set_rxn = input_rxn[input_rxn.actual_smiles.isin(val_smiles)]

    # Create molecule set with approximately the same size as the rxn set
    n_samples_train, n_samples_test, n_samples_val = int(len(train_set_rxn)/2/6), int(len(test_set_rxn)/2/6), int(len(val_set_rxn)/2/6)

    train_set_mol = make_nmr_mol_set(nmr_df[nmr_df.index.isin(train_smiles)], mode,
                                     hnmr_mode=hnmr_mode,
                                     token_space=token_space,
                                     non_matching=non_matching,
                                     n_samples=n_samples_train)
    
    test_set_mol = make_nmr_mol_set(nmr_df[nmr_df.index.isin(test_smiles)], mode,
                                     hnmr_mode=hnmr_mode,
                                     token_space=token_space,
                                     non_matching=non_matching,
                                     n_samples=n_samples_test)

    val_set_mol = make_nmr_mol_set(nmr_df[nmr_df.index.isin(val_smiles)], mode,
                                        hnmr_mode=hnmr_mode,
                                        token_space=token_space,
                                        non_matching=non_matching,
                                        n_samples=n_samples_val)
    
    # Concatenate training and validation set
    combined_train_set = pd.concat((train_set_rxn, train_set_mol))
    combined_val_set = pd.concat((val_set_rxn, val_set_mol))

    # Save training data
    save_set(train_set_rxn, out_path, "train-rxn")
    save_set(train_set_mol, out_path, "train-mol")
    save_set(combined_train_set, out_path, 'train')

    save_set(combined_val_set, out_path, "val")

    save_set(test_set_mol, out_path, "test-mol")
    save_set(test_set_rxn, out_path, "test-rxn")
