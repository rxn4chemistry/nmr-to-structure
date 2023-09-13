import logging
import os
import random
import time
from functools import partial
from multiprocessing import Manager, Process
from pathlib import Path
from typing import List

import click
import pandas as pd
from rxn.chemutils.tokenization import tokenize_smiles
from rxn.utilities.logging import setup_console_and_file_logger
from tqdm.auto import tqdm

from nmr_to_structure.prepare_input.nmr_utils import (
    DEFAULT_SEED,
    DEFAULT_NON_MATCHING_TOKEN,
    evaluate_molecule,
    log_file_name_from_time,
    make_nmr,
    save_set,
    split_data,
    tanimoto_set_worker,
)

DEFAULT_TEST_SIZE = 0.1
DEFAULT_VAL_SIZE = 0.05
DEFAULT_MAX_SEQ_LEN = 600

random.seed(DEFAULT_SEED)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def make_nmr_tanimoto_set(
    nmr_df: pd.DataFrame,
    mode: str,
    n_samples: int,
    logger: logging.Logger,
    cores: int = 8,
    hnmr_mode: str = "range",
    token_space: str = "shared",
    non_matching=True,
    tanimoto_bins: int = 5,
) -> pd.DataFrame:
    all_smiles = list(nmr_df.index)
    random.shuffle(all_smiles)

    smiles_chunks = [
        all_smiles[
            i
            * len(all_smiles)
            // cores : min((i + 1) * len(all_smiles) // cores, len(all_smiles))
        ]
        for i in range(cores)
    ]

    manager = Manager()
    return_list = manager.list()
    status_list = manager.list([0] * len(smiles_chunks))
    jobs = []

    worker_fn = partial(
        tanimoto_set_worker,
        nmr_df=nmr_df,
        mode=mode,
        n_samples=n_samples // cores,
        return_list=return_list,
        status_list=status_list,
        hnmr_mode=hnmr_mode,
        token_space=token_space,
        non_matching=non_matching,
        tanimoto_bins=tanimoto_bins,
    )

    for i, chunk in enumerate(smiles_chunks):
        logger.info(f"Starting job {i+1}/{cores}")
        process = Process(target=worker_fn, args=(i, chunk))
        jobs.append(process)
        process.start()

    pbar = tqdm(total=n_samples)
    while len(return_list) != len(smiles_chunks):
        status = 0
        for value in status_list:
            status += value
        pbar.update(status - pbar.n)
        time.sleep(5)
    pbar.close()

    combined_df = pd.concat(return_list)

    logger.info("Printing Tanimoto statistics:")
    for key in pd.unique(combined_df.tanimoto_bin):
        logger.info(
            "Tanimoto bin {}: {:.3f}%".format(
                key, (combined_df.tanimoto_bin == key).sum() / len(combined_df) * 100
            )
        )

    return combined_df


def make_nmr_rxn_set(
    nmr_df: pd.DataFrame,
    rxn_set: pd.DataFrame,
    n_samples: int,
    mode: str,
    hnmr_mode: str = "range",
    token_space: str = "shared",
    non_matching: bool = True,
) -> pd.DataFrame:
    rxns = list(rxn_set.index)
    random.shuffle(rxns)

    mols = set(nmr_df.index)
    mols_sample = tuple(mols)

    src_tgt_pairs: List[dict] = list()

    idx = 0
    pbar = tqdm(total=n_samples)
    while len(src_tgt_pairs) < n_samples:
        rxn = rxns[idx].replace(">", ".").replace("..", "")
        components = rxn.split(".")
        components = list(filter(None, components))

        allowed_mask = [evaluate_molecule(smiles) for smiles in components]
        allowed_components = [
            components[i]
            for i in range(len(allowed_mask))
            if allowed_mask[i] and len(components[i]) > 5
        ]

        components_tokenized = tokenize_smiles(".".join(allowed_components))

        for component in allowed_components:
            if component not in mols or len(component) < 5:
                continue

            # Make matching set
            nmr_input = components_tokenized
            nmr = make_nmr(
                mode,
                component,
                hnmr=nmr_df.loc[component]["1H_NMR_sim"]["peaks"],
                cnmr=nmr_df.loc[component]["13C_NMR_sim"],
                hnmr_mode=hnmr_mode,
                token_space=token_space,
            )
            nmr_input = components_tokenized + nmr

            if nmr_input.count(" ") + 1 > DEFAULT_MAX_SEQ_LEN:
                continue

            src_tgt_pairs.append(
                {"nmr_input": nmr_input, "smiles": tokenize_smiles(component)}
            )

            if not non_matching:
                continue

            # Create non matching input
            random_mol = component
            while random_mol == component:
                random_mol = random.choice(mols_sample)

            nmr = make_nmr(
                mode,
                random_mol,
                hnmr=nmr_df.loc[random_mol]["1H_NMR_sim"]["peaks"],
                cnmr=nmr_df.loc[random_mol]["13C_NMR_sim"],
                hnmr_mode=hnmr_mode,
                token_space=token_space,
            )
            nmr_input_non_matching = components_tokenized + nmr
            src_tgt_pairs.append(
                {
                    "nmr_input": nmr_input_non_matching,
                    "smiles": DEFAULT_NON_MATCHING_TOKEN,
                }
            )

        idx += 1
        pbar.update(len(src_tgt_pairs) - pbar.n)

        if idx == len(rxns) - 1:
            logging.warning(
                f"Ran out of reactions to consider before reaching desired training samples. Generated {len(src_tgt_pairs)} samples instead of {n_samples} desired."
            )
            break

    return pd.DataFrame(src_tgt_pairs)


def make_nmr_mol_set(
    nmr_df: pd.DataFrame,
    mode: str,
    n_samples: int,
    hnmr_mode: str = "range",
    token_space: str = "shared",
    non_matching=True,
    n_max_mol_set_size: int = 7,
) -> pd.DataFrame:
    mols = tuple(set(nmr_df.index))
    src_tgt_pairs: List[dict] = list()

    # Default mol_distribution to cover equal number of examples for the different set sizes, if non_matching is set to true -> divide by two as two examples will be added per pass
    mol_distribution = [
        [
            i,
            int(n_samples / ((n_max_mol_set_size - 1) * 2))
            if non_matching
            else int(n_samples / (n_max_mol_set_size - 1)),
        ]
        for i in range(2, n_max_mol_set_size + 1)
    ]

    for n_mols, n_examples in tqdm(mol_distribution):
        for _ in range(n_examples):
            # Create set of molecules
            mol_set = random.sample(mols, k=n_mols)

            # Matching set
            nmr_mol = random.choice(mol_set)
            nmr = make_nmr(
                mode,
                nmr_mol,
                hnmr=nmr_df.loc[nmr_mol]["1H_NMR_sim"]["peaks"],
                cnmr=nmr_df.loc[nmr_mol]["13C_NMR_sim"],
                hnmr_mode=hnmr_mode,
                token_space=token_space,
            )
            nmr_input_matching = tokenize_smiles(".".join(mol_set)) + " " + nmr

            if nmr_input_matching.count(" ") + 1 > DEFAULT_MAX_SEQ_LEN:
                continue

            src_tgt_pairs.append(
                {"nmr_input": nmr_input_matching, "smiles": tokenize_smiles(nmr_mol)}
            )

            if not non_matching:
                continue

            # Create not matching set
            random_mol = nmr_mol
            while random_mol in mol_set:
                random_mol = random.choice(mols)

            nmr = make_nmr(
                mode,
                nmr_mol,
                hnmr=nmr_df.loc[random_mol]["1H_NMR_sim"]["peaks"],
                cnmr=nmr_df.loc[random_mol]["13C_NMR_sim"],
                hnmr_mode=hnmr_mode,
                token_space=token_space,
            )
            nmr_input_non_matching = tokenize_smiles(".".join(mol_set)) + " " + nmr
            src_tgt_pairs.append(
                {
                    "nmr_input": nmr_input_non_matching,
                    "smiles": DEFAULT_NON_MATCHING_TOKEN,
                }
            )

    return pd.DataFrame(src_tgt_pairs)


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
@click.option("--cores", "-c", type=int, default=8)
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["hnmr", "cnmr", "combined"]),
    required=True,
    help="Which mode to use. Choose from either solely 1H, 13C NMR or both combined.",
)
@click.option("--sample_size", "-s", type=int, default=3000000)
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
    cores: int,
    mode: str,
    sample_size: int,
    hnmr_mode: str = "range",
    token_space: str = "shared",
    non_matching: bool = True,
):
    if not out_path.exists():
        os.makedirs(out_path)

    logfile = out_path / log_file_name_from_time("nmr_matching_preprocess")
    setup_console_and_file_logger(logfile)

    # Load dataframe containing nmr data
    logger.info("Reading data.")
    nmr_df = pd.read_pickle(nmr_data)
    #nmr_df.drop(columns=["1H_NMR_exp", "13C_NMR_exp"], inplace=True)
    nmr_df.dropna(inplace=True)

    # Split into train, test, val
    logger.info("Splitting into train, test and validation set.")
    nmr_df_train, nmr_df_test, nmr_df_val = split_data(
        nmr_df, test_size=DEFAULT_TEST_SIZE, val_size=DEFAULT_VAL_SIZE
    )

    # Determine number of examples to sample for each of the three sets: RXN, Random Sample and Tanimoto
    n_samples_train, n_samples_test, n_samples_val = (
        int(sample_size / 3),
        int(sample_size / 3 * DEFAULT_TEST_SIZE),
        int(sample_size / 3 * DEFAULT_VAL_SIZE),
    )

    # Make tanimoto sets
    logger.info("Making tanimoto sets.")
    tanimoto_train = make_nmr_tanimoto_set(
        nmr_df_train,
        mode=mode,
        n_samples=n_samples_train,
        logger=logger,
        cores=cores,
        hnmr_mode=hnmr_mode,
        token_space=token_space,
        non_matching=non_matching,
    )
    tanimoto_test = make_nmr_tanimoto_set(
        nmr_df_test,
        mode=mode,
        n_samples=n_samples_test,
        logger=logger,
        cores=cores,
        hnmr_mode=hnmr_mode,
        token_space=token_space,
        non_matching=non_matching,
    )
    tanimoto_val = make_nmr_tanimoto_set(
        nmr_df_val,
        mode=mode,
        n_samples=n_samples_val,
        logger=logger,
        cores=cores,
        hnmr_mode=hnmr_mode,
        token_space=token_space,
        non_matching=non_matching,
    )

    # Sample RXN sets

    logging.info("Making RXN sets.")
    rxn_df = pd.read_pickle(rxn_data)

    rxn_train = make_nmr_rxn_set(
        nmr_df_train,
        rxn_df,
        n_samples_train,
        mode,
        hnmr_mode=hnmr_mode,
        token_space=token_space,
        non_matching=non_matching,
    )

    rxn_test = make_nmr_rxn_set(
        nmr_df_test,
        rxn_df,
        n_samples_test,
        mode,
        hnmr_mode=hnmr_mode,
        token_space=token_space,
        non_matching=non_matching,
    )

    rxn_val = make_nmr_rxn_set(
        nmr_df_val,
        rxn_df,
        n_samples_val,
        mode,
        hnmr_mode=hnmr_mode,
        token_space=token_space,
        non_matching=non_matching,
    )

    # Sample random sets for train, test and val
    logging.info("Making Molecule sets.")
    mol_train = make_nmr_mol_set(
        nmr_df_train,
        mode,
        hnmr_mode=hnmr_mode,
        token_space=token_space,
        non_matching=non_matching,
        n_samples=n_samples_train,
    )

    mol_test = make_nmr_mol_set(
        nmr_df_test,
        mode,
        hnmr_mode=hnmr_mode,
        token_space=token_space,
        non_matching=non_matching,
        n_samples=n_samples_test,
    )

    mol_val = make_nmr_mol_set(
        nmr_df_val,
        mode,
        hnmr_mode=hnmr_mode,
        token_space=token_space,
        non_matching=non_matching,
        n_samples=n_samples_val,
    )

    # Concatenate training and validation set
    combined_train_set = pd.concat((rxn_train, mol_train, tanimoto_train))
    combined_train_set = combined_train_set.sample(frac=1, random_state=DEFAULT_SEED)
    combined_val_set = pd.concat((rxn_val, mol_val, tanimoto_val))
    combined_val_set = combined_val_set.sample(frac=1, random_state=DEFAULT_SEED)

    # Save training data
    logging.info("Saving data.")
    save_set(combined_train_set, out_path, "train")

    # Save validation data
    combined_val_set_small = combined_val_set.sample(n=min(10000, len(combined_val_set)), random_state=DEFAULT_SEED)
    save_set(combined_val_set_small, out_path, "val")
    save_set(combined_val_set, out_path, "val-big")

    # Save test data
    save_set(rxn_test, out_path, "test-rxn")
    save_set(mol_test, out_path, "test-mol")

    for key in pd.unique(tanimoto_test.tanimoto_bin):
        subset = tanimoto_test[tanimoto_test.tanimoto_bin == key]
        save_set(subset, out_path, f"test-tanimoto-{key}")


if __name__ == "__main__":
    main()
