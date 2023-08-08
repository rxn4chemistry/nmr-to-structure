from pathlib import Path

import click
import pandas as pd
from rxn.chemutils.tokenization import tokenize_smiles
from tqdm.auto import tqdm

from .nmr_utils import (build_cnmr_string, build_hnmr_string, save_set,
                        split_data)


def make_nmr_rxn_set(
    nmr_set: pd.DataFrame,
    rxn_set: pd.DataFrame,
    mode: str,
    hnmr_mode: str = "range",
    token_space="shared",
    not_include_formula=False,
) -> pd.DataFrame:
    rxns = list(rxn_set.index)
    mols = set(nmr_set.index)

    src_tgt_pairs = list()

    for rxn in tqdm(rxns):
        rxn = rxn.replace(">", ".").replace("..", "")
        components = rxn.split(".")
        components = list(filter(None, components))

        components_tokenized = tokenize_smiles(".".join(components))

        for component in components:
            if component not in mols or len(component) < 5:
                continue

            nmr_input = components_tokenized

            if mode == "combined":
                hnmr_string = build_hnmr_string(
                    smiles=component,
                    peak_dict=nmr_set.loc[component]["1H_NMR_sim"]["peaks"],
                    mode=hnmr_mode,
                    header=not not_include_formula,
                    token_space=token_space,
                )[0]

                cnmr_string = build_cnmr_string(
                    nmr_set.loc[component]["13C_NMR_sim"],
                    header=False,
                    token_space=token_space,
                )[0]

                nmr_input += " {} {}".format(hnmr_string.strip(), cnmr_string)

            elif mode == "hnmr":
                hnmr_string = build_hnmr_string(
                    smiles=component,
                    peak_dict=nmr_set.loc[component]["1H_NMR_sim"]["peaks"],
                    mode=hnmr_mode,
                    header=not not_include_formula,
                    token_space=token_space,
                )[0]

                nmr_input += " " + hnmr_string
            elif mode == "cnmr":
                cnmr_string = build_cnmr_string(
                    nmr_set.loc[component]["13C_NMR_sim"],
                    header=False,
                    token_space=token_space,
                )[0]
                nmr_input += " " + cnmr_string

            src_tgt_pairs.append({"nmr_input": nmr_input, "smiles": component})

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
@click.option("--not_include_formula", is_flag=True, help="")
def main(
    nmr_data: Path,
    rxn_data: Path,
    out_path: Path,
    mode: str,
    hnmr_mode: str = "range",
    token_space: str = "shared",
    not_include_formula: bool = False,
):
    nmr_df = pd.read_pickle(nmr_data)
    rxn_df = pd.read_pickle(rxn_data)

    # Make the training data
    input_df = make_nmr_rxn_set(
        nmr_df,
        rxn_df,
        mode,
        hnmr_mode=hnmr_mode,
        token_space=token_space,
        not_include_formula=not_include_formula,
    )

    # Split into train, test, val on the molecules
    all_smiles = pd.unique(input_df.smiles)
    train_smiles, test_smiles, val_smiles = split_data(all_smiles)

    train_set = input_df[input_df.smiles.isin(train_smiles)]
    test_set = input_df[input_df.smiles.isin(test_smiles)]
    val_set = input_df[input_df.smiles.isin(val_smiles)]

    # Save training data
    save_set(train_set, out_path, "train")
    save_set(test_set, out_path, "test")
    save_set(val_set, out_path, "val")
