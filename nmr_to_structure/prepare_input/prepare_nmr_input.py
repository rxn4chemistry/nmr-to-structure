from pathlib import Path

import click
import pandas as pd
from tqdm.auto import tqdm

from .nmr_utils import build_cnmr_string, build_hnmr_string, save_set, split_data


def make_nmr_set(
    nmr_set: pd.DataFrame,
    mode: str,
    hnmr_mode: str = "range",
    token_space="shared",
    n_aug=0,
    not_include_formula=False,
):
    input_list = list()

    for i in tqdm(range(len(nmr_set))):
        nmr_strings = list()
        if mode == "combined":
            h_nmr_strings = build_hnmr_string(
                smiles=nmr_set.index[i],
                peak_dict=nmr_set.iloc[i]["1H_NMR_sim"]["peaks"],
                mode=hnmr_mode,
                header=not not_include_formula,
                token_space=token_space,
                n_aug=n_aug,
            )

            c_nmr_strings = build_cnmr_string(
                nmr_set.iloc[i]["13C_NMR_sim"],
                header=False,
                n_aug=n_aug,
                token_space=token_space,
            )

            nmr_strings = [
                h_nmr + c_nmr for h_nmr, c_nmr in zip(h_nmr_strings, c_nmr_strings)
            ]

        elif mode == "hnmr":
            nmr_strings = build_hnmr_string(
                smiles=nmr_set.index[i],
                peak_dict=nmr_set.iloc[i]["1H_NMR_sim"]["peaks"],
                mode=hnmr_mode,
                header=not not_include_formula,
                token_space=token_space,
                n_aug=n_aug,
            )

        elif mode == "cnmr":
            nmr_strings = build_cnmr_string(
                nmr_set.iloc[i]["13C_NMR_sim"],
                smiles=nmr_set.index[i],
                header=not not_include_formula,
                token_space=token_space,
                n_aug=n_aug,
            )

        for nmr_string in nmr_strings:
            if nmr_string is None:
                continue
            input_list.append({"smiles": nmr_set.index[i], "nmr_input": nmr_string})

    input_df = pd.DataFrame(input_list)
    input_df = input_df.drop_duplicates(subset="nmr_input")

    return input_df


@click.command()
@click.option(
    "--nmr_data",
    "-n",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the NMR dataframe",
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
    help="Wether the token space between the 1H and 13C NMR is shared or not.",
)
@click.option(
    "--not_include_formula",
    is_flag=True,
    help="Wether the formula is included with the NMR or not.",
)
@click.option("--augment", is_flag=True, help="")
def main(
    nmr_data: Path,
    out_path: Path,
    mode: str,
    hnmr_mode: str = "range",
    token_space: str = "shared",
    not_include_formula: bool = False,
    augment: bool = False,
):
    nmr_data = pd.read_pickle(nmr_data)

    train_data, test_data, val_data = split_data(nmr_data)

    # Make the training data
    if augment:
        n_aug = 1
    else:
        n_aug = 0

    train_set = make_nmr_set(
        train_data,
        mode,
        hnmr_mode=hnmr_mode,
        token_space=token_space,
        not_include_formula=not_include_formula,
        n_aug=n_aug,
    )
    test_set = make_nmr_set(
        test_data,
        mode,
        hnmr_mode=hnmr_mode,
        token_space=token_space,
        not_include_formula=not_include_formula,
        n_aug=n_aug,
    )
    val_set = make_nmr_set(
        val_data,
        mode,
        hnmr_mode=hnmr_mode,
        token_space=token_space,
        not_include_formula=not_include_formula,
        n_aug=n_aug,
    )

    # Save training data
    save_set(train_set, out_path, "train")
    save_set(test_set, out_path, "test")
    save_set(val_set, out_path, "val")
