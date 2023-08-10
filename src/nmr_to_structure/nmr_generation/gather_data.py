import os
from io import StringIO
from pathlib import Path
from typing import Tuple

import click
import pandas as pd
from tqdm.contrib.concurrent import process_map


def process_spectrum(spectrum: str) -> dict:
    spectrum_df = pd.read_csv(StringIO(spectrum), names=["ppm", "real", "imag"])
    return {"real": spectrum_df["real"], "ppm": spectrum_df["ppm"]}


def parse_file_13C(args: Tuple[str, str]) -> Tuple[dict, bool]:
    def process_peaks_13C(peaks: str) -> Tuple[str, int]:
        peak_df = pd.read_csv(StringIO(peaks), index_col=0)
        return peak_df.to_json(), len(peak_df[peak_df["delta (ppm)"] > -20])

    file_path, index = args
    with open(file_path, "r") as f:
        data = f.read()
    try:
        smiles, spectrum, peaks = data.split("##############\n")
    except ValueError:
        return dict(), False

    peak_dict, n_peaks = process_peaks_13C(peaks)
    spectrum_dict = process_spectrum(spectrum)

    sim_result = {
        "index": index,
        "smiles": smiles,
        "spectrum": spectrum_dict,
        "peaks": peak_dict,
    }

    n_c = smiles.lower().count("c")
    if n_c < n_peaks:
        return sim_result, True
    else:
        return sim_result, False


def parse_file_1H(args: Tuple[str, str]) -> Tuple[dict, bool]:
    def process_peaks_1H(peaks: str) -> str:
        peak_df = pd.read_csv(StringIO(peaks), index_col=0)
        return peak_df.to_json()

    file_path, index = args
    with open(file_path, "r") as f:
        data = f.read()
    try:
        smiles, spectrum, peaks = data.split("##############\n")
    except ValueError:
        return dict(), False

    peak_dict = process_peaks_1H(peaks)
    spectrum_dict = process_spectrum(spectrum)

    sim_result = {
        "index": index,
        "smiles": smiles,
        "spectrum": spectrum_dict,
        "peaks": peak_dict,
    }

    return sim_result, False


def parse_folder(
    folder_path: Path,
    sim_type: str = "1H",
) -> pd.DataFrame:
    success = list()

    failure = 0
    total = 0

    file_list = list()

    for file in os.listdir(folder_path):
        index = file.split(".")[0].replace(sim_type, "")

        if file.endswith(".png"):
            failure += 1
            # to_redo.append(file)
            continue
        elif not file.endswith(".csv"):
            continue

        file_list.append((os.path.join(folder_path, file), index))

    # Multiprocessing
    if sim_type == "1H":
        parse_results = list(
            process_map(parse_file_1H, file_list, max_workers=12, chunksize=100)
        )
    elif sim_type == "13C":
        parse_results = list(
            process_map(parse_file_13C, file_list, max_workers=12, chunksize=100)
        )

    for result_dict, redo in parse_results:
        if len(result_dict) == 0 or redo:
            total += 1
            failure += 1

            continue

        success.append(result_dict)
        total += 1

    print(
        "Total: {}; Failure: {} -> {:.4f}".format(
            total, failure, (total - failure) / total
        )
    )

    # Format output dict to make it compatible with downstream tasks
    sim_type_key = "1H_NMR_sim" if sim_type == "1H" else "13C_NMR_sim"
    success_dict = {
        out_dict.pop("smiles"): {sim_type_key: out_dict} for out_dict in success
    }

    results_df = pd.DataFrame.from_dict(success_dict, orient="index")
    return results_df


@click.command()
@click.option(
    "--results_folder",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to a folder containing the output of a simulation.",
    multiple=True,
)
@click.option(
    "--out_path",
    required=True,
    type=click.Path(path_type=Path),
    help="Where to save the gathered data.",
)
@click.option(
    "--add_to_existing_df",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Option to add the gathered data to another dataframe.",
)
@click.option(
    "--sim_type",
    type=click.Choice(["1H", "13C"]),
    default="1H",
    help="Simulation type.",
)
def main(
    results_folder: Tuple[Path], out_path: Path, add_to_existing_df: Path, sim_type: str
):
    print(results_folder, len(results_folder), type(results_folder))

    # Gather data
    results_df = pd.DataFrame()
    if len(results_folder) > 1:
        for folder in results_folder:
            out_df = parse_folder(folder, sim_type=sim_type)
            results_df = (
                out_df if results_df is None else pd.concat((results_df, out_df))
            )
    else:
        results_df = parse_folder(results_folder[0], sim_type=sim_type)

    # Save data
    if add_to_existing_df is not None:
        existing_df = pd.read_pickle(add_to_existing_df)
        merged_df = existing_df.join(results_df, how="outer")
        pd.to_pickle(merged_df, out_path)
    else:
        pd.to_pickle(results_df, out_path)
