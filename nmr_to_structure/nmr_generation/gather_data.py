import pandas as pd
from io import StringIO
from typing import Tuple
import os
import lzma
import gzip
import json
import pickle
import click
import tqdm
from tqdm.contrib.concurrent import process_map


def process_spectrum(spectrum: str) -> dict:
    spectrum_df = pd.read_csv(StringIO(spectrum), names=["ppm", "real", "imag"])
    return {"real": spectrum_df["real"], "ppm": spectrum_df["ppm"]}


def parse_file_13C(args: Tuple[str, str]) -> Tuple[dict, bool]:
    def process_peaks_13C(peaks: str) -> Tuple[dict, int]:
        peak_df = pd.read_csv(StringIO(peaks), index_col=0)
        return peak_df.to_json(), len(peak_df[peak_df["delta (ppm)"] > -20])

    file_path, index = args
    with open(file_path, "r") as f:
        data = f.read()
    try:
        smiles, spectrum, peaks = data.split("##############\n")
    except:
        return None, False

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
    def process_peaks_1H(peaks: str) -> Tuple[dict, int]:
        peak_df = pd.read_csv(StringIO(peaks), index_col=0)
        return peak_df.to_json()

    file_path, index = args
    with open(file_path, "r") as f:
        data = f.read()
    try:
        smiles, spectrum, peaks = data.split("##############\n")
    except:
        return None, False

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
    folder_path: str,
    out_path: str,
    file_names: str,
    return_results: bool = False,
    sim_type: str = "1H",
) -> None:
    success = list()
    to_redo = list()

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
        if result_dict == None:
            total += 1
            failure += 1
            to_redo.append(result_dict)
            continue
        if redo:
            to_redo.append(result_dict)
        else:
            success.append(result_dict)

        total += 1

    print(
        "Total: {}; Failure: {} -> {:.4f}".format(
            total, failure, (total - failure) / total
        )
    )
    print(
        "Total: {}; Redo {} -> {:.4f}".format(total, len(to_redo), len(to_redo) / total)
    )

    to_redo_df = dict()
    for entry in to_redo:
        if entry == None:
            continue
        to_redo_df[entry["index"]] = entry["smiles"].strip()

    to_redo_df = pd.DataFrame.from_dict(to_redo_df, orient="index", columns=["Smiles"])
    to_redo_df.to_csv(os.path.join(out_path, f"redo_{file_names}.csv"))

    if return_results:
        return success
    else:
        for i in range(len(success) // 50000 + 1):
            with lzma.open(
                os.path.join(out_path, f"success_{file_names}_{i}.xz"), "wb"
            ) as f:
                pickle.dump(success[i * 50000 : (i + 1) * 50000], f)


@click.command()
@click.option("--results_folder", required=True, help="Output folder", multiple=True)
@click.option("--out_folder", required=True, help="Output folder")
@click.option("--out_name", default=None, help="Output folder")
@click.option("--sim_type", default="1H", help="Output folder")
def main(results_folder: str, out_folder: str, out_name: str, sim_type: str):
    if out_name == None:
        out_name = results_folder[0]
    print(results_folder, len(results_folder), type(results_folder))
    if len(results_folder) > 1:
        success = list()
        for folder in results_folder:
            parse_folder(
                folder, out_folder, folder, return_results=False, sim_type=sim_type
            )

        # with lzma.open(os.path.join(out_folder, 'success_all.xz'), "wb") as f:
        #    pic

    else:
        parse_folder(results_folder[0], out_folder, out_name, sim_type=sim_type)