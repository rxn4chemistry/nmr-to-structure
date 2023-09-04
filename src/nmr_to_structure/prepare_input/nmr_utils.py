import datetime
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, KeysView, List, Optional, Tuple

import numpy as np
import pandas as pd
import regex as re
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors
from rxn.chemutils.tokenization import tokenize_smiles
from sklearn.model_selection import train_test_split

DEFAULT_SEED = 3246
DEFAULT_NON_MATCHING_TOKEN = "<no_match>"


# General Utilities #
def tokenize_formula(formula: str) -> list:
    return re.findall("[A-Z][a-z]?|\d+|.", formula)


def jitter(value: float, jitter_range: float = 2) -> float:
    jitter_value = np.random.uniform(-jitter_range, +jitter_range)
    return value + jitter_value


def split_data(
    input_data: Any, test_size: float = 0.1, val_size: float = 0.05
) -> Tuple[Any, Any, Any]:
    train_data, test_data = train_test_split(
        input_data, test_size=test_size, random_state=DEFAULT_SEED
    )
    train_data, val_data = train_test_split(
        train_data, test_size=val_size, random_state=DEFAULT_SEED
    )

    return (train_data, test_data, val_data)


def save_set(data_set: pd.DataFrame, out_path: Path, set_type: str) -> None:
    os.makedirs(out_path, exist_ok=True)

    smiles = list(data_set.smiles)
    with open(out_path / f"tgt-{set_type}.txt", "w") as f:
        for item in smiles:
            f.write(f"{item}\n")

    nmr_input = data_set.nmr_input
    with open(out_path / f"src-{set_type}.txt", "w") as f:
        for item in nmr_input:
            f.write(f"{item}\n")


# Copied from github.com/rxn4chemistry/rxn-onmt-models/blob/main/src/rxn/onmt_models/utils.py
def log_file_name_from_time(prefix: Optional[str] = None) -> str:
    """
    Get the name of a log file (typically to create it) from the current
    date and time.

    Returns:
        String for a file name in the format "20221231-1425.log", or
        "{prefix}-20221231-1425.log" if the prefix is specified.
    """
    now = datetime.datetime.now()
    now_formatted = now.strftime("%Y%m%d-%H%M")
    if prefix is None:
        return now_formatted + ".log"
    else:
        return prefix + "-" + now_formatted + ".log"


# Functions for making NMR strings #
def build_1H_peak(
    HNMR_sim_peaks: dict,
    peak: str,
    jitter_peaks: bool = False,
    mode: str = "adaptive",
    token_space: str = "separate",
) -> Tuple[float, str]:
    if (
        HNMR_sim_peaks[peak]["rangeMax"] - HNMR_sim_peaks[peak]["rangeMin"] > 0.15
        and mode == "adaptive"
    ) or mode == "range":
        max_val = (
            jitter(round(HNMR_sim_peaks[peak]["rangeMax"], 2), jitter_range=0.2)
            if jitter_peaks
            else round(HNMR_sim_peaks[peak]["rangeMax"], 2)
        )
        min_val = (
            jitter(round(HNMR_sim_peaks[peak]["rangeMin"], 2), jitter_range=0.2)
            if jitter_peaks
            else round(HNMR_sim_peaks[peak]["rangeMin"], 2)
        )

        if token_space == "separate":
            max_val = "1H{:.2f}".format(max_val)
            min_val = "1H{:.2f}".format(min_val)
        elif token_space == "shared":
            max_val = "{:.1f}".format(max_val * 10)
            min_val = "{:.1f}".format(min_val * 10)

        peak_string = "| {} {} {} {}H ".format(
            min_val,
            max_val,
            HNMR_sim_peaks[peak]["category"],
            HNMR_sim_peaks[peak]["nH"],
        )

        return HNMR_sim_peaks[peak]["rangeMax"], peak_string

    else:
        centroid = (
            jitter(round(HNMR_sim_peaks[peak]["centroid"], 2), jitter_range=0.2)
            if jitter_peaks
            else round(HNMR_sim_peaks[peak]["centroid"], 2)
        )

        if token_space == "separate":
            centroid = "1H{:.2f}".format(centroid)
        elif token_space == "shared":
            centroid = "{:.1f}".format(centroid * 10)

        peak_string = "| {} {} {}H ".format(
            centroid, HNMR_sim_peaks[peak]["category"], HNMR_sim_peaks[peak]["nH"]
        )

        return HNMR_sim_peaks[peak]["centroid"], peak_string


def build_hnmr_string(
    smiles: str,
    peak_dict: dict,
    mode: str = "adaptive",
    header: bool = True,
    token_space: str = "same",
    n_aug: int = 0,
) -> List[str]:
    # Construct NMR string

    mol = Chem.MolFromSmiles(smiles)
    formula = rdMolDescriptors.CalcMolFormula(mol)

    if header:
        formula_split = tokenize_formula(formula)
        formula_tokenized = " ".join(list(filter(None, formula_split)))
        nmr_header = "{} 1HNMR ".format(formula_tokenized)
    else:
        nmr_header = "1HNMR "

    peak_strings = list()

    for i in range(n_aug + 1):
        # No augmentation for the first set
        processed_peak = dict()
        for peak in peak_dict.keys():
            peak_pos, peak_string = build_1H_peak(
                peak_dict,
                peak,
                jitter_peaks=True if i > 0 else False,
                mode=mode,
                token_space=token_space,
            )
            processed_peak[peak_pos] = peak_string

        # Order such that peaks are in ascending order
        peak_string = nmr_header
        for _, peak in sorted(processed_peak.items()):
            peak_string = peak_string + peak

        peak_strings.append(peak_string)

    return peak_strings


def build_cnmr_string(
    C_NMR_entry: dict,
    header: bool = False,
    smiles: Optional[str] = None,
    token_space="shared",
    n_aug: int = 0,
) -> List[str]:
    if header:
        mol = Chem.MolFromSmiles(smiles)
        formula = rdMolDescriptors.CalcMolFormula(mol)
        formula_split = tokenize_formula(formula)
        formula_tokenized = " ".join(list(filter(None, formula_split)))

        nmr_header = "{} 13C_NMR".format(formula_tokenized)

    else:
        nmr_header = "13C_NMR"

    nmr_strings = list()
    for i in range(n_aug + 1):
        peaks = list()

        for peak in C_NMR_entry["peaks"].values():
            if peak["delta (ppm)"] > 230 or peak["delta (ppm)"] < -20:
                continue

            value = float(round(peak["delta (ppm)"], 1))
            value_str = str(jitter(value, jitter_range=0.5) if i > 0 else value)

            if token_space == "separate":
                value_str = "13C" + str(value)

            peaks.append(value_str)

        peaks = sorted(peaks)

        nmr_string = nmr_header
        for peak in peaks:
            nmr_string += f" {peak}"
        nmr_strings.append(nmr_string)

    return nmr_strings


def make_nmr(
    mode: str,
    component: str,
    hnmr: Optional[dict] = None,
    cnmr: Optional[dict] = None,
    hnmr_mode: str = "range",
    token_space: str = "shared",
) -> str:
    if mode == "combined":
        if hnmr is None or cnmr is None:
            raise ValueError("For mode combined both hnmr and cnmr have to be defined.")

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
        if hnmr is None:
            raise ValueError("For mode hnmr hnmr can't be None.")

        hnmr_string = build_hnmr_string(
            smiles=component,
            peak_dict=hnmr,
            mode=hnmr_mode,
            header=False,
            token_space=token_space,
        )[0]

        nmr_string = " " + hnmr_string

    elif mode == "cnmr":
        if cnmr is None:
            raise ValueError("For mode cnmr cnmr can't be None.")

        cnmr_string = build_cnmr_string(
            cnmr,
            header=False,
            token_space=token_space,
        )[0]
        nmr_string = " " + cnmr_string

    return nmr_string


# Functions and classes supporting the generation of Tanimoto sets #
class TanimotoContainer:
    def __init__(self, n_tanimoto_bins: int) -> None:
        tanimoto_boxes = np.linspace(0, 1, n_tanimoto_bins + 1)

        self.length = 0
        self.tanimoto_results_container: Dict[str, list] = {
            "{:.3f}".format(tanimoto_boxes[i]): list() for i in range(n_tanimoto_bins)
        }

    def add(self, to_add_dict: Dict) -> None:
        for key in to_add_dict:
            self.tanimoto_results_container[key].extend(to_add_dict[key])
            self.length += len(to_add_dict[key])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, key: str) -> List:
        return self.tanimoto_results_container[key]

    def get_keys(self) -> KeysView:
        return self.tanimoto_results_container.keys()

    def to_dataframe(self) -> pd.DataFrame:
        results_combined = list()

        for key in self.tanimoto_results_container:
            for sample in self.tanimoto_results_container[key]:
                sample["tanimoto_bin"] = key
            results_combined.extend(self.tanimoto_results_container[key])

        return pd.DataFrame(results_combined)


@lru_cache(maxsize=5000000)
def calculate_fingerprint(smiles: str) -> DataStructs.cDataStructs.ExplicitBitVect:
    mol = Chem.MolFromSmiles(smiles)
    mol_fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, useChirality=True, radius=2, nBits=1024
    )
    return mol_fp


def calculate_tanimoto(smiles1: str, smiles2: str) -> float:
    smiles1_fp, smiles2_fp = calculate_fingerprint(smiles1), calculate_fingerprint(
        smiles2
    )

    return DataStructs.FingerprintSimilarity(smiles1_fp, smiles2_fp)


def get_tanimoto_set(
    tgt_smiles: str,
    nmr_df: pd.DataFrame,
    mode: str,
    hnmr_mode: str,
    token_space: str,
    non_matching: bool,
    n_tanimoto_bins: int,
) -> Dict[str, list]:
    """Creates tanimoto sets for a single molecule."""

    similarities = list()
    for smiles in nmr_df.index:
        similarities.append(calculate_tanimoto(tgt_smiles, smiles))

    similarities_np = np.array(similarities)

    nmr = make_nmr(
        mode,
        tgt_smiles,
        hnmr=nmr_df.loc[tgt_smiles]["1H_NMR_sim"]["peaks"],
        cnmr=nmr_df.loc[tgt_smiles]["13C_NMR_sim"],
        hnmr_mode=hnmr_mode,
        token_space=token_space,
    )

    tanimoto_boxes = np.linspace(0, 1, n_tanimoto_bins + 1)

    src_tgt_pairs: Dict[str, list] = {
        "{:.3f}".format(tanimoto_boxes[i]): list() for i in range(n_tanimoto_bins)
    }
    for i in range(len(tanimoto_boxes) - 1):
        sim_smiles = nmr_df.index[
            np.logical_and(
                similarities_np >= tanimoto_boxes[i],
                similarities_np < tanimoto_boxes[i + 1],
            )
        ]

        # If smaller than 4 can't constuct proper sets
        if len(sim_smiles) < 4:
            continue

        # Matching set
        selected_smiles = [tgt_smiles] + np.random.choice(sim_smiles, 3, replace=False)

        nmr_input = tokenize_smiles(".".join(selected_smiles)) + nmr
        src_tgt_pairs["{:.3f}".format(tanimoto_boxes[i])].append(
            {"nmr_input": nmr_input, "smiles": tokenize_smiles(tgt_smiles)}
        )

        if not non_matching:
            continue

        # Non matching set
        selected_smiles = [tgt_smiles] + np.random.choice(sim_smiles, 4, replace=False)
        nmr_input = tokenize_smiles(".".join(selected_smiles)) + nmr

        src_tgt_pairs["{:.3f}".format(tanimoto_boxes[i])].append(
            {"nmr_input": nmr_input, "smiles": DEFAULT_NON_MATCHING_TOKEN}
        )

    return src_tgt_pairs


def tanimoto_set_worker(
    worker_id: int,
    smiles_chunks: list,
    nmr_df: pd.DataFrame,
    mode: str,
    n_samples: int,
    return_list: List,
    status_list: List,
    hnmr_mode: str = "range",
    token_space: str = "shared",
    non_matching=True,
    tanimoto_bins: int = 5,
) -> None:
    container = TanimotoContainer(n_tanimoto_bins=tanimoto_bins)

    for smiles in smiles_chunks:
        tanimoto_dict = get_tanimoto_set(
            smiles,
            nmr_df,
            mode,
            hnmr_mode=hnmr_mode,
            token_space=token_space,
            non_matching=non_matching,
            n_tanimoto_bins=tanimoto_bins,
        )
        container.add(tanimoto_dict)

        status_list[worker_id] = len(container)
        if len(container) > n_samples:
            break

    container_df = container.to_dataframe()
    container_df["merged_input_output"] = (
        container_df["nmr_input"] + container_df["smiles"]
    )
    container_df = container_df.drop_duplicates(subset="merged_input_output")
    container_df.drop(columns="merged_input_output", inplace=True)

    return_list.append(container_df)
