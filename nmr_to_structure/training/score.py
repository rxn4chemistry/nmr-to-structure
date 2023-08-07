from ctypes import ArgumentError
from typing import Dict, List, Tuple

import click
import pandas as pd
import tqdm
from rdkit import Chem, RDLogger
from rdkit.Chem import Fragments, rdMolDescriptors
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol

functional_groups = {
    "Alcohol": Chem.MolFromSmarts("[OX2H][CX4;!$(C([OX2H])[O,S,#7,#15])]"),
    "Carboxylic Acid": Chem.MolFromSmarts("[CX3](=O)[OX2H1]"),
    "Ester": Chem.MolFromSmarts("[#6][CX3](=O)[OX2H0][#6]"),
    "Ether": Fragments.fr_ether,
    "Aldehyde": Chem.MolFromSmarts("[CX3H1](=O)[#6]"),
    "Ketone": Chem.MolFromSmarts("[#6][CX3](=O)[#6]"),
    "Alkene": Chem.MolFromSmarts("[CX3]=[CX3]"),
    "Alkyne": Chem.MolFromSmarts("[$([CX2]#C)]"),
    "Benzene": Fragments.fr_benzene,
    "Primary Amine": Chem.MolFromSmarts("[NX3;H2;!$(NC=[!#6]);!$(NC#[!#6])][#6]"),
    "Secondary Amine": Fragments.fr_NH1,
    "Tertiary Amine": Fragments.fr_NH0,
    "Amide": Chem.MolFromSmarts("[NX3][CX3](=[OX1])[#6]"),
    "Cyano": Chem.MolFromSmarts("[NX1]#[CX2]"),
    "Fluorine": Chem.MolFromSmarts("[#6][F]"),
    "Chlorine": Chem.MolFromSmarts("[#6][Cl]"),
    "Iodine": Chem.MolFromSmarts("[#6][I]"),
    "Bromine": Chem.MolFromSmarts("[#6][Br]"),
    "Sulfonamide": Chem.MolFromSmarts("[#16X4]([NX3])(=[OX1])(=[OX1])[#6]"),
    "Sulfone": Chem.MolFromSmarts("[#16X4](=[OX1])(=[OX1])([#6])[#6]"),
    "Sulfide": Chem.MolFromSmarts("[#16X2H0]"),
    "Phosphoric Acid": Chem.MolFromSmarts(
        "[$(P(=[OX1])([$([OX2H]),$([OX1-]),$([OX2]P)])([$([OX2H]),$([OX1-]),$([OX2]P)])[$([OX2H]),$([OX1-]),$([OX2]P)]),$([P+]([OX1-])([$([OX2H]),$([OX1-]),$([OX2]P)])([$([OX2H]),$([OX1-]),$([OX2]P)])[$([OX2H]),$([OX1-]),$([OX2]P)])]"
    ),
    "Phosphoester": Chem.MolFromSmarts(
        "[$(P(=[OX1])([OX2][#6])([$([OX2H]),$([OX1-]),$([OX2][#6])])[$([OX2H]),$([OX1-]),$([OX2][#6]),$([OX2]P)]),$([P+]([OX1-])([OX2][#6])([$([OX2H]),$([OX1-]),$([OX2][#6])])[$([OX2H]),$([OX1-]),$([OX2][#6]),$([OX2]P)])]"
    ),
}


def strip(data: List[str]) -> List[str]:
    return [smiles.replace(" ", "") for smiles in data]


def load_data(
    inference_path: str, tgt_path: str, n_beams: int = 10
) -> Tuple[List[List[str]], List[str]]:
    # Load tgt
    with open(tgt_path, "r") as f:
        tgt = f.readlines()
    tgt = strip(tgt)

    # Load preds
    with open(inference_path, "r") as f:
        preds = f.readlines()
    preds_stripped = strip(preds)
    pred_batched = [
        preds_stripped[i * n_beams : (i + 1) * n_beams]
        for i in range(len(preds_stripped) // n_beams)
    ]

    return pred_batched, tgt


def match_group(mol: Chem.Mol, func_group) -> int:
    if isinstance(type(func_group), Chem.Mol):
        n = len(mol.GetSubstructMatches(func_group))
    else:
        n = func_group(mol)
    return 0 if n == 0 else 1


def get_functional_groups(mol):
    func_groups = dict()
    for func_group_name, smarts in functional_groups.items():
        func_groups[func_group_name] = match_group(mol, smarts)
    return func_groups


def match_smiles(
    tgt_smiles: str, pred_smiles: list, suffix: str = ""
) -> Dict[str, int]:
    if len(pred_smiles) == 0:
        return {
            "top1" + suffix: 0,
            "top5" + suffix: 0,
            "top10" + suffix: 0,
        }

    if tgt_smiles == pred_smiles[0]:
        return {"top1" + suffix: 1, "top5" + suffix: 1, "top10" + suffix: 1}
    elif len(pred_smiles) < 5 and tgt_smiles in pred_smiles:
        return {"top1" + suffix: 0, "top5" + suffix: 1, "top10" + suffix: 1}
    elif tgt_smiles in pred_smiles[:5]:
        return {"top1" + suffix: 0, "top5" + suffix: 1, "top10" + suffix: 1}
    elif tgt_smiles in pred_smiles:
        return {"top1" + suffix: 0, "top5" + suffix: 0, "top10" + suffix: 1}
    else:
        return {"top1" + suffix: 0, "top5" + suffix: 0, "top10" + suffix: 0}


def score(preds: List[List[str]], tgt: List[str]) -> pd.DataFrame:
    results = dict()

    for pred_smiles, tgt_smiles in tqdm.tqdm(zip(preds, tgt), total=len(tgt)):
        mol = Chem.MolFromSmiles(tgt_smiles)
        tgt_smiles = Chem.MolToSmiles(mol)
        hac = rdMolDescriptors.CalcNumHeavyAtoms(mol)
        functional_groups = get_functional_groups(mol)

        tgt_scaffold = GetScaffoldForMol(mol)
        tgt_scaffold_smiles = Chem.MolToSmiles(tgt_scaffold)

        pred_smiles_canon = list()
        pred_scaffold = list()
        for pred in pred_smiles:
            try:
                pred_mol = Chem.MolFromSmiles(pred)

                if pred_mol is None:
                    raise ArgumentError("Invalid Smiles")

                pred_smiles_canon.append(Chem.MolToSmiles(pred_mol))

                pred_scaffold_mol = GetScaffoldForMol(pred_mol)
                pred_scaffold.append(Chem.MolToSmiles(pred_scaffold_mol))
            except ArgumentError:
                continue

        results[tgt_smiles] = {
            "hac": hac,
            **functional_groups,
            "predictions": pred_smiles_canon,
        }

        # Score match of the smiles string
        results_smiles = match_smiles(tgt_smiles, pred_smiles_canon)
        results[tgt_smiles].update(results_smiles)

        # Score scaffold match
        results_scaffold = match_smiles(
            tgt_scaffold_smiles, pred_scaffold, suffix="_scaffold"
        )
        results[tgt_smiles].update(results_scaffold)

    return pd.DataFrame.from_dict(results, orient="index")


@click.command()
@click.option("--tgt_path", required=True, help="Path to the tgt file")
@click.option("--inference_path", required=True, help="Path to the inference file")
@click.option("--n_beams", default=10, help="Number of beams")
def main(inference_path: str, tgt_path: str, n_beams: int = 10) -> None:
    RDLogger.DisableLog("rdApp.*")

    preds, tgt = load_data(
        inference_path=inference_path, tgt_path=tgt_path, n_beams=n_beams
    )
    results_score = score(preds, tgt)

    print("Results: Smiles Match")
    print(
        "Top 1: {:.3f}, Top 5: {:.3f}, Top 10: {:.3f}".format(
            results_score["top1"].sum() / len(results_score) * 100,
            results_score["top5"].sum() / len(results_score) * 100,
            results_score["top10"].sum() / len(results_score) * 100,
        )
    )

    print("Results: Scaffold Match")
    print(
        "Top 1 Scaffold: {:.3f}, Top 5 Scaffold: {:.3f}, Top 10 Scaffold: {:.3f}".format(
            results_score["top1_scaffold"].sum() / len(results_score) * 100,
            results_score["top5_scaffold"].sum() / len(results_score) * 100,
            results_score["top10_scaffold"].sum() / len(results_score) * 100,
        )
    )


if __name__ == "__main__":
    main()
