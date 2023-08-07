import datetime
import getpass
import logging
import os
import subprocess
import time
from pathlib import Path

import click
import pandas as pd
import psutil
import pyautogui

# Orchestrate


def open_mnova(mnova_path: Path) -> subprocess.Popen:
    logging.info("Opening MestreNova")
    process = subprocess.Popen([mnova_path], preexec_fn=os.setsid)
    time.sleep(2.5)
    return process


def kill_mnova() -> None:
    logging.info("Killing MestreNova")

    for proc in psutil.process_iter():
        with proc.oneshot():
            name = proc.name()
            username = proc.username()

        if name == "MestReNova" and username == getpass.getuser():
            proc.terminate()
            time.sleep(0.5)
            try:
                proc.kill()
            except psutil.NoSuchProcess:
                pass


def reopen_mnova(mnova_path: Path) -> subprocess.Popen:
    kill_mnova()
    return open_mnova(mnova_path)


def run_simulations(
    smiles_df: pd.DataFrame,
    out_dir: Path,
    sim_type: str = "1H",
    index_start: int = 0,
    mnova_path: Path = Path("/Applications/MestReNova.app/Contents/MacOS/MestReNova"),
    script_path: Path = Path("/Users/arv/Projects/nmr/nmr_generation/scripts/"),
):
    open_mnova(mnova_path)
    for i in range(len(smiles_df)):
        smiles = smiles_df.iloc[i]["Smiles"]
        index = smiles_df.index[i]

        output_path = out_dir / f"{sim_type}_{index}.csv"

        logging.info(
            "Running: index {} smiles {} output_path {}".format(
                index, smiles, output_path
            )
        )

        if sim_type == "1H":
            subprocess.Popen(
                [
                    mnova_path,
                    script_path / "1H_predictor.qs",
                    "-sf",
                    f"predict_1H,{smiles},{output_path}",
                ]
            )
        elif sim_type == "13C":
            subprocess.Popen(
                [
                    mnova_path,
                    script_path / "13C_predictor.qs",
                    "-sf",
                    f"predict_13C,{smiles},{output_path}",
                ]
            )

        start_time = datetime.datetime.now()
        while not os.path.isfile(output_path):
            now = datetime.datetime.now()
            if (now - start_time).seconds > 30:
                pyautogui.screenshot(
                    os.path.join(out_dir, f"{sim_type}_{index}_failure.png")
                )
                logging.warning(
                    f"Smiles: {smiles}, index: {index} timed out reopening MestreNova and skipping."
                )

                reopen_mnova(mnova_path)

                break
            time.sleep(0.5)
    time.sleep(2.5)
    kill_mnova()
    time.sleep(2.5)


@click.command()
@click.option(
    "--smiles_csv",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Input smiles csv",
)
@click.option(
    "--out_folder", type=click.Path(path_type=Path), required=True, help="Output folder"
)
@click.option(
    "--sim_type",
    default="1H",
    type=click.Choice(["1H", "13C"]),
    help="Simulation type. Either 1H or 13C",
)
@click.option(
    "--chunk_size",
    default=100,
    type=int,
    help="Size of each chunk before MestreNova is restarted.",
)
@click.option(
    "--start_index",
    default=None,
    type=int,
    help="At which index to start in the csv file. Useful for restarting if a run fails midway.",
)
@click.option(
    "--mnova_path",
    type=click.Path(exists=True, path_type=Path),
    default=Path("/opt/MestReNova/bin/MestReNova"),
    help="Path to the MestreNova executable.",
)
@click.option(
    "--script_path",
    type=click.Path(exists=True, path_type=Path),
    default=Path("/home/vnc1/Projects/NMR_gen_mestre_nova/scripts"),
    help="Path to folder containing the 1H and 13C prediction scripts.",
)
def main(
    smiles_csv: Path,
    out_folder: Path,
    sim_type: str,
    chunk_size: int,
    start_index: int,
    mnova_path: Path,
    script_path: Path,
):
    smiles_data = pd.read_csv(smiles_csv, index_col=0)
    if start_index is not None:
        smiles_data = smiles_data.loc[start_index:]

    os.makedirs(out_folder, exist_ok=True)

    log_file = os.path.join(out_folder, "log.txt")
    logging.basicConfig(
        filename=log_file,
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    chunked_dfs = [
        smiles_data[i : i + chunk_size]
        for i in range(0, smiles_data.shape[0], chunk_size)
    ]
    for i, df in enumerate(chunked_dfs):
        t1 = datetime.datetime.now()

        run_simulations(
            df,
            out_folder,
            sim_type=sim_type,
            mnova_path=mnova_path,
            script_path=script_path,
        )

        t2 = datetime.datetime.now()
        logging.info(f"Set: {i} with batch_size {chunk_size} took: {t2-t1}")


if __name__ == "__main__":
    main()
