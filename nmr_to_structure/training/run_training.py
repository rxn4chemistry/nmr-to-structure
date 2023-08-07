import logging
import os
import subprocess

import click

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def create_input(
    template_path: str,
    output_path: str,
    data_path: str,
    src_train_path: str,
    tgt_train_path: str,
    src_val_path: str,
    tgt_val_path: str,
    log_path: str,
) -> str:
    # Create nmt yaml
    with open(template_path, "r") as f:
        template = f.read()

    src_vocab_path = os.path.join(output_path, "data", "vocab", "vocab.src")
    tgt_vocab_path = os.path.join(output_path, "data", "vocab", "vocab.tgt")

    save_model_path = os.path.join(output_path, "model")

    input_file = template.format(
        data_path,
        src_vocab_path,
        tgt_vocab_path,
        src_train_path,
        tgt_train_path,
        src_val_path,
        tgt_val_path,
        log_path,
        save_model_path,
    )

    input_file_path = os.path.join(output_path, "input.yaml")
    with open(input_file_path, "w") as f:
        f.write(input_file)

    return input_file_path


def gen_vocab(log_path: str, input_file_path: str) -> None:
    # Create vocab
    with open(os.path.join(log_path, "vocab.log"), "w") as out:
        subprocess.call(
            ["onmt_build_vocab", "-config", input_file_path, "-n_sample", "-1"],
            stdout=out,
            stderr=out,
        )


@click.command()
@click.option("--template_path", required=True, help="Path to the config template")
@click.option("--data_folder", required=True, help="Data folder")
def main(template_path: str, data_folder: str):
    logging.basicConfig(level="INFO")

    log_path = os.path.join(data_folder, "logs")
    os.makedirs(log_path, exist_ok=True)

    data_path = os.path.join(data_folder, "data")

    src_train_path = os.path.join(data_path, "src-train.txt")
    tgt_train_path = os.path.join(data_path, "tgt-train.txt")
    src_val_path = os.path.join(data_path, "src-val.txt")
    tgt_val_path = os.path.join(data_path, "tgt-val.txt")

    # Create input yaml
    logger.info("Creating input...")
    input_file_path = create_input(
        template_path,
        data_folder,
        data_path,
        src_train_path,
        tgt_train_path,
        src_val_path,
        tgt_val_path,
        log_path,
    )

    # Create vocab files
    logger.info("Creating vocab...")
    gen_vocab(log_path, input_file_path)

    input_file_path = os.path.join(data_folder, "input.yaml")

    # Start trainig
    logger.info("Starting training...")
    train_logs_path = os.path.join(log_path, "train")
    out_train = os.path.join(train_logs_path, "out.txt")
    err_train = os.path.join(train_logs_path, "err.txt")

    os.makedirs(train_logs_path, exist_ok=True)

    with open(out_train, "w") as out_file, open(err_train, "w") as err_file:
        subprocess.call(
            ["onmt_train", "-config", input_file_path], stdout=out_file, stderr=err_file
        )


if __name__ == "__main__":
    main()
