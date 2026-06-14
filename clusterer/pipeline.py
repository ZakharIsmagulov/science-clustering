from HdbscanModel import HdbscanModel
from ClusterizationModel import ClusterizationModel
import logging
import argparse
from pathlib import Path
from file_io import load_parquet_batches
from Config import ConfigItem
import pandas as pd
import cupy as cp
import uuid
import joblib


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interpreting clusterization"
    )

    parser.add_argument(
        "--parquet-name",
        required=True,
        help="Name of the parquet file with embeddings (float32)",
    )
    parser.add_argument(
        "--params-name",
        required=True,
        help="Name of the json file with parameters",
    )
    parser.add_argument(
        "--duckdb-name",
        required=True,
        help="Name of the duckdb file with metadata",
    )
    return parser.parse_args()


def clear_directory(path: str, only_temp: bool = False):
    path = Path(path)

    if not path.exists():
        path.mkdir(parents=True)
        return

    for item in path.iterdir():
        if not item.is_dir() and (not only_temp or "_umap" in item.name or "_hdbscan" in item.name):
            item.unlink()


def set_logger():
    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)

    handler = logging.FileHandler("logger.log")
    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False

    return logger


def get_latest_id():
    directory = Path("temp/")
    latest_id = None
    latest_mtime = None
    for path in directory.glob("*.parquet"):
        stem = path.stem
        if stem.endswith("_umap"):
            current_id = stem.removesuffix("_umap")
        elif stem.endswith("_hdbscan"):
            current_id = stem.removesuffix("_hdbscan")
        else:
            current_id = stem
        mtime = path.stat().st_mtime
        if latest_mtime is None or mtime > latest_mtime:
            latest_mtime = mtime
            latest_id = current_id
    return latest_id


def get_config_ids():
    directory = Path("temp/")

    ids: set[str] = set()

    for path in directory.glob("*.parquet"):
        stem = path.stem

        if stem.endswith("_umap") or stem.endswith("_hdbscan"):
            continue

        ids.add(stem)

    return sorted(ids)


def find_last_checkpoint():
    last_id = get_latest_id()

    if Path(f"temp/{last_id}.parquet").exists():
        return f"temp/{last_id}.parquet"

    if Path(f"temp/{last_id}_hdbscan.parquet").exists():
        return f"temp/{last_id}_hdbscan.parquet"

    if Path(f"temp/{last_id}_umap.parquet").exists():
        return f"temp/{last_id}_umap.parquet"

    return None


def is_unique_id(config_id: str):
    config_ids = get_config_ids()
    if config_id in config_ids:
        return False
    return True


def generate_new_id(config_id: str):
    new_id = config_id
    while not is_unique_id(new_id):
        new_id = config_id + "_" + uuid.uuid4().hex[:12]
    return new_id


def cluster_pipeline(args, clust_model: ClusterizationModel | None, logger_name: str = "logger"):
    logger = logging.getLogger(logger_name)
    logger.info("Starting clusterization")

    logger.info("Clearing temporary umap and hdbscan files in temp/")
    clear_directory("temp/", True)

    logger.info("Loading config")
    config = ConfigItem.from_json_file(args.params_name)

    if not is_unique_id(config.id):
        logger.warning("Config id is not unique. Changing...")
        config.id = generate_new_id(config.id)
        logger.info(f"New config id is {config.id}")

    if clust_model is None:
        logger.info("Loading embeddings from parquet")
        ids, embeds = load_parquet_batches(args.parquet_name)

        logger.info("Creating clusterization model")
        clust_model = ClusterizationModel(
            ids=ids,
            embeddings=embeds,
            config=config,
            duckdb_path=args.duckdb_name,
        )
    else:
        logger.info("Updating pipeline parameters")
        clust_model.update_config(config)

    try:
        logger.info("Processing clusterization")
        clust_model.process_clustering()
    except Exception:
        logger.exception("Clusterization failed")
        raise

    print("-------------------------Clusterization finished-------------------------")
    logger.info("-------------------------Clusterization finished-------------------------")

    return clust_model


def list_pipeline():
    config_ids = get_config_ids()
    print("\n".join([f"{i}. {_id}" for i, _id in enumerate(config_ids)]))


def interpret_pipeline(args, config_num: int, logger_name: str = "logger"):
    logger = logging.getLogger(logger_name)
    logger.info("Starting interpretation")

    try:
        config_id = get_config_ids()[config_num]
    except Exception:
        logger.info(f"No such number: {config_num}")
        print(f"No such number: {config_num}")
        raise FileNotFoundError(f"No such number: {config_num}")
    file_path = f"temp/{config_id}.parquet"
    if not Path(file_path).exists():
        logger.info(f"No such file: {file_path}")
        print(f"No such file: {file_path}")
        raise FileNotFoundError(f"No such file: {file_path}")
    logger.info(f"Loading data from {file_path}")
    clusters_df = pd.read_parquet(file_path, engine="pyarrow")
    clust_model = ClusterizationModel(
        ids=None,
        embeddings=None,
        config=None,
        duckdb_path=args.duckdb_name,
    )
    clust_model.id = config_id

    try:
        logger.info("Interpreting clusters")
        clust_model.process_interpreting(clusters_df)
    except Exception:
        logger.exception("Interpreting failed")
        raise

    print("-------------------------Interpretation finished-------------------------")
    logger.info("-------------------------Interpretation finished-------------------------")


def continue_failed_clusterization(args, checkpoint_file: str, logger_name: str = "logger"):
    logger = logging.getLogger(logger_name)
    logger.info("Starting interpretation")

    logger.info("Loading config")
    config = ConfigItem.from_json_file(args.params_name)

    if "umap" in checkpoint_file:
        logger.info("Found UMAP checkpoint. Continue from HDBSCAN")
        if not Path(checkpoint_file).exists():
            logger.info(f"No such file: {checkpoint_file}")
            print(f"No such file: {checkpoint_file}")
            raise FileNotFoundError(f"No such file: {checkpoint_file}")

        config_id = Path(checkpoint_file).stem.replace("_umap", "")
        config.id = config_id

        logger.info("Loading embeddings from parquet")
        ids, embeds = load_parquet_batches(args.parquet_name)

        logger.info("Recovering clusterization model")
        clust_model = ClusterizationModel(
            ids=ids,
            embeddings=embeds,
            config=config,
            duckdb_path=args.duckdb_name,
        )
        clust_model.start_from = "hdbscan"
        clust_model.done_state = "umap"
        _, reduced = load_parquet_batches(checkpoint_file, "id", "embeddings", 16, 60_000)
        clust_model.reduced = reduced
    elif "hdbscan" in checkpoint_file:
        logger.info("Found HDBSCAN checkpoint. Continue from periphery")
        checkpoint_file = checkpoint_file.replace("parquet", "joblib")
        if not Path(checkpoint_file).exists():
            logger.info(f"No such file: {checkpoint_file}")
            print(f"No such file: {checkpoint_file}")
            raise FileNotFoundError(f"No such file: {checkpoint_file}")

        config_id = Path(checkpoint_file).stem.replace("_hdbscan", "")
        config.id = config_id

        logger.info("Loading embeddings from parquet")
        ids, embeds = load_parquet_batches(args.parquet_name)

        logger.info("Recovering clusterization model")
        clust_model = ClusterizationModel(
            ids=ids,
            embeddings=embeds,
            config=config,
            duckdb_path=args.duckdb_name,
        )
        clust_model.start_from = "periphery"
        clust_model.done_state = "hdbscan"
        loaded_model = joblib.load(checkpoint_file)
        hdbscan_model = HdbscanModel(config=None, model=loaded_model)
        clust_model.hdbscan_model = hdbscan_model
    else:
        logger.info("Found periphery checkpoint. Continue from metrics")
        if not Path(checkpoint_file).exists():
            logger.info(f"No such file: {checkpoint_file}")
            print(f"No such file: {checkpoint_file}")
            raise FileNotFoundError(f"No such file: {checkpoint_file}")

        config_id = Path(checkpoint_file).stem.replace("_hdbscan", "")
        config.id = config_id

        logger.info("Loading embeddings from parquet")
        ids, embeds = load_parquet_batches(args.parquet_name)

        logger.info("Recovering clusterization model")
        clust_model = ClusterizationModel(
            ids=ids,
            embeddings=embeds,
            config=config,
            duckdb_path=args.duckdb_name,
        )
        clust_model.start_from = "metrics"
        clust_model.done_state = "periphery"

    try:
        logger.info("Processing clusterization")
        clust_model.process_clustering()
    except Exception:
        logger.exception("Clusterization failed")
        raise

    print("-------------------------Recovering clusterization finished-------------------------")
    logger.info("-------------------------Recovering clusterization finished-------------------------")

    return clust_model


def interactive_pipeline(args, logger_name: str = "logger"):
    logger = logging.getLogger(logger_name)

    logger.info("Starting interactive pipeline")

    clust_model = None

    while True:
        print("Interactive clusterization pipeline")
        print("Commands:")
        print("  cluster              - run clustering with current config (clears temp files from umap and hdbscan)")
        print("  list                 - show saved clusterizations and metrics")
        print("  interpret <num>      - interpret selected clusterization (clears res/ folder, you should backup)")
        print("  continue-failed      - continue latest failed checkpoint")
        print("  exit                 - stop")

        try:
            command = input("> ").strip()
        except EOFError:
            logger.info("Stopping interactive pipeline")
            return

        if not command:
            continue

        if command == "cluster":
            try:
                clust_model = cluster_pipeline(args, clust_model, logger_name)
            except Exception:
                logger.exception("Cluster command failed")
            continue

        if command == "list":
            try:
                list_pipeline()
            except Exception:
                logger.exception("List command failed")
            continue

        if command.startswith("interpret "):
            config_num = command.removeprefix("interpret ").strip()

            if not config_num:
                print("Usage: interpret <num>")
                continue

            try:
                interpret_pipeline(args, int(config_num), logger_name)
            except Exception:
                logger.exception("Interpret command failed")
            continue

        if command.startswith("continue-failed"):
            checkpoint_file = find_last_checkpoint()

            if checkpoint_file is None:
                print("No checkpoint found in temp/")
                continue

            try:
                clust_model = continue_failed_clusterization(args, checkpoint_file, logger_name)
            except Exception:
                logger.exception("Continue-failed command failed")
            continue

        if command == "exit":
            logger.info("User stopped interactive pipeline")
            return

        print(f"Unknown command: {command}")


if __name__ == "__main__":
    _ = set_logger()
    interactive_pipeline(parse_args())
