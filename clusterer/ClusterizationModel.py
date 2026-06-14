import gc
import time
import numpy as np
import json
import pandas as pd
import cupy as cp
from logging import getLogger
from typing import Literal
from AppException import *
from Config import ConfigItem
from UMAPModel import UmapModel
from HdbscanModel import HdbscanModel
from PeripheryModel import PeripheryModel
from clusterization_metrics import simple_metrics, get_interpretable_metrics
from topic_metrics import get_topic_metrics
from clusters_annotation import annotate_clusters
from file_io import save_temp_parquet, save_temp_joblib, del_temp_parquet, ReducedData, ClustersData, TempData


class ClusterizationModel:
    def __init__(self, ids: list | None, embeddings: np.ndarray | None, config: ConfigItem | None,
                 duckdb_path: str,
                 logger_name: str = "logger"):
        self.logger = getLogger(logger_name)
        self.duckdb_path = duckdb_path

        self.start_from: Literal["umap", "hdbscan", "periphery", "metrics"] = "umap"
        self.done_state: Literal["idle", "umap", "hdbscan", "periphery"] = "idle"
        self.id = config.id if config else None
        self.umap_config = config.config.umap if config else None
        self.hdbscan_config = config.config.hdbscan if config else None
        self.periphery_config = config.config.periphery if config else None

        self.hdbscan_model: HdbscanModel | None = None

        self.ids = ids
        self.embeddings = embeddings
        self.reduced: np.ndarray | cp.ndarray | None = None

        self.tau = None
        self.delta = None

        self.metrics = {
            "clusterization_id": self.id,
        }
        self.timings = {
            "clusterization_id": self.id,
        }

    def save_model_files(self):
        if self.done_state == "idle" and self.reduced is not None:
            save_temp_parquet(
                self.id,
                TempData(value=ReducedData(
                    ids=self.ids,
                    mat=self.reduced,
                ))
            )
        elif self.done_state == "umap" and self.hdbscan_model is not None:
            save_temp_parquet(
                self.id,
                TempData(value=ClustersData(
                    ids=self.ids,
                    topics=self.hdbscan_model.get_labels(),
                    is_core=self.hdbscan_model.get_is_core(),
                    probs=self.hdbscan_model.get_probs(),
                    file_type="hdbscan",
                ))
            )
            save_temp_joblib(self.id, self.hdbscan_model.model)
            del_temp_parquet(self.id, "umap")
        elif self.done_state == "hdbscan" and self.tau is not None:
            save_temp_parquet(
                self.id,
                TempData(value=ClustersData(
                    ids=self.ids,
                    topics=self.hdbscan_model.get_labels(),
                    is_core=self.hdbscan_model.get_is_core(),
                    probs=self.hdbscan_model.get_probs(),
                    file_type="periphery",
                ))
            )
            del_temp_parquet(self.id, "hdbscan")

    def process_clustering(self):
        t0 = time.perf_counter()

        self.free_pool()

        if self.start_from == "umap":
            self.logger.info("Starting UMAP")
            self.reduce_umap()
            self.logger.info("Finished UMAP")
            self.save_model_files()

            self.free_pool()

        self.done_state = "umap"

        if self.start_from in {"umap", "hdbscan"}:
            self.logger.info("Starting HDBSCAN clustering")
            self.cluster_hdbscan()
            self.logger.info("Finished HDBSCAN clustering")

            self.logger.info("Starting HDBSCAN computing probabilities")
            self.count_probs()
            self.logger.info("Finished HDBSCAN computing probabilities")
            self.save_model_files()

            self.free_pool()

            self.logger.info("Sending reduced embeddings to GPU")
            self.reduced_to_cp()
            self.logger.info("Finished sending embeddings to GPU")

        self.done_state = "hdbscan"

        if self.start_from in {"umap", "hdbscan", "periphery"}:
            self.logger.info("Starting periphery attachment")
            self.attach_periphery()
            self.logger.info("Finished periphery attachment")
            self.save_model_files()

        self.done_state = "periphery"

        self.logger.info("Starting clusterization metrics")
        self.count_clusterization_metrics()
        self.logger.info("Finished clusterization metrics")

        self.timings["clusterization"] = str(int(time.perf_counter() - t0))

        with open(f"temp/{self.id}_metrics.json", "w") as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=4)

        with open(f"temp/{self.id}_timings.json", "w") as f:
            json.dump(self.timings, f, ensure_ascii=False, indent=4)

    def process_interpreting(self, clusters_df):
        t0 = time.perf_counter()

        self.logger.info("Starting LLM annotation")
        self.llm_annotation(clusters_df)
        self.logger.info("Finished LLM annotation")

        self.logger.info("Starting topic metrics")
        self.count_topic_metrics(clusters_df)
        self.logger.info("Finished topic metrics")

        self.timings["interpreting"] = str(int(time.perf_counter() - t0))

        with open(f"res/{self.id}_timings.json", "w") as f:
            json.dump(self.timings, f, ensure_ascii=False, indent=4)

    def update_config(self, config: ConfigItem):
        self.id = config.id
        self.done_state = "idle"
        self.tau = None
        self.delta = None

        self.metrics = {
            "clusterization_id": self.id,
        }
        self.timings = {
            "clusterization_id": self.id,
        }

        if self.umap_config != config.config.umap:
            self.start_from = "umap"
            self.umap_config = config.config.umap
            self.hdbscan_config = config.config.hdbscan
            self.periphery_config = config.config.periphery

            self.hdbscan_model = None
            self.reduced = None

            self.free_pool()
            return

        self.done_state = "umap"

        if self.hdbscan_config != config.config.hdbscan:
            self.start_from = "hdbscan"
            self.hdbscan_config = config.config.hdbscan
            self.periphery_config = config.config.periphery

            self.hdbscan_model = None

            self.free_pool()
            return

        self.done_state = "hdbscan"

        if self.periphery_config != config.config.periphery:
            self.start_from = "periphery"
            self.periphery_config = config.config.periphery
            return

        self.done_state = "periphery"
        self.start_from = "metrics"

    @staticmethod
    def free_pool():
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    def reduced_to_cp(self):
        t0 = time.perf_counter()
        try:
            self.reduced = cp.asarray(self.reduced, dtype=cp.float16)
        except Exception as e:
            raise ReducedToCPException from e
        self.timings["reduced_to_gpu"] = str(int(time.perf_counter() - t0))

    def reduce_umap(self):
        t0 = time.perf_counter()
        try:
            model = UmapModel(self.umap_config)
        except Exception as e:
            raise BuildUMAPException from e
        self.timings["build_umap"] = str(int(time.perf_counter() - t0))

        t0 = time.perf_counter()
        try:
            self.reduced = model.reduce(self.embeddings)
        except Exception as e:
            raise ReduceUMAPException from e
        self.timings["reduce_umap"] = str(int(time.perf_counter() - t0))

    def cluster_hdbscan(self):
        t0 = time.perf_counter()
        try:
            self.hdbscan_model = HdbscanModel(config=self.hdbscan_config, model=None)
        except Exception as e:
            raise BuildHdbscanException from e
        self.timings["build_hdbscan"] = str(int(time.perf_counter() - t0))

        t0 = time.perf_counter()
        try:
            self.hdbscan_model.cluster(self.reduced)
        except Exception as e:
            raise ClusterHdbscanException from e
        self.timings["cluster_hdbscan"] = str(int(time.perf_counter() - t0))

    def count_probs(self):
        t0 = time.perf_counter()
        try:
            self.hdbscan_model.generate_probs()
            self.hdbscan_model.calculate_cupy_init_probs()
        except Exception as e:
            raise ProbsHdbscanException from e
        self.timings["hdbscan_probs"] = str(int(time.perf_counter() - t0))

    def attach_periphery(self):
        t0 = time.perf_counter()
        try:
            perif_model = PeripheryModel(self.periphery_config)
            perif_model.calibrate_thresholds_from_core(self.hdbscan_model, self.reduced)
            perif_model.attach_noise_by_membership(self.hdbscan_model, self.reduced)
        except Exception as e:
            raise PeriferyException from e
        else:
            self.timings["periphery"] = str(int(time.perf_counter() - t0))
            self.tau = perif_model.tau
            self.delta = perif_model.delta
            self.metrics["tau"] = str(self.tau)
            self.metrics["delta"] = str(self.delta)

    def count_clusterization_metrics(self):
        t0 = time.perf_counter()
        self.metrics.update(simple_metrics(self.hdbscan_model))
        self.timings["simple_metrics"] = str(int(time.perf_counter() - t0))

        t0 = time.perf_counter()
        clusters_df = pd.DataFrame({
            "id": self.ids,
            "topic": cp.asnumpy(self.hdbscan_model.get_labels()),
            "is_core": cp.asnumpy(self.hdbscan_model.get_is_core()),
        })
        timings, metrics = get_interpretable_metrics(self.duckdb_path, clusters_df)
        self.metrics["interpretable_metrics"] = metrics
        self.timings["interpretable_metrics"] = str(int(time.perf_counter() - t0))
        self.timings["interpretable_metrics_detail"] = timings

    def llm_annotation(self, clusters_df: pd.DataFrame):
        timings = annotate_clusters(self.duckdb_path, clusters_df)
        self.timings["clusters_annotation"] = timings

    def count_topic_metrics(self, clusters_df: pd.DataFrame):
        timings = get_topic_metrics(self.duckdb_path, clusters_df)
        self.timings["topic_metrics"] = timings
