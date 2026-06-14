from cuml.cluster import HDBSCAN
from cuml.cluster.hdbscan import membership_vector
from Config import HdbscanConfig
import numpy as np
import cupy as cp


class HdbscanModel:
    def __init__(self, config: HdbscanConfig | None, model: HDBSCAN | None = None):
        if config is not None:
            self.model = HDBSCAN(**config.model_dump())
            self.labels: cp.array | None = None
            self.is_core: cp.array | None = None
            self.probs: cp.array | None = None
        elif model is not None:
            self.model = model
            self.labels = self.get_labels()
            self.is_core = self.get_is_core()
            self.probs = self.get_probs()

    def get_labels(self):
        if self.labels is None:
            self.labels = cp.asarray(getattr(self.model, "labels_", None))
        return self.labels

    def get_is_core(self):
        if self.is_core is None:
            labels = self.get_labels()
            self.is_core = (labels != -1)
        return self.is_core

    def get_probs(self):
        if self.probs is None:
            self.calculate_cupy_init_probs()
        return self.probs

    def cluster(self, embeddings):
        self.model.fit(embeddings)

    def generate_probs(self):
        self.model.generate_prediction_data()

    def calculate_cupy_init_probs(self):
        probs = cp.asarray(self.model.probabilities_).astype(cp.float16, copy=False)
        init_probs = cp.zeros(self.get_labels().shape[0], dtype=cp.float16)
        init_probs[self.get_is_core()] = probs[self.get_is_core()]
        self.probs = init_probs

    def get_cupy_membership_probs(self, cp_array: cp.array, batch_size: int) -> cp.array:
        return cp.asarray(membership_vector(self.model, cp_array, batch_size=batch_size))
