from cuml.manifold import UMAP
from Config import UmapConfig
import numpy as np


class UmapModel:
    def __init__(self, config: UmapConfig):
        self.model = UMAP(**config.model_dump())

    def reduce(self, embeddings: np.ndarray) -> np.ndarray:
        reduced = self.model.fit_transform(embeddings)
        return reduced
