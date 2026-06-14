from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, computed_field, model_validator, RootModel


class JsonConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_json_file(cls, path: str | Path):
        path = Path(path)
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


class UmapConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_neighbors: int = 50
    min_dist: float = 0.1

    @computed_field
    @property
    def n_components(self) -> int:
        return 16

    @computed_field
    @property
    def metric(self) -> str:
        return "cosine"

    @computed_field
    @property
    def init(self) -> str:
        return "spectral"


class HdbscanConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_cluster_size: int = 100
    min_samples: int = 25
    cluster_selection_method: Literal["eom", "leaf"] = "eom"

    @computed_field
    @property
    def prediction_data(self) -> bool:
        return True

    @computed_field
    @property
    def build_algo(self) -> str:
        return "brute_force"


class PeripheryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    core_sample_frac: float = 0.2
    core_sample_min: int = 0
    core_sample_max: int = 999_999_999
    core_quantile: float = 0.05
    mv_batch_size: int = 2048
    batch_size: int = 50_000

    @model_validator(mode="after")
    def validate_params(self) -> "PeripheryConfig":
        if self.core_sample_min > self.core_sample_max:
            raise ValueError("core_sample_min должен быть не больше core_sample_max")

        if self.mv_batch_size > self.batch_size:
            raise ValueError("mv_batch_size должен быть не больше batch_size")

        return self


class AppConfig(BaseModel):
    umap: UmapConfig
    hdbscan: HdbscanConfig
    periphery: PeripheryConfig


class ConfigItem(JsonConfigModel):
    id: str
    config: AppConfig
