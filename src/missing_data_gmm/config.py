"""All the general configuration of the project."""

from pathlib import Path
from typing import NamedTuple

import pandas as pd
from pytask import DataCatalog

pd.set_option("mode.copy_on_write", True)
pd.set_option("future.infer_string", True)
pd.set_option("future.no_silent_downcasting", True)
pd.set_option("plotting.backend", "plotly")

SRC = Path(__file__).parent.resolve()
ROOT = SRC.joinpath("..", "..").resolve()
BLD = SRC.joinpath("..", "..", "bld").resolve()
DATA = BLD.joinpath("data").resolve()
DOCUMENTS = ROOT.joinpath("documents").resolve()

TEST_DIR = SRC.joinpath("..", "..", "tests").resolve()

# 0 is the specification of the final paper, 1-8 are from the appendix
MC_DESIGNS = list(range(9))

METHODS = ["Complete case method", "Dummy case method", "Dagenais (FGLS)", "GMM"]

DATA_CATALOGS = {
    "regression": {
        "missreg": DataCatalog(name="missreg"),
        "raw": DataCatalog(name="raw"),
    },
    "simulation": DataCatalog(name="MC"),
}


class Dataset(NamedTuple):
    name: str

    @property
    def path(self) -> Path:
        return DATA / "missreg" / f"{self.name}.dta"

    @property
    def independent_variables(self) -> list[str]:
        if self.name == "wls-data":
            return ["bmirating", "iq", "bmimissing", "constant"]
        return ["bmirating", "iq", "educ", "bmimissing", "constant"]

    @property
    def dependent_variable(self) -> str:
        if self.name == "wls-data":
            return "educ"
        return "adultbmi"


class Sex(NamedTuple):
    name: str


DATASETS = [Dataset("wls-data"), Dataset("wls-data-adultbmi")]
SEXES = [Sex("male"), Sex("not-male")]


class Experiment(NamedTuple):
    dataset: Dataset
    sex: Sex
    methods: tuple[str]

    @property
    def name(self) -> str:
        return f"{self.sex.name}-{self.dataset.name}"

    @property
    def complete_model_name(self) -> str:
        return f"complete-{self.name}"

    @property
    def dummy_model_name(self) -> str:
        return f"dummy-{self.name}"

    @property
    def gmm_model_name(self) -> str:
        return f"gmm-{self.name}"


EXPERIMENTS = [
    Experiment(dataset, sex, ("complete", "dummy", "gmm"))
    for dataset in DATASETS
    for sex in SEXES
]


__all__ = [
    "BLD",
    "DATA",
    "DATA_CATALOGS",
    "DOCUMENTS",
    "EXPERIMENTS",
    "MC_DESIGNS",
    "pd",
    "ROOT",
    "SRC",
    "TEST_DIR",
]
