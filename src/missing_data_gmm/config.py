"""All the general configuration of the project."""

from pathlib import Path

import pandas as pd
from pytask import DataCatalog

pd.set_option("mode.copy_on_write", True)
pd.set_option("future.infer_string", True)
pd.set_option("future.no_silent_downcasting", True)
pd.set_option("plotting.backend", "plotly")

SRC = Path(__file__).parent.resolve()
DATA = SRC.joinpath("..", "..", "data").resolve()

TEST_DIR = SRC.joinpath("..", "..", "tests").resolve()

SEX_NAMES = ("male", "not-male")
DATA_NAMES = ("wls-data", "wls-data-adultbmi")

DATA_CATALOGS = {
    "regression": {
        data_name: {
            sex_name: DataCatalog(name=f"{data_name}-{sex_name}")
            for sex_name in SEX_NAMES
        }
        for data_name in DATA_NAMES
    },
    "simulation": DataCatalog(name="MC"),
}


__all__ = [
    "pd",
    "DATA",
    "SRC",
    "TEST_DIR",
    "SEX_NAMES",
    "DATA_NAMES",
    "DATA_CATALOGS",
]
