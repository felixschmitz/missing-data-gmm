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

# DATA_CATALOGS = 


__all__ = [
    "pd",
    "DATA",
    "SRC",
    "TEST_DIR",
    # "DATA_CATALOGS",
]
