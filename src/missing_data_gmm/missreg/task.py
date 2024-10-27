from pathlib import Path
from typing import Annotated

from missing_data_gmm.config import DATA, pd


def task_do_something(raw_data: Annotated[Path, DATA / "wls-data.dta"]) -> None:
    df = pd.read_stata(DATA / "wls-data.dta")
