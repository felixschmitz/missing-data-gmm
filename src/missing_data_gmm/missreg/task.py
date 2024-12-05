"""Task related to missing data regression."""

from pathlib import Path
from typing import Annotated

from missing_data_gmm.config import DATA


def task_do_something(
    raw_data: Annotated[Path, DATA / "missreg" / "wls-data.dta"],
) -> None:
    """Real world data example for missing data regression.

    Args:
        raw_data: Path to the raw data file.
    """
