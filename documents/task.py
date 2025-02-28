"""Tasks for compiling the paper and presentation(s)."""

import shutil
from pathlib import Path

import pytask
from pytask_latex import compilation_steps as cs

from missing_data_gmm.config import BLD, DOCUMENTS, ROOT

figures = [
    "simulation_results_gamma20_heteroskedastic_regression.png",
    "simulation_results_gamma20_heteroskedastic_imputation.png",
    "simulation_results_gamma20_homoskedastic.png",
]

input_files = [
    "introduction.tex",
    "literature_review.tex",
    "methodology.tex",
    "mc_simulations.tex",
    "empirical_application.tex",
    "conclusion.tex",
    "appendix.tex",
]

tables = [
    "simulation_results_design1.tex",
    "simulation_results_design2.tex",
    "simulation_results_design3.tex",
]

DOCUMENTS_KWARGS = {
    "paper": {
        "depends_on": [BLD / "figures" / figure for figure in figures]
        + [DOCUMENTS / "resources" / input_file for input_file in input_files]
        + [BLD / "tables" / table for table in tables]
    },
    "presentation": {"depends_on": None},
}

for document, kwargs in DOCUMENTS_KWARGS.items():

    @pytask.mark.latex(
        script=DOCUMENTS / f"{document}.tex",
        document=BLD / "documents" / f"{document}.pdf",
        compilation_steps=cs.latexmk(
            options=("--pdf", "--interaction=nonstopmode", "--synctex=1", "--cd"),
        ),
    )
    @pytask.task(id=document, kwargs=kwargs)
    def task_compile_document(depends_on: None | list[Path]) -> None:
        """Compile the document specified in the latex decorator."""

    copy_to_root_kwargs = {
        "depends_on": BLD / "documents" / f"{document}.pdf",
        "produces": ROOT / f"{document}.pdf",
    }

    @pytask.task(id=document, kwargs=copy_to_root_kwargs)
    def task_copy_to_root(depends_on: Path, produces: Path) -> None:
        """Copy a document to the root directory for easier retrieval."""
        shutil.copy(depends_on, produces)
