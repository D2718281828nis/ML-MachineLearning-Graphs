#!/usr/bin/env python3
"""Inventory notebooks across the repository and enforce the root policy."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


CLASSIFICATION = {
    "07_vitiligo_image_clustering_exploratory.ipynb": "Clustering / medical imaging",
    "05_genomic_population_gcn_exploratory.ipynb": "Graph neural networks / genomics",
    "Graph from DataFrame tSNE.ipynb": "Graph construction / dimensionality reduction",
    "Gromov-Waserstein graph clastering.ipynb": "Gromov-Wasserstein / clustering",
    "Gromov-Wasserstein distance Transport Task.ipynb": "Gromov-Wasserstein / optimal transport",
    "06_hyperbolic_tp53_sequences_exploratory.ipynb": "Hyperbolic learning / genomics",
    "Hyperbolic Graph Neural Network.ipynb": "Hyperbolic learning / GNN",
    "LLM Graph triplets visualisation.ipynb": "Knowledge graphs / LLM visualization",
    "ML task table PCA ICA tSNE DBSCAN AggClustering SVM.ipynb": "Classical ML / method survey",
    "Poincare ball for Graph.ipynb": "Hyperbolic learning / visualization",
    "example Gated Graph Attention Network.ipynb": "Graph neural networks / attention",
    "example_Interactive_Graph_Visualisation.ipynb": "Graph visualization / interactive",
    "erd_to_mermaid.ipynb": "Utilities / ERD conversion",
    "erd_graph_visualisation.ipynb": "Utilities / ERD visualization",
    "erd_interactive_graph_visualisation.ipynb": "Utilities / ERD interactive visualization",
    "01_patient_similarity_mi.ipynb": "Healthcare / patient similarity",
    "02_ptbxl_ecg_graph.ipynb": "Healthcare / ECG",
    "03_mimic_ehr_graph.ipynb": "Healthcare / EHR",
    "04_brain_hypergraph_optional.ipynb": "Healthcare / brain hypergraph",
    "ai_medical_devices_graph_visualizations.ipynb": "Examples / medical devices",
    "graph_learning_basics.ipynb": "Examples / graph basics",
    "hyperbolic_graph_basics.ipynb": "Examples / hyperbolic basics",
    "graph_connectivity_chains_circles.ipynb": "Examples / graph connectivity vs. local methods",
}


@dataclass(frozen=True)
class NotebookAudit:
    file: str
    category: str
    size_mb: float
    cells: int
    code_cells: int
    markdown_cells: int
    executed_code_cells: int
    stored_outputs: int
    stored_errors: list[str]
    has_colab_badge: bool
    uses_colab_features: bool
    setup_commands: int
    external_urls: int


def audit_notebook(path: Path) -> NotebookAudit:
    """Return reproducibility signals without executing notebook code."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])
    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    markdown_cells = [cell for cell in cells if cell.get("cell_type") == "markdown"]
    source = "\n".join("".join(cell.get("source", [])) for cell in cells)
    errors = [
        f"{output.get('ename', 'Error')}: {output.get('evalue', '')}".strip()
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    first_cell = "".join(cells[0].get("source", [])) if cells else ""
    setup_pattern = re.compile(r"(?m)^\s*(?:!|%)(?:pip|apt(?:-get)?|wget|git)\b")

    return NotebookAudit(
        file=path.name,
        category=CLASSIFICATION.get(path.name, "Unclassified"),
        size_mb=round(path.stat().st_size / 1_000_000, 2),
        cells=len(cells),
        code_cells=len(code_cells),
        markdown_cells=len(markdown_cells),
        executed_code_cells=sum(
            cell.get("execution_count") is not None for cell in code_cells
        ),
        stored_outputs=sum(len(cell.get("outputs", [])) for cell in code_cells),
        stored_errors=errors,
        has_colab_badge="colab.research.google.com" in first_cell,
        uses_colab_features="google.colab" in source or "/content/" in source,
        setup_commands=len(setup_pattern.findall(source)),
        external_urls=len(set(re.findall(r"https?://[^\s\)\]\"'<>]+", source))),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [path for path in args.root.rglob("*.ipynb") if ".git" not in path.parts]
    audits = [audit_notebook(path) for path in sorted(paths)]
    if args.json:
        print(json.dumps([asdict(audit) for audit in audits], ensure_ascii=False, indent=2))
        return

    print("file\tcategory\tcells\texecuted\toutputs\terrors\tsize_mb")
    for audit in audits:
        print(
            f"{audit.file}\t{audit.category}\t{audit.cells}\t"
            f"{audit.executed_code_cells}/{audit.code_cells}\t"
            f"{audit.stored_outputs}\t{len(audit.stored_errors)}\t{audit.size_mb:.2f}"
        )
    root_notebooks = [path for path in paths if path.parent == args.root]
    print(f"\nNotebooks: {len(audits)}; root notebooks: {len(root_notebooks)}")
    unclassified = [audit.file for audit in audits if audit.category == "Unclassified"]
    if unclassified:
        raise SystemExit(f"Unclassified notebooks: {', '.join(unclassified)}")
    invalid_root = [
        path.name
        for path in root_notebooks
        if not path.name.lower().startswith(("demo", "example"))
    ]
    if invalid_root:
        raise SystemExit(
            "Only demo/example notebooks may remain in root: "
            + ", ".join(invalid_root)
        )


if __name__ == "__main__":
    main()
