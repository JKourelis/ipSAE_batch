"""
ipsae_batch - Batch analysis of protein structure predictions.

This package provides tools for analyzing protein structure prediction outputs
from multiple backends (AlphaFold3, ColabFold, Boltz2, IntelliFold).

Main modules:
- data_readers: Backend-specific file parsers
- extractors: Data extraction (PAE, PMC, contacts, PTM)
- core: Scoring algorithms (ipSAE, ipTM, etc.)
- graphics: Visualization (matrix plots, ribbon diagrams)
- output_writers: CSV and report output

Command-line interface:
    python -m ipsae_batch <input_folder> [options]
    ipsae-batch <input_folder> [options]  # If installed via pip
"""

__version__ = "1.0.0"
__author__ = "Jiorgos Kourelis"

# Expose main classes and functions at package level
from .data_readers import get_reader, list_backends, FoldingResult
from .core import calculate_scores_from_result
from .extractors import (
    extract_pae,
    extract_pmc,
    extract_contact_probs,
    get_geometric_interfaces,
    cluster_domains_from_result,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Data readers
    "get_reader",
    "list_backends",
    "FoldingResult",
    # Core scoring
    "calculate_scores_from_result",
    # Extractors
    "extract_pae",
    "extract_pmc",
    "extract_contact_probs",
    "get_geometric_interfaces",
    "cluster_domains_from_result",
]
