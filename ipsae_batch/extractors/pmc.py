"""
PMC (Predicted Merged Confidence) matrix computation.

PMC combines PAE and pLDDT into a single confidence matrix,
as used in AlphaBridge.

Formula: PMC = symmetric_pae + plddt_penalty_matrix / 3
where plddt_penalty_matrix[i,j] = -((plddt[i] + plddt[j])/2 - 100)
for |i-j| > 2, else 0

All functions work with FoldingResult from data_readers.
"""

from typing import Optional
import numpy as np

from ..data_readers import FoldingResult
from .pae import symmetrize_pae


def compute_plddt_penalty_matrix(plddt: np.ndarray, diagonal_offset: int = 2) -> np.ndarray:
    """
    Compute pLDDT penalty matrix.

    The penalty is based on average pLDDT of residue pairs,
    with diagonal and near-diagonal elements set to zero.

    Args:
        plddt: Array of pLDDT values (0-100)
        diagonal_offset: Number of positions from diagonal to set to zero

    Returns:
        pLDDT penalty matrix
    """
    n = len(plddt)

    # Vectorized computation
    plddt_i = plddt[:, np.newaxis]
    plddt_j = plddt[np.newaxis, :]
    penalty = -((plddt_i + plddt_j) / 2 - 100)

    # Zero out diagonal and near-diagonal
    for offset in range(-diagonal_offset, diagonal_offset + 1):
        np.fill_diagonal(penalty[max(0, -offset):, max(0, offset):], 0)

    return penalty


def extract_pmc(
    result: FoldingResult,
    pae_weight: float = 1.0,
    plddt_weight: float = 1/3,
    max_value: float = 32.0,
    symmetrize: bool = True,
) -> np.ndarray:
    """
    Compute PMC (Predicted Merged Confidence) matrix.

    PMC combines PAE and pLDDT information into a single confidence metric.
    Lower values indicate higher confidence.

    Args:
        result: FoldingResult instance
        pae_weight: Weight for PAE component
        plddt_weight: Weight for pLDDT penalty component
        max_value: Maximum value to clip PMC at
        symmetrize: Whether to symmetrize the PAE matrix first

    Returns:
        PMC matrix
    """
    pae = result.pae_matrix
    plddt = result.plddt

    # Symmetrize PAE
    if symmetrize:
        symmetric_pae = symmetrize_pae(pae, method="max")
    else:
        symmetric_pae = pae

    # Compute pLDDT penalty
    plddt_penalty = compute_plddt_penalty_matrix(plddt)

    # Combine into PMC
    pmc = pae_weight * symmetric_pae + plddt_weight * plddt_penalty

    # Clip to max value
    pmc = np.clip(pmc, 0, max_value)

    return pmc


def pmc_to_contact_matrix(pmc: np.ndarray, threshold: float = 16.0) -> np.ndarray:
    """
    Convert PMC matrix to binary contact matrix.

    Args:
        pmc: PMC matrix
        threshold: PMC threshold for contact (lower = contact)

    Returns:
        Binary contact matrix
    """
    return (pmc < threshold).astype(float)
