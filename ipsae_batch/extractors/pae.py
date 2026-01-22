"""
PAE (Predicted Aligned Error) matrix extraction and utilities.

Provides functions for extracting, processing, and transforming PAE matrices.
All functions work with FoldingResult from data_readers.
"""

from typing import Optional, Dict
import numpy as np

from ..data_readers import FoldingResult


def extract_pae(result: FoldingResult) -> np.ndarray:
    """
    Extract PAE matrix from FoldingResult.

    Args:
        result: FoldingResult instance

    Returns:
        PAE matrix (NxN)
    """
    return result.pae_matrix


def symmetrize_pae(pae: np.ndarray, method: str = "max") -> np.ndarray:
    """
    Symmetrize a PAE matrix.

    Args:
        pae: PAE matrix (NxN)
        method: Symmetrization method:
            - "max": Take maximum of (i,j) and (j,i) at each position
            - "min": Take minimum of (i,j) and (j,i)
            - "mean": Take average of (i,j) and (j,i)

    Returns:
        Symmetrized PAE matrix
    """
    if method == "max":
        return np.maximum(pae, pae.T)
    elif method == "min":
        return np.minimum(pae, pae.T)
    elif method == "mean":
        return (pae + pae.T) / 2
    else:
        raise ValueError(f"Unknown symmetrization method: {method}")


def get_chain_pair_pae(
    result: FoldingResult,
    chain1: str,
    chain2: str,
) -> np.ndarray:
    """
    Extract PAE submatrix for a specific chain pair.

    Args:
        result: FoldingResult instance
        chain1: First chain ID
        chain2: Second chain ID

    Returns:
        PAE submatrix (M x N where M=len(chain1), N=len(chain2))
    """
    idx1 = result.get_chain_indices(chain1)
    idx2 = result.get_chain_indices(chain2)
    return result.pae_matrix[np.ix_(idx1, idx2)]


def compute_pae_statistics(pae: np.ndarray) -> Dict:
    """
    Compute statistics for a PAE matrix.

    Args:
        pae: PAE matrix

    Returns:
        Dictionary with statistics:
            - mean: Mean PAE value
            - median: Median PAE value
            - min: Minimum PAE value
            - max: Maximum PAE value
            - std: Standard deviation
            - fraction_below_threshold: Dict of thresholds to fractions
    """
    flat = pae.flatten()

    thresholds = [5, 10, 15, 20]
    fractions = {t: float(np.mean(flat < t)) for t in thresholds}

    return {
        "mean": float(np.mean(flat)),
        "median": float(np.median(flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "std": float(np.std(flat)),
        "fraction_below_threshold": fractions,
    }


def apply_pae_threshold(
    pae: np.ndarray,
    threshold: float,
    fill_value: float = np.inf,
) -> np.ndarray:
    """
    Apply threshold to PAE matrix, setting values above threshold to fill_value.

    Args:
        pae: PAE matrix
        threshold: PAE cutoff value
        fill_value: Value to use for PAE >= threshold

    Returns:
        Thresholded PAE matrix
    """
    result = pae.copy()
    result[pae >= threshold] = fill_value
    return result


def pae_to_contact_probability(
    pae: np.ndarray,
    steepness: float = 0.5,
    midpoint: float = 5.0,
) -> np.ndarray:
    """
    Convert PAE values to contact probabilities using sigmoid function.

    Lower PAE = higher contact probability.

    Args:
        pae: PAE matrix
        steepness: Steepness of sigmoid transition
        midpoint: PAE value at 50% probability

    Returns:
        Contact probability matrix (0-1)
    """
    return 1.0 / (1.0 + np.exp(steepness * (pae - midpoint)))
