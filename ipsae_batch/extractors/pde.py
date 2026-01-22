"""
Contact probability and PDE (Predicted Distance Error) matrix calculation.

Contact probability:
- AF3: Uses NATIVE contact_probs (more accurate for multi-interface detection)
- Other backends: Calculated from PAE and CB-CB distance

PDE:
- Boltz2: Uses native PDE
- Other backends: Estimated from PAE and distance

All functions work with FoldingResult from data_readers.
"""

from typing import Optional, Dict
import numpy as np

from ..data_readers import FoldingResult


def _compute_distance_matrix(result: FoldingResult) -> np.ndarray:
    """Compute CB-CB distance matrix from FoldingResult."""
    coords = result.cb_coordinates
    return np.sqrt(
        ((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2).sum(axis=2)
    )


def calculate_contact_probs(
    result: FoldingResult,
    distance_cutoff: float = 8.0,
    distance_steepness: float = 2.0,
    pae_cutoff: float = 10.0,
    pae_steepness: float = 2.0,
) -> np.ndarray:
    """
    CALCULATE contact probability matrix from PAE and CB-CB distance.

    This function ALWAYS calculates - it NEVER uses native backend values.
    This ensures UNIFORM figures across all backends.

    Formula: contact_prob = sigmoid((dist_cutoff - distance) / dist_steep)
                          * sigmoid((pae_cutoff - PAE) / pae_steep)

    This formula:
    - Weights distance and PAE roughly equally
    - Produces values that correctly separate multiple interfaces
    - Tested against AF3 native contact_probs (matches interface detection)

    Note: Use threshold=0.65 for interface detection (validated to match
    AF3 native at 0.5 for detecting multiple interfaces).

    Args:
        result: FoldingResult instance
        distance_cutoff: Distance midpoint for sigmoid (default 8.0Å)
        distance_steepness: Steepness of distance sigmoid (default 2.0)
        pae_cutoff: PAE midpoint for sigmoid (default 10.0)
        pae_steepness: Steepness of PAE sigmoid (default 2.0)

    Returns:
        Contact probability matrix (NxN, values 0-1)
    """
    pae = result.pae_matrix
    if pae is None:
        raise ValueError("PAE matrix required to calculate contact probabilities")

    # Compute CB-CB distance matrix
    distance_matrix = _compute_distance_matrix(result)

    # Sigmoid function
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    # Distance component: closer = higher probability
    # sigmoid((cutoff - distance) / steepness)
    # When distance < cutoff: positive argument -> prob > 0.5
    # When distance > cutoff: negative argument -> prob < 0.5
    dist_component = sigmoid((distance_cutoff - distance_matrix) / distance_steepness)

    # PAE component: lower PAE = higher confidence = higher probability
    # sigmoid((cutoff - PAE) / steepness)
    # When PAE < cutoff: positive argument -> prob > 0.5
    # When PAE > cutoff: negative argument -> prob < 0.5
    pae_component = sigmoid((pae_cutoff - pae) / pae_steepness)

    # Combined probability (both must be high for contact)
    contact_prob = dist_component * pae_component

    # Ensure symmetry (take max of both directions for PAE asymmetry)
    contact_prob = np.maximum(contact_prob, contact_prob.T)

    return contact_prob


def extract_contact_probs(result: FoldingResult) -> np.ndarray:
    """
    Get contact probability matrix - ALWAYS CALCULATED for uniform results.

    This function ALWAYS calculates contact probabilities from PAE and distance.
    This ensures UNIFORM figures across ALL backends (AF3, ColabFold, Boltz2, etc.).

    We do NOT use native backend values (even when available) because:
    1. Each backend calculates contact_probs differently
    2. We need consistent interface detection across all backends
    3. Our formula has been validated to correctly detect multiple interfaces

    Args:
        result: FoldingResult instance

    Returns:
        Contact probability matrix (NxN, values 0-1)
    """
    return calculate_contact_probs(result)


def extract_pde(result: FoldingResult) -> Optional[np.ndarray]:
    """
    Extract native PDE matrix from FoldingResult (only Boltz2 has this).

    Args:
        result: FoldingResult instance

    Returns:
        PDE matrix (NxN) or None if not available
    """
    return result.pde_matrix


def pde_to_contact_probability(
    pde: np.ndarray,
    steepness: float = 0.3,
    midpoint: float = 3.0,
) -> np.ndarray:
    """
    Convert PDE values to contact probabilities.

    Lower PDE values indicate more confident distance predictions,
    which can be interpreted as higher contact probability.

    Args:
        pde: PDE matrix
        steepness: Steepness of sigmoid transition
        midpoint: PDE value at 50% probability

    Returns:
        Contact probability matrix (0-1)
    """
    return 1.0 / (1.0 + np.exp(steepness * (pde - midpoint)))


def compute_pde_statistics(pde: np.ndarray) -> Dict:
    """
    Compute statistics for a PDE matrix.

    Args:
        pde: PDE matrix

    Returns:
        Dictionary with statistics
    """
    flat = pde.flatten()

    return {
        "mean": float(np.mean(flat)),
        "median": float(np.median(flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "std": float(np.std(flat)),
    }


def estimate_pde_from_pae(
    pae: np.ndarray,
    distance_matrix: np.ndarray,
    distance_scale: float = 8.0,
) -> np.ndarray:
    """
    Estimate PDE-like matrix from PAE and distance information.

    This is a rough approximation for backends that don't provide PDE.

    Args:
        pae: PAE matrix
        distance_matrix: CB-CB distance matrix
        distance_scale: Scale factor for distance contribution

    Returns:
        Estimated PDE matrix
    """
    distance_factor = 1 + np.log1p(distance_matrix / distance_scale)
    return pae * distance_factor


def extract_or_estimate_pde(
    result: FoldingResult,
    distance_scale: float = 8.0,
) -> np.ndarray:
    """
    Extract native PDE or estimate it from PAE and distances.

    For Boltz2, returns native PDE. For other backends,
    estimates PDE from PAE and CB-CB distances.

    Args:
        result: FoldingResult instance
        distance_scale: Scale factor for distance contribution in estimation

    Returns:
        PDE matrix (native or estimated)
    """
    # Return native PDE if available (Boltz2)
    if result.pde_matrix is not None:
        return result.pde_matrix

    # Compute CB-CB distance matrix and estimate PDE from PAE
    distance_matrix = _compute_distance_matrix(result)
    return estimate_pde_from_pae(result.pae_matrix, distance_matrix, distance_scale)


def extract_both_pde(
    result: FoldingResult,
    distance_scale: float = 8.0,
) -> Dict:
    """
    Extract both PDE and contact probability matrices.

    IMPORTANT: contact_probs is ALWAYS CALCULATED from PAE and distance.
    PDE is native for Boltz2, estimated for other backends.

    Args:
        result: FoldingResult instance
        distance_scale: Scale factor for PDE estimation

    Returns:
        Dict with 'pde' and 'contact_probs' matrices
    """
    return {
        'pde': extract_or_estimate_pde(result, distance_scale),
        'contact_probs': extract_contact_probs(result),  # Always calculated
    }
