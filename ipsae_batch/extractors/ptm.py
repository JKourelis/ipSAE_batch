"""
PTM (predicted TM-score) calculator from PAE matrix.

Recalculates pTM and ipTM scores directly from the PAE matrix,
providing our own calculation independent of backend-reported values.

The calculation is DIRECTIONAL for ipTM - each direction (A→B vs B→A)
is calculated separately.
"""

import numpy as np
from typing import Dict, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from data_readers import FoldingResult


def calculate_d0(length: int) -> float:
    """
    Calculate d0 parameter for TM-score normalization.

    Uses the Yang & Skolnick formula.

    Args:
        length: Number of residues

    Returns:
        d0 normalization value
    """
    if length <= 21:
        return 0.5
    else:
        return 1.24 * ((length - 15) ** (1.0 / 3.0)) - 1.8


def ptm_term(pae_value: float, d0: float) -> float:
    """
    Calculate single PTM term from PAE value.

    Args:
        pae_value: PAE (predicted aligned error) value
        d0: Normalization distance

    Returns:
        PTM term (0-1)
    """
    return 1.0 / (1.0 + (pae_value / d0) ** 2)


def calculate_ptm_from_pae(
    pae_matrix: np.ndarray,
    d0: Optional[float] = None
) -> float:
    """
    Calculate global pTM from PAE matrix.

    Takes the maximum TM-score over all possible alignment centers.

    Args:
        pae_matrix: NxN PAE matrix
        d0: Normalization distance (calculated from matrix size if None)

    Returns:
        pTM score (0-1)
    """
    n_res = pae_matrix.shape[0]
    if n_res == 0:
        return 0.0

    if d0 is None:
        d0 = calculate_d0(n_res)

    max_tm_score = 0.0

    for i in range(n_res):
        tm_terms = []
        for j in range(n_res):
            expected_distance = pae_matrix[i, j]
            tm_term_val = ptm_term(expected_distance, d0)
            tm_terms.append(tm_term_val)

        tm_score = np.mean(tm_terms)
        max_tm_score = max(max_tm_score, tm_score)

    return max_tm_score


def calculate_iptm_from_pae(
    pae_matrix: np.ndarray,
    chains: np.ndarray,
    d0: Optional[float] = None
) -> float:
    """
    Calculate global ipTM from PAE matrix for multi-chain complexes.

    Only considers inter-chain residue pairs.

    Args:
        pae_matrix: NxN PAE matrix
        chains: Length-N array of chain IDs per residue
        d0: Normalization distance (calculated from matrix size if None)

    Returns:
        ipTM score (0-1)
    """
    n_res = pae_matrix.shape[0]
    unique_chains = np.unique(chains)

    if len(unique_chains) < 2:
        return 0.0

    if d0 is None:
        d0 = calculate_d0(n_res)

    # Create inter-chain mask
    inter_chain_mask = chains[:, None] != chains[None, :]

    max_iptm_score = 0.0

    for i in range(n_res):
        tm_terms = []
        for j in range(n_res):
            if inter_chain_mask[i, j]:
                expected_distance = pae_matrix[i, j]
                tm_term_val = ptm_term(expected_distance, d0)
                tm_terms.append(tm_term_val)

        if tm_terms:
            tm_score = np.mean(tm_terms)
            max_iptm_score = max(max_iptm_score, tm_score)

    return max_iptm_score


def calculate_iptm_directional(
    pae_matrix: np.ndarray,
    chains: np.ndarray,
    chain1: str,
    chain2: str,
    d0: Optional[float] = None
) -> float:
    """
    Calculate DIRECTIONAL ipTM from chain1 to chain2.

    This is the asymmetric ipTM where we only consider alignment
    centers in chain1 and score against chain2.

    Args:
        pae_matrix: NxN PAE matrix
        chains: Length-N array of chain IDs per residue
        chain1: Source chain ID (alignment center)
        chain2: Target chain ID (scored residues)
        d0: Normalization distance (calculated from matrix size if None)

    Returns:
        Directional ipTM score (0-1)
    """
    n_res = pae_matrix.shape[0]

    if d0 is None:
        d0 = calculate_d0(n_res)

    max_iptm_score = 0.0

    for i in range(n_res):
        if chains[i] != chain1:
            continue

        tm_terms = []
        for j in range(n_res):
            if chains[j] == chain2:
                expected_distance = pae_matrix[i, j]
                tm_term_val = ptm_term(expected_distance, d0)
                tm_terms.append(tm_term_val)

        if tm_terms:
            tm_score = np.mean(tm_terms)
            max_iptm_score = max(max_iptm_score, tm_score)

    return max_iptm_score


def calculate_ptm_per_chain(
    pae_matrix: np.ndarray,
    chains: np.ndarray
) -> Dict[str, float]:
    """
    Calculate per-chain pTM scores with chain-specific d0.

    Args:
        pae_matrix: NxN PAE matrix
        chains: Length-N array of chain IDs per residue

    Returns:
        Dict mapping chain_id -> pTM score
    """
    unique_chains = np.unique(chains)
    n_res = pae_matrix.shape[0]
    results = {}

    for chain_id in unique_chains:
        chain_indices = np.where(chains == chain_id)[0]
        chain_length = len(chain_indices)

        if chain_length <= 1:
            results[chain_id] = 0.0
            continue

        # Calculate d0 based on chain length
        chain_d0 = calculate_d0(chain_length)

        # Create weight mask (1 for chain residues, 0 otherwise)
        residue_weights = np.zeros(n_res)
        residue_weights[chain_indices] = 1.0

        max_tm_score = 0.0

        for i in range(n_res):
            if residue_weights[i] == 0:
                continue

            tm_terms = []
            weights = []

            for j in range(n_res):
                if residue_weights[j] > 0:
                    expected_distance = pae_matrix[i, j]
                    tm_term_val = ptm_term(expected_distance, chain_d0)
                    tm_terms.append(tm_term_val)
                    weights.append(residue_weights[j])

            if tm_terms:
                tm_score = np.average(tm_terms, weights=weights)
                max_tm_score = max(max_tm_score, tm_score)

        results[chain_id] = max_tm_score

    return results


def calculate_all_ptm_scores(result: 'FoldingResult') -> Dict[str, any]:
    """
    Calculate all pTM/ipTM scores from a FoldingResult.

    Returns both global and per-chain scores, as well as
    directional ipTM for each chain pair.

    Args:
        result: FoldingResult instance

    Returns:
        Dictionary with:
            'pTM_calc': Global pTM score
            'ipTM_calc': Global ipTM score
            'pTM_per_chain': Dict of per-chain pTM scores
            'ipTM_directional': Dict of (chain1, chain2) -> directional ipTM
    """
    pae_matrix = result.pae_matrix
    chains = result.chains
    unique_chains = result.unique_chains

    n_res = pae_matrix.shape[0]
    global_d0 = calculate_d0(n_res)

    results = {
        'pTM_calc': calculate_ptm_from_pae(pae_matrix, global_d0),
        'ipTM_calc': calculate_iptm_from_pae(pae_matrix, chains, global_d0),
        'pTM_per_chain': calculate_ptm_per_chain(pae_matrix, chains),
        'ipTM_directional': {}
    }

    # Calculate directional ipTM for each chain pair
    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1 == chain2:
                continue
            iptm_dir = calculate_iptm_directional(
                pae_matrix, chains, chain1, chain2, global_d0
            )
            results['ipTM_directional'][(chain1, chain2)] = iptm_dir

    return results


def extract_ptm_scores(result: 'FoldingResult') -> Tuple[float, float, Dict[str, float]]:
    """
    Convenience function to extract main PTM scores.

    Args:
        result: FoldingResult instance

    Returns:
        Tuple of (pTM_calc, ipTM_calc, ipTM_directional_dict)
    """
    scores = calculate_all_ptm_scores(result)
    return (
        scores['pTM_calc'],
        scores['ipTM_calc'],
        scores['ipTM_directional']
    )
