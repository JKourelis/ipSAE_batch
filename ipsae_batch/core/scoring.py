"""
Core scoring functions for ipSAE calculations.

Based on ipSAE by Roland Dunbrack, Fox Chase Cancer Center
https://www.biorxiv.org/content/10.1101/2025.02.10.637595v1

This module provides the core scoring algorithms for calculating:
- ipSAE (interface predicted Structural Alignment Error)
- ipTM (interface predicted TM-score)
- pDockQ and pDockQ2 (predicted DockQ scores)
- LIS (Local Interaction Score)
- Per-contact scores for visualization

PER-CONTACT SCORES:
-------------------
For each contact (residue_i, residue_j), the following scores are available:
- ipSAE: PTM-based score using d0 normalized to number of valid pairs for residue_i
- ipSAE_d0chn: PTM-based score using d0 normalized to chain pair length
- ipTM: PTM-based score for TM-score calculation
- pae: Raw PAE value for the contact
- distance: CB-CB distance in Angstroms
- plddt_i, plddt_j: pLDDT values for each residue

The standard metric for contact line coloring is ipSAE (configurable).
"""

import math
import numpy as np
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from data_readers import FoldingResult

# Import extractors for additional calculations
from ..extractors.ptm import calculate_ptm_from_pae, calculate_iptm_directional
from ..extractors.contacts import find_interface_contacts


# =============================================================================
# Core scoring primitives
# =============================================================================

def ptm_func(x: np.ndarray, d0: float) -> np.ndarray:
    """
    Calculate PTM (predicted TM-score) score.

    Args:
        x: PAE values
        d0: Normalization distance

    Returns:
        PTM scores
    """
    return 1.0 / (1 + (x / d0) ** 2.0)


ptm_func_vec = np.vectorize(ptm_func)


def calc_d0(L: int, pair_type: str) -> float:
    """
    Calculate d0 for TM-score normalization (Yang & Skolnick).

    Args:
        L: Length for normalization (typically number of residues)
        pair_type: 'protein' or 'nucleic_acid'

    Returns:
        d0 normalization value
    """
    L = float(max(27, L))
    min_value = 2.0 if pair_type == 'nucleic_acid' else 1.0
    d0 = 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8
    return max(min_value, d0)


def calc_d0_array(L: np.ndarray, pair_type: str) -> np.ndarray:
    """
    Calculate d0 for array of lengths.

    Args:
        L: Array of lengths
        pair_type: 'protein' or 'nucleic_acid'

    Returns:
        Array of d0 normalization values
    """
    L = np.maximum(27, np.array(L, dtype=float))
    min_value = 2.0 if pair_type == 'nucleic_acid' else 1.0
    return np.maximum(min_value, 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8)


# =============================================================================
# Per-contact scoring
# =============================================================================

def calculate_per_contact_scores(
    result: 'FoldingResult',
    pae_cutoff: float = 10.0,
    dist_cutoff: float = 10.0,
) -> List[Dict]:
    """
    Calculate scores for each individual contact.

    This returns a list of dicts, one per inter-chain contact, with various
    metrics that can be used for visualization (e.g., coloring contact lines).

    Args:
        result: FoldingResult object
        pae_cutoff: PAE cutoff for valid contacts
        dist_cutoff: Distance cutoff for contacts (Angstroms)

    Returns:
        List of dicts, each with:
            - residue_i: Global residue index (0-based)
            - residue_j: Global residue index (0-based)
            - local_i: Local residue index within chain (0-based)
            - local_j: Local residue index within chain (0-based)
            - chain_i: Chain ID for residue_i
            - chain_j: Chain ID for residue_j
            - resnum_i: Residue number for residue_i
            - resnum_j: Residue number for residue_j
            - distance: CB-CB distance (Angstroms)
            - pae: PAE value for this contact (0-31.75)
            - pae_symmetric: max(PAE[i,j], PAE[j,i])
            - plddt_i: pLDDT of residue_i (0-100)
            - plddt_j: pLDDT of residue_j (0-100)
            - ipSAE: ipSAE score (d0res normalization, 0-1, higher=better)
            - ipSAE_d0chn: ipSAE score (d0chn normalization, 0-1)
            - ipTM: ipTM-style score (0-1)
            - pmc: PMC value (0-32, lower=better)
            - contact_prob: Contact probability (0-1, higher=better, AlphaBridge metric)
            - AB_score: AlphaBridge score = sqrt(contact_prob * ipTM) (0-1, higher=better)
    """
    numres = result.num_residues
    if numres == 0:
        return []

    coordinates = result.cb_coordinates
    chains = result.chains
    unique_chains = np.unique(chains)
    plddt = result.plddt
    pae_matrix = result.pae_matrix
    residue_numbers = result.residue_numbers

    # Get contact probability matrix (native from AF3, derived for others)
    contact_probs_matrix = result.contact_probs
    if contact_probs_matrix is None:
        # Try to derive from PDE or PAE
        try:
            from extractors.pde import extract_contact_probs
            contact_probs_matrix = extract_contact_probs(result)
        except (ImportError, Exception):
            pass

    # Calculate distance matrix
    distances = np.sqrt(
        ((coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]) ** 2).sum(axis=2)
    )

    # Calculate PMC matrix (symmetric PAE + pLDDT penalty)
    # PMC = max(PAE[i,j], PAE[j,i]) + pLDDT_penalty/3
    symmetric_pae = np.maximum(pae_matrix, pae_matrix.T)

    # pLDDT penalty: -1 * ((plddt_i + plddt_j)/2 - 100), 0 if within 2 residues
    plddt_avg = (plddt[:, np.newaxis] + plddt[np.newaxis, :]) / 2.0
    plddt_penalty = -1.0 * (plddt_avg - 100.0)

    # Zero out penalty for residues within 2 positions on same chain
    for i in range(numres):
        for j in range(numres):
            if chains[i] == chains[j] and abs(i - j) <= 2:
                plddt_penalty[i, j] = 0.0

    pmc_matrix = np.minimum(32.0, symmetric_pae + plddt_penalty / 3.0)

    # Chain classification for d0 calculation
    chain_dict = result.chain_types
    chain_pair_type = {}
    for chain1 in unique_chains:
        chain_pair_type[chain1] = {}
        for chain2 in unique_chains:
            if chain1 == chain2:
                continue
            if chain_dict.get(chain1) == 'nucleic_acid' or chain_dict.get(chain2) == 'nucleic_acid':
                chain_pair_type[chain1][chain2] = 'nucleic_acid'
            else:
                chain_pair_type[chain1][chain2] = 'protein'

    # Build chain index mappings (global -> local)
    chain_indices = {}
    global_to_local = {}
    for chain in unique_chains:
        indices = np.where(chains == chain)[0]
        chain_indices[chain] = indices
        for local_idx, global_idx in enumerate(indices):
            global_to_local[global_idx] = (chain, local_idx)

    # Pre-calculate d0chn for each chain pair
    d0chn = {}
    for chain1 in unique_chains:
        d0chn[chain1] = {}
        for chain2 in unique_chains:
            if chain1 == chain2:
                continue
            n0chn = len(chain_indices[chain1]) + len(chain_indices[chain2])
            d0chn[chain1][chain2] = calc_d0(n0chn, chain_pair_type[chain1][chain2])

    # Pre-calculate d0res for each residue (number of valid pairs from that residue)
    d0res_cache = {}
    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1 == chain2:
                continue
            valid_pairs_matrix = (chains == chain2) & (pae_matrix < pae_cutoff)
            for i in chain_indices[chain1]:
                n_valid = np.sum(valid_pairs_matrix[i])
                d0res_cache[(i, chain2)] = calc_d0(n_valid, chain_pair_type[chain1][chain2])

    contacts = []

    # Find all inter-chain contacts
    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1 >= chain2:  # Only process each pair once (A-B, not B-A)
                continue

            indices1 = chain_indices[chain1]
            indices2 = chain_indices[chain2]

            for i in indices1:
                for j in indices2:
                    dist = distances[i, j]
                    pae_ij = pae_matrix[i, j]

                    # Check contact criteria
                    if dist > dist_cutoff:
                        continue
                    if pae_ij >= pae_cutoff:
                        continue

                    # Calculate scores for this contact
                    d0_chn = d0chn[chain1][chain2]
                    d0_res_i = d0res_cache.get((i, chain2), d0_chn)
                    d0_res_j = d0res_cache.get((j, chain1), d0_chn)

                    # ipSAE scores (use average of both directions for symmetry)
                    ipsae_ij = ptm_func(np.array([pae_ij]), d0_res_i)[0]
                    ipsae_ji = ptm_func(np.array([pae_matrix[j, i]]), d0_res_j)[0]
                    ipsae = (ipsae_ij + ipsae_ji) / 2.0

                    # ipSAE with chain normalization
                    ipsae_d0chn_ij = ptm_func(np.array([pae_ij]), d0_chn)[0]
                    ipsae_d0chn_ji = ptm_func(np.array([pae_matrix[j, i]]), d0_chn)[0]
                    ipsae_d0chn = (ipsae_d0chn_ij + ipsae_d0chn_ji) / 2.0

                    # ipTM-style score (same as d0chn but conceptually different)
                    iptm = ipsae_d0chn

                    # Get local indices
                    _, local_i = global_to_local[i]
                    _, local_j = global_to_local[j]

                    # Get contact probability if available
                    contact_prob = None
                    if contact_probs_matrix is not None:
                        contact_prob = float(contact_probs_matrix[i, j])

                    # Calculate AB_score (AlphaBridge style)
                    # AB_score = sqrt(contact_prob * ipTM)
                    ab_score = None
                    if contact_prob is not None and contact_prob > 0:
                        ab_score = float(math.sqrt(contact_prob * iptm))

                    contacts.append({
                        'residue_i': int(i),
                        'residue_j': int(j),
                        'local_i': int(local_i),
                        'local_j': int(local_j),
                        'chain_i': chain1,
                        'chain_j': chain2,
                        'resnum_i': int(residue_numbers[i]),
                        'resnum_j': int(residue_numbers[j]),
                        'distance': float(dist),
                        'pae': float(pae_ij),
                        'pae_symmetric': float(max(pae_ij, pae_matrix[j, i])),
                        'plddt_i': float(plddt[i]),
                        'plddt_j': float(plddt[j]),
                        'ipSAE': float(ipsae),
                        'ipSAE_d0chn': float(ipsae_d0chn),
                        'ipTM': float(iptm),
                        'pmc': float(pmc_matrix[i, j]),
                        'contact_prob': contact_prob,
                        'AB_score': ab_score,
                    })

    return contacts


def get_per_contact_scores_for_interface(
    contacts: List[Dict],
    chain1: str,
    chain2: str,
) -> List[Dict]:
    """
    Filter per-contact scores for a specific interface.

    Args:
        contacts: Full list from calculate_per_contact_scores()
        chain1: First chain ID
        chain2: Second chain ID

    Returns:
        Filtered list of contacts for this interface
    """
    return [
        c for c in contacts
        if (c['chain_i'] == chain1 and c['chain_j'] == chain2) or
           (c['chain_i'] == chain2 and c['chain_j'] == chain1)
    ]


# =============================================================================
# Main scoring function
# =============================================================================

def calculate_scores_from_result(result: 'FoldingResult', pae_cutoff: float,
                                  dist_cutoff: float, return_per_residue: bool = False
                                  ) -> Tuple[List[Dict], Optional[List[Dict]]]:
    """
    Calculate all ipSAE scores from a FoldingResult object.

    This function calculates interface scores between all chain pairs in a
    protein structure prediction, including:
    - ipSAE: interface predicted Structural Alignment Error
    - ipTM: interface predicted TM-score
    - pDockQ/pDockQ2: predicted DockQ scores
    - LIS: Local Interaction Score

    Args:
        result: FoldingResult object from a reader
        pae_cutoff: PAE cutoff for ipSAE calculation (residue pairs with PAE
                    above this are excluded)
        dist_cutoff: Distance cutoff for interface residues in Angstroms
        return_per_residue: Whether to return per-residue results

    Returns:
        Tuple of (aggregate_results, per_residue_results)
        - aggregate_results: List of dicts with scores per chain pair
        - per_residue_results: List of dicts with per-residue scores (or None)
    """
    # Extract data from FoldingResult
    numres = result.num_residues
    if numres == 0:
        return [], None

    coordinates = result.cb_coordinates
    chains = result.chains
    unique_chains = np.unique(chains)
    residue_types = result.residue_types
    plddt = result.plddt
    cb_plddt = result.metadata.get('cb_plddt', plddt)
    pae_matrix = result.pae_matrix

    # Build residues list for per-residue output
    residues = [{'resnum': result.residue_numbers[i], 'res': result.residue_types[i]}
                for i in range(numres)]

    # Chain classification
    chain_dict = result.chain_types
    chain_pair_type = {}
    for chain1 in unique_chains:
        chain_pair_type[chain1] = {}
        for chain2 in unique_chains:
            if chain1 == chain2:
                continue
            if chain_dict.get(chain1) == 'nucleic_acid' or chain_dict.get(chain2) == 'nucleic_acid':
                chain_pair_type[chain1][chain2] = 'nucleic_acid'
            else:
                chain_pair_type[chain1][chain2] = 'protein'

    # Calculate distance matrix
    distances = np.sqrt(((coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]) ** 2).sum(axis=2))

    # Get AF3 ipTM scores (convert from (chain1, chain2) tuple keys to nested dict)
    iptm_af3 = {}
    for chain1 in unique_chains:
        iptm_af3[chain1] = {}
        for chain2 in unique_chains:
            if chain1 == chain2:
                continue
            iptm_af3[chain1][chain2] = result.chain_pair_iptm.get((chain1, chain2), 0.0)

    # Initialize score dictionaries
    def init_chainpair_zeros():
        return {c1: {c2: 0 for c2 in unique_chains if c1 != c2} for c1 in unique_chains}

    def init_chainpair_set():
        return {c1: {c2: set() for c2 in unique_chains if c1 != c2} for c1 in unique_chains}

    def init_chainpair_npzeros():
        return {c1: {c2: np.zeros(numres) for c2 in unique_chains if c1 != c2} for c1 in unique_chains}

    # Score arrays
    iptm_d0chn_byres = init_chainpair_npzeros()
    ipsae_d0chn_byres = init_chainpair_npzeros()
    ipsae_d0dom_byres = init_chainpair_npzeros()
    ipsae_d0res_byres = init_chainpair_npzeros()

    n0chn = init_chainpair_zeros()
    n0dom = init_chainpair_zeros()
    n0res = init_chainpair_zeros()
    n0res_byres = init_chainpair_npzeros()

    d0chn = init_chainpair_zeros()
    d0dom = init_chainpair_zeros()
    d0res = init_chainpair_zeros()
    d0res_byres = init_chainpair_npzeros()

    unique_residues_chain1 = init_chainpair_set()
    unique_residues_chain2 = init_chainpair_set()
    dist_unique_residues_chain1 = init_chainpair_set()
    dist_unique_residues_chain2 = init_chainpair_set()
    pDockQ_unique_residues = init_chainpair_set()

    pDockQ = init_chainpair_zeros()
    pDockQ2 = init_chainpair_zeros()
    LIS = init_chainpair_zeros()

    # Calculate pDockQ
    pDockQ_cutoff = 8.0
    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1 == chain2:
                continue
            npairs = 0
            for i in range(numres):
                if chains[i] != chain1:
                    continue
                valid_pairs = (chains == chain2) & (distances[i] <= pDockQ_cutoff)
                npairs += np.sum(valid_pairs)
                if valid_pairs.any():
                    pDockQ_unique_residues[chain1][chain2].add(i)
                    for residue in np.where(valid_pairs)[0]:
                        pDockQ_unique_residues[chain1][chain2].add(residue)

            if npairs > 0:
                mean_plddt = cb_plddt[list(pDockQ_unique_residues[chain1][chain2])].mean()
                x = mean_plddt * math.log10(npairs)
                pDockQ[chain1][chain2] = 0.724 / (1 + math.exp(-0.052 * (x - 152.611))) + 0.018
            else:
                pDockQ[chain1][chain2] = 0.0

    # Calculate pDockQ2
    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1 == chain2:
                continue
            npairs = 0
            total = 0.0
            for i in range(numres):
                if chains[i] != chain1:
                    continue
                valid_pairs = (chains == chain2) & (distances[i] <= pDockQ_cutoff)
                if valid_pairs.any():
                    npairs += np.sum(valid_pairs)
                    pae_list = pae_matrix[i][valid_pairs]
                    pae_list_ptm = ptm_func_vec(pae_list, 10.0)
                    total += pae_list_ptm.sum()

            if npairs > 0:
                mean_plddt = cb_plddt[list(pDockQ_unique_residues[chain1][chain2])].mean()
                mean_ptm = total / npairs
                x = mean_plddt * mean_ptm
                pDockQ2[chain1][chain2] = 1.31 / (1 + math.exp(-0.075 * (x - 84.733))) + 0.005
            else:
                pDockQ2[chain1][chain2] = 0.0

    # Calculate LIS
    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1 == chain2:
                continue
            mask = (chains[:, None] == chain1) & (chains[None, :] == chain2)
            selected_pae = pae_matrix[mask]
            if selected_pae.size > 0:
                valid_pae = selected_pae[selected_pae <= 12]
                if valid_pae.size > 0:
                    scores = (12 - valid_pae) / 12
                    LIS[chain1][chain2] = np.mean(scores)
                else:
                    LIS[chain1][chain2] = 0.0
            else:
                LIS[chain1][chain2] = 0.0

    # Calculate ipTM/ipSAE scores
    valid_pair_counts = init_chainpair_zeros()
    dist_valid_pair_counts = init_chainpair_zeros()

    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1 == chain2:
                continue

            n0chn[chain1][chain2] = np.sum(chains == chain1) + np.sum(chains == chain2)
            d0chn[chain1][chain2] = calc_d0(n0chn[chain1][chain2], chain_pair_type[chain1][chain2])
            ptm_matrix_d0chn = ptm_func_vec(pae_matrix, d0chn[chain1][chain2])

            valid_pairs_iptm = (chains == chain2)
            valid_pairs_matrix = (chains == chain2) & (pae_matrix < pae_cutoff)

            for i in range(numres):
                if chains[i] != chain1:
                    continue

                valid_pairs_ipsae = valid_pairs_matrix[i]
                iptm_d0chn_byres[chain1][chain2][i] = ptm_matrix_d0chn[i, valid_pairs_iptm].mean() if valid_pairs_iptm.any() else 0.0
                ipsae_d0chn_byres[chain1][chain2][i] = ptm_matrix_d0chn[i, valid_pairs_ipsae].mean() if valid_pairs_ipsae.any() else 0.0

                valid_pair_counts[chain1][chain2] += np.sum(valid_pairs_ipsae)
                if valid_pairs_ipsae.any():
                    unique_residues_chain1[chain1][chain2].add(residues[i]['resnum'])
                    for j in np.where(valid_pairs_ipsae)[0]:
                        unique_residues_chain2[chain1][chain2].add(residues[j]['resnum'])

                valid_pairs = (chains == chain2) & (pae_matrix[i] < pae_cutoff) & (distances[i] < dist_cutoff)
                dist_valid_pair_counts[chain1][chain2] += np.sum(valid_pairs)

                if valid_pairs.any():
                    dist_unique_residues_chain1[chain1][chain2].add(residues[i]['resnum'])
                    for j in np.where(valid_pairs)[0]:
                        dist_unique_residues_chain2[chain1][chain2].add(residues[j]['resnum'])

    # Calculate d0dom scores
    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1 == chain2:
                continue

            residues_1 = len(unique_residues_chain1[chain1][chain2])
            residues_2 = len(unique_residues_chain2[chain1][chain2])
            n0dom[chain1][chain2] = residues_1 + residues_2
            d0dom[chain1][chain2] = calc_d0(n0dom[chain1][chain2], chain_pair_type[chain1][chain2])

            ptm_matrix_d0dom = ptm_func_vec(pae_matrix, d0dom[chain1][chain2])
            valid_pairs_matrix = (chains == chain2) & (pae_matrix < pae_cutoff)

            n0res_byres_all = np.sum(valid_pairs_matrix, axis=1)
            d0res_byres_all = calc_d0_array(n0res_byres_all, chain_pair_type[chain1][chain2])
            n0res_byres[chain1][chain2] = n0res_byres_all
            d0res_byres[chain1][chain2] = d0res_byres_all

            for i in range(numres):
                if chains[i] != chain1:
                    continue
                valid_pairs = valid_pairs_matrix[i]
                ipsae_d0dom_byres[chain1][chain2][i] = ptm_matrix_d0dom[i, valid_pairs].mean() if valid_pairs.any() else 0.0

                ptm_row_d0res = ptm_func_vec(pae_matrix[i], d0res_byres[chain1][chain2][i])
                ipsae_d0res_byres[chain1][chain2][i] = ptm_row_d0res[valid_pairs].mean() if valid_pairs.any() else 0.0

    # Compute aggregate scores
    results_list = []

    ipsae_d0res_asym = init_chainpair_zeros()
    ipsae_d0chn_asym = init_chainpair_zeros()
    ipsae_d0dom_asym = init_chainpair_zeros()
    iptm_d0chn_asym = init_chainpair_zeros()

    # Calculate pTM (global) using our recalculation
    pTM_calculated = calculate_ptm_from_pae(pae_matrix)

    # Calculate directional ipTM and n_contacts for each chain pair
    iptm_calculated = init_chainpair_zeros()
    n_contacts = init_chainpair_zeros()
    contact_threshold = 8.0  # Standard contact distance

    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1 == chain2:
                continue
            # Directional ipTM (our recalculation)
            iptm_calculated[chain1][chain2] = calculate_iptm_directional(
                pae_matrix, chains, chain1, chain2
            )
            # Contact count
            contacts = find_interface_contacts(result, chain1, chain2, contact_threshold)
            n_contacts[chain1][chain2] = len(contacts)

    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1 == chain2:
                continue

            max_idx = np.argmax(iptm_d0chn_byres[chain1][chain2])
            iptm_d0chn_asym[chain1][chain2] = iptm_d0chn_byres[chain1][chain2][max_idx]

            max_idx = np.argmax(ipsae_d0chn_byres[chain1][chain2])
            ipsae_d0chn_asym[chain1][chain2] = ipsae_d0chn_byres[chain1][chain2][max_idx]

            max_idx = np.argmax(ipsae_d0dom_byres[chain1][chain2])
            ipsae_d0dom_asym[chain1][chain2] = ipsae_d0dom_byres[chain1][chain2][max_idx]

            max_idx = np.argmax(ipsae_d0res_byres[chain1][chain2])
            ipsae_d0res_asym[chain1][chain2] = ipsae_d0res_byres[chain1][chain2][max_idx]
            n0res[chain1][chain2] = n0res_byres[chain1][chain2][max_idx]
            d0res[chain1][chain2] = d0res_byres[chain1][chain2][max_idx]

    # Build results for each chain pair
    chainpairs = set()
    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1 < chain2:
                chainpairs.add((chain1, chain2))

    for chain_a, chain_b in sorted(chainpairs):
        for pair in [(chain_a, chain_b), (chain_b, chain_a)]:
            chain1, chain2 = pair

            residues_1 = len(unique_residues_chain1[chain1][chain2])
            residues_2 = len(unique_residues_chain2[chain1][chain2])
            dist_residues_1 = len(dist_unique_residues_chain1[chain1][chain2])
            dist_residues_2 = len(dist_unique_residues_chain2[chain1][chain2])

            results_list.append({
                'chain1': chain1,
                'chain2': chain2,
                'type': 'asym',
                'ipSAE': ipsae_d0res_asym[chain1][chain2],
                'ipSAE_d0chn': ipsae_d0chn_asym[chain1][chain2],
                'ipSAE_d0dom': ipsae_d0dom_asym[chain1][chain2],
                'ipTM_extracted': iptm_af3[chain1][chain2],
                'ipTM_calculated': iptm_calculated[chain1][chain2],
                'ipTM_d0chn': iptm_d0chn_asym[chain1][chain2],
                'pTM_calculated': pTM_calculated,
                'pDockQ': pDockQ[chain1][chain2],
                'pDockQ2': pDockQ2[chain1][chain2],
                'LIS': LIS[chain1][chain2],
                'n_contacts': n_contacts[chain1][chain2],
                'n0res': int(n0res[chain1][chain2]),
                'n0chn': int(n0chn[chain1][chain2]),
                'n0dom': int(n0dom[chain1][chain2]),
                'd0res': d0res[chain1][chain2],
                'd0chn': d0chn[chain1][chain2],
                'd0dom': d0dom[chain1][chain2],
                'nres1': residues_1,
                'nres2': residues_2,
                'dist1': dist_residues_1,
                'dist2': dist_residues_2
            })

        # Add max values
        max_ipsae = max(ipsae_d0res_asym[chain_a][chain_b], ipsae_d0res_asym[chain_b][chain_a])
        max_ipsae_d0chn = max(ipsae_d0chn_asym[chain_a][chain_b], ipsae_d0chn_asym[chain_b][chain_a])
        max_ipsae_d0dom = max(ipsae_d0dom_asym[chain_a][chain_b], ipsae_d0dom_asym[chain_b][chain_a])
        max_iptm_d0chn = max(iptm_d0chn_asym[chain_a][chain_b], iptm_d0chn_asym[chain_b][chain_a])
        max_pDockQ2 = max(pDockQ2[chain_a][chain_b], pDockQ2[chain_b][chain_a])
        avg_LIS = (LIS[chain_a][chain_b] + LIS[chain_b][chain_a]) / 2.0

        # Determine which direction has max values for n0/d0
        if ipsae_d0res_asym[chain_a][chain_b] >= ipsae_d0res_asym[chain_b][chain_a]:
            max_n0res = n0res[chain_a][chain_b]
            max_d0res = d0res[chain_a][chain_b]
        else:
            max_n0res = n0res[chain_b][chain_a]
            max_d0res = d0res[chain_b][chain_a]

        if ipsae_d0dom_asym[chain_a][chain_b] >= ipsae_d0dom_asym[chain_b][chain_a]:
            max_n0dom = n0dom[chain_a][chain_b]
            max_d0dom = d0dom[chain_a][chain_b]
        else:
            max_n0dom = n0dom[chain_b][chain_a]
            max_d0dom = d0dom[chain_b][chain_a]

        residues_1 = max(len(unique_residues_chain2[chain_a][chain_b]), len(unique_residues_chain1[chain_b][chain_a]))
        residues_2 = max(len(unique_residues_chain1[chain_a][chain_b]), len(unique_residues_chain2[chain_b][chain_a]))
        dist_residues_1 = max(len(dist_unique_residues_chain2[chain_a][chain_b]), len(dist_unique_residues_chain1[chain_b][chain_a]))
        dist_residues_2 = max(len(dist_unique_residues_chain1[chain_a][chain_b]), len(dist_unique_residues_chain2[chain_b][chain_a]))

        # Max ipTM calculated and total contacts for this pair
        max_iptm_calculated = max(iptm_calculated[chain_a][chain_b], iptm_calculated[chain_b][chain_a])
        total_contacts = n_contacts[chain_a][chain_b] + n_contacts[chain_b][chain_a]

        results_list.append({
            'chain1': chain_b,
            'chain2': chain_a,
            'type': 'max',
            'ipSAE': max_ipsae,
            'ipSAE_d0chn': max_ipsae_d0chn,
            'ipSAE_d0dom': max_ipsae_d0dom,
            'ipTM_extracted': iptm_af3[chain_a][chain_b],
            'ipTM_calculated': max_iptm_calculated,
            'ipTM_d0chn': max_iptm_d0chn,
            'pTM_calculated': pTM_calculated,
            'pDockQ': pDockQ[chain_a][chain_b],
            'pDockQ2': max_pDockQ2,
            'LIS': avg_LIS,
            'n_contacts': total_contacts,
            'n0res': int(max_n0res),
            'n0chn': int(n0chn[chain_a][chain_b]),
            'n0dom': int(max_n0dom),
            'd0res': max_d0res,
            'd0chn': d0chn[chain_a][chain_b],
            'd0dom': max_d0dom,
            'nres1': residues_1,
            'nres2': residues_2,
            'dist1': dist_residues_1,
            'dist2': dist_residues_2
        })

    # Build per-residue results if requested
    per_residue_results = None
    if return_per_residue:
        # Try to get domain clusters (optional, requires igraph)
        domain_clusters = None
        try:
            from extractors.clustering import cluster_domains_from_result, HAS_IGRAPH
            if HAS_IGRAPH:
                domain_clusters = cluster_domains_from_result(result)
        except (ImportError, Exception):
            pass

        per_residue_results = []
        for chain1 in unique_chains:
            for chain2 in unique_chains:
                if chain1 == chain2:
                    continue
                for i in range(numres):
                    if chains[i] != chain1:
                        continue

                    # Get domain cluster if available
                    cluster = domain_clusters[i] if domain_clusters is not None else None

                    per_residue_results.append({
                        'residue_idx': i + 1,
                        'align_chain': chain1,
                        'scored_chain': chain2,
                        'resnum': residues[i]['resnum'],
                        'restype': residues[i]['res'],
                        'plddt': plddt[i],
                        'domain_cluster': cluster,
                        'n0chn': int(n0chn[chain1][chain2]),
                        'n0dom': int(n0dom[chain1][chain2]),
                        'n0res': int(n0res_byres[chain1][chain2][i]),
                        'd0chn': d0chn[chain1][chain2],
                        'd0dom': d0dom[chain1][chain2],
                        'd0res': d0res_byres[chain1][chain2][i],
                        'ipTM_d0chn': iptm_d0chn_byres[chain1][chain2][i],
                        'ipSAE_d0chn': ipsae_d0chn_byres[chain1][chain2][i],
                        'ipSAE_d0dom': ipsae_d0dom_byres[chain1][chain2][i],
                        'ipSAE': ipsae_d0res_byres[chain1][chain2][i]
                    })

    return results_list, per_residue_results
