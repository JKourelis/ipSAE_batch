"""
CSV output writers for ipSAE scoring results.

Provides functions to write aggregate, per-residue, and per-contact scoring results to CSV files.
"""

import csv
from typing import List, Dict


# Field definitions for output CSV files
AGGREGATE_FIELDS = [
    'job_name', 'model', 'chain1', 'chain2', 'pae_cutoff', 'dist_cutoff', 'type',
    'ipSAE', 'ipSAE_d0chn', 'ipSAE_d0dom', 'ipTM_extracted', 'ipTM_calculated', 'ipTM_d0chn',
    'pTM_calculated', 'pDockQ', 'pDockQ2', 'LIS', 'n_contacts',
    'n0res', 'n0chn', 'n0dom', 'd0res', 'd0chn', 'd0dom',
    'nres1', 'nres2', 'dist1', 'dist2'
]

PER_RESIDUE_FIELDS = [
    'job_name', 'model', 'residue_idx', 'align_chain', 'scored_chain',
    'resnum', 'restype', 'plddt', 'domain_cluster',
    'n0chn', 'n0dom', 'n0res', 'd0chn', 'd0dom', 'd0res',
    'ipTM_d0chn', 'ipSAE_d0chn', 'ipSAE_d0dom', 'ipSAE'
]

# Per-contact fields - one row per contact pair
# These are the available metrics for coloring contact lines in visualizations
#
# THEORETICAL RANGES:
#   ipSAE:        0-1 (higher = better)
#   ipSAE_d0chn:  0-1 (higher = better)
#   ipTM:         0-1 (higher = better)
#   contact_prob: 0-1 (higher = better, AlphaBridge uses cutoff 0.5)
#   AB_score:     0-1 (higher = better, AlphaBridge = sqrt(contact_prob * ipTM))
#   pae:          0-31.75 (lower = better)
#   pmc:          0-32 (lower = better)
#   distance:     0-∞ Å (lower = closer contact)
#   plddt:        0-100 (higher = better)
#
PER_CONTACT_FIELDS = [
    'job_name', 'model',
    # Contact identifiers
    'residue_i', 'residue_j',           # Global indices (0-based)
    'local_i', 'local_j',               # Local indices within chain (0-based)
    'chain_i', 'chain_j',               # Chain IDs
    'resnum_i', 'resnum_j',             # Residue numbers
    # Contact properties
    'distance',                          # CB-CB distance (Angstroms)
    'pae',                               # PAE value (i->j direction, 0-31.75)
    'pae_symmetric',                     # max(PAE[i,j], PAE[j,i])
    'plddt_i', 'plddt_j',               # pLDDT values (0-100)
    # Calculated scores - use for line coloring
    'ipSAE',                             # ipSAE score (d0res, 0-1) - STANDARD
    'ipSAE_d0chn',                       # ipSAE score (d0chn, 0-1)
    'ipTM',                              # ipTM-style score (0-1)
    'pmc',                               # PMC value (0-32)
    'contact_prob',                      # Contact probability (0-1, AlphaBridge)
    'AB_score',                          # AlphaBridge score = sqrt(contact_prob * ipTM)
]


def write_aggregate_csv(results: List[Dict], output_file: str) -> None:
    """
    Write aggregate scoring results to CSV.

    Args:
        results: List of dictionaries containing aggregate scores per chain pair
        output_file: Path to output CSV file
    """
    if not results:
        return

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=AGGREGATE_FIELDS)
        writer.writeheader()
        writer.writerows(results)


def write_per_residue_csv(results: List[Dict], output_file: str) -> None:
    """
    Write per-residue scoring results to CSV.

    Args:
        results: List of dictionaries containing per-residue scores
        output_file: Path to output CSV file
    """
    if not results:
        return

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=PER_RESIDUE_FIELDS)
        writer.writeheader()
        writer.writerows(results)


def write_per_contact_csv(results: List[Dict], output_file: str) -> None:
    """
    Write per-contact scoring results to CSV.

    Each row represents one contact (residue pair) with all available metrics.
    These can be used for visualization (e.g., coloring contact lines by ipSAE).

    Args:
        results: List of dictionaries containing per-contact scores
                 (from calculate_per_contact_scores)
        output_file: Path to output CSV file
    """
    if not results:
        return

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=PER_CONTACT_FIELDS)
        writer.writeheader()
        writer.writerows(results)
