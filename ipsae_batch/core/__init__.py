"""
Core scoring module for ipSAE calculations.

This module provides the core algorithms for calculating interface scores
from protein structure predictions.

Per-contact scoring:
- calculate_per_contact_scores(): Returns scores for each individual contact
- get_per_contact_scores_for_interface(): Filter contacts for a specific chain pair
"""

from .scoring import (
    ptm_func,
    ptm_func_vec,
    calc_d0,
    calc_d0_array,
    calculate_scores_from_result,
    calculate_per_contact_scores,
    get_per_contact_scores_for_interface,
)

from .residue_selection import (
    parse_residue_selection,
    is_residue_selected,
    is_contact_selected,
    filter_contacts_by_selection,
    calculate_ipsae_selection_metrics,
    display_selection_summary,
)

__all__ = [
    'ptm_func',
    'ptm_func_vec',
    'calc_d0',
    'calc_d0_array',
    'calculate_scores_from_result',
    'calculate_per_contact_scores',
    'get_per_contact_scores_for_interface',
    # Residue selection
    'parse_residue_selection',
    'is_residue_selected',
    'is_contact_selected',
    'filter_contacts_by_selection',
    'calculate_ipsae_selection_metrics',
    'display_selection_summary',
]
