"""
Extractors module for ipSAE_batch.

This module provides functions for extracting and computing various
metrics from FoldingResult data:

- pae: PAE matrix extraction and utilities
- pmc: PMC (Predicted Merged Confidence) computation
- pde: PDE (Predicted Distance Error) extraction

All extractors work with FoldingResult from data_readers.
"""

from .pae import (
    extract_pae,
    symmetrize_pae,
    get_chain_pair_pae,
    compute_pae_statistics,
    apply_pae_threshold,
    pae_to_contact_probability,
)

from .pmc import (
    extract_pmc,
    compute_plddt_penalty_matrix,
    pmc_to_contact_matrix,
)

from .pde import (
    extract_contact_probs,
    extract_pde,
    pde_to_contact_probability,
    compute_pde_statistics,
    estimate_pde_from_pae,
    extract_or_estimate_pde,
    extract_both_pde,
)

from .ptm import (
    calculate_d0,
    calculate_ptm_from_pae,
    calculate_iptm_from_pae,
    calculate_iptm_directional,
    calculate_ptm_per_chain,
    calculate_all_ptm_scores,
    extract_ptm_scores,
)

from .contacts import (
    compute_distance_matrix,
    find_contacts,
    find_interface_contacts,
    identify_interface_regions,
    get_interface_links,
    get_leiden_interfaces,
    get_alphabridge_interfaces,
    get_geometric_interfaces,
    get_proximity_contacts,
    get_contact_matrix,
    summarize_interfaces,
)

# Clustering is optional (requires igraph)
try:
    from .clustering import (
        cluster_domains,
        cluster_domains_from_result,
        get_domain_boundaries,
        get_interacting_domains,
        get_domain_info,
        HAS_IGRAPH,
    )
except ImportError:
    HAS_IGRAPH = False

__all__ = [
    # PAE
    'extract_pae',
    'symmetrize_pae',
    'get_chain_pair_pae',
    'compute_pae_statistics',
    'apply_pae_threshold',
    'pae_to_contact_probability',
    # PMC
    'extract_pmc',
    'compute_plddt_penalty_matrix',
    'pmc_to_contact_matrix',
    # PDE / Contact Probability
    'extract_contact_probs',
    'extract_pde',
    'pde_to_contact_probability',
    'compute_pde_statistics',
    'estimate_pde_from_pae',
    'extract_or_estimate_pde',
    'extract_both_pde',
    # PTM
    'calculate_d0',
    'calculate_ptm_from_pae',
    'calculate_iptm_from_pae',
    'calculate_iptm_directional',
    'calculate_ptm_per_chain',
    'calculate_all_ptm_scores',
    'extract_ptm_scores',
    # Contacts
    'compute_distance_matrix',
    'find_contacts',
    'find_interface_contacts',
    'identify_interface_regions',
    'get_interface_links',
    'get_leiden_interfaces',
    'get_alphabridge_interfaces',
    'get_geometric_interfaces',
    'get_proximity_contacts',
    'get_contact_matrix',
    'summarize_interfaces',
    # Clustering (optional, requires igraph)
    'HAS_IGRAPH',
    'cluster_domains',
    'cluster_domains_from_result',
    'get_domain_boundaries',
    'get_interacting_domains',
    'get_domain_info',
]
