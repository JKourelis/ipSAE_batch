"""
Domain clustering module.

Identifies co-evolutionary domains using graph-based clustering
on the PMC (Predicted Merged Confidence) matrix.

Uses the Leiden algorithm from igraph for community detection.

All functions work with FoldingResult from data_readers.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np

try:
    import igraph
    HAS_IGRAPH = True
except ImportError:
    HAS_IGRAPH = False

from ..data_readers import FoldingResult
from .pmc import extract_pmc


def _check_igraph():
    """Check if igraph is available."""
    if not HAS_IGRAPH:
        raise ImportError(
            "igraph is required for domain clustering. "
            "Install it with: pip install igraph"
        )


def cluster_domains(
    pmc_matrix: np.ndarray,
    resolution: float = 0.25,
    cutoff: float = 27.0,
    pae_power: float = 1.0,
) -> np.ndarray:
    """
    Cluster residues into co-evolutionary domains.

    Uses the Leiden algorithm on a graph where edges connect
    residues with PMC below the cutoff, weighted by 1/PMC^power.

    Args:
        pmc_matrix: PMC (or confidence) matrix
        resolution: Leiden resolution parameter (lower = fewer clusters)
        cutoff: PMC cutoff for creating edges
        pae_power: Power to raise weights to

    Returns:
        Array of cluster assignments (one per residue)
    """
    _check_igraph()

    # Build weighted graph
    weights = 1.0 / (pmc_matrix ** pae_power)

    g = igraph.Graph()
    size = weights.shape[0]
    g.add_vertices(range(size))

    # Add edges where PMC is below cutoff
    edges = np.argwhere(pmc_matrix < cutoff)
    sel_weights = weights[edges[:, 0], edges[:, 1]]

    g.add_edges(edges.tolist())
    g.es['weight'] = sel_weights.tolist()

    # Run Leiden clustering
    # Note: resolution_parameter is scaled by 100 in original code
    vc = g.community_leiden(
        weights='weight',
        resolution=resolution / 100,
        n_iterations=-1
    )

    return np.array(vc.membership)


def cluster_domains_from_result(
    result: FoldingResult,
    resolution: Optional[float] = None,
    cutoff: Optional[float] = None,
) -> np.ndarray:
    """
    Cluster domains directly from a FoldingResult.

    Automatically selects parameters based on backend/version.

    Args:
        result: FoldingResult instance
        resolution: Override resolution parameter
        cutoff: Override PMC cutoff

    Returns:
        Array of cluster assignments (one per residue)
    """
    # Extract PMC matrix
    pmc = extract_pmc(result)

    # Select parameters based on backend
    # AF2-style has different optimal parameters than AF3-style
    backend = result.metadata.get('backend', 'alphafold3')

    if resolution is None:
        resolution = 0.5 if backend == 'colabfold' else 0.25
    if cutoff is None:
        cutoff = 2.6 if backend == 'colabfold' else 27.0

    return cluster_domains(pmc, resolution=resolution, cutoff=cutoff)


def get_domain_boundaries(
    clusters: np.ndarray,
    chains: np.ndarray,
) -> Dict[str, List[Tuple[int, int]]]:
    """
    Get domain boundaries per chain.

    Args:
        clusters: Cluster assignments from cluster_domains()
        chains: Chain IDs per residue

    Returns:
        Dict mapping chain_id -> list of (start, end) tuples for each domain
    """
    unique_chains = np.unique(chains)
    domain_boundaries = {}

    for chain in unique_chains:
        chain_mask = chains == chain
        chain_indices = np.where(chain_mask)[0]
        chain_clusters = clusters[chain_mask]

        boundaries = []
        current_cluster = None
        start_idx = None

        for i, (idx, cluster) in enumerate(zip(chain_indices, chain_clusters)):
            if cluster != current_cluster:
                if current_cluster is not None:
                    # End previous domain
                    boundaries.append((start_idx, chain_indices[i-1]))
                # Start new domain
                current_cluster = cluster
                start_idx = idx

        # Close last domain
        if current_cluster is not None:
            boundaries.append((start_idx, chain_indices[-1]))

        domain_boundaries[chain] = boundaries

    return domain_boundaries


def get_interacting_domains(
    clusters: np.ndarray,
    chains: np.ndarray,
) -> Dict[str, Dict[str, List[Tuple[int, int]]]]:
    """
    Find domains that span multiple chains (interacting domains).

    Args:
        clusters: Cluster assignments from cluster_domains()
        chains: Chain IDs per residue

    Returns:
        Dict mapping cluster_name -> {chain_id: [(start, end), ...]}
    """
    unique_clusters = np.unique(clusters)
    unique_chains = np.unique(chains)
    interacting = {}

    for cluster in unique_clusters:
        cluster_mask = clusters == cluster
        cluster_chains = np.unique(chains[cluster_mask])

        # Only include clusters that span multiple chains
        if len(cluster_chains) > 1:
            cluster_name = f"cluster_{cluster}"
            interacting[cluster_name] = {}

            for chain in cluster_chains:
                chain_cluster_mask = cluster_mask & (chains == chain)
                indices = np.where(chain_cluster_mask)[0]

                # Group consecutive indices into ranges
                ranges = []
                start = indices[0]
                prev = indices[0]

                for idx in indices[1:]:
                    if idx != prev + 1:
                        ranges.append((start, prev))
                        start = idx
                    prev = idx
                ranges.append((start, prev))

                interacting[cluster_name][chain] = ranges

    return interacting


def get_domain_info(result: FoldingResult) -> Dict:
    """
    Get complete domain clustering information.

    Args:
        result: FoldingResult instance

    Returns:
        Dictionary with:
            'clusters': Array of cluster assignments
            'n_domains': Number of unique domains
            'boundaries': Domain boundaries per chain
            'interacting': Domains spanning multiple chains
    """
    clusters = cluster_domains_from_result(result)
    chains = result.chains

    return {
        'clusters': clusters,
        'n_domains': len(np.unique(clusters)),
        'boundaries': get_domain_boundaries(clusters, chains),
        'interacting': get_interacting_domains(clusters, chains),
    }
