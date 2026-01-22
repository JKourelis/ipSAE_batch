"""
Contact detection and interface identification.

Identifies structural contacts from CB-CB distances and groups them
into interfaces for visualization (e.g., ribbon diagrams).

All functions work with FoldingResult from data_readers.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np

from ..data_readers import FoldingResult


def compute_distance_matrix(result: FoldingResult) -> np.ndarray:
    """
    Compute CB-CB distance matrix from FoldingResult.

    Args:
        result: FoldingResult instance

    Returns:
        NxN distance matrix in Angstroms
    """
    coords = result.cb_coordinates
    return np.sqrt(
        ((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2).sum(axis=2)
    )


def find_contacts(
    result: FoldingResult,
    distance_threshold: float = 8.0,
    min_sequence_separation: int = 6,
) -> List[Tuple[int, int, float]]:
    """
    Find all residue pairs in structural contact.

    Args:
        result: FoldingResult instance
        distance_threshold: Maximum CB-CB distance for contact (Angstroms)
        min_sequence_separation: Minimum sequence separation for same-chain contacts

    Returns:
        List of (residue_i, residue_j, distance) tuples for contacts
    """
    dist_matrix = compute_distance_matrix(result)
    chains = result.chains
    n_res = len(chains)

    contacts = []
    for i in range(n_res):
        for j in range(i + 1, n_res):
            if dist_matrix[i, j] <= distance_threshold:
                # Check sequence separation for same chain
                if chains[i] == chains[j]:
                    if abs(j - i) < min_sequence_separation:
                        continue
                contacts.append((i, j, dist_matrix[i, j]))

    return contacts


def find_interface_contacts(
    result: FoldingResult,
    chain1: str,
    chain2: str,
    distance_threshold: float = 8.0,
) -> List[Tuple[int, int, float]]:
    """
    Find contacts between two specific chains.

    Args:
        result: FoldingResult instance
        chain1: First chain ID
        chain2: Second chain ID
        distance_threshold: Maximum CB-CB distance for contact

    Returns:
        List of (residue_i, residue_j, distance) tuples for contacts
        where residue_i is from chain1 and residue_j is from chain2
    """
    dist_matrix = compute_distance_matrix(result)
    chains = result.chains

    indices1 = np.where(chains == chain1)[0]
    indices2 = np.where(chains == chain2)[0]

    contacts = []
    for i in indices1:
        for j in indices2:
            if dist_matrix[i, j] <= distance_threshold:
                contacts.append((i, j, dist_matrix[i, j]))

    return contacts


def identify_interface_regions(
    contacts: List[Tuple[int, int, float]],
    chains: np.ndarray,
    gap_threshold: int = 5,
) -> Dict[str, List[Tuple[int, int]]]:
    """
    Group contacts into contiguous interface regions per chain.

    Args:
        contacts: List of (residue_i, residue_j, distance) from find_contacts()
        chains: Chain IDs array from FoldingResult.chains
        gap_threshold: Maximum gap between residues to consider same region

    Returns:
        Dict mapping chain_id -> list of (start_idx, end_idx) regions
    """
    # Collect contacting residue indices per chain
    chain_residues: Dict[str, set] = {}

    for i, j, _ in contacts:
        chain_i = chains[i]
        chain_j = chains[j]

        if chain_i != chain_j:  # Inter-chain contacts only
            if chain_i not in chain_residues:
                chain_residues[chain_i] = set()
            if chain_j not in chain_residues:
                chain_residues[chain_j] = set()

            chain_residues[chain_i].add(i)
            chain_residues[chain_j].add(j)

    # Group into contiguous regions
    regions = {}
    for chain_id, indices in chain_residues.items():
        sorted_indices = sorted(indices)
        if not sorted_indices:
            continue

        chain_regions = []
        start = sorted_indices[0]
        prev = sorted_indices[0]

        for idx in sorted_indices[1:]:
            if idx - prev > gap_threshold:
                chain_regions.append((start, prev))
                start = idx
            prev = idx
        chain_regions.append((start, prev))

        regions[chain_id] = chain_regions

    return regions


def get_interface_links(
    result: FoldingResult,
    distance_threshold: float = 8.0,
) -> List[Dict]:
    """
    Get interface links suitable for ribbon diagram visualization.

    Returns links in a format compatible with AlphaBridge-style ribbon plots.

    Args:
        result: FoldingResult instance
        distance_threshold: Maximum CB-CB distance for contact

    Returns:
        List of interface dicts, each with:
            'chain1': str
            'chain2': str
            'links': list of {'residue1': int, 'residue2': int, 'distance': float}
            'regions1': list of (start, end) tuples for chain1
            'regions2': list of (start, end) tuples for chain2
    """
    unique_chains = result.unique_chains
    chains = result.chains
    interfaces = []

    for i, chain1 in enumerate(unique_chains):
        for chain2 in unique_chains[i + 1:]:
            contacts = find_interface_contacts(
                result, chain1, chain2, distance_threshold
            )

            if not contacts:
                continue

            # Get local residue indices (within chain)
            indices1 = np.where(chains == chain1)[0]
            indices2 = np.where(chains == chain2)[0]

            # Convert global indices to local
            idx1_to_local = {g: l for l, g in enumerate(indices1)}
            idx2_to_local = {g: l for l, g in enumerate(indices2)}

            links = []
            local_contacts1 = set()
            local_contacts2 = set()

            for global_i, global_j, dist in contacts:
                local_i = idx1_to_local[global_i]
                local_j = idx2_to_local[global_j]
                links.append({
                    'residue1': local_i,
                    'residue2': local_j,
                    'distance': dist
                })
                local_contacts1.add(local_i)
                local_contacts2.add(local_j)

            # Group into regions
            regions1 = _group_into_regions(sorted(local_contacts1))
            regions2 = _group_into_regions(sorted(local_contacts2))

            interfaces.append({
                'chain1': chain1,
                'chain2': chain2,
                'links': links,
                'regions1': regions1,
                'regions2': regions2,
                'n_contacts': len(contacts),
            })

    return interfaces


def _group_into_regions(
    indices: List[int],
    gap_threshold: int = 5
) -> List[Tuple[int, int]]:
    """Group sorted indices into contiguous regions."""
    if not indices:
        return []

    regions = []
    start = indices[0]
    prev = indices[0]

    for idx in indices[1:]:
        if idx - prev > gap_threshold:
            regions.append((start, prev))
            start = idx
        prev = idx
    regions.append((start, prev))

    return regions


def get_contact_matrix(
    result: FoldingResult,
    distance_threshold: float = 8.0,
    min_sequence_separation: int = 6,
) -> np.ndarray:
    """
    Create a binary contact matrix from CB-CB distances.

    Args:
        result: FoldingResult instance
        distance_threshold: Maximum CB-CB distance for contact
        min_sequence_separation: Minimum sequence separation for same-chain contacts

    Returns:
        NxN binary contact matrix (1 = contact, 0 = no contact)
    """
    dist_matrix = compute_distance_matrix(result)
    chains = result.chains
    n_res = len(chains)

    contact_matrix = (dist_matrix <= distance_threshold).astype(float)

    # Zero out short-range same-chain contacts
    for i in range(n_res):
        for j in range(n_res):
            if chains[i] == chains[j] and abs(j - i) < min_sequence_separation:
                contact_matrix[i, j] = 0

    return contact_matrix


def summarize_interfaces(result: FoldingResult, distance_threshold: float = 8.0) -> Dict:
    """
    Get a summary of all interfaces in the structure.

    Args:
        result: FoldingResult instance
        distance_threshold: Maximum CB-CB distance for contact

    Returns:
        Dictionary with:
            'n_interfaces': int
            'total_contacts': int
            'interfaces': list of interface summaries
    """
    interfaces = get_interface_links(result, distance_threshold)

    return {
        'n_interfaces': len(interfaces),
        'total_contacts': sum(iface['n_contacts'] for iface in interfaces),
        'interfaces': [
            {
                'chains': f"{iface['chain1']}-{iface['chain2']}",
                'n_contacts': iface['n_contacts'],
                'n_regions_chain1': len(iface['regions1']),
                'n_regions_chain2': len(iface['regions2']),
            }
            for iface in interfaces
        ]
    }


def get_leiden_interfaces(
    result: FoldingResult,
    distance_threshold: float = 8.0,
) -> List[Dict]:
    """
    Get interface links based on Leiden clustering.

    Uses Leiden clustering to identify co-evolutionary domains that span
    multiple chains. Each such cluster represents a distinct "interacting
    surface" between chains.

    Args:
        result: FoldingResult instance
        distance_threshold: Maximum CB-CB distance for contact

    Returns:
        List of interface dicts, each with:
            'interface_id': str (e.g., 'I1', 'I2')
            'cluster_id': int
            'chain1': str
            'chain2': str
            'links': list of contact dicts
            'regions1': list of (start, end) tuples for chain1
            'regions2': list of (start, end) tuples for chain2
            'n_contacts': int
    """
    try:
        from .clustering import cluster_domains_from_result, get_interacting_domains, HAS_IGRAPH
        if not HAS_IGRAPH:
            # Fall back to standard interface detection
            return get_interface_links(result, distance_threshold)
    except ImportError:
        return get_interface_links(result, distance_threshold)

    # Get Leiden clusters
    clusters = cluster_domains_from_result(result)
    interacting = get_interacting_domains(clusters, result.chains)

    if not interacting:
        # No interacting domains found, fall back
        return get_interface_links(result, distance_threshold)

    # Get distance matrix for contact detection
    dist_matrix = compute_distance_matrix(result)
    chains = result.chains
    unique_chains = result.unique_chains

    interfaces = []
    interface_count = 0

    for cluster_name, chain_ranges in interacting.items():
        # Get chains involved in this cluster
        involved_chains = list(chain_ranges.keys())

        # For each pair of chains in this cluster
        for i, chain1 in enumerate(involved_chains):
            for chain2 in involved_chains[i + 1:]:
                interface_count += 1

                # Get global indices for each chain in this cluster
                indices1 = result.get_chain_indices(chain1)
                indices2 = result.get_chain_indices(chain2)

                # Filter to only cluster regions
                cluster_ranges1 = chain_ranges.get(chain1, [])
                cluster_ranges2 = chain_ranges.get(chain2, [])

                cluster_indices1 = set()
                for start, end in cluster_ranges1:
                    cluster_indices1.update(range(start, end + 1))

                cluster_indices2 = set()
                for start, end in cluster_ranges2:
                    cluster_indices2.update(range(start, end + 1))

                # Find contacts within cluster regions
                links = []
                local_contacts1 = set()
                local_contacts2 = set()

                idx1_to_local = {g: l for l, g in enumerate(indices1)}
                idx2_to_local = {g: l for l, g in enumerate(indices2)}

                for global_i in cluster_indices1:
                    if global_i not in idx1_to_local:
                        continue
                    for global_j in cluster_indices2:
                        if global_j not in idx2_to_local:
                            continue
                        dist = dist_matrix[global_i, global_j]
                        if dist <= distance_threshold:
                            local_i = idx1_to_local[global_i]
                            local_j = idx2_to_local[global_j]
                            links.append({
                                'residue1': local_i,
                                'residue2': local_j,
                                'distance': float(dist)
                            })
                            local_contacts1.add(local_i)
                            local_contacts2.add(local_j)

                if not links:
                    continue

                # Group into regions
                regions1 = _group_into_regions(sorted(local_contacts1))
                regions2 = _group_into_regions(sorted(local_contacts2))

                interfaces.append({
                    'interface_id': f'I{interface_count}',
                    'cluster_id': int(cluster_name.split('_')[-1]) if '_' in cluster_name else interface_count,
                    'chain1': chain1,
                    'chain2': chain2,
                    'links': links,
                    'regions1': regions1,
                    'regions2': regions2,
                    'n_contacts': len(links),
                })

    # If Leiden didn't find any interfaces, fall back
    if not interfaces:
        return get_interface_links(result, distance_threshold)

    return interfaces


def get_alphabridge_interfaces(
    result: FoldingResult,
    contact_probs: np.ndarray,
    threshold: float = 0.65,
    gap_merge: int = 3,
) -> List[Dict]:
    """
    Get interface links using AlphaBridge's connected-component approach.

    Uses scipy.ndimage.label() on the contact probability matrix to find
    spatially SEPARATE interface regions. This can find MULTIPLE interfaces
    between the same chain pair if the contacts form distinct clusters.

    After finding connected components, interfaces are MERGED if they share
    overlapping regions on either chain (using graph connectivity analysis).
    This matches AlphaBridge's merge_split_interfaces() behavior.

    Args:
        result: FoldingResult instance
        contact_probs: Contact probability matrix (NxN, values 0-1)
        threshold: Minimum contact probability to consider (default 0.65)
                   Validated to match AF3 native interface detection at 0.5
        gap_merge: Maximum gap between regions to merge (default 3)

    Returns:
        List of interface dicts, each with:
            'interface_id': str (e.g., 'I1', 'I2')
            'chain1': str
            'chain2': str
            'links': list of {'residue1': int, 'residue2': int, 'contact_prob': float}
            'regions1': list of (start, end) tuples for chain1
            'regions2': list of (start, end) tuples for chain2
            'n_contacts': int
    """
    from scipy import ndimage

    unique_chains = result.unique_chains
    interfaces = []
    interface_count = 0

    for i, chain1 in enumerate(unique_chains):
        for chain2 in unique_chains[i + 1:]:
            # Get indices for each chain
            indices1 = result.get_chain_indices(chain1)
            indices2 = result.get_chain_indices(chain2)

            if len(indices1) == 0 or len(indices2) == 0:
                continue

            # Extract submatrix for this chain pair
            submatrix = contact_probs[np.ix_(indices1, indices2)]

            # Apply threshold to get binary matrix
            binary = submatrix > threshold

            if not binary.any():
                continue

            # Use ndimage.label to find connected components
            structure = ndimage.generate_binary_structure(2, 2)
            labeled, n_components = ndimage.label(binary, structure=structure)

            if n_components == 0:
                continue

            # Collect all contact blobs
            contact_blobs = []
            for comp_idx in range(n_components):
                comp_label = comp_idx + 1
                comp_mask = (labeled == comp_label)
                contact_positions = np.argwhere(comp_mask)

                if len(contact_positions) == 0:
                    continue

                # Get region ranges for this blob
                local_i_vals = contact_positions[:, 0]
                local_j_vals = contact_positions[:, 1]

                # Create region tuples for this blob
                region1 = (int(local_i_vals.min()), int(local_i_vals.max()))
                region2 = (int(local_j_vals.min()), int(local_j_vals.max()))

                links = []
                for pos in contact_positions:
                    local_i, local_j = int(pos[0]), int(pos[1])
                    prob = float(submatrix[local_i, local_j])
                    links.append({
                        'residue1': local_i,
                        'residue2': local_j,
                        'contact_prob': prob,
                    })

                contact_blobs.append({
                    'region1': region1,
                    'region2': region2,
                    'links': links,
                })

            # Merge blobs into interfaces based on region overlap
            # Two blobs belong to same interface if their regions overlap/touch
            merged_interfaces = _merge_contact_blobs(contact_blobs, gap_merge)

            # Create interface dicts
            for merged in merged_interfaces:
                interface_count += 1

                # Collect all links and regions
                all_links = []
                all_local1 = set()
                all_local2 = set()

                for blob in merged:
                    all_links.extend(blob['links'])
                    for link in blob['links']:
                        all_local1.add(link['residue1'])
                        all_local2.add(link['residue2'])

                # Group into contiguous regions
                regions1 = _group_into_regions_with_gap_merge(
                    sorted(all_local1), gap_merge
                )
                regions2 = _group_into_regions_with_gap_merge(
                    sorted(all_local2), gap_merge
                )

                interfaces.append({
                    'interface_id': f'I{interface_count}',
                    'chain1': chain1,
                    'chain2': chain2,
                    'links': all_links,
                    'regions1': regions1,
                    'regions2': regions2,
                    'n_contacts': len(all_links),
                })

    return interfaces


def _merge_contact_blobs(
    blobs: List[Dict],
    gap_threshold: int = 3
) -> List[List[Dict]]:
    """
    Merge contact blobs that share overlapping or nearby regions.

    This implements AlphaBridge's merge_split_interfaces() logic using
    graph connectivity: blobs that share overlapping regions on chain1
    OR chain2 are grouped into the same interface.

    Args:
        blobs: List of {'region1': (start, end), 'region2': (start, end), 'links': [...]}
        gap_threshold: Maximum gap to consider regions as overlapping

    Returns:
        List of lists, where each inner list contains blobs belonging to same interface
    """
    if not blobs:
        return []

    if len(blobs) == 1:
        return [blobs]

    # Build adjacency: two blobs are connected if their regions overlap
    n = len(blobs)
    adjacency = [set() for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            # Check if regions overlap (with gap tolerance)
            if _regions_overlap(blobs[i]['region1'], blobs[j]['region1'], gap_threshold) or \
               _regions_overlap(blobs[i]['region2'], blobs[j]['region2'], gap_threshold):
                adjacency[i].add(j)
                adjacency[j].add(i)

    # Find connected components using BFS
    visited = [False] * n
    components = []

    for start in range(n):
        if visited[start]:
            continue

        # BFS from this node
        component = []
        queue = [start]
        visited[start] = True

        while queue:
            node = queue.pop(0)
            component.append(blobs[node])

            for neighbor in adjacency[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)

        components.append(component)

    return components


def _regions_overlap(region1: Tuple[int, int], region2: Tuple[int, int], gap: int = 3) -> bool:
    """Check if two regions overlap or are within gap distance."""
    start1, end1 = region1
    start2, end2 = region2

    # Extend regions by gap
    return not (end1 + gap < start2 or end2 + gap < start1)


def _group_into_regions_with_gap_merge(
    indices: List[int],
    gap_threshold: int = 3
) -> List[Tuple[int, int]]:
    """
    Group sorted indices into contiguous regions, merging small gaps.

    This matches AlphaBridge's fix_intervals() behavior which merges
    regions separated by gaps <= 3 residues.

    Args:
        indices: Sorted list of residue indices
        gap_threshold: Maximum gap to merge (default 3, as in AlphaBridge)

    Returns:
        List of (start, end) tuples
    """
    if not indices:
        return []

    regions = []
    start = indices[0]
    prev = indices[0]

    for idx in indices[1:]:
        if idx - prev > gap_threshold:
            regions.append((start, prev))
            start = idx
        prev = idx
    regions.append((start, prev))

    return regions


def get_geometric_interfaces(
    result: FoldingResult,
    distance_threshold: float = 10.0,
    pae_threshold: float = 10.0,
    gap_merge: int = 3,
) -> List[Dict]:
    """
    Get interface links using DISTANCE + PAE threshold.

    This function uses distance AND PAE for interface DETECTION.
    Contacts are identified where:
        distance < distance_threshold AND max(PAE[i,j], PAE[j,i]) < pae_threshold

    The PAE threshold should match the PAE cutoff used for calculations (default 10Å).
    High PAE → filtered out → creates NATURAL GAPS for interface separation.

    Uses scipy.ndimage.label() on binary contact matrix to find spatially
    separate interface regions.

    Args:
        result: FoldingResult instance
        distance_threshold: Maximum CB-CB distance for contact (default 10.0Å)
        pae_threshold: Maximum PAE for contact (default 10.0Å, matches calculation cutoff)
        gap_merge: Maximum gap between regions to merge (default 3)

    Returns:
        List of interface dicts, each with:
            'interface_id': str (e.g., 'I1', 'I2')
            'chain1': str
            'chain2': str
            'links': list of {'residue1': int, 'residue2': int, 'distance': float, 'pae_symmetric': float}
            'regions1': list of (start, end) tuples for chain1
            'regions2': list of (start, end) tuples for chain2
            'n_contacts': int
    """
    from scipy import ndimage

    # Compute CB-CB distance matrix
    dist_matrix = compute_distance_matrix(result)
    pae_matrix = result.pae_matrix

    unique_chains = result.unique_chains
    interfaces = []
    interface_count = 0

    for i, chain1 in enumerate(unique_chains):
        for chain2 in unique_chains[i + 1:]:
            # Get indices for each chain
            indices1 = result.get_chain_indices(chain1)
            indices2 = result.get_chain_indices(chain2)

            if len(indices1) == 0 or len(indices2) == 0:
                continue

            # Extract distance and PAE submatrices for this chain pair
            dist_sub = dist_matrix[np.ix_(indices1, indices2)]
            pae_sub = pae_matrix[np.ix_(indices1, indices2)]
            pae_sub_T = pae_matrix[np.ix_(indices2, indices1)].T

            # Symmetric PAE: max(PAE[i,j], PAE[j,i]) - conservative estimate
            pae_symmetric = np.maximum(pae_sub, pae_sub_T)

            # Apply COMBINED threshold: distance AND PAE
            # This creates sparse matrix with natural gaps
            binary = (dist_sub <= distance_threshold) & (pae_symmetric < pae_threshold)

            if not binary.any():
                continue

            # Use ndimage.label to find connected components
            structure = ndimage.generate_binary_structure(2, 2)
            labeled, n_components = ndimage.label(binary, structure=structure)

            if n_components == 0:
                continue

            # Collect all contact blobs
            contact_blobs = []
            for comp_idx in range(n_components):
                comp_label = comp_idx + 1
                comp_mask = (labeled == comp_label)
                contact_positions = np.argwhere(comp_mask)

                if len(contact_positions) == 0:
                    continue

                # Get region ranges for this blob
                local_i_vals = contact_positions[:, 0]
                local_j_vals = contact_positions[:, 1]

                # Create region tuples for this blob
                region1 = (int(local_i_vals.min()), int(local_i_vals.max()))
                region2 = (int(local_j_vals.min()), int(local_j_vals.max()))

                links = []
                for pos in contact_positions:
                    local_i, local_j = int(pos[0]), int(pos[1])
                    dist = float(dist_sub[local_i, local_j])
                    pae_sym = float(pae_symmetric[local_i, local_j])
                    links.append({
                        'residue1': local_i,
                        'residue2': local_j,
                        'distance': dist,
                        'pae_symmetric': pae_sym,
                    })

                contact_blobs.append({
                    'region1': region1,
                    'region2': region2,
                    'links': links,
                })

            # Merge blobs into interfaces based on region overlap
            merged_interfaces = _merge_contact_blobs(contact_blobs, gap_merge)

            # Create interface dicts
            for merged in merged_interfaces:
                interface_count += 1

                # Collect all links and regions
                all_links = []
                all_local1 = set()
                all_local2 = set()

                for blob in merged:
                    all_links.extend(blob['links'])
                    for link in blob['links']:
                        all_local1.add(link['residue1'])
                        all_local2.add(link['residue2'])

                # Group into contiguous regions
                regions1 = _group_into_regions_with_gap_merge(
                    sorted(all_local1), gap_merge
                )
                regions2 = _group_into_regions_with_gap_merge(
                    sorted(all_local2), gap_merge
                )

                interfaces.append({
                    'interface_id': f'I{interface_count}',
                    'chain1': chain1,
                    'chain2': chain2,
                    'links': all_links,
                    'regions1': regions1,
                    'regions2': regions2,
                    'n_contacts': len(all_links),
                })

    return interfaces


def get_proximity_contacts(
    result: FoldingResult,
    distance_threshold: float = 10.0,
) -> List[Dict]:
    """
    Get ALL proximity contacts for VISUALIZATION only.

    This function returns ALL contacts within distance threshold, regardless of PAE.
    Used for ribbon plot visualization where contacts are colored by PAE quality:
    - High confidence (PAE < threshold): colored
    - Low confidence (PAE > threshold): grey

    NOTE: This is for VISUALIZATION only. For SCORING, use get_geometric_interfaces()
    which filters by both distance AND PAE.

    Args:
        result: FoldingResult instance
        distance_threshold: Maximum CB-CB distance for contact (default 10.0Å)

    Returns:
        List of contact dicts, each with:
            'chain1': str
            'chain2': str
            'links': list of {'residue1': int, 'residue2': int, 'distance': float, 'pae_symmetric': float}
    """
    # Compute CB-CB distance matrix
    dist_matrix = compute_distance_matrix(result)
    pae_matrix = result.pae_matrix

    unique_chains = result.unique_chains
    contacts = []

    for i, chain1 in enumerate(unique_chains):
        for chain2 in unique_chains[i + 1:]:
            # Get indices for each chain
            indices1 = result.get_chain_indices(chain1)
            indices2 = result.get_chain_indices(chain2)

            if len(indices1) == 0 or len(indices2) == 0:
                continue

            # Extract distance and PAE submatrices for this chain pair
            dist_sub = dist_matrix[np.ix_(indices1, indices2)]
            pae_sub = pae_matrix[np.ix_(indices1, indices2)]
            pae_sub_T = pae_matrix[np.ix_(indices2, indices1)].T

            # Symmetric PAE: max(PAE[i,j], PAE[j,i]) - conservative estimate
            pae_symmetric = np.maximum(pae_sub, pae_sub_T)

            # Distance only - no PAE filter
            contact_mask = (dist_sub <= distance_threshold)

            if not contact_mask.any():
                continue

            # Collect all contacts
            contact_positions = np.argwhere(contact_mask)
            links = []
            for pos in contact_positions:
                local_i, local_j = int(pos[0]), int(pos[1])
                links.append({
                    'residue1': local_i,
                    'residue2': local_j,
                    'distance': float(dist_sub[local_i, local_j]),
                    'pae_symmetric': float(pae_symmetric[local_i, local_j]),
                })

            contacts.append({
                'chain1': chain1,
                'chain2': chain2,
                'links': links,
                'n_contacts': len(links),
            })

    return contacts
