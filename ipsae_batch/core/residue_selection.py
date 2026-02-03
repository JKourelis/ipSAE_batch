"""
Residue selection utilities for focused interface analysis.

Allows users to analyze specific residues (e.g., active sites, binding pockets, mutation sites)
using standard structural biology syntax: chain:residue format.

Syntax examples:
    "A:100-105"           Chain A, residues 100-105 (inclusive range)
    "A:100,102,104"       Chain A, specific residues
    "A:100-105,B:203"     Multiple chains and mixed formats
    "A:57,102,195"        Enzyme catalytic triad

Contact selection uses OR logic:
    A contact is SELECTED if EITHER residue is in the selection set.
    This enables finding ALL contacts involving a binding pocket residue.
"""

from typing import Dict, Set, List, Optional
import sys


def parse_residue_selection(selection_str: str) -> Dict[str, Set[int]]:
    """
    Parse residue selection string using standard structural biology syntax.

    Supports chain:residue format with ranges and individual residues:
    - "A:100-105" (chain A, residues 100-105)
    - "A:100,102,104" (chain A, specific residues)
    - "A:100-105,B:203,C:50-60" (multiple chains and ranges)

    Args:
        selection_str: Selection string in standard format

    Returns:
        Dictionary mapping chain_id -> set of residue numbers
        Empty dict if selection_str is None or empty
    """
    selections: Dict[str, Set[int]] = {}

    if not selection_str or not selection_str.strip():
        return selections

    try:
        # Split by comma, handling chain:residue groups
        parts = selection_str.split(',')
        current_chain = None

        for part in parts:
            part = part.strip()
            if not part:
                continue

            if ':' in part:
                # New chain:residue specification
                chain, residues_str = part.split(':', 1)
                current_chain = chain.strip()

                if current_chain not in selections:
                    selections[current_chain] = set()

                # Parse the residue part
                _parse_residue_spec(residues_str.strip(), selections[current_chain])
            elif current_chain is not None:
                # Continuation of previous chain (e.g., "A:100,102,104")
                _parse_residue_spec(part, selections[current_chain])
            else:
                # No chain specified yet
                print(f"Warning: Residue '{part}' has no chain, skipping")

    except (ValueError, IndexError) as e:
        print(f"Error parsing residue selection '{selection_str}': {e}")
        print("Expected format: 'A:100-105,B:203,C:50-60'")
        sys.exit(1)

    return selections


def _parse_residue_spec(spec: str, residue_set: Set[int]) -> None:
    """
    Parse a single residue specification and add to the set.

    Handles:
    - Single residue: "100"
    - Range: "100-105"

    Args:
        spec: Residue specification string
        residue_set: Set to add residues to
    """
    spec = spec.strip()
    if not spec:
        return

    if '-' in spec:
        # Range: 100-105
        start_str, end_str = spec.split('-', 1)
        start = int(start_str.strip())
        end = int(end_str.strip())
        for i in range(start, end + 1):
            residue_set.add(i)
    else:
        # Single residue
        residue_set.add(int(spec))


def is_residue_selected(
    chain: str,
    resnum: int,
    residue_selection: Dict[str, Set[int]]
) -> bool:
    """
    Check if a single residue is in the selection.

    Args:
        chain: Chain ID
        resnum: Residue number
        residue_selection: Dict from parse_residue_selection()

    Returns:
        True if residue is selected, False otherwise
    """
    if not residue_selection:
        return False
    return chain in residue_selection and resnum in residue_selection[chain]


def is_contact_selected(
    contact: Dict,
    residue_selection: Dict[str, Set[int]]
) -> bool:
    """
    Check if a contact involves any selected residue (OR logic).

    A contact is SELECTED if EITHER of its residues is in the selection.
    This enables finding ALL contacts involving a binding pocket/active site.

    Args:
        contact: Contact dict with chain_i/chain_j and resnum_i/resnum_j
                 (or chain1/chain2 and res1/res2 for older formats)
        residue_selection: Dict from parse_residue_selection()

    Returns:
        True if either residue is selected, False otherwise
    """
    if not residue_selection:
        return False

    # Support both naming conventions
    chain1 = contact.get('chain_i') or contact.get('chain1')
    chain2 = contact.get('chain_j') or contact.get('chain2')
    res1 = contact.get('resnum_i') or contact.get('res1')
    res2 = contact.get('resnum_j') or contact.get('res2')

    # Convert to int if needed
    if res1 is not None:
        res1 = int(res1)
    if res2 is not None:
        res2 = int(res2)

    # OR logic: selected if EITHER residue is in selection
    selected1 = (chain1 in residue_selection and res1 in residue_selection[chain1])
    selected2 = (chain2 in residue_selection and res2 in residue_selection[chain2])

    return selected1 or selected2


def filter_contacts_by_selection(
    contacts: List[Dict],
    residue_selection: Dict[str, Set[int]]
) -> List[Dict]:
    """
    Filter contacts to only those involving selected residues.

    Args:
        contacts: List of contact dicts
        residue_selection: Dict from parse_residue_selection()

    Returns:
        Filtered list of contacts (subset of input)
    """
    if not residue_selection:
        return contacts

    return [c for c in contacts if is_contact_selected(c, residue_selection)]


def calculate_ipsae_selection_metrics(
    all_contacts: List[Dict],
    residue_selection: Dict[str, Set[int]],
) -> Dict:
    """
    Calculate ipSAE-style metrics for selected contacts only.

    For ipSAE_batch which focuses on ipSAE/ipTM metrics.

    Args:
        all_contacts: All contacts (from per-contact analysis)
        residue_selection: Dict from parse_residue_selection()

    Returns:
        Dict with selection-specific ipSAE metrics:
        - n_contacts_selection: Total selected contacts
        - ipSAE_selection: Average ipSAE for selected contacts
        - ipTM_selection: Average ipTM for selected contacts
        - mean_pae_selection: Average PAE
        - mean_plddt_selection: Average pLDDT
    """
    selected = filter_contacts_by_selection(all_contacts, residue_selection)

    if not selected:
        return {
            'n_contacts_selection': 0,
            'ipSAE_selection': None,
            'ipTM_selection': None,
            'mean_pae_selection': None,
            'mean_plddt_selection': None,
        }

    # ipSAE scores
    ipsae_values = []
    for c in selected:
        ipsae = c.get('ipSAE')
        if ipsae is not None:
            ipsae_values.append(ipsae)

    # ipTM scores
    iptm_values = []
    for c in selected:
        iptm = c.get('ipTM')
        if iptm is not None:
            iptm_values.append(iptm)

    # PAE values
    pae_values = []
    for c in selected:
        pae = c.get('pae') or c.get('pae_symmetric')
        if pae is not None:
            pae_values.append(pae)

    # pLDDT values
    plddt_values = []
    for c in selected:
        plddt_i = c.get('plddt_i')
        plddt_j = c.get('plddt_j')
        if plddt_i is not None and plddt_j is not None:
            plddt_values.append((plddt_i + plddt_j) / 2.0)

    return {
        'n_contacts_selection': len(selected),
        'ipSAE_selection': sum(ipsae_values) / len(ipsae_values) if ipsae_values else None,
        'ipTM_selection': sum(iptm_values) / len(iptm_values) if iptm_values else None,
        'mean_pae_selection': sum(pae_values) / len(pae_values) if pae_values else None,
        'mean_plddt_selection': sum(plddt_values) / len(plddt_values) if plddt_values else None,
    }


def display_selection_summary(residue_selection: Dict[str, Set[int]]) -> None:
    """
    Display a summary of the residue selection to the user.

    Args:
        residue_selection: Dict from parse_residue_selection()
    """
    if not residue_selection:
        return

    total_residues = sum(len(residues) for residues in residue_selection.values())
    print(f"Residue selection enabled:")
    print(f"  Chains: {list(residue_selection.keys())}")
    print(f"  Total selected residues: {total_residues}")

    for chain, residues in sorted(residue_selection.items()):
        residue_list = sorted(residues)
        if len(residue_list) <= 10:
            print(f"  Chain {chain}: {residue_list}")
        else:
            # Truncate for display
            print(f"  Chain {chain}: {residue_list[:5]}...{residue_list[-2:]} ({len(residue_list)} total)")
