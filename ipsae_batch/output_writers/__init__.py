"""
Output writers for ipSAE scoring results.

This module provides writers for different output formats.

Output types:
- Aggregate: One row per chain pair with summary scores
- Per-residue: One row per residue with scores against partner chains
- Per-contact: One row per contact pair with all available metrics (for visualization)
"""

from .csv_writer import (
    write_aggregate_csv,
    write_per_residue_csv,
    write_per_contact_csv,
    AGGREGATE_FIELDS,
    PER_RESIDUE_FIELDS,
    PER_CONTACT_FIELDS,
)

__all__ = [
    'write_aggregate_csv',
    'write_per_residue_csv',
    'write_per_contact_csv',
    'AGGREGATE_FIELDS',
    'PER_RESIDUE_FIELDS',
    'PER_CONTACT_FIELDS',
]
