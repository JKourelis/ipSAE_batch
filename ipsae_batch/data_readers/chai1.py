"""
Chai-1 data reader.

Reads structural data from Chai-1 outputs:
- Structure: CIF files (pred.model_idx_{N}.cif)
- Scores: NPZ files (scores.model_idx_{N}.npz)
  Keys: ptm(1,), iptm(1,), per_chain_pair_iptm(1,nchains,nchains), aggregate_score(1,)
- PAE: NPZ files (pae.model_idx_{N}.npz) — REQUIRED but not yet available

PAE files are required for all-atom contact scoring. Without PAE, the reader
will detect Chai-1 files but fail with a clear error message instructing the
user to provide PAE data.

Chai-1 CIF has 'id' field at position 2 (not near end like other backends),
but the field-name parser handles this correctly.
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import numpy as np

from .base import BaseReader, FoldingResult, ReaderError
from . import register_reader


@register_reader('chai1')
class Chai1Reader(BaseReader):
    """Reader for Chai-1 CIF structure files with NPZ confidence data."""

    def __init__(self, backend_name: str = 'chai1'):
        super().__init__(backend_name)

    def find_models(self, job_folder: Path) -> List[Dict[str, Any]]:
        """
        Find Chai-1 model files in a job folder.

        Searches for pred.model_idx_*.cif files. Requires corresponding
        scores.model_idx_*.npz and pae.model_idx_*.npz files.

        Args:
            job_folder: Path to folder containing Chai-1 output

        Returns:
            List of model info dicts
        """
        job_folder = Path(job_folder)

        # Strip common suffixes from job name
        job_name = job_folder.name
        if job_name.endswith('_results'):
            job_name = job_name[:-8]

        # Find the folder containing Chai-1 files (may be in output/ subfolder)
        required_patterns = [
            ["pred.model_idx_*.cif"],
            ["scores.model_idx_*.npz"],
        ]
        model_folder = self._find_model_folder_recursive(job_folder, required_patterns)

        if model_folder is None:
            return []

        models = []

        # Find all CIF files and extract model indices dynamically
        cif_files = sorted(model_folder.glob("pred.model_idx_*.cif"))

        for cif_file in cif_files:
            match = re.search(r'pred\.model_idx_(\d+)\.cif$', cif_file.name)
            if not match:
                continue
            model_idx = int(match.group(1))

            # Require scores NPZ
            scores_file = model_folder / f"scores.model_idx_{model_idx}.npz"
            if not scores_file.exists():
                print(f"  Warning: Skipping model {model_idx} - no scores NPZ")
                continue

            # Require PAE NPZ (may not exist yet)
            pae_file = model_folder / f"pae.model_idx_{model_idx}.npz"

            models.append({
                'job_name': job_name,
                'model_num': model_idx,
                'structure_file': cif_file,
                'scores_file': scores_file,
                'pae_file': pae_file,
                'model_folder': str(model_folder),
            })

        return models

    def read_model(self, model_info: Dict[str, Any]) -> Optional[FoldingResult]:
        """
        Read and parse a Chai-1 model into a FoldingResult.

        Args:
            model_info: Dict from find_models() containing file paths

        Returns:
            FoldingResult with all standardized data, or None if parsing fails

        Raises:
            ReaderError: If PAE file is missing (required for scoring)
        """
        try:
            job_name = model_info['job_name']
            model_num = model_info['model_num']
            structure_file = Path(model_info['structure_file'])
            scores_file = Path(model_info['scores_file'])
            pae_file = Path(model_info['pae_file'])

            # Read structure from CIF
            ca_residues, cb_residues, chain_index_map = self._read_cif_structure(structure_file)

            if not ca_residues:
                return None

            num_residues = len(ca_residues)

            # Convert to numpy arrays
            ca_coordinates = np.array([res['coor'] for res in ca_residues])
            cb_coordinates = np.array([res['coor'] for res in cb_residues])

            chains = np.array([str(res['chainid']) for res in ca_residues], dtype=object)
            residue_types = np.array([str(res['res']) for res in ca_residues], dtype=object)
            residue_numbers = np.array([res['resnum'] for res in ca_residues])

            # Load PAE matrix (REQUIRED)
            if not pae_file.exists():
                raise ReaderError(
                    f"Chai-1 PAE file not found: {pae_file}\n"
                    f"  PAE data is required for contact scoring.\n"
                    f"  Expected file: pae.model_idx_{model_num}.npz\n"
                    f"  Chai-1 does not export PAE by default. You may need to:\n"
                    f"  1. Re-run Chai-1 with PAE export enabled, or\n"
                    f"  2. Generate PAE using chai-lab's Python API with output_pae=True"
                )

            pae_data = np.load(pae_file)
            # Try common key names
            if 'pae' in pae_data:
                pae_matrix = pae_data['pae']
            elif 'predicted_aligned_error' in pae_data:
                pae_matrix = pae_data['predicted_aligned_error']
            else:
                available_keys = list(pae_data.files)
                raise ReaderError(
                    f"Chai-1 PAE NPZ has unexpected keys: {available_keys}\n"
                    f"  Expected 'pae' or 'predicted_aligned_error'"
                )

            # Squeeze batch dimension if present
            if pae_matrix.ndim == 3 and pae_matrix.shape[0] == 1:
                pae_matrix = pae_matrix.squeeze(0)

            pae_matrix = pae_matrix.astype(np.float64)

            # pLDDT from CIF B-factors (already per-residue, 0-100 scale)
            plddt = np.array([res['plddt'] for res in ca_residues])

            # Handle dimension mismatch between PAE tokens and CIF residues.
            # Chai-1 PAE is per-token: each protein residue = 1 token, each
            # ligand ATOM = 1 token. But the CIF parser collapses each ligand
            # into 1 representative entry. So PAE can be larger than num_residues
            # when ligands are present (e.g., ADP: 27 non-H atoms = 27 tokens,
            # 1 CIF entry).
            #
            # Strategy: polymer tokens are first (1:1 with polymer residues),
            # ligand atom tokens follow. Keep one token per ligand CIF entry
            # (the first), discard the rest.
            if pae_matrix.shape[0] != num_residues:
                is_ligand_mask = np.array([r.get('is_ligand', False) for r in ca_residues])
                polymer_indices = np.where(~is_ligand_mask)[0]
                n_polymer = len(polymer_indices)
                n_ligand_entries = int(np.sum(is_ligand_mask))

                if pae_matrix.shape[0] == n_polymer:
                    # PAE only covers polymer residues - pad for ligands
                    pae_full = np.full((num_residues, num_residues), 30.0)
                    pae_full[np.ix_(polymer_indices, polymer_indices)] = pae_matrix
                    pae_matrix = pae_full

                elif pae_matrix.shape[0] > num_residues and n_ligand_entries > 0:
                    # PAE has more tokens than CIF residues — ligand atoms are
                    # expanded (1 token per atom) but CIF collapses them to 1 entry.
                    # Token order: polymer tokens first, then ligand atom tokens.
                    n_tokens = pae_matrix.shape[0]
                    n_ligand_tokens = n_tokens - n_polymer
                    # Build keep mask: keep all polymer tokens, then keep only
                    # the first token for each ligand CIF entry.
                    keep = np.ones(n_tokens, dtype=bool)

                    if n_ligand_entries == 1:
                        # Single ligand: keep first token, discard rest
                        keep[n_polymer + 1:] = False
                    else:
                        # Multiple ligand entries: count non-H atoms per ligand
                        # from CIF to determine token block sizes.
                        ligand_atom_counts = self._count_ligand_atoms(
                            Path(model_info['structure_file']))
                        offset = n_polymer
                        for count in ligand_atom_counts:
                            # Keep first token of this ligand group
                            keep[offset + 1:offset + count] = False
                            offset += count

                    if np.sum(keep) == num_residues:
                        pae_matrix = pae_matrix[np.ix_(keep, keep)]
                    else:
                        print(f"  Warning: PAE token trim failed: expected {num_residues}, got {np.sum(keep)}")

                else:
                    print(f"  Warning: PAE dimension ({pae_matrix.shape[0]}) != residue count ({num_residues})")

            # Load scores NPZ
            scores_data = np.load(scores_file)

            global_iptm = float(scores_data['iptm'].squeeze()) if 'iptm' in scores_data else None
            global_ptm = float(scores_data['ptm'].squeeze()) if 'ptm' in scores_data else None
            ranking_score = float(scores_data['aggregate_score'].squeeze()) if 'aggregate_score' in scores_data else None

            # Extract chain_pair_iptm from per_chain_pair_iptm array
            chain_pair_iptm = {}
            unique_chains = list(dict.fromkeys(chains))

            if 'per_chain_pair_iptm' in scores_data:
                pair_iptm_array = scores_data['per_chain_pair_iptm'].squeeze(0)  # Remove batch dim → (nchains, nchains)

                for idx1, chain1 in enumerate(unique_chains):
                    for idx2, chain2 in enumerate(unique_chains):
                        if chain1 != chain2:
                            if idx1 < pair_iptm_array.shape[0] and idx2 < pair_iptm_array.shape[1]:
                                chain_pair_iptm[(chain1, chain2)] = float(pair_iptm_array[idx1, idx2])
                            else:
                                chain_pair_iptm[(chain1, chain2)] = global_iptm if global_iptm else 0.0
            else:
                for chain1 in unique_chains:
                    for chain2 in unique_chains:
                        if chain1 != chain2:
                            chain_pair_iptm[(chain1, chain2)] = global_iptm if global_iptm else 0.0

            # CB pLDDT from B-factors
            cb_plddt = np.array([res['plddt'] for res in cb_residues])

            # Classify chains
            chain_types = self.classify_chains(chains, residue_types)

            # Metadata
            metadata = {
                'cb_plddt': cb_plddt,
                'chain_index_map': chain_index_map,
            }

            return FoldingResult(
                job_name=job_name,
                model_num=model_num,
                structure_path=structure_file,
                ca_coordinates=ca_coordinates,
                cb_coordinates=cb_coordinates,
                chains=chains,
                residue_types=residue_types,
                residue_numbers=residue_numbers,
                plddt=plddt,
                pae_matrix=pae_matrix,
                chain_pair_iptm=chain_pair_iptm,
                global_iptm=global_iptm,
                global_ptm=global_ptm,
                ranking_score=ranking_score,
                chain_types=chain_types,
                metadata=metadata,
            )

        except ReaderError:
            raise  # Let ReaderError propagate with its clear message
        except Exception as e:
            print(f"Error reading Chai-1 model {model_info.get('model_num', '?')}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _read_cif_structure(self, cif_file: Path) -> Tuple[List[Dict], List[Dict], Dict[str, int]]:
        """
        Read Chai-1 CIF structure file.

        Returns:
            Tuple of (ca_residues, cb_residues, chain_index_map)
        """
        ca_residues = []
        cb_residues = []
        seen_ligand_residues = {}

        atomsitefield_dict = {}
        atomsitefield_num = 0

        chain_order = []

        with open(cif_file, 'r') as f:
            for line in f:
                if line.startswith("_atom_site."):
                    line = line.strip()
                    _, fieldname = line.split(".")
                    atomsitefield_dict[fieldname] = atomsitefield_num
                    atomsitefield_num += 1

                elif line.startswith("ATOM") or line.startswith("HETATM"):
                    atom = self._parse_cif_atom_line(line, atomsitefield_dict)
                    if atom is None:
                        continue

                    if atom['chain_id'] not in chain_order:
                        chain_order.append(atom['chain_id'])

                    if atom.get('is_ligand', False):
                        lig_key = (atom['chain_id'], atom['residue_name'])
                        if lig_key not in seen_ligand_residues:
                            if atom['atom_name'][0] != 'H':
                                seen_ligand_residues[lig_key] = True
                                entry = {
                                    'coor': np.array([atom['x'], atom['y'], atom['z']]),
                                    'res': atom['residue_name'],
                                    'chainid': atom['chain_id'],
                                    'resnum': atom['residue_seq_num'],
                                    'plddt': atom.get('b_factor', 0.0),
                                    'is_ligand': True,
                                }
                                ca_residues.append(entry)
                                cb_residues.append(entry)
                        continue

                    if atom['atom_name'] == "CA" or "C1" in atom['atom_name']:
                        ca_residues.append({
                            'coor': np.array([atom['x'], atom['y'], atom['z']]),
                            'res': atom['residue_name'],
                            'chainid': atom['chain_id'],
                            'resnum': atom['residue_seq_num'],
                            'plddt': atom.get('b_factor', 0.0),
                        })

                    if (atom['atom_name'] == "CB" or "C3" in atom['atom_name'] or
                        (atom['residue_name'] == "GLY" and atom['atom_name'] == "CA")):
                        cb_residues.append({
                            'coor': np.array([atom['x'], atom['y'], atom['z']]),
                            'res': atom['residue_name'],
                            'chainid': atom['chain_id'],
                            'resnum': atom['residue_seq_num'],
                            'plddt': atom.get('b_factor', 0.0),
                        })

        chain_index_map = {chain: idx for idx, chain in enumerate(chain_order)}

        return ca_residues, cb_residues, chain_index_map

    def _parse_cif_atom_line(self, line: str, fielddict: Dict[str, int]) -> Optional[Dict]:
        """Parse mmCIF ATOM/HETATM line for Chai-1 format."""
        linelist = line.split()
        required_fields = ['label_seq_id', 'label_atom_id', 'label_comp_id',
                          'label_asym_id', 'Cartn_x', 'Cartn_y', 'Cartn_z', 'B_iso_or_equiv']
        max_required = max(fielddict.get(f, 0) for f in required_fields if f in fielddict)
        if len(linelist) <= max_required:
            return None

        try:
            residue_seq_num_str = linelist[fielddict['label_seq_id']]
            is_ligand = residue_seq_num_str == "."

            if is_ligand:
                if 'auth_seq_id' in fielddict:
                    auth_seq_str = linelist[fielddict['auth_seq_id']]
                    residue_seq_num = int(auth_seq_str) if auth_seq_str != "." else 1
                else:
                    residue_seq_num = 1
            else:
                residue_seq_num = int(residue_seq_num_str)

            chain_id = linelist[fielddict['label_asym_id']]

            return {
                'atom_num': int(linelist[fielddict['id']]),
                'atom_name': linelist[fielddict['label_atom_id']],
                'residue_name': linelist[fielddict['label_comp_id']],
                'chain_id': chain_id,
                'residue_seq_num': residue_seq_num,
                'x': float(linelist[fielddict['Cartn_x']]),
                'y': float(linelist[fielddict['Cartn_y']]),
                'z': float(linelist[fielddict['Cartn_z']]),
                'b_factor': float(linelist[fielddict['B_iso_or_equiv']]) if 'B_iso_or_equiv' in fielddict else 0.0,
                'is_ligand': is_ligand,
            }
        except (ValueError, IndexError, KeyError):
            return None

    def _count_ligand_atoms(self, cif_file: Path) -> List[int]:
        """Count non-H atoms per ligand entry in CIF, in chain order.

        Returns a list with one count per unique (chain_id, residue_name)
        ligand group, preserving the order they appear in the CIF file.
        This matches the token order in the PAE matrix.
        """
        atomsitefield_dict = {}
        atomsitefield_num = 0
        ligand_groups = []  # ordered list of (chain_id, residue_name)
        ligand_counts = {}  # (chain_id, residue_name) -> non-H atom count

        with open(cif_file, 'r') as f:
            for line in f:
                if line.startswith("_atom_site."):
                    _, fieldname = line.strip().split(".")
                    atomsitefield_dict[fieldname] = atomsitefield_num
                    atomsitefield_num += 1
                elif line.startswith("ATOM") or line.startswith("HETATM"):
                    linelist = line.split()
                    try:
                        seq_id = linelist[atomsitefield_dict['label_seq_id']]
                        if seq_id != ".":
                            continue
                        atom_name = linelist[atomsitefield_dict['label_atom_id']]
                        if atom_name[0] == 'H':
                            continue
                        chain_id = linelist[atomsitefield_dict['label_asym_id']]
                        comp_id = linelist[atomsitefield_dict['label_comp_id']]
                        key = (chain_id, comp_id)
                        if key not in ligand_counts:
                            ligand_groups.append(key)
                            ligand_counts[key] = 0
                        ligand_counts[key] += 1
                    except (KeyError, IndexError):
                        continue

        return [ligand_counts[g] for g in ligand_groups]
