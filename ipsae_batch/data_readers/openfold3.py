"""
OpenFold3 data reader.

Reads structural data from OpenFold3 outputs:
- Structure: CIF files ({job_name}_seed_{seed}_sample_{N}_model.cif)
- Confidence: JSON files ({job_name}_seed_{seed}_sample_{N}_confidences.json)
  Keys: pde (token-pair, symmetric), plddt (atom-level)
- Summary: JSON files ({job_name}_seed_{seed}_sample_{N}_confidences_aggregated.json)
  Keys: chain_pair_iptm (string keys like "(A, B)"), sample_ranking_score

IMPORTANT: OpenFold3 provides PDE (Predicted Distance Error), not PAE. The PDE
matrix is symmetric, which means the symmetry bonus in contact_score will always
trigger (1.25x inflation). This is acceptable because:
- Rankings between models are preserved (all equally inflated)
- PDE is a reasonable proxy for PAE (both measure structural confidence)
- The alternative (no PAE scoring) would be worse

Sample numbering starts at 1 (not 0).
CIF has pdbx_formal_charge column (value '?') — field-name parser handles this.
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import numpy as np

from .base import BaseReader, FoldingResult, ReaderError
from . import register_reader


@register_reader('openfold3')
class OpenFold3Reader(BaseReader):
    """Reader for OpenFold3 CIF structure files with JSON confidence data."""

    def __init__(self, backend_name: str = 'openfold3'):
        super().__init__(backend_name)

    def find_models(self, job_folder: Path) -> List[Dict[str, Any]]:
        """
        Find OpenFold3 model files in a job folder.

        Recursively searches for directories containing OpenFold3 output.
        Handles multiple seeds by flattening into sequential model numbers,
        sorted by seed directory path then sample number.

        Args:
            job_folder: Path to folder containing OpenFold3 output

        Returns:
            List of model info dicts
        """
        job_folder = Path(job_folder)

        # Strip common suffixes from job name
        job_name = job_folder.name
        if job_name.endswith('_results'):
            job_name = job_name[:-8]

        # Find all seed directories containing OpenFold3 files
        seed_dirs = self._find_all_seed_dirs(job_folder)

        if not seed_dirs:
            # Fallback: try finding a single model folder
            required_patterns = [
                ["*_model.cif"],
                ["*_confidences.json"],
            ]
            model_folder = self._find_model_folder_recursive(job_folder, required_patterns)
            if model_folder:
                seed_dirs = [model_folder]
            else:
                return []

        models = []
        model_num = 0

        for seed_dir in sorted(seed_dirs):
            # Find all model CIF files in this directory
            cif_files = sorted(seed_dir.glob("*_model.cif"))

            for cif_file in cif_files:
                # Extract seed and sample numbers from filename
                match = re.search(r'_seed_(\d+)_sample_(\d+)_model\.cif$', cif_file.name)
                if not match:
                    # Try simpler pattern without seed
                    match = re.search(r'_sample_(\d+)_model\.cif$', cif_file.name)
                    if not match:
                        continue
                    sample_num = int(match.group(1))
                    seed_num = None
                    base_name = cif_file.name[:match.start()]
                else:
                    seed_num = int(match.group(1))
                    sample_num = int(match.group(2))
                    base_name = cif_file.name[:match.start()]

                # Build file name prefix for this sample
                if seed_num is not None:
                    prefix = f"{base_name}_seed_{seed_num}_sample_{sample_num}"
                else:
                    prefix = f"{base_name}_sample_{sample_num}"

                # Find confidences JSON (PDE source - required)
                conf_file = seed_dir / f"{prefix}_confidences.json"
                if not conf_file.exists():
                    candidates = list(seed_dir.glob(f"*_sample_{sample_num}_confidences.json"))
                    conf_file = candidates[0] if candidates else None

                if not conf_file or not conf_file.exists():
                    print(f"  Warning: Skipping sample {sample_num} in {seed_dir} - no confidences JSON (PDE required)")
                    continue

                # Find aggregated confidences JSON (optional but expected)
                agg_file = seed_dir / f"{prefix}_confidences_aggregated.json"
                if not agg_file.exists():
                    candidates = list(seed_dir.glob(f"*_sample_{sample_num}_confidences_aggregated.json"))
                    agg_file = candidates[0] if candidates else None

                models.append({
                    'job_name': job_name,
                    'model_num': model_num,
                    'structure_file': cif_file,
                    'confidence_file': conf_file,
                    'summary_file': agg_file,
                    'seed_num': seed_num,
                    'sample_num': sample_num,
                })
                model_num += 1

        return models

    def _find_all_seed_dirs(self, root: Path, max_depth: int = 5) -> List[Path]:
        """
        Find all directories containing OpenFold3 model files.

        Searches for directories with both *_model.cif and *_confidences.json.
        """
        seed_dirs = []

        def search(folder: Path, depth: int):
            if depth > max_depth:
                return

            has_cif = bool(list(folder.glob("*_model.cif"))[:1])
            has_json = bool(list(folder.glob("*_confidences.json"))[:1])

            if has_cif and has_json:
                seed_dirs.append(folder)
                return

            try:
                for subfolder in sorted(folder.iterdir()):
                    if subfolder.is_dir() and not subfolder.name.startswith('.'):
                        search(subfolder, depth + 1)
            except PermissionError:
                pass

        search(root, 0)
        return seed_dirs

    def read_model(self, model_info: Dict[str, Any]) -> Optional[FoldingResult]:
        """
        Read and parse an OpenFold3 model into a FoldingResult.

        Args:
            model_info: Dict from find_models() containing file paths

        Returns:
            FoldingResult with all standardized data, or None if parsing fails
        """
        try:
            job_name = model_info['job_name']
            model_num = model_info['model_num']
            structure_file = Path(model_info['structure_file'])
            confidence_file = Path(model_info['confidence_file'])
            summary_file = model_info.get('summary_file')

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

            # Load confidence data from JSON
            with open(confidence_file, 'r') as f:
                conf_data = json.load(f)

            # PDE matrix (symmetric) — used as BOTH pae_matrix and pde_matrix
            pde_raw = np.array(conf_data['pde'])
            pae_matrix = pde_raw  # PDE serves as PAE proxy
            pde_matrix = pde_raw.copy()

            # pLDDT from CIF B-factors (already per-residue, 0-100 scale)
            plddt = np.array([res['plddt'] for res in ca_residues])

            # Handle dimension mismatch between PDE tokens and CIF residues.
            # OpenFold3 PDE is per-token: each protein residue = 1 token, each
            # ligand ATOM = 1 token. But the CIF parser collapses each ligand
            # into 1 representative entry. So PDE can be larger than num_residues
            # when ligands are present (e.g., ADP: 27 atoms = 27 tokens, 1 CIF entry).
            if pae_matrix.shape[0] != num_residues:
                is_ligand_mask = np.array([r.get('is_ligand', False) for r in ca_residues])
                polymer_indices = np.where(~is_ligand_mask)[0]
                n_polymer = len(polymer_indices)
                n_ligand_entries = int(np.sum(is_ligand_mask))

                if pae_matrix.shape[0] == n_polymer:
                    # PDE only covers polymer residues - pad for ligands
                    pae_full = np.full((num_residues, num_residues), 30.0)
                    pae_full[np.ix_(polymer_indices, polymer_indices)] = pae_matrix
                    pae_matrix = pae_full

                    pde_full = np.full((num_residues, num_residues), 30.0)
                    pde_full[np.ix_(polymer_indices, polymer_indices)] = pde_matrix
                    pde_matrix = pde_full

                elif pae_matrix.shape[0] > num_residues and n_ligand_entries > 0:
                    # PDE has more tokens than CIF residues — ligand atoms are
                    # expanded (1 token per atom) but CIF collapses them to 1 entry.
                    # Token order: polymer tokens first, then ligand atom tokens.
                    n_tokens = pae_matrix.shape[0]
                    keep = np.ones(n_tokens, dtype=bool)

                    if n_ligand_entries == 1:
                        # Single ligand: keep first token after polymer, discard rest
                        keep[n_polymer + 1:] = False
                    else:
                        # Multiple ligand entries: count atoms per ligand from CIF
                        ligand_atom_counts = self._count_ligand_atoms(structure_file)
                        offset = n_polymer
                        for count in ligand_atom_counts:
                            keep[offset + 1:offset + count] = False
                            offset += count

                    if np.sum(keep) == num_residues:
                        pae_matrix = pae_matrix[np.ix_(keep, keep)]
                        pde_matrix = pde_matrix[np.ix_(keep, keep)]
                    else:
                        print(f"  Warning: PDE token trim failed: expected {num_residues}, got {np.sum(keep)}")

                else:
                    print(f"  Warning: PDE dimension ({pde_raw.shape[0]}) != residue count ({num_residues})")

            # Load summary confidence data if available
            global_iptm = None
            global_ptm = None
            chain_pair_iptm = {}
            ranking_score = None

            if summary_file and Path(summary_file).exists():
                with open(summary_file, 'r') as f:
                    summary_data = json.load(f)

                global_iptm = summary_data.get('iptm')
                global_ptm = summary_data.get('ptm')
                ranking_score = summary_data.get('sample_ranking_score')

                # Parse chain_pair_iptm from string keys like "(A, B)"
                pair_iptm_raw = summary_data.get('chain_pair_iptm', {})
                for key_str, value in pair_iptm_raw.items():
                    match = re.match(r'\((\w+),\s*(\w+)\)', key_str)
                    if match:
                        c1, c2 = match.group(1), match.group(2)
                        chain_pair_iptm[(c1, c2)] = float(value)
                        # Also set reverse direction if not present
                        if (c2, c1) not in pair_iptm_raw:
                            chain_pair_iptm[(c2, c1)] = float(value)

                # Fill in any missing chain pairs
                unique_chains = list(dict.fromkeys(chains))
                for c1 in unique_chains:
                    for c2 in unique_chains:
                        if c1 != c2 and (c1, c2) not in chain_pair_iptm:
                            chain_pair_iptm[(c1, c2)] = global_iptm if global_iptm else 0.0
            else:
                unique_chains = list(dict.fromkeys(chains))
                for chain1 in unique_chains:
                    for chain2 in unique_chains:
                        if chain1 != chain2:
                            chain_pair_iptm[(chain1, chain2)] = 0.0

            # CB pLDDT from B-factors
            cb_plddt = np.array([res['plddt'] for res in cb_residues])

            # Classify chains
            chain_types = self.classify_chains(chains, residue_types)

            # Metadata
            metadata = {
                'cb_plddt': cb_plddt,
                'chain_index_map': chain_index_map,
                'pae_is_symmetric': True,  # PDE is symmetric — symmetry bonus always triggers
                'seed_num': model_info.get('seed_num'),
                'sample_num': model_info.get('sample_num'),
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
                pde_matrix=pde_matrix,
                chain_pair_iptm=chain_pair_iptm,
                global_iptm=global_iptm,
                global_ptm=global_ptm,
                ranking_score=ranking_score,
                chain_types=chain_types,
                metadata=metadata,
            )

        except Exception as e:
            print(f"Error reading OpenFold3 model {model_info.get('model_num', '?')}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _read_cif_structure(self, cif_file: Path) -> Tuple[List[Dict], List[Dict], Dict[str, int]]:
        """
        Read OpenFold3 CIF structure file.

        Note: OpenFold3 CIF includes pdbx_formal_charge column (value '?').
        The field-name parser handles this correctly.

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
        """Parse mmCIF ATOM/HETATM line for OpenFold3 format."""
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
        """Count ligand atoms per ligand entry in CIF, in chain order.

        OpenFold3 CIF does not include hydrogen atoms, so all HETATM atoms
        are counted (unlike Chai-1 which filters H atoms).

        Returns a list with one count per unique (chain_id, residue_name)
        ligand group, preserving the order they appear in the CIF file.
        This matches the token order in the PDE matrix.
        """
        atomsitefield_dict = {}
        atomsitefield_num = 0
        ligand_groups = []  # ordered list of (chain_id, residue_name)
        ligand_counts = {}  # (chain_id, residue_name) -> atom count

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
