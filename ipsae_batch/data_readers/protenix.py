"""
Protenix data reader.

Reads structural data from Protenix outputs:
- Structure: CIF files ({job_name}_sample_{N}.cif)
- Full data: JSON files ({job_name}_full_data_sample_{N}.json)
  Keys: token_pair_pae, token_pair_pde, contact_probs, atom_plddt
- Summary: JSON files ({job_name}_summary_confidence_sample_{N}.json)
  Keys: chain_pair_iptm (2D list), ranking_score, iptm, ptm

Protenix format is similar to AlphaFold3 but uses:
- _sample_ (underscore) instead of IntelliFold's _sample- (dash)
- _summary_confidence_ (singular) instead of AF3's _summary_confidences_ (plural)
- token_pair_pae (not 'pae') in full_data JSON
- Nested folder structure: run_*/job_name/seed_*/predictions/
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import numpy as np

from .base import BaseReader, FoldingResult, ReaderError
from . import register_reader


@register_reader('protenix')
class ProtenixReader(BaseReader):
    """Reader for Protenix CIF structure files with JSON confidence data."""

    def __init__(self, backend_name: str = 'protenix'):
        super().__init__(backend_name)

    def find_models(self, job_folder: Path) -> List[Dict[str, Any]]:
        """
        Find Protenix model files in a job folder.

        Recursively searches for prediction directories containing Protenix output.
        Handles multiple seeds/runs by flattening into sequential model numbers,
        sorted by seed directory path then sample number.

        Args:
            job_folder: Path to folder containing Protenix output

        Returns:
            List of model info dicts with keys:
                job_name, model_num, structure_file, confidence_file, summary_file,
                seed_path, sample_num
        """
        job_folder = Path(job_folder)

        # Strip common suffixes from job name
        job_name = job_folder.name
        if job_name.endswith('_results'):
            job_name = job_name[:-8]

        # Find all prediction directories containing Protenix files
        prediction_dirs = self._find_all_prediction_dirs(job_folder)

        if not prediction_dirs:
            # Fallback: try finding a single model folder
            required_patterns = [
                ["*_sample_*.cif"],
                ["*_full_data_sample_*.json"],
            ]
            model_folder = self._find_model_folder_recursive(job_folder, required_patterns)
            if model_folder:
                prediction_dirs = [model_folder]
            else:
                return []

        models = []
        model_num = 0

        for pred_dir in sorted(prediction_dirs):
            # Find all CIF files in this directory
            cif_files = sorted(pred_dir.glob("*_sample_*.cif"))

            for cif_file in cif_files:
                # Extract sample number from filename
                match = re.search(r'_sample_(\d+)\.cif$', cif_file.name)
                if not match:
                    continue
                sample_num = int(match.group(1))

                # Derive the base name (everything before _sample_N.cif)
                base_name = cif_file.name[:match.start()]

                # Find full_data JSON (PAE source - required)
                full_data_file = pred_dir / f"{base_name}_full_data_sample_{sample_num}.json"
                if not full_data_file.exists():
                    # Try glob fallback
                    candidates = list(pred_dir.glob(f"*_full_data_sample_{sample_num}.json"))
                    full_data_file = candidates[0] if candidates else None

                if not full_data_file or not full_data_file.exists():
                    print(f"  Warning: Skipping sample {sample_num} in {pred_dir} - no full_data JSON (PAE required)")
                    continue

                # Find summary confidence JSON (optional but expected)
                summary_file = pred_dir / f"{base_name}_summary_confidence_sample_{sample_num}.json"
                if not summary_file.exists():
                    candidates = list(pred_dir.glob(f"*_summary_confidence_sample_{sample_num}.json"))
                    summary_file = candidates[0] if candidates else None

                models.append({
                    'job_name': job_name,
                    'model_num': model_num,
                    'structure_file': cif_file,
                    'confidence_file': full_data_file,
                    'summary_file': summary_file,
                    'seed_path': str(pred_dir),
                    'sample_num': sample_num,
                })
                model_num += 1

        return models

    def _find_all_prediction_dirs(self, root: Path, max_depth: int = 5) -> List[Path]:
        """
        Find all directories containing Protenix prediction files.

        Searches for directories with both *_sample_*.cif and *_full_data_sample_*.json.
        """
        prediction_dirs = []

        def search(folder: Path, depth: int):
            if depth > max_depth:
                return

            # Check if this folder has Protenix files
            has_cif = bool(list(folder.glob("*_sample_*.cif"))[:1])
            has_json = bool(list(folder.glob("*_full_data_sample_*.json"))[:1])

            if has_cif and has_json:
                prediction_dirs.append(folder)
                return  # Don't recurse into prediction dirs

            try:
                for subfolder in sorted(folder.iterdir()):
                    if subfolder.is_dir() and not subfolder.name.startswith('.'):
                        search(subfolder, depth + 1)
            except PermissionError:
                pass

        search(root, 0)
        return prediction_dirs

    def read_model(self, model_info: Dict[str, Any]) -> Optional[FoldingResult]:
        """
        Read and parse a Protenix model into a FoldingResult.

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

            # Load full_data JSON (PAE, PDE, contact_probs)
            with open(confidence_file, 'r') as f:
                conf_data = json.load(f)

            # PAE matrix - Protenix uses 'token_pair_pae' (not 'pae')
            pae_matrix = np.array(conf_data['token_pair_pae'])

            # PDE matrix (optional)
            pde_matrix = None
            if 'token_pair_pde' in conf_data:
                pde_matrix = np.array(conf_data['token_pair_pde'])

            # Contact probabilities (optional)
            contact_probs = None
            if 'contact_probs' in conf_data:
                contact_probs = np.array(conf_data['contact_probs'])

            # pLDDT from CIF B-factors (already per-residue, 0-100 scale)
            plddt = np.array([res['plddt'] for res in ca_residues])

            # Handle dimension mismatch if PAE doesn't include ligand tokens
            if pae_matrix.shape[0] != num_residues:
                is_ligand_mask = np.array([r.get('is_ligand', False) for r in ca_residues])
                polymer_indices = np.where(~is_ligand_mask)[0]

                if pae_matrix.shape[0] == len(polymer_indices):
                    # PAE only covers polymer residues - pad for ligands
                    pae_full = np.full((num_residues, num_residues), 30.0)
                    pae_full[np.ix_(polymer_indices, polymer_indices)] = pae_matrix
                    pae_matrix = pae_full

                    if pde_matrix is not None and pde_matrix.shape[0] == len(polymer_indices):
                        pde_full = np.full((num_residues, num_residues), 30.0)
                        pde_full[np.ix_(polymer_indices, polymer_indices)] = pde_matrix
                        pde_matrix = pde_full

                    if contact_probs is not None and contact_probs.shape[0] == len(polymer_indices):
                        cp_full = np.full((num_residues, num_residues), 0.0)
                        cp_full[np.ix_(polymer_indices, polymer_indices)] = contact_probs
                        contact_probs = cp_full
                else:
                    print(f"  Warning: PAE dimension ({pae_matrix.shape[0]}) != residue count ({num_residues})")

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
                ranking_score = summary_data.get('ranking_score')

                # Extract chain_pair_iptm (2D list, same format as IntelliFold)
                pair_iptm_data = summary_data.get('chain_pair_iptm', [])
                unique_chains = list(dict.fromkeys(chains))

                for idx1, chain1 in enumerate(unique_chains):
                    for idx2, chain2 in enumerate(unique_chains):
                        if chain1 != chain2:
                            try:
                                if idx1 < len(pair_iptm_data) and idx2 < len(pair_iptm_data[idx1]):
                                    chain_pair_iptm[(chain1, chain2)] = float(pair_iptm_data[idx1][idx2])
                                else:
                                    chain_pair_iptm[(chain1, chain2)] = global_iptm if global_iptm else 0.0
                            except (IndexError, TypeError):
                                chain_pair_iptm[(chain1, chain2)] = global_iptm if global_iptm else 0.0
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
                'token_asym_id': conf_data.get('token_asym_id'),
                'seed_path': model_info.get('seed_path'),
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
                contact_probs=contact_probs,
                chain_pair_iptm=chain_pair_iptm,
                global_iptm=global_iptm,
                global_ptm=global_ptm,
                ranking_score=ranking_score,
                chain_types=chain_types,
                metadata=metadata,
            )

        except Exception as e:
            print(f"Error reading Protenix model {model_info.get('model_num', '?')}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _read_cif_structure(self, cif_file: Path) -> Tuple[List[Dict], List[Dict], Dict[str, int]]:
        """
        Read Protenix CIF structure file.

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
        """Parse mmCIF ATOM/HETATM line for Protenix format."""
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
