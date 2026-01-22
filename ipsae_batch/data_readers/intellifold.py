"""
IntelliFold data reader.

Reads structural data from IntelliFold outputs:
- Structure: CIF files ({job_name}_seed-42_sample-N.cif)
- Confidence: JSON files ({job_name}_seed-42_sample-N_confidences.json)
- Summary: JSON files ({job_name}_seed-42_sample-N_summary_confidences.json)

IntelliFold format is similar to AlphaFold3 but uses sample-N numbering
and has a deeply nested folder structure.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import numpy as np

from .base import BaseReader, FoldingResult, ReaderError
from . import register_reader


@register_reader('intellifold')
class IntelliFoldReader(BaseReader):
    """Reader for IntelliFold CIF structure files with JSON confidence data."""

    def __init__(self, backend_name: str = 'intellifold'):
        super().__init__(backend_name)

    def find_models(self, job_folder: Path) -> List[Dict[str, Any]]:
        """
        Find IntelliFold model files in a job folder.

        Recursively searches for folders containing IntelliFold output files.

        Args:
            job_folder: Path to folder containing IntelliFold output (typically *_results)

        Returns:
            List of model info dicts with keys:
                job_name, model_num, structure_file, confidence_file, summary_file
        """
        job_folder = Path(job_folder)

        # IntelliFold folder typically ends with _results
        job_name = job_folder.name
        if job_name.endswith('_results'):
            job_name = job_name[:-8]

        # Define required file patterns for IntelliFold (at least one model)
        required_patterns = [
            ["*_seed-*_sample-*.cif", "*_sample-*.cif"],  # Structure file
            ["*_seed-*_sample-*_confidences.json", "*_sample-*_confidences.json"],  # Confidence file
        ]

        # Find the folder containing model files (recursive search)
        model_folder = self._find_model_folder_recursive(job_folder, required_patterns)
        if model_folder is None:
            return []

        models = []

        # IntelliFold produces samples 0-4 with seed-42
        for model_num in range(5):
            # Find CIF structure file
            cif_patterns = [
                f"{job_name}_seed-42_sample-{model_num}.cif",
                f"*_seed-*_sample-{model_num}.cif",
            ]
            cif_file = self._find_file_by_patterns(model_folder, cif_patterns)

            # Find confidences JSON (PAE, pLDDT per atom/token)
            conf_patterns = [
                f"{job_name}_seed-42_sample-{model_num}_confidences.json",
                f"*_seed-*_sample-{model_num}_confidences.json",
            ]
            conf_file = self._find_file_by_patterns(model_folder, conf_patterns)

            # Find summary confidences JSON (chain_pair_iptm, ranking_score)
            summary_patterns = [
                f"{job_name}_seed-42_sample-{model_num}_summary_confidences.json",
                f"*_seed-*_sample-{model_num}_summary_confidences.json",
            ]
            summary_file = self._find_file_by_patterns(model_folder, summary_patterns)

            if cif_file and conf_file:
                models.append({
                    'job_name': job_name,
                    'model_num': model_num,
                    'structure_file': cif_file,
                    'confidence_file': conf_file,
                    'summary_file': summary_file,  # May be None
                })

        return models

    def read_model(self, model_info: Dict[str, Any]) -> Optional[FoldingResult]:
        """
        Read and parse an IntelliFold model into a FoldingResult.

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

            # Use object dtype for string arrays
            chains = np.array([str(res['chainid']) for res in ca_residues], dtype=object)
            residue_types = np.array([str(res['res']) for res in ca_residues], dtype=object)
            residue_numbers = np.array([res['resnum'] for res in ca_residues])

            # Load confidence data from JSON
            with open(confidence_file, 'r') as f:
                conf_data = json.load(f)

            # PAE matrix (already in token/residue space)
            pae_matrix = np.array(conf_data['pae'])

            # pLDDT - use from CIF B-factors (already per-residue, 0-100 scale)
            plddt_from_cif = np.array([res['plddt'] for res in ca_residues])
            plddt = plddt_from_cif

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

                # Extract chain_pair_iptm (2D array)
                pair_iptm_data = summary_data.get('chain_pair_iptm', [])
                unique_chains = list(dict.fromkeys(chains))

                # Map numeric indices to chain names
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
                # No summary file - use default values
                unique_chains = list(dict.fromkeys(chains))
                for chain1 in unique_chains:
                    for chain2 in unique_chains:
                        if chain1 != chain2:
                            chain_pair_iptm[(chain1, chain2)] = 0.0

            # CB pLDDT - extract from structure B-factors
            cb_plddt = np.array([res['plddt'] for res in cb_residues])

            # Classify chains
            chain_types = self.classify_chains(chains, residue_types)

            # Metadata
            metadata = {
                'cb_plddt': cb_plddt,
                'chain_index_map': chain_index_map,
                'token_chain_ids': conf_data.get('token_chain_ids'),
                'token_res_ids': conf_data.get('token_res_ids'),
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

        except Exception as e:
            print(f"Error reading IntelliFold model {model_info.get('model_num', '?')}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _read_cif_structure(self, cif_file: Path) -> Tuple[List[Dict], List[Dict], Dict[str, int]]:
        """
        Read IntelliFold CIF structure file.

        Returns:
            Tuple of (ca_residues, cb_residues, chain_index_map)
            - ca_residues: List of dicts with CA/C1' atom info
            - cb_residues: List of dicts with CB/C3'/CA(GLY) atom info
            - chain_index_map: Dict mapping chain name -> index (0, 1, 2, ...)
        """
        ca_residues = []
        cb_residues = []

        atomsitefield_dict = {}
        atomsitefield_num = 0

        # Track unique chains in order
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

                    # Track chain order
                    if atom['chain_id'] not in chain_order:
                        chain_order.append(atom['chain_id'])

                    # CA atoms (or C1' for nucleic acids)
                    if atom['atom_name'] == "CA" or "C1" in atom['atom_name']:
                        ca_residues.append({
                            'coor': np.array([atom['x'], atom['y'], atom['z']]),
                            'res': atom['residue_name'],
                            'chainid': atom['chain_id'],
                            'resnum': atom['residue_seq_num'],
                            'plddt': atom.get('b_factor', 0.0),
                        })

                    # CB atoms (or C3' for nucleic acids, or CA for GLY)
                    if (atom['atom_name'] == "CB" or "C3" in atom['atom_name'] or
                        (atom['residue_name'] == "GLY" and atom['atom_name'] == "CA")):
                        cb_residues.append({
                            'coor': np.array([atom['x'], atom['y'], atom['z']]),
                            'res': atom['residue_name'],
                            'chainid': atom['chain_id'],
                            'resnum': atom['residue_seq_num'],
                            'plddt': atom.get('b_factor', 0.0),
                        })

        # Build chain index map
        chain_index_map = {chain: idx for idx, chain in enumerate(chain_order)}

        return ca_residues, cb_residues, chain_index_map

    def _parse_cif_atom_line(self, line: str, fielddict: Dict[str, int]) -> Optional[Dict]:
        """Parse mmCIF ATOM/HETATM line for IntelliFold format."""
        linelist = line.split()
        # Only check for required fields (up to B_iso_or_equiv, index 17)
        required_fields = ['label_seq_id', 'label_atom_id', 'label_comp_id',
                          'label_asym_id', 'Cartn_x', 'Cartn_y', 'Cartn_z', 'B_iso_or_equiv']
        max_required = max(fielddict.get(f, 0) for f in required_fields if f in fielddict)
        if len(linelist) <= max_required:
            return None

        try:
            residue_seq_num = linelist[fielddict['label_seq_id']]
            if residue_seq_num == ".":
                return None  # Ligand atom

            # Get chain ID
            chain_id = linelist[fielddict['label_asym_id']]

            return {
                'atom_num': int(linelist[fielddict['id']]),
                'atom_name': linelist[fielddict['label_atom_id']],
                'residue_name': linelist[fielddict['label_comp_id']],
                'chain_id': chain_id,
                'residue_seq_num': int(residue_seq_num),
                'x': float(linelist[fielddict['Cartn_x']]),
                'y': float(linelist[fielddict['Cartn_y']]),
                'z': float(linelist[fielddict['Cartn_z']]),
                'b_factor': float(linelist[fielddict['B_iso_or_equiv']]) if 'B_iso_or_equiv' in fielddict else 0.0,
            }
        except (ValueError, IndexError, KeyError):
            return None
