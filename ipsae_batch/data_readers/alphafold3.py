"""
AlphaFold3 data reader.

Reads structural data from AlphaFold3 outputs:
- Structure: CIF files (*_model_N.cif)
- Confidence: JSON files (*_full_data_N.json) containing PAE, pLDDT, contact_probs
- Summary: JSON files (*_summary_confidences_N.json) containing chain_pair_iptm
"""

import glob
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import numpy as np

from .base import BaseReader, FoldingResult, ReaderError
from . import register_reader


@register_reader('alphafold3')
class AlphaFold3Reader(BaseReader):
    """Reader for AlphaFold3 CIF structure files with JSON confidence data."""

    def __init__(self, backend_name: str = 'alphafold3'):
        super().__init__(backend_name)

    def find_models(self, job_folder: Path) -> List[Dict[str, Any]]:
        """
        Find AlphaFold3 model files in a job folder.

        Recursively searches for folders containing AF3 output files.

        Args:
            job_folder: Path to folder containing AF3 output files

        Returns:
            List of model info dicts with keys:
                job_name, model_num, structure_file, confidence_file, summary_file
        """
        job_folder = Path(job_folder)
        job_name = job_folder.name

        # Define required file patterns for AF3 (at least one model)
        required_patterns = [
            ["*_model_*.cif", "fold_*_model_*.cif"],  # Structure file
            ["*_full_data_*.json", "fold_*_full_data_*.json"],  # Confidence file
            ["*_summary_confidences_*.json", "fold_*_summary_confidences_*.json"],  # Summary file
        ]

        # Find the folder containing model files (recursive search)
        model_folder = self._find_model_folder_recursive(job_folder, required_patterns)
        if model_folder is None:
            return []

        models = []

        # AF3 typically produces models 0-4
        for model_num in range(5):
            # Find CIF file
            cif_patterns = [
                f"*_model_{model_num}.cif",
                f"fold_*_model_{model_num}.cif",
            ]
            cif_file = self._find_file_by_patterns(model_folder, cif_patterns)

            # Find full_data JSON (confidence data)
            conf_patterns = [
                f"*_full_data_{model_num}.json",
                f"fold_*_full_data_{model_num}.json",
            ]
            conf_file = self._find_file_by_patterns(model_folder, conf_patterns)

            # Find summary_confidences JSON
            summary_patterns = [
                f"*_summary_confidences_{model_num}.json",
                f"fold_*_summary_confidences_{model_num}.json",
            ]
            summary_file = self._find_file_by_patterns(model_folder, summary_patterns)

            if cif_file and conf_file and summary_file:
                models.append({
                    'job_name': job_name,
                    'model_num': model_num,
                    'structure_file': cif_file,
                    'confidence_file': conf_file,
                    'summary_file': summary_file,
                })

        return models

    def read_model(self, model_info: Dict[str, Any]) -> Optional[FoldingResult]:
        """
        Read and parse an AF3 model into a FoldingResult.

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
            summary_file = Path(model_info['summary_file'])

            # Read structure
            ca_residues, cb_residues, token_mask = self._read_structure(structure_file)

            if not ca_residues:
                return None

            num_residues = len(ca_residues)

            # Convert to numpy arrays
            ca_atom_nums = np.array([res['atom_num'] - 1 for res in ca_residues])
            cb_atom_nums = np.array([res['atom_num'] - 1 for res in cb_residues])

            ca_coordinates = np.array([res['coor'] for res in ca_residues])
            cb_coordinates = np.array([res['coor'] for res in cb_residues])

            # Use object dtype to store regular Python strings (not np.str_)
            chains = np.array([str(res['chainid']) for res in ca_residues], dtype=object)
            residue_types = np.array([str(res['res']) for res in ca_residues], dtype=object)
            residue_numbers = np.array([res['resnum'] for res in ca_residues])

            # Load confidence data
            with open(confidence_file, 'r') as f:
                conf_data = json.load(f)

            atom_plddts = np.array(conf_data['atom_plddts'])
            plddt = atom_plddts[ca_atom_nums]

            # Extract PAE matrix and apply token mask
            pae_matrix_full = np.array(conf_data['pae'])
            token_mask_bool = token_mask.astype(bool)
            pae_matrix = pae_matrix_full[np.ix_(token_mask_bool, token_mask_bool)]

            # Extract contact_probs if available
            contact_probs = None
            if 'contact_probs' in conf_data:
                contact_probs_full = np.array(conf_data['contact_probs'])
                contact_probs = contact_probs_full[np.ix_(token_mask_bool, token_mask_bool)]

            # Load summary data
            with open(summary_file, 'r') as f:
                summary_data = json.load(f)

            # Extract chain_pair_iptm
            unique_chains = list(dict.fromkeys(chains))  # Preserve order
            chain_pair_iptm = {}
            af3_iptm_data = summary_data.get('chain_pair_iptm', [])

            for chain1 in unique_chains:
                nchain1 = ord(chain1) - ord('A')
                for chain2 in unique_chains:
                    if chain1 == chain2:
                        continue
                    nchain2 = ord(chain2) - ord('A')
                    try:
                        iptm_value = af3_iptm_data[nchain1][nchain2]
                        if iptm_value is not None:
                            chain_pair_iptm[(chain1, chain2)] = float(iptm_value)
                        else:
                            chain_pair_iptm[(chain1, chain2)] = 0.0
                    except (IndexError, TypeError):
                        chain_pair_iptm[(chain1, chain2)] = 0.0

            # Extract global scores
            global_iptm = summary_data.get('iptm')
            global_ptm = summary_data.get('ptm')
            ranking_score = summary_data.get('ranking_score')

            # Classify chains
            chain_types = self.classify_chains(chains, residue_types)

            # Build metadata with atom numbers for potential advanced use
            metadata = {
                'ca_atom_nums': ca_atom_nums,
                'cb_atom_nums': cb_atom_nums,
                'token_mask': token_mask,
                'cb_plddt': atom_plddts[cb_atom_nums],
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
                contact_probs=contact_probs,
                chain_pair_iptm=chain_pair_iptm,
                global_iptm=global_iptm,
                global_ptm=global_ptm,
                ranking_score=ranking_score,
                chain_types=chain_types,
                metadata=metadata,
            )

        except Exception as e:
            print(f"Error reading AF3 model {model_info.get('model_num', '?')}: {e}")
            return None

    def _read_structure(self, cif_file: Path) -> Tuple[List[Dict], List[Dict], np.ndarray]:
        """
        Read AF3 CIF structure file.

        Returns:
            Tuple of (ca_residues, cb_residues, token_mask)
            - ca_residues: List of dicts with CA/C1' atom info
            - cb_residues: List of dicts with CB/C3'/CA(GLY) atom info
            - token_mask: Array marking which atoms are residue representatives
        """
        ca_residues = []
        cb_residues = []
        token_mask = []

        atomsitefield_dict = {}
        atomsitefield_num = 0

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
                        token_mask.append(0)
                        continue

                    # CA or C1' atoms (residue representatives)
                    if atom['atom_name'] == "CA" or "C1" in atom['atom_name']:
                        token_mask.append(1)
                        ca_residues.append({
                            'atom_num': atom['atom_num'],
                            'coor': np.array([atom['x'], atom['y'], atom['z']]),
                            'res': atom['residue_name'],
                            'chainid': atom['chain_id'],
                            'resnum': atom['residue_seq_num'],
                        })

                    # CB, C3', or CA for GLY (for distance calculations)
                    if (atom['atom_name'] == "CB" or "C3" in atom['atom_name'] or
                        (atom['residue_name'] == "GLY" and atom['atom_name'] == "CA")):
                        cb_residues.append({
                            'atom_num': atom['atom_num'],
                            'coor': np.array([atom['x'], atom['y'], atom['z']]),
                            'res': atom['residue_name'],
                            'chainid': atom['chain_id'],
                            'resnum': atom['residue_seq_num'],
                        })

                    # Non-CA atoms in PTM/modified residues
                    if (atom['atom_name'] != "CA" and "C1" not in atom['atom_name'] and
                        atom['residue_name'] not in self.POLYMER_RESIDUES):
                        token_mask.append(0)

        return ca_residues, cb_residues, np.array(token_mask)

    def _parse_cif_atom_line(self, line: str, fielddict: Dict[str, int]) -> Optional[Dict]:
        """Parse mmCIF ATOM/HETATM line."""
        linelist = line.split()
        if len(linelist) <= max(fielddict.values()):
            return None

        try:
            residue_seq_num = linelist[fielddict['label_seq_id']]
            if residue_seq_num == ".":
                return None  # Ligand atom

            return {
                'atom_num': int(linelist[fielddict['id']]),
                'atom_name': linelist[fielddict['label_atom_id']],
                'residue_name': linelist[fielddict['label_comp_id']],
                'chain_id': linelist[fielddict['label_asym_id']],
                'residue_seq_num': int(residue_seq_num),
                'x': float(linelist[fielddict['Cartn_x']]),
                'y': float(linelist[fielddict['Cartn_y']]),
                'z': float(linelist[fielddict['Cartn_z']]),
            }
        except (ValueError, IndexError, KeyError):
            return None
