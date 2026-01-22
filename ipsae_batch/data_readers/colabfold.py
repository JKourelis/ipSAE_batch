"""
ColabFold (AlphaFold2-multimer) data reader.

Reads structural data from ColabFold outputs:
- Structure: PDB files (*_unrelaxed_rank_NNN_*.pdb or *_relaxed_rank_NNN_*.pdb)
- Confidence: JSON files (*_scores_rank_NNN_*.json) containing PAE, pLDDT, ptm, iptm

Note: ColabFold does not provide chain_pair_iptm, only global iptm.
"""

import glob
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import numpy as np

from .base import BaseReader, FoldingResult, ReaderError
from . import register_reader


@register_reader('colabfold')
class ColabFoldReader(BaseReader):
    """Reader for ColabFold PDB structure files with JSON confidence data."""

    def __init__(self, backend_name: str = 'colabfold'):
        super().__init__(backend_name)

    def find_models(self, job_folder: Path) -> List[Dict[str, Any]]:
        """
        Find ColabFold model files in a job folder.

        Recursively searches for folders containing ColabFold output files.
        ColabFold produces ranked models (rank_001 through rank_005).

        Args:
            job_folder: Path to folder containing ColabFold output files

        Returns:
            List of model info dicts with keys:
                job_name, model_num, structure_file, confidence_file
        """
        job_folder = Path(job_folder)
        job_name = job_folder.name

        # Define required file patterns for ColabFold (at least one model)
        required_patterns = [
            ["*_unrelaxed_rank_*.pdb", "*_relaxed_rank_*.pdb", "*_rank_*.pdb"],  # Structure file
            ["*_scores_rank_*.json", "*_confidence_rank_*.json"],  # Confidence file
        ]

        # Find the folder containing model files (recursive search)
        model_folder = self._find_model_folder_recursive(job_folder, required_patterns)
        if model_folder is None:
            return []

        models = []

        # ColabFold produces rank_001 through rank_005 (1-indexed)
        for model_num in range(5):
            rank_num = model_num + 1
            rank_str = f"{rank_num:03d}"

            # Find PDB structure file
            pdb_patterns = [
                f"*_unrelaxed_rank_{rank_str}_*.pdb",
                f"*_relaxed_rank_{rank_str}_*.pdb",
                f"*_rank_{rank_str}_*.pdb",
            ]
            pdb_file = self._find_file_by_patterns(model_folder, pdb_patterns)

            # Find scores JSON file (contains pLDDT, PAE, ptm, iptm)
            scores_patterns = [
                f"*_scores_rank_{rank_str}_*.json",
                f"*_confidence_rank_{rank_str}_*.json",
            ]
            scores_file = self._find_file_by_patterns(model_folder, scores_patterns)

            if pdb_file and scores_file:
                models.append({
                    'job_name': job_name,
                    'model_num': model_num,
                    'structure_file': pdb_file,
                    'confidence_file': scores_file,
                })

        return models

    def read_model(self, model_info: Dict[str, Any]) -> Optional[FoldingResult]:
        """
        Read and parse a ColabFold model into a FoldingResult.

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

            # Read structure from PDB
            ca_residues, cb_residues = self._read_pdb_structure(structure_file)

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

            # pLDDT from PDB B-factor (per-residue average)
            plddt_from_pdb = np.array([res['plddt'] for res in ca_residues])

            # Load confidence data from JSON
            with open(confidence_file, 'r') as f:
                conf_data = json.load(f)

            # pLDDT from JSON (should match PDB but JSON is more precise)
            plddt_json = np.array(conf_data['plddt'])

            # Use JSON pLDDT if available and length matches
            if len(plddt_json) == num_residues:
                plddt = plddt_json
            else:
                plddt = plddt_from_pdb

            # PAE matrix
            pae_matrix = np.array(conf_data['pae'])

            # Global scores (ColabFold doesn't have chain_pair_iptm)
            global_iptm = conf_data.get('iptm')
            global_ptm = conf_data.get('ptm')

            # Build chain_pair_iptm using global iptm for all pairs
            # (ColabFold doesn't provide per-chain-pair values)
            unique_chains = list(dict.fromkeys(chains))
            chain_pair_iptm = {}
            for chain1 in unique_chains:
                for chain2 in unique_chains:
                    if chain1 != chain2:
                        # Use global iptm as estimate for all pairs
                        chain_pair_iptm[(chain1, chain2)] = global_iptm if global_iptm else 0.0

            # CB pLDDT (from PDB B-factors)
            cb_plddt = np.array([res['plddt'] for res in cb_residues])

            # Classify chains
            chain_types = self.classify_chains(chains, residue_types)

            # Metadata
            metadata = {
                'cb_plddt': cb_plddt,
                'max_pae': conf_data.get('max_pae'),
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
                chain_types=chain_types,
                metadata=metadata,
            )

        except Exception as e:
            print(f"Error reading ColabFold model {model_info.get('model_num', '?')}: {e}")
            return None

    def _read_pdb_structure(self, pdb_file: Path) -> Tuple[List[Dict], List[Dict]]:
        """
        Read PDB structure file.

        Returns:
            Tuple of (ca_residues, cb_residues)
            Each is a list of dicts with atom info
        """
        ca_residues = []
        cb_residues = []

        # Track which residues we've seen (to handle multiple atoms per residue)
        seen_ca = set()
        seen_cb = set()

        with open(pdb_file, 'r') as f:
            for line in f:
                if not (line.startswith('ATOM') or line.startswith('HETATM')):
                    continue

                # Parse PDB ATOM line (fixed format)
                try:
                    atom_name = line[12:16].strip()
                    residue_name = line[17:20].strip()
                    chain_id = line[21:22].strip()
                    residue_num = int(line[22:26].strip())
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    b_factor = float(line[60:66].strip())  # pLDDT in AlphaFold PDBs
                except (ValueError, IndexError):
                    continue

                # Skip if not a standard residue
                if residue_name not in self.POLYMER_RESIDUES:
                    continue

                residue_key = (chain_id, residue_num)

                # CA atoms (or C1' for nucleic acids)
                if atom_name == 'CA' or atom_name == "C1'":
                    if residue_key not in seen_ca:
                        seen_ca.add(residue_key)
                        ca_residues.append({
                            'coor': np.array([x, y, z]),
                            'res': residue_name,
                            'chainid': chain_id,
                            'resnum': residue_num,
                            'plddt': b_factor,
                        })

                # CB atoms (or C3' for nucleic acids, or CA for GLY)
                if atom_name == 'CB' or atom_name == "C3'" or (residue_name == 'GLY' and atom_name == 'CA'):
                    if residue_key not in seen_cb:
                        seen_cb.add(residue_key)
                        cb_residues.append({
                            'coor': np.array([x, y, z]),
                            'res': residue_name,
                            'chainid': chain_id,
                            'resnum': residue_num,
                            'plddt': b_factor,
                        })

        return ca_residues, cb_residues
