"""
Boltz2 data reader.

Reads structural data from Boltz2 outputs:
- Structure: CIF files ({job_name}_model_N.cif)
- pLDDT: NPZ files (plddt_{job_name}_model_N.npz)
- PAE: NPZ files (pae_{job_name}_model_N.npz)
- PDE: NPZ files (pde_{job_name}_model_N.npz) - Native PDE from Boltz2
- Confidence: JSON files (confidence_{job_name}_model_N.json)

Boltz2 provides native PDE matrices, which are stored in the FoldingResult.
"""

import glob
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import numpy as np

from .base import BaseReader, FoldingResult, ReaderError
from . import register_reader


@register_reader('boltz2')
class Boltz2Reader(BaseReader):
    """Reader for Boltz2 CIF structure files with NPZ confidence data."""

    def __init__(self, backend_name: str = 'boltz2'):
        super().__init__(backend_name)

    def find_models(self, job_folder: Path) -> List[Dict[str, Any]]:
        """
        Find Boltz2 model files in a job folder.

        Recursively searches for folders containing Boltz2 output files.

        Args:
            job_folder: Path to folder containing Boltz2 output (typically *_results)

        Returns:
            List of model info dicts with keys:
                job_name, model_num, structure_file, plddt_file, pae_file, pde_file, confidence_file
        """
        job_folder = Path(job_folder)

        # Boltz2 folder typically ends with _results
        job_name = job_folder.name
        if job_name.endswith('_results'):
            job_name = job_name[:-8]

        # Define required file patterns for Boltz2 (at least one model)
        required_patterns = [
            ["*_model_*.cif"],  # Structure file
            ["plddt_*_model_*.npz"],  # pLDDT file
            ["pae_*_model_*.npz"],  # PAE file
        ]

        # Find the folder containing model files (recursive search)
        model_folder = self._find_model_folder_recursive(job_folder, required_patterns)
        if model_folder is None:
            return []

        models = []

        # Boltz2 typically produces models 0-4
        for model_num in range(5):
            # Find CIF structure file
            cif_patterns = [
                f"{job_name}_model_{model_num}.cif",
                f"*_model_{model_num}.cif",
            ]
            cif_file = self._find_file_by_patterns(model_folder, cif_patterns)

            # Find pLDDT NPZ
            plddt_patterns = [
                f"plddt_{job_name}_model_{model_num}.npz",
                f"plddt_*_model_{model_num}.npz",
            ]
            plddt_file = self._find_file_by_patterns(model_folder, plddt_patterns)

            # Find PAE NPZ
            pae_patterns = [
                f"pae_{job_name}_model_{model_num}.npz",
                f"pae_*_model_{model_num}.npz",
            ]
            pae_file = self._find_file_by_patterns(model_folder, pae_patterns)

            # Find PDE NPZ (native Boltz2 PDE)
            pde_patterns = [
                f"pde_{job_name}_model_{model_num}.npz",
                f"pde_*_model_{model_num}.npz",
            ]
            pde_file = self._find_file_by_patterns(model_folder, pde_patterns)

            # Find confidence JSON
            conf_patterns = [
                f"confidence_{job_name}_model_{model_num}.json",
                f"confidence_*_model_{model_num}.json",
            ]
            conf_file = self._find_file_by_patterns(model_folder, conf_patterns)

            if cif_file and plddt_file and pae_file:
                models.append({
                    'job_name': job_name,
                    'model_num': model_num,
                    'structure_file': cif_file,
                    'plddt_file': plddt_file,
                    'pae_file': pae_file,
                    'pde_file': pde_file,  # May be None
                    'confidence_file': conf_file,  # May be None
                })

        return models

    def read_model(self, model_info: Dict[str, Any]) -> Optional[FoldingResult]:
        """
        Read and parse a Boltz2 model into a FoldingResult.

        Args:
            model_info: Dict from find_models() containing file paths

        Returns:
            FoldingResult with all standardized data, or None if parsing fails
        """
        try:
            job_name = model_info['job_name']
            model_num = model_info['model_num']
            structure_file = Path(model_info['structure_file'])
            plddt_file = Path(model_info['plddt_file'])
            pae_file = Path(model_info['pae_file'])
            pde_file = model_info.get('pde_file')
            confidence_file = model_info.get('confidence_file')

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

            # Load pLDDT from NPZ (Boltz2 stores as 0-1, we need 0-100)
            plddt_data = np.load(plddt_file)
            plddt = plddt_data['plddt'] * 100.0  # Convert to 0-100 scale

            # Load PAE from NPZ
            pae_data = np.load(pae_file)
            pae_matrix = pae_data['pae']

            # Load PDE from NPZ if available (native Boltz2 PDE)
            pde_matrix = None
            if pde_file and Path(pde_file).exists():
                pde_data = np.load(pde_file)
                pde_matrix = pde_data['pde']

            # Load confidence data from JSON if available
            global_iptm = None
            global_ptm = None
            chain_pair_iptm = {}
            ranking_score = None
            complex_pde = None
            complex_ipde = None

            if confidence_file and Path(confidence_file).exists():
                with open(confidence_file, 'r') as f:
                    conf_data = json.load(f)

                global_iptm = conf_data.get('iptm')
                global_ptm = conf_data.get('ptm')
                ranking_score = conf_data.get('confidence_score')
                complex_pde = conf_data.get('complex_pde')
                complex_ipde = conf_data.get('complex_ipde')

                # Extract pair_chains_iptm and map to chain names
                pair_iptm_data = conf_data.get('pair_chains_iptm', {})
                unique_chains = list(dict.fromkeys(chains))

                # Map numeric indices to chain names
                for idx1, chain1 in enumerate(unique_chains):
                    for idx2, chain2 in enumerate(unique_chains):
                        if chain1 != chain2:
                            try:
                                iptm_val = pair_iptm_data.get(str(idx1), {}).get(str(idx2))
                                if iptm_val is not None:
                                    chain_pair_iptm[(chain1, chain2)] = float(iptm_val)
                                else:
                                    chain_pair_iptm[(chain1, chain2)] = global_iptm if global_iptm else 0.0
                            except (KeyError, TypeError):
                                chain_pair_iptm[(chain1, chain2)] = global_iptm if global_iptm else 0.0
            else:
                # No confidence file - use default values
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
                'complex_pde': complex_pde,
                'complex_ipde': complex_ipde,
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
                pde_matrix=pde_matrix,  # Native Boltz2 PDE
                chain_pair_iptm=chain_pair_iptm,
                global_iptm=global_iptm,
                global_ptm=global_ptm,
                ranking_score=ranking_score,
                chain_types=chain_types,
                metadata=metadata,
            )

        except Exception as e:
            print(f"Error reading Boltz2 model {model_info.get('model_num', '?')}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _read_cif_structure(self, cif_file: Path) -> Tuple[List[Dict], List[Dict], Dict[str, int]]:
        """
        Read Boltz2 CIF structure file.

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
        """Parse mmCIF ATOM/HETATM line for Boltz2 format."""
        linelist = line.split()
        if len(linelist) <= max(fielddict.values()):
            return None

        try:
            residue_seq_num = linelist[fielddict['label_seq_id']]
            if residue_seq_num == ".":
                return None  # Ligand atom

            # Get chain ID - Boltz2 uses longer chain names
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
