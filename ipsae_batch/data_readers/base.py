"""
Base classes for multi-backend structure prediction data readers.

This module provides the common interface and data structures for reading
structural data from different protein folding programs (AlphaFold3, ColabFold,
Boltz2, IntelliFold).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import numpy as np


@dataclass
class FoldingResult:
    """
    Standardized output from any protein folding backend.

    This dataclass contains all data needed for ipSAE and related score calculations,
    regardless of which backend produced the prediction.

    Attributes:
        job_name: Name of the prediction job/folder
        model_num: Model number (typically 0-4)
        structure_path: Path to the structure file (CIF or PDB)

        # Residue-level data (indexed 0 to N-1 where N = number of residues)
        ca_coordinates: Nx3 array of CA (or C1' for nucleic acids) coordinates
        cb_coordinates: Nx3 array of CB (or C3' for nucleic, CA for GLY) coordinates
        chains: Length-N array of chain IDs for each residue
        residue_types: Length-N array of residue names (e.g., 'ALA', 'GLY', 'DA')
        residue_numbers: Length-N array of residue sequence numbers
        plddt: Length-N array of per-residue pLDDT scores (0-100)

        # Matrices
        pae_matrix: NxN Predicted Aligned Error matrix
        contact_probs: NxN contact probability matrix (optional, AF3 only)
        pde_matrix: NxN Predicted Distance Error matrix (optional, Boltz2 native)

        # Global/chain-level scores
        chain_pair_iptm: Dict mapping (chain1, chain2) -> ipTM score
        global_iptm: Global interface pTM score (if available)
        global_ptm: Global pTM score (if available)
        ranking_score: Model ranking score (backend-specific)

        # Chain classification
        chain_types: Dict mapping chain_id -> 'protein' | 'nucleic_acid' | 'ligand'

        # Backend-specific metadata
        metadata: Dict for any backend-specific data not covered above
    """
    # Identification
    job_name: str
    model_num: int
    structure_path: Path

    # Residue-level arrays
    ca_coordinates: np.ndarray  # (N, 3)
    cb_coordinates: np.ndarray  # (N, 3)
    chains: np.ndarray  # (N,) string array
    residue_types: np.ndarray  # (N,) string array
    residue_numbers: np.ndarray  # (N,) int array
    plddt: np.ndarray  # (N,)

    # Matrices
    pae_matrix: np.ndarray  # (N, N)
    contact_probs: Optional[np.ndarray] = None  # (N, N), AF3 only
    pde_matrix: Optional[np.ndarray] = None  # (N, N), Boltz2 native

    # Global scores
    chain_pair_iptm: Dict[Tuple[str, str], float] = field(default_factory=dict)
    global_iptm: Optional[float] = None
    global_ptm: Optional[float] = None
    ranking_score: Optional[float] = None

    # Chain classification
    chain_types: Dict[str, str] = field(default_factory=dict)

    # Backend metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_residues(self) -> int:
        """Number of residues in the structure."""
        return len(self.chains)

    @property
    def unique_chains(self) -> List[str]:
        """List of unique chain IDs."""
        return list(dict.fromkeys(self.chains))  # Preserves order

    def get_chain_indices(self, chain_id: str) -> np.ndarray:
        """Get residue indices for a specific chain."""
        return np.where(self.chains == chain_id)[0]

    def get_chain_pair_type(self, chain1: str, chain2: str) -> str:
        """
        Get the interaction type for a chain pair.

        Returns 'nucleic_acid' if either chain is nucleic acid, else 'protein'.
        """
        type1 = self.chain_types.get(chain1, 'protein')
        type2 = self.chain_types.get(chain2, 'protein')
        if type1 == 'nucleic_acid' or type2 == 'nucleic_acid':
            return 'nucleic_acid'
        return 'protein'


class BaseReader(ABC):
    """
    Abstract base class for protein structure prediction data readers.

    Each backend (AF3, ColabFold, Boltz2, IntelliFold) should implement a
    subclass that knows how to find and parse that backend's output files.

    Usage:
        reader = AlphaFold3Reader()
        models = reader.find_models(job_folder)
        for model_info in models:
            result = reader.read_model(model_info)
            # result is a FoldingResult with all standardized data
    """

    # Standard amino acid residues
    PROTEIN_RESIDUES = {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"
    }

    # Nucleic acid residues
    NUCLEIC_RESIDUES = {"DA", "DC", "DT", "DG", "A", "C", "U", "G"}

    # All standard polymer residues
    POLYMER_RESIDUES = PROTEIN_RESIDUES | NUCLEIC_RESIDUES

    def __init__(self, backend_name: str):
        """
        Initialize the reader.

        Args:
            backend_name: Name of the backend (e.g., 'alphafold3', 'colabfold')
        """
        self.backend_name = backend_name

    @abstractmethod
    def find_models(self, job_folder: Path) -> List[Dict[str, Any]]:
        """
        Find all model files in a job folder.

        Args:
            job_folder: Path to the folder containing prediction output files

        Returns:
            List of model info dicts, each containing:
                - 'job_name': str
                - 'model_num': int
                - 'structure_file': Path
                - 'confidence_file': Path
                - 'summary_file': Path (may be same as confidence_file)
                - Any backend-specific keys needed for read_model()
        """
        pass

    @abstractmethod
    def read_model(self, model_info: Dict[str, Any]) -> Optional[FoldingResult]:
        """
        Read and parse a single model into a FoldingResult.

        Args:
            model_info: Dict from find_models() containing file paths

        Returns:
            FoldingResult with all standardized data, or None if parsing fails
        """
        pass

    def read_all_models(self, job_folder: Path) -> List[FoldingResult]:
        """
        Convenience method to find and read all models in a folder.

        Args:
            job_folder: Path to the folder containing prediction output files

        Returns:
            List of FoldingResult objects for all successfully parsed models
        """
        models = self.find_models(job_folder)
        results = []
        for model_info in models:
            result = self.read_model(model_info)
            if result is not None:
                results.append(result)
        return results

    def classify_chains(self, chains: np.ndarray, residue_types: np.ndarray) -> Dict[str, str]:
        """
        Classify chains as protein, nucleic_acid, or ligand.

        Args:
            chains: Array of chain IDs for each residue
            residue_types: Array of residue type names

        Returns:
            Dict mapping chain_id -> 'protein' | 'nucleic_acid' | 'ligand'
        """
        chain_types = {}

        for chain in np.unique(chains):
            indices = np.where(chains == chain)[0]
            chain_residues = set(residue_types[indices])

            # Check for nucleic acids first
            if chain_residues & self.NUCLEIC_RESIDUES:
                chain_types[chain] = 'nucleic_acid'
            # Then proteins
            elif chain_residues & self.PROTEIN_RESIDUES:
                chain_types[chain] = 'protein'
            # Otherwise ligand
            else:
                chain_types[chain] = 'ligand'

        return chain_types

    def _find_file_by_patterns(self, folder: Path, patterns: List[str]) -> Optional[Path]:
        """Find first file matching any of the given patterns in a folder."""
        for pattern in patterns:
            files = list(folder.glob(pattern))
            if files:
                return files[0]
        return None

    def _find_model_folder_recursive(
        self,
        start_folder: Path,
        required_patterns: List[List[str]],
        max_depth: int = 5
    ) -> Optional[Path]:
        """
        Recursively search for a folder containing required model files.

        Searches start_folder and its subfolders (up to max_depth) for a folder
        that contains files matching ALL required pattern groups.

        Args:
            start_folder: Folder to start searching from
            required_patterns: List of pattern groups. Each group is a list of
                              glob patterns (OR within group, AND between groups).
                              E.g., [["*.cif"], ["*_full_data_*.json"]] requires
                              both a CIF file and a JSON file.
            max_depth: Maximum folder depth to search

        Returns:
            Path to folder containing required files, or None if not found
        """
        def check_folder(folder: Path) -> bool:
            """Check if folder contains files matching all required pattern groups."""
            for pattern_group in required_patterns:
                found = False
                for pattern in pattern_group:
                    if list(folder.glob(pattern)):
                        found = True
                        break
                if not found:
                    return False
            return True

        def search(folder: Path, depth: int) -> Optional[Path]:
            if depth > max_depth:
                return None

            # Check current folder first
            if check_folder(folder):
                return folder

            # Search subfolders
            try:
                for subfolder in sorted(folder.iterdir()):
                    if subfolder.is_dir() and not subfolder.name.startswith('.'):
                        result = search(subfolder, depth + 1)
                        if result:
                            return result
            except PermissionError:
                pass

            return None

        return search(start_folder, 0)


class ReaderError(Exception):
    """Exception raised for errors during structure reading."""
    pass
