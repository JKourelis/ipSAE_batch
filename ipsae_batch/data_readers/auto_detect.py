"""
Automatic backend detection from file patterns.

Detects the prediction backend based on files present in a job folder.
"""

from pathlib import Path
from typing import Optional, List, Tuple


def detect_backend(job_folder: Path) -> Optional[str]:
    """
    Auto-detect the prediction backend from files in a job folder.

    Detection priority and patterns:
    1. Boltz2: NPZ files (plddt_*.npz, pae_*.npz) - most distinctive
    2. ColabFold: PDB + *_scores_rank_*.json pattern
    3. IntelliFold: *_seed-*_sample-*.cif + *_confidences.json pattern
    4. AlphaFold3: *_model_*.cif + *_full_data_*.json pattern

    Args:
        job_folder: Path to the job folder to analyze

    Returns:
        Backend name ('alphafold3', 'colabfold', 'boltz2', 'intellifold')
        or None if detection fails
    """
    job_folder = Path(job_folder)

    # Collect all files recursively (up to 5 levels deep)
    all_files = _collect_files_recursive(job_folder, max_depth=5)

    # Get just the filenames for pattern matching
    filenames = [f.name for f in all_files]
    extensions = set(f.suffix.lower() for f in all_files)

    # 1. Check for Boltz2 - NPZ files are very distinctive
    if '.npz' in extensions:
        npz_files = [f for f in filenames if f.endswith('.npz')]
        # Boltz2 has plddt_*.npz and pae_*.npz
        has_plddt_npz = any(f.startswith('plddt_') for f in npz_files)
        has_pae_npz = any(f.startswith('pae_') for f in npz_files)
        if has_plddt_npz and has_pae_npz:
            return 'boltz2'

    # 2. Check for ColabFold - PDB files with rank pattern
    if '.pdb' in extensions:
        pdb_files = [f for f in filenames if f.endswith('.pdb')]
        has_rank_pdb = any('_rank_' in f for f in pdb_files)

        json_files = [f for f in filenames if f.endswith('.json')]
        has_scores_json = any('_scores_rank_' in f for f in json_files)

        if has_rank_pdb and has_scores_json:
            return 'colabfold'

    # 3. Check for IntelliFold - seed/sample pattern in CIF/JSON
    if '.cif' in extensions:
        cif_files = [f for f in filenames if f.endswith('.cif')]
        has_sample_cif = any('_sample-' in f for f in cif_files)

        json_files = [f for f in filenames if f.endswith('.json')]
        has_confidences_json = any('_confidences.json' in f for f in json_files)

        if has_sample_cif and has_confidences_json:
            return 'intellifold'

    # 4. Check for AlphaFold3 - model pattern in CIF + full_data JSON
    if '.cif' in extensions:
        cif_files = [f for f in filenames if f.endswith('.cif')]
        has_model_cif = any('_model_' in f for f in cif_files)

        json_files = [f for f in filenames if f.endswith('.json')]
        has_full_data_json = any('_full_data_' in f for f in json_files)
        has_summary_json = any('_summary_confidences_' in f for f in json_files)

        if has_model_cif and has_full_data_json and has_summary_json:
            return 'alphafold3'

    return None


def detect_backend_with_confidence(job_folder: Path) -> Tuple[Optional[str], float, str]:
    """
    Auto-detect backend with confidence score and reason.

    Args:
        job_folder: Path to the job folder to analyze

    Returns:
        Tuple of (backend_name, confidence_score, reason_string)
        - backend_name: detected backend or None
        - confidence_score: 0.0 to 1.0 indicating detection confidence
        - reason_string: explanation of why this backend was detected
    """
    job_folder = Path(job_folder)

    all_files = _collect_files_recursive(job_folder, max_depth=5)
    filenames = [f.name for f in all_files]
    extensions = set(f.suffix.lower() for f in all_files)

    # Check for Boltz2
    if '.npz' in extensions:
        npz_files = [f for f in filenames if f.endswith('.npz')]
        has_plddt = any(f.startswith('plddt_') for f in npz_files)
        has_pae = any(f.startswith('pae_') for f in npz_files)
        has_pde = any(f.startswith('pde_') for f in npz_files)

        if has_plddt and has_pae:
            confidence = 0.95 if has_pde else 0.90
            reason = f"Found Boltz2 NPZ files: plddt_*.npz, pae_*.npz" + (", pde_*.npz" if has_pde else "")
            return 'boltz2', confidence, reason

    # Check for ColabFold
    if '.pdb' in extensions:
        pdb_files = [f for f in filenames if f.endswith('.pdb')]
        json_files = [f for f in filenames if f.endswith('.json')]

        rank_pdbs = [f for f in pdb_files if '_rank_' in f]
        scores_jsons = [f for f in json_files if '_scores_rank_' in f]

        if rank_pdbs and scores_jsons:
            confidence = 0.95
            reason = f"Found ColabFold files: {len(rank_pdbs)} ranked PDBs, {len(scores_jsons)} scores JSONs"
            return 'colabfold', confidence, reason

    # Check for IntelliFold
    if '.cif' in extensions:
        cif_files = [f for f in filenames if f.endswith('.cif')]
        json_files = [f for f in filenames if f.endswith('.json')]

        sample_cifs = [f for f in cif_files if '_sample-' in f]
        conf_jsons = [f for f in json_files if '_confidences.json' in f]
        summary_jsons = [f for f in json_files if '_summary_confidences.json' in f]

        if sample_cifs and conf_jsons:
            confidence = 0.90 if summary_jsons else 0.85
            reason = f"Found IntelliFold files: {len(sample_cifs)} sample CIFs, {len(conf_jsons)} confidence JSONs"
            return 'intellifold', confidence, reason

    # Check for AlphaFold3
    if '.cif' in extensions:
        cif_files = [f for f in filenames if f.endswith('.cif')]
        json_files = [f for f in filenames if f.endswith('.json')]

        model_cifs = [f for f in cif_files if '_model_' in f]
        full_data_jsons = [f for f in json_files if '_full_data_' in f]
        summary_jsons = [f for f in json_files if '_summary_confidences_' in f]

        if model_cifs and full_data_jsons and summary_jsons:
            confidence = 0.95
            reason = f"Found AF3 files: {len(model_cifs)} model CIFs, {len(full_data_jsons)} full_data JSONs"
            return 'alphafold3', confidence, reason

    # No match
    return None, 0.0, "Could not detect backend from file patterns"


def detect_backend_for_input_folder(input_folder: Path) -> Optional[str]:
    """
    Auto-detect backend by scanning job folders in the input directory.

    Checks each subdirectory and returns the first detected backend.

    Args:
        input_folder: Root directory containing job folders

    Returns:
        Backend name or None if no backend detected
    """
    input_folder = Path(input_folder)

    for item in sorted(input_folder.iterdir()):
        if item.is_dir() and not item.name.startswith('.'):
            backend = detect_backend(item)
            if backend:
                return backend

    return None


def _collect_files_recursive(folder: Path, max_depth: int = 5) -> List[Path]:
    """Collect all files recursively up to max_depth."""
    files = []

    def search(path: Path, depth: int):
        if depth > max_depth:
            return

        try:
            for item in path.iterdir():
                if item.is_file():
                    files.append(item)
                elif item.is_dir() and not item.name.startswith('.'):
                    search(item, depth + 1)
        except PermissionError:
            pass

    search(folder, 0)
    return files
