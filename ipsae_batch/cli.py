#!/usr/bin/env python3
"""
ipsae_batch CLI - Batch processing of protein structure predictions for ipSAE scoring

This script processes output folders from multiple structure prediction backends
(AlphaFold3, ColabFold, Boltz2, IntelliFold) and calculates interaction scores
for all models. Outputs per-folder CSV files with aggregate scores.

Based on ipSAE by Roland Dunbrack, Fox Chase Cancer Center
https://www.biorxiv.org/content/10.1101/2025.02.10.637595v1

Usage:
    python -m ipsae_batch <input_folder> [options]
    ipsae-batch <input_folder> [options]  # If installed via pip

Example:
    ipsae-batch ./data --pae_cutoff 10 --dist_cutoff 10 --workers 4
    ipsae-batch ./data --backend colabfold --per_residue
"""

import sys
import os
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from multiprocessing import Pool, cpu_count

# Import data readers (relative imports for package)
from .data_readers import get_reader, list_backends, FoldingResult, detect_backend_for_input_folder

# Import parallel processing
from .core.parallel_processing import process_folders_parallel
AVAILABLE_BACKENDS = list_backends()

# Import core scoring module
from .core import calculate_scores_from_result
from .core.scoring import calculate_per_contact_scores
from .core.residue_selection import (
    parse_residue_selection,
    calculate_ipsae_selection_metrics,
    display_selection_summary,
)

# Import output writers
from .output_writers import write_aggregate_csv, write_per_residue_csv, write_per_contact_csv

# Import extractors for graphics
from .extractors import extract_pmc, cluster_domains_from_result, get_geometric_interfaces, get_proximity_contacts
from .extractors.pde import extract_contact_probs

# Import graphics (lazy load matplotlib)
HAS_GRAPHICS = False
HAS_BATCH_COMPARISON = False
plt = None
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .graphics import plot_alphabridge_combined, plot_ribbon
    from .graphics.config import load_config_from_csv, set_config, get_config, GraphicsConfig
    from .graphics.pdf_export import (
        ModelPDFReport, generate_combined_matrix_png,
        generate_combined_pae_png, generate_combined_ribbon_png
    )
    HAS_GRAPHICS = True
except ImportError:
    pass

try:
    from .graphics.batch_comparison import generate_batch_comparison_from_results
    HAS_BATCH_COMPARISON = True
except ImportError:
    pass


# =============================================================================
# File discovery
# =============================================================================

def find_job_folders_for_backend(input_folder: str, backend: str) -> List[Tuple[str, List[Dict]]]:
    """
    Find all job folders for a specific backend in the input directory.

    Args:
        input_folder: Root directory containing job folders
        backend: Backend name ('alphafold3', 'colabfold', 'boltz2', 'intellifold')

    Returns:
        List of (folder_path, model_infos) tuples for folders with valid models
    """
    reader = get_reader(backend)
    job_folders = []

    for item in sorted(os.listdir(input_folder)):
        item_path = os.path.join(input_folder, item)
        if os.path.isdir(item_path):
            models = reader.find_models(Path(item_path))
            if models:
                job_folders.append((item_path, models))

    return job_folders


# =============================================================================
# Model processing
# =============================================================================

def process_single_model(args: Tuple) -> Tuple[List[Dict], Optional[List[Dict]], Optional[List[Dict]], Dict]:
    """
    Worker function to process a single model.
    Args is a tuple of (model_dict, pae_cutoff, dist_cutoff, per_residue, per_contact, residue_selection)
    Returns (aggregate_results, per_residue_results, per_contact_results, model_info)
    If residue_selection is provided, selection-specific metrics are added to aggregate results.
    """
    model, pae_cutoff, dist_cutoff, per_residue, per_contact, residue_selection = args

    job_name = model['job_name']
    model_num = model['model_num']

    try:
        folding_result = model['folding_result']
        results, per_res_results = calculate_scores_from_result(
            folding_result,
            pae_cutoff,
            dist_cutoff,
            return_per_residue=per_residue
        )

        # Add job and model info to each result
        for r in results:
            r['job_name'] = job_name
            r['model'] = model_num
            r['pae_cutoff'] = pae_cutoff
            r['dist_cutoff'] = dist_cutoff

        if per_res_results:
            for r in per_res_results:
                r['job_name'] = job_name
                r['model'] = model_num

        # Calculate per-contact scores if requested (or needed for selection metrics)
        per_contact_results = None
        if per_contact or residue_selection:
            per_contact_results = calculate_per_contact_scores(
                folding_result, pae_cutoff=pae_cutoff, dist_cutoff=dist_cutoff
            )
            for c in per_contact_results:
                c['job_name'] = job_name
                c['model'] = model_num

        # Calculate selection-specific metrics if residue selection is provided
        if residue_selection and per_contact_results:
            selection_metrics = calculate_ipsae_selection_metrics(
                per_contact_results, residue_selection
            )
            # Add selection metrics to each aggregate result
            for r in results:
                r.update(selection_metrics)

        return results, per_res_results, per_contact_results if per_contact else None, model

    except Exception as e:
        print(f"  Error processing {job_name} model {model_num}: {e}")
        import traceback
        traceback.print_exc()
        return [], None, None, model


def process_batch(input_folder: str, pae_cutoff: float, dist_cutoff: float,
                  output_dir: str, num_workers: int = None,
                  per_residue: bool = False, per_contact: bool = False,
                  png: bool = False, pdf: bool = False, config_path: str = None,
                  backend: str = 'alphafold3',
                  residue_selection: Optional[Dict] = None,
                  cores: int = None) -> None:
    """Process all job folders and output results to CSV using parallel processing.

    Args:
        input_folder: Directory containing job folders
        pae_cutoff: PAE cutoff for interface scoring
        dist_cutoff: Distance cutoff for interface contacts
        output_dir: Output directory for CSV files
        num_workers: Number of parallel workers
        per_residue: Whether to output per-residue scores
        per_contact: Whether to output per-contact scores
        png: Whether to generate PNG graphics for each model
        pdf: Whether to generate PDF report with side-by-side model comparison
        config_path: Path to graphics config CSV (optional)
        backend: Backend name ('alphafold3', 'colabfold', 'boltz2', 'intellifold')
        residue_selection: Optional dict mapping chain_id -> set of residue numbers
                          for focused analysis of specific residues
    """
    start_time = time.time()

    # Load graphics config if provided
    graphics_config = None
    if config_path and HAS_GRAPHICS:
        graphics_config = load_config_from_csv(config_path)
        set_config(graphics_config)
        print(f"Loaded config: {config_path}")
    elif (png or pdf) and not HAS_GRAPHICS:
        print("Warning: matplotlib/pycirclize not available, --png/--pdf disabled")
        png = False
        pdf = False

    reader = get_reader(backend)
    job_folders_with_models = find_job_folders_for_backend(input_folder, backend)
    if not job_folders_with_models:
        print(f"No {backend} folders found in {input_folder}")
        return
    print(f"Found {len(job_folders_with_models)} {backend} folder(s)")

    # Create output directory if needed
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Build config dict for parallel processing
    config_dict = {
        'backend': backend,
        'pae_cutoff': pae_cutoff,
        'dist_cutoff': dist_cutoff,
        'output_dir': output_dir,
        'per_residue': per_residue,
        'per_contact': per_contact,
        'png': png,
        'pdf': pdf,
        'residue_selection': residue_selection,
        'graphics_config': graphics_config,
    }

    # Process folders (parallel or sequential based on cores)
    all_combined_results, all_per_contact_results, total_models = process_folders_parallel(
        job_folders_with_models,
        config_dict,
        cores=cores,
    )

    elapsed_time = time.time() - start_time

    # Write combined output
    if all_combined_results:
        if output_dir:
            combined_output = os.path.join(output_dir, "ipSAE_combined.csv")
        else:
            combined_output = os.path.join(input_folder, "ipSAE_combined.csv")

        write_aggregate_csv(all_combined_results, combined_output)
        print(f"\nCombined results: {combined_output}")

    # Write combined per-contact output
    if per_contact and all_per_contact_results:
        if output_dir:
            combined_contacts = os.path.join(output_dir, "ipSAE_contacts_combined.csv")
        else:
            combined_contacts = os.path.join(input_folder, "ipSAE_contacts_combined.csv")

        write_per_contact_csv(all_per_contact_results, combined_contacts)
        print(f"Combined contacts: {combined_contacts}")

    # Generate batch comparison HTML (always generated when CSV is produced)
    if all_combined_results and HAS_BATCH_COMPARISON:
        try:
            if output_dir:
                html_path = os.path.join(output_dir, "ipSAE_comparison.html")
            else:
                html_path = os.path.join(input_folder, "ipSAE_comparison.html")

            generate_batch_comparison_from_results(
                all_combined_results,
                html_path,
                default_x_metric='ipTM',
                default_y_metric='ipSAE',
                jitter_threshold=0.3
            )
        except Exception as e:
            print(f"Warning: Failed to generate batch comparison HTML: {e}")

    print(f"\nTotal folders processed: {len(job_folders_with_models)}")
    print(f"Total models processed: {total_models}")
    print(f"Total rows: {len(all_combined_results)}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    if total_models > 0:
        print(f"Average per model: {elapsed_time/total_models:.2f} seconds")


def main():
    """Main entry point"""
    backends_help = ', '.join(AVAILABLE_BACKENDS)
    parser = argparse.ArgumentParser(
        description='Batch processing of protein structure predictions for ipSAE scoring',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Supported backends: {backends_help}

Examples:
  python ipSAE_batch.py ./data
  python ipSAE_batch.py ./data --backend colabfold
  python ipSAE_batch.py ./data --backend boltz2 --pae_cutoff 12 --dist_cutoff 8
  python ipSAE_batch.py ./data --per_residue --workers 4
  python ipSAE_batch.py ./data --output_dir ./results
  python ipSAE_batch.py ./data --png --config graphics_config.csv
  python ipSAE_batch.py ./data --per_contact --png --output_dir ./output
        """
    )

    parser.add_argument('input_folder',
                        help='Folder containing job output folders')
    parser.add_argument('--backend', type=str, default='auto',
                        choices=['auto'] + AVAILABLE_BACKENDS,
                        help=f'Structure prediction backend (default: auto-detect). Options: auto, {backends_help}')
    parser.add_argument('--pae_cutoff', type=float, default=10.0,
                        help='PAE cutoff for ipSAE calculation (default: 10)')
    parser.add_argument('--dist_cutoff', type=float, default=10.0,
                        help='Distance cutoff for interface residues in Angstroms (default: 10)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results (default: same as input folders)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: number of CPU cores)')
    parser.add_argument('--per_residue', action='store_true',
                        help='Also output per-residue scores (_byres.csv files)')
    parser.add_argument('--per_contact', action='store_true',
                        help='Also output per-contact scores (_contacts.csv files)')
    parser.add_argument('--png', action='store_true',
                        help='Generate PNG graphics for each model (matrix + ribbon plots)')
    parser.add_argument('--pdf', action='store_true',
                        help='Generate PDF report with side-by-side model comparison per job')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to graphics configuration CSV file')
    parser.add_argument('--select-residues', type=str, default=None,
                        dest='select_residues',
                        help='Focus analysis on specific residues (e.g., "A:100-105,B:203,C:50-60")')
    parser.add_argument('--cores', type=int, default=None,
                        help='Number of CPU cores for folder-level parallelization (default: all - 1)')

    args = parser.parse_args()

    if not os.path.isdir(args.input_folder):
        print(f"Error: {args.input_folder} is not a directory")
        sys.exit(1)

    # Auto-detect backend if set to 'auto'
    backend = args.backend
    if backend == 'auto':
        from pathlib import Path
        detected = detect_backend_for_input_folder(Path(args.input_folder))
        if detected:
            backend = detected
            print(f"Auto-detected backend: {backend}")
        else:
            print("Error: Could not auto-detect backend from file patterns.")
            print("Please specify --backend manually.")
            print(f"Available backends: {', '.join(AVAILABLE_BACKENDS)}")
            sys.exit(1)

    # Parse residue selection if provided
    residue_selection = None
    if args.select_residues:
        residue_selection = parse_residue_selection(args.select_residues)

    print(f"ipSAE Batch Processor")
    print(f"=====================")
    print(f"Backend: {backend}")
    print(f"Input folder: {args.input_folder}")
    print(f"PAE cutoff: {args.pae_cutoff}")
    print(f"Distance cutoff: {args.dist_cutoff}")
    if args.output_dir:
        print(f"Output directory: {args.output_dir}")
    print(f"Per-residue output: {args.per_residue}")
    print(f"Per-contact output: {args.per_contact}")
    print(f"PNG graphics: {args.png}")
    print(f"PDF reports: {args.pdf}")
    if args.config:
        print(f"Config file: {args.config}")
    if args.cores:
        print(f"Parallel cores: {args.cores}")
    else:
        print(f"Parallel cores: auto (CPU count - 1)")
    if residue_selection:
        display_selection_summary(residue_selection)

    process_batch(
        args.input_folder,
        args.pae_cutoff,
        args.dist_cutoff,
        args.output_dir,
        args.workers,
        args.per_residue,
        args.per_contact,
        args.png,
        args.pdf,
        args.config,
        backend,
        residue_selection,
        args.cores
    )


if __name__ == "__main__":
    main()
