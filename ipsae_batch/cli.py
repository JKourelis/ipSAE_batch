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
AVAILABLE_BACKENDS = list_backends()

# Import core scoring module
from .core import calculate_scores_from_result
from .core.scoring import calculate_per_contact_scores

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
    Args is a tuple of (model_dict, pae_cutoff, dist_cutoff, per_residue, per_contact)
    Returns (aggregate_results, per_residue_results, per_contact_results, model_info)
    """
    model, pae_cutoff, dist_cutoff, per_residue, per_contact = args

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

        # Calculate per-contact scores if requested
        per_contact_results = None
        if per_contact:
            per_contact_results = calculate_per_contact_scores(
                folding_result, pae_cutoff=pae_cutoff, dist_cutoff=dist_cutoff
            )
            for c in per_contact_results:
                c['job_name'] = job_name
                c['model'] = model_num

        return results, per_res_results, per_contact_results, model

    except Exception as e:
        print(f"  Error processing {job_name} model {model_num}: {e}")
        import traceback
        traceback.print_exc()
        return [], None, None, model


def process_batch(input_folder: str, pae_cutoff: float, dist_cutoff: float,
                  output_dir: str, num_workers: int = None,
                  per_residue: bool = False, per_contact: bool = False,
                  png: bool = False, pdf: bool = False, config_path: str = None,
                  backend: str = 'alphafold3') -> None:
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
    """
    start_time = time.time()

    # Load graphics config if provided
    if config_path and HAS_GRAPHICS:
        config = load_config_from_csv(config_path)
        set_config(config)
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

    # Process each folder
    all_combined_results = []
    all_per_contact_results = []
    total_models = 0

    for folder, model_infos in job_folders_with_models:
        job_name = model_infos[0]['job_name'] if model_infos else os.path.basename(folder)

        # Read all models using the reader
        models = []
        for model_info in model_infos:
            folding_result = reader.read_model(model_info)
            if folding_result:
                models.append({
                    'job_name': folding_result.job_name,
                    'model_num': folding_result.model_num,
                    'folding_result': folding_result,
                })

        if not models:
            print(f"\n{job_name}: No complete model sets found, skipping")
            continue

        print(f"\nProcessing: {job_name} ({len(models)} models)")
        total_models += len(models)

        # Prepare arguments for processing (per_contact always true if PNG/PDF requested)
        need_per_contact = per_contact or png or pdf
        work_args = [(model, pae_cutoff, dist_cutoff, per_residue, need_per_contact) for model in models]

        # Determine number of workers for this folder
        n_workers = num_workers
        if n_workers is None:
            n_workers = min(cpu_count(), len(models))
        n_workers = max(1, min(n_workers, len(models)))

        # Process models
        folder_results = []
        folder_per_res_results = []
        folder_per_contact_results = []
        model_results_map = {}  # model_num -> (agg_res, per_contact_res, folding_result)

        if n_workers == 1:
            for i, args in enumerate(work_args):
                print(f"  Model {args[0]['model_num']}...")
                agg_res, per_res, per_contact_res, model_info = process_single_model(args)
                folder_results.extend(agg_res)
                if per_res:
                    folder_per_res_results.extend(per_res)
                if per_contact_res:
                    folder_per_contact_results.extend(per_contact_res)
                model_results_map[model_info['model_num']] = (agg_res, per_contact_res, model_info['folding_result'])
        else:
            print(f"  Processing {len(models)} models in parallel ({n_workers} workers)...")
            with Pool(processes=n_workers) as pool:
                results_list = pool.map(process_single_model, work_args)

            for agg_res, per_res, per_contact_res, model_info in results_list:
                folder_results.extend(agg_res)
                if per_res:
                    folder_per_res_results.extend(per_res)
                if per_contact_res:
                    folder_per_contact_results.extend(per_contact_res)
                model_results_map[model_info['model_num']] = (agg_res, per_contact_res, model_info['folding_result'])

        # Write per-folder output
        if output_dir:
            folder_output = os.path.join(output_dir, f"{job_name}_ipSAE.csv")
        else:
            folder_output = os.path.join(folder, f"{job_name}_ipSAE.csv")

        write_aggregate_csv(folder_results, folder_output)
        print(f"  Wrote: {folder_output}")

        # Write per-residue output if requested
        if per_residue and folder_per_res_results:
            if output_dir:
                per_res_output = os.path.join(output_dir, f"{job_name}_ipSAE_byres.csv")
            else:
                per_res_output = os.path.join(folder, f"{job_name}_ipSAE_byres.csv")

            write_per_residue_csv(folder_per_res_results, per_res_output)
            print(f"  Wrote: {per_res_output}")

        # Write per-contact output if requested
        if per_contact and folder_per_contact_results:
            if output_dir:
                per_contact_output = os.path.join(output_dir, f"{job_name}_ipSAE_contacts.csv")
            else:
                per_contact_output = os.path.join(folder, f"{job_name}_ipSAE_contacts.csv")

            write_per_contact_csv(folder_per_contact_results, per_contact_output)
            print(f"  Wrote: {per_contact_output}")

        # Generate PNG graphics if requested (combined side-by-side for all models)
        if png and HAS_GRAPHICS:
            config = get_config()

            # Pre-compute data for all models
            precomputed_data = {}
            folder_folding_results = []

            for model_num, (agg_res, contact_scores, folding_result) in model_results_map.items():
                try:
                    folder_folding_results.append(folding_result)

                    pmc = extract_pmc(folding_result)
                    contact_probs = extract_contact_probs(folding_result)
                    clusters = cluster_domains_from_result(folding_result)

                    interfaces = get_geometric_interfaces(
                        folding_result,
                        distance_threshold=dist_cutoff,
                        pae_threshold=pae_cutoff,
                        gap_merge=config.interface.gap_merge
                    )

                    proximity_contacts = get_proximity_contacts(
                        folding_result,
                        distance_threshold=dist_cutoff
                    )

                    precomputed_data[model_num] = {
                        'pmc': pmc,
                        'contact_probs': contact_probs,
                        'clusters': clusters,
                        'interfaces': interfaces,
                        'proximity_contacts': proximity_contacts,
                        'contact_scores': contact_scores,
                    }
                except Exception as e:
                    print(f"  Warning: Failed to precompute data for model {model_num}: {e}")

            # Sort by model number
            folder_folding_results.sort(key=lambda r: r.model_num)

            # Generate combined PNG outputs
            try:
                if output_dir:
                    matrix_path = os.path.join(output_dir, f"{job_name}_matrices_combined.png")
                    pae_path = os.path.join(output_dir, f"{job_name}_pae_combined.png")
                    ribbon_path = os.path.join(output_dir, f"{job_name}_ribbons_combined.png")
                else:
                    matrix_path = os.path.join(folder, f"{job_name}_matrices_combined.png")
                    pae_path = os.path.join(folder, f"{job_name}_pae_combined.png")
                    ribbon_path = os.path.join(folder, f"{job_name}_ribbons_combined.png")

                # Combined PMC/contact_prob matrices
                generate_combined_matrix_png(
                    folder_folding_results,
                    matrix_path,
                    precomputed_data=precomputed_data,
                    dpi=config.dpi
                )
                print(f"  Wrote: {matrix_path}")

                # Combined PAE matrices
                generate_combined_pae_png(
                    folder_folding_results,
                    pae_path,
                    dpi=config.dpi
                )
                print(f"  Wrote: {pae_path}")

                # Combined ribbon plots with shared legend
                generate_combined_ribbon_png(
                    folder_folding_results,
                    ribbon_path,
                    precomputed_data=precomputed_data,
                    dpi=config.dpi
                )
                print(f"  Wrote: {ribbon_path}")

            except Exception as e:
                print(f"  Warning: Failed to generate combined graphics: {e}")
                import traceback
                traceback.print_exc()

        # Generate PDF report if requested
        if pdf and HAS_GRAPHICS:
            try:
                config = get_config()

                # Use precomputed data from PNG section if available, otherwise compute
                if not png:
                    # Need to compute precomputed_data (PNG section didn't run)
                    precomputed_data = {}
                    folder_folding_results = []

                    for model_num, (agg_res, contact_scores, folding_result) in model_results_map.items():
                        folder_folding_results.append(folding_result)

                        pmc = extract_pmc(folding_result)
                        contact_probs = extract_contact_probs(folding_result)
                        clusters = cluster_domains_from_result(folding_result)

                        interfaces = get_geometric_interfaces(
                            folding_result,
                            distance_threshold=dist_cutoff,
                            pae_threshold=pae_cutoff,
                            gap_merge=config.interface.gap_merge
                        )

                        proximity_contacts = get_proximity_contacts(
                            folding_result,
                            distance_threshold=dist_cutoff
                        )

                        precomputed_data[model_num] = {
                            'pmc': pmc,
                            'contact_probs': contact_probs,
                            'clusters': clusters,
                            'interfaces': interfaces,
                            'proximity_contacts': proximity_contacts,
                            'contact_scores': contact_scores,
                        }

                    folder_folding_results.sort(key=lambda r: r.model_num)
                # else: reuse precomputed_data and folder_folding_results from PNG section

                # Generate PDF
                if output_dir:
                    pdf_path = os.path.join(output_dir, f"{job_name}_report.pdf")
                else:
                    pdf_path = os.path.join(folder, f"{job_name}_report.pdf")

                report = ModelPDFReport(
                    folder_folding_results,
                    title=f"{job_name} Model Analysis",
                    precomputed_data=precomputed_data
                )
                report.generate(
                    pdf_path,
                    include_pae=True,
                    include_joint=True,
                    include_ribbon=True,
                )
                print(f"  Wrote: {pdf_path}")

            except Exception as e:
                print(f"  Warning: Failed to generate PDF report: {e}")
                import traceback
                traceback.print_exc()

        all_combined_results.extend(folder_results)
        all_per_contact_results.extend(folder_per_contact_results)

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
        backend
    )


if __name__ == "__main__":
    main()
