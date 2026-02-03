"""
Folder-level parallel processing for ipSAE_batch.

Option B: Simplified multiprocessing wrapper using Pool.imap().
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from multiprocessing import Pool, cpu_count


def process_single_folder(args: Tuple) -> Dict[str, Any]:
    """
    Process a single folder - module-level function for multiprocessing.

    MUST be module-level (not a method) for pickle serialization.
    Creates all instances fresh inside to avoid shared state issues.

    Args:
        args: Tuple of (folder_path, model_infos, config_dict)
              config_dict contains all processing parameters

    Returns:
        Dict with 'folder_results', 'per_contact_results', 'job_name', 'n_models', 'error'
    """
    folder_path, model_infos, config_dict = args

    # Extract config
    backend = config_dict['backend']
    pae_cutoff = config_dict['pae_cutoff']
    dist_cutoff = config_dict['dist_cutoff']
    output_dir = config_dict['output_dir']
    per_residue = config_dict['per_residue']
    per_contact = config_dict['per_contact']
    png = config_dict['png']
    pdf = config_dict['pdf']
    residue_selection = config_dict['residue_selection']
    graphics_config = config_dict.get('graphics_config')

    job_name = model_infos[0]['job_name'] if model_infos else os.path.basename(folder_path)

    result = {
        'folder_results': [],
        'per_contact_results': [],
        'job_name': job_name,
        'n_models': 0,
        'error': None,
    }

    try:
        # Import modules inside function for multiprocessing safety
        from ..data_readers import get_reader
        from ..core import calculate_scores_from_result
        from ..core.scoring import calculate_per_contact_scores
        from ..core.residue_selection import calculate_ipsae_selection_metrics
        from ..output_writers import write_aggregate_csv, write_per_residue_csv, write_per_contact_csv

        # Graphics imports (optional)
        HAS_GRAPHICS = False
        try:
            import matplotlib
            matplotlib.use('Agg')
            from ..extractors import extract_pmc, cluster_domains_from_result, get_geometric_interfaces, get_proximity_contacts
            from ..extractors.pde import extract_contact_probs
            from ..graphics.config import set_config, get_config, GraphicsConfig
            from ..graphics.pdf_export import (
                ModelPDFReport, generate_combined_matrix_png,
                generate_combined_pae_png, generate_combined_ribbon_png
            )
            HAS_GRAPHICS = True
        except ImportError:
            pass

        # Set graphics config if provided
        if graphics_config and HAS_GRAPHICS:
            set_config(graphics_config)

        # Create fresh reader instance
        reader = get_reader(backend)

        # Read all models
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
            result['error'] = 'No complete model sets found'
            return result

        result['n_models'] = len(models)

        # Process models (sequentially within folder - parallel at folder level)
        folder_results = []
        folder_per_res_results = []
        folder_per_contact_results = []
        model_results_map = {}

        need_per_contact = per_contact or png or pdf or residue_selection

        for model in models:
            model_num = model['model_num']
            folding_result = model['folding_result']

            try:
                agg_results, per_res_results = calculate_scores_from_result(
                    folding_result,
                    pae_cutoff,
                    dist_cutoff,
                    return_per_residue=per_residue
                )

                # Add job and model info
                for r in agg_results:
                    r['job_name'] = job_name
                    r['model'] = model_num
                    r['pae_cutoff'] = pae_cutoff
                    r['dist_cutoff'] = dist_cutoff

                if per_res_results:
                    for r in per_res_results:
                        r['job_name'] = job_name
                        r['model'] = model_num

                # Per-contact scores
                per_contact_res = None
                if need_per_contact:
                    per_contact_res = calculate_per_contact_scores(
                        folding_result, pae_cutoff=pae_cutoff, dist_cutoff=dist_cutoff
                    )
                    for c in per_contact_res:
                        c['job_name'] = job_name
                        c['model'] = model_num

                # Selection metrics
                if residue_selection and per_contact_res:
                    selection_metrics = calculate_ipsae_selection_metrics(
                        per_contact_res, residue_selection
                    )
                    for r in agg_results:
                        r.update(selection_metrics)

                folder_results.extend(agg_results)
                if per_res_results:
                    folder_per_res_results.extend(per_res_results)
                if per_contact_res:
                    folder_per_contact_results.extend(per_contact_res)

                model_results_map[model_num] = (agg_results, per_contact_res, folding_result)

            except Exception as e:
                print(f"  Error processing {job_name} model {model_num}: {e}")

        # Write per-folder output
        if output_dir:
            folder_output = os.path.join(output_dir, f"{job_name}_ipSAE.csv")
        else:
            folder_output = os.path.join(folder_path, f"{job_name}_ipSAE.csv")

        write_aggregate_csv(folder_results, folder_output)

        # Write per-residue output
        if per_residue and folder_per_res_results:
            if output_dir:
                per_res_output = os.path.join(output_dir, f"{job_name}_ipSAE_byres.csv")
            else:
                per_res_output = os.path.join(folder_path, f"{job_name}_ipSAE_byres.csv")
            write_per_residue_csv(folder_per_res_results, per_res_output)

        # Write per-contact output
        if per_contact and folder_per_contact_results:
            if output_dir:
                per_contact_output = os.path.join(output_dir, f"{job_name}_ipSAE_contacts.csv")
            else:
                per_contact_output = os.path.join(folder_path, f"{job_name}_ipSAE_contacts.csv")
            write_per_contact_csv(folder_per_contact_results, per_contact_output)

        # Generate graphics
        if (png or pdf) and HAS_GRAPHICS:
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
                    pass  # Skip model if precompute fails

            folder_folding_results.sort(key=lambda r: r.model_num)

            # Generate PNG
            if png:
                try:
                    if output_dir:
                        matrix_path = os.path.join(output_dir, f"{job_name}_matrices_combined.png")
                        pae_path = os.path.join(output_dir, f"{job_name}_pae_combined.png")
                        ribbon_path = os.path.join(output_dir, f"{job_name}_ribbons_combined.png")
                    else:
                        matrix_path = os.path.join(folder_path, f"{job_name}_matrices_combined.png")
                        pae_path = os.path.join(folder_path, f"{job_name}_pae_combined.png")
                        ribbon_path = os.path.join(folder_path, f"{job_name}_ribbons_combined.png")

                    generate_combined_matrix_png(
                        folder_folding_results, matrix_path,
                        precomputed_data=precomputed_data, dpi=config.dpi
                    )
                    generate_combined_pae_png(
                        folder_folding_results, pae_path, dpi=config.dpi
                    )
                    generate_combined_ribbon_png(
                        folder_folding_results, ribbon_path,
                        precomputed_data=precomputed_data, dpi=config.dpi
                    )
                except Exception as e:
                    pass  # Continue on graphics failure

            # Generate PDF
            if pdf:
                try:
                    if output_dir:
                        pdf_path = os.path.join(output_dir, f"{job_name}_report.pdf")
                    else:
                        pdf_path = os.path.join(folder_path, f"{job_name}_report.pdf")

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
                except Exception as e:
                    pass  # Continue on PDF failure

        result['folder_results'] = folder_results
        result['per_contact_results'] = folder_per_contact_results if per_contact else []

    except Exception as e:
        import traceback
        result['error'] = f"{e}\n{traceback.format_exc()}"

    return result


def process_folders_parallel(
    job_folders_with_models: List[Tuple[str, List[Dict]]],
    config_dict: Dict[str, Any],
    cores: int = None,
) -> Tuple[List[Dict], List[Dict], int]:
    """
    Process multiple folders in parallel.

    Args:
        job_folders_with_models: List of (folder_path, model_infos) tuples
        config_dict: Dict with all processing parameters
        cores: Number of CPU cores to use (default: cpu_count - 1)

    Returns:
        (all_combined_results, all_per_contact_results, total_models)
    """
    if cores is None:
        cores = max(1, cpu_count() - 1)
    cores = min(cores, len(job_folders_with_models))

    # Prepare arguments for each folder
    job_args = [
        (folder_path, model_infos, config_dict)
        for folder_path, model_infos in job_folders_with_models
    ]

    all_combined_results = []
    all_per_contact_results = []
    total_models = 0

    if cores > 1:
        print(f"\nProcessing {len(job_folders_with_models)} folders in parallel ({cores} cores)...")

        # Use imap for progress tracking
        from tqdm import tqdm
        with Pool(processes=cores) as pool:
            results = list(tqdm(
                pool.imap(process_single_folder, job_args),
                total=len(job_args),
                desc="Folders"
            ))

        for res in results:
            if res['error']:
                print(f"  {res['job_name']}: Error - {res['error']}")
            else:
                all_combined_results.extend(res['folder_results'])
                all_per_contact_results.extend(res['per_contact_results'])
                total_models += res['n_models']
    else:
        # Sequential fallback
        for args in job_args:
            job_name = args[1][0]['job_name'] if args[1] else os.path.basename(args[0])
            print(f"\nProcessing: {job_name}")

            res = process_single_folder(args)

            if res['error']:
                print(f"  Error: {res['error']}")
            else:
                print(f"  Processed {res['n_models']} models")
                all_combined_results.extend(res['folder_results'])
                all_per_contact_results.extend(res['per_contact_results'])
                total_models += res['n_models']

    return all_combined_results, all_per_contact_results, total_models
