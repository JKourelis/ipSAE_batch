"""
Graphics module for ipSAE_batch.

This module provides plotting functions for:

- matrix_plot: PAE, PMC, PDE matrix visualization
- ribbon_plot: Circular ribbon diagrams with pLDDT/contact coloring
- pdf_export: Multi-page PDF report generation
- config: Configurable color schemes and settings

All components work with FoldingResult from data_readers.
"""

from .config import (
    GraphicsConfig,
    MatrixColorConfig,
    InterfaceConfig,
    RibbonConfig,
    get_config,
    set_config,
    load_config_from_csv,
    save_config_to_csv,
    create_default_config_csv,
)

from .matrix_plot import (
    plot_pae_matrix,
    plot_pmc_matrix,
    plot_pde_matrix,
    plot_contact_probability_matrix,
    plot_alphabridge_combined,
    plot_pmc_pde_combined,
    plot_pae_pde_combined,
    plot_combined_triangles,
    plot_all_matrices_separate,
    plot_model_matrices,
    _get_chain_boundaries,
)

from .ribbon_plot import (
    RibbonPlot,
    plot_ribbon,
    default_pae_to_color,
    plddt_to_color,
    set_pae_color_function,
    get_pae_color,
)

from .pdf_export import (
    ModelPDFReport,
    generate_model_images,
    generate_combined_figure,
    generate_model_comparison_page,
)

# Batch comparison (requires plotly)
try:
    from .batch_comparison import (
        generate_batch_comparison_html,
        generate_batch_comparison_from_results,
        get_best_models,
        calculate_correlation_matrix,
    )
    HAS_BATCH_COMPARISON = True
except ImportError:
    HAS_BATCH_COMPARISON = False

__all__ = [
    # Config
    'GraphicsConfig',
    'MatrixColorConfig',
    'InterfaceConfig',
    'RibbonConfig',
    'get_config',
    'set_config',
    'load_config_from_csv',
    'save_config_to_csv',
    'create_default_config_csv',
    # Matrix plots
    'plot_pae_matrix',
    'plot_pmc_matrix',
    'plot_pde_matrix',
    'plot_contact_probability_matrix',
    'plot_alphabridge_combined',
    'plot_pmc_pde_combined',
    'plot_pae_pde_combined',
    'plot_combined_triangles',
    'plot_all_matrices_separate',
    'plot_model_matrices',
    '_get_chain_boundaries',
    # Ribbon plots
    'RibbonPlot',
    'plot_ribbon',
    'default_pae_to_color',
    'plddt_to_color',
    'set_pae_color_function',
    'get_pae_color',
    # PDF export
    'ModelPDFReport',
    'generate_model_images',
    'generate_combined_figure',
    'generate_model_comparison_page',
    # Batch comparison
    'HAS_BATCH_COMPARISON',
    'generate_batch_comparison_html',
    'generate_batch_comparison_from_results',
    'get_best_models',
    'calculate_correlation_matrix',
]
