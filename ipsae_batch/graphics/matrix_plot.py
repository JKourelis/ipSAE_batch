"""
Matrix plotting module.

Provides functions for plotting PAE, PMC, PDE matrices individually
and in various combinations. Uses configurable color schemes.
Works with FoldingResult from data_readers.
"""

from typing import Optional, Tuple, List
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from ..data_readers import FoldingResult


def _check_matplotlib():
    """Check if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install matplotlib"
        )


def _get_chain_boundaries(result: FoldingResult) -> Tuple[List[int], List[str]]:
    """Get chain boundaries and labels from FoldingResult."""
    chain_boundaries = [0]
    chain_labels = []

    for chain_id in result.unique_chains:
        indices = result.get_chain_indices(chain_id)
        chain_boundaries.append(chain_boundaries[-1] + len(indices))
        chain_labels.append(chain_id)

    return chain_boundaries, chain_labels


def _add_chain_boundaries(
    ax: "plt.Axes",
    chain_boundaries: List[int],
    chain_labels: List[str],
    fontsize: int = 10
) -> None:
    """Add chain boundary lines and labels to axes."""
    # Add boundary lines
    for boundary in chain_boundaries[1:]:
        ax.axhline(y=boundary - 0.5, color='black', linewidth=0.5, alpha=0.7)
        ax.axvline(x=boundary - 0.5, color='black', linewidth=0.5, alpha=0.7)

    # Add chain labels
    if chain_labels and chain_boundaries:
        midpoints = []
        for i in range(len(chain_boundaries) - 1):
            start = chain_boundaries[i]
            end = chain_boundaries[i + 1]
            midpoints.append((start + end) / 2)

        if midpoints:
            ax.set_xticks(midpoints)
            ax.set_xticklabels(chain_labels[:len(midpoints)], fontsize=fontsize)
            ax.set_yticks(midpoints)
            ax.set_yticklabels(chain_labels[:len(midpoints)], fontsize=fontsize)


def _get_tick_interval(chain_length: int) -> int:
    """
    Get appropriate tick interval based on chain length.
    Uses same logic as ribbon plot from config.
    """
    from .config import get_config
    config = get_config()

    tick_interval = config.ribbon.tick_interval  # Fallback
    for max_len, interval in config.ribbon.tick_intervals:
        if chain_length <= max_len:
            tick_interval = interval
            break
    return tick_interval


def _generate_matrix_ticks(
    chain_boundaries: List[int],
    chain_labels: List[str],
    gap_size: int = 0,
    for_x_axis: bool = False
) -> Tuple[List[float], List[str], List[int]]:
    """
    Generate tick positions and labels for matrix axes.

    At position 0 of each chain: shows chain label with direction arrow (▼)
    - X-axis (top, rotated): "A▼" so arrow appears above chain name
    - Y-axis (left, horizontal): "▼A" so arrow indicates downward direction
    At other positions: shows residue number

    Args:
        chain_boundaries: Original chain boundaries [0, len_A, len_A+len_B, ...]
        chain_labels: Chain identifiers ['A', 'B', ...]
        gap_size: Gap inserted between chains in expanded matrix
        for_x_axis: True for x-axis format, False for y-axis format

    Returns:
        Tuple of (tick_positions, tick_labels, chain_start_indices)
        chain_start_indices: list of indices in tick_positions that are chain starts (for bold styling)
    """
    tick_positions = []
    tick_labels = []
    chain_start_indices = []

    n_chains = len(chain_labels)

    for i in range(n_chains):
        chain_start_orig = chain_boundaries[i]
        chain_end_orig = chain_boundaries[i + 1]
        chain_length = chain_end_orig - chain_start_orig

        # Calculate offset due to gaps
        gap_offset = i * gap_size

        # Get tick interval for this chain length
        tick_interval = _get_tick_interval(chain_length)

        # Generate tick positions within this chain
        for pos in range(0, chain_length, tick_interval):
            # Position in expanded matrix
            matrix_pos = chain_start_orig + pos + gap_offset
            tick_positions.append(matrix_pos)

            if pos == 0:
                # Chain start: show chain label with arrow
                # Arrow ▼ indicates N→C direction (downward/rightward)
                chain_start_indices.append(len(tick_labels))
                if for_x_axis:
                    # X-axis (rotated 90°): "A▼" puts arrow above chain name
                    tick_labels.append(f"{chain_labels[i]}▼")
                else:
                    # Y-axis (horizontal): "▼A" puts arrow before chain name
                    tick_labels.append(f"▼{chain_labels[i]}")
            else:
                # Regular position: show residue number
                tick_labels.append(str(pos))

    return tick_positions, tick_labels, chain_start_indices


def _expand_matrix_with_gaps(
    matrix: np.ndarray,
    chain_boundaries: List[int],
    gap_size: int = 3,
    gap_value: float = np.nan
) -> Tuple[np.ndarray, List[int]]:
    """
    Expand a matrix by inserting gaps between chains.

    Args:
        matrix: Original NxN matrix
        chain_boundaries: List of boundary indices [0, len_A, len_A+len_B, ...]
        gap_size: Number of rows/cols to insert as gap
        gap_value: Value to fill gaps with (np.nan renders as white)

    Returns:
        Tuple of (expanded_matrix, new_boundaries)
    """
    n_chains = len(chain_boundaries) - 1
    if n_chains <= 1:
        return matrix, chain_boundaries

    n_gaps = n_chains - 1
    old_n = matrix.shape[0]
    new_n = old_n + n_gaps * gap_size

    # Create expanded matrix filled with gap value
    expanded = np.full((new_n, new_n), gap_value, dtype=float)

    # Calculate new boundaries (accounting for gaps)
    new_boundaries = [0]
    for i in range(1, len(chain_boundaries)):
        # Original boundary + gaps inserted before this point
        new_boundaries.append(chain_boundaries[i] + (i - 1) * gap_size + gap_size)
    # Adjust last boundary (no gap after last chain)
    new_boundaries[-1] = chain_boundaries[-1] + (n_chains - 1) * gap_size

    # Copy data blocks into expanded matrix
    for i in range(n_chains):
        for j in range(n_chains):
            # Source block coordinates
            src_row_start = chain_boundaries[i]
            src_row_end = chain_boundaries[i + 1]
            src_col_start = chain_boundaries[j]
            src_col_end = chain_boundaries[j + 1]

            # Destination block coordinates (shifted by gaps)
            dst_row_start = src_row_start + i * gap_size
            dst_row_end = src_row_end + i * gap_size
            dst_col_start = src_col_start + j * gap_size
            dst_col_end = src_col_end + j * gap_size

            # Copy the block
            expanded[dst_row_start:dst_row_end, dst_col_start:dst_col_end] = \
                matrix[src_row_start:src_row_end, src_col_start:src_col_end]

    return expanded, new_boundaries


# =============================================================================
# Individual Matrix Plots
# =============================================================================

def plot_pae_matrix(
    pae: np.ndarray,
    ax: Optional["plt.Axes"] = None,
    cmap: Optional[str] = None,
    vmin: float = 0,
    vmax: Optional[float] = None,
    chain_boundaries: Optional[List[int]] = None,
    chain_labels: Optional[List[str]] = None,
    title: str = "PAE",
    show_colorbar: bool = True,
) -> Tuple["Figure", "plt.Axes"]:
    """
    Plot a PAE matrix with configurable colors.

    Args:
        pae: PAE matrix (NxN)
        ax: Matplotlib axes to plot on (creates new if None)
        cmap: Colormap name (uses config default if None)
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap (uses config default if None)
        chain_boundaries: List of residue indices where chains start
        chain_labels: Labels for each chain
        title: Plot title
        show_colorbar: Whether to show colorbar

    Returns:
        Tuple of (figure, axes)
    """
    _check_matplotlib()
    from .config import get_config

    config = get_config()
    if cmap is None:
        cmap = config.matrix.pae_cmap
    if vmax is None:
        vmax = config.matrix.pae_vmax

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    im = ax.imshow(pae, cmap=cmap, vmin=vmin, vmax=vmax, origin='upper')
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Scored residue")
    ax.set_ylabel("Aligned residue")

    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Expected position error (Å)")

    if chain_boundaries:
        _add_chain_boundaries(ax, chain_boundaries, chain_labels or [])

    return fig, ax


def plot_pae_matrix_with_gaps(
    result: "FoldingResult",
    ax: Optional["plt.Axes"] = None,
    cmap: Optional[str] = None,
    vmax: Optional[float] = None,
    title: Optional[str] = None,
) -> Tuple["Figure", "plt.Axes"]:
    """
    Plot a PAE matrix with gaps between chains.

    Uses the same gap configuration as the combined matrix plots.

    Args:
        result: FoldingResult instance
        ax: Matplotlib axes to plot on (creates new if None)
        cmap: Colormap name (uses config default if None)
        vmax: Maximum value for colormap (uses config default if None)
        title: Plot title

    Returns:
        Tuple of (figure, axes)
    """
    _check_matplotlib()
    from .config import get_config
    from matplotlib.patches import Rectangle
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    config = get_config()
    if cmap is None:
        cmap = config.matrix.pae_cmap
    if vmax is None:
        vmax = config.matrix.pae_vmax

    pae = result.pae_matrix
    chain_boundaries, chain_labels = _get_chain_boundaries(result)

    # Calculate gap size from config
    total_size = pae.shape[0]
    if config.matrix.chain_gap_percent > 0:
        gap_size = max(1, int(total_size * config.matrix.chain_gap_percent / 100))
    else:
        gap_size = config.matrix.chain_gap_size

    # Expand matrix with gaps
    pae_expanded, new_boundaries = _expand_matrix_with_gaps(
        pae, chain_boundaries, gap_size=gap_size
    )

    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.figure

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.5)

    # Plot PAE
    im = ax.imshow(pae_expanded, cmap=cmap, vmin=0, vmax=vmax, origin='upper')

    # Draw boxes around ALL chain-pair blocks
    # Use original chain_boundaries with gap offsets to get correct data block positions
    n_chains = len(chain_labels)
    for i in range(n_chains):
        for j in range(n_chains):
            # Calculate actual data block positions using original boundaries + gap offset
            row_start = chain_boundaries[i] + i * gap_size
            row_end = chain_boundaries[i + 1] + i * gap_size
            col_start = chain_boundaries[j] + j * gap_size
            col_end = chain_boundaries[j + 1] + j * gap_size
            height = row_end - row_start
            width = col_end - col_start
            rect = Rectangle(
                (col_start - 0.5, row_start - 0.5), width, height,
                linewidth=1.5, edgecolor='black', facecolor='none', zorder=10
            )
            ax.add_patch(rect)

    # Generate tick positions and labels with amino acid numbering
    # X-axis (top)
    x_positions, x_labels, x_chain_starts = _generate_matrix_ticks(
        chain_boundaries, chain_labels, gap_size, for_x_axis=True
    )
    # Y-axis (left)
    y_positions, y_labels, y_chain_starts = _generate_matrix_ticks(
        chain_boundaries, chain_labels, gap_size, for_x_axis=False
    )

    # X-axis ticks on TOP
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=7, rotation=90, ha='center')

    # Y-axis ticks on LEFT
    ax.yaxis.set_ticks_position('left')
    ax.yaxis.set_label_position('left')
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=7, rotation=0, ha='right')

    # Make chain start labels bold
    for idx in x_chain_starts:
        ax.get_xticklabels()[idx].set_fontweight('bold')
    for idx in y_chain_starts:
        ax.get_yticklabels()[idx].set_fontweight('bold')

    # Add padding between ticks and plot border
    ax.tick_params(axis='both', which='major', pad=3, length=3)

    # Colorbar
    fig.colorbar(im, cax=cax, label='PAE (Å)')

    # Title
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f'PAE - {result.job_name} model {result.model_num}', fontsize=14)

    return fig, ax


def plot_pmc_matrix(
    pmc: np.ndarray,
    ax: Optional["plt.Axes"] = None,
    cmap: Optional[str] = None,
    vmin: float = 0,
    vmax: Optional[float] = None,
    chain_boundaries: Optional[List[int]] = None,
    chain_labels: Optional[List[str]] = None,
    title: str = "PMC",
    show_colorbar: bool = True,
) -> Tuple["Figure", "plt.Axes"]:
    """
    Plot a PMC (Predicted Merged Confidence) matrix.

    Args:
        pmc: PMC matrix (NxN)
        ax: Matplotlib axes to plot on
        cmap: Colormap name (uses config default if None)
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        chain_boundaries: Residue indices where chains start
        chain_labels: Labels for each chain
        title: Plot title
        show_colorbar: Whether to show colorbar

    Returns:
        Tuple of (figure, axes)
    """
    _check_matplotlib()
    from .config import get_config

    config = get_config()
    if cmap is None:
        cmap = config.matrix.pmc_cmap
    if vmax is None:
        vmax = config.matrix.pmc_vmax

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    im = ax.imshow(pmc, cmap=cmap, vmin=vmin, vmax=vmax, origin='upper')
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Residue index")
    ax.set_ylabel("Residue index")

    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Confidence score")

    if chain_boundaries:
        _add_chain_boundaries(ax, chain_boundaries, chain_labels or [])

    return fig, ax


def plot_pde_matrix(
    pde: np.ndarray,
    ax: Optional["plt.Axes"] = None,
    cmap: Optional[str] = None,
    vmin: float = 0,
    vmax: Optional[float] = None,
    chain_boundaries: Optional[List[int]] = None,
    chain_labels: Optional[List[str]] = None,
    title: str = "PDE",
    show_colorbar: bool = True,
) -> Tuple["Figure", "plt.Axes"]:
    """
    Plot a PDE (Predicted Distance Error) matrix.

    Args:
        pde: PDE matrix (NxN)
        ax: Matplotlib axes to plot on
        cmap: Colormap name (uses config default if None)
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        chain_boundaries: Residue indices where chains start
        chain_labels: Labels for each chain
        title: Plot title
        show_colorbar: Whether to show colorbar

    Returns:
        Tuple of (figure, axes)
    """
    _check_matplotlib()
    from .config import get_config

    config = get_config()
    if cmap is None:
        cmap = config.matrix.pde_cmap
    if vmax is None:
        vmax = config.matrix.pde_vmax

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    im = ax.imshow(pde, cmap=cmap, vmin=vmin, vmax=vmax, origin='upper')
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Residue index")
    ax.set_ylabel("Residue index")

    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Distance error (Å)")

    if chain_boundaries:
        _add_chain_boundaries(ax, chain_boundaries, chain_labels or [])

    return fig, ax


def plot_contact_probability_matrix(
    contact_probs: np.ndarray,
    ax: Optional["plt.Axes"] = None,
    cmap: Optional[str] = None,
    vmin: float = 0,
    vmax: float = 1.0,
    chain_boundaries: Optional[List[int]] = None,
    chain_labels: Optional[List[str]] = None,
    title: str = "Contact Probability",
    show_colorbar: bool = True,
) -> Tuple["Figure", "plt.Axes"]:
    """
    Plot a contact probability matrix.

    Contact probability is native from AF3 or derived from PDE for Boltz2.
    Values range from 0 (no contact) to 1 (confident contact).

    Args:
        contact_probs: Contact probability matrix (NxN), values 0-1
        ax: Matplotlib axes to plot on
        cmap: Colormap name (uses RdPu if None)
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        chain_boundaries: Residue indices where chains start
        chain_labels: Labels for each chain
        title: Plot title
        show_colorbar: Whether to show colorbar

    Returns:
        Tuple of (figure, axes)
    """
    _check_matplotlib()
    from .config import get_config

    config = get_config()
    if cmap is None:
        cmap = config.matrix.pde_cmap  # Use same colormap as PDE (RdPu)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    im = ax.imshow(contact_probs, cmap=cmap, vmin=vmin, vmax=vmax, origin='upper')
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Residue index")
    ax.set_ylabel("Residue index")

    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Contact probability")

    if chain_boundaries:
        _add_chain_boundaries(ax, chain_boundaries, chain_labels or [])

    return fig, ax


# =============================================================================
# Combined Triangle Plots
# =============================================================================

def plot_alphabridge_combined(
    result: "FoldingResult",
    pmc_matrix: Optional[np.ndarray] = None,
    contact_probs: Optional[np.ndarray] = None,
    clusters: Optional[np.ndarray] = None,
    ax: Optional["plt.Axes"] = None,
    show_clusters: bool = True,
    title: Optional[str] = None,
    upper_metric: Optional[str] = None,
    lower_metric: Optional[str] = None,
) -> Tuple["Figure", "plt.Axes"]:
    """
    Plot combined matrix in AlphaBridge style.

    Upper/lower triangle metrics are configurable via:
    - Function parameters (upper_metric, lower_metric)
    - Config CSV settings (joint_matrix_upper, joint_matrix_lower)

    Available metrics: 'pae', 'pmc', 'contact_prob'

    Includes:
    - Chain labels on top with cluster coloring (if provided)
    - Chain boundary lines
    - Proper colorbars for both matrices

    IMPORTANT: This function does NOT recalculate matrices. All data must be
    pre-computed and passed in. Use extractors to compute data before calling.

    Args:
        result: FoldingResult instance (for chain info and PAE)
        pmc_matrix: Pre-computed PMC matrix (required if using 'pmc')
        contact_probs: Pre-computed contact probability matrix (required if using 'contact_prob')
        clusters: Pre-computed Leiden cluster assignments (array of cluster IDs per residue)
        ax: Matplotlib axes to plot on
        show_clusters: Show Leiden cluster coloring on top bar
        title: Plot title
        upper_metric: Override config for upper triangle ('pae', 'pmc', 'contact_prob')
        lower_metric: Override config for lower triangle ('pae', 'pmc', 'contact_prob')

    Returns:
        Tuple of (figure, axes)
    """
    _check_matplotlib()
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.colors as mcolors
    from .config import get_config

    config = get_config()

    # Get metrics from config or parameters
    upper_type = upper_metric or config.matrix.joint_matrix_upper
    lower_type = lower_metric or config.matrix.joint_matrix_lower

    # Fallback if contact_probs not available
    if contact_probs is None:
        if upper_type == 'contact_prob':
            upper_type = 'pmc'  # Fallback to PMC
        if lower_type == 'contact_prob':
            lower_type = 'pae'  # Fallback to PAE

    def get_matrix_info(metric_type: str):
        """Get matrix, colormap, vmax, and label for a metric type."""
        if metric_type == 'pae':
            matrix = result.pae_matrix
            cmap = config.matrix.pae_cmap
            vmax = config.matrix.pae_vmax
            label = "PAE"
        elif metric_type == 'pmc':
            if pmc_matrix is None:
                raise ValueError("pmc_matrix is required when using 'pmc'. "
                               "Use extractors.extract_pmc() to compute it first.")
            matrix = pmc_matrix
            cmap = config.matrix.pmc_cmap
            vmax = config.matrix.pmc_vmax
            label = "PMC"
        elif metric_type == 'contact_prob':
            if contact_probs is None:
                raise ValueError("contact_probs is required when using 'contact_prob'. "
                               "Use extractors.pde.extract_contact_probs() to compute it first.")
            matrix = contact_probs
            cmap = config.matrix.pde_cmap
            vmax = 1.0
            label = "Contact Prob"
        else:
            raise ValueError(f"Unknown metric type: {metric_type}. "
                           "Options: 'pae', 'pmc', 'contact_prob'")
        return matrix, cmap, vmax, label

    # Get upper and lower matrix info
    upper_matrix, upper_cmap, upper_vmax, upper_label = get_matrix_info(upper_type)
    lower_matrix, lower_cmap, lower_vmax, lower_label = get_matrix_info(lower_type)

    if upper_matrix is None:
        raise ValueError(f"No matrix available for upper triangle ({upper_type})")
    if lower_matrix is None:
        raise ValueError(f"No matrix available for lower triangle ({lower_type})")

    chain_boundaries, chain_labels_list = _get_chain_boundaries(result)

    # Calculate gap size from config
    # If chain_gap_percent > 0, use percentage of total matrix size
    # Otherwise use absolute chain_gap_size
    total_size = upper_matrix.shape[0]
    if config.matrix.chain_gap_percent > 0:
        gap_size = max(1, int(total_size * config.matrix.chain_gap_percent / 100))
    else:
        gap_size = config.matrix.chain_gap_size

    # Expand matrices with gaps between chains
    upper_expanded, new_boundaries = _expand_matrix_with_gaps(
        upper_matrix, chain_boundaries, gap_size=gap_size
    )
    lower_expanded, _ = _expand_matrix_with_gaps(
        lower_matrix, chain_boundaries, gap_size=gap_size
    )

    n = upper_expanded.shape[0]

    # Create figure with proper axes for colorbars
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 12))
    else:
        fig = ax.figure

    divider = make_axes_locatable(ax)
    cax_right = divider.append_axes("right", size="5%", pad=0.5)
    cax_bottom = divider.append_axes("bottom", size="5%", pad=0.5)

    # Create masked matrices for upper/lower triangles
    # Lower triangle
    mask_upper = np.triu(np.ones((n, n), dtype=bool), k=0)
    lower_masked = np.ma.array(lower_expanded, mask=mask_upper)

    # Upper triangle
    mask_lower = np.tril(np.ones((n, n), dtype=bool), k=0)
    upper_masked = np.ma.array(upper_expanded, mask=mask_lower)

    # Plot lower triangle
    im_lower = ax.imshow(lower_masked, cmap=lower_cmap,
                         vmin=0, vmax=lower_vmax, origin='upper')

    # Plot upper triangle
    im_upper = ax.imshow(upper_masked, cmap=upper_cmap,
                         vmin=0, vmax=upper_vmax, origin='upper')

    # Draw boxes around ALL chain-pair blocks (not just diagonal)
    # Use original chain_boundaries with gap offsets to get correct data block positions
    from matplotlib.patches import Rectangle
    n_chains = len(chain_labels_list)
    for i in range(n_chains):
        for j in range(n_chains):
            # Calculate actual data block positions using original boundaries + gap offset
            row_start = chain_boundaries[i] + i * gap_size
            row_end = chain_boundaries[i + 1] + i * gap_size
            col_start = chain_boundaries[j] + j * gap_size
            col_end = chain_boundaries[j + 1] + j * gap_size
            height = row_end - row_start
            width = col_end - col_start
            # Draw rectangle around this chain-pair block
            rect = Rectangle(
                (col_start - 0.5, row_start - 0.5), width, height,
                linewidth=1.5, edgecolor='black', facecolor='none', zorder=10
            )
            ax.add_patch(rect)

    # Generate tick positions and labels with amino acid numbering
    # X-axis (top)
    x_positions, x_labels, x_chain_starts = _generate_matrix_ticks(
        chain_boundaries, chain_labels_list, gap_size, for_x_axis=True
    )
    # Y-axis (left)
    y_positions, y_labels, y_chain_starts = _generate_matrix_ticks(
        chain_boundaries, chain_labels_list, gap_size, for_x_axis=False
    )

    # X-axis ticks on TOP (for upper triangle / PMC readability)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=7, rotation=90, ha='center')

    # Y-axis ticks on LEFT
    ax.yaxis.set_ticks_position('left')
    ax.yaxis.set_label_position('left')
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=7, rotation=0, ha='right')

    # Make chain start labels bold
    for idx in x_chain_starts:
        ax.get_xticklabels()[idx].set_fontweight('bold')
    for idx in y_chain_starts:
        ax.get_yticklabels()[idx].set_fontweight('bold')

    # Add padding between ticks and plot border
    ax.tick_params(axis='both', which='major', pad=3, length=3)

    # Add colorbars
    upper_vmax_str = f"{upper_vmax:.0f}" if upper_vmax >= 1 else f"{upper_vmax:.1f}"
    lower_vmax_str = f"{lower_vmax:.0f}" if lower_vmax >= 1 else f"{lower_vmax:.1f}"
    fig.colorbar(im_upper, cax=cax_right, label=f'{upper_label} (0-{upper_vmax_str})')
    fig.colorbar(im_lower, cax=cax_bottom, orientation='horizontal',
                 label=f'{lower_label} (0-{lower_vmax_str})')

    # Set title - use ax.set_title when ax was provided (grid context), suptitle otherwise
    if title:
        ax.set_title(title, fontsize=12, pad=10)
    else:
        ax.set_title(f'{upper_label} (upper) / {lower_label} (lower)', fontsize=12, pad=10)

    return fig, ax


def plot_combined_triangles(
    upper_matrix: np.ndarray,
    lower_matrix: np.ndarray,
    ax: Optional["plt.Axes"] = None,
    upper_cmap: Optional[str] = None,
    lower_cmap: Optional[str] = None,
    upper_vmax: float = 31.75,
    lower_vmax: float = 10.0,
    upper_label: str = "PAE",
    lower_label: str = "PDE",
    chain_boundaries: Optional[List[int]] = None,
    chain_labels: Optional[List[str]] = None,
    title: str = "PAE (upper) / PDE (lower)",
) -> Tuple["Figure", "plt.Axes"]:
    """
    Plot combined triangle matrix with configurable colors.

    Args:
        upper_matrix: Matrix for upper triangle (NxN)
        lower_matrix: Matrix for lower triangle (NxN)
        ax: Matplotlib axes to plot on
        upper_cmap: Colormap for upper triangle
        lower_cmap: Colormap for lower triangle
        upper_vmax: Maximum value for upper colormap
        lower_vmax: Maximum value for lower colormap
        upper_label: Label for upper triangle legend
        lower_label: Label for lower triangle legend
        chain_boundaries: Residue indices where chains start
        chain_labels: Labels for each chain
        title: Plot title

    Returns:
        Tuple of (figure, axes)
    """
    _check_matplotlib()
    from .config import get_config

    config = get_config()

    n = upper_matrix.shape[0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.figure

    # Normalize to 0-1 for color mapping
    upper_norm = np.clip(upper_matrix / upper_vmax, 0, 1)
    lower_norm = np.clip(lower_matrix / lower_vmax, 0, 1)

    # Get colormaps
    upper_cmap_obj = plt.cm.get_cmap(upper_cmap or config.matrix.pae_cmap)
    lower_cmap_obj = plt.cm.get_cmap(lower_cmap or config.matrix.pde_cmap)

    # Create RGB image
    rgb = np.zeros((n, n, 3))

    # Apply upper colormap to upper triangle
    for i in range(n):
        for j in range(i + 1, n):
            rgb[i, j, :] = upper_cmap_obj(upper_norm[i, j])[:3]

    # Apply lower colormap to lower triangle
    for i in range(n):
        for j in range(i):
            rgb[i, j, :] = lower_cmap_obj(lower_norm[i, j])[:3]

    # Diagonal
    for i in range(n):
        rgb[i, i, :] = [0.8, 0.8, 0.8]

    ax.imshow(rgb, origin='upper', aspect='equal')
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Residue index")
    ax.set_ylabel("Residue index")

    if chain_boundaries:
        _add_chain_boundaries(ax, chain_boundaries, chain_labels or [])

    # Create legend
    legend_elements = [
        Patch(facecolor=upper_cmap_obj(0.7)[:3], label=f'{upper_label} (0-{upper_vmax:.0f})'),
        Patch(facecolor=lower_cmap_obj(0.7)[:3], label=f'{lower_label} (0-{lower_vmax:.0f})'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    return fig, ax


def plot_pae_pde_combined(
    pae: np.ndarray,
    pde: np.ndarray,
    ax: Optional["plt.Axes"] = None,
    chain_boundaries: Optional[List[int]] = None,
    chain_labels: Optional[List[str]] = None,
    title: str = "PAE (upper) / PDE (lower)",
) -> Tuple["Figure", "plt.Axes"]:
    """
    Plot combined PAE (upper) and PDE (lower) triangle matrix.

    Args:
        pae: PAE matrix (NxN)
        pde: PDE matrix (NxN)
        ax: Matplotlib axes to plot on
        chain_boundaries: Residue indices where chains start
        chain_labels: Labels for each chain
        title: Plot title

    Returns:
        Tuple of (figure, axes)
    """
    from .config import get_config
    config = get_config()

    return plot_combined_triangles(
        upper_matrix=pae,
        lower_matrix=pde,
        ax=ax,
        upper_cmap=config.matrix.pae_cmap,
        lower_cmap=config.matrix.pde_cmap,
        upper_vmax=config.matrix.pae_vmax,
        lower_vmax=config.matrix.pde_vmax,
        upper_label="PAE",
        lower_label="PDE",
        chain_boundaries=chain_boundaries,
        chain_labels=chain_labels,
        title=title,
    )


def plot_pmc_pde_combined(
    pmc: np.ndarray,
    pde: Optional[np.ndarray] = None,
    ax: Optional["plt.Axes"] = None,
    pmc_cmap: Optional[str] = None,
    pde_cmap: Optional[str] = None,
    pmc_vmax: Optional[float] = None,
    pde_vmax: Optional[float] = None,
    chain_boundaries: Optional[List[int]] = None,
    chain_labels: Optional[List[str]] = None,
    title: str = "PMC (upper) / PDE (lower)",
) -> Tuple["Figure", "plt.Axes"]:
    """
    Plot combined PMC (upper) and PDE (lower) triangle matrix.

    Args:
        pmc: PMC matrix (NxN)
        pde: PDE matrix (NxN), uses PMC if None
        ax: Matplotlib axes to plot on
        pmc_cmap: Colormap for PMC
        pde_cmap: Colormap for PDE
        pmc_vmax: Maximum value for PMC
        pde_vmax: Maximum value for PDE
        chain_boundaries: Residue indices where chains start
        chain_labels: Labels for each chain
        title: Plot title

    Returns:
        Tuple of (figure, axes)
    """
    from .config import get_config
    config = get_config()

    if pde is None:
        pde = pmc

    return plot_combined_triangles(
        upper_matrix=pmc,
        lower_matrix=pde,
        ax=ax,
        upper_cmap=pmc_cmap or config.matrix.pmc_cmap,
        lower_cmap=pde_cmap or config.matrix.pde_cmap,
        upper_vmax=pmc_vmax or config.matrix.pmc_vmax,
        lower_vmax=pde_vmax or config.matrix.pde_vmax,
        upper_label="PMC",
        lower_label="PDE",
        chain_boundaries=chain_boundaries,
        chain_labels=chain_labels,
        title=title,
    )


# =============================================================================
# Convenience Functions
# =============================================================================

def plot_all_matrices_separate(
    result: FoldingResult,
    pmc_matrix: Optional[np.ndarray] = None,
    pde_matrix: Optional[np.ndarray] = None,
    include_pae: bool = True,
    include_pmc: bool = True,
    include_pde: bool = True,
    figsize_per_plot: Tuple[int, int] = (8, 8),
) -> "Figure":
    """
    Plot all matrices as separate side-by-side plots.

    IMPORTANT: This function does NOT recalculate matrices. All data must be
    pre-computed and passed in. Use extractors to compute data before calling.

    Args:
        result: FoldingResult instance (for chain info and PAE)
        pmc_matrix: Pre-computed PMC matrix (required if include_pmc=True)
        pde_matrix: Pre-computed PDE/contact prob matrix (required if include_pde=True)
        include_pae: Include PAE plot
        include_pmc: Include PMC plot
        include_pde: Include PDE plot
        figsize_per_plot: Size per individual plot

    Returns:
        Figure with side-by-side plots
    """
    _check_matplotlib()

    chain_boundaries, chain_labels = _get_chain_boundaries(result)

    plots = []
    if include_pae and result.pae_matrix is not None:
        plots.append(('pae', result.pae_matrix))
    if include_pmc and pmc_matrix is not None:
        plots.append(('pmc', pmc_matrix))
    if include_pde and pde_matrix is not None:
        plots.append(('pde', pde_matrix))

    if not plots:
        return None

    n_plots = len(plots)
    fig, axes = plt.subplots(1, n_plots, figsize=(figsize_per_plot[0] * n_plots, figsize_per_plot[1]))
    if n_plots == 1:
        axes = [axes]

    for i, (plot_type, matrix) in enumerate(plots):
        if plot_type == 'pae':
            plot_pae_matrix(matrix, ax=axes[i], chain_boundaries=chain_boundaries,
                           chain_labels=chain_labels,
                           title=f"PAE - {result.job_name} model {result.model_num}")
        elif plot_type == 'pmc':
            plot_pmc_matrix(matrix, ax=axes[i], chain_boundaries=chain_boundaries,
                           chain_labels=chain_labels,
                           title=f"PMC - {result.job_name} model {result.model_num}")
        elif plot_type == 'pde':
            plot_pde_matrix(matrix, ax=axes[i], chain_boundaries=chain_boundaries,
                           chain_labels=chain_labels,
                           title=f"PDE - {result.job_name} model {result.model_num}")

    plt.tight_layout()
    return fig


def plot_model_matrices(
    result: FoldingResult,
    pmc_matrix: Optional[np.ndarray] = None,
    pde_matrix: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    include_pae: bool = True,
    include_pmc_pde: bool = True,
    dpi: int = 150,
) -> Optional["Figure"]:
    """
    Create a comprehensive matrix plot for a model.

    IMPORTANT: This function does NOT recalculate matrices. All data must be
    pre-computed and passed in. Use extractors to compute data before calling.

    Args:
        result: FoldingResult instance (for chain info and PAE)
        pmc_matrix: Pre-computed PMC matrix (required if include_pmc_pde=True)
        pde_matrix: Pre-computed PDE matrix (optional, uses PMC if None)
        output_path: Path to save figure (displays if None)
        include_pae: Include standalone PAE plot
        include_pmc_pde: Include PMC/PDE combined plot
        dpi: DPI for saved figure

    Returns:
        Figure object if not saving to file
    """
    _check_matplotlib()

    n_plots = include_pae + include_pmc_pde
    if n_plots == 0:
        return None

    fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 8))
    if n_plots == 1:
        axes = [axes]

    chain_boundaries, chain_labels = _get_chain_boundaries(result)

    plot_idx = 0

    if include_pae and result.pae_matrix is not None:
        plot_pae_matrix(
            result.pae_matrix,
            ax=axes[plot_idx],
            chain_boundaries=chain_boundaries,
            chain_labels=chain_labels,
            title=f"PAE - {result.job_name} model {result.model_num}"
        )
        plot_idx += 1

    if include_pmc_pde and pmc_matrix is not None:
        plot_pmc_pde_combined(
            pmc_matrix,
            pde=pde_matrix,
            ax=axes[plot_idx],
            chain_boundaries=chain_boundaries,
            chain_labels=chain_labels,
            title=f"PMC/PDE - {result.job_name} model {result.model_num}"
        )

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        return None
    else:
        return fig
