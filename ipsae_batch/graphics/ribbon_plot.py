"""
Ribbon diagram plotting module.

Creates circular ribbon diagrams showing chain interactions,
with interface contacts colored by interface assignment.
Works with FoldingResult from data_readers.

IMPORTANT: This module does NOT recalculate data. All interfaces
must be pre-computed and passed in as parameters.

=============================================================================
KEY CONCEPTS: INTERFACES vs PROXIMITY CONTACTS
=============================================================================

There are TWO distinct concepts that must not be confused:

1. INTERFACES (from get_geometric_interfaces):
   - Contacts filtered by BOTH distance AND PAE threshold (PAE < 10)
   - Used for SCORING (ipSAE calculations)
   - Grouped into distinct interface regions using connected components
   - Shown as colored REGIONS on the inner track of the ribbon plot
   - Each interface has its own color and contact count
   - Contact LINES are colored by interface assignment (uniform color)

2. PROXIMITY CONTACTS (from get_proximity_contacts):
   - ALL contacts within distance threshold (NO PAE filter)
   - Used for VISUALIZATION only
   - Proximity contacts NOT in any interface drawn in GREY
   - NOT used for scoring calculations

The distinction exists because:
- We want to SHOW all structural proximity (grey lines for context)
- But only SCORE confident interactions (PAE < 10)

=============================================================================
LEGEND FORMAT
=============================================================================

The legend shows:

1. pLDDT (outer ring): Standard AlphaFold coloring for confidence

2. Interface regions (inner ring):
   - Each interface shows: "I1 (A-B): 15 contacts"
   - These are ALWAYS scored (by definition, interfaces are PAE-filtered)
   - Each interface gets a distinct color from Paul Tol palette
   - Contact lines use the SAME color as interface regions

3. Unscored contacts (grey):
   - Contacts within distance but PAE > 10
   - Shown as grey lines (low confidence, not scored)

The "Unscored: X/Y" format shows:
- X = number of unscored contacts (proximity - sum of interface contacts)
- Y = total proximity contacts for all chain pairs
"""

from typing import Optional, Callable, List, Tuple, Dict
import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.figure import Figure
    from matplotlib.colors import rgb2hex, Normalize
    import matplotlib.cm as cm
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from pycirclize import Circos
    HAS_PYCIRCLIZE = True
except ImportError:
    HAS_PYCIRCLIZE = False

from ..data_readers import FoldingResult


# Import config for accessing settings
from .config import get_config

# Theoretical ranges for per-contact metrics (used for colormap normalization)
# Format: metric_name -> (vmin, vmax, higher_is_better)
METRIC_RANGES = {
    'ipSAE': (0.0, 1.0, True),           # ipSAE score (higher = better)
    'ipSAE_d0chn': (0.0, 1.0, True),     # ipSAE chain-normalized (higher = better)
    'ipTM': (0.0, 1.0, True),            # ipTM score (higher = better)
    'contact_prob': (0.0, 1.0, True),    # Contact probability (higher = better)
    'AB_score': (0.0, 1.0, True),        # AlphaBridge = sqrt(contact_prob * ipTM) (higher = better)
    'pae': (0.0, 31.75, False),          # PAE (lower = better)
    'pae_symmetric': (0.0, 31.75, False), # Symmetric PAE (lower = better)
    'pmc': (0.0, 32.0, False),           # PMC (lower = better)
    'distance': (0.0, 15.0, False),      # CB-CB distance (lower = closer)
    'plddt_i': (0.0, 100.0, True),       # pLDDT (higher = better)
    'plddt_j': (0.0, 100.0, True),       # pLDDT (higher = better)
}


def _get_quality_threshold(metric: str) -> Optional[float]:
    """Get quality threshold from config."""
    config = get_config()
    return config.get_quality_threshold(metric)


def _get_color_bins(metric: str) -> Optional[Tuple[float, float, float]]:
    """Get color bin thresholds from config."""
    config = get_config()
    bins = config.get_color_bins(metric)
    if bins and len(bins) >= 3:
        return tuple(bins[:3])
    return None


def _get_plddt_colors() -> Dict[str, str]:
    """Get pLDDT-style colors from config."""
    config = get_config()
    return config.get_plddt_colors()


def _get_interface_colors() -> List[str]:
    """Get distinct interface colors from config."""
    config = get_config()
    return config.ribbon.interface_colors


def metric_to_plddt_color(value: float, metric: str) -> Optional[str]:
    """
    Convert a metric value to pLDDT-style 4-bin color.

    Uses colors and thresholds from config.

    Args:
        value: The metric value
        metric: The metric name

    Returns:
        Hex color string, or None if below quality threshold
    """
    if value is None:
        return None

    # Check quality threshold first (from config)
    threshold = _get_quality_threshold(metric)
    if threshold is not None:
        higher_is_better = METRIC_RANGES.get(metric, (0, 1, True))[2]
        if higher_is_better and value < threshold:
            return None  # Below quality threshold
        if not higher_is_better and value > threshold:
            return None  # Above quality threshold (for metrics where lower is better)

    # Get bin thresholds from config
    bins = _get_color_bins(metric)
    colors = _get_plddt_colors()

    if bins is None:
        return colors['high']  # Default color if no bins defined

    t_high, t_mid, t_low = bins
    higher_is_better = METRIC_RANGES.get(metric, (0, 1, True))[2]

    if higher_is_better:
        # Higher is better: >high=blue, mid-high=cyan, low-mid=yellow, <low=orange
        if value >= t_high:
            return colors['very_high']
        elif value >= t_mid:
            return colors['high']
        elif value >= t_low:
            return colors['low']
        else:
            return colors['very_low']
    else:
        # Lower is better: <low=blue, low-mid=cyan, mid-high=yellow, >high=orange
        if value <= t_high:  # Note: for lower-is-better, t_high is actually the lowest threshold
            return colors['very_high']
        elif value <= t_mid:
            return colors['high']
        elif value <= t_low:
            return colors['low']
        else:
            return colors['very_low']


def _check_dependencies():
    """Check if required dependencies are available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install matplotlib"
        )
    if not HAS_PYCIRCLIZE:
        raise ImportError(
            "pycirclize is required for ribbon plots. "
            "Install it with: pip install pycirclize"
        )


# =============================================================================
# PAE Color Mapping Hook
# =============================================================================

def default_pae_to_color(pae_value: float, pae_max: float = 31.75) -> str:
    """
    Default PAE to color mapping function.

    Args:
        pae_value: PAE value (expected range: 0 to pae_max)
        pae_max: Maximum PAE value for normalization

    Returns:
        Hex color string
    """
    # Normalize to 0-1
    normalized = np.clip(pae_value / pae_max, 0, 1)

    # Simple gradient: green (low PAE) -> yellow -> red (high PAE)
    if normalized < 0.5:
        # Green to yellow
        r = int(255 * (normalized * 2))
        g = 255
    else:
        # Yellow to red
        r = 255
        g = int(255 * (2 - normalized * 2))
    b = 0

    return f"#{r:02x}{g:02x}{b:02x}"


def plddt_to_color(plddt: float) -> str:
    """
    Convert pLDDT to AlphaFold-style color.

    Args:
        plddt: pLDDT value (0-100)

    Returns:
        Hex color string
    """
    if plddt < 50:
        return '#ff7d45'  # Orange - Very low
    elif plddt < 70:
        return '#ffdb13'  # Yellow - Low
    elif plddt < 90:
        return '#65cbf3'  # Light blue - High
    else:
        return '#0053d6'  # Dark blue - Very high


# Global hook for PAE color mapping
PAE_COLOR_FUNCTION: Callable[[float, float], str] = default_pae_to_color


def set_pae_color_function(func: Callable[[float, float], str]) -> None:
    """
    Set the global PAE color mapping function.

    Args:
        func: Function that takes (pae_value, pae_max) and returns hex color
    """
    global PAE_COLOR_FUNCTION
    PAE_COLOR_FUNCTION = func


def get_pae_color(pae_value: float, pae_max: float = 31.75) -> str:
    """
    Get color for a PAE value using the current color function.

    Args:
        pae_value: PAE value
        pae_max: Maximum PAE for normalization

    Returns:
        Hex color string
    """
    return PAE_COLOR_FUNCTION(pae_value, pae_max)


# =============================================================================
# Ribbon Diagram
# =============================================================================

def _get_distinct_colors(n: int) -> List[str]:
    """Get n visually distinct colors from config."""
    if n == 0:
        return []

    # Get colors from config
    config_colors = _get_interface_colors()

    # If we need more colors than config provides, generate additional ones
    if n <= len(config_colors):
        return config_colors[:n]
    else:
        # Use config colors first, then generate additional ones
        colors = list(config_colors)
        remaining = n - len(config_colors)
        hues = np.linspace(0, 1, remaining + 2, endpoint=False)[1:-1]  # Skip first/last to avoid similar colors
        for h in hues:
            rgb = plt.cm.hsv(h)[:3]
            colors.append(rgb2hex(rgb))
        return colors[:n]


class RibbonPlot:
    """
    Ribbon diagram creator for protein complexes.

    Creates circular diagrams showing chains as sectors with:
    - pLDDT coloring for each residue
    - Interface contact regions highlighted
    - Links between contacting regions (optionally colored by metric)
    - Optional domain clustering coloring

    IMPORTANT: This class does NOT recalculate data. Pre-compute interfaces
    and domain clusters using extractors before passing to this class.
    """

    def __init__(
        self,
        result: FoldingResult,
        pae_color_func: Optional[Callable[[float, float], str]] = None,
    ):
        """
        Initialize ribbon plot.

        Args:
            result: FoldingResult instance
            pae_color_func: Custom PAE to color function (uses global if None)
        """
        _check_dependencies()

        self.result = result
        self.pae_color_func = pae_color_func or get_pae_color

        # Build sector sizes (chain lengths)
        self.sectors = {}
        for chain_id in result.unique_chains:
            indices = result.get_chain_indices(chain_id)
            self.sectors[chain_id] = len(indices)

    def create_plot(
        self,
        interfaces: Optional[List[Dict]] = None,
        proximity_contacts: Optional[List[Dict]] = None,
        figsize: Tuple[int, int] = (10, 10),
        space: float = 0.75,
        plddt_track: bool = True,
        show_contacts: bool = True,
        link_alpha: float = 0.25,
    ) -> "Figure":
        """
        Create the ribbon plot.

        IMPORTANT: This method does NOT recalculate data. All data must be
        pre-computed and passed in.

        Tracks (from inside to outside):
        - Inner track: Interface regions (colored by interface)
        - Outer track: pLDDT values (AlphaFold-style coloring)

        Args:
            interfaces: Pre-computed interfaces from extractors.get_geometric_interfaces()
                       (PAE-filtered) - used for interface REGIONS and contact lines
            proximity_contacts: Pre-computed proximity contacts from
                               extractors.get_proximity_contacts() (distance only)
                               Contacts NOT in interfaces drawn in GREY
                               If None, uses interfaces for lines too
            figsize: Figure size
            space: Space between sectors
            plddt_track: Show pLDDT coloring track
            show_contacts: Show interface contacts and links
            link_alpha: Alpha (transparency) for links

        Returns:
            Matplotlib Figure
        """
        circos = Circos(self.sectors, space=space)

        # Use provided interfaces or empty list
        interface_list = interfaces if interfaces is not None else []
        interface_colors = {}

        if show_contacts and interface_list:
            colors = _get_distinct_colors(len(interface_list))
            for i, iface in enumerate(interface_list):
                # Use interface_id if available, otherwise use index
                # This ensures DIFFERENT colors for multiple interfaces between same chain pair
                iface_id = iface.get('interface_id', f"I{i+1}")
                key = iface_id  # Unique key per interface
                interface_colors[key] = colors[i] if colors else '#808080'
                # Store the key back in the interface for reference
                iface['_color_key'] = key

        # Track positions: [interface regions (inner), pLDDT (outer)]
        track_positions = [(75, 85), (88, 93)]

        # Get config
        config = get_config()

        # Add tracks to each sector
        for sector in circos.sectors:
            chain_id = sector.name
            chain_indices = self.result.get_chain_indices(chain_id)
            chain_length = len(chain_indices)

            # Determine tick interval based on chain length
            tick_interval = config.ribbon.tick_interval  # Fallback
            for max_len, interval in config.ribbon.tick_intervals:
                if chain_length <= max_len:
                    tick_interval = interval
                    break

            # Add tracks
            for track_pos in track_positions:
                track = sector.add_track(track_pos)
                track.axis()

            # Add tick marks at interval
            if tick_interval > 0:
                # Generate tick positions: 0, tick_interval, 2*tick_interval, ...
                tick_positions = list(range(0, chain_length, tick_interval))
                # Numeric labels only (chain name drawn separately with bold)
                tick_labels = ['' if p == 0 else str(p) for p in tick_positions]
                if tick_positions:
                    track.xticks(
                        tick_positions,
                        tick_labels,
                        label_orientation='vertical',
                        label_size=5  # Smaller font to avoid overlap
                    )

            # Add chain name at position 0 (bold, separate from numeric ticks)
            label_r = track_positions[-1][1] + 5  # Position for chain label
            track.text(chain_id, x=0, color="black", size=8, r=label_r,
                      orientation='vertical', adjust_rotation=True,
                      fontweight='bold')

            # Add direction indicator (triangle pointing along arc = N→C direction)
            # With orientation='horizontal' and adjust_rotation=True:
            # - The triangle rotates to follow the circular arc
            # - Points in clockwise direction (N→C)
            arrow_r = label_r + 5  # Just above the chain label
            track.text("\u25B6", x=0, color="black", size=7, r=arrow_r,
                      orientation='horizontal', adjust_rotation=True)

            # Color by pLDDT (outer track)
            if plddt_track:
                plddt_values = self.result.plddt[chain_indices]
                for i, plddt in enumerate(plddt_values):
                    color = plddt_to_color(plddt)
                    sector.rect(
                        start=i,
                        end=i + 1,
                        r_lim=track_positions[1],
                        color=color,
                        lw=0
                    )

            # Highlight interface regions (track 0)
            if show_contacts:
                for iface in interface_list:
                    # Use unique interface key for color lookup
                    key = iface.get('_color_key', iface.get('interface_id', f"{iface['chain1']}-{iface['chain2']}"))
                    color = interface_colors.get(key, '#808080')

                    # Draw interface regions on this chain
                    if chain_id == iface['chain1']:
                        for start, end in iface['regions1']:
                            sector.rect(
                                start=start,
                                end=end + 1,
                                r_lim=track_positions[0],
                                color=color,
                                ec='black',
                                lw=0.5
                            )
                    elif chain_id == iface['chain2']:
                        for start, end in iface['regions2']:
                            sector.rect(
                                start=start,
                                end=end + 1,
                                r_lim=track_positions[0],
                                color=color,
                                ec='black',
                                lw=0.5
                            )

        # Add links between interface regions
        # Color by metric using pLDDT-style bins
        # Use proximity_contacts for LINES (all distance-based contacts, colored by quality)
        # Use interfaces for REGIONS (PAE-filtered, for scoring)
        scored_counts = {}  # Track number of SCORED contacts (PAE < threshold) per interface
        total_counts = {}   # Track TOTAL proximity contacts per chain pair

        # Determine what to use for drawing links
        # If proximity_contacts provided, use those (shows ALL contacts, colored by PAE)
        # Otherwise fall back to interfaces
        contacts_for_links = proximity_contacts if proximity_contacts else interface_list

        # Build chain pair -> total proximity contacts lookup
        proximity_counts_by_chain_pair = {}
        if proximity_contacts:
            for pc in proximity_contacts:
                chain_pair = (pc['chain1'], pc['chain2'])
                proximity_counts_by_chain_pair[chain_pair] = pc.get('n_contacts', len(pc.get('links', [])))

        if show_contacts:
            # Initialize counts for interfaces
            # scored = interface contacts (PAE-filtered)
            # total = all proximity contacts for that chain pair
            for iface in interface_list:
                key = iface.get('_color_key', iface.get('interface_id', f"{iface['chain1']}-{iface['chain2']}"))
                # Scored = PAE-filtered contacts in this interface
                scored_counts[key] = iface.get('n_contacts', len(iface.get('links', [])))
                # Total = ALL proximity contacts for this chain pair
                chain_pair = (iface['chain1'], iface['chain2'])
                total_counts[key] = proximity_counts_by_chain_pair.get(chain_pair, scored_counts[key])

            # Build lookup of interface contacts for quick checking
            interface_contact_set = set()
            for iface in interface_list:
                chain1, chain2 = iface['chain1'], iface['chain2']
                if 'links' in iface:
                    for link in iface['links']:
                        res1, res2 = link['residue1'], link['residue2']
                        interface_contact_set.add((chain1, chain2, res1, res2))

            # Get low confidence color from config
            low_conf_color = config.ribbon.low_confidence_color

            # Draw contact LINES
            # Interface contacts use interface color, proximity-only contacts use grey
            for contact in contacts_for_links:
                chain1, chain2 = contact['chain1'], contact['chain2']

                # Find matching interface for color lookup
                matching_iface = None
                for iface in interface_list:
                    if iface['chain1'] == chain1 and iface['chain2'] == chain2:
                        matching_iface = iface
                        break

                key = matching_iface.get('_color_key', f"{chain1}-{chain2}") if matching_iface else f"{chain1}-{chain2}"
                interface_color = interface_colors.get(key, '#808080')

                if 'links' in contact:
                    # Draw individual contact lines
                    for link in contact['links']:
                        res1 = link['residue1']  # local index
                        res2 = link['residue2']  # local index

                        # Check if this contact is in an interface (PAE-filtered)
                        is_interface_contact = (chain1, chain2, res1, res2) in interface_contact_set

                        if is_interface_contact:
                            # Interface contact - use interface color
                            circos.link(
                                (chain1, res1, res1 + 1),
                                (chain2, res2, res2 + 1),
                                color=interface_color,
                                alpha=link_alpha
                            )
                        else:
                            # Proximity-only contact (PAE > threshold) - use grey
                            circos.link(
                                (chain1, res1, res1 + 1),
                                (chain2, res2, res2 + 1),
                                color=low_conf_color,
                                alpha=link_alpha * 0.7  # Slightly more transparent
                            )
                elif 'regions1' in contact and 'regions2' in contact:
                    # Draw links between regions (uniform interface color)
                    for region1 in contact['regions1']:
                        for region2 in contact['regions2']:
                            circos.link(
                                (chain1, region1[0], region1[1] + 1),
                                (chain2, region2[0], region2[1] + 1),
                                color=interface_color,
                                alpha=link_alpha
                            )

        # Create figure
        fig = circos.plotfig()

        # Calculate total proximity contacts for legend
        total_proximity_all = sum(proximity_counts_by_chain_pair.values()) if proximity_counts_by_chain_pair else 0

        # Add legend
        self._add_legend(circos.ax, plddt_track, interface_list, interface_colors,
                        scored_counts=scored_counts,
                        total_counts=total_counts,
                        has_proximity_contacts=(proximity_contacts is not None and len(proximity_contacts) > 0),
                        total_proximity_all=total_proximity_all)

        return fig

    def _add_legend(
        self,
        ax: "plt.Axes",
        show_plddt: bool,
        interfaces: List[Dict],
        interface_colors: Dict[str, str],
        scored_counts: Optional[Dict[str, int]] = None,
        total_counts: Optional[Dict[str, int]] = None,
        has_proximity_contacts: bool = False,
        total_proximity_all: int = 0,
    ) -> None:
        """Add legend to the plot."""
        legend_elements = []

        # Get low-confidence color from config
        config = get_config()
        low_conf_color = config.ribbon.low_confidence_color

        # pLDDT legend (outer track)
        if show_plddt:
            legend_elements.append(Patch(facecolor='none', edgecolor='none', label='pLDDT (outer ring):'))
            plddt_colors = ['#0053d6', '#65cbf3', '#ffdb13', '#ff7d45']
            plddt_labels = ['>90 (very high)', '70-90 (high)', '50-70 (low)', '<50 (very low)']
            for c, l in zip(plddt_colors, plddt_labels):
                legend_elements.append(Patch(facecolor=c, edgecolor='black', linewidth=0.5, label=f'  {l}'))

        # Interface regions legend (inner track)
        # Calculate total unscored contacts (proximity - sum of interface contacts)
        total_unscored = 0
        total_proximity = 0
        if total_counts and interfaces:
            # Track proximity and interface contacts per chain pair
            chain_pair_proximity = {}  # (chain1, chain2) -> total proximity contacts
            chain_pair_interface = {}  # (chain1, chain2) -> sum of interface contacts

            for iface in interfaces:
                chain_pair = (iface['chain1'], iface['chain2'])
                key = iface.get('_color_key', iface.get('interface_id', f"{iface['chain1']}-{iface['chain2']}"))

                # Get proximity count for this chain pair (only set once per chain pair)
                if chain_pair not in chain_pair_proximity:
                    chain_pair_proximity[chain_pair] = total_counts.get(key, 0)

                # Accumulate interface contacts for this chain pair
                chain_pair_interface[chain_pair] = chain_pair_interface.get(chain_pair, 0) + iface.get('n_contacts', 0)

            # Total proximity = sum across all chain pairs
            total_proximity = sum(chain_pair_proximity.values())
            # Total unscored = sum across all chain pairs of (proximity - interface)
            total_unscored = sum(
                chain_pair_proximity[cp] - chain_pair_interface.get(cp, 0)
                for cp in chain_pair_proximity
            )

        if interfaces or has_proximity_contacts:
            legend_elements.append(Patch(facecolor='none', edgecolor='none', label=''))  # Spacer
            legend_elements.append(Patch(facecolor='none', edgecolor='none', label='Interfaces (inner ring + lines):'))

            if interfaces:
                for iface in interfaces:
                    # Use unique interface key
                    key = iface.get('_color_key', iface.get('interface_id', f"{iface['chain1']}-{iface['chain2']}"))
                    color = interface_colors.get(key, '#808080')
                    # Show interface ID, chain pair, and contact count (interfaces are always scored)
                    iface_id = iface.get('interface_id', key)
                    chain_pair = f"{iface['chain1']}-{iface['chain2']}"
                    n_contacts = iface.get('n_contacts', 0)
                    label = f"  {iface_id} ({chain_pair}): {n_contacts} contacts"
                    legend_elements.append(Patch(facecolor=color, edgecolor='black', linewidth=0.5, label=label))

            # Show unscored contacts with grey color box
            if total_unscored > 0 and total_proximity > 0:
                legend_elements.append(Patch(facecolor=low_conf_color, edgecolor='black', linewidth=0.5,
                                            label=f'  Unscored (PAE >10Å): {total_unscored}'))
            elif not interfaces and has_proximity_contacts and total_proximity_all > 0:
                # No interfaces at all - all contacts are unscored
                legend_elements.append(Patch(facecolor=low_conf_color, edgecolor='black', linewidth=0.5,
                                            label=f'  Unscored (PAE >10Å): {total_proximity_all}'))

        if legend_elements:
            ax.legend(
                handles=legend_elements,
                bbox_to_anchor=(1.3, 0.5),
                loc="center",
                fontsize=7,
                title="Legend",
                ncol=1,
                frameon=True,
                fancybox=True,
            )

    def _add_plddt_legend(self, ax: "plt.Axes") -> None:
        """Add pLDDT legend to the plot (legacy method)."""
        plddt_colors = ['#0053d6', '#65cbf3', '#ffdb13', '#ff7d45']
        plddt_labels = ['Very high (>90)', 'High (70-90)',
                        'Low (50-70)', 'Very Low (<50)']

        legend_elements = [
            Patch(facecolor=c, label=l)
            for c, l in zip(plddt_colors, plddt_labels)
        ]

        ax.legend(
            handles=legend_elements,
            bbox_to_anchor=(1.2, 0.55),
            loc="center",
            fontsize=8,
            title="pLDDT",
            ncol=1,
        )

    def save(
        self,
        output_path: str,
        dpi: int = 150,
        **kwargs
    ) -> None:
        """
        Create and save the ribbon plot.

        Args:
            output_path: Path to save figure
            dpi: DPI for saved figure
            **kwargs: Additional arguments for create_plot()
        """
        fig = self.create_plot(**kwargs)
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)


def plot_ribbon(
    result: FoldingResult,
    interfaces: Optional[List[Dict]] = None,
    proximity_contacts: Optional[List[Dict]] = None,
    output_path: Optional[str] = None,
    show_contacts: bool = True,
    pae_color_func: Optional[Callable[[float, float], str]] = None,
    dpi: int = 150,
) -> Optional["Figure"]:
    """
    Convenience function to create a ribbon plot.

    IMPORTANT: This function does NOT recalculate data. Pre-compute interfaces
    using extractors before calling.

    Tracks (from inside to outside):
    - Inner track: Interface regions (colored by interface)
    - Outer track: pLDDT values (AlphaFold-style coloring)

    Contact lines are colored by interface assignment (uniform color per interface).
    Proximity contacts NOT in any interface are shown in grey.

    Args:
        result: FoldingResult instance
        interfaces: Pre-computed interfaces from extractors.get_geometric_interfaces()
                   (PAE-filtered) - used for interface REGIONS and contact lines
        proximity_contacts: Pre-computed proximity contacts from
                           extractors.get_proximity_contacts() (distance only)
                           Contacts NOT in interfaces drawn in GREY
                           If None, uses interfaces for lines too
        output_path: Path to save figure (displays if None)
        show_contacts: Show interface contacts and links
        pae_color_func: Custom PAE color function
        dpi: DPI for saved figure

    Returns:
        Figure if not saving to file
    """
    ribbon = RibbonPlot(
        result,
        pae_color_func=pae_color_func
    )

    fig = ribbon.create_plot(
        interfaces=interfaces,
        proximity_contacts=proximity_contacts,
        show_contacts=show_contacts,
    )

    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        return None

    return fig
