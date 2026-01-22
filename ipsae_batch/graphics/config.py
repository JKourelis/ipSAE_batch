"""
Graphics configuration module.

Provides configurable settings for colors, confidence levels,
and other plotting parameters. Settings can be loaded from CSV
or set programmatically.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List
from pathlib import Path
import csv


@dataclass
class MatrixColorConfig:
    """Color configuration for matrix plots."""
    # Colormaps for different matrices (using matplotlib colormap names)
    pae_cmap: str = "Greens_r"      # PAE matrix - green (AlphaBridge default)
    pmc_cmap: str = "Blues_r"       # PMC matrix - blue (AlphaBridge default)
    pde_cmap: str = "RdPu"          # PDE/contact_prob matrix - red-purple

    # Value ranges for colormaps
    pae_vmin: float = 0.0
    pae_vmax: float = 31.75
    pmc_vmin: float = 0.0
    pmc_vmax: float = 32.0
    pde_vmin: float = 0.0
    pde_vmax: float = 10.0

    # Joint matrix configuration
    # Options: 'pae', 'pmc', 'contact_prob'
    joint_matrix_upper: str = "pmc"           # Upper triangle metric
    joint_matrix_lower: str = "contact_prob"  # Lower triangle metric

    # Chain gap configuration for matrix plots
    # Gap between chains in matrix visualization
    # If chain_gap_percent > 0, gap = percent * total_size (recommended)
    # If chain_gap_percent = 0, uses chain_gap_size as absolute pixels
    chain_gap_percent: float = 1.0  # Gap as % of total matrix size (1.0 = 1%)
    chain_gap_size: int = 5         # Absolute gap in pixels (used if percent=0)


@dataclass
class InterfaceConfig:
    """Configuration for interface detection."""
    # Confidence levels for interface detection
    # low: more sensitive (more interfaces detected)
    # default: balanced
    # high: more stringent (fewer, more confident interfaces)
    confidence_level: str = "default"

    # Distance thresholds for each confidence level (CB-CB in Angstroms)
    distance_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "low": 10.0,      # More permissive - catches weaker interactions
        "default": 10.0,  # Standard contact threshold (matches PAE cutoff)
        "high": 8.0,      # Stringent - only close contacts
    })

    # PAE thresholds for interface detection (combined with distance)
    # Using PAE threshold creates sparse matrices with natural gaps
    # Should match the PAE cutoff used for calculations (default 10Å)
    pae_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "low": 12.0,      # More permissive
        "default": 10.0,  # Standard threshold (matches calculation cutoff)
        "high": 8.0,      # Stringent
    })

    # Minimum contacts to consider an interface
    min_contacts: Dict[str, int] = field(default_factory=lambda: {
        "low": 3,
        "default": 5,
        "high": 10,
    })

    # Gap threshold for merging interface regions
    # Controls how separate contact blobs are merged into interfaces
    # gap=0: Strict - only merge directly overlapping regions (shows more interfaces)
    # gap=3: Permissive - merge regions within 3 residues (AlphaBridge default)
    gap_merge: int = 0  # Default: strict, shows separate interfaces

    @property
    def distance_threshold(self) -> float:
        """Get current distance threshold based on confidence level."""
        return self.distance_thresholds.get(self.confidence_level, 10.0)

    @property
    def pae_threshold(self) -> float:
        """Get current PAE threshold based on confidence level."""
        return self.pae_thresholds.get(self.confidence_level, 10.0)

    @property
    def min_contact_count(self) -> int:
        """Get minimum contact count based on confidence level."""
        return self.min_contacts.get(self.confidence_level, 5)


@dataclass
class MetricConfig:
    """Configuration for per-contact metrics and coloring."""
    # pLDDT-style color scheme (4 bins: very_high, high, low, very_low)
    color_very_high: str = '#0053d6'  # Dark blue - excellent
    color_high: str = '#65cbf3'       # Light blue - good
    color_low: str = '#ffdb13'        # Yellow - fair
    color_very_low: str = '#ff7d45'   # Orange - poor

    # Quality thresholds - contacts below this are NOT drawn
    # For higher_is_better: minimum acceptable value
    # For lower_is_better: maximum acceptable value
    # ipSAE threshold 0.1 ≈ PAE < 10Å for typical chain sizes
    threshold_ipSAE: float = 0.1
    threshold_ipSAE_d0chn: float = 0.1
    threshold_ipTM: float = 0.1
    threshold_contact_prob: float = 0.5
    threshold_AB_score: float = 0.1
    threshold_pae: float = 10.0
    threshold_pae_symmetric: float = 10.0
    threshold_pmc: float = 25.0

    # Color bin thresholds for each metric (high, mid, low)
    # For higher_is_better: >high=blue, mid-high=cyan, low-mid=yellow, <low=orange
    # For lower_is_better: <high=blue, high-mid=cyan, mid-low=yellow, >low=orange
    # ipSAE bins: >0.5 (PAE<4Å), 0.3-0.5 (PAE<6Å), 0.15-0.3 (PAE<8Å), <0.15 (PAE<10Å)
    bins_ipSAE: List[float] = field(default_factory=lambda: [0.5, 0.3, 0.15])
    bins_ipSAE_d0chn: List[float] = field(default_factory=lambda: [0.5, 0.3, 0.15])
    bins_ipTM: List[float] = field(default_factory=lambda: [0.5, 0.3, 0.15])
    bins_contact_prob: List[float] = field(default_factory=lambda: [0.9, 0.75, 0.5])
    bins_AB_score: List[float] = field(default_factory=lambda: [0.5, 0.3, 0.15])
    bins_pae: List[float] = field(default_factory=lambda: [3.0, 5.0, 8.0])
    bins_pae_symmetric: List[float] = field(default_factory=lambda: [3.0, 5.0, 8.0])
    bins_pmc: List[float] = field(default_factory=lambda: [5.0, 15.0, 22.0])
    bins_distance: List[float] = field(default_factory=lambda: [3.0, 6.0, 10.0])


@dataclass
class RibbonConfig:
    """Configuration for ribbon plots."""
    # pLDDT color thresholds and colors (for pLDDT track)
    plddt_thresholds: List[float] = field(default_factory=lambda: [50, 70, 90])
    plddt_colors: List[str] = field(default_factory=lambda: [
        '#ff7d45',  # Very low (<50) - orange-red
        '#ffdb13',  # Low (50-70) - yellow
        '#65cbf3',  # High (70-90) - light blue
        '#0053d6',  # Very high (>90) - dark blue
    ])

    # Color for low-confidence contacts (within distance cutoff but PAE > threshold)
    # Light grey to show spatial proximity without implying confident interaction
    low_confidence_color: str = '#d3d3d3'  # Light grey

    # Tick marks on chain sectors - adaptive based on chain length
    # Format: list of (max_length, tick_interval) tuples, processed in order
    # Last entry should have a high max_length to catch all remaining
    tick_intervals: List[tuple] = field(default_factory=lambda: [
        (100, 10),   # Chains up to 100aa: tick every 10 residues
        (300, 20),   # Chains up to 300aa: tick every 20 residues
        (10000, 50), # Larger chains: tick every 50 residues
    ])
    tick_interval: int = 10  # Fallback if tick_intervals not used (deprecated)

    # Interface link settings
    link_alpha: float = 0.25

    # Contact line coloring metric
    # Options: 'pae_symmetric' (default), 'pae', 'ipSAE', 'ipSAE_d0chn', 'ipTM', 'contact_prob', 'AB_score', 'pmc', None
    # When set, contact lines are colored using pLDDT-style 4-bin colors
    # When None, uses uniform interface colors
    # pae_symmetric = max(PAE[i,j], PAE[j,i]) - conservative estimate
    line_color_metric: Optional[str] = 'pae_symmetric'

    # AlphaBridge interface detection threshold
    # Use 0.65 with our calculated contact_prob formula
    # (validated to match AF3 native interface detection at 0.5)
    interface_contact_prob_threshold: float = 0.65

    # Track sizes (radial positions)
    interface_track: tuple = (75, 85)
    plddt_track: tuple = (88, 93)

    # Space between sectors
    sector_space: float = 0.75

    # Distinct interface colors - Paul Tol's qualitative palette (colorblind-friendly)
    # https://personal.sron.nl/~pault/
    interface_colors: List[str] = field(default_factory=lambda: [
        '#77AADD',  # Light blue
        '#EE8866',  # Orange
        '#EEDD88',  # Light yellow
        '#FFAABB',  # Pink
        '#99DDFF',  # Cyan
        '#44BB99',  # Teal
        '#BBCC33',  # Yellow-green
        '#AAAA00',  # Olive
        '#DDDDDD',  # Light grey
    ])


@dataclass
class GraphicsConfig:
    """Main graphics configuration."""
    matrix: MatrixColorConfig = field(default_factory=MatrixColorConfig)
    interface: InterfaceConfig = field(default_factory=InterfaceConfig)
    ribbon: RibbonConfig = field(default_factory=RibbonConfig)
    metric: MetricConfig = field(default_factory=MetricConfig)

    # Output settings
    dpi: int = 150
    image_format: str = "png"

    def set_confidence_level(self, level: str) -> None:
        """Set interface detection confidence level."""
        if level not in ["low", "default", "high"]:
            raise ValueError(f"Invalid confidence level: {level}. Use 'low', 'default', or 'high'.")
        self.interface.confidence_level = level

    def get_quality_threshold(self, metric: str) -> Optional[float]:
        """Get quality threshold for a metric."""
        return getattr(self.metric, f'threshold_{metric}', None)

    def get_color_bins(self, metric: str) -> Optional[List[float]]:
        """Get color bin thresholds for a metric."""
        return getattr(self.metric, f'bins_{metric}', None)

    def get_plddt_colors(self) -> Dict[str, str]:
        """Get pLDDT-style color scheme."""
        return {
            'very_high': self.metric.color_very_high,
            'high': self.metric.color_high,
            'low': self.metric.color_low,
            'very_low': self.metric.color_very_low,
        }


# Global default configuration
_default_config = GraphicsConfig()


def get_config() -> GraphicsConfig:
    """Get the current graphics configuration."""
    return _default_config


def set_config(config: GraphicsConfig) -> None:
    """Set the global graphics configuration."""
    global _default_config
    _default_config = config


def load_config_from_csv(csv_path: str) -> GraphicsConfig:
    """
    Load graphics configuration from a CSV file.

    CSV format:
        setting,value
        pae_cmap,Greens_r
        pmc_cmap,Blues_r
        pde_cmap,RdPu
        confidence_level,default
        dpi,150
        ...

    Args:
        csv_path: Path to configuration CSV file

    Returns:
        GraphicsConfig instance
    """
    config = GraphicsConfig()

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            setting = row.get('setting', '').strip()
            value = row.get('value', '').strip()

            if not setting or not value or setting.startswith('#'):
                continue

            # Matrix colors
            if setting == 'pae_cmap':
                config.matrix.pae_cmap = value
            elif setting == 'pmc_cmap':
                config.matrix.pmc_cmap = value
            elif setting == 'pde_cmap':
                config.matrix.pde_cmap = value
            elif setting == 'pae_vmax':
                config.matrix.pae_vmax = float(value)
            elif setting == 'pmc_vmax':
                config.matrix.pmc_vmax = float(value)
            elif setting == 'pde_vmax':
                config.matrix.pde_vmax = float(value)
            elif setting == 'joint_matrix_upper':
                config.matrix.joint_matrix_upper = value
            elif setting == 'joint_matrix_lower':
                config.matrix.joint_matrix_lower = value
            elif setting == 'chain_gap_percent':
                config.matrix.chain_gap_percent = float(value)
            elif setting == 'chain_gap_size':
                config.matrix.chain_gap_size = int(value)

            # Interface settings
            elif setting == 'confidence_level':
                config.set_confidence_level(value)
            elif setting == 'distance_threshold_low':
                config.interface.distance_thresholds['low'] = float(value)
            elif setting == 'distance_threshold_default':
                config.interface.distance_thresholds['default'] = float(value)
            elif setting == 'distance_threshold_high':
                config.interface.distance_thresholds['high'] = float(value)
            elif setting == 'pae_threshold_low':
                config.interface.pae_thresholds['low'] = float(value)
            elif setting == 'pae_threshold_default':
                config.interface.pae_thresholds['default'] = float(value)
            elif setting == 'pae_threshold_high':
                config.interface.pae_thresholds['high'] = float(value)
            elif setting == 'gap_merge':
                config.interface.gap_merge = int(value)
            elif setting == 'interface_contact_prob_threshold':
                config.ribbon.interface_contact_prob_threshold = float(value)

            # Output settings
            elif setting == 'dpi':
                config.dpi = int(value)
            elif setting == 'image_format':
                config.image_format = value

            # Ribbon settings
            elif setting == 'link_alpha':
                config.ribbon.link_alpha = float(value)
            elif setting == 'line_color_metric':
                config.ribbon.line_color_metric = None if value.lower() == 'none' else value
            elif setting == 'sector_space':
                config.ribbon.sector_space = float(value)
            elif setting == 'tick_interval':
                config.ribbon.tick_interval = int(value)
            elif setting == 'tick_intervals':
                # Format: "100:10,300:20,10000:50"
                intervals = []
                for pair in value.split(','):
                    if ':' in pair:
                        max_len, interval = pair.strip().split(':')
                        intervals.append((int(max_len), int(interval)))
                if intervals:
                    config.ribbon.tick_intervals = intervals
            elif setting == 'low_confidence_color':
                config.ribbon.low_confidence_color = value
            elif setting == 'interface_colors':
                config.ribbon.interface_colors = [c.strip() for c in value.split(',')]

            # pLDDT-style colors
            elif setting == 'color_very_high':
                config.metric.color_very_high = value
            elif setting == 'color_high':
                config.metric.color_high = value
            elif setting == 'color_low':
                config.metric.color_low = value
            elif setting == 'color_very_low':
                config.metric.color_very_low = value

            # Quality thresholds
            elif setting.startswith('threshold_'):
                metric = setting[10:]  # Remove 'threshold_' prefix
                setattr(config.metric, setting, float(value))

            # Color bin thresholds
            elif setting.startswith('bins_'):
                metric = setting[5:]  # Remove 'bins_' prefix
                bins = [float(x.strip()) for x in value.split(',')]
                setattr(config.metric, setting, bins)

    return config


def save_config_to_csv(config: GraphicsConfig, csv_path: str) -> None:
    """
    Save graphics configuration to a CSV file.

    Args:
        config: GraphicsConfig instance
        csv_path: Path to save configuration CSV
    """
    # Get current line color metric for documentation
    line_metric = config.ribbon.line_color_metric or 'none'
    line_bins = config.get_color_bins(line_metric) if line_metric != 'none' else None
    line_threshold = config.get_quality_threshold(line_metric) if line_metric != 'none' else None

    settings = [
        # Interface Detection (most important - at top)
        ('# Interface Detection', ''),
        ('# Contacts are identified by: distance < threshold AND PAE < threshold', ''),
        ('confidence_level', config.interface.confidence_level),
        ('distance_threshold_low', str(config.interface.distance_thresholds['low'])),
        ('distance_threshold_default', str(config.interface.distance_thresholds['default'])),
        ('distance_threshold_high', str(config.interface.distance_thresholds['high'])),
        ('# PAE thresholds for interface detection (should match calculation cutoff)', ''),
        ('pae_threshold_low', str(config.interface.pae_thresholds['low'])),
        ('pae_threshold_default', str(config.interface.pae_thresholds['default'])),
        ('pae_threshold_high', str(config.interface.pae_thresholds['high'])),
        ('# Gap merge: 0=strict (more interfaces), 3=permissive (merge nearby regions)', ''),
        ('gap_merge', str(config.interface.gap_merge)),

        # Contact Line Coloring (second most important)
        ('# Contact Line Coloring', ''),
        ('# Metric used for coloring contact lines in ribbon plot', ''),
        ('# Options: pae_symmetric (default), pae, ipSAE, ipTM, contact_prob, AB_score, pmc, none', ''),
        ('line_color_metric', line_metric),
        ('# Color bins for current metric (high/mid/low thresholds)', ''),
        (f'# Current: bins_{line_metric}', ','.join(str(x) for x in line_bins) if line_bins else 'N/A'),
        (f'# Cutoff (below not drawn): threshold_{line_metric}', str(line_threshold) if line_threshold else 'N/A'),

        # Matrix settings
        ('# Matrix Colormaps', ''),
        ('pae_cmap', config.matrix.pae_cmap),
        ('pmc_cmap', config.matrix.pmc_cmap),
        ('pde_cmap', config.matrix.pde_cmap),
        ('pae_vmax', str(config.matrix.pae_vmax)),
        ('pmc_vmax', str(config.matrix.pmc_vmax)),
        ('pde_vmax', str(config.matrix.pde_vmax)),
        ('# Joint Matrix (options: pae, pmc, contact_prob)', ''),
        ('joint_matrix_upper', config.matrix.joint_matrix_upper),
        ('joint_matrix_lower', config.matrix.joint_matrix_lower),
        ('# Chain gap in matrix plots (percent of total size, or absolute if percent=0)', ''),
        ('chain_gap_percent', str(config.matrix.chain_gap_percent)),
        ('chain_gap_size', str(config.matrix.chain_gap_size)),

        # Output settings
        ('# Output Settings', ''),
        ('dpi', str(config.dpi)),
        ('image_format', config.image_format),

        # Ribbon plot settings
        ('# Ribbon Plot Settings', ''),
        ('link_alpha', str(config.ribbon.link_alpha)),
        ('sector_space', str(config.ribbon.sector_space)),
        ('# Tick intervals: format is max_length:interval pairs', ''),
        ('# e.g., 100:10,300:20,10000:50 means every 10 up to 100aa, every 20 up to 300aa, every 50 for larger', ''),
        ('tick_intervals', ','.join(f'{m}:{i}' for m, i in config.ribbon.tick_intervals)),
        ('# Color for low-confidence contacts (within distance but PAE > threshold)', ''),
        ('low_confidence_color', config.ribbon.low_confidence_color),
        ('interface_colors', ','.join(config.ribbon.interface_colors)),
        ('interface_contact_prob_threshold', str(config.ribbon.interface_contact_prob_threshold)),

        # pLDDT-style colors
        ('# pLDDT-Style Color Scheme (4-bin coloring)', ''),
        ('color_very_high', config.metric.color_very_high),
        ('color_high', config.metric.color_high),
        ('color_low', config.metric.color_low),
        ('color_very_low', config.metric.color_very_low),

        # Quality thresholds
        ('# Quality Thresholds (contacts beyond threshold are NOT drawn)', ''),
        ('threshold_pae', str(config.metric.threshold_pae)),
        ('threshold_pae_symmetric', str(config.metric.threshold_pae_symmetric)),
        ('threshold_ipSAE', str(config.metric.threshold_ipSAE)),
        ('threshold_ipSAE_d0chn', str(config.metric.threshold_ipSAE_d0chn)),
        ('threshold_ipTM', str(config.metric.threshold_ipTM)),
        ('threshold_contact_prob', str(config.metric.threshold_contact_prob)),
        ('threshold_AB_score', str(config.metric.threshold_AB_score)),
        ('threshold_pmc', str(config.metric.threshold_pmc)),

        # Color bin thresholds
        ('# Color Bin Thresholds', ''),
        ('# For lower-is-better (PAE/PMC): <high=blue, high-mid=cyan, mid-low=yellow, >low=orange', ''),
        ('# For higher-is-better (ipSAE/ipTM): >high=blue, mid-high=cyan, low-mid=yellow, <low=orange', ''),
        ('bins_pae', ','.join(str(x) for x in config.metric.bins_pae)),
        ('bins_pae_symmetric', ','.join(str(x) for x in config.metric.bins_pae_symmetric)),
        ('bins_ipSAE', ','.join(str(x) for x in config.metric.bins_ipSAE)),
        ('bins_ipSAE_d0chn', ','.join(str(x) for x in config.metric.bins_ipSAE_d0chn)),
        ('bins_ipTM', ','.join(str(x) for x in config.metric.bins_ipTM)),
        ('bins_contact_prob', ','.join(str(x) for x in config.metric.bins_contact_prob)),
        ('bins_AB_score', ','.join(str(x) for x in config.metric.bins_AB_score)),
        ('bins_pmc', ','.join(str(x) for x in config.metric.bins_pmc)),
        ('bins_distance', ','.join(str(x) for x in config.metric.bins_distance)),
    ]

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['setting', 'value'])
        for setting, value in settings:
            if not setting.startswith('#'):
                writer.writerow([setting, value])
            else:
                writer.writerow([setting, value])  # Keep comments


def create_default_config_csv(csv_path: str) -> None:
    """Create a default configuration CSV file."""
    save_config_to_csv(GraphicsConfig(), csv_path)
