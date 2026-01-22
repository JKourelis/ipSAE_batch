"""
Batch comparison plots module.

Generates interactive HTML reports for comparing multiple jobs/models:
- Scatter plots: Best model per job, configurable X/Y metrics with R² calculation
- Jitter plots: All models for high-scoring jobs (reveals model consistency)
- Correlation matrix: R coefficients between all metrics

Uses Plotly for interactive visualizations with client-side JavaScript controls.
Reference: OLD/AF3_BATCH_Score_Extractor.Rmd

=============================================================================
KEY DESIGN DECISIONS
=============================================================================

1. CHAIN PAIR HANDLING:
   Each chain pair (A-B, A-C, B-C) is shown as a SEPARATE point.
   This preserves information about which interfaces score well.

2. COLOR = NUMBER OF CHAINS:
   Points are colored by the number of chains in the complex.
   Uses Paul Tol palette (same as interface coloring):
   - 2 chains: #4477AA (blue)
   - 3 chains: #EE6677 (red)
   - 4 chains: #228833 (green)
   - 5+ chains: #CCBB44 (yellow)

3. BEST MODEL SELECTION:
   Best model selected by ipSAE (max) by default.
   X/Y axes are independently configurable.

4. INTERACTIVE CONTROLS (JavaScript, no server):
   - X-axis metric dropdown
   - Y-axis metric dropdown
   - R² auto-updates when axes change
   - ID filter (regex)
   - Score threshold slider for jitter plot

=============================================================================
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# Paul Tol colors for number of chains (same as interface coloring)
CHAIN_COUNT_COLORS = {
    2: '#4477AA',  # Blue
    3: '#EE6677',  # Red
    4: '#228833',  # Green
    5: '#CCBB44',  # Yellow (5+ chains)
}

# Metrics available for X/Y axes
# Maps canonical name -> (higher_is_better, display_name, description)
METRIC_DEFINITIONS = {
    'ipSAE': (True, 'ipSAE', 'Interface Predicted SAE (d0res normalization)'),
    'ipSAE_d0chn': (True, 'ipSAE (chain)', 'ipSAE with chain length normalization'),
    'ipSAE_d0dom': (True, 'ipSAE (domain)', 'ipSAE with domain normalization'),
    'ipTM': (True, 'ipTM', 'Interface predicted TM-score'),
    'pTM': (True, 'pTM', 'Global predicted TM-score'),
    'contact_prob_mean': (True, 'Contact Prob', 'Mean contact probability'),
    'n_contacts': (True, 'N Contacts', 'Number of interface contacts'),
    'pae_mean': (False, 'PAE (mean)', 'Mean PAE at interface (lower is better)'),
    'pDockQ': (True, 'pDockQ', 'Predicted DockQ score'),
    'pDockQ2': (True, 'pDockQ2', 'Enhanced pDockQ with PAE'),
    'LIS': (True, 'LIS', 'Local Interaction Score'),
}


def _check_dependencies():
    """Check if required dependencies are available."""
    if not HAS_PANDAS:
        raise ImportError(
            "pandas is required for batch comparison. "
            "Install it with: pip install pandas"
        )
    if not HAS_PLOTLY:
        raise ImportError(
            "plotly is required for batch comparison. "
            "Install it with: pip install plotly"
        )


def _get_metric_info(metric: str) -> Tuple[bool, str, str]:
    """Get metric info: (higher_is_better, display_name, description)."""
    defn = METRIC_DEFINITIONS.get(metric)
    if defn:
        return defn
    return (True, metric, metric)


def _get_available_metrics(df: "pd.DataFrame") -> List[str]:
    """Get list of available numeric metrics in dataframe for scoring comparisons."""
    # Metrics we want to show for PPI scoring (in order of priority)
    scoring_metrics = [
        'ipSAE', 'ipSAE_d0chn', 'ipSAE_d0dom',
        'ipTM', 'ipTM_extracted', 'ipTM_calculated', 'ipTM_d0chn',
        'pTM', 'pTM_extracted', 'pTM_calculated',
        'pDockQ', 'pDockQ2', 'LIS',
        'contact_prob_mean', 'n_contacts',
        'pae_mean',
    ]

    # Internal parameters to exclude from dropdowns
    internal_params = [
        'model', 'n_chains', 'pae_cutoff', 'dist_cutoff',
        'n0res', 'n0chn', 'n0dom',
        'd0res', 'd0chn', 'd0dom',
        'nres1', 'nres2', 'dist1', 'dist2',
    ]

    available = []

    # First add scoring metrics that exist
    for m in scoring_metrics:
        if m in df.columns and pd.api.types.is_numeric_dtype(df[m]):
            available.append(m)

    # Then add any other numeric columns not already included and not internal
    for col in df.select_dtypes(include=[np.number]).columns:
        if col not in available and col not in internal_params:
            available.append(col)

    return available


def _count_chains_in_job(df: "pd.DataFrame") -> "pd.DataFrame":
    """Add n_chains column based on unique chains per job."""
    df = df.copy()

    # Try to determine chain count from chain_pair or chain columns
    if 'chain1' in df.columns and 'chain2' in df.columns:
        # Count unique chains per job
        def count_unique_chains(group):
            chains = set(group['chain1'].unique()) | set(group['chain2'].unique())
            return len(chains)

        chain_counts = df.groupby('job_name').apply(count_unique_chains, include_groups=False)
        df['n_chains'] = df['job_name'].map(chain_counts)
    elif 'chain_pair' in df.columns:
        # Parse chain pairs to count unique chains
        def count_from_pairs(group):
            chains = set()
            for pair in group['chain_pair'].unique():
                if '-' in str(pair):
                    c1, c2 = str(pair).split('-', 1)
                    chains.add(c1)
                    chains.add(c2)
            return len(chains) if chains else 2

        chain_counts = df.groupby('job_name').apply(count_from_pairs, include_groups=False)
        df['n_chains'] = df['job_name'].map(chain_counts)
    else:
        df['n_chains'] = 2  # Default to 2 chains

    return df


def _get_chain_color(n_chains: int) -> str:
    """Get Paul Tol color for chain count."""
    if n_chains >= 5:
        return CHAIN_COUNT_COLORS[5]
    return CHAIN_COUNT_COLORS.get(n_chains, CHAIN_COUNT_COLORS[2])


def load_combined_csv(csv_path: str) -> "pd.DataFrame":
    """
    Load the combined CSV file with aggregate scores.

    Args:
        csv_path: Path to ipSAE_combined.csv

    Returns:
        DataFrame with all results
    """
    _check_dependencies()
    df = pd.read_csv(csv_path)

    # Ensure chain_pair column exists
    if 'chain_pair' not in df.columns:
        if 'chain1' in df.columns and 'chain2' in df.columns:
            df['chain_pair'] = df['chain1'].astype(str) + '-' + df['chain2'].astype(str)
        else:
            df['chain_pair'] = 'A-B'

    # Add chain count
    df = _count_chains_in_job(df)

    return df


def get_best_models(
    df: "pd.DataFrame",
    metric: str = 'ipSAE',
) -> "pd.DataFrame":
    """
    Get the best model per job AND chain pair based on a metric.

    Each chain pair is kept as a separate entry (not aggregated).
    For metrics where higher is better, selects the maximum.
    For metrics where lower is better (PAE), selects the minimum.

    Args:
        df: DataFrame with all results
        metric: Metric to use for selection (default: ipSAE)

    Returns:
        DataFrame with best model per job per chain pair
    """
    _check_dependencies()

    if metric not in df.columns:
        # Fallback to first available metric
        available = _get_available_metrics(df)
        if available:
            metric = available[0]
            print(f"Warning: Metric not found, using '{metric}'")
        else:
            return df.drop_duplicates(['job_name', 'chain_pair']).reset_index(drop=True)

    df = df.copy()
    higher_is_better = _get_metric_info(metric)[0]

    # Group by job_name AND chain_pair, get best model for each
    try:
        if higher_is_better:
            idx = df.groupby(['job_name', 'chain_pair'])[metric].idxmax()
        else:
            idx = df.groupby(['job_name', 'chain_pair'])[metric].idxmin()

        return df.loc[idx.dropna()].reset_index(drop=True)
    except (KeyError, ValueError) as e:
        print(f"Warning: Error selecting best models: {e}")
        return df.drop_duplicates(['job_name', 'chain_pair']).reset_index(drop=True)


def calculate_correlation_matrix(
    df: "pd.DataFrame",
    metrics: Optional[List[str]] = None
) -> "pd.DataFrame":
    """
    Calculate Pearson correlation coefficients between metrics.

    Args:
        df: DataFrame with metric values
        metrics: List of metrics to include (default: all available)

    Returns:
        Correlation matrix DataFrame
    """
    _check_dependencies()

    if metrics is None:
        metrics = _get_available_metrics(df)

    # Filter to available metrics with data
    available = [m for m in metrics if m in df.columns and df[m].notna().sum() > 1]

    if len(available) < 2:
        return pd.DataFrame()

    return df[available].corr()


def generate_batch_comparison_html(
    csv_path: str,
    output_path: str,
    default_x_metric: str = 'ipTM',
    default_y_metric: str = 'ipSAE',
    best_model_metric: str = 'ipSAE',
    jitter_threshold: float = 0.3,
) -> str:
    """
    Generate an interactive HTML report for batch comparison.

    Features:
    - Scatter plot with configurable X/Y axes and R² calculation
    - Color by number of chains (Paul Tol palette)
    - Jitter plot showing all models for high-scoring jobs
    - Correlation matrix heatmap
    - Interactive JavaScript controls (no server needed)

    Args:
        csv_path: Path to ipSAE_combined.csv
        output_path: Path to save HTML report
        default_x_metric: Default metric for X axis
        default_y_metric: Default metric for Y axis
        best_model_metric: Metric for selecting best model (default: ipSAE)
        jitter_threshold: Default threshold for jitter plot

    Returns:
        Path to generated HTML file
    """
    _check_dependencies()

    # Load data
    df = load_combined_csv(csv_path)

    if df.empty:
        raise ValueError("No data in CSV file")

    # Get available metrics
    available_metrics = _get_available_metrics(df)
    if not available_metrics:
        raise ValueError("No numeric metrics found in CSV")

    # Filter to 'max' type only if present (best asymmetric scores)
    if 'type' in df.columns:
        df_max = df[df['type'] == 'max']
        if len(df_max) > 0:
            df = df_max

    # Prepare data for JavaScript (convert to list of dicts)
    # Best model selection and correlation computed dynamically in JS
    # All models included - best model selection done dynamically in JS
    all_data_json = df.to_dict(orient='records')

    # Resolve default metrics to actual column names
    # Try to find ipTM variant if 'ipTM' requested
    def resolve_metric(requested, available):
        if requested in available:
            return requested
        # Try common variants
        variants = [
            requested,
            f"{requested}_extracted",
            f"{requested}_calculated",
            f"{requested}_d0chn",
        ]
        for v in variants:
            if v in available:
                return v
        # Fallback to first available
        return available[0] if available else requested

    resolved_x = resolve_metric(default_x_metric, available_metrics)
    resolved_y = resolve_metric(default_y_metric, available_metrics)

    # Build metric options HTML with resolved defaults
    def build_options(selected_metric):
        options = ""
        for m in available_metrics:
            display_name = _get_metric_info(m)[1]
            selected = ' selected' if m == selected_metric else ''
            options += f'<option value="{m}"{selected}>{display_name}</option>\n'
        return options

    x_metric_options = build_options(resolved_x)
    y_metric_options = build_options(resolved_y)

    # Build chain color legend HTML
    chain_legend_html = ""
    for n, color in sorted(CHAIN_COUNT_COLORS.items()):
        label = f"{n} chains" if n < 5 else "5+ chains"
        chain_legend_html += f'''
            <span style="display: inline-flex; align-items: center; margin-right: 15px;">
                <span style="width: 12px; height: 12px; background: {color}; border-radius: 50%; margin-right: 5px;"></span>
                {label}
            </span>
        '''

    # Generate HTML
    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>ipSAE Batch Comparison</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f7fa;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            margin-bottom: 15px;
        }}
        .controls {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            align-items: center;
        }}
        .control-group {{
            display: flex;
            flex-direction: column;
            gap: 5px;
        }}
        .control-group label {{
            font-weight: 600;
            color: #555;
            font-size: 12px;
            text-transform: uppercase;
        }}
        select, input[type="text"] {{
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
            min-width: 150px;
        }}
        select:focus, input:focus {{
            outline: none;
            border-color: #3498db;
        }}
        .stats-box {{
            background: #e8f4f8;
            padding: 15px 20px;
            border-radius: 8px;
            display: flex;
            gap: 30px;
            align-items: center;
        }}
        .stat {{
            text-align: center;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .stat-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
        }}
        .plot-container {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            min-height: 400px;
        }}
        .legend {{
            background: #f8f9fa;
            padding: 10px 15px;
            border-radius: 4px;
            margin-bottom: 15px;
            font-size: 13px;
        }}
        .legend-title {{
            font-weight: 600;
            margin-right: 15px;
        }}
        .info {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 10px 15px;
            margin-bottom: 20px;
            border-radius: 0 4px 4px 0;
        }}
        .slider-container {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        input[type="range"] {{
            width: 150px;
        }}
        .threshold-value {{
            min-width: 40px;
            text-align: center;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <h1>ipSAE Batch Comparison Report</h1>

    <div class="info">
        <strong>Source:</strong> {Path(csv_path).name} |
        <strong>Jobs:</strong> {df['job_name'].nunique()} |
        <strong>Total entries:</strong> {len(df)}
    </div>

    <div class="controls">
        <div class="control-group">
            <label>Best Model By</label>
            <select id="best-metric" onchange="updateScatterPlot()">
                {y_metric_options}
            </select>
        </div>
        <div class="control-group">
            <label>X-Axis Metric</label>
            <select id="x-metric" onchange="updateScatterPlot()">
                {x_metric_options}
            </select>
        </div>
        <div class="control-group">
            <label>Y-Axis Metric</label>
            <select id="y-metric" onchange="updateScatterPlot()">
                {y_metric_options}
            </select>
        </div>
        <div class="control-group">
            <label>Filter Jobs (regex)</label>
            <input type="text" id="id-filter" placeholder="e.g., avr4.*" oninput="updateAllPlots()">
        </div>
        <div class="stats-box">
            <div class="stat">
                <div class="stat-value" id="r-squared">-</div>
                <div class="stat-label">R² value</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="n-points">-</div>
                <div class="stat-label">Data points</div>
            </div>
        </div>
    </div>

    <h2>Scatter Plot: Best Model per Job (per Chain Pair)</h2>
    <div class="legend">
        <span class="legend-title">Chain Count:</span>
        {chain_legend_html}
    </div>
    <div class="plot-container" id="scatter-plot"></div>

    <h2>All Models (Jitter Plot)</h2>
    <div class="controls" style="margin-bottom: 10px;">
        <div class="control-group">
            <label>Jitter Metric</label>
            <select id="jitter-metric" onchange="updateJitterPlot()">
                {y_metric_options}
            </select>
        </div>
        <div class="control-group">
            <label>Min Threshold</label>
            <div class="slider-container">
                <input type="range" id="threshold-slider" min="0" max="1" step="0.05" value="{jitter_threshold}" oninput="updateJitterPlot()">
                <span class="threshold-value" id="threshold-display">{jitter_threshold}</span>
            </div>
        </div>
    </div>
    <div class="plot-container" id="jitter-plot"></div>

    <h2>Metric Correlation Matrix</h2>
    <div class="controls" style="margin-bottom: 10px;">
        <div class="control-group">
            <label>Data Source</label>
            <select id="corr-source" onchange="updateCorrelationMatrix()">
                <option value="best" selected>Best model per job</option>
                <option value="all">All models</option>
                <option value="average">Average per job</option>
            </select>
        </div>
        <div class="control-group">
            <label>Filter Metric</label>
            <select id="corr-filter-metric" onchange="updateCorrelationMatrix()">
                <option value="none">No filter</option>
                {y_metric_options}
            </select>
        </div>
        <div class="control-group">
            <label>Filter Threshold</label>
            <div class="slider-container">
                <select id="corr-filter-direction" onchange="updateCorrelationMatrix()" style="min-width: 50px;">
                    <option value="gte">≥</option>
                    <option value="lte">≤</option>
                </select>
                <input type="range" id="corr-threshold-slider" min="0" max="1" step="0.05" value="0" oninput="updateCorrelationMatrix()">
                <span class="threshold-value" id="corr-threshold-display">0.00</span>
            </div>
        </div>
        <div class="stats-box" style="margin-left: 20px;">
            <div class="stat">
                <div class="stat-value" id="corr-n-jobs">-</div>
                <div class="stat-label">Predictions</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="corr-n-interfaces">-</div>
                <div class="stat-label">Interfaces</div>
            </div>
        </div>
    </div>
    <div id="corr-description" style="background: #f0f4f8; padding: 8px 15px; border-radius: 4px; margin-bottom: 10px; font-size: 13px; color: #555;"></div>
    <div class="plot-container" id="correlation-plot"></div>

    <script>
        // Embedded data
        const allData = {json.dumps(all_data_json)};

        // Metrics available for correlation (numeric scoring metrics)
        const corrMetrics = {json.dumps(available_metrics)};

        // Metrics where lower is better (for best model selection)
        const lowerIsBetter = ['pae_mean'];

        // Chain colors (Paul Tol)
        const chainColors = {{
            2: '#4477AA',
            3: '#EE6677',
            4: '#228833',
            5: '#CCBB44'
        }};

        function getChainColor(n) {{
            return chainColors[Math.min(n, 5)] || chainColors[2];
        }}

        function filterByRegex(data, pattern) {{
            if (!pattern) return data;
            try {{
                const regex = new RegExp(pattern, 'i');
                return data.filter(d => regex.test(d.job_name || ''));
            }} catch (e) {{
                return data;  // Invalid regex, return all
            }}
        }}

        // Compute best model per job+chain_pair based on metric
        function getBestModels(data, metric) {{
            const dominated = lowerIsBetter.includes(metric);
            const best = {{}};

            data.forEach(d => {{
                const key = d.job_name + '|' + d.chain_pair;
                const val = d[metric];
                if (val == null || isNaN(val)) return;

                if (!best[key]) {{
                    best[key] = d;
                }} else {{
                    const currentBest = best[key][metric];
                    if (dominated ? val < currentBest : val > currentBest) {{
                        best[key] = d;
                    }}
                }}
            }});

            return Object.values(best);
        }}

        function calculateR2(data, xMetric, yMetric) {{
            const valid = data.filter(d =>
                d[xMetric] != null && d[yMetric] != null &&
                !isNaN(d[xMetric]) && !isNaN(d[yMetric])
            );
            if (valid.length < 3) return null;

            const x = valid.map(d => d[xMetric]);
            const y = valid.map(d => d[yMetric]);

            const n = x.length;
            const sumX = x.reduce((a, b) => a + b, 0);
            const sumY = y.reduce((a, b) => a + b, 0);
            const sumXY = x.reduce((acc, xi, i) => acc + xi * y[i], 0);
            const sumX2 = x.reduce((acc, xi) => acc + xi * xi, 0);
            const sumY2 = y.reduce((acc, yi) => acc + yi * yi, 0);

            const num = n * sumXY - sumX * sumY;
            const den = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));

            if (den === 0) return null;
            const r = num / den;
            return r * r;
        }}

        function updateScatterPlot() {{
            try {{
            const bestMetric = document.getElementById('best-metric').value;
            const xMetric = document.getElementById('x-metric').value;
            const yMetric = document.getElementById('y-metric').value;
            const idFilter = document.getElementById('id-filter').value;

            // Get best models based on selected metric
            const bestData = getBestModels(allData, bestMetric);
            console.log('Best models by', bestMetric, ':', bestData.length);

            let data = filterByRegex(bestData, idFilter);

            // Filter to valid data points
            data = data.filter(d =>
                d[xMetric] != null && d[yMetric] != null &&
                !isNaN(d[xMetric]) && !isNaN(d[yMetric])
            );

            // Calculate R²
            const r2 = calculateR2(data, xMetric, yMetric);
            document.getElementById('r-squared').textContent = r2 !== null ? r2.toFixed(3) : 'N/A';
            document.getElementById('n-points').textContent = data.length;

            // Group by chain count for coloring
            const traces = [];
            const chainCounts = [...new Set(data.map(d => Math.min(d.n_chains || 2, 5)))].sort();

            chainCounts.forEach(n => {{
                const subset = data.filter(d => Math.min(d.n_chains || 2, 5) === n);
                const label = n < 5 ? `${{n}} chains` : '5+ chains';

                traces.push({{
                    x: subset.map(d => d[xMetric]),
                    y: subset.map(d => d[yMetric]),
                    mode: 'markers',
                    type: 'scatter',
                    name: label,
                    marker: {{
                        color: getChainColor(n),
                        size: 10,
                        opacity: 0.7
                    }},
                    text: subset.map(d =>
                        `Job: ${{d.job_name}}<br>` +
                        `Model: ${{d.model}}<br>` +
                        `Chain pair: ${{d.chain_pair}}<br>` +
                        `${{xMetric}}: ${{(d[xMetric] || 0).toFixed(3)}}<br>` +
                        `${{yMetric}}: ${{(d[yMetric] || 0).toFixed(3)}}`
                    ),
                    hoverinfo: 'text'
                }});
            }});

            // Add trend line if enough data
            if (data.length > 2 && r2 !== null) {{
                const x = data.map(d => d[xMetric]);
                const y = data.map(d => d[yMetric]);

                // Simple linear regression
                const n = x.length;
                const sumX = x.reduce((a, b) => a + b, 0);
                const sumY = y.reduce((a, b) => a + b, 0);
                const sumXY = x.reduce((acc, xi, i) => acc + xi * y[i], 0);
                const sumX2 = x.reduce((acc, xi) => acc + xi * xi, 0);

                const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
                const intercept = (sumY - slope * sumX) / n;

                const xMin = Math.min(...x);
                const xMax = Math.max(...x);

                traces.push({{
                    x: [xMin, xMax],
                    y: [slope * xMin + intercept, slope * xMax + intercept],
                    mode: 'lines',
                    type: 'scatter',
                    name: `R² = ${{r2.toFixed(3)}}`,
                    line: {{
                        color: 'rgba(0,0,0,0.3)',
                        dash: 'dash',
                        width: 2
                    }},
                    hoverinfo: 'skip'
                }});
            }}

            const layout = {{
                xaxis: {{ title: xMetric, rangemode: 'tozero' }},
                yaxis: {{ title: yMetric, rangemode: 'tozero' }},
                hovermode: 'closest',
                legend: {{ x: 1.02, y: 1 }},
                margin: {{ r: 150 }}
            }};

            console.log('Rendering scatter plot with', traces.length, 'traces');
            console.log('Data points:', data.length);

            // Clear and create new plot
            Plotly.newPlot('scatter-plot', traces, layout, {{responsive: true}}).then(function() {{
                console.log('Scatter plot rendered successfully');
            }}).catch(function(err) {{
                console.error('Plotly error:', err);
            }});
            }} catch (e) {{
                console.error('Error in updateScatterPlot:', e);
                document.getElementById('scatter-plot').innerHTML = '<p style="color:red;">Error: ' + e.message + '</p>';
            }}
        }}

        // Normalize chain pair so A-B and B-A become the same (alphabetically sorted)
        function normalizeChainPair(chain1, chain2) {{
            return [chain1, chain2].sort().join('-');
        }}

        // Paul Tol colors for chain pairs
        const chainPairColors = [
            '#4477AA', '#EE6677', '#228833', '#CCBB44',
            '#66CCEE', '#AA3377', '#BBBBBB', '#EE7733'
        ];

        function getChainPairColor(idx) {{
            return chainPairColors[idx % chainPairColors.length];
        }}

        function updateJitterPlot() {{
            const metric = document.getElementById('jitter-metric').value;
            const threshold = parseFloat(document.getElementById('threshold-slider').value);
            const idFilter = document.getElementById('id-filter').value;

            document.getElementById('threshold-display').textContent = threshold.toFixed(2);

            let data = filterByRegex(allData, idFilter);

            // Filter to valid data and add normalized chain pair
            data = data.filter(d => d[metric] != null && !isNaN(d[metric]));
            data = data.map(d => ({{
                ...d,
                norm_chain_pair: normalizeChainPair(d.chain1 || 'A', d.chain2 || 'B')
            }}));

            // Get BEST model score per job (max across ALL models and ALL chain pairs)
            // This determines if the job qualifies based on threshold
            const jobBest = {{}};
            data.forEach(d => {{
                const job = d.job_name;
                if (!jobBest[job] || d[metric] > jobBest[job]) {{
                    jobBest[job] = d[metric];
                }}
            }});

            // Filter to jobs where BEST model >= threshold
            // Then show ALL models for qualifying jobs (even if some score below threshold)
            const validJobs = Object.entries(jobBest)
                .filter(([job, best]) => best >= threshold)
                .sort((a, b) => b[1] - a[1])  // Sort by best descending
                .slice(0, 50)  // Limit to 50 jobs
                .map(([job]) => job);

            if (validJobs.length === 0) {{
                Plotly.react('jitter-plot', [], {{
                    annotations: [{{
                        text: `No jobs with best model ${{metric}} ≥ ${{threshold}}`,
                        xref: 'paper', yref: 'paper',
                        x: 0.5, y: 0.5, showarrow: false,
                        font: {{ size: 16 }}
                    }}]
                }}, {{responsive: true}});
                return;
            }}

            // Include ALL models for qualifying jobs
            data = data.filter(d => validJobs.includes(d.job_name));

            // Create jitter positions
            const jobIndex = {{}};
            validJobs.forEach((job, i) => {{ jobIndex[job] = i; }});

            // Get unique chain pairs and assign colors
            const uniquePairs = [...new Set(data.map(d => d.norm_chain_pair))].sort();
            const pairColorMap = {{}};
            uniquePairs.forEach((pair, i) => {{ pairColorMap[pair] = i; }});

            // Group by normalized chain pair for coloring
            const traces = [];

            uniquePairs.forEach((pair, pairIdx) => {{
                const subset = data.filter(d => d.norm_chain_pair === pair);

                // Add tight jitter (reduced from 0.4 to 0.15)
                const xJitter = subset.map(d =>
                    jobIndex[d.job_name] + (Math.random() - 0.5) * 0.15
                );

                traces.push({{
                    x: xJitter,
                    y: subset.map(d => d[metric]),
                    mode: 'markers',
                    type: 'scatter',
                    name: pair,
                    marker: {{
                        color: getChainPairColor(pairIdx),
                        size: 7,
                        opacity: 0.7
                    }},
                    text: subset.map(d =>
                        `Job: ${{d.job_name}}<br>` +
                        `Model: ${{d.model}}<br>` +
                        `Chain pair: ${{d.norm_chain_pair}}<br>` +
                        `${{metric}}: ${{(d[metric] || 0).toFixed(3)}}`
                    ),
                    hoverinfo: 'text'
                }});
            }});

            const layout = {{
                xaxis: {{
                    tickmode: 'array',
                    tickvals: validJobs.map((_, i) => i),
                    ticktext: validJobs,
                    tickangle: 45,
                    range: [-0.7, validJobs.length - 0.3],  // Asymmetric padding - more on left
                    autorange: false,
                    showgrid: false,
                    zeroline: false,
                    showline: false
                }},
                yaxis: {{
                    title: metric,
                    rangemode: 'tozero',
                    gridcolor: '#e0e0e0',
                    zeroline: true,           // Show the y=0 baseline
                    zerolinecolor: '#888',    // Visible gray line at 0
                    zerolinewidth: 1,
                    showline: false
                }},
                hovermode: 'closest',
                legend: {{ x: 1.02, y: 1, title: {{ text: 'Chain Pair' }} }},
                margin: {{ l: 70, r: 150, b: 120, t: 20 }},
                plot_bgcolor: '#fafafa'   // Slight off-white for depth
            }};

            Plotly.react('jitter-plot', traces, layout, {{responsive: true}});
        }}

        // Compute Pearson correlation between two arrays
        function pearsonCorr(x, y) {{
            const n = x.length;
            if (n < 3) return NaN;

            const sumX = x.reduce((a, b) => a + b, 0);
            const sumY = y.reduce((a, b) => a + b, 0);
            const sumXY = x.reduce((acc, xi, i) => acc + xi * y[i], 0);
            const sumX2 = x.reduce((acc, xi) => acc + xi * xi, 0);
            const sumY2 = y.reduce((acc, yi) => acc + yi * yi, 0);

            const num = n * sumXY - sumX * sumY;
            const den = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));

            return den === 0 ? NaN : num / den;
        }}

        // Compute average values per job (for correlation)
        function getAveragePerJob(data, metrics) {{
            const jobData = {{}};

            data.forEach(d => {{
                const key = d.job_name + '|' + normalizeChainPair(d.chain1 || 'A', d.chain2 || 'B');
                if (!jobData[key]) {{
                    jobData[key] = {{ count: 0, sums: {{}} }};
                    metrics.forEach(m => {{ jobData[key].sums[m] = 0; }});
                }}
                jobData[key].count++;
                metrics.forEach(m => {{
                    if (d[m] != null && !isNaN(d[m])) {{
                        jobData[key].sums[m] += d[m];
                    }}
                }});
            }});

            return Object.entries(jobData).map(([key, val]) => {{
                const avg = {{ job_key: key }};
                metrics.forEach(m => {{
                    avg[m] = val.sums[m] / val.count;
                }});
                return avg;
            }});
        }}

        function updateCorrelationMatrix() {{
            const source = document.getElementById('corr-source').value;
            const filterMetric = document.getElementById('corr-filter-metric').value;
            const filterDir = document.getElementById('corr-filter-direction').value;
            const threshold = parseFloat(document.getElementById('corr-threshold-slider').value);

            document.getElementById('corr-threshold-display').textContent = threshold.toFixed(2);

            let data = [...allData];

            // Apply filter if metric selected
            if (filterMetric !== 'none') {{
                data = data.filter(d => {{
                    const val = d[filterMetric];
                    if (val == null || isNaN(val)) return false;
                    return filterDir === 'gte' ? val >= threshold : val <= threshold;
                }});
            }}

            // Count unique jobs and interfaces from filtered raw data
            const uniqueJobs = new Set(data.map(d => d.job_name));
            const uniqueInterfaces = new Set(data.map(d =>
                d.job_name + '|' + normalizeChainPair(d.chain1 || 'A', d.chain2 || 'B')
            ));
            const nJobs = uniqueJobs.size;
            const nInterfaces = uniqueInterfaces.size;
            const nModels = data.length;
            const modelsPerInterface = nInterfaces > 0 ? Math.round(nModels / nInterfaces) : 0;

            // Get data based on source
            let processedData;
            let description = '';
            if (source === 'best') {{
                processedData = getBestModels(data, 'ipSAE');
                description = `Based on <b>best model</b> (by ipSAE) per interface. ` +
                    `${{nJobs}} predictions × ${{Math.round(nInterfaces/nJobs) || 1}} interfaces/prediction = ${{nInterfaces}} data points.`;
            }} else if (source === 'average') {{
                processedData = getAveragePerJob(data, corrMetrics);
                description = `Based on <b>average</b> of ${{modelsPerInterface}} models per interface. ` +
                    `${{nJobs}} predictions × ${{Math.round(nInterfaces/nJobs) || 1}} interfaces/prediction = ${{nInterfaces}} data points.`;
            }} else {{
                processedData = data;
                description = `Based on <b>all models</b>. ` +
                    `${{nJobs}} predictions × ${{Math.round(nInterfaces/nJobs) || 1}} interfaces × ${{modelsPerInterface}} models = ${{nModels}} data points. ` +
                    `<i>(Note: models within a job are correlated)</i>`;
            }}

            // Update stats display
            document.getElementById('corr-n-jobs').textContent = nJobs;
            document.getElementById('corr-n-interfaces').textContent = nInterfaces;
            document.getElementById('corr-description').innerHTML = description;

            if (processedData.length < 3) {{
                document.getElementById('correlation-plot').innerHTML =
                    '<p style="text-align:center;color:#666;">Not enough data points (need ≥3) for correlation matrix</p>';
                return;
            }}

            // Compute correlation matrix
            const matrix = [];
            const labels = corrMetrics.map(m => m.replace('_extracted', '').replace('_calculated', ''));

            for (let i = 0; i < corrMetrics.length; i++) {{
                const row = [];
                for (let j = 0; j < corrMetrics.length; j++) {{
                    const xVals = [];
                    const yVals = [];
                    processedData.forEach(d => {{
                        const x = d[corrMetrics[i]];
                        const y = d[corrMetrics[j]];
                        if (x != null && y != null && !isNaN(x) && !isNaN(y)) {{
                            xVals.push(x);
                            yVals.push(y);
                        }}
                    }});
                    row.push(pearsonCorr(xVals, yVals));
                }}
                matrix.push(row);
            }}

            const trace = {{
                z: matrix,
                x: labels,
                y: labels,
                type: 'heatmap',
                colorscale: 'RdBu',
                zmid: 0,
                zmin: -1,
                zmax: 1,
                text: matrix.map(row => row.map(v => isNaN(v) ? 'N/A' : v.toFixed(2))),
                texttemplate: '%{{text}}',
                hovertemplate: '%{{x}} vs %{{y}}<br>R = %{{z:.3f}}<extra></extra>'
            }};

            const layout = {{
                width: 600,
                height: 500,
                margin: {{ l: 120, r: 50, t: 30, b: 120 }}
            }};

            Plotly.newPlot('correlation-plot', [trace], layout, {{responsive: true}});
        }}

        function updateAllPlots() {{
            updateScatterPlot();
            updateJitterPlot();
            updateCorrelationMatrix();
        }}

        // Initialize plots
        updateScatterPlot();
        updateJitterPlot();
        updateCorrelationMatrix();
    </script>
</body>
</html>'''

    # Write HTML
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html_content)

    print(f"Batch comparison HTML saved to: {output_path}")
    return str(output_path)


def generate_batch_comparison_from_results(
    all_results: List[Dict],
    output_path: str,
    default_x_metric: str = 'ipTM',
    default_y_metric: str = 'ipSAE',
    jitter_threshold: float = 0.3,
) -> Optional[str]:
    """
    Generate batch comparison directly from results list.

    Args:
        all_results: List of result dictionaries (from process_batch)
        output_path: Path to save HTML report
        default_x_metric: Default metric for X axis
        default_y_metric: Default metric for Y axis
        jitter_threshold: Threshold for jitter plot

    Returns:
        Path to generated HTML file, or None if failed
    """
    _check_dependencies()

    if not all_results:
        print("Warning: No results provided for batch comparison")
        return None

    df = pd.DataFrame(all_results)
    if df.empty:
        print("Warning: Empty dataframe for batch comparison")
        return None

    # Save temporary CSV
    temp_csv = Path(output_path).parent / "_temp_combined.csv"
    df.to_csv(temp_csv, index=False)

    try:
        result = generate_batch_comparison_html(
            str(temp_csv),
            output_path,
            default_x_metric=default_x_metric,
            default_y_metric=default_y_metric,
            jitter_threshold=jitter_threshold,
        )
        return result
    finally:
        # Clean up temp file
        if temp_csv.exists():
            temp_csv.unlink()
