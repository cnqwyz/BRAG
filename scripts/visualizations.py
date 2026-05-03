"""
Visualization scripts for BRAG paper.

Figures:
- Figure 1: Tg prediction scatter plot (predicted vs experimental)
- Figure 2: High-Tg vs Low-Tg polymer performance comparison
- Figure 3: BRAG vs baseline models comparison
- Figure 4: Aggregation function heatmap
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Define colors - unified blue-green color scheme matching heatmap
# Unified font sizes for consistent appearance
FONT_SIZES = {
    'title': 16,
    'axis_label': 14,
    'tick_label': 12,
    'legend': 12,
    'value_label': 12,
    'heatmap_label': 14
}

COLORS = {
    'BRAG': '#4a7c7c',        # Dark teal (main method)
    'BaseGNN': '#8cb8b8',     # Light teal (baseline)
    'AttnGNN': '#6b9a9a',     # Medium-dark teal (attention)
    'Contrastive': '#a8d0d0', # Very light teal (contrastive)
    'High_Tg': '#2d6a6a',     # Very dark teal for high-Tg points (high contrast)
    'Low_Tg': '#88c0c0',      # Light teal for low-Tg points
}

# Model name mapping for display (updated for paper)
MODEL_NAMES = {
    'VanillaGNN': 'BaseGNN',
    'AtomAttentionGNN': 'AttnGNN',
    'BRAG': 'BRAG',
    'Contrastive': 'Contrastive'
}


def load_results(table_num):
    """Load results from JSON file."""
    filepath = Path(__file__).parent.parent / 'results' / f'table{table_num}_results.json'
    with open(filepath, 'r') as f:
        return json.load(f)


def figure2_high_low_temp_comparison():
    """
    Figure 2: High-Tg vs Low-Tg polymer performance comparison.

    Data source: table1_results.json
    Shows BRAG's advantage on high-Tg polymers
    """
    print("Generating Figure 2: High-Tg vs Low-Tg Comparison")

    results = load_results(1)

    models = ['VanillaGNN', 'AtomAttentionGNN', 'BRAG', 'Contrastive']
    model_names = [MODEL_NAMES[m] for m in models]  # Updated: BaseGNN, AttnGNN

    low_tg_mae = [results[m]['low_tg_mae_mean'] for m in models]
    high_tg_mae = [results[m]['high_tg_mae_mean'] for m in models]
    low_tg_std = [results[m]['low_tg_mae_std'] for m in models]
    high_tg_std = [results[m]['high_tg_mae_std'] for m in models]

    fig, ax = plt.subplots(figsize=(11, 6))

    x = np.arange(len(models))
    width = 0.35

    # Low-Tg bars (blue) - no error bars
    bars1 = ax.bar(x - width/2, low_tg_mae, width,
                   label='Low-$T_g$ Polymers (<508K)', color=COLORS['Low_Tg'],
                   alpha=0.85, edgecolor='black', linewidth=1)

    # High-Tg bars (red) - no error bars
    bars2 = ax.bar(x + width/2, high_tg_mae, width,
                   label='High-$T_g$ Polymers (≥508K)', color=COLORS['High_Tg'],
                   alpha=0.85, edgecolor='black', linewidth=1)

    # Add value labels
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=FONT_SIZES['value_label'], fontweight='bold')

    add_labels(bars1)
    add_labels(bars2)

    ax.set_ylabel('Mean Absolute Error (K)', fontsize=FONT_SIZES['axis_label'], fontweight='bold')
    ax.set_xlabel('Model', fontsize=FONT_SIZES['axis_label'], fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=FONT_SIZES['tick_label'])
    ax.legend(fontsize=FONT_SIZES['legend'], loc='upper right', framealpha=0.9)

    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([23, 28])

    plt.tight_layout()

    # Save figure
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'figure2_high_low_temp_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure2_high_low_temp_comparison.pdf', bbox_inches='tight')

    print(f"Figure 2 saved to {output_dir}")
    print(f"  BRAG Low-Tg MAE: {low_tg_mae[2]:.2f} ± {low_tg_std[2]:.2f} K")
    print(f"  BRAG High-Tg MAE: {high_tg_mae[2]:.2f} ± {high_tg_std[2]:.2f} K")
    brag_bias = high_tg_mae[2] - low_tg_mae[2]
    print(f"  BRAG Bias Δ: {brag_bias:.2f} K")
    plt.close()

    return fig


def figure3_baseline_comparison():
    """
    Figure 3: BRAG vs baseline models comprehensive comparison.

    Data source: table1_results.json
    Radar chart showing performance across multiple metrics
    """
    print("\nGenerating Figure 3: BRAG vs Baseline Comparison")

    results = load_results(1)

    models = ['VanillaGNN', 'AtomAttentionGNN', 'BRAG', 'Contrastive']
    model_names = [MODEL_NAMES[m] for m in models]

    # Normalize metrics for radar chart (0-1 scale)
    # Higher is better for all metrics except MAE/RMSE (invert these)
    mae = [results[m]['mae_mean'] for m in models]
    rmse = [results[m]['rmse_mean'] for m in models]
    r2 = [results[m]['r2_mean'] for m in models]
    high_tg_mae = [results[m]['high_tg_mae_mean'] for m in models]
    norm_bias = [results[m]['normalized_bias_mean'] for m in models]  # Lower is better

    # Normalize metrics for radar chart
    # For metrics where HIGHER is better (R²): normalize to [0,1] using min-max scaling
    # For metrics where LOWER is better (MAE, RMSE, Bias): convert to score = (max - value) / range
    # This ensures all metrics are "higher is better" on the radar chart

    def normalize_higher_better(values):
        """Normalize metrics where higher values are better (e.g., R²)"""
        values = np.array(values)
        return (values - values.min()) / (values.max() - values.min() + 1e-6)

    def normalize_lower_better(values):
        """Normalize metrics where lower values are better (e.g., MAE, RMSE, Bias)
        Convert to score so that higher score = better performance"""
        values = np.array(values)
        # Score = (max_value - current_value) / range
        # Best (lowest) value gets score = 1.0
        # Worst (highest) value gets score = 0.0
        return (values.max() - values) / (values.max() - values.min() + 1e-6)

    # Apply normalization
    # MAE/RMSE: lower is better, so invert
    mae_norm = normalize_lower_better(mae)
    rmse_norm = normalize_lower_better(rmse)
    # R²: higher is better
    r2_norm = normalize_higher_better(r2)
    # High-Tg MAE: lower is better, so invert
    high_tg_mae_norm = normalize_lower_better(high_tg_mae)
    # Normalized Bias: lower is better (closer to 0 or negative), so invert
    norm_bias_norm = normalize_lower_better(norm_bias)

    metrics = ['MAE\n(Lower↓)', 'RMSE\n(Lower↓)', 'R²\n(Higher↑)',
               'High-Tg\nMAE\n(Lower↓)', 'Norm.\nBias\n(Lower↓)']
    categories = metrics

    # Radar chart
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection='polar')

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    # Map old model names to color keys for backward compatibility
    color_mapping = {
        'VanillaGNN': 'BaseGNN',
        'AtomAttentionGNN': 'AttnGNN',
        'BRAG': 'BRAG',
        'Contrastive': 'Contrastive'
    }
    colors = [COLORS[color_mapping[m]] for m in models]

    for idx, model in enumerate(models):
        values = [mae_norm[idx], rmse_norm[idx], r2_norm[idx],
                 high_tg_mae_norm[idx], norm_bias_norm[idx]]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2.5, label=MODEL_NAMES[model], color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=FONT_SIZES['axis_label'], fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=FONT_SIZES['tick_label'])
    ax.grid(True, linestyle='--', alpha=0.4)

    # Legend - use less crowded layout
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.05), fontsize=FONT_SIZES['legend'], framealpha=0.9)

    plt.tight_layout()

    # Save figure
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'figure3_baseline_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure3_baseline_comparison.pdf', bbox_inches='tight')

    print(f"Figure 3 saved to {output_dir}")
    print(f"  BRAG R²: {r2[2]:.3f}")
    print(f"  BRAG MAE: {mae[2]:.2f} K")
    print(f"  BRAG Normalized Bias: {norm_bias[2]:.3f}")
    plt.close()

    return fig


def figure1_scatter_plot():
    """
    Figure 1: Tg prediction vs experimental values scatter plot comparison.

    Data source: results/brag_predictions.json and results/basegnn_predictions.json
    Shows side-by-side comparison between BRAG and BaseGNN
    """
    print("\nGenerating Figure 1: BRAG vs BaseGNN Tg Prediction Scatter Plot")

    # Load BRAG predictions
    brag_path = Path(__file__).parent.parent / 'results/brag_predictions.json'

    if not brag_path.exists():
        print(f"ERROR: {brag_path} not found!")
        print("Please run 'python scripts/generate_predictions.py' first.")
        return None

    with open(brag_path, 'r') as f:
        brag_data = json.load(f)

    # Try to load BaseGNN predictions, if not exist, skip
    basegnn_path = Path(__file__).parent.parent / 'results/basegnn_predictions.json'
    basegnn_data = None
    if basegnn_path.exists():
        with open(basegnn_path, 'r') as f:
            basegnn_data = json.load(f)
        print("Loaded BaseGNN predictions for comparison")
    else:
        print("BaseGNN predictions not found, showing BRAG only")

    # BRAG data
    brag_predictions = np.array(brag_data['predictions'])
    brag_targets = np.array(brag_data['targets'])
    brag_high_mask = np.array(brag_data['high_tg_mask'])
    brag_low_mask = np.array(brag_data['low_tg_mask'])

    # Get BRAG metrics
    brag_metrics = brag_data['metrics']
    brag_mae = brag_metrics['mae']
    brag_rmse = brag_metrics['rmse']
    brag_r2 = brag_metrics['r2']
    high_tg_threshold = brag_metrics['high_tg_threshold']

    # Create figure with 1 or 2 subplots
    if basegnn_data is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        axes = [ax1, ax2]

        # BaseGNN data
        basegnn_predictions = np.array(basegnn_data['predictions'])
        basegnn_targets = np.array(basegnn_data['targets'])
        basegnn_high_mask = np.array(basegnn_data['high_tg_mask'])
        basegnn_low_mask = np.array(basegnn_data['low_tg_mask'])
        basegnn_metrics = basegnn_data['metrics']

        # BaseGNN regression
        basegnn_slope, basegnn_intercept, basegnn_r, _, _ = stats.linregress(basegnn_targets, basegnn_predictions)

        # Plot BRAG (ax1)
        ax1.set_title('(a) BRAG Model', fontsize=FONT_SIZES['title'], fontweight='bold', color='black')
        plot_scatter(ax1, brag_targets, brag_predictions, brag_high_mask, brag_low_mask,
                     brag_mae, brag_rmse, brag_r2, high_tg_threshold, COLORS['BRAG'])

        # Plot BaseGNN (ax2)
        ax2.set_title('(b) BaseGNN Model', fontsize=FONT_SIZES['title'], fontweight='bold', color='black')
        plot_scatter(ax2, basegnn_targets, basegnn_predictions, basegnn_high_mask, basegnn_low_mask,
                     basegnn_metrics['mae'], basegnn_metrics['rmse'], basegnn_metrics['r2'],
                     basegnn_metrics['high_tg_threshold'], COLORS['BaseGNN'])

    else:
        # Single plot for BRAG only
        fig, ax1 = plt.subplots(figsize=(10, 8))
        axes = [ax1]

        ax1.set_title('BRAG Model', fontsize=FONT_SIZES['title'], fontweight='bold', color='black')
        plot_scatter(ax1, brag_targets, brag_predictions, brag_high_mask, brag_low_mask,
                     brag_mae, brag_rmse, brag_r2, high_tg_threshold, COLORS['BRAG'])

    # Save figure
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)

    plt.savefig(output_dir / 'figure1_scatter_plot.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure1_scatter_plot.pdf', bbox_inches='tight')

    print(f"Figure 1 saved to {output_dir}")
    print(f"  BRAG High-Tg samples: {brag_high_mask.sum()}, Low-Tg samples: {brag_low_mask.sum()}")

    if basegnn_data is not None:
        improvement = ((basegnn_metrics['mae'] - brag_mae) / basegnn_metrics['mae']) * 100
        print(f"  BRAG improvement: {improvement:.1f}% vs BaseGNN")

    plt.close()

    return fig


def plot_scatter(ax, targets, predictions, high_mask, low_mask, mae, rmse, r2, threshold, color):
    """Helper function to plot scatter plot for a single model."""

    # Plot high-Tg polymers (red)
    ax.scatter(targets[high_mask], predictions[high_mask],
              c=COLORS['High_Tg'], alpha=0.7, s=50, edgecolors='black', linewidth=0.5,
              label=f'High-$T_g$ (≥{threshold:.0f} K, {high_mask.sum()})')

    # Plot low-Tg polymers (blue)
    ax.scatter(targets[low_mask], predictions[low_mask],
              c=COLORS['Low_Tg'], alpha=0.7, s=50, edgecolors='black', linewidth=0.5,
              label=f'Low-$T_g$ (<{threshold:.0f} K, {low_mask.sum()})')

    # Add diagonal line (perfect prediction)
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val],
             'k--', linewidth=1.5, alpha=0.5, label='Perfect (y=x)')

    # Add threshold line (high-Tg/low-Tg boundary)
    ax.axvline(x=threshold, color='red', linestyle='--', linewidth=1.5, alpha=0.6)
    ax.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5, alpha=0.6)

    # Calculate and plot regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(targets, predictions)
    regression_line = slope * targets + intercept
    ax.plot(targets, regression_line, 'g-', linewidth=2, alpha=0.8,
            label=f'Fit: y={slope:.3f}x{intercept:+.1f}')

    # Set labels
    ax.set_xlabel('Experimental $T_g$ (K)', fontsize=FONT_SIZES['axis_label'], fontweight='bold')
    ax.set_ylabel('Predicted $T_g$ (K)', fontsize=FONT_SIZES['axis_label'], fontweight='bold')

    # Set axis limits
    ax.set_xlim([min_val - 10, max_val + 10])
    ax.set_ylim([min_val - 10, max_val + 10])

    # Legend
    ax.legend(loc='lower right', fontsize=FONT_SIZES['legend'], framealpha=0.9)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')


def figure4_aggregation_heatmap():
    """
    Figure 4: Aggregation function performance heatmap.

    Data source: table5_results.json
    Shows MAE for different aggregation strategies
    """
    print("\nGenerating Figure 4: Aggregation Function Heatmap")

    results = load_results(5)

    # Parse aggregation methods
    methods = list(results.keys())

    # Extract aggregation type (add/mean/max) and interaction (cat/diff/abs_diff/add+diff)
    agg_types = []
    interactions = []
    mae_values = []

    for method in methods:
        if method == 'add_add+diff':
            agg_types.append('Add')
            interactions.append('Add+Diff')
        elif method == 'mean_add+diff':
            agg_types.append('Mean')
            interactions.append('Add+Diff')
        elif method == 'max_add+diff':
            agg_types.append('Max')
            interactions.append('Add+Diff')
        elif method.startswith('add_'):
            agg_types.append('Add')
            interactions.append(method[4:])
        elif method.startswith('mean_'):
            agg_types.append('Mean')
            interactions.append(method[5:])
        elif method.startswith('max_'):
            agg_types.append('Max')
            interactions.append(method[4:])

        mae_values.append(results[method]['mae_mean'])

    # Create DataFrame for heatmap
    agg_order = ['Add', 'Mean', 'Max']
    inter_order = ['cat', 'diff', 'abs_diff', 'Add+Diff']

    # Build matrix
    mae_matrix = np.zeros((len(agg_order), len(inter_order)))
    for agg, inter, mae in zip(agg_types, interactions, mae_values):
        agg_idx = agg_order.index(agg)
        inter_idx = inter_order.index(inter)
        mae_matrix[agg_idx, inter_idx] = mae


    # Create heatmap
    fig, ax = plt.subplots(figsize=(11, 6))

    # Create custom colormap
    # Colors: lighter green-blue -> darker (best to worst performance)
    custom_purple = LinearSegmentedColormap.from_list(
        'custom_colors',
        ['#f5faf3',  # Lightest green-white (best performance)
         '#aed2ca',  # Light blue-green
         '#99c7bf',  # Medium blue-green
         '#789999']   # Medium-dark blue-green (worst performance)
    )

    # Lower MAE = better performance = lighter colors
    # Higher MAE = worse performance = darker colors
    im = ax.imshow(mae_matrix, cmap=custom_purple, aspect='auto', vmin=24.3, vmax=30.0)

    # Set ticks (remove grid lines)
    ax.set_xticks(np.arange(len(inter_order)))
    ax.set_yticks(np.arange(len(agg_order)))
    ax.set_xticklabels(inter_order, fontsize=14, fontweight='bold')
    ax.set_yticklabels(agg_order, fontsize=14, fontweight='bold')
    ax.grid(False)  # Remove grid lines

    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    # Find best method first (need min_mae for both loops)
    min_mae = mae_matrix.min()

    # Add text annotations (skip the best cell)
    for i in range(len(agg_order)):
        for j in range(len(inter_order)):
            if mae_matrix[i, j] != min_mae:
                # Use contrasting text color based on background brightness
                value = mae_matrix[i, j]
                if value < 25.5:
                    text_color = 'black'  # Light background
                else:
                    text_color = 'black'  # Darker background but still readable
                text = ax.text(j, i, f'{value:.2f}',
                              ha="center", va="center", color=text_color,
                              fontsize=13, fontweight='bold')

    # Highlight best method with bold border
    for i in range(len(agg_order)):
        for j in range(len(inter_order)):
            if mae_matrix[i, j] == min_mae:
                # Add bold white text with black outline for best cell
                ax.text(j, i, f'{min_mae:.2f}', ha="center", va="center",
                       color="black", fontsize=15, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor='red', linewidth=2.5, alpha=0.9))

    ax.set_xlabel('Interaction Method', fontsize=14, fontweight='bold')
    ax.set_ylabel('Aggregation Type', fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean Absolute Error (K)', rotation=270, labelpad=20,
                  fontsize=13, fontweight='bold')

    plt.tight_layout()

    # Save figure
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'figure4_aggregation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure4_aggregation_heatmap.pdf', bbox_inches='tight')

    print(f"Figure 4 saved to {output_dir}")
    # Find best method for print output
    best_method_idx = np.unravel_index(mae_matrix.argmin(), mae_matrix.shape)
    best_agg = agg_order[best_method_idx[0]]
    best_inter = inter_order[best_method_idx[1]]
    print(f"  Best aggregation: {best_agg} + {best_inter}")
    print(f"  Best MAE: {min_mae:.2f} K")
    plt.close()

    return fig


def figure5_bias_analysis():
    """
    Figure 5: Bias analysis comparison across models.

    Data source: table2_results.json
    Shows BRAG's bias reduction advantage over baselines
    """
    print("\nGenerating Figure 5: Bias Analysis")

    results = load_results(2)

    models = ['VanillaGNN', 'AtomAttentionGNN', 'BRAG', 'Contrastive']
    model_names = [MODEL_NAMES[m] for m in models]

    # Bias metrics
    bias_delta = [results[m]['bias_delta_mean'] for m in models]
    bias_delta_std = [results[m]['bias_delta_std'] for m in models]
    normalized_bias = [results[m]['normalized_bias_mean'] for m in models]
    normalized_bias_std = [results[m]['normalized_bias_std'] for m in models]

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(models))
    width = 0.6

    # Color mapping
    color_mapping = {
        'VanillaGNN': 'BaseGNN',
        'AtomAttentionGNN': 'AttnGNN',
        'BRAG': 'BRAG',
        'Contrastive': 'Contrastive'
    }
    colors = [COLORS[color_mapping[m]] for m in models]

    # Subplot 1: Bias Delta (High-Tg MAE - Low-Tg MAE)
    bars1 = ax1.bar(x, bias_delta, width,
                    color=colors, alpha=0.85, edgecolor='black', linewidth=1)

    # Add zero line
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, bias_delta)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:+.2f}',
                ha='center', va='bottom' if val > 0 else 'top',
                fontsize=14, fontweight='bold')

    ax1.set_ylabel(r'$\Delta_{bias}$ [K]', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, fontsize=12)
    ax1.set_title('(a) Prediction Bias (High-$T_g$ vs Low-$T_g$)', fontsize=15, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Subplot 2: Normalized Bias
    bars2 = ax2.bar(x, normalized_bias, width,
                    color=colors, alpha=0.85, edgecolor='black', linewidth=1)

    # Add value labels
    for bar, val in zip(bars2, normalized_bias):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom',
                fontsize=14, fontweight='bold')

    ax2.set_ylabel('Normalized Bias', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, fontsize=12)
    ax2.set_title('(b) Normalized Bias Magnitude', fontsize=15, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim([0.25, 0.35])

    plt.tight_layout()

    # Save figure
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'figure5_bias_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure5_bias_analysis.pdf', bbox_inches='tight')

    print(f"Figure 5 saved to {output_dir}")
    # Calculate bias metrics for print output
    brag_bias = bias_delta[2]
    others_avg = np.mean([bias_delta[i] for i in [0, 1, 3]])
    print(f"  BRAG Bias Δ: {brag_bias:+.2f} K")
    print(f"  Baseline avg: {others_avg:+.2f} K")
    print(f"  Improvement: {abs(brag_bias - others_avg):.2f} K")
    plt.close()

    return fig


def main():
    """Generate all figures."""
    print("=" * 60)
    print("Generating Figures for BRAG Paper")
    print("=" * 60)

    # Try to generate Figure 1
    try:
        figure1_scatter_plot()
    except Exception as e:
        print(f"\nWarning: Could not generate Figure 1")
        print(f"  Error: {e}")
        print(f"  To fix: Run 'python scripts/generate_predictions.py' first")
        print("")

    # Generate figures 2, 3, 4, 5
    figure2_high_low_temp_comparison()
    figure3_baseline_comparison()
    figure4_aggregation_heatmap()
    figure5_bias_analysis()

    print("\n" + "=" * 60)
    print("All available figures generated successfully!")
    print("Output directory: figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
