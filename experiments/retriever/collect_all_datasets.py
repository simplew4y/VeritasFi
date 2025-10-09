import json
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict

# Dataset configurations matching eval_all.sh
DATASET_CONFIGS = {
    "zeekr": "/work/xinyu/RAG_Agent/experiments/zeekr/eval",
    "lotus": "/work/xinyu/RAG_Agent/experiments/lotus/eval",
    "financebench": "/work/xinyu/RAG_Agent/experiments/financebench/eval",
    "finder": "/work/xinyu/RAG_Agent/experiments/finder/eval",
    "finqa": "/work/xinyu/RAG_Agent/experiments/finqa/eval"
}

def collect_statistics_for_dataset(dataset_name, eval_dir, step_dir='step3_threshold_0.8'):
    """
    Collect avg_hit_rate and avg_chunk from all statistic.json files for a dataset.
    
    Structure expected: eval_dir/step3/version_name/exp_name/statistic.json
    Example: eval_dir/step3/20_chunks/faiss/statistic.json
    
    Args:
        dataset_name: Name of the dataset
        eval_dir: Path to the eval directory
        step_dir: Name of the step directory (default: step3)
    
    Returns:
        dict: Nested dictionary {version_name: {exp_name: {'hit_rate': float, 'avg_chunk': float}}}
    """
    statistics = defaultdict(dict)
    
    eval_path = Path(eval_dir)
    if not eval_path.exists():
        print(f"Warning: Eval directory not found: {eval_dir}")
        return statistics
    
    # Navigate to step3 directory directly
    step_path = eval_path / step_dir
    if not step_path.exists():
        print(f"Warning: {step_path} not found")
        return statistics
    
    print(f"\n{dataset_name}: Using directory: {step_path}")
    
    # Iterate through version directories (e.g., 20_chunks, 30_chunks, etc.)
    for version_dir in step_path.iterdir():
        if not version_dir.is_dir():
            continue
            
        version_name = version_dir.name
        
        # Iterate through experiment type directories (e.g., faiss, faiss_bm25, etc.)
        for exp_dir in version_dir.iterdir():
            if not exp_dir.is_dir():
                continue
                
            exp_name = exp_dir.name
            stat_file = exp_dir / 'statistic.json'
            
            # Read statistic.json if it exists
            if stat_file.exists():
                try:
                    with open(stat_file, 'r') as f:
                        data = json.load(f)
                        hit_rate = data.get('avg_hit_rate')
                        avg_chunk = data.get('avg_chunk')
                        
                        if hit_rate is not None:
                            statistics[version_name][exp_name] = {
                                'hit_rate': hit_rate,
                                'avg_chunk': avg_chunk
                            }
                            chunks_str = f", avg_chunk: {avg_chunk:.2f}" if avg_chunk is not None else ""
                            print(f"  Loaded: {version_name}/{exp_name} -> avg_hit_rate: {hit_rate:.4f}{chunks_str}")
                        else:
                            print(f"  Warning: No 'avg_hit_rate' in {stat_file}")
                except json.JSONDecodeError:
                    print(f"  Error: Could not parse JSON in {stat_file}")
                except Exception as e:
                    print(f"  Error reading {stat_file}: {e}")
    
    return dict(statistics)

def save_to_csv(statistics, output_file, dataset_name):
    """
    Save statistics to CSV file.
    
    Args:
        statistics: Dictionary of {version_name: {exp_name: {'hit_rate': float, 'avg_chunk': float}}}
        output_file: Output CSV filename
        dataset_name: Name of the dataset
    """
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'version_name', 'exp_name', 'avg_hit_rate', 'avg_chunk'])
        
        for version_name, experiments in sorted(statistics.items()):
            for exp_name, data in sorted(experiments.items()):
                hit_rate = data['hit_rate']
                avg_chunk = data.get('avg_chunk', '')
                writer.writerow([dataset_name, version_name, exp_name, hit_rate, avg_chunk])
    
    print(f"  CSV saved to: {output_file}")

def plot_statistics(statistics, output_dir, dataset_name):
    """
    Create bar plots for each version showing hit rates and avg_chunk by experiment.
    
    Args:
        statistics: Dictionary of {version_name: {exp_name: {'hit_rate': float, 'avg_chunk': float}}}
        output_dir: Directory to save plots
        dataset_name: Name of the dataset
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Calculate grid dimensions based on number of versions
    num_versions = len(statistics)
    if num_versions == 0:
        print(f"  No data to plot for {dataset_name}")
        return
    
    # Determine grid size
    cols = min(3, num_versions)
    rows = (num_versions + cols - 1) // cols
    
    # Create figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(8*cols, 6*rows))
    fig.suptitle(f'{dataset_name.upper()} - Average Hit Rates and Chunks by Version', 
                 fontsize=16, fontweight='bold')
    
    # Flatten axes for easier iteration
    if num_versions == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Plot each version
    for idx, (version_name, experiments) in enumerate(sorted(statistics.items())):
        ax1 = axes[idx]
        
        # Sort experiments by name
        exp_names = sorted(experiments.keys())
        hit_rates = [experiments[exp]['hit_rate'] for exp in exp_names]
        avg_chunks_list = [experiments[exp].get('avg_chunk') for exp in exp_names]
        
        # Check if we have avg_chunk data
        has_chunks = any(nc is not None for nc in avg_chunks_list)
        
        x_pos = np.arange(len(exp_names))
        width = 0.35 if has_chunks else 0.6
        
        # Plot hit rates
        bars1 = ax1.bar(x_pos - width/2 if has_chunks else x_pos, hit_rates, 
                        width, label='Avg Hit Rate', color='steelblue', alpha=0.7, edgecolor='black')
        
        ax1.set_xlabel('Experiment', fontsize=10, fontweight='bold')
        ax1.set_ylabel('Average Hit Rate', fontsize=10, fontweight='bold', color='steelblue')
        ax1.set_title(f'{version_name}', fontsize=12, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=8)
        ax1.tick_params(axis='y', labelcolor='steelblue')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on hit rate bars
        for bar, rate in zip(bars1, hit_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                   f'{rate:.3f}',
                   ha='center', va='bottom', fontsize=7)
        
        # Plot avg_chunk if available
        if has_chunks:
            ax2 = ax1.twinx()
            bars2 = ax2.bar(x_pos + width/2, [nc if nc is not None else 0 for nc in avg_chunks_list], 
                           width, label='Avg Chunks', color='coral', alpha=0.7, edgecolor='black')
            
            ax2.set_ylabel('Average Chunks', fontsize=10, fontweight='bold', color='coral')
            ax2.tick_params(axis='y', labelcolor='coral')
            
            # Add value labels on avg_chunk bars
            for bar, chunks in zip(bars2, avg_chunks_list):
                if chunks is not None:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                           f'{chunks:.1f}',
                           ha='center', va='bottom', fontsize=7)
            
            # Add legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
        
        # Set y-limit for hit rate
        ax1.set_ylim(0, max(hit_rates) * 1.15 if hit_rates else 1)
    
    # Hide unused subplots
    for idx in range(num_versions, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Save combined plot
    combined_path = Path(output_dir) / f'{dataset_name}_all_versions_grid.png'
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"  Grid plot saved to: {combined_path}")
    plt.close()
    
    # Also save individual plots for each version
    for version_name, experiments in sorted(statistics.items()):
        fig_single, ax1_single = plt.subplots(figsize=(12, 7))
        
        exp_names = sorted(experiments.keys())
        hit_rates = [experiments[exp]['hit_rate'] for exp in exp_names]
        avg_chunks_list = [experiments[exp].get('avg_chunk') for exp in exp_names]
        
        has_chunks = any(nc is not None for nc in avg_chunks_list)
        
        x_pos = np.arange(len(exp_names))
        width = 0.35 if has_chunks else 0.6
        
        # Plot hit rates
        bars1 = ax1_single.bar(x_pos - width/2 if has_chunks else x_pos, hit_rates, 
                              width, label='Avg Hit Rate', color='steelblue', alpha=0.7, edgecolor='black')
        
        ax1_single.set_xlabel('Experiment', fontsize=12, fontweight='bold')
        ax1_single.set_ylabel('Average Hit Rate', fontsize=12, fontweight='bold', color='steelblue')
        ax1_single.set_title(f'{dataset_name.upper()} - Average Hit Rates and Chunks - {version_name}', 
                            fontsize=14, fontweight='bold')
        ax1_single.set_xticks(x_pos)
        ax1_single.set_xticklabels(exp_names, rotation=45, ha='right')
        ax1_single.tick_params(axis='y', labelcolor='steelblue')
        ax1_single.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on hit rate bars
        for bar, rate in zip(bars1, hit_rates):
            height = bar.get_height()
            ax1_single.text(bar.get_x() + bar.get_width()/2., height,
                          f'{rate:.3f}',
                          ha='center', va='bottom', fontsize=9)
        
        # Plot avg_chunk if available
        if has_chunks:
            ax2_single = ax1_single.twinx()
            bars2 = ax2_single.bar(x_pos + width/2, [nc if nc is not None else 0 for nc in avg_chunks_list], 
                                  width, label='Avg Chunks', color='coral', alpha=0.7, edgecolor='black')
            
            ax2_single.set_ylabel('Average Chunks', fontsize=12, fontweight='bold', color='coral')
            ax2_single.tick_params(axis='y', labelcolor='coral')
            
            # Add value labels on avg_chunk bars
            for bar, chunks in zip(bars2, avg_chunks_list):
                if chunks is not None:
                    height = bar.get_height()
                    ax2_single.text(bar.get_x() + bar.get_width()/2., height,
                                  f'{chunks:.1f}',
                                  ha='center', va='bottom', fontsize=9)
            
            # Add legend
            lines1, labels1 = ax1_single.get_legend_handles_labels()
            lines2, labels2 = ax2_single.get_legend_handles_labels()
            ax1_single.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        ax1_single.set_ylim(0, max(hit_rates) * 1.15 if hit_rates else 1)
        
        plt.tight_layout()
        
        version_path = Path(output_dir) / f'{dataset_name}_{version_name}.png'
        plt.savefig(version_path, dpi=300, bbox_inches='tight')
        print(f"  Individual plot saved to: {version_path}")
        plt.close()

def process_dataset(dataset_name, eval_dir):
    """
    Process a single dataset: collect statistics, save CSV, and generate plots.
    
    Args:
        dataset_name: Name of the dataset
        eval_dir: Path to the eval directory
    """
    print(f"\n{'='*70}")
    print(f"Processing Dataset: {dataset_name.upper()}")
    print(f"{'='*70}")
    print(f"Eval directory: {eval_dir}")
    
    # Collect statistics
    print("\nCollecting statistics...")
    statistics = collect_statistics_for_dataset(dataset_name, eval_dir)
    
    if not statistics:
        print(f"\nNo statistics found for {dataset_name}. Skipping.")
        return
    
    print(f"\nFound {len(statistics)} versions")
    for version in statistics:
        print(f"  - {version}: {len(statistics[version])} experiments")
    
    # Determine output directory (in the dataset's eval folder)
    output_dir = Path(eval_dir) / 'collected_results'
    output_dir.mkdir(exist_ok=True)
    
    # Save to CSV
    print("\nSaving to CSV...")
    csv_path = output_dir / f'{dataset_name}_hit_rates.csv'
    save_to_csv(statistics, str(csv_path), dataset_name)
    
    # Create plots
    print("\nGenerating plots...")
    plots_dir = output_dir / 'plots'
    plot_statistics(statistics, str(plots_dir), dataset_name)
    
    print(f"\n{dataset_name.upper()} processing complete!")
    print(f"Results saved to: {output_dir}")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Extract and visualize hit rates from all datasets in eval_all.sh'
    )
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        default=['zeekr', 'lotus', 'financebench', 'finder'],
        choices=['zeekr', 'lotus', 'financebench', 'finder', 'finqa'],
        help='Datasets to process (default: all three)'
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("Multi-Dataset Hit Rate Analysis Script")
    print("=" * 70)
    print(f"Datasets to process: {', '.join(args.datasets)}")
    
    # Process each dataset
    for dataset_name in args.datasets:
        eval_dir = DATASET_CONFIGS.get(dataset_name)
        if not eval_dir:
            print(f"\nWarning: Unknown dataset '{dataset_name}', skipping.")
            continue
        
        process_dataset(dataset_name, eval_dir)
    
    print("\n" + "=" * 70)
    print("All Datasets Analysis Complete!")
    print("=" * 70)
    print("\nResults saved in respective dataset folders:")
    for dataset_name in args.datasets:
        eval_dir = DATASET_CONFIGS.get(dataset_name)
        if eval_dir:
            print(f"  - {dataset_name}: {eval_dir}/collected_results/")

if __name__ == "__main__":
    main()
