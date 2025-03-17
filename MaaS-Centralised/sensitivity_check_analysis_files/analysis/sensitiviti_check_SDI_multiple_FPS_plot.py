import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

def plot_stepped_allocation_comparison(data_files_dir='.', output_dir='step_trend_plots'):
    """
    Creates a step-like visualization of different allocation strategies using saved result files,
    with distinct colors (blue, green, red, pink, yellow) and wider steps.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Identify and load result files
    result_files = [f for f in os.listdir(data_files_dir) if f.startswith('sdi_results_') and f.endswith('.pkl')]
    combined_file = [f for f in os.listdir(data_files_dir) if f == 'sdi_multiple_allocation_results.pkl']
    
    all_results = {}
    
    # Try loading the combined file first
    if combined_file:
        try:
            with open(os.path.join(data_files_dir, combined_file[0]), 'rb') as f:
                all_results = pickle.load(f)
            print(f"Loaded combined results file: {combined_file[0]}")
        except Exception as e:
            print(f"Error loading combined file: {e}")
            all_results = {}
    
    # If combined file didn't work, load individual files
    if not all_results:
        for file in result_files:
            try:
                strategy_name = file.replace('sdi_results_', '').replace('.pkl', '')
                with open(os.path.join(data_files_dir, file), 'rb') as f:
                    results = pickle.load(f)
                all_results[strategy_name] = results
                print(f"Loaded {file} for strategy: {strategy_name}")
            except Exception as e:
                print(f"Error loading {file}: {e}")
    
    if not all_results:
        print("No valid result files found!")
        return {}
    
    # Set up figure with explicit color settings
    plt.figure(figsize=(14, 10))
    
    # Define distinct colors with pink instead of purple and yellow instead of orange
    distinct_colors = {
        'Default': '#0000FF',       # Pure Blue
        'MaaS_Focused': '#00CC00',  # Bright Green
        'Bike_Focused': '#FF0000',  # Pure Red
        'Balanced': '#FF00FF',      # Pink (changed from purple)
        'Car_Focused': '#FFCC00'    # Yellow (changed from orange)
    }
    
    # Get number of strategies for consistent segment size calculation
    strategies = list(all_results.keys())
    num_strategies = len(strategies)
    
    # Track optimal points for backwards compatibility
    optimal_points = {}
    
    # Plot each strategy with guaranteed distinct colors
    for strategy_idx, strategy_name in enumerate(strategies):
        results = all_results[strategy_name]
        
        # Get color for this strategy - fallback to a distinct color if not found
        fallback_colors = ['#0000FF', '#00CC00', '#FF0000', '#FF00FF', '#FFCC00']
        color = distinct_colors.get(strategy_name, fallback_colors[strategy_idx % len(fallback_colors)])
        
        # Extract data for low income group
        x_values = []
        y_values = []
        
        for result in results:
            if 'subsidy_pool' in result and 'low' in result:
                x_values.append(result['subsidy_pool'])
                sdi_value = result['low']['sdi']
                if isinstance(sdi_value, tuple):
                    sdi_value = sdi_value[0]
                y_values.append(sdi_value)
        
        if not x_values:
            print(f"No valid data for {strategy_name}, skipping...")
            continue
            
        # Sort by subsidy pool size
        sorted_data = sorted(zip(x_values, y_values))
        x_sorted, y_sorted = zip(*sorted_data)
        
        # Scatter points with color specified using guaranteed color
        plt.scatter(x_sorted, y_sorted, 
                   color=color, marker='o', alpha=0.5, s=30)
        
        # Create wider step segments (fewer segments means wider steps)
        if len(x_sorted) >= 3:
            x_min, x_max = min(x_sorted), max(x_sorted)
            
            # Reduced number of segments for wider steps
            num_segments = min(8, max(5, len(x_sorted) // 3))
            
            # Create segment boundaries
            boundaries = np.linspace(x_min, x_max, num_segments+1)
            
            # Initialize segments
            segment_x = []
            segment_y = []
            
            # Calculate average value in each segment
            for i in range(len(boundaries)-1):
                start, end = boundaries[i], boundaries[i+1]
                
                # Find points in this segment
                segment_indices = [j for j, x in enumerate(x_sorted) if start <= x < end]
                if i == len(boundaries)-2:  # Include the last point in the last segment
                    segment_indices.extend([j for j, x in enumerate(x_sorted) if x == end])
                
                if segment_indices:
                    segment_values = [y_sorted[j] for j in segment_indices]
                    segment_y.append(np.mean(segment_values))
                    segment_point_x = [x_sorted[j] for j in segment_indices]
                    segment_x.append(np.mean(segment_point_x))
            
            # Only create steps if we have segments
            if segment_x:
                # Add start and end points if needed
                if segment_x[0] > x_min:
                    segment_x.insert(0, x_min)
                    segment_y.insert(0, segment_y[0] if segment_y else 0)
                
                if segment_x[-1] < x_max:
                    segment_x.append(x_max)
                    segment_y.append(segment_y[-1] if segment_y else 0)
                
                # Draw step line with guaranteed color - thicker line for better visibility
                plt.step(segment_x, segment_y, 
                        where='post',
                        color=color,
                        linewidth=2.5,
                        alpha=1.0)
                
                # Find optimal point for backward compatibility
                max_idx = np.argmax(y_sorted)
                optimal_points[strategy_name] = (x_sorted[max_idx], y_sorted[max_idx])
    
    # Customize plot with explicit labels
    plt.title('SDI Score vs Subsidy Pool Size for Different Allocation Strategies\n(Low Income Commuters)', 
              fontsize=16, pad=20)
    plt.xlabel('Subsidy Pool Size', fontsize=14)
    plt.ylabel('SDI Score', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Create custom legend with explicitly colored lines
    legend_elements = []
    for strategy_name in strategies:
        color = distinct_colors.get(strategy_name, fallback_colors[strategies.index(strategy_name) % len(fallback_colors)])
        legend_elements.append(
            Line2D([0], [0], color=color, lw=2.5, label=strategy_name.replace('_', ' '))
        )
    
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), 
               loc='upper left', fontsize=12)
    
    # Add annotation explaining the analysis
    explanation = (
        "This plot compares different subsidy allocation strategies using a fine-grained stepped approach\n"
        "that better represents changes in performance across different subsidy ranges.\n"
        "Each line shows the trend of Social Distribution Index (SDI) for low-income commuters\n"
        "with different allocation strategies."
    )
    plt.figtext(0.5, 0.01, explanation, ha='center', fontsize=12, 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.2))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save high-resolution outputs
    plt.savefig(os.path.join(output_dir, 'stepped_allocation_comparison.png'), 
               dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'stepped_allocation_comparison.pdf'), 
               bbox_inches='tight')
    plt.close()
    
    # Create additional summary visualization with updated distinct colors
    create_comparative_trend_analysis(all_results, output_dir, distinct_colors)
    
    return optimal_points

def create_comparative_trend_analysis(all_results, output_dir, distinct_colors):
    """
    Creates additional visualization comparing trends across allocation strategies
    using pink instead of purple and yellow instead of orange.
    """
    # For each strategy, calculate average SDI score in different subsidy ranges
    subsidy_ranges = [(0, 40000), (40000, 80000), (80000, 120000), (120000, 160000)]
    range_labels = ['0-40K', '40K-80K', '80K-120K', '120K-160K']
    
    strategies = list(all_results.keys())
    range_performance = {strategy: [] for strategy in strategies}
    
    # Updated fallback colors
    fallback_colors = ['#0000FF', '#00CC00', '#FF0000', '#FF00FF', '#FFCC00']
    
    for strategy, results in all_results.items():
        # Extract data for low income
        subsidy_sdi_pairs = []
        
        for result in results:
            if 'subsidy_pool' in result and 'low' in result:
                subsidy = result['subsidy_pool']
                sdi_value = result['low']['sdi']
                if isinstance(sdi_value, tuple):
                    sdi_value = sdi_value[0]
                subsidy_sdi_pairs.append((subsidy, sdi_value))
        
        # Calculate average SDI in each subsidy range
        for subsidy_range in subsidy_ranges:
            min_val, max_val = subsidy_range
            range_values = [sdi for subsidy, sdi in subsidy_sdi_pairs 
                          if min_val <= subsidy < max_val]
            
            if range_values:
                avg_sdi = sum(range_values) / len(range_values)
            else:
                avg_sdi = None
                
            range_performance[strategy].append(avg_sdi)
    
    # Create comparative visualization
    plt.figure(figsize=(14, 8))
    
    # Plot range performance for each strategy with updated colors
    bar_width = 0.15
    index = np.arange(len(range_labels))
    
    for i, strategy in enumerate(strategies):
        # Get explicit color with fallback
        color = distinct_colors.get(strategy, fallback_colors[i % len(fallback_colors)])
        
        valid_indices = []
        valid_values = []
        
        for j, value in enumerate(range_performance[strategy]):
            if value is not None:
                valid_indices.append(j)
                valid_values.append(value)
                
        if valid_values:
            plt.bar(index[valid_indices] + i*bar_width, valid_values, 
                   bar_width, label=strategy.replace('_', ' '), 
                   color=color, edgecolor='black', linewidth=0.5)
    
    plt.xlabel('Subsidy Pool Size Range', fontsize=14)
    plt.ylabel('Average SDI Score', fontsize=14)
    plt.title('Comparative Performance of Allocation Strategies Across Subsidy Ranges', fontsize=16)
    plt.xticks(index + bar_width * (len(strategies)-1)/2, range_labels)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=len(strategies))
    plt.grid(axis='y', alpha=0.3)
    
    # Set the y-axis to start from 0
    plt.ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'strategy_range_comparison.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()

# Usage example
if __name__ == '__main__':
    optimal_points = plot_stepped_allocation_comparison()
    print("Optimal points by strategy:")
    for strategy, (x, y) in optimal_points.items():
        print(f"{strategy}: Subsidy = {int(x):,}, SDI = {y:.3f}")