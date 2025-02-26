# sensitivity_check_SDI_multiple_allocations.py
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sensitivity_check_SDI import run_sdi_analysis, FPS_SUBSIDY_DEFAULTS
import multiprocessing as mp
from agent_subsidy_pool import SubsidyPoolConfig
import os
from matplotlib.lines import Line2D

def create_combined_visualization(all_results, output_dir='sdi_multiple_allocations_results'):
    """
    Create visualization comparing different allocation strategies.
    
    Args:
        all_results: Dictionary mapping strategy names to result lists
        output_dir: Directory to save visualization outputs
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot setup
    plt.figure(figsize=(14, 10))
    
    # Color and marker setup for distinguishing strategies
    colors = ['#2563eb', '#059669', '#dc2626', '#9333ea', '#f59e0b', '#0284c7']
    markers = ['o', 's', '^', 'D', 'P', '*']
    
    # Line style cycling for trend lines
    line_styles = ['-', '--', '-.', ':']
    
    # Plot each strategy's results
    legend_elements = []
    for idx, (strategy_name, results) in enumerate(all_results.items()):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        line_style = line_styles[idx % len(line_styles)]
        
        # Extract data for low income only
        x_values = []
        y_values = []
        
        for result in results:
            if 'subsidy_pool' in result and 'low' in result:
                x_values.append(result['subsidy_pool'])
                # Handle both tuple and scalar SDI values
                sdi_value = result['low']['sdi']
                if isinstance(sdi_value, tuple):
                    sdi_value = sdi_value[0]
                y_values.append(sdi_value)
        
        if not x_values or not y_values:
            print(f"No valid data for {strategy_name}, skipping...")
            continue
            
        # Sort by subsidy pool size
        sorted_data = sorted(zip(x_values, y_values))
        x_sorted, y_sorted = zip(*sorted_data)
        
        # Plot data points
        plt.scatter(x_sorted, y_sorted, marker=marker, color=color, alpha=0.5, s=60)
        
        if len(x_sorted) >= 3:  # Need at least 3 points for quadratic fit
            # Add trend line
            z = np.polyfit(x_sorted, y_sorted, 2)
            p = np.poly1d(z)
            x_smooth = np.linspace(min(x_sorted), max(x_sorted), 100)
            y_smooth = p(x_smooth)
            plt.plot(x_smooth, y_smooth, linestyle=line_style, color=color, linewidth=2.5)
            
            # Find and mark optimal point
            optimal_x = x_smooth[np.argmax(y_smooth)]
            optimal_y = np.max(y_smooth)
            plt.scatter(optimal_x, optimal_y, marker='*', s=150, color=color, edgecolor='black', zorder=5)
            
            plt.annotate(f"Optimal ({strategy_name}):\n({int(optimal_x)}, {optimal_y:.3f})",
                        xy=(optimal_x, optimal_y),
                        xytext=(20, 10 + 10*idx),  # Offset annotation to prevent overlap
                        textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, edgecolor=color),
                        arrowprops=dict(arrowstyle="->", color=color))
                        
            # Add to legend elements
            legend_elements.append(Line2D([0], [0], color=color, lw=2.5, linestyle=line_style,
                                         label=f"{strategy_name}"))
    
    # Customize plot
    plt.title('SDI Score vs Subsidy Pool Size for Different Allocation Strategies\n(Low Income Commuters)', 
             fontsize=16, pad=20)
    plt.xlabel('Subsidy Pool Size', fontsize=14)
    plt.ylabel('SDI Score', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    
    # Add annotation explaining the analysis
    explanation = (
        "This plot compares different subsidy allocation strategies and their impact on\n"
        "Social Distribution Index (SDI) for low-income commuters. Each line represents\n"
        "a different allocation strategy, with optimal points marked by stars."
    )
    plt.figtext(0.5, 0.01, explanation, ha='center', fontsize=12, 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.2))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust layout to make room for the explanation
    
    # Save plot with high resolution
    plt.savefig(os.path.join(output_dir, 'allocation_strategies_comparison.png'), 
               dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'allocation_strategies_comparison.pdf'), 
               bbox_inches='tight')
    
    plt.close()
    print(f"Visualization saved to {output_dir}")
    
    # Create additional analysis comparing strategies
    create_strategy_comparison_analysis(all_results, output_dir)

def create_strategy_comparison_analysis(all_results, output_dir):
    """
    Create additional analysis plots comparing allocation strategies.
    
    Args:
        all_results: Dictionary mapping strategy names to result lists
        output_dir: Directory to save visualization outputs
    """
    # Prepare data for analysis
    strategy_metrics = {}
    
    for strategy_name, results in all_results.items():
        # Extract metrics for each strategy
        sdi_values = []
        for result in results:
            if 'low' in result:
                sdi_value = result['low']['sdi']
                if isinstance(sdi_value, tuple):
                    sdi_value = sdi_value[0]
                sdi_values.append(sdi_value)
        
        if sdi_values:
            strategy_metrics[strategy_name] = {
                'mean_sdi': np.mean(sdi_values),
                'max_sdi': np.max(sdi_values),
                'min_sdi': np.min(sdi_values),
                'std_sdi': np.std(sdi_values),
                'values': sdi_values
            }
    
    # 1. Bar chart comparison of mean SDI values
    plt.figure(figsize=(12, 6))
    strategies = list(strategy_metrics.keys())
    means = [strategy_metrics[s]['mean_sdi'] for s in strategies]
    std_devs = [strategy_metrics[s]['std_sdi'] for s in strategies]
    
    bars = plt.bar(strategies, means, yerr=std_devs, capsize=8, 
                  color=['#3b82f6', '#10b981', '#ef4444', '#8b5cf6', '#f97316'])
    
    # Annotate bars with values
    for bar, value in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.title('Comparison of Mean SDI Scores Across Allocation Strategies', fontsize=14)
    plt.xlabel('Allocation Strategy', fontsize=12)
    plt.ylabel('Mean SDI Score', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0, max(means) * 1.2)  # Add some space at the top for annotations
    
    plt.savefig(os.path.join(output_dir, 'mean_sdi_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Box plot comparison of SDI distributions
    plt.figure(figsize=(12, 6))
    
    data_to_plot = [strategy_metrics[s]['values'] for s in strategies]
    plt.boxplot(data_to_plot, labels=strategies, patch_artist=True,
               boxprops=dict(facecolor='#d1d5db'),
               medianprops=dict(color='#1e40af', linewidth=2),
               flierprops=dict(marker='o', markerfacecolor='#ef4444'))
    
    plt.title('Distribution of SDI Scores Across Allocation Strategies', fontsize=14)
    plt.xlabel('Allocation Strategy', fontsize=12)
    plt.ylabel('SDI Score', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, 'sdi_distribution_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Summary table as image
    fig, ax = plt.subplots(figsize=(12, len(strategies)*0.8 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = [[s, 
                  f"{strategy_metrics[s]['mean_sdi']:.3f}",
                  f"{strategy_metrics[s]['max_sdi']:.3f}",
                  f"{strategy_metrics[s]['min_sdi']:.3f}",
                  f"{strategy_metrics[s]['std_sdi']:.3f}"] 
                 for s in strategies]
    
    # Create table
    table = ax.table(cellText=table_data, 
                    colLabels=['Strategy', 'Mean SDI', 'Max SDI', 'Min SDI', 'Std Dev'],
                    loc='center', cellLoc='center')
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    # Style header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#1e40af')
        else:
            cell.set_facecolor('#f3f4f6' if row % 2 == 0 else 'white')
    
    plt.title('Summary Statistics for Different Allocation Strategies', fontsize=14, pad=20)
    plt.savefig(os.path.join(output_dir, 'strategy_summary_table.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Main script execution
if __name__ == '__main__':
    # Define base parameters (using your existing structure)
    base_parameters = {
        'num_commuters': 120,
        'grid_width': 55,
        'grid_height': 55,
        'data_income_weights': [0.5, 0.3, 0.2],
        'data_health_weights': [0.9, 0.1],
        'data_payment_weights': [0.8, 0.2],
        'data_age_distribution': {(18, 25): 0.2, (26, 35): 0.3, (36, 45): 0.2, 
                                (46, 55): 0.15, (56, 65): 0.1, (66, 75): 0.05},
        'data_disability_weights': [0.2, 0.8],
        'data_tech_access_weights': [0.95, 0.05],
        'CHANCE_FOR_INSERTING_RANDOM_TRAFFIC': 0.2,
        'ASC_VALUES': {'car': 100, 'bike': 100, 'public': 100, 
                      'walk': 100, 'maas': 100, 'default': 0},
        'UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS': {
            'beta_C': -0.05,
            'beta_T': -0.06
        },
        'UTILITY_FUNCTION_BASE_COEFFICIENTS': {
            'beta_C': -0.05, 'beta_T': -0.06, 
            'beta_W': -0.01, 'beta_A': -0.01, 'alpha': -0.01
        },
        'PENALTY_COEFFICIENTS': {
            'disability_bike_walk': 0.8,
            'age_health_bike_walk': 0.3,
            'no_tech_access_car_bike': 0.1
        },
        'AFFORDABILITY_THRESHOLDS': {'low': 25, 'middle': 85, 'high': 250},
        'FLEXIBILITY_ADJUSTMENTS': {'low': 1.05, 'medium': 1.0, 'high': 0.95},
        'VALUE_OF_TIME': {'low': 9.64, 'middle': 23.7, 'high': 67.2},
        'ALPHA_VALUES': {
            'UberLike1': 0.5,
            'UberLike2': 0.5,
            'BikeShare1': 0.5,
            'BikeShare2': 0.5
        },
        'CONGESTION_T_IJ_FREE_FLOW': 2,
        'BACKGROUND_TRAFFIC_AMOUNT': 70,
        'simulation_steps': 144,
    }
    
    # Define different allocation strategies to test
    allocation_strategies = {
        "Default": {
            'low': {'bike': 0.317, 'car': 0.176, 'MaaS_Bundle': 0.493},
            'middle': {'bike': 0.185, 'car': 0.199, 'MaaS_Bundle': 0.383},
            'high': {'bike': 0.201, 'car': 0.051, 'MaaS_Bundle': 0.297}
        },
        "MaaS Focused": {
            'low': {'bike': 0.25, 'car': 0.15, 'MaaS_Bundle': 0.60},
            'middle': {'bike': 0.15, 'car': 0.15, 'MaaS_Bundle': 0.50},
            'high': {'bike': 0.15, 'car': 0.05, 'MaaS_Bundle': 0.40}
        },
        "Bike Focused": {
            'low': {'bike': 0.50, 'car': 0.10, 'MaaS_Bundle': 0.40},
            'middle': {'bike': 0.40, 'car': 0.15, 'MaaS_Bundle': 0.30},
            'high': {'bike': 0.35, 'car': 0.05, 'MaaS_Bundle': 0.25}
        },
        "Balanced": {
            'low': {'bike': 0.33, 'car': 0.33, 'MaaS_Bundle': 0.34},
            'middle': {'bike': 0.33, 'car': 0.33, 'MaaS_Bundle': 0.34},
            'high': {'bike': 0.33, 'car': 0.33, 'MaaS_Bundle': 0.34}
        },
        "Car Focused": {
            'low': {'bike': 0.20, 'car': 0.55, 'MaaS_Bundle': 0.25},
            'middle': {'bike': 0.15, 'car': 0.65, 'MaaS_Bundle': 0.20},
            'high': {'bike': 0.10, 'car': 0.75, 'MaaS_Bundle': 0.15}
        }
    }
    
    # Simulation parameters
    num_simulations_per_strategy = 30
    num_cpus = 8
    
    # Storage for all results
    all_results = {}
    
    # Run analysis for each allocation strategy
    for strategy_name, allocations in allocation_strategies.items():
        print(f"\n{'='*60}")
        print(f"Running analysis for {strategy_name} allocation strategy...")
        print(f"{'='*60}")
        
        # Create a copy of base parameters with this allocation strategy
        custom_params = base_parameters.copy()
        custom_params['subsidy_dataset'] = allocations
        custom_params['varied_mode'] = 'subsidy'  # Focus on subsidy variations
        
        try:
            # Run the analysis using your existing function
            results, _ = run_sdi_analysis('FPS', custom_params, num_simulations_per_strategy, num_cpus)
            
            # Store results
            all_results[strategy_name] = results
            
            # Save intermediate results in case of failure in later runs
            with open(f'sdi_results_{strategy_name}.pkl', 'wb') as f:
                pickle.dump(results, f)
                
            print(f"✓ Successfully completed {len(results)} simulations for {strategy_name}")
            print(f"  Results saved to: sdi_results_{strategy_name}.pkl")
            
        except Exception as e:
            print(f"✗ Error running analysis for {strategy_name}: {e}")
            print("  Continuing with next strategy...")
    
    # Save complete set of results
    if all_results:
        with open('sdi_multiple_allocation_results.pkl', 'wb') as f:
            pickle.dump(all_results, f)
        print("\nAll results saved to: sdi_multiple_allocation_results.pkl")
        
        # Create comprehensive visualizations
        create_combined_visualization(all_results)
    else:
        print("\nNo valid results to analyze.")