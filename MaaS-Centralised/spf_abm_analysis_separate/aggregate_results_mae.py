import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def aggregate_simulation_results():
    """Aggregate and analyze results from multiple simulation runs"""
    print("Starting aggregation of simulation results...")
    
    # Get parent directory (MaaS-Centralised) where the result folders are located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    os.chdir(parent_dir)  # Change to parent directory where results are located
    print(f"Looking for results in: {parent_dir}")
    
    # Find all equity results directories
    result_dirs = glob.glob("mae_equity_results_*")
    print(f"Found {len(result_dirs)} simulation result directories")
    
    if not result_dirs:
        print("No simulation result directories found. Please check the path.")
        return None
    
    # Create output directory for aggregated results
    output_dir = os.path.join(parent_dir, f"aggregated_results_mae_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect data by FPS value
    fps_results = {}
    optimal_solutions = []
    # Add collection for subsidy usage data
    subsidy_usage_by_fps = {}
    
    # Process each simulation directory
    for dir_path in result_dirs:
        print(f"Processing {dir_path}...")
        
        # Load all_results.pkl
        all_results_path = os.path.join(dir_path, 'all_results.pkl')
        if os.path.exists(all_results_path):
            try:
                with open(all_results_path, 'rb') as f:
                    results = pickle.load(f)
                    
                print(f"  Loaded results with {len(results)} FPS values")
                
                # Add results to main collection
                for fps, result in results.items():
                    fps_float = float(fps)
                    if fps_float not in fps_results:
                        fps_results[fps_float] = []
                    
                    # Add directory info for tracking
                    result['directory'] = dir_path
                    fps_results[fps_float].append(result)
                    
                    # Extract subsidy usage data
                    subsidy_data = None
                    if 'subsidy_usage' in result:
                        subsidy_data = result['subsidy_usage']
                        print(f"  Found subsidy_usage directly in result for FPS {fps}")
                    elif 'full_results' in result and 'subsidy_usage' in result['full_results']:
                        subsidy_data = result['full_results']['subsidy_usage']
                        print(f"  Found subsidy_usage in full_results for FPS {fps}")
                    
                    if subsidy_data:
                        if fps_float not in subsidy_usage_by_fps:
                            subsidy_usage_by_fps[fps_float] = []
                        subsidy_usage_by_fps[fps_float].append(subsidy_data)
                    else:
                        print(f"  No subsidy_usage data for FPS {fps}")
                        
            except Exception as e:
                print(f"  Error loading results from {dir_path}: {e}")
        
        # Parse optimal solution
        opt_path = os.path.join(dir_path, 'optimal_solution.txt')
        if os.path.exists(opt_path):
            try:
                opt_data = {}
                with open(opt_path, 'r') as f:
                    lines = f.readlines()
                    
                    # Parse FPS and equity score
                    opt_data['fps'] = float(lines[0].split(':')[1].strip())
                    opt_data['equity'] = float(lines[1].split(':')[1].strip())
                    
                    # Parse allocations
                    allocations = {}
                    in_allocation_section = False
                    for line in lines:
                        if line.strip() == "Optimal Subsidy Allocations:":
                            in_allocation_section = True
                            continue
                        elif line.strip() == "Equity Score by Income Level:":
                            in_allocation_section = False
                            continue
                            
                        if in_allocation_section and line.strip().startswith("  "):
                            parts = line.strip().split(":")
                            key = parts[0].strip()
                            value = float(parts[1].strip())
                            allocations[key] = value
                    
                    opt_data['allocations'] = allocations
                    
                    # Parse income-level equity scores
                    income_scores = {}
                    for line in lines:
                        for level in ["Low", "Middle", "High"]:
                            if line.strip().startswith(f"  {level}:"):
                                income_scores[level.lower()] = float(line.split(':')[1].strip())
                    
                    opt_data['income_scores'] = income_scores
                    opt_data['directory'] = dir_path
                    optimal_solutions.append(opt_data)
                    
                print(f"  Parsed optimal solution: FPS={opt_data['fps']}, Equity={opt_data['equity']:.4f}")
            except Exception as e:
                print(f"  Error parsing optimal solution from {dir_path}: {e}")
    
    # Create combined analysis of FPS results
    fps_analysis = []
    for fps, results_list in fps_results.items():
        # Get equity scores for this FPS across all runs
        equity_scores = [r.get('avg_equity', float('inf')) for r in results_list]
        valid_scores = [s for s in equity_scores if s != float('inf')]
        
        # Get income-specific scores
        low_scores = [r.get('equity_scores', {}).get('low', float('inf')) for r in results_list]
        mid_scores = [r.get('equity_scores', {}).get('middle', float('inf')) for r in results_list]
        high_scores = [r.get('equity_scores', {}).get('high', float('inf')) for r in results_list]
        
        if valid_scores:
            fps_analysis.append({
                'fps': fps,
                'mean_equity': np.mean(valid_scores),
                'std_equity': np.std(valid_scores),
                'min_equity': np.min(valid_scores),
                'max_equity': np.max(valid_scores),
                'count': len(valid_scores),
                'mean_low': np.mean([s for s in low_scores if s != float('inf')]),
                'mean_middle': np.mean([s for s in mid_scores if s != float('inf')]),
                'mean_high': np.mean([s for s in high_scores if s != float('inf')])
            })
    
    # Process subsidy usage data
    if subsidy_usage_by_fps:
        print(f"Processing subsidy usage data for {len(subsidy_usage_by_fps)} FPS values")
        subsidy_analysis = []
        for fps, usage_list in subsidy_usage_by_fps.items():
            # Calculate average percentage used
            pct_used = [data['percentage_used'] for data in usage_list]
            avg_pct_used = np.mean(pct_used) if pct_used else 0
            
            # Calculate income group distributions
            low_pct = []
            middle_pct = []
            high_pct = []
            
            for data in usage_list:
                if 'subsidy_by_income' in data:
                    income_data = data['subsidy_by_income']
                    low_pct.append(income_data.get('low', {}).get('percentage_of_used', 0))
                    middle_pct.append(income_data.get('middle', {}).get('percentage_of_used', 0))
                    high_pct.append(income_data.get('high', {}).get('percentage_of_used', 0))
            
            subsidy_analysis.append({
                'fps': fps,
                'avg_pct_used': avg_pct_used,
                'avg_low_pct': np.mean(low_pct) if low_pct else 0,
                'avg_middle_pct': np.mean(middle_pct) if middle_pct else 0,
                'avg_high_pct': np.mean(high_pct) if high_pct else 0
            })
        
        # Convert to DataFrame
        subsidy_df = pd.DataFrame(subsidy_analysis).sort_values('fps')
        
        # Save to CSV
        subsidy_df.to_csv(os.path.join(output_dir, 'subsidy_usage_analysis.csv'), index=False)
        
        # Create visualizations
        if len(subsidy_df) > 0:
            create_fps_usage_plot(subsidy_df, output_dir)
            create_income_distribution_plot(subsidy_df, output_dir)
            print(f"Created subsidy usage visualizations with {len(subsidy_df)} data points")
    else:
        print("No subsidy usage data found in results")
    
    # Handle empty fps_analysis
    if not fps_analysis:
        print("No valid results found to analyze.")
        return None
    
    # Convert to DataFrame and sort by FPS
    fps_df = pd.DataFrame(fps_analysis).sort_values('fps')
    fps_df.to_csv(os.path.join(output_dir, 'fps_analysis.csv'), index=False)
    
    # Create visualizations
    create_fps_comparison_plot(fps_df, output_dir)
    
    print(f"Aggregation complete! Results saved to {output_dir}")
    return output_dir

def create_fps_usage_plot(subsidy_df, output_dir):
    """Create a plot showing percentage of FPS budget used"""
    plt.figure(figsize=(12, 8))
    
    # Plot the percentage used
    plt.plot(subsidy_df['fps'], subsidy_df['avg_pct_used'], 'o-', 
             color='green', linewidth=2, markersize=8)
    
    # Add data labels
    for i, row in subsidy_df.iterrows():
        plt.annotate(f"{row['avg_pct_used']:.1f}%",
                    xy=(row['fps'], row['avg_pct_used']),
                    xytext=(5, 5),
                    textcoords='offset points')
    
    # Set axis labels and title
    plt.title('Percentage of FPS Budget Used', fontsize=14)
    plt.xlabel('Fixed Pool Subsidy (FPS)', fontsize=12)
    plt.ylabel('Percentage Used (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    # Set y-axis to start from 0 and go to 105% to include annotations
    plt.ylim(0, max(105, subsidy_df['avg_pct_used'].max() * 1.1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'subsidy_usage_percentage.png'), dpi=300)
    plt.close()

def create_income_distribution_plot(subsidy_df, output_dir):
    """Create a plot showing distribution of used subsidy by income group"""
    plt.figure(figsize=(12, 8))
    
    # Plot lines for each income level
    plt.plot(subsidy_df['fps'], subsidy_df['avg_low_pct'], 'o-', 
             color='#1f77b4', linewidth=2, markersize=8, 
             label='Low Income')
    
    plt.plot(subsidy_df['fps'], subsidy_df['avg_middle_pct'], 'o-', 
             color='#ff7f0e', linewidth=2, markersize=8, 
             label='Middle Income')
    
    plt.plot(subsidy_df['fps'], subsidy_df['avg_high_pct'], 'o-', 
             color='#2ca02c', linewidth=2, markersize=8, 
             label='High Income')
    
    # Set axis labels and title
    plt.title('Distribution of Used Subsidy by Income Group', fontsize=14)
    plt.xlabel('Fixed Pool Subsidy (FPS)', fontsize=12)
    plt.ylabel('Percentage of Used Subsidy (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    # Add legend
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'subsidy_distribution_by_income.png'), dpi=300)
    plt.close()
    
def create_fps_comparison_plot(fps_df, output_dir):
    """Create FPS comparison plot similar to the reference plot"""
    plt.figure(figsize=(12, 8))
    
    # Plot lines for each income level with markers
    plt.plot(fps_df['fps'], fps_df['mean_low'], 'o-', 
             color='#1f77b4', linewidth=2, markersize=8, 
             label='Low Income')
    
    plt.plot(fps_df['fps'], fps_df['mean_middle'], 'o-', 
             color='#ff7f0e', linewidth=2, markersize=8, 
             label='Middle Income')
    
    plt.plot(fps_df['fps'], fps_df['mean_high'], 'o-', 
             color='#2ca02c', linewidth=2, markersize=8, 
             label='High Income')
    
    # Plot Sum Equity with more emphasis
    plt.plot(fps_df['fps'], fps_df['mean_equity'], 'o-', 
             color='#d62728', linewidth=3, markersize=8, 
             label='Sum Equity')
    
    # Add trend line using polynomial fit
    try:
        # Use log of fps for better fitting
        z = np.polyfit(np.log(fps_df['fps']), fps_df['mean_equity'], 2)
        p = np.poly1d(z)
        
        # Create smooth x values for the trend line
        x_smooth = np.geomspace(fps_df['fps'].min(), fps_df['fps'].max(), 100)
        y_smooth = p(np.log(x_smooth))
        
        plt.plot(x_smooth, y_smooth, '--', color='gray', alpha=0.6, label='Trend Line')
        
        # Find optimal point based on trend
        opt_idx = np.argmin(y_smooth)
        opt_fps = x_smooth[opt_idx]
        
        # Add vertical line at optimal point
        plt.axvline(x=opt_fps, color='red', linestyle='--', alpha=0.5)
    except Exception as e:
        print(f"Error fitting trend line: {e}")
    
    # Add point labels to Sum Equity line
    for i, (idx, row) in enumerate(fps_df.iterrows()):
        if i % 2 == 0:  # Label every other point
            fps = row['fps']
            equity = row['mean_equity']
            
            plt.annotate(f'Point {i+1}', 
                        xy=(fps, equity),
                        xytext=(5, 5),
                        textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
    
    # Set axis labels and title
    plt.xlabel('Fixed Pool Subsidy (FPS)', fontsize=12)
    plt.ylabel('Equity Score (Lower is Better)', fontsize=12)
    plt.title('Optimal Equity Scores vs Fixed Pool Subsidy (FPS) Values', fontsize=14)
    
    # Set log scale for x-axis
    plt.xscale('log')
    
    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Add note about allocation table
    plt.figtext(0.02, 0.02, 
               "Note: See 'equity_allocation_table.csv' for the complete allocations table",
               fontsize=8)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'optimal_equity_vs_fps.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    aggregate_simulation_results()