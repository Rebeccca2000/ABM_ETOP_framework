import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import traceback

def aggregate_mae_results():
    """Aggregate and analyze mode share equity results from multiple simulation runs"""
    print("Starting aggregation of mode share equity results...")
    
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
    output_dir = os.path.join(parent_dir, f"aggregated_results_mae")
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect data by FPS value
    fps_results = {}
    optimal_solutions = []
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
                traceback.print_exc()
        
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
                        elif line.strip() == "Equity Score by Income Level:" or "Mode Share Distribution:" in line:
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
                    in_income_section = False
                    for line in lines:
                        if line.strip() == "Equity Score by Income Level:":
                            in_income_section = True
                            continue
                        elif "Mode Share Distribution:" in line or "Overall Equity Score:" in line:
                            in_income_section = False
                            continue
                            
                        if in_income_section and line.strip().startswith("  "):
                            for level in ["Low", "Middle", "High"]:
                                if line.strip().startswith(f"  {level}:"):
                                    income_scores[level.lower()] = float(line.split(':')[1].strip())
                    
                    opt_data['income_scores'] = income_scores
                    
                    # Try to parse mode share distribution
                    mode_shares = {}
                    in_mode_section = False
                    current_mode = None

                    for line in lines:
                        if "Mode Share Distribution:" in line:
                            in_mode_section = True
                            continue
                        elif "Optimal Subsidy Allocations:" in line:
                            in_mode_section = False
                            continue
                            
                        if in_mode_section and ": " in line:
                            parts = line.strip().split(": ")
                            if len(parts) == 2:
                                mode = parts[0].strip()
                                pct_parts = parts[1].strip().split()
                                try:
                                    if len(pct_parts) > 0:
                                        pct = float(pct_parts[0].strip('%'))
                                        mode_shares[mode] = pct
                                except ValueError:
                                    pass
                    
                    opt_data['mode_shares'] = mode_shares
                    opt_data['directory'] = dir_path
                    optimal_solutions.append(opt_data)
                    
                print(f"  Parsed optimal solution: FPS={opt_data['fps']}, Equity={opt_data['equity']:.4f}")
            except Exception as e:
                print(f"  Error parsing optimal solution from {dir_path}: {e}")
                traceback.print_exc()
    
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
        
        # Extract mode share differences if available
        mode_share_diffs = []
        for r in results_list:
            if 'mode_share_differences' in r:
                mode_share_diffs.append(r['mode_share_differences'])
        
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
                'mean_high': np.mean([s for s in high_scores if s != float('inf')]),
                'mode_share_diffs': mode_share_diffs[0] if mode_share_diffs else None
            })
    
    # Process subsidy usage data
    if subsidy_usage_by_fps:
        print(f"Processing subsidy usage data for {len(subsidy_usage_by_fps)} FPS values")
        subsidy_analysis = []
        for fps, usage_list in subsidy_usage_by_fps.items():
            # Calculate average percentage used
            pct_used = [data.get('percentage_used', 0) for data in usage_list]
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
            create_subsidy_usage_plot(subsidy_df, output_dir)
            create_subsidy_income_distribution_plot(subsidy_df, output_dir)
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
    create_equity_comparison_plot(fps_df, output_dir)
    create_income_equity_plots(fps_df, output_dir)
    
    if optimal_solutions:
        create_optimal_allocation_analysis(optimal_solutions, output_dir)
        create_mode_share_analysis(optimal_solutions, output_dir)
    
    print(f"Aggregation complete! Results saved to {output_dir}")
    return output_dir, fps_results

def create_subsidy_usage_plot(subsidy_df, output_dir):
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

def create_subsidy_income_distribution_plot(subsidy_df, output_dir):
    """Create plots showing distribution of used subsidy by income group"""
    # Line plot
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
    
    # Create stacked area chart to clearly show relative proportions
    plt.figure(figsize=(12, 8))
    plt.stackplot(subsidy_df['fps'], 
                 [subsidy_df['avg_low_pct'], subsidy_df['avg_middle_pct'], subsidy_df['avg_high_pct']],
                 labels=['Low Income', 'Middle Income', 'High Income'],
                 colors=['#1f77b4', '#ff7f0e', '#2ca02c'],
                 alpha=0.7)
    
    plt.title('Relative Distribution of Used Subsidy by Income Group', fontsize=14)
    plt.xlabel('Fixed Pool Subsidy (FPS)', fontsize=12)
    plt.ylabel('Percentage of Used Subsidy (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'subsidy_distribution_stacked.png'), dpi=300)
    plt.close()
    
def create_equity_comparison_plot(fps_df, output_dir):
    """Create equity comparison plot with improved optimal point detection methods"""
    plt.figure(figsize=(14, 10))
    
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
             label='Overall Equity')
    
    # Add error bands using standard deviation
    plt.fill_between(fps_df['fps'],
                    fps_df['mean_equity'] - fps_df['std_equity'],
                    fps_df['mean_equity'] + fps_df['std_equity'],
                    color='#d62728', alpha=0.2)
    
    # Find the raw data optimal point (minimum equity score)
    raw_optimal_idx = fps_df['mean_equity'].idxmin()
    raw_optimal_fps = fps_df.loc[raw_optimal_idx, 'fps']
    raw_optimal_equity = fps_df.loc[raw_optimal_idx, 'mean_equity']
    
    # Add vertical line for raw data optimal
    plt.axvline(x=raw_optimal_fps, color='blue', linestyle='--', alpha=0.7,
               label=f'Optimal FPS (Raw): {raw_optimal_fps:.0f}')
    
    # Add marker for optimal point
    plt.scatter([raw_optimal_fps], [raw_optimal_equity], color='blue', s=150, zorder=10)
    
    # -------- Elbow Point Detection Method --------
    # Calculate normalized metrics for elbow detection
    x = fps_df['fps'].values
    y = fps_df['mean_equity'].values
    
    # Normalize both x and y to [0,1] range
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())
    
    # Calculate distances from points to the line connecting first and last points
    first_point = np.array([x_norm[0], y_norm[0]])
    last_point = np.array([x_norm[-1], y_norm[-1]])
    line_vec = last_point - first_point
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
    
    vec_from_first = np.array([(x_norm[i], y_norm[i]) for i in range(len(x_norm))]) - first_point
    
    # Calculate distances to the line
    scalar_projection = np.sum(vec_from_first * line_vec_norm, axis=1)
    vec_to_line = vec_from_first - scalar_projection.reshape(-1, 1) * line_vec_norm
    distances = np.sqrt(np.sum(vec_to_line**2, axis=1))
    
    # Find the point with maximum distance - this is the elbow point
    elbow_idx = np.argmax(distances)
    elbow_fps = x[elbow_idx]
    elbow_equity = y[elbow_idx]
    
    # Add elbow point to plot
    plt.axvline(x=elbow_fps, color='green', linestyle='--', alpha=0.7,
               label=f'Elbow Point: {elbow_fps:.0f}')
    plt.scatter([elbow_fps], [elbow_equity], color='green', s=150, zorder=10)
    
    # -------- Improvement Rate Analysis --------
    # Calculate percent improvement between adjacent points
    improvements = []
    for i in range(1, len(fps_df)):
        prev_val = fps_df.iloc[i-1]['mean_equity']
        curr_val = fps_df.iloc[i]['mean_equity']
        pct_change = (prev_val - curr_val) / prev_val * 100
        improvements.append({
            'fps': fps_df.iloc[i]['fps'],
            'improvement': pct_change
        })
    
    # Find the point where improvement rate drops below threshold (e.g., 5%)
    threshold = 5  # 5% improvement threshold
    threshold_fps = None
    for imp in improvements:
        if imp['improvement'] < threshold:
            threshold_fps = imp['fps']
            break
    
    if threshold_fps:
        threshold_idx = fps_df[fps_df['fps'] == threshold_fps].index[0]
        threshold_equity = fps_df.loc[threshold_idx, 'mean_equity']
        
        # Add threshold point to plot
        plt.axvline(x=threshold_fps, color='purple', linestyle=':', alpha=0.7,
                   label=f'Threshold ({threshold}% improvement): {threshold_fps:.0f}')
        plt.scatter([threshold_fps], [threshold_equity], color='purple', s=150, zorder=10)
    
    # Add point labels to help explain the plot and provide reference points
    for i, (idx, row) in enumerate(fps_df.iterrows()):
        if i % 2 == 0:  # Only annotate every other point
            fps = row['fps']
            equity = row['mean_equity']
            
            plt.annotate(f'Point {i+1}', 
                        xy=(fps, equity),
                        xytext=(5, 5),
                        textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
    
    # Add note about optimal values
    optimal_text = (f"Raw data optimal FPS = {raw_optimal_fps:.0f}\n"
                   f"Elbow point FPS = {elbow_fps:.0f}\n")
    if threshold_fps:
        optimal_text += f"Threshold ({threshold}%) FPS = {threshold_fps:.0f}"
    
    plt.figtext(0.02, 0.02, optimal_text,
               bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
               fontsize=10)
    
    # Set axis labels and title
    plt.xlabel('Fixed Pool Subsidy (FPS)', fontsize=12)
    plt.ylabel('Equity Score (Lower is Better)', fontsize=12)
    plt.title('Mode Share Equity Scores vs Fixed Pool Subsidy (FPS) Values', fontsize=14)
    
    # Set log scale for x-axis
    plt.xscale('log')
    
    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'optimal_equity_vs_fps.png'), dpi=300)
    plt.close()
    
    # Create a text file with detailed analysis
    with open(os.path.join(output_dir, 'optimal_mae_fps_analysis.txt'), 'w') as f:
        f.write("Mode Share Equity Optimization Analysis\n")
        f.write("=======================================\n\n")
        f.write(f"Raw Data Optimal FPS: {raw_optimal_fps:.0f} (Equity Score: {raw_optimal_equity:.4f})\n")
        f.write(f"Elbow Point FPS: {elbow_fps:.0f} (Equity Score: {elbow_equity:.4f})\n")
        if threshold_fps:
            f.write(f"Improvement Threshold ({threshold}%) FPS: {threshold_fps:.0f} (Equity Score: {threshold_equity:.4f})\n\n")
        
        # Add FPS values ranked by equity score
        f.write("FPS Values Ranked by Equity Score (best to worst):\n")
        f.write("----------------------------------------------\n")
        for _, row in fps_df.sort_values('mean_equity').iterrows():
            f.write(f"FPS: {row['fps']:.0f}, Equity Score: {row['mean_equity']:.4f}\n")
            
        # Add improvement rates table
        f.write("\nImprovement Rates Between FPS Points:\n")
        f.write("----------------------------------\n")
        for imp in improvements:
            f.write(f"FPS: {imp['fps']:.0f}, Improvement: {imp['improvement']:.2f}%\n")
    
    return {
        'raw_optimal': raw_optimal_fps,
        'elbow_point': elbow_fps,
        'threshold': threshold_fps if threshold_fps else None
    }

def create_income_equity_plots(fps_df, output_dir):
    """Create visualizations showing equity scores by income level"""
    # Create stacked area chart
    plt.figure(figsize=(12, 8))
    
    # Calculate proportion of equity score by income level
    for i, row in fps_df.iterrows():
        total = row['mean_low'] + row['mean_middle'] + row['mean_high']
        if total > 0:  # Avoid division by zero
            fps_df.at[i, 'low_pct'] = row['mean_low'] / total * 100
            fps_df.at[i, 'middle_pct'] = row['mean_middle'] / total * 100
            fps_df.at[i, 'high_pct'] = row['mean_high'] / total * 100
    
    # Plot stacked areas for income level contribution
    plt.stackplot(fps_df['fps'], 
                 [fps_df['low_pct'], fps_df['middle_pct'], fps_df['high_pct']],
                 labels=['Low Income', 'Middle Income', 'High Income'],
                 colors=['#1f77b4', '#ff7f0e', '#2ca02c'],
                 alpha=0.7)
    
    # Set axis labels and title
    plt.title('Income Group Contribution to Total Equity Score', fontsize=14)
    plt.xlabel('Fixed Pool Subsidy (FPS)', fontsize=12)
    plt.ylabel('Percentage of Total Equity Score (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'equity_percentage_by_income.png'), dpi=300)
    plt.close()
    
    # Create a bar chart comparing income-specific equity scores
    plt.figure(figsize=(14, 8))
    
    x = np.arange(len(fps_df))
    width = 0.25
    
    # Create grouped bar chart
    plt.bar(x - width, fps_df['mean_low'], width, 
           label='Low Income', color='#1f77b4', alpha=0.8)
    plt.bar(x, fps_df['mean_middle'], width,
           label='Middle Income', color='#ff7f0e', alpha=0.8)
    plt.bar(x + width, fps_df['mean_high'], width,
           label='High Income', color='#2ca02c', alpha=0.8)
    
    # Add total line on top
    plt.plot(x, fps_df['mean_equity'], 'o-', color='#d62728', 
             linewidth=2, label='Total Equity Score')
    
    # Set x-axis labels
    plt.xticks(x, [f"{fps:.0f}" for fps in fps_df['fps']], rotation=45)
    
    # Set axis labels and title
    plt.title('Equity Scores by Income Level', fontsize=14)
    plt.xlabel('Fixed Pool Subsidy (FPS)', fontsize=12)
    plt.ylabel('Equity Score (Lower is Better)', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'equity_by_income_bars.png'), dpi=300)
    plt.close()
    
    # Create heatmap to visualize changes across FPS values
    plt.figure(figsize=(12, 8))
    
    # Prepare data for heatmap
    heatmap_data = fps_df[['fps', 'mean_low', 'mean_middle', 'mean_high']].copy()
    heatmap_data.set_index('fps', inplace=True)
    heatmap_data.columns = ['Low Income', 'Middle Income', 'High Income']
    
    # Create heatmap
    sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd_r', fmt=".2f", 
               cbar_kws={'label': 'Equity Score (Lower is Better)'})
    
    plt.title('Equity Scores by Income Group and FPS', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'equity_heatmap.png'), dpi=300)
    plt.close()

def create_optimal_allocation_analysis(optimal_solutions, output_dir):
    """Create visualizations analyzing optimal subsidy allocations"""
    if not optimal_solutions:
        print("No optimal solutions to analyze")
        return
    
    # Find the overall best solution (minimum equity score)
    best_solution = min(optimal_solutions, key=lambda x: x['equity'])
    
    if 'allocations' in best_solution and best_solution['allocations']:
        # Extract allocation data
        income_mode_data = []
        
        for key, value in best_solution['allocations'].items():
            if '_' in key:
                parts = key.split('_')
                income_level = parts[0].capitalize()
                mode = '_'.join(parts[1:]).replace('_', ' ')
                
                income_mode_data.append({
                    'income_level': income_level,
                    'mode': mode,
                    'allocation': value
                })
        
        alloc_df = pd.DataFrame(income_mode_data)
        
        # Create heatmap of allocations
        plt.figure(figsize=(12, 8))
        
        # Pivot data for heatmap
        pivot_table = alloc_df.pivot(index='income_level', columns='mode', values='allocation')
        
        # Create heatmap
        sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt=".2f", 
                   cbar_kws={'label': 'Allocation Percentage'})
        
        plt.title(f'Optimal Subsidy Allocations (FPS={best_solution["fps"]:.0f}, Equity Score={best_solution["equity"]:.4f})', 
                 fontsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'optimal_allocations_heatmap.png'), dpi=300)
        plt.close()
        
        # Create radar chart for better visualization of allocation patterns
        plt.figure(figsize=(10, 8))
        
        # Get unique modes
        modes = sorted(alloc_df['mode'].unique())
        
        # Setup radar chart
        angles = np.linspace(0, 2*np.pi, len(modes), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        ax = plt.subplot(111, polar=True)
        
        # Plot each income level
        for income_level in ['Low', 'Middle', 'High']:
            values = []
            for mode in modes:
                value = alloc_df[(alloc_df['income_level'] == income_level) & 
                               (alloc_df['mode'] == mode)]['allocation'].values
                values.append(value[0] if len(value) > 0 else 0)
            
            # Close the loop
            values += values[:1]
            
            # Plot the income level allocations
            color = {'Low': '#1f77b4', 'Middle': '#ff7f0e', 'High': '#2ca02c'}[income_level]
            ax.plot(angles, values, 'o-', linewidth=2, label=f'{income_level} Income', color=color)
            ax.fill(angles, values, alpha=0.1, color=color)
        
        # Set chart properties
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(modes)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8'])
        ax.set_ylim(0, 0.8)
        
        plt.title(f'Optimal Allocation Distribution by Income Level (FPS={best_solution["fps"]:.0f})', fontsize=14)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'optimal_allocation_radar.png'), dpi=300)
        plt.close()
        
        # Save the optimal allocation data for further analysis
        with open(os.path.join(output_dir, 'optimal_allocation_details.txt'), 'w') as f:
            f.write(f"Optimal FPS Value: {best_solution['fps']:.0f}\n")
            f.write(f"Minimum Equity Score: {best_solution['equity']:.4f}\n\n")
            
            f.write("Optimal Subsidy Allocations:\n")
            f.write("---------------------------\n")
            for key, value in sorted(best_solution['allocations'].items()):
                f.write(f"  {key}: {value:.4f}\n")
            
            if 'income_scores' in best_solution:
                f.write("\nEquity Scores by Income Level:\n")
                f.write("-----------------------------\n")
                for level, score in best_solution['income_scores'].items():
                    f.write(f"  {level.capitalize()}: {score:.4f}\n")

def create_mode_share_analysis(optimal_solutions, output_dir):
    """Create visualizations for mode share analysis"""
    if not optimal_solutions:
        print("No optimal solutions to analyze")
        return
    
    # Find the solution with the best (lowest) equity score
    best_solution = min(optimal_solutions, key=lambda x: x['equity'])
    
    if 'mode_shares' in best_solution and best_solution['mode_shares']:
        # Extract mode share data
        modes = []
        shares = []
        
        for mode, share in best_solution['mode_shares'].items():
            modes.append(mode)
            shares.append(share)
        
        # Sort by share value for better visualization
        sorted_data = sorted(zip(modes, shares), key=lambda x: x[1], reverse=True)
        modes = [item[0] for item in sorted_data]
        shares = [item[1] for item in sorted_data]
        
        # Create pie chart
        plt.figure(figsize=(12, 8))
        
        # Create color map
        colors = plt.cm.tab10(np.linspace(0, 1, len(modes)))
        
        # Plot pie chart
        plt.pie(shares, labels=modes, autopct='%1.1f%%', 
               startangle=90, colors=colors, shadow=False)
        plt.axis('equal')
        plt.title(f'Mode Share Distribution at Optimal FPS={best_solution["fps"]:.0f}', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'optimal_mode_share_pie.png'), dpi=300)
        plt.close()
        
        # Create bar chart
        plt.figure(figsize=(12, 8))
        
        # Plot horizontal bars
        bars = plt.barh(modes, shares, color=colors, alpha=0.7)
        
        # Add percentage labels
        for bar, share in zip(bars, shares):
            plt.text(min(share + 1, 95), bar.get_y() + bar.get_height()/2, 
                    f"{share:.1f}%", va='center')
        
        plt.title(f'Mode Share Distribution at Optimal FPS={best_solution["fps"]:.0f}', fontsize=14)
        plt.xlabel('Share Percentage (%)', fontsize=12)
        plt.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'optimal_mode_share_bars.png'), dpi=300)
        plt.close()

def plot_normalized_performance(fps_df, output_dir):
    """Create visualization showing diminishing returns analysis"""
    plt.figure(figsize=(12, 8))
    
    # Sort DataFrame by FPS
    df_sorted = fps_df.sort_values('fps')
    
    # Get baseline (worst) value
    baseline = df_sorted['mean_equity'].max()
    
    # Get best value
    best_value = df_sorted['mean_equity'].min()
    
    # Calculate improvement from baseline for each FPS value (lower equity is better)
    improvements = (baseline - df_sorted['mean_equity']) / (baseline - best_value)
    
    # Plot normalized improvement
    plt.plot(df_sorted['fps'], improvements, 'o-', 
             color='#d62728', linewidth=2, markersize=8)
    
    # Add reference lines
    plt.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Add annotations for key points
    plt.annotate(f"90% of max benefit", xy=(df_sorted['fps'].iloc[0]*1.1, 0.9),
               xytext=(df_sorted['fps'].iloc[0]*1.1, 0.9 + 0.05),
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Identify the FPS value where we get 90% of the benefit
    if any(improvements >= 0.9):
        fps_90pct = df_sorted.loc[improvements >= 0.9, 'fps'].iloc[0]
        plt.axvline(x=fps_90pct, color='green', linestyle='--', alpha=0.5)
        plt.annotate(f"FPS={fps_90pct:.0f}", 
                   xy=(fps_90pct, 0.1),
                   xytext=(fps_90pct*1.1, 0.1),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Set axis labels and title
    plt.xlabel('Fixed Pool Subsidy (FPS)', fontsize=12)
    plt.ylabel('Normalized Equity Improvement', fontsize=12)
    plt.title('Mode Share Equity - Diminishing Returns Analysis', fontsize=14)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'diminishing_returns_analysis.png'), dpi=300)
    plt.close()

def save_optimal_allocations(fps_results, output_dir):
    """Save optimal allocations to a text file for cross-policy analysis."""
    if not fps_results:
        print("No results to save")
        return
        
    # Find the FPS with lowest average equity score
    # First, calculate average equity score for each FPS
    avg_equity_scores = {}
    for fps, results_list in fps_results.items():
        # Extract equity scores from all results for this FPS
        equity_scores = []
        for result in results_list:
            if isinstance(result, dict) and 'avg_equity' in result:
                equity_scores.append(result['avg_equity'])
        
        if equity_scores:
            avg_equity_scores[fps] = sum(equity_scores) / len(equity_scores)
    
    # Find optimal FPS (lowest average equity score)
    if not avg_equity_scores:
        print("No valid equity score data found")
        return
        
    optimal_fps = min(avg_equity_scores.items(), key=lambda x: x[1])[0]
    optimal_results_list = fps_results[optimal_fps]
    
    # Get the result with the lowest equity score from the list
    best_result = None
    min_equity_score = float('inf')
    for result in optimal_results_list:
        if isinstance(result, dict) and 'avg_equity' in result:
            if result['avg_equity'] < min_equity_score:
                min_equity_score = result['avg_equity']
                best_result = result
    
    if not best_result:
        print("No valid best result found")
        return
    
    with open(os.path.join(output_dir, 'optimal_allocations.txt'), 'w') as f:
        f.write(f"Optimal FPS Value: {optimal_fps}\n")
        f.write(f"Mode Share Equity Score: {min_equity_score:.4f}\n\n")
        
        # Save allocations in the expected format
        f.write("Optimal Subsidy Allocations:\n")
        if 'optimal_allocations' in best_result and best_result['optimal_allocations']:
            for key, value in sorted(best_result['optimal_allocations'].items()):
                f.write(f"  {key}: {value}\n")
        
        # Write allocations directly if optimal_allocations doesn't exist but fixed_allocations does
        elif 'fixed_allocations' in best_result and best_result['fixed_allocations']:
            for key, value in sorted(best_result['fixed_allocations'].items()):
                f.write(f"  {key}: {value}\n")
        
        # Add equity scores by income level if available
        if 'equity_scores' in best_result and best_result['equity_scores']:
            f.write("\nEquity Score by Income Level:\n")
            for level, score in best_result['equity_scores'].items():
                f.write(f"  {level.capitalize()}: {score}\n")
                    
        # Add mode share distribution if available
        if 'mode_share_differences' in best_result and best_result['mode_share_differences']:
            f.write("\nMode Share Distribution:\n")
            for mode, share in best_result['mode_share_differences'].items():
                f.write(f"  {mode}: {share}%\n")
        
if __name__ == "__main__":
    output_dir, results = aggregate_mae_results()
    # Check if the output directory was created successfully
    if output_dir:
        try:
            # Save optimal allocations for cross-policy analysis
            save_optimal_allocations(results, output_dir)
            print("Optimal allocations saved to text file for cross-policy analysis")
        except Exception as e:
            print(f"Error: {e}")