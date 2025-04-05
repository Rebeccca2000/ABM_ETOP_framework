import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import traceback

def aggregate_travel_time_equity_results():
    """Aggregate and analyze results from multiple travel time equity optimization runs"""
    print("Starting aggregation of travel time equity optimization results...")
    
    # Get parent directory (MaaS-Centralised) where the result folders are located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    os.chdir(parent_dir)  # Change to parent directory where results are located
    print(f"Looking for results in: {parent_dir}")
    
    # Find all travel time equity results directories
    result_dirs = glob.glob("travel_time_equity_results_*")
    print(f"Found {len(result_dirs)} simulation result directories")
    
    if not result_dirs:
        print("No simulation result directories found. Please check the path.")
        return None
    
    # Create output directory for aggregated results
    output_dir = os.path.join(parent_dir, f"aggregated_results_travel_time_equity")
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
                    if 'subsidy_usage' in result:
                        if fps_float not in subsidy_usage_by_fps:
                            subsidy_usage_by_fps[fps_float] = []
                        subsidy_usage_by_fps[fps_float].append(result['subsidy_usage'])
                        
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
                    opt_data['travel_time_equity_index'] = float(lines[1].split(':')[1].strip())
                    
                    # Parse allocations
                    allocations = {}
                    in_allocation_section = False
                    for line in lines:
                        if line.strip() == "Optimal Subsidy Allocations:":
                            in_allocation_section = True
                            continue
                        elif line.strip() == "Travel Time Deviation by Income Level:":
                            in_allocation_section = False
                            continue
                            
                        if in_allocation_section and line.strip().startswith("  "):
                            parts = line.strip().split(":")
                            key = parts[0].strip()
                            value = float(parts[1].strip())
                            allocations[key] = value
                    
                    opt_data['allocations'] = allocations
                    
                    # Parse travel time deviations
                    deviations = {}
                    in_deviation_section = False
                    for line in lines:
                        if line.strip() == "Travel Time Deviation by Income Level:":
                            in_deviation_section = True
                            continue
                        elif line.strip() == "Average Travel Times by Income Level:":
                            in_deviation_section = False
                            continue
                            
                        if in_deviation_section and line.strip().startswith("  "):
                            parts = line.strip().split(":")
                            level = parts[0].strip().lower()
                            value = float(parts[1].split()[0].strip())  # Extract numeric value
                            deviations[level] = value
                    
                    # Parse average travel times
                    avg_travel_times = {}
                    in_travel_time_section = False
                    for line in lines:
                        if line.strip() == "Average Travel Times by Income Level:":
                            in_travel_time_section = True
                            continue
                        elif line.strip().startswith("Overall Average Travel Time:"):
                            # Extract overall average travel time
                            opt_data['overall_avg_travel_time'] = float(line.split(':')[1].split()[0].strip())
                            in_travel_time_section = False
                            continue
                            
                        if in_travel_time_section and line.strip().startswith("  "):
                            parts = line.strip().split(":")
                            level = parts[0].strip().lower()
                            value = float(parts[1].split()[0].strip())  # Extract numeric value
                            avg_travel_times[level] = value
                    
                    opt_data['deviations'] = deviations
                    opt_data['avg_travel_times'] = avg_travel_times
                    opt_data['directory'] = dir_path
                    optimal_solutions.append(opt_data)
                    
                print(f"  Parsed optimal solution: FPS={opt_data['fps']}, Travel Time Equity Index={opt_data['travel_time_equity_index']:.4f}")
            except Exception as e:
                print(f"  Error parsing optimal solution from {dir_path}: {e}")
                traceback.print_exc()
    
    # Create combined analysis of FPS results
    fps_analysis = []
    for fps, results_list in fps_results.items():
        # Get equity scores for this FPS across all runs
        equity_scores = [r.get('travel_time_equity_index', float('inf')) for r in results_list]
        valid_scores = [s for s in equity_scores if s != float('inf')]
        
        # Get income-specific deviation scores
        low_devs = [r.get('deviations', {}).get('low', float('inf')) for r in results_list]
        mid_devs = [r.get('deviations', {}).get('middle', float('inf')) for r in results_list]
        high_devs = [r.get('deviations', {}).get('high', float('inf')) for r in results_list]
        
        # Get average travel times from full results
        avg_low_times = []
        avg_mid_times = []
        avg_high_times = []
        overall_avg_times = []
        
        for r in results_list:
            if 'full_results' in r:
                for level, key in [('low', 'avg_low_times'), ('middle', 'avg_mid_times'), 
                                  ('high', 'avg_high_times')]:
                    if level in r['full_results']:
                        locals()[key].append(r['full_results'][level].get('avg_travel_time', 0))
                
                overall_avg_times.append(r['full_results'].get('overall_avg_travel_time', 0))
        
        if valid_scores:
            fps_analysis.append({
                'fps': fps,
                'mean_equity': np.mean(valid_scores),
                'std_equity': np.std(valid_scores),
                'min_equity': np.min(valid_scores),
                'max_equity': np.max(valid_scores),
                'count': len(valid_scores),
                'mean_low_dev': np.mean([d for d in low_devs if d != float('inf')]),
                'mean_middle_dev': np.mean([d for d in mid_devs if d != float('inf')]),
                'mean_high_dev': np.mean([d for d in high_devs if d != float('inf')]),
                'mean_low_time': np.mean(avg_low_times) if avg_low_times else 0,
                'mean_middle_time': np.mean(avg_mid_times) if avg_mid_times else 0,
                'mean_high_time': np.mean(avg_high_times) if avg_high_times else 0,
                'mean_overall_time': np.mean(overall_avg_times) if overall_avg_times else 0
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
    create_travel_time_equity_comparison_plot(fps_df, output_dir)
    create_travel_time_deviation_plot(fps_df, output_dir)
    create_average_travel_time_plot(fps_df, output_dir)
    create_optimal_allocation_analysis(optimal_solutions, output_dir)
    
    print(f"Aggregation complete! Results saved to {output_dir}")
    return output_dir, fps_results

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
    
    # Create a stacked area chart version
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

def create_travel_time_equity_comparison_plot(fps_df, output_dir):
    """Create travel time equity comparison plot with improved optimal point detection"""
    plt.figure(figsize=(14, 10))
    
    # Plot travel time equity index with error bands
    plt.plot(fps_df['fps'], fps_df['mean_equity'], 'o-', 
             color='#d62728', linewidth=3, markersize=8, 
             label='Travel Time Equity Index')
    
    plt.fill_between(fps_df['fps'],
                    fps_df['mean_equity'] - fps_df['std_equity'],
                    fps_df['mean_equity'] + fps_df['std_equity'],
                    color='#d62728', alpha=0.2)
    
    # Plot deviations for each income level
    plt.plot(fps_df['fps'], fps_df['mean_low_dev'], 'o-', 
             color='#1f77b4', linewidth=2, markersize=8, 
             label='Low Income Deviation')
    
    plt.plot(fps_df['fps'], fps_df['mean_middle_dev'], 'o-', 
             color='#ff7f0e', linewidth=2, markersize=8, 
             label='Middle Income Deviation')
    
    plt.plot(fps_df['fps'], fps_df['mean_high_dev'], 'o-', 
             color='#2ca02c', linewidth=2, markersize=8, 
             label='High Income Deviation')
    
    # Find the raw data optimal point (minimum equity index)
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
    
    # Add point labels
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
    
    # Add trend line using polynomial fit
    try:
        z = np.polyfit(np.log(fps_df['fps']), fps_df['mean_equity'], 2)
        p = np.poly1d(z)
        x_smooth = np.geomspace(fps_df['fps'].min(), fps_df['fps'].max(), 100)
        y_smooth = p(np.log(x_smooth))
        plt.plot(x_smooth, y_smooth, '--', color='gray', alpha=0.6, label='Trend Line')
    except Exception as e:
        print(f"Error fitting trend line: {e}")
    
    # Set axis labels and title
    plt.xlabel('Fixed Pool Subsidy (FPS)', fontsize=12)
    plt.ylabel('Travel Time Equity Index (Lower is Better)', fontsize=12)
    plt.title('Travel Time Equity Index vs Fixed Pool Subsidy (FPS) Values', fontsize=14)
    
    # Set log scale for x-axis
    plt.xscale('log')
    
    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'travel_time_equity_comparison.png'), dpi=300)
    plt.close()
    
    # Create a text file with detailed analysis
    with open(os.path.join(output_dir, 'optimal_fps_analysis.txt'), 'w') as f:
        f.write("Travel Time Equity Optimization Analysis\n")
        f.write("=======================================\n\n")
        f.write(f"Raw Data Optimal FPS: {raw_optimal_fps:.0f} (Equity Index: {raw_optimal_equity:.4f})\n")
        f.write(f"Elbow Point FPS: {elbow_fps:.0f} (Equity Index: {elbow_equity:.4f})\n")
        if threshold_fps:
            f.write(f"Improvement Threshold ({threshold}%) FPS: {threshold_fps:.0f} (Equity Index: {threshold_equity:.4f})\n\n")
        
        # Add FPS values ranked by equity score
        f.write("FPS Values Ranked by Equity Index (best to worst):\n")
        f.write("----------------------------------------------\n")
        for _, row in fps_df.sort_values('mean_equity').iterrows():
            f.write(f"FPS: {row['fps']:.0f}, Equity Index: {row['mean_equity']:.4f}\n")
            
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

def create_travel_time_deviation_plot(fps_df, output_dir):
    """Create a detailed plot of travel time deviations across income groups"""
    # Create bar chart for comparing deviations
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set width of bars
    barWidth = 0.25
    
    # Set positions of the bars on X axis
    r1 = np.arange(len(fps_df))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    
    # Create bars
    ax.bar(r1, fps_df['mean_low_dev'], width=barWidth, label='Low Income', color='#1f77b4', alpha=0.7)
    ax.bar(r2, fps_df['mean_middle_dev'], width=barWidth, label='Middle Income', color='#ff7f0e', alpha=0.7)
    ax.bar(r3, fps_df['mean_high_dev'], width=barWidth, label='High Income', color='#2ca02c', alpha=0.7)
    
    # Add labels and title
    ax.set_xlabel('Fixed Pool Subsidy (FPS)', fontsize=12)
    ax.set_ylabel('Travel Time Deviation (Minutes)', fontsize=12)
    ax.set_title('Travel Time Deviations by Income Group Across FPS Values', fontsize=14)
    
    # Set x-ticks at the center of grouped bars
    ax.set_xticks([r + barWidth for r in range(len(fps_df))])
    ax.set_xticklabels([f"{fps:.0f}" for fps in fps_df['fps']], rotation=45)
    
    # Add grid and legend
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(fontsize=10)
    
    # Add value labels on bars
    for i, value in enumerate(fps_df['mean_low_dev']):
        ax.text(r1[i], value + 0.1, f"{value:.2f}", ha='center', va='bottom', fontsize=8)
    for i, value in enumerate(fps_df['mean_middle_dev']):
        ax.text(r2[i], value + 0.1, f"{value:.2f}", ha='center', va='bottom', fontsize=8)
    for i, value in enumerate(fps_df['mean_high_dev']):
        ax.text(r3[i], value + 0.1, f"{value:.2f}", ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'travel_time_deviations_by_fps.png'), dpi=300)
    plt.close()
    
    # Create a heatmap of deviations
    plt.figure(figsize=(12, 8))
    
    # Reshape data for heatmap
    heatmap_data = fps_df[['fps', 'mean_low_dev', 'mean_middle_dev', 'mean_high_dev']].copy()
    heatmap_data.set_index('fps', inplace=True)
    heatmap_data.columns = ['Low Income', 'Middle Income', 'High Income']
    
    # Create heatmap
    sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd_r', fmt=".2f", 
               cbar_kws={'label': 'Travel Time Deviation (Minutes)'})
    
    plt.title('Travel Time Deviations by Income Group and FPS', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'travel_time_deviation_heatmap.png'), dpi=300)
    plt.close()

def create_average_travel_time_plot(fps_df, output_dir):
    """Create plots analyzing average travel times across income groups"""
    plt.figure(figsize=(12, 8))
    
    # Plot average travel times for each income group
    plt.plot(fps_df['fps'], fps_df['mean_low_time'], 'o-', 
             color='#1f77b4', linewidth=2, markersize=8, 
             label='Low Income')
    
    plt.plot(fps_df['fps'], fps_df['mean_middle_time'], 'o-', 
             color='#ff7f0e', linewidth=2, markersize=8, 
             label='Middle Income')
    
    plt.plot(fps_df['fps'], fps_df['mean_high_time'], 'o-', 
             color='#2ca02c', linewidth=2, markersize=8, 
             label='High Income')
    
    # Plot overall average travel time
    plt.plot(fps_df['fps'], fps_df['mean_overall_time'], 'o-', 
             color='purple', linewidth=3, markersize=10, 
             label='Overall Average')
    
    # Add trend line
    try:
        z = np.polyfit(np.log(fps_df['fps']), fps_df['mean_overall_time'], 2)
        p = np.poly1d(z)
        x_smooth = np.geomspace(fps_df['fps'].min(), fps_df['fps'].max(), 100)
        y_smooth = p(np.log(x_smooth))
        plt.plot(x_smooth, y_smooth, '--', color='gray', alpha=0.6)
    except Exception as e:
        print(f"Error fitting trend line: {e}")
    
    plt.title('Average Travel Times by Income Group', fontsize=14)
    plt.xlabel('Fixed Pool Subsidy (FPS)', fontsize=12)
    plt.ylabel('Average Travel Time (Minutes)', fontsize=12)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'average_travel_times.png'), dpi=300)
    plt.close()
    
    # Create travel time vs equity scatter plot
    plt.figure(figsize=(12, 8))
    
    # Plot how travel time equity changes with average travel time
    scatter = plt.scatter(fps_df['mean_overall_time'], fps_df['mean_equity'], 
                         c=fps_df['fps'], cmap='viridis', 
                         s=100, alpha=0.7)
    
    plt.colorbar(scatter, label='FPS Value')
    
    # Add FPS annotations
    for i, row in fps_df.iterrows():
        plt.annotate(f"FPS={row['fps']:.0f}",
                    xy=(row['mean_overall_time'], row['mean_equity']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8)
    
    # Add trend line
    try:
        z = np.polyfit(fps_df['mean_overall_time'], fps_df['mean_equity'], 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(fps_df['mean_overall_time'].min(), fps_df['mean_overall_time'].max(), 100)
        y_smooth = p(x_smooth)
        plt.plot(x_smooth, y_smooth, '--', color='gray', alpha=0.6)
    except Exception as e:
        print(f"Error fitting trend line: {e}")
    
    plt.title('Relationship Between Average Travel Time and Travel Time Equity', fontsize=14)
    plt.xlabel('Average Travel Time (Minutes)', fontsize=12)
    plt.ylabel('Travel Time Equity Index', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'travel_time_vs_equity.png'), dpi=300)
    plt.close()

def create_optimal_allocation_analysis(optimal_solutions, output_dir):
    """Analyze and visualize optimal subsidy allocations"""
    if not optimal_solutions:
        print("No optimal solutions to analyze")
        return
    
    # Convert to DataFrame for easier analysis
    opt_df = pd.DataFrame(optimal_solutions)
    
    # Check if we have data to proceed
    if opt_df.empty:
        print("Empty optimal solutions DataFrame")
        return
        
    # Sort by travel time equity index to find the best solution
    if 'travel_time_equity_index' in opt_df.columns:
        opt_df = opt_df.sort_values('travel_time_equity_index')
        best_solution = opt_df.iloc[0]
        print(f"Best solution FPS: {best_solution['fps']}, Equity: {best_solution['travel_time_equity_index']:.4f}")
    else:
        print("Warning: 'travel_time_equity_index' not found in optimal solutions")
        # Try to use the first solution as the best one
        best_solution = opt_df.iloc[0]
    
    # Check if 'allocations' exists in the best solution
    if 'allocations' not in best_solution or not best_solution['allocations']:
        print("Warning: No allocation data found in the best solution")
        return
    
    try:
        # Extract allocation data for the best solution
        alloc_data = []
        for income_level in ['low', 'middle', 'high']:
            for mode in ['bike', 'car', 'MaaS_Bundle', 'public', 'walk']:
                key = f"{income_level}_{mode}"
                if key in best_solution['allocations']:
                    alloc_data.append({
                        'income_level': income_level,
                        'mode': mode,
                        'allocation_value': best_solution['allocations'][key]
                    })
        
        # Check if we have allocation data
        if not alloc_data:
            print("No allocation data extracted")
            return
            
        alloc_df = pd.DataFrame(alloc_data)
        
        # Create heatmap of optimal allocations
        plt.figure(figsize=(12, 8))
        
        # Pivot data for heatmap
        pivot_alloc = alloc_df.pivot_table(
            values='allocation_value',
            index='income_level',
            columns='mode'
        )
        
        # Sort index to ensure consistent order
        pivot_alloc = pivot_alloc.reindex(['low', 'middle', 'high'])
        
        # Create heatmap
        sns.heatmap(pivot_alloc, annot=True, cmap='YlGnBu', fmt=".2f", 
                   cbar_kws={'label': 'Allocation Percentage'})
        
        plt.title(f'Optimal Subsidy Allocations (FPS={best_solution["fps"]:.0f}, Equity Index={best_solution.get("travel_time_equity_index", 0):.4f})', 
                 fontsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'optimal_allocations_heatmap.png'), dpi=300)
        plt.close()
        
        # Create radar chart of optimal allocations by income level
        plt.figure(figsize=(10, 8))
        
        # Get all unique modes
        modes = sorted(alloc_df['mode'].unique())
        
        # Setup radar chart
        angles = np.linspace(0, 2*np.pi, len(modes), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        ax = plt.subplot(111, polar=True)
        
        # Plot each income level
        for income_level in ['low', 'middle', 'high']:
            values = []
            for mode in modes:
                value = alloc_df[(alloc_df['income_level'] == income_level) & 
                               (alloc_df['mode'] == mode)]['allocation_value'].values
                values.append(value[0] if len(value) > 0 else 0)
            
            # Close the loop
            values += values[:1]
            
            # Plot the income level allocations
            color = {'low': '#1f77b4', 'middle': '#ff7f0e', 'high': '#2ca02c'}[income_level]
            ax.plot(angles, values, 'o-', linewidth=2, label=f'{income_level.capitalize()} Income', color=color)
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
    
    except Exception as e:
        print(f"Error in allocation analysis: {e}")
        traceback.print_exc()
    
    try:
        # Plot comparative equity metrics for the solutions
        plt.figure(figsize=(12, 8))
        
        # Check if required columns exist
        if 'fps' in opt_df.columns and 'travel_time_equity_index' in opt_df.columns:
            # Create scatter plot with FPS vs Equity
            sizes = opt_df.get('count', pd.Series([1] * len(opt_df))).fillna(1) * 50
            scatter = plt.scatter(opt_df['fps'], opt_df['travel_time_equity_index'],
                                 s=sizes, c=opt_df['travel_time_equity_index'],
                                 cmap='coolwarm_r', alpha=0.7)
            
            plt.colorbar(scatter, label='Travel Time Equity Index (lower is better)')
            
            # Add a line connecting the points
            plt.plot(opt_df['fps'], opt_df['travel_time_equity_index'], 
                     '-', color='gray', alpha=0.5)
            
            # Highlight the best solution
            plt.scatter([best_solution['fps']], [best_solution['travel_time_equity_index']],
                       s=200, c='none', edgecolors='black', linewidths=2,
                       marker='o')
            
            plt.annotate('Best Solution',
                        xy=(best_solution['fps'], best_solution['travel_time_equity_index']),
                        xytext=(20, -20),
                        textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'),
                        fontsize=12)
            
            plt.title('Comparison of Solutions by FPS and Travel Time Equity', fontsize=14)
            plt.xlabel('Fixed Pool Subsidy (FPS)', fontsize=12)
            plt.ylabel('Travel Time Equity Index', fontsize=12)
            plt.xscale('log')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'solution_comparison.png'), dpi=300)
            plt.close()
            
            # Save the best solution details to a CSV
            try:
                best_data = {
                    'fps': best_solution['fps'],
                    'travel_time_equity_index': best_solution['travel_time_equity_index']
                }
                
                # Add deviation data if available
                if 'deviations' in best_solution and best_solution['deviations']:
                    for level in ['low', 'middle', 'high']:
                        if level in best_solution['deviations']:
                            best_data[f'{level}_deviation'] = best_solution['deviations'][level]
                
                # Add average travel time data if available            
                if 'avg_travel_times' in best_solution and best_solution['avg_travel_times']:
                    for level in ['low', 'middle', 'high']:
                        if level in best_solution['avg_travel_times']:
                            best_data[f'{level}_avg_time'] = best_solution['avg_travel_times'][level]
                
                # Add overall average travel time if available
                if 'overall_avg_travel_time' in best_solution:
                    best_data['overall_avg_time'] = best_solution['overall_avg_travel_time']
                
                # Add allocations if available
                if 'allocations' in best_solution and best_solution['allocations']:
                    for k, v in best_solution['allocations'].items():
                        best_data[k] = v
                
                best_allocation_df = pd.DataFrame([best_data])
                best_allocation_df.to_csv(os.path.join(output_dir, 'best_solution_details.csv'), index=False)
            except Exception as e:
                print(f"Error saving best solution details: {e}")
                traceback.print_exc()
        else:
            print("Warning: Missing required columns for solution comparison plot")
    except Exception as e:
        print(f"Error in solution comparison: {e}")
        traceback.print_exc()
        
if __name__ == "__main__":
    aggregate_travel_time_equity_results()