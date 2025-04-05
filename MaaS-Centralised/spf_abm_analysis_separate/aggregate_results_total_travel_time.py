import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def aggregate_travel_time_results():
    """Aggregate and analyze total system travel time results from multiple simulation runs"""
    print("Starting aggregation of total system travel time results...")
    
    # Get parent directory (MaaS-Centralised) where the result folders are located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    os.chdir(parent_dir)  # Change to parent directory where results are located
    print(f"Looking for results in: {parent_dir}")
    
    # Find all total system travel time results directories
    result_dirs = glob.glob("total_system_travel_time_results_*")
    print(f"Found {len(result_dirs)} simulation result directories")
    
    if not result_dirs:
        print("No simulation result directories found. Please check the path.")
        return None
    
    # Create output directory for aggregated results
    output_dir = os.path.join(parent_dir, f"aggregated_results_total_travel_time")
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
        
        # Parse optimal solution
        opt_path = os.path.join(dir_path, 'optimal_solution.txt')
        if os.path.exists(opt_path):
            try:
                opt_data = {}
                with open(opt_path, 'r') as f:
                    lines = f.readlines()
                    
                    # Parse FPS and travel time
                    opt_data['fps'] = float(lines[0].split(':')[1].strip())
                    opt_data['total_system_travel_time'] = float(lines[1].split(':')[1].split()[0].strip())
                    print(f"1 {opt_data['total_system_travel_time']}")
                    opt_data['avg_trip_time'] = float(lines[2].split(':')[1].split()[0].strip())
                    print(f"2 {opt_data['avg_trip_time']}")

                    opt_data['total_trips'] = int(lines[3].split(':')[1].split('.')[0].strip())
                    print(f"3 {opt_data['total_trips']}")
                    # Parse allocations
                    allocations = {}
                    in_allocation_section = False
                    for i, line in enumerate(lines):
                        if line.strip() == "Optimal Subsidy Allocations:":
                            in_allocation_section = True
                            continue
                        elif "Travel Time Breakdown by Income Level:" in line or "Travel Time Breakdown by Mode:" in line:
                            in_allocation_section = False
                            continue
                            
                        if in_allocation_section and line.strip().startswith("  "):
                            parts = line.strip().split(":")
                            key = parts[0].strip()
                            value = float(parts[1].strip())
                            print(f"4 {parts[1].strip()}")
                            allocations[key] = value
                    
                    opt_data['allocations'] = allocations
                    
                    # Try to parse income breakdown
                    income_breakdown = {}
                    in_income_section = False
                    current_income = None
                    
                    for line in lines:
                        if "Travel Time Breakdown by Income Level:" in line:
                            in_income_section = True
                            continue
                        elif "Travel Time Breakdown by Mode:" in line:
                            in_income_section = False
                            continue
                            
                        if in_income_section:
                            if line.strip().endswith("Income:"):
                                current_income = line.strip().split()[0].lower()
                                income_breakdown[current_income] = {}
                            elif current_income and "Total Travel Time:" in line:
                                parts = line.strip().split(":")
                                time_parts = parts[1].strip().split()
                                print(f"5 time_parts {time_parts}")
                                income_breakdown[current_income]['total_travel_time'] = float(time_parts[0])
                                print(f"5 {income_breakdown[current_income]['total_travel_time']}")
                                if len(time_parts) > 1:
                                    # Extract percentage if available
                                    pct = time_parts[2].strip("()%")
                                    print(f"5 {pct}")
                                    income_breakdown[current_income]['percentage'] = float(pct)
                                    print(f"5 {income_breakdown[current_income]['percentage']}")
                            elif current_income and "Average Trip Time:" in line:
                                print(f"6 {line}")
                                parts = line.strip().split(":")
                                print(f"6 {parts}")
                                time_parts = parts[1].strip().split()
                                print(f"6 time_parts {time_parts}")
                                income_breakdown[current_income]['avg_trip_time'] = float(time_parts[0])
                                print(f"6 {income_breakdown[current_income]['avg_trip_time']}")
                            elif current_income and "Number of Trips:" in line:
                                parts = line.strip().split(":")
                                print(f"7 {parts}")
                                income_breakdown[current_income]['trip_count'] = int(parts[1].split('.')[0].strip())
                    
                    opt_data['income_breakdown'] = income_breakdown
                    
                    # Try to parse mode breakdown
                    mode_breakdown = {}
                    in_mode_section = False
                    current_mode = None

                    for line in lines:
                        if "Travel Time Breakdown by Mode:" in line:
                            in_mode_section = True
                            continue
                        elif "Optimal Subsidy Allocations:" in line:
                            in_mode_section = False
                            continue
                            
                        if in_mode_section:
                            if line.strip().endswith(":") and not line.strip().startswith("Travel"):
                                current_mode = line.strip().strip(":")
                                mode_breakdown[current_mode] = {}
                            elif current_mode and "Total Travel Time:" in line:
                                print(f"8 line {line}")
                                parts = line.strip().split(":")
                                time_parts = parts[1].strip().split()
                                
                                # Safely parse travel time with error handling
                                try:
                                    if time_parts[0] == "nan":
                                        mode_breakdown[current_mode]['total_travel_time'] = 0
                                    else:
                                        mode_breakdown[current_mode]['total_travel_time'] = float(time_parts[0])
                                    print(f"8 {mode_breakdown[current_mode]['total_travel_time']}")
                                except (ValueError, IndexError):
                                    mode_breakdown[current_mode]['total_travel_time'] = 0
                                    print("8 0 (value error fixed)")
                                
                                # Safely parse percentage with proper index
                                try:
                                    if len(time_parts) > 2:  # Format: "X.XX minutes (Y.Y%)"
                                        pct_str = time_parts[2].strip("()%")
                                        if pct_str != "nan":
                                            mode_breakdown[current_mode]['percentage'] = float(pct_str)
                                        else:
                                            mode_breakdown[current_mode]['percentage'] = 0
                                    else:
                                        mode_breakdown[current_mode]['percentage'] = 0
                                    print(f"9 {mode_breakdown[current_mode].get('percentage', 0)}")
                                except (ValueError, IndexError):
                                    mode_breakdown[current_mode]['percentage'] = 0
                                    print("9 0 (percentage parsing error fixed)")
                                    
                            elif current_mode and "Number of Trips:" in line:
                                parts = line.strip().split(":")
                                try:
                                    if len(parts) > 1 and parts[1].strip() != "nan":
                                        mode_breakdown[current_mode]['trip_count'] = int(parts[1].split('.')[0].strip())
                                    else:
                                        mode_breakdown[current_mode]['trip_count'] = 0
                                except (ValueError, IndexError):
                                    mode_breakdown[current_mode]['trip_count'] = 0

                    opt_data['mode_breakdown'] = mode_breakdown
                    opt_data['directory'] = dir_path
                    optimal_solutions.append(opt_data)
                    print(f"  Parsed optimal solution: FPS={opt_data['fps']}, Travel Time={opt_data['total_system_travel_time']:.2f}")
                    
                print(f"  Parsed optimal solution: FPS={opt_data['fps']}, Travel Time={opt_data['total_system_travel_time']:.2f}")
            except Exception as e:
                print(f"  Error parsing optimal solution from {dir_path}: {e}")
    
    # Create combined analysis of FPS results
    fps_analysis = []
    for fps, results_list in fps_results.items():
        # Get travel time metrics for this FPS across all runs
        travel_times = [r.get('total_system_travel_time', float('inf')) for r in results_list]
        avg_trip_times = [r.get('avg_trip_time', 0) for r in results_list]
        trip_counts = [r.get('total_trips', 0) for r in results_list]
        valid_times = [t for t in travel_times if t != float('inf')]
        
        # Get income-specific travel times
        low_times = []
        mid_times = []
        high_times = []
        
        for r in results_list:
            if 'income_breakdown' in r:
                if 'low' in r['income_breakdown']:
                    low_times.append(r['income_breakdown']['low'].get('total_travel_time', 0))
                if 'middle' in r['income_breakdown']:
                    mid_times.append(r['income_breakdown']['middle'].get('total_travel_time', 0))
                if 'high' in r['income_breakdown']:
                    high_times.append(r['income_breakdown']['high'].get('total_travel_time', 0))
        
        if valid_times:
            fps_analysis.append({
                'fps': fps,
                'mean_travel_time': np.mean(valid_times),
                'std_travel_time': np.std(valid_times),
                'min_travel_time': np.min(valid_times),
                'max_travel_time': np.max(valid_times),
                'mean_avg_trip_time': np.mean([t for t in avg_trip_times if t > 0]),
                'mean_trip_count': np.mean([t for t in trip_counts if t > 0]),
                'count': len(valid_times),
                'mean_low_time': np.mean(low_times) if low_times else 0,
                'mean_middle_time': np.mean(mid_times) if mid_times else 0,
                'mean_high_time': np.mean(high_times) if high_times else 0
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
    fps_df.to_csv(os.path.join(output_dir, 'fps_travel_time_analysis.csv'), index=False)
    
    # Create visualizations
    create_travel_time_comparison_plot(fps_df, output_dir)
    create_income_breakdown_plot(fps_df, output_dir)
    create_income_linear_plot(fps_df, output_dir)
    create_trip_time_plot(fps_df, output_dir)
    
    if optimal_solutions:
        create_mode_breakdown_plot(optimal_solutions, output_dir)
    
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
    
def create_travel_time_comparison_plot(fps_df, output_dir):
    """Create total system travel time comparison plot with improved optimal point detection"""
    plt.figure(figsize=(14, 10))
    
    # Plot total system travel time with error bands
    plt.plot(fps_df['fps'], fps_df['mean_travel_time'], 'o-', 
             color='#d62728', linewidth=3, markersize=8, 
             label='Total System Travel Time')
    
    plt.fill_between(fps_df['fps'],
                    fps_df['mean_travel_time'] - fps_df['std_travel_time'],
                    fps_df['mean_travel_time'] + fps_df['std_travel_time'],
                    color='#d62728', alpha=0.2)
    
    # Find the raw data optimal point (minimum travel time)
    raw_optimal_idx = fps_df['mean_travel_time'].idxmin()
    raw_optimal_fps = fps_df.loc[raw_optimal_idx, 'fps']
    raw_optimal_travel_time = fps_df.loc[raw_optimal_idx, 'mean_travel_time']
    
    # Add vertical line for raw data optimal
    plt.axvline(x=raw_optimal_fps, color='blue', linestyle='--', alpha=0.7,
               label=f'Optimal FPS (Raw): {raw_optimal_fps:.0f}')
    
    # Add marker for optimal point
    plt.scatter([raw_optimal_fps], [raw_optimal_travel_time], color='blue', s=150, zorder=10)
    
    # -------- Elbow Point Detection Method --------
    # Calculate normalized metrics for elbow detection
    x = fps_df['fps'].values
    y = fps_df['mean_travel_time'].values
    
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
    elbow_travel_time = y[elbow_idx]
    
    # Add elbow point to plot
    plt.axvline(x=elbow_fps, color='green', linestyle='--', alpha=0.7,
               label=f'Elbow Point: {elbow_fps:.0f}')
    plt.scatter([elbow_fps], [elbow_travel_time], color='green', s=150, zorder=10)
    
    # -------- Improvement Rate Analysis --------
    # Calculate percent improvement between adjacent points
    improvements = []
    for i in range(1, len(fps_df)):
        prev_val = fps_df.iloc[i-1]['mean_travel_time']
        curr_val = fps_df.iloc[i]['mean_travel_time']
        pct_change = (prev_val - curr_val) / prev_val * 100
        improvements.append({
            'fps': fps_df.iloc[i]['fps'],
            'improvement': pct_change
        })
    
    # Find the point where improvement rate drops below threshold (e.g., 2%)
    threshold = 2  # 2% improvement threshold - can be lower for travel time
    threshold_fps = None
    for imp in improvements:
        if imp['improvement'] < threshold:
            threshold_fps = imp['fps']
            break
    
    if threshold_fps:
        threshold_idx = fps_df[fps_df['fps'] == threshold_fps].index[0]
        threshold_travel_time = fps_df.loc[threshold_idx, 'mean_travel_time']
        
        # Add threshold point to plot
        plt.axvline(x=threshold_fps, color='purple', linestyle=':', alpha=0.7,
                   label=f'Threshold ({threshold}% improvement): {threshold_fps:.0f}')
        plt.scatter([threshold_fps], [threshold_travel_time], color='purple', s=150, zorder=10)
    
    # Add point labels
    for i, (idx, row) in enumerate(fps_df.iterrows()):
        if i % 2 == 0:  # Only annotate every other point
            fps = row['fps']
            travel_time = row['mean_travel_time']
            
            plt.annotate(f'Point {i+1}', 
                        xy=(fps, travel_time),
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
    plt.ylabel('Total System Travel Time (minutes)', fontsize=12)
    plt.title('Total System Travel Time vs Fixed Pool Subsidy (FPS) Values', fontsize=14)
    
    # Set log scale for x-axis
    plt.xscale('log')
    
    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'total_system_travel_time_vs_fps.png'), dpi=300)
    plt.close()
    
    # Create a text file with detailed analysis
    with open(os.path.join(output_dir, 'optimal_travel_time_fps_analysis.txt'), 'w') as f:
        f.write("Total System Travel Time Optimization Analysis\n")
        f.write("============================================\n\n")
        f.write(f"Raw Data Optimal FPS: {raw_optimal_fps:.0f} (Travel Time: {raw_optimal_travel_time:.2f} minutes)\n")
        f.write(f"Elbow Point FPS: {elbow_fps:.0f} (Travel Time: {elbow_travel_time:.2f} minutes)\n")
        if threshold_fps:
            f.write(f"Improvement Threshold ({threshold}%) FPS: {threshold_fps:.0f} (Travel Time: {threshold_travel_time:.2f} minutes)\n\n")
        
        # Add FPS values ranked by travel time
        f.write("FPS Values Ranked by Total Travel Time (best to worst):\n")
        f.write("--------------------------------------------------\n")
        for _, row in fps_df.sort_values('mean_travel_time').iterrows():
            f.write(f"FPS: {row['fps']:.0f}, Travel Time: {row['mean_travel_time']:.2f} minutes\n")
            
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

def create_income_breakdown_plot(fps_df, output_dir):
    """Create visualization of travel time breakdown by income level"""
    plt.figure(figsize=(12, 8))
    
    # Plot stacked areas for each income level
    income_columns = ['mean_low_time', 'mean_middle_time', 'mean_high_time']
    income_labels = ['Low Income', 'Middle Income', 'High Income']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    plt.stackplot(fps_df['fps'], 
                 [fps_df[col] for col in income_columns],
                 labels=income_labels,
                 colors=colors,
                 alpha=0.7)
    
    # Plot total travel time with line
    plt.plot(fps_df['fps'], fps_df['mean_travel_time'], 'o-', 
            color='#d62728', linewidth=2, label='Total Travel Time')
    
    # Add data labels for total at each point
    for i, row in fps_df.iterrows():
        plt.annotate(f"{row['mean_travel_time']:.0f}", 
                    xy=(row['fps'], row['mean_travel_time']),
                    xytext=(0, 5),
                    textcoords='offset points',
                    ha='center',
                    fontsize=8)
    
    # Set axis labels and title
    plt.title('Travel Time Breakdown by Income Level', fontsize=14)
    plt.xlabel('Fixed Pool Subsidy (FPS)', fontsize=12)
    plt.ylabel('Travel Time (minutes)', fontsize=12)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'travel_time_by_income_stack.png'), dpi=300)
    plt.close()
    
    # Create percentage contribution plot
    plt.figure(figsize=(12, 8))
    
    # Calculate percentage contribution
    for i, row in fps_df.iterrows():
        total = row[income_columns].sum()
        if total > 0:
            for col in income_columns:
                fps_df.at[i, f"{col}_pct"] = row[col] / total * 100
    
    # Create stacked percentage plot
    pct_columns = [f"{col}_pct" for col in income_columns]
    plt.stackplot(fps_df['fps'], 
                 [fps_df[col] for col in pct_columns],
                 labels=income_labels,
                 colors=colors,
                 alpha=0.7)
    
    # Set axis labels and title
    plt.title('Income Group Contribution to Total Travel Time', fontsize=14)
    plt.xlabel('Fixed Pool Subsidy (FPS)', fontsize=12)
    plt.ylabel('Percentage of Total Travel Time (%)', fontsize=12)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'travel_time_percentage_by_income.png'), dpi=300)
    plt.close()

def create_income_linear_plot(fps_df, output_dir):
    """Create linear visualization of travel time by income level"""
    plt.figure(figsize=(12, 8))
    
    # Plot separate lines for each income level's total travel time
    plt.plot(fps_df['fps'], fps_df['mean_low_time'], 'o-', 
             color='#1f77b4', linewidth=2, markersize=8, 
             label='Low Income')
    
    plt.plot(fps_df['fps'], fps_df['mean_middle_time'], 'o-', 
             color='#ff7f0e', linewidth=2, markersize=8, 
             label='Middle Income')
    
    plt.plot(fps_df['fps'], fps_df['mean_high_time'], 'o-', 
             color='#2ca02c', linewidth=2, markersize=8, 
             label='High Income')
    
    # Plot total system travel time
    plt.plot(fps_df['fps'], fps_df['mean_travel_time'], 'o-', 
             color='#d62728', linewidth=3, markersize=10, 
             label='Total Travel Time')
    
    # Add data labels for each income level at the rightmost point
    last_idx = fps_df['fps'].idxmax()
    for level, color in zip(['low', 'middle', 'high'], ['#1f77b4', '#ff7f0e', '#2ca02c']):
        col = f'mean_{level}_time'
        plt.annotate(f"{level.capitalize()}: {fps_df.loc[last_idx, col]:.0f} min",
                     xy=(fps_df.loc[last_idx, 'fps'], fps_df.loc[last_idx, col]),
                     xytext=(10, 0),
                     textcoords='offset points',
                     fontsize=10,
                     color=color,
                     fontweight='bold')
    
    # Set axis labels and title
    plt.title('Travel Time by Income Level (Linear View)', fontsize=14)
    plt.xlabel('Fixed Pool Subsidy (FPS)', fontsize=12)
    plt.ylabel('Travel Time (minutes)', fontsize=12)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Add annotation explaining the visualization
    plt.figtext(0.02, 0.02, 
                "This linear view shows absolute travel time for each income group,\n"
                "allowing for direct comparison of travel time changes across FPS values.",
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
                fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'travel_time_by_income_linear.png'), dpi=300)
    plt.close()

def create_trip_time_plot(fps_df, output_dir):
    """Create visualization of average trip time and trip counts"""
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot average trip time
    color1 = 'tab:blue'
    ax1.set_xlabel('Fixed Pool Subsidy (FPS)', fontsize=12)
    ax1.set_ylabel('Average Trip Time (minutes)', fontsize=12, color=color1)
    line1 = ax1.plot(fps_df['fps'], fps_df['mean_avg_trip_time'], 'o-', 
                    color=color1, linewidth=2, label='Average Trip Time')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xscale('log')
    
    # Create second y-axis for trip count
    ax2 = ax1.twinx()
    color2 = 'tab:green'
    ax2.set_ylabel('Total Number of Trips', fontsize=12, color=color2)
    line2 = ax2.plot(fps_df['fps'], fps_df['mean_trip_count'], 's-', 
                    color=color2, linewidth=2, label='Total Trips')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Add legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=10, loc='upper center')
    
    # Set title and grid
    plt.title('Average Trip Time and Total Trips vs FPS', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'avg_trip_time_and_counts.png'), dpi=300)
    plt.close()

def create_mode_breakdown_plot(optimal_solutions, output_dir):
    """Create visualization of travel time breakdown by mode for optimal solutions"""
    # First, find the overall best solution (minimum total system travel time)
    best_solution = min(optimal_solutions, key=lambda x: x['total_system_travel_time'])
    
    if 'mode_breakdown' in best_solution and best_solution['mode_breakdown']:
        # Extract mode data
        modes = []
        travel_times = []
        percentages = []
        trips = []
        
        for mode, data in best_solution['mode_breakdown'].items():
            modes.append(mode)
            travel_times.append(data.get('total_travel_time', 0))
            percentages.append(data.get('percentage', 0))
            trips.append(data.get('trip_count', 0))
        
        # Sort by travel time (descending)
        sorted_indices = np.argsort(travel_times)[::-1]
        modes = [modes[i] for i in sorted_indices]
        travel_times = [travel_times[i] for i in sorted_indices]
        percentages = [percentages[i] for i in sorted_indices]
        trips = [trips[i] for i in sorted_indices]
        
        # Create pie chart of mode contribution
        plt.figure(figsize=(12, 8))
        
        # Create color map
        colors = plt.cm.tab10(np.linspace(0, 1, len(modes)))
        
        # Plot pie chart
        plt.pie(travel_times, labels=modes, autopct='%1.1f%%', 
               startangle=90, colors=colors, shadow=False)
        plt.axis('equal')
        plt.title(f'Travel Time Breakdown by Mode (FPS={best_solution["fps"]})', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'optimal_mode_breakdown_pie.png'), dpi=300)
        plt.close()
        
        # Create bar chart comparing travel time vs trips
        fig, ax1 = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(modes))
        width = 0.35
        
        # Plot travel time bars
        color1 = 'tab:blue'
        ax1.set_ylabel('Travel Time (minutes)', fontsize=12, color=color1)
        bars1 = ax1.bar(x - width/2, travel_times, width, label='Travel Time', color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)
        
        # Add second y-axis for trip count
        ax2 = ax1.twinx()
        color2 = 'tab:orange'
        ax2.set_ylabel('Number of Trips', fontsize=12, color=color2)
        bars2 = ax2.bar(x + width/2, trips, width, label='Number of Trips', color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # Set x-axis labels
        ax1.set_xticks(x)
        ax1.set_xticklabels(modes, rotation=45, ha='right')
        
        # Add legend
        fig.legend([bars1, bars2], ['Travel Time', 'Number of Trips'], 
                  loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2)
        
        # Set title
        plt.title(f'Travel Time and Trip Count by Mode (FPS={best_solution["fps"]})', 
                 fontsize=14, pad=30)
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(output_dir, 'optimal_mode_travel_time_trips.png'), dpi=300)
        plt.close()
        
        # Create efficiency chart (avg time per trip by mode)
        avg_times = []
        for tt, tp in zip(travel_times, trips):
            avg_times.append(tt / tp if tp > 0 else 0)
        
        plt.figure(figsize=(12, 6))
        plt.bar(modes, avg_times, color='tab:green')
        plt.ylabel('Average Time per Trip (minutes)', fontsize=12)
        plt.title(f'Average Trip Time by Mode (FPS={best_solution["fps"]})', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on top of bars
        for i, v in enumerate(avg_times):
            plt.text(i, v + 0.1, f"{v:.1f}", ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'optimal_mode_efficiency.png'), dpi=300)
        plt.close()

def save_optimal_allocations(fps_results, output_dir):
    """Save optimal allocations to a text file for cross-policy analysis."""
    if not fps_results:
        print("No results to save")
        return
        
    # Find the FPS with lowest average travel time
    # First, calculate average travel time for each FPS
    avg_travel_times = {}
    for fps, results_list in fps_results.items():
        # Extract travel times from all results for this FPS
        travel_times = []
        for result in results_list:
            if isinstance(result, dict) and 'total_system_travel_time' in result:
                travel_times.append(result['total_system_travel_time'])
        
        if travel_times:
            avg_travel_times[fps] = sum(travel_times) / len(travel_times)
    
    # Find optimal FPS (lowest average travel time)
    if not avg_travel_times:
        print("No valid travel time data found")
        return
        
    optimal_fps = min(avg_travel_times.items(), key=lambda x: x[1])[0]
    optimal_results_list = fps_results[optimal_fps]
    
    # Get the result with the lowest travel time from the list
    best_result = None
    min_travel_time = float('inf')
    for result in optimal_results_list:
        if isinstance(result, dict) and 'total_system_travel_time' in result:
            if result['total_system_travel_time'] < min_travel_time:
                min_travel_time = result['total_system_travel_time']
                best_result = result
    
    if not best_result:
        print("No valid best result found")
        return
    
    with open(os.path.join(output_dir, 'optimal_allocations.txt'), 'w') as f:
        f.write(f"Optimal FPS Value: {optimal_fps}\n")
        f.write(f"Total System Travel Time: {min_travel_time:.2f} minutes\n\n")
        
        # Save allocations in the expected format
        f.write("Optimal Subsidy Allocations:\n")
        if 'optimal_allocations' in best_result and best_result['optimal_allocations']:
            for key, value in sorted(best_result['optimal_allocations'].items()):
                f.write(f"  {key}: {value}\n")
        
        # Write allocations directly if optimal_allocations doesn't exist but fixed_allocations does
        elif 'fixed_allocations' in best_result and best_result['fixed_allocations']:
            for key, value in sorted(best_result['fixed_allocations'].items()):
                f.write(f"  {key}: {value}\n")
        
        # Add income breakdown if available
        if any(income in best_result for income in ['low', 'middle', 'high']):
            f.write("\nTravel Time Breakdown by Income Level:\n")
            for level in ['low', 'middle', 'high']:
                if level in best_result:
                    level_data = best_result[level]
                    if isinstance(level_data, dict) and 'total_travel_time' in level_data:
                        f.write(f"  {level.capitalize()} Income: {level_data['total_travel_time']} minutes\n")
                    
        # Add overall average travel time if available
        if 'avg_trip_time' in best_result:
            f.write(f"\nOverall Average Travel Time: {best_result['avg_trip_time']} minutes\n")
                      
if __name__ == "__main__":
    output_dir, results = aggregate_travel_time_results()
    # Check if the output directory was created successfully
    if output_dir:
        # Get the results dictionary to pass to the save function
        # This needs to use the global fps_results dictionary from the aggregation function
        try:
            # Assuming fps_results is defined in the aggregation function
            # We may need to make it accessible or pass it as a return value
            save_optimal_allocations(results, output_dir)
            print("Optimal allocations saved to text file for cross-policy analysis")
        except NameError as e:
            print(f"Error: {e}. Make sure 'results' is defined or returned by aggregate_travel_time_results()")