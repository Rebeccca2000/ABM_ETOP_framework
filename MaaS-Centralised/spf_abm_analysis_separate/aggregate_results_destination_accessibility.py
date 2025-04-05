import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def aggregate_accessibility_results():
    """Aggregate and analyze destination accessibility results from multiple simulation runs"""
    print("Starting aggregation of destination accessibility results...")
    
    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)  # MaaS-Centralised
    os.chdir(parent_dir)
    print(f"Looking for results in: {parent_dir}")
    
    # Find all accessibility results directories
    result_dirs = glob.glob("accessibility_results_*")
    print(f"Found {len(result_dirs)} simulation result directories")
    
    if not result_dirs:
        print("No simulation result directories found. Please check the path.")
        return None
    
    # Create output directory for aggregated results
    output_dir = os.path.join(parent_dir, "spf_abm_analysis_separate", 
                             f"aggregated_accessibility_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect data by FPS value
    fps_results = {}
    optimal_solutions = []
    subsidy_usage_by_fps = {}
    
    # Process each simulation directory
    for dir_path in result_dirs:
        print(f"Processing {dir_path}...")
        
        # Load all_results.pkl
        all_results_path = os.path.join(dir_path, 'all_accessibility_results.pkl')
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
                    
            except Exception as e:
                print(f"  Error loading results from {dir_path}: {e}")
        
        # Parse optimal solution
        opt_path = os.path.join(dir_path, 'optimal_accessibility_solution.txt')
        if os.path.exists(opt_path):
            try:
                opt_data = {}
                with open(opt_path, 'r') as f:
                    lines = f.readlines()
                    
                    # Parse FPS and potential increase
                    opt_data['fps'] = float(lines[0].split(':')[1].strip())
                    opt_data['potential_increase'] = float(lines[1].split(':')[1].strip())
                    opt_data['effectiveness'] = float(lines[2].split(':')[1].strip())
                    
                    # Parse allocations
                    allocations = {}
                    in_allocation_section = False
                    for line in lines:
                        if line.strip() == "Optimal Subsidy Allocations:":
                            in_allocation_section = True
                            continue
                        elif line.strip() == "Potential Increase by Income Level:":
                            in_allocation_section = False
                            continue
                            
                        if in_allocation_section and line.strip().startswith("  "):
                            parts = line.strip().split(":")
                            key = parts[0].strip()
                            value = float(parts[1].strip())
                            allocations[key] = value
                    
                    opt_data['allocations'] = allocations
                    
                    # Parse income-level scores
                    potential_scores = {}
                    effectiveness_scores = {}
                    in_potential_section = False
                    in_effectiveness_section = False
                    
                    for line in lines:
                        if line.strip() == "Potential Increase by Income Level:":
                            in_potential_section = True
                            in_effectiveness_section = False
                            continue
                        elif line.strip() == "Effectiveness by Income Level:":
                            in_potential_section = False
                            in_effectiveness_section = True
                            continue
                            
                        if (in_potential_section or in_effectiveness_section) and line.strip().startswith("  "):
                            parts = line.strip().split(":")
                            level = parts[0].strip().lower()
                            value = float(parts[1].strip())
                            
                            if in_potential_section:
                                potential_scores[level] = value
                            else:
                                effectiveness_scores[level] = value
                    
                    opt_data['potential_scores'] = potential_scores
                    opt_data['effectiveness_scores'] = effectiveness_scores
                    opt_data['directory'] = dir_path
                    optimal_solutions.append(opt_data)
                    
                print(f"  Parsed optimal solution: FPS={opt_data['fps']}, Potential={opt_data['potential_increase']:.4f}")
            except Exception as e:
                print(f"  Error parsing optimal solution from {dir_path}: {e}")
    
    # Create combined analysis of FPS results
    fps_analysis = []
    for fps, results_list in fps_results.items():
        # Get potential increase scores for this FPS across all runs
        potential_scores = [r.get('total_potential_increase', float('-inf')) for r in results_list]
        valid_scores = [s for s in potential_scores if s != float('-inf')]
        
        # Get income-specific scores
        low_scores = [r.get('potential_scores', {}).get('low', float('-inf')) for r in results_list]
        mid_scores = [r.get('potential_scores', {}).get('middle', float('-inf')) for r in results_list]
        high_scores = [r.get('potential_scores', {}).get('high', float('-inf')) for r in results_list]
        
        # Get effectiveness scores
        effectiveness_scores = [r.get('overall_effectiveness', 0) for r in results_list]
        
        if valid_scores:
            fps_analysis.append({
                'fps': fps,
                'mean_potential': np.mean(valid_scores),
                'std_potential': np.std(valid_scores),
                'min_potential': np.min(valid_scores),
                'max_potential': np.max(valid_scores),
                'mean_effectiveness': np.mean(effectiveness_scores),
                'std_effectiveness': np.std(effectiveness_scores),
                'count': len(valid_scores),
                'mean_low': np.mean([s for s in low_scores if s != float('-inf')]),
                'mean_middle': np.mean([s for s in mid_scores if s != float('-inf')]),
                'mean_high': np.mean([s for s in high_scores if s != float('-inf')])
            })
    
    # Handle empty fps_analysis
    if not fps_analysis:
        print("No valid results found to analyze.")
        return None
    
    # Convert to DataFrame and sort by FPS
    fps_df = pd.DataFrame(fps_analysis).sort_values('fps')
    fps_df.to_csv(os.path.join(output_dir, 'fps_accessibility_analysis.csv'), index=False)
    
    # Create visualizations
    create_accessibility_comparison_plot(fps_df, output_dir)
    create_effectiveness_analysis_plot(fps_df, output_dir)
    create_income_distribution_plot(fps_df, output_dir)
    create_subsidy_potential_plot(fps_results, output_dir)
    create_effectiveness_ratio_plot(fps_results, output_dir)
    
    print(f"Aggregation complete! Results saved to {output_dir}")
    return output_dir

def create_accessibility_comparison_plot(fps_df, output_dir):
    """Create plot comparing potential increase across FPS values"""
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
    
    # Plot total potential with more emphasis
    plt.plot(fps_df['fps'], fps_df['mean_potential'], 'o-', 
             color='#d62728', linewidth=3, markersize=8, 
             label='Total Potential')
    
    # Add error bands for total potential
    plt.fill_between(fps_df['fps'],
                    fps_df['mean_potential'] - fps_df['std_potential'],
                    fps_df['mean_potential'] + fps_df['std_potential'],
                    color='#d62728', alpha=0.2)
    
    # Add trend line
    try:
        z = np.polyfit(np.log(fps_df['fps']), fps_df['mean_potential'], 2)
        p = np.poly1d(z)
        x_smooth = np.geomspace(fps_df['fps'].min(), fps_df['fps'].max(), 100)
        y_smooth = p(np.log(x_smooth))
        plt.plot(x_smooth, y_smooth, '--', color='gray', alpha=0.6, label='Trend')
        
        # Find optimal point based on trend
        opt_idx = np.argmax(y_smooth)
        opt_fps = x_smooth[opt_idx]
        plt.axvline(x=opt_fps, color='red', linestyle='--', alpha=0.5)
    except Exception as e:
        print(f"Error fitting trend line: {e}")
    
    plt.title('Destination Potential Increase vs Fixed Pool Subsidy', fontsize=14)
    plt.xlabel('Fixed Pool Subsidy (FPS)', fontsize=12)
    plt.ylabel('Potential Increase', fontsize=12)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'potential_increase_comparison.png'), dpi=300)
    plt.close()

def create_effectiveness_analysis_plot(fps_df, output_dir):
    """Create visualization of effectiveness metrics"""
    plt.figure(figsize=(12, 8))
    
    # Plot effectiveness with error bands
    plt.plot(fps_df['fps'], fps_df['mean_effectiveness'], 'o-',
             color='#1f77b4', linewidth=2, label='Mean Effectiveness')
    
    plt.fill_between(fps_df['fps'],
                    fps_df['mean_effectiveness'] - fps_df['std_effectiveness'],
                    fps_df['mean_effectiveness'] + fps_df['std_effectiveness'],
                    color='#1f77b4', alpha=0.2)
    
    # Add annotations for key points
    max_idx = fps_df['mean_effectiveness'].idxmax()
    plt.annotate(f'Peak: {fps_df.iloc[max_idx]["mean_effectiveness"]:.4f}',
                xy=(fps_df.iloc[max_idx]['fps'], fps_df.iloc[max_idx]['mean_effectiveness']),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))
    
    plt.title('Subsidy Effectiveness vs Fixed Pool Subsidy', fontsize=14)
    plt.xlabel('Fixed Pool Subsidy (FPS)', fontsize=12)
    plt.ylabel('Effectiveness (Potential Increase per Subsidy Unit)', fontsize=12)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'effectiveness_analysis.png'), dpi=300)
    plt.close()

def create_subsidy_potential_plot(fps_results, output_dir):
    """Create scatter plot showing relationship between subsidy amount and potential increase"""
    plt.figure(figsize=(12, 8))
    
    # Process data for each income level
    for income_level in ['low', 'middle', 'high']:
        x_data = []  # subsidy amounts
        y_data = []  # potential increases
        fps_labels = []  # for annotations
        
        for fps, results_list in fps_results.items():
            for result in results_list:
                if 'full_results' in result and income_level in result['full_results']:
                    income_data = result['full_results'][income_level]
                    if 'total_subsidy' in income_data and 'potential_increase' in income_data:
                        x_data.append(income_data['total_subsidy'])
                        y_data.append(income_data['potential_increase'])
                        fps_labels.append(str(fps))
        
        if x_data and y_data:
            # Plot scatter points
            color = {'low': '#1f77b4', 'middle': '#ff7f0e', 'high': '#2ca02c'}[income_level]
            plt.scatter(x_data, y_data, label=f'{income_level.capitalize()} Income', 
                       color=color, alpha=0.7)
            
            # Add FPS labels
            for i, (x, y, fps) in enumerate(zip(x_data, y_data, fps_labels)):
                plt.annotate(fps, (x, y), xytext=(3, 3), textcoords='offset points', 
                           fontsize=8)
            
            # Add trend line
            try:
                z = np.polyfit(x_data, y_data, 1)
                p = np.poly1d(z)
                x_line = np.linspace(min(x_data), max(x_data), 100)
                plt.plot(x_line, p(x_line), '--', color=color, alpha=0.5)
            except Exception as e:
                print(f"Error fitting trend line for {income_level}: {e}")
    
    plt.title('Relationship Between Subsidy Amount and Potential Increase')
    plt.xlabel('Total Subsidy Amount')
    plt.ylabel('Potential Increase')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'subsidy_potential_relationship.png'), dpi=300)
    plt.close()

def create_effectiveness_ratio_plot(fps_results, output_dir):
    """Create bar plot showing average effectiveness ratio by income level"""
    # Collect effectiveness ratios for each income level
    effectiveness_data = {
        'low': [],
        'middle': [],
        'high': [],
        'overall': []
    }
    
    for fps, results_list in fps_results.items():
        for result in results_list:
            if 'full_results' in result:
                # Collect income-specific ratios
                for level in ['low', 'middle', 'high']:
                    if level in result['full_results']:
                        ratio = result['full_results'][level].get('effectiveness_ratio', 0)
                        effectiveness_data[level].append(ratio)
                
                # Collect overall effectiveness
                if 'overall_effectiveness' in result:
                    effectiveness_data['overall'].append(result['overall_effectiveness'])
    
    # Calculate means and standard deviations
    means = {level: np.mean(ratios) for level, ratios in effectiveness_data.items() if ratios}
    stds = {level: np.std(ratios) for level, ratios in effectiveness_data.items() if ratios}
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    
    # Set positions and width
    positions = range(len(means))
    width = 0.7
    
    # Create bars with error bars
    colors = {'low': '#1f77b4', 'middle': '#ff7f0e', 'high': '#2ca02c', 'overall': '#d62728'}
    bars = plt.bar(positions, means.values(), width, 
                  yerr=list(stds.values()),
                  color=[colors[level] for level in means.keys()],
                  capsize=5)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + stds[list(means.keys())[bars.index(bar)]],
                f'{height:.6f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.title('Average Effectiveness Ratio by Income Level')
    plt.ylabel('Effectiveness Ratio (Potential Increase per Subsidy Unit)')
    plt.xticks(positions, [level.capitalize() for level in means.keys()])
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add explanation note
    plt.figtext(0.02, 0.02, 
                "Note: Effectiveness Ratio represents the destination potential increase per subsidy unit",
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'effectiveness_ratio_by_income.png'), dpi=300)
    plt.close()

def create_income_distribution_plot(fps_df, output_dir):
    """Create visualization of potential increase distribution across income levels"""
    income_cols = ['mean_low', 'mean_middle', 'mean_high']
    income_labels = ['Low Income', 'Middle Income', 'High Income']
    
    # Calculate percentage distribution
    pct_df = fps_df.copy()
    for fps_idx in pct_df.index:
        total = pct_df.loc[fps_idx, income_cols].sum()
        if total > 0:
            for col in income_cols:
                pct_df.loc[fps_idx, f'{col}_pct'] = pct_df.loc[fps_idx, col] / total * 100
    
    plt.figure(figsize=(12, 8))
    
    # Create stacked area plot
    plt.stackplot(pct_df['fps'],
                 [pct_df[f'{col}_pct'] for col in income_cols],
                 labels=income_labels,
                 alpha=0.7)
    
    plt.title('Distribution of Potential Increase Across Income Levels', fontsize=14)
    plt.xlabel('Fixed Pool Subsidy (FPS)', fontsize=12)
    plt.ylabel('Percentage of Total Potential Increase', fontsize=12)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'income_distribution.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    aggregate_accessibility_results()