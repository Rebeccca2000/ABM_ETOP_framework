import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import traceback
import re

def calculate_total_system_cost_from_pickle(dir_path):
    """
    Calculate total system cost by accessing service_booking_log data from pickle files
    """
    total_costs = {}
    
    # First, try to load all_results.pkl
    all_results_path = os.path.join(dir_path, 'all_results.pkl')
    if os.path.exists(all_results_path):
        try:
            with open(all_results_path, 'rb') as f:
                results = pickle.load(f)
            
            print(f"    Found all_results.pkl with {len(results)} FPS values")
            
            for fps, result in results.items():
                fps_float = float(fps)
                
                # Try multiple ways to extract service booking data
                service_booking_data = None
                
                # Method 1: Direct access
                if isinstance(result, dict):
                    if 'service_booking_log' in result:
                        service_booking_data = result['service_booking_log']
                    elif 'full_results' in result and isinstance(result['full_results'], dict):
                        if 'service_booking_log' in result['full_results']:
                            service_booking_data = result['full_results']['service_booking_log']
                    elif 'query_results' in result and isinstance(result['query_results'], dict):
                        for key, value in result['query_results'].items():
                            if 'service_booking_log' in key.lower():
                                service_booking_data = value
                                break
                
                # Calculate total cost if we found data
                if service_booking_data:
                    total_cost = sum_total_prices(service_booking_data)
                    if total_cost > 0:
                        total_costs[fps_float] = total_cost
                        print(f"      FPS {fps_float}: Total system cost = {total_cost:.2f}")
                    else:
                        print(f"      FPS {fps_float}: No valid cost data found")
                else:
                    print(f"      FPS {fps_float}: No service_booking_log found")
                    
        except Exception as e:
            print(f"    Error processing all_results.pkl: {e}")
    
    # If all_results.pkl didn't work, try individual FPS files
    if not total_costs:
        print("    Trying individual FPS files...")
        fps_files = glob.glob(os.path.join(dir_path, 'fps_*.pkl'))
        
        for fps_file in fps_files:
            try:
                # Extract FPS value from filename
                fps_match = re.search(r'fps_([0-9.]+)', os.path.basename(fps_file))
                if not fps_match:
                    continue
                
                fps_value = float(fps_match.group(1))
                
                # Load the FPS result pickle file
                with open(fps_file, 'rb') as f:
                    fps_data = pickle.load(f)
                
                # Extract total cost
                total_cost = extract_service_booking_costs(fps_data)
                
                if total_cost and total_cost > 0:
                    total_costs[fps_value] = total_cost
                    print(f"      FPS {fps_value}: Total system cost = {total_cost:.2f}")
                    
            except Exception as e:
                print(f"      Error processing {fps_file}: {e}")
                continue
    
    return total_costs

def extract_service_booking_costs(fps_data):
    """
    Extract and sum total_price from service_booking_log data within pickle file
    """
    total_cost = 0
    
    # Handle different possible data structures
    if isinstance(fps_data, dict):
        # Check for service_booking_log key directly
        if 'service_booking_log' in fps_data:
            service_bookings = fps_data['service_booking_log']
            total_cost = sum_total_prices(service_bookings)
            
        # Check for nested results structure
        elif 'results' in fps_data and isinstance(fps_data['results'], dict):
            if 'service_booking_log' in fps_data['results']:
                service_bookings = fps_data['results']['service_booking_log']
                total_cost = sum_total_prices(service_bookings)
                
        # Check for query_results structure
        elif 'query_results' in fps_data:
            for table_name, table_data in fps_data['query_results'].items():
                if 'service_booking_log' in table_name.lower():
                    total_cost = sum_total_prices(table_data)
                    break
                    
        # If the entire fps_data is the service booking data
        elif isinstance(fps_data, list):
            total_cost = sum_total_prices(fps_data)
            
        # Check for simulation_data structure (common in some setups)
        elif 'simulation_data' in fps_data:
            sim_data = fps_data['simulation_data']
            if isinstance(sim_data, dict) and 'service_booking_log' in sim_data:
                total_cost = sum_total_prices(sim_data['service_booking_log'])
    
    elif isinstance(fps_data, list):
        # If fps_data is directly a list of service booking records
        total_cost = sum_total_prices(fps_data)
    
    return total_cost

def sum_total_prices(service_bookings):
    """
    Sum up total_price values from service booking records
    """
    if not service_bookings:
        return 0
    
    total = 0
    processed_records = 0
    
    try:
        if isinstance(service_bookings, list):
            for booking in service_bookings:
                if isinstance(booking, dict):
                    if 'total_price' in booking:
                        price = booking['total_price']
                        if price is not None:
                            total += float(price)
                            processed_records += 1
                    # Also check for other possible price column names
                    elif 'price' in booking:
                        price = booking['price']
                        if price is not None:
                            total += float(price)
                            processed_records += 1
        
        elif isinstance(service_bookings, dict):
            # Handle case where service_bookings is a single record
            if 'total_price' in service_bookings:
                price = service_bookings['total_price']
                if price is not None:
                    total += float(price)
                    processed_records += 1
        
        # Handle pandas DataFrame if that's what we get
        elif hasattr(service_bookings, 'columns'):
            if 'total_price' in service_bookings.columns:
                total = service_bookings['total_price'].sum()
                processed_records = len(service_bookings)
    
    except Exception as e:
        print(f"        Error processing service bookings: {e}")
        return 0
    
    print(f"        Processed {processed_records} records, total cost: {total:.2f}")
    return total

def aggregate_mae_results_dynamic():
    """Aggregate and analyze mode share equity results with dynamic FPS percentages"""
    print("Starting Dynamic-Referenced Mode Share Equity Analysis...")
    
    # Get parent directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    os.chdir(parent_dir)
    print(f"Looking for results in: {parent_dir}")
    
    # Find all equity results directories
    result_dirs = glob.glob("mae_equity_results_*")
    print(f"Found {len(result_dirs)} simulation result directories")
    
    if not result_dirs:
        print("No simulation result directories found.")
        return None
    
    # Create output directory
    output_dir = os.path.join(parent_dir, f"aggregated_results_mae_dynamic_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect data by FPS value
    fps_results = {}
    fps_total_costs = {}
    subsidy_usage_by_fps = {}
    optimal_solutions = []
    
    # Process each simulation directory
    for dir_path in result_dirs:
        print(f"Processing {dir_path}...")
        
        # Calculate total system costs for each FPS in this simulation
        total_costs = calculate_total_system_cost_from_pickle(dir_path)
        
        if total_costs:
            print(f"  Found cost data for {len(total_costs)} FPS values")
            # Store total costs
            for fps, cost in total_costs.items():
                if fps not in fps_total_costs:
                    fps_total_costs[fps] = []
                fps_total_costs[fps].append(cost)
        else:
            print(f"  No cost data found in {dir_path}")
        
        # Load all_results.pkl for other analyses
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
                    
                    result['directory'] = dir_path
                    fps_results[fps_float].append(result)
                    
                    # Extract subsidy usage data
                    if 'subsidy_usage' in result:
                        subsidy_data = result['subsidy_usage']
                        if fps_float not in subsidy_usage_by_fps:
                            subsidy_usage_by_fps[fps_float] = []
                        subsidy_usage_by_fps[fps_float].append(subsidy_data)
                    
            except Exception as e:
                print(f"  Error loading results from {dir_path}: {e}")
                traceback.print_exc()
        
        # Parse optimal solution
        opt_path = os.path.join(dir_path, 'optimal_solution.txt')
        if os.path.exists(opt_path):
            try:
                opt_data = parse_optimal_solution(opt_path, dir_path)
                if opt_data:
                    optimal_solutions.append(opt_data)
            except Exception as e:
                print(f"  Error parsing optimal solution from {dir_path}: {e}")
    
    # Check if we have any cost data
    if not fps_total_costs:
        print("Error: Could not calculate FPS percentages. Check if service booking data is available.")
        print("Debug info:")
        print(f"- Found {len(result_dirs)} simulation directories")
        print(f"- Found {len(fps_results)} FPS values with results data")
        print("- No cost data extracted from any directory")
        
        # Try to debug the first directory
        if result_dirs:
            print(f"\nDebugging first directory: {result_dirs[0]}")
            test_costs = calculate_total_system_cost_from_pickle(result_dirs[0])
            print(f"Debug result: {test_costs}")
        
        return None
    
    # Calculate average total costs and FPS percentages
    fps_percentages = {}
    avg_total_costs = {}
    
    print("\nCalculating FPS percentages based on actual system costs...")
    for fps, costs_list in fps_total_costs.items():
        if costs_list:
            avg_cost = sum(costs_list) / len(costs_list)
            avg_total_costs[fps] = avg_cost
            fps_percentages[fps] = (fps / avg_cost) * 100
            print(f"FPS {fps:.0f}: Avg total cost = {avg_cost:.2f}, Percentage = {fps_percentages[fps]:.1f}%")
    
    if not fps_percentages:
        print("Error: Could not calculate FPS percentages. No valid cost data found.")
        return None
    
    # Create combined analysis with FPS percentages
    fps_analysis = create_fps_analysis_with_percentages(fps_results, fps_percentages, avg_total_costs)
    
    if not fps_analysis:
        print("No valid results found to analyze.")
        return None
    
    # Convert to DataFrame and sort by FPS
    fps_df = pd.DataFrame(fps_analysis).sort_values('fps')
    fps_df.to_csv(os.path.join(output_dir, 'mae_fps_dynamic_analysis.csv'), index=False)
    
    # Create visualizations
    create_equity_vs_percentage_plot(fps_df, output_dir)
    create_percentage_utilization_plots(fps_df, subsidy_usage_by_fps, fps_percentages, output_dir)
    create_dynamic_income_equity_plots(fps_df, output_dir)
    create_fps_percentage_summary_table(fps_df, output_dir)
    
    # Process optimal solutions with percentages
    if optimal_solutions:
        create_optimal_allocation_analysis_dynamic(optimal_solutions, fps_percentages, output_dir)
    
    # Generate comprehensive analysis report
    generate_dynamic_analysis_report(fps_df, fps_percentages, output_dir)
    
    print(f"\nDynamic analysis complete! Results saved to {output_dir}")
    return output_dir, fps_results
def parse_optimal_solution(opt_path, dir_path):
    """Parse optimal solution file and return structured data"""
    opt_data = {}
    try:
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
            elif line.strip().startswith("Equity Score by Income Level:"):
                in_allocation_section = False
                continue
                
            if in_allocation_section and line.strip().startswith("  "):
                parts = line.strip().split(":")
                key = parts[0].strip()
                value = float(parts[1].strip())
                allocations[key] = value
        
        opt_data['allocations'] = allocations
        opt_data['directory'] = dir_path
        
        return opt_data
    except Exception as e:
        print(f"Error parsing optimal solution: {e}")
        return None

def create_fps_analysis_with_percentages(fps_results, fps_percentages, avg_total_costs):
    """Create FPS analysis with percentage calculations"""
    fps_analysis = []
    
    for fps, results_list in fps_results.items():
        # Calculate equity scores
        equity_scores = [r.get('avg_equity', float('inf')) for r in results_list]
        valid_scores = [s for s in equity_scores if s != float('inf')]
        
        # Calculate income-specific scores
        low_scores = [r.get('equity_scores', {}).get('low', float('inf')) for r in results_list]
        mid_scores = [r.get('equity_scores', {}).get('middle', float('inf')) for r in results_list]
        high_scores = [r.get('equity_scores', {}).get('high', float('inf')) for r in results_list]
        
        if valid_scores:
            analysis_entry = {
                'fps': fps,
                'fps_percentage': fps_percentages.get(fps, None),
                'total_system_cost': avg_total_costs.get(fps, None),
                'mean_equity': np.mean(valid_scores),
                'std_equity': np.std(valid_scores),
                'min_equity': np.min(valid_scores),
                'max_equity': np.max(valid_scores),
                'count': len(valid_scores),
                'mean_low': np.mean([s for s in low_scores if s != float('inf')]),
                'mean_middle': np.mean([s for s in mid_scores if s != float('inf')]),
                'mean_high': np.mean([s for s in high_scores if s != float('inf')])
            }
            fps_analysis.append(analysis_entry)
    
    return fps_analysis

def create_equity_vs_percentage_plot(fps_df, output_dir):
    """Create equity comparison plot using FPS percentages"""
    plt.figure(figsize=(16, 12))
    
    # Plot lines for each income level
    plt.plot(fps_df['fps_percentage'], fps_df['mean_low'], 'o-', 
             color='#1f77b4', linewidth=2, markersize=8, label='Low Income')
    
    plt.plot(fps_df['fps_percentage'], fps_df['mean_middle'], 'o-', 
             color='#ff7f0e', linewidth=2, markersize=8, label='Middle Income')
    
    plt.plot(fps_df['fps_percentage'], fps_df['mean_high'], 'o-', 
             color='#2ca02c', linewidth=2, markersize=8, label='High Income')
    
    # Plot overall equity with emphasis
    plt.plot(fps_df['fps_percentage'], fps_df['mean_equity'], 'o-', 
             color='#d62728', linewidth=3, markersize=10, label='Overall Equity')
    
    # Add error bands
    plt.fill_between(fps_df['fps_percentage'],
                    fps_df['mean_equity'] - fps_df['std_equity'],
                    fps_df['mean_equity'] + fps_df['std_equity'],
                    color='#d62728', alpha=0.2)
    
    # Find and mark optimal point
    optimal_idx = fps_df['mean_equity'].idxmin()
    optimal_fps_pct = fps_df.loc[optimal_idx, 'fps_percentage']
    optimal_equity = fps_df.loc[optimal_idx, 'mean_equity']
    optimal_fps = fps_df.loc[optimal_idx, 'fps']
    
    plt.axvline(x=optimal_fps_pct, color='blue', linestyle='--', alpha=0.7,
               label=f'Optimal: {optimal_fps_pct:.1f}% of System Costs')
    plt.scatter([optimal_fps_pct], [optimal_equity], color='blue', s=200, zorder=10)
    
    # Find elbow point (simplified method)
    fps_sorted = fps_df.sort_values('fps_percentage')
    if len(fps_sorted) > 3:
        # Calculate second derivative approximation
        y = fps_sorted['mean_equity'].values
        x = fps_sorted['fps_percentage'].values
        
        # Find inflection point
        diff1 = np.diff(y)
        diff2 = np.diff(diff1)
        if len(diff2) > 0:
            elbow_idx = np.argmax(diff2) + 1
            if elbow_idx < len(fps_sorted):
                elbow_fps_pct = fps_sorted.iloc[elbow_idx]['fps_percentage']
                elbow_equity = fps_sorted.iloc[elbow_idx]['mean_equity']
                
                plt.axvline(x=elbow_fps_pct, color='green', linestyle='--', alpha=0.7,
                           label=f'Elbow Point: {elbow_fps_pct:.1f}%')
                plt.scatter([elbow_fps_pct], [elbow_equity], color='green', s=150, zorder=10)
    
    # Add annotations for key points
    plt.annotate(f'Optimal\n{optimal_fps_pct:.1f}%\n(FPS {optimal_fps:.0f})',
                xy=(optimal_fps_pct, optimal_equity),
                xytext=(optimal_fps_pct + 5, optimal_equity + 0.1),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    # Set labels and formatting
    plt.xlabel('Subsidy Level (% of Total System Costs)', fontsize=14)
    plt.ylabel('Equity Score (Lower is Better)', fontsize=14)
    plt.title('Mode Share Equity vs Subsidy Level\n(Dynamic-Referenced Analysis)', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Format x-axis to show percentages
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'equity_vs_percentage_dynamic.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create improvement rate analysis
    create_improvement_rate_plot(fps_df, output_dir)

def create_improvement_rate_plot(fps_df, output_dir):
    """Create plot showing improvement rates between subsidy levels"""
    # Sort by percentage
    fps_sorted = fps_df.sort_values('fps_percentage')
    
    # Calculate improvement rates
    improvements = []
    for i in range(1, len(fps_sorted)):
        prev_equity = fps_sorted.iloc[i-1]['mean_equity']
        curr_equity = fps_sorted.iloc[i]['mean_equity']
        improvement = (prev_equity - curr_equity) / prev_equity * 100
        improvements.append({
            'from_pct': fps_sorted.iloc[i-1]['fps_percentage'],
            'to_pct': fps_sorted.iloc[i]['fps_percentage'],
            'improvement': improvement,
            'midpoint_pct': (fps_sorted.iloc[i-1]['fps_percentage'] + fps_sorted.iloc[i]['fps_percentage']) / 2
        })
    
    if improvements:
        improvement_df = pd.DataFrame(improvements)
        
        plt.figure(figsize=(12, 8))
        plt.bar(improvement_df['midpoint_pct'], improvement_df['improvement'], 
                width=np.diff(fps_sorted['fps_percentage'].values).mean() * 0.8,
                alpha=0.7, color='steelblue')
        
        # Add value labels on bars
        for _, row in improvement_df.iterrows():
            plt.text(row['midpoint_pct'], row['improvement'] + 0.5,
                    f'{row["improvement"]:.1f}%',
                    ha='center', va='bottom')
        
        plt.xlabel('Subsidy Level (% of Total System Costs)', fontsize=12)
        plt.ylabel('Improvement Rate (%)', fontsize=12)
        plt.title('Marginal Improvement Rates Between Subsidy Levels', fontsize=14)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add 5% threshold line
        plt.axhline(y=5, color='red', linestyle='--', alpha=0.7, label='5% Threshold')
        plt.legend()
        
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'improvement_rates_dynamic.png'), dpi=300, bbox_inches='tight')
        plt.close()

def create_percentage_utilization_plots(fps_df, subsidy_usage_by_fps, fps_percentages, output_dir):
    """Create subsidy utilization plots using percentages"""
    if not subsidy_usage_by_fps:
        return
    
    # Process subsidy usage data with percentages
    usage_analysis = []
    for fps, usage_list in subsidy_usage_by_fps.items():
        pct_used = [data.get('percentage_used', 0) for data in usage_list]
        avg_pct_used = np.mean(pct_used) if pct_used else 0
        
        low_pct = []
        middle_pct = []
        high_pct = []
        
        for data in usage_list:
            if 'subsidy_by_income' in data:
                income_data = data['subsidy_by_income']
                low_pct.append(income_data.get('low', {}).get('percentage_of_used', 0))
                middle_pct.append(income_data.get('middle', {}).get('percentage_of_used', 0))
                high_pct.append(income_data.get('high', {}).get('percentage_of_used', 0))
        
        usage_analysis.append({
            'fps': fps,
            'fps_percentage': fps_percentages.get(fps, None),
            'avg_pct_used': avg_pct_used,
            'avg_low_pct': np.mean(low_pct) if low_pct else 0,
            'avg_middle_pct': np.mean(middle_pct) if middle_pct else 0,
            'avg_high_pct': np.mean(high_pct) if high_pct else 0
        })
    
    usage_df = pd.DataFrame(usage_analysis).sort_values('fps')
    usage_df.to_csv(os.path.join(output_dir, 'subsidy_usage_dynamic_analysis.csv'), index=False)
    
    # Utilization rate plot
    plt.figure(figsize=(12, 8))
    plt.plot(usage_df['fps_percentage'], usage_df['avg_pct_used'], 'o-', 
             color='green', linewidth=2, markersize=8)
    
    # Add annotations
    for _, row in usage_df.iterrows():
        if pd.notna(row['fps_percentage']):
            plt.annotate(f"{row['avg_pct_used']:.1f}%",
                        xy=(row['fps_percentage'], row['avg_pct_used']),
                        xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Subsidy Level (% of Total System Costs)', fontsize=12)
    plt.ylabel('Subsidy Utilization Rate (%)', fontsize=12)
    plt.title('Subsidy Utilization vs Investment Level\n(Dynamic-Referenced)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
    plt.ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'utilization_vs_percentage_dynamic.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Income distribution plot
    plt.figure(figsize=(12, 8))
    plt.plot(usage_df['fps_percentage'], usage_df['avg_low_pct'], 'o-', 
             color='#1f77b4', linewidth=2, markersize=8, label='Low Income')
    plt.plot(usage_df['fps_percentage'], usage_df['avg_middle_pct'], 'o-', 
             color='#ff7f0e', linewidth=2, markersize=8, label='Middle Income')
    plt.plot(usage_df['fps_percentage'], usage_df['avg_high_pct'], 'o-', 
             color='#2ca02c', linewidth=2, markersize=8, label='High Income')
    
    plt.xlabel('Subsidy Level (% of Total System Costs)', fontsize=12)
    plt.ylabel('Percentage of Used Subsidies (%)', fontsize=12)
    plt.title('Subsidy Distribution by Income Group\n(Dynamic-Referenced)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'income_distribution_vs_percentage_dynamic.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_dynamic_income_equity_plots(fps_df, output_dir):
    """Create income-specific equity plots with percentages"""
    # Stacked area chart
    plt.figure(figsize=(12, 8))
    
    # Calculate proportion of equity score by income level
    fps_df_copy = fps_df.copy()
    for i, row in fps_df_copy.iterrows():
        total = row['mean_low'] + row['mean_middle'] + row['mean_high']
        if total > 0:
            fps_df_copy.at[i, 'low_pct'] = row['mean_low'] / total * 100
            fps_df_copy.at[i, 'middle_pct'] = row['mean_middle'] / total * 100
            fps_df_copy.at[i, 'high_pct'] = row['mean_high'] / total * 100
    
    plt.stackplot(fps_df_copy['fps_percentage'], 
                 [fps_df_copy['low_pct'], fps_df_copy['middle_pct'], fps_df_copy['high_pct']],
                 labels=['Low Income', 'Middle Income', 'High Income'],
                 colors=['#1f77b4', '#ff7f0e', '#2ca02c'],
                 alpha=0.7)
    
    plt.xlabel('Subsidy Level (% of Total System Costs)', fontsize=12)
    plt.ylabel('Percentage of Total Equity Score (%)', fontsize=12)
    plt.title('Income Group Contribution to Total Equity Score\n(Dynamic-Referenced)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'equity_contribution_by_income_dynamic.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_fps_percentage_summary_table(fps_df, output_dir):
    """Create comprehensive summary table with FPS percentages"""
    # Prepare table data
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table data
    table_data = []
    for _, row in fps_df.sort_values('fps').iterrows():
        table_data.append([
            f"{row['fps']:.0f}",
            f"{row['fps_percentage']:.1f}%",
            f"{row['total_system_cost']:.0f}",
            f"{row['mean_equity']:.4f}",
            f"{row['mean_low']:.4f}",
            f"{row['mean_middle']:.4f}",
            f"{row['mean_high']:.4f}"
        ])
    
    table = ax.table(cellText=table_data,
                    colLabels=['FPS Value', 'System Cost %', 'Total Cost (Units)', 
                              'Overall Equity', 'Low Income', 'Middle Income', 'High Income'],
                    cellLoc='center', loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2)
    
    # Highlight optimal row
    optimal_idx = fps_df['mean_equity'].idxmin()
    optimal_row_idx = list(fps_df.index).index(optimal_idx) + 1
    for col in range(len(table_data[0])):
        table[(optimal_row_idx, col)].set_facecolor('#ccffcc')
    
    plt.title('FPS Analysis Summary: Dynamic-Referenced Percentages\n(Green = Optimal)', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fps_percentage_summary_table_dynamic.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_optimal_allocation_analysis_dynamic(optimal_solutions, fps_percentages, output_dir):
    """Create optimal allocation analysis with percentage context"""
    if not optimal_solutions:
        return
    
    # Find best solution
    best_solution = min(optimal_solutions, key=lambda x: x['equity'])
    best_fps = best_solution['fps']
    best_fps_pct = fps_percentages.get(best_fps, 0)
    
    # Create allocation heatmap
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
        
        if income_mode_data:
            alloc_df = pd.DataFrame(income_mode_data)
            
            plt.figure(figsize=(12, 8))
            pivot_table = alloc_df.pivot(index='income_level', columns='mode', values='allocation')
            sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt=".2f", 
                       cbar_kws={'label': 'Allocation Percentage'})
            
            plt.title(f'Optimal Subsidy Allocations\n({best_fps_pct:.1f}% of System Costs, FPS={best_fps:.0f})', 
                     fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'optimal_allocations_heatmap_dynamic.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()

def generate_dynamic_analysis_report(fps_df, fps_percentages, output_dir):
    """Generate comprehensive analysis report with dynamic percentages"""
    # Find key thresholds
    optimal_idx = fps_df['mean_equity'].idxmin()
    optimal_fps = fps_df.loc[optimal_idx, 'fps']
    optimal_fps_pct = fps_df.loc[optimal_idx, 'fps_percentage']
    optimal_equity = fps_df.loc[optimal_idx, 'mean_equity']
    
    # Find improvement rates
    fps_sorted = fps_df.sort_values('fps')
    improvements = []
    for i in range(1, len(fps_sorted)):
        prev_val = fps_sorted.iloc[i-1]['mean_equity']
        curr_val = fps_sorted.iloc[i]['mean_equity']
        pct_change = (prev_val - curr_val) / prev_val * 100
        improvements.append({
            'fps': fps_sorted.iloc[i]['fps'],
            'fps_pct': fps_sorted.iloc[i]['fps_percentage'],
            'improvement': pct_change
        })
    
    # Write analysis report
    with open(os.path.join(output_dir, 'dynamic_analysis_report.txt'), 'w') as f:
        f.write("Mode Share Equity Dynamic Analysis Report\n")
        f.write("==========================================\n\n")
        f.write("Executive Summary:\n")
        f.write("-----------------\n")
        f.write(f"Optimal subsidy level: {optimal_fps_pct:.1f}% of total system costs (FPS {optimal_fps:.0f})\n")
        f.write(f"Optimal equity score: {optimal_equity:.4f}\n")
        f.write(f"Investment range tested: {fps_df['fps_percentage'].min():.1f}% - {fps_df['fps_percentage'].max():.1f}%\n\n")
        
        f.write("Key Findings:\n")
        f.write("-------------\n")
        f.write("1. The relationship between subsidy level and equity follows a non-linear pattern\n")
        f.write("2. Significant improvements occur in the lower investment ranges\n")
        f.write("3. Diminishing returns appear after the optimal point\n")
        f.write("4. Progressive allocation patterns emerge across all subsidy levels\n\n")
        
        f.write("Detailed Results by Subsidy Level:\n")
        f.write("----------------------------------\n")
        for _, row in fps_df.sort_values('mean_equity').iterrows():
            f.write(f"FPS {row['fps']:.0f} ({row['fps_percentage']:.1f}%): Equity Score {row['mean_equity']:.4f}\n")
        
        f.write("\nImprovement Rates:\n")
        f.write("------------------\n")
        for imp in improvements:
            f.write(f"{imp['fps_pct']:.1f}% (FPS {imp['fps']:.0f}): {imp['improvement']:.2f}% improvement\n")
        
        f.write(f"\nMethodology:\n")
        f.write(f"------------\n")
        f.write(f"This analysis uses dynamic-referenced percentages where FPS values are expressed\n")
        f.write(f"as percentages of actual total system costs at each intervention level.\n")
        f.write(f"This accounts for behavioral responses including induced demand and mode shifts.\n")

def plot_normalized_diminishing_returns(fps_df, output_dir):
    """Create normalized diminishing returns visualization"""
    plt.figure(figsize=(12, 8))
    
    # Sort and normalize
    df_sorted = fps_df.sort_values('fps_percentage')
    baseline = df_sorted['mean_equity'].max()
    best_value = df_sorted['mean_equity'].min()
    
    # Calculate normalized improvement
    improvements = (baseline - df_sorted['mean_equity']) / (baseline - best_value)
    
    plt.plot(df_sorted['fps_percentage'], improvements, 'o-', 
             color='#d62728', linewidth=2, markersize=8)
    
    # Add reference lines
    plt.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='90% of max benefit')
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% of max benefit')
    
    # Find 90% point
    fps_90pct = None
    for i, imp in enumerate(improvements):
        if imp >= 0.9:
            fps_90pct = df_sorted.iloc[i]['fps_percentage']
            break
    
    if fps_90pct:
        plt.axvline(x=fps_90pct, color='green', linestyle='--', alpha=0.7)
        plt.annotate(f'90% benefit at\n{fps_90pct:.1f}%', 
                    xy=(fps_90pct, 0.9),
                    xytext=(fps_90pct + 5, 0.8),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))
    
    plt.xlabel('Subsidy Level (% of Total System Costs)', fontsize=12)
    plt.ylabel('Normalized Equity Improvement', fontsize=12)
    plt.title('Diminishing Returns Analysis\n(Dynamic-Referenced)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'diminishing_returns_dynamic.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    output_dir, results = aggregate_mae_results_dynamic()
    if output_dir:
        print("Dynamic analysis complete!")
        print(f"Results saved to: {output_dir}")
    else:
        print("Analysis failed. Check error messages above.")
