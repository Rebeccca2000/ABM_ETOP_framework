"""
FPS-PBS Optimization Analysis

This script performs a comprehensive analysis of the relationship between Fixed Pool Subsidy (FPS) 
values and optimal Percentage-Based Subsidy (PBS) allocations to maximize the Social Distribution 
Index (SDI).

For each FPS value in a defined range:
1. Runs PBS analysis with the FPS value as a constraint
2. Finds optimal percentage allocations that maximize SDI
3. Records the FPS value, optimal allocations, and corresponding SDI score

Finally, it generates visualizations showing how the optimal SDI and allocations change with FPS.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
import os
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
import traceback
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from agent_subsidy_pool import SubsidyPoolConfig
from run_visualisation_03 import MobilityModel
from agent_service_provider_initialisation_03 import reset_database, CommuterInfoLog, ServiceBookingLog
from scipy.optimize import differential_evolution

# Import functions from sensitivity_check_SDI
from sensitivity_check_SDI import calculate_sur, calculate_mae, calculate_upi, calculate_sdi, calculate_sdi_metrics

# Constants
SIMULATION_STEPS = 15  # Reduced for faster runs during optimization
NUM_CPUS = 8  # Adjust based on your system

def run_single_simulation(params):
    """Run a single simulation and calculate SDI metrics"""
    print(f"Starting simulation with PID {os.getpid()} - FPS: {params.get('fps_value', 'N/A')}")
    db_path = f"service_provider_database_{os.getpid()}.db"
    db_connection_string = f"sqlite:///{db_path}"
    
    try:
        # Extract analysis parameters
        fps_value = params.pop('fps_value', 0)
        fixed_allocations = params.pop('fixed_allocations', None)
        simulation_steps = params.pop('simulation_steps', SIMULATION_STEPS)
        
        # Initialize database and simulation
        engine = create_engine(db_connection_string)
        Session = sessionmaker(bind=engine)
        session = Session()

        reset_db_params = {k: params[k] for k in [
            'uber_like1_capacity', 'uber_like1_price',
            'uber_like2_capacity', 'uber_like2_price',
            'bike_share1_capacity', 'bike_share1_price',
            'bike_share2_capacity', 'bike_share2_price'
        ]}
        
        reset_database(engine=engine, session=session, **reset_db_params)
        
        # Set up subsidy configuration
        # If fixed_allocations is provided, use it for PBS with the FPS constraint
        if fixed_allocations:
            # Create subsidy dataset
            subsidy_dataset = {}
            for income_level in ['low', 'middle', 'high']:
                subsidy_dataset[income_level] = {}
                for mode in ['bike', 'car', 'MaaS_Bundle']:
                    key = f"{income_level}_{mode}"
                    if key in fixed_allocations:
                        subsidy_dataset[income_level][mode] = fixed_allocations[key]
                    else:
                        # Default allocation if not specified
                        subsidy_dataset[income_level][mode] = 0.1
                        
            params['subsidy_dataset'] = subsidy_dataset
            params['subsidy_config'] = SubsidyPoolConfig('daily', float(fps_value))
        else:
            # Regular FPS
            params['subsidy_config'] = SubsidyPoolConfig('daily', float(fps_value))
        
        # Run simulation
        model = MobilityModel(db_connection_string=db_connection_string, **params)
        model.run_model(simulation_steps)
        
        # Calculate SDI metrics
        results = calculate_sdi_metrics(session)
        
        # Add simulation parameters to results
        results['fps_value'] = fps_value
        results['fixed_allocations'] = fixed_allocations
        
        return results
        
    except Exception as e:
        print(f"Error in simulation {os.getpid()}: {str(e)}")
        traceback.print_exc()
        return None
        
    finally:
        session.close()
        if os.path.exists(db_path):
            os.remove(db_path)

def optimize_subsidy_allocation(fps_value, base_parameters, n_generations=10, population_size=20):
    """
    Optimize subsidy allocation percentages for a given FPS value
    
    Args:
        fps_value: Fixed Pool Subsidy value to use as constraint
        base_parameters: Base simulation parameters
        n_generations: Number of generations for optimization
        population_size: Population size for optimization
        
    Returns:
        dict: Optimal allocation and corresponding SDI scores
    """
    print(f"\nOptimizing subsidy allocation for FPS = {fps_value}")
    
    # Define parameter bounds for optimization
    # For each income level and mode, we need to optimize allocation percentage
    # Format: [low_bike, low_car, low_maas, middle_bike, middle_car, middle_maas, high_bike, high_car, high_maas]
    param_bounds = [(0.05, 0.6) for _ in range(9)]
    
    # Define objective function (negative of average SDI to minimize)
    def objective_function(params):
        # Convert optimization parameters to allocation dictionary
        allocations = {
            'low_bike': params[0],
            'low_car': params[1],
            'low_MaaS_Bundle': params[2],
            'middle_bike': params[3],
            'middle_car': params[4],
            'middle_MaaS_Bundle': params[5],
            'high_bike': params[6],
            'high_car': params[7],
            'high_MaaS_Bundle': params[8]
        }
        
        # Create simulation parameters
        sim_params = base_parameters.copy()
        sim_params['fps_value'] = fps_value
        sim_params['fixed_allocations'] = allocations
        sim_params['simulation_steps'] = SIMULATION_STEPS
        
        # Run simulation with these allocations
        results = run_single_simulation(sim_params)
        
        if results:
            # Calculate average SDI across income levels
            sdi_values = [results[level]['sdi'] for level in ['low', 'middle', 'high']]
            avg_sdi = np.mean(sdi_values)
            print(f"Tested allocation with avg SDI: {avg_sdi:.4f}")
            return -avg_sdi  # Negative because we want to maximize
        else:
            # Return a large value if simulation failed
            return 100.0
    
    # Run optimization
    print("Starting optimization process...")
    optimizer_args = {
        'bounds': param_bounds,
        'strategy': 'best1bin',
        'maxiter': n_generations,
        'popsize': population_size,
        'tol': 0.01,
        'mutation': (0.5, 1.0),
        'recombination': 0.7,
        'seed': 42,
        'disp': True,
        'polish': True
    }
    
    try:
        result = differential_evolution(objective_function, **optimizer_args)
        
        # Extract optimal parameters
        optimal_allocations = {
            'low_bike': result.x[0],
            'low_car': result.x[1],
            'low_MaaS_Bundle': result.x[2],
            'middle_bike': result.x[3],
            'middle_car': result.x[4],
            'middle_MaaS_Bundle': result.x[5],
            'high_bike': result.x[6],
            'high_car': result.x[7],
            'high_MaaS_Bundle': result.x[8]
        }
        
        # Run final simulation with optimal allocations to get full results
        sim_params = base_parameters.copy()
        sim_params['fps_value'] = fps_value
        sim_params['fixed_allocations'] = optimal_allocations
        sim_params['simulation_steps'] = SIMULATION_STEPS
        final_results = run_single_simulation(sim_params)
        
        return {
            'fps_value': fps_value,
            'optimal_allocations': optimal_allocations,
            'sdi_scores': {level: final_results[level]['sdi'] for level in ['low', 'middle', 'high']},
            'avg_sdi': np.mean([final_results[level]['sdi'] for level in ['low', 'middle', 'high']]),
            'full_results': final_results
        }
        
    except Exception as e:
        print(f"Optimization error for FPS {fps_value}: {str(e)}")
        traceback.print_exc()
        return {
            'fps_value': fps_value,
            'error': str(e),
            'optimal_allocations': None,
            'sdi_scores': None,
            'avg_sdi': 0.0
        }

def fps_pbs_sweep_analysis(fps_values, base_parameters):
    """
    Perform a sweep analysis across multiple FPS values to find optimal PBS allocations
    
    Args:
        fps_values: List of FPS values to analyze
        base_parameters: Base simulation parameters
        
    Returns:
        dict: Results for each FPS value
    """
    results = {}
    
    for fps in fps_values:
        print(f"\n{'='*50}")
        print(f"Running analysis for FPS = {fps}")
        print(f"{'='*50}")
        
        # Run optimization for this FPS value
        opt_result = optimize_subsidy_allocation(fps, base_parameters)
        
        # Store results
        results[fps] = opt_result
        
        # Save intermediate results to avoid losing progress
        with open(f'fps_pbs_sweep_intermediate_{fps}.pkl', 'wb') as f:
            pickle.dump(opt_result, f)
            
        print(f"Completed analysis for FPS = {fps}")
        print(f"Optimal Average SDI: {opt_result.get('avg_sdi', 'N/A')}")
        
    return results

def visualize_results(results, output_dir='fps_pbs_analysis_results'):
    """
    Create enhanced visualizations from the sweep analysis results
    specifically targeting the main graph requested with allocation tables
    
    Args:
        results: Dictionary of results from fps_pbs_sweep_analysis
        output_dir: Directory to save visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert results to DataFrame for easier visualization
    rows = []
    for fps, result in results.items():
        if 'optimal_allocations' in result and result['optimal_allocations']:
            row = {'fps_value': fps, 'avg_sdi': result['avg_sdi']}
            
            # Add SDI scores for each income level
            for level in ['low', 'middle', 'high']:
                if 'sdi_scores' in result and result['sdi_scores']:
                    row[f'sdi_{level}'] = result['sdi_scores'][level]
            
            # Add allocation percentages
            for key, value in result['optimal_allocations'].items():
                row[f'alloc_{key}'] = value
                
            rows.append(row)
    
    if not rows:
        print("No valid results to visualize")
        return
        
    df = pd.DataFrame(rows)
    
    # Save full data table that will be referenced in the visualizations
    data_table = df.copy()
    # Format allocation percentages as percentages
    for col in data_table.columns:
        if col.startswith('alloc_'):
            data_table[col] = data_table[col].map(lambda x: f"{x*100:.1f}%")
    
    data_table.to_csv(os.path.join(output_dir, 'fps_pbs_allocation_table.csv'), index=False)
    
    # 1. Create main visualization with all income levels and allocation references
    plt.figure(figsize=(15, 10))
    
    # Plot lines for each income level
    colors = {'low': '#1f77b4', 'middle': '#ff7f0e', 'high': '#2ca02c', 'avg': '#d62728'}
    markers = {'low': 'o', 'middle': 's', 'high': '^', 'avg': 'D'}
    
    # Plot income-specific SDI values
    for level in ['low', 'middle', 'high']:
        plt.plot(df['fps_value'], df[f'sdi_{level}'], 
                 marker=markers[level], linestyle='-', 
                 color=colors[level], 
                 label=f'{level.capitalize()} Income', 
                 alpha=0.8)
    
    # Plot average SDI with bolder line
    plt.plot(df['fps_value'], df['avg_sdi'], 
             marker=markers['avg'], linestyle='-', 
             color=colors['avg'],
             linewidth=3, 
             label='Average SDI', 
             alpha=0.9)
    
    # Add trend line using polynomial fit
    if len(df) > 2:
        try:
            z = np.polyfit(df['fps_value'], df['avg_sdi'], 3)
            p = np.poly1d(z)
            x_smooth = np.linspace(df['fps_value'].min(), df['fps_value'].max(), 100)
            y_smooth = p(x_smooth)
            plt.plot(x_smooth, y_smooth, '--', color='gray', alpha=0.6, label='Trend Line')
            
            # Find and mark optimal point
            optimal_idx = np.argmax(y_smooth)
            optimal_fps = x_smooth[optimal_idx]
            optimal_sdi = y_smooth[optimal_idx]
            
            plt.scatter([optimal_fps], [optimal_sdi], 
                        marker='*', s=300, color='red', zorder=10,
                        label=f'Optimal Point (FPS={optimal_fps:.0f})')
            
            # Annotate optimal point
            plt.annotate(f'Optimal: SDI={optimal_sdi:.4f}',
                        xy=(optimal_fps, optimal_sdi),
                        xytext=(10, -30), 
                        textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))
        except Exception as e:
            print(f"Error fitting trend line: {e}")
    
    # Add point annotations for allocation references
    # We'll add annotations to every other point to avoid overcrowding
    for i, (idx, row) in enumerate(df.iterrows()):
        if i % 2 == 0:  # Only annotate every other point
            fps = row['fps_value']
            avg_sdi = row['avg_sdi']
            
            # Create a simplified allocation text
            alloc_text = f"FPS {fps}\n"
            for level in ['low', 'middle', 'high']:
                mode_values = []
                for mode in ['bike', 'car', 'MaaS_Bundle']:
                    col = f'alloc_{level}_{mode}'
                    if col in row:
                        mode_values.append(f"{mode.split('_')[0]}: {row[col]:.2f}")
                alloc_text += f"{level}: {', '.join(mode_values)}\n"
            
            # Add annotation with arrow pointing to point
            plt.annotate(f'Point {i+1}',  # Use numbers to reference the table
                        xy=(fps, avg_sdi),
                        xytext=(10, 10 + (i % 3) * 20),  # Stagger annotations
                        textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'),
                        fontsize=8)
    
    # Add table reference
    plt.figtext(0.02, 0.02, 
               "Note: See 'fps_pbs_allocation_table.csv' for the complete allocations table",
               bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
    
    # Improve graph appearance
    plt.xscale('log')  # Log scale for FPS values
    plt.title('Optimal SDI Scores vs Fixed Pool Subsidy (FPS) Values', fontsize=16)
    plt.xlabel('Fixed Pool Subsidy (FPS)', fontsize=14)
    plt.ylabel('Social Distribution Index (SDI)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='best')
    
    # Add table with optimal allocations for key FPS values
    # Select representative points for the table
    if len(df) > 0:
        table_rows = []
        fps_indices = []
        
        # Get indices for interesting points
        if len(df) <= 5:
            fps_indices = list(range(len(df)))
        else:
            # Get first, last, and some points in between
            step = max(1, len(df) // 5)
            fps_indices = list(range(0, len(df), step))
            if len(df) - 1 not in fps_indices:
                fps_indices.append(len(df) - 1)
        
        # Extract data for table
        for idx in fps_indices:
            row = df.iloc[idx]
            table_row = [
                f"{row['fps_value']:.0f}",
                f"{row['avg_sdi']:.4f}"
            ]
            
            # Add allocations for each income level and mode
            for level in ['low', 'middle', 'high']:
                for mode in ['bike', 'car', 'MaaS_Bundle']:
                    col = f'alloc_{level}_{mode}'
                    if col in row:
                        table_row.append(f"{row[col]:.2f}")
                    else:
                        table_row.append("N/A")
            
            table_rows.append(table_row)
        
        # Create table at the bottom of the figure
        table_data = np.array(table_rows)
        cols = ['FPS', 'Avg SDI'] + [
            f"{level} {mode.split('_')[0]}" 
            for level in ['low', 'middle', 'high'] 
            for mode in ['bike', 'car', 'MaaS_Bundle']
        ]
        
        # Only add table if we have data
        if len(table_data) > 0:
            table_ax = plt.gcf().add_axes([0.1, -0.3, 0.8, 0.2])  # x, y, width, height
            table_ax.axis('off')
            table = table_ax.table(
                cellText=table_data,
                colLabels=cols,
                loc='center',
                cellLoc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)
            plt.gcf().set_size_inches(15, 15)  # Make figure taller to accommodate table
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)  # Adjust to make room for the table
    plt.savefig(os.path.join(output_dir, 'sdi_vs_fps_with_allocations.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Create detailed allocation line graphs for each income level
    income_levels = ['low', 'middle', 'high']
    modes = ['bike', 'car', 'MaaS_Bundle']
    
    # Create a separate figure for allocation trends
    plt.figure(figsize=(15, 15))
    
    for i, income in enumerate(income_levels):
        plt.subplot(3, 1, i+1)
        
        for mode in modes:
            key = f'alloc_{income}_{mode}'
            if key in df.columns:
                plt.plot(df['fps_value'], df[key], 'o-', 
                         label=f'{mode.replace("_", " ")}', 
                         linewidth=2)
        
        plt.title(f'Optimal Subsidy Allocation vs FPS - {income.capitalize()} Income')
        plt.xlabel('Fixed Pool Subsidy (FPS)')
        plt.ylabel('Allocation Percentage')
        plt.xscale('log')  # Log scale for better visualization
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'allocation_trends.png'))
    plt.close()
    
    # 3. Create summary figures showing overall trends
    
    # Create a 3D visualization to see relationships across all dimensions
    try:
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot each income level as a separate series
        for level in ['low', 'middle', 'high']:
            ax.scatter(df['fps_value'], 
                      df[f'alloc_{level}_MaaS_Bundle'], 
                      df[f'sdi_{level}'],
                      label=f'{level.capitalize()} Income',
                      s=50)
        
        ax.set_xlabel('FPS Value')
        ax.set_ylabel('MaaS Allocation %')
        ax.set_zlabel('SDI Score')
        ax.set_xscale('log')
        ax.set_title('3D Relationship: FPS, MaaS Allocation, and SDI')
        ax.legend()
        
        plt.savefig(os.path.join(output_dir, '3d_relationship.png'))
        plt.close()
    except:
        print("Could not create 3D visualization (Matplotlib 3D plotting might not be available)")
    
    # Find and save the optimal solution
    if 'avg_sdi' in df.columns:
        optimal_row = df.loc[df['avg_sdi'].idxmax()]
        
        with open(os.path.join(output_dir, 'optimal_solution.txt'), 'w') as f:
            f.write(f"Optimal FPS Value: {optimal_row['fps_value']}\n")
            f.write(f"Optimal Average SDI: {optimal_row['avg_sdi']:.4f}\n\n")
            f.write("Optimal Subsidy Allocations:\n")
            
            for key in sorted([k for k in optimal_row.index if k.startswith('alloc_')]):
                f.write(f"  {key.replace('alloc_', '')}: {optimal_row[key]:.4f}\n")
                
            f.write("\nSDI by Income Level:\n")
            for level in ['low', 'middle', 'high']:
                f.write(f"  {level.capitalize()}: {optimal_row[f'sdi_{level}']:.4f}\n")

                
def run_optimized_simulation(optimal_fps, optimal_allocations, base_parameters):
    """
    Run a simulation with the optimal FPS and allocations for comprehensive metrics
    """
    # Set up parameters with optimal values
    params = base_parameters.copy()
    params['fps_value'] = optimal_fps
    params['fixed_allocations'] = optimal_allocations
    params['simulation_steps'] = SIMULATION_STEPS * 2  # Run longer for better results
    
    # Run simulation
    results = run_single_simulation(params)
    
    # Save detailed results
    with open('optimal_solution_detailed_results.pkl', 'wb') as f:
        pickle.dump(results, f)
        
    return results

def main():
    """Main function to run the complete analysis"""
    # Define base parameters
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
        'public_price_table': {
            'train': {'on_peak': 2, 'off_peak': 1.5},
            'bus': {'on_peak': 1, 'off_peak': 0.8}
        },
        'ALPHA_VALUES': {
            'UberLike1': 0.5,
            'UberLike2': 0.5,
            'BikeShare1': 0.5,
            'BikeShare2': 0.5
        },
        'DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS': {
            'S_base': 0.08,
            'alpha': 0.2,
            'delta': 0.5
        },
        'BACKGROUND_TRAFFIC_AMOUNT': 70,
        'CONGESTION_ALPHA': 0.25,
        'CONGESTION_BETA': 4,
        'CONGESTION_CAPACITY': 4,
        'CONGESTION_T_IJ_FREE_FLOW': 2,
        'uber_like1_capacity': 8,
        'uber_like1_price': 6,
        'uber_like2_capacity': 9,
        'uber_like2_price': 6.5,
        'bike_share1_capacity': 10,
        'bike_share1_price': 1,
        'bike_share2_capacity': 12,
        'bike_share2_price': 1.2,
    }
    
    # Define FPS values to analyze (use a logarithmic scale for better coverage)
    # For initial testing, use a smaller set of values
    # fps_values = [1, 10, 100, 1000, 5000, 10000, 20000, 50000, 100000]
    
    # For actual analysis, use a more comprehensive set
    fps_values = np.logspace(0, 5, 10).astype(int)  # 10 points from 1 to 100,000 on log scale
    
    try:
        # Run sweep analysis
        print(f"Starting FPS-PBS sweep analysis with {len(fps_values)} FPS values")
        print(f"FPS values: {fps_values}")
        
        results = fps_pbs_sweep_analysis(fps_values, base_parameters)
        
        # Save full results
        with open('fps_pbs_sweep_results.pkl', 'wb') as f:
            pickle.dump(results, f)
            
        # Create visualizations
        visualize_results(results)
        
        # Find optimal FPS and allocations
        optimal_fps = None
        max_sdi = 0
        
        for fps, result in results.items():
            if 'avg_sdi' in result and result['avg_sdi'] > max_sdi:
                max_sdi = result['avg_sdi']
                optimal_fps = fps
                
        if optimal_fps:
            print(f"\nOptimal FPS Value: {optimal_fps}")
            print(f"Optimal Average SDI: {max_sdi:.4f}")
            
            # Run final simulation with optimal values
            optimal_allocations = results[optimal_fps]['optimal_allocations']
            optimal_results = run_optimized_simulation(optimal_fps, optimal_allocations, base_parameters)
            
            print("\nDetailed results for optimal solution saved.")
            
        print("\nAnalysis complete.")
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()