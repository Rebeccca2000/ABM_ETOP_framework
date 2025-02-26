from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm, qmc
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
from functools import partial


# Constants
SIMULATION_STEPS = 144  # Reduced for faster runs during optimization
NUM_CPUS = 8  # Adjust based on your system

def calculate_equity_indicator(session):
    """
    Calculate equity indicator with amplified differences.
    
    Args:
        session: SQLAlchemy session with simulation data
        
    Returns:
        dict: Results containing equity indicators for each income level
    """
    results = {}
    income_levels = ['low', 'middle', 'high']
    
    # Get mode shares for each income level
    mode_shares = {}
    for income_level in income_levels:
        mode_shares[income_level] = {}
        
        # Query mode choices
        mode_choices = session.query(
            ServiceBookingLog.record_company_name,
            func.count(ServiceBookingLog.request_id).label('count')
        ).join(
            CommuterInfoLog,
            ServiceBookingLog.commuter_id == CommuterInfoLog.commuter_id
        ).filter(
            CommuterInfoLog.income_level == income_level
        ).group_by(
            ServiceBookingLog.record_company_name
        ).all()
        
        # Calculate total trips and mode shares
        total_trips = sum(choice[1] for choice in mode_choices) or 1  # Avoid division by zero
        
        for mode, count in mode_choices:
            # Normalize company names to mode categories
            if 'Bike' in mode or mode == 'bike':
                category = 'bike'
            elif 'Uber' in mode or mode == 'car':
                category = 'car'
            elif mode == 'MaaS_Bundle':
                category = 'MaaS'
            elif mode == 'public':
                category = 'public'
            elif mode == 'walk':
                category = 'walk'
            else:
                category = mode
                
            # Store mode share
            if category in mode_shares[income_level]:
                mode_shares[income_level][category] += count / total_trips
            else:
                mode_shares[income_level][category] = count / total_trips
    
    # Ensure all modes are represented in each income level
    all_modes = set()
    for income_shares in mode_shares.values():
        all_modes.update(income_shares.keys())
    
    for income_level in income_levels:
        # Fill in missing modes with zero share
        for mode in all_modes:
            if mode not in mode_shares[income_level]:
                mode_shares[income_level][mode] = 0
    
    # Calculate average mode share for each mode across all income levels
    avg_mode_shares = {}
    for mode in all_modes:
        shares = [mode_shares[income][mode] for income in income_levels if mode in mode_shares[income]]
        avg_mode_shares[mode] = sum(shares) / len(shares) if shares else 0
    
    # Calculate equity indicator for each income level using absolute difference
    # and apply scaling to amplify differences
    equity_indicators = {}
    total_difference = 0
    scaling_factor = 10  # Adjust this value to amplify differences
    transformation_power = 0.5  # Use square root to amplify small differences
    
    # Store detailed breakdown for each income level
    for income_level in income_levels:
        mode_details = []
        income_difference = 0
        
        for mode in all_modes:
            mode_share = mode_shares[income_level].get(mode, 0)
            avg_share = avg_mode_shares.get(mode, 0)
            
            # Calculate absolute difference and apply transformation
            abs_diff = abs(mode_share - avg_share)
            transformed_diff = abs_diff ** transformation_power  # Square root to amplify small differences
            scaled_diff = transformed_diff * scaling_factor
            
            income_difference += scaled_diff
            
            mode_details.append({
                'mode': mode,
                'mode_share': mode_share,
                'avg_share': avg_share,
                'difference': scaled_diff
            })
        
        # Store the indicator for this income level
        equity_indicators[income_level] = income_difference
        
        # Add to total difference
        total_difference += income_difference
        
        # Store detailed results
        results[income_level] = {
            'equity_indicator': income_difference,
            'mode_shares': mode_shares[income_level],
            'mode_details': mode_details
        }
    
    # Calculate total equity indicator
    results['total_equity_indicator'] = total_difference
    
    return results

def run_single_simulation(params):
    """Run a single simulation and calculate equity metrics"""
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
                for mode in ['bike', 'car', 'MaaS_Bundle', 'public', 'walk']:
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
        
        # Calculate equity metrics using our new function
        results = calculate_equity_indicator(session)
        
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
def optimize_allocation_sequential(fps_value, base_parameters):
    """
    Optimize subsidy allocation for a given FPS value using sequential processing
    with income-specific boundary constraints for more realistic optimization.
    """
    print(f"\nOptimizing subsidy allocation for FPS = {fps_value}")
    
    # Define income-specific bounds for better policy control
    bounds = {
        # Low income - higher public transit and bike subsidies, moderate MaaS, lower car
        'low_bike': (0.10, 0.45),      
        'low_car': (0.1, 0.7),       
        'low_MaaS_Bundle': (0.10, 0.45),
        'low_public': (0.1, 0.7),    
        
        # Middle income - balanced approach
        'middle_bike': (0.10, 0.35),
        'middle_car': (0.05, 0.40),
        'middle_MaaS_Bundle': (0.1, 0.40),
        'middle_public': (0.15, 0.45),
        
        # High income - lower subsidies overall as less subsidy-dependent
        'high_bike': (0.05, 0.50),
        'high_car': (0.05, 0.20),      
        'high_MaaS_Bundle': (0.15, 0.40),
        'high_public': (0.05, 0.40)
    }
    
    # Create arrays of lower and upper bounds for sampling
    lower_bounds = []
    upper_bounds = []
    param_names = [
        'low_bike', 'low_car', 'low_MaaS_Bundle', 'low_public',
        'middle_bike', 'middle_car', 'middle_MaaS_Bundle', 'middle_public',
        'high_bike', 'high_car', 'high_MaaS_Bundle', 'high_public'
    ]
    
    for param in param_names:
        lower_bounds.append(bounds[param][0])
        upper_bounds.append(bounds[param][1])
    
    # Use Latin Hypercube Sampling with specific bounds
    sampler = qmc.LatinHypercube(d=12, seed=42)
    X_init = sampler.random(n=10)  # 10 initial points for better coverage
    
    # Scale each parameter according to its specific bounds
    X_init = qmc.scale(X_init, lower_bounds, upper_bounds)
    
    # Apply total subsidy constraints
    for i in range(len(X_init)):
        # Get subsidies by income level
        low_subsidies = X_init[i, 0:4]
        middle_subsidies = X_init[i, 4:8]
        high_subsidies = X_init[i, 8:12]
        
        # Check if total subsidies exceed maximum allowed (1.0)
        max_total = 1.0  # Maximum total subsidy per income level
        
        # Scale down if necessary while preserving proportions
        if np.sum(low_subsidies) > max_total:
            scale_factor = max_total / np.sum(low_subsidies)
            X_init[i, 0:4] = low_subsidies * scale_factor
            
        if np.sum(middle_subsidies) > max_total:
            scale_factor = max_total / np.sum(middle_subsidies)
            X_init[i, 4:8] = middle_subsidies * scale_factor
            
        if np.sum(high_subsidies) > max_total:
            scale_factor = max_total / np.sum(high_subsidies)
            X_init[i, 8:12] = high_subsidies * scale_factor
    
    best_equity = float('inf') 
    best_allocations = None
    best_results = None

    # Prepare parameters for initial points
    initial_param_sets = []
    for params in X_init:
        allocations = {
            'low_bike': params[0],
            'low_car': params[1],
            'low_MaaS_Bundle': params[2],
            'low_public': params[3],
            'middle_bike': params[4],
            'middle_car': params[5],
            'middle_MaaS_Bundle': params[6],
            'middle_public': params[7],
            'high_bike': params[8],
            'high_car': params[9],
            'high_MaaS_Bundle': params[10],
            'high_public': params[11]
        }
        
        sim_params = base_parameters.copy()
        sim_params['fps_value'] = fps_value
        sim_params['fixed_allocations'] = allocations
        sim_params['simulation_steps'] = SIMULATION_STEPS
        initial_param_sets.append(sim_params)
    
    # Evaluate initial points sequentially
    initial_results = []
    for params in initial_param_sets:
        result = run_single_simulation(params)
        initial_results.append(result)
    
    # Process initial results
    y_init = []
    for i, results in enumerate(initial_results):
        if results:
            # Use total equity indicator as optimization target (lower is better)
            equity_indicator = results.get('total_equity_indicator', float('inf'))
            y_init.append(equity_indicator)
            
            if equity_indicator < best_equity:
                best_equity = equity_indicator
                best_allocations = {
                    'low_bike': X_init[i][0],
                    'low_car': X_init[i][1],
                    'low_MaaS_Bundle': X_init[i][2],
                    'low_public': X_init[i][3],
                    'middle_bike': X_init[i][4],
                    'middle_car': X_init[i][5],
                    'middle_MaaS_Bundle': X_init[i][6],
                    'middle_public': X_init[i][7],
                    'high_bike': X_init[i][8],
                    'high_car': X_init[i][9],
                    'high_MaaS_Bundle': X_init[i][10],
                    'high_public': X_init[i][11]
                }
                best_results = results
                print(f"New best equity indicator from initial points: {best_equity:.6f}")
        else:
            y_init.append(0)

    # Setup Gaussian Process
    kernel = Matern(nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42)
    
    X = X_init
    y = np.array(y_init)
    
    # Run optimization iterations
    n_iterations = 15  # Additional iterations after initial points
    
    for iteration in range(n_iterations):
        # Fit GP to current data
        gp.fit(X, y)
        
        # Generate candidates for Expected Improvement
        candidates = sampler.random(n=500)  # Generate more candidates
        
        # Scale candidates to their specific bounds
        candidates = qmc.scale(candidates, lower_bounds, upper_bounds)
        
        # Apply total subsidy constraints to candidates
        for i in range(len(candidates)):
            # Get subsidies by income level
            low_subsidies = candidates[i, 0:4]
            middle_subsidies = candidates[i, 4:8]
            high_subsidies = candidates[i, 8:12]
            
            # Scale down if necessary while preserving proportions
            if np.sum(low_subsidies) > max_total:
                scale_factor = max_total / np.sum(low_subsidies)
                candidates[i, 0:4] = low_subsidies * scale_factor
                
            if np.sum(middle_subsidies) > max_total:
                scale_factor = max_total / np.sum(middle_subsidies)
                candidates[i, 4:8] = middle_subsidies * scale_factor
                
            if np.sum(high_subsidies) > max_total:
                scale_factor = max_total / np.sum(high_subsidies)
                candidates[i, 8:12] = high_subsidies * scale_factor
        
        # Calculate EI for all candidates (minimizing, not maximizing)
        ei_values = []
        for candidate in candidates:
            mu, sigma = gp.predict(candidate.reshape(1, -1), return_std=True)
            mu = mu.reshape(-1)
            sigma = sigma.reshape(-1)
            
            imp = np.min(y) - mu  # Changed for minimization
            Z = imp / (sigma + 1e-9)
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei_values.append(float(ei))
        
        # Select best candidate
        best_idx = np.argmax(ei_values)
        next_point = candidates[best_idx]
        
        # Prepare and run simulation with this candidate
        allocations = {
            'low_bike': next_point[0],
            'low_car': next_point[1],
            'low_MaaS_Bundle': next_point[2],
            'low_public': next_point[3],
            'middle_bike': next_point[4],
            'middle_car': next_point[5],
            'middle_MaaS_Bundle': next_point[6],
            'middle_public': next_point[7],
            'high_bike': next_point[8],
            'high_car': next_point[9],
            'high_MaaS_Bundle': next_point[10],
            'high_public': next_point[11]
        }
        
        sim_params = base_parameters.copy()
        sim_params['fps_value'] = fps_value
        sim_params['fixed_allocations'] = allocations
        sim_params['simulation_steps'] = SIMULATION_STEPS
        
        results = run_single_simulation(sim_params)
        
        # Process result
        if results:
            equity_indicator = results.get('total_equity_indicator', float('inf'))
            
            # Update data
            X = np.vstack([X, next_point])
            y = np.append(y, equity_indicator)
            
            if equity_indicator < best_equity:  # Changed from > to < for minimization
                best_equity = equity_indicator
                best_allocations = allocations
                best_results = results
                print(f"Iteration {iteration + 1}, New best equity indicator: {best_equity:.6f}")

    # Return results
    return {
        'fps_value': fps_value,
        'optimal_allocations': best_allocations,
        'equity_scores': {level: best_results[level]['equity_indicator'] for level in ['low', 'middle', 'high']},
        'avg_equity': best_equity,
        'full_results': best_results
    }

def visualize_results(results, output_dir='equity_optimization_results'):
    """
    Create visualizations from the sweep analysis results
    
    Args:
        results: Dictionary of results from optimization
        output_dir: Directory to save visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert results to DataFrame for easier visualization
    rows = []
    for fps, result in results.items():
        if 'optimal_allocations' in result and result['optimal_allocations']:
            row = {'fps_value': fps, 'avg_equity': result['avg_equity']}
            
            # Add equity scores for each income level
            for level in ['low', 'middle', 'high']:
                if 'equity_scores' in result and result['equity_scores']:
                    row[f'equity_{level}'] = result['equity_scores'][level]
            
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
    
    data_table.to_csv(os.path.join(output_dir, 'equity_allocation_table.csv'), index=False)
    
    # 1. Create main visualization with all income levels and allocation references
    plt.figure(figsize=(15, 10))
    
    # Plot lines for each income level
    colors = {'low': '#1f77b4', 'middle': '#ff7f0e', 'high': '#2ca02c', 'avg': '#d62728'}
    markers = {'low': 'o', 'middle': 's', 'high': '^', 'avg': 'D'}
    
    # Plot income-specific equity values
    for level in ['low', 'middle', 'high']:
        plt.plot(df['fps_value'], df[f'equity_{level}'], 
                 marker=markers[level], linestyle='-', 
                 color=colors[level], 
                 label=f'{level.capitalize()} Income', 
                 alpha=0.8)
    
    # Plot average equity with bolder line
    plt.plot(df['fps_value'], df['avg_equity'], 
             marker=markers['avg'], linestyle='-', 
             color=colors['avg'],
             linewidth=3, 
             label='Sum Equity', 
             alpha=0.9)
    
    # Add trend line using polynomial fit
    if len(df) > 2:
        try:
            z = np.polyfit(df['fps_value'], df['avg_equity'], 3)
            p = np.poly1d(z)
            x_smooth = np.linspace(df['fps_value'].min(), df['fps_value'].max(), 100)
            y_smooth = p(x_smooth)
            plt.plot(x_smooth, y_smooth, '--', color='gray', alpha=0.6, label='Trend Line')
            
            # Find and mark optimal point
            optimal_idx = np.argmin(y_smooth)
            optimal_fps = x_smooth[optimal_idx]
            optimal_equity = y_smooth[optimal_idx]
            
            plt.scatter([optimal_fps], [optimal_equity], 
                        marker='*', s=300, color='red', zorder=10,
                        label=f'Optimal Point (FPS={optimal_fps:.0f})')
            
            # Annotate optimal point
            plt.annotate(f'Optimal: Equity={optimal_equity:.4f}',
                        xy=(optimal_fps, optimal_equity),
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
            avg_equity = row['avg_equity']
            
            # Create a simplified allocation text
            alloc_text = f"FPS {fps}\n"
            for level in ['low', 'middle', 'high']:
                mode_values = []
                for mode in ['bike', 'car', 'MaaS_Bundle', 'public']:
                    col = f'alloc_{level}_{mode}'
                    if col in row:
                        mode_values.append(f"{mode.split('_')[0]}: {row[col]:.2f}")
                alloc_text += f"{level}: {', '.join(mode_values)}\n"
            
            # Add annotation with arrow pointing to point
            plt.annotate(f'Point {i+1}',  # Use numbers to reference the table
                        xy=(fps, avg_equity),
                        xytext=(10, 10 + (i % 3) * 20),  # Stagger annotations
                        textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'),
                        fontsize=8)
    
    # Add table reference
    plt.figtext(0.02, 0.02, 
               "Note: See 'equity_allocation_table.csv' for the complete allocations table",
               bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
    
    # Improve graph appearance
    plt.xscale('log')  # Log scale for FPS values
    plt.title('Optimal Equity Scores vs Fixed Pool Subsidy (FPS) Values', fontsize=16)
    plt.xlabel('Fixed Pool Subsidy (FPS)', fontsize=14)
    plt.ylabel('Equity Score (Higher is Better)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='best')
    
    # Add vertical line at the FPS value with highest equity score
    if not df.empty:
        best_fps = df.loc[df['avg_equity'].idxmax()]['fps_value']
        plt.axvline(x=best_fps, color='red', linestyle='--', alpha=0.5,
                   label=f'Best FPS: {best_fps}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'equity_vs_fps_with_allocations.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Create detailed allocation line graphs for each income level
    income_levels = ['low', 'middle', 'high']
    modes = ['bike', 'car', 'MaaS_Bundle', 'public']
    
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
    
    # 3. Create visualization of mode shares and equity
    # Extract mode share details for visualization
    try:
        mode_data = []
        for fps, result in results.items():
            if 'full_results' in result:
                full_results = result['full_results']
                for income_level in ['low', 'middle', 'high']:
                    if income_level in full_results:
                        mode_shares = full_results[income_level].get('mode_shares', {})
                        for mode, share in mode_shares.items():
                            mode_data.append({
                                'fps_value': fps,
                                'income_level': income_level,
                                'mode': mode,
                                'mode_share': share,
                                'equity_score': full_results[income_level].get('equity_score', 0)
                            })
        
        if mode_data:
            mode_df = pd.DataFrame(mode_data)
            
            # Create mode share visualization
            plt.figure(figsize=(15, 10))
            
            # Create a grid of subplots by income level and mode
            income_levels = mode_df['income_level'].unique()
            modes = mode_df['mode'].unique()
            
            # Calculate number of rows and columns for subplot grid
            n_rows = len(income_levels)
            n_cols = len(modes)
            
            # Create a subplot for each income-mode combination
            for i, income in enumerate(income_levels):
                for j, mode in enumerate(modes):
                    ax = plt.subplot(n_rows, n_cols, i * n_cols + j + 1)
                    
                    # Filter data for this income-mode combination
                    data = mode_df[(mode_df['income_level'] == income) & (mode_df['mode'] == mode)]
                    
                    if not data.empty:
                        # Sort by FPS value
                        data = data.sort_values('fps_value')
                        
                        # Plot mode share vs FPS
                        ax.plot(data['fps_value'], data['mode_share'], 'o-', 
                               label=f'{mode} Share', color='blue')
                        
                        # Set labels and title
                        ax.set_xlabel('FPS Value')
                        ax.set_ylabel('Mode Share')
                        ax.set_title(f'{income.capitalize()} - {mode}')
                        ax.set_xscale('log')
                        ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'equity_scores_by_income.png'))
            plt.close()
            
            # 4. Create heatmap showing squared differences for optimal solution
            best_fps = df.loc[df['avg_equity'].idxmax()]['fps_value']
            best_result = results[best_fps]
            
            if 'full_results' in best_result:
                full_results = best_result['full_results']
                
                # Extract mode details
                diff_data = []
                for income in ['low', 'middle', 'high']:
                    if income in full_results and 'mode_details' in full_results[income]:
                        for detail in full_results[income]['mode_details']:
                            diff_data.append({
                                'income_level': income,
                                'mode': detail['mode'],
                                'mode_share': detail['mode_share'],
                                'avg_share': detail['avg_share'],
                                'squared_diff': detail['difference']  # Using the new key name
                            })
                
                if diff_data:
                    diff_df = pd.DataFrame(diff_data)
                    
                    # Pivot data for heatmap
                    pivot_diff = diff_df.pivot_table(
                        values='squared_diff',
                        index='income_level',
                        columns='mode'
                    )
                    
                    # Create heatmap
                    plt.figure(figsize=(12, 8))
                    sns.heatmap(pivot_diff, annot=True, cmap='YlOrRd', fmt=".4f")
                    plt.title(f'Squared Differences for Optimal Solution (FPS={best_fps})')
                    plt.ylabel('Income Level')
                    plt.xlabel('Mode')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'mode_shares_radar.png'))
                    plt.close()
    
    except Exception as e:
        print(f"Error creating mode share visualizations: {e}")
        traceback.print_exc()
        
    # Find and save the optimal solution
    if 'avg_equity' in df.columns:
        optimal_row = df.loc[df['avg_equity'].idxmax()]
        
        with open(os.path.join(output_dir, 'optimal_solution.txt'), 'w') as f:
            f.write(f"Optimal FPS Value: {optimal_row['fps_value']}\n")
            f.write(f"Optimal Sum Equity Score: {optimal_row['avg_equity']:.4f}\n\n")
            f.write("Optimal Subsidy Allocations:\n")
            
            for key in sorted([k for k in optimal_row.index if k.startswith('alloc_')]):
                f.write(f"  {key.replace('alloc_', '')}: {optimal_row[key]:.4f}\n")
                
            f.write("\nEquity Score by Income Level:\n")
            for level in ['low', 'middle', 'high']:
                f.write(f"  {level.capitalize()}: {optimal_row[f'equity_{level}']:.4f}\n")

def main():
    """Main function to run the complete equity optimization analysis"""
    # Define base parameters
    base_parameters = {
        'num_commuters': 120,
        'grid_width': 65,
        'grid_height': 65,
        'data_income_weights': [0.5, 0.3, 0.2],
        'data_health_weights': [0.9, 0.1],
        'data_payment_weights': [0.8, 0.2],
        'data_age_distribution': {(18, 25): 0.2, (26, 35): 0.3, (36, 45): 0.2, 
                                (46, 55): 0.15, (56, 65): 0.1, (66, 75): 0.05},
        'data_disability_weights': [0.2, 0.8],
        'data_tech_access_weights': [0.95, 0.05],
        'ASC_VALUES': {'car': 0, 'bike': 0, 'public': 0, 
                      'walk': 0, 'maas': 0, 'default': 0},
        'UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS': {
            'beta_C': -0.02,
            'beta_T': -0.09
        },
        'UTILITY_FUNCTION_BASE_COEFFICIENTS': {
            'beta_C': -0.15, 'beta_T': -0.09, 
            'beta_W': -0.04, 'beta_A': -0.04, 'alpha': -0.01
        },
        'PENALTY_COEFFICIENTS': {
            'disability_bike_walk': 0.8,
            'age_health_bike_walk': 0.3,
            'no_tech_access_car_bike': 0.1
        },
        'AFFORDABILITY_THRESHOLDS': {'low': 65, 'middle': 110, 'high': 300},
        'FLEXIBILITY_ADJUSTMENTS': {'low': 1.15, 'medium': 1.0, 'high': 0.85},
        'VALUE_OF_TIME': {'low': 15, 'middle': 24, 'high': 95},
        'public_price_table': {
            'train': {'on_peak': 3, 'off_peak': 2.6},
            'bus': {'on_peak': 2, 'off_peak': 1.8}
        },
        'ALPHA_VALUES': {
            'UberLike1': 0.3,
            'UberLike2': 0.3,
            'BikeShare1': 0.25,
            'BikeShare2': 0.25
        },
        'DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS': {
            'S_base': 0.03,# Base surcharge (10%)
            'alpha': 0.15,# Sensitivity coefficient
            'delta': 0.5 # Reduction factor for subscription model
        },
        'CHANCE_FOR_INSERTING_RANDOM_TRAFFIC': 0.25,
        'BACKGROUND_TRAFFIC_AMOUNT': 70,
        'CONGESTION_ALPHA': 0.03,
        'CONGESTION_BETA': 1.5,  
        'CONGESTION_CAPACITY': 10,
        'CONGESTION_T_IJ_FREE_FLOW': 1.5,
        'uber_like1_capacity': 11,
        'uber_like1_price': 13,
        'uber_like2_capacity': 9,
        'uber_like2_price': 15,
        'bike_share1_capacity': 8,
        'bike_share1_price': 4,
        'bike_share2_capacity': 12,
        'bike_share2_price': 6 
    }
    
    # Define FPS values to analyze (using a logarithmic scale for better coverage)
    # Start with a smaller set of values for testing
    fps_values = [100, 1000, 2000, 5000, 10000, 20000, 30000, 50000]

    # For actual analysis, use a more comprehensive set
    # fps_values = np.logspace(0, 5, 10).astype(int)  # 10 points from 1 to 100,000 on log scale
    
    try:
        # Run sweep analysis with top-level parallelism
        print(f"Starting Equity Optimization with {len(fps_values)} FPS values")
        print(f"FPS values: {fps_values}")
        
        results = {}
        # Create a pool of workers for top-level parallelism
        with mp.Pool(processes=NUM_CPUS) as pool:
            # Use partial to create a function with preset base_parameters
            optimizer_function = partial(optimize_allocation_sequential, base_parameters=base_parameters)
            fps_results = pool.map(optimizer_function, fps_values)
            
            # Combine results into a dictionary
            for fps, result in zip(fps_values, fps_results):
                results[fps] = result
                
                # Save intermediate results to avoid losing progress
                with open(f'equity_optimization_intermediate_{fps}.pkl', 'wb') as f:
                    pickle.dump(result, f)
                    
                print(f"Completed analysis for FPS = {fps}")
                print(f"Optimal Sum Equity Score: {result.get('avg_equity', 'N/A')}")
        
        # Save full results
        with open('equity_optimization_results.pkl', 'wb') as f:
            pickle.dump(results, f)
            
        # Create visualizations
        visualize_results(results)
        
        # Find optimal FPS and allocations
        optimal_fps = None
        max_equity = 0
        
        for fps, result in results.items():
            if 'avg_equity' in result and result['avg_equity'] > max_equity:
                max_equity = result['avg_equity']
                optimal_fps = fps
                
        if optimal_fps:
            print(f"\nOptimal FPS Value: {optimal_fps}")
            print(f"Optimal Sum Equity Score: {max_equity:.4f}")
            
            # Print detailed allocation percentages
            optimal_allocations = results[optimal_fps]['optimal_allocations']
            print("\nOptimal Subsidy Allocations:")
            for key, value in sorted(optimal_allocations.items()):
                print(f"  {key}: {value:.4f}")
            
            print("\nEquity Scores by Income Level:")
            for level in ['low', 'middle', 'high']:
                print(f"  {level.capitalize()}: {results[optimal_fps]['equity_scores'][level]:.4f}")
            
        print("\nEquity optimization analysis complete.")
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()