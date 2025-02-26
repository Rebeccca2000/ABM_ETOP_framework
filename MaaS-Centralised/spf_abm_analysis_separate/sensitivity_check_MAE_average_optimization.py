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
# Import functions from sensitivity_check_SDI
from sensitivity_check_SDI import calculate_sur, calculate_mae, calculate_upi, calculate_sdi, calculate_sdi_metrics

# Constants
SIMULATION_STEPS = 144 # Reduced for faster runs during optimization
NUM_CPUS = 8  # Adjust based on your system

def run_single_simulation(params):
    """Run a single simulation and calculate MAE metrics"""
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
                for mode in ['bike', 'car', 'MaaS_Bundle', 'public','walk']:
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
        
        # Calculate metrics
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

def optimize_allocation_sequential(fps_value, base_parameters):
    """
    Optimize subsidy allocation for a given FPS value using sequential processing
    with income-specific boundary constraints for more realistic optimization.
    """
    print(f"\nOptimizing subsidy allocation for FPS = {fps_value}")
    
    # Define income-specific bounds for better policy control
    bounds = {
        # Low income - increase car and MaaS upper bounds significantly
        'low_bike': (0.20, 0.50),      # Increased lower bound to ensure basic bike access
        'low_car': (0.35, 0.80),       # Much higher bounds to make car truly accessible
        'low_MaaS_Bundle': (0.25, 0.75),  # Higher bounds to encourage MaaS adoption
        'low_public': (0.15, 0.50),    # Reduced upper bound to avoid over-dependence
        
        # Middle income - adjust for balanced approach
        'middle_bike': (0.15, 0.40),    # Slightly higher bounds for sustainability
        'middle_car': (0.10, 0.35),     # Moderate subsidy needs
        'middle_MaaS_Bundle': (0.15, 0.45),  # Increased to encourage MaaS adoption
        'middle_public': (0.10, 0.35),   # Reduced to avoid over-reliance
        
        # High income - lower subsidies overall as less subsidy-dependent
        'high_bike': (0.05, 0.50),
        'high_car': (0.05, 0.10),      
        'high_MaaS_Bundle': (0.15, 0.40),
        'high_public': (0.05, 0.30)
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
    
    best_mae = 0
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
            # mae_values = [results[level]['mae'] for level in ['low', 'middle', 'high']]
            # avg_mae = np.mean(mae_values)
            weights = {'low': 0.5, 'middle': 0.35, 'high': 0.15}
            mae_values = {level: results[level]['mae'] for level in ['low', 'middle', 'high']}
            avg_mae = sum(mae_values[level] * weights[level] for level in weights.keys())
            y_init.append(avg_mae)
            
            if avg_mae > best_mae:
                best_mae = avg_mae
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
                print(f"New best MAE from initial points: {best_mae:.4f}")
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
        
        # Calculate EI for all candidates
        ei_values = []
        for candidate in candidates:
            mu, sigma = gp.predict(candidate.reshape(1, -1), return_std=True)
            mu = mu.reshape(-1)
            sigma = sigma.reshape(-1)
            
            imp = mu - np.max(y)
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
            mae_values = [results[level]['mae'] for level in ['low', 'middle', 'high']]
            avg_mae = np.mean(mae_values)
            
            # Update data
            X = np.vstack([X, next_point])
            y = np.append(y, avg_mae)
            
            if avg_mae > best_mae:
                best_mae = avg_mae
                best_allocations = allocations
                best_results = results
                print(f"Iteration {iteration + 1}, New best MAE: {best_mae:.4f}")

    # Return results in same format as original function
    return {
        'fps_value': fps_value,
        'optimal_allocations': best_allocations,
        'mae_scores': {level: best_results[level]['mae'] for level in ['low', 'middle', 'high']},
        'avg_mae': best_mae,
        'full_results': best_results
    }
# Add this inside your create_visualizations function or create_pbs_visualizations function
def add_mode_share_visualization(results, output_dir):
    """Add visualizations showing mode share metrics and adherence score relationships"""
    # Extract FPS values and mode details
    data_rows = []
    
    for fps, result in results.items():
        # Check if the result has 'optimal_allocations' and 'full_results'
        if isinstance(result, dict) and 'full_results' in result:
            full_results = result['full_results']
            
            # Process each income level
            for income in ['low', 'middle', 'high']:
                if income in full_results and 'mode_details' in full_results[income]:
                    for mode_data in full_results[income]['mode_details']:
                        data_rows.append({
                            'fps_value': fps,
                            'income_level': income,
                            'mode': mode_data['mode'],
                            'target_share': mode_data['target_share'],
                            'actual_share': mode_data['actual_share'],
                            'share_ratio': mode_data.get('share_ratio', 0),
                            'adherence_score': mode_data.get('adherence_score', 0)
                        })
    
    if not data_rows:
        print("No mode share details available for visualization")
        return
        
    mode_df = pd.DataFrame(data_rows)
    
    # Create two visualizations:
    # 1. Mode shares comparison (actual vs target)
    # 2. Relationship between share_ratio and adherence_score
    
    # 1. Mode shares by income and mode
    plt.figure(figsize=(15, 15))
    income_levels = mode_df['income_level'].unique()
    
    if 'mode' in mode_df.columns:
        modes = mode_df['mode'].unique()
    else:
        print("No mode data found in results")
        return
    
    plot_idx = 1
    for income in income_levels:
        for mode in modes:
            plt.subplot(len(income_levels), len(modes), plot_idx)
            plot_idx += 1
            
            filtered_df = mode_df[(mode_df['income_level'] == income) & 
                                 (mode_df['mode'] == mode)]
            
            if not filtered_df.empty:
                # Sort by FPS value
                filtered_df = filtered_df.sort_values('fps_value')
                
                # Plot actual share
                plt.plot(filtered_df['fps_value'], filtered_df['actual_share'], 
                         'bo-', label='Actual')
                
                # Plot target share as horizontal line if available
                if 'target_share' in filtered_df.columns and len(filtered_df) > 0:
                    plt.axhline(y=filtered_df['target_share'].iloc[0], 
                                color='r', linestyle='--', label='Target')
                
                plt.title(f"{income.capitalize()} - {mode}")
                plt.xscale('log')
                plt.ylim(0, max(filtered_df['actual_share'].max() * 1.2, 0.7))
                
                # Add labels
                if income == income_levels[-1]:
                    plt.xlabel('FPS Value')
                if mode == modes[0]:
                    plt.ylabel('Mode Share')
                    
                plt.legend(loc='best', fontsize='small')
                plt.grid(True, alpha=0.3)
    
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(output_dir, 'mode_shares_vs_targets.png'))
    plt.close()
    
    # 2. Create the NEW visualization: share_ratio vs adherence_score by income level and mode
    plt.figure(figsize=(18, 12))
    
    income_colors = {'low': 'blue', 'middle': 'green', 'high': 'red'}
    markers = {'MaaS_Bundle': 'o', 'public': 's', 'bike': '^', 'car': 'D', 'walk':'*'}
    
    # First subplot: Share Ratio vs Adherence Score
    plt.subplot(1, 2, 1)
    
    # Create a scatter plot with different colors for income levels and markers for modes
    for income in income_levels:
        for mode in modes:
            filtered_df = mode_df[(mode_df['income_level'] == income) & 
                                 (mode_df['mode'] == mode)]
            
            if not filtered_df.empty and 'share_ratio' in filtered_df.columns:
                plt.scatter(filtered_df['share_ratio'], 
                           filtered_df['adherence_score'],
                           marker=markers.get(mode, 'o'),
                           color=income_colors.get(income, 'black'),
                           alpha=0.7,
                           label=f"{income.capitalize()} - {mode}")
    
    # Add the exponential curve showing the adherence score formula
    x_formula = np.linspace(0, 2, 100)  # Share ratios from 0 to 2
    # Use the actual formula from your MAE calculation
    y_formula = np.exp(-((x_formula - 1) ** 2) / 0.5)  # Using 0.5 in the denominator
    plt.plot(x_formula, y_formula, 'k--', linewidth=2, 
             label='Adherence Score Formula')
    
    plt.title('Share Ratio vs Adherence Score', fontsize=14)
    plt.xlabel('Share Ratio (Actual/Target)', fontsize=12)
    plt.ylabel('Adherence Score', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    plt.text(1.01, 0.02, "Perfect Match (Ratio=1)", rotation=90)
    plt.ylim(0, 1.05)
    
    # Add a box explaining how the adherence score works
    explanation_text = (
        "Adherence Score Formula:\n"
        "score = exp(-((ratio-1)²)/0.5)\n\n"
        "Perfect match (ratio=1): score=1.0\n"
        "50% of target (ratio=0.5): score≈0.5\n"
        "200% of target (ratio=2.0): score≈0.5"
    )
    plt.text(1.5, 0.2, explanation_text, 
             bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    
    # Second subplot: Target vs Actual Share by mode and income
    plt.subplot(1, 2, 2)
    
    for income in income_levels:
        for mode in modes:
            filtered_df = mode_df[(mode_df['income_level'] == income) & 
                                 (mode_df['mode'] == mode)]
            
            if not filtered_df.empty:
                plt.scatter(filtered_df['target_share'], 
                           filtered_df['actual_share'],
                           marker=markers.get(mode, 'o'),
                           color=income_colors.get(income, 'black'),
                           alpha=0.7,
                           label=f"{income.capitalize()} - {mode}")
    
    # Add a diagonal line representing perfect match
    max_value = max(mode_df['target_share'].max(), mode_df['actual_share'].max())
    plt.plot([0, max_value], [0, max_value], 'k--', alpha=0.5, 
             label='Perfect Match Line')
    
    plt.title('Target Share vs Actual Share', fontsize=14)
    plt.xlabel('Target Share', fontsize=12)
    plt.ylabel('Actual Share', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Combine legends and place it outside the plots
    handles, labels = plt.gca().get_legend_handles_labels()
    # Remove duplicates while preserving order
    by_label = dict(zip(labels, handles))
    plt.figlegend(by_label.values(), by_label.keys(), 
                 loc='lower center', ncol=4, fontsize=10)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95], pad=2.0)
    plt.savefig(os.path.join(output_dir, 'adherence_score_analysis.png'))
    plt.close()
    
    # 3. Create a third visualization showing the relationship between
    # FPS values, share ratios, and adherence scores for each income-mode combination
    plt.figure(figsize=(15, 15))
    
    plot_idx = 1
    for income in income_levels:
        for mode in modes:
            plt.subplot(len(income_levels), len(modes), plot_idx)
            plot_idx += 1
            
            filtered_df = mode_df[(mode_df['income_level'] == income) & 
                                 (mode_df['mode'] == mode)]
            
            if not filtered_df.empty and len(filtered_df) > 0:
                # Sort by FPS value
                filtered_df = filtered_df.sort_values('fps_value')
                
                # Primary Y-axis for share_ratio
                color = 'blue'
                ax1 = plt.gca()
                ax1.set_xlabel('FPS Value')
                ax1.set_ylabel('Share Ratio', color=color)
                ax1.plot(filtered_df['fps_value'], filtered_df['share_ratio'], 
                         'o-', color=color, label='Share Ratio')
                ax1.tick_params(axis='y', labelcolor=color)
                ax1.set_xscale('log')
                
                # Add horizontal line at ratio=1 (perfect match)
                ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
                
                # Secondary Y-axis for adherence_score
                color = 'red'
                ax2 = ax1.twinx()
                ax2.set_ylabel('Adherence Score', color=color)
                ax2.plot(filtered_df['fps_value'], filtered_df['adherence_score'], 
                         's-', color=color, label='Adherence Score')
                ax2.tick_params(axis='y', labelcolor=color)
                ax2.set_ylim(0, 1.05)
                
                # Title
                plt.title(f"{income.capitalize()} - {mode}")
                
                # Legend
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize='small')
    
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(output_dir, 'fps_ratio_adherence_analysis.png'))
    plt.close()

def visualize_results(results, output_dir='fps_pbs_analysis_mae_results'):
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
            row = {'fps_value': fps, 'avg_mae': result['avg_mae']}
            
            # Add MAE scores for each income level
            for level in ['low', 'middle', 'high']:
                if 'mae_scores' in result and result['mae_scores']:
                    row[f'mae_{level}'] = result['mae_scores'][level]
            
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
    
    # Plot income-specific MAE values
    for level in ['low', 'middle', 'high']:
        plt.plot(df['fps_value'], df[f'mae_{level}'], 
                 marker=markers[level], linestyle='-', 
                 color=colors[level], 
                 label=f'{level.capitalize()} Income', 
                 alpha=0.8)
    
    # Plot average MAE with bolder line
    plt.plot(df['fps_value'], df['avg_mae'], 
             marker=markers['avg'], linestyle='-', 
             color=colors['avg'],
             linewidth=3, 
             label='Average MAE', 
             alpha=0.9)
    
    # Add trend line using polynomial fit
    if len(df) > 2:
        try:
            z = np.polyfit(df['fps_value'], df['avg_mae'], 3)
            p = np.poly1d(z)
            x_smooth = np.linspace(df['fps_value'].min(), df['fps_value'].max(), 100)
            y_smooth = p(x_smooth)
            plt.plot(x_smooth, y_smooth, '--', color='gray', alpha=0.6, label='Trend Line')
            
            # Find and mark optimal point
            optimal_idx = np.argmax(y_smooth)
            optimal_fps = x_smooth[optimal_idx]
            optimal_mae = y_smooth[optimal_idx]
            
            plt.scatter([optimal_fps], [optimal_mae], 
                        marker='*', s=300, color='red', zorder=10,
                        label=f'Optimal Point (FPS={optimal_fps:.0f})')
            
            # Annotate optimal point
            plt.annotate(f'Optimal: MAE={optimal_mae:.4f}',
                        xy=(optimal_fps, optimal_mae),
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
            avg_mae = row['avg_mae']
            
            # Create a simplified allocation text
            alloc_text = f"FPS {fps}\n"
            for level in ['low', 'middle', 'high']:
                mode_values = []
                for mode in ['bike', 'car', 'MaaS_Bundle', 'public','walk']:
                    col = f'alloc_{level}_{mode}'
                    if col in row:
                        mode_values.append(f"{mode.split('_')[0]}: {row[col]:.2f}")
                alloc_text += f"{level}: {', '.join(mode_values)}\n"
            
            # Add annotation with arrow pointing to point
            plt.annotate(f'Point {i+1}',  # Use numbers to reference the table
                        xy=(fps, avg_mae),
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
    plt.title('Optimal MAE Scores vs Fixed Pool Subsidy (FPS) Values', fontsize=16)
    plt.xlabel('Fixed Pool Subsidy (FPS)', fontsize=14)
    plt.ylabel('Modal Access Equity (MAE)', fontsize=14)
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
                f"{row['avg_mae']:.4f}"
            ]
            
            # Add allocations for each income level and mode
            for level in ['low', 'middle', 'high']:
                for mode in ['bike', 'car', 'MaaS_Bundle', 'public','walk']:
                    col = f'alloc_{level}_{mode}'
                    if col in row:
                        table_row.append(f"{row[col]:.2f}")
                    else:
                        table_row.append("N/A")
            
            table_rows.append(table_row)
        
        # Create table at the bottom of the figure
        table_data = np.array(table_rows)
        cols = ['FPS', 'Avg MAE'] + [
            f"{level} {mode.split('_')[0]}" 
            for level in ['low', 'middle', 'high'] 
            for mode in ['bike', 'car', 'MaaS_Bundle', 'public','walk']
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
    plt.savefig(os.path.join(output_dir, 'mae_vs_fps_with_allocations.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Create detailed allocation line graphs for each income level
    income_levels = ['low', 'middle', 'high']
    modes = ['bike', 'car', 'MaaS_Bundle','walk','public']
    
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
                      df[f'mae_{level}'],
                      label=f'{level.capitalize()} Income',
                      s=50)
        
        ax.set_xlabel('FPS Value')
        ax.set_ylabel('MaaS Allocation %')
        ax.set_zlabel('MAE Score')
        ax.set_xscale('log')
        ax.set_title('3D Relationship: FPS, MaaS Allocation, and MAE')
        ax.legend()
        
        plt.savefig(os.path.join(output_dir, '3d_relationship.png'))
        plt.close()
    except:
        print("Could not create 3D visualization (Matplotlib 3D plotting might not be available)")
    
    # Add this call to your visualization function
    add_mode_share_visualization(results, output_dir)

    # Find and save the optimal solution
    if 'avg_mae' in df.columns:
        optimal_row = df.loc[df['avg_mae'].idxmax()]
        
        with open(os.path.join(output_dir, 'optimal_solution.txt'), 'w') as f:
            f.write(f"Optimal FPS Value: {optimal_row['fps_value']}\n")
            f.write(f"Optimal Average MAE: {optimal_row['avg_mae']:.4f}\n\n")
            f.write("Optimal Subsidy Allocations:\n")
            
            for key in sorted([k for k in optimal_row.index if k.startswith('alloc_')]):
                f.write(f"  {key.replace('alloc_', '')}: {optimal_row[key]:.4f}\n")
                
            f.write("\nMAE by Income Level:\n")
            for level in ['low', 'middle', 'high']:
                f.write(f"  {level.capitalize()}: {optimal_row[f'mae_{level}']:.4f}\n")
                
def run_optimized_simulation(optimal_fps, optimal_allocations, base_parameters):
    """
    Run a simulation with the optimal FPS and allocations for comprehensive metrics
    """
    # Set up parameters with optimal values
    params = base_parameters.copy()
    params['fps_value'] = optimal_fps
    params['fixed_allocations'] = optimal_allocations
    params['simulation_steps'] = SIMULATION_STEPS # Run longer for better results
    
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
        'ASC_VALUES': {'car': 0, 'bike': 0, 'public': 0, 
                      'walk': 0, 'maas': 0, 'default': 0},
        'UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS': {
            'beta_C': -0.02,
            'beta_T': -0.09
        },
        'UTILITY_FUNCTION_BASE_COEFFICIENTS': {
            'beta_C': -0.1, 'beta_T': -0.11, 
            'beta_W': -0.06, 'beta_A': -0.04, 'alpha': -0.01
        },
        'PENALTY_COEFFICIENTS': {
            'disability_bike_walk': 0.8,
            'age_health_bike_walk': 0.3,
            'no_tech_access_car_bike': 0.1
        },
        'AFFORDABILITY_THRESHOLDS': {'low': 40, 'middle': 85, 'high': 245},
        'FLEXIBILITY_ADJUSTMENTS': {'low': 1.15, 'medium': 1.0, 'high': 0.85},
        'VALUE_OF_TIME': {'low': 5.5, 'middle': 15, 'high': 85},
        'public_price_table': {
            'train': {'on_peak': 2.3, 'off_peak': 1.9},
            'bus': {'on_peak': 1.2, 'off_peak': 1}
        },
        'ALPHA_VALUES': {
            'UberLike1': 0.3,
            'UberLike2': 0.3,
            'BikeShare1': 0.25,
            'BikeShare2': 0.25
        },
        'DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS': {
            'S_base': 0.03,
            'alpha': 0.15,
            'delta': 0.6
        },
        'BACKGROUND_TRAFFIC_AMOUNT': 55,
        'CONGESTION_ALPHA': 0.15,
        'CONGESTION_BETA': 2,
        'CONGESTION_CAPACITY': 15,
        'CONGESTION_T_IJ_FREE_FLOW': 1.5,
        'uber_like1_capacity': 10,
        'uber_like1_price': 12,
        'uber_like2_capacity': 9,
        'uber_like2_price': 10,
        'bike_share1_capacity': 8,
        'bike_share1_price': 4,
        'bike_share2_capacity': 12,
        'bike_share2_price': 6 
    }
    
    # Define FPS values to analyze (use a logarithmic scale for better coverage)
    # For initial testing, use a smaller set of values
    fps_values = [1000, 4000, 8000, 10000, 25000, 50000, 80000, 100000]

    # For actual analysis, use a more comprehensive set
    # fps_values = np.logspace(0, 5, 10).astype(int)  # 10 points from 1 to 100,000 on log scale
    
    try:
        # Run sweep analysis with top-level parallelism
        print(f"Starting FPS-PBS sweep analysis with {len(fps_values)} FPS values")
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
                with open(f'fps_pbs_sweep_intermediate_{fps}.pkl', 'wb') as f:
                    pickle.dump(result, f)
                    
                print(f"Completed analysis for FPS = {fps}")
                print(f"Optimal Average MAE: {result.get('avg_mae', 'N/A')}")
        
        # Save full results
        with open('fps_pbs_mae_sweep_results.pkl', 'wb') as f:
            pickle.dump(results, f)
            
        # Create visualizations
        visualize_results(results)
        
        # Find optimal FPS and allocations
        optimal_fps = None
        max_mae = 0
        
        for fps, result in results.items():
            if 'avg_mae' in result and result['avg_mae'] > max_mae:
                max_mae = result['avg_mae']
                optimal_fps = fps
                
        if optimal_fps:
            print(f"\nOptimal FPS Value: {optimal_fps}")
            print(f"Optimal Average MAE: {max_mae:.4f}")
            
            # Run final simulation with optimal values
            optimal_allocations = results[optimal_fps]['optimal_allocations']
            #optimal_results = run_optimized_simulation(optimal_fps, optimal_allocations, base_parameters)
            
            print("\nDetailed results for optimal solution saved.")
            
        print("\nAnalysis complete.")
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()