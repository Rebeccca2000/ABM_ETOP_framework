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
    Optimize subsidy allocation for a given FPS value focusing ONLY on low-income users.
    Modified to prioritize MAE scores for low-income population only.
    """
    print(f"\nOptimizing subsidy allocation for FPS = {fps_value} - Focus: LOW INCOME")
    
    # Define income-specific bounds, with expanded ranges for low-income
    bounds = {
        # Low income - expanded ranges for optimization focus
        'low_bike': (0.10, 0.60),      # Increased upper bound
        'low_car': (0.10, 0.70),       # Increased upper bound
        'low_MaaS_Bundle': (0.10, 0.60), # Increased upper bound
        'low_public': (0.15, 0.80),    # Increased upper bound
        
        # Middle and high income are still considered but with narrower ranges
        'middle_bike': (0.05, 0.30),    # Reduced range
        'middle_car': (0.05, 0.30),     # Reduced range
        'middle_MaaS_Bundle': (0.05, 0.30), # Reduced range
        'middle_public': (0.10, 0.35),  # Reduced range
        
        'high_bike': (0.05, 0.20),      # Reduced range
        'high_car': (0.05, 0.20),       # Reduced range
        'high_MaaS_Bundle': (0.05, 0.20), # Reduced range
        'high_public': (0.05, 0.20)     # Reduced range
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
    
    best_mae = 0  # Best low-income MAE score
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
    
    # Process initial results - FOCUS ONLY ON LOW INCOME
    y_init = []
    for i, results in enumerate(initial_results):
        if results:
            # Extract only the low income MAE score
            low_income_mae = results['low']['mae']
            y_init.append(low_income_mae)
            
            # Track best result based only on low income MAE
            if low_income_mae > best_mae:
                best_mae = low_income_mae
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
                print(f"New best LOW INCOME MAE from initial points: {best_mae:.4f}")
        else:
            y_init.append(0)

    # Setup Gaussian Process for optimization
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
        
        # Process result - FOCUS ONLY ON LOW INCOME
        if results:
            # Extract only low income MAE score
            low_income_mae = results['low']['mae']
            
            # Update data
            X = np.vstack([X, next_point])
            y = np.append(y, low_income_mae)
            
            if low_income_mae > best_mae:
                best_mae = low_income_mae
                best_allocations = allocations
                best_results = results
                print(f"Iteration {iteration + 1}, New best LOW INCOME MAE: {best_mae:.4f}")

    # Return results with low income focus
    return {
        'fps_value': fps_value,
        'optimal_allocations': best_allocations,
        'mae_scores': {level: best_results[level]['mae'] for level in ['low', 'middle', 'high']},
        'low_income_mae': best_mae,  # Explicitly track low income MAE 
        'full_results': best_results
    }


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
def visualize_results(results, output_dir='fps_pbs_analysis_low_income_mae_results'):
    """
    Create visualizations from the sweep analysis results
    focusing on low-income MAE optimization
    
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
            row = {'fps_value': fps, 'low_income_mae': result['low_income_mae']}
            
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
    
    # Save full data table
    data_table = df.copy()
    # Format allocation percentages as percentages
    for col in data_table.columns:
        if col.startswith('alloc_'):
            data_table[col] = data_table[col].map(lambda x: f"{x*100:.1f}%")
    
    data_table.to_csv(os.path.join(output_dir, 'fps_pbs_low_income_allocation_table.csv'), index=False)
    
    # 1. Create main visualization focused on low-income MAE
    plt.figure(figsize=(15, 10))
    
    # Plot lines for each income level with low income highlighted
    colors = {'low': '#1f77b4', 'middle': '#A0A0A0', 'high': '#A0A0A0'}
    line_styles = {'low': '-', 'middle': '--', 'high': '--'}
    line_widths = {'low': 3, 'middle': 1, 'high': 1}
    markers = {'low': 'o', 'middle': '.', 'high': '.'}
    
    # Plot income-specific MAE values
    for level in ['low', 'middle', 'high']:
        plt.plot(df['fps_value'], df[f'mae_{level}'], 
                 marker=markers[level], linestyle=line_styles[level], 
                 color=colors[level], 
                 linewidth=line_widths[level],
                 label=f'{level.capitalize()} Income', 
                 alpha=0.8)
    
    # Add trend line for low income using polynomial fit
    if len(df) > 2:
        try:
            z = np.polyfit(df['fps_value'], df['mae_low'], 3)
            p = np.poly1d(z)
            x_smooth = np.linspace(df['fps_value'].min(), df['fps_value'].max(), 100)
            y_smooth = p(x_smooth)
            plt.plot(x_smooth, y_smooth, '--', color='blue', alpha=0.6, label='Low Income Trend')
            
            # Find and mark optimal point
            optimal_idx = np.argmax(y_smooth)
            optimal_fps = x_smooth[optimal_idx]
            optimal_mae = y_smooth[optimal_idx]
            
            plt.scatter([optimal_fps], [optimal_mae], 
                        marker='*', s=300, color='blue', zorder=10,
                        label=f'Optimal Point (FPS={optimal_fps:.0f})')
            
            # Annotate optimal point
            plt.annotate(f'Optimal Low Income: MAE={optimal_mae:.4f}',
                        xy=(optimal_fps, optimal_mae),
                        xytext=(10, -30), 
                        textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))
        except Exception as e:
            print(f"Error fitting trend line: {e}")
    
    # Improve graph appearance
    plt.xscale('log')  # Log scale for FPS values
    plt.title('Low-Income MAE Optimization vs Fixed Pool Subsidy (FPS) Values', fontsize=16)
    plt.xlabel('Fixed Pool Subsidy (FPS)', fontsize=14)
    plt.ylabel('Modal Access Equity (MAE)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='best')
    
    # Add note about optimization focus
    plt.figtext(0.02, 0.02, 
               "Note: Optimization focused specifically on maximizing low-income MAE scores",
               bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'low_income_mae_vs_fps.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Create visualization of low-income allocation proportions
    plt.figure(figsize=(12, 8))
    
    # Extract only low-income allocations
    low_income_columns = [col for col in df.columns if col.startswith('alloc_low_')]
    modes = [col.replace('alloc_low_', '') for col in low_income_columns]
    
    # Plot low-income allocation proportions
    for col, mode in zip(low_income_columns, modes):
        plt.plot(df['fps_value'], df[col], 'o-', 
                 linewidth=2, label=mode)
    
    plt.title('Low-Income Subsidy Allocation Proportions', fontsize=16)
    plt.xlabel('Fixed Pool Subsidy (FPS)', fontsize=14)
    plt.ylabel('Allocation Proportion', fontsize=14)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='best')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'low_income_allocation_proportions.png'),
               dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Create a stacked area chart showing how low-income allocations change with FPS
    plt.figure(figsize=(12, 8))
    
    # Create stacked area chart
    df_sorted = df.sort_values('fps_value')
    
    # Get data for stacked area chart
    x = df_sorted['fps_value']
    y_data = [df_sorted[col] for col in low_income_columns]
    
    # Create stacked area chart
    plt.stackplot(x, y_data, labels=modes, alpha=0.7)
    
    plt.title('Low-Income Subsidy Allocation Distribution', fontsize=16)
    plt.xlabel('Fixed Pool Subsidy (FPS)', fontsize=14)
    plt.ylabel('Allocation Proportion', fontsize=14)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='best')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'low_income_allocation_stacked.png'),
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Create scatter plot of low-income MAE vs subsidy proportions
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, mode in enumerate(modes):
        if i < len(axes):
            col = f'alloc_low_{mode}'
            axes[i].scatter(df[col], df['mae_low'], s=50, alpha=0.7)
            axes[i].set_title(f'Low-Income MAE vs {mode} Allocation', fontsize=14)
            axes[i].set_xlabel(f'{mode} Allocation Proportion', fontsize=12)
            axes[i].set_ylabel('Low-Income MAE', fontsize=12)
            axes[i].grid(True, alpha=0.3)
            
            # Add trend line
            if len(df) > 2:
                try:
                    z = np.polyfit(df[col], df['mae_low'], 2)
                    p = np.poly1d(z)
                    x_smooth = np.linspace(df[col].min(), df[col].max(), 100)
                    y_smooth = p(x_smooth)
                    axes[i].plot(x_smooth, y_smooth, '--', color='red', alpha=0.6)
                except Exception as e:
                    print(f"Error fitting trend line for {mode}: {e}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'low_income_mae_vs_allocations.png'),
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Create Line Plot: Mode Share vs FPS for Low Income
    plt.figure(figsize=(12, 8))

    # Extract mode share data for low income from full results
    mode_share_data = []
    for fps, result in results.items():
        if 'full_results' in result and result['full_results']:
            full_results = result['full_results']
            
            if 'low' in full_results and 'mode_details' in full_results['low']:
                for mode_data in full_results['low']['mode_details']:
                    mode_share_data.append({
                        'fps_value': fps,
                        'mode': mode_data['mode'],
                        'actual_share': mode_data['actual_share'],
                        'target_share': mode_data['target_share']
                    })

    if mode_share_data:
        # Convert to DataFrame for easier manipulation
        mode_df = pd.DataFrame(mode_share_data)
        
        # Sort by FPS value for line plotting
        mode_df = mode_df.sort_values('fps_value')
        
        # Plot each mode
        for mode in sorted(mode_df['mode'].unique()):
            mode_data = mode_df[mode_df['mode'] == mode]
            plt.plot(mode_data['fps_value'], mode_data['actual_share'], 
                    'o-', linewidth=2, label=f'{mode}')
        
        plt.title('Low-Income Mode Share vs Fixed Pool Subsidy', fontsize=16)
        plt.xlabel('Fixed Pool Subsidy (FPS)', fontsize=14)
        plt.ylabel('Mode Share', fontsize=14)
        plt.xscale('log')  # Log scale for better visualization
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12, loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'low_income_mode_share_vs_fps.png'), 
                dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Create Small Multiples Plot: Target vs Actual Share for Each Mode
        modes = sorted(mode_df['mode'].unique())
        n_modes = len(modes)
        
        # Calculate grid dimensions
        n_cols = min(3, n_modes)
        n_rows = (n_modes + n_cols - 1) // n_cols
        
        plt.figure(figsize=(5*n_cols, 4*n_rows))
        
        for i, mode in enumerate(modes):
            plt.subplot(n_rows, n_cols, i+1)
            
            mode_data = mode_df[mode_df['mode'] == mode]
            
            # Plot actual share
            plt.plot(mode_data['fps_value'], mode_data['actual_share'], 
                    'bo-', linewidth=2, label='Actual', alpha=0.8)
            
            # Plot target share as horizontal line
            if not mode_data.empty:
                target = mode_data['target_share'].iloc[0]
                plt.axhline(y=target, color='r', linestyle='--', label=f'Target: {target:.3f}')
            
            plt.title(f'Mode: {mode}')
            plt.xlabel('FPS Value')
            plt.ylabel('Mode Share')
            plt.xscale('log')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'low_income_mode_target_vs_actual.png'), 
                dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save the mode share data for reference
        mode_df.to_csv(os.path.join(output_dir, 'low_income_mode_share_data.csv'), index=False)
    else:
        print("Warning: No mode share data found in results")
    # Find and save the optimal solution
    optimal_row = df.loc[df['low_income_mae'].idxmax()]
    
    with open(os.path.join(output_dir, 'optimal_low_income_solution.txt'), 'w') as f:
        f.write(f"Optimal FPS Value for Low-Income: {optimal_row['fps_value']}\n")
        f.write(f"Optimal Low-Income MAE: {optimal_row['low_income_mae']:.4f}\n\n")
        f.write("Optimal Low-Income Subsidy Allocations:\n")
        
        for mode in modes:
            key = f'alloc_low_{mode}'
            f.write(f"  {mode}: {optimal_row[key]:.4f}\n")
            
        f.write("\nMAE by Income Level with this solution:\n")
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
        'AFFORDABILITY_THRESHOLDS': {'low': 45, 'middle': 85, 'high': 245},
        'FLEXIBILITY_ADJUSTMENTS': {'low': 1.15, 'medium': 1.0, 'high': 0.85},
        'VALUE_OF_TIME': {'low': 6.5, 'middle': 15, 'high': 85},
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
    fps_values = [100, 1000, 4000, 8000, 10000, 25000, 50000, 100000]

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