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
from datetime import datetime
import traceback
import math
from sqlalchemy import create_engine, func, text
from agent_subsidy_pool import SubsidyPoolConfig
from run_visualisation_03 import MobilityModel
from agent_service_provider_initialisation_03 import reset_database, CommuterInfoLog, ServiceBookingLog
from functools import partial
import time
import random

# Constants
SIMULATION_STEPS = 144  # Reduced for faster runs during optimization
NUM_CPUS = 15  # Adjust based on your system

def calculate_destination_potential(session, fps_value, schema=None):
    """
    Calculate how subsidies improve destination potential accessibility
    within a single simulation, corrected for ABM time units.
    """
    if schema:
        session.execute(text(f"SET search_path TO {schema}"))
        session.commit()
    
    # Get correct value of time from your model parameters
    # These should match the VALUES from your ABM initialization
    vot = {
        'low': 5,     # VALUE_OF_TIME['low'] in your model
        'middle': 10, # VALUE_OF_TIME['middle'] in your model
        'high': 20    # VALUE_OF_TIME['high'] in your model
    }
    
    # Correct impedance parameter calibrated for your model's scale
    # Based on Kwan (1998) but adjusted for your grid size and step definition
    beta = 0.05  # Smaller value appropriate for your model's cost scale
    
    # Query all completed trips with subsidy information
    trip_query = text("""
        SELECT
            cil.income_level,
            sbl.total_time,
            sbl.total_price,
            sbl.government_subsidy,
            sbl.record_company_name,
            sbl.commuter_id,
            sbl.origin_coordinates,
            sbl.destination_coordinates
        FROM
            service_booking_log sbl
        JOIN
            commuter_info_log cil ON sbl.commuter_id = cil.commuter_id
        WHERE
            sbl.status = 'finished'
    """)
    
    # Process trips by income group
    results = {}
    income_trip_data = {'low': [], 'middle': [], 'high': []}
    
    for row in session.execute(trip_query):
        income = row[0]
        time = float(row[1])  # Ensure numeric type
        price = float(row[2])
        subsidy = float(row[3] or 0)  # Handle NULL values
        mode = row[4]
        commuter_id = row[5]
        
        if income not in income_trip_data:
            continue
            
        # CORRECTED: Time conversion factor for 1 step = 10 minutes
        # Convert time (in steps) to hours: time_steps * (10 min/step) / (60 min/hour)
        time_cost = time * vot[income] / 6  # Correct conversion for your time steps
        
        # Calculate generalized cost with and without subsidy
        gen_cost_with_subsidy = price - subsidy + time_cost
        gen_cost_without_subsidy = price + time_cost
        
        # Handle potential negative costs (if subsidy > price + time_cost)
        if gen_cost_with_subsidy < 0:
            gen_cost_with_subsidy = 0  # Floor at zero
            
        # Calculate accessibility using negative exponential function
        # Based on Handy & Niemeier (1997)
        acc_with_subsidy = math.exp(-beta * gen_cost_with_subsidy)
        acc_without_subsidy = math.exp(-beta * gen_cost_without_subsidy)
        
        income_trip_data[income].append({
            'commuter_id': commuter_id,
            'accessibility_with': acc_with_subsidy,
            'accessibility_without': acc_without_subsidy,
            'improvement': acc_with_subsidy - acc_without_subsidy,
            'subsidy': subsidy,
            'time': time,
            'price': price
        })
    
    # Calculate key metrics by income group
    for income, trips in income_trip_data.items():
        if not trips:
            continue
            
        # Aggregate metrics
        total_trips = len(trips)
        total_subsidy = sum(t['subsidy'] for t in trips)
        avg_subsidy_per_trip = total_subsidy / total_trips if total_trips > 0 else 0
        
        # Calculate accessibility metrics
        destination_potential = sum(t['accessibility_with'] for t in trips)
        baseline_potential = sum(t['accessibility_without'] for t in trips)
        potential_increase = destination_potential - baseline_potential
        
        # Calculate effectiveness ratios
        subsidy_effectiveness = potential_increase / total_subsidy if total_subsidy > 0 else 0
        
        # Store results
        results[income] = {
            'trips': total_trips,
            'total_subsidy': total_subsidy,
            'avg_subsidy_per_trip': avg_subsidy_per_trip,
            'destination_potential_with': destination_potential,
            'destination_potential_without': baseline_potential,
            'potential_increase': potential_increase,
            'effectiveness_ratio': subsidy_effectiveness
        }
    
    # Calculate overall metrics across all income groups
    total_trips = sum(data['trips'] for income, data in results.items())
    total_subsidy = sum(data['total_subsidy'] for income, data in results.items())
    total_potential_increase = sum(data['potential_increase'] for income, data in results.items())
    overall_effectiveness = total_potential_increase / total_subsidy if total_subsidy > 0 else 0
    
    # Add FPS value and aggregate results
    results['fps_value'] = fps_value
    results['total_trips'] = total_trips
    results['total_subsidy'] = total_subsidy
    results['total_potential_increase'] = total_potential_increase
    results['overall_effectiveness'] = overall_effectiveness
    
    return results

def run_single_simulation(params):
    """Run a single simulation and calculate accessibility metrics"""
    pid = os.getpid()
    print(f"Starting simulation with PID {pid} - FPS: {params.get('fps_value', 'N/A')}")
    
    # Seed initialization for consistent results
    np.random.seed(pid + int(params.get('fps_value', 0)))
    random.seed(pid + int(params.get('fps_value', 0)))
    
    # Create a unique schema name for this process
    schema_name = f"sim_{pid}_{int(time.time())}"
    
    # Initialize session and engine variables outside try block
    session = None
    engine = None
    
    try:
        # PostgreSQL connection string with appropriate credentials
        db_connection_string = f"postgresql://z5247491@localhost:15441/postgres"
        
        # Extract analysis parameters
        fps_value = params.pop('fps_value', 0)
        fixed_allocations = params.pop('fixed_allocations', None)
        simulation_steps = params.pop('simulation_steps', SIMULATION_STEPS)
        
        # Initialize database connection
        from sqlalchemy import create_engine, text
        engine = create_engine(db_connection_string)
        
        # Create a unique schema for this process
        with engine.connect() as connection:
            connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema_name}"))
            connection.commit()  # Make sure to commit the schema creation
        
        # Create session with the specified schema
        from sqlalchemy.orm import sessionmaker
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Set the search path to use our schema
        with engine.connect() as connection:
            connection.execute(text(f"SET search_path TO {schema_name}"))
            connection.commit()

        # Reset the database tables with the schema parameter
        reset_db_params = {k: params[k] for k in [
            'uber_like1_capacity', 'uber_like1_price',
            'uber_like2_capacity', 'uber_like2_price',
            'bike_share1_capacity', 'bike_share1_price',
            'bike_share2_capacity', 'bike_share2_price'
        ]}
        
        # Pass the schema parameter to reset_database
        reset_database(
            engine=engine, 
            session=session, 
            schema=schema_name,
            **reset_db_params
        )
        
        # Set up subsidy configuration
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
        
        # Configure the model with schema info
        params['db_connection_string'] = db_connection_string
        params['schema'] = schema_name
        
        # Run simulation with the schema-specific configuration
        # Remove simulation_id if it exists
        if 'simulation_id' in params:
            del params['simulation_id']
        model = MobilityModel(**params)
        model.run_model(simulation_steps)
        
        # Calculate accessibility metrics from data in the current schema
        results = calculate_destination_potential(session, fps_value, schema=schema_name)
        
        # Add simulation parameters to results
        if results:
            results['fps_value'] = fps_value
            results['fixed_allocations'] = fixed_allocations
            return results
        else:
            print(f"Warning: No results from simulation with FPS {fps_value}")
            # Return a default result structure to avoid NoneType errors
            return {
                'fps_value': fps_value,
                'fixed_allocations': fixed_allocations,
                'total_potential_increase': 0,
                'overall_effectiveness': 0,
                'low': {'destination_potential_with': 0, 'effectiveness_ratio': 0},
                'middle': {'destination_potential_with': 0, 'effectiveness_ratio': 0},
                'high': {'destination_potential_with': 0, 'effectiveness_ratio': 0}
            }
        
    except Exception as e:
        print(f"Error in simulation {pid}: {str(e)}")
        traceback.print_exc()
        # Return a default result structure to avoid NoneType errors
        return {
            'fps_value': fps_value if 'fps_value' in locals() else 0,
            'fixed_allocations': fixed_allocations if 'fixed_allocations' in locals() else None,
            'total_potential_increase': 0,
            'overall_effectiveness': 0,
            'low': {'destination_potential_with': 0, 'effectiveness_ratio': 0},
            'middle': {'destination_potential_with': 0, 'effectiveness_ratio': 0},
            'high': {'destination_potential_with': 0, 'effectiveness_ratio': 0}
        }
        
    finally:
        # Safely close session if it was created
        if session:
            session.close()
            
        # Clean up by dropping the schema
        if engine:
            try:
                with engine.connect() as connection:
                    connection.execute(text(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE"))
                    connection.commit()
                print(f"Successfully cleaned up schema {schema_name}")
            except Exception as e:
                print(f"Error cleaning up schema {schema_name}: {str(e)}")

def run_sequential_fps_optimization(fps_values, base_parameters, num_cpus=NUM_CPUS):
    """
    Process FPS values one at a time, using parallelism within each optimization.
    
    Args:
        fps_values: List of FPS values to optimize
        base_parameters: Base simulation parameters
        num_cpus: Number of CPUs to use for parallel simulation
    
    Returns:
        Dictionary mapping FPS values to optimization results
    """
    print(f"Processing {len(fps_values)} FPS values sequentially with parallel simulations")
    
    results = {}
    # Sort FPS values to ensure sequential processing in ascending order
    fps_values = sorted(fps_values)
    
    for fps_index, fps_value in enumerate(fps_values):
        print(f"\n[{fps_index+1}/{len(fps_values)}] Starting optimization for FPS={fps_value}")
        
        # Run parallel-enabled optimization for this FPS value
        result = optimize_allocation_parallel(fps_value, base_parameters, num_cpus)
        results[float(fps_value)] = result
        
        print(f"Completed FPS={fps_value}, Potential Increase={result['total_potential_increase']:.4f}")
        
        # Save partial results after each FPS completes
        with open(f"accessibility_results_fps_{fps_value}_{time.strftime('%Y%m%d_%H%M%S')}.pkl", "wb") as f:
            pickle.dump(results, f)
    
    return results

def optimize_allocation_parallel(fps_value, base_parameters, num_cpus=NUM_CPUS):
    """
    Optimize subsidy allocation using parallel simulations within a single optimization run,
    using destination accessibility as the objective function - maximise the total_potential_increase
    """
    print(f"\nOptimizing subsidy allocation for FPS = {fps_value} with {num_cpus} parallel simulations")
    
    # Define bounds and parameters for optimization
    bounds = {
        'low_bike': (0.45, 0.65),      
        'low_car': (0.5, 0.7),       
        'low_MaaS_Bundle': (0.45, 0.65),
        'low_public': (0.45, 0.65),    
        'middle_bike': (0.25, 0.45),
        'middle_car': (0.25, 0.45),
        'middle_MaaS_Bundle': (0.25, 0.45),
        'middle_public': (0.2, 0.4),
        'high_bike': (0.05, 0.20),
        'high_car': (0, 0.15),      
        'high_MaaS_Bundle': (0.05, 0.20),
        'high_public': (0.05, 0.20)
    }
    
    # Set up parameter space and sampling
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
    X_init = sampler.random(n=10)  
    X_init = qmc.scale(X_init, lower_bounds, upper_bounds)
    
    # Set up early termination parameters
    min_improvement_threshold = 0.01
    target_potential_threshold = 5.0  # Target value for potential increase
    no_improvement_iterations = 0
    patience = 6
    
    best_potential_increase = float('-inf')  # We want to maximize potential increase
    best_allocations = None
    best_results = None
    previous_best = float('-inf')

    # Prepare parameter sets for initial points
    initial_param_sets = []
    for i, params in enumerate(X_init):
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
        sim_params['simulation_id'] = i  # Add ID for tracking
        initial_param_sets.append(sim_params)
    
    # Run initial simulations in parallel
    print(f"Running {len(initial_param_sets)} initial simulations in parallel with {num_cpus} CPUs")
    with mp.Pool(processes=num_cpus) as pool:
        initial_results = list(pool.map(run_single_simulation, initial_param_sets))
    
    # Process initial results
    y_init = []
    for i, results in enumerate(initial_results):
        if results:
            potential_increase = results.get('total_potential_increase', float('-inf'))
            y_init.append(-potential_increase)  # Negate for minimization
            
            if potential_increase > best_potential_increase:
                best_potential_increase = potential_increase
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
                print(f"New best potential increase from initial points: {best_potential_increase:.6f}")
                
                if best_potential_increase > target_potential_threshold:
                    print(f"Early termination: Target potential threshold {target_potential_threshold} reached!")
                    return {
                        'fps_value': fps_value,
                        'optimal_allocations': best_allocations,
                        'potential_scores': {level: best_results[level]['potential_increase'] 
                                             for level in ['low', 'middle', 'high'] if level in best_results},
                        'total_potential_increase': best_potential_increase,
                        'overall_effectiveness': best_results.get('overall_effectiveness', 0),
                        'full_results': best_results,
                        'terminated_early': True,
                        'termination_reason': 'Target threshold reached'
                    }
        else:
            y_init.append(float('inf'))  # Penalize failed simulations

    # Setup Gaussian Process
    kernel = Matern(nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42)
    
    X = X_init
    y = np.array(y_init)  # Negated potential increase for minimization
    
    # Run optimization iterations
    max_iterations = 10
    
    for iteration in range(max_iterations):
        previous_best = best_potential_increase
        
        # Fit GP to current data
        gp.fit(X, y)
        
        # Generate candidates for Expected Improvement
        candidates = sampler.random(n=500)
        candidates = qmc.scale(candidates, lower_bounds, upper_bounds)
        
        # Calculate EI for all candidates (minimizing negative potential)
        ei_values = []
        for candidate in candidates:
            mu, sigma = gp.predict(candidate.reshape(1, -1), return_std=True)
            mu = mu.reshape(-1)
            sigma = sigma.reshape(-1)
            
            # For maximization of potential, we negate the values
            imp = np.min(y) - mu
            Z = imp / (sigma + 1e-9)
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei_values.append(float(ei.item()) if hasattr(ei, 'item') else float(ei))
        
        # Select top candidates for parallel evaluation
        top_n = min(num_cpus, len(candidates))
        top_indices = np.argsort(ei_values)[-top_n:]
        top_candidates = [candidates[i] for i in top_indices]
        
        # Prepare parameter sets for parallel simulation
        iteration_param_sets = []
        for i, candidate in enumerate(top_candidates):
            allocations = {
                'low_bike': candidate[0],
                'low_car': candidate[1],
                'low_MaaS_Bundle': candidate[2],
                'low_public': candidate[3],
                'middle_bike': candidate[4],
                'middle_car': candidate[5],
                'middle_MaaS_Bundle': candidate[6],
                'middle_public': candidate[7],
                'high_bike': candidate[8],
                'high_car': candidate[9],
                'high_MaaS_Bundle': candidate[10],
                'high_public': candidate[11]
            }
            
            sim_params = base_parameters.copy()
            sim_params['fps_value'] = fps_value
            sim_params['fixed_allocations'] = allocations
            sim_params['simulation_steps'] = SIMULATION_STEPS
            sim_params['simulation_id'] = i
            iteration_param_sets.append(sim_params)
        
        # Run simulations in parallel
        print(f"Iteration {iteration+1}: Running {len(iteration_param_sets)} simulations in parallel")
        with mp.Pool(processes=min(num_cpus, len(iteration_param_sets))) as pool:
            simulation_results = list(pool.map(run_single_simulation, iteration_param_sets))
        
        # Process results and update best allocation
        found_improvement = False
        for i, result in enumerate(simulation_results):
            if result:
                potential_increase = result.get('total_potential_increase', float('-inf'))
                
                # Add to GP data
                X = np.vstack([X, top_candidates[i]])
                y = np.append(y, -potential_increase)  # Negate for minimization
                
                improvement = potential_increase - previous_best
                
                if potential_increase > best_potential_increase:
                    best_potential_increase = potential_increase
                    best_allocations = {
                        'low_bike': top_candidates[i][0],
                        'low_car': top_candidates[i][1],
                        'low_MaaS_Bundle': top_candidates[i][2],
                        'low_public': top_candidates[i][3],
                        'middle_bike': top_candidates[i][4],
                        'middle_car': top_candidates[i][5],
                        'middle_MaaS_Bundle': top_candidates[i][6],
                        'middle_public': top_candidates[i][7],
                        'high_bike': top_candidates[i][8],
                        'high_car': top_candidates[i][9],
                        'high_MaaS_Bundle': top_candidates[i][10],
                        'high_public': top_candidates[i][11]
                    }
                    best_results = result
                    found_improvement = True
                    print(f"Iteration {iteration + 1}, New best potential increase: {best_potential_increase:.6f}, Improvement: {improvement:.6f}")
                    
                    if improvement > min_improvement_threshold:
                        no_improvement_iterations = 0
                    else:
                        no_improvement_iterations += 1
                        print(f"  Minimal improvement: {improvement:.6f} < {min_improvement_threshold}")
                    
                    if best_potential_increase > target_potential_threshold:
                        print(f"Early termination after {iteration + 1} iterations: Target potential threshold reached!")
                        return {
                            'fps_value': fps_value,
                            'optimal_allocations': best_allocations,
                            'potential_scores': {level: best_results[level]['potential_increase'] 
                                               for level in ['low', 'middle', 'high'] if level in best_results},
                            'total_potential_increase': best_potential_increase,
                            'overall_effectiveness': best_results.get('overall_effectiveness', 0),
                            'full_results': best_results,
                            'terminated_early': True,
                            'termination_reason': 'Target threshold reached'
                        }
        
        if not found_improvement:
            print(f"Iteration {iteration + 1}, No improvement. Current best: {best_potential_increase:.6f}")
            no_improvement_iterations += 1
        
        if no_improvement_iterations >= patience:
            print(f"Early termination after {iteration + 1} iterations: No significant improvement for {patience} consecutive iterations")
            return {
                'fps_value': fps_value,
                'optimal_allocations': best_allocations,
                'potential_scores': {level: best_results[level]['potential_increase'] 
                                   for level in ['low', 'middle', 'high'] if level in best_results},
                'total_potential_increase': best_potential_increase,
                'overall_effectiveness': best_results.get('overall_effectiveness', 0),
                'full_results': best_results,
                'terminated_early': True,
                'termination_reason': 'Convergence'
            }

    return {
        'fps_value': fps_value,
        'optimal_allocations': best_allocations,
        'potential_scores': {level: best_results[level]['potential_increase'] 
                           for level in ['low', 'middle', 'high'] if level in best_results},
        'total_potential_increase': best_potential_increase,
        'overall_effectiveness': best_results.get('overall_effectiveness', 0),
        'full_results': best_results,
        'terminated_early': False
    }

def visualize_accessibility_results(results, output_dir):
    """
    Create visualizations from the accessibility analysis results
    
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
            row = {'fps_value': fps, 'total_potential_increase': result['total_potential_increase']}
            
            # Add effectiveness and potential scores for each income level
            for level in ['low', 'middle', 'high']:
                if 'potential_scores' in result and level in result['potential_scores']:
                    row[f'potential_increase_{level}'] = result['potential_scores'][level]
                
                # Add effectiveness ratios if available
                if 'full_results' in result and level in result['full_results']:
                    row[f'effectiveness_{level}'] = result['full_results'][level].get('effectiveness_ratio', 0)
            
            # Add allocation percentages
            for key, value in result['optimal_allocations'].items():
                row[f'alloc_{key}'] = value
                
            # Add overall effectiveness
            row['overall_effectiveness'] = result.get('overall_effectiveness', 0)
                
            rows.append(row)
    
    if not rows:
        print("No valid results to visualize")
        return
        
    df = pd.DataFrame(rows)
    if not df.empty:
        # Ensure FPS values are numeric
        df['fps_value'] = pd.to_numeric(df['fps_value'])
        
        # Sort DataFrame by FPS value to ensure proper plotting order
        df = df.sort_values('fps_value')
        
        # Reset index after sorting
        df = df.reset_index(drop=True)
        
        print(f"Processing data for {len(df)} FPS values: {sorted(df['fps_value'].tolist())}")
    
    # Save full data table that will be referenced in the visualizations
    data_table = df.copy()
    # Format allocation percentages as percentages
    for col in data_table.columns:
        if col.startswith('alloc_'):
            data_table[col] = data_table[col].map(lambda x: f"{x*100:.1f}%")
    
    data_table.to_csv(os.path.join(output_dir, 'accessibility_allocation_table.csv'), index=False)
    
    # 1. Create main visualization of potential increase by income level across FPS values
    plt.figure(figsize=(15, 10))
    
    # Plot lines for each income level
    colors = {'low': '#1f77b4', 'middle': '#ff7f0e', 'high': '#2ca02c', 'total': '#d62728'}
    markers = {'low': 'o', 'middle': 's', 'high': '^', 'total': 'D'}
    
    # Plot income-specific potential increases
    for level in ['low', 'middle', 'high']:
        plt.plot(df['fps_value'], df[f'potential_increase_{level}'], 
                 marker=markers[level], linestyle='-', 
                 color=colors[level], 
                 label=f'{level.capitalize()} Income', 
                 alpha=0.8)
    
    # Plot total potential increase with bolder line
    plt.plot(df['fps_value'], df['total_potential_increase'], 
             marker=markers['total'], linestyle='-', 
             color=colors['total'],
             linewidth=3, 
             label='Total Potential Increase', 
             alpha=0.9)
    
    # Add trend line using polynomial fit
    if len(df) > 2:
        try:
            from scipy import stats
            x = df['fps_value']
            y = df['total_potential_increase']
            z = np.polyfit(np.log(x), y, 2)  # Fit against log of FPS
            p = np.poly1d(z)
            x_smooth = np.geomspace(df['fps_value'].min(), df['fps_value'].max(), 100)
            y_smooth = p(np.log(x_smooth))
            
            # Add smoother constraints
            y_smooth = np.clip(y_smooth, 0, y.max()*1.2)
            
            plt.plot(x_smooth, y_smooth, '--', color='gray', alpha=0.6, label='Trend Line')
        except Exception as e:
            print(f"Error fitting trend line: {e}")
    
    # Add point annotations for allocation references
    for i, (idx, row) in enumerate(df.iterrows()):
        if i % 2 == 0:  # Only annotate every other point
            fps = row['fps_value']
            potential = row['total_potential_increase']
            
            # Add annotation with arrow pointing to point
            plt.annotate(f'Point {i+1}',  # Use numbers to reference the table
                        xy=(fps, potential),
                        xytext=(10, 10 + (i % 3) * 20),  # Stagger annotations
                        textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'),
                        fontsize=8)
    
    # Add table reference
    plt.figtext(0.02, 0.02, 
               "Note: See 'accessibility_allocation_table.csv' for the complete allocations table",
               bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
    
    # Improve graph appearance
    plt.xscale('log')  # Log scale for FPS values
    plt.title('Destination Potential Increase vs Fixed Pool Subsidy (FPS) Values', fontsize=16)
    plt.xlabel('Fixed Pool Subsidy (FPS)', fontsize=14)
    plt.ylabel('Destination Potential Increase (Higher is Better)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='best')
    
    # Add vertical line at the FPS value with highest potential increase
    if not df.empty:
        best_fps = df.loc[df['total_potential_increase'].idxmax()]['fps_value']
        plt.axvline(x=best_fps, color='red', linestyle='--', alpha=0.5,
                   label=f'Best FPS: {best_fps}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'potential_increase_vs_fps.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Create visualization of average effectiveness by income level
    plt.figure(figsize=(10, 8))

    # Calculate average effectiveness for each income level
    income_levels = ['low', 'middle', 'high']
    avg_effectiveness = {}
    std_effectiveness = {}

    for level in income_levels:
        avg_effectiveness[level] = df[f'effectiveness_{level}'].mean()
        std_effectiveness[level] = df[f'effectiveness_{level}'].std()

    # Add overall average
    avg_effectiveness['overall'] = df['overall_effectiveness'].mean()
    std_effectiveness['overall'] = df['overall_effectiveness'].std()

    # Set up bar positions and width
    all_levels = income_levels + ['overall']
    positions = np.arange(len(all_levels))
    width = 0.7

    # Create bars with the same color scheme as before
    bars = plt.bar(positions, 
                [avg_effectiveness[level] for level in all_levels],
                width, 
                color=[colors[level] if level != 'overall' else colors['total'] for level in all_levels],
                alpha=0.8,
                yerr=[std_effectiveness[level] for level in all_levels], 
                capsize=5)

    # Add value labels above each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                f'{avg_effectiveness[all_levels[i]]:.6f}',
                ha='center', va='bottom', fontsize=10)

    # Improve appearance
    plt.title('Average Effectiveness Ratio by Income Level', fontsize=16)
    plt.ylabel('Effectiveness Ratio (Potential Increase per Subsidy Unit)', fontsize=14)
    plt.xticks(positions, [level.capitalize() if level != 'overall' else 'Overall' 
                        for level in all_levels])
    plt.grid(True, alpha=0.3, axis='y')

    # Add explanation text
    plt.figtext(0.02, 0.02, 
            "Note: Effectiveness Ratio represents the destination potential increase per subsidy unit",
            bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'effectiveness_vs_fps.png'), 
            dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Create detailed allocation line graphs for each income level
    income_levels = ['low', 'middle', 'high']
    modes = ['bike', 'car', 'MaaS_Bundle', 'public', 'walk']
    
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
    
    # 4. Create accessibility vs equity scatter plot
    if 'equity_scores' in results[list(results.keys())[0]].get('full_results', {}):
        try:
            equity_df = pd.DataFrame([
                {
                    'fps_value': fps,
                    'potential_increase': result['total_potential_increase'],
                    'equity_score': result['full_results'].get('total_equity_indicator', 0)
                }
                for fps, result in results.items()
                if 'total_potential_increase' in result
            ])
            
            if not equity_df.empty:
                plt.figure(figsize=(12, 8))
                scatter = plt.scatter(equity_df['equity_score'], equity_df['potential_increase'], 
                                     c=equity_df['fps_value'], cmap='viridis', 
                                     s=100, alpha=0.7)
                
                # Add FPS annotations
                for i, row in equity_df.iterrows():
                    plt.annotate(f"FPS={row['fps_value']}", 
                                xy=(row['equity_score'], row['potential_increase']),
                                xytext=(5, 5),
                                textcoords="offset points")
                
                plt.colorbar(scatter, label='FPS Value')
                plt.title('Trade-off: Equity vs Accessibility', fontsize=16)
                plt.xlabel('Equity Score (Lower is Better)', fontsize=14)
                plt.ylabel('Destination Potential Increase (Higher is Better)', fontsize=14)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'equity_vs_accessibility.png'))
                plt.close()
        except Exception as e:
            print(f"Error creating equity vs accessibility plot: {e}")
    
    # 5. Create a radar chart comparison of optimal allocation
    # Get the optimal allocation
    if not df.empty:
        best_idx = df['total_potential_increase'].idxmax()
        optimal_row = df.iloc[best_idx]
        
        # Prepare data for radar chart
        modes = ['bike', 'car', 'MaaS_Bundle', 'public']
        income_levels = ['low', 'middle', 'high']
        
        radar_data = {}
        for income in income_levels:
            radar_data[income] = [optimal_row[f'alloc_{income}_{mode}'] for mode in modes]
        
        # Create radar chart
        plt.figure(figsize=(10, 8))
        angles = np.linspace(0, 2*np.pi, len(modes), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        ax = plt.subplot(111, polar=True)
        
        for income, values in radar_data.items():
            values += values[:1]  # Close the loop
            ax.plot(angles, values, 'o-', linewidth=2, label=f'{income.capitalize()} Income')
            ax.fill(angles, values, alpha=0.1)
        
        # Set labels
        ax.set_thetagrids(np.degrees(angles[:-1]), modes)
        ax.set_ylim(0, 1)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title(f'Optimal Allocation Distribution for FPS={optimal_row["fps_value"]}', fontsize=16)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'optimal_allocation_radar.png'))
        plt.close()
    
    # Find and save the optimal solution
    if 'total_potential_increase' in df.columns:
        optimal_row = df.loc[df['total_potential_increase'].idxmax()]
        
        with open(os.path.join(output_dir, 'optimal_accessibility_solution.txt'), 'w') as f:
            f.write(f"Optimal FPS Value for Accessibility: {optimal_row['fps_value']}\n")
            f.write(f"Optimal Total Potential Increase: {optimal_row['total_potential_increase']:.4f}\n")
            f.write(f"Overall Effectiveness: {optimal_row['overall_effectiveness']:.4f}\n\n")
            f.write("Optimal Subsidy Allocations:\n")
            
            for key in sorted([k for k in optimal_row.index if k.startswith('alloc_')]):
                f.write(f"  {key.replace('alloc_', '')}: {optimal_row[key]:.4f}\n")
                
            f.write("\nPotential Increase by Income Level:\n")
            for level in ['low', 'middle', 'high']:
                f.write(f"  {level.capitalize()}: {optimal_row[f'potential_increase_{level}']:.4f}\n")
                
            f.write("\nEffectiveness by Income Level:\n")
            for level in ['low', 'middle', 'high']:
                f.write(f"  {level.capitalize()}: {optimal_row[f'effectiveness_{level}']:.4f}\n")

def visualize_destination_insights(results, output_dir):
    """
    Create visualizations focusing on destination accessibility insights.
    
    Args:
        results: Dictionary of results from optimization
        output_dir: Directory to save visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract detailed accessibility data from results
    detailed_data = []
    fps_values = []
    
    for fps, result in results.items():
        if 'full_results' in result and result['full_results']:
            fps_values.append(fps)
            
            for income_level in ['low', 'middle', 'high']:
                if income_level in result['full_results']:
                    income_data = result['full_results'][income_level]
                    
                    # Extract key metrics
                    detailed_data.append({
                        'fps_value': fps,
                        'income_level': income_level,
                        'destination_potential_with': income_data.get('destination_potential_with', 0),
                        'destination_potential_without': income_data.get('destination_potential_without', 0),
                        'potential_increase': income_data.get('potential_increase', 0),
                        'effectiveness_ratio': income_data.get('effectiveness_ratio', 0),
                        'total_subsidy': income_data.get('total_subsidy', 0),
                        'trips': income_data.get('trips', 0),
                        'avg_subsidy_per_trip': income_data.get('avg_subsidy_per_trip', 0)
                    })
    
    if not detailed_data:
        print("No detailed data available for visualization")
        return
    
    detailed_df = pd.DataFrame(detailed_data)
    
    # 1. Create visualization of accessibility with vs without subsidies
    plt.figure(figsize=(15, 10))
    
    # Set up subplot grid
    income_levels = ['low', 'middle', 'high']
    colors = {'low': '#1f77b4', 'middle': '#ff7f0e', 'high': '#2ca02c'}
    
    for i, income in enumerate(income_levels):
        # Filter data for this income level
        income_data = detailed_df[detailed_df['income_level'] == income]
        
        if income_data.empty:
            continue
            
        # Sort by FPS value
        income_data = income_data.sort_values('fps_value')
        
        # Create subplot
        plt.subplot(3, 1, i+1)
        
        # Plot stacked bars showing with/without subsidy
        x = range(len(income_data))
        plt.bar(x, income_data['destination_potential_without'], 
                color='lightgray', label='Baseline Potential')
        plt.bar(x, income_data['potential_increase'], 
                bottom=income_data['destination_potential_without'],
                color=colors[income], label='Potential Increase from Subsidy')
        
        # Add labels
        plt.title(f'Destination Potential for {income.capitalize()} Income Group')
        plt.xticks(x, [f"{fps}" for fps in income_data['fps_value']])
        plt.xlabel('FPS Value')
        plt.ylabel('Destination Potential')
        
        # Add value labels on bars
        for j, (base, increase) in enumerate(zip(income_data['destination_potential_without'], 
                                               income_data['potential_increase'])):
            total = base + increase
            plt.text(j, total + 0.1, f"{total:.1f}", ha='center')
            
            # Show increase percentage
            if base > 0:
                pct_increase = (increase / base) * 100
                plt.text(j, base + (increase/2), f"+{pct_increase:.1f}%", 
                        ha='center', color='white', fontweight='bold')
        
        plt.legend()
        plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'destination_potential_breakdown.png'))
    plt.close()
    
    # 2. Effectiveness ratio comparison across income groups and FPS values
    plt.figure(figsize=(12, 8))
    
    # Pivot data for heatmap
    pivot_df = detailed_df.pivot_table(
        values='effectiveness_ratio',
        index='income_level',
        columns='fps_value'
    )
    
    # Create heatmap
    sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt=".3f")
    plt.title('Subsidy Effectiveness by Income Level and FPS Value')
    plt.xlabel('FPS Value')
    plt.ylabel('Income Level')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'effectiveness_heatmap.png'))
    plt.close()
    
    # 3. Relationship between subsidy amount and potential increase
    plt.figure(figsize=(12, 8))
    
    for income in income_levels:
        income_data = detailed_df[detailed_df['income_level'] == income]
        if not income_data.empty:
            plt.scatter(income_data['total_subsidy'], income_data['potential_increase'],
                       label=f'{income.capitalize()} Income', 
                       color=colors[income], s=80, alpha=0.7)
            
            # Add FPS values as labels
            for _, row in income_data.iterrows():
                plt.annotate(f"{row['fps_value']}", 
                            xy=(row['total_subsidy'], row['potential_increase']),
                            xytext=(5, 5),
                            textcoords='offset points')
    
    plt.title('Relationship Between Subsidy Amount and Potential Increase')
    plt.xlabel('Total Subsidy Amount')
    plt.ylabel('Potential Increase')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Try to fit trend lines if enough data points
    for income in income_levels:
        income_data = detailed_df[detailed_df['income_level'] == income]
        if len(income_data) >= 3:
            try:
                # Simple polynomial fit
                x = income_data['total_subsidy']
                y = income_data['potential_increase']
                z = np.polyfit(x, y, 2)
                p = np.poly1d(z)
                
                # Generate smooth points for the line
                x_smooth = np.linspace(x.min(), x.max(), 100)
                y_smooth = p(x_smooth)
                
                plt.plot(x_smooth, y_smooth, '--', color=colors[income], alpha=0.5)
            except Exception as e:
                print(f"Error fitting trend line for {income} income: {e}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'subsidy_vs_potential.png'))
    plt.close()
    
    # 4. Average subsidy per trip across income levels
    plt.figure(figsize=(12, 8))
    
    # Pivot data
    avg_subsidy_pivot = detailed_df.pivot_table(
        values='avg_subsidy_per_trip',
        index='fps_value',
        columns='income_level'
    )
    
    # Plot
    avg_subsidy_pivot.plot(kind='bar', figsize=(12, 8))
    plt.title('Average Subsidy per Trip by Income Level')
    plt.xlabel('FPS Value')
    plt.ylabel('Average Subsidy per Trip')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Income Level')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'avg_subsidy_per_trip.png'))
    plt.close()
    
    # 5. Efficiency vs Equity analysis
    if 'equity_scores' in results[list(results.keys())[0]].get('full_results', {}):
        try:
            # Extract equity and efficiency data
            efficiency_equity_data = []
            
            for fps, result in results.items():
                if 'full_results' in result and 'total_equity_indicator' in result['full_results']:
                    equity_score = result['full_results']['total_equity_indicator']
                    
                    # Get income-specific data
                    for income in income_levels:
                        if income in result['full_results']:
                            efficiency_equity_data.append({
                                'fps_value': fps,
                                'income_level': income,
                                'equity_score': equity_score,
                                'effectiveness_ratio': result['full_results'][income].get('effectiveness_ratio', 0),
                                'destination_potential': result['full_results'][income].get('destination_potential_with', 0)
                            })
            
            if efficiency_equity_data:
                eff_eq_df = pd.DataFrame(efficiency_equity_data)
                
                # Create scatter plot
                plt.figure(figsize=(12, 8))
                
                for income in income_levels:
                    income_data = eff_eq_df[eff_eq_df['income_level'] == income]
                    if not income_data.empty:
                        plt.scatter(income_data['equity_score'], income_data['effectiveness_ratio'],
                                  label=f'{income.capitalize()} Income', 
                                  color=colors[income], s=80, alpha=0.7)
                        
                        # Size points by destination potential
                        sizes = income_data['destination_potential'] * 50
                        sizes = np.clip(sizes, 50, 500)  # Limit size range
                        
                        # Create bubbles sized by destination potential
                        plt.scatter(income_data['equity_score'], income_data['effectiveness_ratio'],
                                  s=sizes, alpha=0.3, color=colors[income])
                        
                        # Add FPS labels
                        for _, row in income_data.iterrows():
                            plt.annotate(f"{row['fps_value']}", 
                                        xy=(row['equity_score'], row['effectiveness_ratio']),
                                        xytext=(5, 5),
                                        textcoords='offset points')
                
                plt.title('Efficiency vs Equity Trade-off')
                plt.xlabel('Equity Score (Lower is Better)')
                plt.ylabel('Effectiveness Ratio (Higher is Better)')
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                plt.figtext(0.02, 0.02, 
                          "Note: Bubble size represents destination potential",
                          bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'efficiency_vs_equity.png'))
                plt.close()
        except Exception as e:
            print(f"Error creating efficiency vs equity plot: {e}")
            traceback.print_exc()

def main():
    """Main function to run the complete accessibility optimization analysis"""
    # Create output directory immediately to ensure it exists
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"accessibility_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Write an initial status file to ensure directory has content
    with open(f"{output_dir}/analysis_status.txt", "w") as f:
        f.write(f"Analysis started at {timestamp}\n")
    
    print(f"Starting parallel optimization with {NUM_CPUS} CPUs")
    print(f"Output directory created: {output_dir}")

    try:
        base_parameters = {
            'num_commuters': 130,
            'grid_width': 85,
            'grid_height': 85,
            'data_income_weights': [0.5, 0.3, 0.2],
            'data_health_weights': [0.9, 0.1],
            'data_payment_weights': [0.8, 0.2],
            'data_age_distribution': {(18, 25): 0.2, (26, 35): 0.3, (36, 45): 0.2, 
                                    (46, 55): 0.15, (56, 65): 0.1, (66, 75): 0.05},
            'data_disability_weights': [0.2, 0.8],
            'data_tech_access_weights': [0.95, 0.05],
            'ASC_VALUES': {'car': 0, 'bike': 0, 'public': 0, 
                          'walk': 0, 'maas':0, 'default': 0},
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
            'AFFORDABILITY_THRESHOLDS': {'low': 25, 'middle': 40, 'high': 130},
            'FLEXIBILITY_ADJUSTMENTS': {'low': 1.15, 'medium': 1.0, 'high': 0.85},
            'VALUE_OF_TIME': {'low': 5, 'middle': 10, 'high': 20},
            'public_price_table': {
                'train': {'on_peak': 3, 'off_peak': 2.6},
                'bus': {'on_peak': 2.4, 'off_peak': 2}
            },
            'ALPHA_VALUES': {
                'UberLike1': 0.3,
                'UberLike2': 0.3,
                'BikeShare1': 0.25,
                'BikeShare2': 0.25
            },
            'DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS': {
                'S_base': 0.02,  # Base surcharge (10%)
                'alpha': 0.10,   # Sensitivity coefficient
                'delta': 0.5     # Reduction factor for subscription model
            },
            'BACKGROUND_TRAFFIC_AMOUNT': 120,
            'CONGESTION_ALPHA': 0.03,
            'CONGESTION_BETA': 1.5,  
            'CONGESTION_CAPACITY': 10,
            'CONGESTION_T_IJ_FREE_FLOW': 1.5,
            'uber_like1_capacity': 15,
            'uber_like1_price': 15.5,
            'uber_like2_capacity': 19,
            'uber_like2_price': 16.5,
            'bike_share1_capacity': 10,
            'bike_share1_price': 2.5,
            'bike_share2_capacity': 12,
            'bike_share2_price': 3 
        }
        
        # Define FPS values to analyze - use a focused set of values
        fps_values = [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 6000, 7000, 8000]
        
        # Save parameters for reference
        with open(f"{output_dir}/parameters.pkl", "wb") as f:
            pickle.dump({
                'base_parameters': base_parameters,
                'fps_values': fps_values,
                'simulation_steps': SIMULATION_STEPS,
                'num_cpus': NUM_CPUS
            }, f)

        # Try to run just one FPS value first to check for issues
        print(f"Running initial test with FPS={fps_values[0]}")
        single_result = optimize_allocation_parallel(fps_values[0], base_parameters, num_cpus=NUM_CPUS)
        
        # Save this initial result
        with open(f"{output_dir}/initial_test_fps_{fps_values[0]}.pkl", "wb") as f:
            pickle.dump(single_result, f)
        
        print(f"Initial test completed successfully. Running full optimization...")
        
        # Run full parallel optimization
        results = run_sequential_fps_optimization(fps_values, base_parameters, num_cpus=NUM_CPUS)
        
        # Save results after each FPS value is processed
        for fps, result in results.items():
            print(f"Saving results for FPS={fps}")
            with open(f"{output_dir}/fps_{fps}_accessibility_result.pkl", "wb") as f:
                pickle.dump(result, f)
        
        # Save combined results
        with open(f"{output_dir}/all_accessibility_results.pkl", "wb") as f:
            pickle.dump(results, f)
        
        # Only try to find the best FPS if we have results
        if results:
            # Identify best FPS for accessibility
            best_fps = max(results.items(), key=lambda x: x[1].get('total_potential_increase', 0))[0]
            print(f"\nOptimal FPS value for accessibility: {best_fps}")
            print(f"Optimal potential increase: {results[best_fps].get('total_potential_increase', 0):.4f}")
            print(f"Overall effectiveness: {results[best_fps].get('overall_effectiveness', 0):.4f}")
            
            # Create visualizations
            try:
                visualize_accessibility_results(results, output_dir)
                visualize_destination_insights(results, output_dir)
            except Exception as viz_e:
                print(f"Error during visualization: {viz_e}")
                with open(f"{output_dir}/visualization_error.txt", "w") as f:
                    f.write(f"Visualization error: {str(viz_e)}\n")
                    f.write(traceback.format_exc())
        else:
            print("No results were obtained. Check for errors in the optimization process.")
            with open(f"{output_dir}/no_results_error.txt", "w") as f:
                f.write("No results were obtained from the optimization process.\n")
        
        # Update status file with completion
        with open(f"{output_dir}/analysis_status.txt", "a") as f:
            f.write(f"Analysis completed at {time.strftime('%Y%m%d_%H%M%S')}\n")
        
        print(f"\nAnalysis complete. Results saved to {output_dir}")
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        traceback.print_exc()
        
        # Write error information to output directory
        with open(f"{output_dir}/error_log.txt", "w") as f:
            f.write(f"Analysis failed with error: {str(e)}\n")
            f.write(traceback.format_exc())
        
        # Update status file with error
        with open(f"{output_dir}/analysis_status.txt", "a") as f:
            f.write(f"Analysis failed at {time.strftime('%Y%m%d_%H%M%S')}\n")
            f.write(f"Error: {str(e)}\n")

if __name__ == "__main__":
    main()