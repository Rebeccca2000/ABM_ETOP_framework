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
from sqlalchemy import create_engine, func, text
from agent_subsidy_pool import SubsidyPoolConfig
from run_visualisation_03 import MobilityModel
from agent_service_provider_initialisation_03 import reset_database, CommuterInfoLog, ServiceBookingLog
from functools import partial
import time
import random

# Constants
SIMULATION_STEPS = 120  # Reduced for faster runs during optimization
NUM_CPUS = 15  # Adjust based on your system

def calculate_equity_indicator(session, schema=None):
    """
    Calculate equity indicator using Mean Absolute Error (MAE) approach.
    This measures how much each income group's mode shares deviate from the average.
    
    Args:
        session: SQLAlchemy session with simulation data
        schema: Optional schema name for PostgreSQL
        
    Returns:
        dict: Results containing equity indicators for each income level and total MAE
    """
    results = {}
    income_levels = ['low', 'middle', 'high']
    
    try:
        # Set schema if provided
        if schema:
            # Create a direct SQL query with explicit schema references
            from sqlalchemy import text
            
            print(f"Setting search path to schema: {schema}")
            session.execute(text(f"SET search_path TO {schema}"))
            session.commit()
            
            # First verify tables exist in the schema
            table_check = session.execute(text(
                f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{schema}'"
            )).fetchall()
            print(f"Tables in schema {schema}: {[row[0] for row in table_check]}")
            
        # Get mode shares and trip counts for each income level
        mode_shares = {}
        total_trips_by_income = {}
        all_mode_choices = {}
        all_trips = 0
        
        for income_level in income_levels:
            mode_shares[income_level] = {}
            
            # Use text-based SQL when schema is provided
            if schema:
                query = text(f"""
                    SELECT record_company_name, COUNT(request_id) as count
                    FROM service_booking_log JOIN commuter_info_log 
                    ON service_booking_log.commuter_id = commuter_info_log.commuter_id
                    WHERE commuter_info_log.income_level = :income_level
                    GROUP BY record_company_name
                """)
                mode_choices = session.execute(query, {"income_level": income_level}).fetchall()
            else:
                # Use ORM query for default schema
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
            
            # Store mode choices for later use
            all_mode_choices[income_level] = mode_choices
            
            # Calculate total trips for this income level
            total_trips_by_income[income_level] = sum(choice[1] for choice in mode_choices) or 1  # Avoid division by zero
            all_trips += total_trips_by_income[income_level]
            
            # Calculate mode shares
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
                    mode_shares[income_level][category] += count / total_trips_by_income[income_level]
                else:
                    mode_shares[income_level][category] = count / total_trips_by_income[income_level]
        
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
            shares = [mode_shares[income][mode] for income in income_levels]
            avg_mode_shares[mode] = sum(shares) / len(shares)
        
        # Calculate MAE for each income level and total MAE
        equity_indicators = {}
        total_mae = 0
        weighted_total = 0
        
        # Calculate population weight for each income level
        income_weights = {income: total_trips_by_income[income] / all_trips 
                         for income in income_levels}
        
        # Store detailed breakdown for each income level
        for income_level in income_levels:
            mode_details = []
            income_total_diff = 0
            
            for mode in all_modes:
                mode_share = mode_shares[income_level].get(mode, 0)
                avg_share = avg_mode_shares.get(mode, 0)
                
                # Simple absolute difference - no transformations
                abs_diff = abs(mode_share - avg_share)
                income_total_diff += abs_diff
                
                # Add to total MAE
                total_mae += abs_diff  # Normalize by total number of comparisons
                
                # Add to weighted total (optional, if you want to keep population weighting)
                weighted_total += abs_diff * income_weights[income_level] / len(all_modes)
                
                mode_details.append({
                    'mode': mode,
                    'mode_share': mode_share,
                    'avg_share': avg_share,
                    'difference': abs_diff
                })
            
            # Store the indicator for this income level
            equity_indicators[income_level] = income_total_diff
            
            # Store detailed results
            results[income_level] = {
                'equity_indicator': income_total_diff,
                'mode_shares': mode_shares[income_level],
                'mode_details': mode_details,
                'weight': income_weights[income_level],
                'trips': total_trips_by_income[income_level]
            }
        
        # Store results
        results['total_equity_indicator'] = total_mae
        results['weighted_equity_indicator'] = weighted_total  # Optional weighted version
        results['income_weights'] = income_weights
        results['total_trips'] = all_trips
        
        return results
        
    except Exception as e:
        print(f"Error in calculate_equity_indicator: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Return a default structure to avoid NoneType errors
        default_results = {}
        for income_level in income_levels:
            default_results[income_level] = {
                'equity_indicator': 0,
                'mode_shares': {},
                'mode_details': [],
                'weight': 1 / len(income_levels),
                'trips': 0
            }
        
        default_results['total_equity_indicator'] = 0
        default_results['weighted_equity_indicator'] = 0
        default_results['income_weights'] = {income: 1/len(income_levels) for income in income_levels}
        default_results['total_trips'] = 0
        
        return default_results

def calculate_subsidy_usage_statistics(session, fps_value, schema=None):
    """
    Calculate statistics about subsidy usage for a given FPS value.
    
    Args:
        session: SQLAlchemy session with simulation data
        fps_value: The FPS value used in the simulation
        schema: Optional schema name for PostgreSQL
        
    Returns:
        dict: Results containing subsidy usage statistics
    """
    try:
        # Set schema if provided
        if schema:
            
            print(f"Setting search path to schema for subsidy stats: {schema}")
            session.execute(text(f"SET search_path TO {schema}"))
            session.commit()
        
        # Use direct SQL for more flexibility with schema
        # Check if table exists in this schema
            check_table = text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = :schema 
                    AND table_name = 'subsidy_usage_log'
                )
            """)
        table_exists = session.execute(check_table, {'schema': schema}).scalar()
        if not table_exists:
            print(f"Warning: subsidy_usage_log table doesn't exist in schema {schema}")
        # Total subsidy usage
        query = text("""
            SELECT SUM(subsidy_amount) as total
            FROM subsidy_usage_log
        """)
            
        result = session.execute(query).fetchone()
        total_subsidy_used = float(result.total) if result and result.total else 0
        
        # Calculate percentage used out of total FPS pool
        percentage_used = (total_subsidy_used / fps_value) * 100 if fps_value > 0 else 0
        
        # Calculate subsidy usage by income group
        subsidy_by_income = {}
        income_levels = ['low', 'middle', 'high']
        
        for income_level in income_levels:
            # SQL query joining SubsidyUsageLog with CommuterInfoLog to get income levels
            query = text("""
                SELECT SUM(sul.subsidy_amount) as income_total
                FROM subsidy_usage_log sul
                JOIN commuter_info_log cil ON sul.commuter_id = cil.commuter_id
                WHERE cil.income_level = :income_level
            """)
                
            result = session.execute(query, {"income_level": income_level}).fetchone()
            amount = float(result.income_total) if result and result.income_total else 0
            
            # Calculate percentage of total used subsidy
            percentage_of_used = (amount / total_subsidy_used) * 100 if total_subsidy_used > 0 else 0
            
            subsidy_by_income[income_level] = {
                'amount': amount,
                'percentage_of_used': percentage_of_used,
                'percentage_of_total_fps': (amount / fps_value) * 100 if fps_value > 0 else 0
            }
            
        return {
            'fps_value': fps_value,
            'total_subsidy_used': total_subsidy_used,
            'percentage_used': percentage_used,
            'subsidy_by_income': subsidy_by_income
        }
        
    except Exception as e:
        print(f"Error calculating subsidy usage statistics: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Return default structure
        default_results = {
            'fps_value': fps_value,
            'total_subsidy_used': 0,
            'percentage_used': 0,
            'subsidy_by_income': {
                'low': {'amount': 0, 'percentage_of_used': 0, 'percentage_of_total_fps': 0},
                'middle': {'amount': 0, 'percentage_of_used': 0, 'percentage_of_total_fps': 0},
                'high': {'amount': 0, 'percentage_of_used': 0, 'percentage_of_total_fps': 0}
            }
        }
        return default_results

def run_single_simulation(params):
    """Run a single simulation and calculate equity metrics"""
    pid = os.getpid()
    print(f"Starting simulation with PID {pid} - FPS: {params.get('fps_value', 'N/A')}")
        # ADD THESE TWO LINES HERE - seed initialization for consistent results
    np.random.seed(pid + int(params.get('fps_value', 0)))
    random.seed(pid + int(params.get('fps_value', 0)))
    # Create a unique schema name for this process
    schema_name = f"sim_{pid}_{int(time.time())}"
    
    # Initialize session and engine variables outside try block
    session = None
    engine = None
    
    try:
        # PostgreSQL connection string with appropriate credentials
        db_connection_string = f"postgresql://z5247491@localhost:15432/postgres"
        
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
        
        # Calculate equity metrics from data in the current schema
        # Modify calculate_equity_indicator to support schema if needed
        results = calculate_equity_indicator(session, schema=schema_name)
        # Add this right after calculating equity indicators in run_single_simulation
        subsidy_stats = calculate_subsidy_usage_statistics(session, fps_value, schema=schema_name)
        results['subsidy_usage'] = subsidy_stats
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
                'total_equity_indicator': 0,
                'low': {'equity_indicator': 0, 'mode_shares': {}},
                'middle': {'equity_indicator': 0, 'mode_shares': {}},
                'high': {'equity_indicator': 0, 'mode_shares': {}}
            }
        
    except Exception as e:
        print(f"Error in simulation {pid}: {str(e)}")
        traceback.print_exc()
        # Return a default result structure to avoid NoneType errors
        return {
            'fps_value': fps_value if 'fps_value' in locals() else 0,
            'fixed_allocations': fixed_allocations if 'fixed_allocations' in locals() else None,
            'total_equity_indicator': 0,
            'low': {'equity_indicator': 0, 'mode_shares': {}},
            'middle': {'equity_indicator': 0, 'mode_shares': {}},
            'high': {'equity_indicator': 0, 'mode_shares': {}}
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
        
        print(f"Completed FPS={fps_value}, Equity={result['avg_equity']:.4f}")
        
        # Save partial results after each FPS completes
        with open(f"partial_results_fps_{fps_value}_{time.strftime('%Y%m%d_%H%M%S')}.pkl", "wb") as f:
            pickle.dump(results, f)
    
    return results

def optimize_allocation_parallel(fps_value, base_parameters, num_cpus=NUM_CPUS):
    """
    Optimize subsidy allocation using parallel simulations within a single optimization run.
    """
    print(f"\nOptimizing subsidy allocation for FPS = {fps_value} with {num_cpus} parallel simulations")
    
    # Define bounds and parameters (same as your current code)
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
    
    # Set up parameter space and sampling (same as your current code)
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
    target_equity_threshold = 0.05
    no_improvement_iterations = 0
    patience = 6
    
    best_equity = float('inf') 
    best_allocations = None
    best_results = None
    previous_best = float('inf')

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
                
                if best_equity < target_equity_threshold:
                    print(f"Early termination: Target equity threshold {target_equity_threshold} reached!")
                    return {
                        'fps_value': fps_value,
                        'optimal_allocations': best_allocations,
                        'equity_scores': {level: best_results[level]['equity_indicator'] for level in ['low', 'middle', 'high']},
                        'avg_equity': best_equity,
                        'full_results': best_results,
                        'subsidy_usage': best_results.get('subsidy_usage', {}),  # Add this line
                        'terminated_early': True,
                        'termination_reason': 'Target threshold reached'
                    }
        else:
            y_init.append(float('inf'))

    # Setup Gaussian Process
    kernel = Matern(nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42)
    
    X = X_init
    y = np.array(y_init)
    
    # Run optimization iterations
    max_iterations = 10
    
    for iteration in range(max_iterations):
        previous_best = best_equity
        
        # Fit GP to current data
        gp.fit(X, y)
        
        # Generate candidates for Expected Improvement
        candidates = sampler.random(n=500)
        candidates = qmc.scale(candidates, lower_bounds, upper_bounds)
        
        # Calculate EI for all candidates (minimizing, not maximizing)
        ei_values = []
        for candidate in candidates:
            mu, sigma = gp.predict(candidate.reshape(1, -1), return_std=True)
            mu = mu.reshape(-1)
            sigma = sigma.reshape(-1)
            
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
                equity_indicator = result.get('total_equity_indicator', float('inf'))
                
                # Add to GP data
                X = np.vstack([X, top_candidates[i]])
                y = np.append(y, equity_indicator)
                
                improvement = previous_best - equity_indicator
                
                if equity_indicator < best_equity:
                    best_equity = equity_indicator
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
                    print(f"Iteration {iteration + 1}, New best equity indicator: {best_equity:.6f}, Improvement: {improvement:.6f}")
                    
                    if improvement > min_improvement_threshold:
                        no_improvement_iterations = 0
                    else:
                        no_improvement_iterations += 1
                        print(f"  Minimal improvement: {improvement:.6f} < {min_improvement_threshold}")
                    
                    if best_equity < target_equity_threshold:
                        print(f"Early termination after {iteration + 1} iterations: Target equity threshold reached!")
                        return {
                            'fps_value': fps_value,
                            'optimal_allocations': best_allocations,
                            'equity_scores': {level: best_results[level]['equity_indicator'] for level in ['low', 'middle', 'high']},
                            'avg_equity': best_equity,
                            'full_results': best_results,
                            'subsidy_usage': best_results.get('subsidy_usage', {}),  # Add this line
                            'terminated_early': True,
                            'termination_reason': 'Target threshold reached'
                        }
        
        if not found_improvement:
            print(f"Iteration {iteration + 1}, No improvement. Current best: {best_equity:.6f}")
            no_improvement_iterations += 1
        
        if no_improvement_iterations >= patience:
            print(f"Early termination after {iteration + 1} iterations: No significant improvement for {patience} consecutive iterations")
            return {
                'fps_value': fps_value,
                'optimal_allocations': best_allocations,
                'equity_scores': {level: best_results[level]['equity_indicator'] for level in ['low', 'middle', 'high']},
                'avg_equity': best_equity,
                'full_results': best_results,
                'subsidy_usage': best_results.get('subsidy_usage', {}),
                'terminated_early': True,
                'termination_reason': 'Convergence'
            }

    return {
        'fps_value': fps_value,
        'optimal_allocations': best_allocations,
        'equity_scores': {level: best_results[level]['equity_indicator'] for level in ['low', 'middle', 'high']},
        'avg_equity': best_equity,
        'full_results': best_results,
        'subsidy_usage': best_results.get('subsidy_usage', {}),
        'terminated_early': False
    }

def visualize_results(results, output_dir):
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
        #     z = np.polyfit(df['fps_value'], df['avg_equity'], 3)
        #     p = np.poly1d(z)
        #     x_smooth = np.linspace(df['fps_value'].min(), df['fps_value'].max(), 100)
        #     y_smooth = p(x_smooth)
        #     plt.plot(x_smooth, y_smooth, '--', color='gray', alpha=0.6, label='Trend Line')
            
        #     # Find and mark optimal point
        #     optimal_idx = np.argmin(y_smooth)
        #     optimal_fps = x_smooth[optimal_idx]
        #     optimal_equity = y_smooth[optimal_idx]
            
        #     plt.scatter([optimal_fps], [optimal_equity], 
        #                 marker='*', s=300, color='red', zorder=10,
        #                 label=f'Optimal Point (FPS={optimal_fps:.0f})')
            
        #     # Annotate optimal point
        #     plt.annotate(f'Optimal: Equity={optimal_equity:.4f}',
        #                 xy=(optimal_fps, optimal_equity),
        #                 xytext=(10, -30), 
        #                 textcoords='offset points',
        #                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        #                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))
        # except Exception as e:
        #     print(f"Error fitting trend line: {e}")
        # Use LOWESS smoothing for more robust trend
            from scipy import stats
            x = df['fps_value']
            y = df['avg_equity']
            z = np.polyfit(np.log(x), y, 2)  # Fit against log of FPS
            p = np.poly1d(z)
            x_smooth = np.geomspace(df['fps_value'].min(), df['fps_value'].max(), 100)
            y_smooth = p(np.log(x_smooth))
            
            # Add smoother constraints
            y_smooth = np.clip(y_smooth, 0, y.max()*1.2)
            
            plt.plot(x_smooth, y_smooth, '--', color='gray', alpha=0.6, label='Trend Line')
            # Rest of optimal point finding code...
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
                for mode in ['bike', 'car', 'MaaS_Bundle', 'public', 'walk']:
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
    plt.ylabel('Equity Score (Lower is Better)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='best')
    
    # Add vertical line at the FPS value with highest equity score
    if not df.empty:
        best_fps = df.loc[df['avg_equity'].idxmin()]['fps_value']  # PROBLEM: Should be idxmin() for minimization
        plt.axvline(x=best_fps, color='red', linestyle='--', alpha=0.5,
                label=f'Best FPS: {best_fps}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'equity_vs_fps_with_allocations.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Create detailed allocation line graphs for each income level
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
            best_fps = df.loc[df['avg_equity'].idxmin()]['fps_value']
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
        optimal_row = df.loc[df['avg_equity'].idxmin()]
        
        with open(os.path.join(output_dir, 'optimal_solution.txt'), 'w') as f:
            f.write(f"Optimal FPS Value: {optimal_row['fps_value']}\n")
            f.write(f"Optimal Sum Equity Score: {optimal_row['avg_equity']:.4f}\n\n")
            f.write("Optimal Subsidy Allocations:\n")
            
            for key in sorted([k for k in optimal_row.index if k.startswith('alloc_')]):
                f.write(f"  {key.replace('alloc_', '')}: {optimal_row[key]:.4f}\n")
                
            f.write("\nEquity Score by Income Level:\n")
            for level in ['low', 'middle', 'high']:
                f.write(f"  {level.capitalize()}: {optimal_row[f'equity_{level}']:.4f}\n")

def visualize_subsidy_usage(results, output_dir):
    """
    Create visualizations for subsidy usage statistics.
    
    Args:
        results: Dictionary of results from optimization
        output_dir: Directory to save visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    # Add this debugging at the beginning
    print("\nStarting subsidy usage visualization...")
    print(f"Number of results entries: {len(results)}")
    
    # Extract subsidy usage data from results
    rows = []
    for fps, result in results.items():
        print(f"Checking result for FPS {fps}...")
        if 'subsidy_usage' in result:
            print(f"  Found subsidy_usage for FPS {fps}")
            subsidy_usage = result['subsidy_usage']
            print(f"  Total subsidy used: {subsidy_usage['total_subsidy_used']}")
            print(f"  Percentage used: {subsidy_usage['percentage_used']}%")
            row = {
                'fps_value': fps,
                'total_subsidy_used': subsidy_usage['total_subsidy_used'],
                'percentage_used': subsidy_usage['percentage_used']
            }
            
            # Add subsidy usage by income level
            for income_level, stats in subsidy_usage['subsidy_by_income'].items():
                row[f'{income_level}_amount'] = stats['amount']
                row[f'{income_level}_percentage_of_used'] = stats['percentage_of_used']
                row[f'{income_level}_percentage_of_total_fps'] = stats['percentage_of_total_fps']
            
            rows.append(row)
        else:
            print(f"  No subsidy_usage data for FPS {fps}")
    if not rows:
        print("No valid subsidy usage data to visualize")
        return
        
    df = pd.DataFrame(rows)
    
    # Ensure FPS values are numeric and sort
    df['fps_value'] = pd.to_numeric(df['fps_value'])
    df = df.sort_values('fps_value')
    
    # 1. Plot total subsidy usage percentage
    plt.figure(figsize=(12, 6))
    plt.plot(df['fps_value'], df['percentage_used'], 'o-', linewidth=2, color='green')
    plt.title('Percentage of FPS Budget Used', fontsize=16)
    plt.xlabel('Fixed Pool Subsidy (FPS)', fontsize=14)
    plt.ylabel('Percentage Used (%)', fontsize=14)
    plt.xscale('log')  # Log scale for FPS values
    plt.grid(True, alpha=0.3)
    
    # Add annotations for exact percentages
    for i, row in df.iterrows():
        plt.annotate(f"{row['percentage_used']:.1f}%",
                    xy=(row['fps_value'], row['percentage_used']),
                    xytext=(5, 5),
                    textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'subsidy_usage_percentage.png'), dpi=300)
    plt.close()
    
    # 2. Plot absolute subsidy usage with bars
    plt.figure(figsize=(14, 8))
    bar_width = 0.35
    x = np.arange(len(df['fps_value']))
    
    plt.bar(x - bar_width/2, df['fps_value'], bar_width, label='Total FPS Available', alpha=0.6)
    plt.bar(x + bar_width/2, df['total_subsidy_used'], bar_width, label='Total Subsidy Used', alpha=0.8)
    
    # Set x-ticks at bar positions
    plt.xticks(x, [f"{val:,.0f}" for val in df['fps_value']])
    
    plt.title('Subsidy Usage: Available vs. Used', fontsize=16)
    plt.xlabel('Fixed Pool Subsidy (FPS)', fontsize=14)
    plt.ylabel('Subsidy Amount', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend()
    
    # Add text labels on bars
    for i, val in enumerate(df['total_subsidy_used']):
        plt.text(i + bar_width/2, val + max(df['fps_value'])*0.01, 
                f"{val:,.0f}", ha='center', va='bottom', fontsize=9)
        
    for i, val in enumerate(df['percentage_used']):
        plt.text(i, df['fps_value'].iloc[i] + max(df['fps_value'])*0.01, 
                f"{val:.1f}%", ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'subsidy_usage_absolute.png'), dpi=300)
    plt.close()
    
    # 3. Plot subsidy distribution by income group (percentage of used subsidy)
    plt.figure(figsize=(12, 8))
    
    income_levels = ['low', 'middle', 'high']
    colors = {'low': '#1f77b4', 'middle': '#ff7f0e', 'high': '#2ca02c'}
    
    for income in income_levels:
        plt.plot(df['fps_value'], df[f'{income}_percentage_of_used'], 'o-', 
                label=f'{income.capitalize()} Income', 
                color=colors[income], linewidth=2)
    
    plt.title('Distribution of Used Subsidy by Income Group', fontsize=16)
    plt.xlabel('Fixed Pool Subsidy (FPS)', fontsize=14)
    plt.ylabel('Percentage of Used Subsidy (%)', fontsize=14)
    plt.xscale('log')  # Log scale for FPS values
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'subsidy_distribution_by_income.png'), dpi=300)
    plt.close()
    
    # 4. Create stacked area chart to show relative proportions
    plt.figure(figsize=(12, 8))
    plt.stackplot(df['fps_value'], 
                [df[f'{income}_percentage_of_used'] for income in income_levels],
                labels=[f'{income.capitalize()} Income' for income in income_levels],
                colors=[colors[income] for income in income_levels],
                alpha=0.7)
    
    plt.title('Relative Distribution of Used Subsidy by Income Group', fontsize=16)
    plt.xlabel('Fixed Pool Subsidy (FPS)', fontsize=14)
    plt.ylabel('Percentage of Used Subsidy (%)', fontsize=14)
    plt.xscale('log')  # Log scale for FPS values
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'subsidy_distribution_stacked.png'), dpi=300)
    plt.close()
    
    # Save data table for reference
    subsidy_table = df[['fps_value', 'percentage_used', 'total_subsidy_used'] + 
                     [f'{income}_{metric}' for income in income_levels 
                      for metric in ['amount', 'percentage_of_used', 'percentage_of_total_fps']]]
    subsidy_table.to_csv(os.path.join(output_dir, 'subsidy_usage_statistics.csv'), index=False)

def main():
    """Main function to run the complete equity optimization analysis"""
    # Define base parameters
    print(f"Starting parallel optimization with {NUM_CPUS} CPUs")

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
            'S_base': 0.02,# Base surcharge (10%)
            'alpha': 0.10,# Sensitivity coefficient
            'delta': 0.5 # Reduction factor for subscription model
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
    
    # Define FPS values to analyze (using a logarithmic scale for better coverage)
    # Start with a smaller set of values for testing
    # Replace your current fps_values with this more focused sampling
    fps_values = [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 6000, 7000, 8000]
    # For actual analysis, use a more comprehensive set
    # fps_values = np.logspace(0, 5, 10).astype(int)  # 10 points from 1 to 100,000 on log scale

    # Run parallel optimization
    results = run_sequential_fps_optimization(fps_values, base_parameters, num_cpus=NUM_CPUS)
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"mae_equity_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save individual FPS results
    for fps, result in results.items():
        with open(f"{output_dir}/fps_{fps}_result.pkl", "wb") as f:
            pickle.dump(result, f)
    
    # Save combined results
    with open(f"{output_dir}/all_results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    # Identify best FPS
    best_fps = min(results.items(), key=lambda x: x[1]['avg_equity'])[0]
    print(f"\nOptimal FPS value: {best_fps}")
    print(f"Optimal equity score: {results[best_fps]['avg_equity']:.4f}")
    
    # Create visualizations
    visualize_results(results, output_dir)
    visualize_subsidy_usage(results, output_dir)
if __name__ == "__main__":
    main()