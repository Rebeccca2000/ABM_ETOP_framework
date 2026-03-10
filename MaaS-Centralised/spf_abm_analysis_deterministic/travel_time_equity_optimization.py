from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm, qmc
import argparse
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
try:
    from spf_abm_analysis_deterministic.deterministic_scenarios import (
        build_scenario,
        save_scenario,
    )
except ModuleNotFoundError:
    from deterministic_scenarios import build_scenario, save_scenario

# Constants
SIMULATION_STEPS = 144  # Reduced for faster runs during optimization
NUM_CPUS = 10  # Adjust based on your system

def calculate_travel_time_equity(session, schema=None):
    """
    Calculate travel time equity using the deviation index approach.
    This measures how much each income group's average travel time deviates from the overall average.
    
    Args:
        session: SQLAlchemy session with simulation data
        schema: Optional schema name for PostgreSQL
        
    Returns:
        dict: Results containing travel time equity indicators
    """
    results = {}
    income_levels = ['low', 'middle', 'high']
    
    try:
        # Set schema if provided
        if schema:
            print(f"Setting search path to schema: {schema}")
            session.execute(text(f"SET search_path TO {schema}"))
            session.commit()
            
            # Verify tables exist in the schema
            table_check = session.execute(text(
                f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{schema}'"
            )).fetchall()
            print(f"Tables in schema {schema}: {[row[0] for row in table_check]}")
        
        # Initialize variables to store results
        travel_times_by_income = {}
        trip_counts_by_income = {}
        total_travel_time = 0
        total_trips = 0
        
        # Query travel times by income level
        for income_level in income_levels:
            if schema:
                # Use text-based SQL when schema is provided
                query = text("""
                    SELECT sbl.total_time, COUNT(sbl.request_id) as trip_count
                    FROM service_booking_log sbl 
                    JOIN commuter_info_log cil ON sbl.commuter_id = cil.commuter_id
                    WHERE cil.income_level = :income_level AND sbl.total_time IS NOT NULL
                    GROUP BY sbl.total_time
                """)
                travel_times = session.execute(query, {"income_level": income_level}).fetchall()
            else:
                # Use ORM query for default schema
                travel_times = session.query(
                    ServiceBookingLog.total_time,
                    func.count(ServiceBookingLog.request_id).label('trip_count')
                ).join(
                    CommuterInfoLog,
                    ServiceBookingLog.commuter_id == CommuterInfoLog.commuter_id
                ).filter(
                    CommuterInfoLog.income_level == income_level,
                    ServiceBookingLog.total_time != None
                ).group_by(
                    ServiceBookingLog.total_time
                ).all()
            
            # Calculate total travel time and trip count for this income level
            income_total_time = sum(time * count for time, count in travel_times)
            income_trip_count = sum(count for _, count in travel_times)
            
            # Store results
            travel_times_by_income[income_level] = income_total_time
            trip_counts_by_income[income_level] = income_trip_count
            
            # Add to overall totals
            total_travel_time += income_total_time
            total_trips += income_trip_count
        
        # Calculate average travel times
        if total_trips > 0:
            overall_avg_travel_time = total_travel_time / total_trips
            
            avg_travel_times = {}
            for income_level in income_levels:
                if trip_counts_by_income[income_level] > 0:
                    avg_travel_times[income_level] = travel_times_by_income[income_level] / trip_counts_by_income[income_level]
                else:
                    avg_travel_times[income_level] = 0
            
            # Calculate travel time deviation index
            travel_time_deviations = {}
            total_deviation = 0
            
            for income_level in income_levels:
                deviation = abs(avg_travel_times[income_level] - overall_avg_travel_time)
                travel_time_deviations[income_level] = deviation
                total_deviation += deviation
            
            # Calculate average deviation (normalized by number of income groups)
            avg_deviation = total_deviation / len(income_levels)
            
            # Store detailed results
            for income_level in income_levels:
                results[income_level] = {
                    'avg_travel_time': avg_travel_times[income_level],
                    'deviation': travel_time_deviations[income_level],
                    'trip_count': trip_counts_by_income[income_level]
                }
            
            # Store overall results
            results['overall_avg_travel_time'] = overall_avg_travel_time
            results['total_travel_time'] = total_travel_time
            results['total_trips'] = total_trips
            results['total_deviation'] = total_deviation
            results['travel_time_equity_index'] = total_deviation  # This is our main optimization metric
            results['avg_deviation'] = avg_deviation
            
            return results
        else:
            print("Warning: No valid trips found in the data")
            # Return default structure
            default_results = {}
            for income_level in income_levels:
                default_results[income_level] = {
                    'avg_travel_time': 0,
                    'deviation': 0,
                    'trip_count': 0
                }
            
            default_results['overall_avg_travel_time'] = 0
            default_results['total_travel_time'] = 0
            default_results['total_trips'] = 0
            default_results['total_deviation'] = 0
            default_results['travel_time_equity_index'] = 0
            default_results['avg_deviation'] = 0
            
            return default_results
            
    except Exception as e:
        print(f"Error in calculate_travel_time_equity: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Return a default structure to avoid NoneType errors
        default_results = {}
        for income_level in income_levels:
            default_results[income_level] = {
                'avg_travel_time': 0,
                'deviation': 0,
                'trip_count': 0
            }
        
        default_results['overall_avg_travel_time'] = 0
        default_results['total_travel_time'] = 0
        default_results['total_trips'] = 0
        default_results['total_deviation'] = 0
        default_results['travel_time_equity_index'] = 0
        default_results['avg_deviation'] = 0
        
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
    """Run a single simulation and calculate travel time equity metrics (ModeShare-plumbing compatible)."""
    pid = os.getpid()

    fps_for_seed = int(params.get("fps_value", 0))
    seed = int(params.get("seed", 0))
    simulation_id = int(params.get("simulation_id", 0))
    bo_iteration = int(params.get("bo_iteration", 0))
    base_seed = int(params.get("base_seed", 12345))
    deterministic_mode = bool(params.get("deterministic_mode", False))

    if deterministic_mode:
        # In deterministic mode, keep seed stream independent of FPS so scenarios are paired.
        final_seed = base_seed + 10_000_000 * seed + 1_000 * bo_iteration + simulation_id
    else:
        final_seed = (
            base_seed
            + 10_000_000 * seed
            + 10_000 * fps_for_seed
            + 1_000 * bo_iteration
            + simulation_id
        )

    print(
        f"Starting simulation PID {pid} | FPS={fps_for_seed} | "
        f"seed={seed} | bo_iter={bo_iteration} | sim_id={simulation_id} | "
        f"final_seed={final_seed} | deterministic_mode={deterministic_mode}"
    )

    np.random.seed(final_seed)
    random.seed(final_seed)
    try:
        import torch
        torch.manual_seed(final_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(final_seed)
    except Exception:
        pass

    schema_name = f"sim_{pid}_{int(time.time())}"
    session = None
    engine = None

    # Keep metadata (requested contract)
    _meta = {
        "simulation_id": simulation_id,
        "bo_iteration": bo_iteration,
        "seed": seed,
        "base_seed": base_seed,
        "final_seed": final_seed,
    }

    try:
        # PostgreSQL connection string with environment-aware defaults (HPC seed-array safe)
        db_port = int(os.environ.get("PGPORT", "15433"))
        db_user = os.environ.get("PGUSER", "z5247491")
        db_host = os.environ.get("PGHOST", "localhost")
        db_name = os.environ.get("PGDATABASE", "postgres")
        db_connection_string = f"postgresql://{db_user}@{db_host}:{db_port}/{db_name}"

        # Extract analysis parameters (do NOT mutate base_parameters permanently)
        fps_value = float(params.pop("fps_value", 0))
        fixed_allocations = params.pop("fixed_allocations", None)
        simulation_steps = int(params.pop("simulation_steps", SIMULATION_STEPS))
        deterministic_mode = bool(params.pop("deterministic_mode", deterministic_mode))
        deterministic_scenario_path = params.pop("deterministic_scenario_path", None)
        deterministic_scenario = params.pop("deterministic_scenario", None)

        # Remove plumbing keys from model kwargs
        params.pop("seed", None)
        params.pop("base_seed", None)
        params.pop("bo_iteration", None)
        params.pop("simulation_id", None)

        # Pilot early-stop controls
        pilot_steps = params.pop("pilot_steps", None)
        best_obj = params.pop("early_stop_best_equity", None)  # naming kept for plumbing parity
        margin = float(params.pop("early_stop_margin", 0.20))
        exhaust_frac = float(params.pop("early_stop_exhaust_frac", 0.90))

        engine = create_engine(db_connection_string)

        # Create schema
        with engine.connect() as connection:
            connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema_name}"))
            connection.commit()

        from sqlalchemy.orm import sessionmaker
        Session = sessionmaker(bind=engine)
        session = Session()

        # Set search_path
        with engine.connect() as connection:
            connection.execute(text(f"SET search_path TO {schema_name}"))
            connection.commit()

        # Reset DB tables inside schema
        reset_db_params = {k: params[k] for k in [
            "uber_like1_capacity", "uber_like1_price",
            "uber_like2_capacity", "uber_like2_price",
            "bike_share1_capacity", "bike_share1_price",
            "bike_share2_capacity", "bike_share2_price"
        ]}
        reset_database(engine=engine, session=session, schema=schema_name, **reset_db_params)

        # Subsidy config + dataset (same shape as Mode Share)
        if fixed_allocations:
            subsidy_dataset = {}
            for income_level in ["low", "middle", "high"]:
                subsidy_dataset[income_level] = {}
                for mode in ["bike", "car", "MaaS_Bundle", "public", "walk"]:
                    key = f"{income_level}_{mode}"
                    subsidy_dataset[income_level][mode] = float(fixed_allocations.get(key, 0.1))
            params["subsidy_dataset"] = subsidy_dataset

        params["subsidy_config"] = SubsidyPoolConfig("daily", float(fps_value))
        params["db_connection_string"] = db_connection_string
        params["schema"] = schema_name
        params["deterministic_mode"] = deterministic_mode
        if deterministic_scenario_path:
            params["deterministic_scenario_path"] = deterministic_scenario_path
        if deterministic_scenario is not None:
            params["deterministic_scenario"] = deterministic_scenario

        if deterministic_mode and not deterministic_scenario_path and deterministic_scenario is None:
            raise ValueError(
                "deterministic_mode=True requires deterministic_scenario_path or deterministic_scenario"
            )

        model = MobilityModel(**params)

        # Dominance-based pilot early-stop (same logic, but objective = travel_time_equity_index)
        if pilot_steps is not None and best_obj is not None and np.isfinite(best_obj):
            model.run_model(int(pilot_steps))

            pilot_results = calculate_travel_time_equity(session, schema=schema_name)
            pilot_metric = float(pilot_results.get("travel_time_equity_index", float("inf")))

            pilot_subsidy = calculate_subsidy_usage_statistics(session, fps_value, schema=schema_name)
            used = float(pilot_subsidy.get("total_subsidy_used", 0.0))
            exhausted_early = (fps_value > 0) and (used >= exhaust_frac * float(fps_value))

            if exhausted_early and (pilot_metric > float(best_obj) * (1.0 + margin)):
                return {
                    "fps_value": float(fps_value),
                    "schema_name": schema_name,
                    "fixed_allocations": fixed_allocations,
                    "terminated_early": True,
                    "termination_reason": (
                        f"pilot_dominated_exhausted (used={used:.2f}, "
                        f"pilot_metric={pilot_metric:.6f}, best={float(best_obj):.6f})"
                    ),
                    "subsidy_usage": pilot_subsidy,
                    "travel_time_equity_index": float("inf"),
                    "low": {"avg_travel_time": float("inf"), "deviation": float("inf")},
                    "middle": {"avg_travel_time": float("inf"), "deviation": float("inf")},
                    "high": {"avg_travel_time": float("inf"), "deviation": float("inf")},
                    **_meta,
                }

            remaining_steps = int(simulation_steps) - int(pilot_steps)
            if remaining_steps > 0:
                model.run_model(remaining_steps)
        else:
            model.run_model(simulation_steps)

        # Final metric + subsidy usage
        results = calculate_travel_time_equity(session, schema=schema_name)
        subsidy_stats = calculate_subsidy_usage_statistics(session, fps_value, schema=schema_name)
        results["subsidy_usage"] = subsidy_stats

        # Attach standard fields (Mode Share compatible)
        results["fps_value"] = float(fps_value)
        results["schema_name"] = schema_name
        results["fixed_allocations"] = fixed_allocations
        results["terminated_early"] = bool(results.get("terminated_early", False))
        results["termination_reason"] = str(results.get("termination_reason", ""))
        results.update(_meta)

        return results

    except Exception as e:
        print(f"Error in simulation PID {pid}: {str(e)}")
        traceback.print_exc()
        return {
            "fps_value": float(params.get("fps_value", 0)),
            "schema_name": schema_name,
            "fixed_allocations": params.get("fixed_allocations", None),
            "terminated_early": False,
            "termination_reason": str(e),
            "subsidy_usage": {},
            "travel_time_equity_index": 0.0,
            "low": {"avg_travel_time": 0.0, "deviation": 0.0},
            "middle": {"avg_travel_time": 0.0, "deviation": 0.0},
            "high": {"avg_travel_time": 0.0, "deviation": 0.0},
            **_meta,
        }
    finally:
        if session:
            session.close()
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
    write_partial_results = os.environ.get("WRITE_PARTIAL_RESULTS", "0") == "1"
    partial_results_dir = os.environ.get("PARTIAL_RESULTS_DIR", "")
    if write_partial_results and partial_results_dir:
        os.makedirs(partial_results_dir, exist_ok=True)

    # Sort FPS values to ensure sequential processing in ascending order
    fps_values = sorted(fps_values)
    
    for fps_index, fps_value in enumerate(fps_values):
        print(f"\n[{fps_index+1}/{len(fps_values)}] Starting optimization for FPS={fps_value}")
        
        # Run parallel-enabled optimization for this FPS value
        result = optimize_allocation_parallel(fps_value, base_parameters, num_cpus)
        results[float(fps_value)] = result
        
        print(f"Completed FPS={fps_value}, Travel Time Equity Index={result['travel_time_equity_index']:.4f}")
        
        # Optional checkpoints (disabled by default to avoid home quota pressure on HPC)
        if write_partial_results:
            partial_name = f"partial_results_fps_{fps_value}_{time.strftime('%Y%m%d_%H%M%S')}.pkl"
            partial_path = os.path.join(partial_results_dir, partial_name) if partial_results_dir else partial_name
            try:
                with open(partial_path, "wb") as f:
                    pickle.dump(results, f)
            except OSError as e:
                print(f"[WARN] Failed to write partial results to {partial_path}: {e}")

    return results

def optimize_allocation_parallel(fps_value, base_parameters, num_cpus=NUM_CPUS):
    """
    ModeShare-plumbing compatible BO wrapper.
    Objective: minimize travel_time_equity_index (lower is better).
    """
    print(f"\nOptimizing subsidy allocation for FPS = {fps_value} with {num_cpus} parallel simulations")

    param_names = [
        "low_bike", "low_car", "low_MaaS_Bundle", "low_public",
        "middle_bike", "middle_car", "middle_MaaS_Bundle", "middle_public",
        "high_bike", "high_car", "high_MaaS_Bundle", "high_public"
    ]

    # Match reference (Mode Share) bounds style
    bounds = {p: (0.0, 0.8) for p in param_names}

    lower_bounds = [bounds[p][0] for p in param_names]
    upper_bounds = [bounds[p][1] for p in param_names]

    sampler = qmc.LatinHypercube(d=len(param_names), seed=42)
    X_init = sampler.random(n=8)
    X_init = qmc.scale(X_init, lower_bounds, upper_bounds)

    min_improvement_threshold = 0.01
    target_equity_threshold = 0.05
    no_improvement_iterations = 0
    patience = 6

    best_obj = float("inf")
    best_allocations = None
    best_results = None
    previous_best = float("inf")

    # ---- Initial LHS evaluations (bo_iteration=0) ----
    initial_param_sets = []
    for i, x in enumerate(X_init):
        allocations = {k: float(v) for k, v in zip(param_names, x)}

        sim_params = base_parameters.copy()
        sim_params["fps_value"] = float(fps_value)
        sim_params["fixed_allocations"] = allocations
        sim_params["simulation_steps"] = int(SIMULATION_STEPS)
        sim_params["simulation_id"] = i
        sim_params["bo_iteration"] = 0
        sim_params["seed"] = int(base_parameters.get("seed", 0))
        sim_params["base_seed"] = int(base_parameters.get("base_seed", 12345))
        initial_param_sets.append(sim_params)

    print(f"Running {len(initial_param_sets)} initial simulations in parallel with {num_cpus} CPUs")
    with mp.Pool(processes=min(num_cpus, len(initial_param_sets))) as pool:
        initial_results = list(pool.map(run_single_simulation, initial_param_sets))

    y_init = []
    for i, r in enumerate(initial_results):
        if isinstance(r, dict):
            obj = float(r.get("travel_time_equity_index", float("inf")))
            y_init.append(obj)

            if obj < best_obj:
                best_obj = obj
                best_allocations = {k: float(v) for k, v in zip(param_names, X_init[i])}
                best_results = r
                print(f"New best travel_time_equity_index from initial points: {best_obj:.6f}")

                if best_obj < target_equity_threshold:
                    return {
                        "fps_value": float(fps_value),
                        "optimal_allocations": best_allocations,
                        "equity_scores": {
                            "low": float(best_results.get("low", {}).get("deviation", np.nan)),
                            "middle": float(best_results.get("middle", {}).get("deviation", np.nan)),
                            "high": float(best_results.get("high", {}).get("deviation", np.nan)),
                        },
                        "deviations": {
                            "low": float(best_results.get("low", {}).get("deviation", np.nan)),
                            "middle": float(best_results.get("middle", {}).get("deviation", np.nan)),
                            "high": float(best_results.get("high", {}).get("deviation", np.nan)),
                        },
                        "avg_equity": float(best_obj),
                        "travel_time_equity_index": float(best_obj),
                        "full_results": best_results,
                        "subsidy_usage": best_results.get("subsidy_usage", {}),
                        "deterministic_mode": deterministic_mode,
                        "deterministic_scenario_path": deterministic_scenario_path or "",
                        "terminated_early": True,
                        "termination_reason": "Target threshold reached",
                    }
        else:
            y_init.append(float("inf"))

    kernel = Matern(nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42)

    X = X_init
    y = np.array(y_init, dtype=float)

    max_iterations = 3

    # ---- EI loop ----
    for iteration in range(max_iterations):
        previous_best = best_obj

        gp.fit(X, y)

        candidates = sampler.random(n=200)
        candidates = qmc.scale(candidates, lower_bounds, upper_bounds)

        ei_values = []
        for c in candidates:
            mu, sigma = gp.predict(c.reshape(1, -1), return_std=True)
            mu = mu.reshape(-1)
            sigma = sigma.reshape(-1)
            imp = np.min(y) - mu
            Z = imp / (sigma + 1e-9)
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei_values.append(float(ei.item()) if hasattr(ei, "item") else float(ei))

        top_n = min(num_cpus, len(candidates))
        top_indices = np.argsort(ei_values)[-top_n:]
        top_candidates = [candidates[i] for i in top_indices]

        iteration_param_sets = []
        for j, c in enumerate(top_candidates):
            allocations = {k: float(v) for k, v in zip(param_names, c)}

            sim_params = base_parameters.copy()
            sim_params["fps_value"] = float(fps_value)
            sim_params["fixed_allocations"] = allocations
            sim_params["simulation_steps"] = int(SIMULATION_STEPS)
            sim_params["simulation_id"] = j
            sim_params["bo_iteration"] = iteration + 1
            sim_params["seed"] = int(base_parameters.get("seed", 0))
            sim_params["base_seed"] = int(base_parameters.get("base_seed", 12345))

            # Pilot early-stop fields (same keys as reference)
            sim_params["pilot_steps"] = 20
            sim_params["early_stop_best_equity"] = float(best_obj)
            sim_params["early_stop_margin"] = 0.20
            sim_params["early_stop_exhaust_frac"] = 0.90

            iteration_param_sets.append(sim_params)

        print(f"Iteration {iteration+1}: Running {len(iteration_param_sets)} simulations in parallel")
        with mp.Pool(processes=min(num_cpus, len(iteration_param_sets))) as pool:
            sim_results = list(pool.map(run_single_simulation, iteration_param_sets))

        found_improvement = False
        for j, r in enumerate(sim_results):
            if not isinstance(r, dict):
                continue

            obj = float(r.get("travel_time_equity_index", float("inf")))

            X = np.vstack([X, top_candidates[j]])
            y = np.append(y, obj)

            improvement = previous_best - obj

            if obj < best_obj:
                best_obj = obj
                best_allocations = {k: float(v) for k, v in zip(param_names, top_candidates[j])}
                best_results = r
                found_improvement = True
                print(
                    f"Iteration {iteration+1}, New best travel_time_equity_index: "
                    f"{best_obj:.6f}, Improvement: {improvement:.6f}"
                )

                if improvement > min_improvement_threshold:
                    no_improvement_iterations = 0
                else:
                    no_improvement_iterations += 1
                    print(f"  Minimal improvement: {improvement:.6f} < {min_improvement_threshold}")

                if best_obj < target_equity_threshold:
                    return {
                        "fps_value": float(fps_value),
                        "optimal_allocations": best_allocations,
                        "equity_scores": {
                            "low": float(best_results.get("low", {}).get("deviation", np.nan)),
                            "middle": float(best_results.get("middle", {}).get("deviation", np.nan)),
                            "high": float(best_results.get("high", {}).get("deviation", np.nan)),
                        },
                        "deviations": {
                            "low": float(best_results.get("low", {}).get("deviation", np.nan)),
                            "middle": float(best_results.get("middle", {}).get("deviation", np.nan)),
                            "high": float(best_results.get("high", {}).get("deviation", np.nan)),
                        },
                        "avg_equity": float(best_obj),
                        "travel_time_equity_index": float(best_obj),
                        "full_results": best_results,
                        "subsidy_usage": best_results.get("subsidy_usage", {}),
                        "deterministic_mode": deterministic_mode,
                        "deterministic_scenario_path": deterministic_scenario_path or "",
                        "terminated_early": True,
                        "termination_reason": "Target threshold reached",
                    }

        if not found_improvement:
            print(f"Iteration {iteration+1}, No improvement. Current best: {best_obj:.6f}")
            no_improvement_iterations += 1

        if no_improvement_iterations >= patience:
            return {
                "fps_value": float(fps_value),
                "optimal_allocations": best_allocations,
                "equity_scores": {
                    "low": float(best_results.get("low", {}).get("deviation", np.nan)) if best_results else np.nan,
                    "middle": float(best_results.get("middle", {}).get("deviation", np.nan)) if best_results else np.nan,
                    "high": float(best_results.get("high", {}).get("deviation", np.nan)) if best_results else np.nan,
                },
                "deviations": {
                    "low": float(best_results.get("low", {}).get("deviation", np.nan)) if best_results else np.nan,
                    "middle": float(best_results.get("middle", {}).get("deviation", np.nan)) if best_results else np.nan,
                    "high": float(best_results.get("high", {}).get("deviation", np.nan)) if best_results else np.nan,
                },
                "avg_equity": float(best_obj),
                "travel_time_equity_index": float(best_obj),
                "full_results": best_results,
                "subsidy_usage": best_results.get("subsidy_usage", {}) if best_results else {},
                "deterministic_mode": bool(base_parameters.get("deterministic_mode", False)),
                "deterministic_scenario_path": str(base_parameters.get("deterministic_scenario_path", "")),
                "terminated_early": True,
                "termination_reason": "Convergence",
            }

    return {
        "fps_value": float(fps_value),
        "optimal_allocations": best_allocations,
        "equity_scores": {
            "low": float(best_results.get("low", {}).get("deviation", np.nan)) if best_results else np.nan,
            "middle": float(best_results.get("middle", {}).get("deviation", np.nan)) if best_results else np.nan,
            "high": float(best_results.get("high", {}).get("deviation", np.nan)) if best_results else np.nan,
        },
        "deviations": {
            "low": float(best_results.get("low", {}).get("deviation", np.nan)) if best_results else np.nan,
            "middle": float(best_results.get("middle", {}).get("deviation", np.nan)) if best_results else np.nan,
            "high": float(best_results.get("high", {}).get("deviation", np.nan)) if best_results else np.nan,
        },
        "avg_equity": float(best_obj),
        "travel_time_equity_index": float(best_obj),
        "full_results": best_results,
        "subsidy_usage": best_results.get("subsidy_usage", {}) if best_results else {},
        "deterministic_mode": bool(base_parameters.get("deterministic_mode", False)),
        "deterministic_scenario_path": str(base_parameters.get("deterministic_scenario_path", "")),
        "terminated_early": False,
    }

def visualize_results(results, output_dir):
    """
    Create visualizations from the travel time equity analysis results
    
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
            row = {
                'fps_value': fps, 
                'travel_time_equity_index': result['travel_time_equity_index']
            }
            
            # Add individual deviations for each income level
            for level in ['low', 'middle', 'high']:
                if 'deviations' in result and result['deviations']:
                    row[f'deviation_{level}'] = result['deviations'][level]
            
            # Add allocation percentages
            for key, value in result['optimal_allocations'].items():
                row[f'alloc_{key}'] = value
                
            # Add average travel times if available
            if 'full_results' in result:
                for level in ['low', 'middle', 'high']:
                    if level in result['full_results']:
                        row[f'avg_travel_time_{level}'] = result['full_results'][level].get('avg_travel_time', 0)
                        
                # Add overall average travel time
                row['overall_avg_travel_time'] = result['full_results'].get('overall_avg_travel_time', 0)
                
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
    
    data_table.to_csv(os.path.join(output_dir, 'travel_time_equity_allocation_table.csv'), index=False)
    
    # 1. Create main visualization with all income levels and allocation references
    plt.figure(figsize=(15, 10))
    
    # Plot lines for each income level's deviation
    colors = {'low': '#1f77b4', 'middle': '#ff7f0e', 'high': '#2ca02c', 'total': '#d62728'}
    markers = {'low': 'o', 'middle': 's', 'high': '^', 'total': 'D'}
    
    # Plot income-specific deviations
    for level in ['low', 'middle', 'high']:
        plt.plot(df['fps_value'], df[f'deviation_{level}'], 
                 marker=markers[level], linestyle='-', 
                 color=colors[level], 
                 label=f'{level.capitalize()} Income Deviation', 
                 alpha=0.8)
    
    # Plot total travel time equity index with bolder line
    plt.plot(df['fps_value'], df['travel_time_equity_index'], 
             marker=markers['total'], linestyle='-', 
             color=colors['total'],
             linewidth=3, 
             label='Total Travel Time Equity Index', 
             alpha=0.9)
    
    # Add trend line using polynomial fit
    if len(df) > 2:
        try:
            # Use LOWESS smoothing for more robust trend
            from scipy import stats
            x = df['fps_value']
            y = df['travel_time_equity_index']
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
    # We'll add annotations to every other point to avoid overcrowding
    for i, (idx, row) in enumerate(df.iterrows()):
        if i % 2 == 0:  # Only annotate every other point
            fps = row['fps_value']
            equity_index = row['travel_time_equity_index']
            
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
                        xy=(fps, equity_index),
                        xytext=(10, 10 + (i % 3) * 20),  # Stagger annotations
                        textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'),
                        fontsize=8)
    
    # Add table reference
    plt.figtext(0.02, 0.02, 
               "Note: See 'travel_time_equity_allocation_table.csv' for the complete allocations table",
               bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
    
    # Improve graph appearance
    plt.xscale('log')  # Log scale for FPS values
    plt.title('Travel Time Equity Index vs Fixed Pool Subsidy (FPS) Values', fontsize=16)
    plt.xlabel('Fixed Pool Subsidy (FPS)', fontsize=14)
    plt.ylabel('Travel Time Equity Index (Lower is Better)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='best')
    
    # Add vertical line at the FPS value with highest equity score
    if not df.empty:
        best_fps = df.loc[df['travel_time_equity_index'].idxmin()]['fps_value']
        plt.axvline(x=best_fps, color='red', linestyle='--', alpha=0.5,
                label=f'Best FPS: {best_fps}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'travel_time_equity_vs_fps_with_allocations.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Create a visualization of average travel times by income level
    plt.figure(figsize=(15, 10))
    
    for level in ['low', 'middle', 'high']:
        if f'avg_travel_time_{level}' in df.columns:
            plt.plot(df['fps_value'], df[f'avg_travel_time_{level}'], 
                     marker=markers[level], linestyle='-', 
                     color=colors[level], 
                     label=f'{level.capitalize()} Income Avg Travel Time', 
                     alpha=0.8)
    
    # Plot overall average travel time
    if 'overall_avg_travel_time' in df.columns:
        plt.plot(df['fps_value'], df['overall_avg_travel_time'], 
                 marker='*', linestyle='-', 
                 color='purple',
                 linewidth=3, 
                 label='Overall Avg Travel Time', 
                 alpha=0.9)
    
    plt.title('Average Travel Times by Income Group vs FPS', fontsize=16)
    plt.xlabel('Fixed Pool Subsidy (FPS)', fontsize=14)
    plt.ylabel('Average Travel Time (minutes)', fontsize=14)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'average_travel_times_by_income.png'), 
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
    
    # 4. Create a heatmap of travel time deviations
    if not df.empty:
        # Get the FPS value with the lowest travel time equity index
        best_fps = df.loc[df['travel_time_equity_index'].idxmin()]['fps_value']
        
        # Extract deviation data for this FPS
        deviation_data = []
        for fps, result in results.items():
            if fps == best_fps and 'full_results' in result:
                for income in ['low', 'middle', 'high']:
                    if income in result['full_results']:
                        deviation_data.append({
                            'income_level': income,
                            'deviation': result['full_results'][income]['deviation'],
                            'avg_travel_time': result['full_results'][income]['avg_travel_time']
                        })
        
        if deviation_data:
            deviation_df = pd.DataFrame(deviation_data)
            
            # Create heatmap for deviations
            plt.figure(figsize=(10, 6))
            deviation_pivot = deviation_df.pivot_table(
                values='deviation',
                index='income_level',
                aggfunc='first'
            ).reindex(['low', 'middle', 'high'])
            
            sns.heatmap(deviation_pivot, annot=True, cmap='YlOrRd', fmt=".2f", cbar_kws={'label': 'Minutes Deviation'})
            plt.title(f'Travel Time Deviations for Optimal Solution (FPS={best_fps})')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'travel_time_deviations_heatmap.png'), dpi=300)
            plt.close()
            
            # Create bar chart for average travel times
            plt.figure(figsize=(10, 6))
            travel_time_pivot = deviation_df.pivot_table(
                values='avg_travel_time',
                index='income_level',
                aggfunc='first'
            ).reindex(['low', 'middle', 'high'])
            
            travel_time_pivot.plot(kind='bar', color='skyblue')
            plt.title(f'Average Travel Times by Income Group (FPS={best_fps})')
            plt.xlabel('Income Level')
            plt.ylabel('Average Travel Time (minutes)')
            plt.xticks(rotation=0)
            plt.grid(axis='y', alpha=0.3)
            
            # Add overall average travel time line
            overall_avg = results[best_fps]['full_results'].get('overall_avg_travel_time', 0)
            plt.axhline(y=overall_avg, color='red', linestyle='--', label=f'Overall Avg: {overall_avg:.2f} min')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'average_travel_times_bar.png'), dpi=300)
            plt.close()
    
    # Find and save the optimal solution
    if 'travel_time_equity_index' in df.columns:
        optimal_row = df.loc[df['travel_time_equity_index'].idxmin()]
        
        with open(os.path.join(output_dir, 'optimal_solution.txt'), 'w') as f:
            f.write(f"Optimal FPS Value: {optimal_row['fps_value']}\n")
            f.write(f"Optimal Travel Time Equity Index: {optimal_row['travel_time_equity_index']:.4f}\n\n")
            f.write("Optimal Subsidy Allocations:\n")
            
            for key in sorted([k for k in optimal_row.index if k.startswith('alloc_')]):
                f.write(f"  {key.replace('alloc_', '')}: {optimal_row[key]:.4f}\n")
                
            f.write("\nTravel Time Deviation by Income Level:\n")
            for level in ['low', 'middle', 'high']:
                f.write(f"  {level.capitalize()}: {optimal_row[f'deviation_{level}']:.4f} minutes\n")
                
            f.write("\nAverage Travel Times by Income Level:\n")
            for level in ['low', 'middle', 'high']:
                f.write(f"  {level.capitalize()}: {optimal_row[f'avg_travel_time_{level}']:.2f} minutes\n")
                
            f.write(f"\nOverall Average Travel Time: {optimal_row['overall_avg_travel_time']:.2f} minutes\n")

def visualize_subsidy_usage(results, output_dir):
    """
    Create visualizations for subsidy usage statistics.
    
    Args:
        results: Dictionary of results from optimization
        output_dir: Directory to save visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
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
    
    # 5. Create a comparison of subsidies vs travel time deviation
    # Join with existing results to correlate subsidy usage with travel time equity
    try:
        # Create a dataframe with travel time equity data
        equity_rows = []
        for fps, result in results.items():
            if 'travel_time_equity_index' in result:
                equity_rows.append({
                    'fps_value': fps,
                    'travel_time_equity_index': result['travel_time_equity_index']
                })
        
        if equity_rows:
            equity_df = pd.DataFrame(equity_rows)
            equity_df['fps_value'] = pd.to_numeric(equity_df['fps_value'])
            
            # Merge with subsidy data
            combined_df = pd.merge(df, equity_df, on='fps_value')
            
            plt.figure(figsize=(12, 8))
            
            # Create a scatter plot with points sized by percentage used
            sizes = combined_df['percentage_used'] * 5  # Scale for better visibility
            
            # Create scatter plot with color gradient based on travel time equity
            scatter = plt.scatter(combined_df['fps_value'], 
                                 combined_df['total_subsidy_used'],
                                 s=sizes,
                                 c=combined_df['travel_time_equity_index'],
                                 cmap='viridis_r',  # Reversed so better equity = brighter color
                                 alpha=0.7)
            
            plt.colorbar(scatter, label='Travel Time Equity Index (lower is better)')
            
            plt.title('Subsidy Usage vs FPS Value (colored by Travel Time Equity)', fontsize=16)
            plt.xlabel('Fixed Pool Subsidy (FPS)', fontsize=14)
            plt.ylabel('Total Subsidy Used', fontsize=14)
            plt.xscale('log')
            plt.grid(True, alpha=0.3)
            
            # Annotate points with their FPS values
            for i, row in combined_df.iterrows():
                plt.annotate(f"{row['fps_value']:.0f}",
                            xy=(row['fps_value'], row['total_subsidy_used']),
                            xytext=(5, 5),
                            textcoords='offset points',
                            fontsize=8)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'subsidy_usage_vs_equity.png'), dpi=300)
            plt.close()
    except Exception as e:
        print(f"Error creating subsidy vs equity visualization: {str(e)}")
        traceback.print_exc()
    
    # Save data table for reference
    subsidy_table = df[['fps_value', 'percentage_used', 'total_subsidy_used'] + 
                     [f'{income}_{metric}' for income in income_levels 
                      for metric in ['amount', 'percentage_of_used', 'percentage_of_total_fps']]]
    subsidy_table.to_csv(os.path.join(output_dir, 'subsidy_usage_statistics.csv'), index=False)

def main():
    """Travel time equity optimisation (ModeShare-plumbing compatible)."""
    global NUM_CPUS, SIMULATION_STEPS

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed0", type=int, default=0, help="Seed index for this optimization run")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory for this run")
    parser.add_argument("--num_cpus", type=int, default=NUM_CPUS, help="Parallel CPUs within this job")
    parser.add_argument("--steps", type=int, default=SIMULATION_STEPS, help="Simulation steps per evaluation")
    parser.add_argument(
        "--fps_values",
        type=str,
        default=None,
        help="Comma-separated FPS values (overrides built-in list), e.g. '3500,4500,5500'",
    )
    parser.add_argument(
        "--deterministic_mode",
        action="store_true",
        help="Use fixed scenario (same commuters/trips/background traffic across FPS)",
    )
    parser.add_argument(
        "--deterministic_scenario",
        type=str,
        default=None,
        help="Path to deterministic scenario JSON. If omitted, one is generated under out_dir.",
    )
    parser.add_argument(
        "--deterministic_scenario_seed",
        type=int,
        default=None,
        help="Scenario seed for auto-generated deterministic scenario JSON.",
    )
    args = parser.parse_args()

    NUM_CPUS = args.num_cpus
    SIMULATION_STEPS = args.steps
    optimizer_seed = args.seed0
    base_seed = 12345

    print(
        f"Optimizer seed0={optimizer_seed}, base_seed={base_seed}, "
        f"NUM_CPUS={NUM_CPUS}, STEPS={SIMULATION_STEPS}"
    )

    # IMPORTANT: match Mode Share base_parameters (commuters + background traffic)
    base_parameters = {
        "num_commuters": 200,
        "grid_width": 100,
        "grid_height": 100,
        "data_income_weights": [0.5, 0.3, 0.2],
        "data_health_weights": [0.9, 0.1],
        "data_payment_weights": [0.8, 0.2],
        "data_age_distribution": {(18, 25): 0.2, (26, 35): 0.3, (36, 45): 0.2,
                                  (46, 55): 0.15, (56, 65): 0.1, (66, 75): 0.05},
        "data_disability_weights": [0.2, 0.8],
        "data_tech_access_weights": [0.95, 0.05],
        "ASC_VALUES": {"car": 0, "bike": 0, "public": 0, "walk": 0, "maas": 0, "default": 0},
        "UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS": {"beta_C": -0.02, "beta_T": -0.09},
        "UTILITY_FUNCTION_BASE_COEFFICIENTS": {"beta_C": -0.15, "beta_T": -0.09, "beta_W": -0.04, "beta_A": -0.04, "alpha": -0.01},
        "PENALTY_COEFFICIENTS": {"disability_bike_walk": 0.8, "age_health_bike_walk": 0.3, "no_tech_access_car_bike": 0.1},
        "AFFORDABILITY_THRESHOLDS": {"low": 25, "middle": 40, "high": 130},
        "FLEXIBILITY_ADJUSTMENTS": {"low": 1.15, "medium": 1.0, "high": 0.85},
        "VALUE_OF_TIME": {"low": 5, "middle": 10, "high": 20},
        "public_price_table": {"train": {"on_peak": 3, "off_peak": 2.6}, "bus": {"on_peak": 2.4, "off_peak": 2}},
        "ALPHA_VALUES": {"UberLike1": 0.3, "UberLike2": 0.3, "BikeShare1": 0.25, "BikeShare2": 0.25},
        "DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS": {"S_base": 0.02, "alpha": 0.10, "delta": 0.5},
        "BACKGROUND_TRAFFIC_AMOUNT": 200,
        "CONGESTION_ALPHA": 0.03,
        "CONGESTION_BETA": 1.5,
        "CONGESTION_CAPACITY": 10,
        "CONGESTION_T_IJ_FREE_FLOW": 1.5,
        "uber_like1_capacity": 15,
        "uber_like1_price": 15.5,
        "uber_like2_capacity": 19,
        "uber_like2_price": 16.5,
        "bike_share1_capacity": 10,
        "bike_share1_price": 2.5,
        "bike_share2_capacity": 12,
        "bike_share2_price": 3,
    }
    base_parameters["seed"] = int(optimizer_seed)
    base_parameters["base_seed"] = int(base_seed)

    if args.fps_values:
        fps_values = [int(float(v.strip())) for v in args.fps_values.split(",") if v.strip()]
    else:
        fps_values = [3500, 4500, 5500, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 15000]

    # Output directory (same serialization contract)
    if args.out_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = f"travel_time_equity_results_{timestamp}"
    else:
        output_dir = args.out_dir
    os.makedirs(output_dir, exist_ok=True)

    if args.deterministic_mode:
        base_parameters["deterministic_mode"] = True
        if args.deterministic_scenario:
            scenario_path = os.path.abspath(args.deterministic_scenario)
            if not os.path.exists(scenario_path):
                raise FileNotFoundError(f"Deterministic scenario not found: {scenario_path}")
        else:
            scenario_seed = (
                int(args.deterministic_scenario_seed)
                if args.deterministic_scenario_seed is not None
                else int(base_seed + optimizer_seed)
            )
            scenario_path = os.path.join(output_dir, "deterministic_scenario.json")
            scenario = build_scenario(
                base_parameters=base_parameters,
                scenario_seed=scenario_seed,
                simulation_steps=SIMULATION_STEPS,
            )
            save_scenario(scenario, scenario_path)
            print(f"Generated deterministic scenario at {scenario_path} (scenario_seed={scenario_seed})")
        base_parameters["deterministic_scenario_path"] = os.path.abspath(scenario_path)
    else:
        base_parameters["deterministic_mode"] = False

    # Persist base parameters for reproducibility
    with open(os.path.join(output_dir, "base_parameters.pkl"), "wb") as f:
        pickle.dump(base_parameters, f)

    results = run_sequential_fps_optimization(fps_values, base_parameters, num_cpus=NUM_CPUS)

    # Per-FPS result files + all_results.pkl
    for fps, result in results.items():
        with open(os.path.join(output_dir, f"fps_{fps}_result.pkl"), "wb") as f:
            pickle.dump(result, f)

    with open(os.path.join(output_dir, "all_results.pkl"), "wb") as f:
        pickle.dump(results, f)

    best_fps = min(results.items(), key=lambda x: x[1].get("avg_equity", float("inf")))[0]
    print(f"\nOptimal FPS value: {best_fps}")
    print(f"Optimal travel_time_equity_index: {results[best_fps].get('travel_time_equity_index', np.nan):.6f}")

    visualize_results(results, output_dir)
    visualize_subsidy_usage(results, output_dir)

if __name__ == "__main__":
    main()
