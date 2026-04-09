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

# Constants
SIMULATION_STEPS = 144  # Reduced for faster runs during optimization
NUM_CPUS = 10  # Adjust based on your system

def calculate_total_system_travel_time(session, schema=None):
    """
    Calculate the total system travel time - sum of all trip durations.
    
    Args:
        session: SQLAlchemy session with simulation data
        schema: Optional schema name for PostgreSQL
        
    Returns:
        dict: Results containing total system travel time and related metrics
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
        
        # Get overall travel time data
        if schema:
            # Use text-based SQL when schema is provided
            query = text("""
                SELECT SUM(total_time) as total_system_time, 
                       COUNT(request_id) as total_trips,
                       AVG(total_time) as avg_trip_time
                FROM service_booking_log
                WHERE total_time IS NOT NULL
            """)
            overall_result = session.execute(query).fetchone()
        else:
            # Use ORM query for default schema
            overall_result = session.query(
                func.sum(ServiceBookingLog.total_time).label('total_system_time'),
                func.count(ServiceBookingLog.request_id).label('total_trips'),
                func.avg(ServiceBookingLog.total_time).label('avg_trip_time')
            ).filter(
                ServiceBookingLog.total_time != None
            ).first()
        
        # Extract overall results
        total_system_time = float(overall_result.total_system_time) if overall_result and overall_result.total_system_time else 0
        total_trips = int(overall_result.total_trips) if overall_result and overall_result.total_trips else 0
        avg_trip_time = float(overall_result.avg_trip_time) if overall_result and overall_result.avg_trip_time else 0
        
        # Store overall results
        results['total_system_travel_time'] = total_system_time
        results['total_trips'] = total_trips
        results['avg_trip_time'] = avg_trip_time
        
        # Get travel time data by income level
        travel_times_by_income = {}
        trip_counts_by_income = {}
        avg_times_by_income = {}
        
        for income_level in income_levels:
            if schema:
                # Use text-based SQL for income-specific queries
                query = text("""
                    SELECT SUM(sbl.total_time) as income_total_time,
                           COUNT(sbl.request_id) as income_trip_count,
                           AVG(sbl.total_time) as income_avg_time
                    FROM service_booking_log sbl
                    JOIN commuter_info_log cil ON sbl.commuter_id = cil.commuter_id
                    WHERE cil.income_level = :income_level AND sbl.total_time IS NOT NULL
                """)
                income_result = session.execute(query, {"income_level": income_level}).fetchone()
            else:
                # Use ORM query for income-specific data
                income_result = session.query(
                    func.sum(ServiceBookingLog.total_time).label('income_total_time'),
                    func.count(ServiceBookingLog.request_id).label('income_trip_count'),
                    func.avg(ServiceBookingLog.total_time).label('income_avg_time')
                ).join(
                    CommuterInfoLog,
                    ServiceBookingLog.commuter_id == CommuterInfoLog.commuter_id
                ).filter(
                    CommuterInfoLog.income_level == income_level,
                    ServiceBookingLog.total_time != None
                ).first()
            
            # Extract income-specific results
            income_total_time = float(income_result.income_total_time) if income_result and income_result.income_total_time else 0
            income_trip_count = int(income_result.income_trip_count) if income_result and income_result.income_trip_count else 0
            income_avg_time = float(income_result.income_avg_time) if income_result and income_result.income_avg_time else 0
            
            # Store income-specific results
            travel_times_by_income[income_level] = income_total_time
            trip_counts_by_income[income_level] = income_trip_count
            avg_times_by_income[income_level] = income_avg_time
        
        # Calculate percentage contribution to total travel time by income level
        for income_level in income_levels:
            percentage = (travel_times_by_income[income_level] / total_system_time * 100) if total_system_time > 0 else 0
            results[f'{income_level}_percentage'] = percentage
        
        # Store detailed results by income level
        for income_level in income_levels:
            results[income_level] = {
                'total_travel_time': travel_times_by_income[income_level],
                'trip_count': trip_counts_by_income[income_level],
                'avg_trip_time': avg_times_by_income[income_level],
                'percentage_of_total': results[f'{income_level}_percentage']
            }
        
        # Get travel time data by mode
        if schema:
            # Use text-based SQL for mode-specific queries
            query = text("""
                SELECT record_company_name as mode,
                       SUM(total_time) as mode_total_time,
                       COUNT(request_id) as mode_trip_count,
                       AVG(total_time) as mode_avg_time
                FROM service_booking_log
                WHERE total_time IS NOT NULL
                GROUP BY record_company_name
                ORDER BY SUM(total_time) DESC
            """)
            mode_results = session.execute(query).fetchall()
        else:
            # Use ORM query for mode-specific data
            mode_results = session.query(
                ServiceBookingLog.record_company_name.label('mode'),
                func.sum(ServiceBookingLog.total_time).label('mode_total_time'),
                func.count(ServiceBookingLog.request_id).label('mode_trip_count'),
                func.avg(ServiceBookingLog.total_time).label('mode_avg_time')
            ).filter(
                ServiceBookingLog.total_time != None
            ).group_by(
                ServiceBookingLog.record_company_name
            ).order_by(
                func.sum(ServiceBookingLog.total_time).desc()
            ).all()
        
        # Store mode-specific results
        results['mode_breakdown'] = {}
        
        for mode_result in mode_results:
            mode = mode_result.mode
            mode_total_time = float(mode_result.mode_total_time) if mode_result.mode_total_time else 0
            mode_trip_count = int(mode_result.mode_trip_count) if mode_result.mode_trip_count else 0
            mode_avg_time = float(mode_result.mode_avg_time) if mode_result.mode_avg_time else 0
            
            # Calculate percentage of total
            mode_percentage = (mode_total_time / total_system_time * 100) if total_system_time > 0 else 0
            
            results['mode_breakdown'][mode] = {
                'total_travel_time': mode_total_time,
                'trip_count': mode_trip_count,
                'avg_trip_time': mode_avg_time,
                'percentage_of_total': mode_percentage
            }
        
        return results
        
    except Exception as e:
        print(f"Error in calculate_total_system_travel_time: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Return a default structure to avoid NoneType errors
        default_results = {
            'total_system_travel_time': 0,
            'total_trips': 0,
            'avg_trip_time': 0
        }
        
        for income_level in income_levels:
            default_results[income_level] = {
                'total_travel_time': 0,
                'trip_count': 0,
                'avg_trip_time': 0,
                'percentage_of_total': 0
            }
            default_results[f'{income_level}_percentage'] = 0
            
        default_results['mode_breakdown'] = {}
        
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
    """Run a single simulation and calculate total system travel time (ModeShare-plumbing compatible)."""
    pid = os.getpid()

    fps_for_seed = int(params.get("fps_value", 0))
    seed = int(params.get("seed", 0))
    simulation_id = int(params.get("simulation_id", 0))
    bo_iteration = int(params.get("bo_iteration", 0))
    base_seed = int(params.get("base_seed", 12345))

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
        f"final_seed={final_seed}"
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

    _meta = {
        "simulation_id": simulation_id,
        "bo_iteration": bo_iteration,
        "seed": seed,
        "base_seed": base_seed,
        "final_seed": final_seed,
    }

    def _empty_income_block(avg_value=0.0):
        return {
            "total_travel_time": 0.0,
            "trip_count": 0,
            "avg_trip_time": float(avg_value),
            "percentage_of_total": 0.0,
        }

    try:
        db_port = int(os.environ.get("PGPORT", "15433"))
        db_user = os.environ.get("PGUSER", "z5247491")
        db_host = os.environ.get("PGHOST", "localhost")
        db_name = os.environ.get("PGDATABASE", "postgres")
        db_connection_string = f"postgresql://{db_user}@{db_host}:{db_port}/{db_name}"

        fps_value = float(params.pop("fps_value", 0))
        fixed_allocations = params.pop("fixed_allocations", None)
        simulation_steps = int(params.pop("simulation_steps", SIMULATION_STEPS))

        params.pop("seed", None)
        params.pop("base_seed", None)
        params.pop("bo_iteration", None)
        params.pop("simulation_id", None)

        pilot_steps = params.pop("pilot_steps", None)
        best_obj = params.pop("early_stop_best_equity", None)
        margin = float(params.pop("early_stop_margin", 0.20))
        exhaust_frac = float(params.pop("early_stop_exhaust_frac", 0.90))

        engine = create_engine(db_connection_string)

        with engine.connect() as connection:
            connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema_name}"))
            connection.commit()

        from sqlalchemy.orm import sessionmaker
        Session = sessionmaker(bind=engine)
        session = Session()

        with engine.connect() as connection:
            connection.execute(text(f"SET search_path TO {schema_name}"))
            connection.commit()

        reset_db_params = {k: params[k] for k in [
            "uber_like1_capacity", "uber_like1_price",
            "uber_like2_capacity", "uber_like2_price",
            "bike_share1_capacity", "bike_share1_price",
            "bike_share2_capacity", "bike_share2_price"
        ]}
        reset_database(engine=engine, session=session, schema=schema_name, **reset_db_params)

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

        model = MobilityModel(**params)

        # Pilot early-stop: objective = total_system_travel_time (lower better)
        if pilot_steps is not None and best_obj is not None and np.isfinite(best_obj):
            model.run_model(int(pilot_steps))

            pilot_results = calculate_total_system_travel_time(session, schema=schema_name)
            pilot_metric = float(pilot_results.get("total_system_travel_time", float("inf")))

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
                    "total_system_travel_time": float("inf"),
                    "total_trips": 0,
                    "avg_trip_time": float("inf"),
                    "avg_equity": float("inf"),
                    "equity_scores": {
                        "low": float("inf"),
                        "middle": float("inf"),
                        "high": float("inf"),
                    },
                    "low": _empty_income_block(avg_value=float("inf")),
                    "middle": _empty_income_block(avg_value=float("inf")),
                    "high": _empty_income_block(avg_value=float("inf")),
                    "income_breakdown": {
                        "low": _empty_income_block(avg_value=float("inf")),
                        "middle": _empty_income_block(avg_value=float("inf")),
                        "high": _empty_income_block(avg_value=float("inf")),
                    },
                    "mode_breakdown": {},
                    **_meta,
                }

            remaining_steps = int(simulation_steps) - int(pilot_steps)
            if remaining_steps > 0:
                model.run_model(remaining_steps)
        else:
            model.run_model(simulation_steps)

        results = calculate_total_system_travel_time(session, schema=schema_name)
        subsidy_stats = calculate_subsidy_usage_statistics(session, fps_value, schema=schema_name)
        results["subsidy_usage"] = subsidy_stats

        results["fps_value"] = float(fps_value)
        results["schema_name"] = schema_name
        results["fixed_allocations"] = fixed_allocations
        results["terminated_early"] = bool(results.get("terminated_early", False))
        results["termination_reason"] = str(results.get("termination_reason", ""))
        results["avg_equity"] = float(results.get("total_system_travel_time", np.nan))
        results["equity_scores"] = {
            "low": float(results.get("low", {}).get("avg_trip_time", np.nan)),
            "middle": float(results.get("middle", {}).get("avg_trip_time", np.nan)),
            "high": float(results.get("high", {}).get("avg_trip_time", np.nan)),
        }
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
            "total_system_travel_time": 0.0,
            "total_trips": 0,
            "avg_trip_time": 0.0,
            "avg_equity": 0.0,
            "equity_scores": {"low": 0.0, "middle": 0.0, "high": 0.0},
            "low": _empty_income_block(avg_value=0.0),
            "middle": _empty_income_block(avg_value=0.0),
            "high": _empty_income_block(avg_value=0.0),
            "income_breakdown": {
                "low": _empty_income_block(avg_value=0.0),
                "middle": _empty_income_block(avg_value=0.0),
                "high": _empty_income_block(avg_value=0.0),
            },
            "mode_breakdown": {},
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
        
        print(f"Completed FPS={fps_value}, Total System Travel Time={result['total_system_travel_time']:.4f}")
        
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
    Objective: minimize total_system_travel_time (lower is better).
    """
    print(f"\nOptimizing subsidy allocation for FPS = {fps_value} with {num_cpus} parallel simulations")

    param_names = [
        "low_bike", "low_car", "low_MaaS_Bundle", "low_public",
        "middle_bike", "middle_car", "middle_MaaS_Bundle", "middle_public",
        "high_bike", "high_car", "high_MaaS_Bundle", "high_public"
    ]

    bounds = {p: (0.0, 0.8) for p in param_names}
    lower_bounds = [bounds[p][0] for p in param_names]
    upper_bounds = [bounds[p][1] for p in param_names]

    sampler = qmc.LatinHypercube(d=len(param_names), seed=42)
    X_init = sampler.random(n=8)
    X_init = qmc.scale(X_init, lower_bounds, upper_bounds)

    min_improvement_threshold = 0.01
    # Set to a float to enable hard-threshold early exit for total system travel time.
    target_total_time_threshold = None
    no_improvement_iterations = 0
    patience = 6

    best_obj = float("inf")
    best_allocations = None
    best_results = None
    previous_best = float("inf")

    def _build_optimizer_output(terminated_early=False, termination_reason=""):
        best_full = best_results if isinstance(best_results, dict) else {}
        low_component = float(best_full.get("low", {}).get("avg_trip_time", np.nan))
        middle_component = float(best_full.get("middle", {}).get("avg_trip_time", np.nan))
        high_component = float(best_full.get("high", {}).get("avg_trip_time", np.nan))
        return {
            "fps_value": float(fps_value),
            "optimal_allocations": best_allocations,
            "equity_scores": {
                "low": low_component,
                "middle": middle_component,
                "high": high_component,
            },
            "avg_equity": float(best_obj),
            "total_system_travel_time": float(best_obj),
            "avg_trip_time": float(best_full.get("avg_trip_time", np.nan)),
            "total_trips": int(best_full.get("total_trips", 0) or 0),
            "income_breakdown": {lvl: best_full.get(lvl, {}) for lvl in ["low", "middle", "high"]},
            "mode_breakdown": best_full.get("mode_breakdown", {}),
            "full_results": best_full,
            "subsidy_usage": best_full.get("subsidy_usage", {}),
            "terminated_early": bool(terminated_early),
            "termination_reason": str(termination_reason),
        }

    # Initial LHS
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
            obj = float(r.get("total_system_travel_time", float("inf")))
            y_init.append(obj)

            if obj < best_obj:
                best_obj = obj
                best_allocations = {k: float(v) for k, v in zip(param_names, X_init[i])}
                best_results = r
                print(f"New best total_system_travel_time from initial points: {best_obj:.6f}")
                if (
                    target_total_time_threshold is not None
                    and np.isfinite(best_obj)
                    and best_obj < float(target_total_time_threshold)
                ):
                    print("Early termination: Target total-system-travel-time threshold reached.")
                    return _build_optimizer_output(
                        terminated_early=True,
                        termination_reason="Target threshold reached",
                    )
        else:
            y_init.append(float("inf"))

    kernel = Matern(nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42)

    X = X_init
    y = np.array(y_init, dtype=float)

    max_iterations = 10

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

            obj = float(r.get("total_system_travel_time", float("inf")))

            X = np.vstack([X, top_candidates[j]])
            y = np.append(y, obj)

            improvement = previous_best - obj

            if obj < best_obj:
                best_obj = obj
                best_allocations = {k: float(v) for k, v in zip(param_names, top_candidates[j])}
                best_results = r
                found_improvement = True
                print(
                    f"Iteration {iteration+1}, New best total_system_travel_time: "
                    f"{best_obj:.6f}, Improvement: {improvement:.6f}"
                )

                if improvement > min_improvement_threshold:
                    no_improvement_iterations = 0
                else:
                    no_improvement_iterations += 1
                    print(f"  Minimal improvement: {improvement:.6f} < {min_improvement_threshold}")

                if (
                    target_total_time_threshold is not None
                    and np.isfinite(best_obj)
                    and best_obj < float(target_total_time_threshold)
                ):
                    print(
                        f"Early termination after {iteration+1} iterations: "
                        "Target total-system-travel-time threshold reached."
                    )
                    return _build_optimizer_output(
                        terminated_early=True,
                        termination_reason="Target threshold reached",
                    )

        if not found_improvement:
            print(f"Iteration {iteration+1}, No improvement. Current best: {best_obj:.6f}")
            no_improvement_iterations += 1

        if no_improvement_iterations >= patience:
            return _build_optimizer_output(
                terminated_early=True,
                termination_reason="Convergence",
            )

    return _build_optimizer_output(
        terminated_early=False,
        termination_reason="",
    )

def visualize_results(results, output_dir):
    """
    Create visualizations from the total system travel time analysis results
    
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
                'total_system_travel_time': result['total_system_travel_time'],
                'avg_trip_time': result['avg_trip_time'],
                'total_trips': result['total_trips']
            }
            
            # Add income-specific travel times
            for level in ['low', 'middle', 'high']:
                if 'income_breakdown' in result and level in result['income_breakdown']:
                    income_data = result['income_breakdown'][level]
                    row[f'{level}_total_time'] = income_data['total_travel_time']
                    row[f'{level}_trip_count'] = income_data['trip_count']
                    row[f'{level}_avg_time'] = income_data['avg_trip_time']
                    row[f'{level}_percentage'] = income_data['percentage_of_total']
            
            # Add allocation percentages
            for key, value in result['optimal_allocations'].items():
                row[f'alloc_{key}'] = value
                
            # Add mode-specific data if available
            if 'mode_breakdown' in result:
                for mode, mode_data in result['mode_breakdown'].items():
                    sanitized_mode = mode.replace(' ', '_').replace('-', '_')
                    row[f'mode_{sanitized_mode}_time'] = mode_data['total_travel_time']
                    row[f'mode_{sanitized_mode}_trips'] = mode_data['trip_count']
                    row[f'mode_{sanitized_mode}_percentage'] = mode_data['percentage_of_total']
                        
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
    
    data_table.to_csv(os.path.join(output_dir, 'total_system_travel_time_table.csv'), index=False)
    
    # 1. Create main visualization of total system travel time vs FPS
    plt.figure(figsize=(15, 10))
    
    # Plot total system travel time
    plt.plot(df['fps_value'], df['total_system_travel_time'], 
             marker='o', linestyle='-', 
             color='#d62728',
             linewidth=3, 
             label='Total System Travel Time', 
             alpha=0.9)
    
    # Add trend line using polynomial fit
    if len(df) > 2:
        try:
            # Use polynomial fit against log of FPS
            from scipy import stats
            x = df['fps_value']
            y = df['total_system_travel_time']
            z = np.polyfit(np.log(x), y, 2)
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
            total_time = row['total_system_travel_time']
            
            # Add annotation with arrow pointing to point
            plt.annotate(f'Point {i+1}',  # Use numbers to reference the table
                        xy=(fps, total_time),
                        xytext=(10, 10 + (i % 3) * 20),  # Stagger annotations
                        textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'),
                        fontsize=8)
    
    # Add table reference
    plt.figtext(0.02, 0.02, 
               "Note: See 'total_system_travel_time_table.csv' for the complete data table",
               bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
    
    # Improve graph appearance
    plt.xscale('log')  # Log scale for FPS values
    plt.title('Total System Travel Time vs Fixed Pool Subsidy (FPS) Values', fontsize=16)
    plt.xlabel('Fixed Pool Subsidy (FPS)', fontsize=14)
    plt.ylabel('Total System Travel Time (minutes)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='best')
    
    # Add vertical line at the FPS value with lowest total travel time
    if not df.empty:
        best_fps = df.loc[df['total_system_travel_time'].idxmin()]['fps_value']
        plt.axvline(x=best_fps, color='green', linestyle='--', alpha=0.5,
                   label=f'Best FPS: {best_fps}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'total_system_travel_time_vs_fps.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Create a visualization of travel time breakdown by income level
    plt.figure(figsize=(15, 10))
    
    # Create stacked bar chart of total travel time by income level
    income_levels = ['low', 'middle', 'high']
    income_colors = {'low': '#1f77b4', 'middle': '#ff7f0e', 'high': '#2ca02c'}
    
    # Convert FPS values to strings for categorical x-axis
    x = [str(int(fps)) for fps in df['fps_value']]
    
    # Plot stacked bars for income-specific travel times
    bottom = np.zeros(len(df))
    for level in income_levels:
        if f'{level}_total_time' in df.columns:
            plt.bar(x, df[f'{level}_total_time'], bottom=bottom, 
                   label=f'{level.capitalize()} Income', color=income_colors[level], alpha=0.7)
            bottom += df[f'{level}_total_time'].values
    
    plt.title('Total Travel Time Breakdown by Income Level', fontsize=16)
    plt.xlabel('Fixed Pool Subsidy (FPS)', fontsize=14)
    plt.ylabel('Total Travel Time (minutes)', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend(fontsize=12)
    
    # Add text labels for total time
    for i, total_time in enumerate(df['total_system_travel_time']):
        plt.text(i, total_time + total_time*0.02, f"{total_time:.0f}", 
                ha='center', va='bottom', fontsize=10, 
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'travel_time_by_income_stacked.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Create a visualization of average trip time by income level
    plt.figure(figsize=(15, 8))
    
    # Plot lines for each income level's average trip time
    for level in income_levels:
        if f'{level}_avg_time' in df.columns:
            plt.plot(df['fps_value'], df[f'{level}_avg_time'], 
                    marker='o', linestyle='-', 
                    color=income_colors[level], 
                    label=f'{level.capitalize()} Income Avg Trip Time', 
                    alpha=0.8)
    
    # Plot overall average trip time
    plt.plot(df['fps_value'], df['avg_trip_time'], 
            marker='*', linestyle='-', 
            color='purple',
            linewidth=3, 
            label='Overall Avg Trip Time', 
            alpha=0.9)
    
    plt.title('Average Trip Time by Income Level vs FPS', fontsize=16)
    plt.xlabel('Fixed Pool Subsidy (FPS)', fontsize=14)
    plt.ylabel('Average Trip Time (minutes)', fontsize=14)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'avg_trip_time_by_income.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Create a visualization of travel time breakdown by mode
    try:
        # Extract mode columns
        mode_columns = [col for col in df.columns if col.startswith('mode_') and col.endswith('_time')]
        if mode_columns:
            # Extract unique mode names
            mode_names = [col.replace('mode_', '').replace('_time', '') for col in mode_columns]
            
            # Create a palette with distinct colors for each mode
            n_modes = len(mode_names)
            mode_colors = sns.color_palette("Set2", n_modes)
            
            # Plot stacked bars for mode-specific travel times
            plt.figure(figsize=(15, 10))
            
            # Convert FPS values to strings for categorical x-axis
            x = [str(int(fps)) for fps in df['fps_value']]
            
            # Plot stacked bars for mode-specific travel times
            bottom = np.zeros(len(df))
            for i, mode in enumerate(mode_names):
                col = f'mode_{mode}_time'
                if col in df.columns:
                    # Format mode name for display (replace underscores with spaces)
                    display_name = mode.replace('_', ' ')
                    plt.bar(x, df[col], bottom=bottom, 
                           label=display_name, color=mode_colors[i], alpha=0.7)
                    bottom += df[col].values
            
            plt.title('Total Travel Time Breakdown by Mode', fontsize=16)
            plt.xlabel('Fixed Pool Subsidy (FPS)', fontsize=14)
            plt.ylabel('Total Travel Time (minutes)', fontsize=14)
            plt.grid(True, alpha=0.3, axis='y')
            plt.legend(fontsize=10, loc='upper center', bbox_to_anchor=(0.5, -0.05),
                      fancybox=True, shadow=True, ncol=min(5, n_modes))
            
            # Add text labels for total time
            for i, total_time in enumerate(df['total_system_travel_time']):
                plt.text(i, total_time + total_time*0.02, f"{total_time:.0f}", 
                        ha='center', va='bottom', fontsize=10, 
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'travel_time_by_mode_stacked.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # 5. Create a visualization of mode percentage contribution
            mode_pct_columns = [col for col in df.columns if col.startswith('mode_') and col.endswith('_percentage')]
            if mode_pct_columns:
                # Get the FPS value with the lowest total travel time
                best_fps = df.loc[df['total_system_travel_time'].idxmin()]['fps_value']
                best_row = df[df['fps_value'] == best_fps].iloc[0]
                
                # Create a pie chart of mode contributions at optimal FPS
                plt.figure(figsize=(12, 12))
                
                # Extract mode percentages and labels
                mode_pcts = [best_row[col] for col in mode_pct_columns]
                mode_labels = [col.replace('mode_', '').replace('_percentage', '').replace('_', ' ') 
                              for col in mode_pct_columns]
                
                # Plot pie chart
                plt.pie(mode_pcts, labels=mode_labels, colors=mode_colors,
                       autopct='%1.1f%%', shadow=True, startangle=90)
                plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                
                plt.title(f'Travel Time Contribution by Mode at Optimal FPS={best_fps}', fontsize=16)
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'mode_contribution_pie_chart.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
    except Exception as e:
        print(f"Error creating mode breakdown visualizations: {e}")
        traceback.print_exc()
    
    # 6. Create detailed allocation line graphs for each income level
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
    
    # Find and save the optimal solution
    if 'total_system_travel_time' in df.columns:
        optimal_row = df.loc[df['total_system_travel_time'].idxmin()]
        
        with open(os.path.join(output_dir, 'optimal_solution.txt'), 'w') as f:
            f.write(f"Optimal FPS Value: {optimal_row['fps_value']}\n")
            f.write(f"Minimum Total System Travel Time: {optimal_row['total_system_travel_time']:.2f} minutes\n")
            f.write(f"Average Trip Time: {optimal_row['avg_trip_time']:.2f} minutes\n")
            f.write(f"Total Number of Trips: {optimal_row['total_trips']}\n\n")
            
            f.write("Travel Time Breakdown by Income Level:\n")
            for level in income_levels:
                total_key = f'{level}_total_time'
                pct_key = f'{level}_percentage'
                avg_key = f'{level}_avg_time'
                count_key = f'{level}_trip_count'
                
                if all(key in optimal_row for key in [total_key, pct_key, avg_key, count_key]):
                    f.write(f"  {level.capitalize()} Income:\n")
                    f.write(f"    Total Travel Time: {optimal_row[total_key]:.2f} minutes ({optimal_row[pct_key]:.1f}%)\n")
                    f.write(f"    Average Trip Time: {optimal_row[avg_key]:.2f} minutes\n")
                    f.write(f"    Number of Trips: {optimal_row[count_key]}\n")
            
            f.write("\nTravel Time Breakdown by Mode:\n")
            mode_columns = [col for col in optimal_row.index if col.startswith('mode_') and col.endswith('_time')]
            for col in mode_columns:
                mode = col.replace('mode_', '').replace('_time', '').replace('_', ' ')
                time_val = optimal_row[col]
                pct_col = col.replace('_time', '_percentage')
                pct_val = optimal_row[pct_col] if pct_col in optimal_row else 0
                trips_col = col.replace('_time', '_trips')
                trips_val = optimal_row[trips_col] if trips_col in optimal_row else 0
                
                f.write(f"  {mode}:\n")
                f.write(f"    Total Travel Time: {time_val:.2f} minutes ({pct_val:.1f}%)\n")
                f.write(f"    Number of Trips: {trips_val}\n")
            
            f.write("\nOptimal Subsidy Allocations:\n")
            for key in sorted([k for k in optimal_row.index if k.startswith('alloc_')]):
                f.write(f"  {key.replace('alloc_', '')}: {optimal_row[key]:.4f}\n")

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
    
    # Extract subsidy usage and trip-count data from results
    rows = []
    income_levels = ['low', 'middle', 'high']
    for fps, result in results.items():
        print(f"Checking result for FPS {fps}...")
        if 'subsidy_usage' in result:
            print(f"  Found subsidy_usage for FPS {fps}")
            subsidy_usage = result.get('subsidy_usage') or {}
            total_subsidy_used = float(subsidy_usage.get('total_subsidy_used', 0.0) or 0.0)
            percentage_used = float(subsidy_usage.get('percentage_used', 0.0) or 0.0)
            income_breakdown = result.get('income_breakdown') or {}
            subsidy_by_income = subsidy_usage.get('subsidy_by_income') or {}

            print(f"  Total subsidy used: {total_subsidy_used}")
            print(f"  Percentage used: {percentage_used}%")
            row = {
                'fps_value': fps,
                'total_system_travel_time': result.get('total_system_travel_time', 0),
                'avg_trip_time': result.get('avg_trip_time', np.nan),
                'total_trips': int(result.get('total_trips', 0) or 0),
                'total_subsidy_used': total_subsidy_used,
                'percentage_used': percentage_used
            }
            
            # Add subsidy usage and trip counts by income level
            for income_level in income_levels:
                stats = subsidy_by_income.get(income_level, {})
                row[f'{income_level}_amount'] = float(stats.get('amount', 0.0) or 0.0)
                row[f'{income_level}_percentage_of_used'] = float(stats.get('percentage_of_used', 0.0) or 0.0)
                row[f'{income_level}_percentage_of_total_fps'] = float(stats.get('percentage_of_total_fps', 0.0) or 0.0)
                income_stats = income_breakdown.get(income_level, {})
                row[f'{income_level}_trip_count'] = int(income_stats.get('trip_count', 0) or 0)
                row[f'{income_level}_avg_trip_time'] = float(income_stats.get('avg_trip_time', np.nan))
            
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
    
    # 5. Create a comparison of subsidies vs total system travel time
    # Join with existing results to correlate subsidy usage with travel time
    try:
        plt.figure(figsize=(12, 8))
        
        # Create a scatter plot with points sized by percentage used
        sizes = df['percentage_used'] * 5  # Scale for better visibility
        
        # Create scatter plot with color gradient based on travel time
        scatter = plt.scatter(df['fps_value'], 
                             df['total_subsidy_used'],
                             s=sizes,
                             c=df['total_system_travel_time'],
                             cmap='viridis_r',  # Reversed so better (lower) time = brighter color
                             alpha=0.7)
        
        plt.colorbar(scatter, label='Total System Travel Time (lower is better)')
        
        plt.title('Subsidy Usage vs FPS Value (colored by Total System Travel Time)', fontsize=16)
        plt.xlabel('Fixed Pool Subsidy (FPS)', fontsize=14)
        plt.ylabel('Total Subsidy Used', fontsize=14)
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
        
        # Annotate points with their FPS values
        for i, row in df.iterrows():
            plt.annotate(f"{row['fps_value']:.0f}",
                        xy=(row['fps_value'], row['total_subsidy_used']),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'subsidy_usage_vs_travel_time.png'), dpi=300)
        plt.close()
        
        # 6. Create line chart showing relationship between travel time and subsidy usage
        plt.figure(figsize=(12, 8))
        
        # Create primary y-axis for travel time
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        color1 = 'tab:red'
        ax1.set_xlabel('Fixed Pool Subsidy (FPS)', fontsize=14)
        ax1.set_ylabel('Total System Travel Time (minutes)', color=color1, fontsize=14)
        ax1.plot(df['fps_value'], df['total_system_travel_time'], color=color1, marker='o', linestyle='-')
        ax1.tick_params(axis='y', labelcolor=color1)
        
        # Create secondary y-axis for subsidy usage
        ax2 = ax1.twinx()
        color2 = 'tab:blue'
        ax2.set_ylabel('Total Subsidy Used', color=color2, fontsize=14)
        ax2.plot(df['fps_value'], df['total_subsidy_used'], color=color2, marker='s', linestyle='-')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # Set log scale for x-axis
        ax1.set_xscale('log')
        
        plt.title('Total System Travel Time vs Subsidy Usage', fontsize=16)
        plt.grid(True, alpha=0.3)
        
        # Create custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=color1, marker='o', label='Total System Travel Time'),
            Line2D([0], [0], color=color2, marker='s', label='Total Subsidy Used')
        ]
        ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  fancybox=True, shadow=True, ncol=2)
        
        fig.tight_layout()
        plt.savefig(os.path.join(output_dir, 'travel_time_vs_subsidy_usage.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 7. Explicit trip-count trend (useful for explaining total-system-time shifts)
        plt.figure(figsize=(12, 8))
        fig, ax1 = plt.subplots(figsize=(12, 8))

        color1 = 'tab:purple'
        ax1.set_xlabel('Fixed Pool Subsidy (FPS)', fontsize=14)
        ax1.set_ylabel('Total Trips', color=color1, fontsize=14)
        ax1.plot(df['fps_value'], df['total_trips'], color=color1, marker='o', linestyle='-')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_xscale('log')
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        color2 = 'tab:orange'
        ax2.set_ylabel('Avg Trip Time (minutes)', color=color2, fontsize=14)
        ax2.plot(df['fps_value'], df['avg_trip_time'], color=color2, marker='s', linestyle='-')
        ax2.tick_params(axis='y', labelcolor=color2)

        plt.title('Trips and Average Trip Time vs FPS', fontsize=16)
        fig.tight_layout()
        plt.savefig(os.path.join(output_dir, 'trip_count_vs_fps.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error creating subsidy vs travel time visualization: {str(e)}")
        traceback.print_exc()
    
    # Save data table for reference
    subsidy_table = df[['fps_value', 'total_system_travel_time', 'avg_trip_time', 'total_trips',
                        'percentage_used', 'total_subsidy_used'] +
                      [f'{income}_{metric}' for income in income_levels
                       for metric in ['amount', 'percentage_of_used', 'percentage_of_total_fps',
                                      'trip_count', 'avg_trip_time']]]
    subsidy_table.to_csv(os.path.join(output_dir, 'subsidy_usage_statistics.csv'), index=False)

    trip_summary = df[['fps_value', 'total_trips', 'avg_trip_time', 'total_system_travel_time',
                       'total_subsidy_used', 'percentage_used',
                       'low_trip_count', 'middle_trip_count', 'high_trip_count']]
    trip_summary.to_csv(os.path.join(output_dir, 'trip_summary_by_fps.csv'), index=False)

def main():
    """Total system travel time optimisation (ModeShare-plumbing compatible)."""
    global NUM_CPUS, SIMULATION_STEPS

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed0", type=int, default=0, help="Seed index for this optimization run")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory for this run")
    parser.add_argument("--num_cpus", type=int, default=NUM_CPUS, help="Parallel CPUs within this job")
    parser.add_argument("--steps", type=int, default=SIMULATION_STEPS, help="Simulation steps per evaluation")
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

    # IMPORTANT: same fps_values set as Mode Share
    fps_values = [3500, 4500, 5500, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 15000]

    # Output directory (same serialization contract)
    if args.out_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = f"total_system_travel_time_results_{timestamp}"
    else:
        output_dir = args.out_dir
    os.makedirs(output_dir, exist_ok=True)

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
    print(f"Optimal total_system_travel_time: {results[best_fps].get('total_system_travel_time', np.nan):.6f}")

    visualize_results(results, output_dir)
    visualize_subsidy_usage(results, output_dir)

if __name__ == "__main__":
    main()
