from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from run_visualisation_03 import MobilityModel
from agent_service_provider_initialisation_03 import reset_database, CommuterInfoLog, ServiceBookingLog
import multiprocessing as mp
import os
from agent_subsidy_pool import SubsidyPoolConfig
SIMULATION_STEPS = 144  # Run for a larger number of steps for better analysis
num_commuters = 120
def calculate_sdi_normalized(subsidy_values, user_counts):
    """
    Calculate normalized Socioeconomic Distribution Index
    
    Args:
        subsidy_values: Dict with income levels as keys and total subsidies as values
        user_counts: Dict with income levels as keys and user counts as values
    """
    s_total = sum(subsidy_values.values())
    u_total = sum(user_counts.values())
    
    if s_total == 0 or u_total == 0:
        print("Warning: Zero total subsidies or users")
        return 0.0
    
    ratios = []
    for income_level in subsidy_values.keys():
        s_i = subsidy_values[income_level]
        u_i = user_counts[income_level]
        if u_i == 0:
            continue
        ratio = (s_i / s_total) / (u_i / u_total)
        ratios.append(ratio)
    
    if not ratios:
        return 0.0
    
    return sum(ratios) / len(ratios)

def calculate_sdi_theil(subsidy_values, user_counts):
    """
    Calculate Theil's Index version of SDI
    """
    s_total = sum(subsidy_values.values())
    u_total = sum(user_counts.values())
    
    if s_total == 0 or u_total == 0:
        print("Warning: Zero total subsidies or users")
        return 0.0
    
    theil_sum = 0
    for income_level in subsidy_values.keys():
        u_i = user_counts[income_level]
        s_i = subsidy_values[income_level]
        
        if u_i == 0 or s_i == 0:
            continue
            
        u_share = u_i / u_total
        s_share = s_i / s_total
        theil_sum += u_share * np.log(u_share / s_share)
    
    return theil_sum

def run_single_simulation(params):
    """
    Run a single simulation and calculate equity metrics
    """
    db_path = f"service_provider_database_{os.getpid()}.db"
    db_connection_string = f"sqlite:///{db_path}"
    engine = create_engine(db_connection_string)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Initialize and run model as in your original code
        reset_database(engine=engine, session=session, **params)
        model = MobilityModel(db_connection_string=db_connection_string, **params)
        model.run_model(params['simulation_steps'])
        
        # Query subsidies by income level
        subsidy_by_income = {}
        user_counts = {}
        
        for income_level in ['low', 'middle', 'high']:
            # Get total subsidies for this income level
            subsidies = (
                session.query(func.sum(ServiceBookingLog.government_subsidy))
                .join(CommuterInfoLog, CommuterInfoLog.commuter_id == ServiceBookingLog.commuter_id)
                .filter(CommuterInfoLog.income_level == income_level)
                .scalar() or 0
            )
            
            # Get user count for this income level
            users = (
                session.query(func.count(func.distinct(CommuterInfoLog.commuter_id)))
                .filter(CommuterInfoLog.income_level == income_level)
                .scalar() or 0
            )
            
            subsidy_by_income[income_level] = float(subsidies)
            user_counts[income_level] = users
        
        # Calculate both SDI metrics
        sdi_norm = calculate_sdi_normalized(subsidy_by_income, user_counts)
        sdi_theil = calculate_sdi_theil(subsidy_by_income, user_counts)
        
        return {
            'sdi_normalized': sdi_norm,
            'sdi_theil': sdi_theil,
            'subsidy_distribution': subsidy_by_income,
            'user_distribution': user_counts
        }
        
    finally:
        session.close()
        os.remove(db_path)

def run_parallel_simulations(parameter_sets, num_cpus):
    """
    Run multiple simulations in parallel and collect results
    """
    with mp.Pool(processes=num_cpus) as pool:
        results = pool.map(run_single_simulation, parameter_sets)
    return results

def plot_equity_results(results, output_dir='equity_analysis_plots'):
    """
    Create visualizations for the equity analysis
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract results
    sdi_norm_values = [r['sdi_normalized'] for r in results]
    sdi_theil_values = [r['sdi_theil'] for r in results]
    
    # Plot SDI distributions
    plt.figure(figsize=(10, 6))
    plt.hist(sdi_norm_values, bins=20, alpha=0.7, color='blue')
    plt.title('Distribution of Normalized SDI Values')
    plt.xlabel('SDI Normalized')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, 'sdi_normalized_dist.png'))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.hist(sdi_theil_values, bins=20, alpha=0.7, color='green')
    plt.title('Distribution of Theil SDI Values')
    plt.xlabel('SDI Theil')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, 'sdi_theil_dist.png'))
    plt.close()
    
    # Create summary statistics
    summary = pd.DataFrame({
        'SDI_Normalized': sdi_norm_values,
        'SDI_Theil': sdi_theil_values
    })
    
    summary.describe().to_csv(os.path.join(output_dir, 'equity_metrics_summary.csv'))

# Example parameter sets with varying subsidy configurations
def generate_parameter_sets(base_parameters, num_sets=20):
    parameter_sets = []
    subsidy_pools = np.linspace(1000, 40000, num_sets)
    
    for pool_size in subsidy_pools:
        params = {
            **base_parameters,
            'subsidy_config': SubsidyPoolConfig('daily', float(pool_size))
        }
        parameter_sets.append(params)
    
    return parameter_sets

if __name__ == "__main__":
    # Load your base parameters
    base_parameters = {
        # 'db_connection_string': DB_CONNECTION_STRING,
    'num_commuters': num_commuters,
    'grid_width': 55,
    'grid_height': 55,
    'data_income_weights': [0.5, 0.3, 0.2],
    'data_health_weights': [0.9, 0.1],
    'data_payment_weights': [0.8, 0.2],
    'data_age_distribution': {(18, 25): 0.2, (26, 35): 0.3, (36, 45): 0.2, (46, 55): 0.15, (56, 65): 0.1, (66, 75): 0.05},
    'data_disability_weights': [0.2, 0.8],
    'data_tech_access_weights': [0.95, 0.05],
    'CHANCE_FOR_INSERTING_RANDOM_TRAFFIC': 0.2,
    'ASC_VALUES': {'car': 0, 'bike': 0, 'public': 0, 'walk': 0, 'maas': 0, 'default': 0},
    'UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS': {'beta_C': -0.05, 'beta_T': -0.06},
    'UTILITY_FUNCTION_BASE_COEFFICIENTS': {'beta_C': -0.05, 'beta_T': -0.06, 'beta_W': -0.01, 'beta_A': -0.01, 'alpha': -0.01},
    'PENALTY_COEFFICIENTS': {'disability_bike_walk': 0.8, 'age_health_bike_walk': 0.3, 'no_tech_access_car_bike': 0.1},
    'AFFORDABILITY_THRESHOLDS': {'low': 25, 'middle': 85, 'high': 250},
    'FLEXIBILITY_ADJUSTMENTS': {'low': 1.05, 'medium': 1.0, 'high': 0.95},
    'VALUE_OF_TIME': {'low': 9.64, 'middle': 23.7, 'high': 67.2},
    'public_price_table': {'train': {'on_peak': 2, 'off_peak': 1.5}, 'bus': {'on_peak': 1, 'off_peak': 0.8}},
    'ALPHA_VALUES': {'UberLike1': 0.5, 'UberLike2': 0.5, 'BikeShare1': 0.5, 'BikeShare2': 0.5},
    'DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS': {'S_base': 0.08, 'alpha': 0.2, 'delta': 0.5},
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
    'simulation_steps': SIMULATION_STEPS,
    'subsidy_dataset': {
            'low': {'bike': 0.1, 'car': 0.05, 'MaaS_Bundle': 0.1},
            'middle': {'bike': 0.3, 'car': 0.01, 'MaaS_Bundle': 0.5},
            'high': {'bike': 0.4, 'car': 0, 'MaaS_Bundle': 0.02}
        }
    }
    
    # Generate parameter sets
    parameter_sets = generate_parameter_sets(base_parameters)
    
    # Run simulations
    num_cpus = mp.cpu_count() - 1  # Leave one CPU free
    results = run_parallel_simulations(parameter_sets, num_cpus)
    
    # Save results
    with open('sensitivity_check_socioeconomic_distribution.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # Generate plots and summary statistics
    plot_equity_results(results)