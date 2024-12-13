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
    print(f"Starting simulation with PID {os.getpid()}")
    db_path = f"service_provider_database_{os.getpid()}.db"
    db_connection_string = f"sqlite:///{db_path}"
    engine = create_engine(db_connection_string)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Extract parameters for reset_database
        reset_db_params = {
            'uber_like1_capacity': params['uber_like1_capacity'],
            'uber_like1_price': params['uber_like1_price'],
            'uber_like2_capacity': params['uber_like2_capacity'],
            'uber_like2_price': params['uber_like2_price'],
            'bike_share1_capacity': params['bike_share1_capacity'],
            'bike_share1_price': params['bike_share1_price'],
            'bike_share2_capacity': params['bike_share2_capacity'],
            'bike_share2_price': params['bike_share2_price']
        }
        
        # Initialize database
        reset_database(engine=engine, session=session, **reset_db_params)
        
        # Extract parameters for MobilityModel
        model_params = params.copy()
        simulation_steps = model_params.pop('simulation_steps')  # Remove and store simulation_steps
        
        # Initialize and run model
        model = MobilityModel(db_connection_string=db_connection_string, **model_params)
        model.run_model(simulation_steps)  # Use simulation_steps here
        
        # Query subsidies by income level
        subsidy_by_income = {}
        user_counts = {}
        
        for income_level in ['low', 'middle', 'high']:
            subsidies = (
                session.query(func.sum(ServiceBookingLog.government_subsidy))
                .join(CommuterInfoLog, CommuterInfoLog.commuter_id == ServiceBookingLog.commuter_id)
                .filter(CommuterInfoLog.income_level == income_level)
                .scalar() or 0
            )
            
            users = (
                session.query(func.count(func.distinct(CommuterInfoLog.commuter_id)))
                .filter(CommuterInfoLog.income_level == income_level)
                .scalar() or 0
            )
            
            subsidy_by_income[income_level] = float(subsidies)
            user_counts[income_level] = users
        
        # Calculate metrics
        sdi_norm = calculate_sdi_normalized(subsidy_by_income, user_counts)
        sdi_theil = calculate_sdi_theil(subsidy_by_income, user_counts)
        
        print(f"Simulation {os.getpid()} completed successfully")
        
        return {
            'sdi_normalized': sdi_norm,
            'sdi_theil': sdi_theil,
            'subsidy_distribution': subsidy_by_income,
            'user_distribution': user_counts
        }
        
    except Exception as e:
        print(f"Error in simulation {os.getpid()}: {str(e)}")
        raise
        
    finally:
        session.close()
        if os.path.exists(db_path):
            os.remove(db_path)

def run_parallel_simulations(parameter_sets, num_cpus):
    """
    Run multiple simulations in parallel and collect results
    """
    with mp.Pool(processes=num_cpus) as pool:
        results = pool.map(run_single_simulation, parameter_sets)
    return results
def create_enhanced_plots(results, output_dir='equity_analysis_plots'):
    """
    Create comprehensive visualizations for SDI analysis using only matplotlib
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create DataFrame from results
    df = pd.DataFrame([{
        'SDI_Normalized': r['sdi_normalized'],
        'SDI_Theil': r['sdi_theil'],
        'subsidy_distribution': r['subsidy_distribution'],  # lowercase
        'user_distribution': r['user_distribution']  # lowercase
    } for r in results])
    
    # 1. Box Plot (instead of violin plot)
    plt.figure(figsize=(12, 6))
    plt.boxplot([df['SDI_Normalized'], df['SDI_Theil']], 
                labels=['SDI_Normalized', 'SDI_Theil'])
    plt.title('Distribution Comparison of SDI Metrics')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'sdi_box_plot.png'))
    plt.close()
    
    # 2. Scatter Plot with Trends
    subsidy_pools = np.linspace(1000, 40000, len(df))
    
    plt.figure(figsize=(12, 6))
    
    # Plot SDI Normalized
    plt.scatter(subsidy_pools, df['SDI_Normalized'], 
               label='SDI Normalized', alpha=0.6, color='blue')
    z = np.polyfit(subsidy_pools, df['SDI_Normalized'], 1)
    p = np.poly1d(z)
    plt.plot(subsidy_pools, p(subsidy_pools), 
            linestyle='--', color='blue', label='Normalized Trend')
    
    # Plot SDI Theil
    plt.scatter(subsidy_pools, df['SDI_Theil'], 
               label='SDI Theil', alpha=0.6, color='red')
    z = np.polyfit(subsidy_pools, df['SDI_Theil'], 1)
    p = np.poly1d(z)
    plt.plot(subsidy_pools, p(subsidy_pools), 
            linestyle='--', color='red', label='Theil Trend')
    
    plt.xlabel('Subsidy Pool Size')
    plt.ylabel('SDI Value')
    plt.title('SDI Metrics vs Subsidy Pool Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'sdi_trends.png'))
    plt.close()
    
    # 3. Alternative to Heatmap: 2D Line Plot
    plt.figure(figsize=(12, 6))
    for income_level in ['low', 'middle', 'high']:
        subsidies = [r['subsidy_distribution'][income_level] for r in results]  # lowercase
        users = [r['user_distribution'][income_level] for r in results]  # lowercase
        plt.plot(subsidies, users, 'o-', label=income_level)
    
    plt.xlabel('Subsidy Amount')
    plt.ylabel('User Count')
    plt.title('Subsidy vs User Distribution by Income Level')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'subsidy_distribution.png'))
    plt.close()
    
    # 4. CDF Plots
    plt.figure(figsize=(12, 6))
    
    # Plot CDF for SDI Normalized
    sorted_norm = np.sort(df['SDI_Normalized'])
    cumulative_prob = np.arange(1, len(sorted_norm) + 1) / len(sorted_norm)
    plt.plot(sorted_norm, cumulative_prob, 
            label='SDI Normalized', linewidth=2, color='blue')
    
    # Plot CDF for SDI Theil
    sorted_theil = np.sort(df['SDI_Theil'])
    cumulative_prob = np.arange(1, len(sorted_theil) + 1) / len(sorted_theil)
    plt.plot(sorted_theil, cumulative_prob, 
            label='SDI Theil', linewidth=2, color='red')
    
    plt.xlabel('SDI Value')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution of SDI Metrics')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'sdi_cdf.png'))
    plt.close()

    # Print summary statistics
    print("\nSummary Statistics:")
    print("\nSDI Normalized:")
    print(df['SDI_Normalized'].describe())
    print("\nSDI Theil:")
    print(df['SDI_Theil'].describe())

def plot_equity_results(results, output_dir='equity_analysis_plots'):
    """
    Create visualizations for the equity analysis with explicit file paths and debug prints
    """
    print("Starting to generate plots...")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Extract results with validation
    sdi_norm_values = [r['sdi_normalized'] for r in results]
    sdi_theil_values = [r['sdi_theil'] for r in results]
    
    print(f"Extracted values for plotting:")
    print(f"SDI Normalized values: {sdi_norm_values}")
    print(f"SDI Theil values: {sdi_theil_values}")
    
    # Plot SDI normalized distribution
    plt.figure(figsize=(10, 6))
    plt.hist(sdi_norm_values, bins=min(20, len(sdi_norm_values)), alpha=0.7, color='blue')
    plt.title('Distribution of Normalized SDI Values')
    plt.xlabel('SDI Normalized')
    plt.ylabel('Frequency')
    
    # Save with full path
    normalized_plot_path = os.path.join(os.getcwd(), output_dir, 'sdi_normalized_dist.png')
    plt.savefig(normalized_plot_path)
    plt.close()
    print(f"Saved normalized SDI plot to: {normalized_plot_path}")
    
    # Plot SDI Theil distribution
    plt.figure(figsize=(10, 6))
    plt.hist(sdi_theil_values, bins=min(20, len(sdi_theil_values)), alpha=0.7, color='green')
    plt.title('Distribution of Theil SDI Values')
    plt.xlabel('SDI Theil')
    plt.ylabel('Frequency')
    
    # Save with full path
    theil_plot_path = os.path.join(os.getcwd(), output_dir, 'sdi_theil_dist.png')
    plt.savefig(theil_plot_path)
    plt.close()
    print(f"Saved Theil SDI plot to: {theil_plot_path}")
    
    # Create summary statistics
    summary = pd.DataFrame({
        'SDI_Normalized': sdi_norm_values,
        'SDI_Theil': sdi_theil_values
    })
    
    # Save summary with full path
    summary_path = os.path.join(os.getcwd(), output_dir, 'equity_metrics_summary.csv')
    summary.describe().to_csv(summary_path)
    print(f"Saved summary statistics to: {summary_path}")
# Example parameter sets with varying subsidy configurations
def generate_parameter_sets(base_parameters, num_sets):
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
            'low': {'bike': 0.3, 'car': 0.35, 'MaaS_Bundle': 0.4},
            'middle': {'bike': 0.3, 'car': 0.15, 'MaaS_Bundle': 0.3},
            'high': {'bike': 0.4, 'car': 0.05, 'MaaS_Bundle': 0.2}
        }
    }
    
    # Generate parameter sets
    parameter_sets = generate_parameter_sets(base_parameters, num_sets=30)
    print(f"Generated {len(parameter_sets)} parameter sets")
    
    # Run simulations
    num_cpus = 4  
    print(f"Starting analysis with {num_cpus} CPUs")
    results = run_parallel_simulations(parameter_sets, num_cpus)
    
    # Save results
    with open('sensitivity_check_socioeconomic_distribution.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # Generate plots and summary statistics
    plot_equity_results(results)
    create_enhanced_plots(results)