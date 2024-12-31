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
from sdi_metrics import calculate_baseline_distribution, calculate_sdi_normalized, calculate_gini, calculate_income_ratios, analyze_temporal_patterns, calculate_efficiency_metrics
SIMULATION_STEPS = 10  # Run for a larger number of steps for better analysis
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
    
    # Weight by population share
    weighted_ratios = []
    for income_level in subsidy_values.keys():
        s_i = subsidy_values[income_level]
        u_i = user_counts[income_level]
        if u_i > 0 and s_total > 0:
            population_share = u_i / u_total
            subsidy_share = s_i / s_total
            weighted_ratios.append((subsidy_share/population_share) * population_share)
            
    return 1 - (sum(weighted_ratios) - min(weighted_ratios)) / (max(weighted_ratios) - min(weighted_ratios))

def calculate_sdi_theil(subsidy_values, user_counts):
    """
    Calculate Theil's Index version of SDI
    """
    s_total = sum(subsidy_values.values())
    u_total = sum(user_counts.values())
    
    theil_sum = 0
    for income_level in subsidy_values.keys():
        s_i = subsidy_values[income_level]
        u_i = user_counts[income_level]
        
        if u_i > 0 and s_i > 0:
            u_share = u_i / u_total
            s_share = s_i / s_total
            theil_sum += s_share * np.log(s_share / u_share)
    
    return theil_sum

def run_single_simulation(params):
    """
    Run a single simulation with comprehensive equity analysis
    """
    print(f"Starting simulation with PID {os.getpid()}")
    db_path = f"service_provider_database_{os.getpid()}.db"
    db_connection_string = f"sqlite:///{db_path}"
    engine = create_engine(db_connection_string)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Define temporal constants
    STEPS_PER_HOUR = 6
    HOURS_PER_DAY = 24
    TOTAL_DAILY_STEPS = STEPS_PER_HOUR * HOURS_PER_DAY
    
    # Initialize enhanced tracking structure
    subsidy_tracking = {
        income_level: {
            'total': 0.0,
            'hourly_subsidies': [0.0] * HOURS_PER_DAY,
            'hourly_users': [0] * HOURS_PER_DAY,
            'step_data': [],
            'outcomes': {
                'co2_reduction': 0.0,
                'completed_trips': 0,
                'total_travel_time': 0.0
            }
        }
        for income_level in ['low', 'middle', 'high']
    }
    
    try:
        # Database initialization remains the same
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
        reset_database(engine=engine, session=session, **reset_db_params)
        
        model_params = params.copy()
        simulation_steps = model_params.pop('simulation_steps')
        model = MobilityModel(db_connection_string=db_connection_string, **model_params)
        model.run_model(simulation_steps)
        
        # Enhanced data collection with outcomes
        for step in range(simulation_steps):
            current_hour = (step % TOTAL_DAILY_STEPS) // STEPS_PER_HOUR
            
            for income_level in ['low', 'middle', 'high']:
                # Query basic metrics
                step_data = (
                    session.query(
                        func.sum(ServiceBookingLog.government_subsidy).label('subsidy'),
                        func.count(func.distinct(ServiceBookingLog.commuter_id)).label('users'),
                        func.sum(ServiceBookingLog.total_time).label('travel_time')
                    )
                    .join(CommuterInfoLog, CommuterInfoLog.commuter_id == ServiceBookingLog.commuter_id)
                    .filter(
                        CommuterInfoLog.income_level == income_level,
                        ServiceBookingLog.start_time == step
                    )
                    .first()
                )
                
                step_subsidy = float(step_data.subsidy or 0.0)
                active_users = int(step_data.users or 0)
                travel_time = float(step_data.travel_time or 0.0)
                
                # Update tracking data with enhanced metrics
                subsidy_tracking[income_level]['step_data'].append({
                    'step': step,
                    'hour': current_hour,
                    'subsidy': step_subsidy,
                    'active_users': active_users,
                    'travel_time': travel_time
                })
                
                # Update aggregates
                subsidy_tracking[income_level]['hourly_subsidies'][current_hour] += step_subsidy
                subsidy_tracking[income_level]['hourly_users'][current_hour] += active_users
                subsidy_tracking[income_level]['total'] += step_subsidy
                subsidy_tracking[income_level]['outcomes']['total_travel_time'] += travel_time
                subsidy_tracking[income_level]['outcomes']['completed_trips'] += active_users
                
        # Calculate final distributions
        baseline = calculate_baseline_distribution()
        subsidy_by_income = {
            level: tracking['total'] 
            for level, tracking in subsidy_tracking.items()
        }
        user_counts = {
            level: len(set(
                record['active_users'] 
                for record in tracking['step_data'] 
                if record['active_users'] > 0
            ))
            for level, tracking in subsidy_tracking.items()
        }
        
        # Calculate comprehensive metrics
        equity_metrics = {
            'distributional': {
                'sdi_normalized': calculate_sdi_normalized(subsidy_by_income, user_counts),  # Remove baseline parameter
                'gini_coefficient': calculate_gini(list(subsidy_by_income.values())),
                'deviation_from_baseline': {
                    level: (subsidy_by_income[level]/sum(subsidy_by_income.values())) - baseline[level]
                    for level in baseline
                }
            },
            'temporal': analyze_temporal_patterns(subsidy_tracking),
            'ratios': {},
            'efficiency': calculate_efficiency_metrics(
                subsidy_by_income,
                {level: tracking['outcomes'] for level, tracking in subsidy_tracking.items()}
            )
        }
        
        # Add income ratios
        ratios, per_user = calculate_income_ratios(subsidy_by_income, user_counts)
        equity_metrics['ratios'] = ratios
        equity_metrics['per_user_subsidies'] = per_user
        
        return {
            'equity_metrics': equity_metrics,
            'raw_data': {
                'subsidy_distribution': subsidy_by_income,
                'user_distribution': user_counts,
                'temporal_data': subsidy_tracking,
                'baseline': baseline
            }
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
def plot_comprehensive_equity_analysis(results, output_dir='socioeconomic_distribution_plots'):
    """
    Create comprehensive visualizations for equity analysis combining both SDI and general equity metrics
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create DataFrame with all metrics
    df = pd.DataFrame([{
        'SDI_Normalized': r['equity_metrics']['distributional']['sdi_normalized'],
        'Gini_Coefficient': r['equity_metrics']['distributional']['gini_coefficient'],
        'Subsidy_Pool': r['raw_data']['baseline'].get('total', 0),
        'Low_Income_Share': r['equity_metrics']['distributional']['deviation_from_baseline']['low'],
        'Middle_Income_Share': r['equity_metrics']['distributional']['deviation_from_baseline']['middle'],
        'High_Income_Share': r['equity_metrics']['distributional']['deviation_from_baseline']['high'],
        'subsidy_distribution': r['raw_data']['subsidy_distribution'],
        'user_distribution': r['raw_data']['user_distribution']
    } for r in results])
    
    # Handle potential zero/invalid values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    # 1. Box Plot of Metrics
    plt.figure(figsize=(12, 6))
    plt.boxplot([df['SDI_Normalized'], df['Gini_Coefficient']], 
                labels=['SDI_Normalized', 'Gini_Coefficient'])
    plt.title('Distribution Comparison of Equity Metrics')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'equity_metrics_box_plot.png'))
    plt.close()
    
    # 2. Scatter Plot with Trends vs Subsidy Pool
    plt.figure(figsize=(12, 6))
    
    # Plot SDI Normalized
    plt.scatter(df['Subsidy_Pool'], df['SDI_Normalized'], 
               label='SDI Normalized', alpha=0.6, color='blue')
    z = np.polyfit(df['Subsidy_Pool'], df['SDI_Normalized'], 1)
    p = np.poly1d(z)
    x_smooth = np.linspace(df['Subsidy_Pool'].min(), df['Subsidy_Pool'].max(), 100)
    plt.plot(x_smooth, p(x_smooth), 
            linestyle='--', color='blue', label='SDI Trend')
    
    # Plot Gini Coefficient
    plt.scatter(df['Subsidy_Pool'], df['Gini_Coefficient'], 
               label='Gini Coefficient', alpha=0.6, color='red')
    z = np.polyfit(df['Subsidy_Pool'], df['Gini_Coefficient'], 1)
    p = np.poly1d(z)
    plt.plot(x_smooth, p(x_smooth), 
            linestyle='--', color='red', label='Gini Trend')
    
    plt.xlabel('Subsidy Pool Size')
    plt.ylabel('Metric Value')
    plt.title('Equity Metrics vs Subsidy Pool Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'metrics_vs_subsidy.png'))
    plt.close()
    
    # 3. Income Share Deviations
    plt.figure(figsize=(12, 6))
    
    income_levels = ['Low_Income_Share', 'Middle_Income_Share', 'High_Income_Share']
    colors = ['blue', 'green', 'red']
    
    for level, color in zip(income_levels, colors):
        plt.plot(df['Subsidy_Pool'], df[level], 'o-', 
                label=level.replace('_', ' '), 
                color=color, 
                alpha=0.6)
    
    plt.xlabel('Subsidy Pool Size')
    plt.ylabel('Deviation from Baseline')
    plt.title('Income Share Deviations by Subsidy Pool Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'income_share_deviations.png'))
    plt.close()
    
    # 4. Subsidy vs User Distribution
    plt.figure(figsize=(12, 6))
    for income_level in ['low', 'middle', 'high']:
        subsidies = [r['raw_data']['subsidy_distribution'][income_level] for r in results]
        users = [r['raw_data']['user_distribution'][income_level] for r in results]
        plt.plot(subsidies, users, 'o-', label=f'{income_level.title()} Income')
    
    plt.xlabel('Subsidy Amount')
    plt.ylabel('User Count')
    plt.title('Subsidy vs User Distribution by Income Level')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'subsidy_distribution.png'))
    plt.close()
    
    # 5. CDF Plots
    plt.figure(figsize=(12, 6))
    
    # Plot CDF for SDI Normalized
    sorted_norm = np.sort(df['SDI_Normalized'])
    cumulative_prob = np.arange(1, len(sorted_norm) + 1) / len(sorted_norm)
    plt.plot(sorted_norm, cumulative_prob, 
            label='SDI Normalized', linewidth=2, color='blue')
    
    # Plot CDF for Gini Coefficient
    sorted_gini = np.sort(df['Gini_Coefficient'])
    cumulative_prob = np.arange(1, len(sorted_gini) + 1) / len(sorted_gini)
    plt.plot(sorted_gini, cumulative_prob, 
            label='Gini Coefficient', linewidth=2, color='red')
    
    plt.xlabel('Metric Value')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution of Equity Metrics')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'equity_metrics_cdf.png'))
    plt.close()

    # Print summary statistics
    print("\nSummary Statistics:")
    print("\nSDI Normalized:")
    print(df['SDI_Normalized'].describe())
    print("\nGini Coefficient:")
    print(df['Gini_Coefficient'].describe())
    print("\nIncome Share Deviations:")
    for level in income_levels:
        print(f"\n{level}:")
        print(df[level].describe())
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
    parameter_sets = generate_parameter_sets(base_parameters, num_sets=5)
    print(f"Generated {len(parameter_sets)} parameter sets")
    
    # Run simulations
    num_cpus = 4  
    print(f"Starting analysis with {num_cpus} CPUs")
    results = run_parallel_simulations(parameter_sets, num_cpus)
    
    # Save results
    with open('sensitivity_check_socioeconomic_distribution.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # Generate plots and summary statistics
plot_comprehensive_equity_analysis(results)