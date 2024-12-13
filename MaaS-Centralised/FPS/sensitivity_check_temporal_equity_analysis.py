from sqlalchemy import create_engine, func, case
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
def get_time_period(step):
    """
    Convert step number to time period based on your system
    Steps per day = 144 (24 hours * 6 steps per hour)
    Morning peak: steps 36-59 (6:00-9:50)
    Evening peak: steps 90-113 (15:00-18:50)
    """
    step = step % 144
    if 36 <= step < 60:
        return 'morning_peak'
    elif 90 <= step < 114:
        return 'evening_peak'
    return 'off_peak'

def calculate_temporal_gini(time_data):
    """
    Calculate Gini coefficient for temporal distribution
    """
    n = len(time_data)
    if n < 2:
        return 0.0
    
    time_data = time_data.sort_values('subsidy_per_user')
    total_subsidy = time_data['subsidy'].sum()
    ranks = np.arange(1, n + 1)
    
    gini = 1 - (1/n) * (2 * np.sum((n - ranks + 1) * time_data['subsidy'] / (n * total_subsidy)))
    return gini

def calculate_entropy(time_data):
    """
    Calculate entropy for temporal distribution
    """
    time_data['p_t'] = time_data['subsidy'] / time_data['subsidy'].sum()
    entropy = -np.sum(time_data['p_t'] * np.log(time_data['p_t'].replace(0, 1)))
    return entropy

def run_single_simulation(params):
    """
    Run a single simulation and calculate temporal equity metrics
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
        
        reset_database(engine=engine, session=session, **reset_db_params)
        
        simulation_steps = params.pop('simulation_steps')
        
        model = MobilityModel(db_connection_string=db_connection_string, **params)
        model.run_model(simulation_steps)
        
        # Query using updated case syntax
        results = (
            session.query(
                ServiceBookingLog.start_time,
                func.count(ServiceBookingLog.request_id).label('users'),
                func.sum(
                    case(
                        (ServiceBookingLog.government_subsidy.isnot(None),
                         func.json_extract(ServiceBookingLog.government_subsidy, '$')),
                        else_=0
                    )
                ).label('subsidy')
            )
            .group_by(ServiceBookingLog.start_time)
            .all()
        )
        
        df = pd.DataFrame(results, columns=['step', 'users', 'subsidy'])
        df['period'] = df['step'].apply(get_time_period)
        
        period_data = df.groupby('period').agg({
            'subsidy': 'sum',
            'users': 'sum'
        }).reset_index()
        
        period_data['subsidy_per_user'] = period_data['subsidy'] / period_data['users'].replace(0, 1)
        
        gini = calculate_temporal_gini(period_data)
        entropy = calculate_entropy(period_data)
        
        time_weights = {
            'morning_peak': 1.2,
            'evening_peak': 1.2,
            'off_peak': 0.8
        }
        weighted_data = period_data.copy()
        weighted_data['subsidy'] = weighted_data['subsidy'] * weighted_data['period'].map(time_weights)
        weighted_tei = 1 - calculate_temporal_gini(weighted_data)
        
        return {
            'temporal_gini': gini,
            'entropy': entropy,
            'weighted_tei': weighted_tei,
            'period_data': period_data.to_dict(),
            'subsidy_config': params['subsidy_config'].total_amount if params['subsidy_config'] else None  # Changed from pool_size to total_amount
        }
        
    except Exception as e:
        print(f"Error in simulation {os.getpid()}: {str(e)}")
        raise
        
    finally:
        session.close()
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except Exception as e:
                print(f"Error removing database file: {str(e)}")
def create_enhanced_tei_plots(results, output_dir='temporal_equity_plots'):
    """
    Create comprehensive visualizations for TEI analysis
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract data into DataFrame
    df = pd.DataFrame([{
        'Temporal_Gini': r['temporal_gini'],
        'Weighted_TEI': r['weighted_tei'],
        'Entropy': r['entropy'],
        'Subsidy_Pool': r['subsidy_config'],
        'Period_Data': r['period_data']
    } for r in results])
    
    # 1. Scatter Plot with Trendlines
    plt.figure(figsize=(12, 6))
    
    # Plot scatter points
    plt.scatter(df['Subsidy_Pool'], df['Temporal_Gini'], 
               alpha=0.6, label='Temporal Gini', color='blue')
    plt.scatter(df['Subsidy_Pool'], df['Weighted_TEI'], 
               alpha=0.6, label='Weighted TEI', color='red')
    
    # Add trendlines
    for metric, color in [('Temporal_Gini', 'blue'), ('Weighted_TEI', 'red')]:
        z = np.polyfit(df['Subsidy_Pool'], df[metric], 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(df['Subsidy_Pool'].min(), df['Subsidy_Pool'].max(), 200)
        plt.plot(x_smooth, p(x_smooth), '--', color=color, alpha=0.8, 
                label=f'{metric} Trend')
    
    plt.xlabel('Subsidy Pool Size')
    plt.ylabel('Equity Metric')
    plt.title('Temporal Equity Metrics vs Subsidy Pool Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'tei_scatter_trends.png'))
    plt.close()
    
    # 2. Average Subsidy per User by Time Period Scatter Plot
    plt.figure(figsize=(12, 6))
    
    colors = {'morning_peak': 'blue', 'evening_peak': 'red', 'off_peak': 'green'}
    markers = {'morning_peak': 'o', 'evening_peak': 's', 'off_peak': '^'}
    
    # Process period data
    period_metrics = {period: {'subsidy': [], 'users': [], 'pool_size': []} 
                     for period in ['morning_peak', 'evening_peak', 'off_peak']}
    
    for idx, row in df.iterrows():
        period_data = pd.DataFrame(row['Period_Data'])
        subsidy_pool = row['Subsidy_Pool']
        
        for period in period_metrics.keys():
            period_row = period_data[period_data['period'] == period]
            if not period_row.empty:
                subsidy = period_row['subsidy'].values[0]
                users = period_row['users'].values[0]
                if users > 0:  # Avoid division by zero
                    period_metrics[period]['subsidy'].append(subsidy / users)
                    period_metrics[period]['pool_size'].append(subsidy_pool)
    
    # Plot each period
    for period, metrics in period_metrics.items():
        if metrics['subsidy']:  # Check if we have data
            plt.scatter(metrics['pool_size'], metrics['subsidy'],
                       label=f'{period.replace("_", " ").title()}',
                       color=colors[period], marker=markers[period], alpha=0.6)
            
            # Add trendline
            if len(metrics['pool_size']) > 1:
                z = np.polyfit(metrics['pool_size'], metrics['subsidy'], 2)
                p = np.poly1d(z)
                x_smooth = np.linspace(min(metrics['pool_size']), 
                                    max(metrics['pool_size']), 200)
                plt.plot(x_smooth, p(x_smooth), '--', color=colors[period], alpha=0.5)
    
    plt.xlabel('Subsidy Pool Size')
    plt.ylabel('Average Subsidy per User')
    plt.title('Average Subsidy per User by Time Period')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'average_subsidy_by_period.png'))
    plt.close()
    
    # 3. Heatmap of Temporal Distribution
    plt.figure(figsize=(10, 6))
    
    # Create heatmap data
    subsidy_ranges = np.linspace(df['Subsidy_Pool'].min(), 
                               df['Subsidy_Pool'].max(), 10)
    gini_ranges = np.linspace(df['Temporal_Gini'].min(), 
                            df['Temporal_Gini'].max(), 10)
    
    heatmap_data = np.zeros((len(gini_ranges)-1, len(subsidy_ranges)-1))
    
    for i in range(len(subsidy_ranges)-1):
        for j in range(len(gini_ranges)-1):
            mask = ((df['Subsidy_Pool'] >= subsidy_ranges[i]) & 
                   (df['Subsidy_Pool'] < subsidy_ranges[i+1]) &
                   (df['Temporal_Gini'] >= gini_ranges[j]) & 
                   (df['Temporal_Gini'] < gini_ranges[j+1]))
            heatmap_data[j, i] = mask.sum()
    
    plt.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    plt.colorbar(label='Count')
    
    # Set labels
    plt.xlabel('Subsidy Pool Size')
    plt.ylabel('Temporal Gini')
    plt.title('Distribution Heatmap of Temporal Equity')
    
    # Set tick labels
    x_ticks = np.linspace(0, len(subsidy_ranges)-2, 5)
    y_ticks = np.linspace(0, len(gini_ranges)-2, 5)
    plt.xticks(x_ticks, 
               [f'{subsidy_ranges[int(i)]:.0f}' for i in x_ticks])
    plt.yticks(y_ticks, 
               [f'{gini_ranges[int(i)]:.2f}' for i in y_ticks])
    
    plt.savefig(os.path.join(output_dir, 'temporal_equity_heatmap.png'))
    plt.close()

    # Save summary statistics
    summary_stats = df[['Temporal_Gini', 'Weighted_TEI', 'Entropy']].describe()
    summary_stats.to_csv(os.path.join(output_dir, 'temporal_equity_summary.csv'))

def plot_temporal_analysis(results, output_dir='temporal_equity_plots'):
    """
    Create visualizations for temporal equity analysis with better empty data handling
    """
    print("Starting temporal analysis plotting...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract metrics and subsidy configs with validation
    metrics_df = pd.DataFrame([{
        'Temporal_Gini': r['temporal_gini'],
        'Entropy': r['entropy'],
        'Weighted_TEI': r['weighted_tei'],
        'Subsidy_Pool': r['subsidy_config']
    } for r in results])
    
    print(f"Collected metrics data shape: {metrics_df.shape}")
    print("Sample of metrics data:")
    print(metrics_df.head())
    
    # Plot metrics vs subsidy pool size if we have valid data
    if not metrics_df.empty and not metrics_df['Subsidy_Pool'].isna().all():
        plt.figure(figsize=(12, 6))
        plt.plot(metrics_df['Subsidy_Pool'], metrics_df['Temporal_Gini'], 'b-', label='Temporal Gini')
        plt.plot(metrics_df['Subsidy_Pool'], metrics_df['Weighted_TEI'], 'r-', label='Weighted TEI')
        plt.xlabel('Subsidy Pool Size')
        plt.ylabel('Equity Metric')
        plt.title('Temporal Equity Metrics vs Subsidy Pool Size')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'temporal_metrics_vs_subsidy.png'))
        plt.close()
    
    # Process period data with validation
    periods = ['morning_peak', 'evening_peak', 'off_peak']
    period_data_collection = []
    
    # Collect and validate period data
    print("\nCollecting period data...")
    for result in results:
        period_data = pd.DataFrame(result['period_data'])
        if not period_data.empty:
            period_data_collection.append(period_data)
            print(f"Found data for periods: {period_data['period'].unique()}")
    
    if period_data_collection:
        # Combine all period data
        all_period_data = pd.concat(period_data_collection, ignore_index=True)
        
        # Calculate averages with error handling
        avg_data = all_period_data.groupby('period').agg({
            'subsidy': lambda x: x.replace([np.inf, -np.inf], np.nan).mean(),
            'users': lambda x: x.replace([np.inf, -np.inf], np.nan).mean()
        }).reset_index()
        
        print("\nAggregated period data:")
        print(avg_data)
        
        # Only create plot if we have valid data
        if not avg_data.empty:
            plt.figure(figsize=(10, 6))
            x = np.arange(len(avg_data))
            width = 0.35
            
            # Create bars with validation
            plt.bar(x - width/2, 
                   avg_data['subsidy'].fillna(0), 
                   width, 
                   label='Average Subsidy')
            plt.bar(x + width/2,
                   avg_data['users'].fillna(0),
                   width, 
                   label='Average Users')
            
            plt.xlabel('Time Period')
            plt.ylabel('Average Value')
            plt.title('Distribution by Time Period')
            plt.xticks(x, avg_data['period'])
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'period_distribution.png'))
            plt.close()
            
            # Save detailed period data
            avg_data.to_csv(os.path.join(output_dir, 'period_summary.csv'), index=False)
    else:
        print("Warning: No valid period data found for plotting")
    
    # Save summary statistics if we have valid metrics
    if not metrics_df.empty:
        metrics_df.describe().to_csv(os.path.join(output_dir, 'temporal_equity_summary.csv'))
        print("\nSummary statistics saved to temporal_equity_summary.csv")

if __name__ == "__main__":
    # Your base parameters here
    base_parameters = {
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
    
    # Generate parameter sets with varying subsidy pools
    parameter_sets = []
    subsidy_pools = np.linspace(1000, 40000, 30)
    for pool_size in subsidy_pools:
        params = {
            **base_parameters,
            'subsidy_config': SubsidyPoolConfig('daily', float(pool_size))
        }
        parameter_sets.append(params)
    
    # Run simulations
    num_cpus = 4
    with mp.Pool(processes=num_cpus) as pool:
        results = pool.map(run_single_simulation, parameter_sets)
    
    # Save results
    with open('temporal_equity_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # Generate plots and summary statistics
    plot_temporal_analysis(results)
    create_enhanced_tei_plots(results)