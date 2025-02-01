from sqlalchemy import create_engine, func, case, and_, or_
from sqlalchemy.orm import sessionmaker
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd
from run_visualisation_03 import MobilityModel
from agent_service_provider_initialisation_03 import reset_database, CommuterInfoLog, ServiceBookingLog
import multiprocessing as mp
import os
from agent_subsidy_pool import SubsidyPoolConfig
import json
from scipy.stats import qmc

SIMULATION_STEPS = 144 # 24 hours * 6 steps per hour
STEPS_PER_HOUR = 6
HOURS_PER_DAY = 24
num_commuters = 120
def get_time_period(step):
    """
    Define time periods more precisely:
    Morning peak: 7:00-10:00 (steps 42-60)
    Evening peak: 16:00-19:00 (steps 96-114)
    """
    step = step % 144
    if 42 <= step < 60:
        return 'morning_peak'
    elif 96 <= step < 114:
        return 'evening_peak'
    return 'off_peak'

def calculate_gini(values):
    """Calculate Gini coefficient with proper handling of edge cases"""
    if len(values) < 2 or sum(values) == 0:
        return 0
    
    values = np.sort(values)
    n = len(values)
    cumsum = np.cumsum(values)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

def calculate_tdi(session, time_step, income_level):
    """
    Calculate Base Temporal Distribution Index (TDI)
    TDI = 1 - Gini(S_t / U_t) where:
    S_t = Subsidy amount in time period t
    U_t = Number of users in time period t
    """
    try:
        # Query subsidies and users per time period for specific income level
        temporal_data = session.query(
            ServiceBookingLog.start_time,
            func.sum(ServiceBookingLog.government_subsidy).label('subsidy'),
            func.count(ServiceBookingLog.commuter_id.distinct()).label('users')
        ).join(
            CommuterInfoLog,  # Join with commuter info to filter by income
            CommuterInfoLog.commuter_id == ServiceBookingLog.commuter_id
        ).filter(
            CommuterInfoLog.income_level == income_level  # Filter by income level
        ).group_by(
            ServiceBookingLog.start_time % 144
        ).all()
        
        # Calculate subsidy per user ratios
        ratios = []
        for _, subsidy, users in temporal_data:
            if users > 0 and subsidy is not None:  # Proper null checking
                ratios.append(subsidy/users)
        
        # Calculate TDI only if we have data
        if ratios:
            return 1 - calculate_gini(ratios)
        return 0
        
    except Exception as e:
        print(f"Error calculating TDI: {e}")
        return 0

def calculate_ppe(session, income_level):
    """Modified PPE calculation with better normalization"""
    try:
        peak_times = [(42, 60), (96, 114)]
        
        # Add income level filter to queries
        peak_query = session.query(
            func.sum(ServiceBookingLog.government_subsidy) /
            func.count(func.distinct(ServiceBookingLog.commuter_id))
        ).join(
            CommuterInfoLog,
            CommuterInfoLog.commuter_id == ServiceBookingLog.commuter_id
        ).filter(
            CommuterInfoLog.income_level == income_level,
            or_(*[
                and_(
                    ServiceBookingLog.start_time % 144 >= start,
                    ServiceBookingLog.start_time % 144 < end
                )
                for start, end in peak_times
            ])
        )
        peak_subsidy_per_user = peak_query.scalar() or 0
        
        # Calculate for off-peak
        offpeak_query = session.query(
            func.sum(ServiceBookingLog.government_subsidy) / 
            func.count(func.distinct(ServiceBookingLog.commuter_id))
        ).filter(
            and_(
                or_(
                    ServiceBookingLog.start_time % 144 < 42,
                    and_(
                        ServiceBookingLog.start_time % 144 >= 60,
                        ServiceBookingLog.start_time % 144 < 96
                    ),
                    ServiceBookingLog.start_time % 144 >= 114
                )
            )
        )
        offpeak_subsidy_per_user = offpeak_query.scalar() or 0
        
        # Calculate equity score
        if peak_subsidy_per_user == 0 and offpeak_subsidy_per_user == 0:
            return 0
            
        max_subsidy = max(peak_subsidy_per_user, offpeak_subsidy_per_user)
        if max_subsidy == 0:
            return 1
            
        difference = abs(peak_subsidy_per_user - offpeak_subsidy_per_user)
        ppe = 1 - (difference / max_subsidy)
        
        return max(0, min(1, ppe))
        
    except Exception as e:
        print(f"Error calculating PPE: {e}")
        return 0

def calculate_tas(session, income_level):
    """Calculate Temporal Access Score focusing on subsidy utilization across time periods"""
    try:
        # Define peak period ratios we want to encourage
        target_ratios = {
            'morning_peak': 0.35,  # Ideally 35% during morning peak
            'evening_peak': 0.35,  # 35% during evening peak
            'off_peak': 0.30       # 30% during off-peak
        }

        # Get subsidized trips for each period
        def get_period_subsidy_stats(start_slot, end_slot):
            return session.query(
                func.sum(ServiceBookingLog.government_subsidy),
                func.count(ServiceBookingLog.request_id)
            ).join(
                CommuterInfoLog,
                CommuterInfoLog.commuter_id == ServiceBookingLog.commuter_id
            ).filter(
                CommuterInfoLog.income_level == income_level,
                ServiceBookingLog.government_subsidy > 0,
                and_(
                    ServiceBookingLog.start_time % 144 >= start_slot,
                    ServiceBookingLog.start_time % 144 < end_slot
                )
            ).first()

        # Calculate actual ratios
        morning_stats = get_period_subsidy_stats(42, 60)
        evening_stats = get_period_subsidy_stats(96, 114)
        off_peak_stats = get_period_subsidy_stats(0, 42)  # Early morning
        off_peak_stats2 = get_period_subsidy_stats(60, 96)  # Mid-day
        off_peak_stats3 = get_period_subsidy_stats(114, 144)  # Late evening

        # Combine off-peak stats
        total_off_peak_subsidy = sum(x[0] or 0 for x in [off_peak_stats, off_peak_stats2, off_peak_stats3])
        total_off_peak_trips = sum(x[1] or 0 for x in [off_peak_stats, off_peak_stats2, off_peak_stats3])

        # Get total subsidized trips
        total_subsidy = sum([
            morning_stats[0] or 0,
            evening_stats[0] or 0,
            total_off_peak_subsidy
        ])
        total_trips = sum([
            morning_stats[1] or 0,
            evening_stats[1] or 0,
            total_off_peak_trips
        ])

        if total_trips == 0 or total_subsidy == 0:
            return 0.3  # Base score if no activity

        # Calculate actual ratios
        actual_ratios = {
            'morning_peak': (morning_stats[1] or 0) / total_trips if total_trips > 0 else 0,
            'evening_peak': (evening_stats[1] or 0) / total_trips if total_trips > 0 else 0,
            'off_peak': total_off_peak_trips / total_trips if total_trips > 0 else 0
        }

        # Calculate score based on how close actual ratios are to target ratios
        score = 0
        for period in target_ratios:
            difference = abs(target_ratios[period] - actual_ratios[period])
            period_score = 1 - (difference / target_ratios[period])
            score += max(0, period_score)

        # Normalize and scale final score
        final_score = (score / 3) * 0.7 + 0.3  # Scale to range [0.3, 1.0]
        
        return final_score

    except Exception as e:
        print(f"Error calculating TAS: {e}")
        return 0.3
    

def calculate_combined_tei(session, income_level, weights=(0.4, 0.3, 0.3)):
    """Calculate combined TEI score with proper normalization"""
    try:
        # Get normalized components
        tdi = calculate_tdi(session, None, income_level)
        ppe = calculate_ppe(session, income_level)
        tas = calculate_tas(session, income_level)
        
        # Handle missing TAS values
        tas = 0 if pd.isna(tas) else tas
        
        # Calculate weighted sum
        tei = (
            weights[0] * tdi +
            weights[1] * ppe +
            weights[2] * tas
        )
        
        return {
            'tei': tei,
            'components': {
                'tdi': tdi,
                'ppe': ppe,
                'tas': tas
            }
        }
        
    except Exception as e:
        print(f"Error calculating TEI: {e}")
        return {
            'tei': 0,
            'components': {
                'tdi': 0,
                'ppe': 0,
                'tas': 0
            }
        }

def calculate_time_period_metrics(session, income_level):
    """Calculate metrics for each time period by income level"""
    try:
        base_query = session.query(
            func.sum(ServiceBookingLog.government_subsidy).label('subsidy'),
            func.count(func.distinct(ServiceBookingLog.commuter_id)).label('users')
        ).join(
            CommuterInfoLog,
            CommuterInfoLog.commuter_id == ServiceBookingLog.commuter_id
        ).filter(
            CommuterInfoLog.income_level == income_level
        )

        # Morning peak query
        morning_peak = base_query.filter(
            and_(
                ServiceBookingLog.start_time % 144 >= 42,
                ServiceBookingLog.start_time % 144 < 60
            )
        ).first()

        # Evening peak query
        evening_peak = base_query.filter(
            and_(
                ServiceBookingLog.start_time % 144 >= 96,
                ServiceBookingLog.start_time % 144 < 114
            )
        ).first()

        # Off-peak query
        off_peak = base_query.filter(
            or_(
                ServiceBookingLog.start_time % 144 < 42,
                and_(
                    ServiceBookingLog.start_time % 144 >= 60,
                    ServiceBookingLog.start_time % 144 < 96
                ),
                ServiceBookingLog.start_time % 144 >= 114
            )
        ).first()

        return {
            'morning_peak_subsidy': morning_peak.subsidy if morning_peak.subsidy else 0,
            'morning_peak_users': morning_peak.users if morning_peak.users else 0,
            'evening_peak_subsidy': evening_peak.subsidy if evening_peak.subsidy else 0,
            'evening_peak_users': evening_peak.users if evening_peak.users else 0,
            'off_peak_subsidy': off_peak.subsidy if off_peak.subsidy else 0,
            'off_peak_users': off_peak.users if off_peak.users else 0
        }
    except Exception as e:
        print(f"Error calculating time period metrics: {e}")
        return {
            'morning_peak_subsidy': 0,
            'morning_peak_users': 0,
            'evening_peak_subsidy': 0,
            'evening_peak_users': 0,
            'off_peak_subsidy': 0,
            'off_peak_users': 0
        }

def run_single_simulation(params):
    """Run a single simulation and calculate TEI metrics"""
    analysis_type = params.pop('_analysis_type', None)
    varied_mode = params.pop('varied_mode', None)
    
    db_path = f"service_provider_database_{os.getpid()}.db"
    db_connection_string = f"sqlite:///{db_path}"
    
    try:
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
        
        simulation_steps = params.pop('simulation_steps')
        model = MobilityModel(db_connection_string=db_connection_string, **params)
        model.run_model(simulation_steps)

        # Calculate TEI metrics for each income level separately
        results = {}
        for income_level in ['low', 'middle', 'high']:
            tei_results = calculate_combined_tei(session, income_level)
            time_period_metrics = calculate_time_period_metrics(session, income_level)  # Add income_level parameter
            
            results[income_level] = {
                'tei': tei_results['tei'],
                'components': tei_results['components'],
                **time_period_metrics
            }

        # Add analysis type specific information
        if analysis_type == 'PBS':
            results['subsidy_percentages'] = params['subsidy_dataset']
            results['varied_mode'] = varied_mode
        else:  # FPS
            results['subsidy_pool'] = params['subsidy_config'].total_amount
            
        results['_analysis_type'] = analysis_type
        
        return results
        
    except Exception as e:
        print(f"Error in simulation {os.getpid()}: {str(e)}")
        raise
    finally:
        session.close()
        if os.path.exists(db_path):
            os.remove(db_path)

def analyze_time_period_subsidies(df, output_dir):
    """Create visualization comparing subsidies across time periods"""
    plt.figure(figsize=(12, 8))
    
        # Set style elements
    plt.style.use('default')  # Use default matplotlib style
    plt.grid(True, linestyle='--', alpha=0.7, which='both')
    
    # Customize the plot style
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10
    })
    
    # Plot morning peak subsidies
    morning_mask = df['time_period'] == 'morning_peak'
    plt.scatter(df[morning_mask]['Subsidy_Pool'], 
               df[morning_mask]['avg_subsidy_per_user'],
               color='blue', marker='o', label='Morning Peak', alpha=0.6)
    
    # Fit trend line for morning peak
    z_morning = np.polyfit(df[morning_mask]['Subsidy_Pool'], 
                          df[morning_mask]['avg_subsidy_per_user'], 2)
    p_morning = np.poly1d(z_morning)
    x_smooth = np.linspace(df['Subsidy_Pool'].min(), df['Subsidy_Pool'].max(), 100)
    plt.plot(x_smooth, p_morning(x_smooth), '--', color='blue', alpha=0.5)
    
    # Plot evening peak subsidies
    evening_mask = df['time_period'] == 'evening_peak'
    plt.scatter(df[evening_mask]['Subsidy_Pool'], 
               df[evening_mask]['avg_subsidy_per_user'],
               color='red', marker='s', label='Evening Peak', alpha=0.6)
    
    # Fit trend line for evening peak
    z_evening = np.polyfit(df[evening_mask]['Subsidy_Pool'], 
                          df[evening_mask]['avg_subsidy_per_user'], 2)
    p_evening = np.poly1d(z_evening)
    plt.plot(x_smooth, p_evening(x_smooth), '--', color='red', alpha=0.5)
    
    # Plot off-peak subsidies
    offpeak_mask = df['time_period'] == 'off_peak'
    plt.scatter(df[offpeak_mask]['Subsidy_Pool'], 
               df[offpeak_mask]['avg_subsidy_per_user'],
               color='green', marker='^', label='Off Peak', alpha=0.6)
    
    # Fit trend line for off-peak
    z_offpeak = np.polyfit(df[offpeak_mask]['Subsidy_Pool'], 
                          df[offpeak_mask]['avg_subsidy_per_user'], 2)
    p_offpeak = np.poly1d(z_offpeak)
    plt.plot(x_smooth, p_offpeak(x_smooth), '--', color='green', alpha=0.5)
    
    # Customize plot
    plt.xlabel('Subsidy Pool Size')
    plt.ylabel('Average Subsidy per User')
    plt.title('Average Subsidy per User by Time Period')
    plt.legend()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, 'time_period_subsidy_analysis.png'))
    plt.close()

def create_fps_visualizations(results, output_dir='tei_analysis_plots'):
    """Create comprehensive visualizations for TEI analysis"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    # Convert results to DataFrame with new structure
    df_rows = []
    for r in results:
        for income_level in ['low', 'middle', 'high']:
            if income_level in r:
                df_rows.append({
                    'Subsidy_Pool': r['subsidy_pool'],
                    'Income_Level': income_level,
                    'TEI': r[income_level]['tei'],
                    'TDI': r[income_level]['components']['tdi'],
                    'PPE': r[income_level]['components']['ppe'],
                    'TAS': r[income_level]['components']['tas'],
                    'morning_peak_subsidy': r[income_level].get('morning_peak_subsidy', 0),
                    'morning_peak_users': r[income_level].get('morning_peak_users', 1),
                    'evening_peak_subsidy': r[income_level].get('evening_peak_subsidy', 0),
                    'evening_peak_users': r[income_level].get('evening_peak_users', 1),
                    'off_peak_subsidy': r[income_level].get('off_peak_subsidy', 0),
                    'off_peak_users': r[income_level].get('off_peak_users', 1)
                })

    df = pd.DataFrame(df_rows)
    
    # Create time period analysis data
    time_period_data = []
    for r in results:
        subsidy_pool = r['subsidy_pool']
        
        # Calculate morning peak averages
        morning_peak_subsidy = r.get('morning_peak_subsidy', 0)
        morning_peak_users = max(r.get('morning_peak_users', 1), 1)
        
        # Calculate evening peak averages
        evening_peak_subsidy = r.get('evening_peak_subsidy', 0)
        evening_peak_users = max(r.get('evening_peak_users', 1), 1)
        
        # Calculate off-peak averages
        off_peak_subsidy = r.get('off_peak_subsidy', 0)
        off_peak_users = max(r.get('off_peak_users', 1), 1)
        
        time_period_data.extend([
            {
                'Subsidy_Pool': subsidy_pool,
                'time_period': 'morning_peak',
                'avg_subsidy_per_user': morning_peak_subsidy / morning_peak_users
            },
            {
                'Subsidy_Pool': subsidy_pool,
                'time_period': 'evening_peak',
                'avg_subsidy_per_user': evening_peak_subsidy / evening_peak_users
            },
            {
                'Subsidy_Pool': subsidy_pool,
                'time_period': 'off_peak',
                'avg_subsidy_per_user': off_peak_subsidy / off_peak_users
            }
        ])
    
    time_period_df = pd.DataFrame(time_period_data)
    
    # Create time period analysis plot
    analyze_time_period_subsidies(time_period_df, output_dir)
    


    # 1. Component Trends Plot
    plt.figure(figsize=(12, 6))
    metrics = ['TDI', 'PPE', 'TAS', 'TEI']
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f1c40f']
    
    for metric, color in zip(metrics, colors):
        plt.scatter(df['Subsidy_Pool'], df[metric], 
                   label=metric, color=color, alpha=0.6)
        
        # Add trend line
        z = np.polyfit(df['Subsidy_Pool'], df[metric], 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(df['Subsidy_Pool'].min(), df['Subsidy_Pool'].max(), 100)
        plt.plot(x_smooth, p(x_smooth), '--', color=color, alpha=0.8)

    plt.title('TEI Components vs Subsidy Pool Size')
    plt.xlabel('Subsidy Pool Size')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'tei_components_trend.png'))
    plt.close()

    # 2. Component Distribution Plot
    plt.figure(figsize=(10, 6))
    box_data = [df[metric] for metric in metrics]
    box_plot = plt.boxplot(box_data, labels=metrics, patch_artist=True)
    
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    plt.title('Distribution of TEI Components')
    plt.ylabel('Score')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'tei_components_distribution.png'))
    plt.close()

    # 3. Correlation Heatmap
    plt.figure(figsize=(8, 6))
    correlation_matrix = df[metrics].corr()
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='coolwarm', 
                vmin=-1, 
                vmax=1,
                fmt='.2f')
    plt.title('Correlation between TEI Components')
    plt.savefig(os.path.join(output_dir, 'tei_correlation_heatmap.png'))
    plt.close()

    # 4. Time Series Analysis (if temporal data available)
    plt.figure(figsize=(12, 6))
    
    # Sort by subsidy pool for proper line plotting
    df = df.sort_values('Subsidy_Pool')
    
    # Plot actual data points
    plt.scatter(df['Subsidy_Pool'], df['TEI'], 
               color='blue', alpha=0.6, 
               label='TEI Scores',
               zorder=3)
    
    # Add polynomial trend line
    z = np.polyfit(df['Subsidy_Pool'], df['TEI'], 2)
    p = np.poly1d(z)
    x_trend = np.linspace(df['Subsidy_Pool'].min(), df['Subsidy_Pool'].max(), 100)
    plt.plot(x_trend, p(x_trend), 'r--', 
            alpha=0.8, label='Trend Line',
            zorder=2)
    
    # Calculate and plot confidence band
    y_pred = p(df['Subsidy_Pool'])
    std_dev = np.std(df['TEI'] - y_pred)
    plt.fill_between(x_trend, 
                    p(x_trend) - std_dev,
                    p(x_trend) + std_dev,
                    alpha=0.2, color='blue',
                    label='Uncertainty Range',
                    zorder=1)
    
    plt.title('TEI Score Trend with Uncertainty')
    plt.xlabel('Subsidy Pool Size')
    plt.ylabel('TEI Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'tei_trend_analysis.png'))
    plt.close()

    # 5. Component Contribution Stacked Area Plot
    plt.figure(figsize=(12, 6))
    x = df['Subsidy_Pool']
    y1 = df['TDI'] * 0.4  # Weight factor
    y2 = df['PPE'] * 0.3  # Weight factor
    y3 = df['TAS'] * 0.3  # Weight factor

    plt.stackplot(x, [y1, y2, y3], 
                 labels=['TDI (40%)', 'PPE (30%)', 'TAS (30%)'],
                 colors=['#2ecc71', '#3498db', '#e74c3c'],
                 alpha=0.6)

    plt.title('Weighted Component Contributions to TEI')
    plt.xlabel('Subsidy Pool Size')
    plt.ylabel('Contribution to TEI Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'tei_component_contributions.png'))
    plt.close()

    # Save summary statistics
    summary = df[metrics].describe()
    summary.to_csv(os.path.join(output_dir, 'tei_summary_statistics.csv'))

    return {
        'average_tei': df['TEI'].mean(),
        'optimal_pool': df.loc[df['TEI'].idxmax(), 'Subsidy_Pool'],
        'component_correlations': correlation_matrix.to_dict(),
        'summary_stats': summary.to_dict(),
        'time_period_analysis': {
            'morning_peak_avg': time_period_df[time_period_df['time_period'] == 'morning_peak']['avg_subsidy_per_user'].mean(),
            'evening_peak_avg': time_period_df[time_period_df['time_period'] == 'evening_peak']['avg_subsidy_per_user'].mean(),
            'off_peak_avg': time_period_df[time_period_df['time_period'] == 'off_peak']['avg_subsidy_per_user'].mean()
        }
    }

def analyze_peak_patterns(results):
    """Analyze patterns in peak vs off-peak equity"""
    df = pd.DataFrame([{
        'Subsidy_Pool': r['subsidy_pool'],
        'PPE': r['components']['ppe']
    } for r in results])

    peak_analysis = {
        'avg_ppe': df['PPE'].mean(),
        'ppe_std': df['PPE'].std(),
        'best_ppe': df['PPE'].max(),
        'worst_ppe': df['PPE'].min(),
        'optimal_pool_for_ppe': df.loc[df['PPE'].idxmax(), 'Subsidy_Pool']
    }

    return peak_analysis

def generate_tei_report(results, output_dir):
    """Generate comprehensive TEI analysis report"""
    vis_results = create_fps_visualizations(results, output_dir)
    peak_analysis = analyze_peak_patterns(results)
    
    report = {
        'overall_metrics': vis_results,
        'peak_analysis': peak_analysis,
        'recommendations': {
            'optimal_pool_size': vis_results['optimal_pool'],
            'peak_optimization': peak_analysis['optimal_pool_for_ppe']
        }
    }

    # Save report
    with open(os.path.join(output_dir, 'tei_analysis_report.json'), 'w') as f:
        json.dump(report, f, indent=4)

    return report

def find_optimal_tei_subsidy(x, y):
    """Find optimal subsidy percentage that maximizes TEI score"""
    z = np.polyfit(x, y, 2)
    p = np.poly1d(z)
    
    a, b, c = z
    if a == 0:  # Linear case
        return x[np.argmax(y)]
    
    critical_point = -b / (2 * a)
    
    # Check if critical point is within range
    x_min, x_max = min(x), max(x)
    if critical_point < x_min:
        return x_min
    elif critical_point > x_max:
        return x_max
    
    return critical_point

def analyze_optimal_tei_subsidies(df):
    """Analyze optimal subsidy percentages for each mode"""
    income_levels = ['low', 'middle', 'high']
    modes = ['MaaS_Bundle', 'car', 'bike']
    subsidy_columns = {
        'MaaS_Bundle': 'MaaS_Subsidy',
        'car': 'Car_Subsidy',
        'bike': 'Bike_Subsidy'
    }
    
    optimal_subsidies = {}
    
    for income in income_levels:
        optimal_subsidies[income] = {}
        
        for mode in modes:
            mode_data = df[
                (df['Income_Level'] == income) & 
                (df['varied_mode'] == mode)
            ]
            
            if not mode_data.empty and len(mode_data) > 2:
                x = mode_data[subsidy_columns[mode]].values
                y = mode_data['TEI'].values
                
                optimal_subsidy = find_optimal_tei_subsidy(x, y)
                
                z = np.polyfit(x, y, 2)
                p = np.poly1d(z)
                optimal_tei = p(optimal_subsidy)
                
                optimal_subsidies[income][mode] = {
                    'optimal_percentage': optimal_subsidy,
                    'predicted_tei': optimal_tei
                }
    
    return optimal_subsidies

def create_tei_score_vs_subsidy_percentage(df, output_dir):
    """Create visualization comparing TEI scores across subsidy percentages"""
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    income_levels = ['low', 'middle', 'high']
    
    subsidy_mapping = {
        'MaaS_Bundle': ('green', 'MaaS Subsidy', 'MaaS_Subsidy'),
        'car': ('coral', 'Car Subsidy', 'Car_Subsidy'),
        'bike': ('blue', 'Bike Subsidy', 'Bike_Subsidy')
    }

    optimal_subsidies = analyze_optimal_tei_subsidies(df)

    for i, income in enumerate(income_levels):
        ax = axes[i]
        
        for mode, (color, label, column) in subsidy_mapping.items():
            mode_data = df[
                (df['Income_Level'] == income) & 
                (df['varied_mode'] == mode)
            ]

            if not mode_data.empty:
                ax.scatter(mode_data[column], mode_data['TEI'], 
                         color=color, alpha=0.6, label=label)
                
                if len(mode_data) > 2:
                    x = mode_data[column].values
                    y = mode_data['TEI'].values
                    x_min = min(x)
                    x_max = max(x)
                    
                    z = np.polyfit(x, y, 2)
                    p = np.poly1d(z)
                    x_smooth = np.linspace(x_min, x_max, 100)
                    ax.plot(x_smooth, p(x_smooth), '--', color=color, alpha=0.7,
                           linewidth=2)

                    if income in optimal_subsidies and mode in optimal_subsidies[income]:
                        opt = optimal_subsidies[income][mode]
                        ax.plot(opt['optimal_percentage'], opt['predicted_tei'], 
                               'o', color=color, 
                               markersize=10, markerfacecolor='none',
                               markeredgewidth=2)
                        
                        ax.annotate(f'Optimal: {opt["optimal_percentage"]:.3f}',
                                  xy=(opt['optimal_percentage'], opt['predicted_tei']),
                                  xytext=(10, 10), textcoords='offset points',
                                  bbox=dict(facecolor='white', edgecolor=color, alpha=0.7),
                                  arrowprops=dict(arrowstyle='->'))

        ax.set_title(f'{income.capitalize()} Income')
        ax.set_ylabel('TEI Score')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        ax.set_ylim(min(df['TEI']) - 0.05, max(df['TEI']) + 0.05)

    plt.xlabel('Subsidy Percentage')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'Income_tei_vs_subsidies_combined.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return optimal_subsidies

def create_temporal_subsidy_analysis(df, output_dir):
    """Create temporal analysis plots for PBS"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    time_periods = ['morning_peak', 'evening_peak', 'off_peak']
    income_levels = ['low', 'middle', 'high']
    
    # Fix: Updated column mapping to match DataFrame structure
    subsidy_columns = {
        'MaaS_Bundle': 'MaaS_Subsidy',  # Match the column names from create_pbs_visualizations
        'car': 'Car_Subsidy',
        'bike': 'Bike_Subsidy'
    }
    
    colors = {'MaaS_Bundle': 'green', 'car': 'coral', 'bike': 'blue'}
    
    for i, period in enumerate(time_periods):
        ax = axes[i]
        
        for mode in ['MaaS_Bundle', 'car', 'bike']:
            for income in income_levels:
                data = df[
                    (df['Income_Level'] == income) &
                    (df['varied_mode'] == mode)
                ]
                
                if not data.empty:
                    subsidy_col = subsidy_columns[mode]  # Use the correct mapping
                    temporal_col = f'{period}_avg_subsidy'
                    
                    ax.scatter(data[subsidy_col], data[temporal_col],
                             label=f'{income.capitalize()} {mode}',
                             color=colors[mode],
                             alpha=0.6)
                    
                    if len(data) > 2:
                        z = np.polyfit(data[subsidy_col], data[temporal_col], 2)
                        p = np.poly1d(z)
                        x_smooth = np.linspace(data[subsidy_col].min(),
                                             data[subsidy_col].max(), 100)
                        ax.plot(x_smooth, p(x_smooth), '--',
                               color=colors[mode], alpha=0.5)
        
        ax.set_title(f'{period.replace("_", " ").title()} Analysis')
        ax.set_xlabel('Subsidy Percentage')
        ax.set_ylabel('Average Subsidy per User')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temporal_subsidy_analysis.png'),
                bbox_inches='tight')
    plt.close()

def create_component_analysis_plot(df, output_dir):
    """Create plot showing TEI component trends across subsidy percentages"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    components = ['TDI', 'PPE', 'TAS']
    income_levels = ['low', 'middle', 'high']
    
    subsidy_columns = {
        'MaaS_Bundle': 'MaaS_Subsidy',
        'car': 'Car_Subsidy',
        'bike': 'Bike_Subsidy'
    }
    
    subsidy_types = ['MaaS_Bundle', 'car', 'bike']
    
    for row, subsidy_type in enumerate(subsidy_types):
        for col, income in enumerate(income_levels):
            ax = axes[row, col]
            
            income_data = df[
                (df['Income_Level'] == income) & 
                (df['varied_mode'] == subsidy_type)
            ]
            
            subsidy_column = subsidy_columns[subsidy_type]
            
            for component in components:
                ax.scatter(income_data[subsidy_column], income_data[component], 
                          label=component, alpha=0.7)
                
                if len(income_data) > 1:
                    z = np.polyfit(income_data[subsidy_column], income_data[component], 2)
                    p = np.poly1d(z)
                    x_smooth = np.linspace(income_data[subsidy_column].min(),
                                         income_data[subsidy_column].max(), 100)
                    ax.plot(x_smooth, p(x_smooth), '--', alpha=0.5)
            
            if row == 0:
                ax.set_title(f'{income.capitalize()} Income')
            if col == 0:
                ax.set_ylabel(f'{subsidy_type}\nComponent Score')
            if row == 2:
                ax.set_xlabel('Subsidy Percentage')
            
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'component_analysis.png'))
    plt.close()

def create_pbs_visualizations(results, output_dir):
    """Create comprehensive visualizations for PBS TEI analysis"""
    # Convert results to DataFrame with new structure
    df_rows = []
    for r in results:
        if isinstance(r, dict) and 'subsidy_percentages' in r:
            percentages = r['subsidy_percentages']
            varied_mode = r.get('varied_mode', None)
            
            # Iterate through income levels which now contain the component metrics
            for income_level in ['low', 'middle', 'high']:
                if income_level in percentages and income_level in r:
                    income_results = r[income_level]  # Get results for this income level
                    
                    df_rows.append({
                        'Income_Level': income_level,
                        'Bike_Subsidy': percentages[income_level]['bike'],
                        'Car_Subsidy': percentages[income_level]['car'],
                        'MaaS_Subsidy': percentages[income_level]['MaaS_Bundle'],
                        'TDI': income_results['components']['tdi'],
                        'PPE': income_results['components']['ppe'],
                        'TAS': income_results['components']['tas'],
                        'TEI': income_results['tei'],
                        'morning_peak_avg_subsidy': income_results.get('morning_peak_subsidy', 0) / 
                                                  max(income_results.get('morning_peak_users', 1), 1),
                        'evening_peak_avg_subsidy': income_results.get('evening_peak_subsidy', 0) / 
                                                  max(income_results.get('evening_peak_users', 1), 1),
                        'off_peak_avg_subsidy': income_results.get('off_peak_subsidy', 0) / 
                                              max(income_results.get('off_peak_users', 1), 1),
                        'varied_mode': varied_mode
                    })
    
    df = pd.DataFrame(df_rows)
    
    # Create visualizations
    optimal_subsidies = create_tei_score_vs_subsidy_percentage(df, output_dir)
    create_temporal_subsidy_analysis(df, output_dir)
    create_component_analysis_plot(df, output_dir)
    
    # Print optimal subsidy analysis
    print("\nOptimal Subsidy Analysis:")
    print("=" * 50)
    for income in optimal_subsidies:
        print(f"\n{income.capitalize()} Income:")
        print("-" * 30)
        for mode in optimal_subsidies[income]:
            opt = optimal_subsidies[income][mode]
            print(f"{mode}:")
            print(f"  Optimal subsidy: {opt['optimal_percentage']:.3f}")
            print(f"  Predicted TEI: {opt['predicted_tei']:.3f}")
    
    return df.describe()

def generate_parameter_sets(analysis_type, base_parameters, num_simulations):
    """
    Generate parameter sets based on analysis type with exact number of simulations
    """
    parameter_sets = []
    
    if analysis_type == 'FPS':
        # Fixed pool simulation remains the same
        subsidy_pools = np.linspace(1000, 40000, num_simulations)
        
        for pool_size in subsidy_pools:
            params = {
                **base_parameters,
                'subsidy_dataset': {
                    'low': {'bike': 0.317, 'car': 0.176, 'MaaS_Bundle': 0.493},
                    'middle': {'bike': 0.185, 'car': 0.199, 'MaaS_Bundle': 0.383},
                    'high': {'bike': 0.201, 'car': 0.051, 'MaaS_Bundle': 0.297}
                },
                'subsidy_config': SubsidyPoolConfig('daily', float(pool_size)),
                '_analysis_type': 'FPS'
            }
            parameter_sets.append(params)
            
    elif analysis_type == 'PBS':
        modes = ['bike', 'car', 'MaaS_Bundle']
        points_per_mode = num_simulations // len(modes)
        
        # Generate parameters for each mode separately
        parameter_sets = []
        for mode in modes:
            for i in range(points_per_mode):
                # Create completely unique subsidy configurations for each run
                subsidy_config = {
                    'low': {
                        'bike': np.random.uniform(0.2, 0.45),
                        'car': np.random.uniform(0.15, 0.35),
                        'MaaS_Bundle': np.random.uniform(0.3, 0.6)
                    },
                    'middle': {
                        'bike': np.random.uniform(0.15, 0.35),
                        'car': np.random.uniform(0.1, 0.25),
                        'MaaS_Bundle': np.random.uniform(0.25, 0.5)
                    },
                    'high': {
                        'bike': np.random.uniform(0.1, 0.3),
                        'car': np.random.uniform(0.05, 0.15),
                        'MaaS_Bundle': np.random.uniform(0.2, 0.4)
                    }
                }
                
                # Only systematically vary the target mode
                target_range = {
                    'low': {'bike': (0.2, 0.45), 'car': (0.15, 0.35), 'MaaS_Bundle': (0.3, 0.6)},
                    'middle': {'bike': (0.15, 0.35), 'car': (0.1, 0.25), 'MaaS_Bundle': (0.25, 0.5)},
                    'high': {'bike': (0.1, 0.3), 'car': (0.05, 0.15), 'MaaS_Bundle': (0.2, 0.4)}
                }
                
                # Override the target mode with systematic variation
                pct = i / (points_per_mode - 1)
                for income in ['low', 'middle', 'high']:
                    min_val, max_val = target_range[income][mode]
                    subsidy_config[income][mode] = min_val + (max_val - min_val) * pct

                params = {
                    **base_parameters,
                    'subsidy_dataset': subsidy_config,
                    'subsidy_config': SubsidyPoolConfig('daily', float('inf')),
                    '_analysis_type': 'PBS',
                    'varied_mode': mode
                }
                parameter_sets.append(params)
    
    return parameter_sets

def create_visualizations(results, analysis_type, output_dir='tei_analysis_plots'):
    """
    Create visualizations based on analysis type
    
    Args:
        results (list): Simulation results
        analysis_type (str): Either 'FPS' or 'PBS'
        output_dir (str): Output directory for plots
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if analysis_type == 'FPS':
        return create_fps_visualizations(results, output_dir)
    else:
        return create_pbs_visualizations(results, output_dir)
    
def run_tei_analysis(analysis_type, base_parameters, num_simulations, num_cpus):
    """Run TEI analysis with specified type and number of simulations"""
    print(f"Starting {analysis_type} analysis with {num_simulations} points...")
    
    parameter_sets = generate_parameter_sets(analysis_type, base_parameters, num_simulations)
    
    with mp.Pool(processes=num_cpus) as pool:
        results = pool.map(run_single_simulation, parameter_sets)
    
    with open(f'tei_{analysis_type.lower()}_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    summary_stats = create_visualizations(results, analysis_type)
    return results, summary_stats


if __name__ == "__main__":
    # Define base parameters
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
        'ASC_VALUES': {'car': 100, 'bike': 100, 'public': 100, 'walk': 100, 'maas': 100, 'default': 0},
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
        'simulation_steps': SIMULATION_STEPS
    }

    # # Generate parameter sets with varying subsidy pools
    # subsidy_pools = np.linspace(1000, 40000, 8)  # Test 20 different subsidy pool sizes
    # print(f"Testing subsidy pools from {subsidy_pools[0]} to {subsidy_pools[-1]}")

    # parameter_sets = []
    # for pool_size in subsidy_pools:
    #     params = {
    #         **base_parameters,
    #         'subsidy_config': SubsidyPoolConfig('daily', float(pool_size))
    #     }
    #     parameter_sets.append(params)

    # # Run parallel simulations
    # print(f"Starting TEI analysis with {len(parameter_sets)} parameter combinations...")
    # num_cpus = min(8, mp.cpu_count())  # Use up to 8 CPUs
    # print(f"Using {num_cpus} CPU cores for parallel processing")

    # try:
    #     with mp.Pool(processes=num_cpus) as pool:
    #         results = pool.map(run_single_simulation, parameter_sets)
            
    #     print("Simulations completed successfully")
        
    #     # Save raw results
    #     with open('tei_analysis_results.pkl', 'wb') as f:
    #         pickle.dump(results, f)
    #     print("Raw results saved to tei_analysis_results.pkl")
        
    #     # Generate analysis and visualizations
    #     print("Generating comprehensive analysis and visualizations...")
    #     report = generate_tei_report(results, 'tei_analysis_plots')
        
    #     # Print key findings
    #     print("\nKey Findings from TEI Analysis:")
    #     print(f"Average TEI Score: {report['overall_metrics']['average_tei']:.3f}")
    #     print(f"Optimal Subsidy Pool Size: {report['recommendations']['optimal_pool_size']:.0f}")
    #     print("\nPeak Period Analysis:")
    #     print(f"Average Peak Period Equity: {report['peak_analysis']['avg_ppe']:.3f}")
    #     print(f"Best Peak Period Equity: {report['peak_analysis']['best_ppe']:.3f}")
    #     print(f"Optimal Pool Size for Peak Equity: {report['peak_analysis']['optimal_pool_for_ppe']:.0f}")
        
    #     # Print component correlations
    #     print("\nComponent Correlations:")
    #     correlations = report['overall_metrics']['component_correlations']
    #     for metric in ['TDI', 'PPE', 'TAS']:
    #         print(f"{metric} correlation with TEI: {correlations[metric]['TEI']:.3f}")
            
    # except Exception as e:
    #     print(f"Error during analysis: {str(e)}")
    #     import traceback
    #     traceback.print_exc()

    # Run analysis for both types
    fps_results, fps_stats = run_tei_analysis('FPS', base_parameters, 25, 8) 
    #pbs_results, pbs_stats = run_tei_analysis('PBS', base_parameters, 75, 8)
    
    # Compare results
    print("\nComparison of FPS and PBS approaches:")
    print("\nFPS Summary:")
    print(fps_stats)
    print("\nPBS Summary:")
    #print(pbs_stats)
