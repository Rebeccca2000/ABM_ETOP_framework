from sqlalchemy import create_engine, func, case, and_, or_, not_
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
from collections import defaultdict  
from matplotlib.gridspec import GridSpec
import json
import traceback

SIMULATION_STEPS = 144 # 24 hours * 6 steps per hour
STEPS_PER_HOUR = 6
HOURS_PER_DAY = 24
num_commuters = 120

def is_peak_time(time_step):
    """
    Determines if a given time step is during peak hours
    Peak hours are:
    - Morning peak: 6:30-10:00 (steps 36-60)
    - Evening peak: 15:00-19:00 (steps 90-114)
    """
    step = time_step % 144  # 144 steps per day (24 hours * 6 steps per hour)
    return (36 <= step < 60) or (90 <= step < 114)

def normalize_with_bounds(value: float, min_val: float, max_val: float) -> float:
    """Normalize values with proper handling of edge cases with bounds protection"""
    if max_val == min_val:
        return 0
    normalized = (value - min_val) / (max_val - min_val)
    return max(0, min(1, normalized))  # Ensure output is between 0 and 1

def calculate_msi(session, income_level):
    """Calculate Mode Shift Index with proper income level filtering"""
    try:
        # Query specifically for this income level's trips
        mode_usage = session.query(
            ServiceBookingLog.record_company_name,
            func.count(ServiceBookingLog.request_id).label('trip_count')
        ).join(
            CommuterInfoLog,
            CommuterInfoLog.commuter_id == ServiceBookingLog.commuter_id
        ).filter(
            CommuterInfoLog.income_level == income_level
        ).group_by(
            ServiceBookingLog.record_company_name
        ).all()

        # Income-specific mode weights
        mode_weights = {
            'low': {
                'walk': 1.0,
                'bike': 0.9,
                'public': 0.8,
                'MaaS_Bundle': 0.7,
                'UberLike1': 0.4,
                'UberLike2': 0.4
            },
            'middle': {
                'walk': 0.95,
                'bike': 0.85,
                'public': 0.75,
                'MaaS_Bundle': 0.65,
                'UberLike1': 0.45,
                'UberLike2': 0.45
            },
            'high': {
                'walk': 0.9,
                'bike': 0.8,
                'public': 0.7,
                'MaaS_Bundle': 0.6,
                'UberLike1': 0.5,
                'UberLike2': 0.5
            }
        }

        total_weighted_score = 0
        total_trips = 0

        for mode, trip_count in mode_usage:
            weight = mode_weights[income_level].get(mode, 0.3)
            total_weighted_score += trip_count * weight
            total_trips += trip_count

        return total_weighted_score / (total_trips * max(mode_weights[income_level].values())) if total_trips > 0 else 0

    except Exception as e:
        print(f"Error calculating MSI for {income_level}: {e}")
        return 0

def calculate_pri(session, income_level):
    """Measure how effectively subsidies enable transport mode accessibility"""
    try:
        priority_weights = {'low': 0.5, 'middle': 0.3, 'high': 0.2}
        
        def get_mode_diversity(has_subsidy):
            query = session.query(
                func.count(func.distinct(ServiceBookingLog.record_company_name))
            ).select_from(ServiceBookingLog).join(
                CommuterInfoLog,
                ServiceBookingLog.commuter_id == CommuterInfoLog.commuter_id
            ).filter(
                CommuterInfoLog.income_level == income_level,
                ServiceBookingLog.government_subsidy > 0 if has_subsidy 
                else ServiceBookingLog.government_subsidy == 0
            ).scalar() or 0
            
            return query / 5.0

        mode_diversity_with_subsidy = get_mode_diversity(True)
        mode_diversity_without_subsidy = get_mode_diversity(False)
        
        impact = max(0, mode_diversity_with_subsidy - mode_diversity_without_subsidy)
        
        fairness = session.query(
            func.avg(ServiceBookingLog.government_subsidy)
        ).select_from(ServiceBookingLog).join(
            CommuterInfoLog,
            ServiceBookingLog.commuter_id == CommuterInfoLog.commuter_id
        ).filter(
            CommuterInfoLog.income_level == income_level,
            ServiceBookingLog.government_subsidy > 0
        ).scalar() or 0
        
        fairness_normalized = min(1.0, fairness / 100)
        
        return impact * priority_weights[income_level] * fairness_normalized

    except Exception as e:
        print(f"Error calculating PRI: {e}")
        return 0
    
def calculate_tud(session, income_level):
    """Calculate Temporal Usage Distribution based on actual trip distribution"""
    try:
        # Get hourly distribution of trips
        hourly_trips = defaultdict(int)
        total_trips = 0
        
        trips_query = session.query(
            ServiceBookingLog.start_time
        ).join(
            CommuterInfoLog,
            CommuterInfoLog.commuter_id == ServiceBookingLog.commuter_id
        ).filter(
            CommuterInfoLog.income_level == income_level
        )

        for trip in trips_query:
            hour = (trip.start_time % 144) // 6  # Convert to hour (0-23)
            hourly_trips[hour] += 1
            total_trips += 1

        if total_trips == 0:
            return 0.0

        # Calculate distribution entropy
        distribution = []
        for hour in range(24):
            prob = hourly_trips[hour] / total_trips if total_trips > 0 else 0
            if prob > 0:
                distribution.append(prob)

        if not distribution:
            return 0.0

        # Calculate entropy
        entropy = -sum(p * np.log(p) for p in distribution)
        max_entropy = -np.log(1/24)  # Maximum possible entropy (uniform distribution)
        
        # Normalize entropy to [0,1] range
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Consider peak/off-peak balance
        peak_count = sum(hourly_trips[h] for h in [7,8,9,16,17,18])  # Peak hours
        offpeak_count = total_trips - peak_count
        balance_ratio = min(peak_count, offpeak_count) / max(peak_count, offpeak_count) if max(peak_count, offpeak_count) > 0 else 0

        # Combine metrics
        return (0.6 * normalized_entropy + 0.4 * balance_ratio)

    except Exception as e:
        print(f"Error calculating TUD for {income_level}: {e}")
        return 0.0

def calculate_bsr(session, weights, income_level):
    """Calculate BSR with income-specific component weights"""
    try:
        # Income-specific component weights
        income_weights = {
            'low': {'msi': 0.45, 'pri': 0.35, 'tud': 0.20},
            'middle': {'msi': 0.40, 'pri': 0.35, 'tud': 0.25},
            'high': {'msi': 0.35, 'pri': 0.35, 'tud': 0.30}
        }

        # Calculate base components
        msi = calculate_msi(session, income_level)
        pri = calculate_pri(session, income_level)
        tud = calculate_tud(session, income_level)

        # Apply income-specific weights
        w = income_weights[income_level]
        bsr = (w['msi'] * msi + 
               w['pri'] * pri + 
               w['tud'] * tud)

        # Store components for analysis
        components = {
            'income_level': income_level,
            'msi_value': msi,
            'pri_value': pri,
            'tud_value': tud
        }

        return bsr, components

    except Exception as e:
        print(f"Error calculating BSR for {income_level}: {e}")
        return 0, {
            'income_level': income_level,
            'msi_value': 0,
            'pri_value': 0,
            'tud_value': 0
        }
    
def create_fps_visualizations(results, output_dir):
    """Create comprehensive visualizations for BSR analysis with enhanced features"""
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fps_dir = os.path.join(output_dir, 'fps_analysis')
        os.makedirs(fps_dir, exist_ok=True)  
        
        df = pd.DataFrame([{
            'Subsidy': r.get('subsidy_pool', 0),
            'BSR': r.get('bsr', 0),  # Use get() with default
            'MSI': r.get('msi', 0),
            'PRI': r.get('pri', 0),
            'TUD': r.get('tud', 0)
        } for r in results if r])  # Filter out None results
        
        if df.empty:
            print("Warning: No valid results to visualize")
            return None
            
        # 1. Enhanced Component Trends Plot
        plt.figure(figsize=(12, 8))
        metrics = ['MSI', 'PRI', 'TUD', 'BSR']
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f1c40f']
        
        for metric, color in zip(metrics, colors):
            # Scatter plot
            plt.scatter(df['Subsidy'], df[metric], 
                       label=metric, color=color, alpha=0.6, s=50)
            
            # Add trend line with polynomial fit
            z = np.polyfit(df['Subsidy'], df[metric], 2)
            p = np.poly1d(z)
            x_smooth = np.linspace(df['Subsidy'].min(), df['Subsidy'].max(), 100)
            plt.plot(x_smooth, p(x_smooth), '--', color=color, alpha=0.8)
            
            # Add confidence intervals
            std_dev = df[metric].std()
            plt.fill_between(x_smooth, 
                            p(x_smooth) - std_dev,
                            p(x_smooth) + std_dev,
                            color=color, alpha=0.1)

        # Add optimal point annotation for BSR
        optimal_point = df.loc[df['BSR'].idxmax()]
        plt.annotate(f'Optimal Pool Size: {optimal_point["Subsidy"]:.0f}\nBSR: {optimal_point["BSR"]:.3f}',
                    xy=(optimal_point['Subsidy'], optimal_point['BSR']),
                    xytext=(10, 20), textcoords='offset points',
                    bbox=dict(facecolor='white', edgecolor='black', alpha=0.7),
                    arrowprops=dict(arrowstyle='->'))

        plt.title('BSR Components vs Subsidy Pool Size', fontsize=14, pad=20)
        plt.xlabel('Subsidy Pool Size', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'bsr_FPS_components_trend.png'), bbox_inches='tight')
        plt.close()

        # 2. Enhanced Component Distribution Plot
        plt.figure(figsize=(10, 8))
        box_data = [df[metric] for metric in metrics]
        
        # Create violin plot with box plot overlay
        violin_parts = plt.violinplot(box_data, showmeans=True)
        box_plot = plt.boxplot(box_data, labels=metrics, patch_artist=True,
                              medianprops=dict(color="black", linewidth=1.5),
                              flierprops=dict(marker='o', markerfacecolor='gray', 
                                            markersize=8, alpha=0.5))
        
        # Customize violin plots
        for vp in violin_parts['bodies']:
            vp.set_alpha(0.3)
        
        # Customize box plots
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        plt.title('Distribution of BSR - FPS Components', fontsize=14, pad=20)
        plt.ylabel('Score', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add summary statistics
        for idx, metric in enumerate(metrics, 1):
            stats = df[metric].describe()
            plt.text(idx, -0.1, 
                    f'Mean: {stats["mean"]:.3f}\nStd: {stats["std"]:.3f}',
                    horizontalalignment='center',
                    verticalalignment='top',
                    bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7))

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'bsr_components_distribution.png'))
        plt.close()

        # 3. Enhanced Correlation Heatmap
        plt.figure(figsize=(10, 8))
        correlation_matrix = df[metrics].corr()
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, 
                    mask=mask,
                    annot=True, 
                    cmap='RdBu_r', 
                    vmin=-1, 
                    vmax=1,
                    fmt='.2f',
                    square=True,
                    linewidths=0.5,
                    cbar_kws={"shrink": .5})

        plt.title('Correlation between BSR Components', fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'bsr_correlation_heatmap.png'))
        plt.close()

        # 4. New: Temporal Evolution Plot
        plt.figure(figsize=(12, 6))
        
        # Sort by subsidy pool size for temporal view
        df_sorted = df.sort_values('Subsidy')
        
        # Plot temporal evolution of components
        for metric, color in zip(metrics, colors):
            plt.plot(range(len(df_sorted)), df_sorted[metric], 
                    marker='o', color=color, label=metric, linewidth=2)
        
        plt.title('Temporal Evolution of BSR Components', fontsize=14, pad=20)
        plt.xlabel('Simulation Step', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'bsr_temporal_evolution.png'))
        plt.close()

        # Save summary statistics
        summary = df[metrics].describe()
        summary.to_csv(os.path.join(output_dir, 'bsr_summary_statistics.csv'))

        # Calculate additional metrics
        optimal_pools = {
            metric: df.loc[df[metric].idxmax(), 'Subsidy']
            for metric in metrics
        }
        
        stability_scores = {
            metric: 1 - df[metric].std() / df[metric].mean()
            for metric in metrics
        }

        return {
            'average_bsr': df['BSR'].mean(),
            'optimal_pool': optimal_point['Subsidy'],
            'component_correlations': correlation_matrix.to_dict(),
            'summary_stats': summary.to_dict(),
            'optimal_pools': optimal_pools,
            'stability_scores': stability_scores
        }

    except Exception as e:
        print(f"Error creating visualizations: {e}")
        return None

def create_pbs_visualizations(results, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df_rows = []
    for result in results:
        varied_mode = result.get('_varied_mode')
        subsidy_percentages = result.get('subsidy_percentages', {})
        
        for income_level in ['low', 'middle', 'high']:
            if income_level in result:
                income_data = result[income_level]
                row = {
                    'Income_Level': income_level,
                    'Varied_Mode': varied_mode,
                    'BSR': income_data['bsr'],
                    'MSI': income_data['components']['msi_value'],
                    'PRI': income_data['components']['pri_value'],
                    'TUD': income_data['components']['tud_value'],
                    'Subsidy': subsidy_percentages[income_level][varied_mode]
                }
                df_rows.append(row)
    
    df = pd.DataFrame(df_rows)
    
    # Mode-specific analysis
    plt.figure(figsize=(15, 10))
    for idx, income in enumerate(['low', 'middle', 'high']):
        plt.subplot(3, 1, idx + 1)
        income_data = df[df['Income_Level'] == income]
        
        for mode in income_data['Varied_Mode'].unique():
            mode_data = income_data[income_data['Varied_Mode'] == mode]
            plt.scatter(mode_data['Subsidy'], mode_data['BSR'], 
                        label=mode, alpha=0.6)
            
            if len(mode_data) > 1:
                z = np.polyfit(mode_data['Subsidy'], mode_data['BSR'], 2)
                p = np.poly1d(z)
                x_smooth = np.linspace(mode_data['Subsidy'].min(), 
                                        mode_data['Subsidy'].max(), 100)
                plt.plot(x_smooth, p(x_smooth), '--', alpha=0.5)
        
        plt.title(f'{income.capitalize()} Income Level')
        plt.xlabel('Subsidy Percentage')
        plt.ylabel('BSR Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'income_level_analysis.png'))
    plt.close()

    # Component analysis 
    components = ['MSI', 'PRI', 'TUD']
    modes = df['Varied_Mode'].unique()
    
    # Component analysis per mode
    for component in components:
        plt.figure(figsize=(15, 15))
        plt.suptitle(f'{component} Component Analysis', fontsize=14)
        
        for mode_idx, mode in enumerate(modes):
            for income_idx, income in enumerate(['low', 'middle', 'high']):
                plt.subplot(3, 1, income_idx + 1)
                mode_data = df[(df['Varied_Mode'] == mode) & (df['Income_Level'] == income)]
                
                jitter = np.random.normal(0, 0.001, size=len(mode_data))
                plt.scatter(mode_data['Subsidy'] + jitter, mode_data[component],
                            label=mode, alpha=0.6)
                
                z = fit_polynomial(mode_data['Subsidy'], mode_data[component])
                if z is not None:
                    p = np.poly1d(z)
                    x_smooth = np.linspace(mode_data['Subsidy'].min(),
                                            mode_data['Subsidy'].max(), 100)
                    plt.plot(x_smooth, p(x_smooth), '--', alpha=0.5)
            
        # Set titles and labels for each subplot
        for idx, income in enumerate(['low', 'middle', 'high']):
            plt.subplot(3, 1, idx + 1)
            plt.title(f'{income.capitalize()} Income Level', fontsize=12, pad=10)
            plt.xlabel('Subsidy Percentage')
            plt.ylabel(f'{component} Score')
            plt.legend()
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.minorticks_on()
            plt.grid(True, which='minor', alpha=0.1, linestyle=':')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{component.lower()}_component_analysis.png'))
        plt.close()


    # Different Matrics analysis
    metrics = ['BSR', 'MSI', 'PRI', 'TUD']
    
    for metric in metrics:
        plt.clf()
        fig = plt.figure(figsize=(15, 14))
        gs = fig.add_gridspec(4, 1, height_ratios=[0.2, 1, 1, 1])
        
        # Add header
        header_ax = fig.add_subplot(gs[0])
        header_ax.text(0.5, 0.5, f'{metric} Score Analysis', 
                      fontsize=16, fontweight='bold', 
                      horizontalalignment='center',
                      verticalalignment='center')
        header_ax.axis('off')
        
        # Create subplots for each income level
        for idx, income in enumerate(['low', 'middle', 'high']):
            ax = fig.add_subplot(gs[idx + 1])
            income_data = df[df['Income_Level'] == income]
            
            for mode in income_data['Varied_Mode'].unique():
                mode_data = income_data[income_data['Varied_Mode'] == mode]
                ax.scatter(mode_data['Subsidy'], mode_data[metric], 
                         label=mode, alpha=0.6)
                
                z = fit_polynomial(mode_data['Subsidy'], mode_data[metric])
                if z is not None:
                    p = np.poly1d(z)
                    x_smooth = np.linspace(mode_data['Subsidy'].min(), 
                                         mode_data['Subsidy'].max(), 100)
                    ax.plot(x_smooth, p(x_smooth), '--', alpha=0.5)
            
            ax.set_title(f'{income.capitalize()} Income Level')
            ax.set_xlabel('Subsidy Percentage')
            ax.set_ylabel(f'{metric} Score')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric.lower()}_analysis.png'))
        plt.close()

    # Comparative analysis
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Varied_Mode', y='BSR', hue='Income_Level')
    plt.title('BSR Distribution by Mode and Income Level')
    plt.savefig(os.path.join(output_dir, 'comparative_analysis.png'))
    plt.close()

    return df.describe()


def run_single_simulation(params):
    """Run single simulation with separated income-level BSR calculations"""
    db_path = f"service_provider_database_{os.getpid()}.db"
    db_connection_string = f"sqlite:///{db_path}"
    
    try:
        analysis_type = params.pop('_analysis_type', None)
        varied_mode = params.pop('varied_mode', None)
        weights = params.pop('weights', None)
        simulation_steps = params.pop('simulation_steps')
        
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
        
        model = MobilityModel(db_connection_string=db_connection_string, **params)
        model.run_model(simulation_steps)

        # Calculate BSR separately for each income level
        results = {}
        for income_level in ['low', 'middle', 'high']:
            income_bsr, components = calculate_bsr(session, weights, income_level)
            
            # Store results with separate component values
            results[income_level] = {
                'bsr': income_bsr,
                'components': components,
                'subsidy_percentage': params['subsidy_dataset'][income_level].get(varied_mode if varied_mode else 'MaaS_Bundle')
            }

        # Add metadata
        results['_analysis_type'] = analysis_type
        results['_varied_mode'] = varied_mode
        if analysis_type == 'PBS':
            results['subsidy_percentages'] = params['subsidy_dataset']
        else:
            results['subsidy_pool'] = params['subsidy_config'].total_amount

        return results

    except Exception as e:
        print(f"Error in simulation {os.getpid()}: {str(e)}")
        traceback.print_exc()
        raise
    finally:
        session.close()
        if os.path.exists(db_path):
            os.remove(db_path)

def generate_parameter_sets(analysis_type, base_parameters, num_simulations):
    """
    Generate parameter sets based on analysis type with controlled number of simulations.
    Handles both Fixed Pool Subsidy (FPS) and Percentage-Based Subsidy (PBS) analyses.
    """
    parameter_sets = []
    
    if analysis_type == 'FPS':
        # Fixed pool simulation with specified number of points
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
                '_analysis_type': 'FPS',
                'weights': {
                    'alpha': 0.4,  # MSI weight
                    'beta': 0.3,   # PRI weight
                    'gamma': 0.3,  # TUD weight
                    'interaction_weight': 0.1,  # Weight for MSI-PRI interaction
                    'temporal_equity_weight': 0.1  # Weight for temporal equity
                }
            }
            parameter_sets.append(params)
            
    elif analysis_type == 'PBS':
        modes = ['bike', 'car', 'MaaS_Bundle']
        points_per_mode = num_simulations // len(modes)
        print(f"Generating {points_per_mode} points per mode")
        
        # Define subsidy ranges for each mode and income level
        subsidy_ranges = {
            'low': {
                'bike': (0.2, 0.45),    # 20-45% subsidy for bikes
                'car': (0.15, 0.35),    # 15-35% subsidy for cars
                'MaaS_Bundle': (0.3, 0.6)  # 30-60% subsidy for MaaS
            },
            'middle': {
                'bike': (0.15, 0.35),   # 15-35% subsidy
                'car': (0.1, 0.25),     # 10-25% subsidy
                'MaaS_Bundle': (0.25, 0.5)  # 25-50% subsidy
            },
            'high': {
                'bike': (0.1, 0.3),     # 10-30% subsidy
                'car': (0.05, 0.15),    # 5-15% subsidy
                'MaaS_Bundle': (0.2, 0.4)   # 20-40% subsidy
            }
        }

        for mode in modes:
            # Generate exactly points_per_mode points for each mode
            for i in range(points_per_mode):
                # Create base subsidy configuration with random values
                subsidy_config = {
                    'low': {},
                    'middle': {},
                    'high': {}
                }
                
                # Set random values for non-varied modes
                for income in ['low', 'middle', 'high']:
                    for m in modes:
                        if m != mode:
                            # Use middle point for non-varied modes
                            min_val, max_val = subsidy_ranges[income][m]
                            subsidy_config[income][m] = np.random.uniform(min_val, max_val)
                
                # Systematically vary the target mode
                pct = i / (points_per_mode - 1) if points_per_mode > 1 else 0
                for income in ['low', 'middle', 'high']:
                    min_val, max_val = subsidy_ranges[income][mode]
                    subsidy_config[income][mode] = min_val + (max_val - min_val) * pct

                params = {
                    **base_parameters,
                    'subsidy_dataset': subsidy_config,
                    'subsidy_config': SubsidyPoolConfig('daily', float('inf')),
                    '_analysis_type': 'PBS',
                    'varied_mode': mode,
                    'weights': {
                        'alpha': 0.4,  # MSI weight
                        'beta': 0.3,   # PRI weight
                        'gamma': 0.3,  # TUD weight
                        'interaction_weight': 0.1,
                        'temporal_equity_weight': 0.1
                    }
                }
                parameter_sets.append(params)

        # Debug prints for verification
        for mode in modes:
            mode_count = sum(1 for p in parameter_sets if p['varied_mode'] == mode)
            print(f"Points generated for {mode}: {mode_count}")
            
    print(f"Total parameter sets generated: {len(parameter_sets)}")
    return parameter_sets

def create_visualizations(results, analysis_type, output_dir):
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
    else:  # PBS
        return create_pbs_visualizations(results, output_dir)
    
def run_bsr_analysis(analysis_type, base_parameters, num_simulations, num_cpus):
    """
    Run BSR analysis with enhanced error handling and logging
    """
    print(f"\nStarting {analysis_type} analysis with {num_simulations} simulations...")
    
    try:
        parameter_sets = generate_parameter_sets(analysis_type, base_parameters, num_simulations)
        print(f"Generated {len(parameter_sets)} parameter combinations")

        with mp.Pool(processes=num_cpus) as pool:
            results = pool.map(run_single_simulation, parameter_sets)
            
        valid_results = [r for r in results if r is not None]
        print(f"Successfully completed {len(valid_results)}/{len(parameter_sets)} simulations")

        output_file = f'bsr_{analysis_type.lower()}_results.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(valid_results, f)
        print(f"Results saved to {output_file}")

        output_dir = os.path.join('bsr_analysis_plots', analysis_type.lower())
        summary_stats = create_visualizations(valid_results, analysis_type, output_dir)
        
        print("\nAnalysis Summary:")
        if valid_results:
            # Calculate average BSR across all income levels
            avg_bsr_by_income = {
                income: np.mean([r[income]['bsr'] for r in valid_results])
                for income in ['low', 'middle', 'high']
            }
            
            print("\nAverage BSR Scores by Income Level:")
            for income, score in avg_bsr_by_income.items():
                print(f"{income.capitalize()}: {score:.3f}")
            
            if analysis_type == 'FPS':
                # Find optimal pool size based on average BSR across income levels
                best_result = max(valid_results, 
                                key=lambda r: sum(r[inc]['bsr'] for inc in ['low', 'middle', 'high']))
                print(f"\nOptimal Subsidy Pool Size: {best_result['subsidy_pool']:.0f}")
            else:
                # Find best mode and configuration
                best_result = max(valid_results, 
                                key=lambda r: sum(r[inc]['bsr'] for inc in ['low', 'middle', 'high']))
                print("\nOptimal Subsidy Configuration:")
                print(f"Best Performing Mode: {best_result.get('_varied_mode', 'N/A')}")
                if 'subsidy_percentages' in best_result:
                    print("Optimal Subsidy Percentages:")
                    print(json.dumps(best_result['subsidy_percentages'], indent=2))

        return valid_results, summary_stats

    except Exception as e:
        print(f"Error in BSR analysis: {str(e)}")
        traceback.print_exc()
        return [], None

def fit_polynomial(x, y):
    try:
        # Suppress polyfit warnings
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', np.RankWarning)
            z = np.polyfit(x, y, 2)
        return z
    except Exception as e:
        print(f"Error fitting polynomial: {e}")
        return None

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
        'simulation_steps': SIMULATION_STEPS
    }

    # # Generate parameter sets with varying subsidy pools
    # subsidy_pools = np.linspace(1000, 40000, 8)  # Test 8 different subsidy pool sizes
    # print(f"Testing subsidy pools from {subsidy_pools[0]} to {subsidy_pools[-1]}")

    # # Define weights for BSR calculation
    # weights_sets = [
    #     {'alpha': 0.4, 'beta': 0.3, 'gamma': 0.3},  # Balanced
    #     {'alpha': 0.5, 'beta': 0.3, 'gamma': 0.2},  # Mode shift focused
    #     {'alpha': 0.3, 'beta': 0.5, 'gamma': 0.2}   # Policy response focused
    # ]

    # # Generate parameter combinations
    # parameter_sets = [
    #     {**base_parameters, 
    #      'subsidy_config': SubsidyPoolConfig('daily', float(pool_size)),
    #      'weights': weights_sets[0]}  # Using balanced weights
    #     for pool_size in subsidy_pools
    # ]

    # print(f"Running {len(parameter_sets)} parameter combinations...")

    # # Run parallel simulations
    # num_cpus = 8  # Adjust based on available CPUs
    # with mp.Pool(processes=num_cpus) as pool:
    #     results = pool.map(run_single_simulation, parameter_sets)

    # # Save results
    # with open('bsr_analysis_results.pkl', 'wb') as f:
    #     pickle.dump(results, f)

    # # Generate visualizations and analysis
    # visualizations = create_FPS_visualizations(results)
    
    # # Print key findings
    # print("\nKey Findings from BSR Analysis:")
    # print(f"Average BSR Score: {visualizations['average_bsr']:.3f}")
    # print(f"Optimal Subsidy Pool Size: {visualizations['optimal_pool']:.0f}")
    
    #fps_results, fps_stats = run_fsir_analysis('FPS', base_parameters, 25, 8)
    print("\nRunning PBS Analysis...")
    pbs_results, pbs_stats = run_bsr_analysis('PBS', base_parameters, 75, 8)

     # Compare results
    print("\nComparison of FPS and PBS approaches:")
    print("\nFPS Summary:")
    # print(fps_stats)  # Commented out as per your example
    print("\nPBS Summary:")
    print(pbs_stats)