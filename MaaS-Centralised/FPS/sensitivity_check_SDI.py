from sqlalchemy import create_engine, func, case, or_, and_
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
import gc
from scipy.stats import qmc
import math
SIMULATION_STEPS = 30
num_commuters = 120

def calculate_sur(session, income_level):
    """
    Calculate Subsidy Utilization Rate by measuring the effectiveness of subsidies 
    in influencing travel behavior and achieving policy goals.
    
    Key improvements:
    - Measures behavioral response to subsidies
    - Considers temporal distribution
    - Accounts for mode-specific effectiveness
    """
    try:
        # Part 1: Mode-specific subsidy effectiveness
        mode_impacts = {}
        total_trips = 0
        
        for mode in ['bike', 'car', 'MaaS_Bundle']:
            # Get all trips and subsidized trips for this mode and income level
            trips_query = session.query(
                ServiceBookingLog
            ).join(
                CommuterInfoLog
            ).filter(
                CommuterInfoLog.income_level == income_level,
                ServiceBookingLog.record_company_name.like(f'%{mode}%')
            )
            
            total_mode_trips = trips_query.count()
            subsidized_trips = trips_query.filter(
                ServiceBookingLog.government_subsidy > 0
            ).count()
            
            if total_mode_trips > 0:
                # Calculate effectiveness as ratio of subsidized to total trips
                mode_impacts[mode] = subsidized_trips / total_mode_trips
                total_trips += total_mode_trips

        # Part 2: Temporal distribution effectiveness
        peak_periods = [(42, 60), (96, 114)]  # Morning and evening peaks
        peak_utilization = session.query(ServiceBookingLog)\
            .join(CommuterInfoLog, ServiceBookingLog.commuter_id == CommuterInfoLog.commuter_id)\
            .filter(
                CommuterInfoLog.income_level == income_level,
                ServiceBookingLog.government_subsidy > 0,
                or_(
                    *[
                        and_(
                            ServiceBookingLog.start_time % 144 >= start,
                            ServiceBookingLog.start_time % 144 < end
                        ) for start, end in peak_periods
                    ]
                )
            ).count()

        # Calculate temporal distribution score
        if total_trips > 0:
            peak_ratio = peak_utilization / total_trips
            temporal_score = 1 - abs(peak_ratio - 0.6)  # Target: 60% peak, 40% off-peak
        else:
            temporal_score = 0

        # Part 3: Calculate final SUR score
        if mode_impacts:
            mode_score = sum(impact for impact in mode_impacts.values()) / len(mode_impacts)
            sur = (0.7 * mode_score + 0.3 * temporal_score)
            return max(0, min(1, sur))  # Normalize to [0,1]
            
        return 0

    except Exception as e:
        print(f"Error calculating SUR for {income_level}: {e}")
        return 0

def calculate_mae(session, income_level, grid_width=55, grid_height=55):
    """
    Calculate Modal Access Equity with improved measurement of mode availability
    and usage patterns across different user groups.
    
    Parameters:
    - session: SQLAlchemy session for database queries
    - income_level: Income level of the user group being analyzed
    - grid_width: Width of the simulation grid (default=55)
    - grid_height: Height of the simulation grid (default=55)
    
    The grid dimensions are used to calculate spatial coverage of different modes.
    We use these to understand how well each transport mode serves different areas
    of the city grid.
    """
    try:
        # Part 1: Calculate base mode shares
        mode_shares = {}
        total_trips = 0
        
        for mode in ['walk', 'bike', 'car', 'public', 'maas']:
            trips = session.query(
                ServiceBookingLog
            ).join(
                CommuterInfoLog
            ).filter(
                CommuterInfoLog.income_level == income_level,
                ServiceBookingLog.record_company_name.like(f'%{mode}%')
            ).count()
            
            mode_shares[mode] = trips
            total_trips += trips

        # Part 2: Calculate mode availability scores
        # These weights represent the baseline availability of each mode
        availability_weights = {
            'walk': 1.0,    # Always available
            'bike': 0.8,    # Weather/infrastructure dependent
            'car': 0.9,     # High availability
            'public': 0.7,  # Schedule dependent
            'maas': 0.85    # Mixed availability
        }

        # Part 3: Calculate spatial coverage
        # Using the provided grid dimensions to normalize spatial coverage
        coverage_scores = {}
        grid_size = grid_width * grid_height
        
        for mode in mode_shares:
            # Count unique origin-destination pairs for each mode
            covered_cells = session.query(
                func.count(func.distinct(
                    func.concat(
                        ServiceBookingLog.origin_coordinates,
                        ServiceBookingLog.destination_coordinates
                    )
                ))
            ).filter(
                ServiceBookingLog.record_company_name.like(f'%{mode}%')
            ).scalar() or 0
            
            # Normalize by total possible grid cells
            coverage_scores[mode] = covered_cells / grid_size

        # Part 4: Calculate MAE score
        if total_trips > 0:
            weighted_shares = {}
            for mode in mode_shares:
                # Calculate weighted share considering:
                # - Base usage (actual trips)
                # - Mode availability
                # - Spatial coverage
                base_share = mode_shares[mode] / total_trips
                weight = availability_weights[mode]
                coverage = coverage_scores[mode]
                
                weighted_shares[mode] = base_share * weight * coverage

            # Calculate deviation from ideal equal distribution
            ideal_share = 1 / len(mode_shares)
            deviations = [
                abs(share - ideal_share)
                for share in weighted_shares.values()
            ]
            
            # Final MAE score: 1 minus average deviation
            mae = 1 - (sum(deviations) / len(deviations))
            return max(0, min(1, mae))  # Normalize to [0,1]
            
        return 0

    except Exception as e:
        print(f"Error calculating MAE for {income_level}: {e}")
        return 0

def calculate_upi(session, income_level):
    """
    Calculate Usage Pattern Index with enhanced measurement of travel behavior diversity
    and temporal consistency.
    
    Key improvements:
    - Separates peak and off-peak patterns
    - Measures usage regularity
    - Considers trip chaining
    """
    try:
        # Part 1: Calculate temporal usage patterns
        time_periods = {
            'morning_peak': (42, 60),    # 7:00-10:00
            'midday': (60, 96),          # 10:00-16:00
            'evening_peak': (96, 114),   # 16:00-19:00
            'night': (114, 42)           # 19:00-7:00
        }
        
        period_patterns = {}
        total_trips = 0
        
        for period_name, (start, end) in time_periods.items():
            # Get trips in this period
            period_trips = session.query(
                ServiceBookingLog
            ).join(
                CommuterInfoLog
            ).filter(
                CommuterInfoLog.income_level == income_level,
                or_(
                    and_(
                        ServiceBookingLog.start_time % 144 >= start,
                        ServiceBookingLog.start_time % 144 < end
                    ),
                    and_(
                        start > end,
                        or_(
                            ServiceBookingLog.start_time % 144 >= start,
                            ServiceBookingLog.start_time % 144 < end
                        )
                    )
                )
            ).count()
            
            period_patterns[period_name] = period_trips
            total_trips += period_trips

        # Part 2: Calculate mode diversity
        unique_modes = session.query(
            func.count(func.distinct(ServiceBookingLog.record_company_name))
        ).join(
            CommuterInfoLog
        ).filter(
            CommuterInfoLog.income_level == income_level
        ).scalar() or 0
        
        mode_diversity = unique_modes / 5  # Normalize by total possible modes

        # Part 3: Calculate trip chaining
        # Identify sequential trips within 30-minute windows
        chain_trips = session.query(
            func.count(ServiceBookingLog.request_id)
        ).join(
            CommuterInfoLog
        ).filter(
            CommuterInfoLog.income_level == income_level,
            ServiceBookingLog.start_time - func.lag(ServiceBookingLog.start_time).over(
                partition_by=ServiceBookingLog.commuter_id,
                order_by=ServiceBookingLog.start_time
            ) <= 3  # 30 minutes (3 time steps)
        ).scalar() or 0

        # Part 4: Calculate final UPI score
        if total_trips > 0:
            # Calculate temporal distribution score
            temporal_shares = {
                period: trips/total_trips 
                for period, trips in period_patterns.items()
            }
            
            # Ideal shares for each period
            ideal_shares = {
                'morning_peak': 0.3,
                'midday': 0.3,
                'evening_peak': 0.3,
                'night': 0.1
            }
            
            temporal_score = 1 - sum(
                abs(temporal_shares.get(period, 0) - share)
                for period, share in ideal_shares.items()
            ) / 2
            
            # Calculate chain score
            chain_score = chain_trips / total_trips
            
            # Combine scores
            upi = (
                0.4 * temporal_score +
                0.3 * mode_diversity +
                0.3 * chain_score
            )
            
            return max(0, min(1, upi))  # Normalize to [0,1]
            
        return 0

    except Exception as e:
        print(f"Error calculating UPI for {income_level}: {e}")
        return 0

def calculate_sdi(sur, mae, upi, weights=(0.4, 0.3, 0.3)):
    """Calculate combined SDI score"""
    return weights[0] * sur + weights[1] * mae + weights[2] * upi

def calculate_sdi_metrics(session):
    """
    Calculate all SDI metrics for all income levels
    
    Args:
        session: SQLAlchemy session

    Returns:
        dict: Results containing metrics for each income level
    """
    results = {}
    for income_level in ['low', 'middle', 'high']:
        # Calculate individual metrics
        sur = calculate_sur(session, income_level)
        mae = calculate_mae(session, income_level)
        upi = calculate_upi(session, income_level)
        
        # Calculate combined SDI score
        sdi = calculate_sdi(sur, mae, upi)
        
        # Store results for this income level
        results[income_level] = {
            'sur': sur,
            'mae': mae,
            'upi': upi,
            'sdi': sdi
        }
    
    return results 

def run_single_simulation(params):
    """Run a single simulation and calculate SDI metrics"""
    print(f"Starting simulation with PID {os.getpid()}")
    db_path = f"service_provider_database_{os.getpid()}.db"
    db_connection_string = f"sqlite:///{db_path}"
    
    try:
        # Extract analysis type and remove from params
        analysis_type = params.pop('_analysis_type', None)
        varied_mode = params.pop('varied_mode', None)  # Store which mode was varied

        # Initialize database and run simulation
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

        # Calculate all metrics using the helper function
        results = calculate_sdi_metrics(session)
        
        # Add additional information based on analysis type
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

def create_fps_visualizations(results, output_dir):
    """Create comprehensive visualizations for SDI analysis"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convert results to DataFrame
    df_rows = []
    for r in results:
        subsidy_pool = r['subsidy_pool']
        for income_level in ['low', 'middle', 'high']:
            metrics = r[income_level]
            df_rows.append({
                'Subsidy_Pool': subsidy_pool,
                'Income_Level': income_level,
                'SUR': metrics['sur'],
                'MAE': metrics['mae'],
                'UPI': metrics['upi'],
                'SDI': metrics['sdi']
            })
    
    df = pd.DataFrame(df_rows)

    # 1. Individual Metric Trends
    metrics = ['SUR', 'MAE', 'UPI']
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    for idx, metric in enumerate(metrics):
        for income in ['low', 'middle', 'high']:
            data = df[df['Income_Level'] == income]
            axes[idx].scatter(data['Subsidy_Pool'], data[metric], 
                            label=f'{income.capitalize()} Income',
                            alpha=0.6)
            
            # Add trend line
            z = np.polyfit(data['Subsidy_Pool'], data[metric], 2)
            p = np.poly1d(z)
            x_smooth = np.linspace(data['Subsidy_Pool'].min(), data['Subsidy_Pool'].max(), 100)
            axes[idx].plot(x_smooth, p(x_smooth), '--', alpha=0.8)
            
        axes[idx].set_title(f'{metric} vs Subsidy Pool Size')
        axes[idx].set_xlabel('Subsidy Pool Size')
        axes[idx].set_ylabel(metric)
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metric_trends.png'))
    plt.close()

    # 2. Combined SDI Score Trends
    plt.figure(figsize=(12, 6))
    for income in ['low', 'middle', 'high']:
        data = df[df['Income_Level'] == income]
        plt.scatter(data['Subsidy_Pool'], data['SDI'], 
                   label=f'{income.capitalize()} Income',
                   alpha=0.6)
        
        # Add trend line
        z = np.polyfit(data['Subsidy_Pool'], data['SDI'], 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(data['Subsidy_Pool'].min(), data['Subsidy_Pool'].max(), 100)
        plt.plot(x_smooth, p(x_smooth), '--', alpha=0.8)
    
    plt.title('Combined SDI Score vs Subsidy Pool Size')
    plt.xlabel('Subsidy Pool Size')
    plt.ylabel('SDI Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'sdi_trends.png'))
    plt.close()

    # 3. Distribution Analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    metrics = ['SUR', 'MAE', 'UPI', 'SDI']
    
    for idx, (metric, ax) in enumerate(zip(metrics, axes.flatten())):
        sns.boxplot(data=df, x='Income_Level', y=metric, ax=ax)
        ax.set_title(f'{metric} Distribution by Income Level')
        ax.set_xlabel('Income Level')
        ax.set_ylabel(metric)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distribution_analysis.png'))
    plt.close()

    # 4. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    correlation_matrix = df[['SUR', 'MAE', 'UPI', 'SDI']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation between SDI Metrics')
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()

    # Save summary statistics
    summary = df.groupby('Income_Level')[['SUR', 'MAE', 'UPI', 'SDI']].describe()
    summary.to_csv(os.path.join(output_dir, 'sdi_summary_statistics.csv'))

def create_combined_sdi_plot(df, output_dir):
    """
    Create visualization for the overall combined SDI score with its weighted components
    """
    plt.figure(figsize=(15, 10))

    # 1. Stacked Bar Chart showing component contributions
    plt.subplot(2, 1, 1)
    
    # Calculate weighted components
    weights = {'SUR': 0.4, 'MAE': 0.3, 'UPI': 0.3}
    weighted_components = pd.DataFrame()
    for component in ['SUR', 'MAE', 'UPI']:
        weighted_components[f'Weighted_{component}'] = df[component] * weights[component]
    
    # Create stacked bar chart
    bar_width = 0.8
    income_levels = ['low', 'middle', 'high']
    x = np.arange(len(income_levels))
    
    bottom = np.zeros(len(income_levels))
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    for idx, component in enumerate(['SUR', 'MAE', 'UPI']):
        means = [weighted_components[f'Weighted_{component}'][df['Income_Level'] == level].mean() 
                for level in income_levels]
        plt.bar(x, means, bar_width, bottom=bottom, label=f'{component} (w={weights[component]})',
               color=colors[idx], alpha=0.7)
        bottom += means
    
    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.3, label='Maximum Possible Score')
    plt.xticks(x, [level.capitalize() for level in income_levels])
    plt.ylabel('Weighted Score Contribution')
    plt.title('Component Contributions to SDI by Income Level')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 2. Combined SDI Score Distribution
    plt.subplot(2, 1, 2)
    
    # Calculate overall SDI for each subsidy combination
    for income in income_levels:
        income_data = df[df['Income_Level'] == income]
        sns.kdeplot(data=income_data['SDI'], label=f'{income.capitalize()} Income',
                   fill=True, alpha=0.3)
    
    plt.axvline(x=df['SDI'].mean(), color='black', linestyle='--', 
                label=f'Overall Mean SDI: {df["SDI"].mean():.3f}')
    
    # Add annotations for mean SDI by income level
    y_pos = 0.1
    for income in income_levels:
        mean_sdi = df[df['Income_Level'] == income]['SDI'].mean()
        plt.text(0.02, y_pos, f'{income.capitalize()} Mean SDI: {mean_sdi:.3f}',
                transform=plt.gca().transAxes)
        y_pos += 0.1

    plt.xlabel('SDI Score')
    plt.ylabel('Density')
    plt.title('Distribution of Combined SDI Scores by Income Level')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_sdi_analysis.png'))
    plt.close()

    # 3. Create a summary table of statistics
    summary_stats = pd.DataFrame({
        'Income_Level': income_levels,
        'Mean_SDI': [df[df['Income_Level'] == level]['SDI'].mean() for level in income_levels],
        'Max_SDI': [df[df['Income_Level'] == level]['SDI'].max() for level in income_levels],
        'Min_SDI': [df[df['Income_Level'] == level]['SDI'].min() for level in income_levels],
        'Std_SDI': [df[df['Income_Level'] == level]['SDI'].std() for level in income_levels]
    })
    
    summary_stats.to_csv(os.path.join(output_dir, 'sdi_summary_stats.csv'))
def find_optimal_subsidy(x, y):
    """
    Find the optimal subsidy percentage that maximizes SDI score
    using polynomial fitting and calculus
    """
    # Fit quadratic polynomial
    z = np.polyfit(x, y, 2)
    p = np.poly1d(z)
    
    # Find critical point (maximum) by taking derivative
    # For quadratic ax^2 + bx + c, maximum occurs at x = -b/(2a)
    a, b, c = z
    if a == 0:  # Linear case
        return x[np.argmax(y)]
    
    critical_point = -b / (2 * a)
    
    # Check if critical point is within the data range
    x_min, x_max = min(x), max(x)
    if critical_point < x_min:
        return x_min
    elif critical_point > x_max:
        return x_max
    
    return critical_point

def analyze_optimal_subsidies(df):
    """
    Analyze and print optimal subsidy percentages for each mode and income level
    """
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
                y = mode_data['SDI'].values
                
                optimal_subsidy = find_optimal_subsidy(x, y)
                
                # Evaluate SDI score at optimal point
                z = np.polyfit(x, y, 2)
                p = np.poly1d(z)
                optimal_sdi = p(optimal_subsidy)
                
                optimal_subsidies[income][mode] = {
                    'optimal_percentage': optimal_subsidy,
                    'predicted_sdi': optimal_sdi
                }
    
    return optimal_subsidies


def create_sdi_score_VS_subsidy_percentage(df, output_dir):
    # Create one figure with three subplots (one per income level)
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    income_levels = ['low', 'middle', 'high']
    
    # Define mode mappings with colors and labels
    subsidy_mapping = {
        'MaaS_Bundle': ('green', 'MaaS Subsidy', 'MaaS_Subsidy'),
        'car': ('coral', 'Car Subsidy', 'Car_Subsidy'),
        'bike': ('blue', 'Bike Subsidy', 'Bike_Subsidy')
    }

    # Find optimal subsidies first
    optimal_subsidies = analyze_optimal_subsidies(df)

    # Create plots for each income level
    for i, income in enumerate(income_levels):
        ax = axes[i]
        
        # Plot data for each mode on the same subplot
        for mode, (color, label, column) in subsidy_mapping.items():
            # Filter data for current mode and income level
            mode_data = df[
                (df['Income_Level'] == income) & 
                (df['varied_mode'] == mode)
            ]

            if not mode_data.empty:
                # Create scatter plot
                ax.scatter(mode_data[column], mode_data['SDI'], 
                         color=color, alpha=0.6, label=label)
                
                # Add polynomial trend line
                if len(mode_data) > 2:
                    x = mode_data[column].values
                    y = mode_data['SDI'].values
                    
                    # Calculate the overall x range for consistent line lengths
                    x_min = min(x)
                    x_max = max(x)
                    
                    # Fit quadratic polynomial
                    z = np.polyfit(x, y, 2)
                    p = np.poly1d(z)
                    
                    # Create smooth curve with more points
                    x_smooth = np.linspace(x_min, x_max, 100)
                    ax.plot(x_smooth, p(x_smooth), '--', color=color, alpha=0.7,
                           linewidth=2)

                    # Plot optimal point if available
                    if income in optimal_subsidies and mode in optimal_subsidies[income]:
                        opt = optimal_subsidies[income][mode]
                        ax.plot(opt['optimal_percentage'], opt['predicted_sdi'], 
                               'o', color=color, 
                               markersize=10, markerfacecolor='none',
                               markeredgewidth=2)
                        
                        # Add annotation for optimal point
                        ax.annotate(f'Optimal: {opt["optimal_percentage"]:.3f}',
                                  xy=(opt['optimal_percentage'], opt['predicted_sdi']),
                                  xytext=(10, 10), textcoords='offset points',
                                  bbox=dict(facecolor='white', edgecolor=color, alpha=0.7),
                                  arrowprops=dict(arrowstyle='->'))

        ax.set_title(f'{income.capitalize()} Income')
        ax.set_ylabel('SDI Score')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Set consistent y-axis limits for each subplot
        ax.set_ylim(min(df['SDI']) - 0.05, max(df['SDI']) + 0.05)

    # Set common x label
    plt.xlabel('Subsidy Percentage')
    
    # Improve spacing between subplots
    plt.tight_layout()
    
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
            print(f"  Predicted SDI: {opt['predicted_sdi']:.3f}")
    
    # Save the figure with high DPI for better quality
    plt.savefig(os.path.join(output_dir, 'Income_sdi_vs_subsidies_combined.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return optimal_subsidies


def create_pbs_visualizations(results, output_dir):
    """Create comprehensive visualizations for PBS analysis"""
    # Convert results to DataFrame
    df_rows = []
    for r in results:
        if isinstance(r, dict) and 'subsidy_percentages' in r:
            percentages = r['subsidy_percentages']
            varied_mode = r.get('varied_mode', None)  # Get varied_mode from results
            # Debug: Print subsidy percentages structure
            print("Subsidy percentages structure:", percentages)
            
            for income_level in ['low', 'middle', 'high']:
                if income_level in r:
                    metrics = r[income_level]
                    df_rows.append({
                        'Income_Level': income_level,
                        'Bike_Subsidy': percentages[income_level]['bike'],
                        'Car_Subsidy': percentages[income_level]['car'],
                        'MaaS_Subsidy': percentages[income_level]['MaaS_Bundle'],
                        'SUR': metrics['sur'],
                        'MAE': metrics['mae'],
                        'UPI': metrics['upi'],
                        'SDI': metrics['sdi'],
                        'varied_mode': varied_mode  # Add varied_mode to the DataFrame
                    })
    
    df = pd.DataFrame(df_rows)
    # Debug: Print final DataFrame structure
    print("DataFrame head:\n", df.head())
    print("\nDataFrame columns:", df.columns.tolist())


    # 1. SDI vs Subsidy Percentages Plot
    create_combined_sdi_plot(df, output_dir)
    create_sdi_score_VS_subsidy_percentage(df, output_dir)

    # 2. Component Analysis Plot
    create_component_analysis_plot(df, output_dir)

    # 3. Subsidy Distribution Heatmap
    create_subsidy_heatmap(df, output_dir)

    # 4. Income Level Comparison
    create_income_comparison_plot(df, output_dir)

    # 5. Correlation Analysis
    create_correlation_analysis(df, output_dir)

    return df.describe()

def create_component_analysis_plot(df, output_dir):
    """Create plot showing how SUR, MAE, and UPI contribute to SDI"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))  # 3x3 grid
    components = ['SUR', 'MAE', 'UPI']
    income_levels = ['low', 'middle', 'high']
    
    # Map mode names to column names
    subsidy_columns = {
        'MaaS_Bundle': 'MaaS_Subsidy',
        'car': 'Car_Subsidy',
        'bike': 'Bike_Subsidy'
    }
    
    subsidy_types = ['MaaS_Bundle', 'car', 'bike']  # Order for rows
    
    for row, subsidy_type in enumerate(subsidy_types):
        for col, income in enumerate(income_levels):
            ax = axes[row, col]
            
            # Filter data for current subsidy type and income level
            income_data = df[
                (df['Income_Level'] == income) & 
                (df['varied_mode'] == subsidy_type)
            ]
            
            # Use the correct column name from the mapping
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
            
            # Set titles and labels
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

def create_subsidy_heatmap(df, output_dir):
    """Create heatmap showing relationship between different subsidy types"""
    plt.figure(figsize=(12, 4))
    
    for idx, income in enumerate(['low', 'middle', 'high']):
        plt.subplot(1, 3, idx+1)
        income_data = df[df['Income_Level'] == income]
        
        # Only create heatmap if we have enough data points
        if len(income_data) >= 9:  # Need at least 9 points for 3x3 heatmap
            try:
                pivot = income_data.pivot_table(
                    values='SDI',
                    index=pd.qcut(income_data['Car_Subsidy'], 3, labels=['Low', 'Medium', 'High'], duplicates='drop'),
                    columns=pd.qcut(income_data['MaaS_Subsidy'], 3, labels=['Low', 'Medium', 'High'], duplicates='drop'),
                    aggfunc='mean'
                )
                
                sns.heatmap(pivot, annot=True, cmap='YlOrRd', fmt='.3f')
                plt.title(f'{income.capitalize()} Income Subsidy Interaction')
                plt.xlabel('MaaS Subsidy Level')
                plt.ylabel('Car Subsidy Level')
            except ValueError as e:
                print(f"Warning: Could not create heatmap for {income} income: {e}")
                plt.text(0.5, 0.5, 'Insufficient unique values\nfor heatmap', 
                        ha='center', va='center')
                plt.title(f'{income.capitalize()} Income')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'subsidy_heatmap.png'))
    plt.close()

def create_income_comparison_plot(df, output_dir):
    """Create violin plots comparing SDI distributions across income levels"""
    plt.figure(figsize=(10, 6))
    
    sns.violinplot(data=df, x='Income_Level', y='SDI')
    plt.title('SDI Distribution by Income Level')
    plt.grid(True, alpha=0.3)
    
    # Add mean SDI values
    for idx, income in enumerate(['low', 'middle', 'high']):
        mean_sdi = df[df['Income_Level'] == income]['SDI'].mean()
        plt.text(idx, df['SDI'].min(), f'Mean: {mean_sdi:.3f}', 
                ha='center', va='bottom')
    
    plt.savefig(os.path.join(output_dir, 'income_comparison.png'))
    plt.close()

def create_correlation_analysis(df, output_dir):
    """Create correlation matrix for subsidy types and metrics"""
    plt.figure(figsize=(10, 8))
    
    correlation_columns = ['Bike_Subsidy', 'Car_Subsidy', 'MaaS_Subsidy', 
                          'SUR', 'MAE', 'UPI', 'SDI']
    correlation_matrix = df[correlation_columns].corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                center=0, vmin=-1, vmax=1, fmt='.2f')
    plt.title('Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_analysis.png'))
    plt.close()

def create_visualizations(results, analysis_type, output_dir='sdi_analysis_plots'):
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


# 8 Jan, edit this parameter set whenever necessary
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
        points_per_mode = num_simulations
        
        # Define ranges for each mode and income level
        subsidy_ranges = {
            'low': {
                'bike': (0.2, 0.4),
                'car': (0.15, 0.25),
                'MaaS_Bundle': (0.3, 0.6)
            },
            'middle': {
                'bike': (0.15, 0.35),
                'car': (0.10, 0.20),
                'MaaS_Bundle': (0.25, 0.5)
            },
            'high': {
                'bike': (0.10, 0.30),
                'car': (0.05, 0.15),
                'MaaS_Bundle': (0.20, 0.4)
            }
        }

        # Generate evenly spaced points for each mode
        for i in range(points_per_mode):
            # Calculate percentage through range for each point
            pct = i / (points_per_mode - 1) if points_per_mode > 1 else 0
            mode_to_vary = modes[i % len(modes)]  # Cycle through modes
            
            # Create a base subsidy configuration with default values
            subsidy_config = {
                'low': {
                    'bike': 0.3,      # Default values
                    'car': 0.2,
                    'MaaS_Bundle': 0.4
                },
                'middle': {
                    'bike': 0.25,
                    'car': 0.15,
                    'MaaS_Bundle': 0.35
                },
                'high': {
                    'bike': 0.2,
                    'car': 0.1,
                    'MaaS_Bundle': 0.3
                }
            }

            # Only vary the subsidies for the current mode
            for income_level in ['low', 'middle', 'high']:
                mode_range = subsidy_ranges[income_level][mode_to_vary]
                subsidy_config[income_level][mode_to_vary] = mode_range[0] + (mode_range[1] - mode_range[0]) * pct

            params = {
                **base_parameters,
                'subsidy_dataset': subsidy_config,
                'subsidy_config': SubsidyPoolConfig('daily', float('inf')),
                '_analysis_type': 'PBS',
                'varied_mode': mode_to_vary
            }
            parameter_sets.append(params)

    print(f"Generated exactly {len(parameter_sets)} parameter combinations")
    return parameter_sets


def run_sdi_analysis(analysis_type, base_parameters, num_simulations, num_cpus):
    """
    Run SDI analysis with specified type and exact number of simulations
    """
    print(f"Starting {analysis_type} analysis with {num_simulations} points...")
    
    # Generate parameter sets based on analysis type
    parameter_sets = generate_parameter_sets(analysis_type, base_parameters, num_simulations)
    print(f"Running exactly {len(parameter_sets)} simulations")
    
    # Run parallel simulations
    with mp.Pool(processes=num_cpus) as pool:
        results = pool.map(run_single_simulation, parameter_sets)
    
    # Save results
    with open(f'sdi_{analysis_type.lower()}_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # Generate visualizations
    summary_stats = create_visualizations(results, analysis_type)
    
    print(f"Completed {len(results)} simulations")
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
        'simulation_steps': SIMULATION_STEPS,
        
    }

    # # Generate parameter sets with varying subsidy pools
    # subsidy_pools = np.linspace(1000, 40000, 15)  # Test 15 different subsidy pool sizes
    # parameter_sets = []
    
    # for pool_size in subsidy_pools:
    #     params = {
    #         **base_parameters,
    #         'subsidy_config': SubsidyPoolConfig('daily', float(pool_size))
    #     }
    #     parameter_sets.append(params)

    # # Run parallel simulations
    # print(f"Starting SDI analysis with {len(parameter_sets)} parameter combinations...")
    # num_cpus = 8  # Adjust based on available CPU cores
    
    # try:
    #     with mp.Pool(processes=num_cpus) as pool:
    #         results = pool.map(run_single_simulation, parameter_sets)
            
    #     print("Simulations completed successfully")
        
    #     # Save raw results
    #     with open('sdi_analysis_results.pkl', 'wb') as f:
    #         pickle.dump(results, f)
    #     print("Results saved to sdi_analysis_results.pkl")
        
    #     # Generate visualizations
    #     print("Generating visualizations...")
    #     create_visualizations(results)
    #     print("Visualizations saved to sdi_analysis_plots directory")
        
    #     # Print summary statistics
    #     print("\nSummary of SDI Analysis:")
    #     df_summary = pd.DataFrame([
    #         {
    #             'Income_Level': income_level,
    #             'Avg_SDI': np.mean([r[income_level]['sdi'] for r in results]),
    #             'Max_SDI': np.max([r[income_level]['sdi'] for r in results]),
    #             'Min_SDI': np.min([r[income_level]['sdi'] for r in results])
    #         }
    #         for income_level in ['low', 'middle', 'high']
    #     ])
    #     print("\nSDI Statistics by Income Level:")
    #     print(df_summary)
        
    #     # Find optimal subsidy pool size
    #     optimal_results = max(results, key=lambda x: np.mean([x[level]['sdi'] for level in ['low', 'middle', 'high']]))
    #     print(f"\nOptimal subsidy pool size: {optimal_results['subsidy_pool']}")
    #     print("Optimal SDI scores:")
    #     for level in ['low', 'middle', 'high']:
    #         print(f"{level.capitalize()} income: {optimal_results[level]['sdi']:.3f}")
            
    # except Exception as e:
    #     print(f"Error during analysis: {str(e)}")
    #     import traceback
    #     traceback.print_exc()

    # Run analysis for both types
    # fps_results, fps_stats = run_sdi_analysis('FPS', base_parameters, 35, 8) #number of simulations, number of CPUS
    pbs_results, pbs_stats = run_sdi_analysis('PBS', base_parameters, 15, 7)
    
    # Compare results
    print("\nComparison of FPS and PBS approaches:")
    print("\nFPS Summary:")
    # print(fps_stats)
    print("\nPBS Summary:")
    print(pbs_stats)