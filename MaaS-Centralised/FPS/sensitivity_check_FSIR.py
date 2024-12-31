from sqlalchemy import create_engine, func, case, and_
from sqlalchemy.orm import sessionmaker
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from run_visualisation_03 import MobilityModel
from agent_service_provider_initialisation_03 import reset_database, CommuterInfoLog, ServiceBookingLog, ShareServiceBookingLog
import multiprocessing as mp
import os
from agent_subsidy_pool import SubsidyPoolConfig
from typing import Dict, List, Tuple
from pandas.plotting import parallel_coordinates

SIMULATION_STEPS = 50
num_commuters = 160

def normalize_with_bounds(value: float, min_val: float, max_val: float) -> float:
    """Normalize values with proper handling of edge cases"""
    if max_val == min_val:
        return 0
    return (value - min_val) / (max_val - min_val)

def calculate_co2_reduction(session):
    """Calculate CO2 reduction with improved error handling and detailed logging"""
    emission_factors = {
        'train': 0.041,
        'bus': 0.089,
        'car': 0.171,
        'bike': 0,
        'walk': 0,
        'MaaS_Bundle': 0
    }
    
    occupancy_rates = {
        'train': 0.7,
        'bus': 0.6,
        'car': 1.2,
        'bike': 1.0,
        'walk': 1.0
    }

    def safely_calculate_distance(route_details):
        """Safely calculate distance with comprehensive error handling"""
        try:
            if not route_details:
                return 0
                
            if isinstance(route_details, str):
                try:
                    route_details = eval(route_details)
                except:
                    return 0
                    
            if not isinstance(route_details, list):
                return 0
                
            total_distance = 0
            
            # Handle nested route structures
            if route_details and isinstance(route_details[0], (list, tuple)):
                for i in range(len(route_details)-1):
                    try:
                        x1, y1 = route_details[i]
                        x2, y2 = route_details[i+1]
                        total_distance += ((x2-x1)**2 + (y2-y1)**2)**0.5
                    except (IndexError, TypeError):
                        continue
            
            return total_distance
            
        except Exception as e:
            return 0

    def process_public_transport_route(route_details):
        """Process public transport route with improved error handling"""
        try:
            mode_distances = {}
            
            if not isinstance(route_details, list):
                return mode_distances
                
            for segment in route_details:
                try:
                    if not isinstance(segment, (list, tuple)) or not segment:
                        continue
                        
                    segment_type = segment[0]
                    
                    if segment_type in ["to station", "transfer", "to destination"]:
                        if len(segment) > 1 and isinstance(segment[1], list):
                            if isinstance(segment[1][0], list):
                                distance = safely_calculate_distance(segment[1][0])
                            else:
                                distance = safely_calculate_distance([segment[1][0], segment[1][1]])
                            mode_distances['walk'] = mode_distances.get('walk', 0) + distance
                            
                    elif segment_type in ["train", "bus"]:
                        if len(segment) > 2:
                            stops = segment[2]
                            distance = len(stops) * 5
                            mode_distances[segment_type] = mode_distances.get(segment_type, 0) + distance
                            
                except Exception as e:
                    continue
                    
            return mode_distances
            
        except Exception as e:
            return {}

    def process_maas_bundle(route_details, to_station, to_destination):
        """Process MaaS bundle with improved error handling"""
        try:
            mode_distances = {}
            
            # Process to_station segment
            if to_station:
                try:
                    mode = to_station[0]
                    route = to_station[3]
                    distance = safely_calculate_distance(route)
                    mode_distances[mode] = mode_distances.get(mode, 0) + distance
                except (IndexError, TypeError):
                    pass
            
            # Process main route
            if isinstance(route_details, list):
                for segment in route_details:
                    try:
                        if not isinstance(segment, (list, tuple)) or not segment:
                            continue
                            
                        segment_type = segment[0]
                        
                        if segment_type in ["train", "bus"]:
                            if len(segment) > 2:
                                stops = segment[2]
                                distance = len(stops) * 5
                                mode_distances[segment_type] = mode_distances.get(segment_type, 0) + distance
                                
                        elif segment_type in ["to station", "transfer", "to destination"]:
                            if len(segment) > 1 and isinstance(segment[1], list):
                                if isinstance(segment[1][0], list):
                                    distance = safely_calculate_distance(segment[1][0])
                                else:
                                    distance = safely_calculate_distance([segment[1][0], segment[1][1]])
                                mode_distances['walk'] = mode_distances.get('walk', 0) + distance
                                
                    except Exception as e:
                        continue
            
            # Process to_destination segment
            if to_destination:
                try:
                    mode = to_destination[0]
                    route = to_destination[3]
                    distance = safely_calculate_distance(route)
                    mode_distances[mode] = mode_distances.get(mode, 0) + distance
                except (IndexError, TypeError):
                    pass
                    
            return mode_distances
            
        except Exception as e:
            return {}

    total_co2_saved = 0
    total_subsidy = 0
    
    try:
        trips = session.query(
            ServiceBookingLog.request_id,
            ServiceBookingLog.record_company_name,
            ServiceBookingLog.route_details,
            ServiceBookingLog.to_station,
            ServiceBookingLog.to_destination,
            ServiceBookingLog.government_subsidy
        ).all()

        for trip in trips:
            try:
                mode_distances = {}
                
                if trip.record_company_name == 'MaaS_Bundle':
                    mode_distances = process_maas_bundle(
                        trip.route_details,
                        trip.to_station,
                        trip.to_destination
                    )
                elif trip.record_company_name == 'public':
                    mode_distances = process_public_transport_route(trip.route_details)
                else:
                    distance = safely_calculate_distance(trip.route_details)
                    mode_distances[trip.record_company_name] = distance

                # Calculate CO2 with occupancy rates
                baseline_co2 = sum(
                    (dist * emission_factors['car']) / occupancy_rates['car']
                    for dist in mode_distances.values()
                )
                
                actual_co2 = sum(
                    (dist * emission_factors.get(mode, 0)) / occupancy_rates.get(mode, 1.0)
                    for mode, dist in mode_distances.items()
                )
                
                total_co2_saved += (baseline_co2 - actual_co2)

                # Process subsidy
                if trip.government_subsidy:
                    try:
                        subsidy = float(trip.government_subsidy) if isinstance(trip.government_subsidy, (int, float)) else \
                                 float(eval(str(trip.government_subsidy))) if isinstance(trip.government_subsidy, str) else 0
                        total_subsidy += subsidy
                    except:
                        pass
                        
            except Exception as e:
                # Instead of printing error, silently continue
                continue
                
    except Exception as e:
        print(f"Database query error: {e}")
        return 0, 0
        
    return total_co2_saved, total_subsidy
def get_mode_id(mode_name):
    """Convert mode/company name to mode_id based on TransportModes table.
    mode_id mappings:
    1: train
    2: bus  
    3: car (UberLike services)
    4: bike (BikeShare services)
    5: walk
    """
    mode_map = {
        # Public transport modes
        'train': 1,
        'bus': 2,
        
        # Car services
        'UberLike1': 3,
        'UberLike2': 3,
        
        # Bike services  
        'BikeShare1': 4,
        'BikeShare2': 4,
        
        # Walking
        'walk': 5,
        'to station': 5,
        'to destination': 5,
        'transfer': 5
    }
    
    return mode_map.get(mode_name, 0)  # Default to car (3) if unknown mode
def calculate_maas_utilization(session):
    """Calculate MaaS bundle utilization rate with peak/off-peak differentiation"""
    try:
        def is_peak_period(time_step):
            time_step = time_step % 144  # 24 hours * 6 steps per hour
            return (36 <= time_step < 60) or (90 <= time_step < 114)

        # Query all trips with their time periods
        query = session.query(
            ServiceBookingLog.record_company_name,
            ServiceBookingLog.start_time
        ).all()

        peak_maas_trips = 0
        peak_total_trips = 0
        offpeak_maas_trips = 0
        offpeak_total_trips = 0

        for record_name, start_time in query:
            is_peak = is_peak_period(start_time)
            is_maas = record_name == 'MaaS_Bundle'
            
            if is_peak:
                peak_total_trips += 1
                if is_maas:
                    peak_maas_trips += 1
            else:
                offpeak_total_trips += 1
                if is_maas:
                    offpeak_maas_trips += 1

        # Calculate weighted utilization
        peak_utilization = peak_maas_trips / peak_total_trips if peak_total_trips > 0 else 0
        offpeak_utilization = offpeak_maas_trips / offpeak_total_trips if offpeak_total_trips > 0 else 0
        
        # Weight peak periods more heavily (60% peak, 40% off-peak)
        weighted_utilization = (peak_utilization * 0.6 + offpeak_utilization * 0.4)
        
        return weighted_utilization

    except Exception as e:
        print(f"Error calculating MaaS utilization: {e}")
        return 0
def calculate_vkt_reduction_enhanced(session):
    """Enhanced VKT reduction calculation using ShareServiceBookingLog data"""
    try:
        def calculate_distance(route_details):
            """Calculate distance from route coordinates"""
            if not route_details or not isinstance(route_details, list):
                return 0
                
            total_distance = 0
            try:
                for i in range(len(route_details)-1):
                    x1, y1 = route_details[i]
                    x2, y2 = route_details[i+1]
                    total_distance += ((x2-x1)**2 + (y2-y1)**2)**0.5
            except:
                return 0
            return total_distance

        # Get car trips
        car_trips = session.query(ShareServiceBookingLog).filter(
            ShareServiceBookingLog.mode_id == 3  # Car mode
        ).all()
        
        print(f"Number of car trips found: {len(car_trips)}")
        
        # Calculate total car distance
        car_vkt = 0
        for trip in car_trips:
            if hasattr(trip, 'route_details') and trip.route_details:
                route = trip.route_details
                if isinstance(route, str):
                    try:
                        route = eval(route)  # Handle JSON string if needed
                    except:
                        continue
                car_vkt += calculate_distance(route)
        
        print(f"Total car VKT: {car_vkt}")
        
        # Get all trips
        all_trips = session.query(ShareServiceBookingLog).all()
        print(f"Total number of trips: {len(all_trips)}")
        
        # Calculate total distance across all modes
        total_distance = 0
        for trip in all_trips:
            if hasattr(trip, 'route_details') and trip.route_details:
                route = trip.route_details
                if isinstance(route, str):
                    try:
                        route = eval(route)  # Handle JSON string if needed
                    except:
                        continue
                total_distance += calculate_distance(route)
                
        print(f"Total distance across all modes: {total_distance}")
        
        # Calculate reduction as percentage
        if total_distance > 0:
            reduction = (total_distance - car_vkt) / total_distance
            print(f"VKT reduction: {reduction * 100:.2f}%")
        else:
            reduction = 0
            print("No distance traveled")
            
        return reduction

    except Exception as e:
        import traceback
        print(f"Error calculating VKT reduction: {str(e)}")
        print("Full error traceback:")
        print(traceback.format_exc())
        return 0

def calculate_time_value(session):
    """Calculate economic value of time saved"""
    value_of_time = {
        'low': 9.64,
        'middle': 23.7,
        'high': 67.2
    }
    
    query = session.query(
        CommuterInfoLog.income_level,
        ServiceBookingLog.total_time
    ).join(
        ServiceBookingLog,
        CommuterInfoLog.commuter_id == ServiceBookingLog.commuter_id
    ).all()
    
    total_value = sum(value_of_time[income] * time for income, time in query)
    return total_value

def calculate_fsir(session, weights):
    """Calculate FSIR with component weights"""
    co2_reduction, total_subsidy = calculate_co2_reduction(session)
    maas_utilization = calculate_maas_utilization(session) 
    vkt_reduction = calculate_vkt_reduction_enhanced(session)
    time_value = calculate_time_value(session)
    
    # Store min/max values for normalization
    component_ranges = {
        'co2': {'min': 0, 'max': max(co2_reduction, 1)},
        'maas': {'min': 0, 'max': 1},
        'vkt': {'min': -1, 'max': 1},
        'time': {'min': 0, 'max': max(time_value, 1)}
    }
    
    # Normalize components
    normalized = {
        'co2': normalize_with_bounds(co2_reduction/total_subsidy if total_subsidy > 0 else 0,
                                   component_ranges['co2']['min'],
                                   component_ranges['co2']['max']),
        'maas': maas_utilization,  
        'vkt': normalize_with_bounds(vkt_reduction,
                                   component_ranges['vkt']['min'],
                                   component_ranges['vkt']['max']),
        'time': normalize_with_bounds(np.log1p(time_value),
                                    component_ranges['time']['min'],
                                    component_ranges['time']['max'])
    }
    
    # Calculate interaction term
    interaction = normalized['maas'] * normalized['vkt']
    
    # Calculate final FSIR
    fsir = (weights['alpha'] * normalized['co2'] +
            weights['beta'] * normalized['maas'] +
            weights['gamma'] * normalized['vkt'] +
            weights['delta'] * np.log1p(normalized['time']) +
            weights.get('eta', 0.1) * interaction)  # Add small weight for interaction
            
    return fsir, {
        'co2_reduction': co2_reduction,
        'maas_utilization': maas_utilization,
        'vkt_reduction': vkt_reduction,
        'time_value': time_value,
        'normalized': normalized,
        'interaction': interaction
    }

def run_single_simulation(params):
    """Run a single simulation with FSIR calculation"""
    print(f"Starting simulation with PID {os.getpid()}")
    db_path = f"service_provider_database_{os.getpid()}.db"
    db_connection_string = f"sqlite:///{db_path}"
    engine = create_engine(db_connection_string)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
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
        weights = params.pop('weights')
        
        model = MobilityModel(db_connection_string=db_connection_string, **params)
        model.run_model(simulation_steps)
        
        fsir, components = calculate_fsir(session, weights)
        
        return {
            'fsir': fsir,
            'components': components,
            'subsidy_config': params['subsidy_config'].total_amount,
            'weights': weights
        }
        
    except Exception as e:
        print(f"Error in simulation {os.getpid()}: {str(e)}")
        raise
        
    finally:
        session.close()
        if os.path.exists(db_path):
            os.remove(db_path)

def create_enhanced_fsir_visualizations(results, output_dir='fsir_analysis_plots'):
    """Create comprehensive visualizations for FSIR analysis with comprehensive error handling"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create DataFrame from results with error handling
    df = pd.DataFrame([{
        'FSIR': r['fsir'],
        'Subsidy': r['subsidy_config'],
        'CO2_Reduction': r['components']['co2_reduction'],
        'MaaS_Utilization': r['components']['maas_utilization'],
        'VKT_Reduction': r['components']['vkt_reduction'],
        'Time_Value': r['components']['time_value']
    } for r in results])
    
    # Handle potential zero/invalid values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    # Safely calculate CO2 per subsidy
    df['CO2_per_subsidy'] = np.where(df['Subsidy'] > 0, 
                                    df['CO2_Reduction'] / df['Subsidy'], 
                                    0)
    
    components = ['CO2_Reduction', 'MaaS_Utilization', 'VKT_Reduction', 'Time_Value']
    
    try:
        # 1. Component Contributions Plot
        plt.figure(figsize=(15, 8))
        
        # Create subsidy groups safely
        unique_subsidies = df['Subsidy'].nunique()
        n_groups = min(max(2, unique_subsidies), 10)
        
        if unique_subsidies > 1:
            subsidy_groups = pd.qcut(df['Subsidy'], 
                                   q=n_groups, 
                                   labels=[f'G{i}' for i in range(n_groups)],
                                   duplicates='drop')
        else:
            # Handle case with single unique value
            subsidy_groups = pd.Series(['G0'] * len(df), index=df.index)
        
        # Normalize components with error handling
        for comp in components:
            range_val = df[comp].max() - df[comp].min()
            df[f'{comp}_norm'] = np.where(range_val > 0,
                                        (df[comp] - df[comp].min()) / range_val,
                                        df[comp])
        
        # Create stacked bar chart
        grouped_data = df.groupby(subsidy_groups)[[f'{comp}_norm' for comp in components]].mean()
        bottom = np.zeros(len(grouped_data))
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f1c40f']
        
        for i, comp in enumerate(components):
            plt.bar(grouped_data.index, 
                   grouped_data[f'{comp}_norm'], 
                   bottom=bottom, 
                   label=comp, 
                   color=colors[i])
            bottom += grouped_data[f'{comp}_norm']
        
        plt.title('FSIR Component Contributions by Subsidy Group')
        plt.xlabel('Subsidy Groups (Low to High)')
        plt.ylabel('Normalized Component Contribution')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'component_contributions.png'))
        plt.close()
        
        # 2. Scatter Plot with Trend Line
        plt.figure(figsize=(12, 8))
        
        # Clean data for regression
        mask = (df['CO2_per_subsidy'].notna() & 
                df['MaaS_Utilization'].notna() & 
                ~np.isinf(df['CO2_per_subsidy']) & 
                ~np.isinf(df['MaaS_Utilization']))
        
        X = df.loc[mask, 'CO2_per_subsidy'].values
        y = df.loc[mask, 'MaaS_Utilization'].values
        
        if len(X) > 1:  # Need at least 2 points for regression
            # Use robust regression
            from sklearn.linear_model import TheilSenRegressor
            reg = TheilSenRegressor(random_state=42)
            X_reshaped = X.reshape(-1, 1)
            reg.fit(X_reshaped, y)
            
            # Create prediction line
            X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
            y_plot = reg.predict(X_plot)
            
            # Plot scatter and regression line
            scatter = plt.scatter(X, y, c=df.loc[mask, 'FSIR'], cmap='viridis', alpha=0.6)
            plt.plot(X_plot, y_plot, 'r--', alpha=0.8)
        else:
            # Fallback to simple scatter plot if regression isn't possible
            scatter = plt.scatter(df['CO2_per_subsidy'], 
                                df['MaaS_Utilization'],
                                c=df['FSIR'],
                                cmap='viridis',
                                alpha=0.6)
        
        plt.colorbar(scatter, label='FSIR Score')
        plt.title('Efficiency Metrics Relationship')
        plt.xlabel('CO₂ Reduction per Subsidy Dollar')
        plt.ylabel('MaaS Utilization Rate')
        plt.savefig(os.path.join(output_dir, 'efficiency_metrics.png'))
        plt.close()
        
        # Save summary statistics
        summary = df.describe()
        summary.to_csv(os.path.join(output_dir, 'fsir_summary_statistics.csv'))
        
    except Exception as e:
        print(f"Error in visualization generation: {e}")
        import traceback
        traceback.print_exc()

def run_fsir_analysis(parameter_sets):
    """Run FSIR analysis with improved error handling"""
    results = []
    
    for params in parameter_sets:
        try:
            result = run_single_simulation(params)
            if result is not None:  # Check if simulation returned valid results
                results.append(result)
        except Exception as e:
            print(f"Error in simulation with parameters {params}: {e}")
            continue
            
    return results
if __name__ == "__main__":
    # Define simulation parameters
    subsidy_pools = np.linspace(1000, 40000, 20)
    weights_sets = [
        {'alpha': 0.4, 'beta': 0.2, 'gamma': 0.2, 'delta': 0.2},  # Environment focused
        {'alpha': 0.25, 'beta': 0.25, 'gamma': 0.25, 'delta': 0.25},  # Balanced
        {'alpha': 0.2, 'beta': 0.4, 'gamma': 0.2, 'delta': 0.2}   # PT utilization focused
    ]

    # In the base_parameters:
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
        'ASC_VALUES': {'car': 0, 'bike': 0, 'public': 0, 'walk': 0, 'maas': 2000, 'default': 0},
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

    # Generate parameter combinations
    parameter_sets = [
        {**base_parameters, 
         'subsidy_config': SubsidyPoolConfig('daily', float(pool_size)),
         'weights': weights}
        for pool_size in subsidy_pools
        for weights in weights_sets
    ]

    print(f"Running {len(parameter_sets)} parameter combinations...")

    # Run parallel simulations
    num_cpus = 4
    with mp.Pool(processes=num_cpus) as pool:
        results = pool.map(run_single_simulation, parameter_sets)

    # Save results
    with open('fsir_analysis_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    # Generate visualizations
    create_enhanced_fsir_visualizations(results)