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
import json
from matplotlib.gridspec import GridSpec
import seaborn as sns
import numpy as np

SIMULATION_STEPS = 15
num_commuters = 120

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
        print("Total trips found:", len(query))
        print("Unique record_company_names:", set(record[0] for record in query))
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
        print(f"Peak MaaS trips: {peak_maas_trips}")
        print(f"Peak total trips: {peak_total_trips}")
        print(f"Offpeak MaaS trips: {offpeak_maas_trips}")
        print(f"Offpeak total trips: {offpeak_total_trips}")


        # Calculate weighted utilization
        peak_utilization = peak_maas_trips / peak_total_trips if peak_total_trips > 0 else 0
        offpeak_utilization = offpeak_maas_trips / offpeak_total_trips if offpeak_total_trips > 0 else 0
        print(f"Peak utilization: {peak_utilization}")
        print(f"Offpeak utilization: {offpeak_utilization}")
        # Weight peak periods more heavily (60% peak, 40% off-peak)
        weighted_utilization = (peak_utilization * 0.6 + offpeak_utilization * 0.4)
        print(f"Final weighted utilization: {weighted_utilization}")
        return weighted_utilization

    except Exception as e:
        print(f"Error calculating MaaS utilization: {e}")
        return 0
def calculate_vkt_reduction_enhanced(session):
    """Enhanced VKT reduction calculation including car segments in MaaS trips"""
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

        # Calculate car VKT from both direct car trips and MaaS car segments
        car_vkt = 0
        
        # Get direct car trips from ShareServiceBookingLog
        car_trips = session.query(ShareServiceBookingLog).filter(
            ShareServiceBookingLog.mode_id == 3  # Car mode
        ).all()
        print(f"Number of direct car trips found: {len(car_trips)}")
        
        # Calculate VKT for direct car trips
        for trip in car_trips:
            if hasattr(trip, 'route_details') and trip.route_details:
                route = trip.route_details
                if isinstance(route, str):
                    try:
                        route = eval(route)  # Handle JSON string if needed
                    except:
                        continue
                car_vkt += calculate_distance(route)
        
        # Get MaaS trips and calculate car segments
        maas_trips = session.query(ServiceBookingLog).filter(
            ServiceBookingLog.record_company_name == 'MaaS_Bundle'
        ).all()
        print(f"Number of MaaS trips found: {len(maas_trips)}")
        
        # Calculate VKT from car segments in MaaS trips
        maas_car_vkt = 0
        for trip in maas_trips:
            # Check to_station segment
            if trip.to_station:
                try:
                    to_station_info = trip.to_station
                    if isinstance(to_station_info, str):
                        to_station_info = eval(to_station_info)
                    if to_station_info and 'UberLike' in to_station_info[0]:
                        route = to_station_info[3]
                        maas_car_vkt += calculate_distance(route)
                except:
                    continue
                    
            # Check to_destination segment
            if trip.to_destination:
                try:
                    to_dest_info = trip.to_destination
                    if isinstance(to_dest_info, str):
                        to_dest_info = eval(to_dest_info)
                    if to_dest_info and 'UberLike' in to_dest_info[0]:
                        route = to_dest_info[3]
                        maas_car_vkt += calculate_distance(route)
                except:
                    continue

        # Total car VKT combines direct car trips and MaaS car segments
        total_car_vkt = car_vkt + maas_car_vkt
        print(f"Total car VKT (direct + MaaS): {total_car_vkt}")
        
        # Calculate total distance across all modes
        total_distance = 0
        
        # Get all regular trips
        all_trips = session.query(ShareServiceBookingLog).all()
        print(f"Total number of regular trips: {len(all_trips)}")
        
        # Calculate total distance from regular trips
        for trip in all_trips:
            if hasattr(trip, 'route_details') and trip.route_details:
                route = trip.route_details
                if isinstance(route, str):
                    try:
                        route = eval(route)
                    except:
                        continue
                total_distance += calculate_distance(route)
        
        # Add distances from MaaS trips
        for trip in maas_trips:
            # Process main route
            if trip.route_details:
                try:
                    route_details = trip.route_details
                    if isinstance(route_details, str):
                        route_details = eval(route_details)
                    for segment in route_details:
                        if isinstance(segment, (list, tuple)) and len(segment) > 1:
                            # Handle different segment types in MaaS routes
                            if segment[0] in ['to station', 'to destination', 'transfer']:
                                if len(segment) > 1 and isinstance(segment[1], list):
                                    total_distance += calculate_distance(segment[1])
                except:
                    continue

            # Add distances from to_station and to_destination segments
            for segment_info in [trip.to_station, trip.to_destination]:
                if segment_info:
                    try:
                        if isinstance(segment_info, str):
                            segment_info = eval(segment_info)
                        if segment_info and len(segment_info) > 3:
                            route = segment_info[3]
                            total_distance += calculate_distance(route)
                    except:
                        continue
                
        print(f"Total distance across all modes: {total_distance}")
        
        # Calculate reduction as percentage
        if total_distance > 0:
            reduction = (total_distance - total_car_vkt) / total_distance
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
    print(f"Debug - MaaS utilization after calculation: {maas_utilization}") 
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
    print(f"Debug - Normalized components: {normalized}")
    # Calculate interaction term
    interaction = normalized['maas'] * normalized['vkt']
    
    # Calculate final FSIR
    fsir = (weights['alpha'] * normalized['co2'] +
            weights['beta'] * normalized['maas'] +
            weights['gamma'] * normalized['vkt'] +
            weights['delta'] * np.log1p(normalized['time']) +
            weights.get('eta', 0.1) * interaction)  # Add small weight for interaction
    components_dict = {
        'co2_reduction': co2_reduction,
        'maas_utilization': maas_utilization,  # This should be 1.0
        'vkt_reduction': vkt_reduction,
        'time_value': time_value,
        'normalized': normalized,
        'interaction': normalized['maas'] * normalized['vkt']
    }
    print(f"Debug - Final components dict: {components_dict}")  # Add this
    return fsir, components_dict

def run_single_simulation(params):
    """Run a single simulation with FSIR calculation"""
    print(f"Starting simulation with PID {os.getpid()}")
    db_path = f"service_provider_database_{os.getpid()}.db"
    db_connection_string = f"sqlite:///{db_path}"
    
    try:
        engine = create_engine(db_connection_string)
        Session = sessionmaker(bind=engine)
        session = Session()

        # Extract and remove analysis parameters before passing to MobilityModel
        analysis_type = params.pop('_analysis_type', None)
        varied_mode = params.pop('varied_mode', None)
        simulation_steps = params.pop('simulation_steps')
        weights = params.pop('weights')
        
        reset_db_params = {k: params[k] for k in [
            'uber_like1_capacity', 'uber_like1_price',
            'uber_like2_capacity', 'uber_like2_price',
            'bike_share1_capacity', 'bike_share1_price',
            'bike_share2_capacity', 'bike_share2_price'
        ]}
        
        reset_database(engine=engine, session=session, **reset_db_params)
        
        model = MobilityModel(db_connection_string=db_connection_string, **params)
        model.run_model(simulation_steps)
        
        fsir, components = calculate_fsir(session, weights)
        
        # Store analysis parameters in result
        result = {
            'fsir': fsir,
            'components': components,
            'subsidy_config': params['subsidy_config'].total_amount,
            'weights': weights
        }
        
        if analysis_type == 'PBS':
            result['subsidy_percentages'] = params['subsidy_dataset']
            result['varied_mode'] = varied_mode
            
        return result

    except Exception as e:
        print(f"Error in simulation {os.getpid()}: {str(e)}")
        raise
        
    finally:
        session.close()
        if os.path.exists(db_path):
            os.remove(db_path)

def create_fps_visualizations(results, output_dir):
    """Create comprehensive visualizations for FSIR analysis with comprehensive error handling"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("\nDebug - Raw results data:")
    for idx, r in enumerate(results):
        print(f"\nResult {idx}:")
        print(f"FSIR: {r['fsir']}")
        print(f"Subsidy: {r['subsidy_config']}")
        print(f"MaaS utilization: {r['components']['maas_utilization']}")
    # Create DataFrame from results with error handling
    df = pd.DataFrame([{
        'FSIR': r['fsir'],
        'Subsidy': r['subsidy_config'],
        'CO2_Reduction': r['components']['co2_reduction'],
        'MaaS_Utilization': r['components']['maas_utilization'],
        'VKT_Reduction': r['components']['vkt_reduction'],
        'Time_Value': r['components']['time_value']
    } for r in results])
    print("\nDebug - Created DataFrame:")
    print(df[['Subsidy', 'MaaS_Utilization']])
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
        
        #Print the plots separately 02/Jan/25
        def fit_curve(x, y, degree=3):
            """Fit a polynomial curve to the data"""
                
            coeffs = np.polyfit(x, y, degree)
            return np.poly1d(coeffs)

        def create_separate_component_plots(df, output_dir='fsir_component_plots'):
            components = ['CO2_Reduction', 'MaaS_Utilization', 'VKT_Reduction', 'Time_Value']
            
            for component in components:
                x = df['Subsidy']
                y = df[component]
                
                plt.figure(figsize=(10, 6))
                plt.scatter(x, y, alpha=0.7, label=f'{component} Data')
                
                # Fit curve
                try:
                    trend = fit_curve(x, y)
                    x_fit = np.linspace(min(x), max(x), 500)
                    y_fit = trend(x_fit)
                    plt.plot(x_fit, y_fit, 'r-', label='Trend Line')
                    
                    # Set y-axis limits specifically for MaaS utilization
                    if component == 'MaaS_Utilization':
                        plt.ylim(0, 1.2)  # Set y-axis from 0 to 1.2 to show full range
                        
                except Exception as e:
                    print(f"Error fitting curve for {component}: {e}")
                
                plt.title(f'{component} Trends Across Subsidy Levels')
                plt.xlabel('Subsidy Pool Size')
                plt.ylabel(component)
                plt.grid(True)  # Add grid for better readability
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{component}_trend.png"))
                plt.close()
        create_separate_component_plots(df)
                    
    except Exception as e:
        print(f"Error in visualization generation: {e}")
        import traceback
        traceback.print_exc()

def normalize_component_values(df):
    """Normalize component values to appropriate scales"""
    # CO2 Reduction per subsidy dollar
    if 'subsidy_config' in df.columns:
        df['CO2_Reduction_per_subsidy'] = df['CO2_Reduction'] / df['subsidy_config']
    
    # Ensure MaaS utilization is between 0 and 1
    df['MaaS_Utilization'] = df['MaaS_Utilization'].clip(0, 1)
    
    # Normalize VKT reduction to percentage
    df['VKT_Reduction'] = df['VKT_Reduction'] * 100
    
    # Scale time value based on your specific needs
    if 'Time_Value' in df.columns:
        df['Time_Value'] = df['Time_Value'] / df['Time_Value'].max()
    
    return df


def run_fsir_analysis(analysis_type, base_parameters, num_simulations, num_cpus):
   """Run FSIR analysis with enhanced error handling and logging"""
   print(f"Starting {analysis_type} analysis with {num_simulations} simulations...")
   
   try:
       # Generate parameter sets
       parameter_sets = generate_parameter_sets(analysis_type, base_parameters, num_simulations)
       print(f"Generated {len(parameter_sets)} parameter combinations")

       # Run parallel simulations
       with mp.Pool(processes=num_cpus) as pool:
           results = pool.map(run_single_simulation, parameter_sets)
           
       valid_results = [r for r in results if r is not None]
       print(f"Successfully completed {len(valid_results)}/{len(parameter_sets)} simulations")

       # Save raw results
       output_file = f'fsir_{analysis_type.lower()}_results.pkl'
       with open(output_file, 'wb') as f:
           pickle.dump(valid_results, f)
       print(f"Results saved to {output_file}")

       # Generate visualizations
       output_dir = os.path.join('fsir_analysis_plots', analysis_type.lower())
       summary_stats = create_visualizations(valid_results, analysis_type, output_dir)
       
       # Calculate and print key metrics
       if valid_results:
           avg_fsir = np.mean([r['fsir'] for r in valid_results])
           max_fsir = np.max([r['fsir'] for r in valid_results])
           
           print("\nKey Analysis Metrics:")
           print(f"Average FSIR Score: {avg_fsir:.3f}")
           print(f"Maximum FSIR Score: {max_fsir:.3f}")
           
           if analysis_type == 'FPS':
               optimal_pool = valid_results[np.argmax([r['fsir'] for r in valid_results])]['subsidy_config']
               print(f"Optimal Subsidy Pool Size: {optimal_pool:.0f}")
           else:
               best_result = valid_results[np.argmax([r['fsir'] for r in valid_results])]
               print(f"Best Performing Mode: {best_result.get('varied_mode', 'N/A')}")
               print("Optimal Subsidy Percentages:")
               print(json.dumps(best_result['subsidy_percentages'], indent=2))

       return valid_results, summary_stats

   except Exception as e:
       print(f"Error in FSIR analysis: {str(e)}")
       import traceback
       traceback.print_exc()
       return [], None
   
def create_temporal_impact_analysis(df, output_dir):
    """Create temporal analysis of FSIR impacts"""
    plt.figure(figsize=(20, 16))
    
    # Debug print
    print("\nDataFrame for temporal analysis:")
    print(df.head())
    print("\nColumns:", df.columns.tolist())
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. Peak vs Off-peak Analysis (Top left)
    ax1 = axes[0, 0]
    modes = df['Varied_Mode'].unique()
    
    for idx, mode in enumerate(modes):
        mode_data = df[df['Varied_Mode'] == mode]
        
        # Calculate mean FSIR for peak and off-peak
        peak_fsir = mode_data[mode_data['MaaS_Utilization'] > mode_data['MaaS_Utilization'].median()]['FSIR']
        off_peak_fsir = mode_data[mode_data['MaaS_Utilization'] <= mode_data['MaaS_Utilization'].median()]['FSIR']
        
        positions = [idx*2, idx*2 + 1]
        bplot = ax1.boxplot([peak_fsir, off_peak_fsir],
                          positions=positions,
                          labels=[f'{mode}\npeak', f'{mode}\noff-peak'],
                          patch_artist=True)
        
        # Color coding for different modes
        colors = {'bike': 'lightblue', 'car': 'lightgreen', 'MaaS_Bundle': 'lightpink'}
        for patch in bplot['boxes']:
            patch.set_facecolor(colors.get(mode, 'lightgray'))
    
    ax1.set_title('Peak vs Off-peak FSIR Distribution')
    ax1.set_ylabel('FSIR Score')
    
    # 2. Temporal Efficiency (Top right)
    ax2 = axes[0, 1]
    for mode in modes:
        mode_data = df[df['Varied_Mode'] == mode]
        ax2.scatter(mode_data['MaaS_Utilization'],
                   mode_data['CO2_Reduction'],
                   label=mode,
                   alpha=0.7)
    ax2.set_title('MaaS Utilization vs CO2 Reduction')
    ax2.set_xlabel('MaaS Utilization Rate')
    ax2.set_ylabel('CO2 Reduction')
    ax2.legend()
    
    # 3. Value Generation (Bottom left)
    ax3 = axes[1, 0]
    for mode in modes:
        mode_data = df[df['Varied_Mode'] == mode]
        ax3.scatter(mode_data['MaaS_Utilization'],
                   mode_data['Time_Value'],
                   label=mode,
                   alpha=0.7)
    ax3.set_title('Value Generation Over Time')
    ax3.set_xlabel('MaaS Utilization Rate')
    ax3.set_ylabel('Time Value')
    ax3.legend()
    
    # 4. Mode Effectiveness (Bottom right)
    ax4 = axes[1, 1]
    effectiveness = df.groupby('Varied_Mode').agg({
        'FSIR': 'mean',
        'CO2_Reduction': 'sum',
        'Time_Value': 'mean'
    }).reset_index()
    
    x = np.arange(len(modes))
    width = 0.25
    
    ax4.bar(x - width, effectiveness['FSIR'], width, label='FSIR')
    ax4.bar(x, effectiveness['CO2_Reduction']/effectiveness['CO2_Reduction'].max(), width, label='CO2 (Normalized)')
    ax4.bar(x + width, effectiveness['Time_Value']/effectiveness['Time_Value'].max(), width, label='Time Value (Normalized)')
    
    ax4.set_xticks(x)
    ax4.set_xticklabels(modes)
    ax4.set_title('Mode Effectiveness Comparison')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temporal_impact_analysis.png'))
    plt.close()

def create_pbs_visualizations(results, output_dir):
    """Create comprehensive visualizations for PBS FSIR analysis"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convert results to DataFrame with error handling
    df_rows = []
    for r in results:
        try:
            if isinstance(r, dict) and 'subsidy_percentages' in r:
                percentages = r['subsidy_percentages']
                varied_mode = r.get('varied_mode', None)
                components = r.get('components', {})
                
                row = {
                    'FSIR': r.get('fsir', 0),
                    'Varied_Mode': varied_mode,
                    'CO2_Reduction': components.get('co2_reduction', 0),
                    'CO2_per_Subsidy': components.get('co2_per_subsidy', 0),
                    'MaaS_Utilization': components.get('maas_utilization', 0),
                    'VKT_Reduction': components.get('vkt_reduction', 0),
                    'Time_Value': components.get('time_value', 0)
                }
                
                # Add subsidy percentages for each mode and income level
                modes = {
                    'Bike': 'bike',
                    'Car': 'car',
                    'MaaS': 'MaaS_Bundle'
                }
                
                for mode_key, mode_value in modes.items():
                    for income in ['low', 'middle', 'high']:
                        try:
                            row[f'{mode_key}_Subsidy_{income}'] = percentages[income][mode_value]
                        except KeyError:
                            print(f"Warning: Missing subsidy data for {mode_key} - {income}")
                            row[f'{mode_key}_Subsidy_{income}'] = 0
                
                df_rows.append(row)
                
        except Exception as e:
            print(f"Error processing result: {str(e)}")
            continue
    
    if not df_rows:
        print("Warning: No valid data to create visualizations")
        return None
    
    try:
        df = pd.DataFrame(df_rows)
        
        # Create all visualizations with error handling
        visualization_functions = [
            ('Mode Specific Analysis', create_mode_specific_analysis),
            ('Component Contribution Analysis', create_component_contribution_analysis),
            ('Efficiency Frontier Analysis', create_efficiency_frontier_analysis),
            ('Temporal Impact Analysis', create_temporal_impact_analysis),
            ('Comparative Subsidy Analysis', create_comparative_subsidy_analysis),
            ('Component Specific Visualizations', create_component_specific_visualizations),
            ('FSIR Component Analysis', lambda df, output_dir: create_fsir_component_analysis(normalize_component_values(df), output_dir))
        ]
        
        for viz_name, viz_func in visualization_functions:
            try:
                print(f"Creating {viz_name}...")
                viz_func(df, output_dir)
            except Exception as e:
                print(f"Error creating {viz_name}: {str(e)}")
                continue
        
        return df.describe()
        
    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")
        return None

def create_mode_specific_analysis(df, output_dir):
    """Create separate analysis files for each mode across income levels"""
    modes = ['bike', 'car', 'MaaS_Bundle']
    income_levels = ['low', 'middle', 'high']
    
    for mode in modes:
        fig, axes = plt.subplots(3, 1, figsize=(15, 20))
        mode_data = df[df['Varied_Mode'] == mode].copy()
        
        print(f"\nProcessing mode: {mode}")
        print(f"Number of data points: {len(mode_data)}")
        
        for idx, income in enumerate(income_levels):
            ax = axes[idx]
            
            # Handle the MaaS_Bundle case specially
            subsidy_col = f"MaaS_Subsidy_{income}" if mode == 'MaaS_Bundle' else f"{mode.capitalize()}_Subsidy_{income}"
            
            print(f"Plotting {mode} for {income} income level")
            print(f"Subsidy values: {mode_data[subsidy_col].values}")
            
            if subsidy_col in mode_data.columns:
                scatter = ax.scatter(mode_data[subsidy_col], 
                                  mode_data['FSIR'],
                                  c=mode_data['CO2_Reduction'],
                                  cmap='viridis',
                                  alpha=0.7)
                
                # Add trend line
                if len(mode_data) > 1:
                    z = np.polyfit(mode_data[subsidy_col], mode_data['FSIR'], 2)
                    p = np.poly1d(z)
                    x_smooth = np.linspace(mode_data[subsidy_col].min(), 
                                         mode_data[subsidy_col].max(), 100)
                    ax.plot(x_smooth, p(x_smooth), '--', color='red', alpha=0.7)
                
                plt.colorbar(scatter, ax=ax, label='CO₂ Reduction')
                ax.set_title(f'{income.capitalize()} Income')
                ax.set_xlabel('Subsidy Percentage')
                ax.set_ylabel('FSIR Score')
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 0.4)
                ax.legend(['FSIR Score', 'Trend'], loc='upper right')
            
        plt.suptitle(f'{mode} Subsidy Impact on FSIR', y=1.02, fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'mode_specific_analysis_{mode}.png'))
        plt.close()

def create_component_specific_visualizations(df, output_dir):
    """Create visualizations for each FSIR component across income levels"""
    components = {
        'CO2_Reduction': 'CO₂ Reduction',
        'MaaS_Utilization': 'MaaS Utilization Rate',
        'VKT_Reduction': 'VKT Reduction',
        'Time_Value': 'Time Value'
    }
    
    # Debug print
    print("\nCreating component visualizations for:")
    print(df['Varied_Mode'].unique())
    
    for component, title in components.items():
        fig, axes = plt.subplots(3, 1, figsize=(15, 20))
        
        for idx, mode in enumerate(df['Varied_Mode'].unique()):
            mode_data = df[df['Varied_Mode'] == mode]
            
            # Print debug info
            print(f"\nPlotting {component} for {mode}")
            print(f"Data points: {len(mode_data)}")
            
            if not mode_data.empty:
                axes[idx].scatter(mode_data['MaaS_Utilization'],
                                mode_data[component],
                                label=mode,
                                alpha=0.7)
                
                # Add trend line
                if len(mode_data) > 1:
                    z = np.polyfit(mode_data['MaaS_Utilization'],
                                 mode_data[component], 2)
                    p = np.poly1d(z)
                    x_smooth = np.linspace(mode_data['MaaS_Utilization'].min(),
                                         mode_data['MaaS_Utilization'].max(), 100)
                    axes[idx].plot(x_smooth, p(x_smooth), '--',
                                 label=f'{mode} trend')
            
            axes[idx].set_title(f'{mode} - {title}')
            axes[idx].set_xlabel('MaaS Utilization Rate')
            axes[idx].set_ylabel(title)
            axes[idx].grid(True, alpha=0.3)
            axes[idx].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{component.lower()}_analysis.png'))
        plt.close()

def create_component_contribution_analysis(df, output_dir):
    """Analyze how different components contribute to FSIR"""
    components = ['CO2_Reduction', 'MaaS_Utilization', 'VKT_Reduction', 'Time_Value']
    
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Component correlations
    ax1 = fig.add_subplot(gs[0, 0])
    correlation_matrix = df[components + ['FSIR']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax1)
    ax1.set_title('Component Correlations')
    
    # Stacked contribution
    ax2 = fig.add_subplot(gs[0, 1])
    normalized_components = df[components].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    normalized_components.plot(kind='area', stacked=True, ax=ax2)
    ax2.set_title('Normalized Component Contributions')
    
    # Component distributions
    ax3 = fig.add_subplot(gs[1, :])
    data_melted = pd.melt(normalized_components)
    sns.violinplot(data=data_melted, x='variable', y='value', ax=ax3)
    ax3.set_title('Component Distribution Analysis')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'component_contributions.png'))
    plt.close()

def create_efficiency_frontier_analysis(df, output_dir):
    """Create efficiency frontier analysis plots"""
    # Debug prints
    print("\nDataFrame for efficiency frontier analysis:")
    print("CO2_Reduction values:", df['CO2_Reduction'].values)
    print("Subsidy values:", df['subsidy_config'].values)
    
    # Calculate CO2 per subsidy with proper normalization
    total_subsidy = df['subsidy_config'].astype(float)
    df['CO2_per_Subsidy'] = np.where(total_subsidy > 0,
                                    df['CO2_Reduction'] / total_subsidy,
                                    0)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Add jitter to prevent vertical alignment
    jitter = np.random.normal(0, 0.0001, size=len(df))
    x_values = df['CO2_per_Subsidy'] + jitter
    
    # CO2 Efficiency vs MaaS Utilization
    scatter1 = ax1.scatter(x_values, 
                          df['MaaS_Utilization'],
                          c=df['FSIR'],
                          cmap='viridis',
                          alpha=0.7)
    
    # Add trend line
    if len(df) > 1:
        z = np.polyfit(x_values, df['MaaS_Utilization'], 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(min(x_values), max(x_values), 100)
        ax1.plot(x_smooth, p(x_smooth), '--', color='red', alpha=0.5)
    
    ax1.set_xlabel('CO₂ Reduction per Subsidy')
    ax1.set_ylabel('MaaS Utilization Rate')
    plt.colorbar(scatter1, ax=ax1, label='FSIR Score')
    
    # VKT Reduction vs Time Value with jitter
    jitter2 = np.random.normal(0, 0.0001, size=len(df))
    scatter2 = ax2.scatter(df['VKT_Reduction'] + jitter2, 
                          df['Time_Value'],
                          c=df['FSIR'],
                          cmap='viridis',
                          alpha=0.7)
    
    # Add trend line for second plot
    if len(df) > 1:
        z2 = np.polyfit(df['VKT_Reduction'] + jitter2, df['Time_Value'], 2)
        p2 = np.poly1d(z2)
        x_smooth2 = np.linspace(min(df['VKT_Reduction']), max(df['VKT_Reduction']), 100)
        ax2.plot(x_smooth2, p2(x_smooth2), '--', color='red', alpha=0.5)
    
    ax2.set_xlabel('VKT Reduction')
    ax2.set_ylabel('Economic Value of Time Saved')
    plt.colorbar(scatter2, ax=ax2, label='FSIR Score')
    
    # Add titles
    ax1.set_title('CO₂ Efficiency vs MaaS Utilization')
    ax2.set_title('VKT Reduction vs Economic Value')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'efficiency_frontier.png'))
    plt.close()

def create_comparative_subsidy_analysis(df, output_dir):
    """Create comparative analysis of subsidy strategies separated by income level"""
    fig, axes = plt.subplots(3, 1, figsize=(15, 20))
    income_levels = ['low', 'middle', 'high']
    modes = ['bike', 'car', 'MaaS_Bundle']  # All three modes
    
    # Define colors and display names for modes
    mode_config = {
        'bike': {'color': '#3498db', 'label': 'Bike'},  # Blue
        'car': {'color': '#e67e22', 'label': 'Car'},    # Orange
        'MaaS_Bundle': {'color': '#2ecc71', 'label': 'MaaS'}  # Green
    }
    
    for idx, income in enumerate(income_levels):
        ax = axes[idx]
        
        for mode in modes:
            # Filter data for current mode
            mode_data = df[df['Varied_Mode'] == mode].copy()
            
            # Set up correct column name based on mode
            if mode == 'MaaS_Bundle':
                col = f'MaaS_Subsidy_{income}'
            else:
                col = f'{mode.capitalize()}_Subsidy_{income}'
                
            if col in mode_data.columns:
                # Create scatter plot
                scatter = ax.scatter(mode_data[col], 
                                   mode_data['FSIR'],
                                   color=mode_config[mode]['color'],
                                   alpha=0.6,
                                   label=mode_config[mode]['label'])
                
                # Add trend line if we have enough points
                if len(mode_data) > 1:
                    z = np.polyfit(mode_data[col], mode_data['FSIR'], 2)
                    p = np.poly1d(z)
                    x_smooth = np.linspace(mode_data[col].min(), mode_data[col].max(), 100)
                    ax.plot(x_smooth, p(x_smooth), '--', 
                           color=mode_config[mode]['color'],
                           alpha=0.5)
        
        # Set labels and title
        ax.set_xlabel('Subsidy Percentage')
        ax.set_ylabel('FSIR Score')
        ax.set_title(f'{income.capitalize()} Income Subsidy Impact')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set y-axis limits for better visualization
        ax.set_ylim(0.21, 0.26)
        
        # Set x-axis limits to cover all subsidy ranges
        ax.set_xlim(0.0, 0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparative_subsidy_analysis.png'))
    plt.close()

def create_fsir_component_analysis(df, output_dir):
    """Create comparative analysis plots for each FSIR component"""
    component_dir = os.path.join(output_dir, 'component_analysis')
    os.makedirs(component_dir, exist_ok=True)  # Corrected line

    components = {
        'CO2_Reduction': {
            'title': 'CO₂ Reduction per Subsidy Dollar',
            'ylabel': 'CO₂ Reduction Rate'
        },
        'MaaS_Utilization': {
            'title': 'MaaS Utilization Rate',
            'ylabel': 'Utilization Rate'
        },
        'VKT_Reduction': {
            'title': 'Vehicle Kilometers Traveled Reduction',
            'ylabel': 'VKT Reduction Rate'
        },
        'Time_Value': {
            'title': 'Economic Value of Time Saved',
            'ylabel': 'Time Value (Currency Units)'
        }
    }
    
    income_levels = ['low', 'middle', 'high']
    modes = ['bike', 'car', 'MaaS_Bundle']
    
    mode_config = {
        'bike': {'color': '#3498db', 'label': 'Bike'},
        'car': {'color': '#e67e22', 'label': 'Car'},
        'MaaS_Bundle': {'color': '#2ecc71', 'label': 'MaaS'}
    }

    for component_name, component_config in components.items():
        print(f"\nDebug - Creating plot for {component_name}")
        print(f"Raw data for {component_name}:")
        print(df[component_name].describe())
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 20))
        
        for idx, income in enumerate(income_levels):
            ax = axes[idx]
            print(f"\nAnalyzing {income} income level:")
            
            for mode in modes:
                mode_data = df[df['Varied_Mode'] == mode].copy()
                print(f"\n{mode} mode data points:")
                
                if mode == 'MaaS_Bundle':
                    col = f'MaaS_Subsidy_{income}'
                else:
                    col = f'{mode.capitalize()}_Subsidy_{income}'
                
                if col in mode_data.columns and component_name in mode_data.columns:
                    # Debug print for each mode's data
                    print(f"{mode} - Subsidy values ({col}):")
                    print(mode_data[col].values)
                    print(f"{mode} - {component_name} values:")
                    print(mode_data[component_name].values)
                    
                    scatter = ax.scatter(
                        mode_data[col],
                        mode_data[component_name],
                        color=mode_config[mode]['color'],
                        alpha=0.6,
                        label=mode_config[mode]['label']
                    )
                    
                    if len(mode_data) > 1:
                        z = np.polyfit(mode_data[col], mode_data[component_name], 2)
                        p = np.poly1d(z)
                        x_smooth = np.linspace(mode_data[col].min(), mode_data[col].max(), 100)
                        ax.plot(x_smooth, p(x_smooth), '--', 
                               color=mode_config[mode]['color'],
                               alpha=0.5)
                else:
                    print(f"Missing data for {mode} - Column {col} or {component_name} not found")
            
            ax.set_xlabel('Subsidy Percentage')
            ax.set_ylabel(component_config['ylabel'])
            ax.set_title(f'{income.capitalize()} Income: {component_config["title"]}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Debug print for axis limits that matplotlib automatically set
            print(f"\nAutomatic axis limits for {income} income:")
            print(f"X-axis limits: {ax.get_xlim()}")
            print(f"Y-axis limits: {ax.get_ylim()}")
        
        plt.suptitle(component_config['title'], y=1.02, fontsize=16)
        plt.tight_layout()
        
        plt.savefig(os.path.join(component_dir, f'{component_name.lower()}_analysis.png'))
        plt.close()

    print(f"Component analysis plots saved in: {component_dir}")

def generate_parameter_sets(analysis_type, base_parameters, num_simulations):
    """
    Generate parameter sets based on analysis type with controlled number of simulations
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
                'weights': {'alpha': 0.3, 'beta': 0.3, 'gamma': 0.2, 'delta': 0.2}  # FSIR-specific weights
            }
            parameter_sets.append(params)
            
    elif analysis_type == 'PBS':
        modes = ['bike', 'car', 'MaaS_Bundle']
        # Calculate points per mode - this is key
        points_per_mode = num_simulations // len(modes)  # With 15 total, this gives 5 points per mode
        print(f"Generating {points_per_mode} points per mode")
        parameter_sets = []
        # Define ranges for each mode and income level
        subsidy_ranges = {
            'low': {
                'bike': (0.2, 0.45),
                'car': (0.15, 0.35),
                'MaaS_Bundle': (0.3, 0.6)
            },
            'middle': {
                'bike': (0.15, 0.35),
                'car': (0.1, 0.25),
                'MaaS_Bundle': (0.25, 0.5)
            },
            'high': {
                'bike': (0.1, 0.3),
                'car': (0.05, 0.15),
                'MaaS_Bundle': (0.2, 0.4)
            }
        }

        for mode in modes:
            # Generate exactly points_per_mode points for each mode
            for i in range(points_per_mode):
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

                # Only vary the current mode's values
                for income in ['low', 'middle', 'high']:
                    for m in modes:
                        if m != mode:
                            min_val = subsidy_ranges[income][m][0]
                            max_val = subsidy_ranges[income][m][1]
                            subsidy_config[income][m] = (min_val + max_val) / 2

                params = {
                    **base_parameters,
                    'subsidy_dataset': subsidy_config,
                    'subsidy_config': SubsidyPoolConfig('daily', float('inf')),
                    '_analysis_type': 'PBS',
                    'varied_mode': mode
                }
                parameter_sets.append(params)

        print(f"Generated total of {len(parameter_sets)} parameter sets")
        # Debug print for verification
        for mode in modes:
            mode_count = sum(1 for p in parameter_sets if p['varied_mode'] == mode)
            print(f"Points for {mode}: {mode_count}")
            
        return parameter_sets

def create_visualizations(results, analysis_type, output_dir):
    """
    Create visualizations based on analysis type
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if analysis_type == 'FPS':
        return create_fps_visualizations(results, output_dir)
    else:  # PBS
        return create_pbs_visualizations(results, output_dir)

if __name__ == "__main__":
    # Define simulation parameters
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
        'ASC_VALUES': {'car': 100, 'bike': 100, 'public': 100, 'walk':100, 'maas': 100, 'default': 0},
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
        'weights': weights_sets[0]
    }

    # # Generate parameter combinations
    # parameter_sets = [
    #     {**base_parameters, 
    #      'subsidy_config': SubsidyPoolConfig('daily', float(pool_size)),
    #      'weights': weights_sets[0]}
    #     for pool_size in subsidy_pools
    # ]

    # print(f"Running {len(parameter_sets)} parameter combinations...")

    # # Run parallel simulations
    # num_cpus = 8
    # with mp.Pool(processes=num_cpus) as pool:
    #     results = pool.map(run_single_simulation, parameter_sets)

    # # Save results
    # with open('fsir_analysis_results.pkl', 'wb') as f:
    #     pickle.dump(results, f)

    # # Generate visualizations
    # create_enhanced_fsir_visualizations(results)
   # Run analysis for both types
    #fps_results, fps_stats = run_fsir_analysis('FPS', base_parameters, 25, 8)
    pbs_results, pbs_stats = run_fsir_analysis('PBS', base_parameters, 15, 8)
    
   # Compare results
    print("\nComparison of FPS and PBS approaches:")

    print("\nFPS Summary:")
   # print(fps_stats)
    print("\nPBS Summary:")
    print(pbs_stats)