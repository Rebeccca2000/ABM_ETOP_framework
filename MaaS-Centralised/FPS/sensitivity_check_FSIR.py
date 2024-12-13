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

SIMULATION_STEPS = 144
num_commuters = 120

def calculate_co2_reduction(session):
    """Calculate CO2 reduction from mode shifts"""
    emission_factors = {
        1: 0.041,  # Train (kg CO2/km)
        2: 0.089,  # Bus
        3: 0.171,  # Car 
        4: 0,      # Bike
        5: 0       # Walk
    }
    
    def calculate_distance(route_details):
        """Calculate distance for a simple coordinate route"""
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

    def process_public_transport_route(route_details):
        """Process public transport route segments"""
        mode_distances = {}
        
        for segment in route_details:
            if segment[0] in ["to station", "transfer", "to destination"]:
                # Walking segments
                if len(segment) > 1 and isinstance(segment[1], list):
                    if isinstance(segment[1][0], list):  # Contains coordinates
                        distance = calculate_distance(segment[1][0])
                    else:  # Contains start and end points
                        distance = calculate_distance([segment[1][0], segment[1][1]])
                    mode_distances['walk'] = mode_distances.get('walk', 0) + distance
                    
            elif segment[0] == "train":
                # Train segments
                stations = segment[2]
                distance = len(stations) * 5  # Standard distance between stations
                mode_distances['train'] = mode_distances.get('train', 0) + distance
                
            elif segment[0] == "bus":
                # Bus segments
                stops = segment[2]
                distance = len(stops) * 5  # Standard distance between stops
                mode_distances['bus'] = mode_distances.get('bus', 0) + distance
                
        return mode_distances

    def process_maas_bundle(route_details, to_station, to_destination):
        """Process MaaS bundle route segments"""
        mode_distances = {}
        
        # Process to_station segment
        if to_station:
            mode = to_station[0]  # First element is mode type
            route = to_station[3]  # Fourth element contains route
            distance = calculate_distance(route)
            mode_distances[mode] = mode_distances.get(mode, 0) + distance
        
        # Process main route
        for segment in route_details:
            if segment[0] in ["train", "bus"]:
                mode = segment[0]
                stations = segment[2]
                distance = len(stations) * 5
                mode_distances[mode] = mode_distances.get(mode, 0) + distance
            elif segment[0] in ["to station", "transfer", "to destination"]:
                if len(segment) > 1 and isinstance(segment[1], list):
                    if isinstance(segment[1][0], list):
                        distance = calculate_distance(segment[1][0])
                    else:
                        distance = calculate_distance([segment[1][0], segment[1][1]])
                    mode_distances['walk'] = mode_distances.get('walk', 0) + distance
        
        # Process to_destination segment
        if to_destination:
            mode = to_destination[0]
            route = to_destination[3]
            distance = calculate_distance(route)
            mode_distances[mode] = mode_distances.get(mode, 0) + distance
            
        return mode_distances

    total_co2_saved = 0
    total_subsidy = 0
    
    try:
        query = session.query(
            ServiceBookingLog.request_id,
            ServiceBookingLog.record_company_name,
            ServiceBookingLog.route_details,
            ServiceBookingLog.to_station,
            ServiceBookingLog.to_destination,
            ServiceBookingLog.government_subsidy
        ).all()

        for trip in query:
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
                    # Direct routes (bike, car, walk)
                    distance = calculate_distance(trip.route_details)
                    mode_distances[trip.record_company_name] = distance
                
                # Calculate CO2 savings
                baseline_co2 = sum(dist * emission_factors[3] for dist in mode_distances.values())
                actual_co2 = sum(dist * emission_factors[get_mode_id(mode)] for mode, dist in mode_distances.items())
                total_co2_saved += (baseline_co2 - actual_co2)
                
                # Add subsidy
                if trip.government_subsidy:
                    try:
                        if isinstance(trip.government_subsidy, (int, float)):
                            total_subsidy += trip.government_subsidy
                        else:
                            total_subsidy += float(eval(str(trip.government_subsidy)))
                    except:
                        pass
                            
            except Exception as e:
                print(f"Error processing trip {trip.request_id}: {e}")
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
        'public': 1,  # Default to train if just "public"
        
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
    
    return mode_map.get(mode_name, 3)  # Default to car (3) if unknown mode
def calculate_pt_utilization(session):
    """Calculate public transport utilization rate including both direct PT and MaaS bundle trips"""
    
    try:
        # First count total trips
        total_trips = session.query(
            func.count(ServiceBookingLog.request_id)
        ).scalar() or 0

        # Count PT trips
        pt_trips = session.query(
            func.count(ServiceBookingLog.request_id)
        ).filter(
            (ServiceBookingLog.record_company_name == 'public') |
            (ServiceBookingLog.record_company_name == 'MaaS_Bundle')
        ).scalar() or 0

        return pt_trips / total_trips if total_trips > 0 else 0

    except Exception as e:
        print(f"Error calculating PT utilization: {e}")
        return 0

def calculate_vkt_reduction(session):
    """Calculate Vehicle Kilometers Traveled reduction"""
    baseline_query = session.query(
        func.sum(ShareServiceBookingLog.duration)
    ).filter(
        ShareServiceBookingLog.mode_id == 3
    ).scalar() or 0
    
    current_query = session.query(
        func.sum(ShareServiceBookingLog.duration)
    ).filter(
        ShareServiceBookingLog.mode_id != 3
    ).scalar() or 0
    
    return baseline_query - current_query

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
    pt_utilization = calculate_pt_utilization(session)
    vkt_reduction = calculate_vkt_reduction(session)
    time_value = calculate_time_value(session)
    
    # Normalize components
    normalized_co2 = co2_reduction / total_subsidy if total_subsidy > 0 else 0
    normalized_vkt = vkt_reduction / 1000  # Assuming 1000km as baseline
    normalized_time = time_value / 10000  # Assuming $10000 as baseline
    
    fsir = (
        weights['alpha'] * normalized_co2 +
        weights['beta'] * pt_utilization +
        weights['gamma'] * normalized_vkt +
        weights['delta'] * normalized_time
    )
    
    return fsir, {
        'co2_reduction': co2_reduction,
        'pt_utilization': pt_utilization,
        'vkt_reduction': vkt_reduction,
        'time_value': time_value
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

def create_fsir_visualizations(results, output_dir='fsir_analysis_plots'):
    """Create visualizations for FSIR analysis"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df = pd.DataFrame([{
        'FSIR': r['fsir'],
        'Subsidy': r['subsidy_config'],
        'CO2_Reduction': r['components']['co2_reduction'],
        'PT_Utilization': r['components']['pt_utilization'],
        'VKT_Reduction': r['components']['vkt_reduction'],
        'Time_Value': r['components']['time_value']
    } for r in results])
    
    # FSIR vs Subsidy Plot
    plt.figure(figsize=(12, 6))
    plt.scatter(df['Subsidy'], df['FSIR'], alpha=0.6)
    plt.xlabel('Subsidy Pool Size')
    plt.ylabel('FSIR Score')
    plt.title('FSIR Score vs Subsidy Pool Size')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'fsir_vs_subsidy.png'))
    plt.close()
    
    # Component Contributions
    components = ['CO2_Reduction', 'PT_Utilization', 'VKT_Reduction', 'Time_Value']
    plt.figure(figsize=(12, 6))
    for component in components:
        plt.plot(df['Subsidy'], df[component], label=component)
    plt.xlabel('Subsidy Pool Size')
    plt.ylabel('Component Value')
    plt.title('FSIR Component Contributions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'component_contributions.png'))
    plt.close()
    
    # Efficiency Metrics
    plt.figure(figsize=(12, 6))
    plt.scatter(df['CO2_Reduction'] / df['Subsidy'], 
               df['PT_Utilization'], 
               c=df['FSIR'], 
               cmap='viridis')
    plt.colorbar(label='FSIR Score')
    plt.xlabel('CO2 Reduction per Subsidy Dollar')
    plt.ylabel('Public Transport Utilization Rate')
    plt.title('Efficiency Metrics')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'efficiency_metrics.png'))
    plt.close()
    
    # Save summary statistics
    summary_stats = df.describe()
    summary_stats.to_csv(os.path.join(output_dir, 'fsir_summary.csv'))

if __name__ == "__main__":
    # Define simulation parameters
    subsidy_pools = np.linspace(1000, 40000, 25)
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
    create_fsir_visualizations(results)