from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from scipy.optimize import curve_fit
import statsmodels.api as sm
from run_visualisation_03 import MobilityModel
from agent_service_provider_initialisation_03 import reset_database
from agent_service_provider_initialisation_03 import CommuterInfoLog, ServiceBookingLog
import multiprocessing as mp 

# Global configuration
DB_CONNECTION_STRING = 'sqlite:///service_provider_database_1.db'
SIMULATION_STEPS = 144  # Run for a larger number of steps for better analysis
num_commuters = 120

def calculate_gini_coefficient(values):
    """
    Computes the Gini coefficient from a list of values.
    """
    # Debugging: Print raw values
    print(f"Raw values: {values}")

    # Handle edge case where there are fewer than two values
    if len(values) < 2:
        print("Warning: Not enough data points to compute Gini coefficient.")
        return 0.0

    # Convert values to numpy array for easier calculations
    values = np.array(values)

    # Calculate the mean of the values
    mean_value = np.mean(values)
    if mean_value == 0:
        return 0.0  # Avoid division by zero if all values are zero

    # Compute the absolute differences between all pairs
    diff_sum = np.sum(np.abs(values[:, None] - values[None, :]))

    # Calculate the Gini coefficient
    gini = diff_sum / (2 * len(values)**2 * mean_value)

    print(f"Calculated Gini: {gini}")
    return gini



def run_single_simulation(params):
    engine = create_engine(params['db_connection_string'])
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        reset_database(
            engine=engine,
            session=session,
            uber_like1_capacity=params['uber_like1_capacity'],
            uber_like1_price=params['uber_like1_price'],
            uber_like2_capacity=params['uber_like2_capacity'],
            uber_like2_price=params['uber_like2_price'],
            bike_share1_capacity=params['bike_share1_capacity'],
            bike_share1_price=params['bike_share1_price'],
            bike_share2_capacity=params['bike_share2_capacity'],
            bike_share2_price=params['bike_share2_price']
        )

        model = MobilityModel(
            db_connection_string=params['db_connection_string'],
            num_commuters=params['num_commuters'],
            grid_width=params['grid_width'],
            grid_height=params['grid_height'],
            data_income_weights=params['data_income_weights'],
            data_health_weights=params['data_health_weights'],
            data_payment_weights=params['data_payment_weights'],
            data_age_distribution=params['data_age_distribution'],
            data_disability_weights=params['data_disability_weights'],
            data_tech_access_weights=params['data_tech_access_weights'],
            CHANCE_FOR_INSERTING_RANDOM_TRAFFIC=params['CHANCE_FOR_INSERTING_RANDOM_TRAFFIC'],
            ASC_VALUES=params['ASC_VALUES'],
            UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS=params['UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS'],
            UTILITY_FUNCTION_BASE_COEFFICIENTS=params['UTILITY_FUNCTION_BASE_COEFFICIENTS'],
            PENALTY_COEFFICIENTS=params['PENALTY_COEFFICIENTS'],
            AFFORDABILITY_THRESHOLDS=params['AFFORDABILITY_THRESHOLDS'],
            FLEXIBILITY_ADJUSTMENTS=params['FLEXIBILITY_ADJUSTMENTS'],
            VALUE_OF_TIME=params['VALUE_OF_TIME'],
            public_price_table=params['public_price_table'],
            ALPHA_VALUES=params['ALPHA_VALUES'],
            DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS=params['DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS'],
            BACKGROUND_TRAFFIC_AMOUNT=params['BACKGROUND_TRAFFIC_AMOUNT'],
            CONGESTION_ALPHA=params['CONGESTION_ALPHA'],
            CONGESTION_BETA=params['CONGESTION_BETA'],
            CONGESTION_CAPACITY=params['CONGESTION_CAPACITY'],
            CONGESTION_T_IJ_FREE_FLOW=params['CONGESTION_T_IJ_FREE_FLOW'],
            uber_like1_capacity=params['uber_like1_capacity'],
            uber_like1_price=params['uber_like1_price'],
            uber_like2_capacity=params['uber_like2_capacity'],
            uber_like2_price=params['uber_like2_price'],
            bike_share1_capacity=params['bike_share1_capacity'],
            bike_share1_price=params['bike_share1_price'],
            bike_share2_capacity=params['bike_share2_capacity'],
            bike_share2_price=params['bike_share2_price'],
            subsidy_dataset=params['subsidy_dataset'],
        )
        model.run_model(params['simulation_steps'])

        # Query: Total subsidy for low-income group
        low_income_subsidy = (
            session.query(
                func.sum(ServiceBookingLog.government_subsidy).label('total_subsidy')
            )
            .join(CommuterInfoLog, CommuterInfoLog.commuter_id == ServiceBookingLog.commuter_id)
            .filter(CommuterInfoLog.income_level == 'low')
            .scalar() or 0
        )

        # Query: Usage per transport mode (via `record_company_name`) for low-income group
        mode_usage = (
            session.query(ServiceBookingLog.record_company_name, func.count(ServiceBookingLog.request_id))
            .join(CommuterInfoLog, CommuterInfoLog.commuter_id == ServiceBookingLog.commuter_id)
            .filter(CommuterInfoLog.income_level == 'low')
            .group_by(ServiceBookingLog.record_company_name)
            .all()
        )

        # Convert query results to a dictionary for easy handling
        usage_dict = dict(mode_usage)  # Example: {'Bike': 10, 'Car': 5, 'MaaS': 20}

        # Calculate the Gini index for mode usage
        gini_usage = calculate_gini_coefficient(list(usage_dict.values()))

        return gini_usage, low_income_subsidy

    finally:
        session.close()

# def run_multiple_simulations(parameter_sets):
#     """
#     Run multiple simulations and collect Gini index and total subsidy for low-income group.
#     """
#     low_income_gini_results = []
#     low_income_total_subsidies = []

#     for i, params in enumerate(parameter_sets):
#         print(f"Running simulation #{i + 1}")
#         usage, subsidy = run_single_simulation(params)
#         low_income_gini_results.append(usage)
#         low_income_total_subsidies.append(subsidy)

#     return low_income_gini_results, low_income_total_subsidies

def plot_gini_vs_subsidies(gini_results, total_subsidies, title, color):
    """
    Plot scatter plot for Gini Coefficient vs Total Subsidies.
    """
    X = np.array(total_subsidies)
    y = np.array(gini_results)

    print(f"Total Subsidies (X): {X}")
    print(f"Gini Coefficients (y): {y}")

    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color=color, label='Gini Coefficients')
    plt.title(title)
    plt.xlabel('Total Subsidy')
    plt.ylabel('Gini Coefficient')
    plt.legend()
    plt.grid(True)
    plt.show()


base_parameters = {
    'db_connection_string': DB_CONNECTION_STRING,
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

parameter_sets = [
    {
        **base_parameters,
        'subsidy_dataset': {
            'low': {'bike': 0.1, 'car': 0.05, 'MaaS_Bundle': 0.1},
            'middle': {'bike': 0.3, 'car': 0.01, 'MaaS_Bundle': 0.5},
            'high': {'bike': 0.4, 'car': 0, 'MaaS_Bundle': 0.02}
        }
    },
    {
        **base_parameters,
        'subsidy_dataset': {
            'low': {'bike': 0.6, 'car': 0.8, 'MaaS_Bundle': 0.55},
            'middle': {'bike': 0.25, 'car': 0.1, 'MaaS_Bundle': 0.45},
            'high': {'bike': 0.05, 'car': 0.05, 'MaaS_Bundle': 0.01}
        }
    },
    {
        **base_parameters,
        'subsidy_dataset': {
            'low': {'bike': 0.8, 'car': 0.6, 'MaaS_Bundle': 0.5},
            'middle': {'bike': 0.2, 'car': 0.2, 'MaaS_Bundle': 0.4},
            'high': {'bike': 0.1, 'car': 0.1, 'MaaS_Bundle': 0.1}
        }
    },
    {
        **base_parameters,
        'subsidy_dataset': {
            'low': {'bike': 0.4, 'car': 0.3, 'MaaS_Bundle': 0.5},
            'middle': {'bike': 0.4, 'car': 0.3, 'MaaS_Bundle': 0.3},
            'high': {'bike': 0.35, 'car': 0.15, 'MaaS_Bundle': 0.4}
        }
    },
    {
        **base_parameters,
        'subsidy_dataset': {
            'low': {'bike': 0.6, 'car': 0.7, 'MaaS_Bundle': 0.65},
            'middle': {'bike': 0.35, 'car': 0.2, 'MaaS_Bundle': 0.3},
            'high': {'bike': 0.01, 'car': 0.01, 'MaaS_Bundle': 0.02}
        }
    },
    {
        **base_parameters,
        'subsidy_dataset': {
            'low': {'bike': 0.2, 'car': 0.1, 'MaaS_Bundle': 0.25},
            'middle': {'bike': 0.3, 'car': 0.15, 'MaaS_Bundle': 0.4},
            'high': {'bike': 0.2, 'car': 0.1, 'MaaS_Bundle': 0.15}
        }
    },
]
more_parameter_sets = [
    {
        **base_parameters,
        'subsidy_dataset': {
            'low': {'bike': 0.1, 'car': 0.02, 'MaaS_Bundle': 0.05},
            'middle': {'bike': 0.25, 'car': 0.05, 'MaaS_Bundle': 0.35},
            'high': {'bike': 0.35, 'car': 0.15, 'MaaS_Bundle': 0.1}
        }
    },
    {
        **base_parameters,
        'subsidy_dataset': {
            'low': {'bike': 0.7, 'car': 0.85, 'MaaS_Bundle': 0.75},
            'middle': {'bike': 0.3, 'car': 0.1, 'MaaS_Bundle': 0.4},
            'high': {'bike': 0.35, 'car': 0.2, 'MaaS_Bundle': 0.2}
        }
    },
    {
        **base_parameters,
        'subsidy_dataset': {
            'low': {'bike': 0.4, 'car': 0.2, 'MaaS_Bundle': 0.7},
            'middle': {'bike': 0.35, 'car': 0.25, 'MaaS_Bundle': 0.4},
            'high': {'bike': 0.25, 'car': 0.1, 'MaaS_Bundle': 0.05}
        }
    },
    {
        **base_parameters,
        'subsidy_dataset': {
            'low': {'bike': 0.2, 'car': 0.8, 'MaaS_Bundle': 0.35},
            'middle': {'bike': 0.35, 'car': 0.1, 'MaaS_Bundle': 0.4},
            'high': {'bike': 0.4, 'car': 0.2, 'MaaS_Bundle': 0.1}
        }
    },
    {
        **base_parameters,
        'subsidy_dataset': {
            'low': {'bike': 0.25, 'car': 0.1, 'MaaS_Bundle': 0.4},
            'middle': {'bike': 0.3, 'car': 0.15, 'MaaS_Bundle': 0.45},
            'high': {'bike': 0.35, 'car': 0.05, 'MaaS_Bundle': 0.2}
        }
    },
    {
        **base_parameters,
        'subsidy_dataset': {
            'low': {'bike': 0.4, 'car': 0.15, 'MaaS_Bundle': 0.6},
            'middle': {'bike': 0.35, 'car': 0.2, 'MaaS_Bundle': 0.2},
            'high': {'bike': 0.25, 'car': 0.25, 'MaaS_Bundle': 0.04}
        }
    },
]

# Combine with the original set if needed
parameter_sets += more_parameter_sets

def run_parallel_simulations(parameter_sets, num_cpus):
    with mp.Pool(processes=num_cpus) as pool:
        results = pool.map(run_single_simulation, parameter_sets)
    
    low_income_gini_results, low_income_total_subsidies = zip(*results)
    return low_income_gini_results, low_income_total_subsidies

# Running the simulations with 12 sets of parameters in parallel
num_cpus = 4  # Adjust based on available CPUs
low_income_gini_results, low_income_total_subsidies = run_parallel_simulations(parameter_sets, num_cpus)

# # Run simulations
# low_income_gini_results, low_income_total_subsidies = run_multiple_simulations(parameter_sets)

# Save results to pickle files
with open('simulation_results_low_income_equity.pkl', 'wb') as f:
    pickle.dump((low_income_gini_results, low_income_total_subsidies), f)
print("Simulation results saved successfully.")

# # Load saved results (if needed)
# with open('simulation_results_low_income_equity.pkl', 'rb') as f:
#     low_income_gini_results, low_income_total_subsidies = pickle.load(f)

# Plot the results
plot_gini_vs_subsidies(
    low_income_gini_results,
    low_income_total_subsidies,
    'Low-Income Gini Coefficient vs Total Subsidy',
    'green'
)