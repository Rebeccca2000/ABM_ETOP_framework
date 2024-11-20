from sqlalchemy import create_engine, func
from run_visualisation_03 import MobilityModel
from agent_service_provider_initialisation_03 import UberLike1, UberLike2, BikeShare1, BikeShare2, reset_database
from sqlalchemy.orm import sessionmaker

import numpy as np
from agent_service_provider_initialisation_03 import CommuterInfoLog, ServiceBookingLog

import matplotlib.pyplot as plt
# Global configuration
DB_CONNECTION_STRING = 'sqlite:///service_provider_database_01.db'
SIMULATION_STEPS = 144  # Run for a larger number of steps for better analysis
num_commuters = 100

engine = create_engine(DB_CONNECTION_STRING)
Session = sessionmaker(bind=engine)

def get_current_prices(session, step_count):
    """Fetches current prices from UberLike1, UberLike2, BikeShare1, and BikeShare2 tables."""
    
    # Query current prices for UberLike1
    uberlike1_price = session.query(UberLike1.current_price_0).filter(UberLike1.step_count == step_count).first()
    uberlike1_price = uberlike1_price[0] if uberlike1_price else None

    # Query current prices for UberLike2
    uberlike2_price = session.query(UberLike2.current_price_0).filter(UberLike2.step_count == step_count).first()
    uberlike2_price = uberlike2_price[0] if uberlike2_price else None

    # Query current prices for BikeShare1
    bikeshare1_price = session.query(BikeShare1.current_price_0).filter(BikeShare1.step_count == step_count).first()
    bikeshare1_price = bikeshare1_price[0] if bikeshare1_price else None

    # Query current prices for BikeShare2
    bikeshare2_price = session.query(BikeShare2.current_price_0).filter(BikeShare2.step_count == step_count).first()
    bikeshare2_price = bikeshare2_price[0] if bikeshare2_price else None

    return {
        'UberLike1': uberlike1_price,
        'UberLike2': uberlike2_price,
        'BikeShare1': bikeshare1_price,
        'BikeShare2': bikeshare2_price
    }
def query_mode_shares(session, step_count):
    """Fetch mode shares at a particular step."""
    total_bookings = session.query(ServiceBookingLog).count()
    if total_bookings == 0:
        return {'UberLike1': 0, 'UberLike2': 0, 'BikeShare1': 0, 'BikeShare2': 0}

    # Calculate mode shares for UberLike1, UberLike2, BikeShare1, BikeShare2
    uberlike1_share = session.query(ServiceBookingLog).filter(ServiceBookingLog.record_company_name == 'UberLike1').count() / total_bookings
    uberlike2_share = session.query(ServiceBookingLog).filter(ServiceBookingLog.record_company_name == 'UberLike2').count() / total_bookings
    bikeshare1_share = session.query(ServiceBookingLog).filter(ServiceBookingLog.record_company_name == 'BikeShare1').count() / total_bookings
    bikeshare2_share = session.query(ServiceBookingLog).filter(ServiceBookingLog.record_company_name == 'BikeShare2').count() / total_bookings

    return {
        'UberLike1': uberlike1_share,
        'UberLike2': uberlike2_share,
        'BikeShare1': bikeshare1_share,
        'BikeShare2': bikeshare2_share
    }
def run_model_with_dynamic_pricing(params):
    """Runs the MobilityModel with dynamic pricing and tracks price fluctuations and mode shares over time."""
    print(f"Running model with parameters: {params}")

    # Create a session
    session = Session()

    # Reset the database
    reset_database(
        engine=params['engine'],
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

    # Initialize the model
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
        subsidy_dataset=params['subsidy_dataset']
    )

    # Run the model and track price fluctuations and mode shares over time
    price_fluctuations = []
    mode_shares_over_time = []

    for step in range(params['simulation_steps']):
        model.step()

        # Query the current prices at this step
        current_prices = get_current_prices(session, step)

        # Append prices to the tracking list
        price_fluctuations.append({
            'step': step,
            'UberLike1': current_prices['UberLike1'],
            'UberLike2': current_prices['UberLike2'],
            'BikeShare1': current_prices['BikeShare1'],
            'BikeShare2': current_prices['BikeShare2']
        })

        # Track mode shares at each step and add them to mode_shares_over_time
        mode_shares = query_mode_shares(session, step)
        mode_shares_over_time.append(mode_shares)

    session.close()
    return price_fluctuations, mode_shares_over_time


# Function to plot price fluctuations for multiple simulations, each in its own subplot on the same figure
def plot_price_fluctuations_combined(price_fluctuations_list, titles, overall_title="Price Fluctuations Comparison"):
    num_tests = len(price_fluctuations_list)
    fig, axs = plt.subplots(num_tests, 1, figsize=(10, 6 * num_tests))  # Create a row of subplots

    for i, price_fluctuations in enumerate(price_fluctuations_list):
        steps = [entry['step'] for entry in price_fluctuations]
        uberlike1_prices = [entry['UberLike1'] for entry in price_fluctuations]
        uberlike2_prices = [entry['UberLike2'] for entry in price_fluctuations]
        bikeshare1_prices = [entry['BikeShare1'] for entry in price_fluctuations]
        bikeshare2_prices = [entry['BikeShare2'] for entry in price_fluctuations]

        axs[i].plot(steps, uberlike1_prices, label='UberLike1')
        axs[i].plot(steps, uberlike2_prices, label='UberLike2')
        axs[i].plot(steps, bikeshare1_prices, label='BikeShare1')
        axs[i].plot(steps, bikeshare2_prices, label='BikeShare2')

        axs[i].set_title(titles[i])
        axs[i].set_xlabel('Simulation Steps')
        axs[i].set_ylabel('Price')
        axs[i].legend()
        axs[i].grid(True)

    fig.suptitle(overall_title)
    plt.tight_layout()
    plt.show()



def plot_price_elasticity_combined(price_fluctuations_list, mode_shares_list, titles):
    """Plots price fluctuations and mode shares on the same figure for each test."""
    num_tests = len(price_fluctuations_list)
    
    # Create a figure for combined price fluctuations and mode shares
    fig, axs = plt.subplots(num_tests, 1, figsize=(8, 5 * num_tests))  # Adjust height for the number of tests
    
    for i, (price_fluctuations, mode_shares, title) in enumerate(zip(price_fluctuations_list, mode_shares_list, titles)):
        steps = [entry['step'] for entry in price_fluctuations]
        uberlike1_prices = [entry['UberLike1'] for entry in price_fluctuations]
        uberlike2_prices = [entry['UberLike2'] for entry in price_fluctuations]
        bikeshare1_prices = [entry['BikeShare1'] for entry in price_fluctuations]
        bikeshare2_prices = [entry['BikeShare2'] for entry in price_fluctuations]

        # Plot price fluctuations
        axs[i].plot(steps, uberlike1_prices, label='UberLike1 Price', color='blue')
        axs[i].plot(steps, uberlike2_prices, label='UberLike2 Price', color='orange')
        axs[i].plot(steps, bikeshare1_prices, label='BikeShare1 Price', color='green')
        axs[i].plot(steps, bikeshare2_prices, label='BikeShare2 Price', color='red')
        axs[i].set_xlabel('Simulation Steps')
        axs[i].set_ylabel('Price')
        axs[i].set_title(title)
        axs[i].legend(loc='upper right')
        axs[i].grid(True)

        # Plot mode shares on secondary y-axis (twinx)
        if len(mode_shares) == len(steps):
            uberlike1_share = [share['UberLike1'] for share in mode_shares]
            uberlike2_share = [share['UberLike2'] for share in mode_shares]
            bikeshare1_share = [share['BikeShare1'] for share in mode_shares]
            bikeshare2_share = [share['BikeShare2'] for share in mode_shares]
            
            ax2 = axs[i].twinx()
            ax2.plot(steps, uberlike1_share, label='UberLike1 Share', linestyle='dotted', color='blue')
            ax2.plot(steps, uberlike2_share, label='UberLike2 Share', linestyle='dotted', color='orange')
            ax2.plot(steps, bikeshare1_share, label='BikeShare1 Share', linestyle='dotted', color='green')
            ax2.plot(steps, bikeshare2_share, label='BikeShare2 Share', linestyle='dotted', color='red')
            ax2.set_ylabel('Mode Share')
            ax2.legend(loc='lower right')
        else:
            print(f"Error: Mode shares data does not match the number of steps for {title}")
    
    # Save the plot to a file instead of showing it
    plt.tight_layout()
    plt.savefig('price_elasticity_combined.png')  # Save the figure as a PNG file
    print("Combined price elasticity and mode shares plot saved as 'price_elasticity_combined.png'.")


def recursive_merge(dict1, dict2):
    """Recursively merges two dictionaries. dict2 values will override dict1 values."""
    merged = dict1.copy()  # Make a copy of the first dictionary
    for key, value in dict2.items():
        if isinstance(value, dict) and key in merged:
            # If the value is a dictionary and the key exists in both, merge recursively
            merged[key] = recursive_merge(merged[key], value)
        else:
            # Otherwise, just override the value
            merged[key] = value
    return merged


# Run dynamic pricing sensitivity tests and plot results with each simulation in a separate subplot
def run_dynamic_pricing_sensitivity_tests():
    """Runs dynamic pricing sensitivity tests and performs expanded analysis."""
    price_fluctuations_list = []
    mode_shares_list = []
    titles = []



    parameter_sets = [
        # Set 1: Baseline congestion levels
        {
            'db_connection_string': DB_CONNECTION_STRING,
            'engine': engine,
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
            'ASC_VALUES': {'car': 0.01, 'bike': -1.5, 'public': -1, 'walk': -2, 'maas': -1, 'default': 0},
            'UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS': {'beta_C': -0.05, 'beta_T': -0.06},
            'UTILITY_FUNCTION_BASE_COEFFICIENTS': {'beta_C': -0.05, 'beta_T': -0.06, 'beta_W': -0.01, 'beta_A': -0.01, 'alpha': -0.01},
            'PENALTY_COEFFICIENTS': {'disability_bike_walk': 0.8, 'age_health_bike_walk': 0.3, 'no_tech_access_car_bike': 0.1},
            'AFFORDABILITY_THRESHOLDS': {'low': 25, 'middle': 85, 'high': 250},
            'FLEXIBILITY_ADJUSTMENTS': {'low': 1.05, 'medium': 1.0, 'high': 0.95},
            'VALUE_OF_TIME': {'low': 9.64, 'middle': 9.64, 'high': 9.64},
            'public_price_table': {'train': {'on_peak': 2, 'off_peak': 1.5}, 'bus': {'on_peak': 1, 'off_peak': 0.8}},
            'ALPHA_VALUES': {'UberLike1': 0.5, 'UberLike2': 0.5, 'BikeShare1': 0.5, 'BikeShare2': 0.5},
            'DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS': {'S_base': 0.08, 'alpha': 0.2, 'delta': 0.5},
            'BACKGROUND_TRAFFIC_AMOUNT': 40,
            'CONGESTION_ALPHA': 0.15,  # Low congestion
            'CONGESTION_BETA': 3,
            'CONGESTION_CAPACITY': 5,
            'CONGESTION_T_IJ_FREE_FLOW': 1,
            'uber_like1_capacity': 8,
            'uber_like1_price': 6,
            'uber_like2_capacity': 9,
            'uber_like2_price': 6.5,
            'bike_share1_capacity': 10,
            'bike_share1_price': 1,
            'bike_share2_capacity': 12,
            'bike_share2_price': 1.2,
            'subsidy_dataset':{
                'low': {'bike': 0.50, 'car': 0.5, 'MaaS_Bundle': 0.5},  # 50% subsidy for bikes, 20% for cars, etc.
                'middle': {'bike': 0.5, 'car': 0.5, 'MaaS_Bundle': 0.5},
                'high': {'bike': 0.5, 'car': 0.5, 'MaaS_Bundle': 0.5},
                },
            'simulation_steps': SIMULATION_STEPS
        },
        # Set 2: Moderate congestion
        {
            'db_connection_string': DB_CONNECTION_STRING,
            'engine': engine,
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
            'ASC_VALUES': {'car': 0.01, 'bike': -1.5, 'public': -1, 'walk': -2, 'maas': -1, 'default': 0},
            'UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS': {'beta_C': -0.05, 'beta_T': -0.06},
            'UTILITY_FUNCTION_BASE_COEFFICIENTS': {'beta_C': -0.05, 'beta_T': -0.06, 'beta_W': -0.01, 'beta_A': -0.01, 'alpha': -0.01},
            'PENALTY_COEFFICIENTS': {'disability_bike_walk': 0.8, 'age_health_bike_walk': 0.3, 'no_tech_access_car_bike': 0.1},
            'AFFORDABILITY_THRESHOLDS': {'low': 25, 'middle': 85, 'high': 250},
            'FLEXIBILITY_ADJUSTMENTS': {'low': 1.05, 'medium': 1.0, 'high': 0.95},
            'VALUE_OF_TIME': {'low': 9.64, 'middle': 9.64, 'high': 9.64},
            'public_price_table': {'train': {'on_peak': 2, 'off_peak': 1.5}, 'bus': {'on_peak': 1, 'off_peak': 0.8}},
            'ALPHA_VALUES': {'UberLike1': 0.5, 'UberLike2': 0.5, 'BikeShare1': 0.5, 'BikeShare2': 0.5},
            'DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS': {'S_base': 0.08, 'alpha': 0.2, 'delta': 0.5},
            'BACKGROUND_TRAFFIC_AMOUNT': 70,  # Higher background traffic
            'CONGESTION_ALPHA': 0.25,  # Moderate congestion
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
            'subsidy_dataset':{
                'low': {'bike': 0.50, 'car': 0.5, 'MaaS_Bundle': 0.5},  # 50% subsidy for bikes, 20% for cars, etc.
                'middle': {'bike': 0.5, 'car': 0.5, 'MaaS_Bundle': 0.5},
                'high': {'bike': 0.5, 'car': 0.5, 'MaaS_Bundle': 0.5},
                },
            'simulation_steps': SIMULATION_STEPS
        },
        # Set 3: High congestion
        {
            'db_connection_string': DB_CONNECTION_STRING,
            'engine': engine,
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
            'ASC_VALUES': {'car': 0.01, 'bike': -1.5, 'public': -1, 'walk': -2, 'maas': -1, 'default': 0},
            'UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS': {'beta_C': -0.05, 'beta_T': -0.06},
            'UTILITY_FUNCTION_BASE_COEFFICIENTS': {'beta_C': -0.05, 'beta_T': -0.06, 'beta_W': -0.01, 'beta_A': -0.01, 'alpha': -0.01},
            'PENALTY_COEFFICIENTS': {'disability_bike_walk': 0.8, 'age_health_bike_walk': 0.3, 'no_tech_access_car_bike': 0.1},
            'AFFORDABILITY_THRESHOLDS': {'low': 25, 'middle': 85, 'high': 250},
            'FLEXIBILITY_ADJUSTMENTS': {'low': 1.05, 'medium': 1.0, 'high': 0.95},
            'VALUE_OF_TIME': {'low': 9.64, 'middle': 9.64, 'high': 9.64},
            'public_price_table': {'train': {'on_peak': 2, 'off_peak': 1.5}, 'bus': {'on_peak': 1, 'off_peak': 0.8}},
            'ALPHA_VALUES': {'UberLike1': 0.5, 'UberLike2': 0.5, 'BikeShare1': 0.5, 'BikeShare2': 0.5},
            'DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS': {'S_base': 0.08, 'alpha': 0.2, 'delta': 0.5},
            'BACKGROUND_TRAFFIC_AMOUNT': 100,  # Maximum background traffic
            'CONGESTION_ALPHA': 0.35,  # High congestion
            'CONGESTION_BETA': 5,
            'CONGESTION_CAPACITY': 3,
            'CONGESTION_T_IJ_FREE_FLOW': 3,
            'uber_like1_capacity': 8,
            'uber_like1_price': 6,
            'uber_like2_capacity': 9,
            'uber_like2_price': 6.5,
            'bike_share1_capacity': 10,
            'bike_share1_price': 1,
            'bike_share2_capacity': 12,
            'bike_share2_price': 1.2,
            'subsidy_dataset':{
                'low': {'bike': 0.50, 'car': 0.5, 'MaaS_Bundle': 0.5},  # 50% subsidy for bikes, 20% for cars, etc.
                'middle': {'bike': 0.5, 'car': 0.5, 'MaaS_Bundle': 0.5},
                'high': {'bike': 0.5, 'car': 0.5, 'MaaS_Bundle': 0.5},
                },
            'simulation_steps': SIMULATION_STEPS
        }
    ]


    for i, params in enumerate(parameter_sets):
        print(f"\n=== Running simulation set {i + 1} ===")
    
        # Use recursive merge to combine the base parameters with the varying ones
        price_fluctuations, mode_shares_over_time = run_model_with_dynamic_pricing(params)

        price_fluctuations_list.append(price_fluctuations)
        mode_shares_list.append(mode_shares_over_time)
        titles.append(f"Test #{i+1}")

    # Plot the combined results
    plot_price_fluctuations_combined(price_fluctuations_list, titles)
    plot_price_elasticity_combined(price_fluctuations_list, mode_shares_list, titles)
# Run the sensitivity tests
if __name__ == "__main__":
    run_dynamic_pricing_sensitivity_tests()
