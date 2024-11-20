from sqlalchemy import create_engine, func
from run_visualisation_03 import MobilityModel
from agent_service_provider_initialisation_03 import UberLike1, UberLike2, BikeShare1, BikeShare2, reset_database
from sqlalchemy.orm import sessionmaker
import numpy as np
import matplotlib.pyplot as plt
from agent_service_provider_initialisation_03 import CommuterInfoLog, ServiceBookingLog

# Global configuration
DB_CONNECTION_STRING = 'sqlite:///service_provider_database_01.db'
SIMULATION_STEPS = 10  # Increase number of steps for better analysis
num_commuters = 100

engine = create_engine(DB_CONNECTION_STRING)
Session = sessionmaker(bind=engine)

# Function to fetch current prices
def get_current_prices(session, step_count):
    """Fetches current prices for UberLike1, UberLike2, BikeShare1, and BikeShare2 at a specific step count."""
    
    uberlike1_price = session.query(UberLike1.current_price_0).filter(UberLike1.step_count == step_count).first()
    uberlike2_price = session.query(UberLike2.current_price_0).filter(UberLike2.step_count == step_count).first()
    bikeshare1_price = session.query(BikeShare1.current_price_0).filter(BikeShare1.step_count == step_count).first()
    bikeshare2_price = session.query(BikeShare2.current_price_0).filter(BikeShare2.step_count == step_count).first()

    return {
        'UberLike1': uberlike1_price[0] if uberlike1_price else None,
        'UberLike2': uberlike2_price[0] if uberlike2_price else None,
        'BikeShare1': bikeshare1_price[0] if bikeshare1_price else None,
        'BikeShare2': bikeshare2_price[0] if bikeshare2_price else None
    }

# Function to query mode shares
def query_mode_shares(session, step_count):
    """Fetch mode shares at a particular step count."""
    total_bookings = session.query(ServiceBookingLog).count()
    if total_bookings == 0:
        return {'UberLike1': 0, 'UberLike2': 0, 'BikeShare1': 0, 'BikeShare2': 0}

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

def calculate_price_elasticity(price_fluctuations, mode_shares_over_time):
    """Calculates the price elasticity of demand for each mode."""
    elasticity = {}

    for mode in ['UberLike1', 'UberLike2', 'BikeShare1', 'BikeShare2']:
        try:
            # Extract prices and mode shares for the current mode
            prices = [entry[mode] for entry in price_fluctuations if entry[mode] is not None]
            shares = [share[mode] for share in mode_shares_over_time if share[mode] is not None]

            if len(prices) >= 2 and len(shares) >= 2:
                # Calculate percentage changes in price and demand
                price_change = (prices[-1] - prices[0]) / prices[0] if prices[0] != 0 else 0
                demand_change = (shares[-1] - shares[0]) / shares[0] if shares[0] != 0 else 0

                # Calculate elasticity (Price Elasticity of Demand formula)
                if price_change != 0:
                    elasticity[mode] = demand_change / price_change
                else:
                    elasticity[mode] = 0
            else:
                elasticity[mode] = 0
        except ZeroDivisionError:
            elasticity[mode] = 0
        except Exception as e:
            print(f"Error calculating elasticity for {mode}: {e}")
            elasticity[mode] = None

    return elasticity


def run_model_with_dynamic_pricing(params):
    """Runs the MobilityModel with dynamic pricing and tracks price fluctuations, mode shares over time."""
    print(f"Running model with parameters: {params}")

    # Create session
    session = Session()

    # Reset the database and initialize the model
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
        bike_share2_price=params['bike_share2_price'],
        subsidy_bike=params['subsidy_bike']
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
        subsidy_bike=params['subsidy_bike']
    )

    price_fluctuations = []
    mode_shares_over_time = []

    for step in range(params['simulation_steps']):
        model.step()

        # Query the current prices at this step
        current_prices = get_current_prices(session, step)

        price_fluctuations.append({
            'step': step,
            'UberLike1': current_prices['UberLike1'],
            'UberLike2': current_prices['UberLike2'],
            'BikeShare1': current_prices['BikeShare1'],
            'BikeShare2': current_prices['BikeShare2']
        })


        # Track mode shares at each step
        mode_shares = query_mode_shares(session, step)
        mode_shares_over_time.append(mode_shares)

    # After the simulation, calculate price elasticity
    price_elasticity = calculate_price_elasticity(price_fluctuations, mode_shares_over_time)

    session.close()
    return price_fluctuations, mode_shares_over_time, price_elasticity

def plot_price_elasticity_combined(price_fluctuations_list, mode_shares_list):
    """Plots price fluctuations and mode shares on the same figure for each test."""
    num_tests = len(price_fluctuations_list)

    # Create a figure for combined price fluctuations and mode shares
    fig, axs = plt.subplots(num_tests, 1, figsize=(8, 5 * num_tests))  # Adjust height for the number of tests
    
    # If there is only one test, axs will not be a list, so we need to handle that case
    if num_tests == 1:
        axs = [axs]  # Convert it to a list for consistency

    for i, (price_fluctuations, mode_shares) in enumerate(zip(price_fluctuations_list, mode_shares_list)):
        steps = [entry.get('step', i) for i, entry in enumerate(price_fluctuations)]
        uberlike1_prices = [entry.get('UberLike1', 0) for entry in price_fluctuations]
        uberlike2_prices = [entry.get('UberLike2', 0) for entry in price_fluctuations]
        bikeshare1_prices = [entry.get('BikeShare1', 0) for entry in price_fluctuations]
        bikeshare2_prices = [entry.get('BikeShare2', 0) for entry in price_fluctuations]

        # Plot price fluctuations
        axs[i].plot(steps, uberlike1_prices, label='UberLike1 Price', color='blue')
        axs[i].plot(steps, uberlike2_prices, label='UberLike2 Price', color='orange')
        axs[i].plot(steps, bikeshare1_prices, label='BikeShare1 Price', color='green')
        axs[i].plot(steps, bikeshare2_prices, label='BikeShare2 Price', color='red')
        axs[i].set_xlabel('Simulation Steps')
        axs[i].set_ylabel('Price')
        axs[i].set_title(f"Price Fluctuations for Test #{i+1}")
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
            print(f"Error: Mode shares data does not match the number of steps for Test #{i+1}")

    plt.tight_layout()
    plt.show()


# Function to run dynamic pricing sensitivity tests
def run_dynamic_pricing_sensitivity_tests():
    """Runs dynamic pricing sensitivity tests and performs expanded analysis."""
    price_fluctuations_list = []
    mode_shares_list = []
    price_elasticity_list = []
    titles = []


    # Define parameter sets for testing (same as before)
    parameter_sets = [
        {
            'db_connection_string': DB_CONNECTION_STRING,
            'engine': engine,
            'num_commuters': 100,
            'grid_width': 55,
            'grid_height': 55,
            'data_income_weights': [0.5, 0.3, 0.2],
            'data_health_weights': [0.9, 0.1],
            'data_payment_weights': [0.8, 0.2],
            'data_age_distribution': {(18, 25): 0.15, (26, 35): 0.175, (36, 45): 0.25, (46, 55): 0.175, (56, 65): 0.15, (66, 75): 0.10},
            'data_disability_weights': [0.15, 0.85],
            'data_tech_access_weights': [0.98, 0.02],
            'CHANCE_FOR_INSERTING_RANDOM_TRAFFIC': 0.2,
            'ASC_VALUES': {'walk': -2, 'bike': -1.5, 'car': 0.01, 'public': -1, 'maas': -1, 'default': 0},
            'UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS': {'beta_C': -0.05, 'beta_T': -0.06},
            'UTILITY_FUNCTION_BASE_COEFFICIENTS': {'beta_C': -0.05, 'beta_T': -0.06, 'beta_W': -0.01, 'beta_A': -0.01, 'alpha': -0.01},
            'PENALTY_COEFFICIENTS': {'disability_bike_walk': 0.8, 'age_health_bike_walk': 0.3, 'no_tech_access_car_bike': 0.1},
            'AFFORDABILITY_THRESHOLDS': {'low': 25, 'middle': 85, 'high': 250, 'default': 65},
            'FLEXIBILITY_ADJUSTMENTS': {'low': 1.05, 'medium': 1.0, 'high': 0.95},
            'VALUE_OF_TIME': {'low': 9.64, 'middle': 9.64, 'high': 9.64},
            'public_price_table': {'train': {'on_peak': 2, 'off_peak': 1.5}, 'bus': {'on_peak': 1, 'off_peak': 0.8}},
            'ALPHA_VALUES': {'UberLike1': 0.5, 'UberLike2': 0.5, 'BikeShare1': 0.5, 'BikeShare2': 0.5},
            'DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS': {'S_base': 0.08, 'alpha': 0.2, 'delta': 0.5},
            'BACKGROUND_TRAFFIC_AMOUNT': 40,
            'CONGESTION_ALPHA': 0.15,
            'CONGESTION_BETA': 4,
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
            'subsidy_bike': 0,
            'simulation_steps': 10  # Adjust steps for more granularity
        },
        # Add more parameter sets as needed...
    ]

    for i, params in enumerate(parameter_sets):
        print(f"\n=== Running dynamic pricing sensitivity test #{i+1} ===")
        price_fluctuations, mode_shares_over_time, price_elasticity = run_model_with_dynamic_pricing(params)

        price_fluctuations_list.append(price_fluctuations)
        mode_shares_list.append(mode_shares_over_time)
        price_elasticity_list.append(price_elasticity)
        titles.append(f"Test #{i+1}")

        # Print price elasticity for debugging
        print(f"Price Elasticity for Test #{i+1}:")
        for mode, elasticity in price_elasticity.items():
            print(f"  {mode}: {elasticity}")

    # Plot the combined results
  
    plot_price_elasticity_combined(price_fluctuations_list, mode_shares_list)

    # You can add functions to plot or further analyze the price elasticity results as needed.

if __name__ == "__main__":
    run_dynamic_pricing_sensitivity_tests()
