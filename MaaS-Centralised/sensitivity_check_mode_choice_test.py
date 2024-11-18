
from sqlalchemy import create_engine, func
from run_visualisation_03 import MobilityModel
from agent_service_provider_initialisation_03 import reset_database
from sqlalchemy.orm import sessionmaker
import matplotlib.pyplot as plt
from agent_service_provider_initialisation_03 import CommuterInfoLog, ServiceBookingLog
import numpy as np
# Global configuration
DB_CONNECTION_STRING = 'sqlite:///service_provider_database_01.db'
SIMULATION_STEPS = 3 # For testing purposes, keep the steps small
num_commuters = 100

engine = create_engine(DB_CONNECTION_STRING)
Session = sessionmaker(bind=engine)

# Function to run the model with parameters
def run_model_with_parameters(params):
    """Runs the MobilityModel with given parameters and returns results from the simulation."""
    print(f"Running model with parameters: {params}")

    # Create a session
    session = Session()

    # Reset the database with dynamic parameters for service providers
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

    # Initialize and run the MobilityModel with provided parameters
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
        uber_like1_capacity=params['uber_like1_capacity'],  # Add missing parameters here
        uber_like1_price=params['uber_like1_price'],
        uber_like2_capacity=params['uber_like2_capacity'],
        uber_like2_price=params['uber_like2_price'],
        bike_share1_capacity=params['bike_share1_capacity'],
        bike_share1_price=params['bike_share1_price'],
        bike_share2_capacity=params['bike_share2_capacity'],
        bike_share2_price=params['bike_share2_price'],
        subsidy_dataset=params['subsidy_dataset']
    )

    # Run the model for the defined number of steps
    model.run_model(params['simulation_steps'])

    # Query and return the simulation results from the database
    results = query_simulation_results(params['db_connection_string'])
    return results


# Function to query the simulation results from the database
# Function to query the simulation results from the database
def query_simulation_results(db_connection_string):
    session = Session()

    # Join commuter_info_log with service_booking_log to get commuter information and mode choices
    result = session.query(
        CommuterInfoLog.income_level,
        CommuterInfoLog.age,
        CommuterInfoLog.health_status,
        ServiceBookingLog.record_company_name,  # Use the correct column name
        func.count(ServiceBookingLog.request_id).label("num_bookings")
    ).join(ServiceBookingLog, CommuterInfoLog.commuter_id == ServiceBookingLog.commuter_id) \
     .group_by(CommuterInfoLog.income_level, ServiceBookingLog.record_company_name) \
     .all()

    session.close()
    return result





# # Function to visualize the results
# def plot_all_mode_choices(all_mode_choices_by_income, titles, n_cols=2):
#     """Plot mode choice results from different simulations in a grid layout.
    
#     Args:
#     - all_mode_choices_by_income: A list of mode choice dictionaries (one per simulation).
#     - titles: A list of titles for each subplot (one per simulation).
#     - n_cols: Number of columns in the plot grid (default is 2).
#     """
#     n_simulations = len(all_mode_choices_by_income)
#     n_rows = (n_simulations + n_cols - 1) // n_cols  # Determine number of rows needed
    
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 6))
    
#     # Flatten axes array to handle both single and multi-row cases
#     axes = axes.flatten()
    
#     for i, mode_choices_by_income in enumerate(all_mode_choices_by_income):
#         ax = axes[i]  # Select the current subplot
        
#         # For each income level, we plot a bar chart
#         for income_level, mode_data in mode_choices_by_income.items():
#             labels = mode_data.keys()
#             values = mode_data.values()

#             ax.bar(labels, values, label=f"{income_level.capitalize()} Income")
        
#         ax.set_title(titles[i])
#         ax.set_xlabel("Mode")
#         ax.set_ylabel("Number of Bookings")
#         ax.legend()
    
#     # Hide any empty subplots
#     for j in range(i+1, len(axes)):
#         fig.delaxes(axes[j])
    
#     plt.tight_layout()  # Adjust layout to avoid overlap
#     plt.savefig('sensitivity_results.png')
#     plt.show()

def process_mode_choice_overall(simulation_data):
    """Process overall mode choice data for pie chart visualization."""
    overall_mode_choices = {
        'UberLike1': 0,
        'UberLike2': 0,
        'BikeShare1': 0,
        'BikeShare2': 0,
        'MaaS_Bundle': 0,
        'walk': 0,
        'public': 0
    }

    # Sum up the bookings for each mode across all income levels
    for row in simulation_data:
        _, _, _, mode, count = row
        if mode in overall_mode_choices:
            overall_mode_choices[mode] += count

    return overall_mode_choices

def plot_pie_chart_mode_choice(all_mode_choices_overall, titles):
    """
    Plot the pie chart for mode choice distribution across different sensitivity tests.
    
    all_mode_choices_overall: List of mode choice data for each sensitivity test.
    titles: Titles for each subplot (one per sensitivity test).
    """
    num_tests = len(all_mode_choices_overall)
    fig, axs = plt.subplots(1, num_tests, figsize=(6 * num_tests, 6))  # Create a subplot for each test
    
    if num_tests == 1:
        axs = [axs]  # If there's only one test, convert axs to a list for consistency
    
    # Set the mode and color scheme
    modes = ['UberLike1', 'UberLike2', 'BikeShare1', 'BikeShare2', 'public', 'walk', 'MaaS_Bundle']
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'grey', 'pink']

    for i, (mode_choices, ax) in enumerate(zip(all_mode_choices_overall, axs)):
        values = [mode_choices[mode] for mode in modes]
        
        # Create the pie chart for each test
        ax.pie(values, labels=modes, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title(titles[i])
    
    plt.tight_layout()
    plt.savefig('mode_choice_pie_chart.png')
    plt.show()



# Function to aggregate the results by mode choice and income level
def process_mode_choice_by_income(data):
    mode_choices_by_income = {
        'low': {
            'UberLike1': 0, 
            'UberLike2': 0, 
            'BikeShare1': 0, 
            'BikeShare2': 0, 
            'MaaS_Bundle': 0,
            'walk': 0, 
            'public': 0
        },
        'middle': {
            'UberLike1': 0, 
            'UberLike2': 0, 
            'BikeShare1': 0, 
            'BikeShare2': 0, 
            'MaaS_Bundle': 0,
            'walk': 0, 
            'public': 0
        },
        'high': {
            'UberLike1': 0, 
            'UberLike2': 0, 
            'BikeShare1': 0, 
            'BikeShare2': 0, 
            'MaaS_Bundle': 0,
            'walk': 0, 
            'public': 0
        }
    }

    for row in data:
        income_level, age, health_status, mode, count = row
        if income_level in mode_choices_by_income and mode in mode_choices_by_income[income_level]:
            mode_choices_by_income[income_level][mode] += count

    return mode_choices_by_income

# Function to process mode choice percentages by income
def process_mode_choice_percentages(mode_choices_by_income):
    percentages_by_income = {
        'low': {},
        'middle': {},
        'high': {}
    }

    # For each income level, calculate the percentage of each mode choice
    for income_level, mode_data in mode_choices_by_income.items():
        total_bookings = sum(mode_data.values())
        if total_bookings > 0:
            for mode, count in mode_data.items():
                percentages_by_income[income_level][mode] = (count / total_bookings) * 100
        else:
            for mode in mode_data.keys():
                percentages_by_income[income_level][mode] = 0  # Avoid division by zero

    return percentages_by_income

# Function to plot the mode choice by income as a stacked percentage bar chart
# Function to plot the mode choice by income as a stacked percentage bar chart
def plot_mode_choice_percentage_by_income(all_mode_choices_by_income, titles):
    """Plot mode choice percentages by income level for all sensitivity tests on a single canvas."""
    num_tests = len(all_mode_choices_by_income)
    
    # Setup the plot with subplots based on the number of sensitivity tests
    fig, axs = plt.subplots(1, num_tests, figsize=(6 * num_tests, 6))  # Create a subplot for each test
    
    if num_tests == 1:
        axs = [axs]  # If there's only one test, convert axs to a list for consistency

    income_levels = ['low', 'middle', 'high']
    # Updated modes and colors as per your request
    modes = ['UberLike1', 'UberLike2', 'BikeShare1', 'BikeShare2', 'public', 'walk', 'MaaS_Bundle']
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'grey', 'pink']  # Color for each mode

    for i, (mode_choices_by_income, ax) in enumerate(zip(all_mode_choices_by_income, axs)):
        # Prepare the data for the stacked bar chart
        income_level_data = {income: [] for income in income_levels}

        for mode in modes:
            for income_level in income_levels:
                income_level_data[income_level].append(mode_choices_by_income[income_level][mode])

        # Plot the stacked bars
        bottom = [0] * len(income_levels)
        for j, mode in enumerate(modes):
            values = [income_level_data[income][j] for income in income_levels]
            total_values = [sum(income_level_data[income]) for income in income_levels]
            percentages = [value / total * 100 if total > 0 else 0 for value, total in zip(values, total_values)]
            
            ax.bar(income_levels, percentages, bottom=bottom, color=colors[j], label=mode)
            bottom = [x + y for x, y in zip(bottom, percentages)]

        ax.set_title(titles[i])
        ax.set_ylabel('Percentage')
        ax.set_ylim(0, 100)
        ax.legend(loc='upper left', title="Mode")

    plt.tight_layout()
    plt.show()




# Sensitivity testing function
def run_sensitivity_tests():
    all_mode_choices_by_income = []  # To store mode choice results for all simulations
    all_mode_choices_overall = []  # To store overall mode choice results for all simulations
    titles = []  # To store the titles for each subplot

    base_parameters = {
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
    'simulation_steps': SIMULATION_STEPS
    }

    # Generate 5 fixed parameter sets for analysis
    parameter_sets_ASC = [
        {
            **base_parameters,
            'ASC_VALUES': {'car': 0, 'bike': 0, 'public': 0, 'walk': 0, 'maas': 0, 'default': 0},
            'subsidy_dataset': {
                'low': {'bike': 0, 'car': 0, 'MaaS_Bundle': 0},
                'middle': {'bike': 0, 'car': 0, 'MaaS_Bundle': 0},
                'high': {'bike': 0, 'car': 0, 'MaaS_Bundle': 0}
            }
        },
        {
            **base_parameters,
            'ASC_VALUES': {'car': 3, 'bike': 5, 'public': 4, 'walk': 5, 'maas': 10, 'default': 0},
            'subsidy_dataset': {
                'low': {'bike': 0, 'car': 0, 'MaaS_Bundle': 0},
                'middle': {'bike': 0, 'car': 0, 'MaaS_Bundle': 0},
                'high': {'bike': 0, 'car': 0, 'MaaS_Bundle': 0}
            }
        },
        {
            **base_parameters,
            'ASC_VALUES': {'car': -3, 'bike': 7, 'public': 5, 'walk': -4, 'maas': -6, 'default': 0},
            'subsidy_dataset': {
                'low': {'bike': 0, 'car': 0, 'MaaS_Bundle': 0},
                'middle': {'bike': 0, 'car': 0, 'MaaS_Bundle': 0},
                'high': {'bike': 0, 'car': 0, 'MaaS_Bundle': 0}
            }
        },

        {
            **base_parameters,
            'ASC_VALUES': {'car': 1, 'bike': 10, 'public': 1, 'walk': 5, 'maas': 7, 'default': 0},
            'subsidy_dataset': {
                'low': {'bike': 0, 'car': 0, 'MaaS_Bundle': 0},
                'middle': {'bike': 0, 'car': 0, 'MaaS_Bundle': 0},
                'high': {'bike': 0, 'car': 0, 'MaaS_Bundle': 0}
            }
        },
        {
            **base_parameters,
            'ASC_VALUES': {'car': 0, 'bike': 0, 'public': -5, 'walk': 6, 'maas': -8, 'default': 0},
            'subsidy_dataset': {
                'low': {'bike': 0, 'car': 0, 'MaaS_Bundle': 0},
                'middle': {'bike': 0, 'car': 0, 'MaaS_Bundle': 0},
                'high': {'bike': 0, 'car': 0, 'MaaS_Bundle': 0}
            }
        }
    ]

    parameter_sets_subsidy = [
        {
            **base_parameters,
            'ASC_VALUES': {'car': 0, 'bike': 0, 'public': 0, 'walk': 0, 'maas': 0, 'default': 0},
            'subsidy_dataset': {
                'low': {'bike': 0, 'car': 0, 'MaaS_Bundle': 0},
                'middle': {'bike': 0, 'car': 0, 'MaaS_Bundle': 0},
                'high': {'bike': 0, 'car': 0, 'MaaS_Bundle': 0}
            }
        },
        {
            **base_parameters,
            'ASC_VALUES': {'car': 0, 'bike': 0, 'public': 0, 'walk': 0, 'maas': 0, 'default': 0},
            'subsidy_dataset': {
                'low': {'bike': 0.60, 'car': 0.45, 'MaaS_Bundle': 0.70},
                'middle': {'bike': 0.25, 'car': 0.40, 'MaaS_Bundle': 0.45},
                'high': {'bike': 0.15, 'car': 0.20, 'MaaS_Bundle': 0.25}
            }
        },
        {
            **base_parameters,
            'ASC_VALUES': {'car': 0, 'bike': 0, 'public': 0, 'walk': 0, 'maas': 0, 'default': 0},
            'subsidy_dataset': {
                'low': {'bike': 0.50, 'car': 0.35, 'MaaS_Bundle': 0.80},
                'middle': {'bike': 0.45, 'car': 0.25, 'MaaS_Bundle': 0.35},
                'high': {'bike': 0.05, 'car': 0.10, 'MaaS_Bundle': 0.30}
            }
        },
        {
            **base_parameters,
            'ASC_VALUES': {'car': 0, 'bike': 0, 'public': 0, 'walk': 0, 'maas': 0, 'default': 0},
            'subsidy_dataset': {
                'low': {'bike': 0.45, 'car': 0.50, 'MaaS_Bundle': 0.3},
                'middle': {'bike': 0.30, 'car': 0.40, 'MaaS_Bundle': 0.35},
                'high': {'bike': 0.15, 'car': 0.20, 'MaaS_Bundle': 0.20}
            }
        },
        {
            **base_parameters,
            'ASC_VALUES': {'car': 0, 'bike': 0, 'public': 0, 'walk': 0, 'maas': 0, 'default': 0},
            'subsidy_dataset': {
                'low': {'bike': 0.60, 'car': 0.50, 'MaaS_Bundle': 0.75},
                'middle': {'bike': 0.45, 'car': 0.30, 'MaaS_Bundle': 0.35},
                'high': {'bike': 0.05, 'car': 0.20, 'MaaS_Bundle': 0.30}
            }
        }
    ]


    # Loop through each parameter set and run the model
    for i, params in enumerate(parameter_sets_ASC):
    # for i, params in enumerate(parameter_sets_subsidy):
        print(f"\n=== Running sensitivity test #{i+1} ===")
        simulation_results = run_model_with_parameters(params)

        # Process the results and analyze mode choices by income
        mode_choices_by_income = process_mode_choice_by_income(simulation_results)  # By income
        mode_choices_overall = process_mode_choice_overall(simulation_results)  # Overall

        all_mode_choices_by_income.append(mode_choices_by_income)
        all_mode_choices_overall.append(mode_choices_overall)
        titles.append(f"Sensitivity Test #{i+1}")

    # Plot mode choices by income as stacked bar charts
    plot_mode_choice_percentage_by_income(all_mode_choices_by_income, titles)

    # Plot overall mode choices as pie charts
    plot_pie_chart_mode_choice(all_mode_choices_overall, titles)
    


# Run sensitivity tests
if __name__ == "__main__":
    run_sensitivity_tests()
