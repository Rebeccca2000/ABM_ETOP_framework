import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from run_visualisation_03 import MobilityModel
from sqlalchemy.orm import Session  # Placeholder for your database session
from agent_service_provider_initialisation_03 import reset_database, CommuterInfoLog, ServiceBookingLog
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from statsmodels.nonparametric.smoothers_lowess import lowess
import pickle

DB_CONNECTION_STRING = 'sqlite:///service_provider_database_01.db'
SIMULATION_STEPS = 36  # Run for a larger number of steps for better analysis
num_commuters = 100

engine = create_engine(DB_CONNECTION_STRING)
Session = sessionmaker(bind=engine)

# Function to calculate total system cost (SC)
def calculate_system_cost(session, VOT_values):
    """
    Calculate the total system cost (SC) using service booking log data and commuter income levels.
    
    VOT_values: A dictionary containing Value of Time (VOT) for low, middle, and high-income commuters.
    """
    service_bookings = session.query(ServiceBookingLog).all()  
    total_cost = 0

    for booking in service_bookings:
        commuter_id = booking.commuter_id  # Fetch commuter-specific data
        travel_time = booking.total_time
        total_price = booking.total_price
        government_subsidy = booking.government_subsidy
        
        # Query the commuter's income level from the CommuterInfoLog based on commuter_id
        commuter_info = session.query(CommuterInfoLog).filter_by(commuter_id=commuter_id).first()
        
        # Determine the income group (low, middle, high)
        if commuter_info:
            income_group = commuter_info.income_level
        else:
            continue  # If no commuter info found, skip to the next booking
        
        # VOT for this commuter based on their income group
        VOT = VOT_values.get(income_group, 0)  # Default VOT is 0 if income group is not found
        
        # Calculate cost for this booking
        booking_cost = total_price + (travel_time * VOT) + government_subsidy
        total_cost += booking_cost
    
    return total_cost  # Return total SC

# 1. Regression: SC vs 9 Independent Subsidy Variables
def regression_sc_vs_subsidies(sc_values, subsidy_data):
    X = np.array(subsidy_data)
    y = np.array(sc_values)
    
    X = sm.add_constant(X)  # Add constant for intercept
    model = sm.OLS(y, X).fit()
    
    print("\nRegression Results for SC vs 9 Subsidy Variables")
    print(model.summary())
    
    # Construct the final regression equation
    intercept, *coefficients = model.params
    equation = f"SC = {intercept:.4f}"
    for i, coef in enumerate(coefficients):
        equation += f" + {coef:.4f}x{i+1}"
    # print("\nFinal Regression Equation:", equation)
    
    return model, equation
# 2. Regression: SC vs ASC_VALUES (5 variables)
def regression_sc_vs_asc_values(sc_values, asc_value_list):
    X = np.array(asc_value_list)  # Extract ASC values for car, bike, public, walk, maas
    y = np.array(sc_values)
    
    X = sm.add_constant(X)  # Add constant for intercept
    model = sm.OLS(y, X).fit()
    
    print("\nRegression Results for SC vs ASC_VALUES (car, bike, public, walk, maas)")
    print(model.summary())
    
    # Construct the final regression equation
    intercept, *coefficients = model.params
    equation = f"SC = {intercept:.4f}"
    for i, coef in enumerate(coefficients):
        equation += f" + {coef:.4f}x{i+1}"
    # print("\nFinal Regression Equation:", equation)
    
    return model, equation

# 3. Regression: SC vs ASC_VALUE_maas (plot with equation and R-squared)
def regression_sc_vs_asc_maas(sc_values, asc_maas_value):
    """
    Perform regression between System Cost (SC) and ASC_VALUE_MaaS, then plot the result 
    with the regression equation and R-squared value on the graph.
    """
    # Reshape and prepare data
    X = np.array(asc_maas_value).reshape(-1, 1)
    y = np.array(sc_values)
    
    # Add a constant term (for intercept)
    X = sm.add_constant(X)
    
    # Perform the regression
    model = sm.OLS(y, X).fit()
    
    # Predict values for plotting
    X_range = np.linspace(min(asc_maas_value), max(asc_maas_value), 100).reshape(-1, 1)
    X_range = sm.add_constant(X_range)  # Add constant for prediction
    y_pred = model.predict(X_range)
    
    # Extract regression parameters
    intercept, slope = model.params
    r_squared = model.rsquared
    
    # Plot the observed data and regression line
    plt.figure(figsize=(8, 6))
    plt.scatter(asc_maas_value, sc_values, color='blue', label='Simulation Data')
    plt.plot(X_range[:, 1], y_pred, color='red', label='Regression Line')
    
    # Add title and labels
    plt.title('System Cost (SC) vs ASC_VALUE_MaaS')
    plt.xlabel('ASC_VALUE_MaaS')
    plt.ylabel('System Cost (SC)')
    
    # Print the equation and R-squared value on the plot
    equation_text = f'y = {intercept:.4f} + {slope:.4f}x'
    plt.text(min(asc_maas_value), max(sc_values) * 0.9, equation_text, fontsize=12)
    plt.text(min(asc_maas_value), max(sc_values) * 0.85, f'R-squared: {r_squared:.4f}', fontsize=12)
    
    # Add legend and grid
    plt.legend()
    plt.grid(True)
    
    # Show the plot
    plt.show()
    
    # Print the regression results summary in the console
    print("\nRegression Results for SC vs ASC_VALUE_MaaS")
    print(model.summary())
    
    return model

def lowess_sc_vs_asc_maas(sc_values, asc_maas_value, frac=0.3):
    """
    Perform LOWESS fit between System Cost (SC) and ASC_VALUE_MaaS, 
    then plot the result with the LOWESS smoothed curve.
    """
    # Prepare the data
    X = np.array(asc_maas_value)
    y = np.array(sc_values)

    # Perform LOWESS smoothing
    lowess_result = lowess(y, X, frac=frac)

    # Extract the fitted curve
    X_fit = lowess_result[:, 0]
    y_fit = lowess_result[:, 1]

    # Plot the observed data and LOWESS fit
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color='blue', label='Simulation Data')
    plt.plot(X_fit, y_fit, color='red', label='LOWESS Fit')

    # Add title and labels
    plt.title('LOWESS Fit: System Cost (SC) vs ASC_VALUE_MaaS')
    plt.xlabel('ASC_VALUE_MaaS')
    plt.ylabel('System Cost (SC)')

    # Add legend and grid
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

def polynomial_fit_sc_vs_asc_maas(sc_values, asc_maas_value, degree=2):
    """
    Perform Polynomial fit between System Cost (SC) and ASC_VALUE_MaaS, 
    then plot the result with the polynomial curve and equation.
    """
    # Prepare the data
    X = np.array(asc_maas_value)
    y = np.array(sc_values)

    # Perform polynomial fitting
    coefficients = np.polyfit(X, y, degree)
    polynomial = np.poly1d(coefficients)

    # Generate predictions for plotting
    X_range = np.linspace(min(X), max(X), 100)
    y_pred = polynomial(X_range)

    # Plot the observed data and polynomial fit
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color='blue', label='Simulation Data')
    plt.plot(X_range, y_pred, color='red', label=f'Polynomial Fit (degree={degree})')

    # Extract the equation for display
    equation_text = f'y = {" + ".join([f"{coeff:.4f}x^{i}" for i, coeff in enumerate(reversed(coefficients))])}'
    plt.text(min(X), max(y) * 0.85, equation_text, fontsize=10)

    # Add title and labels
    plt.title(f'Polynomial Fit: System Cost (SC) vs ASC_VALUE_MaaS (Degree {degree})')
    plt.xlabel('ASC_VALUE_MaaS')
    plt.ylabel('System Cost (SC)')

    # Add legend and grid
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()


# Function to run the system efficiency simulation and return the System Cost (SC)
def run_simulation(params):
    """
    Run the system efficiency simulation using MobilityModel and calculate SC (System Cost).
    
    This function will initialize the MobilityModel, run the simulation, and calculate SC 
    using the service booking log, travel time, cost, and government subsidy.
    """
    session = Session()

    # Reset the database before running the model
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
    )

    # Initialize the MobilityModel with system efficiency related parameters
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

    # Run the simulation for the number of steps
    model.run_model(params['simulation_steps'])

    # Calculate SC (System Cost) using the service booking log
    VOT_values = params['VALUE_OF_TIME']  # Extract Value of Time (VOT) values from the parameter set
    SC = calculate_system_cost(session, VOT_values)  # Function to calculate SC from booking log
    
    session.close()  # Close the session after running the simulation

    return SC  # Return the calculated System Cost (SC)

# Main function to run simulations and perform regressions
def run_simulation_and_analyze_sc(parameter_sets):
    sc_values = []
    subsidy_data = []
    asc_maas_values = []
    asc_value_list = []  # Collect all ASC values (car, bike, public, walk, maas)

    for i, params in enumerate(parameter_sets):
        print(f"\n=== Running simulation #{i+1} ===")
        
        session = Session()  # Assuming session is initialized for the database connection
        
        # Run the simulation and collect results (in this case SC)
        run_simulation(params)  # This runs the actual simulation
        
        # Calculate SC using service booking log and VOT values
        VOT_values = params['VALUE_OF_TIME']  # Extract VOT values from the parameter set
        SC = calculate_system_cost(session, VOT_values)
        sc_values.append(SC)
        
        # Collect subsidy percentages for regression analysis
        subsidy_data.append([
            params['subsidy_dataset']['low']['bike'],
            params['subsidy_dataset']['low']['car'],
            params['subsidy_dataset']['low']['MaaS_Bundle'],
            params['subsidy_dataset']['middle']['bike'],
            params['subsidy_dataset']['middle']['car'],
            params['subsidy_dataset']['middle']['MaaS_Bundle'],
            params['subsidy_dataset']['high']['bike'],
            params['subsidy_dataset']['high']['car'],
            params['subsidy_dataset']['high']['MaaS_Bundle'],
        ])
        
        # Collect ASC_VALUES for regression analysis
        asc_values = [
            params['ASC_VALUES']['car'],
            params['ASC_VALUES']['bike'],
            params['ASC_VALUES']['public'],
            params['ASC_VALUES']['walk'],
            params['ASC_VALUES']['maas']
        ]
        asc_value_list.append(asc_values)  # Append all ASC values to list
        
        # Collect ASC_VALUE_MaaS for the third analysis
        asc_maas_values.append(params['ASC_VALUES']['maas'])
        
        session.close()  # Close the session after each simulation

    # Ensure all collected data arrays have the same length
    if len(sc_values) != len(subsidy_data) or len(sc_values) != len(asc_value_list) or len(sc_values) != len(asc_maas_values):
        raise ValueError("Mismatch in lengths of SC values, subsidy data, or ASC values.")
    
    return sc_values, subsidy_data, asc_maas_values, asc_value_list




# Placeholder for parameter sets (you can input them here)
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

# 20 different subsidy datasets
parameter_sets_test_ASC = [
    {
        **base_parameters,
        'ASC_VALUES': {
            'car': np.random.uniform(-10, 10),
            'bike': np.random.uniform(-10, 10),
            'public': np.random.uniform(-10, 10),
            'walk': np.random.uniform(-10, 10),
            'maas': np.random.uniform(-10, 10),
            'default': 0
        },
        'subsidy_dataset': {
            'low': {'bike': 0, 'car': 0, 'MaaS_Bundle': 0},
            'middle': {'bike': 0, 'car': 0, 'MaaS_Bundle': 0},
            'high': {'bike': 0, 'car': 0, 'MaaS_Bundle': 0}
        }
    }
    for _ in range(20)  # Generate 20 different parameter sets
]


parameter_sets_test_maas = [
    {
        **base_parameters,
        'ASC_VALUES': {
            'car': 0,
            'bike':0,
            'public': 0,
            'walk': 0,
            'maas': np.random.uniform(-20, 20),
            'default': 0
        },
        'subsidy_dataset': {
            'low': {'bike': 0, 'car': 0, 'MaaS_Bundle': 0},
            'middle': {'bike': 0, 'car': 0, 'MaaS_Bundle': 0},
            'high': {'bike': 0, 'car': 0, 'MaaS_Bundle': 0}
        }
    }
    for _ in range(20)  # Generate 20 different parameter sets
]


parameter_sets_test_subsidy = [
    {
        **base_parameters,
        'ASC_VALUES': {'car': 0, 'bike': 0, 'public': 0, 'walk': 0, 'maas': 0, 'default': 0},
        'subsidy_dataset': {
            'low': {
                'bike': np.random.uniform(0, 1), 
                'car': np.random.uniform(0, 1), 
                'MaaS_Bundle': np.random.uniform(0, 1)
            },
            'middle': {
                'bike': np.random.uniform(0, 1), 
                'car': np.random.uniform(0, 1), 
                'MaaS_Bundle': np.random.uniform(0, 1)
            },
            'high': {
                'bike': np.random.uniform(0, 1), 
                'car': np.random.uniform(0, 1), 
                'MaaS_Bundle': np.random.uniform(0, 1)
            }
        }
    }
    for _ in range(20)  # Generate 20 different parameter sets
]

def get_analysis_result(sc_values, subsidy_data, asc_maas_values, asc_value_list):
    # 1. SC vs 9 Subsidy Variables
    subsidy_model, subsidy_equation = regression_sc_vs_subsidies(sc_values, subsidy_data)
    
    # 2. SC vs ASC_VALUES (car, bike, public, walk, maas)
    asc_model, asc_equation = regression_sc_vs_asc_values(sc_values, asc_value_list)
    
    # 3. SC vs ASC_VALUE_MaaS (plot)
    # regression_sc_vs_asc_maas(sc_values, asc_maas_values)
    lowess_sc_vs_asc_maas(sc_values, asc_maas_values, frac=0.3)
    polynomial_fit_sc_vs_asc_maas(sc_values, asc_maas_values, degree=2)
    
    (' SC vs 9 Subsidy Variables equation:')
    print(subsidy_equation)
    print('SC vs ASC_VALUES equation:')
    print(asc_equation)

# Call the function to run simulations and analyze SC
# sc_values, subsidy_data, asc_maas_values, asc_value_list = run_simulation_and_analyze_sc(parameter_sets_test_ASC)

"""1st time running, uncomment the lines below"""
sc_values, subsidy_data, asc_maas_values, asc_value_list = run_simulation_and_analyze_sc(parameter_sets_test_maas)

with open('simulation_results_efficiency.pkl', 'wb') as f:
    pickle.dump((sc_values, subsidy_data, asc_maas_values, asc_value_list), f)

print("Simulation results saved successfully.")

"""2nd time running onwards, comment the lines above and uncomment the lines below"""
# # Load the saved simulation results
# with open('simulation_results_efficiency.pkl', 'rb') as f:
#     sc_values, subsidy_data, asc_maas_values, asc_value_list = pickle.load(f)

# print("Simulation results loaded successfully.")

# # Now run the analysis using the loaded data

get_analysis_result(sc_values, subsidy_data, asc_maas_values, asc_value_list)