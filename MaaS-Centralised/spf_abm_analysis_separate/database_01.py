from agent_subsidy_pool import SubsidyPoolConfig
DB_CONNECTION_STRING = 'sqlite:///service_provider_database.db'
SIMULATION_STEPS = 144
# #####################################Control for commuters#################################
num_commuters = 130
income_weights = [0.5, 0.3, 0.2]# income_levels = ['low', 'middle', 'high']
health_weights = [0.9, 0.1]# health_statuses = ['good', 'poor']
payment_weights = [0.8, 0.2] # payment_schemes = ['PAYG', 'subscription']
disability_weights = [0.2, 0.8] #[is disabled, not]
tech_access_weights = [0.95, 0.05] #[has_access, not


# # Define health statuses and their respective weights


# # Define payment schemes and their respective weights

age_distribution = {
    (18, 25): 0.2, 
    (26, 35): 0.3, 
    (36, 45): 0.2, 
    (46, 55): 0.15, 
    (56, 65): 0.1, 
    (66, 75): 0.05,
}


############################################Control for mode choice##################################
PENALTY_COEFFICIENTS = {
    'disability_bike_walk': 0.8,
    'age_health_bike_walk': 0.3,
    'no_tech_access_car_bike': 0.1
}
# Disability Penalty for Bike/Walk (0.5): 
# Individuals with disabilities face significant challenges 
# using non-motorized modes of transport such as biking or walking. 
# A high penalty coefficient reflects the substantial additional effort 
# or impossibility for these individuals to use such modes.

#Age/Health Penalty for Bike/Walk (0.2): 
# Older individuals or those with poor health are less likely to use biking or 
# walking due to physical limitations or health concerns. 
# This penalty is lower than the disability penalty but still significant.

#No Tech Access Penalty for Car/Bike (0.1): 
# Access to technology is crucial for using certain modes of transport effectively, 
# especially ride-sharing services and bike rentals. 
# While not as prohibitive as physical disabilities or health issues, 
# the lack of tech access still adds some inconvenience.

AFFORDABILITY_THRESHOLDS = {
    'low': 25, 'middle': 40, 'high': 130
}


VALUE_OF_TIME = {
    'low': 5, 'middle': 10, 'high': 20
}

FLEXIBILITY_ADJUSTMENTS = {
    'low': 1.15, 'medium': 1.0, 'high': 0.85
}

UTILITY_FUNCTION_BASE_COEFFICIENTS = {
    'beta_C': -0.15, 
    'beta_T': -0.09, 
    'beta_W': -0.04, 
    'beta_A': -0.04, 
    'alpha': -0.01
}


# Dictionary to store ASC values for each mode
ASC_VALUES = {
    'walk': 0,      # Refined value for walking
    'bike': 0,    # Refined value for biking
    'car': 0,     # Refined value for car
    'public': 0,    # Refined value for public transport
    'maas': 8,      # Refined value for MaaS bundle
    'default': 0     # Default value for any mode not covered
}





# UTILITY_FUNCTION_BASE_COEFFICIENTS = {
#     'beta_C': -0.03,
#     'beta_T': -0.04,
#     'beta_W': -0.04,
#     'beta_A': -0.02,
#     'alpha': -0.03
# }
        # U_j = (
        #     coefficients['alpha'] +
        #     (coefficients['beta_C'] * price) +
        #     (coefficients['beta_T'] * (time / 6 * value_of_time)) +
        #     (coefficients['beta_W'] * penalty) +
        #     (coefficients['beta_A'] * affordability_adjustment)
        # )


UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS = {
    'beta_C': -0.02,
    'beta_T': -0.09
}

##################################### Congestion Control #################################################
CONGESTION_ALPHA = 0.03  # Congestion scaling factor
CONGESTION_BETA = 1.5      # Congestion exponent
# Define capacity and free-flow time for segments
CONGESTION_CAPACITY = 10  # Example capacity for all segments
CONGESTION_T_IJ_FREE_FLOW = 1.5  # Free-flow travel time for each segment
BACKGROUND_TRAFFIC_AMOUNT = 120
CHANCE_FOR_INSERTING_RANDOM_TRAFFIC = 0.2
####################################Control for MaaS######################################################
DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS = {
    'S_base': 0.02,  # Base surcharge (2%)
            'alpha': 0.10,   # Sensitivity coefficient
            'delta': 0.5  
}


####################################Control of the map############################################
grid_width = 55
grid_height = 55

####################################Government Subsidy##################################
subsidy_dataset = {
    'low': {'bike': 0.50, 'car': 0.5, 'MaaS_Bundle': 0.5},  # 50% subsidy for bikes, 20% for cars, etc.
    'middle': {'bike': 0.5, 'car': 0.5, 'MaaS_Bundle': 0.5},
    'high': {'bike': 0.5, 'car': 0.5, 'MaaS_Bundle': 0.5},
}
#More policy can be applied here, such as subsidy for the disabled ppel to have more access to Uber
# For daily subsidy pool
daily_config = SubsidyPoolConfig('daily', 4000)  # 1000 units per day

# For weekly subsidy pool (Mon-Fri)
weekly_config = SubsidyPoolConfig('weekly', 22000)  # 5000 units per week

# For monthly subsidy pool
monthly_config = SubsidyPoolConfig('monthly', 80000)  # 20000 units per month
####################################Control for providers#############################################
public_price_table = {
    'train': {'on_peak': 3, 'off_peak': 2.6},
    'bus': {'on_peak': 2.4, 'off_peak': 2}
}

# Sensitivity coefficients for dynamic pricing
ALPHA_VALUES = {
    'UberLike1': 0.3,
            'UberLike2': 0.3,
            'BikeShare1': 0.25,
            'BikeShare2': 0.25
}

# Baseline demand for each service (this could be based on historical data or set as an initial assumption)
BASELINE_DEMANDS = {
    # 'UberLike1': 40,  # Baseline demand for UberLike1
    # 'UberLike2': 35,  # Baseline demand for UberLike2
    # 'BikeShare1': None, # Baseline demand for BikeShare1
    # 'BikeShare2': None  # Baseline demand for BikeShare2
}
UberLike1_cpacity = 15
UberLike1_price = 15.5

UberLike2_cpacity = 19
UberLike2_price = 16.5


BikeShare1_capacity = 10
BikeShare1_price = 2.5


BikeShare2_capacity = 12
BikeShare2_price = 3






stations = {
    'train': {
        # T2 Airport, Inner West & South Line (keeping only stations within 30x30 grid)
        'T2-1': (10, 5), 'T2-2': (15, 10), 'T2-3': (20, 15), 'T2-4': (25, 20), 'T2-5': (30, 25),
        
        # T3 Bankstown Line (keeping only stations within 30x30 grid)
        'T3-1': (20, 5), 'T3-2': (20, 10), 'T3-3': (20, 15), 'T3-4': (20, 20), 'T3-5': (20, 25),
        'T3-6': (25, 30),
        
        # T5 Cumberland Line (keeping only stations within 30x30 grid)
        'T5-1': (10, 15), 'T5-2': (10, 20), 'T5-3': (10, 25), 'T5-4': (10, 30),
        
        # T6 Carlingford Line (keeping only stations within 30x30 grid)
        'T6-1': (25, 5), 'T6-2': (25, 10), 'T6-3': (25, 15), 'T6-4': (25, 20), 'T6-5': (25, 25),
        'T6-6': (30, 30)
    },
    'bus': {
        # Horizontal bus routes (keeping only stations within 30x30 grid)
        'B21': (5, 7), 'B22': (10, 7), 'B23': (15, 7), 'B24': (20, 7), 'B25': (25, 7), 'B26': (30, 7),
        'B28': (5, 17), 'B29': (10, 17), 'B30': (15, 17), 'B31': (20, 17), 'B32': (25, 17), 'B33': (30, 17),
        'B35': (5, 27), 'B36': (10, 27), 'B37': (15, 27), 'B38': (20, 27), 'B39': (25, 27), 'B40': (30, 27),
        
        # Vertical bus routes (keeping only stations within 30x30 grid)
        'B49': (6, 5), 'B50': (6, 10), 'B51': (6, 15), 'B52': (6, 20), 'B53': (6, 25), 'B54': (6, 30),
        'B56': (16, 5), 'B57': (16, 10), 'B58': (16, 15), 'B59': (16, 20), 'B60': (16, 25), 'B61': (16, 30),
        'B63': (26, 5), 'B64': (26, 10), 'B65': (26, 15), 'B66': (26, 20), 'B67': (26, 25), 'B68': (26, 30)
    }
}

routes = {
    'train': {
        # Adjusted routes to only include stations within the 30x30 grid
        'RT2': ['T2-1', 'T2-2', 'T2-3', 'T2-4', 'T2-5'],
        'RT3': ['T3-1', 'T3-2', 'T3-3', 'T3-4', 'T3-5', 'T3-6'],
        'RT5': ['T5-1', 'T5-2', 'T5-3', 'T5-4'],
        'RT6': ['T6-1', 'T6-2', 'T6-3', 'T6-4', 'T6-5', 'T6-6']
    },
    'bus': {
        # Adjusted horizontal bus routes
        'RB6': ['B21', 'B22', 'B23', 'B24', 'B25', 'B26'],
        'RB7': ['B28', 'B29', 'B30', 'B31', 'B32', 'B33'],
        'RB8': ['B35', 'B36', 'B37', 'B38', 'B39', 'B40'],
        
        # Adjusted vertical bus routes
        'RB10': ['B49', 'B50', 'B51', 'B52', 'B53', 'B54'],
        'RB11': ['B56', 'B57', 'B58', 'B59', 'B60', 'B61'],
        'RB12': ['B63', 'B64', 'B65', 'B66', 'B67', 'B68']
    }
}

# Keeping only transfers between stations that remain in the grid
transfers = {
    # Bus-to-bus transfers
    ('B42', 'B55'): 0.3,
    
    # Train-bus transfers
    ('T2-1', 'B22'): 0.02, ('T5-1', 'B29'): 0.41, ('T5-3', 'B36'): 0.32, 
    ('T2-2', 'B57'): 0.34, 
    ('T3-1', 'B24'): 0.5, ('T2-3', 'T3-3'): 0.21, ('T2-3', 'B31'): 0.65, ('T3-3', 'B31'): 0.48, ('T3-5', 'B38'): 0.28,
    ('T6-1', 'B63'): 0.7, ('T6-1', 'B25'): 0.31, ('T6-2', 'B64'): 0.36, ('T6-3', 'B65'): 0.23, ('T6-4', 'T2-4'): 0.1,
    ('T2-4', 'B66'): 0.18, ('T6-4', 'B66'): 0.21, ('T6-5', 'B67'): 0.31, ('T6-5', 'B39'): 0.25, ('T3-6', 'B68'): 0.11,
    ('T2-5', 'B40'): 0.39,
    
    # Reverse transfers (same stations, opposite direction)
    ('B22', 'T2-1'): 0.21, ('B29', 'T5-1'): 0.22, ('B36', 'T5-3'): 0.22,
    ('B24', 'T3-1'): 0.17, ('T3-3', 'T2-3'): 0.11, ('B31', 'T2-3'): 0.36, ('B31', 'T3-3'): 0.25, ('B38', 'T3-5'): 0.3,
    ('B63', 'T6-1'): 0.3, ('B25', 'T6-1'): 0.31, ('B64', 'T6-2'): 0.24, ('B65', 'T6-3'): 0.17, ('T6-4', 'T2-4'): 0.24,
    ('B66', 'T2-4'): 0.21, ('B66', 'T6-4'): 0.23, ('B67', 'T6-5'): 0.1, ('B39', 'T6-5'): 0.37, ('B68', 'T3-6'): 0.11,
    ('B40', 'T2-5'): 0.3
}