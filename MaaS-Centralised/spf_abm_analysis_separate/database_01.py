from agent_subsidy_pool import SubsidyPoolConfig
DB_CONNECTION_STRING = 'sqlite:///service_provider_database_2.db'
SIMULATION_STEPS = 3
# #####################################Control for commuters#################################
num_commuters = 100
income_weights = [0.5, 0.3, 0.2]# income_levels = ['low', 'middle', 'high']
health_weights = [0.9, 0.1]# health_statuses = ['good', 'poor']
payment_weights = [0.8, 0.2] # payment_schemes = ['PAYG', 'subscription']
disability_weights = [0.15, 0.85] #[is disabled, not]
tech_access_weights = [0.98, 0.02] #[has_access, not


# # Define health statuses and their respective weights


# # Define payment schemes and their respective weights

age_distribution = {
    (18, 25): 0.15,
    (26, 35): 0.175,
    (36, 45): 0.25,
    (46, 55): 0.175,
    (56, 65): 0.15,
    (66, 75): 0.10,
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
    'low': 25,
    'middle': 85,
    'high': 250,
    'default': 65
}


VALUE_OF_TIME = {
    'low': 9.64,
    'middle': 23.7,
    'high': 67.2
}

FLEXIBILITY_ADJUSTMENTS = {
    'low': 1.05,
    'medium': 1.0,  # Assuming medium flexibility doesn't change the base value
    'high': 0.95
}

UTILITY_FUNCTION_BASE_COEFFICIENTS = {
    'beta_C': -0.05,
    'beta_T': -0.06,
    'beta_W': -0.01,
    'beta_A': -0.01,
    'alpha': -0.01,
}


# Dictionary to store ASC values for each mode
ASC_VALUES = {
    'walk': 0,      # Refined value for walking
    'bike': 0,    # Refined value for biking
    'car': 0,     # Refined value for car
    'public': 0,    # Refined value for public transport
    'maas': 10000,      # Refined value for MaaS bundle
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
    'beta_C': -0.05,
    'beta_T': -0.06
}

##################################### Congestion Control #################################################
CONGESTION_ALPHA = 0.15  # Congestion scaling factor
CONGESTION_BETA = 4      # Congestion exponent
# Define capacity and free-flow time for segments
CONGESTION_CAPACITY = 5  # Example capacity for all segments
CONGESTION_T_IJ_FREE_FLOW = 1  # Free-flow travel time for each segment
BACKGROUND_TRAFFIC_AMOUNT = 40
CHANCE_FOR_INSERTING_RANDOM_TRAFFIC = 0.2
####################################Control for MaaS######################################################
DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS = {
    'S_base': 0.08, # Base surcharge (10%)
    'alpha': 0.2,  # Sensitivity coefficient
    'delta': 0.5 # Reduction factor for subscription model

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
daily_config = SubsidyPoolConfig('daily', 1000)  # 1000 units per day

# For weekly subsidy pool (Mon-Fri)
weekly_config = SubsidyPoolConfig('weekly', 5000)  # 5000 units per week

# For monthly subsidy pool
monthly_config = SubsidyPoolConfig('monthly', 20000)  # 20000 units per month
####################################Control for providers#############################################
public_price_table = {
    'train': {'on_peak': 2, 'off_peak': 1.5},
    'bus': {'on_peak': 1, 'off_peak': 0.8}
}

# Sensitivity coefficients for dynamic pricing
ALPHA_VALUES = {
    'UberLike1': 0.5,  # Sensitivity for UberLike1
    'UberLike2': 0.5,  # Sensitivity for UberLike2
    'BikeShare1': 0.5, # Sensitivity for BikeShare1
    'BikeShare2': 0.5  # Sensitivity for BikeShare2
}

# Baseline demand for each service (this could be based on historical data or set as an initial assumption)
BASELINE_DEMANDS = {
    # 'UberLike1': 40,  # Baseline demand for UberLike1
    # 'UberLike2': 35,  # Baseline demand for UberLike2
    # 'BikeShare1': None, # Baseline demand for BikeShare1
    # 'BikeShare2': None  # Baseline demand for BikeShare2
}
UberLike1_cpacity = 20
UberLike1_price = 6

UberLike2_cpacity = 25
UberLike2_price = 6.5


BikeShare1_capacity = 15
BikeShare1_price = 1


BikeShare2_capacity = 12
BikeShare2_price = 1.2






stations = {
    'train': {
        # T1 North Shore, Northern & Western Line
        'T1-1': (5, 5), 'T1-2': (5, 10), 'T1-3': (5, 15), 'T1-4': (5, 20), 'T1-5': (5, 25),
        'T1-6': (10, 30), 'T1-7': (15, 35), 'T1-8': (20, 40), 'T1-9': (25, 45),
        
        # T2 Airport, Inner West & South Line
        'T2-1': (10, 5), 'T2-2': (15, 10), 'T2-3': (20, 15), 'T2-4': (25, 20), 'T2-5': (30, 25),
        'T2-6': (35, 30), 'T2-7': (40, 35), 'T2-8': (45, 40), 'T2-9': (45, 45),
        
        # T3 Bankstown Line
        'T3-1': (20, 5), 'T3-2': (20, 10), 'T3-3': (20, 15), 'T3-4': (20, 20), 'T3-5': (20, 25),
        'T3-6': (25, 30), 'T3-7': (30, 35), 'T3-8': (35, 40), 'T3-9': (40, 45),
        
        # T4 Eastern Suburbs & Illawarra Line
        'T4-1': (15, 5), 'T4-2': (15, 10), 'T4-3': (15, 15), 'T4-4': (15, 20), 'T4-5': (15, 25),
        'T4-6': (20, 30), 'T4-7': (25, 35), 'T4-8': (30, 40), 'T4-9': (35, 45),
        
        # T5 Cumberland Line
        'T5-1': (10, 15), 'T5-2': (10, 20), 'T5-3': (10, 25), 'T5-4': (10, 30), 'T5-5': (10, 35),
        'T5-6': (15, 40), 'T5-7': (20, 45), 'T5-8': (25, 45), 'T5-9': (30, 45),
        
        # T6 Carlingford Line
        'T6-1': (25, 5), 'T6-2': (25, 10), 'T6-3': (25, 15), 'T6-4': (25, 20), 'T6-5': (25, 25),
        'T6-6': (30, 30), 'T6-7': (35, 35), 'T6-8': (40, 40), 'T6-9': (49, 49),
        
        # T7 Olympic Park Line
        'T7-1': (30, 5), 'T7-2': (30, 10), 'T7-3': (30, 15), 'T7-4': (30, 20), 'T7-5': (30, 25),
        'T7-6': (40, 28), 'T7-7': (43, 33), 'T7-8': (45, 37), 'T7-9': (45, 44)
    },
    'bus': {
    'B21': (5, 7), 'B22': (10, 7), 'B23': (15, 7), 'B24': (20, 7), 'B25': (25, 7), 'B26': (30, 7), 'B27': (35, 7),
    'B28': (5, 17), 'B29': (10, 17), 'B30': (15, 17), 'B31': (20, 17), 'B32': (25, 17), 'B33': (30, 17), 'B34': (35, 17),
    'B35': (5, 27), 'B36': (10, 27), 'B37': (15, 27), 'B38': (20, 27), 'B39': (25, 27), 'B40': (30, 27), 'B41': (35, 27),
    'B42': (5, 37), 'B43': (10, 37), 'B44': (15, 37), 'B45': (20, 37), 'B46': (25, 37), 'B47': (30, 37), 'B48': (35, 37),
    # Additional vertical bus stops
    'B49': (6, 5), 'B50': (6, 10), 'B51': (6, 15), 'B52': (6, 20), 'B53': (6, 25), 'B54': (6, 30), 'B55': (6, 35),
    'B56': (16, 5), 'B57': (16, 10), 'B58': (16, 15), 'B59': (16, 20), 'B60': (16, 25), 'B61': (16, 30), 'B62': (16, 35),
    'B63': (26, 5), 'B64': (26, 10), 'B65': (26, 15), 'B66': (26, 20), 'B67': (26, 25), 'B68': (26, 30), 'B69': (26, 35),
    'B70': (36, 5), 'B71': (36, 10), 'B72': (36, 15), 'B73': (36, 20), 'B74': (36, 25), 'B75': (36, 30), 'B76': (36, 35),
    # addtional  bus stops
    'B76': (40, 7), 'B77': (40, 17), 'B78': (40, 27), 'B79':(40, 37), 
    'B80': (6, 40), 'B81': (16, 40 ), 'B82': (26, 40), 'B83':(36, 40),
    'B84':(45, 7), 'B85': (45, 17), 'B86': (45, 27), 'B87': (45, 37),
    'B88': (6, 45), 'B89': (16, 45), 'B90': (26, 45), 'B91':(36, 45),
    }
}

routes = {
    'train': {
        'RT1': ['T1-1', 'T1-2', 'T1-3', 'T1-4', 'T1-5', 'T1-6', 'T1-7', 'T1-8', 'T1-9'],
        'RT2': ['T2-1', 'T2-2', 'T2-3', 'T2-4', 'T2-5', 'T2-6', 'T2-7', 'T2-8', 'T2-9'],
        'RT3': ['T3-1', 'T3-2', 'T3-3', 'T3-4', 'T3-5', 'T3-6', 'T3-7', 'T3-8', 'T3-9'],
        'RT4': ['T4-1', 'T4-2', 'T4-3', 'T4-4', 'T4-5', 'T4-6', 'T4-7', 'T4-8', 'T4-9'],
        'RT5': ['T5-1', 'T5-2', 'T5-3', 'T5-4', 'T5-5', 'T5-6', 'T5-7', 'T5-8', 'T5-9'],
        'RT6': ['T6-1', 'T6-2', 'T6-3', 'T6-4', 'T6-5', 'T6-6', 'T6-7', 'T6-8', 'T6-9'],
        'RT7': ['T7-1', 'T7-2', 'T7-3', 'T7-4', 'T7-5', 'T7-6', 'T7-7', 'T7-8', 'T7-9']
    },
    'bus': {
        'RB6': ['B21', 'B22', 'B23', 'B24', 'B25', 'B26', 'B27', 'B76', 'B84'],
    'RB7': ['B28', 'B29', 'B30', 'B31', 'B32', 'B33', 'B34', 'B77', 'B85'],
    'RB8': ['B35', 'B36', 'B37', 'B38', 'B39', 'B40', 'B41', 'B78', 'B86'],
    'RB9': ['B42', 'B43', 'B44', 'B45', 'B46', 'B47', 'B48', 'B79', 'B87'],
    'RB10': ['B49', 'B50', 'B51', 'B52', 'B53', 'B54', 'B55', 'B80', 'B88'],
    'RB11': ['B56', 'B57', 'B58', 'B59', 'B60', 'B61', 'B62', 'B81', 'B89'],
    'RB12': ['B63', 'B64', 'B65', 'B66', 'B67', 'B68', 'B69', 'B82', 'B90'],
    'RB13': ['B70', 'B71', 'B72', 'B73', 'B74', 'B75', 'B83', 'B91']}
}

transfers = {
    ('B42', 'B55'): 0.6, ('B43', 'T5-5'): 0.61, ('T1-6', 'T5-4'): 0.1, ('T1-3', 'B28'): 0.6, ('T1-1', 'B21'): 0.52,
    ('B49', 'T1-1'): 0.8, ('T2-1', 'B22'): 1, ('T5-1', 'B29'): 0.71, ('T5-3', 'B36'): 0.32, ('T5-5', 'B43'): 0.53,
    ('B81', 'T5-6'): 0.3, ('T1-7', 'B62'): 0.31, ('T4-5', 'B60'): 0.36, ('T4-5', 'B37'): 0.61, ('T4-4', 'B59'): 0.67,('T4-3', 'B58'): 0.69, ('T4-3', 'B30'): 0.25, ('T2-2', 'T4-2'): 0, ('T4-2', 'B57'): 1, ('T2-2', 'B57'): 0.34, ('T4-1', 'B56'): 0.77,('T4-1', 'B23'): 0.25,
    ('T3-1', 'B24'): 0.5, ('T2-3', 'T3-3'): 0.21, ('T2-3', 'B31'):0.85, ('T3-3', 'B31'): 0.48, ('T3-5', 'B38'): 0.28,
    ('T6-1', 'B63'): 0.7, ('T6-1', 'B25'): 0.81, ('T6-2', 'B64'): 0.26, ('T6-3', 'B65'): 0.23, ('T6-3', 'B65'): 0.23, ('T2-4', 'T6-4'): 0.1, ('T2-4', 'B66'): 0.58,('T6-4', 'B66'): 1,('T6-5', 'B67'): 1,('T6-5', 'B39'): 0.25,('T3-6', 'B68'): 1,('T4-7', 'B69'): 0.33,('B46', 'B69'): 0.34, ('T1-9', 'T5-8'): 0.56,('T1-9', 'B90'): 0.44, ('B90', 'T5-8'): 0.32, 
    ('T7-1', 'B26'): 1.5, ('T7-3', 'B33'): 1.2, ('T2-5','T7-5'): 0.12, ('T2-5','B40'): 0.59, ('B40','T7-5'): 0.44, ('B47','T3-7'): 0.13,
    ('B27', 'B70'): 1.2, ('B34', 'B72'): 0.61, ('B41', 'B74'): 0.65, ('T2-6', 'B75'): 0.12, ('T6-7', 'B48'): 0.39, ('T3-8', 'B83'): 0.2, ('T4-9', 'B91'): 0.32,
    ('B79', 'T2-7'): 1.1, ('T7-6', 'B78'): 0.41, ('T6-7', 'B48'): 1.3, ('T7-8', 'B87'): 0.68, ('T2-9', 'T7-9'): 0.06,
     ('B55', 'B42'): 1.2, ('T5-5', 'B43'): 0.31, ('T5-4', 'T1-6'): 0.05, ('B28', 'T1-3'): 0.95, ('B21', 'T1-1'): 0.07,
    ('T1-1', 'B49'): 0.6, ('B22', 'T2-1'): 0.21, ('B29', 'T5-1'): 1.02, ('B36', 'T5-3'): 0.22, ('B43', 'T5-5'): 0.6,
    ('T5-6', 'B81'): 0.3, ('B62', 'T1-7'): 0.41, ('B60', 'T4-5'): 0.52, ('B37', 'T4-5'): 0.36, ('B59', 'T4-4'): 0.24, ('B58', 'T4-3'): 0.36, ('B30', 'T4-3'): 0.26, ('T4-2', 'T2-2'): 0.08, ('B57', 'T4-2'): 1, ('B57', 'T2-2'): 0.6, ('B56', 'T4-1'): 0.32, ('B23', 'T4-1'): 0.69,
    ('B24', 'T3-1'): 0.7, ('T3-3', 'T2-3'): 0.71, ('B31', 'T2-3'): 0.36, ('B31', 'T3-3'): 0.25, ('B38', 'T3-5'): 0.3,
    ('B63', 'T6-1'): 0.6, ('B25', 'T6-1'): 0.31, ('B64', 'T6-2'): 0.24, ('B65', 'T6-3'): 0.47, ('B65', 'T6-3'): 0.59, ('T6-4', 'T2-4'): 0.54, ('B66', 'T2-4'): 0.41, ('B66', 'T6-4'): 0.43, ('B67', 'T6-5'): 1, ('B39', 'T6-5'): 0.47, ('B68', 'T3-6'): 0.11, ('B69', 'T4-7'): 0.45, ('B69', 'B46'): 0.56, ('T5-8', 'T1-9'): 0.47, ('B90', 'T1-9'): 0.2, ('T5-8', 'B90'): 0.3,
    ('B26', 'T7-1'): 1.8, ('B33', 'T7-3'): 1.12, ('T7-5', 'T2-5'): 0.06, ('B40', 'T2-5'): 0.6, ('T7-5', 'B40'): 0.88, ('T3-7', 'B47'): 1.01,
    ('B70', 'B27'): 0.5, ('B72', 'B34'): 0.51, ('B74', 'B41'): 1.3, ('B75', 'T2-6'): 0.28, ('B48', 'T6-7'): 0.96, ('B83', 'T3-8'): 0.45, ('B91', 'T4-9'): 0.23,
    ('T2-7', 'B79'): 0.46, ('B78', 'T7-6'): 0.21, ('B48', 'T6-7'): 0.12, ('B87', 'T7-8'): 0.69, ('T7-9', 'T2-9'): 0.04,
}

