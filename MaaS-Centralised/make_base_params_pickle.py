import os, pickle

output_dir = "mae_equity_results_20260217_185340"

base_parameters = {
    'num_commuters': 200,
    'grid_width': 100,
    'grid_height': 100,
    'data_income_weights': [0.5, 0.3, 0.2],
    'data_health_weights': [0.9, 0.1],
    'data_payment_weights': [0.8, 0.2],
    'data_age_distribution': {(18, 25): 0.2, (26, 35): 0.3, (36, 45): 0.2,
                              (46, 55): 0.15, (56, 65): 0.1, (66, 75): 0.05},
    'data_disability_weights': [0.2, 0.8],
    'data_tech_access_weights': [0.95, 0.05],
    'ASC_VALUES': {'car': 0, 'bike': 0, 'public': 0, 'walk': 0, 'maas':0, 'default': 0},
    'UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS': {'beta_C': -0.02, 'beta_T': -0.09},
    'UTILITY_FUNCTION_BASE_COEFFICIENTS': {'beta_C': -0.15, 'beta_T': -0.09, 'beta_W': -0.04, 'beta_A': -0.04, 'alpha': -0.01},
    'PENALTY_COEFFICIENTS': {'disability_bike_walk': 0.8, 'age_health_bike_walk': 0.3, 'no_tech_access_car_bike': 0.1},
    'AFFORDABILITY_THRESHOLDS': {'low': 25, 'middle': 40, 'high': 130},
    'FLEXIBILITY_ADJUSTMENTS': {'low': 1.15, 'medium': 1.0, 'high': 0.85},
    'VALUE_OF_TIME': {'low': 5, 'middle': 10, 'high': 20},
    'public_price_table': {'train': {'on_peak': 3, 'off_peak': 2.6}, 'bus': {'on_peak': 2.4, 'off_peak': 2}},
    'ALPHA_VALUES': {'UberLike1': 0.3, 'UberLike2': 0.3, 'BikeShare1': 0.25, 'BikeShare2': 0.25},
    'DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS': {'S_base': 0.02, 'alpha': 0.10, 'delta': 0.5},
    'BACKGROUND_TRAFFIC_AMOUNT': 200,
    'CONGESTION_ALPHA': 0.03,
    'CONGESTION_BETA': 1.5,
    'CONGESTION_CAPACITY': 10,
    'CONGESTION_T_IJ_FREE_FLOW': 1.5,
    'uber_like1_capacity': 15,
    'uber_like1_price': 15.5,
    'uber_like2_capacity': 19,
    'uber_like2_price': 16.5,
    'bike_share1_capacity': 10,
    'bike_share1_price': 2.5,
    'bike_share2_capacity': 12,
    'bike_share2_price': 3
}

os.makedirs(output_dir, exist_ok=True)
path = os.path.join(output_dir, "base_parameters.pkl")
with open(path, "wb") as f:
    pickle.dump(base_parameters, f)

print("Wrote:", path)
