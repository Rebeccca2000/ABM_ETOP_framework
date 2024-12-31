import numpy as np
import pandas as pd

def calculate_baseline_distribution():
    """
    Define expected fair distribution based on policy goals
    """
    return {
        'low': 0.5,    # 50% target for low income 
        'middle': 0.3, # 30% target for middle income
        'high': 0.2    # 20% target for high income
    }

def calculate_sdi_normalized(subsidy_values, user_counts):
    """
    Calculate normalized Socioeconomic Distribution Index with robust error handling
    
    Args:
        subsidy_values: Dict with income levels as keys and total subsidies as values
        user_counts: Dict with income levels as keys and user counts as values
    """
    try:
        s_total = sum(subsidy_values.values())
        u_total = sum(user_counts.values())
        
        # Debug prints
        print(f"Subsidy values: {subsidy_values}")
        print(f"User counts: {user_counts}")
        print(f"Total subsidies: {s_total}")
        print(f"Total users: {u_total}")
        
        # Early return checks
        if s_total == 0 or u_total == 0:
            print("Early return: total subsidies or users is zero")
            return 0.0
        
        # Calculate weighted ratios with validation
        weighted_ratios = []
        for income_level in subsidy_values.keys():
            s_i = subsidy_values.get(income_level, 0)
            u_i = user_counts.get(income_level, 0)
            
            # Debug prints
            print(f"Income level {income_level}: subsidies={s_i}, users={u_i}")
            
            if u_i > 0:
                population_share = u_i / u_total
                subsidy_share = s_i / s_total if s_total > 0 else 0
                ratio = (subsidy_share/population_share) if population_share > 0 else 0
                weighted_ratio = ratio * population_share
                weighted_ratios.append(weighted_ratio)
                print(f"Added weighted ratio: {weighted_ratio}")
        
        # Handle empty or invalid ratios
        if not weighted_ratios:
            print("No valid weighted ratios calculated")
            return 0.0
        
        ratio_min = min(weighted_ratios)
        ratio_max = max(weighted_ratios)
        
        # Debug prints
        print(f"Weighted ratios: {weighted_ratios}")
        print(f"Min ratio: {ratio_min}")
        print(f"Max ratio: {ratio_max}")
        
        # Handle equal min and max
        if ratio_max == ratio_min:
            print("Max equals min - returning 0")
            return 0.0
        
        numerator = sum(weighted_ratios) - ratio_min
        denominator = ratio_max - ratio_min
        
        # Final validation
        if denominator == 0:
            print("Denominator is zero after calculations")
            return 0.0
            
        result = 1 - (numerator / denominator)
        print(f"Final result: {result}")
        return result
        
    except Exception as e:
        print(f"Error in calculate_sdi_normalized: {str(e)}")
        return 0.0

def calculate_gini(values):
    """
    Calculate Gini coefficient for any distribution
    """
    if not values or all(v == 0 for v in values):
        return 0
        
    array = np.array(list(values))
    if array.shape[0] < 2:
        return 0
    
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return ((2 * np.sum(index * array)) / (n * np.sum(array))) - ((n + 1) / n)

def calculate_income_ratios(subsidy_values, user_counts):
    """
    Calculate subsidy ratios between income groups
    """
    per_user = {}
    for level in subsidy_values:
        if user_counts[level] > 0:
            per_user[level] = subsidy_values[level] / user_counts[level]
        else:
            per_user[level] = 0
            
    ratios = {
        'low_to_high': per_user['low'] / per_user['high'] if per_user['high'] > 0 else 0,
        'low_to_middle': per_user['low'] / per_user['middle'] if per_user['middle'] > 0 else 0
    }
    
    return ratios, per_user

def analyze_temporal_patterns(temporal_data):
    """
    Comprehensive temporal equity analysis
    """
    def get_period(hour):
        if 6 <= hour < 10:
            return 'morning_peak'
        elif 15 <= hour < 19:
            return 'evening_peak'
        return 'off_peak'
    
    patterns = {
        income_level: {
            'peak_ratios': {},
            'hourly_gini': 0,
            'peak_share': 0
        }
        for income_level in ['low', 'middle', 'high']
    }
    
    for income_level in patterns:
        hourly_data = temporal_data[income_level]['hourly_subsidies']
        
        # Calculate metrics by time period
        period_totals = {
            period: sum(
                hourly_data[h] 
                for h in range(24) 
                if get_period(h) == period
            )
            for period in ['morning_peak', 'evening_peak', 'off_peak']
        }
        
        total_subsidy = sum(hourly_data)
        if total_subsidy > 0:
            patterns[income_level]['peak_share'] = (
                (period_totals['morning_peak'] + period_totals['evening_peak']) / 
                total_subsidy
            )
            
        # Calculate hourly Gini
        patterns[income_level]['hourly_gini'] = calculate_gini(hourly_data)
        
        # Calculate peak ratios
        if period_totals['off_peak'] > 0:
            patterns[income_level]['peak_ratios'] = {
                'morning': period_totals['morning_peak'] / period_totals['off_peak'],
                'evening': period_totals['evening_peak'] / period_totals['off_peak']
            }
            
    return patterns

def calculate_efficiency_metrics(subsidy_values, outcomes):
    """
    Calculate outcome-based efficiency metrics
    """
    efficiency = {}
    for level in subsidy_values:
        if subsidy_values[level] > 0:
            efficiency[level] = {
                'co2_per_subsidy': outcomes[level]['co2_reduction'] / subsidy_values[level],
                'trips_per_subsidy': outcomes[level]['completed_trips'] / subsidy_values[level]
            }
    return efficiency