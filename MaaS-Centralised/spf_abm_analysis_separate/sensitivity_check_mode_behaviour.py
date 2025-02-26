from sqlalchemy import create_engine, func, case, and_, or_
from sqlalchemy.orm import sessionmaker
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd
from run_visualisation_03 import MobilityModel
from agent_service_provider_initialisation_03 import reset_database, CommuterInfoLog, ServiceBookingLog
import multiprocessing as mp
import os
from agent_subsidy_pool import SubsidyPoolConfig
import json
import seaborn as sns
from typing import Dict, List, Optional, Any
import multiprocessing as mp
import pickle
from pathlib import Path
from datetime import datetime
from scipy import stats
from scipy.stats import qmc, chi2_contingency, ttest_ind
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sensitivity_check_parameter_config import (
    ParameterTracker, PARAMETER_RANGES, 
    FPS_SUBSIDY_DEFAULTS, PBS_SUBSIDY_RANGES
)
import traceback

SIMULATION_STEPS = 20
STEPS_PER_HOUR = 6
HOURS_PER_DAY = 24
num_commuters = 120

class ModeBehaviorAnalyzer:
    """
    Enhanced analyzer for mode choice behaviors and equity in transportation.
    Combines mode analysis with equity metrics for comprehensive system evaluation.
    """
    def __init__(self, session):
        self.session = session
        self.modes = ['bike', 'car', 'MaaS_Bundle', 'public', 'walk']
        self.income_levels = ['low', 'middle', 'high']
        
    def analyze_mode_shares(self):
        """
        Analyzes mode shares across income levels with robust statistical testing.
        Handles zero frequencies and sparse data appropriately.
        """
        try:
            mode_shares = {}
            statistical_tests = {}
            
            # Calculate mode shares for each income level
            for income_level in self.income_levels:
                mode_choices = self.session.query(
                    ServiceBookingLog.record_company_name,
                    func.count(ServiceBookingLog.request_id).label('count')
                ).join(
                    CommuterInfoLog,
                    ServiceBookingLog.commuter_id == CommuterInfoLog.commuter_id
                ).filter(
                    CommuterInfoLog.income_level == income_level
                ).group_by(
                    ServiceBookingLog.record_company_name
                ).all()
                
                total_trips = sum(choice[1] for choice in mode_choices)
                if total_trips > 0:
                    mode_shares[income_level] = {
                        choice[0]: choice[1]/total_trips 
                        for choice in mode_choices
                    }
                else:
                    mode_shares[income_level] = {mode: 0 for mode in self.modes}

            # Create contingency table with handling for zero frequencies
            contingency_table = self._create_contingency_table(mode_shares)
            
            # Only perform chi-square test if we have sufficient non-zero frequencies
            if np.all(contingency_table > 0):
                try:
                    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                    statistical_tests['chi_square'] = {
                        'statistic': chi2,
                        'p_value': p_value,
                        'dof': dof
                    }
                except Exception as e:
                    print(f"Warning: Chi-square test failed: {e}")
                    statistical_tests['chi_square'] = None
            else:
                # Use Fisher's exact test for small/sparse samples
                try:
                    from scipy.stats import fisher_exact
                    # For 2x2 tables only
                    if contingency_table.shape == (2, 2):
                        statistic, p_value = fisher_exact(contingency_table)
                        statistical_tests['fisher_exact'] = {
                            'statistic': statistic,
                            'p_value': p_value
                        }
                    else:
                        statistical_tests['note'] = "Sample too sparse for statistical testing"
                except Exception as e:
                    print(f"Warning: Fisher's exact test failed: {e}")
                    statistical_tests['fisher_exact'] = None
            
            # Calculate Gini coefficient for mode distribution
            gini_coefficients = {}
            for income_level in mode_shares:
                values = list(mode_shares[income_level].values())
                if any(v > 0 for v in values):  # Only calculate if we have non-zero values
                    gini_coefficients[income_level] = self._calculate_gini(values)
                else:
                    gini_coefficients[income_level] = None
                    
            statistical_tests['gini'] = gini_coefficients
            
            return {
                'mode_shares': mode_shares,
                'statistical_tests': statistical_tests
            }
        
        except Exception as e:
            print(f"Error in analyze_mode_shares: {e}")
            traceback.print_exc()
            return {
                'mode_shares': {},
                'statistical_tests': {}
            }
    def analyze_mode_shifts(self):
        """
        Analyzes patterns of mode shifts with advanced statistical metrics.
        Includes transition probabilities and trigger analysis.
        """
        shift_patterns = {}
        
        for income_level in self.income_levels:
            # Get sequential trips
            trips = self.session.query(
                ServiceBookingLog.commuter_id,
                ServiceBookingLog.record_company_name,
                ServiceBookingLog.start_time,
                ServiceBookingLog.total_price,
                ServiceBookingLog.government_subsidy,
                ServiceBookingLog.total_time
            ).join(
                CommuterInfoLog,
                ServiceBookingLog.commuter_id == CommuterInfoLog.commuter_id
            ).filter(
                CommuterInfoLog.income_level == income_level
            ).order_by(
                ServiceBookingLog.commuter_id,
                ServiceBookingLog.start_time
            ).all()
            
            # Analyze shifts with statistical tests
            shifts = self._analyze_sequential_shifts(trips)
            
            # Add Markov transition matrix
            transition_matrix = self._calculate_transition_matrix(shifts['shift_matrix'])
            
            # Calculate shift trigger significance
            trigger_significance = self._analyze_trigger_significance(shifts['shift_triggers'])
            
            shift_patterns[income_level] = {
                **shifts,
                'transition_matrix': transition_matrix,
                'trigger_significance': trigger_significance
            }
            
        return shift_patterns

    def analyze_price_sensitivity(self):
        """
        Analyzes price elasticity and sensitivity with robust statistical measures.
        Includes confidence intervals and significance testing.
        """
        price_sensitivity = {}
        
        for income_level in self.income_levels:
            # Get price and mode choice data with temporal information
            choices = self.session.query(
                ServiceBookingLog.record_company_name,
                ServiceBookingLog.total_price,
                ServiceBookingLog.government_subsidy,
                ServiceBookingLog.start_time
            ).join(
                CommuterInfoLog,
                ServiceBookingLog.commuter_id == CommuterInfoLog.commuter_id
            ).filter(
                CommuterInfoLog.income_level == income_level
            ).all()
            
            # Calculate elasticity with confidence intervals
            mode_elasticity = {}
            for mode in self.modes:
                mode_choices = [c for c in choices if c.record_company_name == mode]
                if mode_choices:
                    elasticity_results = self._calculate_price_elasticity_with_ci(mode_choices)
                    mode_elasticity[mode] = elasticity_results
            
            price_sensitivity[income_level] = {
                'elasticity': mode_elasticity,
                'temporal_variation': self._analyze_temporal_price_sensitivity(choices)
            }
            
        return price_sensitivity

    def analyze_equity_metrics(self):
        """
        Calculates comprehensive equity metrics including Gini coefficient
        and distributional fairness measures.
        """
        equity_metrics = {}
        
        # Calculate mode access equity
        access_equity = self._calculate_mode_access_equity()
        
        # Calculate subsidy distribution equity
        subsidy_equity = self._calculate_subsidy_equity()
        
        # Calculate temporal equity
        temporal_equity = self._calculate_temporal_equity()
        
        # Combine metrics with statistical significance
        equity_metrics = {
            'access_equity': access_equity,
            'subsidy_equity': subsidy_equity,
            'temporal_equity': temporal_equity,
            'composite_score': self._calculate_composite_equity_score(
                access_equity, subsidy_equity, temporal_equity
            )
        }
        
        return equity_metrics

    def _calculate_gini(self, values):
        """
        Calculate Gini coefficient with proper handling of edge cases.
        """
        if len(values) < 2 or sum(values) == 0:
            return 0
            
        values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(values)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    
    def _calculate_theil_index(self, values):
        """Calculate Theil index for measuring inequality"""
        if not values or sum(values) == 0:
            return 0
            
        values = np.array(values)
        values = values[values > 0]  # Remove zeros
        n = len(values)
        
        if n == 0:
            return 0
            
        mean = np.mean(values)
        theil = np.sum((values / mean) * np.log(values / mean)) / n
        return theil

    def _analyze_sequential_shifts(self, trips):
        """Analyze sequential mode shifts with comprehensive data handling"""
        try:
            # Initialize with all possible modes
            shift_matrix = {mode: {m: 0 for m in self.modes} for mode in self.modes}
            shift_triggers = []
            
            # Group trips by commuter (keep original simple structure)
            commuter_trips = {}
            for trip in trips:
                if trip.commuter_id not in commuter_trips:
                    commuter_trips[trip.commuter_id] = []
                commuter_trips[trip.commuter_id].append(trip)
            
            # Analyze sequential trips
            for commuter_id, trip_list in commuter_trips.items():
                sorted_trips = sorted(trip_list, key=lambda x: x.start_time)
                
                for i in range(len(sorted_trips)-1):
                    from_mode = sorted_trips[i].record_company_name
                    to_mode = sorted_trips[i+1].record_company_name
                    
                    # Keep original direct comparison
                    if from_mode != to_mode:
                        # Safely update matrix
                        if from_mode in shift_matrix and to_mode in shift_matrix[from_mode]:
                            shift_matrix[from_mode][to_mode] += 1
                            
                            # Record triggers using original structure with safe access
                            shift_triggers.append({
                                'from_mode': from_mode,
                                'to_mode': to_mode,
                                'price_change': sorted_trips[i+1].total_price - sorted_trips[i].total_price,
                                'subsidy_change': (sorted_trips[i+1].government_subsidy or 0) - 
                                                (sorted_trips[i].government_subsidy or 0),
                                'time_difference': sorted_trips[i+1].total_time - sorted_trips[i].total_time
                            })
            
            return {
                'shift_matrix': shift_matrix,
                'shift_triggers': shift_triggers
            }
            
        except Exception as e:
            print(f"Error analyzing sequential shifts: {e}")
            return {'shift_matrix': {}, 'shift_triggers': []}
    def _create_contingency_table(self, mode_shares):
        """
        Create contingency table for chi-square testing.
        """
        modes = self.modes
        table = []
        for income_level in self.income_levels:
            row = [mode_shares[income_level].get(mode, 0) for mode in modes]
            table.append(row)
        return np.array(table)

    def _calculate_transition_matrix(self, shift_matrix):
        """
        Calculate Markov transition matrix with statistical properties.
        """
        modes = list(shift_matrix.keys())
        n_modes = len(modes)
        transition_probs = np.zeros((n_modes, n_modes))
        
        for i, from_mode in enumerate(modes):
            total_shifts = sum(shift_matrix[from_mode].values())
            if total_shifts > 0:
                for j, to_mode in enumerate(modes):
                    transition_probs[i,j] = shift_matrix[from_mode].get(to_mode, 0) / total_shifts
                    
        return {
            'probabilities': transition_probs,
            'modes': modes,
            'eigenvalues': np.linalg.eigvals(transition_probs)
        }

    def _analyze_trigger_significance(self, triggers):
        """
        Analyze statistical significance of mode shift triggers.
        """
        if not triggers:
            return None
            
        # Convert triggers to DataFrame for analysis
        df = pd.DataFrame(triggers)
        
        # Perform multiple regression
        X = df[['price_change', 'subsidy_change', 'time_difference']]
        y = pd.get_dummies(df['to_mode'])  # Target variable: choice of new mode
        
        regression_results = {}
        for mode in y.columns:
            model = sm.OLS(y[mode], sm.add_constant(X))
            results = model.fit()
            regression_results[mode] = {
                'coefficients': results.params.to_dict(),
                'p_values': results.pvalues.to_dict(),
                'r_squared': results.rsquared
            }
            
        return regression_results

    def _calculate_price_elasticity_with_ci(self, mode_choices):
        """
        Calculate price elasticity with confidence intervals using bootstrap.
        """
        if len(mode_choices) < 2:
            return None
            
        prices = [choice.total_price for choice in mode_choices]
        quantities = range(len(mode_choices))
        
        # Bootstrap for confidence intervals
        n_bootstrap = 1000
        elasticities = []
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(prices), size=len(prices), replace=True)
            bootstrap_prices = [prices[i] for i in indices]
            bootstrap_quantities = [quantities[i] for i in indices]
            
            if np.std(bootstrap_prices) > 0:
                elasticity = (np.corrcoef(bootstrap_prices, bootstrap_quantities)[0,1] * 
                            (np.mean(prices)/len(mode_choices)))
                elasticities.append(elasticity)
                
        if elasticities:
            return {
                'mean': np.mean(elasticities),
                'ci_lower': np.percentile(elasticities, 2.5),
                'ci_upper': np.percentile(elasticities, 97.5)
            }
        return None

    def _analyze_temporal_price_sensitivity(self, choices):
        """
        Analyze how price sensitivity varies across time periods.
        """
        peak_periods = [(42, 60), (96, 114)]  # Morning and evening peak
        
        def is_peak(time_step):
            time = time_step % 144
            return any(start <= time < end for start, end in peak_periods)
            
        # Separate peak and off-peak choices
        peak_choices = [c for c in choices if is_peak(c.start_time)]
        off_peak_choices = [c for c in choices if not is_peak(c.start_time)]
        
        # Calculate elasticities for each period
        peak_elasticity = self._calculate_price_elasticity_with_ci(peak_choices)
        off_peak_elasticity = self._calculate_price_elasticity_with_ci(off_peak_choices)
        
        # Perform statistical comparison
        if peak_elasticity and off_peak_elasticity:
            t_stat, p_value = ttest_ind(
                [peak_elasticity['mean']], [off_peak_elasticity['mean']]
            )
            
            return {
                'peak_elasticity': peak_elasticity,
                'off_peak_elasticity': off_peak_elasticity,
                'difference_significance': {
                    't_statistic': t_stat,
                    'p_value': p_value
                }
            }
        return None

    def _calculate_mode_access_equity(self):
        """
        Calculate equity in mode accessibility across income groups.
        """
        # Query mode availability by income level
        mode_availability = {}
        for income_level in self.income_levels:
            available_modes = self.session.query(
                func.count(func.distinct(ServiceBookingLog.record_company_name))
            ).join(
                CommuterInfoLog,
                ServiceBookingLog.commuter_id == CommuterInfoLog.commuter_id
            ).filter(
                CommuterInfoLog.income_level == income_level
            ).scalar()
            
            mode_availability[income_level] = available_modes
            
        # Calculate Theil index for mode availability
        total_modes = len(self.modes)
        theil_index = self._calculate_theil_index(
            [count/total_modes for count in mode_availability.values()]
        )
        
        return {
            'availability': mode_availability,
            'theil_index': theil_index
        }

    def _calculate_subsidy_equity(self):
        """
        Calculate equity in subsidy distribution across income groups.
        """
        subsidy_distribution = {}
        for income_level in self.income_levels:
            # Calculate total subsidies and trips
            subsidies = self.session.query(
                func.sum(ServiceBookingLog.government_subsidy).label('total_subsidy'),
                func.count(ServiceBookingLog.request_id).label('total_trips')
            ).join(
                CommuterInfoLog,
                ServiceBookingLog.commuter_id == CommuterInfoLog.commuter_id
            ).filter(
                CommuterInfoLog.income_level == income_level
            ).first()
            
            if subsidies.total_trips > 0:
                subsidy_distribution[income_level] = {
                    'total_subsidy': subsidies.total_subsidy or 0,
                    'subsidy_per_trip': (subsidies.total_subsidy or 0) / subsidies.total_trips
                }
            else:
                subsidy_distribution[income_level] = {
                    'total_subsidy': 0,
                    'subsidy_per_trip': 0
                }
                
        # Calculate Gini coefficient for subsidy distribution
        gini = self._calculate_gini([d['subsidy_per_trip'] for d in subsidy_distribution.values()])
        
        return {
            'distribution': subsidy_distribution,
            'gini_coefficient': gini
        }

    def _calculate_temporal_equity(self):
        """Calculate temporal equity with robust error handling"""
        try:
            # Query temporal data
            temporal_data = self.session.query(
                ServiceBookingLog.start_time,
                CommuterInfoLog.income_level,
                func.count(ServiceBookingLog.request_id).label('trip_count')
            ).join(
                CommuterInfoLog,
                ServiceBookingLog.commuter_id == CommuterInfoLog.commuter_id
            ).group_by(
                ServiceBookingLog.start_time,
                CommuterInfoLog.income_level
            ).all()

            if not temporal_data:
                print("Warning: No temporal data available")
                return {
                    'temporal_gini': 0,
                    'peak_equity': 0,
                    'statistical_tests': {'anova_p_value': None}
                }

            # Convert to DataFrame for easier processing
            df = pd.DataFrame(temporal_data, columns=['time', 'income_level', 'trip_count'])
            
            # Ensure we have data for all income levels
            if len(df['income_level'].unique()) < len(self.income_levels):
                print("Warning: Missing data for some income levels")
                return {
                    'temporal_gini': 0,
                    'peak_equity': 0,
                    'statistical_tests': {'anova_p_value': None}
                }

            # Calculate temporal Gini coefficient
            temporal_gini = self._calculate_gini(df['trip_count'].values)

            # Calculate peak vs off-peak equity
            peak_periods = df[df['time'].between(7*STEPS_PER_HOUR, 9*STEPS_PER_HOUR) | 
                             df['time'].between(16*STEPS_PER_HOUR, 18*STEPS_PER_HOUR)]
            off_peak = df[~df['time'].between(7*STEPS_PER_HOUR, 9*STEPS_PER_HOUR) & 
                         ~df['time'].between(16*STEPS_PER_HOUR, 18*STEPS_PER_HOUR)]

            # Calculate peak equity only if we have data
            if not peak_periods.empty and not off_peak.empty:
                peak_ratio = peak_periods.groupby('income_level')['trip_count'].mean()
                off_peak_ratio = off_peak.groupby('income_level')['trip_count'].mean()
                peak_equity = abs(1 - (peak_ratio / off_peak_ratio).mean())
            else:
                peak_equity = 0

            # Perform statistical test if we have enough data
            try:
                groups = [group['trip_count'].values for _, group in df.groupby('income_level')]
                if all(len(g) > 0 for g in groups):
                    f_stat, p_value = stats.f_oneway(*groups)
                else:
                    p_value = None
            except Exception as e:
                print(f"Warning: ANOVA test failed: {e}")
                p_value = None

            return {
                'temporal_gini': temporal_gini,
                'peak_equity': peak_equity,
                'statistical_tests': {'anova_p_value': p_value}
            }

        except Exception as e:
            print(f"Error in temporal equity calculation: {str(e)}")
            return {
                'temporal_gini': 0,
                'peak_equity': 0,
                'statistical_tests': {'anova_p_value': None}
            }

    def _calculate_composite_equity_score(self, access_equity, subsidy_equity, temporal_equity):
        """
        Calculate a composite equity score combining multiple equity dimensions.
        Uses weighted combination with uncertainty quantification.
        """
        try:
            # Define weights for different components
            weights = {
                'access': 0.4,
                'subsidy': 0.3,
                'temporal': 0.3
            }
            
            # Normalize component scores
            normalized_scores = {
                'access': 1 - (access_equity.get('theil_index', 0) or 0),
                'subsidy': 1 - (subsidy_equity.get('gini_coefficient', 0) or 0),
                'temporal': 1 - (temporal_equity.get('temporal_gini', 0) or 0)
            }
            
            # Calculate weighted score
            composite_score = sum(
                weights[component] * score
                for component, score in normalized_scores.items()
            )
            
            # Calculate uncertainty using bootstrap
            n_bootstrap = 1000
            bootstrap_scores = []
            
            for _ in range(n_bootstrap):
                # Randomly perturb weights while maintaining sum = 1
                perturbed_weights = np.random.dirichlet(
                    [10 * w for w in weights.values()]
                )
                
                bootstrap_score = sum(
                    weight * normalized_scores[component]
                    for component, weight in zip(weights.keys(), perturbed_weights)
                )
                
                bootstrap_scores.append(bootstrap_score)
                
            return {
                'score': composite_score,
                'ci_lower': np.percentile(bootstrap_scores, 2.5),
                'ci_upper': np.percentile(bootstrap_scores, 97.5),
                'component_scores': normalized_scores
            }
        except Exception as e:
            print(f"Error calculating composite score: {str(e)}")
            return {
                'score': 0,
                'ci_lower': 0,
                'ci_upper': 0,
                'component_scores': {
                    'access': 0,
                    'subsidy': 0,
                    'temporal': 0
                }
            }

def create_visualization_suite(results,output_dir):
    """
    Create comprehensive visualization suite for mode behavior analysis.
    
    Args:
        results: Dictionary containing analysis results
        output_dir: Directory to save visualizations
        
    The function creates multiple plots:
    - Mode share analysis
    - Mode shift patterns
    - Price sensitivity analysis
    - Equity metrics
    - Summary dashboard
    """
    try:
        # Use a valid seaborn style
        plt.style.use('fivethirtyeight')  # This is the correct style name
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Validate results
        if not results or not isinstance(results, dict):
            raise ValueError("Invalid results format")
            
        required_keys = ['mode_shares', 'mode_shifts', 'price_sensitivity', 'equity_metrics']
        missing_keys = [key for key in required_keys if key not in results]
        
        if missing_keys:
            raise KeyError(f"Missing required data: {missing_keys}")
        
        # 1. Mode Share Analysis
        try:
            plot_mode_shares(results['mode_shares'], output_dir)
        except Exception as e:
            print(f"Error plotting mode shares: {e}")
            traceback.print_exc()
        
        # 2. Mode Shift Analysis
        try:
            plot_mode_shifts(results['mode_shifts'], output_dir)
        except Exception as e:
            print(f"Error plotting mode shifts: {e}")
            traceback.print_exc()
        
        # 3. Price Sensitivity Analysis
        try:
            plot_price_sensitivity(results['price_sensitivity'], output_dir)
        except Exception as e:
            print(f"Error plotting price sensitivity: {e}")
            traceback.print_exc()
        
        # 4. Equity Analysis
        try:
            plot_equity_metrics(results['equity_metrics'], output_dir)
        except Exception as e:
            print(f"Error plotting equity metrics: {e}")
            traceback.print_exc()
        
        # 5. Create Summary Dashboard
        try:
            create_summary_dashboard(results, output_dir)
        except Exception as e:
            print(f"Error creating summary dashboard: {e}")
            traceback.print_exc()
            
        print(f"Visualization suite completed. Output saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error in visualization suite: {e}")
        traceback.print_exc()

    
def plot_mode_shares(mode_shares, output_dir):
    """Create detailed mode share visualizations with statistical annotations."""
    plt.style.use('fivethirtyeight')
    fig = plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(2, 2)
    
    # 1. Stacked bar chart by income level
    ax1 = fig.add_subplot(gs[0, :])
    if 'mode_shares' in mode_shares:
        data = pd.DataFrame(mode_shares['mode_shares']).T
        data.plot(kind='bar', stacked=True, ax=ax1)
        ax1.set_title('Mode Shares by Income Level')
        ax1.set_ylabel('Share')
        ax1.legend(title='Mode', bbox_to_anchor=(1.05, 1))
    
    # 2. Statistical test results - with error handling
    ax2 = fig.add_subplot(gs[1, 0])
    if 'statistical_tests' in mode_shares and 'chi_square' in mode_shares['statistical_tests']:
        chi2_results = mode_shares['statistical_tests']['chi_square']
        ax2.text(0.1, 0.9, f"Chi-square test:\np-value: {chi2_results['p_value']:.3f}\n" +
                 f"statistic: {chi2_results['statistic']:.3f}")
    else:
        ax2.text(0.1, 0.9, "Statistical tests not available")
    ax2.axis('off')
    
    # 3. Gini coefficients - with error handling
    ax3 = fig.add_subplot(gs[1, 1])
    if 'statistical_tests' in mode_shares and 'gini' in mode_shares['statistical_tests']:
        gini_data = pd.DataFrame(mode_shares['statistical_tests']['gini'], 
                               index=['Gini']).T
        sns.heatmap(gini_data, annot=True, cmap='YlOrRd', ax=ax3)
        ax3.set_title('Gini Coefficients by Income Level')
    else:
        ax3.text(0.5, 0.5, "Gini coefficients not available", 
                ha='center', va='center')
        ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mode_shares.png'))
    plt.close()

def create_summary_dashboard(results, output_dir):
    """Create summary dashboard with robust data validation"""
    try:
        fig = plt.figure(figsize=(20, 15))
        gs = plt.GridSpec(3, 3)
        
        # 1. Mode distribution plot
        ax1 = fig.add_subplot(gs[0, 0])
        mode_data = results.get('mode_shares', {}).get('mode_shares', {})
        if mode_data and all(isinstance(v, (int, float)) for v in mode_data.values()):
            plt.pie(list(mode_data.values()), labels=list(mode_data.keys()), 
                   autopct='%1.1f%%')
            ax1.set_title('Overall Mode Distribution')
        else:
            ax1.text(0.5, 0.5, 'No valid mode data', ha='center')
        
        # 2. Equity metrics
        ax2 = fig.add_subplot(gs[0, 1:])
        equity_data = results.get('equity_metrics', {}).get('composite_score', {}) \
                            .get('component_scores', {})
        if equity_data and all(isinstance(v, (int, float)) for v in equity_data.values()):
            plt.bar(list(equity_data.keys()), list(equity_data.values()))
            ax2.set_title('Equity Component Scores')
            plt.xticks(rotation=45)
        else:
            ax2.text(0.5, 0.5, 'No valid equity data', ha='center')
            
        # 3. Temporal distribution
        ax4 = fig.add_subplot(gs[2, :])
        temporal_data = {}
        if 'equity_metrics' in results and 'temporal_equity' in results['equity_metrics']:
            temp_equity = results['equity_metrics']['temporal_equity'].get('access_by_income', {})
            for income, data in temp_equity.items():
                if isinstance(data, dict) and 'period_shares' in data:
                    shares = data['period_shares']
                    if all(isinstance(v, (int, float)) for v in shares.values()):
                        temporal_data[income] = shares
        
        if temporal_data:
            temp_df = pd.DataFrame(temporal_data)
            temp_df.plot(kind='bar', ax=ax4)
            ax4.set_title('Temporal Distribution by Income Level')
            plt.xticks(rotation=45)
        else:
            ax4.text(0.5, 0.5, 'No valid temporal data', ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'summary_dashboard.png'))
        
    except Exception as e:
        print(f"Error creating dashboard: {e}")
    finally:
        plt.close()

def plot_mode_shifts(mode_shifts, output_dir):
    """Plot mode shift analysis results with error handling"""
    try:
        plt.figure(figsize=(15, 10))
        
        # Plot transition matrix if available
        plt.subplot(211)
        for income_level, data in mode_shifts.items():
            if 'transition_matrix' in data:
                matrix_data = data['transition_matrix']
                modes = ['bike', 'car', 'MaaS_Bundle', 'public', 'walk']
                
                # Create empty transition matrix
                transition_array = np.zeros((len(modes), len(modes)))
                
                # Fill the matrix with available data
                if isinstance(matrix_data, dict):
                    for i, from_mode in enumerate(modes):
                        for j, to_mode in enumerate(modes):
                            if from_mode in matrix_data and isinstance(matrix_data[from_mode], dict):
                                transition_array[i, j] = matrix_data[from_mode].get(to_mode, 0)
                
                # Create DataFrame with consistent dimensions
                transition_data = pd.DataFrame(
                    transition_array,
                    index=modes,
                    columns=modes
                )
                
                # Replace zeros with NaN to avoid formatting warnings
                transition_data = transition_data.replace(0, np.nan)
                
                # Plot heatmap with custom formatting for NaN values
                sns.heatmap(transition_data, 
                           annot=True, 
                           cmap='YlOrRd', 
                           ax=plt.gca(),
                           fmt='.2f')  # Format numbers to 2 decimal places
                plt.title(f'Mode Transition Matrix - {income_level.capitalize()}')
                break  # Just plot one income level for clarity
        
        # Plot trigger analysis if available
        plt.subplot(212)
        for income_level, data in mode_shifts.items():
            if 'trigger_significance' in data:
                trigger_results = data['trigger_significance']
                
                # Convert trigger results to plottable format
                trigger_data = []
                if isinstance(trigger_results, dict):
                    for trigger, value in trigger_results.items():
                        if isinstance(value, (int, float)):
                            trigger_data.append({'trigger': trigger, 'significance': value})
                
                if trigger_data:
                    trigger_df = pd.DataFrame(trigger_data)
                    sns.barplot(data=trigger_df, 
                              x='trigger', 
                              y='significance',
                              ax=plt.gca())
                    plt.title(f'Shift Trigger Significance - {income_level.capitalize()}')
                    plt.xticks(rotation=45)
                    break  # Just plot one income level for clarity
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'mode_shifts.png'))
        plt.close()
        
    except Exception as e:
        print(f"Error plotting mode shifts: {str(e)}")
        traceback.print_exc()
        # Create a simple fallback plot
        try:
            plt.figure(figsize=(10, 5))
            plt.text(0.5, 0.5, 'Mode shift visualization unavailable\nInsufficient data', 
                    ha='center', va='center')
            plt.savefig(os.path.join(output_dir, 'mode_shifts.png'))
            plt.close()
        except:
            print("Could not create fallback visualization")

def plot_price_sensitivity(price_sensitivity, output_dir):
    """Plot price sensitivity analysis results"""
    plt.figure(figsize=(15, 10))
    
    # 1. Elasticity by mode and income
    plt.subplot(211)
    elasticity_data = {}
    for income, data in price_sensitivity.items():
        for mode, elastic in data['elasticity'].items():
            if elastic:
                elasticity_data.setdefault(mode, {})[income] = elastic['mean']
    
    elasticity_df = pd.DataFrame(elasticity_data)
    sns.heatmap(elasticity_df, annot=True, cmap='RdBu', center=0)
    plt.title('Price Elasticity by Mode and Income Level')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'price_sensitivity.png'))
    plt.close()

def plot_equity_metrics(equity_metrics, output_dir):
    """Plot equity analysis results"""
    plt.figure(figsize=(15, 15))
    
    # 1. Access equity
    plt.subplot(221)
    access_data = pd.DataFrame(equity_metrics['access_equity']['availability'], 
                             index=['Modes Available'])
    sns.barplot(data=access_data.T)
    plt.title('Mode Accessibility by Income Level')
    
    # 2. Subsidy equity
    plt.subplot(222)
    subsidy_data = pd.DataFrame({
        income: data['subsidy_per_trip']
        for income, data in equity_metrics['subsidy_equity']['distribution'].items()
    }, index=['Subsidy per Trip'])
    sns.barplot(data=subsidy_data.T)
    plt.title('Subsidy Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'equity_metrics.png'))
    plt.close()
def generate_parameter_sets(analysis_type, base_parameters, num_simulations):
    """
    Generate parameter sets for mode behavior analysis.
    
    Args:
        analysis_type: Either 'FPS' or 'PBS'
        base_parameters: Dictionary of base simulation parameters
        num_simulations: Number of simulation runs to generate parameters for
        
    Returns:
        list: List of parameter dictionaries for each simulation
    """
    parameter_sets = []
    
    if analysis_type == 'FPS':
        # Generate Fixed Pool Subsidy parameter sets
        subsidy_pools = np.linspace(1000, 40000, num_simulations)
        
        for pool_size in subsidy_pools:
            params = base_parameters.copy()
            
            # Set default subsidy percentages
            params['subsidy_dataset'] = {
                'low': {'bike': 0.317, 'car': 0.176, 'MaaS_Bundle': 0.493},
                'middle': {'bike': 0.185, 'car': 0.199, 'MaaS_Bundle': 0.383},
                'high': {'bike': 0.201, 'car': 0.051, 'MaaS_Bundle': 0.297}
            }
            
            # Set subsidy pool configuration
            params['subsidy_config'] = SubsidyPoolConfig('daily', float(pool_size))
            params['_analysis_type'] = 'FPS'
            
            # Vary other parameters randomly within ranges
            params['UTILITY_FUNCTION_BASE_COEFFICIENTS']['beta_C'] = np.random.uniform(-0.08, -0.02)
            params['UTILITY_FUNCTION_BASE_COEFFICIENTS']['beta_T'] = np.random.uniform(-0.08, -0.02)
            params['DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS']['S_base'] = np.random.uniform(0.05, 0.15)
            
            parameter_sets.append(params)
            
    elif analysis_type == 'PBS':
        # Generate Percentage Based Subsidy parameter sets
        modes = ['bike', 'car', 'MaaS_Bundle']
        points_per_mode = num_simulations // len(modes)
        
        subsidy_ranges = {
            'low': {
                'bike': (0.2, 0.45),
                'car': (0.15, 0.35),
                'MaaS_Bundle': (0.3, 0.6)
            },
            'middle': {
                'bike': (0.15, 0.35),
                'car': (0.1, 0.25),
                'MaaS_Bundle': (0.25, 0.5)
            },
            'high': {
                'bike': (0.1, 0.3),
                'car': (0.05, 0.15),
                'MaaS_Bundle': (0.2, 0.4)
            }
        }
        
        for mode in modes:
            for i in range(points_per_mode):
                params = base_parameters.copy()
                
                # Create subsidy configuration
                subsidy_config = {income_level: {} for income_level in ['low', 'middle', 'high']}
                
                # Calculate subsidy percentages
                for income_level in ['low', 'middle', 'high']:
                    for m in modes:
                        if m == mode:
                            # Systematically vary the target mode
                            min_val, max_val = subsidy_ranges[income_level][m]
                            pct = i / (points_per_mode - 1) if points_per_mode > 1 else 0
                            subsidy_config[income_level][m] = min_val + (max_val - min_val) * pct
                        else:
                            # Randomly set other modes within their ranges
                            min_val, max_val = subsidy_ranges[income_level][m]
                            subsidy_config[income_level][m] = np.random.uniform(min_val, max_val)
                
                params['subsidy_dataset'] = subsidy_config
                params['subsidy_config'] = SubsidyPoolConfig('daily', float('inf'))
                params['_analysis_type'] = 'PBS'
                params['varied_mode'] = mode
                
                # Vary other parameters
                params['UTILITY_FUNCTION_BASE_COEFFICIENTS']['beta_C'] = np.random.uniform(-0.08, -0.02)
                params['UTILITY_FUNCTION_BASE_COEFFICIENTS']['beta_T'] = np.random.uniform(-0.08, -0.02)
                params['DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS']['S_base'] = np.random.uniform(0.05, 0.15)
                
                parameter_sets.append(params)
    
    else:
        raise ValueError(f"Invalid analysis type: {analysis_type}")
    
    # Add service provider variations
    for params in parameter_sets:
        # Vary Uber-like service parameters
        params['uber_like1_capacity'] = np.random.randint(6, 11)
        params['uber_like1_price'] = np.random.uniform(4, 8)
        params['uber_like2_capacity'] = np.random.randint(7, 12)
        params['uber_like2_price'] = np.random.uniform(4.5, 8.5)
        
        # Vary bike-share service parameters
        params['bike_share1_capacity'] = np.random.randint(8, 13)
        params['bike_share1_price'] = np.random.uniform(0.8, 1.2)
        params['bike_share2_capacity'] = np.random.randint(10, 15)
        params['bike_share2_price'] = np.random.uniform(1.0, 1.4)
        
        # Add simulation steps
        params['simulation_steps'] = SIMULATION_STEPS
    
    print(f"Generated {len(parameter_sets)} parameter sets for {analysis_type} analysis")
    return parameter_sets

def run_mode_behavior_analysis(analysis_type, 
                             base_parameters,
                             num_simulations,
                             num_cpus):
    """
    Run comprehensive mode behavior analysis with parallel processing.
    
    Args:
        analysis_type: Either 'FPS' or 'PBS'
        base_parameters: Dictionary of simulation parameters
        num_simulations: Number of simulations to run
        num_cpus: Number of CPU cores to use
        
    Returns:
        List of simulation results
        
    The function:
    1. Validates inputs
    2. Generates parameter sets
    3. Runs parallel simulations
    4. Saves results and generates visualizations
    """
    try:
        # Input validation
        if analysis_type not in ['FPS', 'PBS']:
            raise ValueError(f"Invalid analysis type: {analysis_type}")
            
        if num_simulations < 1:
            raise ValueError("num_simulations must be positive")
            
        if num_cpus < 1:
            raise ValueError("num_cpus must be positive")
            
        # Set up logging and output directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f'mode_behavior_{analysis_type.lower()}')
        output_dir.mkdir(parents=True, exist_ok=True)
        
                # Define param_file path early
        param_file = output_dir / f'parameters_{analysis_type.lower()}.json'
        
        print(f"\nStarting {analysis_type} analysis")
        print(f"Number of simulations: {num_simulations}")
        print(f"Number of CPUs: {num_cpus}")
        print(f"Output directory: {output_dir}")
        
        # Generate parameter sets
        try:
            parameter_sets = generate_parameter_sets(analysis_type, base_parameters, num_simulations)
            print(f"Generated {len(parameter_sets)} parameter sets")
            
            if not parameter_sets:
                raise ValueError("No valid parameter sets generated")
        except Exception as e:
            print(f"Error generating parameter sets: {e}")
            traceback.print_exc()
            return []
        
        # Run parallel simulations with progress tracking
        print("\nRunning simulations...")
        try:
            with mp.Pool(processes=num_cpus) as pool:
                results = pool.map(run_single_simulation, parameter_sets)
                
            # Filter and validate results
            valid_results = [r for r in results if r is not None]
            print(f"\nCompleted {len(valid_results)} valid simulations out of {num_simulations} attempted")
            
            if not valid_results:
                raise ValueError("No valid simulation results obtained")
                
        except Exception as e:
            print(f"Error in parallel simulation: {e}")
            traceback.print_exc()
            return []
        
        # Save results
        try:
            output_file = output_dir / f'results_{analysis_type.lower()}.pkl'
            with open(output_file, 'wb') as f:
                pickle.dump(valid_results, f)
            print(f"\nSaved results to: {output_file}")
            
            def tuple_to_string(obj):
                """Convert tuple keys to strings for JSON serialization"""
                if isinstance(obj, dict):
                    return {str(key) if isinstance(key, tuple) else key: tuple_to_string(value) 
                            for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [tuple_to_string(item) for item in obj]
                return obj

            # Use when saving to JSON:
            with open(param_file, 'w') as f:
                json.dump(tuple_to_string({
                    'analysis_type': analysis_type,
                    'num_simulations': num_simulations,
                    'base_parameters': base_parameters
                }), f, indent=4)

            print(f"Saved parameters to: {param_file}")
                
        except Exception as e:
            print(f"Error saving results: {e}")
            traceback.print_exc()
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        for i, result in enumerate(valid_results):
            try:
                vis_dir = output_dir / f'visualization_{i+1}'
                create_visualization_suite(result, str(vis_dir))
            except Exception as e:
                print(f"Error creating visualization for simulation {i+1}: {e}")
                continue
        
        # Create final summary visualization
        try:
            create_aggregated_summary(valid_results, output_dir)
        except Exception as e:
            print(f"Error creating aggregated summary: {e}")
            
        print(f"\nAnalysis complete. Results and visualizations saved to: {output_dir}")
        return valid_results
        
    except Exception as e:
        print(f"Error in mode behavior analysis: {e}")
        traceback.print_exc()
        return []
def create_aggregated_summary(results, output_dir):
    """Creates aggregated summary with comprehensive visualization"""
    try:
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 15))
        
        # Process mode shares with original validation
        mode_shares = []
        for r in results:
            if isinstance(r, dict) and 'mode_shares' in r:
                shares = r.get('mode_shares', {}).get('mode_shares', {})
                if shares and all(isinstance(v, (int, float)) for v in shares.values()):
                    mode_shares.append({k: float(v) for k, v in shares.items()})
        
        if mode_shares:
            # Create DataFrame and plot distributions
            mode_shares_df = pd.DataFrame(mode_shares)
            
            plt.subplot(221)
            mode_shares_df.mean().plot(kind='bar')
            plt.title('Average Mode Shares')
            plt.ylabel('Share')
            
            plt.subplot(222)
            sns.boxplot(data=mode_shares_df)
            plt.title('Mode Share Distribution')
            plt.xticks(rotation=45)
            
            # Plot equity scores with error handling
            plt.subplot(223)
            equity_scores = []
            for r in results:
                try:
                    score = r.get('equity_metrics', {}).get('composite_score', {}).get('score')
                    if isinstance(score, (int, float)):
                        equity_scores.append(score)
                except Exception:
                    continue
                    
            if equity_scores:
                plt.hist(equity_scores, bins=20)
                plt.title('Distribution of Equity Scores')
                plt.xlabel('Equity Score')
                plt.ylabel('Frequency')
            
            # Save statistics with proper DataFrame operations
            summary_stats = {
                'mode_shares': mode_shares_df.describe().to_dict(),
                'equity_scores': {
                    'mean': np.mean(equity_scores) if equity_scores else 0,
                    'std': np.std(equity_scores) if equity_scores else 0,
                    'min': np.min(equity_scores) if equity_scores else 0,
                    'max': np.max(equity_scores) if equity_scores else 0
                }
            }
            
            with open(os.path.join(output_dir, 'summary_statistics.json'), 'w') as f:
                json.dump(summary_stats, f, indent=4)
        else:
            plt.text(0.5, 0.5, 'Insufficient data for visualization', 
                    ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'aggregate_summary.png'))
        
    except Exception as e:
        print(f"Error creating aggregated summary: {e}")
        traceback.print_exc()
    finally:
        plt.close()
def run_single_simulation(params): 
    """This function is referenced but not defined"""
    db_path = f"service_provider_database_{os.getpid()}.db"
    db_connection_string = f"sqlite:///{db_path}"
    
    try:
        engine = create_engine(db_connection_string)
        Session = sessionmaker(bind=engine)
        session = Session()

        # Extract analysis parameters
        analysis_type = params.pop('_analysis_type', None)
        varied_mode = params.pop('varied_mode', None)
        simulation_steps = params.pop('simulation_steps')

        # Initialize model and run simulation
        model = MobilityModel(db_connection_string=db_connection_string, **params)
        model.run_model(simulation_steps)

        # Create analyzer and get results
        analyzer = ModeBehaviorAnalyzer(session)
        results = {
            'mode_shares': analyzer.analyze_mode_shares(),
            'mode_shifts': analyzer.analyze_mode_shifts(),
            'price_sensitivity': analyzer.analyze_price_sensitivity(),
            'equity_metrics': analyzer.analyze_equity_metrics()
        }

        # Add metadata
        results['_analysis_type'] = analysis_type
        results['varied_mode'] = varied_mode

        return results

    except Exception as e:
        print(f"Error in simulation {os.getpid()}: {str(e)}")
        raise
    finally:
        session.close()
        if os.path.exists(db_path):
            os.remove(db_path)

class ModeSensitivityAnalyzer:
    """
    Enhanced analyzer for comprehensive mode choice sensitivity analysis.
    """
    def __init__(self, results, parameter_history):
        self.results = results
        self.params = pd.DataFrame(parameter_history)
        self.metrics = ['mode_shares', 'mode_shifts', 'price_sensitivity']
        
    def analyze_parameter_impacts(self):
        """Analyze how parameter variations impact mode choices"""
        impacts = {}
        
        for param in self.params.columns:
            if not pd.api.types.is_numeric_dtype(self.params[param]):
                continue
                
            param_impacts = {
                'mode_choice_correlation': self._calculate_mode_choice_correlation(param),
                'elasticity': self._calculate_mode_elasticity(param),
                'threshold_effects': self._find_threshold_points(param),
                'temporal_stability': self._analyze_temporal_stability(param)
            }
            impacts[param] = param_impacts
            
        return impacts
        
    def _calculate_mode_choice_correlation(self, param):
        """Calculate correlation between parameter values and mode choices"""
        correlations = {}
        
        for result in self.results:
            mode_shares = result.get('mode_shares', {}).get('mode_shares', {})
            if not mode_shares:
                continue
                
            for mode, share in mode_shares.items():
                if mode not in correlations:
                    correlations[mode] = {'values': [], 'params': []}
                correlations[mode]['values'].append(share)
                correlations[mode]['params'].append(self.params[param].iloc[0])
                
        # Calculate correlations and p-values
        mode_correlations = {}
        for mode, data in correlations.items():
            if len(data['values']) > 1:
                corr, p_value = stats.pearsonr(data['params'], data['values'])
                mode_correlations[mode] = {
                    'correlation': corr,
                    'p_value': p_value
                }
                
        return mode_correlations
        
    def _calculate_mode_elasticity(self, param):
        """Calculate elasticity of mode choice with respect to parameter changes"""
        elasticities = {}
        
        for mode in ['bike', 'car', 'MaaS_Bundle', 'public', 'walk']:
            mode_shares = []
            param_values = []
            
            for result, param_value in zip(self.results, self.params[param]):
                shares = result.get('mode_shares', {}).get('mode_shares', {})
                if mode in shares:
                    mode_shares.append(shares[mode])
                    param_values.append(param_value)
                    
            if len(mode_shares) > 1:
                # Calculate arc elasticity
                delta_share = np.diff(mode_shares)
                delta_param = np.diff(param_values)
                avg_share = (np.array(mode_shares[1:]) + np.array(mode_shares[:-1])) / 2
                avg_param = (np.array(param_values[1:]) + np.array(param_values[:-1])) / 2
                
                elasticity = np.mean((delta_share / delta_param) * (avg_param / avg_share))
                elasticities[mode] = elasticity
                
        return elasticities
        
    def _find_threshold_points(self, param):
        """Identify critical threshold points where mode choice behavior changes"""
        thresholds = {}
        
        for result in self.results:
            mode_shares = result.get('mode_shares', {}).get('mode_shares', {})
            if not mode_shares:
                continue
                
            for mode, share in mode_shares.items():
                if mode not in thresholds:
                    thresholds[mode] = []
                    
                param_value = self.params[param].iloc[0]
                thresholds[mode].append((param_value, share))
                
        # Find points of significant change
        critical_points = {}
        for mode, points in thresholds.items():
            if len(points) < 3:
                continue
                
            points = sorted(points, key=lambda x: x[0])
            param_values, shares = zip(*points)
            
            # Calculate rate of change
            deltas = np.diff(shares) / np.diff(param_values)
            mean_delta = np.mean(deltas)
            std_delta = np.std(deltas)
            
            # Find points where rate of change exceeds 2 standard deviations
            critical_indices = np.where(abs(deltas - mean_delta) > 2 * std_delta)[0]
            critical_points[mode] = [
                {'param_value': param_values[i],
                 'share_value': shares[i],
                 'delta': deltas[i]}
                for i in critical_indices
            ]
            
        return critical_points
        
    def _analyze_temporal_stability(self, param):
        """Analyze how parameter impacts vary over simulation time"""
        temporal_stability = {}
        
        for step in range(SIMULATION_STEPS):
            correlations = {}
            
            for result in self.results:
                mode_shares = result.get('mode_shares', {}).get('mode_shares', {})
                if not mode_shares:
                    continue
                    
                for mode, share in mode_shares.items():
                    if mode not in correlations:
                        correlations[mode] = {'shares': [], 'params': []}
                    correlations[mode]['shares'].append(share)
                    correlations[mode]['params'].append(self.params[param].iloc[0])
                    
            # Calculate stability metrics for this time step
            step_stability = {}
            for mode, data in correlations.items():
                if len(data['shares']) > 1:
                    corr, p_value = stats.pearsonr(data['params'], data['shares'])
                    step_stability[mode] = {
                        'correlation': corr,
                        'p_value': p_value,
                        'variance': np.var(data['shares'])
                    }
                    
            temporal_stability[step] = step_stability
            
        return temporal_stability

if __name__ == "__main__":
    # Define base parameters
    base_parameters = {
        'num_commuters': num_commuters,
        'grid_width': 55,
        'grid_height': 55,
        'data_income_weights': [0.5, 0.3, 0.2],
        'data_health_weights': [0.9, 0.1],
        'data_payment_weights': [0.8, 0.2],
        'data_age_distribution': {
            (18, 25): 0.2, (26, 35): 0.3, (36, 45): 0.2,
            (46, 55): 0.15, (56, 65): 0.1, (66, 75): 0.05
        },
        'data_disability_weights': [0.2, 0.8],
        'data_tech_access_weights': [0.95, 0.05],
        'CHANCE_FOR_INSERTING_RANDOM_TRAFFIC': 0.2,
        'ASC_VALUES': {
            'car': 100, 'bike': 100, 'public': 100, 
            'walk': 100, 'maas': 100, 'default': 0
        },
        'UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS': {
            'beta_C': -0.05,
            'beta_T': -0.06
        },
        'UTILITY_FUNCTION_BASE_COEFFICIENTS': {
            'beta_C': -0.05, 'beta_T': -0.06, 
            'beta_W': -0.01, 'beta_A': -0.01, 'alpha': -0.01
        },
        'PENALTY_COEFFICIENTS': {
            'disability_bike_walk': 0.8,
            'age_health_bike_walk': 0.3,
            'no_tech_access_car_bike': 0.1
        },
        'AFFORDABILITY_THRESHOLDS': {'low': 25, 'middle': 85, 'high': 250},
        'FLEXIBILITY_ADJUSTMENTS': {'low': 1.05, 'medium': 1.0, 'high': 0.95},
        'VALUE_OF_TIME': {'low': 9.64, 'middle': 23.7, 'high': 67.2},
        'public_price_table': {
            'train': {'on_peak': 2, 'off_peak': 1.5},
            'bus': {'on_peak': 1, 'off_peak': 0.8}
        },
        'ALPHA_VALUES': {
            'UberLike1': 0.5,
            'UberLike2': 0.5,
            'BikeShare1': 0.5,
            'BikeShare2': 0.5
        },
        'DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS': {
            'S_base': 0.08,
            'alpha': 0.2,
            'delta': 0.5
        },
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

    # Run analysis for both FPS and PBS
    fps_results = run_mode_behavior_analysis('FPS', base_parameters, 8, 8)
    # pbs_results = run_mode_behavior_analysis('PBS', base_parameters, 6, 8)
    
    print("\nAnalysis complete for both FPS and PBS approaches")