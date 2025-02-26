"""
Comprehensive Sensitivity Pattern Analysis for MaaS System

This module provides end-to-end analysis of parameter impacts on mode choice behavior:
- Generates parameters for different focus groups (subsidy, utility, service, etc.)
- Runs simulations with those parameters
- Analyzes patterns and relationships
- Creates visualizations and statistical summaries
- Saves organized results and tracking information
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
from scipy import stats
import multiprocessing as mp
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from agent_service_provider_initialisation_03 import reset_database, CommuterInfoLog, ServiceBookingLog
import pickle
import os
import traceback
from typing import Dict, List, Optional
from collections import defaultdict
# Import configurations and required components
from sensitivity_check_parameter_config import (
    PARAMETER_RANGES, FPS_SUBSIDY_DEFAULTS, PBS_SUBSIDY_RANGES, ParameterTracker
)
from run_visualisation_03 import MobilityModel
from agent_service_provider_initialisation_03 import reset_database
from agent_subsidy_pool import SubsidyPoolConfig
from sqlalchemy import func
# Constants
SIMULATION_STEPS = 20
NUM_SIMULATIONS = 30  # Number of simulations per parameter group
NUM_CPUS = 8

class ParameterFocusAnalyzer:
    """
    Handles parameter generation, simulation running, and analysis for different focus groups.
    """
    def __init__(self, base_parameters: dict, output_dir: str = 'sensitivity_focus_analysis'):
        self.base_parameters = base_parameters
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Store results during analysis
        self.current_analysis_results = {}
        # Define valid parameter groups and their corresponding ranges
        self.parameter_groups = {
            'utility': PARAMETER_RANGES['utility'],
            'service': PARAMETER_RANGES['service'],
            'maas': PARAMETER_RANGES['maas'],
            'value_of_time': PARAMETER_RANGES['value_of_time'],
            'public_transport': PARAMETER_RANGES['public_transport'],
            'congestion': PARAMETER_RANGES['congestion'],
            'subsidies': {'pool_size': (1000, 40000)}  # Add specific range for subsidies
        }
        
    

    def _generate_focused_parameters(self, focus_group: str, analysis_type: str, num_simulations: int) -> List[Dict]:
        """
        Generate parameter combinations for sensitivity analysis.
        
        This function handles all parameter generation cases:
        1. FPS subsidy analysis (preserve working logic)
        2. PBS subsidy analysis 
        3. Other parameter groups with systematic variation
        
        Args:
            focus_group: Parameter group to analyze ('subsidies', 'utility', etc.)
            analysis_type: Type of analysis ('FPS' or 'PBS')
            num_simulations: Number of parameter sets to generate
            
        Returns:
            List of parameter dictionaries for simulations
        """
        parameters = []
        
        # Define required default values that all parameter sets should have
        required_defaults = {
            'public_price_table': {
                'train': {'on_peak': 2, 'off_peak': 1.5},
                'bus': {'on_peak': 1, 'off_peak': 0.8}
            },
            'DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS': {
                'S_base': 0.08, 'alpha': 0.2, 'delta': 0.5
            },
            'uber_like1_capacity': 8,
            'uber_like1_price': 6,
            'uber_like2_capacity': 9,
            'uber_like2_price': 6.5,
            'bike_share1_capacity': 10,
            'bike_share1_price': 1,
            'bike_share2_capacity': 12,
            'bike_share2_price': 1.2
        }
        
        # Define parameter ranges for each focus group
        parameter_ranges = {
            'utility': {
                'beta_C': (-0.08, -0.02),
                'beta_T': (-0.08, -0.02),
                'beta_W': (-0.02, 0),
                'beta_A': (-0.02, 0),
                'alpha': (-0.02, 0)
            },
            'service': {
                'uber_like1_capacity': (6, 11),
                'uber_like1_price': (4, 8),
                'uber_like2_capacity': (7, 12),
                'uber_like2_price': (4.5, 8.5),
                'bike_share1_capacity': (8, 13),
                'bike_share1_price': (0.8, 1.2),
                'bike_share2_capacity': (10, 15),
                'bike_share2_price': (1.0, 1.4)
            },
            'maas': {
                'S_base': (0.05, 0.15),
                'alpha': (0.1, 0.3),
                'delta': (0.3, 0.7)
            },
            'public_transport': {
                'train_peak': (1.5, 3.0),
                'train_offpeak': (0.8, 2.0),
                'bus_peak': (1.5, 3.0),
                'bus_offpeak': (0.8, 2.0)
            },
            'value_of_time': {
                'low': (7, 12),
                'middle': (18, 28),
                'high': (50, 80)
            },
            'congestion': {
                'CONGESTION_ALPHA': (0.2, 0.3),
                'CONGESTION_BETA': (3, 5),
                'CONGESTION_CAPACITY': (3, 5),
                'CONGESTION_T_IJ_FREE_FLOW': (1.5, 2.5)
            }
        }

        # Handle FPS subsidy analysis (preserve existing working logic)
        if focus_group == 'subsidies' and analysis_type == 'FPS':
            subsidy_pools = np.linspace(1000, 40000, num_simulations)
            for pool_size in subsidy_pools:
                params = self.base_parameters.copy()
                
                # Add required defaults
                for field, default in required_defaults.items():
                    if field not in params:
                        params[field] = default.copy() if isinstance(default, dict) else default
                        
                # Set up subsidy configuration (preserve working logic)
                params['subsidy_dataset'] = FPS_SUBSIDY_DEFAULTS
                params['subsidy_config'] = SubsidyPoolConfig('daily', float(pool_size))
                params['_analysis_type'] = 'FPS'
                params['varied_mode'] = focus_group
                params['simulation_steps'] = SIMULATION_STEPS
                parameters.append(params)
        
        # Handle all other parameter groups with systematic variation
        else:
            for i in range(num_simulations):
                params = self.base_parameters.copy()
                
                # Add required defaults
                for field, default in required_defaults.items():
                    if field not in params:
                        params[field] = default.copy() if isinstance(default, dict) else default
                
                # For non-subsidy analysis, use fixed middle-value subsidy pool
                if focus_group != 'subsidies':
                    params['subsidy_dataset'] = FPS_SUBSIDY_DEFAULTS
                    params['subsidy_config'] = SubsidyPoolConfig('daily', 20000.0)
                
                # Generate systematically varying parameters based on focus group
                if focus_group in parameter_ranges:
                    ranges = parameter_ranges[focus_group]
                    
                    if focus_group == 'utility':
                        # Generate parameter sets for utility coefficients
                        coefficients = ['beta_T', 'beta_C', 'beta_W', 'beta_A', 'alpha']
                        points_per_coeff = num_simulations // len(coefficients)  # Ensure even distribution
                        parameters = []
                        
                        for coefficient in coefficients:
                            ranges = parameter_ranges[focus_group]
                            min_val, max_val = ranges[coefficient]
                            
                            # Generate evenly spaced values for this coefficient
                            values = np.linspace(min_val, max_val, points_per_coeff)
                            
                            for value in values:
                                params = self.base_parameters.copy()
                                
                                # Add required defaults
                                for field, default in required_defaults.items():
                                    if field not in params:
                                        params[field] = default.copy() if isinstance(default, dict) else default

                                # Start with base utility coefficients
                                params['UTILITY_FUNCTION_BASE_COEFFICIENTS'] = {
                                    'beta_T': -0.06,
                                    'beta_C': -0.05,
                                    'beta_W': -0.01,
                                    'beta_A': -0.01,
                                    'alpha': -0.01
                                }

                                # Only modify the coefficient we're analyzing
                                params['UTILITY_FUNCTION_BASE_COEFFICIENTS'][coefficient] = value
                                
                                # Add metadata for tracking
                                params['_analysis_type'] = analysis_type
                                params['varied_mode'] = focus_group
                                params['_varied_coefficient'] = coefficient
                                params['simulation_steps'] = SIMULATION_STEPS
                                
                                # Use fixed subsidy pool
                                params['subsidy_dataset'] = FPS_SUBSIDY_DEFAULTS
                                params['subsidy_config'] = SubsidyPoolConfig('daily', 20000.0)
                                
                                parameters.append(params)

                        print(f"Generated {len(parameters)} parameter sets for utility analysis")
                        print(f"Number of variations per coefficient: {num_simulations}")
                            
                    elif focus_group == 'maas':
                        for param, (min_val, max_val) in ranges.items():
                            values = np.linspace(min_val, max_val, num_simulations)
                            params['DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS'][param] = values[i]
                            
                    elif focus_group == 'public_transport':
                        for mode in ['train', 'bus']:
                            peak_range = ranges[f'{mode}_peak']
                            offpeak_range = ranges[f'{mode}_offpeak']
                            peak_values = np.linspace(peak_range[0], peak_range[1], num_simulations)
                            offpeak_values = np.linspace(offpeak_range[0], offpeak_range[1], num_simulations)
                            params['public_price_table'][mode]['on_peak'] = peak_values[i]
                            params['public_price_table'][mode]['off_peak'] = offpeak_values[i]
                            
                    elif focus_group == 'value_of_time':
                        for income, (min_val, max_val) in ranges.items():
                            values = np.linspace(min_val, max_val, num_simulations)
                            params['VALUE_OF_TIME'][income] = values[i]
                            
                    else:  # service and congestion parameters
                        for param, (min_val, max_val) in ranges.items():
                            values = np.linspace(min_val, max_val, num_simulations)
                            if 'capacity' in param:
                                values = values.astype(int)
                            params[param] = values[i]
                
                params['_analysis_type'] = analysis_type
                params['varied_mode'] = focus_group
                params['simulation_steps'] = SIMULATION_STEPS
                parameters.append(params)

        # Debug information
        print(f"Generated {len(parameters)} parameter sets for {focus_group} analysis")
        if focus_group == 'subsidies':
            print("Subsidy pool values:", [p['subsidy_config'].total_amount for p in parameters])

        return parameters

    def run_focus_analysis(self, analysis_type: str, base_parameters: dict):
        """Run analysis focused on different parameter groups."""
        all_results = {}
        varied_mode = base_parameters.get('varied_mode', 'all')
        analysis_groups = list(self.parameter_groups.keys()) if varied_mode == 'all' else [varied_mode]

        for group in analysis_groups:
            print(f"\nAnalyzing {group} parameters...")
            
            # Generate parameters
            param_sets = self._generate_focused_parameters(group, analysis_type, NUM_SIMULATIONS)
            print(f"Generated {len(param_sets)} parameter sets for {group} analysis")
            
            # Store the raw parameter sets before running simulations
            self.current_analysis_results = {
                'parameters': param_sets,
                'simulation_results': [],
                'varied_mode': group
            }
            
            # Create and use parameter tracker with simulation IDs
            tracker = ParameterTracker(f"mode_pattern_{group}")
            for sim_idx, params in enumerate(param_sets):
                # Add simulation ID when recording parameters
                tracker.record_parameters(params, simulation_id=sim_idx)
            
            # Run simulations
            simulation_results = self._run_simulations(param_sets)
            
            if simulation_results:
                # Update results while keeping the parameter data
                self.current_analysis_results['simulation_results'] = simulation_results
                self.current_analysis_results['analysis'] = self._analyze_results(simulation_results, group)
                
                # Store in overall results
                all_results[group] = self.current_analysis_results
                
                # Create visualizations
                self._create_visualizations(self.current_analysis_results, group)
                
                print(f"\n{group.capitalize()} Parameter Group:")
                print(f"- Number of simulations: {len(simulation_results)}")
                print(f"- Analysis metrics computed: {list(self.current_analysis_results['analysis'].keys())}")
                
                # Save parameter history
                tracker.save_parameter_history()

        return all_results
    
    def _calculate_parameter_impacts(self, param_values, mode_choices, group):
        """Calculate impact of parameter variations on mode choices"""
        impacts = {}
        
        for param, values in param_values.items():
            if len(values) < 2:  # Skip if not enough variation
                continue
                
            # Calculate correlation
            correlation = stats.pearsonr(values, mode_choices)[0]
            
            # Calculate elasticity
            param_pct_change = np.diff(values) / values[:-1]
            choice_pct_change = np.diff(mode_choices) / mode_choices[:-1]
            elasticity = np.mean(choice_pct_change / param_pct_change)
            
            impacts[param] = {
                'correlation': correlation,
                'elasticity': elasticity
            }
        
        return impacts

    def _calculate_service_impacts(self, param_variations, results, group):
        """Calculate impact on service utilization"""
        service_impacts = {}
        
        for service in ['uber', 'bike', 'maas']:
            service_impacts[service] = {
                'utilization': np.mean([r.get('service_utilization', {}).get(service, 0) 
                                    for r in results]),
                'elasticity': {}
            }
            
            # Calculate elasticity for each parameter
            for param, values in param_variations.items():
                if len(values) < 2:
                    continue
                    
                utilization = [r.get('service_utilization', {}).get(service, 0) 
                            for r in results]
                
                param_pct_change = np.diff(values) / values[:-1]
                util_pct_change = np.diff(utilization) / utilization[:-1]
                
                elasticity = np.mean(util_pct_change / param_pct_change)
                service_impacts[service]['elasticity'][param] = elasticity
                
        return service_impacts
    
    def _analyze_mode_patterns(self, results: List[Dict]) -> Dict:
        """
        Analyze mode choice patterns across simulations.
        
        This function examines:
        - Mode share distributions by income level
        - Mode shift patterns over time
        - Mode choice elasticities with respect to parameters
        """
        patterns = {}
        
        # Analyze mode shares
        mode_shares = self._calculate_mode_shares(results)
        patterns['mode_shares'] = mode_shares
        
        # Analyze mode transitions
        mode_transitions = self._calculate_mode_transitions(results)
        patterns['transitions'] = mode_transitions
        
        # Calculate elasticities
        elasticities = self._calculate_mode_elasticities(results)
        patterns['elasticities'] = elasticities
        
        return patterns
    def _calculate_mode_shares(self, results):
        """
        Calculate the distribution of mode choices across different simulations.
        
        This method analyzes how different transportation modes (bike, car, MaaS, etc.)
        are being used by aggregating data from service booking logs. It breaks down
        the analysis by income level to understand equity aspects of mode choices.
        
        Args:
            results: List of simulation results containing service booking data
            
        Returns:
            Dictionary containing mode shares by income level and overall statistics
        """
        mode_shares = {
            'low': {},
            'middle': {},
            'high': {}
        }
        
        try:
            for result in results:
                if 'mode_shares' in result:
                    # Aggregate mode shares from individual simulations
                    for income_level in ['low', 'middle', 'high']:
                        if income_level not in mode_shares:
                            mode_shares[income_level] = {}
                            
                        shares = result['mode_shares'].get(income_level, {})
                        for mode, share in shares.items():
                            if mode not in mode_shares[income_level]:
                                mode_shares[income_level][mode] = []
                            mode_shares[income_level][mode].append(share)
            
            # Calculate statistics for each mode and income level
            statistics = {}
            for income_level in mode_shares:
                statistics[income_level] = {}
                for mode in mode_shares[income_level]:
                    shares = mode_shares[income_level][mode]
                    if shares:
                        statistics[income_level][mode] = {
                            'mean': np.mean(shares),
                            'std': np.std(shares),
                            'min': np.min(shares),
                            'max': np.max(shares)
                        }
            
            return {
                'raw_shares': mode_shares,
                'statistics': statistics,
                'income_levels': ['low', 'middle', 'high'],
                'modes': list(set(mode for income_data in mode_shares.values() 
                                for mode in income_data.keys()))
            }
            
        except Exception as e:
            print(f"Error calculating mode shares: {str(e)}")
            traceback.print_exc()
            return {
                'raw_shares': mode_shares,
                'statistics': {},
                'income_levels': ['low', 'middle', 'high'],
                'modes': []
            }
    def _calculate_sensitivities(self, results: List[Dict], group: str) -> Dict:
        """Calculate sensitivity metrics with proper structure handling"""
        sensitivities = {}
        param_variations = self._extract_parameter_variations(group)
        
        # Calculate mode choice sensitivities using mode_shares instead
        mode_sensitivities = {}
        for mode in ['bike', 'car', 'MaaS_Bundle', 'public', 'walk']:
            mode_impacts = self._calculate_parameter_impacts(
                param_variations,
                [r.get('mode_shares', {}).get('middle', {}).get(mode, 0) for r in results],
                group
            )
            mode_sensitivities[mode] = mode_impacts
        
        sensitivities['mode_choice'] = mode_sensitivities
        
        # Calculate service utilization sensitivities
        service_impacts = self._calculate_service_impacts(param_variations, results, group)
        sensitivities['service'] = service_impacts
        
        return sensitivities

    def _analyze_temporal_patterns(self, results: List[Dict]) -> Dict:
        """
        Analyze temporal patterns in mode choices and system performance.
        
        This examines:
        - Peak vs off-peak behavior
        - Time-of-day variations
        - Temporal stability of patterns
        """
        temporal = {}
        
        # Analyze peak period patterns
        peak_patterns = self._analyze_peak_patterns(results)
        temporal['peak'] = peak_patterns
        
        # Analyze time-of-day variations
        time_variations = self._analyze_time_variations(results)
        temporal['time_of_day'] = time_variations
        
        # Calculate temporal stability metrics
        stability = self._calculate_temporal_stability(results)
        temporal['stability'] = stability
        
        return temporal
    def _calculate_temporal_stability(self, results):
        """Calculate temporal stability metrics"""
        stability = {}
        modes = ['bike', 'car', 'MaaS_Bundle', 'public', 'walk']
        
        for r in results:
            temporal_data = r.get('temporal_data', {})
            
            # Calculate coefficient of variation for each mode
            for mode in modes:
                if mode not in stability:
                    stability[mode] = []
                    
                mode_shares = []
                for period in ['morning_peak', 'evening_peak', 'off_peak']:
                    counts = temporal_data.get(period, {}).get('mode_counts', {})
                    total = sum(counts.values())
                    if total > 0:
                        mode_shares.append(counts.get(mode, 0) / total)
                        
                if mode_shares:
                    cv = np.std(mode_shares) / np.mean(mode_shares) if np.mean(mode_shares) > 0 else 0
                    stability[mode].append(cv)
        
        return {mode: np.mean(cvs) if cvs else 0 for mode, cvs in stability.items()}
    def _analyze_time_variations(self, results):
        """Analyze time-of-day variations in mode choices"""
        variations = {}
        
        for r in results:
            temporal_data = r.get('temporal_data', {})
            for hour in range(24):
                if hour not in variations:
                    variations[hour] = {'mode_shares': defaultdict(list)}
                
                # Convert simulation steps to hours (6 steps per hour)
                step_range = range(hour * 6, (hour + 1) * 6)
                mode_counts = defaultdict(int)
                
                # Aggregate counts for this hour
                for step in step_range:
                    step_data = temporal_data.get(step, {}).get('mode_counts', {})
                    for mode, count in step_data.items():
                        mode_counts[mode] += count
                
                # Calculate shares
                total = sum(mode_counts.values())
                if total > 0:
                    for mode, count in mode_counts.items():
                        variations[hour]['mode_shares'][mode].append(count/total)
        
        return variations
    
    
    def _run_simulations(self, parameters: List[Dict]) -> List[Dict]:
        """Run parallel simulations with given parameters"""
        print(f"\nRunning {len(parameters)} simulations...")
        
        with mp.Pool(processes=NUM_CPUS) as pool:
            results = pool.map(self.run_single_simulation, parameters)
            
        valid_results = [r for r in results if r is not None]
        print(f"Completed {len(valid_results)} valid simulations")
        
        return valid_results

    def _analyze_results(self, results, group):
        """
        Analyze simulation results for a specific parameter group.
        """
        analysis = {}
        
        try:
            # Calculate mode choice patterns
            mode_patterns = self._calculate_mode_shares(results)
            analysis['mode_patterns'] = mode_patterns
            
            # Add other analyses as needed
            if hasattr(self, '_calculate_sensitivities'):
                analysis['sensitivities'] = self._calculate_sensitivities(results, group)
                
            if hasattr(self, '_analyze_temporal_patterns'):
                analysis['temporal'] = self._analyze_temporal_patterns(results)
                
        except Exception as e:
            print(f"Error in analysis for group {group}: {str(e)}")
            traceback.print_exc()
        
        return analysis
    def _analyze_peak_patterns(self, results):
        """Analyze peak vs off-peak patterns"""
        patterns = {}
        
        for r in results:
            temporal_data = r.get('temporal_data', {})
            for period in ['morning_peak', 'evening_peak', 'off_peak']:
                if period not in patterns:
                    patterns[period] = []
                mode_counts = temporal_data.get(period, {}).get('mode_counts', {})
                total = sum(mode_counts.values())
                if total > 0:
                    patterns[period].append({mode: count/total 
                                        for mode, count in mode_counts.items()})
        
        return patterns

    def _plot_mode_choice_evolution(self, results: Dict, vis_dir: Path):
        """
        Create small multiples visualization of mode share evolution by parameter.
        Handles both FPS subsidy analysis and utility coefficient analysis.
        """
        try:
            vis_dir.mkdir(parents=True, exist_ok=True)
            
            # Get parameter group and simulation results
            param_group = results.get('varied_mode', 'all')
            sim_results = results.get('simulation_results', [])

            # Special handling for utility analysis
            if param_group == 'utility':
                for coeff in ['beta_T', 'beta_C', 'beta_W', 'beta_A', 'alpha']:
                    plt.figure(figsize=(15, 12))
                    
                    # Get all unique parameter values
                    param_values = []
                    for params in results.get('parameters', []):
                        if params.get('_varied_coefficient') == coeff:  # Only get values for current coefficient
                            param_values.append(params['UTILITY_FUNCTION_BASE_COEFFICIENTS'].get(coeff, 0))
                    
                    modes = ['WALK', 'PUBLIC', 'MAAS', 'BIKE', 'CAR']
                    for idx, mode in enumerate(modes):
                        plt.subplot(3, 2, idx+1 if idx < 4 else 5)
                        
                        for income_level in ['low', 'middle', 'high']:
                            # Get mode shares only for current coefficient variations
                            mode_shares = []
                            for result, params in zip(sim_results, results.get('parameters', [])):
                                if params.get('_varied_coefficient') == coeff:
                                    share = result.get('mode_shares', {}).get(income_level, {}).get(mode.lower(), 0)
                                    mode_shares.append(share)
                            
                            if mode_shares and param_values:
                                color = {'low': '#2563eb', 'middle': '#059669', 'high': '#dc2626'}[income_level]
                                
                                # Plot scatter points
                                plt.scatter(param_values, mode_shares, 
                                        color=color, alpha=0.4, label=f'{income_level.capitalize()} Income')
                                
                                # Add trend line
                                if len(param_values) > 2:
                                    z = np.polyfit(param_values, mode_shares, 2)
                                    p = np.poly1d(z)
                                    x_smooth = np.linspace(min(param_values), max(param_values), 100)
                                    plt.plot(x_smooth, p(x_smooth), '-', color=color, alpha=0.7)
                        
                        plt.title(f'{mode} Mode Share')
                        plt.xlabel(f'{coeff} Value')
                        plt.ylabel('Mode Share')
                        plt.grid(True, alpha=0.3)
                        plt.ylim(0, 1)
                        plt.legend()
                    
                    plt.suptitle(f'Mode Share Evolution by {coeff}', y=1.02, fontsize=14)
                    plt.tight_layout()
                    plt.savefig(vis_dir / f'mode_shares_{coeff}.png', bbox_inches='tight', dpi=300)
                    plt.close()
            elif param_group == 'subsidy':
                # Original FPS analysis code
                param_variations = self._extract_parameter_variations(param_group)
                if not param_variations:
                    print(f"No valid parameter variations for {param_group}")
                    return

                param_name = self._get_primary_parameter(param_group)
                param_values = param_variations.get(param_name, [])
                if not param_values:
                    print(f"No parameter values for {param_group} - {param_name}")
                    return

                # Rest of your original FPS plotting code...
                modes = ['walk', 'public', 'maas', 'bike', 'car']
                fig = plt.figure(figsize=(15, 12))
                gs = plt.GridSpec(3, 2, figure=fig)
                
                for idx, mode in enumerate(modes):
                    ax = fig.add_subplot(gs[idx // 2, idx % 2])
                    
                    for income_level in ['low', 'middle', 'high']:
                        mode_shares = []
                        for result in sim_results:
                            shares = result.get('mode_shares', {}).get(income_level, {})
                            mode_shares.append(shares.get(mode, 0))
                        
                        color = {'low': '#2563eb', 'middle': '#059669', 'high': '#dc2626'}[income_level]
                        ax.plot(param_values, mode_shares, 'o-',
                            label=f'{income_level.capitalize()} Income',
                            color=color, linewidth=2, alpha=0.7)
                        
                        z = np.polyfit(param_values, mode_shares, 2)
                        p = np.poly1d(z)
                        x_smooth = np.linspace(min(param_values), max(param_values), 100)
                        ax.plot(x_smooth, p(x_smooth), '--', color=color, alpha=0.3)
                    
                    ax.set_title(f'{mode.upper()} Mode Share')
                    ax.set_xlabel(self._get_parameter_label(param_group, param_name))
                    ax.set_ylabel('Mode Share')
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    ax.set_ylim(0, 0.8)
                
                plt.suptitle(f'Mode Share Evolution by {param_group.capitalize()}', y=1.02, fontsize=14)
                plt.tight_layout()
                plt.savefig(vis_dir / f'mode_shares_{param_group}.png',
                        bbox_inches='tight', dpi=300)
                plt.close()

        except Exception as e:
            print(f"Error plotting mode choice evolution: {e}")
            traceback.print_exc()

    def _get_primary_parameter(self, group: str) -> str:
        """Get the primary parameter to plot for each group"""
        return {
            'utility': 'beta_T',
            'service': 'uber_like1_price',
            'maas': 'S_base',
            'value_of_time': 'vot_middle',
            'public_transport': 'train_on_peak',
            'congestion': 'CONGESTION_ALPHA',
            'subsidies': 'pool_size'
        }.get(group, '')

    def _get_parameter_label(self, group: str, param_name: str) -> str:
        """Get display label for parameter based on analysis type"""
        labels = {
            'utility': {
                'beta_T': 'Time Coefficient',
                'beta_C': 'Cost Coefficient',
                'beta_W': 'Wait Time Coefficient',
                'beta_A': 'Access Time Coefficient'
            },
            'service': {
                'uber_price': 'Car Service Price',
                'bike_price': 'Bike Share Price',
                'uber_capacity': 'Car Service Capacity',
                'bike_capacity': 'Bike Share Capacity'
            },
            'maas': {
                'S_base': 'Base Surcharge',
                'alpha': 'Dynamic Pricing Factor',
                'delta': 'Subscription Discount'
            },
            'value_of_time': {
                'vot_low': 'Low Income VOT',
                'vot_middle': 'Middle Income VOT',
                'vot_high': 'High Income VOT'
            },
            'public_transport': {
                'train_on_peak': 'Train Peak Price',
                'bus_on_peak': 'Bus Peak Price',
                'train_off_peak': 'Train Off-Peak Price',
                'bus_off_peak': 'Bus Off-Peak Price'
            },
            'congestion': {
                'CONGESTION_ALPHA': 'Congestion Sensitivity',
                'CONGESTION_BETA': 'Congestion Factor',
                'CONGESTION_CAPACITY': 'Road Capacity',
                'CONGESTION_T_IJ_FREE_FLOW': 'Free Flow Time'
            },
            'subsidies': {
                'pool_size': 'Subsidy Pool Size'
            }
        }
        
        return labels.get(group, {}).get(param_name, param_name)

    def _plot_parameter_impact(self, param_values, mode_shares, label, income_level):
        """
        Helper function to plot impact of a specific parameter on mode shares.
        
        Args:
            param_values: List of parameter values that were varied
            mode_shares: Dictionary mapping modes to their share values
            label: String label for the parameter being analyzed
            income_level: Income level being analyzed ('low', 'middle', 'high')
        """
        try:
            
            print(f"\nPlotting parameter impact:")
            print(f"Parameter values: {param_values}")
            print(f"Number of modes with data: {sum(1 for shares in mode_shares.values() if shares)}")

            if not param_values or not mode_shares:
                print("No data to plot")
                return

            param_values = np.array(param_values)
            sort_idx = np.argsort(param_values)
            param_values = param_values[sort_idx]

            # Plot each mode's shares
            for mode, shares in mode_shares.items():
                if shares and len(shares) == len(param_values):
                    # Sort shares according to parameter values
                    shares = np.array(shares)[sort_idx]
                    
                    # Plot with different styles for different modes
                    if 'MaaS' in mode:
                        plt.plot(param_values, shares, 'o-', 
                                label=mode, linewidth=2.5, color='green', alpha=0.8)
                    elif 'Uber' in mode:
                        plt.plot(param_values, shares, 's-',
                                label=mode, linewidth=2, alpha=0.7)
                    elif 'Bike' in mode:
                        plt.plot(param_values, shares, '^-',
                                label=mode, linewidth=2, alpha=0.7)
                    elif mode == 'public':
                        plt.plot(param_values, shares, 'D-',
                                label=mode, linewidth=2, color='red', alpha=0.7)
                    else:
                        plt.plot(param_values, shares, 'o-',
                                label=mode, linewidth=2, alpha=0.7)

                    # Add trend line for clearer pattern visualization
                    z = np.polyfit(param_values, shares, 2)
                    p = np.poly1d(z)
                    x_smooth = np.linspace(param_values.min(), param_values.max(), 100)
                    plt.plot(x_smooth, p(x_smooth), '--', color='gray', alpha=0.3)

            # Customize plot appearance
            plt.xlabel(label)
            plt.ylabel('Mode Share')
            plt.title(f'Mode Share Evolution - {income_level.capitalize()} Income')
            plt.grid(True, alpha=0.3)
            
            # Add legend with better positioning and transparency
            plt.legend(bbox_to_anchor=(1.05, 1), 
                    loc='upper left', 
                    framealpha=0.9,
                    borderaxespad=0.)
            
            # Add annotations for key points
            for mode, shares in mode_shares.items():
                if shares and len(shares) == len(param_values):
                    shares = np.array(shares)[sort_idx]
                    max_idx = np.argmax(shares)
                    if shares[max_idx] > 0.1:  # Only annotate significant shares
                        plt.annotate(f'{shares[max_idx]:.2f}',
                                xy=(param_values[max_idx], shares[max_idx]),
                                xytext=(5, 5), textcoords='offset points',
                                fontsize=8, alpha=0.7)

            # Set y-axis limits with some padding
            plt.ylim(0, max(max(shares) for shares in mode_shares.values() if shares) * 1.1)
            
            # Adjust layout to prevent label cutoff
            plt.tight_layout()

        except Exception as e:
            print(f"Error in parameter impact plotting: {e}")
            traceback.print_exc()
    def _plot_parameter_interaction_effects(self, param_variations, mode_shares, vis_dir):
        """
        Create visualizations showing how parameter interactions affect mode shares.
        """
        try:
            plt.figure(figsize=(15, 15))
            param_names = list(param_variations.keys())
            
            if len(param_names) >= 2:
                # Select first two parameters for 2D visualization
                param1, param2 = param_names[:2]
                
                for idx, income_level in enumerate(['low', 'middle', 'high']):
                    plt.subplot(3, 1, idx + 1)
                    
                    # Create scatter plot colored by dominant mode
                    x_values = param_variations[param1]
                    y_values = param_variations[param2]
                    
                    # Get dominant mode for each point
                    dominant_modes = []
                    colors = []
                    for i, r in enumerate(mode_shares):
                        if isinstance(r, dict) and 'mode_shares' in r:
                            shares = r['mode_shares'].get(income_level, {})
                            if shares:
                                dominant_mode = max(shares.items(), key=lambda x: x[1])[0]
                                dominant_modes.append(dominant_mode)
                    
                    # Create scatter plot
                    unique_modes = list(set(dominant_modes))
                    color_map = dict(zip(unique_modes, sns.color_palette(n_colors=len(unique_modes))))
                    
                    for mode in unique_modes:
                        mask = [m == mode for m in dominant_modes]
                        plt.scatter(
                            [x_values[i] for i, m in enumerate(mask) if m],
                            [y_values[i] for i, m in enumerate(mask) if m],
                            label=mode,
                            alpha=0.7
                        )
                    
                    plt.title(f'Parameter Interaction Effects - {income_level.capitalize()} Income')
                    plt.xlabel(param1)
                    plt.ylabel(param2)
                    plt.legend()
                    plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(vis_dir / 'parameter_interaction_effects.png')
            plt.close()
            
        except Exception as e:
            print(f"Error plotting parameter interactions: {e}")
            traceback.print_exc()

    def _plot_parameter_impacts(self, results: Dict, vis_dir: Path):
        """
        Create comprehensive visualizations showing how different parameters impact mode choices.
        
        This method analyzes and visualizes:
        1. Direct parameter effects on mode choices for each income level
        2. Parameter sensitivity analysis
        3. Comparative impact strength across different parameters
        
        Args:
            results: Dictionary containing simulation results and parameter variations
            vis_dir: Path to directory for saving visualizations
        """
        try:
            # Ensure visualization directory exists
            vis_dir.mkdir(parents=True, exist_ok=True)
            
            # Get parameter group and simulation results
            param_group = results.get('varied_mode', 'all')
            analysis_type = results.get('_analysis_type', 'FPS')
            sim_results = results.get('simulation_results', [])
            params_list = results.get('parameters', [])
            if not isinstance(sim_results, list):
                sim_results = [sim_results]
            
            if not sim_results:
                print("No simulation results available for parameter impact analysis")
                return
                    
            # Extract parameter variations based on parameter group
            param_variations = self._extract_parameter_variations(param_group)
            
            # Define consistent sets of modes and income levels
            modes = ['walk', 'public', 'maas', 'bike', 'car']
            income_levels = ['low', 'middle', 'high']
            
            # Create figure for parameter impacts
            plt.figure(figsize=(15, 10))

            # Handle different parameter groups
            if param_group == 'utility':
                # Create figure with subplots for each utility parameter
                utility_params = ['beta_T', 'beta_C', 'beta_W', 'beta_A', 'alpha']
                fig, axes = plt.subplots(len(utility_params), 1, figsize=(12, 4*len(utility_params)))
                
                for idx, param in enumerate(utility_params):
                    ax = axes[idx] if len(utility_params) > 1 else axes
                    
                    # Extract parameter values from the parameters list
                    param_values = [
                        params['UTILITY_FUNCTION_BASE_COEFFICIENTS'][param]
                        for params in params_list
                    ]
                    
                    # Plot for each income level
                    for income_level in ['low', 'middle', 'high']:
                        mode_shares = {mode: [] for mode in ['walk', 'public', 'maas', 'bike', 'car']}
                        
                        # Collect mode shares for each simulation result
                        for result in sim_results:
                            shares = result.get('mode_shares', {}).get(income_level, {})
                            for mode in mode_shares:
                                mode_shares[mode].append(shares.get(mode, 0))
                        
                        # Calculate total shares
                        total_shares = [sum(shares) for shares in zip(*mode_shares.values())]
                        
                        # Plot with different colors for each income level
                        color = {'low': '#2563eb', 'middle': '#059669', 'high': '#dc2626'}[income_level]
                        ax.scatter(param_values, total_shares, 
                                label=f'{income_level.capitalize()} Income',
                                color=color, alpha=0.6)
                        
                        # Add trend line if we have enough points
                        if len(param_values) > 2:
                            z = np.polyfit(param_values, total_shares, 2)
                            p = np.poly1d(z)
                            x_smooth = np.linspace(min(param_values), max(param_values), 100)
                            ax.plot(x_smooth, p(x_smooth), '--', color=color, alpha=0.3)
                    
                    ax.set_xlabel(f'Utility Coefficient ({param})')
                    ax.set_ylabel('Total Mode Share')
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    ax.set_ylim(0, 1)  # Set y-axis limits for better visualization
                
                plt.suptitle('Utility Parameter Impacts on Mode Choice')
                plt.tight_layout()
                

            elif param_group == 'service':
                # Service parameters analysis
                service_params = ['uber_price', 'bike_price']
                plt.suptitle('Service Price Impacts on Mode Choice', size=14)
                
                for idx, param in enumerate(service_params, 1):
                    plt.subplot(1, 2, idx)
                    param_values = param_variations.get(param, [])
                    
                    for mode in modes:
                        shares = []
                        for result in sim_results:
                            # Average across income levels
                            avg_share = np.mean([
                                result.get('mode_shares', {}).get(inc, {}).get(mode, 0)
                                for inc in income_levels
                            ])
                            shares.append(avg_share)
                        
                        if shares and len(shares) == len(param_values):
                            plt.plot(param_values, shares, 'o-',
                                label=mode.capitalize(),
                                linewidth=2, alpha=0.7)
                            
                            # Add trend line
                            z = np.polyfit(param_values, shares, 2)
                            p = np.poly1d(z)
                            x_smooth = np.linspace(min(param_values), max(param_values), 100)
                            plt.plot(x_smooth, p(x_smooth), '--', alpha=0.3)
                    
                    plt.xlabel(f'Service Price ({param})')
                    plt.ylabel('Mode Share')
                    plt.grid(True, alpha=0.3)
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.ylim(0, 1)

            elif param_group == 'maas':
                # MaaS parameters analysis
                maas_params = ['S_base', 'alpha', 'delta']
                plt.suptitle('MaaS Parameter Impacts on Mode Choice', size=14)
                
                for idx, param in enumerate(maas_params, 1):
                    plt.subplot(1, 3, idx)
                    param_values = param_variations.get(param, [])
                    
                    for income_level in income_levels:
                        shares = []
                        for result in sim_results:
                            mode_shares = result.get('mode_shares', {}).get(income_level, {})
                            maas_share = mode_shares.get('maas', 0)
                            shares.append(maas_share)
                        
                        if shares and len(shares) == len(param_values):
                            plt.plot(param_values, shares, 'o-',
                                label=f'{income_level.capitalize()} Income',
                                linewidth=2, alpha=0.7)
                            
                            # Add trend line
                            z = np.polyfit(param_values, shares, 2)
                            p = np.poly1d(z)
                            x_smooth = np.linspace(min(param_values), max(param_values), 100)
                            plt.plot(x_smooth, p(x_smooth), '--', alpha=0.3)
                    
                    plt.xlabel(f'MaaS Parameter ({param})')
                    plt.ylabel('MaaS Share')
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    plt.ylim(0, 1)

            elif param_group == 'value_of_time':
                # Value of time analysis
                plt.suptitle('Value of Time Impacts on Mode Choice', size=14)
                
                for idx, income_level in enumerate(income_levels, 1):
                    plt.subplot(1, 3, idx)
                    param_values = param_variations.get(f'vot_{income_level}', [])
                    
                    for mode in modes:
                        shares = []
                        for result in sim_results:
                            mode_shares = result.get('mode_shares', {}).get(income_level, {})
                            share = mode_shares.get(mode, 0)
                            shares.append(share)
                        
                        if shares and len(shares) == len(param_values):
                            plt.plot(param_values, shares, 'o-',
                                label=mode.capitalize(),
                                linewidth=2, alpha=0.7)
                            
                            # Add trend line
                            z = np.polyfit(param_values, shares, 2)
                            p = np.poly1d(z)
                            x_smooth = np.linspace(min(param_values), max(param_values), 100)
                            plt.plot(x_smooth, p(x_smooth), '--', alpha=0.3)
                    
                    plt.xlabel(f'Value of Time ({income_level.capitalize()})')
                    plt.ylabel('Mode Share')
                    plt.grid(True, alpha=0.3)
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.ylim(0, 1)

            elif param_group == 'subsidies':
                # Previous subsidies code remains the same
                plt.suptitle(f'Subsidy Parameter Impacts on Mode Choice', size=14)
                
                if analysis_type == 'FPS':
                    param_name = 'pool_size'
                    param_values = param_variations.get(param_name, [])
                    x_label = 'Subsidy Pool Size'
                else:  # PBS
                    param_values = []
                    for mode in ['bike', 'car', 'MaaS_Bundle']:
                        for income_level in income_levels:
                            key = f'{mode.lower()}_subsidy_{income_level}'
                            values = param_variations.get(key, [])
                            if values:
                                param_values.extend(values)
                    x_label = 'Subsidy Percentage'
                
                for idx, income_level in enumerate(income_levels, 1):
                    plt.subplot(1, 3, idx)
                    
                    for mode in modes:
                        shares = []
                        for result in sim_results:
                            mode_shares = result.get('mode_shares', {}).get(income_level, {})
                            share = mode_shares.get(mode, 0)
                            shares.append(share)
                        
                        if shares and len(shares) == len(param_values):
                            plt.plot(param_values, shares, 'o-',
                                label=mode.capitalize(),
                                linewidth=2, alpha=0.7)
                            
                            # Add trend line
                            z = np.polyfit(param_values, shares, 2)
                            p = np.poly1d(z)
                            x_smooth = np.linspace(min(param_values), max(param_values), 100)
                            plt.plot(x_smooth, p(x_smooth), '--', alpha=0.3)
                    
                    plt.xlabel(x_label)
                    plt.ylabel('Mode Share')
                    plt.title(f'{income_level.capitalize()} Income')
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    plt.ylim(0, 1)

            # Save the figure
            plt.tight_layout()
            plt.savefig(vis_dir / f'parameter_impacts_{param_group}.png', 
                    bbox_inches='tight', dpi=300)
            plt.close()

        except Exception as e:
            print(f"Error plotting parameter impacts: {e}")
            traceback.print_exc()
    def _plot_temporal_patterns(self, results: Dict, vis_dir: Path):
        """
        Create visualizations showing temporal patterns.
        """
        temporal = results['analysis']['temporal']
        
        # Plot peak vs off-peak patterns
        plt.figure(figsize=(12, 6))
        peak_patterns = temporal['peak']
        
        plt.subplot(1, 2, 1)
        self._plot_peak_distributions(peak_patterns)
        plt.title('Peak vs Off-peak Distribution')
        
        plt.subplot(1, 2, 2)
        self._plot_peak_transitions(peak_patterns)
        plt.title('Peak Period Transitions')
        
        plt.tight_layout()
        plt.savefig(vis_dir / 'temporal_patterns.png')
        plt.close()
        
        # Plot time-of-day variations
        self._plot_time_variations(temporal['time_of_day'], vis_dir)

    def _plot_peak_distributions(self, peak_patterns):
        """Plot distribution of mode shares during peak/off-peak periods"""
        for period, data in peak_patterns.items():
            if data:  # Check if we have data
                df = pd.DataFrame(data)
                sns.boxplot(data=df)
                plt.title(f'Mode Distribution - {period}')
                plt.xticks(rotation=45)
                plt.ylabel('Mode Share')

    def _plot_peak_transitions(self, peak_patterns):
        """Plot transitions between peak and off-peak periods"""
        periods = ['morning_peak', 'evening_peak', 'off_peak']
        modes = set()
        for period_data in peak_patterns.values():
            for pattern in period_data:
                modes.update(pattern.keys())
                
        transition_matrix = np.zeros((len(modes), len(periods)))
        for i, mode in enumerate(modes):
            for j, period in enumerate(periods):
                if period in peak_patterns and peak_patterns[period]:
                    shares = [pattern.get(mode, 0) for pattern in peak_patterns[period]]
                    transition_matrix[i,j] = np.mean(shares)
                    
        sns.heatmap(transition_matrix, 
                    xticklabels=periods,
                    yticklabels=list(modes),
                    annot=True,
                    fmt='.2f')




    def _plot_time_variations(self, time_variations, vis_dir):
        """Plot time-of-day variations in mode choices"""
        plt.figure(figsize=(12, 6))
        
        for hour in range(24):
            if hour in time_variations:
                mode_shares = time_variations[hour]['mode_shares']
                for mode, shares in mode_shares.items():
                    if shares:  # Check if we have data
                        plt.plot(hour, np.mean(shares), 'o', label=mode)
                        
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Mode Share')
        plt.title('Time-of-Day Mode Share Variations')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(vis_dir / 'time_variations.png')
        plt.close()
    
    def _extract_parameter_variations(self, group: str) -> Dict:
        """
        Extract and organize parameter variations from tracked parameter history for all analysis types.
        
        This function systematically processes the parameter history to extract variations
        for each parameter group (subsidies, utility, service, etc.). It handles nested parameters,
        different data structures, and ensures proper type conversion and validation.
        
        Args:
            group: Parameter group being analyzed (e.g., 'subsidies', 'utility', etc.)
            
        Returns:
            Dictionary containing parameter variations organized by parameter type
        """
        variations = {}
        
        try:
            # Get parameters from tracked history
            params = self.current_analysis_results.get('parameters', [])
            if not params:
                print(f"No parameter data found for group: {group}")
                return variations

            print(f"\nExtracting parameters for {group}:")
            print(f"Number of parameter sets: {len(params)}")
            
            # Handle each parameter group with specific logic
            if group == 'subsidies':
                # For subsidies, extract pool sizes carefully
                pool_sizes = []
                for param_set in params:
                    # Debug output
                    print(f"\nProcessing parameter set:")
                    print(f"Subsidy config type: {type(param_set.get('subsidy_config'))}")
                    
                    config = param_set.get('subsidy_config')
                    if config is not None:
                        if hasattr(config, 'total_amount'):
                            pool_sizes.append(float(config.total_amount))
                            print(f"Added pool size from object: {config.total_amount}")
                        elif isinstance(config, dict) and 'total_amount' in config:
                            pool_sizes.append(float(config['total_amount']))
                            print(f"Added pool size from dict: {config['total_amount']}")
                
                if pool_sizes:
                    variations['pool_size'] = sorted(pool_sizes)
                    print(f"\nExtracted pool sizes: {variations['pool_size']}")
                else:
                    print("No pool sizes extracted")
                    
            elif group == 'utility':
                # Extract utility function coefficients
                coefficient_keys = [
                    'beta_C',  # Cost coefficient
                    'beta_T',  # Time coefficient
                    'beta_W',  # Wait time coefficient
                    'beta_A',  # Access time coefficient
                    'alpha'    # Base utility coefficient
                ]
                
                for coeff in coefficient_keys:
                    values = []
                    for param_set in params:
                        coeff_value = param_set.get('UTILITY_FUNCTION_BASE_COEFFICIENTS', {}).get(coeff)
                        if coeff_value is not None:
                            values.append(float(coeff_value))
                    if values:
                        variations[coeff] = values
                        
            elif group == 'service':
                # Extract service provider parameters (prices and capacities)
                service_params = {
                    'uber_price': ['uber_like1_price', 'uber_like2_price'],
                    'bike_price': ['bike_share1_price', 'bike_share2_price'],
                    'uber_capacity': ['uber_like1_capacity', 'uber_like2_capacity'],
                    'bike_capacity': ['bike_share1_capacity', 'bike_share2_capacity']
                }
                
                for consolidated_param, original_params in service_params.items():
                    values = []
                    for param_set in params:
                        # Take average of related parameters
                        param_values = [float(param_set.get(p, 0)) for p in original_params]
                        if param_values:
                            values.append(np.mean(param_values))
                    if values:
                        variations[consolidated_param] = values
                        
            elif group == 'maas':
                # Extract MaaS-specific parameters
                maas_params = {
                    'S_base': 'Base surcharge',
                    'alpha': 'Dynamic pricing sensitivity',
                    'delta': 'Subscription discount factor'
                }
                
                for param in maas_params:
                    values = []
                    for param_set in params:
                        value = param_set.get('DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS', {}).get(param)
                        if value is not None:
                            values.append(float(value))
                    if values:
                        variations[param] = values
                        
            elif group == 'value_of_time':
                # Extract value of time parameters for each income level
                income_levels = ['low', 'middle', 'high']
                
                for income in income_levels:
                    values = []
                    for param_set in params:
                        vot = param_set.get('VALUE_OF_TIME', {}).get(income)
                        if vot is not None:
                            values.append(float(vot))
                    if values:
                        variations[f'vot_{income}'] = values
                        
            elif group == 'public_transport':
                # Extract public transport pricing parameters
                price_params = {
                    'train_peak': ('train', 'on_peak'),
                    'train_offpeak': ('train', 'off_peak'),
                    'bus_peak': ('bus', 'on_peak'),
                    'bus_offpeak': ('bus', 'off_peak')
                }
                
                for param_name, (mode, period) in price_params.items():
                    values = []
                    for param_set in params:
                        price = param_set.get('public_price_table', {}).get(mode, {}).get(period)
                        if price is not None:
                            values.append(float(price))
                    if values:
                        variations[param_name] = values
                        
            elif group == 'congestion':
                # Extract congestion-related parameters
                congestion_params = [
                    'CONGESTION_ALPHA',
                    'CONGESTION_BETA',
                    'CONGESTION_CAPACITY',
                    'CONGESTION_T_IJ_FREE_FLOW'
                ]
                
                for param in congestion_params:
                    values = []
                    for param_set in params:
                        value = param_set.get(param)
                        if value is not None:
                            values.append(float(value))
                    if values:
                        variations[param] = values

            # Print summary of extracted variations
            print("\nExtracted parameter variations:")
            for param, values in variations.items():
                if values:
                    print(f"{param}: {len(values)} values, range: [{min(values):.2f}, {max(values):.2f}]")

            return variations
                    
        except Exception as e:
            print(f"Error extracting parameter variations: {e}")
            traceback.print_exc()
            return variations
    
    def _create_visualizations(self, results: Dict, group: str):
        """Create comprehensive visualizations for analysis results"""
        # Create output directory for this group
        vis_dir = self.output_dir / group / 'visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)
        # Store results temporarily for parameter extraction
        self.results = {
            group: {
                'parameters': results.get('parameters', []),
                'simulation_results': results.get('simulation_results', []),
                'analysis': results.get('analysis', {})
            }
        }
        print(f"\nCreating visualizations for {group}...")
        print(f"Output directory: {vis_dir}")
        # Mode choice evolution plots
        print("Starting mode choice evolution plots...")
        self._plot_mode_choice_evolution(results, vis_dir)
        
        # Parameter impact plots
        self._plot_parameter_impacts(results, vis_dir)
        
        # Temporal pattern plots
        if 'analysis' in results and 'temporal' in results['analysis']:
            self._plot_temporal_patterns(results, vis_dir)
        
    def _save_results(self, all_results: Dict, analysis_type: str):
        """Save comprehensive results and metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save results pickle
        results_file = self.output_dir / f'focus_analysis_{analysis_type}_{timestamp}.pkl'
        with open(results_file, 'wb') as f:
            pickle.dump(all_results, f)
            
        # Save analysis summary
        summary = self._create_analysis_summary(all_results)
        summary_file = self.output_dir / f'analysis_summary_{timestamp}.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4)
            
        print(f"\nResults saved to: {self.output_dir}")



    def extract_temporal_patterns(self, session) -> Dict:
        """Extract temporal pattern data from simulation results."""
        patterns = {}
        
        # Query patterns by time period
        for period in ['morning_peak', 'evening_peak', 'off_peak']:
            patterns[period] = self._extract_period_data(session, period)
        
        return patterns
    def _extract_period_data(self, session, period):
        """
        Extract temporal pattern data for a specific time period.
        
        Args:
            session: SQLAlchemy session
            period: String indicating time period ('morning_peak', 'evening_peak', 'off_peak')
        
        Returns:
            Dict containing pattern data for the specified period
        """
        period_ranges = {
            'morning_peak': (42, 60),   # 7:00-10:00
            'evening_peak': (96, 114),  # 16:00-19:00
            'off_peak': None            # All other times
        }
        
        try:
            if period == 'off_peak':
                # Query for off-peak periods (not in morning or evening peak)
                query = session.query(
                    ServiceBookingLog.record_company_name,
                    func.count(ServiceBookingLog.request_id)
                ).filter(
                    ~((ServiceBookingLog.start_time % 144).between(42, 60)) &
                    ~((ServiceBookingLog.start_time % 144).between(96, 114))
                ).group_by(
                    ServiceBookingLog.record_company_name
                )
            else:
                # Query for peak periods
                start, end = period_ranges[period]
                query = session.query(
                    ServiceBookingLog.record_company_name,
                    func.count(ServiceBookingLog.request_id)
                ).filter(
                    (ServiceBookingLog.start_time % 144).between(start, end)
                ).group_by(
                    ServiceBookingLog.record_company_name
                )
            
            results = query.all()
            return {
                'mode_counts': {mode: count for mode, count in results},
                'period': period
            }
            
        except Exception as e:
            print(f"Error extracting period data for {period}: {e}")
            return {
                'mode_counts': {},
                'period': period
            }
    def calculate_equity_metrics(self, session) -> Dict:
        """Calculate equity metrics from simulation results."""
        metrics = {}
        
        # Calculate mode access equity
        metrics['mode_access'] = self._calculate_mode_access_equity(session)
        
        # Calculate temporal equity
        metrics['temporal'] = self._calculate_temporal_equity(session)
        
        # Calculate subsidy distribution equity
        metrics['subsidy'] = self._calculate_subsidy_equity(session)
        
        return metrics

    def run_single_simulation(self, params):
        """Run single simulation with cleaned parameters"""
        db_path = f"service_provider_database_{os.getpid()}.db"
        db_connection_string = f"sqlite:///{db_path}"
        engine = None
        session = None
        try:
            # Remove tracking parameters before model initialization
            analysis_type = params.pop('_analysis_type', None)
            varied_mode = params.pop('varied_mode', None)
            varied_coefficient = params.pop('_varied_coefficient', None)
            simulation_steps = params.pop('simulation_steps')
            
            engine = create_engine(db_connection_string)
            Session = sessionmaker(bind=engine)
            session = Session()
            
            model = MobilityModel(db_connection_string=db_connection_string, **params)
            model.run_model(simulation_steps)
            
            # Extract and return results
            results = {
                'mode_shares': self.extract_mode_shares(session),
                'temporal_data': self.extract_temporal_patterns(session),
                'varied_coefficient': varied_coefficient
            }
            
            return results
            
        except Exception as e:
            print(f"Error in simulation {os.getpid()}: {str(e)}")
            return None
        finally:
            if session is not None:
                session.close()
            if os.path.exists(db_path):
                os.remove(db_path)

    def extract_mode_shares(self, session):
        """
        Extract consolidated mode shares from session data, combining similar modes.
        Returns mode shares for walk, public, maas, bike and car modes.
        """
        mode_shares = {}
        for income_level in ['low', 'middle', 'high']:
            # Get all bookings for this income level
            query = session.query(
                ServiceBookingLog.record_company_name,
                func.count(ServiceBookingLog.request_id)
            ).select_from(ServiceBookingLog).join(
                CommuterInfoLog,
                ServiceBookingLog.commuter_id == CommuterInfoLog.commuter_id
            ).filter(
                CommuterInfoLog.income_level == income_level
            ).group_by(
                ServiceBookingLog.record_company_name
            ).all()
            
            # Initialize counters for consolidated modes
            consolidated_counts = {
                'walk': 0,
                'public': 0,
                'maas': 0,
                'bike': 0,
                'car': 0
            }
            
            # Aggregate counts by consolidated mode
            total = 0
            for mode, count in query:
                total += count
                # Map detailed modes to consolidated modes
                if 'BikeShare' in mode:
                    consolidated_counts['bike'] += count
                elif 'UberLike' in mode:
                    consolidated_counts['car'] += count
                elif mode == 'public':
                    consolidated_counts['public'] += count
                elif mode == 'walk':
                    consolidated_counts['walk'] += count
                elif mode == 'MaaS_Bundle':
                    consolidated_counts['maas'] += count
            
            # Calculate shares
            if total > 0:
                mode_shares[income_level] = {
                    mode: count/total
                    for mode, count in consolidated_counts.items()
                }
            else:
                mode_shares[income_level] = {
                    mode: 0 for mode in consolidated_counts.keys()
                }
        
        return mode_shares
    def _save_results(self, all_results, analysis_type):
        """Simplified results saving"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f'focus_analysis_{analysis_type}_{timestamp}.pkl'
        
        with open(output_file, 'wb') as f:
            pickle.dump(all_results, f)
def main():
    """Run complete sensitivity analysis across different parameter groups"""
    
    # Define comprehensive base parameters
    base_parameters = {
        'num_commuters': 120,
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
        'ALPHA_VALUES': {
            'UberLike1': 0.5, 'UberLike2': 0.5,
            'BikeShare1': 0.5, 'BikeShare2': 0.5
        },
        'BACKGROUND_TRAFFIC_AMOUNT': 70,
        'CONGESTION_ALPHA': 0.25,
        'CONGESTION_BETA': 4,
        'CONGESTION_CAPACITY': 4,
        'CONGESTION_T_IJ_FREE_FLOW': 2,
        'simulation_steps': SIMULATION_STEPS
    }

    try:
        print("\nStarting Sensitivity Pattern Analysis...")
        print("----------------------------------------")
        
        # Initialize analyzer with base parameters
        analyzer = ParameterFocusAnalyzer(base_parameters)
        
        # Run FPS analysis for each parameter group
        print("\nRunning FPS Analysis...")
        fps_results = analyzer.run_focus_analysis('FPS', {
            **base_parameters, 
            'varied_mode': 'service'  # ['utility', 'service', 'maas', 'value_of_time','public_transport', 'congestion', 'subsidies']
        })
        
        print("\nFPS Analysis Results Summary:")
        for group, results in fps_results.items():
            analyzer._create_visualizations(results, group)
            print(f"\n{group.capitalize()} Parameter Group:")
            print(f"- Number of simulations: {len(results['simulation_results'])}")
            print(f"- Analysis metrics computed: {list(results['analysis'].keys())}")
        
        # Run PBS analysis for each parameter group
        print("\nRunning PBS Analysis...")
        # pbs_results = analyzer.run_focus_analysis('PBS')
        
        # print("\nPBS Analysis Results Summary:")
        # for group, results in pbs_results.items():
            # analyzer._create_visualizations(results, group)
        #     print(f"\n{group.capitalize()} Parameter Group:")
        #     print(f"- Number of simulations: {len(results['simulation_results'])}")
        #     print(f"- Analysis metrics computed: {list(results['analysis'].keys())}")
        
        # Save comprehensive results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(f'sensitivity_analysis_results_{timestamp}.pkl')
        
        with open(output_file, 'wb') as f:
            pickle.dump({
                'fps_results': fps_results,
                # 'pbs_results': pbs_results,
                'base_parameters': base_parameters,
                'timestamp': timestamp
            }, f)
        
        print("\nAnalysis complete!")
        print(f"Results saved to: {output_file}")
        print("\nGenerated visualizations can be found in:")
        print(f"- FPS: {analyzer.output_dir}/fps")
        print(f"- PBS: {analyzer.output_dir}/pbs")

    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()