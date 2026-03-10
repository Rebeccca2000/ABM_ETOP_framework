"""
ABM-ETOP Sensitivity Analysis Module

This script provides specialized sensitivity analysis functionality for the 
Agent-Based Model for Equity-Transportation Optimization and Policy (ABM-ETOP) 
framework.

It includes tools for parameter interaction analysis and stability testing to
enhance the robustness of findings from the ABM-ETOP model.

Usage:
    python abm_sensitivity_analysis.py --db_path path/to/database.db --output_dir path/to/output
    
    or
    
    python abm_sensitivity_analysis.py --run_simulation --steps 100 --output_dir path/to/output
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import os
import argparse
import warnings
import traceback
import random
import copy
import time
from datetime import datetime
from collections import defaultdict
import json
import sys

# Database tools
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# Add parent directory to path to import model modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import model modules - these will only work if the script is run from the correct directory
try:
    from run_visualisation_03 import MobilityModel
    from database_01 import (DB_CONNECTION_STRING, num_commuters, grid_width, grid_height, 
                            income_weights, health_weights, payment_weights, age_distribution, 
                            disability_weights, tech_access_weights, subsidy_dataset, daily_config,
                            SIMULATION_STEPS, ASC_VALUES, UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS,
                            UTILITY_FUNCTION_BASE_COEFFICIENTS, PENALTY_COEFFICIENTS,
                            AFFORDABILITY_THRESHOLDS, FLEXIBILITY_ADJUSTMENTS, VALUE_OF_TIME,
                            public_price_table, ALPHA_VALUES, DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS,
                            BACKGROUND_TRAFFIC_AMOUNT, CONGESTION_ALPHA, CONGESTION_BETA,
                            CONGESTION_CAPACITY, CONGESTION_T_IJ_FREE_FLOW,
                            UberLike1_cpacity, UberLike1_price, UberLike2_cpacity, UberLike2_price,
                            BikeShare1_capacity, BikeShare1_price, BikeShare2_capacity, BikeShare2_price)
    from agent_subsidy_pool import SubsidyPoolConfig
    from travel_time_equity_optimization import calculate_travel_time_equity
    MODEL_IMPORTS_AVAILABLE = True
except ImportError:
    warnings.warn("Model modules could not be imported. Some functionality may be limited.")
    MODEL_IMPORTS_AVAILABLE = False


class ABMSensitivityAnalyzer:
    """
    A class for conducting sensitivity analyses on the ABM-ETOP model.
    This class provides methods to analyze parameter interactions and 
    assess model stability across different conditions.
    """
    
    def __init__(self, model=None, db_connection_string=None, output_dir='sensitivity_analysis_outputs'):
        """
        Initialize the sensitivity analyzer with either a running model instance 
        or a database connection string.
        
        Args:
            model: A MobilityModel instance
            db_connection_string: Connection string to the simulation database
            output_dir: Directory to save analysis outputs
        """
        self.model = model
        self.db_connection_string = db_connection_string
        
        # Initialize database connection if string provided
        if db_connection_string and not model:
            try:
                self.engine = create_engine(db_connection_string)
                # Test connection
                with self.engine.connect() as conn:
                    result = conn.execute(text("SELECT 1"))
                    if not result.fetchone():
                        raise Exception("Database connection test failed")
                print(f"Successfully connected to database: {db_connection_string}")
            except Exception as e:
                print(f"Error connecting to database: {e}")
                self.engine = None
        elif model:
            self.engine = model.db_engine if hasattr(model, 'db_engine') else None
        
        # Define color schemes for income groups
        self.income_colors = {
            'low': '#1f77b4',    # Blue
            'middle': '#ff7f0e', # Orange
            'high': '#2ca02c'    # Green
        }
        
        # Define color schemes for transportation modes
        self.mode_colors = {
            'walk': '#7f7f7f',   # Gray
            'bike': '#17becf',   # Cyan
            'car': '#d62728',    # Red
            'bus': '#9467bd',    # Purple
            'train': '#8c564b',  # Brown
            'MaaS_Bundle': '#e377c2'  # Pink
        }
        
        # Create output directory for visualizations
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Get grid dimensions
        if model:
            self.grid_width = model.grid_width
            self.grid_height = model.grid_height
        else:
            # Default dimensions if not available
            self.grid_width = 55
            self.grid_height = 55
            
        # Timestamp for logging and filenames
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create log file
        self.log_file = os.path.join(self.output_dir, f"sensitivity_analysis_log_{self.timestamp}.txt")
        with open(self.log_file, "w") as f:
            f.write(f"ABM-ETOP Sensitivity Analysis Log\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    def _log(self, message):
        """Log a message to console and log file"""
        print(message)
        with open(self.log_file, "a") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
            
    def _execute_query(self, query, params=None):
        """
        Execute a SQL query and return the results as a DataFrame.
        
        Args:
            query: SQL query text
            params: Parameters for the query
            
        Returns:
            pandas DataFrame with query results
        """
        if not self.engine:
            self._log("No database connection available.")
            return pd.DataFrame()
        
        try:
            with self.engine.connect() as conn:
                if params:
                    result = conn.execute(text(query), params)
                else:
                    result = conn.execute(text(query))
                return pd.DataFrame(result.fetchall(), columns=result.keys())
        except Exception as e:
            self._log(f"Error executing query: {e}")
            return pd.DataFrame()
    
    def _get_default_model_params(self):
        """
        Helper method to get default model parameters
        
        Returns:
            Dictionary of default parameters for MobilityModel
        """
        if not MODEL_IMPORTS_AVAILABLE:
            return {}
            
        return {
            'db_connection_string': DB_CONNECTION_STRING,
            'num_commuters': num_commuters,
            'grid_width': grid_width,
            'grid_height': grid_height,
            'data_income_weights': income_weights,
            'data_health_weights': health_weights,
            'data_payment_weights': payment_weights,
            'data_age_distribution': age_distribution,
            'data_disability_weights': disability_weights,
            'data_tech_access_weights': tech_access_weights,
            'ASC_VALUES': ASC_VALUES,
            'UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS': UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS,
            'UTILITY_FUNCTION_BASE_COEFFICIENTS': UTILITY_FUNCTION_BASE_COEFFICIENTS,
            'PENALTY_COEFFICIENTS': PENALTY_COEFFICIENTS,
            'AFFORDABILITY_THRESHOLDS': AFFORDABILITY_THRESHOLDS,
            'FLEXIBILITY_ADJUSTMENTS': FLEXIBILITY_ADJUSTMENTS,
            'VALUE_OF_TIME': VALUE_OF_TIME,
            'public_price_table': public_price_table,
            'ALPHA_VALUES': ALPHA_VALUES,
            'DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS': DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS,
            'BACKGROUND_TRAFFIC_AMOUNT': BACKGROUND_TRAFFIC_AMOUNT,
            'CONGESTION_ALPHA': CONGESTION_ALPHA,
            'CONGESTION_BETA': CONGESTION_BETA,
            'CONGESTION_CAPACITY': CONGESTION_CAPACITY,
            'CONGESTION_T_IJ_FREE_FLOW': CONGESTION_T_IJ_FREE_FLOW,
            'uber_like1_capacity': UberLike1_cpacity,
            'uber_like1_price': UberLike1_price,
            'uber_like2_capacity': UberLike2_cpacity,
            'uber_like2_price': UberLike2_price,
            'bike_share1_capacity': BikeShare1_capacity,
            'bike_share1_price': BikeShare1_price,
            'bike_share2_capacity': BikeShare2_capacity,
            'bike_share2_price': BikeShare2_price,
            'subsidy_dataset': subsidy_dataset,
            'subsidy_config': daily_config
        }
    
    def _update_nested_dict(self, original_dict, keys_path, new_value):
        """
        Update a nested dictionary using a path of keys
        
        Args:
            original_dict: Dictionary to update
            keys_path: List or string with period-separated keys for path
            new_value: New value to set
            
        Returns:
            Updated dictionary
        """
        if isinstance(keys_path, str):
            keys_path = keys_path.split('.')
        
        current = original_dict
        for key in keys_path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys_path[-1]] = new_value
        return original_dict
    
    def _get_nested_value(self, nested_dict, keys_path):
        """
        Get a value from a nested dictionary using a path of keys
        
        Args:
            nested_dict: Dictionary to retrieve from
            keys_path: List or string with period-separated keys for path
            
        Returns:
            Value at the specified path
        """
        if isinstance(keys_path, str):
            keys_path = keys_path.split('.')
        
        current = nested_dict
        for key in keys_path:
            if key not in current:
                return None
            current = current[key]
        
        return current
    
    def _calculate_model_metrics(self, model, metrics=None):
        """
        Calculate specified metrics from a model
        
        Args:
            model: MobilityModel instance
            metrics: List of metrics to calculate
            
        Returns:
            Dictionary of calculated metrics
        """
        if metrics is None:
            metrics = ['travel_time_equity_index']
        
        results = {}
        
        try:
            with model.Session() as session:
                if 'travel_time_equity_index' in metrics:
                    tte_results = calculate_travel_time_equity(session)
                    results['travel_time_equity_index'] = tte_results.get('travel_time_equity_index', float('nan'))
                    
                    # Also store income-specific travel times
                    for income in ['low', 'middle', 'high']:
                        if income in tte_results:
                            results[f'{income}_avg_travel_time'] = tte_results[income].get('avg_travel_time', float('nan'))
                
                # Add other metrics as needed
                # For example:
                if 'mode_share_equity' in metrics:
                    # This would need to be implemented based on your model
                    # For now, using a placeholder
                    results['mode_share_equity'] = 0.5
                
                if 'total_system_travel_time' in metrics:
                    # This would need to be implemented based on your model
                    # For now, using a placeholder
                    results['total_system_travel_time'] = 4000
                    
        except Exception as e:
            self._log(f"Error calculating model metrics: {e}")
            traceback.print_exc()
            
            # Fill with NaN values
            for metric in metrics:
                results[metric] = float('nan')
        
        return results
    
    def plot_parameter_interaction(self, param1_name, param1_range, param2_name, param2_range, 
                                  outcome_metric='travel_time_equity_index', runs_per_cell=3, save=True):
        """
        Generate a heatmap showing how two parameters jointly influence a specified outcome metric.
        
        Args:
            param1_name: Name of first parameter
            param1_range: List of values for first parameter
            param2_name: Name of second parameter
            param2_range: List of values for second parameter
            outcome_metric: Metric to evaluate (e.g., 'travel_time_equity_index')
            runs_per_cell: Number of simulation runs per parameter combination
            save: Whether to save the visualization
            
        Returns:
            Matplotlib figure
        """
        if not MODEL_IMPORTS_AVAILABLE:
            self._log("Model modules not available. Cannot run parameter interaction analysis.")
            return None
        
        self._log(f"Starting parameter interaction analysis for {param1_name} vs {param2_name} on {outcome_metric}")
        
        # Clean parameter names for display and filenames
        param1_display = param1_name.split('_')[-1] if '_' in param1_name else param1_name
        param2_display = param2_name.split('_')[-1] if '_' in param2_name else param2_name
        
        # For complex parameters, generate simplified labels
        if isinstance(param1_range[0], dict):
            param1_labels = [f"Set {i+1}" for i in range(len(param1_range))]
        else:
            param1_labels = [str(val) for val in param1_range]
            
        if isinstance(param2_range[0], dict):
            param2_labels = [f"Set {i+1}" for i in range(len(param2_range))]
        else:
            param2_labels = [str(val) for val in param2_range]
        
        # Create results grid
        results = np.zeros((len(param1_range), len(param2_range)))
        std_devs = np.zeros((len(param1_range), len(param2_range)))
        
        # Store detailed results for later analysis
        detailed_results = []
        
        # Get base parameters
        base_params = self._get_default_model_params()
        
        # For each parameter combination
        start_time = time.time()
        for i, p1 in enumerate(param1_range):
            for j, p2 in enumerate(param2_range):
                self._log(f"Testing {param1_display} = {param1_labels[i]}, {param2_display} = {param2_labels[j]}")
                
                # Store results for this cell
                cell_results = []
                
                # Run multiple simulations
                for run in range(runs_per_cell):
                    # Set random seed for reproducibility while maintaining variation
                    seed = hash(f"{param1_name}_{i}_{param2_name}_{j}_{run}") % 10000
                    np.random.seed(seed)
                    random.seed(seed)
                    
                    # Create model with these parameter values
                    model_params = copy.deepcopy(base_params)
                    
                    # Apply parameter values
                    if isinstance(p1, dict):
                        # For dictionary parameters, update the entire dict
                        model_params[param1_name] = copy.deepcopy(p1)
                    else:
                        # For scalar parameters, just update the value
                        model_params[param1_name] = p1
                    
                    if isinstance(p2, dict):
                        model_params[param2_name] = copy.deepcopy(p2)
                    else:
                        model_params[param2_name] = p2
                    
                    # Run simulation and evaluate outcome
                    try:
                        self._log(f"  Run {run+1}/{runs_per_cell}: Starting simulation...")
                        
                        # Create model with parameters
                        model = MobilityModel(**model_params)
                        
                        # Run shortened simulation for efficiency
                        model.run_model(SIMULATION_STEPS // 4)
                        
                        # Calculate metrics
                        metrics = self._calculate_model_metrics(model, [outcome_metric])
                        value = metrics.get(outcome_metric, float('nan'))
                        
                        self._log(f"  Run {run+1}/{runs_per_cell}: {outcome_metric} = {value:.4f}")
                        
                        # Store result
                        cell_results.append(value)
                        
                        # Store detailed result for logging
                        detailed_results.append({
                            'param1_name': param1_name,
                            'param1_value': param1_labels[i],
                            'param2_name': param2_name,
                            'param2_value': param2_labels[j],
                            'run': run,
                            'metric': outcome_metric,
                            'value': value
                        })
                        
                    except Exception as e:
                        self._log(f"  Error in simulation: {e}")
                        traceback.print_exc()
                        cell_results.append(float('nan'))
                
                # Calculate mean and standard deviation for this cell
                if cell_results:
                    results[i, j] = np.nanmean(cell_results)
                    std_devs[i, j] = np.nanstd(cell_results)
                    self._log(f"  Cell results: Mean = {results[i, j]:.4f}, StdDev = {std_devs[i, j]:.4f}")
        
        elapsed_time = time.time() - start_time
        self._log(f"Parameter interaction analysis completed in {elapsed_time:.2f} seconds")
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # Create custom colormaps based on outcome metric
        # For equity metrics, lower values are better, so use reversed colormap
        if 'equity' in outcome_metric.lower():
            cmap_mean = 'viridis_r'
        else:
            cmap_mean = 'viridis'
        
        # Plot mean values
        im1 = axes[0].imshow(results, cmap=cmap_mean, origin='lower', aspect='auto')
        axes[0].set_title(f'Mean {outcome_metric}', fontsize=14)
        axes[0].set_xlabel(param2_display, fontsize=12)
        axes[0].set_ylabel(param1_display, fontsize=12)
        
        # Set tick labels
        axes[0].set_xticks(range(len(param2_range)))
        axes[0].set_yticks(range(len(param1_range)))
        axes[0].set_xticklabels(param2_labels)
        axes[0].set_yticklabels(param1_labels)
        
        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=axes[0])
        cbar1.set_label(outcome_metric)
        
        # Add text annotations for cell values
        for i in range(len(param1_range)):
            for j in range(len(param2_range)):
                axes[0].text(j, i, f"{results[i, j]:.2f}", 
                           ha="center", va="center", color="white", fontweight="bold",
                           fontsize=9)
        
        # Plot standard deviations
        im2 = axes[1].imshow(std_devs, cmap='plasma', origin='lower', aspect='auto')
        axes[1].set_title(f'Standard Deviation of {outcome_metric}', fontsize=14)
        axes[1].set_xlabel(param2_display, fontsize=12)
        axes[1].set_ylabel(param1_display, fontsize=12)
        
        # Set tick labels
        axes[1].set_xticks(range(len(param2_range)))
        axes[1].set_yticks(range(len(param1_range)))
        axes[1].set_xticklabels(param2_labels)
        axes[1].set_yticklabels(param1_labels)
        
        # Add colorbar
        cbar2 = plt.colorbar(im2, ax=axes[1])
        cbar2.set_label(f'Standard Deviation of {outcome_metric}')
        
        # Add text annotations for cell values
        for i in range(len(param1_range)):
            for j in range(len(param2_range)):
                axes[1].text(j, i, f"{std_devs[i, j]:.2f}", 
                           ha="center", va="center", color="white", fontweight="bold",
                           fontsize=9)
        
        # Add overall title
        fig.suptitle(f'Parameter Interaction Analysis: {param1_display} vs {param2_display}', 
                    fontsize=16)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Save results
        if save:
            # Save figure
            filename = f'parameter_interaction_{param1_display}_{param2_display}_{self.timestamp}'
            plt.savefig(os.path.join(self.output_dir, f'{filename}.png'), 
                       dpi=300, bbox_inches='tight')
            
            # Save detailed results as CSV
            if detailed_results:
                results_df = pd.DataFrame(detailed_results)
                results_df.to_csv(os.path.join(self.output_dir, f'{filename}_data.csv'), index=False)
            
            self._log(f"Parameter interaction analysis saved to {filename}.png")
        
        return fig
    
    def plot_stability_analysis(self, num_runs=20, key_parameters=None, metrics=None, save=True):
        """
        Run multiple simulations with different random seeds to assess model stability.
        
        Args:
            num_runs: Number of simulation runs
            key_parameters: Dictionary of parameter variations to test
            metrics: List of metrics to track
            save: Whether to save the visualization
            
        Returns:
            Matplotlib figure
        """
        if not MODEL_IMPORTS_AVAILABLE:
            self._log("Model modules not available. Cannot run stability analysis.")
            return None
        
        self._log(f"Starting stability analysis with {num_runs} runs per parameter set")
        
        if metrics is None:
            metrics = ['travel_time_equity_index']
        
        # Store results
        results = {metric: [] for metric in metrics}
        parameter_sets = []
        
        # If we're testing parameter variations
        if key_parameters:
            # Create base parameter set
            base_params = self._get_default_model_params()
            
            # For each parameter variation
            for param_name, values in key_parameters.items():
                for value in values:
                    # Create new parameter set
                    params = copy.deepcopy(base_params)
                    
                    # Apply parameter value
                    if isinstance(value, dict):
                        params[param_name] = copy.deepcopy(value)
                    else:
                        params[param_name] = value
                    
                    # Create label
                    if isinstance(value, dict):
                        label = f"{param_name} Set"
                    else:
                        label = f"{param_name}={value}"
                    
                    parameter_sets.append((label, params))
        else:
            # Just use default parameters
            parameter_sets = [("Default", self._get_default_model_params())]
        
        # Store detailed results
        detailed_results = []
        
        # For each parameter set
        start_time = time.time()
        for param_label, params in parameter_sets:
            self._log(f"Testing parameter set: {param_label}")
            
            # Run simulations with different random seeds
            run_results = {metric: [] for metric in metrics}
            
            for run in range(num_runs):
                # Set random seed
                seed = run * 1000 + hash(param_label) % 1000
                np.random.seed(seed)
                random.seed(seed)
                
                self._log(f"  Run {run+1}/{num_runs} with seed {seed}")
                
                try:
                    # Create model with parameters
                    model = MobilityModel(**params)
                    
                    # Run shortened simulation for efficiency
                    model.run_model(SIMULATION_STEPS // 2)
                    
                    # Calculate metrics
                    model_metrics = self._calculate_model_metrics(model, metrics)
                    
                    # Store results
                    for metric in metrics:
                        value = model_metrics.get(metric, float('nan'))
                        run_results[metric].append(value)
                        
                        self._log(f"    {metric} = {value:.4f}")
                        
                        # Store detailed result
                        detailed_results.append({
                            'parameter_set': param_label,
                            'run': run,
                            'seed': seed,
                            'metric': metric,
                            'value': value
                        })
                
                except Exception as e:
                    self._log(f"  Error in simulation run {run} with {param_label}: {e}")
                    traceback.print_exc()
                    
                    # Add NaN for this run
                    for metric in metrics:
                        run_results[metric].append(float('nan'))
            
            # Store results for this parameter set
            for metric in metrics:
                values = run_results[metric]
                
                # Calculate statistics
                mean_value = np.nanmean(values)
                std_value = np.nanstd(values)
                cv_value = std_value / mean_value if mean_value != 0 else float('nan')
                
                self._log(f"  {metric} - Mean: {mean_value:.4f}, StdDev: {std_value:.4f}, CV: {cv_value:.4f}")
                
                results[metric].append({
                    'label': param_label,
                    'values': values,
                    'mean': mean_value,
                    'std': std_value,
                    'cv': cv_value
                })
        
        elapsed_time = time.time() - start_time
        self._log(f"Stability analysis completed in {elapsed_time:.2f} seconds")
        
        # Create figure - one row per metric
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 5 * len(metrics)))
        
        # Handle single metric case
        if len(metrics) == 1:
            axes = [axes]
        
        # For each metric
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Extract data for this metric
            metric_results = results[metric]
            
            # Calculate positions for box plots
            positions = np.arange(len(metric_results))
            
            # Create box plots
            boxplot = ax.boxplot([r['values'] for r in metric_results], positions=positions, 
                               patch_artist=True, widths=0.6)
            
            # Set colors
            for box in boxplot['boxes']:
                box.set(facecolor='lightblue')
            
            # Add individual points for better visualization
            for j, result in enumerate(metric_results):
                # Add jittered points
                x_jitter = np.random.normal(j, 0.05, size=len(result['values']))
                ax.scatter(x_jitter, result['values'], alpha=0.5, c='blue', s=20)
                
                # Add mean and std text
                ax.text(j, ax.get_ylim()[1] * 0.95, 
                       f"Mean: {result['mean']:.2f}\nStd: {result['std']:.2f}\nCV: {result['cv']:.2f}",
                       ha='center', va='top', fontsize=9,
                       bbox=dict(facecolor='white', alpha=0.8))
            
            # Add horizontal mean line for reference
            all_values = [val for result in metric_results for val in result['values'] if not np.isnan(val)]
            overall_mean = np.mean(all_values) if all_values else 0
            ax.axhline(y=overall_mean, color='red', linestyle='--', alpha=0.5, 
                      label=f'Overall Mean: {overall_mean:.2f}')
            
            # Set labels and title
            ax.set_title(f'Stability Analysis: {metric}', fontsize=14)
            ax.set_ylabel(metric, fontsize=12)
            ax.set_xticks(positions)
            ax.set_xticklabels([r['label'] for r in metric_results], rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Add overall title
        if key_parameters:
            param_names = ', '.join(key_parameters.keys())
            fig.suptitle(f'Stability Analysis: Varying {param_names}', fontsize=16)
        else:
            fig.suptitle('Stability Analysis: Random Seed Variation', fontsize=16)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95, bottom=0.15)
        
        # Save results
        if save:
            # Save figure
            filename = f'stability_analysis_{self.timestamp}'
            if key_parameters:
                param_keys = '_'.join(key_parameters.keys())
                filename = f'stability_analysis_{param_keys}_{self.timestamp}'
            
            plt.savefig(os.path.join(self.output_dir, f'{filename}.png'), 
                       dpi=300, bbox_inches='tight')
            
            # Save detailed results as CSV
            if detailed_results:
                results_df = pd.DataFrame(detailed_results)
                results_df.to_csv(os.path.join(self.output_dir, f'{filename}_data.csv'), index=False)
            
            self._log(f"Stability analysis saved to {filename}.png")
        
        return fig

    def plot_parameter_sensitivity(self, parameter_name, parameter_values, 
                                 metrics=None, runs_per_value=5, save=True):
        """
        Analyze sensitivity of model outcomes to variations in a single parameter.
        
        Args:
            parameter_name: Name of parameter to vary
            parameter_values: List of values to test
            metrics: List of metrics to track
            runs_per_value: Number of runs per parameter value
            save: Whether to save the visualization
            
        Returns:
            Matplotlib figure
        """
        if not MODEL_IMPORTS_AVAILABLE:
            self._log("Model modules not available. Cannot run parameter sensitivity analysis.")
            return None
        
        # Convert parameter name for display
        param_display = parameter_name.split('_')[-1] if '_' in parameter_name else parameter_name
        
        self._log(f"Starting parameter sensitivity analysis for {param_display}")
        
        if metrics is None:
            metrics = ['travel_time_equity_index']
        
        # Create parameter labels
        if isinstance(parameter_values[0], dict):
            param_labels = [f"Set {i+1}" for i in range(len(parameter_values))]
        else:
            param_labels = [str(val) for val in parameter_values]
        
        # Get base parameters
        base_params = self._get_default_model_params()
        
        # Store results
        results = []
        
        # For each parameter value
        start_time = time.time()
        for i, value in enumerate(parameter_values):
            self._log(f"Testing {param_display} = {param_labels[i]}")
            
            # Create parameter set
            params = copy.deepcopy(base_params)
            
            # Apply parameter value
            if isinstance(value, dict):
                params[parameter_name] = copy.deepcopy(value)
            else:
                params[parameter_name] = value
            
            # Run multiple simulations
            for run in range(runs_per_value):
                # Set random seed
                seed = run * 1000 + hash(f"{parameter_name}_{i}") % 1000
                np.random.seed(seed)
                random.seed(seed)
                
                self._log(f"  Run {run+1}/{runs_per_value} with seed {seed}")
                
                try:
                    # Create model with parameters
                    model = MobilityModel(**params)
                    
                    # Run simulation
                    model.run_model(SIMULATION_STEPS // 2)
                    
                    # Calculate metrics
                    model_metrics = self._calculate_model_metrics(model, metrics)
                    
                    # Store results
                    row = {
                        'parameter': parameter_name,
                        'parameter_label': param_labels[i],
                        'parameter_value': value if not isinstance(value, dict) else f"Set {i+1}",
                        'run': run,
                        'seed': seed
                    }
                    
                    # Add metrics
                    for metric in metrics:
                        row[metric] = model_metrics.get(metric, float('nan'))
                        self._log(f"    {metric} = {row[metric]:.4f}")
                    
                    results.append(row)
                
                except Exception as e:
                    self._log(f"  Error in simulation: {e}")
                    traceback.print_exc()
                    
                    # Add row with NaN values
                    row = {
                        'parameter': parameter_name,
                        'parameter_label': param_labels[i],
                        'parameter_value': value if not isinstance(value, dict) else f"Set {i+1}",
                        'run': run,
                        'seed': seed
                    }
                    
                    # Add NaN metrics
                    for metric in metrics:
                        row[metric] = float('nan')
                    
                    results.append(row)
        
        elapsed_time = time.time() - start_time
        self._log(f"Parameter sensitivity analysis completed in {elapsed_time:.2f} seconds")
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate statistics by parameter value
        stats_df = results_df.groupby('parameter_label')[metrics].agg(['mean', 'std', 'count'])
        
        # Calculate elasticity (fractional change in output / fractional change in input)
        # Only for numeric parameter values
        elasticities = {}
        if not isinstance(parameter_values[0], dict) and len(parameter_values) > 1:
            # Convert values to numeric if possible
            try:
                numeric_values = np.array([float(v) for v in parameter_values])
                
                for metric in metrics:
                    # Get mean values
                    mean_values = stats_df[metric]['mean'].values
                    
                    # Calculate percentage changes
                    value_pct_changes = np.diff(numeric_values) / numeric_values[:-1]
                    output_pct_changes = np.diff(mean_values) / mean_values[:-1]
                    
                    # Calculate elasticities
                    metric_elasticities = output_pct_changes / value_pct_changes
                    
                    # Average elasticity
                    elasticities[metric] = np.nanmean(np.abs(metric_elasticities))
                    
                    self._log(f"Elasticity for {metric}: {elasticities[metric]:.4f}")
            except:
                self._log("Could not calculate elasticities for non-numeric parameter values")
        
        # Create figure - one subplot per metric
        fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 6 * len(metrics)))
        
        # Handle single metric case
        if len(metrics) == 1:
            axes = [axes]
        
        # For each metric
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Extract metric data
            metric_data = results_df.pivot(index='run', columns='parameter_label', values=metric)
            
            # Create box plot
            sns.boxplot(data=metric_data, ax=ax, palette='viridis')
            
            # Add scatter points for individual runs
            sns.stripplot(data=metric_data, ax=ax, color='black', alpha=0.5, jitter=True)
            
            # Add elasticity if available
            if metric in elasticities:
                ax.text(0.02, 0.95, f"Elasticity: {elasticities[metric]:.4f}", 
                       transform=ax.transAxes, fontsize=12,
                       bbox=dict(facecolor='white', alpha=0.8))
            
            # Add stats
            for j, param_label in enumerate(param_labels):
                if param_label in stats_df.index:
                    stats = stats_df.loc[param_label, metric]
                    ax.text(j, ax.get_ylim()[1] * 0.9, 
                          f"Mean: {stats['mean']:.2f}\nStd: {stats['std']:.2f}",
                          ha='center', fontsize=9,
                          bbox=dict(facecolor='white', alpha=0.8))
            
            # Set labels
            ax.set_title(f'Sensitivity of {metric} to {param_display}', fontsize=14)
            ax.set_xlabel(param_display, fontsize=12)
            ax.set_ylabel(metric, fontsize=12)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save results
        if save:
            # Save figure
            filename = f'parameter_sensitivity_{param_display}_{self.timestamp}'
            plt.savefig(os.path.join(self.output_dir, f'{filename}.png'), 
                       dpi=300, bbox_inches='tight')
            
            # Save detailed results
            results_df.to_csv(os.path.join(self.output_dir, f'{filename}_data.csv'), index=False)
            
            self._log(f"Parameter sensitivity analysis saved to {filename}.png")
        
        return fig, results_df

    def run_comprehensive_sensitivity_analysis(self, save=True):
        """
        Run a comprehensive set of sensitivity analyses on key model parameters.
        
        Args:
            save: Whether to save visualizations
            
        Returns:
            Dictionary of analysis results
        """
        self._log("Starting comprehensive sensitivity analysis")
        
        results = {}
        
        # 1. Parameter interaction analysis
        self._log("1. Parameter Interaction Analysis")
        
        # 1.1 Price vs Time Sensitivity
        price_sensitivity_range = [
            {'beta_C': -0.02, 'beta_T': -0.06, 'beta_W': -0.01, 'beta_A': -0.01, 'alpha': -0.01},
            {'beta_C': -0.05, 'beta_T': -0.06, 'beta_W': -0.01, 'beta_A': -0.01, 'alpha': -0.01},
            {'beta_C': -0.10, 'beta_T': -0.06, 'beta_W': -0.01, 'beta_A': -0.01, 'alpha': -0.01},
            {'beta_C': -0.15, 'beta_T': -0.06, 'beta_W': -0.01, 'beta_A': -0.01, 'alpha': -0.01}
        ]
        
        time_sensitivity_range = [
            {'beta_C': -0.02, 'beta_T': -0.03},
            {'beta_C': -0.02, 'beta_T': -0.06},
            {'beta_C': -0.02, 'beta_T': -0.09},
            {'beta_C': -0.02, 'beta_T': -0.12}
        ]
        
        try:
            results['price_time_interaction'] = self.plot_parameter_interaction(
                param1_name='UTILITY_FUNCTION_BASE_COEFFICIENTS',
                param1_range=price_sensitivity_range,
                param2_name='UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS',
                param2_range=time_sensitivity_range,
                outcome_metric='travel_time_equity_index',
                runs_per_cell=2,
                save=save
            )
        except Exception as e:
            self._log(f"Error in price vs time sensitivity analysis: {e}")
            traceback.print_exc()
        
        # 1.2 FPS vs Congestion Parameter
        fps_range = [
            SubsidyPoolConfig('daily', 1000),
            SubsidyPoolConfig('daily', 2000),
            SubsidyPoolConfig('daily', 3000),
            SubsidyPoolConfig('daily', 4000)
        ]
        
        congestion_alpha_range = [0.01, 0.05, 0.10, 0.15]
        
        try:
            results['fps_congestion_interaction'] = self.plot_parameter_interaction(
                param1_name='subsidy_config',
                param1_range=fps_range,
                param2_name='CONGESTION_ALPHA',
                param2_range=congestion_alpha_range,
                outcome_metric='travel_time_equity_index',
                runs_per_cell=2,
                save=save
            )
        except Exception as e:
            self._log(f"Error in FPS vs congestion parameter analysis: {e}")
            traceback.print_exc()
        
        # 2. Stability Analysis
        self._log("2. Stability Analysis")
        
        # 2.1 Basic stability with different random seeds
        try:
            results['basic_stability'] = self.plot_stability_analysis(
                num_runs=10,
                metrics=['travel_time_equity_index'],
                save=save
            )
        except Exception as e:
            self._log(f"Error in basic stability analysis: {e}")
            traceback.print_exc()
        
        # 2.2 Stability across different parameter values
        try:
            results['parameter_stability'] = self.plot_stability_analysis(
                num_runs=5,
                key_parameters={'CONGESTION_BETA': [3.0, 4.0, 5.0]},
                metrics=['travel_time_equity_index'],
                save=save
            )
        except Exception as e:
            self._log(f"Error in parameter stability analysis: {e}")
            traceback.print_exc()
        
        # 3. Parameter Sensitivity Analysis
        self._log("3. Parameter Sensitivity Analysis")
        
        # 3.1 Sensitivity to FPS value
        fps_values = [1000, 2000, 3000, 4000, 5000]
        fps_configs = [SubsidyPoolConfig('daily', v) for v in fps_values]
        
        try:
            results['fps_sensitivity'], _ = self.plot_parameter_sensitivity(
                parameter_name='subsidy_config',
                parameter_values=fps_configs,
                metrics=['travel_time_equity_index'],
                runs_per_value=3,
                save=save
            )
        except Exception as e:
            self._log(f"Error in FPS sensitivity analysis: {e}")
            traceback.print_exc()
        
        # 3.2 Sensitivity to congestion parameters
        congestion_beta_values = [2.0, 3.0, 4.0, 5.0, 6.0]
        
        try:
            results['congestion_sensitivity'], _ = self.plot_parameter_sensitivity(
                parameter_name='CONGESTION_BETA',
                parameter_values=congestion_beta_values,
                metrics=['travel_time_equity_index'],
                runs_per_value=3,
                save=save
            )
        except Exception as e:
            self._log(f"Error in congestion parameter sensitivity analysis: {e}")
            traceback.print_exc()
        
        self._log("Comprehensive sensitivity analysis completed")
        
        return results


def run_simulation_with_analyzer(steps=100, output_dir='sensitivity_analysis_outputs'):
    """
    Run a simulation and initialize the sensitivity analyzer.
    
    Args:
        steps: Number of simulation steps
        output_dir: Directory to save visualization outputs
        
    Returns:
        Tuple of (model, analyzer)
    """
    if not MODEL_IMPORTS_AVAILABLE:
        print("Model modules not available. Cannot run simulation.")
        return None, None
    
    try:
        # Initialize model with default parameters
        model = MobilityModel(
            db_connection_string=DB_CONNECTION_STRING,
            num_commuters=num_commuters,
            grid_width=grid_width,
            grid_height=grid_height,
            data_income_weights=income_weights,
            data_health_weights=health_weights,
            data_payment_weights=payment_weights,
            data_age_distribution=age_distribution,
            data_disability_weights=disability_weights,
            data_tech_access_weights=tech_access_weights,
            ASC_VALUES=ASC_VALUES,
            UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS=UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS,
            UTILITY_FUNCTION_BASE_COEFFICIENTS=UTILITY_FUNCTION_BASE_COEFFICIENTS,
            PENALTY_COEFFICIENTS=PENALTY_COEFFICIENTS,
            AFFORDABILITY_THRESHOLDS=AFFORDABILITY_THRESHOLDS,
            FLEXIBILITY_ADJUSTMENTS=FLEXIBILITY_ADJUSTMENTS,
            VALUE_OF_TIME=VALUE_OF_TIME,
            public_price_table=public_price_table,
            ALPHA_VALUES=ALPHA_VALUES,
            DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS=DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS,
            BACKGROUND_TRAFFIC_AMOUNT=BACKGROUND_TRAFFIC_AMOUNT,
            CONGESTION_ALPHA=CONGESTION_ALPHA,
            CONGESTION_BETA=CONGESTION_BETA,
            CONGESTION_CAPACITY=CONGESTION_CAPACITY,
            CONGESTION_T_IJ_FREE_FLOW=CONGESTION_T_IJ_FREE_FLOW,
            uber_like1_capacity=UberLike1_cpacity,
            uber_like1_price=UberLike1_price,
            uber_like2_capacity=UberLike2_cpacity,
            uber_like2_price=UberLike2_price,
            bike_share1_capacity=BikeShare1_capacity,
            bike_share1_price=BikeShare1_price,
            bike_share2_capacity=BikeShare2_capacity,
            bike_share2_price=BikeShare2_price,
            subsidy_dataset=subsidy_dataset,
            subsidy_config=daily_config
        )
        
        print(f"Running simulation for {steps} steps...")
        model.run_model(steps)
        print("Simulation complete.")
        
        # Create analyzer
        analyzer = ABMSensitivityAnalyzer(model=model, output_dir=output_dir)
        
        return model, analyzer
    
    except Exception as e:
        print(f"Error running simulation: {e}")
        traceback.print_exc()
        return None, None


def demo_parameter_interaction_analysis():
    """Run a demonstration of parameter interaction analysis"""
    print("Running parameter interaction analysis demo...")
    
    # Create output directory
    output_dir = "sensitivity_demo_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize analyzer with database connection
    analyzer = ABMSensitivityAnalyzer(
        db_connection_string=DB_CONNECTION_STRING, 
        output_dir=output_dir
    )
    
    # Example 1: Price vs Time Sensitivity (simplified test)
    print("Example 1: Price vs Time Sensitivity")
    price_sensitivity_range = [
        {'beta_C': -0.02, 'beta_T': -0.06, 'beta_W': -0.01, 'beta_A': -0.01, 'alpha': -0.01},
        {'beta_C': -0.15, 'beta_T': -0.06, 'beta_W': -0.01, 'beta_A': -0.01, 'alpha': -0.01}
    ]
    
    time_sensitivity_range = [
        {'beta_C': -0.02, 'beta_T': -0.03},
        {'beta_C': -0.02, 'beta_T': -0.12}
    ]
    
    analyzer.plot_parameter_interaction(
        param1_name='UTILITY_FUNCTION_BASE_COEFFICIENTS',
        param1_range=price_sensitivity_range,
        param2_name='UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS',
        param2_range=time_sensitivity_range,
        outcome_metric='travel_time_equity_index',
        runs_per_cell=2,
        save=True
    )
    
    print("Parameter interaction analysis demo complete!")
    print(f"Results saved to {output_dir}")


def demo_stability_analysis():
    """Run a demonstration of stability analysis"""
    print("Running stability analysis demo...")
    
    # Create output directory
    output_dir = "sensitivity_demo_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize analyzer with database connection
    analyzer = ABMSensitivityAnalyzer(
        db_connection_string=DB_CONNECTION_STRING, 
        output_dir=output_dir
    )
    
    # Basic stability analysis
    print("Basic stability analysis")
    analyzer.plot_stability_analysis(
        num_runs=5,  # Use fewer runs for the demo
        metrics=['travel_time_equity_index'],
        save=True
    )
    
    print("Stability analysis demo complete!")
    print(f"Results saved to {output_dir}")


def main():
    """
    Main function to handle command-line arguments and run sensitivity analyses.
    """
    parser = argparse.ArgumentParser(description='ABM-ETOP Sensitivity Analysis Tool')
    
    # Database connection options
    parser.add_argument('--db_path', type=str, help='Path to the simulation database')
    parser.add_argument('--db_connection_string', type=str, help='Database connection string')
    
    # Simulation options
    parser.add_argument('--run_simulation', action='store_true', help='Run a new simulation')
    parser.add_argument('--steps', type=int, default=100, help='Number of simulation steps')
    
    # Analysis options
    parser.add_argument('--output_dir', type=str, default='sensitivity_analysis_outputs', 
                      help='Directory to save analysis outputs')
    
    # Analysis types
    parser.add_argument('--parameter_interaction', action='store_true', 
                      help='Run parameter interaction analysis')
    parser.add_argument('--stability', action='store_true', 
                      help='Run stability analysis')
    parser.add_argument('--parameter_sensitivity', action='store_true',
                      help='Run parameter sensitivity analysis')
    parser.add_argument('--comprehensive', action='store_true', 
                      help='Run comprehensive sensitivity analysis')
    parser.add_argument('--demo', action='store_true',
                      help='Run a quick demonstration')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize analyzer
    analyzer = None
    model = None
    
    # Run simulation if requested
    if args.run_simulation:
        if MODEL_IMPORTS_AVAILABLE:
            print("Running a new simulation...")
            model, analyzer = run_simulation_with_analyzer(
                steps=args.steps, output_dir=args.output_dir)
        else:
            print("Cannot run simulation: model modules not available.")
            sys.exit(1)
    
    # Connect to existing database if no simulation
    elif args.db_path:
        db_connection_string = f'sqlite:///{args.db_path}'
        analyzer = ABMSensitivityAnalyzer(db_connection_string=db_connection_string, 
                                        output_dir=args.output_dir)
    
    elif args.db_connection_string:
        analyzer = ABMSensitivityAnalyzer(db_connection_string=args.db_connection_string, 
                                        output_dir=args.output_dir)
    
    # Print error if no data source provided
    if analyzer is None and not args.demo:
        print("Error: Please provide a database path (--db_path), connection string (--db_connection_string), or run a simulation (--run_simulation).")
        sys.exit(1)
    
    # Run demo if requested
    if args.demo:
        print("Running demonstration with simplified analysis...")
        demo_parameter_interaction_analysis()
        demo_stability_analysis()
        return
    
    # Run analyses
    if args.comprehensive:
        print("Running comprehensive sensitivity analysis...")
        analyzer.run_comprehensive_sensitivity_analysis()
    else:
        if args.parameter_interaction:
            print("Running parameter interaction analysis...")
            
            # Use reasonable defaults for parameter ranges
            price_sensitivity_range = [
                {'beta_C': -0.02, 'beta_T': -0.06, 'beta_W': -0.01, 'beta_A': -0.01, 'alpha': -0.01},
                {'beta_C': -0.05, 'beta_T': -0.06, 'beta_W': -0.01, 'beta_A': -0.01, 'alpha': -0.01},
                {'beta_C': -0.10, 'beta_T': -0.06, 'beta_W': -0.01, 'beta_A': -0.01, 'alpha': -0.01},
                {'beta_C': -0.15, 'beta_T': -0.06, 'beta_W': -0.01, 'beta_A': -0.01, 'alpha': -0.01}
            ]
            
            time_sensitivity_range = [
                {'beta_C': -0.02, 'beta_T': -0.03},
                {'beta_C': -0.02, 'beta_T': -0.06},
                {'beta_C': -0.02, 'beta_T': -0.09},
                {'beta_C': -0.02, 'beta_T': -0.12}
            ]
            
            analyzer.plot_parameter_interaction(
                param1_name='UTILITY_FUNCTION_BASE_COEFFICIENTS',
                param1_range=price_sensitivity_range,
                param2_name='UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS',
                param2_range=time_sensitivity_range,
                outcome_metric='travel_time_equity_index',
                runs_per_cell=3
            )
        
        if args.stability:
            print("Running stability analysis...")
            analyzer.plot_stability_analysis(
                num_runs=15,
                metrics=['travel_time_equity_index']
            )
            
        if args.parameter_sensitivity:
            print("Running parameter sensitivity analysis...")
            
            # FPS sensitivity
            fps_values = [1000, 2000, 3000, 4000, 5000]
            fps_configs = [SubsidyPoolConfig('daily', v) for v in fps_values]
            
            analyzer.plot_parameter_sensitivity(
                parameter_name='subsidy_config',
                parameter_values=fps_configs,
                metrics=['travel_time_equity_index'],
                runs_per_value=3
            )
    
    print(f"All analyses saved to {args.output_dir}")


if __name__ == "__main__":
    main()