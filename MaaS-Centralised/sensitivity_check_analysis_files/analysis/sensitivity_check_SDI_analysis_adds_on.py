import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import seaborn as sns
from pathlib import Path
import os
import traceback


class SDISensitivityAnalyzer:
    """
    Analyzes sensitivity of SDI metrics to individual parameter variations
    while keeping other parameters fixed.
    """
    def __init__(self, results_data, parameter_history, output_dir):
        """
        Initialize the analyzer with simulation results and parameter history.
        
        Args:
            results_data: List of simulation results
            parameter_history: List of parameter configurations used
            output_dir: Directory to save analysis outputs (passed from SensitivityAnalysisRunner)
        """
        self.results = pd.DataFrame(results_data)
        self.params = pd.DataFrame(parameter_history)
        self.output_dir = Path(output_dir)  # Use the output_dir passed from SensitivityAnalysisRunner
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metric_columns = ['sdi', 'sur', 'mae', 'upi']
    
    def analyze_parameter_impact(self, param, results_df, params_df):
        try:
            if not np.issubdtype(params_df[param].dtype, np.number):
                print(f"Skipping non-numeric parameter: {param}")
                return None
                
            if params_df[param].nunique() <= 1:
                print(f"Skipping constant parameter: {param}")
                return None
                
            # Initialize income_level as None first
            income_level = None
            
            # Extract income level from parameter name if applicable
            param_parts = param.lower().split('_')
            for level in ['high', 'middle', 'low']:
                if level in param_parts:
                    income_level = level
                    results_subset = results_df[results_df['income_level'] == income_level]
                    break
            
            # If no income level found in parameter name, use full results
            if income_level is None:
                results_subset = results_df

            correlations = {}
            for metric in self.metric_columns:
                if metric in results_subset.columns:
                    try:
                        metric_values = results_subset[metric].values
                        param_values = params_df[param].values[:len(results_subset)]
                        
                        if len(np.unique(metric_values)) <= 1 or len(np.unique(param_values)) <= 1:
                            print(f"Skipping correlation for {metric} - constant values")
                            continue
                            
                        corr_coef, p_value = stats.pearsonr(param_values, metric_values)
                        correlations[metric] = {
                            'correlation': corr_coef,
                            'p_value': p_value,
                            'sample_size': len(metric_values)
                        }
                    except Exception as e:
                        print(f"Warning: Could not calculate correlation for {metric}: {e}")
                        continue

            if correlations and len(params_df) >= 5:
                try:
                    plt.figure(figsize=(15, 10))
                    
                    for i, metric in enumerate(self.metric_columns):
                        if metric in results_subset.columns:
                            plt.subplot(2, 2, i + 1)
                            
                            plot_data = pd.DataFrame({
                                'param': params_df[param].values[:len(results_subset)],
                                'metric': results_subset[metric].values
                            })
                            
                            # Simple scatter plot without income level hue
                            sns.scatterplot(
                                data=plot_data,
                                x='param',
                                y='metric',
                                alpha=0.6
                            )
                            
                            z = np.polyfit(plot_data['param'], plot_data['metric'], 2)
                            p = np.poly1d(z)
                            x_range = np.linspace(plot_data['param'].min(), plot_data['param'].max(), 100)
                            plt.plot(x_range, p(x_range), '--', color='red', alpha=0.8)
                            
                            title_suffix = f" ({income_level.capitalize()} Income)" if income_level else ""
                            plt.title(f'{metric.upper()} vs {param}{title_suffix}')
                            plt.xlabel(param)
                            plt.ylabel(metric.upper())
                            
                            if metric in correlations:
                                corr = correlations[metric]['correlation']
                                p_val = correlations[metric]['p_value']
                                text = f'Correlation: {corr:.3f}\np-value: {p_val:.3f}'
                                plt.text(0.05, 0.95, text, 
                                    transform=plt.gca().transAxes,
                                    bbox=dict(facecolor='white', alpha=0.8))

                    plt.tight_layout()
                    plt.savefig(os.path.join(self.output_dir, f'impact_{param}.png'))
                    plt.close()

                    # Only create parameter distribution
                    plt.figure(figsize=(8, 6))
                    sns.histplot(params_df[param], kde=True)
                    plt.title(f'Distribution of {param}')
                    plt.xlabel(param)
                    plt.ylabel('Count')
                    plt.savefig(os.path.join(self.output_dir, f'histogram_{param}.png'))
                    plt.close()

                except Exception as plot_err:
                    print(f"Warning: Error creating plots for {param}: {plot_err}")
                
                return {
                    'correlations': correlations,
                    'parameter_range': {
                        'min': float(params_df[param].min()),
                        'max': float(params_df[param].max()),
                        'mean': float(params_df[param].mean()),
                        'std': float(params_df[param].std())
                    },
                    'sample_sizes': {
                        'total': len(params_df),
                        'filtered': len(results_subset)
                    }
                }
                
            return None
            
        except Exception as e:
            print(f"Error analyzing parameter {param}: {e}")
            traceback.print_exc()
            return None
    def _plot_vot_specific_analysis(self, param, results_df, params_df):
        """Helper function for VOT-specific visualizations"""
        try:
            plt.figure(figsize=(10, 6))
            
            # Create scatter plot with regression line for VOT impact
            income_level = param.split('_')[1]
            sns.regplot(
                x=params_df[param].values[:len(results_df)],
                y=results_df['sdi'],
                scatter_kws={'alpha':0.5},
                line_kws={'color': 'red'}
            )
            
            plt.title(f'SDI vs Value of Time ({income_level.capitalize()} Income)')
            plt.xlabel('Value of Time')
            plt.ylabel('SDI Score')
            
            plt.savefig(os.path.join(self.output_dir, f'vot_specific_{param}.png'))
            plt.close()
            
        except Exception as e:
            print(f"Warning: Could not create VOT-specific plot for {param}: {e}")

    def _calculate_elasticity(self, param_values, metric_scores):
        """Calculate elasticity (% change in output / % change in input)."""
        param_pct_change = np.diff(param_values) / param_values[:-1]
        metric_pct_change = np.diff(metric_scores) / metric_scores[:-1]
        
        # Handle division by zero
        valid_mask = ~np.isnan(param_pct_change) & ~np.isnan(metric_pct_change)
        if not np.any(valid_mask):
            return 0
            
        elasticity = np.mean(metric_pct_change[valid_mask] / 
                           param_pct_change[valid_mask])
        return elasticity

    def _find_threshold_points(self, param_values, metric_scores):
        """Identify critical threshold points where metric behavior changes."""
        # Calculate rate of change
        dx = np.diff(param_values)
        dy = np.diff(metric_scores)
        slopes = dy / dx
        
        # Find points where slope changes significantly
        mean_slope = np.mean(slopes)
        std_slope = np.std(slopes)
        threshold_indices = np.where(abs(slopes - mean_slope) > 2 * std_slope)[0]
        
        thresholds = []
        for idx in threshold_indices:
            thresholds.append({
                'parameter_value': param_values[idx],
                'metric_value': metric_scores[idx],
                'slope_change': slopes[idx]
            })
            
        return thresholds

    def _calculate_stability_range(self, param_values, metric_scores):
        """Calculate range where metric remains stable."""
        # Define stability as range where metric varies within 1 std dev
        mean_score = np.mean(metric_scores)
        std_score = np.std(metric_scores)
        
        stable_mask = (metric_scores >= mean_score - std_score) & \
                     (metric_scores <= mean_score + std_score)
                     
        if not np.any(stable_mask):
            return None
            
        return {
            'min_param': np.min(param_values[stable_mask]),
            'max_param': np.max(param_values[stable_mask]),
            'mean_metric': mean_score,
            'std_metric': std_score
        }
