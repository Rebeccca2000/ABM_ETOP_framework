import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import seaborn as sns
from pathlib import Path

class SDISensitivityAnalyzer:
    """
    Analyzes sensitivity of SDI metrics to individual parameter variations
    while keeping other parameters fixed.
    """
    def __init__(self, results_data, parameter_history, output_dir='sdi_sensitivity_analysis'):
        """
        Initialize the analyzer with simulation results and parameter history.
        
        Args:
            results_data: List of simulation results
            parameter_history: List of parameter configurations used
            output_dir: Directory to save analysis outputs
        """
        self.results = pd.DataFrame(results_data)
        self.params = pd.DataFrame(parameter_history)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def analyze_parameter_impact(self, param_name, metric='SDI'):
        """
        Analyze how a single parameter affects SDI metrics.
        
        Args:
            param_name: Name of parameter to analyze
            metric: SDI component to analyze ('SDI', 'SUR', 'MAE', 'UPI')
            
        Returns:
            dict: Analysis results including statistical measures
        """
        # Extract parameter values and corresponding metric scores
        param_values = self.params[param_name].values
        metric_scores = self.results[metric].values
        
        # Calculate correlation
        correlation = stats.pearsonr(param_values, metric_scores)
        
        # Fit regression model
        def sigmoid(x, L, k, x0):
            return L / (1 + np.exp(-k * (x - x0)))
            
        try:
            popt, _ = curve_fit(sigmoid, param_values, metric_scores)
            fitted_curve = sigmoid(np.sort(param_values), *popt)
        except:
            popt = None
            fitted_curve = None
            
        # Calculate sensitivity metrics
        sensitivity = {
            'correlation': correlation[0],
            'p_value': correlation[1],
            'elasticity': self._calculate_elasticity(param_values, metric_scores),
            'threshold_points': self._find_threshold_points(param_values, metric_scores),
            'stability_range': self._calculate_stability_range(param_values, metric_scores)
        }
        
        # Create visualization
        self._plot_parameter_impact(param_values, metric_scores, 
                                  fitted_curve, param_name, metric)
        
        return sensitivity

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

    def _plot_parameter_impact(self, param_values, metric_scores, 
                             fitted_curve, param_name, metric):
        """Create visualization of parameter impact."""
        plt.figure(figsize=(10, 6))
        
        # Scatter plot of actual data
        plt.scatter(param_values, metric_scores, alpha=0.6, 
                   label='Simulation Data')
        
        # Add fitted curve if available
        if fitted_curve is not None:
            plt.plot(np.sort(param_values), fitted_curve, 'r--', 
                    label='Fitted Curve')
        
        # Add stability range
        stability = self._calculate_stability_range(param_values, metric_scores)
        if stability:
            plt.axhspan(stability['mean_metric'] - stability['std_metric'],
                       stability['mean_metric'] + stability['std_metric'],
                       alpha=0.2, color='green', label='Stability Range')
        
        plt.xlabel(f'{param_name} Value')
        plt.ylabel(f'{metric} Score')
        plt.title(f'Impact of {param_name} on {metric}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig(self.output_dir / f'{param_name}_{metric}_impact.png')
        plt.close()

    def analyze_time_dependence(self, param):
        """Analyze how parameter impact evolves over simulation time"""
        try:
            # Ensure we have time series data
            if 'step' not in self.results_df.columns:
                print("Error: No time step data available")
                return None
                
            # Group data by timestep and calculate metrics
            temporal_data = []
            for step in self.results_df['step'].unique():
                step_data = self.results_df[self.results_df['step'] == step]
                param_values = step_data[param].values
                sdi_values = step_data['sdi'].values
                
                # Calculate correlation only if we have sufficient variation
                if len(np.unique(param_values)) > 1:
                    corr_coef, p_value = stats.pearsonr(param_values, sdi_values)
                    temporal_data.append({
                        'step': step,
                        'correlation': corr_coef,
                        'p_value': p_value,
                        'num_samples': len(param_values),
                        'param_std': np.std(param_values),
                        'sdi_std': np.std(sdi_values)
                    })
            
            return pd.DataFrame(temporal_data) if temporal_data else None

        except Exception as e:
            print(f"Error in temporal analysis for {param}: {str(e)}")
            return None