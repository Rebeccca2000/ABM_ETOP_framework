import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wasserstein_distance, entropy
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from statsmodels.tsa.stattools import acf, ccf
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import logging
from datetime import datetime
import json
import pickle
from dataclasses import dataclass
from tqdm import tqdm
import pickle
#22 Jan 2024

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ValidationMetrics:
    """Container for validation metrics"""
    r2: float
    rmse: float
    mae: float
    mape: Optional[float]
    correlation: float
    ks_statistic: float
    ks_pvalue: float
    wasserstein_distance: float
    temporal_metrics: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]

    def to_dict(self) -> Dict:
        """Convert validation metrics to dictionary"""
        return {
            'r2': self.r2,
            'rmse': self.rmse,
            'mae': self.mae,
            'mape': self.mape,
            'correlation': self.correlation,
            'ks_statistic': self.ks_statistic,
            'ks_pvalue': self.ks_pvalue,
            'wasserstein_distance': self.wasserstein_distance,
            'temporal_metrics': self.temporal_metrics,
            'confidence_intervals': self.confidence_intervals
        }

class ValidationFramework:
    """
    Comprehensive validation framework for SPF analysis
    
    Attributes:
        spf_analyzer: SPF analysis object
        abm_results: ABM simulation results
        metrics: List of metrics to validate
        temporal_periods: List of temporal periods
        output_dir: Directory for validation outputs
    """
    
    def __init__(self, 
                 spf_analyzer: Any,
                 abm_results: Dict,
                 output_dir: Optional[str] = None,
                 confidence_level: float = 0.95,
                 n_bootstrap: int = 1000,
                 random_state: int = 42):
        """
        Initialize validation framework
        
        Args:
            spf_analyzer: SPF analysis object
            abm_results: Dictionary containing ABM simulation results
            output_dir: Directory for validation outputs
            confidence_level: Confidence level for intervals
            n_bootstrap: Number of bootstrap iterations
            random_state: Random state for reproducibility
        """
        self.spf = spf_analyzer
        self.abm_results = abm_results
        self.metrics = ['SDI', 'TEI', 'FSIR', 'BSR']
        self.temporal_periods = ['morning_peak', 'evening_peak', 'off_peak']
        self.output_dir = Path(output_dir) if output_dir else Path('validation_outputs')
        self.confidence_level = confidence_level
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Validate inputs
        self._validate_inputs()
        
        np.random.seed(random_state)
        logger.info("Validation Framework initialized successfully")

    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = self.output_dir / f'validation_{datetime.now():%Y%m%d_%H%M%S}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)

    def _validate_inputs(self):
        """Validate input data"""
        try:
            # Check ABM results
            for metric in self.metrics:
                assert metric in self.abm_results, f"Missing ABM results for metric: {metric}"
                
            # Check temporal data
            assert 'temporal' in self.abm_results, "Missing temporal data in ABM results"
            for period in self.temporal_periods:
                assert period in self.abm_results['temporal'], f"Missing data for period: {period}"
                
            # Check SPF analyzer
            assert hasattr(self.spf, 'predict'), "SPF analyzer must have predict method"
            
        except AssertionError as e:
            logger.error(f"Validation error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during validation: {str(e)}")
            raise

    def validate_comprehensive(self) -> Dict[str, Any]:
        """
        Perform comprehensive validation of SPF against ABM results
        
        Returns:
            Dict containing all validation results
        """
        logger.info("Starting comprehensive validation")
        
        try:
            validation_results = {
                'statistical': self.statistical_validation(),
                'behavioral': self.behavioral_validation(),
                'temporal': self.temporal_validation(),
                'robustness': self.robustness_validation(),
                'cross_validation': self.perform_cross_validation()
            }
            
            # Add summary metrics
            validation_results['summary'] = self._calculate_summary_metrics(
                validation_results
            )
            
            # Generate visualizations
            self.generate_validation_plots(validation_results)
            
            # Save results
            self._save_validation_results(validation_results)
            
            logger.info("Comprehensive validation completed successfully")
            return validation_results
            
        except Exception as e:
            logger.error(f"Error in comprehensive validation: {str(e)}")
            raise

    def statistical_validation(self) -> Dict[str, ValidationMetrics]:
        """
        Perform comprehensive statistical validation
        
        Returns:
            Dict containing statistical validation metrics for each metric
        """
        logger.info("Starting statistical validation")
        statistical_metrics = {}

        try:
            for metric in tqdm(self.metrics, desc="Performing statistical validation"):
                # Get predictions and actual values
                predictions = self.spf.predict_metric(metric)
                actual = self.abm_results[metric]
                
                # Basic regression metrics
                r2 = r2_score(actual, predictions)
                rmse = np.sqrt(mean_squared_error(actual, predictions))
                mae = mean_absolute_error(actual, predictions)
                
                # Calculate MAPE if no zero values
                if not np.any(actual == 0):
                    mape = np.mean(np.abs((actual - predictions) / actual)) * 100
                else:
                    mape = None
                
                # Correlation analysis
                correlation = np.corrcoef(actual, predictions)[0, 1]
                
                # Distribution comparison
                ks_statistic, ks_pvalue = stats.ks_2samp(actual, predictions)
                w_distance = wasserstein_distance(actual, predictions)
                
                # Temporal metrics
                temporal_metrics = self._calculate_temporal_metrics(metric)
                
                # Confidence intervals
                confidence_intervals = self._calculate_confidence_intervals(
                    actual, predictions
                )
                
                # Store metrics
                statistical_metrics[metric] = ValidationMetrics(
                    r2=r2,
                    rmse=rmse,
                    mae=mae,
                    mape=mape,
                    correlation=correlation,
                    ks_statistic=ks_statistic,
                    ks_pvalue=ks_pvalue,
                    wasserstein_distance=w_distance,
                    temporal_metrics=temporal_metrics,
                    confidence_intervals=confidence_intervals
                )
                
                logger.info(f"Completed statistical validation for {metric}")
                
            return statistical_metrics
            
        except Exception as e:
            logger.error(f"Error in statistical validation: {str(e)}")
            raise

    def behavioral_validation(self) -> Dict[str, Dict[str, Any]]:
        """
        Validate behavioral patterns and emergent properties
        
        Returns:
            Dict containing behavioral validation results
        """
        logger.info("Starting behavioral validation")
        behavioral_metrics = {}

        try:
            for metric in tqdm(self.metrics, desc="Performing behavioral validation"):
                # Compare distributions
                distribution_metrics = self._compare_distributions(metric)
                
                # Validate patterns
                pattern_metrics = self._validate_patterns(metric)
                
                # Validate emergent properties
                emergent_properties = self._validate_emergent_properties(metric)
                
                # Validate response characteristics
                response_characteristics = self._validate_response_characteristics(metric)
                
                behavioral_metrics[metric] = {
                    'distribution_comparison': distribution_metrics,
                    'pattern_validation': pattern_metrics,
                    'emergent_properties': emergent_properties,
                    'response_characteristics': response_characteristics,
                    'summary': self._summarize_behavioral_metrics(
                        distribution_metrics,
                        pattern_metrics,
                        emergent_properties,
                        response_characteristics
                    )
                }
                
            return behavioral_metrics
            
        except Exception as e:
            logger.error(f"Error in behavioral validation: {str(e)}")
            raise

    def temporal_validation(self) -> Dict[str, Dict[str, Any]]:
        """
        Validate temporal aspects of predictions
        
        Returns:
            Dict containing temporal validation results
        """
        logger.info("Starting temporal validation")
        temporal_metrics = {}

        try:
            for period in tqdm(self.temporal_periods, desc="Analyzing temporal periods"):
                period_metrics = {}
                
                for metric in self.metrics:
                    # Get temporal predictions and actual values
                    predictions = self.spf.predict_temporal(metric, period)
                    actual = self.abm_results['temporal'][period][metric]
                    
                    # Calculate temporal accuracy metrics
                    accuracy_metrics = self._calculate_temporal_accuracy(
                        actual, predictions
                    )
                    
                    # Analyze temporal patterns
                    pattern_analysis = self._analyze_temporal_patterns(
                        actual, predictions
                    )
                    
                    # Validate temporal stability
                    stability_metrics = self._validate_temporal_stability(
                        actual, predictions
                    )
                    
                    period_metrics[metric] = {
                        'accuracy': accuracy_metrics,
                        'patterns': pattern_analysis,
                        'stability': stability_metrics
                    }
                
                temporal_metrics[period] = period_metrics
                
            # Add cross-period analysis
            temporal_metrics['cross_period'] = self._analyze_cross_period_validation()
            
            return temporal_metrics
            
        except Exception as e:
            logger.error(f"Error in temporal validation: {str(e)}")
            raise

    def _calculate_temporal_metrics(self, metric: str) -> Dict[str, float]:
        """Calculate temporal validation metrics"""
        temporal_metrics = {}
        
        try:
            for period in self.temporal_periods:
                # Get temporal data
                actual = self.abm_results['temporal'][period][metric]
                predicted = self.spf.predict_temporal(metric, period)
                
                # Calculate metrics for this period
                temporal_metrics[period] = {
                    'rmse': np.sqrt(mean_squared_error(actual, predicted)),
                    'correlation': np.corrcoef(actual, predicted)[0, 1],
                    'autocorr_diff': self._compare_autocorrelation(actual, predicted),
                    'pattern_similarity': self._calculate_pattern_similarity(actual, predicted)
                }
                
            return temporal_metrics
            
        except Exception as e:
            logger.error(f"Error calculating temporal metrics: {str(e)}")
            return {}

    def _compare_distributions(self, metric: str) -> Dict[str, float]:
        """Compare distributions between ABM and SPF results"""
        try:
            actual = self.abm_results[metric]
            predicted = self.spf.predict_metric(metric)
            
            return {
                'ks_statistic': stats.ks_2samp(actual, predicted)[0],
                'wasserstein': wasserstein_distance(actual, predicted),
                'jensen_shannon': self._calculate_jensen_shannon(actual, predicted),
                'histogram_intersection': self._calculate_histogram_intersection(
                    actual, predicted
                )
            }
            
        except Exception as e:
            logger.error(f"Error comparing distributions: {str(e)}")
            return {}
    
    def robustness_validation(self) -> Dict[str, Dict[str, Any]]:
        """
        Validate robustness of SPF predictions
        
        Returns:
            Dict containing robustness validation results
        """
        logger.info("Starting robustness validation")
        robustness_results = {}

        try:
            for metric in tqdm(self.metrics, desc="Performing robustness validation"):
                # Parameter variation analysis
                parameter_sensitivity = self._analyze_parameter_sensitivity(metric)
                
                # Noise resistance analysis
                noise_resistance = self._analyze_noise_resistance(metric)
                
                # Stability analysis
                stability_metrics = self._analyze_prediction_stability(metric)
                
                # Boundary condition analysis
                boundary_analysis = self._analyze_boundary_conditions(metric)
                
                # Bootstrap analysis
                bootstrap_results = self._perform_bootstrap_analysis(
                    metric, n_iterations=self.n_bootstrap
                )
                
                robustness_results[metric] = {
                    'parameter_sensitivity': parameter_sensitivity,
                    'noise_resistance': noise_resistance,
                    'stability': stability_metrics,
                    'boundary_analysis': boundary_analysis,
                    'bootstrap_results': bootstrap_results,
                    'summary': self._calculate_robustness_score(
                        parameter_sensitivity,
                        noise_resistance,
                        stability_metrics,
                        boundary_analysis,
                        bootstrap_results
                    )
                }
                
            return robustness_results
            
        except Exception as e:
            logger.error(f"Error in robustness validation: {str(e)}")
            raise

    def perform_cross_validation(self, n_splits: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Perform k-fold cross-validation
        
        Args:
            n_splits: Number of folds for cross-validation
            
        Returns:
            Dict containing cross-validation results
        """
        logger.info(f"Starting {n_splits}-fold cross-validation")
        cv_results = {}

        try:
            for metric in tqdm(self.metrics, desc="Performing cross-validation"):
                fold_metrics = []
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
                
                # Get all data for this metric
                X = self._get_feature_data()
                y = self.abm_results[metric]
                
                for train_idx, test_idx in kf.split(X):
                    # Train SPF on training data
                    self.spf.train(X[train_idx], y[train_idx])
                    
                    # Make predictions on test data
                    predictions = self.spf.predict(X[test_idx])
                    actual = y[test_idx]
                    
                    # Calculate metrics for this fold
                    fold_metrics.append({
                        'r2': r2_score(actual, predictions),
                        'rmse': np.sqrt(mean_squared_error(actual, predictions)),
                        'mae': mean_absolute_error(actual, predictions)
                    })
                
                # Calculate average metrics across folds
                cv_results[metric] = {
                    metric: np.mean([fold[metric] for fold in fold_metrics])
                    for metric in fold_metrics[0].keys()
                }
                
                # Add standard deviation of metrics
                cv_results[metric].update({
                    f"{metric}_std": np.std([fold[metric] for fold in fold_metrics])
                    for metric in fold_metrics[0].keys()
                })
                
            return cv_results
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {str(e)}")
            raise

    def _analyze_parameter_sensitivity(self, metric: str) -> Dict[str, Any]:
        """Analyze sensitivity to parameter variations"""
        try:
            # Get baseline predictions
            baseline = self.spf.predict_metric(metric)
            parameter_ranges = self.spf.get_parameter_ranges()
            
            sensitivity_results = {}
            for param, range_info in parameter_ranges.items():
                variations = np.linspace(
                    range_info['min'],
                    range_info['max'],
                    10
                )
                
                predictions = []
                for value in variations:
                    # Make prediction with varied parameter
                    pred = self.spf.predict_with_param(metric, param, value)
                    predictions.append(pred)
                
                # Calculate sensitivity metrics
                sensitivity_results[param] = {
                    'variation_range': variations.tolist(),
                    'prediction_range': [pred.tolist() for pred in predictions],
                    'sensitivity_score': self._calculate_sensitivity_score(
                        baseline, predictions
                    )
                }
                
            return sensitivity_results
            
        except Exception as e:
            logger.error(f"Error in parameter sensitivity analysis: {str(e)}")
            return {}

    def _analyze_noise_resistance(self, metric: str) -> Dict[str, Any]:
        """Analyze resistance to input noise"""
        try:
            # Get baseline predictions
            baseline = self.spf.predict_metric(metric)
            
            noise_levels = [0.01, 0.05, 0.1, 0.2]
            noise_results = {}
            
            for noise in noise_levels:
                noisy_predictions = []
                for _ in range(50):  # Multiple trials per noise level
                    # Add noise to input data
                    noisy_data = self._add_noise(self._get_feature_data(), noise)
                    
                    # Make prediction with noisy data
                    pred = self.spf.predict(noisy_data)
                    noisy_predictions.append(pred)
                
                # Calculate stability metrics
                noise_results[noise] = {
                    'mean_deviation': np.mean([
                        np.abs(pred - baseline) for pred in noisy_predictions
                    ]),
                    'std_deviation': np.std([
                        np.abs(pred - baseline) for pred in noisy_predictions
                    ]),
                    'max_deviation': np.max([
                        np.abs(pred - baseline) for pred in noisy_predictions
                    ])
                }
                
            return noise_results
            
        except Exception as e:
            logger.error(f"Error in noise resistance analysis: {str(e)}")
            return {}

    def _analyze_prediction_stability(self, metric: str) -> Dict[str, float]:
        """Analyze stability of predictions"""
        try:
            predictions = self.spf.predict_metric(metric)
            
            return {
                'coefficient_of_variation': np.std(predictions) / np.mean(predictions),
                'range_ratio': (np.max(predictions) - np.min(predictions)) / np.mean(predictions),
                'stability_score': self._calculate_stability_score(predictions)
            }
            
        except Exception as e:
            logger.error(f"Error in prediction stability analysis: {str(e)}")
            return {}
        
    def generate_validation_plots(self, results: Dict[str, Any]):
        """Generate comprehensive validation visualization suite"""
        logger.info("Generating validation visualizations")
        
        try:
            # Create visualization directories
            plot_dirs = {
                'statistical': self.output_dir / 'statistical_validation',
                'behavioral': self.output_dir / 'behavioral_validation',
                'temporal': self.output_dir / 'temporal_validation',
                'robustness': self.output_dir / 'robustness_validation',
                'summary': self.output_dir / 'validation_summary'
            }
            
            for directory in plot_dirs.values():
                directory.mkdir(exist_ok=True)

            # Generate plots for each validation type
            self._plot_statistical_validation(results['statistical'], plot_dirs['statistical'])
            self._plot_behavioral_validation(results['behavioral'], plot_dirs['behavioral'])
            self._plot_temporal_validation(results['temporal'], plot_dirs['temporal'])
            self._plot_robustness_validation(results['robustness'], plot_dirs['robustness'])
            self._plot_validation_summary(results, plot_dirs['summary'])
            
        except Exception as e:
            logger.error(f"Error generating validation plots: {str(e)}")
            raise

    def _plot_statistical_validation(self, results: Dict, output_dir: Path):
        """Generate statistical validation plots"""
        for metric in self.metrics:
            # Scatter plot with regression line
            plt.figure(figsize=(10, 8))
            actual = self.abm_results[metric]
            predicted = self.spf.predict_metric(metric)
            
            plt.scatter(actual, predicted, alpha=0.5)
            
            # Add regression line
            z = np.polyfit(actual, predicted, 1)
            p = np.poly1d(z)
            plt.plot(actual, p(actual), "r--", alpha=0.8)
            
            # Add perfect prediction line
            plt.plot([min(actual), max(actual)], 
                    [min(actual), max(actual)], 
                    'k:', alpha=0.5)
            
            plt.title(f'Predicted vs Actual Values - {metric}')
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            
            # Add metrics to plot
            metrics_text = (
                f"R² = {results[metric].r2:.3f}\n"
                f"RMSE = {results[metric].rmse:.3f}\n"
                f"MAE = {results[metric].mae:.3f}"
            )
            plt.text(0.05, 0.95, metrics_text,
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(output_dir / f'scatter_{metric}.png')
            plt.close()

    def _plot_temporal_validation(self, results: Dict, output_dir: Path):
        """Generate temporal validation plots"""
        for period in self.temporal_periods:
            for metric in self.metrics:
                plt.figure(figsize=(15, 6))
                
                # Get temporal data
                actual = self.abm_results['temporal'][period][metric]
                predicted = self.spf.predict_temporal(metric, period)
                
                # Plot time series
                plt.plot(actual, 'b-', label='Actual', alpha=0.7)
                plt.plot(predicted, 'r--', label='Predicted', alpha=0.7)
                
                plt.title(f'Temporal Validation - {metric} ({period})')
                plt.xlabel('Time Step')
                plt.ylabel('Value')
                plt.legend()
                
                # Add metrics
                metrics = results[period][metric]['accuracy']
                metrics_text = '\n'.join([
                    f"{k}: {v:.3f}" for k, v in metrics.items()
                ])
                plt.text(0.05, 0.95, metrics_text,
                        transform=plt.gca().transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                plt.tight_layout()
                plt.savefig(output_dir / f'temporal_{metric}_{period}.png')
                plt.close()

    def _save_validation_results(self, results: Dict):
        """Save validation results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Save full results as pickle
            with open(self.output_dir / f'validation_results_{timestamp}.pkl', 'wb') as f:
                pickle.dump(results, f)
            
            # Save summary as JSON
            summary = self._create_validation_summary(results)
            with open(self.output_dir / f'validation_summary_{timestamp}.json', 'w') as f:
                json.dump(summary, f, indent=4)
            
            # Save detailed metrics as CSV
            detailed_metrics = self._extract_detailed_metrics(results)
            pd.DataFrame(detailed_metrics).to_csv(
                self.output_dir / f'validation_metrics_{timestamp}.csv'
            )
            
            logger.info(f"Validation results saved to {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving validation results: {str(e)}")
            raise

    def _create_validation_summary(self, results: Dict) -> Dict:
        """Create summary of validation results"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'metrics_validated': self.metrics,
            'overall_performance': {},
            'temporal_performance': {},
            'robustness_metrics': {},
            'key_findings': []
        }
        
        for metric in self.metrics:
            # Overall performance metrics
            summary['overall_performance'][metric] = {
                'r2': results['statistical'][metric].r2,
                'rmse': results['statistical'][metric].rmse,
                'correlation': results['statistical'][metric].correlation
            }
            
            # Temporal performance
            summary['temporal_performance'][metric] = {
                period: results['temporal'][period][metric]['accuracy']
                for period in self.temporal_periods
            }
            
            # Robustness metrics
            summary['robustness_metrics'][metric] = results['robustness'][metric]['summary']
        
        # Add key findings
        summary['key_findings'] = self._generate_key_findings(results)
        
        return summary

    @staticmethod
    def _generate_key_findings(results: Dict) -> List[str]:
        """Generate key findings from validation results"""
        findings = []
        
        # Analyze overall performance
        avg_r2 = np.mean([
            results['statistical'][m].r2 for m in results['statistical']
        ])
        findings.append(f"Average R² across all metrics: {avg_r2:.3f}")
        
        # Analyze temporal performance
        temporal_accuracy = np.mean([
            np.mean([
                results['temporal'][p][m]['accuracy']['rmse']
                for p in results['temporal'] if p != 'cross_period'
            ])
            for m in results['statistical']
        ])
        findings.append(f"Average temporal RMSE: {temporal_accuracy:.3f}")
        
        # Add robustness findings
        findings.append("Robustness Analysis Findings:")
        for metric in results['robustness']:
            score = results['robustness'][metric]['summary']
            findings.append(f"- {metric}: Robustness Score = {score:.3f}")
        
        return findings

if __name__ == "__main__":
    # Example usage
    from your_spf_analyzer import SPFAnalyzer
    from your_abm_results import ABMResults
    
    # Initialize validation framework
    validator = ValidationFramework(
        spf_analyzer=SPFAnalyzer(),
        abm_results=ABMResults(),
        output_dir='validation_results',
        confidence_level=0.95,
        n_bootstrap=1000,
        random_state=42
    )
    
    # Run comprehensive validation
    validation_results = validator.validate_comprehensive()
    
    # Access specific results
    statistical_results = validation_results['statistical']
    temporal_results = validation_results['temporal']
    robustness_results = validation_results['robustness']