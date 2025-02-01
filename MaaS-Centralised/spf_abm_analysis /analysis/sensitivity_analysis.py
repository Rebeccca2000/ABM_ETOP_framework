from SALib.sample import saltelli, latin, fast_sampler
from SALib.analyze import sobol, delta, fast, rbd_fast
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, gaussian_kde
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
import os
from pathlib import Path
import logging
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import json
from tqdm import tqdm
import pickle
import itertools


# 22th Jan. 
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SensitivityResult:
    """Container for sensitivity analysis results"""
    first_order: Dict[str, float]
    total_order: Dict[str, float]
    interaction_effects: Dict[str, Dict[str, float]]
    temporal_variations: Dict[str, Dict[str, float]]
    confidence_intervals: Dict[str, Tuple[float, float]]
    parameter_rankings: Dict[str, int]
    stability_metrics: Dict[str, float]
    
    def to_dict(self) -> Dict:
        """Convert results to dictionary for serialization"""
        return {
            'first_order': self.first_order,
            'total_order': self.total_order,
            'interaction_effects': self.interaction_effects,
            'temporal_variations': self.temporal_variations,
            'confidence_intervals': {k: list(v) for k, v in self.confidence_intervals.items()},
            'parameter_rankings': self.parameter_rankings,
            'stability_metrics': self.stability_metrics
        }

class SensitivityAnalyzer:
    """
    Comprehensive sensitivity analysis for transportation system parameters
    
    Attributes:
        parameter_space: Parameter space definition
        results_data: Simulation results data
        metrics: List of metrics to analyze
        temporal_periods: List of temporal periods
        output_dir: Directory for output files
    """
    
    def __init__(self, 
                 parameter_space: Any,
                 results_data: Dict,
                 output_dir: Optional[str] = None,
                 n_samples: int = 1024,
                 n_bootstrap: int = 100,
                 confidence_level: float = 0.95):
        """
        Initialize sensitivity analyzer
        
        Args:
            parameter_space: Parameter space definition
            results_data: Dictionary containing simulation results
            output_dir: Directory for output files
            n_samples: Number of samples for sensitivity analysis
            n_bootstrap: Number of bootstrap iterations
            confidence_level: Confidence level for intervals
        """
        self.parameter_space = parameter_space
        self.results_data = results_data
        self.metrics = ['SDI', 'TEI', 'FSIR', 'BSR']
        self.temporal_periods = ['morning_peak', 'evening_peak', 'off_peak']
        self.output_dir = Path(output_dir) if output_dir else Path('sensitivity_analysis_outputs')
        self.n_samples = n_samples
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self._setup_logging()
        
        # Validate inputs
        self._validate_inputs()
        
        logger.info("Sensitivity Analyzer initialized successfully")

    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = self.output_dir / f'sensitivity_analysis_{datetime.now():%Y%m%d_%H%M%S}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)

    def _validate_inputs(self):
        """Validate input data and parameters"""
        try:
            # Check parameter space
            assert hasattr(self.parameter_space, 'get_bounds'), "Parameter space must have get_bounds method"
            
            # Check results data
            for metric in self.metrics:
                assert metric in self.results_data, f"Missing results for metric: {metric}"
            
            # Check temporal data
            for period in self.temporal_periods:
                assert period in self.results_data['temporal'], f"Missing temporal data for period: {period}"
                
        except AssertionError as e:
            logger.error(f"Validation error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during validation: {str(e)}")
            raise

    def conduct_comprehensive_analysis(self) -> Dict[str, Any]:
        """
        Conduct comprehensive sensitivity analysis across all metrics
        
        Returns:
            Dict containing all sensitivity analysis results
        """
        try:
            logger.info("Starting comprehensive sensitivity analysis")
            
            sensitivity_results = {
                'global': self.perform_global_sensitivity(),
                'local': self.perform_local_sensitivity(),
                'temporal': self.analyze_temporal_sensitivity(),
                'interaction': self.analyze_parameter_interactions(),
                'robustness': self.perform_robustness_analysis()
            }
            
            # Additional analyses
            sensitivity_results.update({
                'parameter_importance': self._analyze_parameter_importance(),
                'cross_metric_effects': self._analyze_cross_metric_effects(),
                'stability_analysis': self._analyze_stability()
            })
            
            # Generate visualizations
            self.generate_sensitivity_plots(sensitivity_results)
            
            # Save results
            self._save_results(sensitivity_results)
            
            logger.info("Comprehensive sensitivity analysis completed successfully")
            return sensitivity_results
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {str(e)}")
            raise

    def perform_global_sensitivity(self) -> Dict[str, SensitivityResult]:
        """
        Perform global sensitivity analysis using multiple methods
        
        Returns:
            Dict containing global sensitivity results for each metric
        """
        logger.info("Starting global sensitivity analysis")
        global_sensitivity = {}

        try:
            for metric in tqdm(self.metrics, desc="Analyzing metrics"):
                # Define problem structure for SALib
                problem = {
                    'num_vars': self._count_parameters(),
                    'names': self._get_parameter_names(),
                    'bounds': self._get_parameter_bounds(),
                    'dists': self._get_parameter_distributions()
                }

                # Generate samples using multiple methods
                samples = {
                    'sobol': saltelli.sample(problem, n=self.n_samples, calc_second_order=True),
                    'fast': fast_sampler.sample(problem, n=self.n_samples),
                    'rbd': latin.sample(problem, n=self.n_samples)
                }

                # Calculate model outputs for each sampling method
                outputs = {}
                for method, sample in samples.items():
                    Y = np.array([self._evaluate_metric(metric, X) for X in sample])
                    outputs[method] = Y

                # Perform analyses using different methods
                sobol_results = self._perform_sobol_analysis(problem, outputs['sobol'])
                fast_results = self._perform_fast_analysis(problem, outputs['fast'])
                rbd_results = self._perform_rbd_analysis(problem, outputs['rbd'])

                # Combine and process results
                combined_results = self._combine_sensitivity_results(
                    sobol_results, fast_results, rbd_results
                )

                # Calculate confidence intervals and rankings
                confidence_intervals = self._calculate_confidence_intervals(
                    problem, outputs, combined_results
                )
                parameter_rankings = self._rank_parameters(combined_results)

                # Create SensitivityResult object
                global_sensitivity[metric] = SensitivityResult(
                    first_order=combined_results['first_order'],
                    total_order=combined_results['total_order'],
                    interaction_effects=combined_results['interaction_effects'],
                    temporal_variations=self._analyze_temporal_variations(metric, samples['sobol']),
                    confidence_intervals=confidence_intervals,
                    parameter_rankings=parameter_rankings,
                    stability_metrics=self._calculate_stability_metrics(combined_results)
                )

                # Log progress
                logger.info(f"Completed global sensitivity analysis for {metric}")

            return global_sensitivity

        except Exception as e:
            logger.error(f"Error in global sensitivity analysis: {str(e)}")
            raise

    def _perform_sobol_analysis(self, problem: Dict, Y: np.ndarray) -> Dict:
        """Perform Sobol sensitivity analysis"""
        try:
            Si = sobol.analyze(problem, Y, print_to_console=False, calc_second_order=True)
            return {
                'first_order': dict(zip(problem['names'], Si['S1'])),
                'total_order': dict(zip(problem['names'], Si['ST'])),
                'second_order': Si['S2'],
                'confidence': {
                    'S1_conf': Si['S1_conf'],
                    'ST_conf': Si['ST_conf']
                }
            }
        except Exception as e:
            logger.error(f"Error in Sobol analysis: {str(e)}")
            raise

    def _perform_fast_analysis(self, problem: Dict, Y: np.ndarray) -> Dict:
        """Perform FAST sensitivity analysis"""
        try:
            fast_results = fast.analyze(problem, Y)
            return {
                'first_order': dict(zip(problem['names'], fast_results['S1'])),
                'total_order': dict(zip(problem['names'], fast_results['St'])),
                'confidence': {
                    'S1_conf': fast_results.get('S1_conf', None)
                }
            }
        except Exception as e:
            logger.error(f"Error in FAST analysis: {str(e)}")
            raise

    def _perform_rbd_analysis(self, problem: Dict, Y: np.ndarray) -> Dict:
        """Perform RBD-FAST sensitivity analysis"""
        try:
            rbd_results = rbd_fast.analyze(problem, Y)
            return {
                'first_order': dict(zip(problem['names'], rbd_results['S1'])),
                'total_order': dict(zip(problem['names'], rbd_results['ST'])),
                'confidence': {
                    'S1_conf': rbd_results.get('S1_conf', None)
                }
            }
        except Exception as e:
            logger.error(f"Error in RBD analysis: {str(e)}")
            raise

    def _combine_sensitivity_results(self, sobol_results: Dict, 
                                  fast_results: Dict, 
                                  rbd_results: Dict) -> Dict:
        """Combine results from different sensitivity analysis methods"""
        combined = {}
        
        # Weight factors for different methods
        weights = {'sobol': 0.5, 'fast': 0.3, 'rbd': 0.2}
        
        try:
            # Combine first-order indices
            first_order = {}
            for param in sobol_results['first_order'].keys():
                first_order[param] = (
                    weights['sobol'] * sobol_results['first_order'][param] +
                    weights['fast'] * fast_results['first_order'][param] +
                    weights['rbd'] * rbd_results['first_order'][param]
                )
            
            # Combine total-order indices
            total_order = {}
            for param in sobol_results['total_order'].keys():
                total_order[param] = (
                    weights['sobol'] * sobol_results['total_order'][param] +
                    weights['fast'] * fast_results['total_order'][param] +
                    weights['rbd'] * rbd_results['total_order'][param]
                )
            
            # Process interaction effects (only from Sobol)
            interaction_effects = self._process_interaction_indices(
                sobol_results['second_order'],
                self._get_parameter_names()
            )
            
            return {
                'first_order': first_order,
                'total_order': total_order,
                'interaction_effects': interaction_effects
            }
            
        except Exception as e:
            logger.error(f"Error combining sensitivity results: {str(e)}")
            raise

    def _calculate_confidence_intervals(self, problem: Dict, 
                                     outputs: Dict,
                                     results: Dict) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for sensitivity indices"""
        confidence_intervals = {}
        
        try:
            for param in problem['names']:
                # Bootstrap samples
                bootstrap_indices = np.random.choice(
                    len(outputs['sobol']),
                    size=(self.n_bootstrap, len(outputs['sobol'])),
                    replace=True
                )
                
                bootstrap_estimates = []
                for indices in bootstrap_indices:
                    Y_bootstrap = outputs['sobol'][indices]
                    Si = sobol.analyze(problem, Y_bootstrap, 
                                     print_to_console=False,
                                     calc_second_order=False)
                    bootstrap_estimates.append(Si['S1'][problem['names'].index(param)])
                
                # Calculate confidence intervals
                ci = np.percentile(bootstrap_estimates, 
                                 [2.5, 97.5])  # 95% confidence interval
                confidence_intervals[param] = (float(ci[0]), float(ci[1]))
            
            return confidence_intervals
            
        except Exception as e:
            logger.error(f"Error calculating confidence intervals: {str(e)}")
            raise
    def perform_local_sensitivity(self) -> Dict[str, Dict[str, float]]:
        """
        Perform local sensitivity analysis using finite differences and advanced methods
        
        Returns:
            Dict containing local sensitivity indices and related metrics
        """
        logger.info("Starting local sensitivity analysis")
        local_sensitivity = {}

        try:
            for metric in tqdm(self.metrics, desc="Analyzing local sensitivity"):
                # Get base metric value
                base_value = self._get_base_metric_value(metric)
                parameter_sensitivities = {}

                # Analyze each parameter
                for param_name, param_range in self._get_parameter_ranges().items():
                    # Calculate basic sensitivity
                    delta = (param_range[1] - param_range[0]) * 0.01
                    basic_sensitivity = self._calculate_local_sensitivity(
                        metric, param_name, delta, base_value
                    )

                    # Calculate normalized sensitivity
                    norm_sensitivity = self._calculate_normalized_sensitivity(
                        metric, param_name, base_value
                    )

                    # Calculate elasticity
                    elasticity = self._calculate_elasticity(
                        metric, param_name, base_value
                    )

                    parameter_sensitivities[param_name] = {
                        'basic_sensitivity': basic_sensitivity,
                        'normalized_sensitivity': norm_sensitivity,
                        'elasticity': elasticity,
                        'importance_score': (basic_sensitivity + norm_sensitivity + elasticity) / 3
                    }

                # Add variance-based decomposition
                vbd_results = self._perform_variance_decomposition(metric)
                
                local_sensitivity[metric] = {
                    'parameter_sensitivities': parameter_sensitivities,
                    'variance_decomposition': vbd_results,
                    'rankings': self._rank_local_sensitivities(parameter_sensitivities),
                    'stability_metrics': self._calculate_local_stability(parameter_sensitivities)
                }

                logger.info(f"Completed local sensitivity analysis for {metric}")

            return local_sensitivity

        except Exception as e:
            logger.error(f"Error in local sensitivity analysis: {str(e)}")
            raise

    def analyze_temporal_sensitivity(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Analyze how sensitivity varies across different time periods
        
        Returns:
            Dict containing temporal sensitivity analysis results
        """
        logger.info("Starting temporal sensitivity analysis")
        temporal_sensitivity = {}

        try:
            for period in tqdm(self.temporal_periods, desc="Analyzing temporal periods"):
                period_sensitivity = {}
                
                for metric in self.metrics:
                    # Calculate period-specific sensitivities
                    base_sensitivities = self._calculate_period_sensitivity(
                        metric, period
                    )
                    
                    # Analyze temporal patterns
                    temporal_patterns = self._analyze_temporal_patterns(
                        metric, period, base_sensitivities
                    )
                    
                    # Calculate temporal stability
                    stability_metrics = self._calculate_temporal_stability(
                        metric, period, base_sensitivities
                    )
                    
                    # Calculate peak vs off-peak differences
                    peak_effects = self._analyze_peak_effects(
                        metric, period, base_sensitivities
                    )
                    
                    period_sensitivity[metric] = {
                        'base_sensitivities': base_sensitivities,
                        'temporal_patterns': temporal_patterns,
                        'stability_metrics': stability_metrics,
                        'peak_effects': peak_effects,
                        'temporal_rankings': self._rank_temporal_sensitivities(
                            base_sensitivities, temporal_patterns
                        )
                    }
                
                temporal_sensitivity[period] = period_sensitivity

            # Add cross-period analysis
            temporal_sensitivity['cross_period_analysis'] = self._analyze_cross_period_effects()
            
            logger.info("Completed temporal sensitivity analysis")
            return temporal_sensitivity

        except Exception as e:
            logger.error(f"Error in temporal sensitivity analysis: {str(e)}")
            raise
    def _calculate_lipschitz_constant(self, metric: str, h: float = 1e-6) -> Optional[float]:
        """Calculate Lipschitz constant for the metric"""
        try:
            samples = self._generate_parameter_samples(100)
            max_ratio = 0.0
            
            for i in range(len(samples)):
                for j in range(i + 1, len(samples)):
                    x1, x2 = samples[i], samples[j]
                    f1 = self._evaluate_spf(metric, x1)
                    f2 = self._evaluate_spf(metric, x2)
                    
                    param_dist = np.linalg.norm(x1 - x2)
                    if param_dist > 0:
                        ratio = abs(f1 - f2) / param_dist
                        max_ratio = max(max_ratio, ratio)
            
            return max_ratio if max_ratio > 0 else None
            
        except Exception as e:
            logger.error(f"Error calculating Lipschitz constant: {str(e)}")
            return None

    def _calculate_condition_number(self, metric: str) -> Optional[float]:
        """Calculate condition number using Hessian"""
        try:
            # Sample point for Hessian calculation
            sample = self._generate_parameter_samples(1)[0]
            
            # Calculate Hessian
            H = self._calculate_hessian(metric, sample)
            
            # Calculate eigenvalues
            eigenvalues = np.linalg.eigvals(H)
            
            # Condition number is ratio of largest to smallest eigenvalue
            max_eig = np.max(np.abs(eigenvalues))
            min_eig = np.min(np.abs(eigenvalues))
            
            return max_eig / min_eig if min_eig > 0 else None
            
        except Exception as e:
            logger.error(f"Error calculating condition number: {str(e)}")
            return None
    def _calculate_normalized_sensitivity(self, metric: str, param_name: str, 
                                       base_value: float) -> float:
        """Calculate normalized sensitivity coefficient"""
        try:
            param_range = self._get_parameter_ranges()[param_name]
            param_nominal = (param_range[1] + param_range[0]) / 2
            delta = param_nominal * 0.01

            # Calculate perturbed values
            param_plus = param_nominal + delta
            param_minus = param_nominal - delta
            
            value_plus = self._evaluate_metric_with_param(metric, param_name, param_plus)
            value_minus = self._evaluate_metric_with_param(metric, param_name, param_minus)

            # Calculate normalized sensitivity
            sensitivity = ((value_plus - value_minus) / (2 * delta)) * (param_nominal / base_value)
            return sensitivity

        except Exception as e:
            logger.error(f"Error calculating normalized sensitivity for {param_name}: {str(e)}")
            return 0.0

    def _calculate_elasticity(self, metric: str, param_name: str, 
                            base_value: float) -> float:
        """Calculate elasticity (percentage change in output per percentage change in input)"""
        try:
            param_range = self._get_parameter_ranges()[param_name]
            param_nominal = (param_range[1] + param_range[0]) / 2
            delta_percent = 0.01  # 1% change

            param_plus = param_nominal * (1 + delta_percent)
            param_minus = param_nominal * (1 - delta_percent)
            
            value_plus = self._evaluate_metric_with_param(metric, param_name, param_plus)
            value_minus = self._evaluate_metric_with_param(metric, param_name, param_minus)

            # Calculate elasticity
            elasticity = ((value_plus - value_minus) / (value_plus + value_minus)) / delta_percent
            return elasticity

        except Exception as e:
            logger.error(f"Error calculating elasticity for {param_name}: {str(e)}")
            return 0.0

    def _analyze_temporal_patterns(self, metric: str, period: str, 
                                 base_sensitivities: Dict[str, float]) -> Dict:
        """Analyze temporal patterns in sensitivity"""
        try:
            # Get time series data for the period
            time_series = self._get_time_series_data(metric, period)
            
            patterns = {
                'trend': self._calculate_temporal_trend(time_series),
                'seasonality': self._extract_seasonality(time_series),
                'volatility': self._calculate_volatility(time_series),
                'autocorrelation': self._calculate_autocorrelation(time_series)
            }
            
            # Add change point detection
            patterns['change_points'] = self._detect_change_points(time_series)
            
            # Calculate temporal stability metrics
            patterns['stability_metrics'] = {
                'coefficient_of_variation': np.std(time_series) / np.mean(time_series),
                'trend_strength': self._calculate_trend_strength(time_series),
                'seasonal_strength': self._calculate_seasonal_strength(time_series)
            }
            
            return patterns

        except Exception as e:
            logger.error(f"Error analyzing temporal patterns: {str(e)}")
            return {}
    def analyze_parameter_interactions(self) -> Dict[str, Dict[str, float]]:
        """
        Comprehensive analysis of parameter interactions
        
        Returns:
            Dict containing interaction analysis results
        """
        logger.info("Starting parameter interaction analysis")
        interaction_analysis = {}

        try:
            for metric in tqdm(self.metrics, desc="Analyzing interactions"):
                metric_interactions = {}
                parameters = self._get_parameter_names()

                # First-order interactions
                for i, param1 in enumerate(parameters[:-1]):
                    for param2 in parameters[i+1:]:
                        interaction = self._calculate_interaction_effect(
                            metric, param1, param2
                        )
                        metric_interactions[f"{param1}_{param2}"] = {
                            'interaction_strength': interaction,
                            'synergy_score': self._calculate_synergy(metric, param1, param2),
                            'antagonism_score': self._calculate_antagonism(metric, param1, param2)
                        }

                # Higher-order interactions
                metric_interactions['higher_order'] = self._analyze_higher_order_interactions(
                    metric, parameters
                )

                # Interaction network analysis
                metric_interactions['network_metrics'] = self._analyze_interaction_network(
                    metric_interactions
                )

                interaction_analysis[metric] = {
                    'interactions': metric_interactions,
                    'summary_statistics': self._calculate_interaction_statistics(metric_interactions),
                    'key_interactions': self._identify_key_interactions(metric_interactions)
                }

            logger.info("Completed parameter interaction analysis")
            return interaction_analysis

        except Exception as e:
            logger.error(f"Error in parameter interaction analysis: {str(e)}")
            raise

    def generate_sensitivity_plots(self, results: Dict):
        """Generate comprehensive set of sensitivity analysis visualizations"""
        logger.info("Generating sensitivity analysis visualizations")
        
        try:
            # Create visualization subdirectories
            plot_dirs = {
                'global': self.output_dir / 'global_sensitivity',
                'local': self.output_dir / 'local_sensitivity',
                'temporal': self.output_dir / 'temporal_sensitivity',
                'interaction': self.output_dir / 'interaction_analysis'
            }
            
            for directory in plot_dirs.values():
                directory.mkdir(exist_ok=True)

            # Generate plots for each analysis type
            self._generate_global_sensitivity_plots(results['global'], plot_dirs['global'])
            self._generate_local_sensitivity_plots(results['local'], plot_dirs['local'])
            self._generate_temporal_sensitivity_plots(results['temporal'], plot_dirs['temporal'])
            self._generate_interaction_plots(results['interaction'], plot_dirs['interaction'])

            # Generate summary plots
            self._generate_summary_plots(results, self.output_dir)
            
            logger.info("Completed generating visualizations")

        except Exception as e:
            logger.error(f"Error generating sensitivity plots: {str(e)}")
            raise

    def _generate_global_sensitivity_plots(self, global_results: Dict, output_dir: Path):
        """Generate global sensitivity analysis visualizations"""
        for metric in self.metrics:
            # Sobol indices plot
            plt.figure(figsize=(12, 8))
            
            # Plot first and total order indices
            indices = pd.DataFrame({
                'First Order': global_results[metric].first_order,
                'Total Order': global_results[metric].total_order
            }).sort_values('Total Order', ascending=True)

            ax = indices.plot(kind='barh', width=0.8)
            plt.title(f'Sensitivity Indices for {metric}')
            plt.xlabel('Sensitivity Index')
            plt.ylabel('Parameter')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(output_dir / f'sobol_indices_{metric}.png')
            plt.close()

            # Interaction heatmap
            plt.figure(figsize=(10, 8))
            interaction_matrix = pd.DataFrame(global_results[metric].interaction_effects)
            sns.heatmap(interaction_matrix, annot=True, cmap='RdBu_r', center=0)
            plt.title(f'Parameter Interactions for {metric}')
            plt.tight_layout()
            plt.savefig(output_dir / f'interaction_heatmap_{metric}.png')
            plt.close()

    def _generate_local_sensitivity_plots(self, local_results: Dict, output_dir: Path):
        """Generate local sensitivity analysis visualizations"""
        for metric, results in local_results.items():
            # Parameter sensitivity rankings
            plt.figure(figsize=(12, 6))
            rankings = pd.DataFrame(results['parameter_sensitivities']).T
            rankings['importance_score'].sort_values().plot(kind='barh')
            plt.title(f'Parameter Importance Ranking for {metric}')
            plt.xlabel('Importance Score')
            plt.tight_layout()
            plt.savefig(output_dir / f'parameter_ranking_{metric}.png')
            plt.close()

            # Elasticity plot
            plt.figure(figsize=(12, 6))
            elasticities = pd.DataFrame({
                param: data['elasticity'] 
                for param, data in results['parameter_sensitivities'].items()
            }, index=[0]).T
            elasticities.plot(kind='bar')
            plt.title(f'Parameter Elasticities for {metric}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / f'elasticity_{metric}.png')
            plt.close()

    def _generate_temporal_sensitivity_plots(self, temporal_results: Dict, output_dir: Path):
        """Generate temporal sensitivity analysis visualizations"""
        for period in self.temporal_periods:
            for metric in self.metrics:
                # Time series plot with change points
                plt.figure(figsize=(15, 6))
                results = temporal_results[period][metric]
                
                time_series = results['temporal_patterns']['time_series']
                change_points = results['temporal_patterns']['change_points']
                
                plt.plot(time_series, label='Sensitivity')
                for cp in change_points:
                    plt.axvline(x=cp, color='r', linestyle='--', alpha=0.5)
                
                plt.title(f'Temporal Sensitivity Pattern for {metric} ({period})')
                plt.xlabel('Time Step')
                plt.ylabel('Sensitivity')
                plt.legend()
                plt.tight_layout()
                plt.savefig(output_dir / f'temporal_pattern_{metric}_{period}.png')
                plt.close()
    def _calculate_synergy(self, metric: str, param1: str, param2: str) -> float:
        """Calculate synergistic effects between parameters"""
        try:
            # Get baseline values
            base_value = self._get_base_metric_value(metric)
            param1_range = self._get_parameter_ranges()[param1]
            param2_range = self._get_parameter_ranges()[param2]
            
            # Calculate individual effects
            delta1 = (param1_range[1] - param1_range[0]) * 0.1
            delta2 = (param2_range[1] - param2_range[0]) * 0.1
            
            effect1 = self._evaluate_metric_with_param(metric, param1, param1_range[0] + delta1) - base_value
            effect2 = self._evaluate_metric_with_param(metric, param2, param2_range[0] + delta2) - base_value
            
            # Calculate combined effect
            combined_effect = self._evaluate_metric_with_params(
                metric, 
                {param1: param1_range[0] + delta1, param2: param2_range[0] + delta2}
            ) - base_value
            
            # Calculate synergy score
            synergy = combined_effect - (effect1 + effect2)
            return float(synergy)
            
        except Exception as e:
            logger.error(f"Error calculating synergy between {param1} and {param2}: {str(e)}")
            return 0.0

    def _analyze_higher_order_interactions(self, metric: str, 
                                        parameters: List[str]) -> Dict[str, float]:
        """Analyze higher-order (3+ way) parameter interactions"""
        higher_order = {}
        try:
            # Generate parameter combinations for higher-order interactions
            for r in range(3, min(len(parameters) + 1, 5)):  # Up to 4-way interactions
                for combo in itertools.combinations(parameters, r):
                    interaction_strength = self._calculate_higher_order_interaction(
                        metric, list(combo)
                    )
                    higher_order['_'.join(combo)] = interaction_strength
                    
            return higher_order
            
        except Exception as e:
            logger.error(f"Error analyzing higher-order interactions: {str(e)}")
            return {}

    def _analyze_interaction_network(self, interactions: Dict) -> Dict:
        """Analyze the network of parameter interactions"""
        try:
            import networkx as nx
            
            # Create interaction network
            G = nx.Graph()
            
            # Add nodes and edges from first-order interactions
            for interaction, data in interactions.items():
                if '_' in interaction and interaction != 'higher_order':
                    param1, param2 = interaction.split('_')
                    G.add_edge(param1, param2, 
                             weight=abs(data['interaction_strength']))
            
            # Calculate network metrics
            network_metrics = {
                'centrality': nx.degree_centrality(G),
                'betweenness': nx.betweenness_centrality(G),
                'clustering': nx.clustering(G),
                'average_clustering': nx.average_clustering(G),
                'density': nx.density(G)
            }
            
            return network_metrics
            
        except Exception as e:
            logger.error(f"Error analyzing interaction network: {str(e)}")
            return {}

    def _save_results(self, results: Dict):
        """Save analysis results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Save full results as pickle
            with open(self.output_dir / f'sensitivity_results_{timestamp}.pkl', 'wb') as f:
                pickle.dump(results, f)
            
            # Save summary as JSON
            summary = self._create_results_summary(results)
            with open(self.output_dir / f'sensitivity_summary_{timestamp}.json', 'w') as f:
                json.dump(summary, f, indent=4)
            
            # Save results for each metric as CSV
            for metric in self.metrics:
                metric_results = self._extract_metric_results(results, metric)
                pd.DataFrame(metric_results).to_csv(
                    self.output_dir / f'sensitivity_{metric}_{timestamp}.csv'
                )
            
            logger.info(f"Results saved to {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise

    @staticmethod
    def _create_results_summary(results: Dict) -> Dict:
        """Create a summary of sensitivity analysis results"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'metrics_analyzed': list(results['global'].keys()),
            'most_influential_parameters': {},
            'key_interactions': {},
            'temporal_patterns': {},
            'stability_metrics': {}
        }
        
        for metric in summary['metrics_analyzed']:
            # Get most influential parameters
            total_effects = results['global'][metric].total_order
            summary['most_influential_parameters'][metric] = dict(
                sorted(total_effects.items(), 
                       key=lambda x: x[1], 
                       reverse=True)[:5]
            )
            
            # Get key interactions
            interactions = results['interaction'][metric]['key_interactions']
            summary['key_interactions'][metric] = dict(
                sorted(interactions.items(), 
                       key=lambda x: abs(x[1]), 
                       reverse=True)[:5]
            )
            
            # Summarize temporal patterns
            if 'temporal' in results:
                temporal = results['temporal']
                summary['temporal_patterns'][metric] = {
                    period: temporal[period][metric]['stability_metrics']
                    for period in temporal if period != 'cross_period_analysis'
                }
            
            # Add stability metrics
            summary['stability_metrics'][metric] = results['global'][metric].stability_metrics
            
        return summary

    def _extract_metric_results(self, results: Dict, metric: str) -> Dict:
        """Extract all results for a specific metric"""
        metric_results = {
            'global_sensitivity': results['global'][metric].to_dict(),
            'local_sensitivity': results['local'][metric],
            'interactions': results['interaction'][metric],
            'temporal': {
                period: results['temporal'][period][metric]
                for period in self.temporal_periods
            }
        }
        return metric_results

if __name__ == "__main__":
    # Example usage
    from your_parameter_space import ParameterSpace
    from your_results_data import ResultsData
    
    # Initialize analyzer
    analyzer = SensitivityAnalyzer(
        parameter_space=ParameterSpace(),
        results_data=ResultsData(),
        output_dir='sensitivity_analysis_results',
        n_samples=1024,
        n_bootstrap=100
    )
    
    # Run analysis
    results = analyzer.conduct_comprehensive_analysis()
    
    # Access specific results
    global_sensitivity = results['global']
    local_sensitivity = results['local']
    temporal_sensitivity = results['temporal']
    interaction_analysis = results['interaction']