import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.stats import norm, gaussian_kde
from sklearn.metrics import r2_score, mean_squared_error
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
from concurrent.futures import ProcessPoolExecutor

# 23 Jan 2025
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SPFProperties:
    """Container for SPF mathematical properties"""
    continuity: bool
    differentiability: bool
    convexity: Optional[bool]
    monotonicity: Dict[str, bool]
    bounds: Tuple[float, float]
    lipschitz_constant: Optional[float]
    condition_number: Optional[float]

    def to_dict(self) -> Dict:
        """Convert properties to dictionary"""
        return {
            'continuity': self.continuity,
            'differentiability': self.differentiability,
            'convexity': self.convexity,
            'monotonicity': self.monotonicity,
            'bounds': list(self.bounds),
            'lipschitz_constant': self.lipschitz_constant,
            'condition_number': self.condition_number
        }

class SPFAnalyzer:
    """
    Comprehensive analyzer for System Performance Function (SPF)
    
    Attributes:
        base_parameters: Base parameter set
        fps_results: Fixed Pool Subsidy results
        pbs_results: Percentage Based Subsidy results
        metrics: List of metrics to analyze
        output_dir: Directory for analysis outputs
    """
    
    def __init__(self,
                 base_parameters: Dict,
                 fps_results: Optional[Dict] = None,
                 pbs_results: Optional[Dict] = None,
                 output_dir: Optional[str] = None,
                 random_state: int = 42):
        """
        Initialize SPF analyzer
        
        Args:
            base_parameters: Dictionary of base parameters
            fps_results: FPS simulation results
            pbs_results: PBS simulation results
            output_dir: Directory for analysis outputs
            random_state: Random state for reproducibility
        """
        self.base_parameters = base_parameters
        self.fps_results = fps_results
        self.pbs_results = pbs_results
        self.metrics = ['SDI', 'TEI', 'FSIR', 'BSR']
        self.output_dir = Path(output_dir) if output_dir else Path('spf_analysis_outputs')
        self.random_state = random_state
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Validate inputs
        self._validate_inputs()
        
        # Initialize analysis components
        self.parameter_bounds = self._initialize_parameter_bounds()
        self.spf_properties = {}
        
        np.random.seed(random_state)
        logger.info("SPF Analyzer initialized successfully")

    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = self.output_dir / f'spf_analysis_{datetime.now():%Y%m%d_%H%M%S}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)

    def _validate_inputs(self):
        """Validate input data and parameters"""
        try:
            # Check base parameters
            required_params = ['simulation_steps', 'grid_dimensions', 'value_of_time']
            for param in required_params:
                assert param in self.base_parameters, f"Missing required parameter: {param}"
            
            # Check results if provided
            if self.fps_results is not None:
                self._validate_results(self.fps_results, 'FPS')
            if self.pbs_results is not None:
                self._validate_results(self.pbs_results, 'PBS')
                
        except AssertionError as e:
            logger.error(f"Validation error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during validation: {str(e)}")
            raise

    def _validate_results(self, results: Dict, policy_type: str):
        """Validate results data"""
        try:
            # Check metrics
            for metric in self.metrics:
                assert metric in results, f"Missing {metric} in {policy_type} results"
                
            # Check required fields
            required_fields = ['parameters', 'metrics', 'temporal']
            for field in required_fields:
                assert field in results, f"Missing {field} in {policy_type} results"
                
        except AssertionError as e:
            logger.error(f"Results validation error: {str(e)}")
            raise

    def _initialize_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Initialize parameter bounds for analysis"""
        return {
            'subsidy_pool': (1000, 40000),
            'maas_adoption': (0, 1),
            'value_of_time': {
                'low': (8, 12),
                'middle': (20, 28),
                'high': (60, 75)
            },
            'utility_coefficients': {
                'beta_C': (-0.08, -0.02),
                'beta_T': (-0.08, -0.02)
            }
        }
    
    def analyze_mathematical_properties(self) -> Dict[str, SPFProperties]:
        """
        Analyze mathematical properties of SPF for each metric
        
        Returns:
            Dict containing mathematical properties for each metric
        """
        logger.info("Starting mathematical properties analysis")
        properties = {}

        try:
            for metric in tqdm(self.metrics, desc="Analyzing mathematical properties"):
                # Analyze basic properties
                continuity = self._analyze_continuity(metric)
                differentiability = self._analyze_differentiability(metric)
                convexity = self._analyze_convexity(metric)
                monotonicity = self._analyze_monotonicity(metric)
                bounds = self._analyze_bounds(metric)
                
                # Analyze advanced properties
                lipschitz_constant = self._calculate_lipschitz_constant(metric)
                condition_number = self._calculate_condition_number(metric)
                
                properties[metric] = SPFProperties(
                    continuity=continuity,
                    differentiability=differentiability,
                    convexity=convexity,
                    monotonicity=monotonicity,
                    bounds=bounds,
                    lipschitz_constant=lipschitz_constant,
                    condition_number=condition_number
                )
                
                logger.info(f"Completed mathematical properties analysis for {metric}")
                
            self.spf_properties = properties
            return properties
            
        except Exception as e:
            logger.error(f"Error in mathematical properties analysis: {str(e)}")
            raise
    def _analyze_bounds(self, metric: str) -> Tuple[float, float]:
        """Analyze upper and lower bounds of SPF for given metric"""
        try:
            samples = self._generate_parameter_samples(1000)
            values = [self._evaluate_spf(metric, params) for params in samples]
            valid_values = [v for v in values if v != float('inf') and v != 0.0]
            
            if not valid_values:
                return (0.0, 1.0)
                
            return (min(valid_values), max(valid_values))
            
        except Exception as e:
            logger.error(f"Error analyzing bounds: {str(e)}")
            return (0.0, 1.0)
    
    def analyze_convergence(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyze convergence properties of SPF
        
        Returns:
            Dict containing convergence analysis results
        """
        logger.info("Starting convergence analysis")
        convergence_results = {}

        try:
            for metric in tqdm(self.metrics, desc="Analyzing convergence"):
                # Analyze FPS convergence
                fps_convergence = self._analyze_fps_convergence(metric)
                
                # Analyze PBS convergence
                pbs_convergence = self._analyze_pbs_convergence(metric)
                
                # Calculate convergence rates
                convergence_rates = self._calculate_convergence_rates(
                    metric, fps_convergence, pbs_convergence
                )
                
                # Analyze stability
                stability_metrics = self._analyze_convergence_stability(
                    metric, fps_convergence, pbs_convergence
                )
                
                convergence_results[metric] = {
                    'fps_convergence': fps_convergence,
                    'pbs_convergence': pbs_convergence,
                    'convergence_rates': convergence_rates,
                    'stability_metrics': stability_metrics,
                    'summary': self._summarize_convergence(
                        fps_convergence,
                        pbs_convergence,
                        convergence_rates,
                        stability_metrics
                    )
                }
                
            return convergence_results
            
        except Exception as e:
            logger.error(f"Error in convergence analysis: {str(e)}")
            raise

    def _analyze_fps_convergence(self, metric: str) -> Dict[str, Any]:
        """Analyze convergence properties for FPS results"""
        try:
            convergence_metrics = {}
            
            # Get FPS data for this metric if available
            if not self.fps_results or metric not in self.fps_results:
                return {
                    'convergence_rate': 0.0,
                    'stability_score': 0.0,
                    'convergence_threshold': None,
                    'iterations_to_converge': None
                }
                
            fps_data = self.fps_results[metric]
            
            # If data is dictionary with iterations
            if isinstance(fps_data, dict) and 'iterations' in fps_data:
                values = fps_data['iterations']
            # If data is list or array
            elif isinstance(fps_data, (list, np.ndarray)):
                values = fps_data
            else:
                return {
                    'convergence_rate': 0.0,
                    'stability_score': 0.0,
                    'convergence_threshold': None,
                    'iterations_to_converge': None
                }
                
            # Convert to numpy array if needed
            if not isinstance(values, np.ndarray):
                values = np.array(values)
                
            # Calculate convergence rate
            if len(values) > 1:
                differences = np.abs(np.diff(values))
                convergence_rate = np.mean(differences)
                
                # Calculate stability score
                stability = 1.0 - (np.std(values) / (np.mean(values) + 1e-10))
                
                # Find convergence threshold and iterations
                threshold = 0.01 * np.mean(values)  # 1% of mean value
                converged_idx = np.where(differences < threshold)[0]
                iterations_to_converge = converged_idx[0] if len(converged_idx) > 0 else len(values)
                
                convergence_metrics = {
                    'convergence_rate': float(convergence_rate),
                    'stability_score': float(stability),
                    'convergence_threshold': float(threshold),
                    'iterations_to_converge': int(iterations_to_converge)
                }
            else:
                convergence_metrics = {
                    'convergence_rate': 0.0,
                    'stability_score': 1.0,
                    'convergence_threshold': None,
                    'iterations_to_converge': 1
                }
                
            return convergence_metrics
            
        except Exception as e:
            logger.error(f"Error analyzing FPS convergence for {metric}: {str(e)}")
            return {
                'convergence_rate': 0.0,
                'stability_score': 0.0,
                'convergence_threshold': None,
                'iterations_to_converge': None
            }
    def analyze_optimization_landscape(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyze optimization landscape characteristics
        
        Returns:
            Dict containing optimization landscape analysis
        """
        logger.info("Starting optimization landscape analysis")
        landscape_analysis = {}

        try:
            for metric in tqdm(self.metrics, desc="Analyzing optimization landscape"):
                # Analyze critical points
                critical_points = self._find_critical_points(metric)
                
                # Analyze local optima
                local_optima = self._analyze_local_optima(metric)
                
                # Calculate gradients
                gradient_analysis = self._analyze_gradients(metric)
                
                # Analyze Hessian
                hessian_analysis = self._analyze_hessian(metric)
                
                landscape_analysis[metric] = {
                    'critical_points': critical_points,
                    'local_optima': local_optima,
                    'gradient_analysis': gradient_analysis,
                    'hessian_analysis': hessian_analysis,
                    'landscape_characteristics': self._characterize_landscape(
                        critical_points,
                        local_optima,
                        gradient_analysis,
                        hessian_analysis
                    )
                }
                
            return landscape_analysis
            
        except Exception as e:
            logger.error(f"Error in optimization landscape analysis: {str(e)}")
            raise

    def _analyze_continuity(self, metric: str) -> bool:
        """Analyze continuity of SPF for given metric"""
        try:
            # Get parameter space samples
            samples = self._generate_parameter_samples(1000)
            
            # Evaluate SPF at sample points
            values = [self._evaluate_spf(metric, params) for params in samples]
            
            # Check for discontinuities
            discontinuities = []
            for i in range(len(samples)-1):
                param_diff = np.linalg.norm(samples[i+1] - samples[i])
                value_diff = abs(values[i+1] - values[i])
                
                if value_diff > 100 * param_diff:  # Threshold for discontinuity
                    discontinuities.append({
                        'location': samples[i],
                        'jump_size': value_diff
                    })
            
            is_continuous = len(discontinuities) == 0
            if not is_continuous:
                logger.warning(f"Found {len(discontinuities)} discontinuities in {metric}")
                
            return is_continuous
            
        except Exception as e:
            logger.error(f"Error analyzing continuity: {str(e)}")
            return False

    def _analyze_differentiability(self, metric: str) -> bool:
        """Analyze differentiability of SPF"""
        try:
            # Get parameter space samples
            samples = self._generate_parameter_samples(1000)
            
            # Calculate numerical derivatives
            non_differentiable_points = []
            for sample in samples:
                derivatives = self._calculate_numerical_derivatives(metric, sample)
                
                # Check for undefined derivatives
                if any(np.isnan(d) or np.isinf(d) for d in derivatives):
                    non_differentiable_points.append(sample)
            
            is_differentiable = len(non_differentiable_points) == 0
            if not is_differentiable:
                logger.warning(
                    f"Found {len(non_differentiable_points)} non-differentiable points in {metric}"
                )
                
            return is_differentiable
            
        except Exception as e:
            logger.error(f"Error analyzing differentiability: {str(e)}")
            return False
        
    def analyze_optimization(self) -> Dict[str, Dict[str, Any]]:
        """
        Perform comprehensive optimization analysis
        
        Returns:
            Dict containing optimization analysis results
        """
        logger.info("Starting optimization analysis")
        optimization_results = {}

        try:
            for metric in tqdm(self.metrics, desc="Analyzing optimization"):
                # Find global optimum
                global_optimum = self._find_global_optimum(metric)
                
                # Analyze parameter sensitivities
                parameter_sensitivities = self._analyze_parameter_sensitivities(metric)
                
                # Analyze trade-offs
                trade_offs = self._analyze_trade_offs(metric)
                
                # Perform constraint analysis
                constraint_analysis = self._analyze_constraints(metric)
                
                optimization_results[metric] = {
                    'global_optimum': global_optimum,
                    'parameter_sensitivities': parameter_sensitivities,
                    'trade_offs': trade_offs,
                    'constraint_analysis': constraint_analysis,
                    'optimization_summary': self._summarize_optimization(
                        global_optimum,
                        parameter_sensitivities,
                        trade_offs,
                        constraint_analysis
                    )
                }
                
            # Add cross-metric optimization analysis
            optimization_results['cross_metric'] = self._analyze_cross_metric_optimization()
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Error in optimization analysis: {str(e)}")
            raise

    def _find_global_optimum(self, metric: str) -> Dict[str, Any]:
        """Find global optimum for given metric"""
        try:
            # Define objective function
            def objective(x):
                return -self._evaluate_spf(metric, x)  # Negative for maximization
            
            # Define constraints
            constraints = self._get_optimization_constraints()
            
            # Try multiple starting points
            starting_points = self._generate_starting_points(10)
            results = []
            
            for start_point in starting_points:
                result = optimize.minimize(
                    objective,
                    start_point,
                    method='SLSQP',
                    constraints=constraints,
                    bounds=self._get_parameter_bounds()
                )
                results.append((result.fun, result.x))
            
            # Find best result
            best_idx = np.argmin([r[0] for r in results])
            best_value = -results[best_idx][0]  # Convert back to maximization
            best_params = results[best_idx][1]
            
            return {
                'optimal_value': best_value,
                'optimal_parameters': {
                    name: value for name, value in zip(self._get_parameter_names(), best_params)
                },
                'convergence_status': True,
                'optimization_path': self._extract_optimization_path(results)
            }
            
        except Exception as e:
            logger.error(f"Error finding global optimum: {str(e)}")
            return {}

    def generate_visualization_suite(self, results: Dict):
        """Generate comprehensive set of visualizations"""
        logger.info("Generating visualization suite")
        
        try:
            # Create visualization directories
            plot_dirs = {
                'properties': self.output_dir / 'mathematical_properties',
                'optimization': self.output_dir / 'optimization_analysis',
                'landscape': self.output_dir / 'optimization_landscape',
                'convergence': self.output_dir / 'convergence_analysis',
                'summary': self.output_dir / 'analysis_summary'
            }
            
            for directory in plot_dirs.values():
                directory.mkdir(exist_ok=True)

            # Generate plots for each analysis type
            self._plot_mathematical_properties(
                results['properties'],
                plot_dirs['properties']
            )
            self._plot_optimization_results(
                results['optimization'],
                plot_dirs['optimization']
            )
            self._plot_landscape_analysis(
                results['landscape'],
                plot_dirs['landscape']
            )
            self._plot_convergence_analysis(
                results['convergence'],
                plot_dirs['convergence']
            )
            
            # Generate summary plots
            self._generate_summary_plots(results, plot_dirs['summary'])
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            raise

    def _plot_mathematical_properties(self, properties: Dict, output_dir: Path):
        """Generate plots for mathematical properties"""
        for metric in self.metrics:
            # Continuity visualization
            plt.figure(figsize=(12, 8))
            self._plot_function_surface(metric)
            plt.title(f'Function Surface - {metric}')
            plt.savefig(output_dir / f'surface_{metric}.png')
            plt.close()
            
            # Gradient visualization
            plt.figure(figsize=(12, 8))
            self._plot_gradient_field(metric)
            plt.title(f'Gradient Field - {metric}')
            plt.savefig(output_dir / f'gradient_{metric}.png')
            plt.close()
            
            # Convexity visualization
            plt.figure(figsize=(12, 8))
            self._plot_convexity_analysis(metric)
            plt.title(f'Convexity Analysis - {metric}')
            plt.savefig(output_dir / f'convexity_{metric}.png')
            plt.close()

    def _plot_optimization_results(self, results: Dict, output_dir: Path):
        """Generate plots for optimization results"""
        for metric in self.metrics:
            # Optimization path
            plt.figure(figsize=(12, 8))
            self._plot_optimization_path(results[metric]['optimization_path'])
            plt.title(f'Optimization Path - {metric}')
            plt.savefig(output_dir / f'optimization_path_{metric}.png')
            plt.close()
            
            # Parameter sensitivity
            plt.figure(figsize=(12, 8))
            self._plot_parameter_sensitivities(
                results[metric]['parameter_sensitivities']
            )
            plt.title(f'Parameter Sensitivities - {metric}')
            plt.savefig(output_dir / f'sensitivities_{metric}.png')
            plt.close()
            
            # Trade-off analysis
            plt.figure(figsize=(12, 8))
            self._plot_trade_offs(results[metric]['trade_offs'])
            plt.title(f'Trade-off Analysis - {metric}')
            plt.savefig(output_dir / f'trade_offs_{metric}.png')
            plt.close()

    
    def _save_analysis_results(self, results: Dict):
        """Save analysis results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Save full results as pickle
            with open(self.output_dir / f'spf_analysis_{timestamp}.pkl', 'wb') as f:
                pickle.dump(results, f)
            
            # Save summary as JSON
            summary = self._create_analysis_summary(results)
            with open(self.output_dir / f'spf_summary_{timestamp}.json', 'w') as f:
                json.dump(summary, f, indent=4)
            
            # Save detailed metrics as CSV
            detailed_metrics = self._extract_detailed_metrics(results)
            pd.DataFrame(detailed_metrics).to_csv(
                self.output_dir / f'spf_metrics_{timestamp}.csv'
            )
            
            logger.info(f"Analysis results saved to {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving analysis results: {str(e)}")
            raise

    def _calculate_numerical_derivatives(self, metric: str, params: np.ndarray,
                                      h: float = 1e-6) -> np.ndarray:
        """Calculate numerical derivatives using central differences"""
        try:
            n_params = len(params)
            derivatives = np.zeros(n_params)
            
            for i in range(n_params):
                params_plus = params.copy()
                params_plus[i] += h
                params_minus = params.copy()
                params_minus[i] -= h
                
                f_plus = self._evaluate_spf(metric, params_plus)
                f_minus = self._evaluate_spf(metric, params_minus)
                
                derivatives[i] = (f_plus - f_minus) / (2 * h)
                
            return derivatives
            
        except Exception as e:
            logger.error(f"Error calculating derivatives: {str(e)}")
            return np.array([np.nan] * n_params)

    def _calculate_hessian(self, metric: str, params: np.ndarray,
                          h: float = 1e-6) -> np.ndarray:
        """Calculate Hessian matrix using finite differences"""
        try:
            n_params = len(params)
            hessian = np.zeros((n_params, n_params))
            
            for i in range(n_params):
                for j in range(n_params):
                    # Calculate mixed partial derivatives
                    params_pp = params.copy()
                    params_pp[i] += h
                    params_pp[j] += h
                    
                    params_pm = params.copy()
                    params_pm[i] += h
                    params_pm[j] -= h
                    
                    params_mp = params.copy()
                    params_mp[i] -= h
                    params_mp[j] += h
                    
                    params_mm = params.copy()
                    params_mm[i] -= h
                    params_mm[j] -= h
                    
                    f_pp = self._evaluate_spf(metric, params_pp)
                    f_pm = self._evaluate_spf(metric, params_pm)
                    f_mp = self._evaluate_spf(metric, params_mp)
                    f_mm = self._evaluate_spf(metric, params_mm)
                    
                    hessian[i,j] = (f_pp - f_pm - f_mp + f_mm) / (4 * h * h)
                    
            return hessian
            
        except Exception as e:
            logger.error(f"Error calculating Hessian: {str(e)}")
            return np.array([[np.nan] * n_params] * n_params)

    @staticmethod
    def _create_analysis_summary(results: Dict) -> Dict:
        """Create summary of analysis results"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'metrics_analyzed': list(results['properties'].keys()),
            'mathematical_properties': {},
            'optimization_results': {},
            'convergence_analysis': {},
            'key_findings': []
        }
        
        # Summarize mathematical properties
        for metric in summary['metrics_analyzed']:
            summary['mathematical_properties'][metric] = {
                'is_continuous': results['properties'][metric].continuity,
                'is_differentiable': results['properties'][metric].differentiability,
                'is_convex': results['properties'][metric].convexity,
                'bounds': results['properties'][metric].bounds
            }
            
            # Add optimization results
            if 'optimization' in results:
                summary['optimization_results'][metric] = {
                    'optimal_value': results['optimization'][metric]['global_optimum']['optimal_value'],
                    'optimal_parameters': results['optimization'][metric]['global_optimum']['optimal_parameters']
                }
            
            # Add convergence results
            if 'convergence' in results:
                summary['convergence_analysis'][metric] = results['convergence'][metric]['summary']
        
        # Generate key findings
        summary['key_findings'] = SPFAnalyzer._generate_key_findings(results)
        
        return summary

    @staticmethod
    def _generate_key_findings(results: Dict) -> List[str]:
        """Generate key findings from analysis results"""
        findings = []
        
        # Analyze mathematical properties
        num_continuous = sum(1 for m in results['properties'].values() if m.continuity)
        findings.append(f"- {num_continuous} out of {len(results['properties'])} metrics are continuous")
        
        # Analyze optimization results
        if 'optimization' in results:
            best_metric = max(results['optimization'].items(),
                            key=lambda x: x[1]['global_optimum']['optimal_value'])[0]
            findings.append(f"- Best performing metric: {best_metric}")
        
        # Analyze convergence
        if 'convergence' in results:
            avg_convergence = np.mean([
                r['summary']['convergence_rate']
                for r in results['convergence'].values()
            ])
            findings.append(f"- Average convergence rate: {avg_convergence:.3f}")
        
        return findings

    def _generate_parameter_samples(self, n_samples: int) -> np.ndarray:
        """Generate parameter samples for analysis"""
        try:
            if not hasattr(self, 'parameter_bounds'):
                self.parameter_bounds = self._define_parameter_bounds()
                
            samples = []
            for _ in range(n_samples):
                sample = []
                for param_name, bounds in self.parameter_bounds.items():
                    if isinstance(bounds, dict):
                        # Handle nested parameters
                        for nested_param, nested_bounds in bounds.items():
                            if isinstance(nested_bounds, tuple):
                                sample.append(np.random.uniform(nested_bounds[0], nested_bounds[1]))
                    elif isinstance(bounds, tuple):
                        sample.append(np.random.uniform(bounds[0], bounds[1]))
                samples.append(sample)
            return np.array(samples)
        except Exception as e:
            logger.error(f"Error generating parameter samples: {str(e)}")
            return np.array([])

    def _define_parameter_bounds(self) -> Dict:
        """Define bounds for each parameter"""
        try:
            # Extract bounds from base parameters
            bounds = {}
            for param_name, value in self.base_parameters.items():
                if isinstance(value, (int, float)):
                    # Add ±20% range for numeric parameters
                    bounds[param_name] = (value * 0.8, value * 1.2)
                elif isinstance(value, dict):
                    # Handle nested parameters
                    bounds[param_name] = {
                        k: (v * 0.8, v * 1.2) if isinstance(v, (int, float)) else v
                        for k, v in value.items()
                    }
            return bounds
        except Exception as e:
            logger.error(f"Error defining parameter bounds: {str(e)}")
            return {}

    def _analyze_convexity(self, metric: str) -> bool:
        """Analyze convexity of SPF for given metric"""
        try:
            samples = self._generate_parameter_samples(100)
            if len(samples) == 0:
                return False
            
            lambdas = np.linspace(0, 1, 10)
            
            for i in range(len(samples)-1):
                x1, x2 = samples[i], samples[i+1]
                for l in lambdas:
                    x_lambda = l*x1 + (1-l)*x2
                    f_lambda = self._evaluate_spf(metric, x_lambda)
                    f_convex = l*self._evaluate_spf(metric, x1) + (1-l)*self._evaluate_spf(metric, x2)
                    if f_lambda > f_convex:
                        return False
            return True
        except Exception as e:
            logging.error(f"Error analyzing convexity: {str(e)}")
            return False

    def _evaluate_spf(self, metric: str, params: np.ndarray) -> float:
        """Evaluate SPF value for given parameters"""
        try:
            # Convert parameters array to dictionary
            param_dict = self._params_array_to_dict(params)
            
            # Get the appropriate results based on metric
            if metric == 'SDI':
                return self._evaluate_sdi(param_dict)
            elif metric == 'TEI':
                return self._evaluate_tei(param_dict)
            elif metric == 'FSIR':
                return self._evaluate_fsir(param_dict)
            elif metric == 'BSR':
                return self._evaluate_bsr(param_dict)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
        except Exception as e:
            logging.error(f"Error evaluating SPF: {str(e)}")
            return float('inf')

    def _params_array_to_dict(self, params: np.ndarray) -> Dict:
        """Convert parameters array to dictionary"""
        try:
            param_names = list(self.parameter_bounds.keys())
            return {name: value for name, value in zip(param_names, params)}
        except Exception as e:
            logging.error(f"Error converting parameters: {str(e)}")
            return {}

    def _evaluate_sdi(self, params: Dict) -> float:
        """Evaluate SDI metric"""
        try:
            if not self.fps_results:
                return 0.0
                
            sdi_data = self.fps_results.get('SDI', {})
            
            # Handle different possible data structures
            if isinstance(sdi_data, dict):
                if 'value' in sdi_data:
                    return float(sdi_data['value'])
                elif 'sdi' in sdi_data:  # Check for nested structure
                    return float(sdi_data['sdi'])
                elif any(isinstance(v, (int, float)) for v in sdi_data.values()):
                    return float(next(v for v in sdi_data.values() if isinstance(v, (int, float))))
            elif isinstance(sdi_data, (list, tuple)):
                # Get first numeric value
                for item in sdi_data:
                    if isinstance(item, (int, float)):
                        return float(item)
                    elif isinstance(item, dict) and 'sdi' in item:
                        return float(item['sdi'])
                        
            return 0.0
                
        except Exception as e:
            logger.error(f"Error evaluating SDI: {str(e)}")
            return 0.0

    def _evaluate_tei(self, params: Dict) -> float:
        """Evaluate TEI metric"""
        try:
            if not self.fps_results:
                return 0.0
            
            tei_results = self.fps_results.get('TEI', None)
            if tei_results is None:
                return 0.0
            
            # Handle both list and dict formats
            if isinstance(tei_results, dict):
                return float(tei_results.get('value', 0.0))
            elif isinstance(tei_results, list):
                return float(tei_results[0]) if tei_results else 0.0
            else:
                return float(tei_results) if tei_results else 0.0
            
        except Exception as e:
            logger.error(f"Error evaluating TEI: {str(e)}")
            return 0.0

    def _evaluate_fsir(self, params: Dict) -> float:
        """Evaluate FSIR metric"""
        try:
            if not self.fps_results:
                return 0.0
            
            fsir_results = self.fps_results.get('FSIR', None)
            if fsir_results is None:
                return 0.0
            
            # Handle both list and dict formats
            if isinstance(fsir_results, dict):
                return float(fsir_results.get('value', 0.0))
            elif isinstance(fsir_results, list):
                return float(fsir_results[0]) if fsir_results else 0.0
            else:
                return float(fsir_results) if fsir_results else 0.0
            
        except Exception as e:
            logger.error(f"Error evaluating FSIR: {str(e)}")
            return 0.0

    def _evaluate_bsr(self, params: Dict) -> float:
        """Evaluate BSR metric"""
        try:
            if not self.fps_results:
                return 0.0
            
            bsr_results = self.fps_results.get('BSR', None)
            if bsr_results is None:
                return 0.0
            
            # If BSR results contain a 'tei' field, extract it
            if isinstance(bsr_results, dict):
                if 'tei' in bsr_results:
                    return float(bsr_results['tei'])
                elif 'value' in bsr_results:
                    return float(bsr_results['value'])
                # If it's a nested dictionary with income levels
                elif any(level in bsr_results for level in ['low', 'middle', 'high']):
                    # Calculate average BSR across income levels
                    bsr_values = []
                    for level in ['low', 'middle', 'high']:
                        if level in bsr_results:
                            level_bsr = bsr_results[level].get('bsr', 0.0)
                            bsr_values.append(float(level_bsr))
                    return sum(bsr_values) / len(bsr_values) if bsr_values else 0.0
                else:
                    return 0.0
            elif isinstance(bsr_results, (int, float)):
                return float(bsr_results)
            elif isinstance(bsr_results, list):
                return float(bsr_results[0]) if bsr_results else 0.0
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error evaluating BSR: {str(e)}")
            return 0.0

    def _analyze_monotonicity(self, metric: str) -> Dict[str, bool]:
        """Analyze monotonicity for each parameter"""
        try:
            samples = self._generate_parameter_samples(100)
            if len(samples) == 0:
                return {'overall': False}
            
            monotonicity = {}
            for param in self.parameter_bounds.keys():
                monotonicity[param] = self._check_parameter_monotonicity(metric, param, samples)
            
            # Add overall monotonicity
            monotonicity['overall'] = any(monotonicity.values())
            return monotonicity
            
        except Exception as e:
            logger.error(f"Error analyzing monotonicity: {str(e)}")
            return {'overall': False}

    def _check_parameter_monotonicity(self, metric: str, param: str, samples: np.ndarray) -> bool:
        """Check if metric is monotonic with respect to parameter"""
        try:
            # Get parameter index
            param_index = list(self.parameter_bounds.keys()).index(param)
            
            # Sort samples by the parameter value
            sorted_indices = np.argsort(samples[:, param_index])
            sorted_samples = samples[sorted_indices]
            
            # Evaluate metric for sorted samples
            values = []
            for sample in sorted_samples:
                value = self._evaluate_spf(metric, sample)
                if value is not None:
                    values.append(value)
                
            if not values:
                return False
            
            # Check both increasing and decreasing monotonicity
            is_increasing = all(values[i] <= values[i+1] for i in range(len(values)-1))
            is_decreasing = all(values[i] >= values[i+1] for i in range(len(values)-1))
            
            return is_increasing or is_decreasing
            
        except Exception as e:
            logger.error(f"Error checking monotonicity for {param}: {str(e)}")
            return False
        
    def _calculate_lipschitz_constant(self, metric: str, h: float = 1e-6) -> Optional[float]:
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
        try:
            sample = self._generate_parameter_samples(1)[0]
            H = self._calculate_hessian(metric, sample)
            eigenvalues = np.linalg.eigvals(H)
            max_eig = np.max(np.abs(eigenvalues))
            min_eig = np.min(np.abs(eigenvalues))
            return max_eig / min_eig if min_eig > 0 else None
        except Exception as e:
            logger.error(f"Error calculating condition number: {str(e)}")
            return None

    def _analyze_bounds(self, metric: str) -> Tuple[float, float]:
        try:
            samples = self._generate_parameter_samples(1000)
            values = [self._evaluate_spf(metric, params) for params in samples]
            valid_values = [v for v in values if v != float('inf') and v != 0.0]
            return (min(valid_values), max(valid_values)) if valid_values else (0.0, 1.0)
        except Exception as e:
            logger.error(f"Error analyzing bounds: {str(e)}")
            return (0.0, 1.0)

    def _calculate_hessian(self, metric: str, params: np.ndarray, h: float = 1e-6) -> np.ndarray:
        try:
            n_params = len(params)
            hessian = np.zeros((n_params, n_params))
            
            for i in range(n_params):
                for j in range(n_params):
                    params_pp = params.copy()
                    params_pp[i] += h; params_pp[j] += h
                    
                    params_pm = params.copy()
                    params_pm[i] += h; params_pm[j] -= h
                    
                    params_mp = params.copy()
                    params_mp[i] -= h; params_mp[j] += h
                    
                    params_mm = params.copy()
                    params_mm[i] -= h; params_mm[j] -= h
                    
                    f_pp = self._evaluate_spf(metric, params_pp)
                    f_pm = self._evaluate_spf(metric, params_pm) 
                    f_mp = self._evaluate_spf(metric, params_mp)
                    f_mm = self._evaluate_spf(metric, params_mm)
                    
                    hessian[i,j] = (f_pp - f_pm - f_mp + f_mm) / (4 * h * h)
                    
            return hessian
        except Exception as e:
            logger.error(f"Error calculating Hessian: {str(e)}")
            return np.zeros((n_params, n_params))

if __name__ == "__main__":
    from main_analysis import SPFAnalysisManager
    
    # Run main analysis first to get results
    manager = SPFAnalysisManager(num_simulations=6, num_cpus=8)
    results = manager.run_analysis()
    
    # Extract required parameters and results from the main analysis 
    base_parameters = manager.base_parameters
    fps_results = results['metrics']['fps']
    pbs_results = results['metrics']['pbs']
    
    # Initialize analyzer with results from main analysis
    analyzer = SPFAnalyzer(
        base_parameters=base_parameters,
        fps_results=fps_results, 
        pbs_results=pbs_results,
        output_dir='spf_analysis_results',
        random_state=42
    )
    
    # Run SPF analysis
    properties = analyzer.analyze_mathematical_properties()
    convergence = analyzer.analyze_convergence()
    landscape = analyzer.analyze_optimization_landscape()
    optimization = analyzer.analyze_optimization()
    
    # Combine results
    analysis_results = {
        'properties': properties,
        'convergence': convergence,
        'landscape': landscape, 
        'optimization': optimization
    }
    
    # Generate visualizations and save results
    analyzer.generate_visualization_suite(analysis_results)
    analyzer._save_analysis_results(analysis_results)
    # Analyze mathematical properties
    properties = analyzer.analyze_mathematical_properties()
    
    # Analyze convergence
    convergence = analyzer.analyze_convergence()
    