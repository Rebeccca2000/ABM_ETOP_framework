import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import multiprocessing as mp
import pickle
import json
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import logging
import traceback
from sensitivity_analysis import SensitivityAnalyzer

# Add the FPS directory to the Python path
current_dir = Path(__file__).resolve().parent
maas_dir = current_dir.parent.parent
fps_dir = maas_dir / 'FPS'
sys.path.append(str(maas_dir))
sys.path.append(str(fps_dir))

# Now import the sensitivity check modules
try:
    # Import FPS modules
    from FPS.run_visualisation_03 import MobilityModel
    from FPS.agent_service_provider_03 import ServiceProvider
    from FPS.agent_MaaS_03 import MaaS
    from FPS.agent_commuter_03 import Commuter
    from FPS.agent_subsidy_pool import SubsidyPoolConfig
    
    # Import analysis modules
    from FPS.sensitivity_check_SDI import run_sdi_analysis
    from FPS.sensitivity_check_TEI import run_tei_analysis
    from FPS.sensitivity_check_FSIR import run_fsir_analysis 
    from FPS.sensitivity_check_BSR import run_bsr_analysis
    from validation_framework import ValidationFramework
    from spf_analysis_framework import SPFAnalyzer

except ImportError as e:
    print(f"Import Error: {e}")
    print(f"Current Python path: {sys.path}")
    print(f"Looking for modules in:")
    print(f"MaaS dir: {maas_dir}")
    print(f"FPS dir: {fps_dir}")
    sys.exit(1)

@dataclass
class ParameterSpace:
    """Define parameter ranges for sensitivity analysis"""
    def __init__(self):
        # Add grid parameters
        self.grid_parameters = {
            'width': 55,
            'height': 55
        }

        # Service provider parameters
        self.service_parameters = {
            'uber_like1_capacity': (6, 10),
            'uber_like1_price': (4, 8),
            'uber_like2_capacity': (7, 11), 
            'uber_like2_price': (4.5, 8.5),
            'bike_share1_capacity': (8, 12),
            'bike_share1_price': (0.8, 1.2),
            'bike_share2_capacity': (10, 14),
            'bike_share2_price': (1.0, 1.4)
        }

        # Utility function parameters
        self.utility_parameters = {
            'beta_C': (-0.08, -0.02),
            'beta_T': (-0.08, -0.02),
            'beta_W': (-0.02, 0),
            'beta_A': (-0.02, 0),
            'alpha': (-0.02, 0)
        }

        # Population parameters 
        self.population_parameters = {
            'income_weights': {
                'low': (0.4, 0.6),
                'middle': (0.2, 0.4),
                'high': (0.1, 0.3)
            },
            'value_of_time': {
                'low': (8, 12),
                'middle': (20, 28), 
                'high': (60, 75)
            }
        }

        # MaaS parameters
        self.maas_parameters = {
            'S_base': (0.05, 0.15),
            'alpha_maas': (0.1, 0.3),
            'delta_maas': (0.3, 0.7)
        }

        # Policy parameters
        self.policy_parameters = {
            'fps_pool_size': (1000, 40000),
            'pbs_rates': {
                'low': {
                    'bike': (0.2, 0.45),
                    'car': (0.15, 0.35), 
                    'maas': (0.3, 0.6)
                },
                'middle': {
                    'bike': (0.15, 0.35),
                    'car': (0.1, 0.25),
                    'maas': (0.25, 0.5)
                },
                'high': {
                    'bike': (0.1, 0.3),
                    'car': (0.05, 0.15),
                    'maas': (0.2, 0.4)
                }
            }
        }

        # Add new simulation parameters
        self.simulation_parameters = {
            'CHANCE_FOR_INSERTING_RANDOM_TRAFFIC': (0.1, 0.3),
            'BACKGROUND_TRAFFIC_AMOUNT': (50, 90),
            'CONGESTION_ALPHA': (0.2, 0.3),
            'CONGESTION_BETA': (3, 5), 
            'CONGESTION_CAPACITY': (3, 5),
            'CONGESTION_T_IJ_FREE_FLOW': (1.5, 2.5),
            'ALPHA_VALUES': {
                'UberLike1': (0.4, 0.6),
                'UberLike2': (0.4, 0.6),
                'BikeShare1': (0.4, 0.6), 
                'BikeShare2': (0.4, 0.6)
            },
            'public_price_table': {
                'train': {
                    'on_peak': (1.5, 2.5),
                    'off_peak': (1.0, 2.0)
                },
                'bus': {
                    'on_peak': (0.8, 1.2),
                    'off_peak': (0.6, 1.0) 
                }
            },
            'FLEXIBILITY_ADJUSTMENTS': {
                'low': (1.0, 1.1),
                'medium': (0.95, 1.05),
                'high': (0.9, 1.0)
            }
        }
    def get_bounds(self) -> Dict:
        """
        Convert our existing parameter definitions into a format suitable for sensitivity analysis.
        Returns a flattened dictionary of all parameter bounds.
        """
        bounds = {}
        
        # Add grid parameters - convert single values to small ranges around them
        for param, value in self.grid_parameters.items():
            bounds[param] = (value - 5, value + 5)
        
        # Add service parameters directly - they're already in (min, max) format
        bounds.update(self.service_parameters)
        
        # Add utility parameters directly
        bounds.update(self.utility_parameters)
        
        # Add MaaS parameters directly
        bounds.update(self.maas_parameters)
        
        # Handle nested population parameters
        for category, params in self.population_parameters.items():
            if isinstance(params, dict):
                for subcategory, value_range in params.items():
                    bounds[f"{category}_{subcategory}"] = value_range
        
        # Handle simulation parameters
        for param, value in self.simulation_parameters.items():
            if isinstance(value, tuple):
                bounds[param] = value
            elif isinstance(value, dict):
                for subparam, subvalue in value.items():
                    if isinstance(subvalue, tuple):
                        bounds[f"{param}_{subparam}"] = subvalue
                    elif isinstance(subvalue, dict):
                        for subsubparam, subsubvalue in subvalue.items():
                            bounds[f"{param}_{subparam}_{subsubparam}"] = subsubvalue
        
        return bounds

    def track_parameter_effects(self, parameters: Dict, results: Dict) -> Dict:
        """
        Track how parameter combinations affect system performance.
        This helps us understand parameter interactions and their impacts.
        """
        effects = {
            'service_effects': self._analyze_service_parameters(parameters, results),
            'behavior_effects': self._analyze_behavior_parameters(parameters, results),
            'system_effects': self._analyze_system_parameters(parameters, results),
            'maas_effects': self._analyze_maas_parameters(parameters, results)
        }
        
        return effects

    def _analyze_service_parameters(self, parameters: Dict, results: Dict) -> Dict:
        """Analyze how service provider parameters affect system performance"""
        return {
            'price_sensitivity': {
                'uber': self._analyze_price_effect('uber', parameters, results),
                'bike': self._analyze_price_effect('bike', parameters, results)
            },
            'capacity_utilization': {
                'uber': parameters['uber_like1_capacity'] / results.get('uber_usage', 1),
                'bike': parameters['bike_share1_capacity'] / results.get('bike_usage', 1)
            }
        }

    def _analyze_maas_parameters(self, parameters: Dict, results: Dict) -> Dict:
        """Analyze MaaS-specific parameter effects"""
        maas_params = parameters['DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS']
        return {
            'surcharge_effectiveness': {
                'base_impact': self._analyze_surcharge_impact(maas_params['S_base'], results),
                'dynamic_adjustment': self._analyze_dynamic_pricing(
                    maas_params['alpha'],
                    maas_params['delta'],
                    results
                )
            },
            'adoption_rates': results.get('maas_adoption', {}),
            'bundle_performance': results.get('bundle_usage', {})
        }
    def analyze_parameter_relationships(self) -> Dict:
        """
        Analyze relationships between parameters and their combined effects.
        This helps identify important parameter interactions.
        """
        relationships = {
            'price_capacity': self._analyze_price_capacity_relationship(),
            'utility_adoption': self._analyze_utility_adoption_relationship(),
            'congestion_maas': self._analyze_congestion_maas_relationship(),
            'subsidy_mode_choice': self._analyze_subsidy_mode_relationship()
        }
        
        return relationships

    def _analyze_price_capacity_relationship(self) -> Dict:
        """
        Analyze how prices and capacities interact to affect service usage.
        This analysis helps understand the balance between pricing and capacity utilization.
        """
        try:
            # Get relevant parameters
            uber_metrics = {
                'uber1': {
                    'price': self.service_parameters['uber_like1_price'],
                    'capacity': self.service_parameters['uber_like1_capacity'],
                    'usage_data': []  # Will be filled from simulation results
                },
                'uber2': {
                    'price': self.service_parameters['uber_like2_price'],
                    'capacity': self.service_parameters['uber_like2_capacity'],
                    'usage_data': []
                }
            }
            
            bike_metrics = {
                'bike1': {
                    'price': self.service_parameters['bike_share1_price'],
                    'capacity': self.service_parameters['bike_share1_capacity'],
                    'usage_data': []
                },
                'bike2': {
                    'price': self.service_parameters['bike_share2_price'],
                    'capacity': self.service_parameters['bike_share2_capacity'],
                    'usage_data': []
                }
            }

            # Calculate price-capacity ratios
            price_capacity_analysis = {
                'uber_services': {
                    'price_per_capacity': {
                        provider: metrics['price'][1] / metrics['capacity'][1]  # Using upper bounds
                        for provider, metrics in uber_metrics.items()
                    },
                    'capacity_utilization': self._calculate_capacity_utilization(uber_metrics),
                    'price_elasticity': self._calculate_price_elasticity(uber_metrics)
                },
                'bike_services': {
                    'price_per_capacity': {
                        provider: metrics['price'][1] / metrics['capacity'][1]
                        for provider, metrics in bike_metrics.items()
                    },
                    'capacity_utilization': self._calculate_capacity_utilization(bike_metrics),
                    'price_elasticity': self._calculate_price_elasticity(bike_metrics)
                }
            }

            return price_capacity_analysis

        except Exception as e:
            self.logger.error(f"Error in price-capacity analysis: {str(e)}")
            return {}

    def _analyze_utility_adoption_relationship(self) -> Dict:
        """
        Analyze how utility coefficients affect MaaS adoption.
        This helps understand user sensitivity to different service attributes.
        """
        try:
            # Extract utility coefficients
            coefficients = {
                'cost_sensitivity': self.utility_parameters['beta_C'],
                'time_sensitivity': self.utility_parameters['beta_T'],
                'waiting_sensitivity': self.utility_parameters['beta_W'],
                'access_sensitivity': self.utility_parameters['beta_A'],
                'mode_preference': self.utility_parameters['alpha']
            }

            # Analyze adoption patterns for different income groups
            adoption_analysis = {
                'income_group_effects': {
                    'low': self._analyze_income_group_adoption('low', coefficients),
                    'middle': self._analyze_income_group_adoption('middle', coefficients),
                    'high': self._analyze_income_group_adoption('high', coefficients)
                },
                'utility_sensitivities': {
                    'cost_impact': abs(coefficients['cost_sensitivity'][1] - coefficients['cost_sensitivity'][0]),
                    'time_impact': abs(coefficients['time_sensitivity'][1] - coefficients['time_sensitivity'][0]),
                    'composite_effects': self._analyze_composite_utility_effects(coefficients)
                },
                'maas_specific_effects': {
                    'bundle_attractiveness': self._calculate_bundle_attractiveness(coefficients),
                    'adoption_thresholds': self._calculate_adoption_thresholds(coefficients)
                }
            }

            return adoption_analysis

        except Exception as e:
            self.logger.error(f"Error in utility-adoption analysis: {str(e)}")
            return {}

    def _validate_price_relationships(self, parameters: Dict) -> bool:
        """
        Validate that price relationships between services make sense.
        For example, ensure premium services aren't cheaper than basic ones.
        """
        try:
            # Validate Uber-like service prices
            if parameters['uber_like1_price'] > parameters['uber_like2_price']:
                self.logger.warning("UberLike1 price should not exceed UberLike2 price")
                return False

            # Validate bike-share service prices
            if parameters['bike_share1_price'] > parameters['bike_share2_price']:
                self.logger.warning("BikeShare1 price should not exceed BikeShare2 price")
                return False

            # Validate price ratios between modes
            if parameters['bike_share1_price'] > parameters['uber_like1_price']:
                self.logger.warning("Bike share price should not exceed ride-hailing price")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error in price relationship validation: {str(e)}")
            return False

    def _validate_capacity_constraints(self, parameters: Dict) -> bool:
        """
        Validate that capacity allocations are feasible and balanced.
        """
        try:
            # Check minimum capacity requirements
            for service in ['uber_like1', 'uber_like2', 'bike_share1', 'bike_share2']:
                if parameters[f'{service}_capacity'] < 5:
                    self.logger.warning(f"Capacity too low for {service}")
                    return False

            # Check capacity balance between services
            uber_ratio = parameters['uber_like2_capacity'] / parameters['uber_like1_capacity']
            bike_ratio = parameters['bike_share2_capacity'] / parameters['bike_share1_capacity']

            if uber_ratio < 0.8 or uber_ratio > 1.2:
                self.logger.warning("Uber-like service capacities are too imbalanced")
                return False

            if bike_ratio < 0.8 or bike_ratio > 1.2:
                self.logger.warning("Bike share capacities are too imbalanced")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error in capacity constraint validation: {str(e)}")
            return False

    def _validate_utility_coefficients(self, parameters: Dict) -> bool:
        """
        Validate that utility coefficients are consistent with economic theory.
        """
        try:
            coeffs = parameters['UTILITY_FUNCTION_BASE_COEFFICIENTS']

            # Check coefficient signs
            if coeffs['beta_C'] > 0 or coeffs['beta_T'] > 0:
                self.logger.warning("Cost and time coefficients should be negative")
                return False

            # Check relative magnitudes
            if abs(coeffs['beta_C']) < abs(coeffs['beta_T']) / 10:
                self.logger.warning("Cost sensitivity seems too low compared to time sensitivity")
                return False

            if abs(coeffs['beta_T']) < abs(coeffs['beta_W']):
                self.logger.warning("Time sensitivity should be higher than waiting sensitivity")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error in utility coefficient validation: {str(e)}")
            return False

class SPFAnalysisManager:
    def __init__(self, num_simulations: int = 6, num_cpus: int = 8):
        self.parameter_space = ParameterSpace()
        self.num_simulations = num_simulations
        self.num_cpus = num_cpus
        self.output_dir = Path('analysis_outputs')

        self.logger = self._setup_logging()
        self.metrics = ['SDI', 'TEI', 'FSIR', 'BSR']
        # Create directory structure
        self.setup_environment()
        # Generate base parameters first
        self.base_parameters = self._sample_base_parameters()
        # Initialize analysis components
        self._initialize_analyzers()

    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = self.output_dir / f'analysis_{datetime.now():%Y%m%d_%H%M%S}.log'
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Setup logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger

    def _initialize_analyzers(self):
        """Initialize analysis components"""
        try:
            # Import required analyzers if not already imported
            from sensitivity_analysis import SensitivityAnalyzer
            from spf_analysis_framework import SPFAnalyzer
            from validation_framework import ValidationFramework

            # Initialize with empty results data structure
            initial_results_data = {
                'SDI': {},
                'TEI': {},
                'FSIR': {},
                'BSR': {},
                'temporal': {
                    'morning_peak': {},
                    'evening_peak': {},
                    'off_peak': {}
                }
            }

            # Initialize sensitivity analyzer
            self.sensitivity_analyzer = SensitivityAnalyzer(
                parameter_space=self.parameter_space,
                results_data=initial_results_data,  # Pass initialized structure instead of None
                output_dir=str(self.output_dir / 'sensitivity')
            )
            
            # Initialize SPF analyzer
            if not hasattr(self, 'base_parameters'):
                self.base_parameters = self._sample_base_parameters()
                
            # Add required grid dimensions to base parameters
            self.base_parameters['grid_dimensions'] = (
                self.base_parameters['grid_width'],
                self.base_parameters['grid_height']
            )
            spf_params = self._prepare_spf_parameters(self.base_parameters)

            self.spf_analyzer = SPFAnalyzer(
                base_parameters=spf_params,
                output_dir=str(self.output_dir / 'spf')
            )
            
            # # Initialize validation framework
            # self.validator = ValidationFramework(
            #     spf_analyzer=self.spf_analyzer,
            #     output_dir=str(self.output_dir / 'validation')
            # )
            # Don't initialize validator yet - will do it after we have results
            self.validator = None

            self.logger.info("Analysis components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing analyzers: {str(e)}")
            raise

    def run_comprehensive_analysis(self):
        try:
            # Phase 1: Parameter Relationship Analysis
            relationship_results = self.parameter_space.analyze_parameter_relationships()
            
            # Phase 2: Run Classic ABM Sensitivity Analysis 
            sensitivity_results = {
                'SDI': self._run_sensitivity_analysis('SDI'),
                'TEI': self._run_sensitivity_analysis('TEI'),
                'FSIR': self._run_sensitivity_analysis('FSIR'),
                'BSR': self._run_sensitivity_analysis('BSR')
            }
            
            # Phase 3: SPF Mathematical Properties Analysis
            spf_analyzer = SPFAnalyzer(
                base_parameters=self.base_parameters,
                fps_results=sensitivity_results,
                pbs_results=None,
                output_dir=str(self.output_dir / 'spf')
            )
            
            spf_properties = spf_analyzer.analyze_mathematical_properties()
            spf_convergence = spf_analyzer.analyze_convergence()
            spf_optimization = spf_analyzer.analyze_optimization_landscape()
            
            # Phase 4: Validation 
            validator = ValidationFramework(
                spf_analyzer=spf_analyzer,
                abm_results=sensitivity_results,
                output_dir=str(self.output_dir / 'validation')
            )
            
            validation_results = validator.validate_comprehensive()
            
            # Combine all results
            final_results = {
                'parameter_relationships': relationship_results,
                'sensitivity_analysis': sensitivity_results,
                'spf_analysis': {
                    'properties': spf_properties,
                    'convergence': spf_convergence,
                    'optimization': spf_optimization
                },
                'validation': validation_results,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'parameter_space': self.parameter_space.__dict__,
                    'config': {
                        'num_simulations': self.num_simulations,
                        'num_cpus': self.num_cpus
                    }
                }
            }
            
            # Save results
            self._save_results(final_results)
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive analysis: {str(e)}")
            traceback.print_exc()
            return None
    
    def _prepare_spf_parameters(self, params):
        """Prepare parameters for SPF analyzer by ensuring correct format"""
        spf_params = params.copy()
        
        # Add grid dimensions
        spf_params['grid_dimensions'] = (params['grid_width'], params['grid_height'])
        
        # Add value of time if not present
        if 'value_of_time' not in spf_params and 'VALUE_OF_TIME' in params:
            spf_params['value_of_time'] = params['VALUE_OF_TIME']
        
        # If neither exists, create it from parameter space
        if 'value_of_time' not in spf_params:
            vot_ranges = self.parameter_space.population_parameters['value_of_time']
            spf_params['value_of_time'] = {
                level: np.mean(range_vals) 
                for level, range_vals in vot_ranges.items()
            }
        
        return spf_params
    def _run_metric_sensitivity(self, metric: str, policy_type: str, n_simulations: int):
        """Run sensitivity analysis for a specific metric and policy type"""
        try:
            # Map metrics to their analysis functions
            analysis_functions = {
                'SDI': run_sdi_analysis,
                'TEI': run_tei_analysis,
                'FSIR': run_fsir_analysis,
                'BSR': run_bsr_analysis
            }
            
            if metric not in analysis_functions:
                raise ValueError(f"Unknown metric: {metric}")
                
            # Get the appropriate analysis function
            analysis_func = analysis_functions[metric]
            
            # Run the analysis with the specified policy type
            return analysis_func(
                policy_type,  # 'FPS' or 'PBS'
                self.base_parameters,
                n_simulations,
                self.num_cpus
            )
                
        except Exception as e:
            self.logger.error(f"Error analyzing {metric} for {policy_type}: {str(e)}")
            raise

    def _run_sensitivity_analysis(self, metric: str):
        """Run sensitivity analysis for a specific metric"""
        try:
            if metric == 'SDI':
                results, stats = run_sdi_analysis(
                    'FPS', self.base_parameters,
                    self.num_simulations, self.num_cpus
                )
            elif metric == 'TEI':
                results, stats = run_tei_analysis(
                    'FPS', self.base_parameters,
                    self.num_simulations, self.num_cpus
                )
            elif metric == 'FSIR':
                results, stats = run_fsir_analysis(
                    'FPS', self.base_parameters,
                    self.num_simulations, self.num_cpus
                )
            elif metric == 'BSR':
                results, stats = run_bsr_analysis(
                    'FPS', self.base_parameters,
                    self.num_simulations, self.num_cpus
                )
            else:
                raise ValueError(f"Unknown metric: {metric}")
                
            return {
                'results': results,
                'stats': stats
            }
            
        except Exception as e:
            self.logger.error(f"Error in sensitivity analysis for {metric}: {str(e)}")
            return None

    def _explore_parameter_space(self):
        """Explore full parameter space beyond just subsidies"""
        explorer = ParameterSpaceExplorer(self.parameter_space)
        
        # Study each parameter category
        grid_effects = explorer.analyze_grid_parameters()
        service_effects = explorer.analyze_service_parameters()
        behavior_effects = explorer.analyze_behavior_parameters()
        maas_effects = explorer.analyze_maas_parameters()
        
        return {
            'grid': grid_effects,
            'service': service_effects,
            'behavior': behavior_effects,
            'maas': maas_effects
        }

    def _analyze_spf_model(self, subsidy_results, parameter_study):
        """Analyze mathematical properties of SPF model"""
        analyzer = SPFAnalyzer(
            base_parameters=self.base_parameters,
            subsidy_results=subsidy_results,
            parameter_study=parameter_study
        )
        
        return {
            'mathematical_properties': analyzer.analyze_mathematical_properties(),
            'convergence': analyzer.analyze_convergence(),
            'optimization': analyzer.analyze_optimization(),
            'stability': analyzer.analyze_stability()
        }
    
    def setup_environment(self):
        """Setup directory structure for outputs"""
        try:
            # Create main output directory
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories for each analysis type
            analysis_dirs = {
                'sensitivity': ['fps', 'pbs'],
                'parameters': ['grid', 'service', 'behavior', 'maas'],
                'spf': ['properties', 'convergence', 'optimization'],
                'validation': ['statistical', 'temporal', 'cross_metric']
            }
            
            for main_dir, subdirs in analysis_dirs.items():
                dir_path = self.output_dir / main_dir
                dir_path.mkdir(exist_ok=True)
                
                for subdir in subdirs:
                    (dir_path / subdir).mkdir(exist_ok=True)
            
            self.logger.info("Analysis environment setup completed")
            
        except Exception as e:
            self.logger.error(f"Error setting up environment: {str(e)}")
            raise

    def _validate_parameters(self, params: Dict) -> bool:
        try:
            # Create copy and handle SubsidyPoolConfig serialization
            params_copy = params.copy()
            if 'subsidy_config' in params_copy:
                params_copy['subsidy_config'] = {
                    'type': params_copy['subsidy_config'].pool_type,
                    'amount': float(params_copy['subsidy_config'].total_amount)
                }

            # Convert age distribution tuples to strings
            if 'data_age_distribution' in params_copy:
                params_copy['data_age_distribution'] = {
                    f"{k[0]}-{k[1]}": v 
                    for k, v in params_copy['data_age_distribution'].items()
                }

            print("Validating parameters:", json.dumps(params_copy, indent=2))
            
            required_params = [
                'UTILITY_FUNCTION_BASE_COEFFICIENTS',
                'UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS'
            ]
            
            required_coeffs = ['beta_C', 'beta_T', 'beta_W', 'beta_A', 'alpha_utility']
            
            # Validate income weights
            if 'data_income_weights' in params:
                weights = params['data_income_weights']
                print(f"Checking income weights: {weights}")
                
                if not (isinstance(weights, list) and len(weights) == 3):
                    print(f"Invalid income weights format: {weights}")
                    return False
                    
                weight_sum = sum(weights)
                if not abs(weight_sum - 1.0) < 0.0001:
                    print(f"Income weights do not sum to 1: {weights} (sum: {weight_sum})")
                    return False
                    
                print("Income weights validation passed")
            else:
                print("Missing data_income_weights parameter")
                return False
            
            # Validate utility coefficients
            for param in required_params:
                if param not in params:
                    print(f"Missing required parameter: {param}")
                    return False
                
                coeffs = params[param]
                for coeff in required_coeffs:
                    if coeff not in coeffs:
                        print(f"Missing coefficient {coeff} in {param}")
                        return False
                        
                print(f"Parameter {param} validation passed")
            
            # Validate num_commuters
            if 'num_commuters' not in params or params['num_commuters'] > 120:
                print(f"Invalid num_commuters: {params.get('num_commuters', 'missing')}")
                params['num_commuters'] = 120

            # Validate required simulation parameters
            required_sim_params = [
                'grid_width', 'grid_height', 'simulation_steps',
                'ALPHA_VALUES', 'public_price_table', 'FLEXIBILITY_ADJUSTMENTS'
            ]
            
            for param in required_sim_params:
                if param not in params:
                    print(f"Missing required simulation parameter: {param}")
                    return False

            # Validate weights for FSIR calculation
            if 'weights' not in params:
                print("Missing FSIR weights")
                return False
                
            required_weights = ['alpha', 'beta', 'gamma', 'delta']
            for weight in required_weights:
                if weight not in params['weights']:
                    print(f"Missing weight parameter: {weight}")
                    return False
                    
            print("All parameters validated successfully")
            return True
            
        except Exception as e:
            print(f"Parameter validation failed: {e}")
            return False

    def generate_parameter_sets(self, analysis_type: str) -> List[Dict]:
        parameter_sets = []
        
        # Add FSIR weights to base parameters
        fsir_weights = {
            'alpha': 0.4,  # CO2 reduction
            'beta': 0.3,   # MaaS utilization 
            'gamma': 0.2,  # VKT reduction
            'delta': 0.1   # Time value
        }
        
        if analysis_type == 'FPS':
            subsidy_pools = np.linspace(
                self.parameter_space.policy_parameters['fps_pool_size'][0],
                self.parameter_space.policy_parameters['fps_pool_size'][1],
                self.num_simulations
            )
            
            for pool_size in subsidy_pools:
                params = self._sample_base_parameters()
                params['subsidy_config'] = SubsidyPoolConfig('daily', float(pool_size))
                params['_analysis_type'] = 'FPS'
                params['weights'] = fsir_weights  # Add weights
                parameter_sets.append(params)

        elif analysis_type == 'PBS':
            modes = ['bike', 'car', 'maas']
            points_per_mode = self.num_simulations // len(modes)
            
            for mode in modes:
                for i in range(points_per_mode):
                    params = self._sample_base_parameters()
                    
                    subsidy_config = {}
                    for income in ['low', 'middle', 'high']:
                        min_val, max_val = self.parameter_space.policy_parameters['pbs_rates'][income][mode]
                        pct = i / (points_per_mode - 1) if points_per_mode > 1 else 0
                        subsidy = min_val + (max_val - min_val) * pct
                        
                        subsidy_config[income] = {
                            'bike': subsidy if mode == 'bike' else self.parameter_space.policy_parameters['pbs_rates'][income]['bike'][0],
                            'car': subsidy if mode == 'car' else self.parameter_space.policy_parameters['pbs_rates'][income]['car'][0],
                            'maas': subsidy if mode == 'maas' else self.parameter_space.policy_parameters['pbs_rates'][income]['maas'][0]
                        }
                    
                    params['subsidy_dataset'] = subsidy_config
                    params['_analysis_type'] = 'PBS'
                    params['varied_mode'] = mode
                    params['weights'] = fsir_weights  # Add weights
                    parameter_sets.append(params)

        # Validate each parameter set
        valid_sets = []
        for params in parameter_sets:
            if self._validate_parameters(params):
                valid_sets.append(params)
            else:
                print(f"Skipping invalid parameter set")
        
        return valid_sets

    def _sample_base_parameters(self) -> Dict:
        try:
            # Initialize with fixed parameters
            params = {
                'num_commuters': 120,
                'grid_width': self.parameter_space.grid_parameters['width'],
                'grid_height': self.parameter_space.grid_parameters['height'],
                'simulation_steps': 15,
                'data_health_weights': [0.9, 0.1],
                'data_payment_weights': [0.8, 0.2],
                'data_disability_weights': [0.2, 0.8],
                'data_tech_access_weights': [0.95, 0.05],
                'ASC_VALUES': {
                    'car': 100, 'bike': 100, 'public': 100,
                    'walk': 100, 'maas': 100, 'default': 0
                },
                'PENALTY_COEFFICIENTS': {
                    'disability_bike_walk': 0.8,
                    'age_health_bike_walk': 0.3,
                    'no_tech_access_car_bike': 0.1
                },
                'AFFORDABILITY_THRESHOLDS': {'low': 25, 'middle': 85, 'high': 250},
                'data_age_distribution': {
                    (18, 25): 0.2, (26, 35): 0.3, (36, 45): 0.2,
                    (46, 55): 0.15, (56, 65): 0.1, (66, 75): 0.05
                }
            }

            # Sample utility parameters
            utility_params = self.parameter_space.utility_parameters.copy()
            alpha_range = utility_params.pop('alpha')
            
            # Sample base coefficients
            base_coeffs = {
                param: np.random.uniform(*range_vals)
                for param, range_vals in utility_params.items()
            }
            base_coeffs['alpha'] = np.random.uniform(*alpha_range)
            base_coeffs['alpha_utility'] = base_coeffs['alpha']
            
            # Use the same coefficients for both utility functions
            params['UTILITY_FUNCTION_BASE_COEFFICIENTS'] = base_coeffs.copy()
            params['UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS'] = base_coeffs.copy()

            # Sample and normalize income weights with explicit order
            income_levels = ['low', 'middle', 'high']
            income_weights = {
                level: np.random.uniform(*self.parameter_space.population_parameters['income_weights'][level])
                for level in income_levels
            }
            total = sum(income_weights.values())
            
            # Ensure weights sum to 1 and maintain order
            params['data_income_weights'] = [income_weights[level]/total for level in income_levels]
            
            # Validate weights
            if not (len(params['data_income_weights']) == 3 and 
                    abs(sum(params['data_income_weights']) - 1.0) < 0.0001):
                raise ValueError(f"Invalid income weights: {params['data_income_weights']}")

            # Debug print
            print("Income weights validation:")
            print(f"Raw weights: {income_weights}")
            print(f"Normalized weights: {params['data_income_weights']}")
            print(f"Sum: {sum(params['data_income_weights'])}")

            # Sample value of time
            params['VALUE_OF_TIME'] = {
                level: np.random.uniform(*range_vals)
                for level, range_vals in self.parameter_space.population_parameters['value_of_time'].items()
            }

            # Sample from simulation parameters
            for param, range_vals in self.parameter_space.simulation_parameters.items():
                if isinstance(range_vals, tuple):
                    if param in ['BACKGROUND_TRAFFIC_AMOUNT', 'CONGESTION_CAPACITY']:
                        params[param] = int(np.random.uniform(*range_vals))
                    else:
                        params[param] = np.random.uniform(*range_vals)
                elif isinstance(range_vals, dict):
                    params[param] = {
                        k: (np.random.uniform(*v) if isinstance(v, tuple) else 
                           {k2: np.random.uniform(*v2) for k2, v2 in v.items()})
                        for k, v in range_vals.items()
                    }

            # Sample service parameters
            for param, range_vals in self.parameter_space.service_parameters.items():
                if '_capacity' in param:
                    params[param] = int(np.random.uniform(*range_vals))
                else:
                    params[param] = np.random.uniform(*range_vals)

            # Add MaaS surcharge coefficients
            params['DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS'] = {
                'S_base': np.random.uniform(*self.parameter_space.maas_parameters['S_base']),
                'alpha': np.random.uniform(*self.parameter_space.maas_parameters['alpha_maas']),
                'delta': np.random.uniform(*self.parameter_space.maas_parameters['delta_maas'])
            }

            return params

        except Exception as e:
            print(f"Error in parameter sampling: {str(e)}")
            raise

    def run_analysis(self) -> Dict:
        try:
            # Create absolute paths for outputs
            analysis_root = Path(self.output_dir).resolve()
            output_dirs = {
                'sdi': analysis_root / 'sdi',
                'tei': analysis_root / 'tei', 
                'fsir': analysis_root / 'fsir',
                'bsr': analysis_root / 'bsr',
                'spf': analysis_root / 'spf',
                'validation': analysis_root / 'validation'
            }

            # Create directories
            for dir_path in output_dirs.values():
                dir_path.mkdir(parents=True, exist_ok=True)
            
            # Setup logging
            log_path = analysis_root / 'analysis_debug.log'
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s - %(levelname)s - %(message)s',
                filename=str(log_path),
                filemode='w'
            )
            
            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            console.setFormatter(formatter)
            logging.getLogger('').addHandler(console)

            results_dict = {'metrics': {}}

            # Run analyses for both FPS and PBS
            for analysis_type in ['FPS', 'PBS']:
                logging.info(f"\nStarting {analysis_type} analysis...")
                try:
                    parameter_sets = self.generate_parameter_sets(analysis_type)
                    if not parameter_sets:
                        raise ValueError(f"No valid parameter sets generated for {analysis_type}")
                    
                    base_params = parameter_sets[0].copy()
                    
                    # Initialize results structure
                    results_dict['metrics'][analysis_type.lower()] = {
                        'parameters': base_params.copy(),
                        'temporal': {},
                        'metrics': {},
                        'stats': {}
                    }
                    
                    # Run individual metric analyses
                    logging.info(f"\nRunning SDI analysis for {analysis_type}...")
                    sdi_base = base_params.copy()
                    sdi_base.pop('weights', None)
                    sdi_results, sdi_stats = run_sdi_analysis(
                        analysis_type, sdi_base, self.num_simulations, self.num_cpus
                    )
                    
                    logging.info(f"\nRunning TEI analysis for {analysis_type}...")
                    tei_base = base_params.copy()
                    tei_base.pop('weights', None)
                    tei_results, tei_stats = run_tei_analysis(
                        analysis_type, tei_base, self.num_simulations, self.num_cpus
                    )
                    
                    logging.info(f"\nRunning FSIR analysis for {analysis_type}...")
                    fsir_results, fsir_stats = run_fsir_analysis(
                        analysis_type, base_params, self.num_simulations, self.num_cpus
                    )
                    
                    logging.info(f"\nRunning BSR analysis for {analysis_type}...")
                    bsr_base = base_params.copy()
                    bsr_base.pop('weights', None)
                    bsr_results, bsr_stats = run_bsr_analysis(
                        analysis_type, bsr_base, self.num_simulations, self.num_cpus
                    )

                    # Store results
                    results_dict['metrics'][analysis_type.lower()].update({
                        'SDI': sdi_results,
                        'TEI': tei_results,
                        'FSIR': fsir_results,
                        'BSR': bsr_results,
                        'metrics': {
                            'SDI': sdi_stats,
                            'TEI': tei_stats,
                            'FSIR': fsir_stats,
                            'BSR': bsr_stats
                        }
                    })

                    if analysis_type == 'FPS':
                        fps_results = results_dict['metrics']['fps']
                    else:
                        pbs_results = results_dict['metrics']['pbs']

                except Exception as e:
                    logging.error(f"Error in {analysis_type} analysis: {str(e)}\n{traceback.format_exc()}")
                    return None

            # Run SPF analysis
            try:
                logging.info("\nPreparing SPF analysis...")
                analyzer_params = {
                    'grid_dimensions': (
                        self.parameter_space.grid_parameters['width'],
                        self.parameter_space.grid_parameters['height']
                    ),
                    'value_of_time': {
                        'low': np.mean(self.parameter_space.population_parameters['value_of_time']['low']),
                        'middle': np.mean(self.parameter_space.population_parameters['value_of_time']['middle']),
                        'high': np.mean(self.parameter_space.population_parameters['value_of_time']['high'])
                    },
                    **parameter_sets[0]
                }

                spf_analyzer = SPFAnalyzer(
                    base_parameters=analyzer_params,
                    fps_results=fps_results,
                    pbs_results=pbs_results,
                    output_dir=str(output_dirs['spf'])
                )

                logging.info("Running SPF analysis...")
                spf_properties = spf_analyzer.analyze_mathematical_properties()
                spf_convergence = spf_analyzer.analyze_convergence()
                spf_optimization = spf_analyzer.analyze_optimization()

            except Exception as e:
                logging.error(f"Error in SPF analysis: {str(e)}\n{traceback.format_exc()}")
                return None

            # Initialize and run validation after we have results
            try:
                logging.info("\nRunning validation framework...")
                if self.validator is None:
                    self.validator = ValidationFramework(
                        spf_analyzer=spf_analyzer,
                        abm_results={
                            'SDI': fps_results['SDI'],
                            'TEI': fps_results['TEI'],
                            'FSIR': fps_results['FSIR'],
                            'BSR': fps_results['BSR']
                        },
                        output_dir=str(output_dirs['validation'])
                    )
                validation_results = self.validator.validate_comprehensive()

            except Exception as e:
                logging.error(f"Error in validation framework: {str(e)}\n{traceback.format_exc()}")
                return None

            # Compile final results
            try:
                results_dict.update({
                    'spf_analysis': {
                        'properties': spf_properties,
                        'convergence': spf_convergence,
                        'optimization': spf_optimization
                    },
                    'validation': validation_results,
                    'metadata': {
                        'timestamp': datetime.now().isoformat(),
                        'parameter_space': self.parameter_space.__dict__,
                        'config': {
                            'num_simulations': self.num_simulations,
                            'num_cpus': self.num_cpus
                        }
                    }
                })

                logging.info("\nSaving results...")
                self._save_results(results_dict, analysis_root)
                logging.info("Analysis completed successfully!")
                return results_dict

            except Exception as e:
                logging.error(f"Error in final results compilation: {str(e)}\n{traceback.format_exc()}")
                return None

        except Exception as e:
            logging.error(f"Fatal error in analysis: {str(e)}\n{traceback.format_exc()}")
            raise

    def _save_results(self, results: Dict):
        """Save analysis results and summaries"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save full results
            results_file = self.output_dir / f'analysis_results_{timestamp}.pkl'
            with open(results_file, 'wb') as f:
                pickle.dump(results, f)
                
            # Save summary
            summary = self._create_analysis_summary(results)
            summary_file = self.output_dir / f'analysis_summary_{timestamp}.json'
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=4)
                
            self.logger.info(f"Results saved to {self.output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            raise

    def _create_analysis_summary(self, results: Dict) -> Dict:
        """Create comprehensive analysis summary"""
        return {
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'num_simulations': self.num_simulations,
                'num_cpus': self.num_cpus,
                'parameter_space': self.parameter_space.__dict__
            },
            'sensitivity_summary': self._summarize_sensitivity(results['sensitivity']),
            'parameter_summary': self._summarize_parameters(results['parameters']),
            'spf_summary': self._summarize_spf(results['spf']),
            'validation_summary': self._summarize_validation(results['validation'])
        }

    def _create_summary(self, results: Dict) -> Dict:
        return {
            'timestamp': results['metadata']['timestamp'],
            'metrics_summary': {
                'fps': {
                    metric: results['metrics']['fps'][metric]['stats']
                    for metric in ['sdi', 'tei', 'fsir', 'bsr']
                },
                'pbs': {
                    metric: results['metrics']['pbs'][metric]['stats']
                    for metric in ['sdi', 'tei', 'fsir', 'bsr']
                }
            },
            'spf_properties': {
                metric: prop.to_dict()
                for metric, prop in results['spf_analysis']['properties'].items()
            },
            'validation_summary': results['validation']['summary']
        }
    
# parameter_space_explorer.py
class ParameterSpaceExplorer:
    """Explores how different parameters affect system behavior"""
    
    def analyze_grid_parameters(self):
        """Study effects of grid size and layout"""
        results = {}
        
        # Vary grid dimensions
        for width, height in self._generate_grid_variations():
            results[f"{width}x{height}"] = self._run_grid_analysis(width, height)
            
        return results
        
    def analyze_service_parameters(self):
        """Study effects of service provider parameters"""
        results = {
            'capacity': self._analyze_capacity_effects(),
            'pricing': self._analyze_pricing_effects(),
            'availability': self._analyze_availability_effects()
        }
        return results
        
    def analyze_behavior_parameters(self):
        """Study effects of user behavior parameters"""
        results = {
            'utility': self._analyze_utility_coefficients(),
            'mode_choice': self._analyze_mode_choice_behavior(),
            'time_value': self._analyze_time_value_effects()
        }
        return results

if __name__ == "__main__":
    manager = SPFAnalysisManager(num_simulations=6, num_cpus=8)  # 8 points per mode for PBS
    results = manager.run_analysis()