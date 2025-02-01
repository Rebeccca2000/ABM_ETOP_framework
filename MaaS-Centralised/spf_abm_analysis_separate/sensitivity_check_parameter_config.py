from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, List, Any
import numpy as np
from typing import Optional
@dataclass
class ParameterTracker:
    """Tracks and records parameter variations for sensitivity analysis"""
    def __init__(self, analysis_type, output_dir='parameter_tracking'):
        self.analysis_type = analysis_type
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.parameter_history = []
        
    def record_parameters(self, parameters: Dict, simulation_id: int):
        """Record parameter set with simulation ID"""
        tracked_params = {
            'simulation_id': simulation_id,
            'timestamp': datetime.now().isoformat(),
            'analysis_type': self.analysis_type,
            'utility_coefficients': {
                'beta_C': parameters['UTILITY_FUNCTION_BASE_COEFFICIENTS']['beta_C'],
                'beta_T': parameters['UTILITY_FUNCTION_BASE_COEFFICIENTS']['beta_T'],
                'beta_W': parameters['UTILITY_FUNCTION_BASE_COEFFICIENTS']['beta_W'],
                'beta_A': parameters['UTILITY_FUNCTION_BASE_COEFFICIENTS']['beta_A']
            },
            'value_of_time': {
                'low': float(parameters['VALUE_OF_TIME']['low']),
                'middle': float(parameters['VALUE_OF_TIME']['middle']),
                'high': float(parameters['VALUE_OF_TIME']['high'])
            },
            'uber_parameters': {
                'uber_like1': {
                    'capacity': parameters['uber_like1_capacity'],
                    'price': parameters['uber_like1_price']
                },
                'uber_like2': {
                    'capacity': parameters['uber_like2_capacity'],
                    'price': parameters['uber_like2_price']
                }
            },
            'bike_parameters': {
                'bike_share1': {
                    'capacity': parameters['bike_share1_capacity'],
                    'price': parameters['bike_share1_price']
                },
                'bike_share2': {
                    'capacity': parameters['bike_share2_capacity'],
                    'price': parameters['bike_share2_price']
                }
            },
            'maas_surcharge': parameters['DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS'],
            'public_transport': parameters['public_price_table'],
            'congestion_params': {
                'alpha': parameters.get('CONGESTION_ALPHA'),
                'beta': parameters.get('CONGESTION_BETA'),
                'capacity': parameters.get('CONGESTION_CAPACITY')
            }
        }
        
        if '_analysis_type' in parameters:
            if parameters['_analysis_type'] == 'FPS':
                tracked_params['subsidy'] = {
                    'type': 'FPS',
                    'pool_size': parameters['subsidy_config'].total_amount
                }
            else:
                tracked_params['subsidy'] = {
                    'type': 'PBS',
                    'percentages': parameters['subsidy_dataset']
                }
        # Record the specific varied mode if available
        if 'varied_mode' in parameters:
            tracked_params['varied_mode'] = parameters['varied_mode']

        self.parameter_history.append(tracked_params)
        
    def save_parameter_history(self):
        """Save parameter history to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f'parameter_history_{self.analysis_type}.json'
        
        # Create a structured output with metadata
        output_data = {
            'metadata': {
                'analysis_type': self.analysis_type,
                'timestamp': timestamp,
                'num_simulations': len(self.parameter_history)
            },
            'parameters': self.parameter_history
        }
        
        # Save with pretty printing for readability
        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=4)
            
        print(f"Parameter history saved to: {filepath}")
        return filepath
    def get_summary_statistics(self):
        """
        Generate summary statistics for parameter variations.
        
        Returns:
            dict: Summary statistics for each parameter
        """
        summary = {
            'analysis_type': self.analysis_type,
            'num_simulations': len(self.parameter_history),
            'parameter_ranges': {}
        }
        
        # Extract all unique parameters
        for param_set in self.parameter_history:
            self._recursively_add_to_summary(param_set, summary['parameter_ranges'])
            
        return summary
    
    def _recursively_add_to_summary(self, param_dict, summary_dict, prefix=''):
        """Helper method to recursively process nested parameter dictionaries."""
        for key, value in param_dict.items():
            if isinstance(value, dict):
                if key not in summary_dict:
                    summary_dict[key] = {}
                self._recursively_add_to_summary(value, summary_dict[key], f"{prefix}{key}_")
            elif isinstance(value, (int, float)):
                full_key = f"{prefix}{key}"
                if full_key not in summary_dict:
                    summary_dict[full_key] = {
                        'min': value,
                        'max': value,
                        'values': [value]
                    }
                else:
                    summary_dict[full_key]['min'] = min(summary_dict[full_key]['min'], value)
                    summary_dict[full_key]['max'] = max(summary_dict[full_key]['max'], value)
                    summary_dict[full_key]['values'].append(value)
                    
# Define parameter ranges
PARAMETER_RANGES = {
    'utility': {
        'beta_C': (-0.08, -0.02),
        'beta_T': (-0.08, -0.02),
        'beta_W': (-0.02, 0),
        'beta_A': (-0.02, 0)
    },
    'service': {
        'uber_like1_capacity': (6, 10),
        'uber_like1_price': (4, 8),
        'uber_like2_capacity': (7, 11),
        'uber_like2_price': (4.5, 8.5),
        'bike_share1_capacity': (8, 12),
        'bike_share1_price': (0.8, 1.2),
        'bike_share2_capacity': (10, 14),
        'bike_share2_price': (1.0, 1.4)
    },
    'value_of_time': {
        'low': (7.0, 12.0),      # Range around default 9.64
        'middle': (20.0, 27.0),  # Range around default 23.7
        'high': (60.0, 75.0)     # Range around default 67.2
    },
    'maas': {
        'S_base': (0.05, 0.15),
        'alpha': (0.1, 0.3),
        'delta': (0.3, 0.7)
    },
    'congestion': {
        'alpha': (0.2, 0.3),
        'beta': (3, 5),
        'capacity': (3, 5)
    },
    'public_transport': {
        'train': {
            'on_peak': (1.5, 2.5),
            'off_peak': (1.0, 2.0)
        },
        'bus': {
            'on_peak': (0.8, 1.2),
            'off_peak': (0.6, 1.0)
        }
    }
}

# FPS subsidy settings
FPS_SUBSIDY_DEFAULTS = {
    'low': {'bike': 0.317, 'car': 0.176, 'MaaS_Bundle': 0.493},
    'middle': {'bike': 0.185, 'car': 0.199, 'MaaS_Bundle': 0.383},
    'high': {'bike': 0.201, 'car': 0.051, 'MaaS_Bundle': 0.297}
}

# PBS subsidy ranges
PBS_SUBSIDY_RANGES = {
    'low': {
        'bike': (0.2, 0.4),
        'car': (0.15, 0.25),
        'MaaS_Bundle': (0.3, 0.6)
    },
    'middle': {
        'bike': (0.15, 0.35),
        'car': (0.10, 0.20),
        'MaaS_Bundle': (0.25, 0.5)
    },
    'high': {
        'bike': (0.10, 0.30),
        'car': (0.05, 0.15),
        'MaaS_Bundle': (0.20, 0.4)
    }
}

class AnalysisMode:
    """Controls which parameters vary in sensitivity analysis"""
    def __init__(self, mode_type: str, parameter_group: Optional[str] = None):
        self.mode_type = mode_type  # 'single', 'group', or 'full'
        self.parameter_group = parameter_group  # specific parameter or group to vary
        
    def should_vary_parameter(self, param_name: str) -> bool:
        valid_groups = ['subsidy', 'utility', 'service', 'maas', 
                       'congestion', 'value_of_time', 'all']
        if self.mode_type == 'single':
            return param_name == self.parameter_group
        elif self.mode_type == 'group':
            if self.parameter_group not in valid_groups:
                raise ValueError(f"Invalid parameter group. Must be one of: {valid_groups}")
            return param_name.startswith(self.parameter_group)
        return self.mode_type == 'full'
