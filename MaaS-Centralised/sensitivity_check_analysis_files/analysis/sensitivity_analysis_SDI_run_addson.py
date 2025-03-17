# sensitivity_analysis_run_adds_on_analysis.py

import pickle
import json
import traceback
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sensitivity_check_SDI_analysis_adds_on import SDISensitivityAnalyzer
from sensitivity_check_visualisation import SensitivityVisualizer
import os
class SensitivityAnalysisRunner:
    def __init__(self, metric_type, analysis_type):
        self.metric_type = metric_type.lower()
        self.analysis_type = analysis_type.upper()  # Store as uppercase for consistent comparison
        self.output_dir = Path(f'{metric_type.lower()}_{analysis_type.lower()}_sensitivity_analysis')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = None
        self.parameter_history = None
        self.params = pd.DataFrame()
        self.metric_columns = ['sdi', 'sur', 'mae', 'upi']
        self.visualizer = SensitivityVisualizer(self.output_dir, self.analysis_type)
        
    def prepare_analysis_data(self, analysis_type=None):
        """
        Prepare and format data for analysis with comprehensive error handling.
        
        Args:
            analysis_type (str, optional): Either 'FPS' or 'PBS'. If None, uses self.analysis_type
            
        Returns:
            tuple: (results_df, params_df) - Processed DataFrames for analysis
        """
        try:
            # Use and validate analysis type
            analysis_type = (analysis_type or self.analysis_type).upper()
            if analysis_type not in ['FPS', 'PBS']:
                raise ValueError(f"Invalid analysis type: {analysis_type}")
                
            # Initialize parameters
            params_df = self.params.copy() if not self.params.empty else pd.DataFrame()
            
            # Extract VOT values if present
            if not params_df.empty and 'VALUE_OF_TIME' in params_df.columns:
                try:
                    vot_data = params_df['VALUE_OF_TIME'].apply(pd.Series)
                    for level in ['low', 'middle', 'high']:
                        if level in vot_data.columns:
                            params_df[f'vot_{level}'] = vot_data[level]
                    params_df = params_df.drop('VALUE_OF_TIME', axis=1)
                except Exception as e:
                    print(f"Warning: Could not process VALUE_OF_TIME data: {e}")

            # Process results data
            results_rows = []
            metrics = self.metric_columns
            income_levels = ['low', 'middle', 'high']
            results = self.results
            
            if not results:
                raise ValueError(f"No {analysis_type} results data available")
                
            # Process each result
            for idx, r in enumerate(results):
                for income_level in income_levels:
                    if income_level in r:
                        try:
                            # Create base row
                            row = {'income_level': income_level.lower()}
                            
                            # Add metrics
                            metrics_data = r[income_level]
                            if isinstance(metrics_data, dict):
                                for metric in metrics:
                                    metric_value = metrics_data.get(metric, 0)
                                    try:
                                        row[metric] = float(metric_value)
                                    except (TypeError, ValueError):
                                        print(f"Warning: Invalid {metric} value for {income_level}: {metric_value}")
                                        row[metric] = 0.0
                            
                            # Add step
                            row['step'] = len(r.get('step_history', [])) - 1 if 'step_history' in r else idx
                            
                            # Add analysis-specific data
                            if analysis_type == 'FPS':
                                if 'subsidy_pool' in r:
                                    try:
                                        row['subsidy_pool'] = float(r['subsidy_pool'])
                                    except (TypeError, ValueError):
                                        print(f"Warning: Invalid subsidy pool value: {r['subsidy_pool']}")
                                        row['subsidy_pool'] = 0.0
                            else:  # PBS
                                if 'subsidy_percentages' in r and income_level in r['subsidy_percentages']:
                                    subsidies = r['subsidy_percentages'][income_level]
                                    for mode, value in subsidies.items():
                                        try:
                                            row[f"{mode.lower()}_subsidy"] = float(value)
                                        except (TypeError, ValueError):
                                            print(f"Warning: Invalid subsidy value for {mode}: {value}")
                                            row[f"{mode.lower()}_subsidy"] = 0.0
                                    
                                    if 'varied_mode' in r:
                                        row['varied_mode'] = str(r.get('varied_mode', ''))
                            
                            results_rows.append(row)
                        except Exception as row_err:
                            print(f"Warning: Error processing row for {income_level}: {str(row_err)}")
                            continue
            
            # Validate and create results DataFrame
            if not results_rows:
                raise ValueError("No valid results data to process")
                
            results_df = pd.DataFrame(results_rows)
            print(f"\nResults data shape: {results_df.shape}")
            print("Available metrics:", list(results_df.columns))
            
            # Process parameter data
            if not params_df.empty:
                try:
                    # Normalize column names
                    params_df.columns = [str(col).lower().replace(' ', '_') for col in params_df.columns]
                    
                    # Convert numeric columns
                    for col in params_df.columns:
                        try:
                            numeric_mask = pd.to_numeric(params_df[col], errors='coerce').notna()
                            if numeric_mask.any():
                                params_df.loc[numeric_mask, col] = pd.to_numeric(
                                    params_df.loc[numeric_mask, col],
                                    errors='coerce'
                                )
                                
                                if params_df[col].nunique() == 1:
                                    print(f"Warning: Parameter '{col}' has constant value")
                                    
                        except Exception as col_err:
                            print(f"Warning: Could not convert column {col}: {str(col_err)}")
                            continue
                    
                    # Handle size mismatch
                    if len(params_df) != len(results_df):
                        print(f"\nAdjusting parameter data dimensions:")
                        print(f"Parameters: {len(params_df)} rows")
                        print(f"Results: {len(results_df)} rows")
                        
                        if len(params_df) > 0:
                            repeat_times = len(results_df) // len(params_df)
                            remainder = len(results_df) % len(params_df)
                            
                            if repeat_times > 0:
                                expanded_params = [params_df] * repeat_times
                                if remainder > 0:
                                    expanded_params.append(params_df.iloc[:remainder])
                                params_df = pd.concat(expanded_params, ignore_index=True)
                            elif len(params_df) > len(results_df):
                                params_df = params_df.iloc[:len(results_df)]
                            
                        print(f"Final parameter dimensions: {params_df.shape}")
                
                except Exception as param_err:
                    print(f"Error processing parameters: {str(param_err)}")
                    params_df = pd.DataFrame()
            
            # Final validations
            if len(results_df) < 5:
                print("Warning: Very small sample size may affect analysis reliability")
                
            if params_df.empty:
                print("Warning: No parameter data available for analysis")
                
            if not params_df.empty and len(params_df) != len(results_df):
                raise ValueError("Parameter and results data dimensions don't match after processing")
                
            print(f"Parameter data shape: {params_df.shape}")
            return results_df, params_df
            
        except Exception as e:
            print(f"Error preparing analysis data: {str(e)}")
            traceback.print_exc()
            return pd.DataFrame(), pd.DataFrame()




    def calculate_effect_sizes(self, results_df):
        """Calculate effect sizes for key relationships"""
        from scipy import stats
        
        def cohens_d(group1, group2):
            n1, n2 = len(group1), len(group2)
            var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
            pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            return (np.mean(group1) - np.mean(group2)) / pooled_se

        effect_sizes = {}
        for metric in ['sdi', 'sur', 'mae', 'upi']:
            try:
                # Use correct column name
                income_groups = results_df.groupby('income_level')[metric]
                effect_sizes[metric] = {
                    'low_vs_middle': cohens_d(
                        income_groups.get_group('low'),
                        income_groups.get_group('middle')
                    ),
                    'middle_vs_high': cohens_d(
                        income_groups.get_group('middle'),
                        income_groups.get_group('high')
                    )
                }
            except KeyError:
                print(f"Warning: Could not calculate effect sizes for {metric}")
                effect_sizes[metric] = {'low_vs_middle': None, 'middle_vs_high': None}
                
        
        return effect_sizes

        
    # Add to SensitivityAnalysisRunner class
    def analyze_statistical_significance(self, results_df):
        """Calculate statistical significance tests"""
        try:
            # Initialize correlation variables
            corr, p_value_corr = None, None
            
            # Between income groups analysis 
            income_groups = results_df.groupby('income_level')['sdi'].apply(list)
            f_stat, p_value = stats.f_oneway(*income_groups)
            
            # Calculate correlation only if subsidy_pool exists
            if 'subsidy_pool' in results_df.columns:
                corr, p_value_corr = stats.pearsonr(
                    results_df['subsidy_pool'],
                    results_df['sdi']
                )
            
            return {
                'anova_results': {'f_stat': f_stat, 'p_value': p_value},
                'correlation_results': {'correlation': corr, 'p_value': p_value_corr}
            }
        except Exception as e:
            print(f"Warning: Statistical analysis failed - {e}")
            return {
                'anova_results': {'f_stat': None, 'p_value': None},
                'correlation_results': {'correlation': None, 'p_value': None}
            }
    def load_results(self, results_path, parameter_history_path):
        """Load results for a single analysis type"""
        try:
            # Load results
            with open(results_path, 'rb') as f:
                self.results = pickle.load(f)
                print(f"- Loaded {len(self.results)} {self.analysis_type} results")

            # Load parameter history
            print(f"Loading parameters from: {parameter_history_path}")
            try:
                with open(parameter_history_path, 'r') as f:
                    param_data = json.load(f)
                    
                # Validate parameter history
                if not self._validate_param_history(param_data):
                    print("Warning: Invalid parameter history structure")
                    self.params = pd.DataFrame()
                    return True
                
                # Filter parameters based on analysis type
                param_list = []
                for param in param_data['parameters']:
                    flat_params = {}
                    
                    # Flatten utility coefficients
                    if 'utility_coefficients' in param:
                        for k, v in param['utility_coefficients'].items():
                            flat_params[f'utility_{k}'] = v
                    
                    # Flatten value of time
                    if 'value_of_time' in param:
                        for level, value in param['value_of_time'].items():
                            flat_params[f'vot_{level}'] = value
                    
                    # Flatten uber parameters
                    if 'uber_parameters' in param:
                        for company, values in param['uber_parameters'].items():
                            for k, v in values.items():
                                flat_params[f'{company}_{k}'] = v
                    
                    # Flatten bike parameters
                    if 'bike_parameters' in param:
                        for company, values in param['bike_parameters'].items():
                            for k, v in values.items():
                                flat_params[f'{company}_{k}'] = v
                    
                    # Flatten MaaS parameters
                    if 'maas' in param:
                        for k, v in param['maas'].items():
                            flat_params[f'maas_{k}'] = v
                    
                    # Flatten public transport parameters
                    if 'public_transport' in param:
                        for mode, times in param['public_transport'].items():
                            for period, value in times.items():
                                flat_params[f'{mode}_{period}'] = value
                    
                    # Flatten congestion parameters
                    if 'congestion_params' in param:
                        for k, v in param['congestion_params'].items():
                            flat_params[f'congestion_{k}'] = v
                    
                    # Handle subsidy information based on analysis type
                    if self.analysis_type == 'FPS':
                        # For FPS analysis, calculate total subsidy amount from PBS percentages
                        if 'subsidy' in param and 'percentages' in param['subsidy']:
                            total_subsidy = 0
                            for income_level, modes in param['subsidy']['percentages'].items():
                                for mode, percentage in modes.items():
                                    # Estimate subsidy pool based on percentages
                                    total_subsidy += percentage * 1000  # Base amount per mode
                            flat_params['subsidy_pool'] = total_subsidy
                    
                    elif self.analysis_type == 'PBS' and 'subsidy' in param:
                        for income, modes in param['subsidy']['percentages'].items():
                            for mode, value in modes.items():
                                flat_params[f'{income}_{mode}_subsidy'] = value
                    
                    # Add other metadata
                    flat_params['simulation_id'] = param.get('simulation_id', 0)
                    flat_params['varied_mode'] = param.get('varied_mode', 'unknown')
                    
                    param_list.append(flat_params)
                
                if param_list:
                    self.params = pd.DataFrame(param_list)
                    print(f"- Processed {len(self.params)} parameter entries for {self.analysis_type} analysis")
                    print("\nDebug: Parameter columns:", self.params.columns.tolist())
                    return True
                else:
                    print(f"Warning: No parameters processed for {self.analysis_type.lower()} analysis")
                    self.params = pd.DataFrame()
                    return True

            except Exception as e:
                print(f"Warning: Error processing parameter history: {str(e)}")
                print("Full error traceback:")
                traceback.print_exc()
                self.params = pd.DataFrame()
                return True

        except Exception as e:
            print(f"Error loading results: {str(e)}")
            traceback.print_exc()
            return False
                    
    def validate_data_sufficiency(self):
        """Validate if enough data points exist for meaningful analysis"""
        min_samples = 20  # Minimum recommended sample size
        
        # Use self.results instead of fps_results/pbs_results
        samples = len(self.results) if self.results else 0
        
        if samples < min_samples:
            print(f"Warning: Sample size ({samples}) below recommended minimum of {min_samples}")
            print("Consider increasing the number of simulations for more reliable results")
            return False
            
        if not self.parameter_history:
            print("Warning: No parameter history available")
            return False
            
        return True

    
    def analyze_parameter_distributions(self, results_df, params_df):
        """Analyze parameter value distributions and their impacts on metrics"""
        analysis_results = {}
        
        # Filter out non-numeric columns first
        numeric_params = params_df.select_dtypes(include=[np.number]).columns
        params_df_numeric = params_df[numeric_params]
        
        metrics = ['sdi', 'sur', 'mae', 'upi']
        
        for param in params_df_numeric.columns:
            try:
                # Basic statistics
                dist_stats = {
                    'mean': float(params_df_numeric[param].mean()),
                    'std': float(params_df_numeric[param].std()),
                    'quartiles': params_df_numeric[param].quantile([0.25, 0.5, 0.75]).to_dict()
                }
                
                # Calculate correlations with all metrics
                correlations = {}
                for metric in metrics:
                    if metric in results_df.columns:
                        try:
                            corr_coef, p_value = stats.pearsonr(
                                params_df_numeric[param].astype(float),
                                results_df[metric]
                            )
                            correlations[metric] = {
                                'correlation': float(corr_coef),
                                'p_value': float(p_value)
                            }
                        except:
                            correlations[metric] = {
                                'correlation': 0.0,
                                'p_value': 1.0
                            }
                
                # Add correlations to the distribution statistics
                dist_stats['correlations'] = correlations
                analysis_results[param] = dist_stats
                    
            except Exception as e:
                print(f"Skipping analysis for parameter {param}: {e}")
                continue
        
        return analysis_results

    def perform_robustness_checks(self, results_df):
        """Improved robustness checks with error handling"""
        robustness_results = {}
        
        # Convert columns to strings first
        results_df.columns = [str(col) for col in results_df.columns]
        
        for metric in self.metric_columns:
            try:
                metric = metric.lower()  # Ensure metric name is lowercase
                if metric not in results_df.columns:
                    print(f"Warning: {metric} not found in results")
                    continue
                
                # Get non-null values
                valid_data = results_df[metric].dropna()
                if len(valid_data) == 0:
                    print(f"Warning: No valid data for {metric}")
                    continue
                    
                bootstrap_means = []
                sample_size = min(len(valid_data), 30)
                
                for _ in range(1000):
                    sample = valid_data.sample(n=sample_size, replace=True)
                    bootstrap_means.append(sample.mean())
                
                robustness_results[metric] = {
                    'bootstrap_ci': np.percentile(bootstrap_means, [2.5, 97.5]),
                    'bootstrap_std': np.std(bootstrap_means),
                    'sample_size': sample_size
                }
                
            except Exception as e:
                print(f"Error in robustness analysis for {metric}: {e}")
                continue
        
        return robustness_results

    def analyze_parameter_interactions(self, params_df, results_df):
        """
        Analyze and visualize parameter interactions using multiple methods
        
        Args:
            params_df (pd.DataFrame): Parameters data with simulation settings
            results_df (pd.DataFrame): Results data containing metrics like SDI
        
        Returns:
            dict: Analysis results containing feature importance and correlations
        """
        try:
            if params_df.empty:
                return {
                    'feature_importance': {},
                    'correlation_matrix': {},
                    'data_summary': {
                        'num_parameters': 0,
                        'num_samples': len(results_df),
                        'explained_variance': 0
                    }
                }
                
            # Convert column names to strings and lowercase
            params_df.columns = [str(col).lower() for col in params_df.columns]
            results_df.columns = [str(col).lower() for col in results_df.columns]
            
            # Select only numeric columns
            numeric_params = params_df.select_dtypes(include=[np.number]).columns
            if len(numeric_params) == 0:
                print("No numeric parameters found for analysis")
                return None
                
            X = params_df[numeric_params].copy()
            
            # Handle data alignment
            if len(X) != len(results_df):
                print(f"\nAdjusting parameter data dimensions:")
                print(f"Original parameter rows: {len(X)}")
                print(f"Required rows (results): {len(results_df)}")
                
                # Calculate required repetitions
                repeat_factor = len(results_df) // len(X)
                if len(results_df) % len(X) != 0:
                    repeat_factor += 1
                
                # Expand parameter data
                X = pd.concat([X] * repeat_factor, ignore_index=True)
                X = X.iloc[:len(results_df)]
                print(f"Final parameter rows after adjustment: {len(X)}")

            # Target variable
            if 'sdi' not in results_df.columns:
                print("Warning: 'sdi' column not found in results, checking for capitalized version")
                if 'SDI' in results_df.columns:
                    y = results_df['sdi']
                else:
                    raise KeyError("Neither 'sdi' nor 'SDI' found in results data")
            else:
                y = results_df['sdi']
            
            print("\nPerforming feature importance analysis...")
            # Scale features for analysis
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Random Forest importance analysis
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_scaled, y)
            
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Calculate correlation matrix
            print("Calculating parameter correlations...")
            correlation_matrix = X.corr()
            
            # Generate visualizations if enough data points
            if len(X) >= 5:  # Minimum threshold for meaningful visualization
                print("\nGenerating interaction visualizations...")
                self.visualizer.plot_interaction_heatmap(
                    correlation_matrix, 
                    'Parameter Correlations'
                )
                self.visualizer.plot_feature_importance(feature_importance)
            else:
                print("Warning: Insufficient data points for meaningful visualization")

            # Prepare analysis results
            results = {
                'feature_importance': feature_importance.to_dict(),
                'correlation_matrix': correlation_matrix.to_dict(),
                'data_summary': {
                    'num_parameters': len(X.columns),
                    'num_samples': len(X),
                    'explained_variance': np.mean(rf.score(X_scaled, y))
                }
            }
            
            print("\nParameter interaction analysis completed successfully")
            return results
                
        except Exception as e:
            print(f"Error in parameter interaction analysis: {str(e)}")
            print("Full error traceback:")
            import traceback
            traceback.print_exc()
            return None


    def analyze_temporal_effects(self, results_df, params_df):
        """
        Analyze how parameter impacts evolve over simulation time with enhanced metrics 
        and better error handling.
        
        Args:
            results_df (pd.DataFrame): Results data containing metrics and time steps
            params_df (pd.DataFrame): Parameter data to analyze
            
        Returns:
            dict: Temporal analysis results for each parameter
        """
        try:
            temporal_analysis = {}
            
            # Add simulation step if not present
            if 'step' not in results_df.columns:
                print("Adding step column based on index")
                results_df['step'] = range(len(results_df))
            
            # Get unique steps and validate time series
            unique_steps = sorted(results_df['step'].unique())
            if len(unique_steps) < 2:
                print("Warning: Insufficient time steps for temporal analysis")
                return {}
                
            # Analyze each parameter
            for param in params_df.columns:
                if params_df[param].dtype in [np.float64, np.int64]:
                    try:
                        step_data = []
                        
                        for step in unique_steps:
                            # Get data for current timestep
                            step_results = results_df[results_df['step'] == step]
                            param_values = params_df[param].values
                            
                            # Ensure data alignment
                            if len(step_results) == len(param_values):
                                # Calculate correlations with each metric
                                metric_correlations = {}
                                for metric in ['sdi', 'sur', 'mae', 'upi']:
                                    if metric in step_results.columns:
                                        metric_values = step_results[metric].values
                                        
                                        # Only calculate if we have sufficient variation
                                        if (len(np.unique(param_values)) > 1 and 
                                            len(np.unique(metric_values)) > 1):
                                            corr_coef, p_value = stats.pearsonr(
                                                param_values, 
                                                metric_values
                                            )
                                            metric_correlations[metric] = {
                                                'correlation': corr_coef,
                                                'p_value': p_value
                                            }
                                
                                # Add comprehensive step data
                                step_info = {
                                    'step': step,
                                    'correlations': metric_correlations,
                                    'param_stats': {
                                        'mean': np.mean(param_values),
                                        'std': np.std(param_values),
                                        'min': np.min(param_values),
                                        'max': np.max(param_values)
                                    },
                                    'num_samples': len(param_values)
                                }
                                step_data.append(step_info)
                        
                        if step_data:
                            temporal_analysis[param] = step_data
                            
                            # Create temporal evolution plot
                            self.plot_temporal_evolution(param, step_data)
                            
                    except Exception as e:
                        print(f"Skipping temporal analysis for {param}: {e}")
                        continue
            
            return temporal_analysis
            
        except Exception as e:
            print(f"Error in temporal analysis: {str(e)}")
            return {}

    def plot_temporal_evolution(self, param, step_data):
        """
        Create visualization of temporal evolution for a parameter
        """
        try:
            plt.figure(figsize=(12, 8))
            
            # Extract data for plotting
            steps = [d['step'] for d in step_data]
            metrics = ['sdi', 'sur', 'mae', 'upi']
            colors = ['b', 'g', 'r', 'purple']
            
            # Plot correlation evolution for each metric
            for metric, color in zip(metrics, colors):
                correlations = [
                    d['correlations'].get(metric, {}).get('correlation', np.nan) 
                    for d in step_data
                ]
                plt.plot(steps, correlations, 
                        marker='o', color=color, label=metric.upper(),
                        linestyle='--', alpha=0.7)
            
            plt.title(f'Temporal Evolution of {param} Impact')
            plt.xlabel('Simulation Step')
            plt.ylabel('Correlation Coefficient')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add parameter value trend
            ax2 = plt.twinx()
            param_means = [d['param_stats']['mean'] for d in step_data]
            ax2.plot(steps, param_means, color='gray', alpha=0.3, 
                    label='Parameter Value', linestyle=':')
            ax2.set_ylabel('Parameter Value', color='gray')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'temporal_evolution_{param}.png'))
            plt.close()
            
        except Exception as e:
            print(f"Error creating temporal plot for {param}: {e}")

    def run_analysis(self):
        """Run sensitivity analysis for the specified analysis type"""
        if not self.validate_data_sufficiency():
            print("Warning: Proceeding with limited data")

        results_summary = {}

        try:
            # Prepare data for specified analysis type 
            results_df, params_df = self.prepare_analysis_data(self.analysis_type)
            
            # Data summary
            data_summary = {
                'num_samples': len(results_df),
                'parameters_analyzed': list(params_df.columns),
                'metrics_analyzed': ['sdi', 'sur', 'mae', 'upi'],
                'income_levels': list(results_df['income_level'].unique())
            }

            # Run parameter impact analysis
            parameter_impacts = {}
            for param in params_df.columns:
                print(f"\nAnalyzing impact of parameter: {param}")
                # Initialize analyzer with current results
                self.analyzer = SDISensitivityAnalyzer(self.results, self.params, self.output_dir)
                
                impact = self.analyzer.analyze_parameter_impact(param, results_df, params_df)
                if impact is not None:
                    parameter_impacts[param] = impact

            # Run main analyses
            param_distributions = self.analyze_parameter_distributions(results_df, params_df)
            interactions = self.analyze_parameter_interactions(params_df, results_df)
            temporal_effects = self.analyze_temporal_effects(results_df, params_df)
            statistical_tests = self.analyze_statistical_significance(results_df)
            effect_sizes = self.calculate_effect_sizes(results_df)
            robustness_results = self.perform_robustness_checks(results_df)

            # Create visualizations
            if temporal_effects:
                for param_name, time_correlations in temporal_effects.items():
                    fig = self.visualizer.plot_temporal_sensitivity(time_correlations, param_name)
                    if fig:
                        fig.savefig(os.path.join(self.output_dir, f'temporal_{param_name}.png'))
                        plt.close(fig)

            if not results_df.empty:
                self.visualizer.plot_summary_dashboard(results_df)
                
                if not params_df.empty:
                    for param in params_df.columns:
                        self.visualizer.create_detailed_parameter_report(
                            param_name=param,
                            param_data=params_df[param],
                            results_df=results_df
                        )
                        
                self.visualizer.plot_robustness_analysis(
                    results_df=results_df,
                    params_df=params_df
                )

            # Store results
            results_summary = {
                'data_summary': data_summary,
                'parameter_impacts': parameter_impacts,
                'parameter_distributions': param_distributions,
                'parameter_interactions': interactions,
                'temporal_effects': temporal_effects,
                'statistical_tests': statistical_tests,
                'effect_sizes': effect_sizes,
                'robustness_results': robustness_results
            }

            # Generate report
            self.generate_analysis_report(results_summary)
            return results_summary

        except Exception as e:
            print(f"Error in {self.analysis_type} analysis: {str(e)}")
            traceback.print_exc()
            return None

    def generate_analysis_report(self, results_summary):
        """Generate comprehensive analysis report with findings and visualizations"""
        report_path = self.output_dir / 'sensitivity_analysis_report.md'
        
        with open(report_path, 'w') as f:
            f.write(f"# {self.metric_type} Sensitivity Analysis Report\n\n")
            
            for analysis_type, results in results_summary.items():
                f.write(f"## {analysis_type} Analysis Results\n\n")
                
                # Data summary
                f.write("### Data Summary\n")
                data_summary = results['data_summary']
                f.write(f"- Number of samples: {data_summary['num_samples']}\n")
                f.write(f"- Parameters analyzed: {', '.join(data_summary['parameters_analyzed'])}\n")
                f.write(f"- Metrics analyzed: {', '.join(data_summary['metrics_analyzed'])}\n\n")
                
                # Parameter impacts
                f.write("### Parameter Impacts\n")
                for param, impact in results['parameter_impacts'].items():
                    f.write(f"\n#### {param}\n")
                    # Add null checks for all dictionary accesses
                    if impact and isinstance(impact, dict):
                        if 'parameter_range' in impact:
                            param_range = impact['parameter_range']
                            f.write(f"- Mean value: {param_range.get('mean', 'N/A'):.3f}\n")
                            f.write(f"- Standard deviation: {param_range.get('std', 'N/A'):.3f}\n")
                        
                        if 'correlations' in impact:
                            f.write("- Correlations with metrics:\n")
                            for metric, corr_data in impact['correlations'].items():
                                if isinstance(corr_data, dict):
                                    corr = corr_data.get('correlation', 'N/A')
                                    p_val = corr_data.get('p_value', 'N/A')
                                    if corr != 'N/A' and p_val != 'N/A':
                                        f.write(f"  - {metric}: r={corr:.3f} (p={p_val:.3f})\n")

                # Parameter interactions
                f.write("\n### Parameter Interactions\n")
                f.write("See interaction heatmap visualization for details.\n")
                
                # Temporal effects
                f.write("\n### Temporal Effects\n")
                f.write("See temporal sensitivity plots for details.\n\n")
                
                f.write("---\n\n")

    def _validate_param_history(self, param_data):
        """Validate parameter history structure and print debug info"""
        if not isinstance(param_data, dict):
            print("Error: Parameter history is not a dictionary")
            return False
        
        if 'metadata' not in param_data:
            print("Warning: No metadata found in parameter history")
        else:
            print("Parameter history metadata:", json.dumps(param_data['metadata'], indent=2))
        
        if 'parameters' not in param_data:
            print("Error: No parameters found in parameter history")
            return False
        
        if not param_data['parameters']:
            print("Error: Parameters list is empty")
            return False
        
        print(f"Found {len(param_data['parameters'])} parameter entries")
        
        # Check parameter structure without being strict about subsidy types
        if 'subsidy' in param_data['parameters'][0]:
            print("Found subsidy information in parameters")
        
        print("First parameter entry structure:", json.dumps(param_data['parameters'][0], indent=2))
        
        return True

def main():
    """Run separate FPS and PBS analyses"""
    try:
        results_summary = {}  # Store both FPS and PBS results

        # FPS Analysis
        fps_runner = SensitivityAnalysisRunner('sdi', 'FPS')
        fps_path = Path('sdi_fps_results.pkl')
        param_path = Path('parameter_tracking/parameter_history_SDI.json')
        
        if fps_path.exists():
            print("\nRunning FPS Analysis...")
            load_success = fps_runner.load_results(
                results_path=str(fps_path),
                parameter_history_path=str(param_path)
            )
            
            if load_success:
                fps_results = fps_runner.run_analysis()
                results_summary['fps'] = fps_results
                print("\nFPS Analysis completed successfully")
            else:
                print("Failed to load FPS data")

        # # PBS Analysis
        # pbs_runner = SensitivityAnalysisRunner('sdi', 'PBS')
        # pbs_path = Path('sdi_pbs_results.pkl')
        
        # if pbs_path.exists():
        #     print("\nRunning PBS Analysis...")
        #     load_success = pbs_runner.load_results(
        #         results_path=str(pbs_path),
        #         parameter_history_path=str(param_path)
        #     )
            
        #     if load_success:
        #         pbs_results = pbs_runner.run_analysis()
        #         results_summary['PBS'] = pbs_results
        #         print("\nPBS Analysis completed successfully")
        #     else:
        #         print("Failed to load PBS data")

        # Print summary if we have any results
        if results_summary:
            print("\nSensitivity Analysis Results Summary:")
            print("=" * 50)
            
            for analysis_type, results in results_summary.items():
                if results is not None:  # Check if analysis produced valid results
                    print(f"\n{analysis_type} Analysis:")
                    print("-" * 30)
                    
                    # Data overview
                    try:
                        data_summary = results.get('data_summary', {})
                        print(f"\nAnalyzed {data_summary.get('num_samples', 'N/A')} samples")
                        parameters = data_summary.get('parameters_analyzed', [])
                        if parameters:
                            print(f"Parameters: {', '.join(parameters)}")
                    except Exception as e:
                        print(f"Error printing data summary: {e}")

                    # Key parameter impacts
                    try:
                        print("\nMost influential parameters:")
                        parameter_impacts = results.get('parameter_impacts', {})
                        for param, impact in parameter_impacts.items():
                            if isinstance(impact, dict):
                                correlations = impact.get('correlations', {})
                                if 'sdi' in correlations:
                                    corr = correlations['sdi']
                                    if isinstance(corr, dict):
                                        correlation = corr.get('correlation', 'N/A')
                                        p_value = corr.get('p_value', 'N/A')
                                        if correlation != 'N/A' and p_value != 'N/A':
                                            print(f"- {param}: r={correlation:.3f} (p={p_value:.3f})")
                    except Exception as e:
                        print(f"Error printing parameter impacts: {e}")

                    # Temporal stability
                    try:
                        print("\nTemporal stability summary:")
                        temporal_effects = results.get('temporal_effects', {})
                        for param, temporal_data in temporal_effects.items():
                            if isinstance(temporal_data, list):
                                correlations = []
                                for d in temporal_data:
                                    if isinstance(d, dict) and 'correlation' in d:
                                        correlations.append(d['correlation'])
                                if correlations:
                                    stability = np.std(correlations)
                                    print(f"- {param}: stability={stability:.3f}")
                    except Exception as e:
                        print(f"Error printing temporal stability: {e}")

            print("\nAnalysis complete. Generated outputs:")
            print("1. Comprehensive visualizations in output directory")
            print("2. Detailed analysis report (sensitivity_analysis_report.md)")
            print("3. Parameter interaction plots")
            print("4. Temporal sensitivity visualizations")
            
            return results_summary
        else:
            print("\nNo analysis results to display")
            return None

    except Exception as e:
        print(f"Error in sensitivity analysis: {str(e)}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
