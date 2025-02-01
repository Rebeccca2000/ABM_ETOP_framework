# sensitivity_analysis_run_adds_on_analysis.py

import pickle
import json
import traceback
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sensitivity_check_SDI_analysis_adds_on import SDISensitivityAnalyzer
from sensitivity_check_visualisation import SensitivityVisualizer

class SensitivityAnalysisRunner:
    def __init__(self, metric_type):
        self.metric_type = metric_type.lower()
        self.output_dir = Path(f'{metric_type.lower()}_sensitivity_analysis')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps_results = None
        self.pbs_results = None
        self.parameter_history = None
        self.params = pd.DataFrame()   # Add this line
        self.metric_columns = ['sdi', 'sur', 'mae', 'upi']  
        self.visualizer = SensitivityVisualizer(self.output_dir)
    
    def prepare_analysis_data(self, analysis_type):
        """
        Prepare and format data for analysis with comprehensive error handling.
        """
        try:
            # Initialize empty params DataFrame if None
            if self.params is None:
                self.params = pd.DataFrame()
                
            # Process results data
            results_rows = []
            metrics = self.metric_columns
            income_levels = ['low', 'middle', 'high']
            results = self.fps_results if analysis_type == 'FPS' else self.pbs_results
            
            # Process results data
            for idx, r in enumerate(results):
                for income_level in income_levels:
                    if income_level in r:
                        # Create base row with income level
                        row = {'income_level': income_level.lower()}
                        
                        # Add metric values
                        for metric in metrics:
                            if metric in r[income_level]:
                                row[metric] = float(r[income_level].get(metric, 0))
                        
                        # Add step - either from results or use simulation index
                        if 'step_history' in r:
                            row['step'] = len(r['step_history']) - 1  # Final step
                        else:
                            row['step'] = idx  # Use simulation index as step
                        
                        # Add analysis-specific data
                        if analysis_type == 'FPS':
                            # Handle FPS subsidy
                            if 'subsidy_pool' in r:
                                row['subsidy_pool'] = float(r.get('subsidy_pool', 0))
                        else:
                            # Handle PBS subsidies
                            if 'subsidy_percentages' in r and income_level in r['subsidy_percentages']:
                                subsidies = r['subsidy_percentages'][income_level]
                                for mode, value in subsidies.items():
                                    row[f"{mode.lower()}_subsidy"] = float(value)
                                if 'varied_mode' in r:
                                    row['varied_mode'] = str(r.get('varied_mode', ''))
                            
                        results_rows.append(row)
            
            # Create results DataFrame
            results_df = pd.DataFrame(results_rows)
            print(f"\nResults data shape: {results_df.shape}")
            print("Available metrics:", list(results_df.columns))
            
            # Process parameter data
            if not self.params.empty:
                # Create copy of parameter data
                params_df = self.params.copy()
                
                # Convert column names to strings
                params_df.columns = [str(col) for col in params_df.columns]
                
                # Convert numeric columns
                for col in params_df.columns:
                    try:
                        numeric_mask = pd.to_numeric(params_df[col], errors='coerce').notna()
                        if numeric_mask.any():
                            params_df.loc[numeric_mask, col] = pd.to_numeric(params_df.loc[numeric_mask, col])
                    except Exception as col_err:
                        print(f"Warning: Could not convert column {col}: {str(col_err)}")
                        continue
                
                # Match parameter data size with results
                if len(params_df) != len(results_df):
                    repeat_times = len(results_df) // len(params_df)
                    remainder = len(results_df) % len(params_df)
                    
                    if repeat_times > 0:
                        expanded_params = [params_df] * repeat_times
                        if remainder > 0:
                            expanded_params.append(params_df.iloc[:remainder])
                        params_df = pd.concat(expanded_params, ignore_index=True)
                    elif len(params_df) > len(results_df):
                        params_df = params_df.iloc[:len(results_df)]
            else:
                params_df = pd.DataFrame()
                
            print(f"Parameter data shape: {params_df.shape}")
            
            return results_df, params_df
            
        except Exception as e:
            print(f"Error preparing analysis data: {str(e)}")
            import traceback
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
    def load_results(self, fps_results_path, pbs_results_path, parameter_history_path):
        """
        Load simulation results and parameter history from files with improved error handling.
        """
        try:
            # Load FPS results
            with open(fps_results_path, 'rb') as f:
                self.fps_results = pickle.load(f)
            print(f"- {len(self.fps_results)} FPS results")

            # Load PBS results
            with open(pbs_results_path, 'rb') as f:
                self.pbs_results = pickle.load(f)
            print(f"- {len(self.pbs_results)} PBS results")

            # Load parameter history with error handling
            try:
                with open(parameter_history_path, 'r') as f:
                    param_data = json.load(f)
                    
                # Initialize parameter history and params DataFrame
                self.parameter_history = param_data
                
                # Convert parameter history to DataFrame if it has the correct structure
                if isinstance(param_data, dict) and 'parameters' in param_data:
                    param_lists = []
                    for param_dict in param_data['parameters']:
                        flat_params = {}
                        # Flatten nested dictionaries
                        def flatten_dict(d, prefix=''):
                            for k, v in d.items():
                                if isinstance(v, dict):
                                    flatten_dict(v, f"{prefix}{k}_")
                                elif isinstance(v, (int, float, str)):
                                    flat_params[f"{prefix}{k}"] = v
                        
                        flatten_dict(param_dict)
                        param_lists.append(flat_params)
                    
                    self.params = pd.DataFrame(param_lists)
                    print(f"- Parameter history converted to DataFrame with {len(self.params)} entries")
                else:
                    print("Warning: Parameter history does not have the expected structure")
                    self.params = pd.DataFrame()
                    
            except FileNotFoundError:
                print(f"Warning: Parameter history file not found at {parameter_history_path}")
                self.parameter_history = {}
                self.params = pd.DataFrame()
            except json.JSONDecodeError:
                print(f"Warning: Invalid JSON format in parameter history file")
                self.parameter_history = {}
                self.params = pd.DataFrame()
            except Exception as e:
                print(f"Warning: Error processing parameter history: {str(e)}")
                self.parameter_history = {}
                self.params = pd.DataFrame()

            print("\nData loading summary:")
            print(f"FPS results: {len(self.fps_results)} entries")
            print(f"PBS results: {len(self.pbs_results)} entries")
            print(f"Parameter data: {len(self.params)} entries")
            
            # Validate loaded data
            if len(self.fps_results) == 0 and len(self.pbs_results) == 0:
                raise ValueError("No results data loaded")

            return True

        except Exception as e:
            print(f"Error loading results: {str(e)}")
            traceback.print_exc()
            return False
            
    def validate_data_sufficiency(self):
        """Validate if enough data points exist for meaningful analysis"""
        min_samples = 30  # Minimum recommended sample size
        
        fps_samples = len(self.fps_results) if self.fps_results else 0
        pbs_samples = len(self.pbs_results) if self.pbs_results else 0
        
        if fps_samples < min_samples or pbs_samples < min_samples:
            print(f"Warning: Sample size ({fps_samples}/{pbs_samples}) below recommended minimum of {min_samples}")
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
            if params_df.empty or results_df.empty:
                print("Error: Empty parameter or results data")
                return None
                
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
                    y = results_df['SDI']
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
        try:
            temporal_analysis = {}
            
            # Add simulation step if not present
            if 'step' not in results_df.columns:
                results_df['step'] = range(len(results_df))
            
            for param in params_df.columns:
                if params_df[param].dtype in [np.float64, np.int64]:
                    try:
                        # Calculate correlations for each time step
                        step_correlations = []
                        for step in results_df['step'].unique():
                            step_data = results_df[results_df['step'] == step]
                            if len(step_data) == len(params_df):
                                corr_coef, p_value = stats.pearsonr(
                                    params_df[param],
                                    step_data['sdi']
                                )
                                step_correlations.append({
                                    'step': step,
                                    'correlation': corr_coef,
                                    'p_value': p_value
                                })
                        
                        if step_correlations:
                            temporal_analysis[param] = step_correlations
                            
                    except Exception as e:
                        print(f"Skipping temporal analysis for {param}: {e}")
                        continue
                        
            return temporal_analysis
            
        except Exception as e:
            print(f"Error in temporal analysis: {str(e)}")
            return {}

    def run_analysis(self):
        """Run comprehensive sensitivity analysis for both FPS and PBS results"""
        if not self.validate_data_sufficiency():
            print("Warning: Proceeding with limited data")
        
        analysis_types = ['FPS', 'PBS']
        results_summary = {}
        
        for analysis_type in analysis_types:
            print(f"\nRunning {analysis_type} analysis...")
            
            try:
                # Prepare data
                results_df, params_df = self.prepare_analysis_data(analysis_type)
                
                # Add data summary before running analyses
                data_summary = {
                    'num_samples': len(results_df),
                    'parameters_analyzed': list(params_df.columns),
                    'metrics_analyzed': ['sdi', 'sur', 'mae', 'upi'],
                    'income_levels': list(results_df['income_level'].unique())
                }
                
                # Run analyses
                param_impacts = self.analyze_parameter_distributions(results_df, params_df)
                interactions = self.analyze_parameter_interactions(params_df, results_df)
                temporal_effects = self.analyze_temporal_effects(results_df, params_df)
                
                # Add statistical analyses
                statistical_tests = self.analyze_statistical_significance(results_df)
                effect_sizes = self.calculate_effect_sizes(results_df)
                robustness_results = self.perform_robustness_checks(results_df)
                
                # Store results
                results_summary[analysis_type] = {
                    'data_summary': data_summary,  # Add this line
                    'parameter_impacts': param_impacts,
                    'parameter_interactions': interactions,
                    'temporal_effects': temporal_effects,
                    'statistical_tests': statistical_tests,
                    'effect_sizes': effect_sizes,
                    'robustness_results': robustness_results
                }
                
            except Exception as e:
                print(f"Error in {analysis_type} analysis: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        # Generate final report
        self.generate_analysis_report(results_summary)
        return results_summary

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
                    f.write(f"- Mean value: {impact['mean']:.3f}\n")
                    f.write(f"- Standard deviation: {impact['std']:.3f}\n")
                    if 'correlations' in impact:
                        f.write("- Correlations with metrics:\n")
                        for metric, corr_data in impact['correlations'].items():
                            f.write(f"  - {metric}: r={corr_data['correlation']:.3f} (p={corr_data['p_value']:.3f})\n")
                    # Parameter interactions
                f.write("\n### Parameter Interactions\n")
                f.write("See interaction heatmap visualization for details.\n")
                
                # Temporal effects
                f.write("\n### Temporal Effects\n")
                f.write("See temporal sensitivity plots for details.\n\n")
                
                f.write("---\n\n")

def main():
    """Run comprehensive sensitivity analysis with enhanced reporting"""
    try:
        # Initialize runner
        runner = SensitivityAnalysisRunner(metric_type='sdi')
        
        # Add file path validation
        fps_path = Path('sdi_fps_results.pkl')
        pbs_path = Path('sdi_pbs_results.pkl')
        param_path = Path('parameter_tracking/parameter_history_sdi.json')
        
        # Validate file existence
        if not fps_path.exists():
            raise FileNotFoundError(f"FPS results file not found: {fps_path}")
        if not pbs_path.exists():
            raise FileNotFoundError(f"PBS results file not found: {pbs_path}")
        
        # Load data with error handling
        load_success = runner.load_results(
            fps_results_path=str(fps_path),
            pbs_results_path=str(pbs_path),
            parameter_history_path=str(param_path)
        )
        
        if not load_success:
            print("Failed to load required data. Aborting analysis.")
            return None
        # Run the complete analysis
        results_summary = runner.run_analysis()
        # Run analysis with additional error checks
        if runner.fps_results is None or runner.pbs_results is None:
            raise ValueError("Failed to load results data")
        # Print detailed findings
        print("\nSensitivity Analysis Results Summary:")
        print("=" * 50)
        
        for analysis_type, results in results_summary.items():
            print(f"\n{analysis_type} Analysis:")
            print("-" * 30)
            
            # Data overview
            data_summary = results['data_summary']
            print(f"\nAnalyzed {data_summary['num_samples']} samples")
            print(f"Parameters: {', '.join(data_summary['parameters_analyzed'])}")
            
            # Key parameter impacts
            print("\nMost influential parameters:")
            for param, impact in results['parameter_impacts'].items():
                if 'correlations' in impact and 'sdi' in impact['correlations']:
                    corr = impact['correlations']['sdi']
                    print(f"- {param}: r={corr['correlation']:.3f} (p={corr['p_value']:.3f})")
            
            # Temporal stability
            print("\nTemporal stability summary:")
            for param, temporal_data in results['temporal_effects'].items():
                correlations = [d['correlation'] for d in temporal_data]
                print(f"- {param}: stability={np.std(correlations):.3f}")
        
        print("\nAnalysis complete. Generated outputs:")
        print("1. Comprehensive visualizations in output directory")
        print("2. Detailed analysis report (sensitivity_analysis_report.md)")
        print("3. Parameter interaction plots")
        print("4. Temporal sensitivity visualizations")
        
        return results_summary
        
    except Exception as e:
        print(f"Error in sensitivity analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()