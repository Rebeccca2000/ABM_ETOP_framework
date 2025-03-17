import numpy as np
import pandas as pd
from scipy import stats
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class AnalysisSummary:
    """Container for analysis summary"""
    metrics_analyzed: List[str]
    performance_metrics: Dict[str, float]
    key_findings: List[str]
    recommendations: List[str]
    limitations: List[str]

    def to_dict(self) -> Dict:
        """Convert summary to dictionary"""
        return {
            'metrics_analyzed': self.metrics_analyzed,
            'performance_metrics': self.performance_metrics,
            'key_findings': self.key_findings,
            'recommendations': self.recommendations,
            'limitations': self.limitations
        }

class ResultsAnalyzer:
    """
    Comprehensive analyzer for all simulation and analysis results
    
    Attributes:
        results: Combined results from all analyses
        metrics: List of metrics to analyze
        temporal_periods: List of temporal periods
        output_dir: Directory for analysis outputs
    """
    
    def __init__(self,
                 results: Dict,
                 output_dir: Optional[str] = None,
                 confidence_level: float = 0.95):
        """
        Initialize results analyzer
        
        Args:
            results: Dictionary containing all analysis results
            output_dir: Directory for analysis outputs
            confidence_level: Confidence level for intervals
        """
        self.results = results
        self.metrics = ['SDI', 'TEI', 'FSIR', 'BSR']
        self.temporal_periods = ['morning_peak', 'evening_peak', 'off_peak']
        self.output_dir = Path(output_dir) if output_dir else Path('results_analysis')
        self.confidence_level = confidence_level
        
        # Create output directories
        self._create_output_directories()
        
        # Setup logging
        self._setup_logging()
        
        # Validate inputs
        self._validate_inputs()
        
        logger.info("Results Analyzer initialized successfully")

    def _create_output_directories(self):
        """Create directory structure for outputs"""
        directories = [
            self.output_dir,
            self.output_dir / 'performance_analysis',
            self.output_dir / 'comparative_analysis',
            self.output_dir / 'temporal_analysis',
            self.output_dir / 'policy_analysis',
            self.output_dir / 'visualizations'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = self.output_dir / f'results_analysis_{datetime.now():%Y%m%d_%H%M%S}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)

    def _validate_inputs(self):
        """Validate input data"""
        try:
            required_components = [
                'spf_analysis', 'sensitivity_analysis',
                'validation_results', 'optimization_results'
            ]
            
            for component in required_components:
                assert component in self.results, f"Missing {component} in results"
            
            for metric in self.metrics:
                assert metric in self.results['spf_analysis'], f"Missing {metric} in SPF analysis"
                
        except AssertionError as e:
            logger.error(f"Validation error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during validation: {str(e)}")
            raise

    def analyze_comprehensive_results(self) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of all results
        
        Returns:
            Dict containing all analysis results
        """
        logger.info("Starting comprehensive results analysis")
        
        try:
            analysis_results = {
                'performance_analysis': self.analyze_performance(),
                'comparative_analysis': self.analyze_comparative_results(),
                'temporal_analysis': self.analyze_temporal_patterns(),
                'policy_analysis': self.analyze_policy_impacts(),
                'cross_metric_analysis': self.analyze_cross_metric_relationships()
            }
            
            # Generate summary
            analysis_results['summary'] = self._generate_analysis_summary(
                analysis_results
            )
            
            # Generate visualizations
            self.generate_visualization_suite(analysis_results)
            
            # Save results
            self._save_analysis_results(analysis_results)
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {str(e)}")
            raise

    def analyze_performance(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze performance metrics across all components
        
        Returns:
            Dict containing performance analysis results
        """
        logger.info("Analyzing performance metrics")
        performance_results = {}

        try:
            for metric in tqdm(self.metrics, desc="Analyzing metrics"):
                # Analyze SPF performance
                spf_performance = self._analyze_spf_performance(metric)
                
                # Analyze sensitivity results
                sensitivity_performance = self._analyze_sensitivity_performance(metric)
                
                # Analyze validation results
                validation_performance = self._analyze_validation_performance(metric)
                
                # Analyze optimization results
                optimization_performance = self._analyze_optimization_performance(metric)
                
                performance_results[metric] = {
                    'spf_metrics': spf_performance,
                    'sensitivity_metrics': sensitivity_performance,
                    'validation_metrics': validation_performance,
                    'optimization_metrics': optimization_performance,
                    'aggregate_performance': self._calculate_aggregate_performance(
                        spf_performance,
                        sensitivity_performance,
                        validation_performance,
                        optimization_performance
                    )
                }
                
            return performance_results
            
        except Exception as e:
            logger.error(f"Error in performance analysis: {str(e)}")
            raise

    def analyze_comparative_results(self) -> Dict[str, Dict[str, Any]]:
        """
        Perform comparative analysis between different approaches
        
        Returns:
            Dict containing comparative analysis results
        """
        logger.info("Performing comparative analysis")
        comparative_results = {}

        try:
            # Compare FPS vs PBS
            policy_comparison = self._compare_policy_approaches()
            
            # Compare metric performances
            metric_comparison = self._compare_metric_performances()
            
            # Compare temporal patterns
            temporal_comparison = self._compare_temporal_patterns()
            
            # Analyze trade-offs
            trade_off_analysis = self._analyze_trade_offs()
            
            comparative_results = {
                'policy_comparison': policy_comparison,
                'metric_comparison': metric_comparison,
                'temporal_comparison': temporal_comparison,
                'trade_offs': trade_off_analysis,
                'summary': self._summarize_comparative_analysis(
                    policy_comparison,
                    metric_comparison,
                    temporal_comparison,
                    trade_off_analysis
                )
            }
            
            return comparative_results
            
        except Exception as e:
            logger.error(f"Error in comparative analysis: {str(e)}")
            raise

    def _analyze_spf_performance(self, metric: str) -> Dict[str, float]:
        """Analyze SPF performance for given metric"""
        try:
            spf_results = self.results['spf_analysis'][metric]
            
            return {
                'r2_score': r2_score(
                    spf_results['actual'],
                    spf_results['predicted']
                ),
                'rmse': np.sqrt(mean_squared_error(
                    spf_results['actual'],
                    spf_results['predicted']
                )),
                'correlation': np.corrcoef(
                    spf_results['actual'],
                    spf_results['predicted']
                )[0,1],
                'explained_variance': self._calculate_explained_variance(
                    spf_results['actual'],
                    spf_results['predicted']
                )
            }
            
        except Exception as e:
            logger.error(f"Error analyzing SPF performance for {metric}: {str(e)}")
            return {}

    def _analyze_sensitivity_performance(self, metric: str) -> Dict[str, float]:
        """Analyze sensitivity analysis performance"""
        try:
            sensitivity_results = self.results['sensitivity_analysis'][metric]
            
            return {
                'average_sensitivity': np.mean(sensitivity_results['sensitivities']),
                'parameter_importance': self._calculate_parameter_importance(
                    sensitivity_results['parameter_effects']
                ),
                'interaction_strength': self._calculate_interaction_strength(
                    sensitivity_results['interactions']
                ),
                'robustness_score': self._calculate_robustness_score(
                    sensitivity_results['robustness']
                )
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sensitivity performance: {str(e)}")
            return {}

    def _compare_policy_approaches(self) -> Dict[str, Any]:
        """Compare FPS and PBS approaches"""
        try:
            fps_results = self._extract_fps_results()
            pbs_results = self._extract_pbs_results()
            
            comparison = {
                'performance_metrics': self._compare_performance_metrics(
                    fps_results, pbs_results
                ),
                'efficiency_metrics': self._compare_efficiency_metrics(
                    fps_results, pbs_results
                ),
                'equity_metrics': self._compare_equity_metrics(
                    fps_results, pbs_results
                ),
                'temporal_metrics': self._compare_temporal_metrics(
                    fps_results, pbs_results
                ),
                'statistical_tests': self._perform_statistical_comparison(
                    fps_results, pbs_results
                )
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing policy approaches: {str(e)}")
            return {}
        
    def analyze_temporal_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Analyze temporal patterns in results"""
        logger.info("Analyzing temporal patterns")
        temporal_results = {}

        try:
            for period in self.temporal_periods:
                period_analysis = {}
                
                for metric in self.metrics:
                    period_analysis[metric] = {
                        'pattern_metrics': self._analyze_temporal_metrics(metric, period),
                        'stability_analysis': self._analyze_temporal_stability(metric, period),
                        'peak_effects': self._analyze_peak_effects(metric, period),
                        'trend_analysis': self._analyze_temporal_trends(metric, period)
                    }
                
                temporal_results[period] = period_analysis

            # Add cross-period analysis
            temporal_results['cross_period'] = self._analyze_cross_period_effects()
            
            return temporal_results
            
        except Exception as e:
            logger.error(f"Error in temporal analysis: {str(e)}")
            raise

    def analyze_policy_impacts(self) -> Dict[str, Dict[str, Any]]:
        """Analyze policy impacts and effectiveness"""
        logger.info("Analyzing policy impacts")
        policy_results = {}

        try:
            # Analyze FPS impacts
            fps_impacts = self._analyze_fps_impacts()
            
            # Analyze PBS impacts
            pbs_impacts = self._analyze_pbs_impacts()
            
            # Compare policy effectiveness
            effectiveness_comparison = self._compare_policy_effectiveness()
            
            # Analyze distributional effects
            distributional_effects = self._analyze_distributional_effects()
            
            policy_results = {
                'fps_impacts': fps_impacts,
                'pbs_impacts': pbs_impacts,
                'effectiveness_comparison': effectiveness_comparison,
                'distributional_effects': distributional_effects,
                'recommendations': self._generate_policy_recommendations(
                    fps_impacts,
                    pbs_impacts,
                    effectiveness_comparison,
                    distributional_effects
                )
            }
            
            return policy_results
            
        except Exception as e:
            logger.error(f"Error in policy impact analysis: {str(e)}")
            raise

    def generate_visualization_suite(self, analysis_results: Dict):
        """Generate comprehensive visualization suite"""
        logger.info("Generating visualization suite")
        
        try:
            vis_dir = self.output_dir / 'visualizations'
            
            # Performance visualizations
            self._plot_performance_metrics(
                analysis_results['performance_analysis'],
                vis_dir / 'performance'
            )
            
            # Comparative visualizations
            self._plot_comparative_analysis(
                analysis_results['comparative_analysis'],
                vis_dir / 'comparative'
            )
            
            # Temporal visualizations
            self._plot_temporal_patterns(
                analysis_results['temporal_analysis'],
                vis_dir / 'temporal'
            )
            
            # Policy impact visualizations
            self._plot_policy_impacts(
                analysis_results['policy_analysis'],
                vis_dir / 'policy'
            )
            
            # Generate summary plots
            self._generate_summary_visualizations(
                analysis_results,
                vis_dir / 'summary'
            )
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            raise

    def _analyze_cross_period_effects(self) -> Dict[str, Any]:
        """Analyze effects across temporal periods"""
        try:
            cross_period_effects = {}
            
            for metric in self.metrics:
                metric_effects = {
                    'correlation_matrix': self._calculate_period_correlations(metric),
                    'transition_patterns': self._analyze_period_transitions(metric),
                    'stability_metrics': self._calculate_cross_period_stability(metric)
                }
                
                cross_period_effects[metric] = metric_effects
            
            return cross_period_effects
            
        except Exception as e:
            logger.error(f"Error analyzing cross-period effects: {str(e)}")
            return {}

    def _analyze_distributional_effects(self) -> Dict[str, Any]:
        """Analyze distributional effects of policies"""
        try:
            distributional_effects = {
                'fps_distribution': self._analyze_fps_distribution(),
                'pbs_distribution': self._analyze_pbs_distribution(),
                'equity_metrics': self._calculate_equity_metrics(),
                'fairness_analysis': self._analyze_fairness()
            }
            
            return distributional_effects
            
        except Exception as e:
            logger.error(f"Error analyzing distributional effects: {str(e)}")
            return {}
        
    def _plot_performance_metrics(self, performance_results: Dict, output_dir: Path):
        """Generate performance visualization plots"""
        output_dir.mkdir(exist_ok=True)
        
        # Overall Performance Plot
        plt.figure(figsize=(12, 8))
        metrics_df = pd.DataFrame(performance_results).applymap(
            lambda x: x['aggregate_performance']
        )
        sns.heatmap(metrics_df, annot=True, cmap='RdYlBu_r', center=0)
        plt.title('Overall Performance Metrics')
        plt.tight_layout()
        plt.savefig(output_dir / 'overall_performance.png')
        plt.close()
        
        # Component-wise Performance
        for metric in self.metrics:
            plt.figure(figsize=(15, 8))
            component_data = performance_results[metric]
            
            plt.subplot(2, 2, 1)
            self._plot_spf_performance(component_data['spf_metrics'])
            
            plt.subplot(2, 2, 2)
            self._plot_sensitivity_performance(component_data['sensitivity_metrics'])
            
            plt.subplot(2, 2, 3)
            self._plot_validation_performance(component_data['validation_metrics'])
            
            plt.subplot(2, 2, 4)
            self._plot_optimization_performance(component_data['optimization_metrics'])
            
            plt.suptitle(f'Component Performance - {metric}')
            plt.tight_layout()
            plt.savefig(output_dir / f'component_performance_{metric}.png')
            plt.close()

    def _plot_comparative_analysis(self, comparative_results: Dict, output_dir: Path):
        """Generate comparative analysis plots"""
        output_dir.mkdir(exist_ok=True)
        
        # Policy Comparison Plot
        plt.figure(figsize=(15, 10))
        policy_data = pd.DataFrame(comparative_results['policy_comparison']['performance_metrics'])
        
        plt.subplot(2, 1, 1)
        policy_data.plot(kind='bar')
        plt.title('Policy Performance Comparison')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 1, 2)
        self._plot_trade_offs(comparative_results['trade_offs'])
        
        plt.tight_layout()
        plt.savefig(output_dir / 'policy_comparison.png')
        plt.close()
        
        # Metric Comparison Plots
        self._plot_metric_comparisons(
            comparative_results['metric_comparison'],
            output_dir
        )

    def _plot_temporal_patterns(self, temporal_results: Dict, output_dir: Path):
        """Generate temporal pattern plots"""
        output_dir.mkdir(exist_ok=True)
        
        for period in self.temporal_periods:
            plt.figure(figsize=(15, 10))
            
            for idx, metric in enumerate(self.metrics, 1):
                plt.subplot(2, 2, idx)
                self._plot_temporal_metrics(
                    temporal_results[period][metric]['pattern_metrics'],
                    metric
                )
                
            plt.suptitle(f'Temporal Patterns - {period}')
            plt.tight_layout()
            plt.savefig(output_dir / f'temporal_patterns_{period}.png')
            plt.close()
        
        # Cross-period analysis plot
        self._plot_cross_period_analysis(
            temporal_results['cross_period'],
            output_dir
        )

    def _plot_policy_impacts(self, policy_results: Dict, output_dir: Path):
        """Generate policy impact plots"""
        output_dir.mkdir(exist_ok=True)
        
        # Impact Distribution Plot
        plt.figure(figsize=(12, 8))
        self._plot_impact_distribution(policy_results['distributional_effects'])
        plt.savefig(output_dir / 'impact_distribution.png')
        plt.close()
        
        # Effectiveness Comparison
        plt.figure(figsize=(12, 8))
        self._plot_effectiveness_comparison(policy_results['effectiveness_comparison'])
        plt.savefig(output_dir / 'effectiveness_comparison.png')
        plt.close()
        
        # Policy Recommendations
        self._create_recommendation_visualizations(
            policy_results['recommendations'],
            output_dir
        )

    def _generate_summary_visualizations(self, analysis_results: Dict, output_dir: Path):
        """Generate summary visualization plots"""
        output_dir.mkdir(exist_ok=True)
        
        # Overall Performance Summary
        plt.figure(figsize=(15, 10))
        self._plot_performance_summary(analysis_results)
        plt.savefig(output_dir / 'performance_summary.png')
        plt.close()
        
        # Key Findings Visualization
        plt.figure(figsize=(12, 8))
        self._plot_key_findings(analysis_results['summary'])
        plt.savefig(output_dir / 'key_findings.png')
        plt.close()
        
        # Recommendations Overview
        plt.figure(figsize=(12, 8))
        self._plot_recommendations_overview(analysis_results['summary'])
        plt.savefig(output_dir / 'recommendations_overview.png')
        plt.close()

    def _calculate_explained_variance(self, actual: np.ndarray, 
                                   predicted: np.ndarray) -> float:
        """Calculate explained variance score"""
        try:
            return 1 - np.var(actual - predicted) / np.var(actual)
        except Exception as e:
            logger.error(f"Error calculating explained variance: {str(e)}")
            return 0.0

    def _calculate_parameter_importance(self, parameter_effects: Dict) -> Dict[str, float]:
        """Calculate parameter importance scores"""
        try:
            total_effect = sum(abs(effect) for effect in parameter_effects.values())
            return {
                param: abs(effect) / total_effect
                for param, effect in parameter_effects.items()
            }
        except Exception as e:
            logger.error(f"Error calculating parameter importance: {str(e)}")
            return {}

    @staticmethod
    def _calculate_interaction_strength(interactions: Dict) -> float:
        """Calculate overall interaction strength"""
        try:
            return np.mean([abs(strength) for strength in interactions.values()])
        except Exception as e:
            logger.error(f"Error calculating interaction strength: {str(e)}")
            return 0.0

if __name__ == "__main__":
    # Example usage
    with open('analysis_results.pkl', 'rb') as f:
        results = pickle.load(f)
    
    analyzer = ResultsAnalyzer(
        results=results,
        output_dir='results_analysis_output',
        confidence_level=0.95
    )
    
    # Run comprehensive analysis
    analysis_results = analyzer.analyze_comprehensive_results()