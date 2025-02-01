# sensitivity_check_visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import pandas as pd
from scipy import stats
import os

class SensitivityVisualizer:
    def __init__(self, output_dir):
        """Initialize visualizer with output directory"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style parameters
        plt.style.use('seaborn-v0_8')
        self.colors = sns.color_palette("husl", 8)
        
    def plot_parameter_impact_detail(self, param_values, results_df, param_name, stats):
        """Create detailed visualization of parameter impacts on metrics"""
        plt.close('all')
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Parameter distribution
        sns.histplot(param_values, kde=True, ax=ax1)
        ax1.set_title(f'Distribution of {param_name}')
        ax1.axvline(stats['mean'], color='r', linestyle='--', label='Mean')
        ax1.axvline(stats['quartiles'][0.5], color='g', linestyle='--', label='Median')
        ax1.legend()
        
        # 2. Impact on SDI
        sns.regplot(x=param_values, y=results_df['sdi'], ax=ax2)  # Changed 'SDI' to 'sdi'
        ax2.set_title(f'Impact on SDI Score\nr={stats["correlations"]["sdi"]["correlation"]:.3f}')
        
        # 3. Impact on components
        for idx, metric in enumerate(['sur', 'mae', 'upi']):  # Changed to lowercase
            sns.regplot(x=param_values, y=results_df[metric], 
                    ax=ax3, color=self.colors[idx], 
                    label=f'{metric.upper()} (r={stats["correlations"][metric]["correlation"]:.3f})')
        ax3.set_title('Impact on Component Metrics')
        ax3.legend()
        
        # 4. Box plot by quartiles
        param_quartiles = pd.qcut(param_values, 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        sns.boxplot(x=param_quartiles, y=results_df['sdi'], ax=ax4)  # Changed 'SDI' to 'sdi'
        ax4.set_title('SDI Distribution by Parameter Quartiles')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{param_name}_impact_detail.png')
        plt.close()
        
    def plot_interaction_heatmap(self, correlation_matrix, title):
        """Create heatmap visualization of parameter interactions"""
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(correlation_matrix))
        
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   vmin=-1, 
                   vmax=1)
        
        plt.title(title)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'parameter_interactions.png')
        plt.close()
        
    def plot_feature_importance(self, feature_importance_df):
        """Create feature importance visualization"""
        plt.figure(figsize=(10, 6))
        
        sns.barplot(x='importance', 
                   y='feature', 
                   data=feature_importance_df,
                   palette='viridis')
        
        plt.title('Parameter Importance (Random Forest)')
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_importance.png')
        plt.close()
        
    def plot_temporal_sensitivity(self, time_correlations, param_name):
        """Create visualization of temporal sensitivity patterns with statistical validation"""
        if time_correlations is None or len(time_correlations) < 2:
            print(f"Insufficient temporal data for parameter: {param_name}")
            return
            
        df = pd.DataFrame(time_correlations)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot correlation strength with error bands
        sns.lineplot(data=df, x='step', y='correlation', 
                    ci=95, err_style='band', ax=ax1)
                    
        # Add sample size indication
        sizes = df['num_samples'] / df['num_samples'].max() * 100
        ax1.scatter(df['step'], df['correlation'], 
                s=sizes, alpha=0.5, 
                label='Sample size')
        
        # Add parameter variation indication
        ax1.fill_between(df['step'], 
                        df['correlation'] - df['param_std'],
                        df['correlation'] + df['param_std'],
                        alpha=0.2, color='gray',
                        label='Parameter variation')
        
        ax1.set_title(f'Temporal Evolution of {param_name} Impact')
        ax1.set_ylabel('Correlation with SDI')
        ax1.legend()
        
        # Plot statistical significance
        sns.scatterplot(data=df, x='step', y='p_value', ax=ax2)
        ax2.axhline(y=0.05, color='r', linestyle='--', 
                    label='p=0.05 threshold')
        ax2.set_yscale('log')
        ax2.set_ylabel('p-value')
        ax2.legend()
        
        plt.tight_layout()
        return fig
        
    def plot_summary_dashboard(self, results_df, params_df, param_impacts, interactions):
        """Create summary dashboard of key findings with error handling"""
        fig = plt.figure(figsize=(20, 15))
        gs = plt.GridSpec(3, 3, figure=fig)
        
        try:
            # 1. Overall SDI Distribution
            ax1 = fig.add_subplot(gs[0, 0])
            sns.histplot(data=results_df, x='sdi', kde=True, ax=ax1)
            ax1.set_title('Distribution of SDI Scores')
            ax1.axvline(results_df['sdi'].mean(), color='r', linestyle='--', label='Mean')
            ax1.axvline(results_df['sdi'].median(), color='g', linestyle='--', label='Median')
            ax1.legend()

            # 2. Income Level Analysis
            ax2 = fig.add_subplot(gs[0, 1])
            sns.boxplot(data=results_df, x='income_level', y='sdi', ax=ax2)
            ax2.set_title('SDI by Income Level')
            ax2.set_xlabel('Income Level')
            ax2.set_ylabel('SDI Score')

            # 3. Component Analysis
            ax3 = fig.add_subplot(gs[1, 0:2])
            component_data = results_df[['sur', 'mae', 'upi']].melt()
            sns.boxplot(data=component_data, x='variable', y='value', ax=ax3)
            ax3.set_title('Distribution of Components')
            ax3.set_xlabel('Component')
            ax3.set_ylabel('Value')

            # 4. Subsidy Impact (FPS) or Mode Analysis (PBS)
            ax4 = fig.add_subplot(gs[2, 0:2])
            if 'subsidy_pool' in results_df.columns:
                # FPS Analysis
                sns.scatterplot(data=results_df, x='subsidy_pool', y='sdi', 
                            hue='income_level', ax=ax4)
                ax4.set_title('SDI vs Subsidy Pool Size')
                ax4.set_xlabel('Subsidy Pool')
            elif 'varied_mode' in results_df.columns:
                # PBS Analysis
                sns.boxplot(data=results_df, x='varied_mode', y='sdi', 
                        hue='income_level', ax=ax4)
                ax4.set_title('SDI by Mode and Income Level')
                ax4.set_xlabel('Mode')

            # 5. Component Correlations
            ax5 = fig.add_subplot(gs[0:2, 2])
            components = ['sdi', 'sur', 'mae', 'upi']
            corr_matrix = results_df[components].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', 
                    vmin=-1, vmax=1, ax=ax5)
            ax5.set_title('Component Correlations')

            # 6. Summary Statistics
            ax6 = fig.add_subplot(gs[2, 2])
            ax6.axis('off')
            summary_text = (
                f"Summary Statistics\n\n"
                f"Mean SDI: {results_df['sdi'].mean():.3f}\n"
                f"Median SDI: {results_df['sdi'].median():.3f}\n"
                f"Std Dev: {results_df['sdi'].std():.3f}\n"
                f"Samples: {len(results_df)}\n"
                f"Income Levels: {len(results_df['income_level'].unique())}"
            )
            ax6.text(0.1, 0.9, summary_text, 
                    transform=ax6.transAxes, 
                    bbox=dict(facecolor='white', alpha=0.8),
                    fontsize=10,
                    verticalalignment='top')

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'summary_dashboard.png'), 
                    dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Error creating dashboard: {str(e)}")
            import traceback
            traceback.print_exc()
            plt.close()  # Ensure figure is closed even if error occurs

    def create_detailed_parameter_report(self, param_name, param_data, results_df):
        """Create comprehensive visualization report for a single parameter"""
        fig = plt.figure(figsize=(15, 20))
        gs = plt.GridSpec(4, 2, figure=fig)

        # 1. Basic Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        sns.histplot(param_data, kde=True, ax=ax1)
        ax1.set_title(f'{param_name} Distribution')

        # 2. QQ Plot
        ax2 = fig.add_subplot(gs[0, 1])
        stats.probplot(param_data, dist="norm", plot=ax2)
        ax2.set_title('Normal Q-Q Plot')

        # 3. Impact on SDI with Income Level
        ax3 = fig.add_subplot(gs[1, :])
        sns.scatterplot(data=results_df, x=param_data, y='sdi', 
                       hue='income_level', style='income_level', ax=ax3)
        ax3.set_title(f'SDI vs {param_name} by Income Level')

        # 4. Component Impacts
        ax4 = fig.add_subplot(gs[2, :])
        components = ['sur', 'mae', 'upi']
        for comp in components:
            sns.regplot(x=param_data, y=results_df[comp], 
                       label=comp, scatter=False, ax=ax4)
        ax4.legend()
        ax4.set_title(f'Component Metrics vs {param_name}')

        # 5. Temporal Evolution
        ax5 = fig.add_subplot(gs[3, :])
        pivot_data = pd.pivot_table(results_df, 
                                  values='SDI', 
                                  index='step',
                                  columns=pd.qcut(param_data, 4, labels=['Q1', 'Q2', 'Q3', 'Q4']))
        sns.heatmap(pivot_data, cmap='viridis', ax=ax5)
        ax5.set_title(f'Temporal Evolution by {param_name} Quartiles')

        plt.tight_layout()
        plt.savefig(self.output_dir / f'{param_name}_detailed_report.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def plot_sensitivity_surface(self, param1, param2, results_df, metric='sdi'):
        """Create 3D surface plot showing interaction between two parameters"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Create mesh grid
        x = np.linspace(param1.min(), param1.max(), 20)
        y = np.linspace(param2.min(), param2.max(), 20)
        X, Y = np.meshgrid(x, y)

        # Fit interplation
        from scipy.interpolate import griddata
        Z = griddata((param1, param2), results_df[metric], 
                    (X, Y), method='cubic')

        # Plot surface
        surf = ax.plot_surface(X, Y, Z, cmap='viridis',
                             linewidth=0, antialiased=False)
        fig.colorbar(surf, ax=ax, label=metric)

        ax.set_xlabel(param1.name)
        ax.set_ylabel(param2.name)
        ax.set_zlabel(metric)
        plt.title(f'{metric} Response Surface')

        plt.tight_layout()
        plt.savefig(self.output_dir / f'surface_{param1.name}_{param2.name}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

    def plot_robustness_analysis(self, results_df, params_df):
        """Create visualization of system robustness to parameter variations"""
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(2, 2, figure=fig)

        # 1. SDI Robustness Overview
        ax1 = fig.add_subplot(gs[0, 0])
        sns.boxplot(data=results_df, y='sdi', x='income_level', ax=ax1)
        ax1.set_title('SDI Robustness by Income Level')

        # 2. Parameter Sensitivity Rankings
        ax2 = fig.add_subplot(gs[0, 1])
        sensitivity_scores = []
        for param in params_df.columns:
            if param != 'simulation_id':
                std_param = (params_df[param] - params_df[param].mean()) / params_df[param].std()
                sensitivity = np.std(results_df['sdi'] * std_param)
                sensitivity_scores.append({'parameter': param, 'sensitivity': sensitivity})
        
        sensitivity_df = pd.DataFrame(sensitivity_scores)
        sns.barplot(data=sensitivity_df, x='sensitivity', y='parameter', ax=ax2)
        ax2.set_title('Parameter Sensitivity Rankings')

        # 3. Stability Analysis
        ax3 = fig.add_subplot(gs[1, :])
        pivot_data = pd.pivot_table(results_df, 
                                  values='SDI',
                                  index='step',
                                  columns='income_level')
        pivot_data.plot(ax=ax3, style=['--', '-', ':'])
        ax3.fill_between(pivot_data.index,
                        pivot_data.min(axis=1),
                        pivot_data.max(axis=1),
                        alpha=0.2)
        ax3.set_title('SDI Stability Over Time')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'robustness_analysis.png',
                   dpi=300, bbox_inches='tight')
        plt.close()