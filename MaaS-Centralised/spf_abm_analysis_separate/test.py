import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Function to load pickle files and extract mode share data
def load_and_extract_data(file_pattern="fps_pbs_sweep_intermediate_*.pkl"):
    """
    Load pickle files matching the pattern and extract low income mode share data.
    
    Args:
        file_pattern: Pattern to match pickle files
        
    Returns:
        DataFrame with FPS values and corresponding mode shares
    """
    data_rows = []
    
    # List all files matching the pattern
    for file_path in Path('.').glob(file_pattern):
        try:
            # Extract FPS value from filename
            fps_value = int(file_path.stem.split('_')[-1])
            
            # Load the pickle file
            with open(file_path, 'rb') as f:
                result = pickle.load(f)
            
            # Check if it has the required structure
            if isinstance(result, dict) and 'full_results' in result:
                full_results = result['full_results']
                
                # Check if there's low income data with mode details
                if 'low' in full_results and 'mode_details' in full_results['low']:
                    # Process each mode's data
                    for mode_data in full_results['low']['mode_details']:
                        # Add row with mode share data
                        data_rows.append({
                            'fps_value': fps_value,
                            'mode': mode_data['mode'],
                            'target_share': mode_data['target_share'],
                            'actual_share': mode_data['actual_share'],
                            'share_ratio': mode_data.get('share_ratio', 0),
                            'adherence_score': mode_data.get('adherence_score', 0)
                        })
            print(f"Processed {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Convert to DataFrame
    if data_rows:
        return pd.DataFrame(data_rows)
    else:
        print("No valid data found in pickle files")
        return pd.DataFrame()

# Try loading from the main results file first
try:
    with open('fps_pbs_mae_sweep_results.pkl', 'rb') as f:
        all_results = pickle.load(f)
        
    data_rows = []
    for fps, result in all_results.items():
        if isinstance(result, dict) and 'full_results' in result:
            full_results = result['full_results']
            
            if 'low' in full_results and 'mode_details' in full_results['low']:
                for mode_data in full_results['low']['mode_details']:
                    data_rows.append({
                        'fps_value': fps,
                        'mode': mode_data['mode'],
                        'target_share': mode_data['target_share'],
                        'actual_share': mode_data['actual_share'],
                        'share_ratio': mode_data.get('share_ratio', 0),
                        'adherence_score': mode_data.get('adherence_score', 0)
                    })
    
    df = pd.DataFrame(data_rows)
    print("Successfully loaded data from main results file")
    
except Exception as e:
    print(f"Error loading main results file: {e}")
    print("Trying to load from intermediate files...")
    df = load_and_extract_data()

# Check if we have data
if df.empty:
    print("No data could be extracted from the files")
else:
    print(f"Loaded data with {len(df)} rows")
    print(f"FPS values found: {sorted(df['fps_value'].unique())}")
    print(f"Modes found: {sorted(df['mode'].unique())}")
    
    # Create visualizations
    # 1. Line plot of actual mode share vs FPS for low income
    plt.figure(figsize=(12, 8))
    
    # Sort by FPS value for line plotting
    df = df.sort_values('fps_value')
    
    # Plot each mode
    for mode in sorted(df['mode'].unique()):
        mode_data = df[df['mode'] == mode]
        plt.plot(mode_data['fps_value'], mode_data['actual_share'], 
                 'o-', linewidth=2, label=f'{mode}')
    
    plt.title('Low-Income Mode Share vs Fixed Pool Subsidy', fontsize=16)
    plt.xlabel('Fixed Pool Subsidy (FPS)', fontsize=14)
    plt.ylabel('Mode Share', fontsize=14)
    plt.xscale('log')  # Log scale for better visualization with wide range of FPS values
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='best')
    plt.tight_layout()
    plt.savefig('low_income_mode_share_vs_fps.png', dpi=300)
    
    # 2. Stacked area chart of mode distribution
    plt.figure(figsize=(12, 8))
    
    # Pivot data for stacked area chart
    pivot_df = df.pivot(index='fps_value', columns='mode', values='actual_share')
    
    # Fill NAs with 0
    pivot_df = pivot_df.fillna(0)
    
    # Plot stacked area
    plt.stackplot(pivot_df.index, 
                 [pivot_df[mode] for mode in pivot_df.columns],
                 labels=pivot_df.columns,
                 alpha=0.7)
    
    plt.title('Low-Income Mode Distribution vs Fixed Pool Subsidy', fontsize=16)
    plt.xlabel('Fixed Pool Subsidy (FPS)', fontsize=14)
    plt.ylabel('Mode Share Proportion', fontsize=14)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='best')
    plt.tight_layout()
    plt.savefig('low_income_mode_distribution_vs_fps.png', dpi=300)
    
    # 3. Combined plot with target vs actual for each mode
    modes = sorted(df['mode'].unique())
    n_modes = len(modes)
    
    # Calculate grid dimensions
    n_cols = min(3, n_modes)
    n_rows = (n_modes + n_cols - 1) // n_cols
    
    plt.figure(figsize=(5*n_cols, 4*n_rows))
    
    for i, mode in enumerate(modes):
        plt.subplot(n_rows, n_cols, i+1)
        
        mode_data = df[df['mode'] == mode]
        
        # Plot actual share
        plt.plot(mode_data['fps_value'], mode_data['actual_share'], 
                'bo-', linewidth=2, label='Actual', alpha=0.8)
        
        # Plot target share as horizontal line
        target = mode_data['target_share'].iloc[0]
        plt.axhline(y=target, color='r', linestyle='--', label=f'Target: {target:.3f}')
        
        plt.title(f'Mode: {mode}')
        plt.xlabel('FPS Value')
        plt.ylabel('Mode Share')
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('low_income_mode_target_vs_actual.png', dpi=300)
    
    # 4. Create a heatmap visualization showing how FPS values affect each mode
    plt.figure(figsize=(12, 8))
    
    # Normalize the mode shares for better visualization
    max_shares = pivot_df.max()
    norm_pivot = pivot_df.copy()
    for col in norm_pivot.columns:
        if max_shares[col] > 0:
            norm_pivot[col] = norm_pivot[col] / max_shares[col]
    
    # Create heatmap
    sns.heatmap(norm_pivot.T, annot=pivot_df.T.round(3), fmt='.3f',
                cmap='YlGnBu', linewidths=0.5, cbar_kws={'label': 'Normalized Share'})
    
    plt.title('FPS Impact on Low-Income Mode Shares', fontsize=16)
    plt.xlabel('FPS Value', fontsize=14)
    plt.ylabel('Mode', fontsize=14)
    plt.tight_layout()
    plt.savefig('low_income_mode_share_heatmap.png', dpi=300)
    
    # Save the data as CSV for reference
    df.to_csv('low_income_mode_share_data.csv', index=False)
    
    print("Analysis complete. Created visualizations:")
    print("1. low_income_mode_share_vs_fps.png")
    print("2. low_income_mode_distribution_vs_fps.png")
    print("3. low_income_mode_target_vs_actual.png")
    print("4. low_income_mode_share_heatmap.png")
    print("5. low_income_mode_share_data.csv (raw data)")