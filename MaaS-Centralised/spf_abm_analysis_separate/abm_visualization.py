"""
ABM-ETOP Visualization Module

This script generates visualizations from the Agent-Based Model for 
Equity-Transportation Optimization and Policy (ABM-ETOP) framework.

It can be run as a standalone script after simulations have been completed
or integrated with the simulation workflow.

Usage:
    python abm_visualization.py --db_path path/to/database.db --output_dir path/to/output
    
    or
    
    python abm_visualization.py --run_simulation --steps 100 --output_dir path/to/output
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation
import networkx as nx
from collections import defaultdict
import os
import argparse
import warnings
import sqlite3
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import json
import sys
import traceback

# Add parent directory to path to import model modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import model modules - these will only work if the script is run from the correct directory
try:
    from run_visualisation_03 import MobilityModel
    from database_01 import (DB_CONNECTION_STRING, num_commuters, grid_width, grid_height, 
                            income_weights, health_weights, payment_weights, age_distribution, 
                            disability_weights, tech_access_weights, subsidy_dataset, daily_config,
                            SIMULATION_STEPS, ASC_VALUES, UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS,
                            UTILITY_FUNCTION_BASE_COEFFICIENTS, PENALTY_COEFFICIENTS,
                            AFFORDABILITY_THRESHOLDS, FLEXIBILITY_ADJUSTMENTS, VALUE_OF_TIME,
                            public_price_table, ALPHA_VALUES, DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS,
                            BACKGROUND_TRAFFIC_AMOUNT, CONGESTION_ALPHA, CONGESTION_BETA,
                            CONGESTION_CAPACITY, CONGESTION_T_IJ_FREE_FLOW,
                            UberLike1_cpacity, UberLike1_price, UberLike2_cpacity, UberLike2_price,
                            BikeShare1_capacity, BikeShare1_price, BikeShare2_capacity, BikeShare2_price)
    MODEL_IMPORTS_AVAILABLE = True
except ImportError:
    warnings.warn("Model modules could not be imported. Some functionality may be limited.")
    MODEL_IMPORTS_AVAILABLE = False


class ABMVisualizer:
    """
    A class for generating visualizations from the ABM-ETOP model results.
    This class provides methods to visualize spatial distributions,
    temporal evolution, and agent trajectories.
    """
    
    def __init__(self, model=None, db_connection_string=None, output_dir='visualization_outputs'):
        """
        Initialize the visualizer with either a running model instance or
        a database connection string.
        
        Args:
            model: A MobilityModel instance
            db_connection_string: Connection string to the simulation database
            output_dir: Directory to save visualizations
        """
        self.model = model
        self.db_connection_string = db_connection_string
        
        # Initialize database connection if string provided
        if db_connection_string and not model:
            try:
                self.engine = create_engine(db_connection_string)
                # Test connection
                with self.engine.connect() as conn:
                    result = conn.execute(text("SELECT 1"))
                    if not result.fetchone():
                        raise Exception("Database connection test failed")
                print(f"Successfully connected to database: {db_connection_string}")
            except Exception as e:
                print(f"Error connecting to database: {e}")
                self.engine = None
        elif model:
            self.engine = model.db_engine if hasattr(model, 'db_engine') else None
        
        # Define color schemes for income groups
        self.income_colors = {
            'low': '#1f77b4',    # Blue
            'middle': '#ff7f0e', # Orange
            'high': '#2ca02c'    # Green
        }
        
        # Define color schemes for transportation modes
        self.mode_colors = {
            'walk': '#7f7f7f',   # Gray
            'bike': '#17becf',   # Cyan
            'car': '#d62728',    # Red
            'bus': '#9467bd',    # Purple
            'train': '#8c564b',  # Brown
            'MaaS_Bundle': '#e377c2'  # Pink
        }
        
        # Create output directory for visualizations
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Get grid dimensions
        if model:
            self.grid_width = model.grid_width
            self.grid_height = model.grid_height
        else:
            # Default dimensions if not available
            self.grid_width = 55
            self.grid_height = 55
    
    def _execute_query(self, query, params=None):
        """
        Execute a SQL query and return the results as a DataFrame.
        
        Args:
            query: SQL query text
            params: Parameters for the query
            
        Returns:
            pandas DataFrame with query results
        """
        if not self.engine:
            print("No database connection available.")
            return pd.DataFrame()
        
        try:
            with self.engine.connect() as conn:
                if params:
                    result = conn.execute(text(query), params)
                else:
                    result = conn.execute(text(query))
                return pd.DataFrame(result.fetchall(), columns=result.keys())
        except Exception as e:
            print(f"Error executing query: {e}")
            return pd.DataFrame()
    
    def _get_station_data(self):
        """
        Get station location data from the database or model.
        
        Returns:
            Dictionary with station data by mode
        """
        stations = {'train': {}, 'bus': {}}
        
        if self.model and hasattr(self.model, 'maas_agent') and hasattr(self.model.maas_agent, 'stations'):
            # Get data directly from model
            return self.model.maas_agent.stations
        
        # Try to get from database if available
        if self.engine:
            try:
                # This is a placeholder - you'll need to adjust based on your database schema
                query = """
                SELECT station_id, x_coord, y_coord, station_type 
                FROM stations
                """
                stations_df = self._execute_query(query)
                
                for _, row in stations_df.iterrows():
                    station_type = row['station_type'].lower()
                    if station_type in stations:
                        stations[station_type][row['station_id']] = (row['x_coord'], row['y_coord'])
                
                if stations['train'] or stations['bus']:
                    return stations
            except Exception as e:
                print(f"Error retrieving station data: {e}")
        
        # If we couldn't get the data, return empty
        return stations
    
    def _get_agent_positions(self, step=None):
        """
        Get agent positions for a specific time step.
        
        Args:
            step: Simulation step (None for latest)
            
        Returns:
            DataFrame with agent positions and attributes
        """
        if self.model:
            # Get from model memory
            data = []
            for agent in self.model.commuter_agents:
                data.append({
                    'agent_id': agent.unique_id,
                    'x': agent.location[0],
                    'y': agent.location[1],
                    'income_level': agent.income_level,
                    'current_mode': agent.current_mode,
                    'has_disability': agent.has_disability,
                    'age': agent.age
                })
            return pd.DataFrame(data)
        
        if self.engine:
            # Try to get from database
            # This needs to be customized based on your database schema
            if step is not None:
                query = """
                SELECT a.commuter_id as agent_id, a.location ->> '$.x' as x, 
                       a.location ->> '$.y' as y, a.income_level, 
                       a.current_mode, a.has_disability, a.age
                FROM agent_positions a
                WHERE a.step = :step
                """
                return self._execute_query(query, {'step': step})
            else:
                query = """
                SELECT c.commuter_id as agent_id, c.location[0] as x, 
                       c.location[1] as y, c.income_level, 
                       CASE WHEN s.status = 'Service Selected' THEN s.record_company_name ELSE NULL END as current_mode,
                       c.has_disability, c.age
                FROM commuter_info_log c
                LEFT JOIN service_booking_log s ON c.commuter_id = s.commuter_id
                """
                return self._execute_query(query)
                
        # Return empty DataFrame if no data available
        return pd.DataFrame(columns=['agent_id', 'x', 'y', 'income_level', 'current_mode', 'has_disability', 'age'])
    
    def _get_trip_data(self):
        """
        Get completed trip data from the database.
        
        Returns:
            DataFrame with trip details
        """
        if not self.engine:
            return pd.DataFrame()
        
        query = """
        SELECT 
            c.income_level,
            s.total_time,
            s.record_company_name as mode,
            s.origin_coordinates,
            s.destination_coordinates
        FROM service_booking_log s
        JOIN commuter_info_log c ON s.commuter_id = c.commuter_id
        WHERE s.status = 'finished'
        """
        
        trip_data = self._execute_query(query)
        
        # Process coordinates if they're stored as JSON strings
        if not trip_data.empty:
            # Process origin coordinates
            if 'origin_coordinates' in trip_data.columns:
                if trip_data['origin_coordinates'].dtype == 'object':
                    try:
                        # Convert from JSON if stored as string
                        trip_data['origin_x'] = trip_data['origin_coordinates'].apply(
                            lambda x: json.loads(x)[0] if isinstance(x, str) else x[0] if isinstance(x, list) else None
                        )
                        trip_data['origin_y'] = trip_data['origin_coordinates'].apply(
                            lambda x: json.loads(x)[1] if isinstance(x, str) else x[1] if isinstance(x, list) else None
                        )
                    except:
                        pass
            
            # Process destination coordinates
            if 'destination_coordinates' in trip_data.columns:
                if trip_data['destination_coordinates'].dtype == 'object':
                    try:
                        # Convert from JSON if stored as string
                        trip_data['dest_x'] = trip_data['destination_coordinates'].apply(
                            lambda x: json.loads(x)[0] if isinstance(x, str) else x[0] if isinstance(x, list) else None
                        )
                        trip_data['dest_y'] = trip_data['destination_coordinates'].apply(
                            lambda x: json.loads(x)[1] if isinstance(x, str) else x[1] if isinstance(x, list) else None
                        )
                    except:
                        pass
        
        return trip_data
    
    def _get_route_data(self):
        """
        Get route data from the database for network usage analysis.
        
        Returns:
            DataFrame with route details
        """
        if not self.engine:
            return pd.DataFrame()
        
        query = """
        SELECT 
            route_details,
            c.income_level
        FROM share_service_booking_log s
        JOIN commuter_info_log c ON s.commuter_id = c.commuter_id
        WHERE route_details IS NOT NULL
        """
        
        routes = self._execute_query(query)
        
        # Process route data if necessary
        # This depends on how routes are stored in your database
        
        return routes
    
    def _get_mode_share_data(self):
        """
        Get mode share data over time.
        
        Returns:
            DataFrame with mode shares by income level over time
        """
        if self.model and hasattr(self.model, 'mode_share_history'):
            # Get from model memory
            mode_share_history = self.model.mode_share_history
            data = []
            
            for step, income_data in mode_share_history.items():
                for income, mode_data in income_data.items():
                    for mode, share in mode_data.items():
                        data.append({
                            'step': step,
                            'income_level': income,
                            'mode': mode,
                            'share': share
                        })
            
            return pd.DataFrame(data)
        
        # Try to get from database if we don't have model data
        if self.engine:
            # This would need to be customized based on your database schema
            query = """
            SELECT 
                step,
                income_level,
                mode,
                share_percentage as share
            FROM mode_shares
            ORDER BY step, income_level, mode
            """
            
            return self._execute_query(query)
        
        # Return empty DataFrame if no data available
        return pd.DataFrame(columns=['step', 'income_level', 'mode', 'share'])
    
    def plot_spatial_distribution(self, step=None, show_stations=True, 
                                 by_income=True, by_mode=False, save=True):
        """
        Generate a spatial distribution map of agents at a given time step.
        
        Args:
            step: Time step to visualize (None for current)
            show_stations: Whether to show station locations
            by_income: Color agents by income level
            by_mode: Color agents by transportation mode
            save: Whether to save the visualization
            
        Returns:
            Matplotlib figure
        """
        # Get agent position data
        agent_data = self._get_agent_positions(step)
        
        if agent_data.empty:
            print("No agent data available.")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Set axis limits with some padding
        ax.set_xlim(-1, self.grid_width + 1)
        ax.set_ylim(-1, self.grid_height + 1)
        
        # Add background grid
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Plot stations if requested
        if show_stations:
            stations = self._get_station_data()
            
            if stations:
                # Extract station coordinates
                train_stations = []
                bus_stations = []
                
                for station_id, coords in stations.get('train', {}).items():
                    train_stations.append(coords)
                
                for station_id, coords in stations.get('bus', {}).items():
                    bus_stations.append(coords)
                
                # Plot stations with appropriate markers
                if train_stations:
                    train_x, train_y = zip(*train_stations)
                    ax.scatter(train_x, train_y, color='black', marker='s', s=30, 
                              label='Train Station', alpha=0.7)
                
                if bus_stations:
                    bus_x, bus_y = zip(*bus_stations)
                    ax.scatter(bus_x, bus_y, color='black', marker='^', s=20, 
                              label='Bus Station', alpha=0.7)
        
        # Convert coordinates to numeric
        agent_data['x'] = pd.to_numeric(agent_data['x'], errors='coerce')
        agent_data['y'] = pd.to_numeric(agent_data['y'], errors='coerce')
        
        # Drop rows with invalid coordinates
        agent_data = agent_data.dropna(subset=['x', 'y'])
        
        # Plot agents with appropriate coloring
        if by_income and not by_mode:
            # Group by income level
            for income, group in agent_data.groupby('income_level'):
                if income in self.income_colors:
                    ax.scatter(group['x'], group['y'], 
                              color=self.income_colors[income],
                              label=f'{income.capitalize()} Income', 
                              alpha=0.7, s=50)
                else:
                    ax.scatter(group['x'], group['y'], 
                              color='gray', label=f'Other ({income})', 
                              alpha=0.7, s=50)
                
        elif by_mode and not by_income:
            # Group by transportation mode
            for mode, group in agent_data.groupby('current_mode'):
                if pd.isnull(mode) or mode is None:
                    label = 'Not Traveling'
                    color = 'gray'
                else:
                    mode_str = str(mode)
                    # Get base mode (e.g., 'car' from 'car_UberLike1')
                    base_mode = mode_str.split('_')[0] if '_' in mode_str else mode_str
                    label = base_mode.capitalize()
                    color = self.mode_colors.get(base_mode, 'gray')
                    
                ax.scatter(group['x'], group['y'], 
                          color=color, label=label, 
                          alpha=0.7, s=50)
        
        else:
            # Default: show all agents in one color
            ax.scatter(agent_data['x'], agent_data['y'], 
                      color='blue', label='Commuters', 
                      alpha=0.7, s=30)
        
        # Add legend, title and labels
        step_str = f"Step {step}" if step is not None else "Current Step"
        
        ax.set_title(f'Spatial Distribution of Agents at {step_str}', 
                    fontsize=16)
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        
        # Create legend with unique entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        if save:
            color_by = 'income' if by_income else 'mode' if by_mode else 'none'
            step_label = f"step{step}" if step is not None else "current"
            plt.savefig(f'{self.output_dir}/spatial_dist_{step_label}_{color_by}.png', 
                       dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_trip_origin_destination_map(self, by_income=True, by_mode=False, save=True):
        """
        Plot origin-destination flows on the map.
        
        Args:
            by_income: Color flows by income level
            by_mode: Color flows by mode
            save: Whether to save the visualization
            
        Returns:
            Matplotlib figure
        """
        # Get trip data
        trip_data = self._get_trip_data()
        
        if trip_data.empty or 'origin_x' not in trip_data.columns:
            print("No suitable trip data available for OD visualization.")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Set axis limits with some padding
        ax.set_xlim(-1, self.grid_width + 1)
        ax.set_ylim(-1, self.grid_height + 1)
        
        # Add background grid
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Plot stations
        stations = self._get_station_data()
        
        if stations:
            # Extract station coordinates
            train_stations = []
            bus_stations = []
            
            for station_id, coords in stations.get('train', {}).items():
                train_stations.append(coords)
            
            for station_id, coords in stations.get('bus', {}).items():
                bus_stations.append(coords)
            
            # Plot stations with appropriate markers
            if train_stations:
                train_x, train_y = zip(*train_stations)
                ax.scatter(train_x, train_y, color='black', marker='s', s=30, 
                          label='Train Station', alpha=0.7)
            
            if bus_stations:
                bus_x, bus_y = zip(*bus_stations)
                ax.scatter(bus_x, bus_y, color='black', marker='^', s=20, 
                          label='Bus Station', alpha=0.7)
        
        # Plot OD flows
        if by_income and not by_mode:
            for income, group in trip_data.groupby('income_level'):
                for _, trip in group.iterrows():
                    ax.plot([trip['origin_x'], trip['dest_x']], 
                           [trip['origin_y'], trip['dest_y']], 
                           color=self.income_colors.get(income, 'gray'), 
                           alpha=0.1, linewidth=0.5)
                
                # Plot a sample line for the legend
                ax.plot([-10, -10], [-10, -10], 
                       color=self.income_colors.get(income, 'gray'), 
                       label=f'{income.capitalize()} Income', 
                       alpha=0.7, linewidth=2)
                
        elif by_mode and not by_income:
            for mode, group in trip_data.groupby('mode'):
                # Get base mode
                base_mode = mode.split('_')[0] if '_' in mode else mode
                
                for _, trip in group.iterrows():
                    ax.plot([trip['origin_x'], trip['dest_x']], 
                           [trip['origin_y'], trip['dest_y']], 
                           color=self.mode_colors.get(base_mode, 'gray'), 
                           alpha=0.1, linewidth=0.5)
                
                # Plot a sample line for the legend
                ax.plot([-10, -10], [-10, -10], 
                       color=self.mode_colors.get(base_mode, 'gray'), 
                       label=f'{base_mode.capitalize()}', 
                       alpha=0.7, linewidth=2)
        
        else:
            # Default: show all trips in one color
            for _, trip in trip_data.iterrows():
                ax.plot([trip['origin_x'], trip['dest_x']], 
                       [trip['origin_y'], trip['dest_y']], 
                       color='blue', alpha=0.1, linewidth=0.5)
        
        # Add legend, title and labels
        ax.set_title('Origin-Destination Flows', fontsize=16)
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        
        # Create legend with unique entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        if save:
            color_by = 'income' if by_income else 'mode' if by_mode else 'none'
            plt.savefig(f'{self.output_dir}/od_flows_{color_by}.png', 
                       dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_network_usage_heatmap(self, save=True):
        """
        Generate a heatmap showing which areas of the network are most heavily used.
        
        Args:
            save: Whether to save the visualization
            
        Returns:
            Matplotlib figure
        """
        # Get route data
        route_data = self._get_route_data()
        
        if route_data.empty:
            print("No route data available for network usage heatmap.")
            return None
        
        # Initialize heat map data
        heat_map = np.zeros((self.grid_height, self.grid_width))
        
        # Process route data to generate the heatmap
        route_counts = 0
        
        for _, row in route_data.iterrows():
            route_details = row['route_details']
            income_level = row['income_level']
            
            # Process route details - this depends on your data format
            try:
                # This might need custom parsing depending on your database schema
                if isinstance(route_details, str):
                    # Try to parse JSON string
                    route_coords = json.loads(route_details)
                elif isinstance(route_details, list):
                    route_coords = route_details
                else:
                    continue
                
                # Process coordinates
                for coord in route_coords:
                    if isinstance(coord, list) or isinstance(coord, tuple):
                        x, y = coord[0], coord[1]
                        if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                            heat_map[int(y), int(x)] += 1
                            route_counts += 1
            except Exception as e:
                # Skip routes that can't be parsed
                continue
        
        if route_counts == 0:
            print("No valid route data could be processed.")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create custom colormap from light to dark
        cmap = LinearSegmentedColormap.from_list(
            'traffic_cmap', ['#f7fbff', '#3182bd', '#08306b'], N=256)
        
        # Plot heatmap
        im = ax.imshow(heat_map, cmap=cmap, interpolation='nearest',
                     extent=[0, self.grid_width, 0, self.grid_height],
                     origin='lower')
        
        # Add stations
        stations = self._get_station_data()
        
        if stations:
            for mode, station_dict in stations.items():
                marker = 's' if mode == 'train' else '^'
                color = 'red' if mode == 'train' else 'orange'
                
                for station_id, (x, y) in station_dict.items():
                    ax.scatter(x, y, color=color, marker=marker, s=30, 
                              edgecolors='black', zorder=5)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Traffic Intensity', fontsize=12)
        
        # Add labels and title
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.set_title('Network Usage Heatmap', fontsize=16)
        
        # Add legend for stations
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='s', color='w', markerfacecolor='red',
                  markersize=10, label='Train Station'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='orange',
                  markersize=10, label='Bus Station')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/network_usage_heatmap.png', 
                       dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_travel_time_distributions(self, save=True):
        """
        Plot the distribution of travel times for different income groups.
        
        Args:
            save: Whether to save the visualization
            
        Returns:
            Matplotlib figure
        """
        # Get trip data
        trip_data = self._get_trip_data()
        
        if trip_data.empty or 'total_time' not in trip_data.columns:
            print("No travel time data available.")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot kernel density estimates for each income group
        for income in ['low', 'middle', 'high']:
            income_trips = trip_data[trip_data['income_level'] == income]
            
            if not income_trips.empty:
                # Convert to numeric if needed
                income_trips['total_time'] = pd.to_numeric(income_trips['total_time'], errors='coerce')
                income_trips = income_trips.dropna(subset=['total_time'])
                
                if not income_trips.empty:
                    sns.kdeplot(income_trips['total_time'], 
                              label=f'{income.capitalize()} Income',
                              color=self.income_colors[income],
                              fill=True, alpha=0.3)
                    
                    # Add markers for means
                    mean_time = income_trips['total_time'].mean()
                    ax.axvline(mean_time, color=self.income_colors[income], 
                              linestyle='--', alpha=0.7)
                    
                    # Get current y-limits for text positioning
                    ylim = ax.get_ylim()
                    ax.text(mean_time, ylim[1]*0.9*(0.7 + income_trips['total_time'].count()/trip_data['total_time'].count()), 
                           f'Mean: {mean_time:.1f}',
                           color=self.income_colors[income],
                           horizontalalignment='center')
        
        # Add labels and legend
        ax.set_xlabel('Travel Time (minutes)', fontsize=14)
        ax.set_ylabel('Density', fontsize=14)
        ax.set_title('Distribution of Travel Times by Income Group', fontsize=16)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Ensure x-axis starts at 0
        xmin, xmax = ax.get_xlim()
        ax.set_xlim(0, xmax)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/travel_time_distributions.png', 
                       dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_mode_share_evolution(self, save=True):
        """
        Plot how mode shares evolve over time, broken down by income group.
        
        Args:
            save: Whether to save the visualization
            
        Returns:
            Matplotlib figure
        """
        # Get mode share data
        mode_share_data = self._get_mode_share_data()
        
        if mode_share_data.empty:
            print("No mode share data available.")
            return None
        
        # Create a multi-panel figure
        fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        for i, income in enumerate(['low', 'middle', 'high']):
            ax = axes[i]
            
            # Filter data for this income group
            income_data = mode_share_data[mode_share_data['income_level'] == income]
            
            if not income_data.empty:
                # Pivot to get mode shares by step
                income_pivot = income_data.pivot_table(
                    index='step', columns='mode', values='share', aggfunc='first'
                ).fillna(0)
                
                # Make sure step is numeric
                income_pivot.index = pd.to_numeric(income_pivot.index)
                
                # Sort by step
                income_pivot = income_pivot.sort_index()
                
                # Plot each mode
                for mode in income_pivot.columns:
                    if mode in self.mode_colors:
                        color = self.mode_colors[mode]
                    elif mode.startswith('car_') or mode.startswith('bike_'):
                        base_mode = mode.split('_')[0]
                        color = self.mode_colors.get(base_mode, 'gray')
                    else:
                        color = 'gray'
                    
                    ax.plot(income_pivot.index, income_pivot[mode], 
                           label=mode.capitalize().replace('_', ' '), 
                           color=color, marker='o', markersize=3, linewidth=2)
                
                ax.set_ylabel(f'{income.capitalize()} Income\nMode Share (%)', 
                             fontsize=12)
                ax.set_title(f'Mode Share Evolution: {income.capitalize()} Income',
                            fontsize=14)
                ax.grid(True, alpha=0.3)
                
                # Create legend with unique entries
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), 
                         loc='center left', bbox_to_anchor=(1, 0.5))
                
                ax.set_ylim(0, 100)
            
        # Set common x-label
        axes[-1].set_xlabel('Simulation Step', fontsize=12)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/mode_share_evolution.png', 
                       dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_subsidy_allocation_comparison(self, full_results=None, save=True):
        """
        Plot subsidy allocation comparison and its impacts.
        
        Args:
            full_results: Optimization results dictionary (optional)
            save: Whether to save the visualization
            
        Returns:
            Matplotlib figure
        """
        if not full_results:
            print("No optimization results provided for subsidy comparison.")
            return None
        
        # Extract data from results
        data = []
        
        for fps, result in full_results.items():
            if 'optimal_allocations' in result:
                fps_data = {'fps': fps}
                
                # Add travel time equity
                if 'travel_time_equity_index' in result:
                    fps_data['equity_index'] = result['travel_time_equity_index']
                
                # Add average travel times
                if 'full_results' in result:
                    for income in ['low', 'middle', 'high']:
                        if income in result['full_results']:
                            fps_data[f'{income}_avg_time'] = result['full_results'][income].get('avg_travel_time', 0)
                
                # Add subsidy allocations
                for key, value in result['optimal_allocations'].items():
                    fps_data[key] = value
                
                data.append(fps_data)
        
        if not data:
            print("No valid data extracted from optimization results.")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Ensure FPS values are numeric and sorted
        df['fps'] = pd.to_numeric(df['fps'])
        df = df.sort_values('fps')
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(15, 18))
        
        # Plot 1: Subsidy allocation by income level
        ax1 = fig.add_subplot(3, 1, 1)
        
        # Calculate aggregate subsidy allocations by income level
        income_levels = ['low', 'middle', 'high']
        modes = ['bike', 'car', 'MaaS_Bundle', 'public']
        
        for income in income_levels:
            # Calculate average allocation across modes
            avg_columns = [col for col in df.columns if col.startswith(f'{income}_') and col.split('_')[1] in modes]
            if avg_columns:
                df[f'{income}_avg'] = df[avg_columns].mean(axis=1)
                
                ax1.plot(df['fps'], df[f'{income}_avg'], 
                        label=f'{income.capitalize()} Income',
                        color=self.income_colors[income],
                        marker='o', markersize=5, linewidth=2)
        
        ax1.set_title('Average Subsidy Allocation by Income Level', fontsize=16)
        ax1.set_xlabel('Fixed Pool Subsidy (FPS)', fontsize=12)
        ax1.set_ylabel('Average Allocation (% of Cost)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Travel time equity index
        if 'equity_index' in df.columns:
            ax2 = fig.add_subplot(3, 1, 2)
            
            ax2.plot(df['fps'], df['equity_index'], 
                    color='purple', marker='o', 
                    markersize=5, linewidth=2)
            
            ax2.set_title('Travel Time Equity Index vs. FPS', fontsize=16)
            ax2.set_xlabel('Fixed Pool Subsidy (FPS)', fontsize=12)
            ax2.set_ylabel('Equity Index (Lower is Better)', fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            # Highlight minimum point
            min_idx = df['equity_index'].idxmin()
            min_fps = df.loc[min_idx, 'fps']
            min_equity = df.loc[min_idx, 'equity_index']
            
            ax2.scatter([min_fps], [min_equity], s=100, color='red', zorder=5)
            ax2.annotate(f'Optimal: FPS={min_fps:.0f}',
                        xy=(min_fps, min_equity),
                        xytext=(10, -30),
                        textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))
        
        # Plot 3: Average travel times by income
        ax3 = fig.add_subplot(3, 1, 3)
        
        for income in income_levels:
            col = f'{income}_avg_time'
            if col in df.columns:
                ax3.plot(df['fps'], df[col], 
                        label=f'{income.capitalize()} Income',
                        color=self.income_colors[income],
                        marker='o', markersize=5, linewidth=2)
        
        ax3.set_title('Average Travel Time by Income Level', fontsize=16)
        ax3.set_xlabel('Fixed Pool Subsidy (FPS)', fontsize=12)
        ax3.set_ylabel('Average Travel Time (minutes)', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/subsidy_allocation_comparison.png', 
                       dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_distance_to_station_by_income(self, save=True):
        """
        Plot distance to nearest station by income level.
        
        Args:
            save: Whether to save the visualization
            
        Returns:
            Matplotlib figure
        """
        # Get agent data
        agent_data = self._get_agent_positions()
        
        if agent_data.empty:
            print("No agent data available.")
            return None
        
        # Get station data
        stations = self._get_station_data()
        
        if not stations:
            print("No station data available.")
            return None
        
        # Combine all stations
        all_stations = []
        for mode, mode_stations in stations.items():
            for station_id, coords in mode_stations.items():
                all_stations.append(coords)
        
        if not all_stations:
            print("No station coordinates available.")
            return None
        
        # Calculate distance to nearest station
        agent_data['distance_to_station'] = agent_data.apply(
            lambda row: min(
                np.sqrt((row['x'] - station[0])**2 + (row['y'] - station[1])**2)
                for station in all_stations
            ) if not pd.isnull(row['x']) and not pd.isnull(row['y']) else np.nan,
            axis=1
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot KDE for each income group
        for income in ['low', 'middle', 'high']:
            income_data = agent_data[agent_data['income_level'] == income]
            
            if not income_data.empty:
                sns.kdeplot(income_data['distance_to_station'], 
                          label=f'{income.capitalize()} Income',
                          color=self.income_colors[income],
                          fill=True, alpha=0.3)
                
                # Add markers for means
                mean_dist = income_data['distance_to_station'].mean()
                ax.axvline(mean_dist, color=self.income_colors[income], 
                          linestyle='--', alpha=0.7)
                
                # Get current y-limits for text positioning
                ylim = ax.get_ylim()
                ax.text(mean_dist, ylim[1]*0.9*(0.7 + 0.1*income_data.index.size/agent_data.index.size), 
                       f'Mean: {mean_dist:.1f}',
                       color=self.income_colors[income],
                       horizontalalignment='center')
        
        # Add labels and legend
        ax.set_xlabel('Distance to Nearest Station', fontsize=14)
        ax.set_ylabel('Density', fontsize=14)
        ax.set_title('Distribution of Distances to Nearest Station by Income Group', fontsize=16)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Ensure x-axis starts at 0
        xmin, xmax = ax.get_xlim()
        ax.set_xlim(0, xmax)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/distance_to_station_by_income.png', 
                       dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_all_visualizations(self, step=None):
        """
        Generate all available visualizations.
        
        Args:
            step: Time step to visualize (None for latest)
            
        Returns:
            Dictionary of figure objects
        """
        print("Generating all visualizations...")
        figures = {}
        
        # Spatial distribution
        print("Generating spatial distribution plot...")
        figures['spatial_distribution'] = self.plot_spatial_distribution(
            step=step, by_income=True, save=True)
        
        # Trip origin-destination map
        print("Generating trip origin-destination map...")
        figures['od_map'] = self.plot_trip_origin_destination_map(
            by_income=True, save=True)
        
        # Network usage heatmap
        print("Generating network usage heatmap...")
        figures['network_usage'] = self.plot_network_usage_heatmap(save=True)
        
        # Travel time distributions
        print("Generating travel time distributions...")
        figures['travel_time'] = self.plot_travel_time_distributions(save=True)
        
        # Mode share evolution
        print("Generating mode share evolution plot...")
        figures['mode_share'] = self.plot_mode_share_evolution(save=True)
        
        # Distance to station by income
        print("Generating distance to station plot...")
        figures['distance_to_station'] = self.plot_distance_to_station_by_income(save=True)
        
        print(f"All visualizations saved to {self.output_dir}")
        return figures


def add_trajectory_tracking(model_class):
    """
    Adds trajectory tracking capability to a MobilityModel class.
    This function monkey-patches the step method to record agent positions.
    
    Args:
        model_class: The MobilityModel class to modify
        
    Returns:
        The modified class
    """
    if not hasattr(model_class, 'original_step'):
        # Store the original method
        model_class.original_step = model_class.step
        
        def step_with_tracking(self, *args, **kwargs):
            """Enhanced step method with trajectory tracking"""
            # Initialize trajectory tracking if not already done
            if not hasattr(self, 'agent_trajectories'):
                self.agent_trajectories = defaultdict(list)
                
            # Initialize mode share history if not already done
            if not hasattr(self, 'mode_share_history'):
                self.mode_share_history = {}
            
            # Record agent positions
            for agent in self.commuter_agents:
                self.agent_trajectories[agent.unique_id].append(
                    (self.current_step, agent.location[0], agent.location[1], 
                     agent.current_mode)
                )
            
            # Record mode shares
            mode_shares = defaultdict(lambda: defaultdict(float))
            
            for income_level in ['low', 'middle', 'high']:
                # Count agents by mode for each income level
                mode_counts = defaultdict(int)
                total_count = 0
                
                for agent in self.commuter_agents:
                    if agent.income_level == income_level:
                        mode = agent.current_mode if agent.current_mode else 'not_traveling'
                        mode_counts[mode] += 1
                        total_count += 1
                
                # Calculate percentages
                if total_count > 0:
                    for mode, count in mode_counts.items():
                        mode_shares[income_level][mode] = 100 * count / total_count
            
            # Save mode shares for this step
            self.mode_share_history[self.current_step] = dict(mode_shares)
            
            # Call the original step method
            return self.original_step(*args, **kwargs)
        
        # Replace the step method
        model_class.step = step_with_tracking
    
    return model_class


def run_simulation_with_tracking(steps=100, output_dir='visualization_outputs'):
    """
    Run a simulation with trajectory tracking and generate visualizations.
    
    Args:
        steps: Number of simulation steps
        output_dir: Directory to save visualizations
        
    Returns:
        Tuple of (model, visualizer)
    """
    if not MODEL_IMPORTS_AVAILABLE:
        print("Model modules not available. Cannot run simulation.")
        return None, None
    
    try:
        # Enable trajectory tracking
        MobilityModel_Tracked = add_trajectory_tracking(MobilityModel)
        
        # Initialize model with default parameters
        model = MobilityModel_Tracked(
            db_connection_string=DB_CONNECTION_STRING,
            num_commuters=num_commuters,
            grid_width=grid_width,
            grid_height=grid_height,
            data_income_weights=income_weights,
            data_health_weights=health_weights,
            data_payment_weights=payment_weights,
            data_age_distribution=age_distribution,
            data_disability_weights=disability_weights,
            data_tech_access_weights=tech_access_weights,
            ASC_VALUES=ASC_VALUES,
            UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS=UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS,
            UTILITY_FUNCTION_BASE_COEFFICIENTS=UTILITY_FUNCTION_BASE_COEFFICIENTS,
            PENALTY_COEFFICIENTS=PENALTY_COEFFICIENTS,
            AFFORDABILITY_THRESHOLDS=AFFORDABILITY_THRESHOLDS,
            FLEXIBILITY_ADJUSTMENTS=FLEXIBILITY_ADJUSTMENTS,
            VALUE_OF_TIME=VALUE_OF_TIME,
            public_price_table=public_price_table,
            ALPHA_VALUES=ALPHA_VALUES,
            DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS=DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS,
            BACKGROUND_TRAFFIC_AMOUNT=BACKGROUND_TRAFFIC_AMOUNT,
            CONGESTION_ALPHA=CONGESTION_ALPHA,
            CONGESTION_BETA=CONGESTION_BETA,
            CONGESTION_CAPACITY=CONGESTION_CAPACITY,
            CONGESTION_T_IJ_FREE_FLOW=CONGESTION_T_IJ_FREE_FLOW,
            uber_like1_capacity=UberLike1_cpacity,
            uber_like1_price=UberLike1_price,
            uber_like2_capacity=UberLike2_cpacity,
            uber_like2_price=UberLike2_price,
            bike_share1_capacity=BikeShare1_capacity,
            bike_share1_price=BikeShare1_price,
            bike_share2_capacity=BikeShare2_capacity,
            bike_share2_price=BikeShare2_price,
            subsidy_dataset=subsidy_dataset,
            subsidy_config=daily_config
        )
        
        print(f"Running simulation for {steps} steps...")
        model.run_model(steps)
        print("Simulation complete.")
        
        # Create visualizer
        visualizer = ABMVisualizer(model=model, output_dir=output_dir)
        
        return model, visualizer
    
    except Exception as e:
        print(f"Error running simulation: {e}")
        traceback.print_exc()
        return None, None


def main():
    """
    Main function to handle command-line arguments and run visualizations.
    """
    parser = argparse.ArgumentParser(description='ABM-ETOP Visualization Tool')
    
    # Database connection options
    parser.add_argument('--db_path', type=str, help='Path to the simulation database')
    parser.add_argument('--db_connection_string', type=str, help='Database connection string')
    
    # Simulation options
    parser.add_argument('--run_simulation', action='store_true', help='Run a new simulation')
    parser.add_argument('--steps', type=int, default=100, help='Number of simulation steps')
    
    # Visualization options
    parser.add_argument('--output_dir', type=str, default='visualization_outputs', 
                      help='Directory to save visualizations')
    parser.add_argument('--step', type=int, help='Specific time step to visualize')
    parser.add_argument('--by_income', action='store_true', default=True, 
                      help='Color by income level')
    parser.add_argument('--by_mode', action='store_true', 
                      help='Color by transportation mode')
    
    # Specific visualizations
    parser.add_argument('--spatial', action='store_true', help='Generate spatial distribution plot')
    parser.add_argument('--od', action='store_true', help='Generate origin-destination map')
    parser.add_argument('--heatmap', action='store_true', help='Generate network usage heatmap')
    parser.add_argument('--travel_time', action='store_true', help='Generate travel time distributions')
    parser.add_argument('--mode_share', action='store_true', help='Generate mode share evolution plot')
    parser.add_argument('--all', action='store_true', help='Generate all visualizations')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize visualizer
    visualizer = None
    model = None
    
    # Run simulation if requested
    if args.run_simulation:
        if MODEL_IMPORTS_AVAILABLE:
            print("Running a new simulation...")
            model, visualizer = run_simulation_with_tracking(
                steps=args.steps, output_dir=args.output_dir)
        else:
            print("Cannot run simulation: model modules not available.")
            sys.exit(1)
    
    # Connect to existing database if no simulation
    elif args.db_path:
        db_connection_string = f'sqlite:///{args.db_path}'
        visualizer = ABMVisualizer(db_connection_string=db_connection_string, 
                                  output_dir=args.output_dir)
    
    elif args.db_connection_string:
        visualizer = ABMVisualizer(db_connection_string=args.db_connection_string, 
                                  output_dir=args.output_dir)
    
    # Print error if no data source provided
    if visualizer is None and not args.run_simulation:
        print("Error: Please provide a database path (--db_path) or run a simulation (--run_simulation).")
        sys.exit(1)
    
    # Generate visualizations
    if args.all or not any([args.spatial, args.od, args.heatmap, args.travel_time, args.mode_share]):
        print("Generating all visualizations...")
        if visualizer:
            visualizer.generate_all_visualizations(step=args.step)
    else:
        if args.spatial and visualizer:
            print("Generating spatial distribution plot...")
            visualizer.plot_spatial_distribution(step=args.step, by_income=args.by_income, 
                                             by_mode=args.by_mode)
        
        if args.od and visualizer:
            print("Generating origin-destination map...")
            visualizer.plot_trip_origin_destination_map(by_income=args.by_income, 
                                                    by_mode=args.by_mode)
        
        if args.heatmap and visualizer:
            print("Generating network usage heatmap...")
            visualizer.plot_network_usage_heatmap()
        
        if args.travel_time and visualizer:
            print("Generating travel time distributions...")
            visualizer.plot_travel_time_distributions()
        
        if args.mode_share and visualizer:
            print("Generating mode share evolution plot...")
            visualizer.plot_mode_share_evolution()
    
    print(f"All visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()