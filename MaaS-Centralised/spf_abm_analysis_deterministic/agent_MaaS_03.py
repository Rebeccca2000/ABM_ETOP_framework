from mesa import Agent
import heapq
import math
from mesa.space import MultiGrid
from agent_service_provider_initialisation_03 import SubsidyUsageLog, ShareServiceBookingLog, ServiceBookingLog
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import create_engine, func
import uuid
import random
class MaaS(Agent):
    def __init__(self, unique_id, model, service_provider_agent, commuter_agents, DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS, \
                 BACKGROUND_TRAFFIC_AMOUNT, stations, routes, transfers, num_commuters, grid_width, grid_height, CONGESTION_ALPHA,\
                CONGESTION_BETA, CONGESTION_CAPACITY, CONGESTION_T_IJ_FREE_FLOW,subsidy_config, schema=None):
        super().__init__(unique_id, model)  # Initialize the base class
        self.service_provider_agent = service_provider_agent  # Store the reference to the service provider agent
        self.commuter_agents = commuter_agents  # Store the reference to the commuter agent
        self.stations = stations  # Dictionary of stations
        self.routes = routes  # Dictionary of routes
        self.transfers = transfers  # Dictionary of transfer times
        self.grid = MultiGrid(width=grid_width, height=grid_height, torus=False)  
        self.shortest_path_cache = {}  # Cache to store shortest paths for optimization
        self.db_engine = create_engine(self.model.db_connection_string)
        self.Session = service_provider_agent.Session  # Reuse the session from ServiceProvider
        self.dynamic_maas_surcharge_base_coefficient = DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS
        self.background_traffic_amount = BACKGROUND_TRAFFIC_AMOUNT
        self.num_commuters = num_commuters
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.congestion_alpha= CONGESTION_ALPHA
        self.congestion_beta= CONGESTION_BETA
        self.congestion_capacity = CONGESTION_CAPACITY
        self.conjestion_t_ij_free_flow = CONGESTION_T_IJ_FREE_FLOW
        self.schema = schema 
        # Existing initialization code...
        self.subsidy_config = subsidy_config
        self.current_subsidy_pool = subsidy_config.total_amount
        self.last_reset_step = 0
        self.total_subsidies_given = 0  # Track subsidies given in current period

    def find_nearest_station_any_mode(self, point):
        min_distance = float('inf')
        nearest_station = None
        nearest_mode = None
        for mode in self.stations:
            for station_id, station_point in self.stations[mode].items():
                distance = ((point[0] - station_point[0]) ** 2 + (point[1] - station_point[1]) ** 2) ** 0.5
                if distance < min_distance:
                    min_distance = distance
                    nearest_station = station_id
                    nearest_mode = mode
        return nearest_mode, nearest_station
    
    ########################################################################################################
    ##################################### Congestion and Shortest Route ########################################
    ########################################################################################################

    def check_is_peak(self, current_ticks):
        if (36 <= current_ticks % 144 < 60) or (90 <= current_ticks % 144 < 114):
            return True
        return False
    
    
    def insert_time_varying_traffic(self, session):
        """
        Insert background traffic with intensity that varies by time of day.
        Uses batch operations and optimized route generation for better performance.
        """
        current_step = self.model.get_current_step()
        if getattr(self.model, "deterministic_mode", False):
            traffic_events = self.model.deterministic_background_traffic_by_step.get(
                int(current_step), []
            )
            if not traffic_events:
                return 0

            is_peak = self.check_is_peak(current_step)
            base_start_time = current_step
            new_bookings = []
            for event in traffic_events:
                start_x, start_y = int(event["start"][0]), int(event["start"][1])
                end_x, end_y = int(event["end"][0]), int(event["end"][1])

                route = [(start_x, start_y)]
                x_dir = 1 if end_x > start_x else -1 if end_x < start_x else 0
                current_x = start_x
                while current_x != end_x:
                    current_x += x_dir
                    route.append((current_x, start_y))

                y_dir = 1 if end_y > start_y else -1 if end_y < start_y else 0
                current_y = start_y
                while current_y != end_y:
                    current_y += y_dir
                    route.append((end_x, current_y))

                route_length = abs(end_x - start_x) + abs(end_y - start_y)
                duration = route_length // 6
                if is_peak:
                    duration = int(duration * 1.3)
                duration = max(1, duration)

                start_offset = int(event.get("start_offset", 0))
                start_time = base_start_time + max(0, min(start_offset, 2))
                affected_steps = list(range(start_time, start_time + duration))

                new_booking = ShareServiceBookingLog(
                    commuter_id=-1,
                    request_id=str(uuid.uuid4()),
                    mode_id=0,
                    provider_id=0,
                    company_name="background_traffic",
                    start_time=start_time,
                    duration=duration,
                    affected_steps=affected_steps,
                    route_details=route,
                )
                new_bookings.append(new_booking)

            session.add_all(new_bookings)
            session.commit()
            return len(new_bookings)

        ticks_in_day = 144
        current_day_tick = current_step % ticks_in_day
        
        # Define traffic intensity based on time of day
        if 36 <= current_day_tick < 48:  # Early morning rush
            traffic_multiplier = 3.0
        elif 48 <= current_day_tick < 60:  # Late morning rush
            traffic_multiplier = 2.5
        elif 90 <= current_day_tick < 102:  # Early evening rush
            traffic_multiplier = 2.5
        elif 102 <= current_day_tick < 114:  # Late evening rush
            traffic_multiplier = 3.0
        elif (60 <= current_day_tick < 90) or (114 <= current_day_tick < 130):
            traffic_multiplier = 1.0
        else:  # Night/early morning
            traffic_multiplier = 0.3
        
        # Calculate traffic amount
        adjusted_traffic_amount = int(self.background_traffic_amount * traffic_multiplier)
        
        # For very small amounts, just return to avoid database operations
        if adjusted_traffic_amount <= 0:
            return 0
        
        # Prepare batch insert
        new_bookings = []
        is_peak = self.check_is_peak(current_step)
        
        # Pre-compute common values
        base_start_time = current_step
        
        # Generate routes in batch
        for _ in range(adjusted_traffic_amount):
            # Generate start and end points
            start_x, start_y = random.randint(0, self.grid_width - 1), random.randint(0, self.grid_height - 1)
            end_x, end_y = random.randint(0, self.grid_width - 1), random.randint(0, self.grid_height - 1)
            
            # Generate route more efficiently
            route = [(start_x, start_y)]
            
            # Move horizontally first
            x_dir = 1 if end_x > start_x else -1 if end_x < start_x else 0
            current_x = start_x
            while current_x != end_x:
                current_x += x_dir
                route.append((current_x, start_y))
            
            # Then move vertically
            y_dir = 1 if end_y > start_y else -1 if end_y < start_y else 0
            current_y = start_y
            while current_y != end_y:
                current_y += y_dir
                route.append((end_x, current_y))
            
            # Calculate duration more efficiently
            route_length = abs(end_x - start_x) + abs(end_y - start_y)
            duration = route_length // 6
            if is_peak:
                duration = int(duration * 1.3)
            duration = max(1, duration)  # Ensure at least 1 time step
            
            # Generate start time (small variation)
            start_time = base_start_time + random.randint(0, 2)
            
            # Create affected steps array efficiently
            affected_steps = list(range(start_time, start_time + duration))
            
            # Create booking object
            new_booking = ShareServiceBookingLog(
                commuter_id=-1,
                request_id=str(uuid.uuid4()),
                mode_id=0,
                provider_id=0,
                company_name="background_traffic",
                start_time=start_time,
                duration=duration,
                affected_steps=affected_steps,
                route_details=route
            )
            new_bookings.append(new_booking)
        
        # Use standard SQLAlchemy batch insert
        if new_bookings:
            # Add all objects to the session at once
            session.add_all(new_bookings)
            # Then commit in a single transaction
            session.commit()
        
        return adjusted_traffic_amount

    ########################################################################################################
    ##################################### Congestion and Shortest Route ########################################
    ########################################################################################################
    def get_current_traffic_volume(self, current_time_step):
        traffic_volume = {}
        cache_key = f"traffic_volume_{current_time_step}"
        
        # Check if we have this traffic volume cached
        if hasattr(self, '_traffic_cache') and cache_key in self._traffic_cache:
            return self._traffic_cache[cache_key]
        
        with self.Session() as session:
            # Optimize query by adding indexing hints and only fetching what we need
            active_bookings = session.query(
                ShareServiceBookingLog.route_details,
                ShareServiceBookingLog.affected_steps
            ).filter(
                ShareServiceBookingLog.start_time <= current_time_step,
                (ShareServiceBookingLog.start_time + ShareServiceBookingLog.duration) > current_time_step,
                ShareServiceBookingLog.mode_id.in_([0, 3])
            ).all()

            # Create a temporary set to store segments we've already processed for each booking
            # to avoid duplicating work
            processed_segments = set()
            
            for route_details, affected_steps in active_bookings:
                if current_time_step not in affected_steps:
                    continue
                    
                route = route_details
                
                # Process route format
                if isinstance(route[0], list) and isinstance(route[0][0], str):
                    route_segments = route[1]
                else:
                    route_segments = route
                
                # Process route segments more efficiently
                for i in range(len(route_segments) - 1):
                    start_point = tuple(route_segments[i])
                    end_point = tuple(route_segments[i + 1])
                    
                    # Create a unique identifier for this segment
                    segment_id = (start_point, end_point)
                    reverse_id = (end_point, start_point)
                    
                    # Skip if we've already processed this segment
                    if segment_id in processed_segments:
                        continue
                    
                    # Mark both directions as processed
                    processed_segments.add(segment_id)
                    processed_segments.add(reverse_id)
                    
                    # Increment volume for both directions
                    traffic_volume[segment_id] = traffic_volume.get(segment_id, 0) + 1
                    traffic_volume[reverse_id] = traffic_volume.get(reverse_id, 0) + 1
        
        # Cache this result for future use
        if not hasattr(self, '_traffic_cache'):
            self._traffic_cache = {}
        self._traffic_cache[cache_key] = traffic_volume
        
        # Limit cache size to prevent memory issues
        if len(self._traffic_cache) > 10:
            oldest_key = min(self._traffic_cache.keys())
            del self._traffic_cache[oldest_key]
        
        return traffic_volume

    def calculate_congested_travel_time(self, V_ij, C_ij, T_ij_free_flow):

        congestion_factor = (1 + self.congestion_alpha * (V_ij / max(C_ij, 1)) ** self.congestion_beta)
        congested_time = T_ij_free_flow * congestion_factor
        return congested_time

    def dijkstra_with_congestion(self, start, end, mode_id):
        # Check if we've already computed this path recently
        cache_key = f"{start}_{end}_{mode_id}_{self.model.get_current_step()//10}"
        if hasattr(self, '_path_cache') and cache_key in self._path_cache:
            return self._path_cache[cache_key]
        
        # If we're not calculating congestion (non-car modes), use a more efficient algorithm
        if mode_id != 3:
            return self._astar_path(start, end)
        
        # For cars, use Dijkstra with congestion
        moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        rows, cols = self.model.grid.width, self.model.grid.height
        
        # Use numpy arrays for better performance with large grids
        import numpy as np
        min_cost = np.full((rows, cols), np.inf)
        min_cost[start[0], start[1]] = 0
        
        # For predecessors, use a dictionary to save memory
        predecessor = {}
        
        # Priority queue for Dijkstra
        queue = [(0, start)]
        
        # Get traffic volume once
        current_time_step = self.model.get_current_step()
        traffic_volume = self.get_current_traffic_volume(current_time_step)
        
        # Cache congestion calculations
        congestion_cache = {}
        
        # Early termination optimization - track best direction
        dx_to_end = end[0] - start[0]
        dy_to_end = end[1] - start[1]
        
        # Main Dijkstra loop
        visited = set()
        while queue:
            current_cost, (x, y) = heapq.heappop(queue)
            
            # Skip if we've already processed this node
            if (x, y) in visited:
                continue
                
            visited.add((x, y))
            
            # Early termination check
            if (x, y) == end:
                break
                
            # Prioritize directions toward the destination
            prioritized_moves = sorted(moves, key=lambda move: 
                                    (dx_to_end * move[0] + dy_to_end * move[1]), reverse=True)
                
            for dx, dy in prioritized_moves:
                nx, ny = x + dx, y + dy
                
                if 0 <= nx < rows and 0 <= ny < cols:
                    segment = ((x, y), (nx, ny))
                    
                    # Calculate travel time with or without congestion
                    if mode_id == 3:  # Only cars use congestion
                        # Check if we've already calculated this segment's travel time
                        if segment in congestion_cache:
                            congested_time = congestion_cache[segment]
                        else:
                            V_ij = traffic_volume.get(segment, 0)
                            congested_time = self.calculate_congested_travel_time(
                                V_ij, self.congestion_capacity, self.conjestion_t_ij_free_flow
                            )
                            congestion_cache[segment] = congested_time
                            
                        next_cost = current_cost + congested_time
                    else:
                        next_cost = current_cost + self.conjestion_t_ij_free_flow
                    
                    if next_cost < min_cost[nx, ny]:
                        min_cost[nx, ny] = next_cost
                        predecessor[(nx, ny)] = (x, y)
                        heapq.heappush(queue, (next_cost, (nx, ny)))
        
        # Reconstruct path
        path = []
        step = end
        
        # Handle case where no path is found
        if step not in predecessor and step != start:
            # Return Manhattan path as fallback
            return self._manhattan_path(start, end)
            
        while step != start:
            path.append(step)
            step = predecessor[step]
        path.append(start)
        path.reverse()
        
        # Cache this result
        if not hasattr(self, '_path_cache'):
            self._path_cache = {}
        self._path_cache[cache_key] = path
        
        # Limit cache size
        if len(self._path_cache) > 100:
            # Remove oldest entries
            oldest = list(self._path_cache.keys())[:10]
            for key in oldest:
                del self._path_cache[key]
        
        return path

    def _astar_path(self, start, end):
        """A* algorithm for non-congested path finding (much faster than Dijkstra)"""
        import numpy as np
        
        def heuristic(a, b):
            # Manhattan distance
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        rows, cols = self.model.grid.width, self.model.grid.height
        moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        # Priority queue
        open_set = [(heuristic(start, end), 0, start)]
        closed_set = set()
        
        # Track g_scores (cost from start) and predecessors
        g_score = {start: 0}
        predecessors = {}
        
        while open_set:
            _, current_g, current = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
                
            if current == end:
                break
                
            closed_set.add(current)
            
            for dx, dy in moves:
                nx, ny = current[0] + dx, current[1] + dy
                neighbor = (nx, ny)
                
                if 0 <= nx < rows and 0 <= ny < cols:
                    # Cost from start to neighbor
                    tentative_g = current_g + self.conjestion_t_ij_free_flow
                    
                    if neighbor in closed_set and tentative_g >= g_score.get(neighbor, float('inf')):
                        continue
                        
                    if tentative_g < g_score.get(neighbor, float('inf')):
                        # This path is better
                        predecessors[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score = tentative_g + heuristic(neighbor, end)
                        heapq.heappush(open_set, (f_score, tentative_g, neighbor))
        
        # Reconstruct path
        path = []
        current = end
        while current != start:
            path.append(current)
            current = predecessors.get(current, start)  # Fallback to start if no path
        path.append(start)
        path.reverse()
        
        return path

    def _manhattan_path(self, start, end):
        """Generate a simple Manhattan path between two points (fallback)"""
        path = [start]
        current_x, current_y = start
        
        # Move horizontally first
        while current_x != end[0]:
            current_x += 1 if end[0] > current_x else -1
            path.append((current_x, current_y))
            
        # Then move vertically
        while current_y != end[1]:
            current_y += 1 if end[1] > current_y else -1
            path.append((current_x, current_y))
            
        return path

    
    ########################################################################################################
    ##################################### For Single Mode ########################################
    ########################################################################################################
    def calculate_single_mode_time_and_price(self, origin, destination, unit_price, unit_speed, mode_id):
        """
        Calculate the shortest route, time, and price for a single mode of transportation.
        This function uses the dijkstra_with_congestion method, which now considers traffic congestion for cars (mode_id == 3).
        
        Parameters:
        - origin: Tuple representing the starting point on the grid.
        - destination: Tuple representing the destination point on the grid.
        - unit_price: Price per unit distance.
        - unit_speed: Speed of the mode of transportation.
        - mode_id: The mode of transport (e.g., car = 3).
        
        Returns:
        - shortest_route: The calculated shortest route.
        - total_time: The time to travel the route.
        - total_price: The price to travel the route.
        """
        # Find the shortest route using the updated dijkstra_with_congestion
        shortest_route = self.dijkstra_with_congestion(origin, destination, mode_id)
        
        # Calculate the total length of the shortest route
        total_length = self.calculate_total_route_length(shortest_route)
        
        # Calculate the total price (without surcharge)
        total_price = unit_price * total_length
        
        # Calculate the total time (adjusting for congestion if mode_id == 3)
        total_time = total_length / unit_speed
        
        return shortest_route, total_time, total_price

    
    # Add MaaS Surcharge
    def get_current_usage(self):
        """
        Calculate the number of active commuters based on their request status.

        Returns:
            int: The number of active commuters.
        """
        active_commuters = 0
        for commuter in self.commuter_agents:
            if any(request['status'] == 'active' for request in commuter.requests.values()):
                active_commuters += 1
        return active_commuters

    def calculate_dynamic_MaaS_surcharge(self, payment_scheme):
        # Parameters from the coefficient dictionary
        coefficients = self.dynamic_maas_surcharge_base_coefficient
        S_base = coefficients['S_base']  # Base surcharge (10%)
        alpha = coefficients['alpha']    # Sensitivity coefficient
        delta = coefficients['delta']    # Reduction factor for subscription model

        # Collect current usage data
        current_usage = self.get_current_usage()
        system_capacity = self.num_commuters  # Imported from MobilityModel

        # Calculate usage ratio
        UR = current_usage / system_capacity

        # Modified formula to encourage adoption when usage is low
        S_dynamic = S_base * (1 - alpha * (1 - UR))  # Decreases surcharge when usage is low
        S_dynamic = max(0, min(S_dynamic, S_base * 1.5))  # Cap the range
        

        if payment_scheme == 'PAYG':
            surcharge_percentage = S_dynamic
        elif payment_scheme == 'subscription':
            surcharge_percentage = S_base * delta  # Fixed reduced surcharge for subscription model
        else:
            surcharge_percentage = 0  # Default to 0 if the payment scheme is not recognized

        return surcharge_percentage



    ########################################################################################################
    ##################################### For Public Transport Mode ########################################
    ########################################################################################################
    def find_optimal_route(self, origin_point, destination_point):
        """Find optimal public transport route with improved efficiency."""
        # Check cache first
        cache_key = f"route_{origin_point}_{destination_point}"
        if hasattr(self, '_route_cache') and cache_key in self._route_cache:
            return self._route_cache[cache_key]
        
        origin_mode, origin_station = self.find_nearest_station_any_mode(origin_point)
        destination_mode, destination_station = self.find_nearest_station_any_mode(destination_point)
        
        # Use defaultdict to avoid initializing all stations
        import collections
        best_times = collections.defaultdict(lambda: float('inf'))
        best_paths = {}
        
        best_times[origin_station] = 0
        
        # Use a more efficient data structure for the queue
        import heapq
        queue = [(0, origin_station)]  # Priority queue based on time
        visited = set()  # Track visited stations to avoid duplicates
        
        while queue:
            current_time, current_station = heapq.heappop(queue)
            
            # Skip if we already found a better path to this station
            if current_station in visited and current_time > best_times[current_station]:
                continue
                
            visited.add(current_station)
            
            # Early termination if we reached the destination
            if current_station == destination_station:
                break
            
            # Process routes more efficiently
            for mode in self.routes:
                route_ids = self.get_routes_through_station(mode, current_station)
                for route_id in route_ids:
                    route_stations = self.routes[mode][route_id]
                    try:
                        station_idx = route_stations.index(current_station)
                        
                        # Process stations in both directions of this route
                        for next_idx in [station_idx-1, station_idx+1]:
                            if 0 <= next_idx < len(route_stations):
                                next_station = route_stations[next_idx]
                                travel_time = 0.8  # 0.8 tick per stop
                                arrival_time = current_time + travel_time
                                
                                if arrival_time < best_times[next_station]:
                                    best_times[next_station] = arrival_time
                                    best_paths[next_station] = current_station
                                    heapq.heappush(queue, (arrival_time, next_station))
                    except ValueError:
                        # Station not found in route, skip
                        continue
            
            # Process transfers more efficiently
            transfers = self.get_transfers_from_station(current_station)
            for next_station, transfer_time in transfers:
                arrival_time = current_time + transfer_time
                if arrival_time < best_times[next_station]:
                    best_times[next_station] = arrival_time
                    best_paths[next_station] = current_station
                    heapq.heappush(queue, (arrival_time, next_station))
        
        # Reconstruct path more efficiently
        path = []
        current = destination_station
        
        while current != origin_station:
            path.append(current)
            current = best_paths.get(current)
            if current is None:
                # Cache negative result to avoid recalculation
                if not hasattr(self, '_route_cache'):
                    self._route_cache = {}
                self._route_cache[cache_key] = None
                return None
                
        path.append(origin_station)
        path.reverse()
        
        # Cache result
        if not hasattr(self, '_route_cache'):
            self._route_cache = {}
        self._route_cache[cache_key] = path
        
        # Limit cache size
        if len(self._route_cache) > 100:
            # Remove a few old entries
            for _ in range(5):
                if self._route_cache:
                    self._route_cache.popitem()
        if not path:
            print(f"No route found between {origin_point} and {destination_point}")
        else:
            print(f"Found route with {len(path)} segments")
        return path

    def build_detailed_itinerary(self, optimal_path, origin_point, destination_point):
        """Build detailed itinerary with improved efficiency."""
        # Check for edge cases early to avoid unnecessary work
        if not optimal_path or len(optimal_path) <= 1:
            return []
            
        # Cache check
        cache_key = f"itinerary_{origin_point}_{destination_point}_{optimal_path[0]}_{optimal_path[-1]}"
        if hasattr(self, '_itinerary_cache') and cache_key in self._itinerary_cache:
            return self._itinerary_cache[cache_key]
        
        detailed_itinerary = []
        
        # Get origin station info
        origin_station_mode, origin_station = self.find_nearest_station_any_mode(origin_point)
        detailed_itinerary.append(('to station', [origin_point, origin_station]))
        
        # Pre-process station information
        station_modes = {}
        for station in optimal_path:
            station_modes[station] = 'bus' if station.startswith('B') else 'train'
        
        # Pre-compute route lookup for stations
        station_routes = {}
        for mode in self.routes:
            for route_id, stations in self.routes[mode].items():
                for station in stations:
                    if station not in station_routes:
                        station_routes[station] = []
                    station_routes[station].append((mode, route_id))
        
        i = 0
        while i < len(optimal_path) - 1:
            current_station = optimal_path[i]
            next_station = optimal_path[i + 1]
            
            current_mode = station_modes[current_station]
            next_mode = station_modes[next_station]
            
            if current_mode == next_mode:
                # Find a route connecting these stations
                connected_route = None
                for mode, route_id in station_routes.get(current_station, []):
                    if mode == current_mode:
                        stations = self.routes[mode][route_id]
                        if current_station in stations and next_station in stations:
                            connected_route = (mode, route_id)
                            break
                
                if connected_route:
                    mode, route_id = connected_route
                    # Check if we need to start a new segment
                    if not detailed_itinerary or detailed_itinerary[-1][0] == 'transfer' or detailed_itinerary[-1][0] != mode:
                        detailed_itinerary.append((mode, route_id, [current_station]))
                    
                    # Add next station to the current segment
                    detailed_itinerary[-1][-1].append(next_station)
                    i += 1  # Move to next station
                else:
                    # No route found, add transfer
                    detailed_itinerary.append(('transfer', [current_station, next_station]))
                    i += 1
            else:
                # Different modes, add transfer
                detailed_itinerary.append(('transfer', [current_station, next_station]))
                i += 1
        
        # Add final leg to destination
        detailed_itinerary.append(('to destination', [optimal_path[-1], destination_point]))
        
        # Cache the result
        if not hasattr(self, '_itinerary_cache'):
            self._itinerary_cache = {}
        self._itinerary_cache[cache_key] = detailed_itinerary
        
        return detailed_itinerary

    def calculate_total_time_and_price_public(self, payment_scheme, detailed_itinerary, walking_speed, 
                                            bus_stop_speed, train_stop_speed, bus_stop_price, train_stop_price):
        """Calculate total time and price with improved efficiency."""
        # Check cache first
        itinerary_hash = hash(str(detailed_itinerary) + payment_scheme)
        cache_key = f"price_time_{itinerary_hash}_{walking_speed}_{bus_stop_speed}_{train_stop_speed}"
        if hasattr(self, '_price_time_cache') and cache_key in self._price_time_cache:
            return self._price_time_cache[cache_key]
        
        total_time = 0
        total_price = 0
        
        # Pre-compute station coordinates to reduce lookups
        station_coordinates = {}
        for mode in self.stations:
            for station, coords in self.stations[mode].items():
                station_coordinates[station] = coords
        
        for segment in detailed_itinerary:
            segment_type = segment[0]
            
            if segment_type == 'to station':
                get_on_station_name = segment[1][1]
                if get_on_station_name in station_coordinates:
                    get_on_station_coordinates = station_coordinates[get_on_station_name]
                else:
                    # Fallback if not found in pre-computed cache
                    if get_on_station_name.startswith('T'):
                        get_on_station_coordinates = self.stations['train'][get_on_station_name]
                    elif get_on_station_name.startswith('B'):
                        get_on_station_coordinates = self.stations['bus'][get_on_station_name]
                    else:
                        raise ValueError(f"Station {get_on_station_name} is not recognized")
                
                # Use simple Manhattan distance if origin is close to station to avoid expensive dijkstra
                origin = segment[1][0]
                manhattan_dist = abs(origin[0] - get_on_station_coordinates[0]) + abs(origin[1] - get_on_station_coordinates[1])
                if manhattan_dist < 10:  # Short distance threshold
                    walk_distance = manhattan_dist
                else:
                    walking_route = self.dijkstra_with_congestion(origin, get_on_station_coordinates, 5)
                    walk_distance = self.calculate_total_route_length(walking_route)
                    
                walk_time = walk_distance / walking_speed
                total_time += walk_time
                
            elif segment_type in ['bus', 'train']:
                mode = segment_type
                route_id = segment[1]
                stops = segment[2]
                
                if len(stops) < 2:
                    continue  # Skip invalid segments
                    
                route_list = self.routes[mode][route_id]
                
                # Get indices efficiently with error handling
                try:
                    get_on_index = route_list.index(stops[0])
                    get_off_index = route_list.index(stops[-1])  # Use last stop in case of multiple
                    number_of_stops = abs(get_off_index - get_on_index)
                except ValueError:
                    # Fallback if stops not found
                    number_of_stops = len(stops) - 1
                
                # Calculate segment specifics
                if mode == 'bus':
                    each_stop_speed = bus_stop_speed
                    price_per_stop = bus_stop_price
                else:  # train
                    each_stop_speed = train_stop_speed
                    price_per_stop = train_stop_price
                    
                travel_time = number_of_stops * each_stop_speed
                waiting_time = 0  # Reduced waiting time
                segment_time = travel_time + waiting_time
                
                total_time += segment_time
                segment_price = number_of_stops * float(price_per_stop)
                total_price += segment_price
                
            elif segment_type == 'transfer':
                transfer_stations = tuple(segment[1])
                if transfer_stations in self.transfers:
                    transfer_time = self.transfers[transfer_stations]
                    total_time += transfer_time
                else:
                    # Fallback if transfer not found - use a default value
                    total_time += 1.0  # Default transfer time
        
        # Apply surcharge based on payment scheme
        if payment_scheme == 'PAYG':
            surcharge_percentage = self.calculate_dynamic_MaaS_surcharge(payment_scheme)
            total_price_with_surcharge = total_price * (1 + surcharge_percentage)
            MaaS_surcharge = total_price_with_surcharge - total_price
        elif payment_scheme == 'subscription':
            surcharge_percentage = self.calculate_dynamic_MaaS_surcharge(payment_scheme)
            total_price_with_surcharge = total_price * (1 + surcharge_percentage)
            MaaS_surcharge = total_price_with_surcharge - total_price
        else:
            total_price_with_surcharge = total_price
            MaaS_surcharge = 0
        
        # Cache the result
        result = (total_time, total_price_with_surcharge, MaaS_surcharge)
        if not hasattr(self, '_price_time_cache'):
            self._price_time_cache = {}
        self._price_time_cache[cache_key] = result
        
        return result

    def get_routes_through_station(self, mode, station):
        """Get routes through a station with caching."""
        # Cache this function's results
        cache_key = f"routes_{mode}_{station}"
        if hasattr(self, '_station_routes_cache') and cache_key in self._station_routes_cache:
            return self._station_routes_cache[cache_key]
        
        # Calculate routes passing through this station
        routes = [route_id for route_id, stations in self.routes[mode].items() 
                if station in stations]
        
        # Cache result
        if not hasattr(self, '_station_routes_cache'):
            self._station_routes_cache = {}
        self._station_routes_cache[cache_key] = routes
        
        return routes

    def get_transfers_from_station(self, station):
        """Get transfers from a station with caching."""
        # Cache this function's results
        cache_key = f"transfers_{station}"
        if hasattr(self, '_transfers_cache') and cache_key in self._transfers_cache:
            return self._transfers_cache[cache_key]
        
        # Find transfers from this station
        transfers = [(dest, time) for (src, dest), time in self.transfers.items() 
                    if src == station]
        
        # Cache result
        if not hasattr(self, '_transfers_cache'):
            self._transfers_cache = {}
        self._transfers_cache[cache_key] = transfers
        
        return transfers

    def calculate_distance(self, point1, point2):
        """Calculate distance with optional caching."""
        # For very frequent calculations, consider caching
        cache_key = f"dist_{point1}_{point2}"
        if hasattr(self, '_distance_cache') and cache_key in self._distance_cache:
            return self._distance_cache[cache_key]
        
        # Direct calculation using squared difference
        x1, y1 = point1
        x2, y2 = point2
        distance = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        
        # Only cache if points have integer coordinates (to limit cache size)
        if all(isinstance(c, int) for c in point1 + point2):
            if not hasattr(self, '_distance_cache'):
                self._distance_cache = {}
            if len(self._distance_cache) < 10000:  # Limit cache size
                self._distance_cache[cache_key] = distance
        
        return distance

    def calculate_total_route_length(self, route):
        """Calculate total route length more efficiently."""
        # Special case for short routes
        if len(route) <= 1:
            return 0.0
        
        # Consider caching for frequently used routes
        route_tuple = tuple(map(tuple, route))
        cache_key = f"route_length_{hash(route_tuple)}"
        if hasattr(self, '_route_length_cache') and cache_key in self._route_length_cache:
            return self._route_length_cache[cache_key]
        
        # Calculate manhattan distance for routes with integer coordinates
        if all(isinstance(p[0], int) and isinstance(p[1], int) for p in route):
            manhattan_length = sum(abs(route[i][0] - route[i-1][0]) + abs(route[i][1] - route[i-1][1]) 
                                for i in range(1, len(route)))
            euclidean_length = sum(self.calculate_distance(route[i-1], route[i]) 
                                for i in range(1, len(route)))
            
            # Cache result for frequently used routes
            if len(route) > 10:  # Only cache longer routes
                if not hasattr(self, '_route_length_cache'):
                    self._route_length_cache = {}
                if len(self._route_length_cache) < 500:  # Limit cache size
                    self._route_length_cache[cache_key] = euclidean_length
            
            return euclidean_length
        else:
            # Fall back to original implementation for non-integer coordinates
            return sum(self.calculate_distance(route[i-1], route[i]) for i in range(1, len(route)))

    def options_without_maas(self, request_id, start_time, origin, destination):
        """Generate travel options without MaaS more efficiently."""
        # Check cache first - important for repeated calculations
        cache_key = f"options_{request_id}_{start_time}_{origin}_{destination}"
        if hasattr(self, '_options_cache') and cache_key in self._options_cache:
            options = self._options_cache[cache_key]
            
            # Update the commuter's request with the cached options
            for commuter in self.commuter_agents:
                if request_id in commuter.requests:
                    commuter.requests[request_id]['travel_options_without_maas'] = options
            
            return options
        
        # Initialize travel options
        travel_options = {'request_id': request_id}
        
        # Calculate single mode options first
        for mode in ['walk', 'bike', 'car']:
            if mode == 'walk':
                # Walking is simple and always available
                unit_price = 0
                unit_speed = self.service_provider_agent.get_travel_speed(mode, start_time)
                shortest_route, total_time, total_price = self.calculate_single_mode_time_and_price(
                    origin, destination, unit_price, unit_speed, 5)
                
                travel_options[f'{mode}_route'] = shortest_route
                travel_options[f'{mode}_time'] = total_time
                travel_options[f'{mode}_price'] = total_price
                travel_options[f'{mode}_availability'] = float('inf')
                
            elif mode in ['bike', 'car']:
                # Get price dictionary for shared services
                price_dict = self.service_provider_agent.get_shared_service_price(mode, start_time)
                if not price_dict:
                    continue
                    
                # Get speed once outside the loop
                unit_speed = self.service_provider_agent.get_travel_speed(mode, start_time)
                
                # Get route once for this mode 
                mode_id = 4 if mode == 'bike' else 3
                route, time_base, _ = self.calculate_single_mode_time_and_price(
                    origin, destination, 1.0, unit_speed, mode_id)
                    
                # Calculate for each service provider
                for company_name, unit_price in price_dict.items():
                    # Calculate price based on the same route
                    total_price = time_base * unit_price
                    total_time = time_base  # Time is already calculated
                    
                    # Check availability
                    availability = self.service_provider_agent.check_shared_availability(company_name, start_time)
                    
                    # Store options
                    travel_options[f'{mode}_{company_name}_route'] = route
                    travel_options[f'{mode}_{company_name}_time'] = total_time
                    travel_options[f'{mode}_{company_name}_price'] = total_price
                    travel_options[f'{mode}_{company_name}_availability'] = availability
        
        # Calculate public transport options second
        try:
            # Find optimal route
            optimal_path = self.find_optimal_route(origin, destination)
            
            if optimal_path:
                # Build detailed itinerary
                detailed_itinerary = self.build_detailed_itinerary(optimal_path, origin, destination)
                
                # Get required parameters for calculation
                walking_speed = self.service_provider_agent.get_travel_speed('walk', start_time)
                bus_stop_speed = self.service_provider_agent.get_travel_speed('bus', start_time)
                train_stop_speed = self.service_provider_agent.get_travel_speed('train', start_time)
                bus_stop_price = self.service_provider_agent.get_public_service_price('bus', start_time)
                train_stop_price = self.service_provider_agent.get_public_service_price('train', start_time)
                
                # Calculate time and price
                public_time, public_price = self.calculate_public_transport_time_and_price(
                    detailed_itinerary, bus_stop_speed, train_stop_speed, bus_stop_price, train_stop_price)
                
                # Store public transport options
                travel_options['public_route'] = detailed_itinerary
                travel_options['public_time'] = public_time
                travel_options['public_price'] = public_price
                travel_options['public_availability'] = float('inf')
        except Exception as e:
            # Handle errors gracefully
            print(f"Error calculating public transport options: {e}")
        
        # Store the options in commuter's request
        for commuter in self.commuter_agents:
            if request_id in commuter.requests:
                commuter.requests[request_id]['travel_options_without_maas'] = travel_options
        
        # Cache the result
        if not hasattr(self, '_options_cache'):
            self._options_cache = {}
        self._options_cache[cache_key] = travel_options
        
        # Limit cache size
        if len(self._options_cache) > 50:
            self._options_cache.pop(next(iter(self._options_cache)))
        
        return travel_options
    
    ########################################################################################################
    ##################################### For MaaS Mode Options ########################################
    ########################################################################################################
    def calculate_time_and_price_to_station_or_destination(self, segment, start_time):
        """Calculate time and price to/from station with optimized performance."""
        # Check cache first
        cache_key = f"segment_{start_time}_{segment[0]}_{segment[1][0]}_{segment[1][-1]}"
        if hasattr(self, '_segment_cache') and cache_key in self._segment_cache:
            return self._segment_cache[cache_key]
        
        options = []
        
        # Extract origin and destination with reduced redundancy
        segment_type = segment[0]
        
        # Pre-compute station coordinates
        if not hasattr(self, '_station_coords_cache'):
            self._station_coords_cache = {}
            for station_type in ['train', 'bus']:
                for station_name, coords in self.stations[station_type].items():
                    self._station_coords_cache[station_name] = coords
        
        # Process segment more efficiently
        try:
            if segment_type == 'to station':
                origin = segment[1][0]
                destination_station = segment[1][1]
                destination = self._station_coords_cache.get(destination_station)
                
                if not destination:
                    # Fallback if not in cache
                    if destination_station.startswith('T'):
                        destination = self.stations['train'][destination_station]
                    elif destination_station.startswith('B'):
                        destination = self.stations['bus'][destination_station]
                    else:
                        raise ValueError(f"Station {destination_station} not recognized")
            
            elif segment_type == 'to destination':
                origin_station = segment[1][0]
                destination = segment[1][1]
                origin = self._station_coords_cache.get(origin_station)
                
                if not origin:
                    # Fallback if not in cache
                    if origin_station.startswith('T'):
                        origin = self.stations['train'][origin_station]
                    elif origin_station.startswith('B'):
                        origin = self.stations['bus'][origin_station]
                    else:
                        raise ValueError(f"Station {origin_station} not recognized")
            else:
                raise ValueError(f"Segment type {segment_type} not recognized")
                
            # Pre-compute distance once
            manhattan_distance = abs(origin[0] - destination[0]) + abs(origin[1] - destination[1])
            
            # If very close (e.g., < 5 units), consider adding a direct walk option regardless of mode
            if manhattan_distance < 5:
                # Very short distance optimization - skip other calculations
                walk_speed = self.service_provider_agent.get_travel_speed('walk', start_time)
                walk_time = manhattan_distance / walk_speed
                walk_route = [origin, destination]  # Simple direct route
                options.append((segment_type, segment[1], 'walk', walk_time, 0, walk_route))
                
                # For very short distances, we might want to return early
                if manhattan_distance < 3:
                    # Cache and return just the walk option for extremely short distances
                    if not hasattr(self, '_segment_cache'):
                        self._segment_cache = {}
                    self._segment_cache[cache_key] = options
                    return options
            
            # Process single modes more efficiently
            single_modes = ['walk', 'bike', 'car']
            mode_ids = {'walk': 5, 'bike': 4, 'car': 3}  # Pre-define mode IDs
            
            for mode in single_modes:
                if mode == 'walk':
                    # Walking is always available and free
                    unit_speed = self.service_provider_agent.get_travel_speed(mode, start_time)
                    route, total_time, total_price = self.calculate_single_mode_time_and_price(
                        origin, destination, 0, unit_speed, mode_ids[mode])
                    options.append((segment_type, segment[1], mode, total_time, total_price, route))
                else:
                    # Get pricing info for shared services
                    price_dict = self.service_provider_agent.get_shared_service_price(mode, start_time)
                    if not price_dict:
                        continue
                    
                    # Get speed only once outside the company loop
                    unit_speed = self.service_provider_agent.get_travel_speed(mode, start_time)
                    
                    # Calculate route once for all companies of this mode
                    route, time_base, _ = self.calculate_single_mode_time_and_price(
                        origin, destination, 1.0, unit_speed, mode_ids[mode])
                    
                    # Process each company with the pre-computed route
                    for company_name, unit_price in price_dict.items():
                        # Check availability first before doing price calculations
                        availability = self.service_provider_agent.check_shared_availability(company_name, start_time)
                        if availability <= 0:
                            continue
                        
                        # Calculate final price based on route length and unit price
                        total_price = time_base * unit_price
                        
                        # Add to options
                        options.append((segment_type, segment[1], company_name, time_base, total_price, route))
        
        except Exception as e:
            print(f"Error calculating options for {segment_type}: {e}")
            # Return at least walk as fallback
            if 'origin' in locals() and 'destination' in locals():
                unit_speed = self.service_provider_agent.get_travel_speed('walk', start_time)
                walk_route = [origin, destination]
                walk_time = manhattan_distance / unit_speed if 'manhattan_distance' in locals() else 10
                options.append((segment_type, segment[1], 'walk', walk_time, 0, walk_route))
        
        # Cache the result
        if not hasattr(self, '_segment_cache'):
            self._segment_cache = {}
        self._segment_cache[cache_key] = options
        
        # Limit cache size
        if len(self._segment_cache) > 200:
            # Remove 20 oldest items
            keys_to_remove = list(self._segment_cache.keys())[:20]
            for k in keys_to_remove:
                del self._segment_cache[k]
                
        return options
    
    def calculate_public_transport_time_and_price(self, detailed_itinerary, bus_stop_speed, 
                                                train_stop_speed, bus_stop_price, train_stop_price):
        """Calculate public transport time and price more efficiently."""
        # Create a cache key based on the itinerary and speeds/prices
        if not detailed_itinerary:
            return 0, 0
            
        cache_key = f"tp_{hash(str(detailed_itinerary))}_{bus_stop_speed}_{train_stop_speed}_{bus_stop_price}_{train_stop_price}"
        if hasattr(self, '_transport_cache') and cache_key in self._transport_cache:
            return self._transport_cache[cache_key]
        
        total_time = 0
        total_price = 0
        
        # Pre-compute needed values to avoid repeated calculations
        mode_params = {
            'bus': {'speed': bus_stop_speed, 'price': bus_stop_price},
            'train': {'speed': train_stop_speed, 'price': train_stop_price}
        }
        
        # Pre-fetch route lists to avoid repeated lookups
        route_lists = {}
        for mode in ['bus', 'train']:
            route_lists[mode] = {}
            for route_id in self.routes[mode]:
                route_lists[mode][route_id] = self.routes[mode][route_id]
        
        # Process all segments in one pass
        for segment in detailed_itinerary:
            segment_type = segment[0]
            
            if segment_type in ['bus', 'train']:
                mode = segment_type
                route_id = segment[1]
                stops = segment[2]
                
                if len(stops) < 2:
                    continue
                
                # Get route list efficiently
                route_list = route_lists[mode].get(route_id)
                if not route_list:
                    continue
                    
                # Calculate stop indices
                try:
                    get_on_index = route_list.index(stops[0])
                    get_off_index = route_list.index(stops[-1])
                    number_of_stops = abs(get_off_index - get_on_index) + 1
                except ValueError:
                    # Fallback - count stops directly
                    number_of_stops = len(stops)
                
                # Get mode-specific parameters
                params = mode_params[mode]
                
                # Calculate times
                travel_time = number_of_stops * params['speed']
                waiting_time = 0.3  # Fixed waiting time
                segment_time = travel_time + waiting_time
                
                # Calculate price
                segment_price = number_of_stops * float(params['price'])
                
                # Update totals
                total_time += segment_time
                total_price += segment_price
                
            elif segment_type == 'transfer':
                transfer_stations = tuple(segment[1])
                # Use direct dictionary lookup with fallback
                transfer_time = self.transfers.get(transfer_stations, 1.0)  # Default 1.0 if not found
                total_time += transfer_time
        
        # Cache the result
        result = (total_time, total_price)
        if not hasattr(self, '_transport_cache'):
            self._transport_cache = {}
        self._transport_cache[cache_key] = result
        
        # Limit cache size
        if len(self._transport_cache) > 100:
            self._transport_cache.popitem()  # Remove oldest item
        
        return result

    def apply_maas_surcharge(self, total_price, payment_scheme):
        """Apply MaaS surcharge with caching for repeated calculations."""
        # Use cache for repeated calls
        cache_key = f"surcharge_{payment_scheme}_{total_price}"
        if hasattr(self, '_surcharge_cache') and cache_key in self._surcharge_cache:
            return self._surcharge_cache[cache_key]
        
        # Calculate surcharge based on payment scheme
        if payment_scheme == 'PAYG':
            surcharge_percentage = self.calculate_dynamic_MaaS_surcharge(payment_scheme)
            total_with_surcharge = total_price * (1 + surcharge_percentage)
            maas_surcharge = total_with_surcharge - total_price
        elif payment_scheme == 'subscription':
            surcharge_percentage = self.calculate_dynamic_MaaS_surcharge(payment_scheme)
            total_with_surcharge = total_price * (1 + surcharge_percentage)
            maas_surcharge = total_with_surcharge - total_price
        else:
            total_with_surcharge = total_price
            maas_surcharge = 0
        
        # Cache result
        result = (total_with_surcharge, maas_surcharge)
        if not hasattr(self, '_surcharge_cache'):
            self._surcharge_cache = {}
        self._surcharge_cache[cache_key] = result
        
        # Keep cache relatively small
        if len(self._surcharge_cache) > 50:
            self._surcharge_cache.popitem()
        
        return result
    def maas_options(self, payment_scheme, request_id, start_time, origin, destination):
        """Generate MaaS options with improved efficiency."""
        # Check cache for exact match first
        cache_key = f"maas_{payment_scheme}_{request_id}_{start_time}_{origin}_{destination}"
        if hasattr(self, '_maas_options_cache') and cache_key in self._maas_options_cache:
            cached_options = self._maas_options_cache[cache_key]
            
            # Update commuter requests with cached options
            for commuter in self.commuter_agents:
                if request_id in commuter.requests:
                    commuter.requests[request_id]['travel_options_with_maas'] = cached_options
            
            return cached_options
        
        # Step 1: Find optimal route - check for missing route early
        optimal_path = self.find_optimal_route(origin, destination)
        if not optimal_path:
            # No public transport route available
            empty_options = []
            for commuter in self.commuter_agents:
                if request_id in commuter.requests:
                    commuter.requests[request_id]['travel_options_with_maas'] = empty_options
            return empty_options
        
        # Step 2: Build itinerary
        detailed_itinerary = self.build_detailed_itinerary(optimal_path, origin, destination)
        if not detailed_itinerary or len(detailed_itinerary) < 2:  # Need at least to_station and to_destination
            empty_options = []
            for commuter in self.commuter_agents:
                if request_id in commuter.requests:
                    commuter.requests[request_id]['travel_options_with_maas'] = empty_options
            return empty_options
        
        # Step 3-5: Get options and combine them
        try:
            # Calculate segments and public transport in parallel
            to_station_options = self.calculate_time_and_price_to_station_or_destination(
                detailed_itinerary[0], start_time)
            
            # Prefetch common speeds and prices
            bus_speed = self.service_provider_agent.get_travel_speed('bus', start_time)
            train_speed = self.service_provider_agent.get_travel_speed('train', start_time)
            bus_price = self.service_provider_agent.get_public_service_price('bus', start_time)
            train_price = self.service_provider_agent.get_public_service_price('train', start_time)
            
            # Calculate public transport segment
            public_time, public_price = self.calculate_public_transport_time_and_price(
                detailed_itinerary, bus_speed, train_speed, bus_price, train_price)
            
            # Get destination options
            to_destination_options = self.calculate_time_and_price_to_station_or_destination(
                detailed_itinerary[-1], start_time)
            
            # Create MaaS options more efficiently
            maas_options = []
            
            # Generate combinations with early filtering
            num_combinations = len(to_station_options) * len(to_destination_options)
            limit_combinations = 20  # Set a reasonable limit
            
            # Change this part to prioritize time efficiency instead of cost
            if num_combinations > limit_combinations:
                # Sort by travel time (faster first), not by price
                to_station_options = sorted(to_station_options, key=lambda x: x[3])[:5]  # x[3] is the time component
                to_destination_options = sorted(to_destination_options, key=lambda x: x[3])[:5]  # x[3] is the time component
                        
            # Combine options
            for to_station_option in to_station_options:
                for to_destination_option in to_destination_options:
                    # Calculate totals
                    total_time = (to_station_option[3] + public_time + to_destination_option[3])
                    total_price = (to_station_option[4] + public_price + to_destination_option[4])
                    
                    # Apply surcharge
                    final_price, surcharge = self.apply_maas_surcharge(total_price, payment_scheme)
                    
                    # Create MaaS option
                    maas_option = [
                        [to_station_option[2], to_station_option[3], to_station_option[4], to_station_option[5]],
                        [to_destination_option[2], to_destination_option[3], to_destination_option[4], to_destination_option[5]],
                        total_time,
                        final_price,
                        surcharge
                    ]
                    
                    # Add to options list
                    maas_options.append([request_id, detailed_itinerary] + [maas_option])
            
            # Update commuter requests
            for commuter in self.commuter_agents:
                if request_id in commuter.requests:
                    commuter.requests[request_id]['travel_options_with_maas'] = maas_options
            
            # Cache results
            if not hasattr(self, '_maas_options_cache'):
                self._maas_options_cache = {}
            self._maas_options_cache[cache_key] = maas_options
            
            # Limit cache size
            if len(self._maas_options_cache) > 50:
                oldest_key = next(iter(self._maas_options_cache))
                del self._maas_options_cache[oldest_key]
            
            return maas_options
        
        except Exception as e:
            print(f"Error generating MaaS options: {e}")
            empty_options = []
            for commuter in self.commuter_agents:
                if request_id in commuter.requests:
                    commuter.requests[request_id]['travel_options_with_maas'] = empty_options
            return empty_options


    #########################################################################################################
    ################################### Confirm Booking Options #############################################
    ########################################################################################################
    def book_service(self, request_id, ranked_options, current_step, availability_dict):
        """Book service with optimized database operations."""
        # Ensure subsidy pool is managed
        self.manage_subsidy_pool()
        
        # Check for existing booking to avoid duplication
        try:
            with self.Session() as session:
                # Use string conversion for request_id
                existing_booking = session.query(ServiceBookingLog).filter_by(
                    request_id=str(request_id)
                ).first()
                
                if existing_booking:
                    print(f"Request {request_id} already has a booking. Skipping.")
                    return True, availability_dict
        except Exception as e:
            print(f"Error checking existing booking: {e}")
        
        # Track if we need to update commuter requests
        booking_made = False
        
        # Find the appropriate commuter
        target_commuter = None
        for commuter in self.commuter_agents:
            if request_id in commuter.requests:
                target_commuter = commuter
                break
        
        if not target_commuter:
            print(f"No commuter found for request {request_id}")
            return False, availability_dict
        
        # Only process if request is active
        if target_commuter.requests[request_id]['status'] != 'active':
            return True, availability_dict
        
        # Track changes to availability_dict
        availability_changes = {}
        
        # Process ranked options
        for option in ranked_options:
            subsidy = option[-1]  # Extract subsidy
            
            try:
                if "maas" in option[1]:  # MaaS option
                    # Retrieve MaaS option details
                    maas_key = option[1]
                    
                    # Efficiently find the matching maas option
                    maas_option = None
                    travel_options = target_commuter.requests[request_id]['travel_options_with_maas']
                    for idx, mo in enumerate(travel_options):
                        if f"maas_{idx}" == maas_key:
                            maas_option = mo
                            break
                    
                    if maas_option:
                        # Process MaaS booking
                        detailed_itinerary = maas_option[1]
                        maas_option_details = maas_option[2]
                        
                        total_time = maas_option_details[2]
                        final_price = maas_option_details[3]
                        maas_surcharge = maas_option_details[4]
                        to_station_info = maas_option_details[0]
                        to_destination_info = maas_option_details[1]
                        
                        # Record booking
                        self.record_maas_or_non_maas_booking(
                            target_commuter,
                            request_id,
                            "MaaS_Bundle",
                            final_price,
                            start_time=target_commuter.requests[request_id]['start_time'],
                            duration=max(1, round(total_time)),
                            route=detailed_itinerary,
                            to_station_info=to_station_info,
                            to_destination_info=to_destination_info,
                            maas_surcharge=maas_surcharge,
                            availability_dict=availability_dict,
                            current_step=current_step,
                            subsidy=subsidy
                        )
                        
                        # Confirm booking
                        if self.confirm_booking_maas(target_commuter, request_id):
                            self.record_subsidy_usage(
                                commuter_id=target_commuter.unique_id,
                                request_id=request_id,
                                subsidy_amount=subsidy,
                                mode='maas',
                                timestamp=current_step
                            )
                            booking_made = True
                            break
                else:  # Non-MaaS option
                    # Extract option details
                    probability, mode_company, route, time = option[:4]
                    
                    # Get availability and price
                    travel_options = target_commuter.requests[request_id]['travel_options_without_maas']
                    availability_key = mode_company.replace('route', 'availability')
                    price_key = mode_company.replace('route', 'price')
                    
                    if availability_key not in travel_options or price_key not in travel_options:
                        continue
                    
                    availability = travel_options[availability_key]
                    price = travel_options[price_key]
                    
                    # Process public or walk options
                    if mode_company in ['public_route', 'walk_route']:
                        if self.confirm_booking_non_maas(target_commuter, request_id, mode_company, route, price):
                            self.record_maas_or_non_maas_booking(
                                target_commuter,
                                request_id,
                                mode_company,
                                price,
                                start_time=target_commuter.requests[request_id]['start_time'],
                                duration=max(1, round(time)),
                                route=route,
                                to_station_info=None,
                                to_destination_info=None,
                                maas_surcharge=0,
                                availability_dict=availability_dict,
                                current_step=current_step,
                                subsidy=subsidy
                            )
                            self.record_subsidy_usage(
                                commuter_id=target_commuter.unique_id,
                                request_id=request_id,
                                subsidy_amount=subsidy,
                                mode='public_subsidy',
                                timestamp=current_step
                            )
                            booking_made = True
                            break
                    else:  # Shared service option
                        chosen_company = mode_company.split('_')[1]
                        start_time = target_commuter.requests[request_id]['start_time']
                        duration = max(1, round(time))
                        end_check_step = min(current_step + 5, start_time + duration)
                        
                        # Check availability efficiently
                        all_steps_available = True
                        steps_to_check = {}
                        
                        for step in range(start_time, end_check_step + 1):
                            step_key = step - current_step
                            if step_key < 0 or step_key > 5:
                                all_steps_available = False
                                break
                                
                            availability_check_key = f'{chosen_company}_{step_key}'
                            current_avail = availability_dict.get(availability_check_key, 0)
                            
                            if current_avail < 1:
                                all_steps_available = False
                                break
                                
                            steps_to_check[step_key] = availability_check_key
                        
                        if all_steps_available:
                            if self.confirm_booking_non_maas(target_commuter, request_id, mode_company, route, price):
                                # Process booking
                                self.record_maas_or_non_maas_booking(
                                    target_commuter,
                                    request_id,
                                    mode_company,
                                    price,
                                    start_time=start_time,
                                    duration=duration,
                                    route=route,
                                    to_station_info=None,
                                    to_destination_info=None,
                                    maas_surcharge=0,
                                    availability_dict=availability_dict,
                                    current_step=current_step,
                                    subsidy=subsidy
                                )
                                self.record_subsidy_usage(
                                    commuter_id=target_commuter.unique_id,
                                    request_id=request_id,
                                    subsidy_amount=subsidy,
                                    mode=mode_company,
                                    timestamp=current_step
                                )
                                
                                # Update availability dict for shared services
                                if 'Uber' in mode_company or 'Bike' in mode_company:
                                    for step_key, avail_key in steps_to_check.items():
                                        availability_dict[avail_key] -= 1
                                        availability_changes[avail_key] = availability_dict[avail_key]
                                    
                                    # Record booking
                                    self.service_provider_agent.record_booking_log(
                                        target_commuter.unique_id,
                                        request_id,
                                        chosen_company,
                                        start_time,
                                        duration,
                                        list(range(start_time, start_time + duration)),
                                        route
                                    )
                                
                                booking_made = True
                                break
            except Exception as e:
                print(f"Error processing option: {e}")
                continue
        
        if not booking_made:
            print(f"Failed to book any service for request {request_id}")
        
        return booking_made, availability_dict

    def record_maas_or_non_maas_booking(self, commuter, request_id, mode_company, final_total_price, start_time, duration, route, to_station_info, to_destination_info, maas_surcharge, availability_dict, current_step, subsidy):

        if mode_company == "MaaS_Bundle":  # This is a MaaS option

            # Correctly use the data from the MaaS option for the record
            selected_route = {
                'route': route,
                'price': final_total_price,  # Correct total price after surcharge
                'MaaS_surcharge': maas_surcharge,
                'time': duration,  # Total time
                'to_station_info': to_station_info,
                'to_destination_info': to_destination_info,
                'route': route,
                'mode': 'MaaS_Bundle',
                'subsidy': subsidy
            }

            # Update the commuter's request with the selected MaaS option
            commuter.requests[request_id]['selected_route'] = selected_route
            commuter.requests[request_id]['status'] = 'Service Selected'
            
            # Call record_service_booking to ensure the service is logged in the ServiceBookingLog
            self.record_service_booking(commuter, request_id, selected_route, subsidy)
            
            # Extract route details for 'to station' and 'to destination'
            to_station_route = next((seg for seg in route if seg[0] == 'to station'), None)
            to_destination_route = next((seg for seg in route if seg[0] == 'to destination'), None)

            # Additionally, record the share service details if applicable
            if to_station_info and ('Uber' in to_station_info[0] or 'Bike' in to_station_info[0]):
                self.record_share_service_booking(
                    commuter,
                    request_id,
                    to_station_info,
                    start_time,
                    availability_dict,
                    current_step,
                    route_details=to_station_route  # Pass the relevant route details for 'to station'
                )
            if to_destination_info and ('Uber' in to_destination_info[0] or 'Bike' in to_destination_info[0]):
                self.record_share_service_booking(
                    commuter,
                    request_id,
                    to_destination_info,
                    start_time + duration,
                    availability_dict,
                    current_step,
                    route_details=to_destination_route  # Pass the relevant route details for 'to destination'
                )

        else:  # This is a non-MaaS option
                    # Gather the non-MaaS booking information
            selected_route = {
                'route': route,
                'price': final_total_price,  # Price of the non-MaaS service
                'time': duration,  # Total time
                'mode': mode_company,
                'subsidy': subsidy  # Include the subsidy in the selected route
            }

            # Call record_service_booking to ensure the service is logged in the ServiceBookingLog
            self.record_service_booking(commuter, request_id, selected_route, subsidy)
            
            self.service_provider_agent.record_booking_log(
                commuter.unique_id,
                request_id,
                mode_company,
                start_time,
                duration,
                list(range(start_time, start_time + duration)),
                route
            )

    def record_share_service_booking(self, commuter, request_id, service_info, start_time, availability_dict, current_step, route_details):
        company_name, time, price, detailed_route = service_info
        duration = max(1, round(time))

        # Prepare the affected steps for the share service
        affected_steps = list(range(start_time, start_time + duration))

        # Mapping of company names to provider IDs
        provider_mapping = {
            'BikeShare1': 3,
            'BikeShare2': 4,
            'UberLike1': 1,
            'UberLike2': 2
        }

        # Mapping of modes to mode IDs
        mode_mapping = {
            'BikeShare1': 4,  # bike
            'BikeShare2': 4,  # bike
            'UberLike1': 3,   # car
            'UberLike2': 3,   # car
            'walk': 5         # walk
        }

        # Get the mode_id and provider_id from the provided mappings
        provider_id = provider_mapping.get(company_name)
        mode_id = mode_mapping.get(company_name if 'walk' not in company_name else 'walk')

        # Convert UUID request_id to a string
        request_id_str = str(request_id)

        # Call to record the booking in ShareServiceBookingLog with the correct route details parameter
        try:
            with self.Session() as session:
                existing_booking = session.query(ShareServiceBookingLog).filter_by(
                    commuter_id=commuter.unique_id,
                    request_id=request_id_str  # Use the string version of request_id here
                ).first()

                if existing_booking:
                    print(f"[INFO] Share service booking already exists for commuter_id={commuter.unique_id}, request_id={request_id_str}. Skipping insert.")
                    return

                # Insert new booking into the ShareServiceBookingLog
                new_booking = ShareServiceBookingLog(
                    commuter_id=commuter.unique_id,
                    request_id=request_id_str,  # Ensure request_id is stored as a string
                    mode_id=mode_id,
                    provider_id=provider_id,
                    company_name=company_name,
                    start_time=start_time,
                    duration=duration,
                    affected_steps=affected_steps,
                    route_details=(route_details, detailed_route)
                )
                session.add(new_booking)
                session.commit()

        except Exception as e:
            print(f"[ERROR] Failed to record share service booking: {e}")

        # Update availability if needed
        if 'Uber' in company_name or 'Bike' in company_name:
            end_check_step = min(current_step + 5, start_time + duration)
            for step in range(start_time, end_check_step + 1):
                step_key = step - current_step
                availability_deduct_key = f'{company_name}_{step_key}'
                availability_dict[availability_deduct_key] -= 1


    def record_service_booking(self, commuter, request_id, selected_route, subsidy):
        payment_scheme = commuter.payment_scheme
        start_time = commuter.requests[request_id]['start_time']
        origin_coordinates = commuter.requests[request_id]['origin']
        destination_coordinates = commuter.requests[request_id]['destination']
        
        # Initialize the to_station and to_destination fields
        to_station_info = None
        to_destination_info = None
        
        # Determine if this is a MaaS option
        is_maas_option = isinstance(selected_route, dict) and 'MaaS_surcharge' in selected_route

        if is_maas_option:
            # MaaS Option Specific Fields
            record_company_name = 'MaaS_Bundle'
            route_details = selected_route['route']
            total_price = selected_route['price']
            maas_surcharge = selected_route['MaaS_surcharge']
            total_time = selected_route['time']

            # Extract 'to station' and 'to destination' information if available
            to_station_info = selected_route.get('to_station_info')
            to_destination_info = selected_route.get('to_destination_info')

        else:
            # Non-MaaS Option Specific Fields
            if 'public' in selected_route['mode']:
                record_company_name = 'public'
            elif 'walk' in selected_route['mode']:
                record_company_name = 'walk'
            else:
                record_company_name = selected_route['mode'].split('_')[1]
                    
            route_details = selected_route['route']
            total_price = selected_route['price']
            maas_surcharge = 0  # No MaaS surcharge for non-MaaS options
            total_time = selected_route['time']

        # Create a new booking log entry
        new_booking = ServiceBookingLog(
            commuter_id=commuter.unique_id,
            payment_scheme=payment_scheme,
            request_id=str(request_id),
            start_time=start_time,
            record_company_name=record_company_name,
            route_details=route_details,
            total_price=total_price,
            maas_surcharge=maas_surcharge,
            total_time=total_time,
            origin_coordinates=origin_coordinates,
            destination_coordinates=destination_coordinates,
            to_station=to_station_info,
            to_destination=to_destination_info,
            government_subsidy=subsidy,
            status='Service Selected'  # Initialize with the Service Selected status
        )

        try:
            Session = sessionmaker(bind=self.db_engine)
            with Session() as session:
                existing_booking = session.query(ServiceBookingLog).filter_by(
                    commuter_id=commuter.unique_id,
                    payment_scheme=payment_scheme,
                    request_id=str(request_id)
                ).first()

                if existing_booking:
                    print(f"Booking already exists for commuter_id {commuter.unique_id}, request_id {request_id}. Updating status.")
                    # Update status if booking exists
                    existing_booking.status = 'Service Selected'
                    session.commit()
                else:
                    session.add(new_booking)
                    session.commit()

        except SQLAlchemyError as e:
            print(f"[ERROR] Error recording service booking: {e}")
    
    def confirm_booking_maas(self, commuter, request_id):


        result = commuter.accept_service(request_id)
        return result


    def confirm_booking_non_maas(self, commuter, request_id, mode, route, price):
        selected_route = {
            'mode': mode,
            'route': route,
            'price': price
        }
        return commuter.accept_service_non_maas(request_id, selected_route)
###################################################################################################################
#######################################Fixed Pool Subsisdy ########################################################
###################################################################################################################
    def manage_subsidy_pool(self):
        current_step = self.model.get_current_step()
        current_day = current_step // 144
        current_week = current_day // 7
        day_of_week = current_day % 7

        # Calculate current usage from log
        with self.Session() as session:
            total_used = session.query(func.sum(SubsidyUsageLog.subsidy_amount))\
                .filter(SubsidyUsageLog.week == current_week)\
                .scalar() or 0

        # Reset pool at beginning of week
        if self.subsidy_config.is_reset_time(current_step, self.last_reset_step):
            self.current_subsidy_pool = self.subsidy_config.total_amount
            self.last_reset_step = current_step
        else:
            self.current_subsidy_pool = self.subsidy_config.total_amount - total_used

        # No subsidies on weekends
        if day_of_week >= 5:
            self.current_subsidy_pool = 0

    def check_subsidy_availability(self, requested_amount):
        """
        Check subsidy availability by calculating total used subsidies from SubsidyUsageLog
        """
        current_step = self.model.get_current_step()
        current_day = current_step // 144
        current_week = current_day // 7
        day_of_week = current_day % 7

        # Check if it's a weekday
        if not self.subsidy_config.is_subsidy_available(day_of_week):
            print("Weekend - No subsidies available")
            return 0

        # Calculate total subsidies already used in current period
        with self.Session() as session:
            # Add this line to ensure fresh data
            session.expire_all()
            if self.subsidy_config.pool_type == 'weekly':
                total_used = session.query(func.sum(SubsidyUsageLog.subsidy_amount))\
                    .filter(SubsidyUsageLog.week == current_week)\
                    .scalar() or 0
            elif self.subsidy_config.pool_type == 'daily':
                total_used = session.query(func.sum(SubsidyUsageLog.subsidy_amount))\
                    .filter(SubsidyUsageLog.day == current_day)\
                    .scalar() or 0
            elif self.subsidy_config.pool_type == 'monthly':
                total_used = session.query(func.sum(SubsidyUsageLog.subsidy_amount))\
                    .filter(SubsidyUsageLog.month == current_day // 30)\
                    .scalar() or 0

        remaining_pool = self.subsidy_config.total_amount - total_used
        #print(f"Total subsidies used: {total_used}, Remaining in pool: {remaining_pool}")

        # If there's enough remaining in the pool
        if remaining_pool >= requested_amount:
            #print(f"Providing subsidy: {requested_amount}")
            return requested_amount
        elif remaining_pool > 0:
            #print(f"Providing partial subsidy: {remaining_pool}")
            return remaining_pool
        else:
            #print("Subsidy pool depleted")
            return 0

    def get_subsidy_statistics(self):
        """
        Get current statistics about subsidy usage
        """
        current_step = self.model.get_current_step()
        steps_per_day = 144
        current_day = current_step // steps_per_day
        day_of_week = current_day % 7
        
        stats = {
            'pool_type': self.subsidy_config.pool_type,
            'total_pool': self.subsidy_config.total_amount,
            'remaining_pool': self.current_subsidy_pool,
            'total_given': self.total_subsidies_given,
            'current_day': current_day,
            'day_of_week': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][day_of_week],
            'subsidies_available_today': self.subsidy_config.is_subsidy_available(day_of_week)
        }
        
        return stats
    
    def record_subsidy_usage(self, commuter_id, request_id, subsidy_amount, mode, timestamp):
        day = timestamp // 144
        week = day // 7
        month = day // 30
         # Convert NumPy float to Python float
        if hasattr(subsidy_amount, 'item'):  # Check if it's a NumPy type
            subsidy_amount = float(subsidy_amount)  # Convert to standard Python float
        
        with self.Session() as session:
            usage_log = SubsidyUsageLog(
                commuter_id=commuter_id,
                request_id=str(request_id),
                subsidy_amount=subsidy_amount,
                mode=mode,
                timestamp=timestamp,
                day=day,
                week=week,
                month=month,
                period_type=self.subsidy_config.pool_type
            )
            session.add(usage_log)
            session.flush()  # Ensure changes are pushed to the database
            session.commit()
