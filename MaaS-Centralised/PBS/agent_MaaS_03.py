from mesa import Agent
import heapq
import math
from mesa.space import MultiGrid
from agent_service_provider_initialisation_03 import ShareServiceBookingLog, ServiceBookingLog
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import create_engine
import uuid
import random
class MaaS(Agent):
    def __init__(self, unique_id, model, service_provider_agent, commuter_agents, DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS, \
                 BACKGROUND_TRAFFIC_AMOUNT, stations, routes, transfers, num_commuters, grid_width, grid_height, CONGESTION_ALPHA,\
                CONGESTION_BETA, CONGESTION_CAPACITY, CONGESTION_T_IJ_FREE_FLOW):
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
    def generate_random_route(self):
        """Generate a random route on the grid from a start point to an end point."""
        start_x, start_y = random.randint(0, self.grid_width - 1), random.randint(0, self.grid_height - 1)
        end_x, end_y = random.randint(0, self.grid_width - 1), random.randint(0, self.grid_height - 1)
        
        # Initialize the route with the start point
        route = [(start_x, start_y)]
        
        # Generate the route from start to end by moving step by step
        current_x, current_y = start_x, start_y
        while current_x != end_x or current_y != end_y:
            if current_x < end_x:
                current_x += 1
            elif current_x > end_x:
                current_x -= 1
            elif current_y < end_y:
                current_y += 1
            elif current_y > end_y:
                current_y -= 1
            route.append((current_x, current_y))
        
        return route

    def insert_random_traffic(self, session):
        """Insert random traffic into the ShareServiceBookingLog as background traffic.
        The start time will be within 5 steps ahead of the current simulation step.
        """
        # Get the current simulation step
        current_step = self.model.get_current_step()
        num_routes= self.background_traffic_amount
        for i in range(num_routes):
            # Generate a random route
            route = self.generate_random_route()
            
            # Generate a start time within 5 steps ahead of the current simulation step
            start_time = random.randint(current_step, current_step + 5)
            
            # Calculate the duration of the route (based on its length)
            duration = len(route)//6
            
            # Generate a unique request ID for each route
            unique_request_id = str(uuid.uuid4())
            # Create a new booking log entry for background traffic
            new_booking = ShareServiceBookingLog(
                commuter_id=-1,  # No commuter ID for background traffic
                request_id=unique_request_id,  # Set request_id to None for background traffic
                mode_id=0,  # No transport mode
                provider_id=0,  # No provider
                company_name="background_traffic",  # Set as background traffic
                start_time=start_time,
                duration=duration,
                affected_steps=[start_time + i for i in range(duration)],  # Each time step is affected during the route
                route_details=route  # The generated route
            )
            
            # Insert the new background traffic entry into the database
            session.add(new_booking)
        
        # Commit the changes to the database
        session.commit()


    ########################################################################################################
    ##################################### Congestion and Shortest Route ########################################
    ########################################################################################################
    def get_current_traffic_volume(self, current_time_step):
        traffic_volume = {}
        
        with self.Session() as session:
            # Query all bookings that are active at the current time step with mode_id as car or background traffic
            active_bookings = session.query(ShareServiceBookingLog).filter(
                ShareServiceBookingLog.start_time <= current_time_step,
                (ShareServiceBookingLog.start_time + ShareServiceBookingLog.duration) > current_time_step,
                ShareServiceBookingLog.mode_id.in_([0, 3])  # Only consider mode_id 0 (background) and 3 (car)
            ).all()

            for booking in active_bookings:
                route = booking.route_details  # Could be in either format
                affected_steps = booking.affected_steps  # List of time steps the booking affects
                if current_time_step in affected_steps:
                    # Check the format of the route_details
                    if isinstance(route[0], list) and isinstance(route[0][0], str):
                        # Type 1: Route contains high-level details followed by coordinates
                        route_segments = route[1]  # Extract the detailed route (e.g., [[15, 17], [15, 18], [16, 18], [17, 18]])
                    else:
                        # Type 2: Route is just a list of coordinates
                        route_segments = route  # e.g., [[48, 7], [47, 7], [46, 7], [45, 7]]

                    # For each segment in the route
                    for i in range(len(route_segments) - 1):
                        segment = (tuple(route_segments[i]), tuple(route_segments[i + 1]))  # ((x1, y1), (x2, y2))
                        # Since the grid is bidirectional, consider both directions
                        reverse_segment = (tuple(route_segments[i + 1]), tuple(route_segments[i]))
                        # Increment the volume on this segment
                        traffic_volume[segment] = traffic_volume.get(segment, 0) + 1
                        traffic_volume[reverse_segment] = traffic_volume.get(reverse_segment, 0) + 1

        return traffic_volume



    def calculate_congested_travel_time(self, V_ij, C_ij, T_ij_free_flow):

        congestion_factor = (1 + self.congestion_alpha * (V_ij / max(C_ij, 1)) ** self.congestion_beta)
        congested_time = T_ij_free_flow * congestion_factor
        return congested_time

    def dijkstra_with_congestion(self, start, end, mode_id):
        moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Adjacent moves
        rows, cols = self.model.grid.width, self.model.grid.height
        min_cost = [[float('inf')] * cols for _ in range(rows)]
        min_cost[start[0]][start[1]] = 0
        predecessor = [[None] * cols for _ in range(rows)]
        queue = [(0, start)]

        # Get current traffic volume
        current_time_step = self.model.get_current_step()
        traffic_volume = self.get_current_traffic_volume(current_time_step)

        while queue:
            current_cost, (x, y) = heapq.heappop(queue)

            if (x, y) == end:
                break

            for dx, dy in moves:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols:
                    # Define the segment
                    segment = ((x, y), (nx, ny))

                    if mode_id == 3:  # Only calculate congestion for cars
                        # Get volume on this segment
                        V_ij = traffic_volume.get(segment, 0)

                        # Calculate congested travel time
                        congested_time = self.calculate_congested_travel_time(
                            V_ij, self.congestion_capacity , self.conjestion_t_ij_free_flow
                        )
                        next_cost = current_cost + congested_time
                    else:
                        # For non-car modes, assume free-flow travel time (no congestion)
                        next_cost = current_cost + self.conjestion_t_ij_free_flow

                    if next_cost < min_cost[nx][ny]:
                        min_cost[nx][ny] = next_cost
                        predecessor[nx][ny] = (x, y)
                        heapq.heappush(queue, (next_cost, (nx, ny)))

        # Reconstruct path
        path = []
        step = end
        while step != start:
            path.append(step)
            step = predecessor[step[0]][step[1]]
        path.append(start)
        path.reverse()

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

        # Calculate dynamic surcharge
        S_dynamic = S_base * (1 + alpha * (UR - 1))

        # Ensure the surcharge is non-negative
        S_dynamic = max(S_dynamic, 0)

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

        origin_mode, origin_station = self.find_nearest_station_any_mode(origin_point)
        destination_mode, destination_station = self.find_nearest_station_any_mode(destination_point)

        best_times = {station: float('inf') for mode in self.stations for station in self.stations[mode]}
        best_paths = {}

        best_times[origin_station] = 0
        queue = [(origin_station, 0)]

        while queue:
            current_station, current_time = queue.pop(0)

            for mode in self.routes:
                for route_id in self.get_routes_through_station(mode, current_station):
                    for next_station in self.routes[mode][route_id]:
                        travel_time = 1  # 1 tick per stop
                        arrival_time = current_time + travel_time

                        if arrival_time < best_times[next_station]:
                            best_times[next_station] = arrival_time
                            best_paths[next_station] = current_station
                            queue.append((next_station, arrival_time))

            for next_station, transfer_time in self.get_transfers_from_station(current_station):
                arrival_time = current_time + transfer_time
                if arrival_time < best_times[next_station]:
                    best_times[next_station] = arrival_time
                    best_paths[next_station] = current_station
                    queue.append((next_station, arrival_time))

        path = []
        current = destination_station
        while current != origin_station:
            path.append(current)
            current = best_paths.get(current)
            if current is None:
                print(f"Path not found from {origin_station} to {destination_station}")  # Debug print
                return None
        path.append(origin_station)
        path.reverse()
        return path


    def build_detailed_itinerary(self, optimal_path, origin_point, destination_point):
        detailed_itinerary = []

        if not optimal_path:
            return detailed_itinerary

        if len(optimal_path) == 1:
            return detailed_itinerary

        origin_station_mode, origin_station = self.find_nearest_station_any_mode(origin_point)
        detailed_itinerary.append(('to station', [origin_point, origin_station]))

        for i in range(len(optimal_path) - 1):
            current_station = optimal_path[i]
            next_station = optimal_path[i + 1]

            current_mode = 'bus' if current_station.startswith('B') else 'train'
            next_mode = 'bus' if next_station.startswith('B') else 'train'

            if current_mode == next_mode:
                for route_id, station_sequence in self.routes[current_mode].items():
                    if current_station in station_sequence and next_station in station_sequence:
        
                        if detailed_itinerary[-1][0] == 'transfer' or detailed_itinerary[-1][0] != current_mode:
                            
                            detailed_itinerary.append((current_mode, route_id, [current_station]))

                        detailed_itinerary[-1][-1].append(next_station)
                        break
            else:
                detailed_itinerary.append(('transfer', [current_station, next_station]))

                

        detailed_itinerary.append(('to destination', [optimal_path[-1], destination_point]))
        return detailed_itinerary

    def calculate_total_time_and_price_public(self, payment_scheme, detailed_itinerary, walking_speed, bus_stop_speed, train_stop_speed, bus_stop_price, train_stop_price):
        total_time = 0
        total_price = 0

        for segment in detailed_itinerary:

            if segment[0] == 'to station':
                get_on_station_name = segment[1][1]
                if get_on_station_name.startswith('T'):
                    get_on_station_coordinates = self.stations['train'][get_on_station_name]
                elif get_on_station_name.startswith('B'):
                    get_on_station_coordinates = self.stations['bus'][get_on_station_name]
                else:
                    raise ValueError(f"Station {get_on_station_name} is not recognized as a train or bus station.")

                walking_route = self.dijkstra_with_congestion(segment[1][0], get_on_station_coordinates,5)
                walk_distance = self.calculate_total_route_length(walking_route)
                walk_time = walk_distance / walking_speed
                total_time += walk_time

            elif segment[0] == 'bus' or segment[0] == 'train':
                mode = segment[0]
                route_id = segment[1]
                stops = segment[2]
                route_list = self.routes[mode][route_id]

                get_on_stop = stops[0]
                get_off_stop = stops[1]
                get_on_index = route_list.index(get_on_stop)
                get_off_index = route_list.index(get_off_stop)
                number_of_stops = abs(get_off_index - get_on_index) + 1
                if segment[0] == 'bus':
                    each_stop_speed = bus_stop_speed
                    price_per_stop = bus_stop_price
                else: 
                    each_stop_speed = train_stop_speed
                    price_per_stop = train_stop_price
                travel_time = number_of_stops * each_stop_speed
                waiting_time = 1 # Default waiting time is 1 step
                segment_time = travel_time + waiting_time

                total_time += segment_time

                segment_price = number_of_stops * float(price_per_stop)

                total_price += segment_price

            elif segment[0] == 'transfer':
                transfer_stations = tuple(segment[1])
                if transfer_stations in self.transfers:
                    transfer_time = self.transfers[transfer_stations]
                else:
                    raise ValueError(f"Transfer from {transfer_stations[0]} to {transfer_stations[1]} is not recognized.")
                total_time += transfer_time
        # Calculate the dynamic PAYG surcharge

        if payment_scheme == 'PAYG':
            PAYG_surcharge_percentage = self.calculate_dynamic_MaaS_surcharge(payment_scheme)
            total_price_with_surcharge = total_price * (1 + PAYG_surcharge_percentage)
            MaaS_surcharge = total_price_with_surcharge - total_price
        elif payment_scheme == 'subscription':
            # Calculate the fixed reduced surcharge for subscription
            subscription_surcharge_percentage = self.calculate_dynamic_MaaS_surcharge(payment_scheme)
            total_price_with_surcharge = total_price * (1 + subscription_surcharge_percentage)
            MaaS_surcharge = total_price_with_surcharge - total_price
        else:
            # Default case (if payment scheme is not recognized)
            total_price_with_surcharge = total_price
            MaaS_surcharge = 0

        return total_time, total_price_with_surcharge, MaaS_surcharge

    def get_routes_through_station(self, mode, station):
        return [route_id for route_id, stations in self.routes[mode].items() if station in stations]

    def get_transfers_from_station(self, station):
        return [(dest, time) for (src, dest), time in self.transfers.items() if src == station]

    @staticmethod
    def calculate_distance(point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        delta_x = x2 - x1
        delta_y = y2 - y1
        return math.sqrt(delta_x**2 + delta_y**2)
    def calculate_total_route_length(self, route):
        total_length = 0.0
        for i in range(len(route) - 1):
            total_length += self.calculate_distance(route[i], route[i + 1])
        return total_length

    def options_without_maas(self, request_id, start_time, origin, destination):
        travel_options = {'request_id': request_id}
        # Calculate single mode options
        single_modes = ['walk', 'bike', 'car']
        for mode in single_modes:
            
            if mode == 'walk':
                unit_price = 0
                unit_speed = self.service_provider_agent.get_travel_speed(mode, start_time)

                shortest_route, total_time, total_price = self.calculate_single_mode_time_and_price(origin, destination, unit_price, unit_speed, 5)
                
                availability = float('inf')  # Walking always available
                travel_options[f'{mode}_route'] = shortest_route
                travel_options[f'{mode}_time'] = total_time
                travel_options[f'{mode}_price'] = total_price
                travel_options[f'{mode}_availability'] = availability

            elif mode == 'bike':
            
                price_dict = self.service_provider_agent.get_shared_service_price(mode, start_time)
              
                if price_dict is None:
                    print(f"No price information available for mode: {mode}")
                    continue

                for company_name, unit_price in price_dict.items():
                    
                    unit_speed = self.service_provider_agent.get_travel_speed(mode, start_time)
                   
                    shortest_route, total_time, total_price = self.calculate_single_mode_time_and_price(origin, destination, unit_price, unit_speed, 4)
                    
                    availability = self.service_provider_agent.check_shared_availability(company_name, start_time)
                    travel_options[f'{mode}_{company_name}_route'] = shortest_route
                    travel_options[f'{mode}_{company_name}_time'] = total_time
                    travel_options[f'{mode}_{company_name}_price'] = total_price
                    travel_options[f'{mode}_{company_name}_availability'] = availability
            else:
                price_dict = self.service_provider_agent.get_shared_service_price(mode, start_time)
              
                if price_dict is None:
                    print(f"No price information available for mode: {mode}")
                    continue

                for company_name, unit_price in price_dict.items():
                    
                    unit_speed = self.service_provider_agent.get_travel_speed(mode, start_time)
                   
                    shortest_route, total_time, total_price = self.calculate_single_mode_time_and_price(origin, destination, unit_price, unit_speed, 3)
                    
                    availability = self.service_provider_agent.check_shared_availability(company_name, start_time)
                    travel_options[f'{mode}_{company_name}_route'] = shortest_route
                    travel_options[f'{mode}_{company_name}_time'] = total_time
                    travel_options[f'{mode}_{company_name}_price'] = total_price
                    travel_options[f'{mode}_{company_name}_availability'] = availability
        # Calculate public transport options
        optimal_path = self.find_optimal_route(origin, destination)

        detailed_itinerary = self.build_detailed_itinerary(optimal_path, origin, destination)
        walking_speed = self.service_provider_agent.get_travel_speed('walk', start_time)
        bus_stop_speed = self.service_provider_agent.get_travel_speed('bus', start_time)
        train_stop_speed = self.service_provider_agent.get_travel_speed('train', start_time)
        bus_stop_price = self.service_provider_agent.get_public_service_price('bus', start_time)
        train_stop_price = self.service_provider_agent.get_public_service_price('train', start_time)

        public_time, public_price = self.calculate_public_transport_time_and_price(
        detailed_itinerary, bus_stop_speed, train_stop_speed, bus_stop_price, train_stop_price
        )

        travel_options['public_route'] = detailed_itinerary
        travel_options['public_time'] = public_time
        travel_options['public_price'] = public_price
        travel_options['public_availability'] = float('inf')  # Assuming always available

        # Store travel_options in the corresponding request
        for commuter in self.commuter_agents:
            if request_id in commuter.requests:
                commuter.requests[request_id]['travel_options_without_maas'] = travel_options
                
        return travel_options
    
    
    ########################################################################################################
    ##################################### For MaaS Options ########################################
    ########################################################################################################
    def calculate_time_and_price_to_station_or_destination(self, segment, start_time):
        
        options = []
        
        # Extract origin and destination from the segment
        if segment[0] == 'to station':
            origin = segment[1][0]  # e.g., (24, 36)
            destination_station = segment[1][1]  # e.g., 'T4-7'
            
            if destination_station.startswith('T'):
                destination = self.stations['train'][destination_station]
            elif destination_station.startswith('B'):
                destination = self.stations['bus'][destination_station]
            else:
                raise ValueError(f"Station {destination_station} is not recognized as a train or bus station.")
        
        elif segment[0] == 'to destination':
            origin_station = segment[1][0]  # e.g., 'B86'
            destination = segment[1][1]  # e.g., (44, 25)
            
            if origin_station.startswith('T'):
                origin = self.stations['train'][origin_station]
            elif origin_station.startswith('B'):
                origin = self.stations['bus'][origin_station]
            else:
                raise ValueError(f"Station {origin_station} is not recognized as a train or bus station.")
        else:
            raise ValueError(f"Segment type {segment[0]} is not recognized.")


        # Loop through the single modes
        single_modes = ['walk', 'bike', 'car']
        for mode in single_modes:
            
            if mode == 'walk':
                unit_price = 0
                unit_speed = self.service_provider_agent.get_travel_speed(mode, start_time)
                
                shortest_route, total_time, total_price = self.calculate_single_mode_time_and_price(origin, destination, unit_price, unit_speed, 5)
                
                options.append((segment[0], segment[1], mode, total_time, total_price, shortest_route))  # Add shortest route here
            elif mode == 'bike':
                price_dict = self.service_provider_agent.get_shared_service_price(mode, start_time)
        
                if price_dict is None:
                    print(f"No price information available for mode: {mode}")
                    continue
                
                for company_name, unit_price in price_dict.items():
                    unit_speed = self.service_provider_agent.get_travel_speed(mode, start_time)
                    
                    shortest_route, total_time, total_price = self.calculate_single_mode_time_and_price(origin, destination, unit_price, unit_speed, 4)
                    
                    # Check availability for this service
                    availability = self.service_provider_agent.check_shared_availability(company_name, start_time)
                    
                    if availability > 0:  # Only consider options that are available
                        options.append((segment[0], segment[1], company_name, total_time, total_price, shortest_route))  # Add shortest route here
                    else:
                        print(f"{company_name} not available at this time.")
            else:
                price_dict = self.service_provider_agent.get_shared_service_price(mode, start_time)
        
                if price_dict is None:
                    print(f"No price information available for mode: {mode}")
                    continue
                
                for company_name, unit_price in price_dict.items():
                    unit_speed = self.service_provider_agent.get_travel_speed(mode, start_time)
                    
                    shortest_route, total_time, total_price = self.calculate_single_mode_time_and_price(origin, destination, unit_price, unit_speed, 3)
                    
                    # Check availability for this service
                    availability = self.service_provider_agent.check_shared_availability(company_name, start_time)
                    
                    if availability > 0:  # Only consider options that are available
                        options.append((segment[0], segment[1], company_name, total_time, total_price, shortest_route))  # Add shortest route here
                    else:
                        print(f"{company_name} not available at this time.")

        return options



    
    def calculate_public_transport_time_and_price(self, detailed_itinerary, bus_stop_speed, train_stop_speed, bus_stop_price, train_stop_price):
        total_time = 0
        total_price = 0

        for segment in detailed_itinerary:
            if segment[0] == 'bus' or segment[0] == 'train':
                mode = segment[0]
                route_id = segment[1]
                stops = segment[2]
                route_list = self.routes[mode][route_id]

                get_on_stop = stops[0]
                get_off_stop = stops[1]
                get_on_index = route_list.index(get_on_stop)
                get_off_index = route_list.index(get_off_stop)
                number_of_stops = abs(get_off_index - get_on_index) + 1
                if segment[0] == 'bus':
                    each_stop_speed = bus_stop_speed
                    price_per_stop = bus_stop_price
                else: 
                    each_stop_speed = train_stop_speed
                    price_per_stop = train_stop_price
                travel_time = number_of_stops * each_stop_speed
                waiting_time = 0.5 # Default waiting time is 1 step
                segment_time = travel_time + waiting_time

                total_time += segment_time

                segment_price = number_of_stops * float(price_per_stop)

                total_price += segment_price

            elif segment[0] == 'transfer':
                transfer_stations = tuple(segment[1])
                if transfer_stations in self.transfers:
                    transfer_time = self.transfers[transfer_stations]
                else:
                    raise ValueError(f"Transfer from {transfer_stations[0]} to {transfer_stations[1]} is not recognized.")
                total_time += transfer_time

        return total_time, total_price    
    
    def apply_maas_surcharge(self, total_price, payment_scheme):
        if payment_scheme == 'PAYG':
            # Calculate the dynamic PAYG surcharge
            PAYG_surcharge_percentage = self.calculate_dynamic_MaaS_surcharge(payment_scheme)
            total_price_with_surcharge = total_price * (1 + PAYG_surcharge_percentage)
            MaaS_surcharge = total_price_with_surcharge - total_price
        elif payment_scheme == 'subscription':
            # Calculate the fixed reduced surcharge for subscription
            subscription_surcharge_percentage = self.calculate_dynamic_MaaS_surcharge(payment_scheme)
            total_price_with_surcharge = total_price * (1 + subscription_surcharge_percentage)
            MaaS_surcharge = total_price_with_surcharge - total_price
        else:
            # Default case (if payment scheme is not recognized)
            total_price_with_surcharge = total_price
            MaaS_surcharge = 0
        
        return total_price_with_surcharge, MaaS_surcharge
    
    def maas_options(self, payment_scheme, request_id, start_time, origin, destination):
        
        # Step 1: Find the optimal public transport route
        detailed_itinerary = self.build_detailed_itinerary(self.find_optimal_route(origin, destination), origin, destination)
        
        # Step 2: Calculate time, price, and route options for 'to station' segment
        to_station_options = self.calculate_time_and_price_to_station_or_destination(detailed_itinerary[0], start_time)
        
        # Step 3: Calculate the total time and price for the public transport segment (train + transfer)
        public_transport_time, public_transport_price = self.calculate_public_transport_time_and_price(
            detailed_itinerary, 
            bus_stop_speed=self.service_provider_agent.get_travel_speed('bus', start_time),
            train_stop_speed=self.service_provider_agent.get_travel_speed('train', start_time),
            bus_stop_price=self.service_provider_agent.get_public_service_price('bus', start_time),
            train_stop_price=self.service_provider_agent.get_public_service_price('train', start_time)
        )
        
        # Step 4: Calculate time, price, and route options for 'to destination' segment
        to_destination_options = self.calculate_time_and_price_to_station_or_destination(detailed_itinerary[-1], start_time)
        
        # Step 5: Combine the options to generate full routes
        maas_options = []

        for to_station_option in to_station_options:
            for to_destination_option in to_destination_options:
                
                total_time = (
                    to_station_option[3]  # Time for 'to station'
                    + public_transport_time  # Time for public transport
                    + to_destination_option[3]  # Time for 'to destination'
                )
                total_price = (
                    to_station_option[4]  # Price for 'to station'
                    + public_transport_price  # Price for public transport
                    + to_destination_option[4]  # Price for 'to destination'
                )
                
                # Apply MaaS surcharge to the total price
                final_total_price, maas_surcharge = self.apply_maas_surcharge(total_price, payment_scheme)
                
                # Extract shortest route for 'to station' and 'to destination'
                to_station_route = to_station_option[5]  # Shortest route for 'to station'
                to_destination_route = to_destination_option[5]  # Shortest route for 'to destination'
                
                # Combine the mode, cost, and route details into a MaaS option
                maas_option = [
                    [to_station_option[2], to_station_option[3], to_station_option[4], to_station_route],  # Mode, time, price, and route for 'to station'
                    [to_destination_option[2], to_destination_option[3], to_destination_option[4], to_destination_route],  # Mode, time, price, and route for 'to destination'
                    total_time,  # Total time
                    final_total_price,  # Final total price after surcharge
                    maas_surcharge  # MaaS surcharge
                ]
                
                # Append the final option list with the request_id, detailed_itinerary, and MaaS options
                maas_options.append([request_id, detailed_itinerary] + [maas_option])

        # Store travel_options in the corresponding request for each commuter
        for commuter in self.commuter_agents:
            if request_id in commuter.requests:
                commuter.requests[request_id]['travel_options_with_maas'] = maas_options
                
        # Return all generated MaaS options
        return maas_options



    #########################################################################################################
    ################################### Confirm Booking Options #############################################
    ########################################################################################################
    def book_service(self, request_id, ranked_options, current_step, availability_dict):
        for commuter in self.commuter_agents:
            if request_id in commuter.requests:
                for option in ranked_options:
                    subsidy = option[-1]  # Extract the subsidy from the ranked option
                    if "maas" in option[1]:  # This indicates a MaaS option
                        # Retrieve the MaaS option details
                        maas_key = option[1]
                        maas_option = next(
                            (mo for mo in commuter.requests[request_id]['travel_options_with_maas'] if f"maas_{commuter.requests[request_id]['travel_options_with_maas'].index(mo)}" == maas_key), None
                        )

                        if maas_option:
                            # Remove overwriting request_id here

                            detailed_itinerary = maas_option[1]
                            maas_option_details = maas_option[2]

                            total_time = maas_option_details[2]
                            final_total_price = maas_option_details[3]
                            maas_surcharge = maas_option_details[4]
                            to_station_info = maas_option_details[0]
                            to_destination_info = maas_option_details[1]

                            # Record the MaaS booking and update availability if necessary
                            self.record_maas_or_non_maas_booking(
                                commuter,
                                request_id,
                                "MaaS_Bundle",
                                final_total_price,
                                start_time=commuter.requests[request_id]['start_time'],
                                duration=max(1, round(total_time)),
                                route=detailed_itinerary,
                                to_station_info=to_station_info,
                                to_destination_info=to_destination_info,
                                maas_surcharge=maas_surcharge,
                                availability_dict=availability_dict,
                                current_step=current_step,
                                subsidy=subsidy  # Pass the subsidy to be recorded
                            )

                            # Call self.confirm_booking to finalize the booking for the MaaS option
                            if self.confirm_booking_maas(
                                commuter, 
                                request_id, 
                            ):
                                return True, availability_dict
                            else:
                                print(f"[DEBUG] MaaS booking confirmation failed for Request ID: {request_id}")

                    else:  # This is a non-MaaS option
                        try:
                            probability, mode_company, route, time= option[:4]  # Extract the basic elements
                        except ValueError as e:
                            print(f"Error unpacking option: {option}, {e}")
                            continue

                        availability_key = mode_company.replace('route', 'availability')
                        price_key = mode_company.replace('route', 'price')

                        try:
                            availability = commuter.requests[request_id]['travel_options_without_maas'][availability_key]
                            price = commuter.requests[request_id]['travel_options_without_maas'][price_key]
                        except KeyError as e:
                            print(f"Error accessing travel options: {e}, request_id: {request_id}, travel_options: {commuter.requests[request_id]['travel_options_without_maas']}")
                            continue


                        if mode_company == 'public_route' or mode_company == 'walk_route':
                            if self.confirm_booking_non_maas(commuter, request_id, mode_company, route, price):  # Pass 'time'
                                self.record_maas_or_non_maas_booking(
                                    commuter,
                                    request_id,
                                    mode_company,
                                    price,
                                    start_time = commuter.requests[request_id]['start_time'],
                                    duration=max(1, round(time)),
                                    route=route,
                                    to_station_info=None,
                                    to_destination_info=None,
                                    maas_surcharge=0,
                                    availability_dict=availability_dict,
                                    current_step=current_step,
                                    subsidy=subsidy  # Save subsidy
                                )
                                return True, availability_dict
                        else:
                            chosen_company = mode_company.split('_')[1]
                            start_time = commuter.requests[request_id]['start_time']
                            duration = max(1, round(time))
                            end_check_step = min(current_step + 5, start_time + duration)

                            all_steps_available = True
                            for step in range(start_time, end_check_step + 1):
                                step_key = step - current_step
                                if step_key < 0 or step_key > 5:
                                    all_steps_available = False
                                    break
                                availability_check_key = f'{chosen_company}_{step_key}'
                                if availability_dict.get(availability_check_key, 0) < 1:
                                    all_steps_available = False
                                    break

                            if all_steps_available:
                                if self.confirm_booking_non_maas(commuter, request_id, mode_company, route, price):  # Pass 'time'
                                    self.record_maas_or_non_maas_booking(
                                        commuter,
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
                                        subsidy=subsidy  # Save subsidy
                                    )
                                    if 'Uber' in mode_company or 'Bike' in mode_company:
                                        for step in range(start_time, end_check_step + 1):
                                            step_key = step - current_step
                                            availability_deduct_key = f'{chosen_company}_{step_key}'
                                            availability_dict[availability_deduct_key] -= 1
                                        self.service_provider_agent.record_booking_log(
                                            commuter.unique_id,
                                            request_id,
                                            chosen_company,
                                            start_time,
                                            duration,
                                            list(range(start_time, start_time + duration)),
                                            route
                                        )
                                        return True, availability_dict
                                else:
                                    print(f"Booking confirmation failed for request {request_id}")

        print(f"Failed to book any service for request {request_id}")
        return False, availability_dict



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
            government_subsidy=subsidy  # Add the government subsidy
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
                    print(f"Booking already exists for commuter_id {commuter.unique_id}, request_id {request_id}. Skipping insert.")
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
