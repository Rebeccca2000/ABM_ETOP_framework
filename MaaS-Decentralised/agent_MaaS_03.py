from mesa import Agent
from database_01 import DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS, stations, routes, transfers, num_commuters, grid_width, grid_height
import heapq
import math
from mesa.space import MultiGrid
from agent_service_provider_initialisation_03 import ServiceBookingLog
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import create_engine
class MaaS(Agent):
    def __init__(self, unique_id, model, service_provider_agent, commuter_agents):
        super().__init__(unique_id, model)  # Initialize the base class
        self.service_provider_agent = service_provider_agent  # Store the reference to the service provider agent
        self.commuter_agents = commuter_agents  # Store the reference to the commuter agent
        self.stations = stations  # Dictionary of stations
        self.routes = routes  # Dictionary of routes
        self.transfers = transfers  # Dictionary of transfer times
        self.grid = MultiGrid(width=grid_width, height=grid_height, torus=False)  
        self.shortest_path_cache = {}  # Cache to store shortest paths for optimization
        self.db_engine = create_engine(self.model.db_connection_string)
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

    def dijkstra_without_diagonals(self, start, end):
        moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Including diagonals
        rows, cols = self.grid.width, self.grid.height
        min_cost = [[float('inf')] * cols for _ in range(rows)]
        min_cost[start[0]][start[1]] = 0
        predecessor = [[None] * cols for _ in range(rows)]
        queue = [(0, start)]

        while queue:
            current_cost, (x, y) = heapq.heappop(queue)

            if (x, y) == end:
                break

            for dx, dy in moves:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols:
                    next_cost = current_cost + 1
                    if next_cost < min_cost[nx][ny]:
                        min_cost[nx][ny] = next_cost
                        predecessor[nx][ny] = (x, y)
                        heapq.heappush(queue, (next_cost, (nx, ny)))

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
    def calculate_single_mode_time_and_price(self, payment_scheme, origin, destination, unit_price, unit_speed):
        # Find the shortest route using dijkstra_without_diagonals
        shortest_route = self.dijkstra_without_diagonals(origin, destination)
        
        # Calculate the total length of the shortest route
        total_length = self.calculate_total_route_length(shortest_route)
        
        if payment_scheme == 'PAYG':
            # Calculate the dynamic PAYG surcharge
            PAYG_surcharge_percentage = self.calculate_dynamic_MaaS_surcharge(payment_scheme)
            updated_unit_price = unit_price * (1 + PAYG_surcharge_percentage)
            total_price = updated_unit_price * total_length
            MaaS_surcharge = (updated_unit_price - unit_price) * total_length
        elif payment_scheme == 'subscription':
            # Calculate the fixed reduced surcharge for subscription
            subscription_surcharge_percentage = self.calculate_dynamic_MaaS_surcharge(payment_scheme)
            updated_unit_price = unit_price * (1 + subscription_surcharge_percentage)
            total_price = updated_unit_price * total_length
            MaaS_surcharge = (updated_unit_price - unit_price) * total_length
        else:
            # Default case (if payment scheme is not recognized)
            total_price = unit_price * total_length
            MaaS_surcharge = 0

        # Calculate the total time
        total_time = total_length / unit_speed 
        
        return shortest_route, total_time, total_price, MaaS_surcharge

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
        coefficients = DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS
        S_base = coefficients['S_base']  # Base surcharge (10%)
        alpha = coefficients['alpha']    # Sensitivity coefficient
        delta = coefficients['delta']    # Reduction factor for subscription model

        # Collect current usage data
        current_usage = self.get_current_usage()
        system_capacity = num_commuters  # Imported from MobilityModel

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
            print("Optimal path is empty, returning empty itinerary.")  # Debug print
            return detailed_itinerary

        if len(optimal_path) == 1:
            print(f"Optimal path has only one station: {optimal_path}. No public transport needed.")  # Debug print
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
        # if detailed_itinerary[0][0] == 'to station':
        #     print("build_detailed_itinerary test 4")
        #     detailed_itinerary[0] = ('to station', [origin_point, detailed_itinerary[1][2][0]])
        #     print(f"Updated first segment to station: {detailed_itinerary[0]}")  # Debug print

        # if detailed_itinerary[-1][0] == 'to destination':
        #     print(f"build_detailed_itinerary test 5: detailed_itinerary is {detailed_itinerary}")
        #     detailed_itinerary[-1] = ('to destination', [detailed_itinerary[-2][2][-1], destination_point])
        #     print(f"Updated last segment to destination: {detailed_itinerary[-1]}")  # Debug print

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

                walking_route = self.dijkstra_without_diagonals(segment[1][0], get_on_station_coordinates)
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

    def generate_travel_options(self, payment_scheme, request_id, start_time, origin, destination):
        
        travel_options = {'request_id': request_id}

        # Calculate single mode options
        single_modes = ['walk', 'bike', 'car']
        for mode in single_modes:
            
            if mode == 'walk':
                unit_price = 0
                unit_speed = self.service_provider_agent.get_travel_speed(mode, start_time)
                
                shortest_route, total_time, total_price, single_MaaS_surcharge = self.calculate_single_mode_time_and_price(payment_scheme, origin, destination, unit_price, unit_speed)
                availability = float('inf')  # Walking always available
                travel_options[f'{mode}_route'] = shortest_route
                travel_options[f'{mode}_time'] = total_time
                travel_options[f'{mode}_price'] = total_price
                travel_options[f'{mode}_availability'] = availability
                travel_options[f'{mode}_walk_MaaS_surcharge'] = single_MaaS_surcharge
            else:
                price_dict = self.service_provider_agent.get_shared_service_price(mode, start_time)
                if price_dict is None:
                    print(f"No price information available for mode: {mode}")
                    continue
                for company_name, unit_price in price_dict.items():
                    
                    unit_speed = self.service_provider_agent.get_travel_speed(mode, start_time)
                    
                    shortest_route, total_time, total_price, single_MaaS_surcharge = self.calculate_single_mode_time_and_price(payment_scheme, origin, destination, unit_price, unit_speed)
                    
                   
                    availability = self.service_provider_agent.check_shared_availability(company_name, start_time)
                    travel_options[f'{mode}_{company_name}_route'] = shortest_route
                    travel_options[f'{mode}_{company_name}_time'] = total_time
                    travel_options[f'{mode}_{company_name}_price'] = total_price
                    travel_options[f'{mode}_{company_name}_availability'] = availability
                    travel_options[f'{mode}_{company_name}_MaaS_surcharge'] = single_MaaS_surcharge
        # Calculate public transport options
        optimal_path = self.find_optimal_route(origin, destination)

        detailed_itinerary = self.build_detailed_itinerary(optimal_path, origin, destination)
        walking_speed = self.service_provider_agent.get_travel_speed('walk', start_time)
        bus_stop_speed = self.service_provider_agent.get_travel_speed('bus', start_time)
        train_stop_speed = self.service_provider_agent.get_travel_speed('train', start_time)
        bus_stop_price = self.service_provider_agent.get_public_service_price('bus', start_time)
        train_stop_price = self.service_provider_agent.get_public_service_price('train', start_time)

        public_time, public_price, public_MaaS_surcharge = self.calculate_total_time_and_price_public(
            payment_scheme, detailed_itinerary, walking_speed, bus_stop_speed, train_stop_speed, bus_stop_price, train_stop_price
        )

        travel_options['public_route'] = detailed_itinerary
        travel_options['public_time'] = public_time
        travel_options['public_price'] = public_price
        travel_options['public_availability'] = float('inf')  # Assuming always available
        travel_options['public_MaaS_surcharge'] = public_MaaS_surcharge
        # Store travel_options in the corresponding request
        for commuter in self.commuter_agents:
            if request_id in commuter.requests:
                commuter.requests[request_id]['travel_options'] = travel_options
        print(travel_options)
        return travel_options

    
    #########################################################################################################
    ################################### Confirm Booking Options #############################################
    ########################################################################################################
    def book_service(self, request_id, ranked_options, current_step, availability_dict):
        for commuter in self.commuter_agents:
            if request_id in commuter.requests:
                for option in ranked_options:
                    try:
                        cost, mode_company, route, time = option  # Extract 'time' from option
                    except ValueError as e:
                        print(f"Error unpacking option: {option}, {e}")
                        continue

                    availability_key = mode_company.replace('route', 'availability')
                    price_key = mode_company.replace('route', 'price')
                    
                    try:
                        availability = commuter.requests[request_id]['travel_options'][availability_key]
                        price = commuter.requests[request_id]['travel_options'][price_key]
                    except KeyError as e:
                        print(f"Error accessing travel options: {e}, request_id: {request_id}, travel_options: {commuter.requests[request_id]['travel_options']}")
                        continue

                    if mode_company == 'public_route':
                        all_steps_available = True
                        if self.confirm_booking(commuter, request_id, mode_company, route, price, time):  # Pass 'time'
                            return True, availability_dict
                    else:
                        chosen_company = mode_company.split('_')[1]
                        start_time = commuter.requests[request_id]['start_time']
                        duration = round(time)
                        end_check_step = min(current_step + 5, start_time + duration)
                        
                        all_steps_available = True
                        for step in range(start_time, end_check_step + 1):
                            step_key = step - current_step
                            if step_key < 0 or step_key > 5:
                                all_steps_available = False
                                break
                            availability_check_key = f'{chosen_company}_{step_key}'
                            print(f"Checking availability for '{availability_check_key}': {availability_dict.get(availability_check_key)}")
                            if availability_dict.get(availability_check_key, 0) < 1:
                                all_steps_available = False
                                break

                        if all_steps_available:
                            if self.confirm_booking(commuter, request_id, mode_company, route, price, time):  # Pass 'time'
                                if 'car' in mode_company or 'bike' in mode_company:
                                    for step in range(start_time, end_check_step + 1):
                                        step_key = step - current_step
                                        availability_deduct_key = f'{chosen_company}_{step_key}'
                                        availability_dict[availability_deduct_key] -= 1
                                    self.service_provider_agent.record_booking_log(
                                        commuter.unique_id,
                                        request_id,
                                        mode_company,
                                        start_time,
                                        duration,
                                        list(range(start_time, start_time + duration)),
                                        route
                                    )
                                    print(f"Booking recorded for request {request_id}")
                                    return True, availability_dict
                            else:
                                print(f"Booking confirmation failed for request {request_id}")

        print(f"Failed to book any service for request {request_id}")
        return False, availability_dict

    def confirm_booking(self, commuter, request_id, mode, route, price, time):
        selected_route = {
            'mode': mode,
            'route': route,
            'price': price,
            'time': time  # Add the 'time' key here
        }
        print(f"Confirming booking with selected route: {selected_route}")
        if commuter.accept_service(request_id, selected_route):
            self.record_service_booking(commuter, request_id, selected_route)
            return True
        else:
            return False

    def record_service_booking(self, commuter, request_id, selected_route):
        payment_scheme = commuter.payment_scheme
        start_time = commuter.requests[request_id]['start_time']
        origin_coordinates = commuter.requests[request_id]['origin']
        destination_coordinates = commuter.requests[request_id]['destination']
        record_company_name = 'public' if 'public' in selected_route['mode'] else selected_route['mode'].split('_')[1]
        route_details = selected_route['route']
        total_price = selected_route['price']
        MaaS_surcharge = commuter.requests[request_id]['travel_options'][f"{selected_route['mode'].replace('route', 'MaaS_surcharge')}"]
        total_time = selected_route['time']  # Assuming the total time is stored in the selected_route dictionary

        new_booking = ServiceBookingLog(
            commuter_id=commuter.unique_id,
            payment_scheme=payment_scheme,
            request_id=str(request_id),
            start_time=start_time,
            record_company_name=record_company_name,
            route_details=route_details,
            total_price=total_price,
            MaaS_surcharge=MaaS_surcharge,
            total_time=total_time,  # Include total time in the booking
            origin_coordinates=origin_coordinates,  # Include origin coordinates
            destination_coordinates=destination_coordinates  # Include destination coordinates
        )

        try:
            Session = sessionmaker(bind=self.db_engine)
            with Session() as session:
                # Check if a record already exists
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
            print(f"Error recording service booking: {e}")
