from mesa import Agent
import math
import random
import numpy as np
from database_01 import UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS, UTILITY_FUNCTION_BASE_COEFFICIENTS, PENALTY_COEFFICIENTS, AFFORDABILITY_THRESHOLDS, FLEXIBILITY_ADJUSTMENTS, VALUE_OF_TIME
class Commuter(Agent):
    def __init__(self, unique_id, model, commuter_location, age, income_level, has_disability,
                 tech_access=True, health_status='good', payment_scheme='PAYG'):
        
        super().__init__(unique_id, model)
        self.requests = {}
        self.services_owned = {}
        self.location = commuter_location # Reflect current possition
        self.income_level = income_level # Enfect the perferred mode choice
        self.preferred_mode_id = None  # Default preferred mode
        # Ethical Consideration
        self.age = age
        self.has_disability = has_disability
        self.tech_access = tech_access
        self.health_status = health_status
        self.current_mode = None  # Initialize current_mode as None
        self.payment_scheme = payment_scheme  # New attribute to determine payment scheme

    ################################################################################################################
    ################################## create_request and related functions 13/05/24 ###############################
    ################################################################################################################
    def create_request(self, request_id, origin, destination, start_time, travel_purpose='work'):
        request = {
            'request_id': request_id,
            'commuter_id': self.unique_id,
            'origin': origin,
            'destination': destination,
            'start_time': start_time,
            'flexible_time': self.determine_schedule_flexibility(travel_purpose),
            'requirements': self.get_personal_requirements(),
            'status': 'active',
            'travel_options': None  # Placeholder for travel options
        }
        self.add_request(request)
        return request
    def determine_schedule_flexibility(self, purpose_of_travel):
        """
        Determines the schedule flexibility based on the purpose of travel.
        
        :param purpose_of_travel: str, the purpose of the travel such as 'work', 'school', 'shopping', 'leisure', 'medical'
        :return: str, the level of schedule flexibility ('high', 'medium', 'low')
        """
        flexibility = {
            'work': 'median',
            'school': 'median',
            'shopping': 'high',
            'medical': 'low',
            'trip': 'high'
        }
        
        # Return the flexibility level for the given purpose, default to 'medium' if not found
        return flexibility.get(purpose_of_travel, 'medium')
    
    def get_personal_requirements(self):
        """
        Returns a dictionary of requirements based on the commuter's personal attributes.
        """
        requirements = {
            'wheelchair_accessible': self.has_disability,
            'priority_seating': self.has_disability or self.age >= 65,
            'additional_time_needed': self.has_disability or self.age >= 65,
            'assistance_required': self.has_disability,
            'tech_support': 'SMS' if not self.tech_access else 'App',
            'health_accommodations': self.health_status != 'good'
        }
        return requirements
    
    def add_request(self, request):
        """
        Add a new request to the commuter's list of requests. If the commuter does not have an entry in
        the requests dictionary, create a new list for this commuter.
        
        Args:
            request (dict): A dictionary containing the details of the travel request.
        
        Raises:
            ValueError: If the request is not formatted correctly.
        """
        # Validate the request input (basic validation, could be expanded as needed)
        if not isinstance(request, dict) or 'origin' not in request or 'destination' not in request:
            raise ValueError("Invalid request format. Please provide a dictionary with 'origin' and 'destination'.")

        request_id = request['request_id']
        self.requests[request_id] = request


    ########################################################################################################
    ############################################## Accept Service ############################################
    #########################################################################################################


    def rank_service_options(self, travel_options, request_id):
        """
        Rank service options based on the probability of mode choice using a logit model.
        
        Parameters:
        travel_options (dict): A dictionary containing different travel options with their details.
        request_id (int): The request ID for which the ranking is being calculated.
        
        Returns:
        list: A list of tuples containing the rank, mode, route, and time, sorted by the generated rank.
        """
        # Calculate the probabilities for each travel option
        probabilities = self.calculate_mode_choice_probabilities(travel_options, request_id)

        # Sort the options based on their probabilities
        sorted_options = sorted(probabilities, key=lambda x: x[0], reverse=True)

        # Generate rankings based on sorted probabilities
        ranked_options = []
        for prob, mode, route, time in sorted_options:
            # Use the probability to rank the option, higher probability means higher rank
            rank = random.uniform(0, prob)
            ranked_options.append((rank, mode, route, time))

        # Sort the ranked options based on the generated rank
        ranked_options.sort(reverse=True, key=lambda x: x[0])
        print("show the ranked options")
        print(f"For a commuter that has the income level of {self.income_level}")
        print(f"This is the ranked options {ranked_options}")
        return ranked_options

    def calculate_mode_choice_probabilities(self, travel_options, request_id):
        utilities = {}
        for mode, details in travel_options.items():
            if 'route' in mode:
                price_key = mode.replace('route', 'price')
                time_key = mode.replace('route', 'time')

                try:
                    price = travel_options[price_key]
                    time = travel_options[time_key]
                    route = travel_options[mode]

                    # Calculate utility
                    utility = self.calculate_generalized_utility(price, time, mode, request_id)
                    utilities[mode] = utility

                except KeyError as e:
                    print(f"KeyError: {e}")

        # Calculate the probability of each mode
        sum_exp_utilities = sum(np.exp(utility) for utility in utilities.values())
        probabilities = {mode: np.exp(utility) / sum_exp_utilities for mode, utility in utilities.items()}

        # Prepare the output list
        probability_for_options = []
        for mode, probability in probabilities.items():
            route = travel_options[mode]
            time = travel_options[mode.replace('route', 'time')]
            probability_for_options.append((probability, mode, route, time))

        return probability_for_options
    
    def calculate_penalty(self, gc, mode=''):
        penalty = 0
        if self.has_disability and mode in ['bike', 'walk']:
            penalty += PENALTY_COEFFICIENTS['disability_bike_walk'] * gc
        if self.age is not None and (self.age >= 65 or self.health_status != 'good'):
            if mode in ['bike', 'walk']:
                penalty += PENALTY_COEFFICIENTS['age_health_bike_walk'] * gc
        if not self.tech_access and mode in ['car', 'bike']:
            penalty += PENALTY_COEFFICIENTS['no_tech_access_car_bike'] * gc
        return penalty



    def calculate_affordability_adjustment(self, price):
        affordability_threshold = AFFORDABILITY_THRESHOLDS.get(self.income_level, AFFORDABILITY_THRESHOLDS['default'])
        if price > affordability_threshold:
            return 2 * price  # Significant adjustment to decrease utility
        else:
            return 0


    def get_value_of_time(self, request_id):
        flexibility = self.requests[request_id]['flexible_time']
        base_value_of_time = VALUE_OF_TIME.get(self.income_level, VALUE_OF_TIME['low'])  # Default to low income value if not found

        flexibility_adjustment = FLEXIBILITY_ADJUSTMENTS.get(flexibility, FLEXIBILITY_ADJUSTMENTS['medium'])  # Default to no adjustment if not found

        return base_value_of_time * flexibility_adjustment

    def calculate_generalized_utility(self, price, time, mode, request_id):
        value_of_time = self.get_value_of_time(request_id)
        generalized_cost = price + (time / 6 * value_of_time)

        penalty = self.calculate_penalty(generalized_cost, mode)
        affordability_adjustment = self.calculate_affordability_adjustment(generalized_cost)

        # Set base coefficients
        coefficients = UTILITY_FUNCTION_BASE_COEFFICIENTS.copy()

        # Adjust coefficients for high-income car users
        if self.income_level == 'high' and mode == 'car':
            coefficients.update(UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS)

        U_j = (
            coefficients['alpha'] +
            (coefficients['beta_C'] * price) +
            (coefficients['beta_T'] * (time / 6 * value_of_time)) +
            (coefficients['beta_W'] * penalty) +
            (coefficients['beta_A'] * affordability_adjustment)
        )

        return U_j


  
    def accept_service(self, request_id, selected_route):
        request = self.requests.get(request_id)
        if request:
            request['selected_route'] = selected_route
            request['status'] = 'Service Selected'
            self.services_owned[request_id] = request
            return True
        else:
            return False

    ########################################################################################################
    ############################################## Update Location ############################################
    def update_location(self):
        """
        Updates the commuter's current location at each step (tick) of the simulation.
        """
        for request_id, request in self.requests.items():
            if request['status'] == 'Service Selected' and request['selected_route']:
                start_time = request['start_time']
                
                if start_time <= self.model.schedule.time:
                    mode = request['selected_route']['mode']
                    self.current_mode = mode.split('_')[0]  # Update current mode

                    if mode != 'public_route':
                        # Handle single mode
                        base_mode = mode.split('_')[0]
                        detailed_itinerary = request['selected_route']['route']
                        travel_speed = self.model.service_provider_agent.get_travel_speed(base_mode, start_time)

                        self.move_along_route_single_mode(detailed_itinerary, travel_speed)
                    elif mode == 'public_route':
                        # Handle public route
                        detailed_itinerary = request['selected_route']['route']
                        self.handle_public_route(detailed_itinerary, start_time, request)
                    else:
                        print("Something wrong with the mode in the request for update location")
                else:
                    #print(f"Commuter {self.unique_id} checking start time: {start_time} <= {self.model.schedule.time}, not in transit")
                    pass

    def handle_public_route(self, detailed_itinerary, start_time, request):
        current_time = self.model.schedule.time
        elapsed_time = current_time - start_time
        current_segment_start_time = start_time
        print("Trouble shooting... for handle_public_route")
        print("The detailed intinary is ")
        print(f"{detailed_itinerary}")
        print("The request list for this commuter is")
        print(f"{self.requests}")
        for segment in detailed_itinerary:
            if segment[0] == 'to station':
                self.current_mode = 'walk'  # Set current mode
                travel_speed = self.model.service_provider_agent.get_travel_speed('walk', current_time)
                get_on_station_coordinates = self.get_station_coordinates(segment[1][1])
                walking_route = self.model.maas_agent.dijkstra_without_diagonals(segment[1][0], get_on_station_coordinates)
                time_to_station = len(walking_route) * travel_speed
                if elapsed_time <= time_to_station:
                    self.move_along_route(walking_route, travel_speed, elapsed_time)
                    return
                else:
                    elapsed_time -= time_to_station
                    current_segment_start_time += time_to_station

            elif segment[0] in ['bus', 'train']:
                self.current_mode = segment[0]  # Set current mode
                travel_speed = self.model.service_provider_agent.get_travel_speed(segment[0], current_time)
                route_list = self.model.maas_agent.routes[segment[0]][segment[1]]
                stops = segment[2]
                get_on_index = route_list.index(stops[0])
                get_off_index = route_list.index(stops[1])
                num_stops = get_off_index - get_on_index
                time_on_transit = num_stops * travel_speed
                if elapsed_time <= time_on_transit:
                    current_stop_index = get_on_index + (elapsed_time // travel_speed)
                    self.move_and_update_location(self.get_station_coordinates(route_list[int(current_stop_index)]))
                    return
                else:
                    elapsed_time -= time_on_transit
                    current_segment_start_time += time_on_transit

            elif segment[0] == 'transfer':
                self.current_mode = 'walk'  # Set current mode
                transfer_time = self.model.maas_agent.transfers[tuple(segment[1])]
                if elapsed_time <= transfer_time:
                    # Stay at transfer location until transfer is complete
                    self.move_and_update_location(self.get_station_coordinates(segment[1][0]))
                    return
                else:
                    elapsed_time -= transfer_time
                    current_segment_start_time += transfer_time

            elif segment[0] == 'to destination':
                self.current_mode = 'walk'  # Set current mode
                travel_speed = self.model.service_provider_agent.get_travel_speed('walk', current_time)
                get_off_station_coordinates = self.get_station_coordinates(segment[1][0])
                destination_coordinates = segment[1][1]
                walking_route = self.model.maas_agent.dijkstra_without_diagonals(get_off_station_coordinates, destination_coordinates)
                time_to_destination = len(walking_route) * travel_speed
                if elapsed_time <= time_to_destination:
                    self.move_along_route(walking_route, travel_speed, elapsed_time)
                    return
                else:
                    elapsed_time -= time_to_destination
                    current_segment_start_time += time_to_destination

        # If we finish the loop and haven't returned, the commuter has arrived at the destination
        request['status'] = 'finished'
        # print("Trouble shooting... for handle_public_route")
        # print("The detailed intinary is ")
        # print(f"{detailed_itinerary}")
        # print("The final destination for the detailed_itinerary is ")
        # print(f"{detailed_itinerary[-1][-1][-1]}")
        # self.location = detailed_itinerary[-1][-1][-1]  # Set location to the final destination

        self.current_mode = None  # Reset mode

    def move_along_route_single_mode(self, route, travel_speed):
        """
        Moves the commuter along the route according to the travel speed.
        """
        current_time = self.model.schedule.time
        start_time = self.requests[next(iter(self.requests))]['start_time']  # Assuming there's only one request for simplification
        distance_traveled = (current_time - start_time) * travel_speed

        total_route_distance = len(route) - 1

        if distance_traveled < total_route_distance:
            current_position_index = int(distance_traveled)
            current_position = route[current_position_index]
        else:
            current_position = route[-1]
            self.requests[next(iter(self.requests))]['status'] = 'finished'
            self.current_mode = None  # Reset mode

        self.move_and_update_location(current_position)

    def move_along_route(self, route, travel_speed, elapsed_time):
        distance_to_move = elapsed_time // travel_speed
        current_position = self.location

        for i in range(len(route) - 1):
            segment_start = route[i]
            segment_end = route[i + 1]
            segment_distance = self.calculate_distance(segment_start, segment_end)

            if distance_to_move <= segment_distance:
                ratio = distance_to_move / segment_distance
                new_x = segment_start[0] + ratio * (segment_end[0] - segment_start[0])
                new_y = segment_start[1] + ratio * (segment_end[1] - segment_start[1])
                current_position = (int(round(new_x)), int(round(new_y)))
                break
            else:
                distance_to_move -= segment_distance
                current_position = segment_end

        self.move_and_update_location(current_position)

    def calculate_distance(self, point1, point2):
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def move_and_update_location(self, next_position):
        if isinstance(next_position, tuple) and len(next_position) == 2 and isinstance(next_position[0], int) and isinstance(next_position[1], int):
            self.model.grid.move_agent(self, next_position)
            self.location = next_position
        else:
            print(f"Error: next_position {next_position} is not a valid coordinate")

    def get_station_coordinates(self, station_name):
        if station_name.startswith('T'):
            return self.model.maas_agent.stations['train'][station_name]
        elif station_name.startswith('B'):
            return self.model.maas_agent.stations['bus'][station_name]
        else:
            raise ValueError(f"Station {station_name} is not recognized as a train or bus station.")

    def check_travel_status(self):
        """
        Purpose: At each simulation step, checks the commuter's travel status, including whether the travel request has been fulfilled and if the commuter has arrived at their destination.
        Inputs: None.
        Outputs: Updated travel status (in transit, arrived, etc.).
        Process:
        If in transit, determine if the current step completes the journey.
        Update the agent's state to "arrived" if the destination is reached.
        """
        for request_id, request in list(self.requests.items()):
            if request['status'] == 'Service Selected' and self.location == request['destination']:
                request['status'] = 'finished'  # Update status to finished when destination is reached
            #print(f"Commuter {self.unique_id} request {request_id} status: {request['status']}")
