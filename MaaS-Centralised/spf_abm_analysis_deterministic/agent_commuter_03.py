from mesa import Agent
import math
import random
import numpy as np
from agent_service_provider_initialisation_03 import ServiceBookingLog
# from database_01 import ASC_VALUES, UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS, UTILITY_FUNCTION_BASE_COEFFICIENTS, PENALTY_COEFFICIENTS, AFFORDABILITY_THRESHOLDS, FLEXIBILITY_ADJUSTMENTS, VALUE_OF_TIME
class Commuter(Agent):
    def __init__(self, unique_id, model, commuter_location, age, income_level, has_disability,
                 tech_access, health_status, payment_scheme,\
                      ASC_VALUES, UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS, \
                        UTILITY_FUNCTION_BASE_COEFFICIENTS, PENALTY_COEFFICIENTS, \
                        AFFORDABILITY_THRESHOLDS, FLEXIBILITY_ADJUSTMENTS, VALUE_OF_TIME, subsidy_dataset):
        
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
        self.asc_values = ASC_VALUES
        self.utility_function_high_income_car_coefficients = UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS
        self.utility_function_base_coefficients = UTILITY_FUNCTION_BASE_COEFFICIENTS
        self.penalty_coefficients = PENALTY_COEFFICIENTS
        self.affordability_thresholds = AFFORDABILITY_THRESHOLDS
        self.flexibility_adjustments = FLEXIBILITY_ADJUSTMENTS
        self.value_of_time = VALUE_OF_TIME
        self.subsidy_dataset = subsidy_dataset
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

    def rank_service_options(self, travel_options_without_MaaS, travel_options_with_MaaS, request_id):
        """
        Rank service options based on the probability of mode choice using a logit model.

        Parameters:
        travel_options_without_MaaS (dict): A dictionary containing different traditional travel options with their details.
        travel_options_with_MaaS (list): A list containing MaaS bundle travel options.
        request_id (int): The request ID for which the ranking is being calculated.

        Returns:
        list: A list of tuples containing the rank, mode, route, and time, sorted by the generated rank.
        """
        # Combine all travel options into a unified structure
        combined_travel_options = {}

        # Add traditional travel options
        for mode in travel_options_without_MaaS:
            if 'route' in mode:
                combined_travel_options[mode] = {
                    'price': travel_options_without_MaaS.get(mode.replace('route', 'price')),
                    'time': travel_options_without_MaaS.get(mode.replace('route', 'time')),
                    'route': travel_options_without_MaaS.get(mode),
                    'mode': mode.split('_')[0]  # Extract mode from the key
                }

        # Add MaaS bundle travel options with unique identifiers (0-24)
        for idx, maas_option in enumerate(travel_options_with_MaaS):
            maas_key = f"maas_{idx}"  # Use the index as part of the key for uniqueness
            
            # Extract information from the MaaS option
            to_station_mode = maas_option[2][0][0]  # e.g., 'walk', 'bike', 'car'
            to_destination_mode = maas_option[2][1][0]  # e.g., 'walk', 'bike', 'car'
            public_transport_mode = ' + '.join([seg[0] for seg in maas_option[1] if seg[0] in ['bus', 'train']])
            
            # Combine the modes into a single string
            combined_mode = f"{to_station_mode} + {public_transport_mode} + {to_destination_mode}"
            
            combined_travel_options[maas_key] = {
                'price': maas_option[2][3],  # Final total price after surcharge
                'time': maas_option[2][2],  # Total time
                'route': maas_option[1],  # The detailed itinerary
                'mode': combined_mode  # The combined mode string
            }

        # Calculate the probabilities for each travel option

        probabilities = self.calculate_mode_choice_probabilities(combined_travel_options, request_id)

        if getattr(self.model, "deterministic_mode", False):
            ranked_options = [
                (float(prob), mode, route, time, subsidy)
                for prob, mode, route, time, subsidy in probabilities
            ]
            ranked_options.sort(
                key=lambda x: (
                    -x[0],
                    float(x[3]) if x[3] is not None else float("inf"),
                    str(x[1]),
                )
            )
            return ranked_options[:5]

        # Generate rankings based on calculated probabilities
        ranked_options = []
        for prob, mode, route, time, subsidy in probabilities:
            # Use the probability to rank the option, higher probability means higher rank
            rank = random.uniform(0, prob)
            ranked_options.append((rank, mode, route, time, subsidy))

        # Sort the ranked options based on generated rank, introducing randomness when prices are the same
        ranked_options.sort(key=lambda x: (x[0], random.random()), reverse=True)

        # Return only the top 5 ranked options
        top_5_ranked_options = ranked_options[:5]

        return top_5_ranked_options

    def calculate_mode_choice_probabilities(self, travel_options, request_id):
        utilities = {}
        subsidies = {}  # Dictionary to store subsidies for each mode
        
        for mode, details in travel_options.items():
            try:
                price = details['price']
                time = details['time']
                route = details['route']

                # Calculate utility and subsidy
                # print("calling calculate_generalized_utility for calculate mode choice probabilities")
                utility, subsidy_amount = self.calculate_generalized_utility(price, time, mode, request_id)
                # print(f"the utility is {utility} for mode {mode}")
                utilities[mode] = utility
                subsidies[mode] = subsidy_amount  # Save the subsidy amount for the mode

            except KeyError as e:
                print(f"KeyError: {e}")

        # Find maximum utility for numerical stability
        if utilities:
            max_utility = max(utilities.values())
            
            # Normalize utilities by subtracting max value (prevents overflow)
            normalized_utilities = {mode: utility - max_utility for mode, utility in utilities.items()}
            
            # Calculate the sum of exp(normalized_utilities)
            sum_exp_utilities = sum(np.exp(utility) for utility in normalized_utilities.values())
            
            # Calculate probabilities using normalized utilities
            probabilities = {mode: np.exp(utility) / sum_exp_utilities 
                            for mode, utility in normalized_utilities.items()}
            
            # Prepare the output list
            probability_for_options = []
            for mode, probability in probabilities.items():
                route = travel_options[mode]['route']
                time = travel_options[mode]['time']
                subsidy = subsidies[mode]
                probability_for_options.append((probability, mode, route, time, subsidy))
            
            return probability_for_options
        
        return []  # Return empty list if no utilities

        
    def calculate_penalty(self, gc, mode=''):
        penalty = 0
        if self.has_disability and mode in ['bike', 'walk']:
            penalty += self.penalty_coefficients['disability_bike_walk'] * gc
        if self.age is not None and (self.age >= 65 or self.health_status != 'good'):
            if mode in ['bike', 'walk']:
                penalty += self.penalty_coefficients['age_health_bike_walk'] * gc
        if not self.tech_access and mode in ['car', 'bike']:
            penalty += self.penalty_coefficients['no_tech_access_car_bike'] * gc
        return penalty



    def calculate_affordability_adjustment(self, price):
        affordability_threshold = self.affordability_thresholds.get(self.income_level, self.affordability_thresholds['default'])
        if price > affordability_threshold:
            return 2 * price  # Significant adjustment to decrease utility
        else:
            return 0


    def get_value_of_time(self, request_id):
        flexibility = self.requests[request_id]['flexible_time']
        base_value_of_time = self.value_of_time.get(self.income_level,self.value_of_time['low'])  # Default to low income value if not found

        flexibility_adjustment = self.flexibility_adjustments.get(flexibility, self.flexibility_adjustments['medium'])  # Default to no adjustment if not found

        return base_value_of_time * flexibility_adjustment


    def calculate_generalized_utility(self, price, time, mode, request_id):
        """
        Calculate utility with income-specific coefficients and mode-specific factors.
        
        Args:
            price: The cost of using the transportation mode
            time: The duration of the trip (in minutes)
            mode: The transportation mode (e.g., 'car', 'bike', 'public', 'MaaS_Bundle')
            request_id: The commuter's request ID for referencing specific details
        """
        # Get base value of time
        value_of_time = self.get_value_of_time(request_id)
        VOT_per_ten_min = value_of_time/6
        
        # Get base coefficients
        base_coefficients = self.utility_function_base_coefficients.copy()
        
        # Adjust coefficients based on income level
        if self.income_level == 'high':
            # High income: Less price sensitive, more time sensitive
            beta_C = base_coefficients['beta_C'] * 0.9  # Reduce price sensitivity
            beta_T = base_coefficients['beta_T'] * 1.2 # Increase time sensitivity
            # comfort_multiplier = 1.5  # Higher value for comfort
        elif self.income_level == 'middle':
            # Middle income: Moderate sensitivity to both
            beta_C = base_coefficients['beta_C'] * 1
            beta_T = base_coefficients['beta_T'] * 1
            # comfort_multiplier = 1.25
        else:  # low income
            # Low income: More price sensitive, less time sensitive
            beta_C = base_coefficients['beta_C'] * 1.5  # Increase price sensitivity
            beta_T = base_coefficients['beta_T'] * 0.85 # Reduce time sensitivity
            # comfort_multiplier = 1.0

        # Get mode-specific ASC and apply income-specific adjustments
        ASC_j = self.set_ASC_values(mode)
        
        # Calculate subsidy if applicable
        income_level = self.income_level
        subsidy_dataset = self.subsidy_dataset.copy()
        subsidy_key = self.map_mode_to_subsidy_key(mode)
        
        if subsidy_key:
            subsidy_percentage = subsidy_dataset.get(income_level, {}).get(subsidy_key, 0)
            subsidy_amount = price * subsidy_percentage
            actual_subsidy = self.model.maas_agent.check_subsidy_availability(subsidy_amount)
            price_after_subsidy = price - actual_subsidy
        else:
            price_after_subsidy = price
            actual_subsidy = 0

        # Calculate final utility
        U_j = (
            ASC_j +  # Mode-specific constant adjusted by comfort
            (beta_C * price_after_subsidy) +  # Income-adjusted price sensitivity
            (beta_T * VOT_per_ten_min * time)  # Income-adjusted time sensitivity
        )

        return U_j, actual_subsidy
    
    def map_mode_to_subsidy_key(self, mode):
            if 'maas' in mode.lower():
                return 'MaaS_Bundle'
            elif 'bike' in mode.lower():
                return 'bike'
            elif 'car' in mode.lower() or 'uber' in mode.lower():  # Assuming 'car' includes Uber-like services
                return 'car'
            elif 'public' in mode.lower() or 'bus' in mode.lower() or 'train' in mode.lower():
                return 'public'  # Add this line to map public transport
            else:
                return None
    
    def set_ASC_values(self, mode):
            """
            Determine the ASC (Alternative-Specific Constant) based on the mode of transport.
            Adjusts the control factor for each mode to influence the mode share.
            """
            # Find the ASC value based on the mode prefix, default to 'default' if not found
            for mode_key in self.asc_values.keys():
                if mode.startswith(mode_key):
                    return self.asc_values[mode_key]
            
            # If no match, return the default ASC value
            return self.asc_values['default']        
    def accept_service(self, request_id):

        request = self.requests.get(request_id)
        
        if request:
    
            request['status'] = 'Service Selected'
            self.services_owned[request_id] = request
            
            return True
        else:
    
            return False

    def accept_service_non_maas(self, request_id,selected_route):
        request = self.requests.get(request_id)
            
        if request:
            request['selected_route'] = selected_route
            request['status'] = 'Service Selected'
            self.services_owned[request_id] = request
        
            return True
        else:
            
            return False
        
    def update_trip_status_in_database(self, request_id, new_status):
        """Update the status of a trip in the ServiceBookingLog database."""
        try:
            from sqlalchemy.orm import sessionmaker
            Session = sessionmaker(bind=self.model.db_engine)
            with Session() as session:
                booking = session.query(ServiceBookingLog).filter_by(
                    commuter_id=self.unique_id,
                    request_id=str(request_id)
                ).first()
                
                if booking:
                    booking.status = new_status
                    session.commit()
                    print(f"Updated status of request {request_id} to {new_status} in database")
                else:
                    print(f"Could not find booking for request {request_id} in database")
                    
        except Exception as e:
            print(f"Error updating trip status in database: {e}")

    ########################################################################################################
    ############################################## Update Location ############################################
    # def update_location(self):
    #     """
    #     Updates the commuter's current location at each step (tick) of the simulation.
    #     """
    #     for request_id, request in self.requests.items():
    #         if request['status'] == 'Service Selected' and request['selected_route']:
    #             start_time = request['start_time']

    #             if start_time <= self.model.schedule.time:
    #                 mode = request['selected_route']['mode']
    #                 self.current_mode = mode.split('_')[0]  # Update current mode

    #                 if mode == 'MaaS_Bundle':
    #                     # Handle MaaS bundle
    #                     detailed_itinerary = request['selected_route']['route']
    #                     self.handle_maas_bundle(detailed_itinerary, start_time, request)
    #                 elif mode != 'public_route':
    #                     # Handle single mode
    #                     base_mode = mode.split('_')[0]
    #                     detailed_itinerary = request['selected_route']['route']
    #                     travel_speed = self.model.service_provider_agent.get_travel_speed(base_mode, start_time)

    #                     self.move_along_route_single_mode(detailed_itinerary, travel_speed)
    #                 elif mode == 'public_route':
    #                     # Handle public route
    #                     detailed_itinerary = request['selected_route']['route']
    #                     self.handle_public_route(detailed_itinerary, start_time, request)
    #                 else:
    #                     print("Something wrong with the mode in the request for update location")
    #             else:
    #                 pass

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

                    if mode == 'MaaS_Bundle':
                        # Handle MaaS bundle
                        detailed_itinerary = request['selected_route']['route']
                        self.handle_maas_bundle(detailed_itinerary, start_time, request)
                    elif mode != 'public_route':
                        # Handle single mode (bike, walk, car, etc.)
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
                    pass


    def handle_maas_bundle(self, detailed_itinerary, start_time, request):
        current_time = self.model.schedule.time
        elapsed_time = current_time - start_time
        total_time_elapsed = 0
        # current_segment_start_time = start_time
       
        # Ensure 'to_station_info' and 'to_destination_info' exist in the selected route
        to_station_info = request['selected_route']['to_station_info']
        to_destination_info = request['selected_route']['to_destination_info']

        if not to_station_info or not to_destination_info:
            print("[ERROR] Missing required info in MaaS bundle")
            return
        
        for segment in detailed_itinerary:
            segment_type = segment[0]
            if segment_type == 'to station':
                company = to_station_info[0]
                # Handle 'to station' segment using the saved data
                mode = 'bike' if 'Bike' in company else 'car' if 'Uber' in company else 'walk' 

                travel_speed = self.model.service_provider_agent.get_travel_speed(mode, current_time)

                # Use saved detailed route from to_station_info
                route = to_station_info[3]  # Assuming [3] is the saved detailed route
                
                time_to_complete = len(route) / travel_speed
                
                if elapsed_time <= total_time_elapsed + time_to_complete:
                    self.current_mode = mode
                    segment_elapsed_time = elapsed_time - total_time_elapsed
                    self.move_along_route(route, travel_speed, segment_elapsed_time)
                    return
                total_time_elapsed += time_to_complete
            
            elif segment_type in ['bus', 'train']:
                self.current_mode = segment_type
                travel_speed = self.model.service_provider_agent.get_travel_speed(segment_type, current_time)
                route_id = segment[1]  # e.g. 'RB11' or 'RT1'
                stops = segment[2]  # e.g. ['B89', 'B62']
                
                route_list = self.model.maas_agent.routes[segment_type][route_id]
                get_on_index = route_list.index(stops[0])
                get_off_index = route_list.index(stops[1])
                num_stops = abs(get_off_index - get_on_index)
                time_to_complete = num_stops * travel_speed

                if elapsed_time <= total_time_elapsed + time_to_complete:
                    progress = elapsed_time - total_time_elapsed
                    current_stop_index = get_on_index + int(progress // travel_speed)
                    current_stop_index = min(current_stop_index, get_off_index)  # Don't overshoot
                    self.move_and_update_location(self.get_station_coordinates(route_list[current_stop_index]))
                    return
                total_time_elapsed += time_to_complete
                   # Transfers between public transport
            elif segment_type == 'transfer':
                self.current_mode = 'walk'
                transfer_stations = tuple(segment[1])  # e.g. ['B62', 'T1-7']
                time_to_complete = self.model.maas_agent.transfers[transfer_stations]

                if elapsed_time <= total_time_elapsed + time_to_complete:
                    # Stay at transfer start station until transfer completes
                    self.move_and_update_location(self.get_station_coordinates(segment[1][0]))
                    return
                total_time_elapsed += time_to_complete
                   # Last mile (to destination)
            elif segment_type == 'to destination':
                company = to_destination_info[0]
                mode = 'bike' if 'Bike' in company else 'car' if 'Uber' in company else 'walk'
                travel_speed = self.model.service_provider_agent.get_travel_speed(mode, current_time)
                route = to_destination_info[3]  # Detailed route coordinates
                time_to_complete = len(route) / travel_speed

                if elapsed_time <= total_time_elapsed + time_to_complete:
                    self.current_mode = mode
                    segment_elapsed_time = elapsed_time - total_time_elapsed
                    self.move_along_route(route, travel_speed, segment_elapsed_time)
                    return
                total_time_elapsed += time_to_complete

        # Journey complete - move to final destination and finish
        # Journey complete
        if elapsed_time >= total_time_elapsed:
            self.move_and_update_location(request['destination'])
            request['status'] = 'finished'
            
            # Extract request_id from the request dictionary
            request_id = request.get('request_id')
            
            # Add this line to update the database
            self.update_trip_status_in_database(request_id, 'finished')
            self.current_mode = None
   


    def handle_public_route(self, detailed_itinerary, start_time, request):
        """Handle movement for public transport journeys including walking segments"""
        current_time = self.model.schedule.time
        elapsed_time = current_time - start_time
        total_time_elapsed = 0

        for segment in detailed_itinerary:
            segment_type = segment[0]

            # Walk to station
            if segment_type == 'to station':
                self.current_mode = 'walk'
                travel_speed = self.model.service_provider_agent.get_travel_speed('walk', current_time)
                origin = segment[1][0]
                station = segment[1][1]
                get_on_coordinates = self.get_station_coordinates(station)
                walking_route = self.model.maas_agent.dijkstra_with_congestion(origin, get_on_coordinates, 5)
                time_to_complete = len(walking_route) / travel_speed

                if elapsed_time <= total_time_elapsed + time_to_complete:
                    segment_elapsed_time = elapsed_time - total_time_elapsed
                    self.move_along_route(walking_route, travel_speed, segment_elapsed_time)
                    return
                total_time_elapsed += time_to_complete

            # Public transport segments 
            elif segment_type in ['bus', 'train']:
                self.current_mode = segment_type
                travel_speed = self.model.service_provider_agent.get_travel_speed(segment_type, current_time)
                route_id = segment[1]
                stops = segment[2]
                route_list = self.model.maas_agent.routes[segment_type][route_id]
                get_on_index = route_list.index(stops[0])
                get_off_index = route_list.index(stops[1])
                num_stops = abs(get_off_index - get_on_index)
                time_to_complete = num_stops * travel_speed

                if elapsed_time <= total_time_elapsed + time_to_complete:
                    progress = elapsed_time - total_time_elapsed
                    current_stop_index = get_on_index + int(progress // travel_speed)
                    current_stop_index = min(current_stop_index, get_off_index)
                    self.move_and_update_location(self.get_station_coordinates(route_list[current_stop_index]))
                    return
                total_time_elapsed += time_to_complete

            # Transfers
            elif segment_type == 'transfer':
                self.current_mode = 'walk'
                transfer_stations = tuple(segment[1])
                time_to_complete = self.model.maas_agent.transfers[transfer_stations]

                if elapsed_time <= total_time_elapsed + time_to_complete:
                    self.move_and_update_location(self.get_station_coordinates(transfer_stations[0]))
                    return
                total_time_elapsed += time_to_complete

            # Walk to destination
            elif segment_type == 'to destination':
                self.current_mode = 'walk'
                travel_speed = self.model.service_provider_agent.get_travel_speed('walk', current_time)
                station = segment[1][0]
                destination = segment[1][1]
                station_coordinates = self.get_station_coordinates(station)
                walking_route = self.model.maas_agent.dijkstra_with_congestion(station_coordinates, destination, 5)
                time_to_complete = len(walking_route) / travel_speed

                if elapsed_time <= total_time_elapsed + time_to_complete:
                    segment_elapsed_time = elapsed_time - total_time_elapsed
                    self.move_along_route(walking_route, travel_speed, segment_elapsed_time)
                    return
                total_time_elapsed += time_to_complete

        # Journey complete
        if elapsed_time >= total_time_elapsed:
            self.move_and_update_location(request['destination'])
            request['status'] = 'finished'
            
            # Extract request_id from the request dictionary
            request_id = request.get('request_id')
            
            # Add this line to update the database
            self.update_trip_status_in_database(request_id, 'finished')
            self.current_mode = None

    def move_along_route_single_mode(self, route, travel_speed):
        """
        Moves the commuter along the route according to the travel speed.
        """
        try:
            if not route or not isinstance(route, list):
                raise ValueError(f"Invalid route format: {route}")
            
            current_time = self.model.schedule.time
            # Get active request
            active_requests = [r for r in self.requests.values() if r['status'] == 'Service Selected']
            if not active_requests:
                print(f"[ERROR] No active requests found for commuter {self.unique_id}")
                return

            
            start_time = active_requests[0]['start_time']  # Assuming there's only one request for simplification
            distance_traveled = (current_time - start_time) * travel_speed

            total_route_distance = len(route) - 1

            if distance_traveled < total_route_distance:
                current_position_index = min(int(distance_traveled), len(route)-1)
                current_position = route[current_position_index]
            else:
                current_position = route[-1]
                active_requests[0]['status'] = 'finished'
                # Add this line to update the database
                request_id = next(iter(active_requests[0].get('request_id', None) for r in active_requests))
                if request_id:
                    self.update_trip_status_in_database(request_id, 'finished')
                self.current_mode = None  # Reset mode

            self.move_and_update_location(current_position)
        except Exception as e:
            print(f"[ERROR] Error in move_along_route_single_mode: {e}")

    def move_along_route(self, route, travel_speed, elapsed_time):
        try:
            if not route or len(route) < 2:
                raise ValueError(f"Invalid route format: {route}")
            distance_to_move = elapsed_time / travel_speed
            current_position = self.location
            total_distance = 0
            for i in range(len(route) - 1):
                segment_start = route[i]
                segment_end = route[i + 1]
                segment_distance = self.calculate_distance(segment_start, segment_end)

                if total_distance + segment_distance >= distance_to_move:
                    # Calculate position within current segment
                    remaining = distance_to_move - total_distance
                    ratio = remaining / segment_distance
                    new_x = segment_start[0] + ratio * (segment_end[0] - segment_start[0])
                    new_y = segment_start[1] + ratio * (segment_end[1] - segment_start[1])
                    current_position = (int(round(new_x)), int(round(new_y)))
                    break
                total_distance += segment_distance
                current_position = segment_end

            self.move_and_update_location(current_position)
        except Exception as e:
            print(f"[ERROR] Error in move_along_route: {e}")

    def calculate_distance(self, point1, point2):
        try:
            if not all(isinstance(p, tuple) and len(p) == 2 for p in [point1, point2]):
                raise ValueError(f"Invalid point format: {point1}, {point2}")
            return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
        except Exception as e:
            print(f"[ERROR] Error calculating distance: {e}")
            return 0

    def move_and_update_location(self, next_position):
        """Updates commuter position on grid and internal location"""
        try:
            if not isinstance(next_position, tuple) or len(next_position) != 2 or \
                not all(isinstance(x, int) for x in next_position):
                raise ValueError(f"Invalid position format: {next_position}")
                
            if not (0 <= next_position[0] < self.model.grid.width and \
                    0 <= next_position[1] < self.model.grid.height):
                raise ValueError(f"Position {next_position} outside grid bounds")
                
            self.model.grid.move_agent(self, next_position)
            self.location = next_position
            
            
        except Exception as e:
            print(f"[ERROR] Failed to update location: {e}")

    def get_station_coordinates(self, station_name):
        """Gets coordinates for train/bus station by ID"""
        try:
            if not isinstance(station_name, str):
                raise ValueError(f"Invalid station name format: {station_name}")
                
            if station_name.startswith('T'):
                coords = self.model.maas_agent.stations['train'].get(station_name)
            elif station_name.startswith('B'):
                coords = self.model.maas_agent.stations['bus'].get(station_name) 
            else:
                raise ValueError(f"Invalid station ID format: {station_name}")
                
            if not coords:
                raise ValueError(f"Station {station_name} not found")
                
            return coords
        except Exception as e:
            print(f"[ERROR] Error getting station coordinates: {e}")
            raise

    def check_travel_status(self):
        """Checks if commuter has reached destination"""
        try:
            for request_id, request in list(self.requests.items()):
                if request['status'] == 'Service Selected':
                    if self.location == request['destination']:
                        request['status'] = 'finished'
                        # Update status in database
                        self.update_trip_status_in_database(request_id, 'finished')
        except Exception as e:
            print(f"[ERROR] Error checking travel status: {e}")
