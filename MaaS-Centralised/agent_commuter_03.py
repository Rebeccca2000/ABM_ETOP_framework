from mesa import Agent
import math
import random
import numpy as np
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
        print("calling calculate_mode_choice_probabilities for rank serviec options")
        print(f" so now the combined travel options are:{combined_travel_options}")
        print("THE END for COMBINED TRAVEL OPTIONS")
        print("THE END for COMBINED TRAVEL OPTIONS")
        print("THE END for COMBINED TRAVEL OPTIONS")
        print("THE END for COMBINED TRAVEL OPTIONS")
        print("THE END for COMBINED TRAVEL OPTIONS")
        print("THE END for COMBINED TRAVEL OPTIONS")
        print("THE END for COMBINED TRAVEL OPTIONS")
        print("THE END for COMBINED TRAVEL OPTIONS")
        print("THE END for COMBINED TRAVEL OPTIONS")
        print("THE END for COMBINED TRAVEL OPTIONS")
        print("THE END for COMBINED TRAVEL OPTIONS")
        print("THE END for COMBINED TRAVEL OPTIONS")

        probabilities = self.calculate_mode_choice_probabilities(combined_travel_options, request_id)

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

        print("Show the ranked options")
        print(f"For a commuter that has the income level of {self.income_level}")
        print(f"This is the top 5 ranked options {top_5_ranked_options}")
        
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
                print("calling calculate_generalized_utility for calculate mode choice probabilities")
                utility, subsidy_amount = self.calculate_generalized_utility(price, time, mode, request_id)
                print(f"the utility is {utility} for mode {mode}")
                utilities[mode] = utility
                subsidies[mode] = subsidy_amount  # Save the subsidy amount for the mode

            except KeyError as e:
                print(f"KeyError: {e}")

        # Calculate the probability of each mode
        sum_exp_utilities = sum(np.exp(utility) for utility in utilities.values())
        print(f"the sum_exp_utilities is {sum_exp_utilities}")
        probabilities = {mode: np.exp(utility) / sum_exp_utilities for mode, utility in utilities.items()}
        print(f"the probabilities is {probabilities}")
        # Prepare the output list including subsidy amounts
        probability_for_options = []
        for mode, probability in probabilities.items():
            route = travel_options[mode]['route']
            time = travel_options[mode]['time']
            subsidy = subsidies[mode]  # Retrieve the saved subsidy for the mode
            
            # Include subsidy in the output
            probability_for_options.append((probability, mode, route, time, subsidy))
        print(f'Test calculate_mode_choice_probabilities {probability_for_options}')
        return probability_for_options

        
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
        Calculate the generalized utility for a given mode based on price, time, ASC, and government subsidies.
        
        Parameters:
        - price: The cost of using the transportation mode.
        - time: The duration of the trip (in minutes).
        - mode: The transportation mode (e.g., 'car', 'bike', 'public').
        - request_id: The commuter's request ID for referencing specific details (e.g., income level, etc.).

        Returns:
        - U_j: The generalized utility for mode j.
        - subsidy_amount: The amount of subsidy applied.
        """
        
        # Step 1: Get the value of time (VOT)
        value_of_time = self.get_value_of_time(request_id)
        
        # Step 2: Set the coefficients based on the mode and context (e.g., income)
        coefficients = self.utility_function_base_coefficients.copy()
        

        # Step 3: Calculate the utility components (price, travel time, and ASC)
        VOT_per_ten_min = value_of_time/6
        
        # Step 4: Use the set_ASC_values function to get the ASC for the mode
        def set_ASC_values(mode):
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
        
        ASC_j = set_ASC_values(mode)
        print(f"ASC_j for mode '{mode}' is {ASC_j}")

        # Step 5: Apply the government subsidy based on the income level and mode
        income_level = self.income_level  # Directly accessing self.income_level
        subsidy_dataset = self.subsidy_dataset.copy()

        def map_mode_to_subsidy_key(mode):
            if 'maas' in mode.lower():
                return 'MaaS_Bundle'
            elif 'bike' in mode.lower():
                return 'bike'
            elif 'car' in mode.lower() or 'uber' in mode.lower():  # Assuming 'car' includes Uber-like services
                return 'car'
            else:
                return None

        subsidy_key = map_mode_to_subsidy_key(mode)
        # If subsidy key is valid, fetch the subsidy percentage; otherwise, default to 0
        if subsidy_key:
            subsidy_percentage = subsidy_dataset.get(income_level, {}).get(subsidy_key, 0)  # Default to 0 if no subsidy available
        else:
            subsidy_percentage = 0
        print(f"the price for the mode {mode} is {price}")
        subsidy_amount = price * subsidy_percentage  # Subsidy is a percentage of the price
        price_after_subsidy = price - subsidy_amount  # Apply the subsidy
            

        # Step 6: Calculate the generalized utility with the modified price
        U_j = (
            ASC_j +  # The ASC for the specific mode
            (coefficients['beta_C'] * price_after_subsidy) +  # Price sensitivity (after subsidy)
            (coefficients['beta_T'] * VOT_per_ten_min * time)  # Travel time sensitivity (includes VOT)
        )

        # Return both the utility value and the subsidy amount
        print(f"The utility value is {U_j} and the subsidy_amount is {subsidy_amount}")
        return U_j, subsidy_amount



  
    def accept_service(self, request_id):
        print(f"[DEBUG] accept_service called for Request ID: {request_id}")

        request = self.requests.get(request_id)
        
        if request:
            print(f"[DEBUG] Request found. Updating selected route and status.")
            request['status'] = 'Service Selected'
            self.services_owned[request_id] = request
            print(f"[DEBUG] Updated request: {request}")
            return True
        else:
            print(f"[DEBUG] No request found for Request ID: {request_id}")
            return False

    def accept_service_non_maas(self, request_id,selected_route):
        request = self.requests.get(request_id)
            
        if request:
            request['selected_route'] = selected_route
            request['status'] = 'Service Selected'
            self.services_owned[request_id] = request
            print(f"[DEBUG] Updated request: {request}")
            return True
        else:
            print(f"[DEBUG] No request found for Request ID: {request_id}")
            return False

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


    # def handle_maas_bundle(self, detailed_itinerary, start_time, request):
    #     current_time = self.model.schedule.time
    #     elapsed_time = current_time - start_time
    #     current_segment_start_time = start_time
    #     print("[DEBUG] Handling MaaS Bundle for Commuter")
    #     print(f"[DEBUG] Selected route in MaaS bundle: {request.get('selected_route', {})}")
    #     # Ensure 'to_station_info' and 'to_destination_info' exist in the selected route
    #     to_station_info = request['selected_route']['to_station_info']
    #     to_destination_info = request['selected_route']['to_destination_info']

    #     if not to_station_info:
    #         print("[ERROR] Missing 'to_station_info' in MaaS bundle")
    #         return

    #     if not to_destination_info:
    #         print("[ERROR] Missing 'to_destination_info' in MaaS bundle")
    #         return

    #     for segment in detailed_itinerary:
    #         if segment[0] == 'to station':
                               
    #             print(f"Info for handle-maas_bundle:")
    #             print(f"to station info for handle-maas_bundle: {to_station_info}")
    #             print(f"segment info for handle-maas_bundle:{segment}")
    #             company = to_station_info[0] 
    #             if 'Bike' in company:
    #                 mode = 'bike'
    #             elif 'Uber' in company:
    #                 mode = 'car'
    #             else:
    #                 mode = 'walk'
    #             travel_speed = self.model.service_provider_agent.get_travel_speed(mode, current_time)
    #             start_location = segment[1][0]
    #             end_location = self.get_station_coordinates(segment[1][1])
    #             route = self.model.maas_agent.dijkstra_with_congestion(start_location, end_location)
    #             #Chaneg this using the saved data.
    #             time_to_complete = len(route) * travel_speed
    #             if elapsed_time <= time_to_complete:
    #                 self.move_along_route(route, travel_speed, elapsed_time)
    #                 return
    #             else:
    #                 elapsed_time -= time_to_complete
    #                 current_segment_start_time += time_to_complete
    #         elif segment[0] == 'to destination':
    #             # Handle the single mode (bike, Uber, etc.)

    #             company =  to_destination_info[0]
    #             if 'Bike' in company:
    #                 mode = 'bike'
    #             elif 'Uber' in company:
    #                 mode = 'car'
    #             else:
    #                 mode = 'walk'
    #             print(f"[INFO] The mode for handle_maas_bundle for to station or destination is :{mode}")
    #             travel_speed = self.model.service_provider_agent.get_travel_speed(mode, current_time)
    #             start_location = self.get_station_coordinates(segment[1][0])
    #             end_location = segment[1][1]
    #             route = self.model.maas_agent.dijkstra_with_congestion(start_location, end_location)

    #             time_to_complete = len(route) * travel_speed
    #             if elapsed_time <= time_to_complete:
    #                 self.move_along_route(route, travel_speed, elapsed_time)
    #                 return
    #             else:
    #                 elapsed_time -= time_to_complete
    #                 current_segment_start_time += time_to_complete

    #         elif segment[0] in ['bus', 'train']:
    #             # Handle the public transport segment
    #             self.current_mode = segment[0]
    #             travel_speed = self.model.service_provider_agent.get_travel_speed(segment[0], current_time)
    #             route_list = self.model.maas_agent.routes[segment[0]][segment[1]]
    #             stops = segment[2]
    #             get_on_index = route_list.index(stops[0])
    #             get_off_index = route_list.index(stops[1])
    #             num_stops = get_off_index - get_on_index
    #             time_on_transit = num_stops * travel_speed

    #             if elapsed_time <= time_on_transit:
    #                 current_stop_index = get_on_index + (elapsed_time // travel_speed)
    #                 self.move_and_update_location(self.get_station_coordinates(route_list[int(current_stop_index)]))
    #                 return
    #             else:
    #                 elapsed_time -= time_on_transit
    #                 current_segment_start_time += time_on_transit

    #         elif segment[0] == 'transfer':
    #             # Handle the transfer segment
    #             self.current_mode = 'walk'
    #             transfer_time = self.model.maas_agent.transfers[tuple(segment[1])]
    #             if elapsed_time <= transfer_time:
    #                 self.move_and_update_location(self.get_station_coordinates(segment[1][0]))
    #                 return
    #             else:
    #                 elapsed_time -= transfer_time
    #                 current_segment_start_time += transfer_time

    #     # If we finish the loop and haven't returned, the commuter has arrived at the destination
    #     request['status'] = 'finished'
    #     self.current_mode = None  # Reset mode
    #     print("[DEBUG] MaaS Bundle: Commuter has finished the journey")

    def handle_maas_bundle(self, detailed_itinerary, start_time, request):
        current_time = self.model.schedule.time
        elapsed_time = current_time - start_time
        current_segment_start_time = start_time
        print("[DEBUG] Handling MaaS Bundle for Commuter")
        print(f"[DEBUG] Selected route in MaaS bundle: {request.get('selected_route', {})}")
        
        # Ensure 'to_station_info' and 'to_destination_info' exist in the selected route
        to_station_info = request['selected_route']['to_station_info']
        to_destination_info = request['selected_route']['to_destination_info']

        if not to_station_info:
            print("[ERROR] Missing 'to_station_info' in MaaS bundle")
            return

        if not to_destination_info:
            print("[ERROR] Missing 'to_destination_info' in MaaS bundle")
            return

        for segment in detailed_itinerary:
            if segment[0] == 'to station':
                # Handle 'to station' segment using the saved data
                company = to_station_info[0] 
                if 'Bike' in company:
                    mode = 'bike'
                elif 'Uber' in company:
                    mode = 'car'
                else:
                    mode = 'walk'

                travel_speed = self.model.service_provider_agent.get_travel_speed(mode, current_time)

                # Use saved detailed route from to_station_info
                route = to_station_info[3]  # Assuming [3] is the saved detailed route
                
                time_to_complete = len(route) * travel_speed
                if elapsed_time <= time_to_complete:
                    self.move_along_route(route, travel_speed, elapsed_time)
                    return
                else:
                    elapsed_time -= time_to_complete
                    current_segment_start_time += time_to_complete

            elif segment[0] == 'to destination':
                # Handle 'to destination' segment using the saved data
                company = to_destination_info[0]
                if 'Bike' in company:
                    mode = 'bike'
                elif 'Uber' in company:
                    mode = 'car'
                else:
                    mode = 'walk'
                    
                travel_speed = self.model.service_provider_agent.get_travel_speed(mode, current_time)

                # Use saved detailed route from to_destination_info
                route = to_destination_info[3]  # Assuming [3] is the saved detailed route

                time_to_complete = len(route) * travel_speed
                if elapsed_time <= time_to_complete:
                    self.move_along_route(route, travel_speed, elapsed_time)
                    return
                else:
                    elapsed_time -= time_to_complete
                    current_segment_start_time += time_to_complete

            elif segment[0] in ['bus', 'train']:
                # Handle the public transport segment
                self.current_mode = segment[0]
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
                # Handle the transfer segment
                self.current_mode = 'walk'
                transfer_time = self.model.maas_agent.transfers[tuple(segment[1])]
                if elapsed_time <= transfer_time:
                    self.move_and_update_location(self.get_station_coordinates(segment[1][0]))
                    return
                else:
                    elapsed_time -= transfer_time
                    current_segment_start_time += transfer_time

        # If we finish the loop and haven't returned, the commuter has arrived at the destination
        request['status'] = 'finished'
        self.current_mode = None  # Reset mode
    print("[DEBUG] MaaS Bundle: Commuter has finished the journey")




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
                walking_route = self.model.maas_agent.dijkstra_with_congestion(segment[1][0], get_on_station_coordinates,5)
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
                walking_route = self.model.maas_agent.dijkstra_with_congestion(get_off_station_coordinates, destination_coordinates,5)
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
