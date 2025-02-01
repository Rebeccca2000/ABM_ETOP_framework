from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import TextElement
from mesa import Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa import Agent
from agent_service_provider_03 import ServiceProvider
from agent_commuter_03 import Commuter
from agent_MaaS_03 import MaaS
from sqlalchemy.orm import sessionmaker, scoped_session
from agent_service_provider_initialisation_03 import reset_database
import uuid
import random
import database_01
from database_01 import num_commuters, grid_width, grid_height, income_weights, \
        health_weights, payment_weights, age_distribution, disability_weights, \
        tech_access_weights,DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS, DB_CONNECTION_STRING, \
        SIMULATION_STEPS, CHANCE_FOR_INSERTING_RANDOM_TRAFFIC, \
        CONGESTION_ALPHA,CONGESTION_BETA, CONGESTION_CAPACITY, CONGESTION_T_IJ_FREE_FLOW, \
        ASC_VALUES, UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS, \
        UTILITY_FUNCTION_BASE_COEFFICIENTS, PENALTY_COEFFICIENTS, \
        AFFORDABILITY_THRESHOLDS, FLEXIBILITY_ADJUSTMENTS, \
        VALUE_OF_TIME, public_price_table, ALPHA_VALUES, BACKGROUND_TRAFFIC_AMOUNT,\
        UberLike1_cpacity, UberLike1_price, \
        UberLike2_cpacity, UberLike2_price, \
        BikeShare1_capacity, BikeShare1_price, \
        BikeShare2_capacity, BikeShare2_price, \
        subsidy_dataset, daily_config,weekly_config,monthly_config
from sqlalchemy import create_engine
from agent_service_provider_initialisation_03 import CommuterInfoLog
import json
class StationAgent(Agent):
    def __init__(self, unique_id, model, location, mode):
        super().__init__(unique_id, model)
        self.location = location
        self.mode = mode

    def step(self):
        pass

class MobilityModel(Model):
    def __init__(self, db_connection_string, num_commuters, grid_width, grid_height, data_income_weights , \
                 data_health_weights, data_payment_weights, data_age_distribution,\
                data_disability_weights, data_tech_access_weights, \
                CHANCE_FOR_INSERTING_RANDOM_TRAFFIC,\
                ASC_VALUES, UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS, \
                UTILITY_FUNCTION_BASE_COEFFICIENTS, PENALTY_COEFFICIENTS, \
                AFFORDABILITY_THRESHOLDS, FLEXIBILITY_ADJUSTMENTS, VALUE_OF_TIME,\
                public_price_table, ALPHA_VALUES,\
                DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS, \
                BACKGROUND_TRAFFIC_AMOUNT, CONGESTION_ALPHA,\
                CONGESTION_BETA, CONGESTION_CAPACITY, CONGESTION_T_IJ_FREE_FLOW, \
                uber_like1_capacity, uber_like1_price, 
                uber_like2_capacity, uber_like2_price, 
                bike_share1_capacity, bike_share1_price, 
                bike_share2_capacity, bike_share2_price, 
                subsidy_dataset,subsidy_config):
        self.db_engine = create_engine(db_connection_string)
        self.Session = scoped_session(sessionmaker(bind=self.db_engine))
        self.session = self.Session()
        # Step 1: Reset the database with dynamic parameters
        reset_database(self.db_engine, self.session,
                       uber_like1_capacity, uber_like1_price, 
                       uber_like2_capacity, uber_like2_price, 
                       bike_share1_capacity, bike_share1_price, 
                       bike_share2_capacity, bike_share2_price)
        
        super().__init__()
        self.db_connection_string = db_connection_string

        self.grid = MultiGrid(grid_width, grid_height, torus=False)
        self.schedule = RandomActivation(self)


        self.num_commuters = num_commuters
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.data_income_weights = data_income_weights
        self.data_health_weights = data_health_weights
        self.data_payment_weights = data_payment_weights
        self.data_age_distribution = data_age_distribution
        self.data_disability_weights = data_disability_weights
        self.data_tech_access_weights = data_tech_access_weights
        self.chance_for_inserting_random_traffic = CHANCE_FOR_INSERTING_RANDOM_TRAFFIC
        self.subsidy_config = subsidy_config
        ##################################################################################
        ################# Initialisation for the commuter agent###########################
        ##################################################################################
        self.asc_values = ASC_VALUES
        self.utility_function_high_income_car_coefficients = UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS
        self.utility_function_base_coefficients = UTILITY_FUNCTION_BASE_COEFFICIENTS
        self.penalty_coefficients = PENALTY_COEFFICIENTS
        self.affordability_thresholds = AFFORDABILITY_THRESHOLDS
        self.flexibility_adjustments = FLEXIBILITY_ADJUSTMENTS
        self.value_of_time = VALUE_OF_TIME
        ###################################################################################
        ################### Initialisation for ServiceProvider Agent ######################
        ###################################################################################
        self.alpha_values = ALPHA_VALUES
        self.public_price_table = public_price_table
        self.service_provider_agent = ServiceProvider(unique_id=1, model=self, \
                                                      db_connection_string=db_connection_string,\
                                                    ALPHA_VALUES = self.alpha_values, \
                                                    public_price_table=self.public_price_table)
        self.schedule.add(self.service_provider_agent)
        # #######################################################################################
        # Initialize current_step to 0
        self.current_step = 0

        # Add station agents
        for mode, mode_stations in database_01.stations.items():
            for station_id, (x, y) in mode_stations.items():
                station_agent = StationAgent(unique_id=f"{mode}_{station_id}", model=self, location=(x, y), mode=mode)
                self.grid.place_agent(station_agent, (x, y))
                self.schedule.add(station_agent)

        self.commuter_agents = []
        income_levels = ['low', 'middle', 'high']

        # Define health statuses and their respective weights
        health_statuses = ['good', 'poor']

        # Define payment schemes and their respective weights
        payment_schemes = ['PAYG', 'subscription']

        # Define age distribution based on the provided data
        age_distribution = self.data_age_distribution
        # Calculate cumulative weights for age distribution
        cumulative_age_weights = []
        current_weight = 0
        for age_range, weight in age_distribution.items():
            current_weight += weight
            cumulative_age_weights.append((age_range, current_weight))

        def get_random_age():
            rnd = random.random()
            for age_range, cumulative_weight in cumulative_age_weights:
                if rnd <= cumulative_weight:
                    return random.randint(age_range[0], age_range[1])
            return random.randint(0, 70)  # Fallback in case of any issues

        for i in range(num_commuters):
            income_level = random.choices(income_levels, self.data_income_weights)[0]  # Weighted random choice for income level
            health_status = random.choices(health_statuses, self.data_health_weights)[0]  # Weighted random choice for health status
            payment_scheme = random.choices(payment_schemes, self.data_payment_weights)[0]  # Weighted random choice for payment scheme
            age = get_random_age()
            has_disability = random.choices([True, False], self.data_disability_weights)[0]  # 15% chance of having a disability
            tech_access = random.choices([True, False], self.data_tech_access_weights)[0]  # 96.2% chance of having tech access

            commuter = Commuter(
                unique_id=i + 2,
                model=self,
                commuter_location=(random.randint(0, self.grid_width - 1), random.randint(0, self.grid_height - 1)),
                age=age,
                income_level=income_level,
                has_disability=has_disability,
                tech_access=tech_access,
                health_status=health_status,
                payment_scheme=payment_scheme,  # Add payment scheme to commuter
                ASC_VALUES = self.asc_values,
                UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS = self.utility_function_high_income_car_coefficients,
                UTILITY_FUNCTION_BASE_COEFFICIENTS = self.utility_function_base_coefficients, 
                PENALTY_COEFFICIENTS = self.penalty_coefficients,
                AFFORDABILITY_THRESHOLDS=self.affordability_thresholds,
                FLEXIBILITY_ADJUSTMENTS = self.flexibility_adjustments,
                VALUE_OF_TIME = self.value_of_time,
                subsidy_dataset = subsidy_dataset
            )
            self.commuter_agents.append(commuter)
            self.schedule.add(commuter)
            self.grid.place_agent(commuter, commuter.location)
            self.record_commuter_info(commuter)  # Record commuter info in the database
        ####################################################################################
        ################### Initialisation for the MaaS agent #############################
        ####################################################################################
        self.congestion_alpha= CONGESTION_ALPHA
        self.congestion_beta= CONGESTION_BETA
        self.congestion_capacity = CONGESTION_CAPACITY
        self.conjestion_t_ij_free_flow = CONGESTION_T_IJ_FREE_FLOW
        self.dynamic_maas_surcharge_base_coefficient = DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS
        self.background_traffic_amount = BACKGROUND_TRAFFIC_AMOUNT
        self.maas_agent = MaaS(unique_id=self.num_commuters + 2, model=self, \
                               service_provider_agent=self.service_provider_agent, \
                                commuter_agents=self.commuter_agents,\
                                DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS = self.dynamic_maas_surcharge_base_coefficient , \
                                BACKGROUND_TRAFFIC_AMOUNT = self.background_traffic_amount,\
                                stations = database_01.stations, routes = database_01.routes, \
                                transfers = database_01.transfers, num_commuters = self.num_commuters, \
                                grid_width = self.grid_width, grid_height = self.grid_height, \
                                CONGESTION_ALPHA = self.congestion_alpha,\
                                CONGESTION_BETA= self.congestion_beta, \
                                CONGESTION_CAPACITY = self.congestion_capacity, \
                                CONGESTION_T_IJ_FREE_FLOW = self.conjestion_t_ij_free_flow,\
                                subsidy_config = self.subsidy_config)
        self.schedule.add(self.maas_agent)
        
        

    def record_commuter_info(self, commuter):
        new_commuter_info = CommuterInfoLog(
            commuter_id=commuter.unique_id,
            location=commuter.location,
            age=commuter.age,
            income_level=commuter.income_level,
            has_disability=1 if commuter.has_disability else 0,  # Convert boolean to integer
            tech_access=1 if commuter.tech_access else 0,  # Convert boolean to integer
            health_status=commuter.health_status,
            payment_scheme=commuter.payment_scheme
        )
        with self.Session() as session:
            existing_commuter = session.query(CommuterInfoLog).filter_by(
                commuter_id=commuter.unique_id
            ).first()
            if existing_commuter:
                print(f"Commuter {commuter.unique_id} already exists in the database. Skipping insert.")
            else:
                session.add(new_commuter_info)
                session.commit()
                print(f"Commuter {commuter.unique_id} info recorded successfully.")


    def update_commuter_info_log(self, commuter):
        """
        Updates the commuter information in the database, including requests and services_owned.
        UUIDs in requests and services_owned will be converted to strings to ensure proper JSON serialization.
        """

        # Convert UUIDs to strings for requests and services_owned before saving
        requests_with_str_keys = {str(k): {**v, 'request_id': str(v['request_id'])} for k, v in commuter.requests.items()}
        services_owned_with_str_keys = {str(k): v for k, v in commuter.services_owned.items()}

        with self.Session() as session:
            # Fetch the commuter's current info from the log
            commuter_log = session.query(CommuterInfoLog).filter_by(commuter_id=commuter.unique_id).first()

            # If the commuter log doesn't exist, create a new entry
            if not commuter_log:
                commuter_log = CommuterInfoLog(
                    commuter_id=commuter.unique_id,  # Use commuter.unique_id directly
                    location={'x': commuter.location[0], 'y': commuter.location[1]},
                    age=commuter.age,
                    income_level=commuter.income_level,
                    has_disability=1 if commuter.has_disability else 0,
                    tech_access=1 if commuter.tech_access else 0,
                    health_status=commuter.health_status,
                    payment_scheme=commuter.payment_scheme,
                    requests=str(requests_with_str_keys),  # Convert UUIDs to strings in requests
                    services_owned=str(services_owned_with_str_keys)  # Convert UUIDs to strings in services_owned
                )
                session.add(commuter_log)
            else:
                # Update the existing commuter log entry
                commuter_log.location = str({'x': commuter.location[0], 'y': commuter.location[1]})
                commuter_log.requests = str(requests_with_str_keys)  # Update requests with converted UUIDs
                commuter_log.services_owned = str(services_owned_with_str_keys)  # Update services_owned with converted UUIDs

            # Commit the session to save changes
            session.commit()


    def check_is_peak(self, current_ticks):
        if (36 <= current_ticks % 144 < 60) or (90 <= current_ticks % 144 < 114):
            return True
        return False

    # def should_create_trip(self, commuter):
    #     # Check the number of trips created in the current day (144 steps)
    #     trips_in_current_day = sum(1 for request in commuter.requests.values() if request['start_time'] // 144 == self.current_step // 144)
    #     if commuter.payment_scheme == 'PAYG':
    #         return trips_in_current_day <= 2
    #     elif commuter.payment_scheme == 'subscription':
    #         return trips_in_current_day <= 6
    #     return False


    def should_create_trip(self, commuter, current_ticks):
        trips_in_current_day = sum(1 for request in commuter.requests.values() if request['start_time'] // 144 == self.current_step // 144)

        # Determine the current peak period
        if 36 <= current_ticks % 144 < 60:
            peak_period = 'morning_peak'
        elif 90 <= current_ticks % 144 < 114:
            peak_period = 'evening_peak'
        else:
            peak_period = 'off_peak'

        # Count trips in the current peak period within the same day
        if peak_period == 'morning_peak':
            trips_in_peak = sum(1 for request in commuter.requests.values() if request['start_time'] // 144 == self.current_step // 144 and 36 <= request['start_time'] % 144 < 60)
        elif peak_period == 'evening_peak':
            trips_in_peak = sum(1 for request in commuter.requests.values() if request['start_time'] // 144 == self.current_step // 144 and 90 <= request['start_time'] % 144 < 114)
        else:
            trips_in_peak = 0  # No specific counting for off-peak

        # Adjust trip limits during peaks and off-peak
        if peak_period in ['morning_peak', 'evening_peak']:
            if commuter.payment_scheme == 'PAYG':
                # Ensure at least one trip is available for evening peak
                if peak_period == 'morning_peak':
                    return trips_in_peak < 1 and trips_in_current_day < 2
                else:
                    return trips_in_peak < 1 and trips_in_current_day < 3
            elif commuter.payment_scheme == 'subscription':
                # Ensure multiple trips can be made throughout the day
                if peak_period == 'morning_peak':
                    return trips_in_peak < 2 and trips_in_current_day < 4
                else:
                    return trips_in_peak < 2 and trips_in_current_day < 6
        else:
            # Allow trips during off-peak if the daily limit has not been reached
            if commuter.payment_scheme == 'PAYG':
                return trips_in_current_day < 3
            elif commuter.payment_scheme == 'subscription':
                return trips_in_current_day < 6

        return False



    def all_requests_finished(self, commuter):
        return all(request['status'] == 'finished' for request in commuter.requests.values())

    def create_new_request(self, current_step, commuter):
        if not commuter.requests or self.all_requests_finished(commuter):
            if not self.should_create_trip(commuter, self.current_step):
                return  # Do not create a new request if the commuter has reached their daily trip limit

            # Check current peak period to adjust trip creation probability
            if 36 <= self.current_step % 144 < 60:
                create_trip_probability = 0.15  # Morning peak
            elif 90 <= self.current_step % 144 < 114:
                create_trip_probability = 0.35  # Evening peak
            else:
                create_trip_probability = 0.03  # Off-peak

            if random.random() < create_trip_probability:
                request_id = uuid.uuid4()
                origin = commuter.location
                destination = (random.randint(0, self.grid.width - 1), random.randint(0, self.grid.height - 1))
                start_time = current_step + random.randint(0, 5)
                travel_purpose = random.choice(['work', 'school', 'shopping', 'leisure', 'medical'])
                commuter.create_request(request_id, origin, destination, start_time, travel_purpose)
                # print(f"Commuter {commuter.unique_id} created request from {origin} to {destination} starting at {start_time}")
                return True
            else:
                return False
            
    def get_current_step(self):
        return self.current_step
    
    def step(self):
        self.current_step += 1
        print(f"Step {self.current_step}")
        # Update the system's time step
        self.service_provider_agent.update_time_steps()

        availability_dict = self.service_provider_agent.initialize_availability(self.current_step - 1)

        for commuter in self.commuter_agents:
            # Create new request based on commuter's needs
            print(f"Creating new request for Commuter {commuter.unique_id} at Step {self.current_step}")
            self.create_new_request(self.current_step, commuter)
            self.update_commuter_info_log(commuter)
            for request_id, request in list(commuter.requests.items()):
                try:
                    if request['status'] == 'active':
                        # MaaS agent generates travel options for each request
                        
                        travel_options_without_MaaS = self.maas_agent.options_without_maas(request_id, request['start_time'], request['origin'], request['destination'])
                        # print(f"Generated travel options for request {request_id}: \nWithout MaaS: {travel_options_without_MaaS}")

                        travel_options_with_MaaS = self.maas_agent.maas_options(commuter.payment_scheme, request_id, request['start_time'], request['origin'], request['destination'])

                        # print(f"Generated travel options for request {request_id}: \nWith MaaS: {travel_options_with_MaaS}")

                        # Commuter ranks the travel options
                        ranked_options = commuter.rank_service_options(travel_options_without_MaaS, travel_options_with_MaaS, request_id)
                        # print(f"Ranked options for request {request_id}: {ranked_options}")

                        if ranked_options != []:  # Only attempt booking if there are ranked options available
                            # print(f"Attempting to book service for request {request_id}")
                            # MaaS agent attempts to book the highest-ranked service using ranked_options
                            booking_success, availability_dict = self.maas_agent.book_service(request_id, ranked_options, self.current_step, availability_dict)
                            if booking_success:
                                print(f"Booking for request {request_id} was successful.")
                            else:
                                print(f"Booking for request {request_id} was not successful.")
                        else:
                            print(f"No viable options for request {request_id}.")
                except Exception as e:
                    print(f"Error in MobilityModel processing request {request_id}: {e}")

            commuter.update_location()
            commuter.check_travel_status()  # Once the commuter arrives at the destination, increase the availability back

        # Update availability based on bookings
        print(f"Updating availability after bookings")
        self.service_provider_agent.update_availability()
        # Call dynamic_pricing_share to update pricing based on demand
        print(f"Calling dynamic pricing update")
        self.service_provider_agent.dynamic_pricing_share()
            # Probability of inserting random traffic
        random_traffic_probability = self.chance_for_inserting_random_traffic  # 20% chance to insert random traffic
        if random.random() < random_traffic_probability:
            with self.Session() as session:
                print(f"Inserting random background traffic at Step {self.current_step}")
                # Insert random traffic with a certain number of routes
                self.maas_agent.insert_random_traffic(session)
        self.schedule.step()

            
    def run_model(self, num_steps):
        for _ in range(num_steps):
            self.step()

    def process_request(self, request):
        try:
            commuter = self.get_commuter_by_id(request.commuter_id)
            if not commuter:
                print(f"Error: Commuter {request.commuter_id} not found")
                return False
            
            # Debug income weights
            print(f"Debug - Processing request for commuter {request.commuter_id}")
            print(f"Income weights: {self.data_income_weights}")
            print(f"Commuter income level index: {commuter.income_level_index}")
            
            if not (0 <= commuter.income_level_index < len(self.data_income_weights)):
                print(f"Invalid income level index: {commuter.income_level_index}")
                return False
            
            # Rest of the processing code
        except Exception as e:
            print(f"Error in process_request: {e}")
            return False


def agent_portrayal(agent):
    portrayal = {}
    if isinstance(agent, Commuter):
        color = ""
        if agent.income_level == 'low':
            color = "green"
        elif agent.income_level == 'middle':
            color = "blue"
        else:  # high income
            color = "red"
        
        if agent.current_mode is None:
            portrayal = {"Shape": "circle", "Color": color, "Filled": "true", "r": 0.7, "Layer": 1, "text": str(agent.unique_id), "text_color": "white"}
        else:
            if agent.current_mode == 'walk':
                portrayal = {"Shape": "circle", "Color": color, "Filled": "true", "r": 0.5, "Layer": 1, "text": str(agent.unique_id), "text_color": "white"}
            elif agent.current_mode == 'bike':
                portrayal = {"Shape": "arrowHead", "Color": color, "Filled": "true","scale": 0.5,"heading_x": 0, "heading_y": 1, "Layer": 1, "text": str(agent.unique_id), "text_color": "white"}
            elif agent.current_mode == 'car':
                portrayal = {"Shape": "arrowHead", "Color": color, "Filled": "true","scale": 0.5,"heading_x": 0, "heading_y": -1, "Layer": 1, "text": str(agent.unique_id), "text_color": "white"}
            elif agent.current_mode == 'bus':
                portrayal = {"Shape": "rect", "Color": color, "Filled": "true", "w": 0.5, "h": 0.5, "Layer": 1, "text": str(agent.unique_id), "text_color": "white"}
            elif agent.current_mode == 'train':
                portrayal = {"Shape": "rect", "Color": color, "Filled": "true", "w": 0.8, "h": 0.3, "Layer": 1, "text": str(agent.unique_id), "text_color": "white"}


    elif isinstance(agent, StationAgent):
        color = "yellow" if agent.mode == 'train' else "orange"
        portrayal = {"Shape": "rect", "Color": color, "Filled": "true", "w": 0.1, "h": 0.3, "Layer": 0}
    return portrayal


class CommuteCountElement(TextElement):
    def __init__(self):
        pass

    def render(self, model):
        return (
            "Number of commuters: " + str(len(model.commuter_agents)) + "<br>" +
            "<span style='color: green;'>●</span> Low income<br>" +
            "<span style='color: blue;'>●</span> Middle income<br>" +
            "<span style='color: red;'>●</span> High income<br>" +
            "<span style='color: black;'>▲</span> Bike<br>" +
            "<span style='color: black;'>▼</span> Car<br>" +
            "<span style='color: black;'>■</span> Bus<br>" +
            "<span style='color: black;'>▭</span> Train<br>" +
            "<span style='color: black;'>●</span> Walk"
        )

grid = CanvasGrid(agent_portrayal, database_01.grid_width, database_01.grid_height, 1500, 1500)

# engine = create_engine(DB_CONNECTION_STRING)
# Session = sessionmaker(bind=engine)

# server = ModularServer(
#     MobilityModel,
#     [grid, CommuteCountElement()],
#     "Mobility Model",
#     {'db_connection_string': DB_CONNECTION_STRING,
#     'num_commuters': num_commuters,
#     'grid_width': 55,
#     'grid_height': 55,
#     'data_income_weights': [0.5, 0.3, 0.2],
#     'data_health_weights': [0.9, 0.1],
#     'data_payment_weights': [0.8, 0.2],
#     'data_age_distribution': {(18, 25): 0.2, (26, 35): 0.3, (36, 45): 0.2, (46, 55): 0.15, (56, 65): 0.1, (66, 75): 0.05},
#     'data_disability_weights': [0.2, 0.8],
#     'data_tech_access_weights': [0.95, 0.05],
#     'subsidy_dataset': {
#             'low': {'bike': 0.1, 'car': 0.05, 'MaaS_Bundle': 0.4},
#             'middle': {'bike': 0.3, 'car': 0.01, 'MaaS_Bundle': 0.5},
#             'high': {'bike': 0.4, 'car': 0, 'MaaS_Bundle': 0.6}
#         },
#     'CHANCE_FOR_INSERTING_RANDOM_TRAFFIC': 0.2,
#     'ASC_VALUES': {'car': 0, 'bike': 0, 'public': 0, 'walk': 0, 'maas': 0, 'default': 0},
#     'UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS': {'beta_C': -0.05, 'beta_T': -0.06},
#     'UTILITY_FUNCTION_BASE_COEFFICIENTS': {'beta_C': -0.05, 'beta_T': -0.06, 'beta_W': -0.01, 'beta_A': -0.01, 'alpha': -0.01},
#     'PENALTY_COEFFICIENTS': {'disability_bike_walk': 0.8, 'age_health_bike_walk': 0.3, 'no_tech_access_car_bike': 0.1},
#     'AFFORDABILITY_THRESHOLDS': {'low': 25, 'middle': 85, 'high': 250},
#     'FLEXIBILITY_ADJUSTMENTS': {'low': 1.05, 'medium': 1.0, 'high': 0.95},
#     'VALUE_OF_TIME': {'low': 9.64, 'middle': 23.7, 'high': 67.2},
#     'public_price_table': {'train': {'on_peak': 2, 'off_peak': 1.5}, 'bus': {'on_peak': 1, 'off_peak': 0.8}},
#     'ALPHA_VALUES': {'UberLike1': 0.5, 'UberLike2': 0.5, 'BikeShare1': 0.5, 'BikeShare2': 0.5},
#     'DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS': {'S_base': 0.08, 'alpha': 0.2, 'delta': 0.5},
#     'BACKGROUND_TRAFFIC_AMOUNT': 70,
#     'CONGESTION_ALPHA': 0.25,
#     'CONGESTION_BETA': 4,
#     'CONGESTION_CAPACITY': 4,
#     'CONGESTION_T_IJ_FREE_FLOW': 2,
#     'uber_like1_capacity': 20,
#     'uber_like1_price': 4,
#     'uber_like2_capacity': 19,
#     'uber_like2_price': 3,
#     'bike_share1_capacity': 15,
#     'bike_share1_price': 0.5,
#     'bike_share2_capacity': 12,
#     'bike_share2_price': 0.2,
#     'subsidy_dataset': subsidy_dataset,
#     'subsidy_config': daily_config}
# )

# server.port = 8521
# server.launch()

# if __name__ == "__main__":
#     model = MobilityModel(db_connection_string=DB_CONNECTION_STRING, num_commuters=num_commuters)
#     model.run_model(SIMULATION_STEPS)

# else:
#     server.launch()

# Runner Code for the MobilityModel
if __name__ == "__main__":
    model = MobilityModel(
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
        CHANCE_FOR_INSERTING_RANDOM_TRAFFIC=CHANCE_FOR_INSERTING_RANDOM_TRAFFIC,
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
        uber_like1_price = UberLike1_price, 
        uber_like2_capacity = UberLike2_cpacity, 
        uber_like2_price = UberLike2_price, 
        bike_share1_capacity = BikeShare1_capacity, 
        bike_share1_price = BikeShare1_price, 
        bike_share2_capacity = BikeShare2_capacity,
        bike_share2_price = BikeShare2_price, 
        subsidy_dataset = subsidy_dataset,
        subsidy_config = daily_config
    )
    model.run_model(SIMULATION_STEPS)