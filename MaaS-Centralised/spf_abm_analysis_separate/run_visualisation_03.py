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
import math
import database_01
from sqlalchemy import create_engine, func, text
from sqlalchemy.orm import sessionmaker
import time
import os
import gc
import traceback
from database_01 import num_commuters, grid_width, grid_height, income_weights, \
        health_weights, payment_weights, age_distribution, disability_weights, \
        tech_access_weights,DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS, DB_CONNECTION_STRING, \
        SIMULATION_STEPS, \
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
                subsidy_dataset,subsidy_config,schema=None):
        self.db_engine = create_engine(db_connection_string)
        self.schema = schema
        
        # If schema is provided, set it in the session
        if self.schema:
            self.engine = create_engine(db_connection_string)
            with self.engine.connect() as connection:
                connection.execute(text(f"SET search_path TO {self.schema}"))
        self.Session = scoped_session(sessionmaker(bind=self.db_engine))
        self.session = self.Session()
        # Step 1: Reset the database with dynamic parameters
        reset_database(self.db_engine, self.session,
                       uber_like1_capacity, uber_like1_price, 
                       uber_like2_capacity, uber_like2_price, 
                       bike_share1_capacity, bike_share1_price, 
                       bike_share2_capacity, bike_share2_price, self.schema)
        
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
        self.service_provider_agent = ServiceProvider(unique_id='service_provider_1', model=self, \
                                                      db_connection_string=db_connection_string,\
                                                    ALPHA_VALUES = self.alpha_values, \
                                                    public_price_table=self.public_price_table,schema=self.schema)
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
        self.maas_agent = MaaS(unique_id="maas_1", model=self, \
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
                                subsidy_config = self.subsidy_config,schema=self.schema)
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
                #print(f"Commuter {commuter.unique_id} info recorded successfully.")


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

    def should_create_trip(self, commuter, current_step):
        """
        Determines if a commuter should create a new trip based on their travel history,
        time of day, demographics, and payment scheme.
        
        Returns:
            bool: True if a new trip should be created, False otherwise
        """
        # Get time context
        ticks_in_day = 144
        current_day_tick = current_step % ticks_in_day
        current_day = current_step // ticks_in_day
        day_of_week = current_day % 7  # 0-4 weekday, 5-6 weekend
        is_weekend = day_of_week >= 5
        
        # Count trips in the current day
        trips_in_current_day = sum(1 for request in commuter.requests.values() 
                                if request['start_time'] // ticks_in_day == current_day)
        
        # Determine the current time period
        if 36 <= current_day_tick < 60:  # 6:30am-10am
            time_period = 'morning_peak'
        elif 60 <= current_day_tick < 90:  # 10am-3pm
            time_period = 'midday'
        elif 90 <= current_day_tick < 114:  # 3pm-7pm
            time_period = 'evening_peak'
        elif 114 <= current_day_tick < 126:  # 7pm-9pm
            time_period = 'evening'
        else:  # Overnight
            time_period = 'night'
        
        # Count trips in the current time period (same day)
        if time_period == 'morning_peak':
            trips_in_period = sum(1 for request in commuter.requests.values() 
                                if request['start_time'] // ticks_in_day == current_day 
                                and 36 <= request['start_time'] % ticks_in_day < 60)
        elif time_period == 'evening_peak':
            trips_in_period = sum(1 for request in commuter.requests.values() 
                                if request['start_time'] // ticks_in_day == current_day 
                                and 90 <= request['start_time'] % ticks_in_day < 114)
        else:
            # For other periods, just check within a 24-tick window (4 hours)
            period_start = max(0, current_day_tick - 12)
            period_end = min(ticks_in_day - 1, current_day_tick + 12)
            trips_in_period = sum(1 for request in commuter.requests.values() 
                                if request['start_time'] // ticks_in_day == current_day 
                                and period_start <= request['start_time'] % ticks_in_day < period_end)
        
        # Base daily limits depending on payment scheme and demographics
        if commuter.payment_scheme == 'subscription':
            base_daily_limit = 8  # Subscription users make more trips
        else:  # PAYG
            base_daily_limit = 6
        
        # Demographic adjustments
        if commuter.income_level == 'high':
            base_daily_limit += 2  # High income makes more trips
        elif commuter.income_level == 'low':
            base_daily_limit -= 1  # Low income makes fewer trips
        
        if commuter.age >= 65 or commuter.has_disability:
            base_daily_limit -= 1  # Older/disabled people may make fewer trips
        
        # Weekend adjustments
        if is_weekend:
            # Fewer trips on weekends, especially for work commuters
            base_daily_limit = max(3, base_daily_limit - 2)
        
        # Determine max trips per time period
        if time_period == 'morning_peak' or time_period == 'evening_peak':
            # During peak, allow more trips for commuters with subscription
            if commuter.payment_scheme == 'subscription':
                period_limit = 3
            else:
                period_limit = 2
        elif time_period == 'midday':
            # Some commuters may do multiple midday trips (lunch, meetings)
            period_limit = 2
        elif time_period == 'evening':
            # Evening activities
            period_limit = 2
        else:  # night
            # Fewer trips at night
            period_limit = 1
        
        # Check if commuter is already traveling (has active non-finished trips)
        has_active_travel = any(request['status'] not in ['finished', 'expired'] 
                            for request in commuter.requests.values())
        if has_active_travel:
            return False  # Can't start a new trip while already traveling
        
        # Special case: If commuter hasn't made any trips yet today, 
        # always allow at least one trip
        if trips_in_current_day == 0:
            # But consider time of day - most people start their day in morning
            if time_period == 'morning_peak' or (is_weekend and time_period == 'midday'):
                return True
            elif time_period == 'night':
                # Very few trips start during night
                return random.random() < 0.1
            else:
                # Some probability of starting first trip later in the day
                return random.random() < 0.4
        
        # Check against limits
        return (trips_in_current_day < base_daily_limit and 
                trips_in_period < period_limit)

    def all_requests_finished(self, commuter):
        return all(request['status'] in ['finished', 'expired'] for request in commuter.requests.values())

    def create_time_based_trip(self, current_step, commuter):
        """Create trip requests with realistic timing patterns based on time of day and commuter attributes"""
        
        # Only create new trips if commuter isn't already traveling
        if not commuter.requests or self.all_requests_finished(commuter):
            # Skip if daily trip limit reached
            if not self.should_create_trip(commuter, current_step):
                return False
                
            # Get time of day context (144 ticks = 1 day, tick 0 = midnight)
            ticks_in_day = 144
            current_day_tick = current_step % ticks_in_day
            current_day = current_step // ticks_in_day
            day_of_week = current_day % 7  # 0-4 weekday, 5-6 weekend
            is_weekend = day_of_week >= 5
            
            # Baseline probabilities influenced by demographics
            base_probability = 0.05
            
            # Income level adjustments (higher income = more trips)
            if commuter.income_level == 'high':
                base_probability *= 1.5
            elif commuter.income_level == 'low':
                base_probability *= 0.8
                
            # Age/disability adjustments
            if commuter.age >= 65 or commuter.has_disability:
                base_probability *= 0.7
                
            # PAYG vs Subscription adjustments
            if commuter.payment_scheme == 'subscription':
                base_probability *= 1.3
                
            # Time of day probability distribution (using normal curves around peak times)
            time_multiplier = 1.0
            
            # Morning peak (centered at 8am = tick 48)
            if not is_weekend:
                morning_peak_center = 48
                morning_intensity = math.exp(-0.5 * ((current_day_tick - morning_peak_center) / 8) ** 2)
                
                # Evening peak (centered at 5:30pm = tick 105)
                evening_peak_center = 105
                evening_intensity = math.exp(-0.5 * ((current_day_tick - evening_peak_center) / 10) ** 2)
                
                # Combine the peaks
                time_multiplier = max(morning_intensity * 3.0, evening_intensity * 2.5, 0.2)
            else:
                # Weekend pattern (midday peak)
                midday_peak_center = 72  # Noon
                midday_intensity = math.exp(-0.5 * ((current_day_tick - midday_peak_center) / 16) ** 2)
                time_multiplier = midday_intensity * 1.5
            
            # Final probability
            trip_probability = base_probability * time_multiplier
            
            # Decide whether to create a trip
            if random.random() < trip_probability:
                # Determine trip purpose based on time of day
                if not is_weekend and current_day_tick < 60:  # Morning on weekday
                    purpose_weights = {'work': 0.7, 'school': 0.2, 'shopping': 0.05, 'medical': 0.03, 'leisure': 0.02}
                elif not is_weekend and current_day_tick >= 90 and current_day_tick < 114:  # Evening on weekday
                    purpose_weights = {'work': 0.1, 'school': 0.05, 'shopping': 0.3, 'leisure': 0.5, 'medical': 0.05}
                elif is_weekend:  # Weekend
                    purpose_weights = {'shopping': 0.4, 'leisure': 0.4, 'medical': 0.05, 'work': 0.1, 'school': 0.05}
                else:  # Middle of weekday
                    purpose_weights = {'work': 0.2, 'school': 0.1, 'shopping': 0.3, 'medical': 0.2, 'leisure': 0.2}
                
                # Select purpose based on weights
                purposes = list(purpose_weights.keys())
                weights = list(purpose_weights.values())
                travel_purpose = random.choices(purposes, weights=weights)[0]
                
                # Generate destination based on purpose
                origin = commuter.location
                destination = self.get_purpose_based_destination(travel_purpose, origin, commuter)
                
                # Set trip timing - more realistic start time distribution
                # Most people don't plan trips exactly at current time - add a buffer
                min_delay = 1
                max_delay = 5
                
                if travel_purpose in ['work', 'school'] and current_day_tick < 30:
                    # Early morning work/school trips might be planned further ahead
                    max_delay = 4
                    
                start_time = current_step + min_delay + random.randint(0, max_delay)
                
                # Double-check that the start_time is valid before creating the request
                if start_time > current_step + 5:
                    # If somehow we ended up with a time too far ahead, adjust it
                    start_time = current_step + 5
                # Create the request
                request_id = uuid.uuid4()
                commuter.create_request(request_id, origin, destination, start_time, travel_purpose)
                return True
            
            return False

    def get_purpose_based_destination(self, purpose, origin, commuter):
        """Generate a realistic destination based on trip purpose"""
        
        # Avoid very short trips
        min_distance = 5
        
        if purpose == 'work' or purpose == 'school':
            # Work trips tend to be to specific locations - can be far away from home
            # Higher income might have longer commutes to specialized jobs
            if commuter.income_level == 'high':
                distance_factor = 0.7  # Can travel farther
            else:
                distance_factor = 0.5
                
            # Get a destination with preference toward business/office districts
            # For simplicity, let's say central areas (middle of grid) are business districts
            grid_center_x = self.grid_width // 2
            grid_center_y = self.grid_height // 2
            
            # Generate with bias toward center
            while True:
                # Biased random point toward center
                x_offset = random.normalvariate(0, self.grid_width * distance_factor)
                y_offset = random.normalvariate(0, self.grid_height * distance_factor)
                
                dest_x = min(max(0, int(grid_center_x + x_offset)), self.grid_width - 1)
                dest_y = min(max(0, int(grid_center_y + y_offset)), self.grid_height - 1)
                destination = (dest_x, dest_y)
                
                # Check if it's far enough away
                distance = math.sqrt((destination[0] - origin[0])**2 + (destination[1] - origin[1])**2)
                if distance >= min_distance:
                    break
            
            return destination
            
        elif purpose == 'shopping':
            # Shopping destinations are often in specific areas like malls or centers
            # Let's say there are a few shopping centers scattered around
            
            # Simple approach - just use a few preset locations
            shopping_centers = [
                (self.grid_width // 4, self.grid_height // 4),
                (self.grid_width // 4, 3 * self.grid_height // 4),
                (3 * self.grid_width // 4, self.grid_height // 4),
                (3 * self.grid_width // 4, 3 * self.grid_height // 4),
                (self.grid_width // 2, self.grid_height // 2)
            ]
            
            # Find closest and furthest shopping centers
            distances = [math.sqrt((center[0] - origin[0])**2 + (center[1] - origin[1])**2) 
                        for center in shopping_centers]
            centers_by_distance = [center for _, center in sorted(zip(distances, shopping_centers))]
            
            # People typically go to reasonably close shopping centers, not always the very closest
            # nor the furthest
            if len(centers_by_distance) >= 3:
                # Skip the very closest and furthest
                choice_centers = centers_by_distance[1:-1]
                return random.choice(choice_centers)
            else:
                return random.choice(shopping_centers)
        
        elif purpose == 'leisure':
            # Leisure trips can be to parks, entertainment venues
            # For simplicity, let's say there are some leisure areas in the outer regions
            
            # Generate a point biased toward edges
            while True:
                if random.random() < 0.5:
                    # Bias toward edges on x-axis
                    x = random.choice([random.randint(0, self.grid_width // 4),
                                    random.randint(3 * self.grid_width // 4, self.grid_width - 1)])
                    y = random.randint(0, self.grid_height - 1)
                else:
                    # Bias toward edges on y-axis
                    y = random.choice([random.randint(0, self.grid_height // 4),
                                    random.randint(3 * self.grid_height // 4, self.grid_height - 1)])
                    x = random.randint(0, self.grid_width - 1)
                    
                destination = (x, y)
                
                # Check if it's far enough away
                distance = math.sqrt((destination[0] - origin[0])**2 + (destination[1] - origin[1])**2)
                if distance >= min_distance:
                    break
                    
            return destination
            
        elif purpose == 'medical':
            # Medical destinations like hospitals are specific and few
            # Let's define a few hospitals
            hospitals = [
                (self.grid_width // 2, self.grid_height // 4),
                (self.grid_width // 4, self.grid_height // 2),
                (3 * self.grid_width // 4, self.grid_height // 2)
            ]
            return random.choice(hospitals)
        
        else:
            # Fallback - completely random
            while True:
                x = random.randint(0, self.grid_width - 1)
                y = random.randint(0, self.grid_height - 1)
                destination = (x, y)
                
                # Check if it's far enough away
                distance = math.sqrt((destination[0] - origin[0])**2 + (destination[1] - origin[1])**2)
                if distance >= min_distance:
                    break
                    
            return destination
            
    def get_current_step(self):
        return self.current_step
    
    def step(self):
        self.current_step += 1
        #print(f"Step {self.current_step}")
        # Update the system's time step
        self.service_provider_agent.update_time_steps()

        availability_dict = self.service_provider_agent.initialize_availability(self.current_step - 1)

        for commuter in self.commuter_agents:
            # Create new request based on commuter's needs
            #print(f"Creating new request for Commuter {commuter.unique_id} at Step {self.current_step}")
            self.create_time_based_trip(self.current_step, commuter)
            
            # First, clean up stale requests
            for request_id, request in list(commuter.requests.items()):
                if request['status'] == 'active' and request['start_time'] < self.current_step:
                    # This request is stale - mark it as expired rather than trying to process it
                    request['status'] = 'expired'
                    print(f"Marking stale request {request_id} as expired (start_time: {request['start_time']}, current_step: {self.current_step})")
                    # Add this code to update database
                    try:
                        from sqlalchemy.orm import sessionmaker
                        from agent_service_provider_initialisation_03 import ServiceBookingLog
                        
                        Session = sessionmaker(bind=self.db_engine)
                        with Session() as session:
                            # Check if this request is already in the database
                            existing = session.query(ServiceBookingLog).filter_by(
                                request_id=str(request_id)
                            ).first()
                            
                            if existing:
                                # Update existing record
                                existing.status = 'expired'
                                session.commit()
                            else:
                                # For expired requests that never got a booking, create a minimal record
                                new_record = ServiceBookingLog(
                                    commuter_id=commuter.unique_id,
                                    payment_scheme=commuter.payment_scheme,
                                    request_id=str(request_id),
                                    start_time=request['start_time'],
                                    record_company_name='none',  # No company was selected
                                    route_details={"route": "none"},  # Minimal route info
                                    total_price=0.0,
                                    maas_surcharge=0.0,
                                    total_time=0.0,
                                    origin_coordinates=request['origin'],
                                    destination_coordinates=request['destination'],
                                    status='expired'  # Mark as expired
                                )
                                session.add(new_record)
                                session.commit()
                    except Exception as e:
                        print(f"Error updating expired status in database: {e}")
            # Now process valid active requests
            for request_id, request in list(commuter.requests.items()):
                try:
                    if request['status'] == 'active' and request['start_time'] >= self.current_step:
                        # MaaS agent generates travel options for each request
                        travel_options_without_MaaS = self.maas_agent.options_without_maas(
                            request_id, request['start_time'], request['origin'], request['destination'])
                        
                        travel_options_with_MaaS = self.maas_agent.maas_options(
                            commuter.payment_scheme, request_id, request['start_time'], 
                            request['origin'], request['destination'])

                        # Process the rest as before...
                        ranked_options = commuter.rank_service_options(
                            travel_options_without_MaaS, travel_options_with_MaaS, request_id)
                        
                        if ranked_options:
                            booking_success, availability_dict = self.maas_agent.book_service(
                                request_id, ranked_options, self.current_step, availability_dict)
                            if not booking_success:
                                print(f"Booking for request {request_id} was not successful.")
                        else:
                            print(f"No viable options for request {request_id}.")
                except Exception as e:
                    print(f"Error in MobilityModel processing request {request_id}: {str(e)}")
                    # Add this to help diagnose the specific request causing problems
                    print(f"Problem request details: status={request['status']}, start_time={request['start_time']}, current_step={self.current_step}")
                    
            self.update_commuter_info_log(commuter)
            commuter.update_location()
            commuter.check_travel_status()  # Once the commuter arrives at the destination, increase the availability back

        # Update availability based on bookings
        #print(f"Updating availability after bookings")
        self.service_provider_agent.update_availability()
        # Call dynamic_pricing_share to update pricing based on demand
        #print(f"Calling dynamic pricing update")
        self.service_provider_agent.dynamic_pricing_share()
            # Probability of inserting random traffic
        # Time-varying background traffic - always insert but vary amount by time of day
        with self.Session() as session:
            #print(f"Inserting random background traffic at Step {self.current_step}")
            # Insert random traffic with a certain number of routes
            num_routes_inserted = self.maas_agent.insert_time_varying_traffic(session)
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