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
import uuid
import random
import database_01 as db
from database_01 import num_commuters, grid_width, grid_height, income_weights, \
        health_weights, payment_weights, age_distribution, disability_weights, \
        tech_access_weights, DB_CONNECTION_STRING,SIMULATION_STEPS
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
    def __init__(self, db_connection_string, num_commuters, width=grid_width, height=grid_height, data_income_weights = income_weights , \
                 data_health_weights = health_weights, data_payment_weights = payment_weights, data_age_distribution = age_distribution,\
                data_disability_weights = disability_weights, data_tech_access_weights = tech_access_weights):
        super().__init__()
        self.db_connection_string = db_connection_string
        self.db_engine = create_engine(db_connection_string)
        self.grid = MultiGrid(width, height, torus=False)
        self.schedule = RandomActivation(self)
        self.service_provider_agent = ServiceProvider(unique_id=1, model=self, db_connection_string=db_connection_string)
        self.schedule.add(self.service_provider_agent)
        self.Session = scoped_session(sessionmaker(bind=self.db_engine))
        # Initialize current_step to 0
        self.current_step = 0

        # Add station agents
        for mode, mode_stations in db.stations.items():
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
        age_distribution = data_age_distribution
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
            income_level = random.choices(income_levels, data_income_weights)[0]  # Weighted random choice for income level
            health_status = random.choices(health_statuses, data_health_weights)[0]  # Weighted random choice for health status
            payment_scheme = random.choices(payment_schemes, data_payment_weights)[0]  # Weighted random choice for payment scheme
            age = get_random_age()
            has_disability = random.choices([True, False], data_disability_weights)[0]  # 15% chance of having a disability
            tech_access = random.choices([True, False], data_tech_access_weights)[0]  # 96.2% chance of having tech access

            commuter = Commuter(
                unique_id=i + 2,
                model=self,
                commuter_location=(random.randint(0, width - 1), random.randint(0, height - 1)),
                age=age,
                income_level=income_level,
                has_disability=has_disability,
                tech_access=tech_access,
                health_status=health_status,
                payment_scheme=payment_scheme  # Add payment scheme to commuter
            )
            self.commuter_agents.append(commuter)
            self.schedule.add(commuter)
            self.grid.place_agent(commuter, commuter.location)
            self.record_commuter_info(commuter)  # Record commuter info in the database

        self.maas_agent = MaaS(unique_id=num_commuters + 2, model=self, service_provider_agent=self.service_provider_agent, commuter_agents=self.commuter_agents)
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

    def check_is_peak(self, current_ticks):
        if (36 <= current_ticks % 144 < 60) or (90 <= current_ticks % 144 < 114):
            return True
        return False

    def should_create_trip(self, commuter):
        # Check the number of trips created in the current day (144 steps)
        trips_in_current_day = sum(1 for request in commuter.requests.values() if request['start_time'] // 144 == self.current_step // 144)
        if commuter.payment_scheme == 'PAYG':
            return trips_in_current_day < 2
        elif commuter.payment_scheme == 'subscription':
            return trips_in_current_day < 4
        return False

    def all_requests_finished(self, commuter):
        return all(request['status'] == 'finished' for request in commuter.requests.values())

    def create_new_request(self, current_step, commuter):
        if not commuter.requests or self.all_requests_finished(commuter):
            if not self.should_create_trip(commuter):
                return  # Do not create a new request if the commuter has reached their daily trip limit

            if self.check_is_peak(self.current_step):
                create_trip_probability = 0.3  # Higher probability during peak hours
            else:
                create_trip_probability = 0.04  # Lower probability during non-peak hours

            if random.random() < create_trip_probability:
                request_id = uuid.uuid4()
                origin = commuter.location
                destination = (random.randint(0, self.grid.width - 1), random.randint(0, self.grid.height - 1))
                start_time = current_step + random.randint(1, 5)
                travel_purpose = random.choice(['work', 'school', 'shopping', 'leisure', 'medical'])
                commuter.create_request(request_id, origin, destination, start_time, travel_purpose)
                print(f"Commuter {commuter.unique_id} created request from {origin} to {destination} starting at {start_time}")
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
            # print(f"The availability_dict current situation is {availability_dict}")
            # Create new request based on commuter's needs
            self.create_new_request(self.current_step, commuter)

            for request_id, request in list(commuter.requests.items()):
                try:
                    if request['status'] == 'active':
                        # MaaS agent generates travel options for each request
                        travel_options = self.maas_agent.generate_travel_options(commuter.payment_scheme, request_id, request['start_time'], request['origin'], request['destination'])
                        print(f"Generated travel options for request {request_id}: {travel_options}")  # Debug print

                        # Commuter ranks the travel options
                        ranked_options = commuter.rank_service_options(travel_options, request_id)
                        print(f"Ranked options for request {request_id}: {ranked_options}")  # Debug print

                        if ranked_options != []:  # Only attempt booking if there are ranked options available
                            print(f"Attempting to book service for request {request_id}")
                            # MaaS agent attempts to book the highest-ranked service using ranked_options
                            booking_success, availability_dict = self.maas_agent.book_service(request_id, ranked_options, self.current_step, availability_dict)
                            if booking_success:
                                print(f"Booking for request {request_id} was successful.")
                            else:
                                print(f"Booking for request {request_id} was not successful.")
                        else:
                            print(f"No viable options for request {request_id}.")
                except Exception as e:
                    print(f"Error processing request {request_id}: {e}")

            commuter.update_location()
            commuter.check_travel_status()  # Once the commuter arrives at the destination, increase the availability back

        # Update availability based on bookings
        self.service_provider_agent.update_availability()
        # Call dynamic_pricing_share to update pricing based on demand
        self.service_provider_agent.dynamic_pricing_share()

        self.schedule.step()
        
    def run_model(self, num_steps):
        for _ in range(num_steps):
            self.step()


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

grid = CanvasGrid(agent_portrayal, grid_width, grid_height, 1500, 1500)



server = ModularServer(
    MobilityModel,
    [grid, CommuteCountElement()],
    "Mobility Model",
    {"db_connection_string": DB_CONNECTION_STRING, "num_commuters": num_commuters}
)

# server.port = 8521
# server.launch()

if __name__ == "__main__":
    model = MobilityModel(db_connection_string=DB_CONNECTION_STRING, num_commuters=num_commuters)
    model.run_model(SIMULATION_STEPS)

else:
    server.launch()