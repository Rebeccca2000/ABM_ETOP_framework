from sqlalchemy import Boolean, Column, Integer, String, Float, ForeignKey, create_engine, PrimaryKeyConstraint
from sqlalchemy.orm import declarative_base
from sqlalchemy.exc import OperationalError
import time
from sqlalchemy.types import JSON
# from database_01 import UberLike1_cpacity, UberLike1_price, UberLike2_cpacity, \
# UberLike2_price, subsidy_bike, BikeShare1_capacity, BikeShare1_price, \
# BikeShare2_capacity, BikeShare2_price, DB_CONNECTION_STRING
Base = declarative_base()

# BikeShare1_price -= subsidy_bike # Apply the subsidy amount
# BikeShare2_price -= subsidy_bike

class TransportModes(Base):
    __tablename__ = 'transport_modes'
    mode_id = Column(Integer, primary_key=True)
    mode_type = Column(String(50), nullable=False)

class UberLike1(Base):
    __tablename__ = 'UberLike1'
    provider_id = Column(Integer)
    mode_id = Column(Integer, ForeignKey('transport_modes.mode_id'))
    company_name = Column(String(100), nullable=False)
    base_price = Column(Float, nullable=False)
    capacity = Column(Integer, nullable=False)
    current_price_0 = Column(Float, nullable=False)
    current_price_1 = Column(Float, nullable=False)
    current_price_2 = Column(Float, nullable=False)
    current_price_3 = Column(Float, nullable=False)
    current_price_4 = Column(Float, nullable=False)
    current_price_5 = Column(Float, nullable=False)
    availability_0 = Column(Integer, nullable=False)
    availability_1 = Column(Integer, nullable=False)
    availability_2 = Column(Integer, nullable=False)
    availability_3 = Column(Integer, nullable=False)
    availability_4 = Column(Integer, nullable=False)
    availability_5 = Column(Integer, nullable=False)
    step_count = Column(Integer, nullable=False)
    __table_args__ = (PrimaryKeyConstraint('provider_id', 'step_count', name='share_service_pk'),)

class UberLike2(Base):
    __tablename__ = 'UberLike2'
    provider_id = Column(Integer)
    mode_id = Column(Integer, ForeignKey('transport_modes.mode_id'))
    company_name = Column(String(100), nullable=False)
    base_price = Column(Float, nullable=False)
    capacity = Column(Integer, nullable=False)
    current_price_0 = Column(Float, nullable=False)
    current_price_1 = Column(Float, nullable=False)
    current_price_2 = Column(Float, nullable=False)
    current_price_3 = Column(Float, nullable=False)
    current_price_4 = Column(Float, nullable=False)
    current_price_5 = Column(Float, nullable=False)
    availability_0 = Column(Integer, nullable=False)
    availability_1 = Column(Integer, nullable=False)
    availability_2 = Column(Integer, nullable=False)
    availability_3 = Column(Integer, nullable=False)
    availability_4 = Column(Integer, nullable=False)
    availability_5 = Column(Integer, nullable=False)
    step_count = Column(Integer, nullable=False)
    __table_args__ = (PrimaryKeyConstraint('provider_id', 'step_count', name='share_service_pk'),)

class BikeShare1(Base):
    __tablename__ = 'BikeShare1'
    provider_id = Column(Integer)
    mode_id = Column(Integer, ForeignKey('transport_modes.mode_id'))
    company_name = Column(String(100), nullable=False)
    base_price = Column(Float, nullable=False)
    capacity = Column(Integer, nullable=False)
    current_price_0 = Column(Float, nullable=False)
    current_price_1 = Column(Float, nullable=False)
    current_price_2 = Column(Float, nullable=False)
    current_price_3 = Column(Float, nullable=False)
    current_price_4 = Column(Float, nullable=False)
    current_price_5 = Column(Float, nullable=False)
    availability_0 = Column(Integer, nullable=False)
    availability_1 = Column(Integer, nullable=False)
    availability_2 = Column(Integer, nullable=False)
    availability_3 = Column(Integer, nullable=False)
    availability_4 = Column(Integer, nullable=False)
    availability_5 = Column(Integer, nullable=False)
    step_count = Column(Integer, nullable=False)
    __table_args__ = (PrimaryKeyConstraint('provider_id', 'step_count', name='share_service_pk'),)

class BikeShare2(Base):
    __tablename__ = 'BikeShare2'
    provider_id = Column(Integer)
    mode_id = Column(Integer, ForeignKey('transport_modes.mode_id'))
    company_name = Column(String(100), nullable=False)
    base_price = Column(Float, nullable=False)
    capacity = Column(Integer, nullable=False)
    current_price_0 = Column(Float, nullable=False)
    current_price_1 = Column(Float, nullable=False)
    current_price_2 = Column(Float, nullable=False)
    current_price_3 = Column(Float, nullable=False)
    current_price_4 = Column(Float, nullable=False)
    current_price_5 = Column(Float, nullable=False)
    availability_0 = Column(Integer, nullable=False)
    availability_1 = Column(Integer, nullable=False)
    availability_2 = Column(Integer, nullable=False)
    availability_3 = Column(Integer, nullable=False)
    availability_4 = Column(Integer, nullable=False)
    availability_5 = Column(Integer, nullable=False)
    step_count = Column(Integer, nullable=False)
    __table_args__ = (PrimaryKeyConstraint('provider_id', 'step_count', name='share_service_pk'),)

class ShareServiceBookingLog(Base):
    __tablename__ = 'share_service_booking_log'
    commuter_id = Column(Integer, nullable=True)
    request_id = Column(String, primary_key=True)  # Store UUID as String
    mode_id = Column(Integer, ForeignKey('transport_modes.mode_id'), nullable=False)
    provider_id = Column(Integer, nullable=False)
    company_name = Column(String(100), nullable=False)
    start_time = Column(Integer, nullable=False)
    duration = Column(Integer, nullable=False)
    affected_steps = Column(JSON, nullable=False)  # Store list as JSON
    route_details = Column(JSON, nullable=False)  # Store detailed route as JSON   

class ServiceBookingLog(Base):
    __tablename__ = 'service_booking_log'
    
    commuter_id = Column(Integer, primary_key=True)
    payment_scheme = Column(String, primary_key=True)
    request_id = Column(String, primary_key=True)  # Store UUID as String
    start_time = Column(Integer, nullable=False)
    record_company_name = Column(String(100), nullable=False)  # if it is public then put public
    route_details = Column(JSON, nullable=False)  # Store detailed route as JSON
    total_price = Column(Float, nullable=False)
    maas_surcharge = Column(Float, nullable=False)
    total_time = Column(Float, nullable=False)  # New column to store the total time taken for each trip
    origin_coordinates = Column(JSON, nullable=False)  # New column to store origin coordinates
    destination_coordinates = Column(JSON, nullable=False)  # New column to store destination coordinates

    # New columns for MaaS options
    to_station = Column(JSON, nullable=True)  # Store 'to station' info as JSON, nullable because non-MaaS options won't use it
    to_destination = Column(JSON, nullable=True)  # Store 'to destination' info as JSON, nullable because non-MaaS options won't use it
    # New column for government subsidy
    government_subsidy = Column(JSON, nullable=True)
    
class CommuterInfoLog(Base):
    __tablename__ = 'commuter_info_log'
    commuter_id = Column(Integer, primary_key=True)
    location = Column(JSON, nullable=False)  # Store location as JSON list [x, y]
    age = Column(Integer, nullable=False)
    income_level = Column(String, nullable=False)
    has_disability = Column(Boolean, nullable=False)
    tech_access = Column(Boolean, nullable=False)
    health_status = Column(String, nullable=False)
    payment_scheme = Column(String, nullable=False)
    # New columns to store requests and services_owned
    requests = Column(JSON, nullable=True)  # Store requests as JSON
    services_owned = Column(JSON, nullable=True)  # Store services_owned as JSON

class GovernmentSubsidyLog(Base):
    __tablename__ = 'government_subsidy_log'
    commuter_id = Column(Integer, primary_key=True)
    service_id = Column(Integer, nullable=False)
    commuter_age = Column(Integer, nullable=False)
    health_condition = Column(String, nullable=False)
    income_level = Column(String, nullable=False)
    mode_choice = Column(String, nullable=False)
    total_original_price = Column(Float, nullable=False)
    subsidy_id = Column(String, nullable=False)  # Can store multiple subsidy IDs as a comma-separated string or use JSON
    price_after_subsidy = Column(Float, nullable=False)
    policy_name = Column(String, nullable=False)  # Can store multiple subsidy IDs as a comma-separated string or use JSON
    target_group = Column(String, nullable=False)

# # Create the engine and reset the database
# engine = create_engine(DB_CONNECTION_STRING)

# # Function to reset the database
# def reset_database(engine):
#     while True:
#         try:
#             Base.metadata.drop_all(engine)  # Drop all tables
#             Base.metadata.create_all(engine)  # Create tables
#             break
#         except OperationalError:
#             print("Database is locked, retrying...")
#             time.sleep(1)

# reset_database(engine)

# Create a new session
# Session = sessionmaker(bind=engine)
# session = Session()

# # Hardcoded data for transport_modes table
# transport_modes_data = [
#     {'mode_id': 1, 'mode_type': 'train'},
#     {'mode_id': 2, 'mode_type': 'bus'},
#     {'mode_id': 3, 'mode_type': 'car'},
#     {'mode_id': 4, 'mode_type': 'bike'},
#     {'mode_id': 5, 'mode_type': 'walk'}
# ]

# Uber_Like_1 = [
#     {'provider_id': 1, 
#     'mode_id': 3, 
#     'company_name': 'UberLike1', 
#     'base_price': UberLike1_price, 
#     'capacity': UberLike1_cpacity, 
#     'current_price_0': UberLike1_price,
#     'current_price_1': UberLike1_price,
#     'current_price_2': UberLike1_price,
#     'current_price_3': UberLike1_price,
#     'current_price_4': UberLike1_price,
#     'current_price_5': UberLike1_price,
#     'availability_0': UberLike1_cpacity,
#     'availability_1': UberLike1_cpacity,
#     'availability_2': UberLike1_cpacity,
#     'availability_3': UberLike1_cpacity,
#     'availability_4': UberLike1_cpacity,
#     'availability_5': UberLike1_cpacity, 
#     'step_count': 0},
# ]

# Uber_Like_2 = [
#     {'provider_id': 2, 
#     'mode_id': 3, 
#     'company_name': 'UberLike2', 
#     'base_price': UberLike2_price, 
#     'capacity': UberLike2_cpacity, 
#     'current_price_0': UberLike2_price,
#     'current_price_1': UberLike2_price,
#     'current_price_2': UberLike2_price,
#     'current_price_3': UberLike2_price,
#     'current_price_4': UberLike2_price,
#     'current_price_5': UberLike2_price,
#     'availability_0': UberLike2_cpacity,
#     'availability_1': UberLike2_cpacity,
#     'availability_2': UberLike2_cpacity,
#     'availability_3': UberLike2_cpacity,
#     'availability_4': UberLike2_cpacity,
#     'availability_5': UberLike2_cpacity, 
#     'step_count': 0},
# ]

# Bike_share_1 = [
#     {'provider_id': 3, 
#     'mode_id': 4, 
#     'company_name': 'BikeShare1', 
#     'base_price': BikeShare1_price, 
#     'capacity': BikeShare1_capacity, 
#     'current_price_0': BikeShare1_price,
#     'current_price_1': BikeShare1_price,
#     'current_price_2': BikeShare1_price,
#     'current_price_3': BikeShare1_price,
#     'current_price_4': BikeShare1_price,
#     'current_price_5': BikeShare1_price,
#     'availability_0': BikeShare1_capacity,
#     'availability_1': BikeShare1_capacity,
#     'availability_2': BikeShare1_capacity,
#     'availability_3': BikeShare1_capacity,
#     'availability_4': BikeShare1_capacity,
#     'availability_5': BikeShare1_capacity, 
#     'step_count': 0},
# ]

# Bike_share_2 = [
#     {'provider_id': 4, 
#     'mode_id': 4, 
#     'company_name': 'BikeShare2', 
#     'base_price': BikeShare2_price, 
#     'capacity': BikeShare2_capacity, 
#     'current_price_0': BikeShare2_price,
#     'current_price_1': BikeShare2_price,
#     'current_price_2': BikeShare2_price,
#     'current_price_3': BikeShare2_price,
#     'current_price_4': BikeShare2_price,
#     'current_price_5': BikeShare2_price,
#     'availability_0': BikeShare2_capacity,
#     'availability_1': BikeShare2_capacity,
#     'availability_2': BikeShare2_capacity,
#     'availability_3': BikeShare2_capacity,
#     'availability_4': BikeShare2_capacity,
#     'availability_5': BikeShare2_capacity, 
#     'step_count': 0},
# ]

# # Hardcoded data for public_service table
# public_service_data = [
#     # public_service_0
#     {'vehicle_id': 1, 'vehicle_name': 'Train 1', 'mode_id': 1, 'capacity': 880, 'unit_price_off_peak': 0.8, 'unit_price_on_peak': 0.8, 'route_id': 'RT1', 'status': 'active', 'availability': 880, 'step_count': 0, 'standard_wait_time': 1, 'unit_travel_time': 1},
#     {'vehicle_id': 2, 'vehicle_name': 'Bus 1', 'mode_id': 2, 'capacity': 80, 'unit_price_off_peak': 0.5, 'unit_price_on_peak': 0.8, 'route_id': 'RB1', 'status': 'active', 'availability': 80, 'step_count': 0, 'standard_wait_time': 1, 'unit_travel_time': 0.5},
#     {'vehicle_id': 3, 'vehicle_name': 'Bus 2', 'mode_id': 2, 'capacity': 80, 'unit_price_off_peak': 0.5, 'unit_price_on_peak': 0.8, 'route_id': 'RB2', 'status': 'active', 'availability': 80, 'step_count': 0, 'standard_wait_time': 1, 'unit_travel_time': 0.5},
#     {'vehicle_id': 4, 'vehicle_name': 'Bus 3', 'mode_id': 2, 'capacity': 80, 'unit_price_off_peak': 0.5, 'unit_price_on_peak': 0.8, 'route_id': 'RB3', 'status': 'active', 'availability': 80, 'step_count': 0, 'standard_wait_time': 1, 'unit_travel_time': 0.5},
# ]


# # Insert data into transport_modes table
# for row in transport_modes_data:
#     mode = TransportModes(mode_id=row['mode_id'], mode_type=row['mode_type'])
#     session.add(mode)
# session.commit()

# # Insert data into Uber_Like_1 table
# for row in Uber_Like_1:
#     service = UberLike1(
#         provider_id=row['provider_id'],
#         mode_id=row['mode_id'],
#         company_name=row['company_name'],
#         base_price=row['base_price'],
#         capacity=row['capacity'],
#         current_price_0 = row['current_price_0'],
#         current_price_1 = row['current_price_1'],
#         current_price_2 = row['current_price_2'],
#         current_price_3 = row['current_price_3'],
#         current_price_4 = row['current_price_4'],
#         current_price_5 = row['current_price_5'],
#         availability_0 = row['availability_0'],
#         availability_1 = row['availability_1'],
#         availability_2 = row['availability_2'],
#         availability_3 = row['availability_3'],
#         availability_4 = row['availability_4'],
#         availability_5 = row['availability_5'],
#         step_count = row['step_count']
#     )
#     session.add(service)
# session.commit()

# # Insert data into Uber_Like_2 table
# for row in Uber_Like_2:
#     service = UberLike2(
#         provider_id=row['provider_id'],
#         mode_id=row['mode_id'],
#         company_name=row['company_name'],
#         base_price=row['base_price'],
#         capacity=row['capacity'],
#         current_price_0 = row['current_price_0'],
#         current_price_1 = row['current_price_1'],
#         current_price_2 = row['current_price_2'],
#         current_price_3 = row['current_price_3'],
#         current_price_4 = row['current_price_4'],
#         current_price_5 = row['current_price_5'],
#         availability_0 = row['availability_0'],
#         availability_1 = row['availability_1'],
#         availability_2 = row['availability_2'],
#         availability_3 = row['availability_3'],
#         availability_4 = row['availability_4'],
#         availability_5 = row['availability_5'],
#         step_count = row['step_count']
#     )
#     session.add(service)
# session.commit()


# # Insert data into Bike_share_1 table
# for row in Bike_share_1:
#     service = BikeShare1(
#         provider_id=row['provider_id'],
#         mode_id=row['mode_id'],
#         company_name=row['company_name'],
#         base_price=row['base_price'],
#         capacity=row['capacity'],
#         current_price_0 = row['current_price_0'],
#         current_price_1 = row['current_price_1'],
#         current_price_2 = row['current_price_2'],
#         current_price_3 = row['current_price_3'],
#         current_price_4 = row['current_price_4'],
#         current_price_5 = row['current_price_5'],
#         availability_0 = row['availability_0'],
#         availability_1 = row['availability_1'],
#         availability_2 = row['availability_2'],
#         availability_3 = row['availability_3'],
#         availability_4 = row['availability_4'],
#         availability_5 = row['availability_5'],
#         step_count = row['step_count']
#     )
#     session.add(service)
# session.commit()


# # Insert data into Bike_share_1 table
# for row in Bike_share_2:
#     service = BikeShare2(
#         provider_id=row['provider_id'],
#         mode_id=row['mode_id'],
#         company_name=row['company_name'],
#         base_price=row['base_price'],
#         capacity=row['capacity'],
#         current_price_0 = row['current_price_0'],
#         current_price_1 = row['current_price_1'],
#         current_price_2 = row['current_price_2'],
#         current_price_3 = row['current_price_3'],
#         current_price_4 = row['current_price_4'],
#         current_price_5 = row['current_price_5'],
#         availability_0 = row['availability_0'],
#         availability_1 = row['availability_1'],
#         availability_2 = row['availability_2'],
#         availability_3 = row['availability_3'],
#         availability_4 = row['availability_4'],
#         availability_5 = row['availability_5'],
#         step_count = row['step_count']
#     )
#     session.add(service)
# session.commit()


# Reset database function with session management
def reset_database(engine, session, uber_like1_capacity, uber_like1_price, uber_like2_capacity, uber_like2_price, 
                   bike_share1_capacity, bike_share1_price, bike_share2_capacity, bike_share2_price):
    while True:
        try:
            # Drop and recreate all tables
            Base.metadata.drop_all(engine)
            Base.metadata.create_all(engine)

            # Insert transport modes and dynamic service provider data
            insert_transport_modes(session)
            insert_uber_like1(session, uber_like1_capacity, uber_like1_price)
            insert_uber_like2(session, uber_like2_capacity, uber_like2_price)
            insert_bike_share1(session, bike_share1_capacity, bike_share1_price)
            insert_bike_share2(session, bike_share2_capacity, bike_share2_price)
            
            break
        except OperationalError:
            print("Database is locked, retrying...")
            time.sleep(1)

# Insert transport modes into the database
def insert_transport_modes(session):
    transport_modes_data = [
        {'mode_id': 1, 'mode_type': 'train'},
        {'mode_id': 2, 'mode_type': 'bus'},
        {'mode_id': 3, 'mode_type': 'car'},
        {'mode_id': 4, 'mode_type': 'bike'},
        {'mode_id': 5, 'mode_type': 'walk'}
    ]
    for row in transport_modes_data:
        mode = TransportModes(mode_id=row['mode_id'], mode_type=row['mode_type'])
        session.add(mode)
    session.commit()

# Insert UberLike1 data into the database
def insert_uber_like1(session, capacity, price):
    Uber_Like_1 = [
        {
            'provider_id': 1,
            'mode_id': 3, 
            'company_name': 'UberLike1', 
            'base_price': price, 
            'capacity': capacity, 
            'current_price_0': price,
            'current_price_1': price,
            'current_price_2': price,
            'current_price_3': price,
            'current_price_4': price,
            'current_price_5': price,
            'availability_0': capacity,
            'availability_1': capacity,
            'availability_2': capacity,
            'availability_3': capacity,
            'availability_4': capacity,
            'availability_5': capacity,
            'step_count': 0
        },
    ]
    for row in Uber_Like_1:
        service = UberLike1(
            provider_id=row['provider_id'],
            mode_id=row['mode_id'],
            company_name=row['company_name'],
            base_price=row['base_price'],
            capacity=row['capacity'],
            current_price_0=row['current_price_0'],
            current_price_1=row['current_price_1'],
            current_price_2=row['current_price_2'],
            current_price_3=row['current_price_3'],
            current_price_4=row['current_price_4'],
            current_price_5=row['current_price_5'],
            availability_0=row['availability_0'],
            availability_1=row['availability_1'],
            availability_2=row['availability_2'],
            availability_3=row['availability_3'],
            availability_4=row['availability_4'],
            availability_5=row['availability_5'],
            step_count=row['step_count']
        )
        session.add(service)
    session.commit()

# Insert UberLike2 data into the database
def insert_uber_like2(session, capacity, price):
    Uber_Like_2 = [
        {
            'provider_id': 2, 
            'mode_id': 3, 
            'company_name': 'UberLike2', 
            'base_price': price, 
            'capacity': capacity, 
            'current_price_0': price,
            'current_price_1': price,
            'current_price_2': price,
            'current_price_3': price,
            'current_price_4': price,
            'current_price_5': price,
            'availability_0': capacity,
            'availability_1': capacity,
            'availability_2': capacity,
            'availability_3': capacity,
            'availability_4': capacity,
            'availability_5': capacity,
            'step_count': 0
        },
    ]
    for row in Uber_Like_2:
        service = UberLike2(
            provider_id=row['provider_id'],
            mode_id=row['mode_id'],
            company_name=row['company_name'],
            base_price=row['base_price'],
            capacity=row['capacity'],
            current_price_0=row['current_price_0'],
            current_price_1=row['current_price_1'],
            current_price_2=row['current_price_2'],
            current_price_3=row['current_price_3'],
            current_price_4=row['current_price_4'],
            current_price_5=row['current_price_5'],
            availability_0=row['availability_0'],
            availability_1=row['availability_1'],
            availability_2=row['availability_2'],
            availability_3=row['availability_3'],
            availability_4=row['availability_4'],
            availability_5=row['availability_5'],
            step_count=row['step_count']
        )
        session.add(service)
    session.commit()

# Insert BikeShare1 data into the database
def insert_bike_share1(session, capacity, price):
    Bike_Share_1 = [
        {
            'provider_id': 3, 
            'mode_id': 4, 
            'company_name': 'BikeShare1', 
            'base_price': price, 
            'capacity': capacity, 
            'current_price_0': price,
            'current_price_1': price,
            'current_price_2': price,
            'current_price_3': price,
            'current_price_4': price,
            'current_price_5': price,
            'availability_0': capacity,
            'availability_1': capacity,
            'availability_2': capacity,
            'availability_3': capacity,
            'availability_4': capacity,
            'availability_5': capacity,
            'step_count': 0
        },
    ]
    for row in Bike_Share_1:
        service = BikeShare1(
            provider_id=row['provider_id'],
            mode_id=row['mode_id'],
            company_name=row['company_name'],
            base_price=row['base_price'],
            capacity=row['capacity'],
            current_price_0=row['current_price_0'],
            current_price_1=row['current_price_1'],
            current_price_2=row['current_price_2'],
            current_price_3=row['current_price_3'],
            current_price_4=row['current_price_4'],
            current_price_5=row['current_price_5'],
            availability_0=row['availability_0'],
            availability_1=row['availability_1'],
            availability_2=row['availability_2'],
            availability_3=row['availability_3'],
            availability_4=row['availability_4'],
            availability_5=row['availability_5'],
            step_count=row['step_count']
        )
        session.add(service)
    session.commit()

# Insert BikeShare2 data into the database
def insert_bike_share2(session, capacity, price):
    Bike_Share_2 = [
        {
            'provider_id': 4, 
            'mode_id': 4, 
            'company_name': 'BikeShare2', 
            'base_price': price, 
            'capacity': capacity, 
            'current_price_0': price,
            'current_price_1': price,
            'current_price_2': price,
            'current_price_3': price,
            'current_price_4': price,
            'current_price_5': price,
            'availability_0': capacity,
            'availability_1': capacity,
            'availability_2': capacity,
            'availability_3': capacity,
            'availability_4': capacity,
            'availability_5': capacity,
            'step_count': 0
        },
    ]
    for row in Bike_Share_2:
        service = BikeShare2(
            provider_id=row['provider_id'],
            mode_id=row['mode_id'],
            company_name=row['company_name'],
            base_price=row['base_price'],
            capacity=row['capacity'],
            current_price_0=row['current_price_0'],
            current_price_1=row['current_price_1'],
            current_price_2=row['current_price_2'],
            current_price_3=row['current_price_3'],
            current_price_4=row['current_price_4'],
            current_price_5=row['current_price_5'],
            availability_0=row['availability_0'],
            availability_1=row['availability_1'],
            availability_2=row['availability_2'],
            availability_3=row['availability_3'],
            availability_4=row['availability_4'],
            availability_5=row['availability_5'],
            step_count=row['step_count']
        )
        session.add(service)
    session.commit()
