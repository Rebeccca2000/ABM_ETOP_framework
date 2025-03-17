from sqlalchemy import Boolean, Column, Integer, String, Float, ForeignKey, create_engine, PrimaryKeyConstraint
from sqlalchemy.orm import declarative_base
from sqlalchemy.exc import OperationalError
import time
from sqlalchemy.types import JSON

Base = declarative_base()

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
    # Give each table a unique constraint name
    __table_args__ = (PrimaryKeyConstraint('provider_id', 'step_count', name='uberlike1_pk'),)

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
    # Give each table a unique constraint name
    __table_args__ = (PrimaryKeyConstraint('provider_id', 'step_count', name='uberlike2_pk'),)

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
    # Give each table a unique constraint name
    __table_args__ = (PrimaryKeyConstraint('provider_id', 'step_count', name='bikeshare1_pk'),)

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
    # Give each table a unique constraint name
    __table_args__ = (PrimaryKeyConstraint('provider_id', 'step_count', name='bikeshare2_pk'),)

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

class SubsidyUsageLog(Base):
    __tablename__ = 'subsidy_usage_log'
    
    id = Column(Integer, primary_key=True)
    commuter_id = Column(Integer, nullable=False)
    request_id = Column(String, nullable=False)
    subsidy_amount = Column(Float, nullable=False)
    mode = Column(String, nullable=False)
    timestamp = Column(Integer, nullable=False)
    day = Column(Integer, nullable=False)
    week = Column(Integer, nullable=False)
    month = Column(Integer, nullable=False)
    period_type = Column(String, nullable=False)  # 'daily', 'weekly', or 'monthly'

# Reset database function with schema support
def reset_database(engine, session, uber_like1_capacity, uber_like1_price, uber_like2_capacity, uber_like2_price, 
                   bike_share1_capacity, bike_share1_price, bike_share2_capacity, bike_share2_price, schema):
    while True:
        try:
            # Set schema if provided
            if schema:
                # When dealing with PostgreSQL schemas
                for table in Base.metadata.tables.values():
                    table.schema = schema
            
            # Drop and recreate all tables
            Base.metadata.drop_all(engine, checkfirst=True)
            Base.metadata.create_all(engine)

            # Insert transport modes and dynamic service provider data
            insert_transport_modes(session)
            insert_uber_like1(session, uber_like1_capacity, uber_like1_price)
            insert_uber_like2(session, uber_like2_capacity, uber_like2_price)
            insert_bike_share1(session, bike_share1_capacity, bike_share1_price)
            insert_bike_share2(session, bike_share2_capacity, bike_share2_price)
            
            break
        except OperationalError as e:
            print(f"Database operation error: {e}")
            time.sleep(1)

# Insert transport modes into the database
def insert_transport_modes(session):
    transport_modes_data = [
        {'mode_id': 0, 'mode_type': 'background_traffic'}, 
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