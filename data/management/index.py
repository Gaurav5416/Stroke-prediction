import os
from dotenv import load_dotenv

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import QueuePool


load_dotenv()

engine = create_engine(
    os.getenv('DB_URL'),
    echo=True, 
    pool_size=5, 
    max_overflow=-1,
    pool_recycle = 3600, 
    pool_pre_ping= True, 
    connect_args={
        "connect_timeout" : 60,
        "keepalives" : 1, 
        "keepalives_idle" : 30, 
        "keepalives_interval" : 10, 
        "keepalives_count" : 5,
    },
)

Session = sessionmaker(bind = engine)
connection = engine.connect()

Base = declarative_base()

from sqlalchemy import String, Column, Sequence, SmallInteger, Integer, Float

class Stroke(Base):
    __tablename__ = "stroke"
    id = Column(SmallInteger, Sequence("stroke_id_seq_1"), primary_key=True)
    gender = Column(String)
    age = Column(SmallInteger)
    hypertension = Column(SmallInteger)
    heart_disease = Column(SmallInteger)
    married = Column(String)
    work_type = Column(String)
    residence = Column(String)
    glucose_level = Column(Float)
    bmi = Column(Float)
    smoking = Column(String)
    stroke = Column(SmallInteger)


Base.metadata.create_all(bind=engine)
connection.close()