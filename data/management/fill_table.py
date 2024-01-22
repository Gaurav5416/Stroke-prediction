import pandas as pd 
from index import Stroke, Session
import os
from dotenv import load_dotenv
load_dotenv()
datapath = os.getenv('datapath')
with Session.begin() as db:
    data = pd.read_csv(datapath)
    for Index, row in data.iterrows():
        stroke = Stroke(
            gender = row[0],
            age = row[1],
            hypertension = row[2],
            heart_disease = row[3],
            married = row[4],
            work_type = row[5],
            residence = row[6],
            glucose_level = row[7],
            bmi = row[8],
            smoking = row[9],
            stroke = row[10],
        )
        db.add(stroke)