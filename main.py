from process import process_data
from fastapi import FastAPI
from pydantic import BaseModel
from constant import MODEL_PATH

import pandas as pd
import numpy as np
import pickle

class ScroingItem(BaseModel):
    gender:str
    SeniorCitizen:str
    Partner:str
    Dependents:str
    tenure:int
    PhoneService:str
    MultipleLines:str
    InternetService:str
    OnlineSecurity:str
    OnlineBackup:str
    DeviceProtection:str
    TechSupport:str
    StreamingTV:str
    StreamingMovies:str
    Contract:str
    PaperlessBilling:str
    PaymentMethod:str
    MonthlyCharges:float
    TotalCharges:float

model = pickle.load(open(MODEL_PATH, "rb"))

app = FastAPI()

@app.post('/')
async def scoring_endpoint(item:ScroingItem):
    df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    yhat = model.predict(process_data(dataframe=df))
    return {"prediction":yhat.tolist()}
