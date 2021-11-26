from fastapi import FastAPI
import tensorflow as tf
from pydantic import BaseModel
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import StandardScaler
import math

app = FastAPI()

model = tf.keras.models.load_model("model_IHZ.h5")

class Data(BaseModel):
    amountTotal_9: str
    amountTotal_8: str
    amountTotal_7: str
    amountTotal_6: str
    amountTotal_5: str
    amountTotal_4: str
    amountTotal_3: str
    amountTotal_2: str
    amountTotal_1: str

    status_9: str
    status_8: str
    status_7: str
    status_6: str
    status_5: str
    status_4: str
    status_3: str
    status_2: str
    status_1: str

    paymentDate_9: float
    paymentDate_8: float
    paymentDate_7: float
    paymentDate_6: float
    paymentDate_5: float
    paymentDate_4: float
    paymentDate_3: float
    paymentDate_2: float
    paymentDate_1: float

    paketradius: str

paket_class = pickle.load(open("encode_paket.p","rb"))

def encode_amountTotal(x):
    amountTotal = float(str(x).replace(".", "").replace(",", "."))
    if math.isnan(amountTotal):
        amountTotal = 0
    elif amountTotal < 0:
        amountTotal = -1 * amountTotal
    return amountTotal

def encode_status(x):
    status = x
    if status == "ZERO BILLING":
        status = 0
    elif status == "UNPAID":
        status = 1
    elif status == "PAID":
        status = 2
    else:
        status = 0
    return status

def encode_paymentDate(x):
    paymentDate = x
    if math.isnan(paymentDate):
        paymentDate = 0
    paymentDate = int(float(paymentDate))
    paymentDate = int(str(paymentDate)[-2:])
    return paymentDate

def encode_paket(x):
    paket = np.zeros(94)
    if x in list(paket_class[0]):
        paket[list(paket_class[0]).index(x)] = 1
    else:
        paket[0] = 1
    return np.array([paket.astype(int)])

def scaling_amountTotal(amountTotal_9, amountTotal_8, amountTotal_7, amountTotal_6, amountTotal_5, amountTotal_4, amountTotal_3, amountTotal_2, amountTotal_1):
    x = np.array([amountTotal_9,
                amountTotal_8,
                amountTotal_7,
                amountTotal_6,
                amountTotal_5,
                amountTotal_4,
                amountTotal_3,
                amountTotal_2,
                amountTotal_1])
    sc = joblib.load("scaler.gz")
    return sc.transform(np.reshape(x, (1, -1)))

@app.get("/")
def home():
    return "Welcome"

@app.get("/predict")
def predict(data: Data):
    amount_total = scaling_amountTotal(encode_amountTotal(data.amountTotal_9), 
                                    encode_amountTotal(data.amountTotal_8), 
                                    encode_amountTotal(data.amountTotal_7), 
                                    encode_amountTotal(data.amountTotal_6), 
                                    encode_amountTotal(data.amountTotal_5), 
                                    encode_amountTotal(data.amountTotal_4), 
                                    encode_amountTotal(data.amountTotal_3), 
                                    encode_amountTotal(data.amountTotal_2), 
                                    encode_amountTotal(data.amountTotal_1))
    
    data1 = np.array([[[amount_total[0][0], encode_status(data.status_9), encode_paymentDate(data.paymentDate_9)],
                [amount_total[0][1], encode_status(data.status_8), encode_paymentDate(data.paymentDate_8)],
                [amount_total[0][2], encode_status(data.status_7), encode_paymentDate(data.paymentDate_7)],
                [amount_total[0][3], encode_status(data.status_6), encode_paymentDate(data.paymentDate_6)],
                [amount_total[0][4], encode_status(data.status_5), encode_paymentDate(data.paymentDate_5)],
                [amount_total[0][5], encode_status(data.status_4), encode_paymentDate(data.paymentDate_4)],
                [amount_total[0][6], encode_status(data.status_3), encode_paymentDate(data.paymentDate_3)],
                [amount_total[0][7], encode_status(data.status_2), encode_paymentDate(data.paymentDate_2)],
                [amount_total[0][8], encode_status(data.status_1), encode_paymentDate(data.paymentDate_1)]]])
    data2 = encode_paket(data.paketradius)

    data_test = [data1, data2]

    result = model.predict(data_test)
    if result < 0.25:
        status = "Churn"
    elif result >= 0.25 and result < 0.45:
        status = "Cenderung Churn"
    elif result >= 0.45 and result < 0.65:
        status = "Ragu-ragu"
    elif result >= 0.65 and result < 0.85:
        status = "Agak Loyal"
    else:
        status = "Loyal"
    return {
        "Percentage": round(float(result), 4) * 100,
        "Predict Description" : status
    }