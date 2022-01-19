from fastapi import FastAPI
import tensorflow as tf
from pydantic import BaseModel
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import StandardScaler
import math

app = FastAPI()

model = tf.keras.models.load_model("model_lstm.h5")
model2 = pickle.load(open('model_timeseriesforest', 'rb'))

class Data(BaseModel):
    billing_11_amountTotal: str
    billing_10_amountTotal: str
    billing_9_amountTotal: str
    billing_8_amountTotal: str
    billing_7_amountTotal: str
    billing_6_amountTotal: str
    billing_5_amountTotal: str
    billing_4_amountTotal: str
    billing_3_amountTotal: str
    billing_2_amountTotal: str

    billing_11_status: str
    billing_10_status: str
    billing_9_status: str
    billing_8_status: str
    billing_7_status: str
    billing_6_status: str
    billing_5_status: str
    billing_4_status: str
    billing_3_status: str
    billing_2_status: str

    billing_11_channel: str
    billing_10_channel: str
    billing_9_channel: str
    billing_8_channel: str
    billing_7_channel: str
    billing_6_channel: str
    billing_5_channel: str
    billing_4_channel: str
    billing_3_channel: str
    billing_2_channel: str

    billing_11_paymentDate: float
    billing_10_paymentDate: float
    billing_9_paymentDate: float
    billing_8_paymentDate: float
    billing_7_paymentDate: float
    billing_6_paymentDate: float
    billing_5_paymentDate: float
    billing_4_paymentDate: float
    billing_3_paymentDate: float
    billing_2_paymentDate: float

    paketradius: str

paket_class = pickle.load(open("encode_paket","rb"))
channel_class = pickle.load(open("encode_channel","rb"))

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

'''
def encode_paket(x):
    paket = np.zeros(94)
    if x in list(paket_class[0]):
        paket[list(paket_class[0]).index(x)] = 1
    else:
        paket[0] = 1
    return np.array([paket.astype(int)])
'''
def encode_paket(x):
    if x in list(paket_class.categories_[0]):
        paket = paket_class.transform([[x]])
    else:
        paket = paket_class.transform([["-"]])
    return paket[0]

def encode_late(x):
    paymentDate = x
    if paymentDate == 0:
        paymentDate = 0
    elif paymentDate > 0 and paymentDate < 21:
        paymentDate = 1
    else:
        paymentDate = 2
    return paymentDate

def scaling_amountTotal(billing_11_amountTotal, billing_10_amountTotal, billing_9_amountTotal, billing_8_amountTotal, billing_7_amountTotal, billing_6_amountTotal, billing_5_amountTotal, billing_4_amountTotal, billing_3_amountTotal, billing_2_amountTotal):
    x = np.array([billing_11_amountTotal,
                billing_10_amountTotal,
                billing_9_amountTotal,
                billing_8_amountTotal,
                billing_7_amountTotal,
                billing_6_amountTotal,
                billing_5_amountTotal,
                billing_4_amountTotal,
                billing_3_amountTotal,
		billing_2_amountTotal])
    sc = joblib.load("scaler.gz")
    return sc.transform(np.reshape(x, (1, -1)))

def encode_channel(x):
    if x in list(channel_class.classes_):
        channel = channel_class.transform([x])
    else:
        channel = channel_class.transform(["-"])
    return channel

@app.get("/predict")
def predict(data: Data):
    amount_total = scaling_amountTotal(encode_amountTotal(data.billing_11_amountTotal), 
                                    encode_amountTotal(data.billing_10_amountTotal), 
                                    encode_amountTotal(data.billing_9_amountTotal), 
                                    encode_amountTotal(data.billing_8_amountTotal), 
                                    encode_amountTotal(data.billing_7_amountTotal), 
                                    encode_amountTotal(data.billing_6_amountTotal), 
                                    encode_amountTotal(data.billing_5_amountTotal), 
                                    encode_amountTotal(data.billing_4_amountTotal), 
                                    encode_amountTotal(data.billing_3_amountTotal), 
                                    encode_amountTotal(data.billing_2_amountTotal))
    
    data1 = np.array([[[amount_total[0][0], encode_status(data.billing_11_status), encode_paymentDate(data.billing_11_paymentDate)],
                [amount_total[0][1], encode_status(data.billing_10_status), encode_paymentDate(data.billing_10_paymentDate)],
                [amount_total[0][2], encode_status(data.billing_9_status), encode_paymentDate(data.billing_9_paymentDate)],
                [amount_total[0][3], encode_status(data.billing_8_status), encode_paymentDate(data.billing_8_paymentDate)],
                [amount_total[0][4], encode_status(data.billing_7_status), encode_paymentDate(data.billing_7_paymentDate)],
                [amount_total[0][5], encode_status(data.billing_6_status), encode_paymentDate(data.billing_6_paymentDate)],
                [amount_total[0][6], encode_status(data.billing_5_status), encode_paymentDate(data.billing_5_paymentDate)],
                [amount_total[0][7], encode_status(data.billing_4_status), encode_paymentDate(data.billing_4_paymentDate)],
                [amount_total[0][8], encode_status(data.billing_3_status), encode_paymentDate(data.billing_3_paymentDate)],
                [amount_total[0][8], encode_status(data.billing_2_status), encode_paymentDate(data.billing_2_paymentDate)]]])
    data2 = encode_paket(data.paketradius)

    data3_1 = [amount_total[0][0], amount_total[0][1], amount_total[0][2], amount_total[0][3], amount_total[0][4], amount_total[0][5], amount_total[0][6], amount_total[0][7], amount_total[0][8], amount_total[0][9],
            encode_channel(data.billing_11_channel), encode_channel(data.billing_10_channel), encode_channel(data.billing_9_channel), encode_channel(data.billing_8_channel), encode_channel(data.billing_7_channel), encode_channel(data.billing_6_channel), encode_channel(data.billing_5_channel), encode_channel(data.billing_4_channel), encode_channel(data.billing_3_channel), encode_channel(data.billing_2_channel),
            encode_paymentDate(data.billing_11_paymentDate), encode_paymentDate(data.billing_10_paymentDate), encode_paymentDate(data.billing_9_paymentDate), encode_paymentDate(data.billing_8_paymentDate), encode_paymentDate(data.billing_7_paymentDate), encode_paymentDate(data.billing_6_paymentDate), encode_paymentDate(data.billing_5_paymentDate), encode_paymentDate(data.billing_4_paymentDate), encode_paymentDate(data.billing_3_paymentDate), encode_paymentDate(data.billing_2_paymentDate),
            encode_late(data.billing_11_paymentDate), encode_late(data.billing_10_paymentDate), encode_late(data.billing_9_paymentDate), encode_late(data.billing_8_paymentDate), encode_late(data.billing_7_paymentDate), encode_late(data.billing_6_paymentDate), encode_late(data.billing_5_paymentDate), encode_late(data.billing_4_paymentDate), encode_late(data.billing_3_paymentDate), encode_late(data.billing_2_paymentDate)]

    data3 = model2.predict([data3_1])

    data_test = [np.reshape(data1[0], (1, data1[0].shape[0], data1[0].shape[1])), np.reshape(data2, (1, data2.shape[0])), data3]

    result = model.predict(data_test)

    if result <= 0.13701576:
        status = "Churn"
    elif result > 0.13701576 and result <= 0.3460211:
        status = "Cenderung Churn"
    elif result > 0.3460211 and result <= 0.5993076:
        status = "Ragu-ragu"
    elif result > 0.5993076 and result <= 0.8439841:
        status = "Agak Loyal"
    else:
        status = "Loyal"
    
    return {
        "Percentage": round(float(result), 4) * 100,
        "Predict Description" : status
    }