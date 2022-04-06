from fastapi import FastAPI
import tensorflow as tf
from pydantic import BaseModel
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import StandardScaler

app = FastAPI()

model = tf.keras.models.load_model("model_lstm_bagging.h5")
model_A = tf.keras.models.load_model("model_lstm_amounttotal.h5")
model_B = tf.keras.models.load_model("model_lstm_paymentdate.h5")
model_C = tf.keras.models.load_model("model_lstm_status.h5")
sc = joblib.load("scaler.gz")
sc_pd = joblib.load("scaler_pd.gz")
sc_stat = joblib.load("scaler_stat.gz")
sc_mean = joblib.load("scaler_mean.gz")

class Data(BaseModel):
    billing_10_amountTotal: str
    billing_9_amountTotal: str
    billing_8_amountTotal: str
    billing_7_amountTotal: str
    billing_6_amountTotal: str
    billing_5_amountTotal: str
    billing_4_amountTotal: str
    billing_3_amountTotal: str
    billing_2_amountTotal: str
    billing_1_amountTotal: str

    billing_10_paymentDate: str
    billing_9_paymentDate: str
    billing_8_paymentDate: str
    billing_7_paymentDate: str
    billing_6_paymentDate: str
    billing_5_paymentDate: str
    billing_4_paymentDate: str
    billing_3_paymentDate: str
    billing_2_paymentDate: str
    billing_1_paymentDate: str

def convert_LTV(billing_10, billing_9, billing_8, billing_7, billing_6, billing_5, billing_4, billing_3, billing_2, billing_1):
    if billing_10 != 0:
        return 10
    elif billing_10 == 0 and billing_9 != 0:
        return 9
    elif billing_10 == 0 and billing_9 == 0 and billing_8 != 0:
        return 8
    elif billing_10 == 0 and billing_9 == 0 and billing_8 == 0 and billing_7 != 0:
        return 7
    elif billing_10 == 0 and billing_9 == 0 and billing_8 == 0 and billing_7 == 0 and billing_6 != 0:
        return 6
    elif billing_10 == 0 and billing_9 == 0 and billing_8 == 0 and billing_7 == 0 and billing_6 == 0 and billing_5 != 0:
        return 5
    elif billing_10 == 0 and billing_9 == 0 and billing_8 == 0 and billing_7 == 0 and billing_6 == 0 and billing_5 == 0 and billing_4 != 0:
        return 4
    elif billing_10 == 0 and billing_9 == 0 and billing_8 == 0 and billing_7 == 0 and billing_6 == 0 and billing_5 == 0 and billing_4 == 0 and billing_3 != 0:
        return 3
    elif billing_10 == 0 and billing_9 == 0 and billing_8 == 0 and billing_7 == 0 and billing_6 == 0 and billing_5 == 0 and billing_4 == 0 and billing_3 == 0 and billing_2 != 0:
        return 2
    elif billing_10 == 0 and billing_9 == 0 and billing_8 == 0 and billing_7 == 0 and billing_6 == 0 and billing_5 == 0 and billing_4 == 0 and billing_3 == 0 and billing_2 == 0 and billing_1 != 0:
        return 1
    else:
        return 0

def convert_status_index(billing_10, billing_9, billing_8, billing_7, billing_6, billing_5, billing_4, billing_3, billing_2, billing_1):
    count = 0
    if billing_10 == 1:
        count += 1
    elif billing_10 == 2:
        count += 2
    else:
        count += 0
    
    if billing_9 == 1:
        count += 1
    elif billing_9 == 2:
        count += 2
    else:
        count += 0
    
    if billing_8 == 1:
        count += 1
    elif billing_8 == 2:
        count += 2
    else:
        count += 0
    
    if billing_7 == 1:
        count += 1
    elif billing_7 == 2:
        count += 2
    else:
        count += 0
    
    if billing_6 == 1:
        count += 1
    elif billing_6 == 2:
        count += 2
    else:
        count += 0
      
    if billing_5 == 1:
        count += 1
    elif billing_5 == 2:
        count += 2
    else:
        count += 0
    
    if billing_4 == 1:
        count += 1
    elif billing_4 == 2:
        count += 2
    else:
        count += 0
    
    if billing_3 == 1:
        count += 1
    elif billing_3 == 2:
        count += 2
    else:
        count += 0

    if billing_2 == 1:
        count += 1
    elif billing_2 == 2:
        count += 2
    else:
        count += 0
    
    if billing_1 == 1:
        count += 1
    elif billing_1 == 2:
        count += 2
    else:
        count += 0
    
    return count

def convert_churn_index(billing_2, billing_1):
    if billing_2 == 0 and billing_1 == 0:
        return 0
    else:
        return 1

def convert_predict(predict):
    temp_arg = predict.argmax(axis = 1)
    temp_max = predict.max(axis = 1)
    temp_mul = np.where(temp_arg == 1, 0.25, np.where(temp_arg == 4, 0.15, 0.20))
    temp = temp_mul * temp_max
    temp_arg = np.where(temp_arg == 0, 0, np.where(temp_arg == 1, 0.25, np.where(temp_arg == 2, 0.45, np.where(temp_arg == 3, 0.65, 0.85))))
    predict = temp + temp_arg

    return predict

@app.get("/predict")
def predict(data: Data):
    amountTotal = [data.billing_10_amountTotal,
                   data.billing_9_amountTotal,
                   data.billing_8_amountTotal,
                   data.billing_7_amountTotal,
                   data.billing_6_amountTotal,
                   data.billing_5_amountTotal,
                   data.billing_4_amountTotal,
                   data.billing_3_amountTotal,
                   data.billing_2_amountTotal,
                   data.billing_1_amountTotal]
    paymentDate = [data.billing_10_paymentDate,
                   data.billing_9_paymentDate,
                   data.billing_8_paymentDate,
                   data.billing_7_paymentDate,
                   data.billing_6_paymentDate,
                   data.billing_5_paymentDate,
                   data.billing_4_paymentDate,
                   data.billing_3_paymentDate,
                   data.billing_2_paymentDate,
                   data.billing_1_paymentDate]
    status = []

    for i in range(10):
        amountTotal[i] = float(amountTotal[i].replace(".", "").replace(",", "."))
        paymentDate[i] = int(paymentDate[i][-2:])
        status.append(2 if paymentDate[i] >= 1 and paymentDate[i] < 22 else (1 if paymentDate[i] >= 22 and paymentDate[i] < 32 else 0))

    data = [[amountTotal[0], paymentDate[0], status[0]],
            [amountTotal[1], paymentDate[1], status[1]],
            [amountTotal[2], paymentDate[2], status[2]],
            [amountTotal[3], paymentDate[3], status[3]],
            [amountTotal[4], paymentDate[4], status[4]],
            [amountTotal[5], paymentDate[5], status[5]],
            [amountTotal[6], paymentDate[6], status[6]],
            [amountTotal[7], paymentDate[7], status[7]],
            [amountTotal[8], paymentDate[8], status[8]],
            [amountTotal[9], paymentDate[9], status[9]]]
            
    data = np.array([data])
    
    data2 = [convert_LTV(status[0], status[1], status[2], status[3], status[4], status[5], status[6], status[7], status[8], status[9]), 
             np.mean(data[:, :, 0]),
	     convert_churn_index(data[:, :, 1][0][8], data[:, :, 1][0][9])]
    data2 = np.array([data2])

    data[:, :, 0] = data[:, :, 0] - data[:, :, 0].mean(axis = 1)

    data[:, :, 0] = sc.transform(data[:, :, 0])
    data[:, :, 1] = sc_pd.transform(data[:, :, 1])
    data[:, :, 2] = sc_stat.transform(data[:, :, 2])

    data2[:, 1] = sc_mean.transform(data2[:, 1].reshape(-1, 1)).reshape(data2.shape[0])

    lstm_A_test = model_A.predict([data[:, :, 0:1]])
    lstm_B_test = model_B.predict([data[:, :, 1:2]])
    lstm_C_test = model_C.predict([data[:, :, 2:3]])
    
    predict = model.predict(np.concatenate((np.concatenate((np.concatenate((data2, lstm_A_test), axis = 1), lstm_B_test), axis = 1), lstm_C_test), axis = 1))
    result = convert_predict(predict)

    if result < 0.25:
        status = "Churn"
    elif result >= 0.25 and result < 0.45:
        status = "Cenderung Churn"
    elif result >= 0.45 and result < 0.65:
        status = "Ragu-ragu"
    elif result >= 0.565 and result < 0.85:
        status = "Agak Loyal"
    else:
        status = "Loyal"
    
    return {
        "Percentage": round(float(result), 4) * 100,
        "Predict Description" : status,
        "Amount Total Predict Description" : "Pola Data Churn" if lstm_A_test < 0.25 else "-",
        "Payment Date Predict Description" : "Pola Data Churn" if lstm_B_test < 0.25 else "-",
        "Status Predict Description" : "Pola Data Churn" if lstm_C_test < 0.25 else "-",
        "Activity Description" : "Tidak Aktif 2 Bulan Terakhir" if data2[:, 2] == 0 else "-"
    }