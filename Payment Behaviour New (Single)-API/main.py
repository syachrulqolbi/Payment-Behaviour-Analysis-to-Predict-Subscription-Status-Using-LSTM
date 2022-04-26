from fastapi import FastAPI
import tensorflow as tf
from pydantic import BaseModel
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from scipy.interpolate import interp1d

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

    billing_10_period: str
    billing_9_period: str
    billing_8_period: str
    billing_7_period: str
    billing_6_period: str
    billing_5_period: str
    billing_4_period: str
    billing_3_period: str
    billing_2_period: str
    billing_1_period: str

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

def convert_churn_index(billing_3, billing_2, billing_1):
    if int(datetime.today().strftime("%d")) > 20 and billing_2 == 0 and billing_1 == 0:
        return 0
    elif int(datetime.today().strftime("%d")) <= 20 and billing_3 == 0 and billing_2 == 0:
        return 0
    else:
        return 1

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
    period = [data.billing_10_period,
              data.billing_9_period,
              data.billing_8_period,
              data.billing_7_period,
              data.billing_6_period,
              data.billing_5_period,
              data.billing_4_period,
              data.billing_3_period,
              data.billing_2_period,
              data.billing_1_period]
    status = []

    for i in range(10):
        paymentDate[i] = int(paymentDate[i][-2:]) if int(paymentDate[i][-4:-2]) == int(period[i][-2:]) and int(paymentDate[i][-2:]) != 0 else (int(32) if int(paymentDate[i][-2:]) == 0 else int(31))
        status.append(3 if paymentDate[i] >= 1 and paymentDate[i] < 11 else (2 if paymentDate[i] >= 11 and paymentDate[i] < 22 else (1 if paymentDate[i] >= 22 and paymentDate[i] < 32 else 0)))
        amountTotal[i] = float(amountTotal[i].replace(".", "").replace(",", ".")) if paymentDate[i] != 32 else float(0)

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
             sum(amountTotal) / len(amountTotal),
             convert_churn_index(status[7], status[8], status[9])]
    data2 = np.array([data2])

    print(data.astype(int))
    print(data2.astype(int))

    if data2[0, 0] < 10 and data2[0, 0] > 1:
        if int(datetime.today().strftime("%d")) <= 20 and data[:, :, 1][0][9] == 0:
            x_temp = np.linspace(1, int(data2[0, 0] - 1), num = int(data2[0, 0] - 1), endpoint = True)
            xnew = np.linspace(1, int(data2[0, 0] - 1), num = 10, endpoint = True)

            #amountTotal
            y_temp = data[0, -int(data2[0, 0]):-1, 0]
            f = interp1d(x_temp, y_temp, kind = "linear")
            data[0, :, 0] = f(xnew)

            #paymentDate
            y_temp = data[0, -int(data2[0, 0]):-1, 1]
            f = interp1d(x_temp, y_temp, kind = "linear")
            data[0, :, 1] = f(xnew)

            #status
            y_temp = data[0, -int(data2[0, 0]):-1, 2]
            f = interp1d(x_temp, y_temp, kind = "linear")
            data[0, :, 2] = f(xnew)
        else:
            x_temp = np.linspace(1, int(data2[0, 0]), num = int(data2[0, 0]), endpoint = True)
            xnew = np.linspace(1, int(data2[0, 0]), num = 10, endpoint = True)

            #amountTotal
            y_temp = data[0, -int(data2[0, 0]):, 0]
            f = interp1d(x_temp, y_temp, kind = "linear")
            data[0, :, 0] = f(xnew)

            #paymentDate
            y_temp = data[0, -int(data2[0, 0]):, 1]
            f = interp1d(x_temp, y_temp, kind = "linear")
            data[0, :, 1] = f(xnew)

            #status
            y_temp = data[0, -int(data2[0, 0]):, 2]
            f = interp1d(x_temp, y_temp, kind = "linear")
            data[0, :, 2] = f(xnew)

    elif data2[0, 0] == 1:
        #amountTotal
        data[0, i, 0] = np.array([data[0, -1, 0] for i in range(10)])

        #paymentDate
        data[0, i, 1] = np.array([data[0, -1, 1] for i in range(10)])

        #status
        data[0, i, 2] = np.array([data[0, -1, 2] for i in range(10)])

    data[:, :, 0] = data[:, :, 0] - data[:, :, 0].mean(axis = 1)
    data[:, :, 1] = data[:, :, 1] - data[:, :, 1].mean(axis = 1)

    data[:, :, 0] = sc.transform(data[:, :, 0])
    data[:, :, 1] = sc_pd.transform(data[:, :, 1])
    data[:, :, 2] = sc_stat.transform(data[:, :, 2])

    data2[:, 1] = sc_mean.transform(data2[:, 1].reshape(-1, 1)).reshape(data2.shape[0])

    lstm_A_test = model_A.predict([data[:, :, 0], data2])
    lstm_B_test = model_B.predict([data[:, :, 1], np.concatenate((data2[:, 0:1], data2[:, 2:3]), axis = 1)])
    lstm_C_test = model_C.predict([data[:, :, 2], np.concatenate((data2[:, 0:1], data2[:, 2:3]), axis = 1)])

    print(lstm_A_test)
    print(lstm_B_test)
    print(lstm_C_test)
    
    result = model.predict(np.concatenate((np.concatenate((lstm_A_test, lstm_B_test), axis = 1), lstm_C_test), axis = 1))

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
        "Predict Description" : status,
        "Amount Total Predict Description" : "Pola Data Churn" if lstm_A_test < 0.25 else "-",
        "Payment Date Predict Description" : "Pola Data Churn" if lstm_B_test < 0.25 else "-",
        "Loyality Description" : "Loyal" if lstm_C_test >= 0.85 else ("Agak Loyal" if (lstm_C_test >= 0.65 and lstm_C_test < 0.85) else "Telat Bayar"),
        "Activity Description" : "Tidak Aktif 2 Bulan Terakhir" if data2[:, 2] == 0 else "-"
    }