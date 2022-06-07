import uvicorn
import csv
import pandas as pd
import tensorflow as tf
from pydantic import BaseModel
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import StandardScaler
import json

from typing import Optional
from fastapi import File, UploadFile, FastAPI
from fastapi import FastAPI, Form
from typing import List

app = FastAPI()
sc = joblib.load("scaler.gz")
sc_pd = joblib.load("scaler_pd.gz")
sc_stat = joblib.load("scaler_stat.gz")
sc_mean = joblib.load("scaler_mean.gz")
model = tf.keras.models.load_model("model_lstm_bagging.h5")
model_A = tf.keras.models.load_model("model_lstm_amounttotal.h5")
model_B = tf.keras.models.load_model("model_lstm_paymentdate.h5")
model_C = tf.keras.models.load_model("model_lstm_status.h5")

def convert_paymentDate(paymentDate, period):
    if paymentDate == 0:
        paymentDate = paymentDate
    elif str(paymentDate).split("-")[1] == "Jan" and str(period)[-2:] != "01":
        paymentDate = "31" + "-" + str(paymentDate).split("-")[1] + "-" + str(paymentDate).split("-")[2]
    elif str(paymentDate).split("-")[1] == "Feb" and str(period)[-2:] != "02":
        paymentDate = "31" + "-" + str(paymentDate).split("-")[1] + "-" + str(paymentDate).split("-")[2]
    elif str(paymentDate).split("-")[1] == "Mar" and str(period)[-2:] != "03":
        paymentDate = "31" + "-" + str(paymentDate).split("-")[1] + "-" + str(paymentDate).split("-")[2]
    elif str(paymentDate).split("-")[1] == "Apr" and str(period)[-2:] != "04":
        paymentDate = "31" + "-" + str(paymentDate).split("-")[1] + "-" + str(paymentDate).split("-")[2]
    elif str(paymentDate).split("-")[1] == "May" and str(period)[-2:] != "05":
        paymentDate = "31" + "-" + str(paymentDate).split("-")[1] + "-" + str(paymentDate).split("-")[2]
    elif str(paymentDate).split("-")[1] == "Jun" and str(period)[-2:] != "06":
        paymentDate = "31" + "-" + str(paymentDate).split("-")[1] + "-" + str(paymentDate).split("-")[2]
    elif str(paymentDate).split("-")[1] == "Jul" and str(period)[-2:] != "07":
        paymentDate = "31" + "-" + str(paymentDate).split("-")[1] + "-" + str(paymentDate).split("-")[2]
    elif str(paymentDate).split("-")[1] == "Aug" and str(period)[-2:] != "08":
        paymentDate = "31" + "-" + str(paymentDate).split("-")[1] + "-" + str(paymentDate).split("-")[2]
    elif str(paymentDate).split("-")[1] == "Sep" and str(period)[-2:] != "09":
        paymentDate = "31" + "-" + str(paymentDate).split("-")[1] + "-" + str(paymentDate).split("-")[2]
    elif str(paymentDate).split("-")[1] == "Oct" and str(period)[-2:] != "10":
        paymentDate = "31" + "-" + str(paymentDate).split("-")[1] + "-" + str(paymentDate).split("-")[2]
    elif str(paymentDate).split("-")[1] == "Nov" and str(period)[-2:] != "11":
        paymentDate = "31" + "-" + str(paymentDate).split("-")[1] + "-" + str(paymentDate).split("-")[2]
    elif str(paymentDate).split("-")[1] == "Dec" and str(period)[-2:] != "12":
        paymentDate = "31" + "-" + str(paymentDate).split("-")[1] + "-" + str(paymentDate).split("-")[2]

    return paymentDate

def check_date(df, array):
    array = array[::-1]
    for i in range(9):
        try:
            if int(str(array[i])[-2:]) == 1 and int(str(array[i + 1])[-2:]) != 12:
                array.insert(i + 1, int(str(int(str(array[i])[:-2]) - 1) + "12"))
                dict_temp = {"PERIODE": int(str(int(str(array[i])[:-2]) - 1) + "12")}
                df = df.append(dict_temp, ignore_index = True)

            elif int(str(array[i])[-2:]) != 1 and int(str(array[i])[-2:]) - 1 != int(str(array[i + 1])[-2:]):
                array.insert(i + 1, int(str(array[i])[:-2] + str(int(str(array[i])[-2:]) - 1).zfill(2)))
                dict_temp = {"PERIODE": int(str(array[i])[:-2] + str(int(str(array[i])[-2:]) - 1).zfill(2))}
                df = df.append(dict_temp, ignore_index = True)
        except:
            continue
        
    while len(array) < 10:
        if int(str(array[-1])[-2:]) != 1:
            array.append(int(str(array[-1])[:-2] + str(int(str(array[-1])[-2:]) - 1).zfill(2)))
            dict_temp = {"PERIODE": int(str(array[-1])[:-2] + str(int(str(array[-1])[-2:]) - 1).zfill(2))}
            df = df.append(dict_temp, ignore_index = True)
        else:
            array.append(int(str(int(str(array[-1])[:-2]) - 1) + "12"))
            dict_temp = {"PERIODE": int(str(int(str(array[-1])[:-2]) - 1) + "12")}
            df = df.append(dict_temp, ignore_index = True)
        
    return df

def shift(a):
    if a >= 1 and a < 11:
        return "LOYAL PAID"
    elif a >= 11 and a < 22:
        return "PAID"
    elif a >= 22 and a < 32:
        return "LATE PAID"
    else:
        return "ZERO BILLING"

def convert_churn_index(billing_2, billing_1):
    if billing_2 == 32 and billing_1 == 32:
        return 0
    else:
        return 1

def convert_LTV(billing_10, billing_9, billing_8, billing_7, billing_6, billing_5, billing_4, billing_3, billing_2, billing_1):
    if billing_10 != "ZERO BILLING":
        return 10
    elif billing_10 == "ZERO BILLING" and billing_9 != "ZERO BILLING":
        return 9
    elif billing_10 == "ZERO BILLING" and billing_9 == "ZERO BILLING" and billing_8 != "ZERO BILLING":
        return 8
    elif billing_10 == "ZERO BILLING" and billing_9 == "ZERO BILLING" and billing_8 == "ZERO BILLING" and billing_7 != "ZERO BILLING":
        return 7
    elif billing_10 == "ZERO BILLING" and billing_9 == "ZERO BILLING" and billing_8 == "ZERO BILLING" and billing_7 == "ZERO BILLING" and billing_6 != "ZERO BILLING":
        return 6
    elif billing_10 == "ZERO BILLING" and billing_9 == "ZERO BILLING" and billing_8 == "ZERO BILLING" and billing_7 == "ZERO BILLING" and billing_6 == "ZERO BILLING" and billing_5 != "ZERO BILLING":
        return 5
    elif billing_10 == "ZERO BILLING" and billing_9 == "ZERO BILLING" and billing_8 == "ZERO BILLING" and billing_7 == "ZERO BILLING" and billing_6 == "ZERO BILLING" and billing_5 == "ZERO BILLING" and billing_4 != "ZERO BILLING":
        return 4
    elif billing_10 == "ZERO BILLING" and billing_9 == "ZERO BILLING" and billing_8 == "ZERO BILLING" and billing_7 == "ZERO BILLING" and billing_6 == "ZERO BILLING" and billing_5 == "ZERO BILLING" and billing_4 == "ZERO BILLING" and billing_3 != "ZERO BILLING":
        return 3
    elif billing_10 == "ZERO BILLING" and billing_9 == "ZERO BILLING" and billing_8 == "ZERO BILLING" and billing_7 == "ZERO BILLING" and billing_6 == "ZERO BILLING" and billing_5 == "ZERO BILLING" and billing_4 == "ZERO BILLING" and billing_3 == "ZERO BILLING" and billing_2 != "ZERO BILLING":
        return 2
    elif billing_10 == "ZERO BILLING" and billing_9 == "ZERO BILLING" and billing_8 == "ZERO BILLING" and billing_7 == "ZERO BILLING" and billing_6 == "ZERO BILLING" and billing_5 == "ZERO BILLING" and billing_4 == "ZERO BILLING" and billing_3 == "ZERO BILLING" and billing_2 == "ZERO BILLING" and billing_1 != "ZERO BILLING":
        return 1
    else:
        return 0

@app.get("/predict/")
def predict(url: str = Form(...)):
    df = pd.read_csv(url)
    df = df[["PERIODE",
         "ND",
         "JUMLAH_TAGIHAN",
         "TANGGAL"]].sort_values(by = ["PERIODE", "TANGGAL"])
    
    df["TANGGAL"] = df.apply(lambda x: convert_paymentDate(x["TANGGAL"],
                                                           x["PERIODE"]), axis = 1)
    
    df = check_date(df, df["PERIODE"].unique())
    df = df.loc[df["PERIODE"] >= np.sort(df["PERIODE"].unique())[-10:][0]]

    df = df.pivot_table(index = "ND", columns = "PERIODE", aggfunc = np.sum).join(df.pivot_table(index = "ND", columns = "PERIODE", values = ["TANGGAL"], aggfunc = "last"), how = "outer")
    df.reset_index(level = 0, inplace = True)
    df = pd.DataFrame(df.values, columns = ["ND",
                                            "billing_10_amountTotal",
                                            "billing_9_amountTotal",
                                            "billing_8_amountTotal",
                                            "billing_7_amountTotal",
                                            "billing_6_amountTotal",
                                            "billing_5_amountTotal",
                                            "billing_4_amountTotal",
                                            "billing_3_amountTotal",
                                            "billing_2_amountTotal",
                                            "billing_1_amountTotal",
                                            "billing_10_paymentDate",
                                            "billing_9_paymentDate",
                                            "billing_8_paymentDate",
                                            "billing_7_paymentDate",
                                            "billing_6_paymentDate",
                                            "billing_5_paymentDate",
                                            "billing_4_paymentDate",
                                            "billing_3_paymentDate",
                                            "billing_2_paymentDate",
                                            "billing_1_paymentDate"])
    df = df.drop(df.loc[(df["ND"].isnull())].index)
    df = df.reset_index(drop = True)

    for month in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        df["billing_{}_paymentDate".format(month)].loc[df["billing_{}_paymentDate".format(month)].isnull()] = 0
        df["billing_{}_paymentDate".format(month)] = df["billing_{}_paymentDate".format(month)].apply(lambda x: str(x).split("-")[0])
        df["billing_{}_paymentDate".format(month)] = df["billing_{}_paymentDate".format(month)].astype(int)
        df["billing_{}_paymentDate".format(month)] = df["billing_{}_paymentDate".format(month)].apply(lambda x: 32 if x == 0 else x)
        df["billing_{}_amountTotal".format(month)].loc[df["billing_{}_amountTotal".format(month)].isnull()] = 0
        df["billing_{}_amountTotal".format(month)] = df["billing_{}_amountTotal".format(month)].astype(float)
        df["billing_{}_status".format(month)] = df.apply(lambda x: shift(x["billing_{}_paymentDate".format(month)]), axis = 1)
        df["billing_{}_status".format(month)] = df["billing_{}_status".format(month)].apply(lambda x: 3 if x == "LOYAL PAID" else (2 if x == "PAID" else (1 if x == "LATE PAID" else 0)))
        
    df["LTV"] = df.apply(lambda x: convert_LTV(x["billing_10_status"], 
                                               x["billing_9_status"], 
                                               x["billing_8_status"], 
                                               x["billing_7_status"], 
                                               x["billing_6_status"], 
                                               x["billing_5_status"], 
                                               x["billing_4_status"], 
                                               x["billing_3_status"], 
                                               x["billing_2_status"], 
                                               x["billing_1_status"]), axis = 1)
    df["churn_index"] = df.apply(lambda x: convert_churn_index(x["billing_2_paymentDate"], 
                                                               x["billing_1_paymentDate"]), axis = 1)
    df["mean"] = df[["billing_10_amountTotal",
                     "billing_9_amountTotal",
                     "billing_8_amountTotal",
                     "billing_7_amountTotal",
                     "billing_6_amountTotal",
                     "billing_5_amountTotal",
                     "billing_4_amountTotal",
                     "billing_3_amountTotal",
                     "billing_2_amountTotal",
                     "billing_1_amountTotal"]].replace(0, np.NaN).mean(axis = 1).values
    df["mean"].loc[df["mean"].isnull()] = 0

    data = np.reshape(df[["billing_10_amountTotal", "billing_10_paymentDate", "billing_10_status", "LTV", "mean", "churn_index",
                          "billing_9_amountTotal", "billing_9_paymentDate", "billing_9_status", "LTV", "mean", "churn_index",
                          "billing_8_amountTotal", "billing_8_paymentDate", "billing_8_status", "LTV", "mean", "churn_index",
                          "billing_7_amountTotal", "billing_7_paymentDate", "billing_7_status", "LTV", "mean", "churn_index",
                          "billing_6_amountTotal", "billing_6_paymentDate", "billing_6_status", "LTV", "mean", "churn_index",
                          "billing_5_amountTotal", "billing_5_paymentDate", "billing_5_status", "LTV", "mean", "churn_index",
                          "billing_4_amountTotal", "billing_4_paymentDate", "billing_4_status", "LTV", "mean", "churn_index",
                          "billing_3_amountTotal", "billing_3_paymentDate", "billing_3_status", "LTV", "mean", "churn_index",
                          "billing_2_amountTotal", "billing_2_paymentDate", "billing_2_status", "LTV", "mean", "churn_index",
                          "billing_1_amountTotal", "billing_1_paymentDate", "billing_1_status", "LTV", "mean", "churn_index"]].values, 
                      (df.values.shape[0], 10, 6))

    for i in range(data[:, :, 0].shape[0]):
        if data[i, 0, 3] < 10 and data[i, 0, 3] > 1:
            x_temp = np.linspace(1, int(data[i, 0, 3]), num = int(data[i, 0, 3]), endpoint = True)
            xnew = np.linspace(1, int(data[i, 0, 3]), num = 10, endpoint = True)

            #amountTotal
            y_temp = data[i, -int(data[i, 0, 3]):, 0]
            f = interp1d(x_temp, y_temp, kind = "linear")
            data[i, :, 0] = f(xnew)

            #paymentDate
            y_temp = data[i, -int(data[i, 0, 3]):, 1]
            f = interp1d(x_temp, y_temp, kind = "linear")
            data[i, :, 1] = f(xnew)

            #status
            y_temp = data[i, -int(data[i, 0, 3]):, 2]
            f = interp1d(x_temp, y_temp, kind = "linear")
            data[i, :, 2] = f(xnew)

        elif data[i, 0, 3] == 1:
            #amountTotal
            data[i, :, 0] = np.array([data[i, -1, 0] for j in range(10)])

            #paymentDate
            data[i, :, 1] = np.array([data[i, -1, 1] for j in range(10)])

            #status
            data[i, :, 2] = np.array([data[i, -1, 2] for j in range(10)])
    
    data[:, :, 0] = data[:, :, 0] - data[:, :, 0].mean(axis = 1).reshape(-1, 1)
    data[:, :, 1] = data[:, :, 1] - data[:, :, 1].mean(axis = 1).reshape(-1, 1)

    for i in range(10):
        data[:, i, 0] = sc.transform(data[:, i, 0].reshape(-1, 1)).reshape(-1)

    for i in range(10):
        data[:, i, 1] = sc_pd.transform(data[:, i, 1].reshape(-1, 1)).reshape(-1)

    for i in range(10):
        data[:, i, 2] = sc_stat.transform(data[:, i, 2].reshape(-1, 1)).reshape(-1)
    
    data = np.array(data)
    data = np.reshape(data, (data.shape[0], data.shape[1], data.shape[2]))
    data2 = data[:, 0, 3:6].astype(float)
    data2[:, 1] = sc_mean.transform(data2[:, 1].reshape(-1, 1)).reshape(data2.shape[0])
    data = data[:, :, :3].astype(float)

    lstm_A_test = model_A.predict([data[:, :, 0], data2])
    lstm_B_test = model_B.predict([data[:, :, 1], np.concatenate((data2[:, 0:1], data2[:, 2:3]), axis = 1)])
    lstm_C_test = model_C.predict([data[:, :, 2], np.concatenate((data2[:, 0:1], data2[:, 2:3]), axis = 1)])

    predict = model.predict([np.concatenate((np.concatenate((lstm_A_test, lstm_B_test), axis = 1), lstm_C_test), axis = 1), np.concatenate((data2[:, 0:1], data2[:, 2:3]), axis = 1)])
    predict_temp = predict
    predict = np.where(predict_temp >= 0.85, "Loyal", np.where(predict_temp >= 0.65, "Agak Loyal", np.where(predict_temp >= 0.45, "Telat Bayar", np.where(predict_temp >= 0.25, "Cenderung Churn", "Churn"))))

    df["lstm_amountTotal"] = lstm_A_test
    df["lstm_paymentDate"] = lstm_B_test
    df["lstm_status"] = lstm_C_test
    df["predict_percentage"] = predict_temp
    df["predict_description"] = predict
    df["amountTotal_prediction_description"] = df["lstm_amountTotal"].apply(lambda x: "Pola Data Churn" if x < 0.25 else "-")
    df["paymentDate_prediction_description"] = df["lstm_paymentDate"].apply(lambda x: "Pola Data Churn" if x < 0.25 else "-")
    df["loyality_description"] = np.where(lstm_C_test >= 0.85, "Loyal", np.where(lstm_C_test >= 0.65, "Agak Loyal", "Telat Bayar"))
    df["activity_description"] = df["churn_index"].apply(lambda x: "Tidak Aktif 2 Bulan Terakhir" if x == 0 else "-")
 
    #df.to_csv("prediction.csv")
    json_dict = json.dumps(df[["ND", 
                               "predict_percentage",
                               "predict_description",
                               "amountTotal_prediction_description",
                               "paymentDate_prediction_description",
                               "loyality_description",
                               "activity_description"]].set_index("ND").to_dict("index"))
    json_file = open("prediction.json", "w")
    json_file.write(json_dict)
    json_file.close()

    return {
              "result": "prediction.json"
           }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000, debug=True)