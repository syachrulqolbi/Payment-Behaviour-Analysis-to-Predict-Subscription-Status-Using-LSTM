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
model = tf.keras.models.load_model("model_lstm.h5")
sc = joblib.load("scaler.gz")
sc_pd = joblib.load("scaler_pd.gz")
sc_stat = joblib.load("scaler_stat.gz")
sc_mean = joblib.load("scaler_mean.gz")

def shift(a):
    if a >= 1 and a < 22:
        return "PAID"
    elif a >= 22 and a < 32:
        return "LATE PAID"
    else:
        return "ZERO BILLING"

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

def check_date(df, array):
    array = array[::-1]
    for i in range(9):
        try:
            if int(str(array[i])[-2:]) == 1 and int(str(array[i + 1])[-2:]) != 12:
                dict_temp = {"PERIODE": int(str(int(str(array[i])[:-2]) - 1) + "12")}
                df = df.append(dict_temp, ignore_index = True)
            elif int(str(array[i])[-2:]) != 1 and int(str(array[i])[-2:]) - 1 != int(str(array[i + 1])[-2:]):
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

@app.get("/predict/")
def predict(url: str = Form(...)):
    df = pd.read_csv(url)
    df = df[["PERIODE",
         "ND",
         "JUMLAH_TAGIHAN",
         "TANGGAL"]].sort_values(by = ["PERIODE", "TANGGAL"])

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
        df["billing_{}_paymentDate".format(month)] = df["billing_{}_paymentDate".format(month)].apply(lambda x: int(str(x)[:2]))
        df["billing_{}_paymentDate".format(month)] = df["billing_{}_paymentDate".format(month)].astype(int)
        df["billing_{}_amountTotal".format(month)].loc[df["billing_{}_amountTotal".format(month)].isnull()] = 0
        df["billing_{}_amountTotal".format(month)] = df["billing_{}_amountTotal".format(month)].astype(float)
        df["billing_{}_status".format(month)] = df.apply(lambda x: shift(x["billing_{}_paymentDate".format(month)]), axis = 1)
        df["billing_{}_status".format(month)] = df["billing_{}_status".format(month)].apply(lambda x: 2 if x == "PAID" else (1 if x == "LATE PAID" else 0))
        
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
    df["mean"] = df[["billing_10_amountTotal",
                     "billing_9_amountTotal",
                     "billing_8_amountTotal",
                     "billing_7_amountTotal",
                     "billing_6_amountTotal",
                     "billing_5_amountTotal",
                     "billing_4_amountTotal",
                     "billing_3_amountTotal",
                     "billing_2_amountTotal",
                     "billing_1_amountTotal"]].mean(axis = 1).values

    data = np.reshape(df[["billing_10_amountTotal", "billing_10_paymentDate", "billing_10_status", "LTV", "mean",
                          "billing_9_amountTotal", "billing_9_paymentDate", "billing_9_status", "LTV", "mean",
                          "billing_8_amountTotal", "billing_8_paymentDate", "billing_8_status", "LTV", "mean",
                          "billing_7_amountTotal", "billing_7_paymentDate", "billing_7_status", "LTV", "mean",
                          "billing_6_amountTotal", "billing_6_paymentDate", "billing_6_status", "LTV", "mean",
                          "billing_5_amountTotal", "billing_5_paymentDate", "billing_5_status", "LTV", "mean",
                          "billing_4_amountTotal", "billing_4_paymentDate", "billing_4_status", "LTV", "mean",
                          "billing_3_amountTotal", "billing_3_paymentDate", "billing_3_status", "LTV", "mean",
                          "billing_2_amountTotal", "billing_2_paymentDate", "billing_2_status", "LTV", "mean",
                          "billing_1_amountTotal", "billing_1_paymentDate", "billing_1_status", "LTV", "mean"]].values, 
                      (df.values.shape[0], 10, 5))

    data[:, :, 0] = data[:, :, 0] - data[:, :, 0].mean(axis = 1).reshape(-1, 1)

    data[:, :, 0] = sc.transform(data[:, :, 0])
    data[:, :, 1] = sc_pd.transform(data[:, :, 1])
    data[:, :, 2] = sc_stat.transform(data[:, :, 2])

    data = np.array(data)
    data = np.reshape(data, (data.shape[0], data.shape[1], data.shape[2]))

    data2 = data[:, 0, 3:5].astype(float)
    data2[:, 1] = sc_mean.transform(data2[:, 1].reshape(-1, 1)).reshape(data2.shape[0])
    data = data[:, :, :3].astype(float)
	
    predict = model.predict([data[:, :, 0:1], data[:, :, 1:2], data[:, :, 2:3], data2[:]]).argmax(axis = 1) * 0.025
    df["predict_percentage"] = np.round(predict * 100, 2)
    df["predict_description"] = np.where(predict >= 0.85, "Loyal", np.where(predict >= 0.65, "Agak Loyal", np.where(predict >= 0.45, "Telat Bayar", np.where(predict >= 0.25, "Cenderung Churn", "Churn"))))
    
    json_dict = json.dumps(df[["ND", "predict_percentage", "predict_description"]].set_index("ND").to_dict("index"))
    json_file = open("prediction.json", "w")
    json_file.write(json_dict)
    json_file.close()

    return {
              "result": "prediction.json"
           }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000, debug=True)