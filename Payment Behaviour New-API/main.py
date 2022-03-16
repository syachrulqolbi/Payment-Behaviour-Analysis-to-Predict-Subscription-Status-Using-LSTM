import uvicorn
import csv
import pandas as pd
import tensorflow as tf
from pydantic import BaseModel
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import StandardScaler
import math

from typing import Optional
from fastapi import File, UploadFile, FastAPI
from fastapi import FastAPI, Form
from typing import List

app = FastAPI()
model = tf.keras.models.load_model("model_lstm.h5")
sc = joblib.load("scaler.gz")

def function(a, b):
    if b >= 1 and b < 22:
        return "PAID"
    elif b >= 22 and b < 32:
        return "LATE PAID"
    else:
        return "ZERO BILLING"

def function_LTV(billing_10, billing_9, billing_8, billing_7, billing_6, billing_5, billing_4, billing_3, billing_2, billing_1):
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

def function_status_index(billing_10, billing_9, billing_8, billing_7, billing_6, billing_5, billing_4, billing_3, billing_2, billing_1):
    count = 0
    if billing_10 == "LATE PAID":
        count += 1
    elif billing_10 == "PAID":
        count += 2
    else:
        count += 0
    
    if billing_9 == "LATE PAID":
        count += 1
    elif billing_9 == "PAID":
        count += 2
    else:
        count += 0
    
    if billing_8 == "LATE PAID":
        count += 1
    elif billing_8 == "PAID":
        count += 2
    else:
        count += 0
    
    if billing_7 == "LATE PAID":
        count += 1
    elif billing_7 == "PAID":
        count += 2
    else:
        count += 0
    
    if billing_6 == "LATE PAID":
        count += 1
    elif billing_6 == "PAID":
        count += 2
    else:
        count += 0
      
    if billing_5 == "LATE PAID":
        count += 1
    elif billing_5 == "PAID":
        count += 2
    else:
        count += 0
    
    if billing_4 == "LATE PAID":
        count += 1
    elif billing_4 == "PAID":
        count += 2
    else:
        count += 0
    
    if billing_3 == "LATE PAID":
        count += 1
    elif billing_3 == "PAID":
        count += 2
    else:
        count += 0

    if billing_2 == "LATE PAID":
        count += 1
    elif billing_2 == "PAID":
        count += 2
    else:
        count += 0
    
    if billing_1 == "LATE PAID":
        count += 1
    elif billing_1 == "PAID":
        count += 2
    else:
        count += 0

    return count

@app.get("/predict/")
def predict(url: str = Form(...)):
    df = pd.read_csv(url)
    df = df[["PERIODE",
         "ND",
         "JUMLAH_TAGIHAN",
         "STATUS_PEMBAYARAN",
         "LOKASI_PEMBAYARAN",
         "TANGGAL"]].sort_values(by = ["PERIODE", "TANGGAL"])
    df = df.pivot_table(index = "ND", columns = "PERIODE", aggfunc = np.sum).join(df.pivot_table(index = "ND", columns = "PERIODE", values = ["STATUS_PEMBAYARAN", "LOKASI_PEMBAYARAN", "TANGGAL"], aggfunc = "last"), how = "outer")
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
                                            "billing_10_channel",
                                            "billing_9_channel",
                                            "billing_8_channel",
                                            "billing_7_channel",
                                            "billing_6_channel",
                                            "billing_5_channel",
                                            "billing_4_channel",
                                            "billing_3_channel",
                                            "billing_2_channel",
                                            "billing_1_channel",
                                            "billing_10_status",
                                            "billing_9_status",
                                            "billing_8_status",
                                            "billing_7_status",
                                            "billing_6_status",
                                            "billing_5_status",
                                            "billing_4_status",
                                            "billing_3_status",
                                            "billing_2_status",
                                            "billing_1_status",
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
    print(df)
    for month in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        df["billing_{}_paymentDate".format(month)].loc[df["billing_{}_paymentDate".format(month)].isnull()] = 0
        df["billing_{}_paymentDate".format(month)] = df["billing_{}_paymentDate".format(month)].apply(lambda x: int(str(x)[:2]))
        df["billing_{}_paymentDate".format(month)] = df["billing_{}_paymentDate".format(month)].astype(int)
    
    for month in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        df["billing_{}_amountTotal".format(month)].loc[df["billing_{}_amountTotal".format(month)].isnull()] = 0
        df["billing_{}_amountTotal".format(month)] = df["billing_{}_amountTotal".format(month)].astype(float)
        df["billing_{}_status".format(month)].loc[df["billing_{}_status".format(month)].isnull()] = "ZERO BILLING"
        df["billing_{}_status".format(month)] = df["billing_{}_status".format(month)].astype(str)
        df["billing_{}_status".format(month)] = df.apply(lambda x: function(x["billing_{}_status".format(month)], x["billing_{}_paymentDate".format(month)]), axis = 1)
        df["billing_{}_channel".format(month)].loc[df["billing_{}_channel".format(month)].isnull()] = "-"
        df["LTV"] = df.apply(lambda x: function_LTV(x["billing_10_status"], 
                                                    x["billing_9_status"], 
                                                    x["billing_8_status"], 
                                                    x["billing_7_status"], 
                                                    x["billing_6_status"], 
                                                    x["billing_5_status"], 
                                                    x["billing_4_status"], 
                                                    x["billing_3_status"], 
                                                    x["billing_2_status"], 
                                                    x["billing_1_status"]), axis = 1)
        df["status_index"] = df.apply(lambda x: function_status_index(x["billing_10_status"], 
                                                                      x["billing_9_status"], 
                                                                      x["billing_8_status"], 
                                                                      x["billing_7_status"], 
                                                                      x["billing_6_status"], 
                                                                      x["billing_5_status"], 
                                                                      x["billing_4_status"], 
                                                                      x["billing_3_status"], 
                                                                      x["billing_2_status"], 
                                                                      x["billing_1_status"]), axis = 1)

    for month in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        df["billing_{}_status".format(month)] = df["billing_{}_status".format(month)].apply(lambda x: 2 if x == "PAID" else (1 if x == "LATE PAID" else 0))

    df = df[["ND",
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
             "billing_10_status",
             "billing_9_status",
             "billing_8_status",
             "billing_7_status",
             "billing_6_status",
             "billing_5_status",
             "billing_4_status",
             "billing_3_status",
             "billing_2_status",
             "billing_1_status",
             "billing_10_paymentDate",
             "billing_9_paymentDate",
             "billing_8_paymentDate",
             "billing_7_paymentDate",
             "billing_6_paymentDate",
             "billing_5_paymentDate",
             "billing_4_paymentDate",
             "billing_3_paymentDate",
             "billing_2_paymentDate",
             "billing_1_paymentDate",
             "billing_10_channel",
             "billing_9_channel",
             "billing_8_channel",
             "billing_7_channel",
             "billing_6_channel",
             "billing_5_channel",
             "billing_4_channel",
             "billing_3_channel",
             "billing_2_channel",
             "billing_1_channel",
             "LTV",
             "status_index"]]

    data = np.reshape(df[["billing_10_amountTotal", "billing_10_paymentDate", "billing_10_status", "billing_10_channel", "LTV", "status_index",
                          "billing_9_amountTotal", "billing_9_paymentDate", "billing_9_status", "billing_9_channel", "LTV", "status_index",
                          "billing_8_amountTotal", "billing_8_paymentDate", "billing_8_status", "billing_8_channel", "LTV", "status_index",
                          "billing_7_amountTotal", "billing_7_paymentDate", "billing_7_status", "billing_7_channel", "LTV", "status_index",
                          "billing_6_amountTotal", "billing_6_paymentDate", "billing_6_status", "billing_6_channel", "LTV", "status_index",
                          "billing_5_amountTotal", "billing_5_paymentDate", "billing_5_status", "billing_5_channel", "LTV", "status_index",
                          "billing_4_amountTotal", "billing_4_paymentDate", "billing_4_status", "billing_4_channel", "LTV", "status_index",
                          "billing_3_amountTotal", "billing_3_paymentDate", "billing_3_status", "billing_3_channel", "LTV", "status_index",
                          "billing_2_amountTotal", "billing_2_paymentDate", "billing_2_status", "billing_2_channel", "LTV", "status_index",
                          "billing_1_amountTotal", "billing_1_paymentDate", "billing_1_status", "billing_1_channel", "LTV", "status_index"]].values, 
                      (df.values.shape[0], 10, 6))

    data[:, :, 0] = sc.transform(data[:, :, 0])
    data = np.array(data)
    data = np.reshape(data, (data.shape[0], data.shape[1], data.shape[2]))
    data2 = data[:, 0, 4:6].astype(float)
    data = data[:, :, :3].astype(float)

    #predict = np.reshape(np.round(model.predict([data.astype(float), data2.astype(float)])).astype(int), (data.shape[0]))
    #print(predict)

    df["predict_percentage"] = np.round(np.reshape(model.predict([data.astype(float), data2.astype(float)]), -1), 2)
    df["predict_description"] = np.where(predict == 0, "UNSUBSCRIBE", np.where(predict == 1, "SUBSCRIBE", predict))

    return df[["ND", "predict_percentage", "predict_description"]].set_index("ND").T.to_dict("list")

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000, debug=True)