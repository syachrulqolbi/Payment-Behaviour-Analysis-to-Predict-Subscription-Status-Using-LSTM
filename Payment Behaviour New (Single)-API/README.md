# Payment Behaviour Analysis to Predict Subscribption Status Using LSTM (Single)

## Data
Payment History from Telkom DDB, feature used in the model:
* Billing Amount Total
* Billing Payment Date

## Team Member
1. Syachrul Qolbi Nur Septi

## Table of Contents
1. [Requirements](#requirements) to install on your system
2. [Results](#results)
3. [Links to google colab](https://colab.research.google.com/drive/17Ews_Ol0RjeU69ewKKElYorFSuWymRtb?usp=sharing)
4. [Tutorial](#tutorial)

## Requirements

The main requirements are listed below:

Tested with 
* Tensorflow 2.7.0
* Python 3.7.10
* Numpy 1.19.5
* Matplotlib 3.2.2
* Pandas 1.1.5
* Scipy 1.8.0

Additional requirements to generate dataset:

* Os
* Sklearn.metrics import classification_report, confusion_matrix
* Sklearn.preprocessing import StandardScaler
* Shutil
* Google.colab import drive
* FastAPI
* Joblib
* Pickle


## Results
These are the results for our models.

### LSTM Model (Unsubscribe/Subscribe)
<div class="tg-wrap"><table class="tg">
  <tr>
    <th class="tg-7btt" colspan="6">Result</th>
  </tr>
  <tr>
    <td class="tg-7btt"></td>
    <td class="tg-7btt">Accuracy (Macro Avg)</td>
    <td class="tg-7btt">Precision</td>
    <td class="tg-7btt">Recall</td>
    <td class="tg-7btt">F1-Score</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Unsubscribe</td>
    <td class="tg-c3ow">82.41%</td>
    <td class="tg-c3ow">50.70%</td>
    <td class="tg-c3ow">93.37%</td>
    <td class="tg-c3ow">65.71%</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Subscribe</td>
    <td class="tg-c3ow">82.41%</td>
    <td class="tg-c3ow">99.88%</td>
    <td class="tg-c3ow">98.33%</td>
    <td class="tg-c3ow">99.10%</td>
  </tr>
</table></div>

## Tutorial
# Instalasi Python

Pastikan sudah terinstall python dan pip dalam system anda, jika system anda mengunakan linux bisa mengikuti command di bawah ini

`
sudo apt install python3-pip
`

jika system mengunakan OS windows atau yang lain bisa minjau situs resmi python untuk instalasi python https://www.python.org/downloads/

# Instalasi Dependency 
Agar code dapat berjalan di perlukan beberapa dependecy, dapat langsung menjalankan command di terminal berikut satu demi satu jika python dan pip sudah terinstall

```
pip install fastapi
pip install uvicorn
pip install tensorflow
```

# Menjalankan API
Untuk menjalankan API cukup mejalankan command berikut di terminal
```
uvicorn main:app --reload
```
Secara default dia akan jalan secara lokal di 127.0.0.1 dengan port 8000 

Output jika runnning berhasil

![image](/Images/Output_Uvicorn.png) 

Jika ingin di jalankan di port dan address host yang berbeda bisa mengunakan option --host dan --port
```
uvicorn --host [host address] --port [nilai port]  main:app --reload 
```

# Menggunakan API
Kita akan memprediksi status berlangganan user, apakah user tersebut akan berlangganan kembali di bulan depan atau tidak. Untuk mengunakan endpoint bisa dengan menyiapkan body parameternya berupa format JSON dengan format seperti berikut

```
{
    "billing_10_amountTotal": "000.000,00",
    "billing_9_amountTotal": "781.000,00",
    "billing_8_amountTotal": "781.000,00",
    "billing_7_amountTotal": "806.300,00",
    "billing_6_amountTotal": "665.500,00",
    "billing_5_amountTotal": "665.500,00",
    "billing_4_amountTotal": "665.500,00",
    "billing_3_amountTotal": "665.500,00",
    "billing_2_amountTotal": "665.500,00",
    "billing_1_amountTotal": "000.000,00",

    "billing_10_paymentDate": "20210100",
    "billing_9_paymentDate": "20210221",
    "billing_8_paymentDate": "20210325",
    "billing_7_paymentDate": "20210421",
    "billing_6_paymentDate": "20210523",
    "billing_5_paymentDate": "20210624",
    "billing_4_paymentDate": "20210715",
    "billing_3_paymentDate": "20210929",
    "billing_2_paymentDate": "20210928",
    "billing_1_paymentDate": "20211000",

    "billing_10_period": "202101",
    "billing_9_period": "202102",
    "billing_8_period": "202103",
    "billing_7_period": "202104",
    "billing_6_period": "202105",
    "billing_5_period": "202106",
    "billing_4_period": "202107",
    "billing_3_period": "202108",
    "billing_2_period": "202109",
    "billing_1_period": "202110"
}
```
dan untuk URL API mengunakan format sebagai berikut
```
http://[Host]:[Port]/predict
```
dan request method yang digunakan adalah **GET** 
API akan mengembalikan variabel Percentage dan Predict Description beserta valuenya dengan tipe data JSON.

## Hasil Retun API
```
{
    "Percentage": 20.8,
    "Predict Description": "Churn",
    "Amount Total Predict Description": "-",
    "Payment Date Predict Description": "-",
    "Loyality Description": "Telat Bayar",
    "Activity Description": "-"
}
```
## Contoh mengunakan POSTMAN
![image](/Images/Contoh_Postman.png)
