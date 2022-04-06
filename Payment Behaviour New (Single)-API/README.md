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
    <td class="tg-7btt">Accuracy</td>
    <td class="tg-7btt">Precision</td>
    <td class="tg-7btt">Recall</td>
    <td class="tg-7btt">F1-Score</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Unsubscribe</td>
    <td class="tg-c3ow">94.96%</td>
    <td class="tg-c3ow">73.00%</td>
    <td class="tg-c3ow">86.39%</td>
    <td class="tg-c3ow">79.14%</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Subscribe</td>
    <td class="tg-c3ow">94.96%</td>
    <td class="tg-c3ow">98.27%</td>
    <td class="tg-c3ow">96.03%</td>
    <td class="tg-c3ow">97.14%</td>
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
    "billing_10_amountTotal": "112.860,00",
    "billing_9_amountTotal": "127.480,00",
    "billing_8_amountTotal": "120.060,00",
    "billing_7_amountTotal": "112.860,00",
    "billing_6_amountTotal": "127.440,00",
    "billing_5_amountTotal": "124.008,00",
    "billing_4_amountTotal": "126.360,00",
    "billing_3_amountTotal": "113.520,00",
    "billing_2_amountTotal": "116.160,00",
    "billing_1_amountTotal": "115.830,00",

    "billing_10_paymentDate": "20201221",
    "billing_9_paymentDate": "20210123",
    "billing_8_paymentDate": "20210225",
    "billing_7_paymentDate": "20210316",
    "billing_6_paymentDate": "20210403",
    "billing_5_paymentDate": "20210503",
    "billing_4_paymentDate": "20210606",
    "billing_3_paymentDate": "20210706",
    "billing_2_paymentDate": "20210819",
    "billing_1_paymentDate": "20210906"
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
    "Percentage": 98.89,
    "Predict Description": "Loyal"
}
```
## Contoh mengunakan POSTMAN
![image](/Images/Contoh_Postman.png)
