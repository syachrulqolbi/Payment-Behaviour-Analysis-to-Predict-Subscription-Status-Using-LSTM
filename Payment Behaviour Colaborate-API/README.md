# Payment Behaviour Analysis to Predict Subscribption Status Using LSTM

## Data
Payment History from Telkom DDB, feature used in the model:
* Billing Amount Total
* Billing Status
* Billing Payment Date
* Billing Channel
* Gladius Paket Radius

## Team Member
1. Syachrul Qolbi Nur Septi
2. Fakhrur Razi

## Table of Contents
1. [Requirements](#requirements) to install on your system
2. Code to create, training, and evaluating the [model](Behaviour_Payment.ipynb)
3. [Results](#results)
4. [Links to google colab](https://colab.research.google.com/drive/17y3vGCJMIQ4a7-We2jWhOlCaDOYyyRSF#scrollTo=NIO8Mq6K9SKQ)
5. [Tutorial](#tutorial)

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
* Math
* IPython.display import clear_output
* Sklearn.metrics import classification_report, confusion_matrix
* Shutil
* Google.colab import drive
* FastAPI
* Pydantic
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
    <td class="tg-c3ow">82.82%</td>
    <td class="tg-c3ow">79.94%</td>
    <td class="tg-c3ow">85.98%</td>
    <td class="tg-c3ow">82.85%</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Subscribe</td>
    <td class="tg-c3ow">82.82%</td>
    <td class="tg-c3ow">85.94%</td>
    <td class="tg-c3ow">79.88%</td>
    <td class="tg-c3ow">82.80%</td>
  </tr>
</table></div>

## Tutorial
# Download Model Time Series Forest

Pastikan sudah mendownload Model Time Series Forest, link download berada pada [Link_Download_Model_Time_Series_Forest](https://drive.google.com/file/d/1yyFohbYDOACLY6e_iknS2-jgQBtL7Jf0/view?usp=sharing) lalu simpan didalam satu folder Payment Behaviour Colaborate-API

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
pip install pyts
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
    "billing_11_amountTotal": "112.860,00",
    "billing_10_amountTotal": "127.480,00",
    "billing_9_amountTotal": "120.060,00",
    "billing_8_amountTotal": "112.860,00",
    "billing_7_amountTotal": "127.440,00",
    "billing_6_amountTotal": "124.008,00",
    "billing_5_amountTotal": "126.360,00",
    "billing_4_amountTotal": "113.520,00",
    "billing_3_amountTotal": "116.160,00",
    "billing_2_amountTotal": "115.830,00",

    "billing_11_status": "PAID",
    "billing_10_status": "PAID",
    "billing_9_status": "PAID",
    "billing_8_status": "PAID",
    "billing_7_status": "PAID",
    "billing_6_status": "PAID",
    "billing_5_status": "PAID",
    "billing_4_status": "PAID",
    "billing_3_status": "PAID",
    "billing_2_status": "PAID",

    "billing_11_channel": "FINNET WAY4 - FINNET",
    "billing_10_channel": "FINNET WAY4 - FINNET",
    "billing_9_channel": "FINNET WAY4 - FINNET",
    "billing_8_channel": "FINNET WAY4 - FINNET",
    "billing_7_channel": "FINNET WAY4 - FINNET",
    "billing_6_channel": "FINNET WAY4 - FINNET",
    "billing_5_channel": "FINNET WAY4 - FINNET",
    "billing_4_channel": "FINNET WAY4 - FINNET",
    "billing_3_channel": "FINNET WAY4 - FINNET",
    "billing_2_channel": "FINNET WAY4 - FINNET",

    "billing_11_paymentDate": "20201221",
    "billing_10_paymentDate": "20210123",
    "billing_9_paymentDate": "20210225",
    "billing_8_paymentDate": "20210316",
    "billing_7_paymentDate": "20210403",
    "billing_6_paymentDate": "20210503",
    "billing_5_paymentDate": "20210606",
    "billing_4_paymentDate": "20210706",
    "billing_3_paymentDate": "20210819",
    "billing_2_paymentDate": "20210906",

    "paketradius": "INET10Q050"
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
    "Percentage": 98.99,
    "Predict Description": "Loyal"
}
```
## Contoh mengunakan POSTMAN
![image](/Images/Contoh_Postman.png)
