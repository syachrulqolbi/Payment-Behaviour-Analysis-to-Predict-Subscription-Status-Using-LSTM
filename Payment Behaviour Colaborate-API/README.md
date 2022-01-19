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
    <td class="tg-c3ow">82.95%</td>
    <td class="tg-c3ow">93.23%</td>
    <td class="tg-c3ow">75.66%</td>
    <td class="tg-c3ow">83.53%</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Subscribe</td>
    <td class="tg-c3ow">82.95%</td>
    <td class="tg-c3ow">74.06%</td>
    <td class="tg-c3ow">92.67%</td>
    <td class="tg-c3ow">82.33%</td>
  </tr>
</table></div>

## Tutorial
# Download Model Time Series Forest

Pastikan sudah mendownload Model Time Series Forest, link download ada di [Link Download model timeseriesforest](Link Download model_timeseriesforest.txt)
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
    "amountTotal_9": "630.300,00", # billing_{}_amountTotal 9 Bulan Sebelumnya
    "amountTotal_8": "630.300,00",
    "amountTotal_7": "630.300,00",
    "amountTotal_6": "630.300,00",
    "amountTotal_5": "630.300,00",
    "amountTotal_4": "651.000,00",
    "amountTotal_3": "630.300,00",
    "amountTotal_2": "630.300,00", # billing_{}_amountTotal 2 Bulan Sebelumnya
    "amountTotal_1": "000.000,00", # billing_{}_amountTotal 1 Bulan Sebelumnya

    "status_9": "PAID", # billing_{}_status 9 Bulan Sebelumnya
    "status_8": "PAID",
    "status_7": "PAID",
    "status_6": "PAID",
    "status_5": "PAID",
    "status_4": "PAID",
    "status_3": "PAID",
    "status_2": "PAID", # billing_{}_status 2 Bulan Sebelumnya
    "status_1": "UNPAID", # billing_{}_status 1 Bulan Sebelumnya

    "paymentDate_9": "20201222", # billing_{}_paymentDate 9 Bulan Sebelumnya
    "paymentDate_8": "20210107", 
    "paymentDate_7": "20210207",
    "paymentDate_6": "20210307",
    "paymentDate_5": "20210404",
    "paymentDate_4": "20210504",
    "paymentDate_3": "20210614",
    "paymentDate_2": "20210714", # billing_{}_paymentDate 2 Bulan Sebelumnya
    "paymentDate_1": "20210800", # billing_{}_paymentDate 1 Bulan Sebelumnya

    "paketradius": "INETNLOY30" # gladius.paketradius
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
    "Percentage": 98.09,
    "Predict Description" : "Loyal"
}
```
## Contoh mengunakan POSTMAN
![image](/Images/Contoh_Postman.png)
