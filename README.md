# Payment Behaviour Analysis to Predict Subscribption Status Using LSTM

## Data
Payment History from Telkom DDB, feature used in the model:
1. Billing Amount Total
2. Billing Status
3. Billing Payment Date
4. Gladius Paket Radius

## Team Member
* Syachrul Qolbi Nur Septi
* Fauzi Arifin

## Table of Contents
1. [Requirements](#requirements) to install on your system
2. Code to create, training, and evaluating the [model](Behaviour_Payment.ipynb)
3. [Results](#results)
4. [Links to google colab](https://colab.research.google.com/drive/1pUasmKKcQWSiSrb4o4oES9xgbO1se5py?authuser=2#scrollTo=v0719deFT8yn)

## Requirements

The main requirements are listed below:

Tested with 
* Tensorflow 2.7.0
* Python 3.7.10
* Numpy 1.19.5
* Matplotlib 3.2.2
* Seaborn 0.11.2
* Pandas 1.1.5

Additional requirements to generate dataset:

* Os
* IPython.display import clear_output
* Sklearn.metrics import classification_report, confusion_matrix
* Shutil
* Google.colab import drive


## Results
These are the results for our models.

### LSTM Model (Unsubscribe/Subscribe)
<div class="tg-wrap"><table class="tg">
  <tr>
    <th class="tg-7btt" colspan="3">Result</th>
  </tr>
  <tr>
    <td class="tg-7btt"></td>
    <td class="tg-7btt">Accuracy</td>
    <td class="tg-7btt">Precision</td>
    <td class="tg-7btt">Recall</td>
    <td class="tg-7btt">Sensitivity</td>
    <td class="tg-7btt">F1-Score</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Unsubscribe</td>
    <td class="tg-c3ow">82.95%</td>
    <td class="tg-c3ow">93.23%</td>
    <td class="tg-c3ow">75.66%</td>
    <td class="tg-c3ow">75.66%</td>
    <td class="tg-c3ow">83.53%</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Subscribe</td>
    <td class="tg-c3ow">82.95%</td>
    <td class="tg-c3ow">74.06%</td>
    <td class="tg-c3ow">92.67%</td>
    <td class="tg-c3ow">92.67%</td>
    <td class="tg-c3ow">82.33%</td>
  </tr>
</table></div>
