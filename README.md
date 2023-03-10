
# Table of contents
* [Introduction](#Introduction)
* [Calling data from web](#Calling-data-from-web)
* [Data](#Data)
* [Setup](#setup)
## Introduction
This work is about estimating the price of a stock in an interval  that has been affected by price manipulation. Therefore, we want to apply some statistical approaches to estimate the price of a stock using information from other stocks. 

## Calling data from web
By using the following scripts, we can download information about stock prices in Iran
```python
import numpy as np
import pandas as pd
import pytse_client as tse
import jdatetime , glob, os, ntpath
from datetime import datetime
from pytse_client import download_client_types_records
```
```python
tickers = tse.download(symbols="all", write_to_csv=True,  include_jdate=True)
```
After executing the above code, the daily transaction information of all stocks will be downloaded and stored in the code directory. Now, we combine  all data together:
```python
path = r'C:\\Users\\masoumifard.kh\\CodsOfprojects\\Public\\PredictionOfPumpAndDump\\tickers_data'
filenames = glob.glob(path + "/*.csv")
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)
Name = [path_leaf(path) for path in filenames]
dfs = []
for filename in filenames:
    dd = pd.read_csv(filename)
    strs = [path_leaf(filename).split(".")[0] for x in range(len(dd))]
    dd.insert(1,"PersianNameOfCode", strs, True)
    dfs.append(dd)
big_frame = pd.concat(dfs, ignore_index=True)
```
