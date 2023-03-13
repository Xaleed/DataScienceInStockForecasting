#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pytse_client as tse
import jdatetime , glob, os, ntpath
from datetime import datetime
from pytse_client import download_client_types_records


# In[2]:
#Download historical data of trades
#tickers = tse.download(symbols="all", write_to_csv=True,  include_jdate=True)


# In[3]:
path = r'C:/Users/masoumifard.kh/CodsOfprojects/Public/DataScienceInStockForecasting/tickers_data/'
#path = r"/home/khaled/Project/DataScienceInStockForecasting/tickers_data/"
filenames = glob.glob(path + "/*.csv")
# In[4]:
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
#%%
big_frame1 = big_frame[big_frame['date'] > big_frame['date'][2500]].copy()
# In[ ]:
big_frame1 = big_frame1[(big_frame1['yesterday'] != 0) & (big_frame1['value'] > 10)]
#big_frame.columns

# %%
def frac(a,d):
    return(a/d)
fr = big_frame1.apply(lambda x:frac(x['adjClose'],x['yesterday']), 
                        axis=1)

#%%
big_frame1.insert(1,"ReturnRate", fr, True)

#%%
big_frame1 = big_frame1[(big_frame1['ReturnRate'] > np.quantile(big_frame1['ReturnRate'],0.001)) & (big_frame1['ReturnRate'] < np.quantile(big_frame1['ReturnRate'],0.999))]

# %%
Return = big_frame1[[ 'PersianNameOfCode', 'ReturnRate', 'date']].copy()
# %%
b = 'Stock'
c = ['Stock' + str(i) for i in range(len(np.unique(Return['PersianNameOfCode'])))]
data = {
  "PersianNameOfCode": np.unique(Return['PersianNameOfCode']),
  "CodeOfPerasianName": c
}
df1 = pd.DataFrame(data)
Return = pd.merge(Return, df1, on = 'PersianNameOfCode', how = 'left')
Return =Return[[ 'ReturnRate', 'date', 'CodeOfPerasianName']]
# %%
G = Return.groupby(['CodeOfPerasianName',  'date'])
Return1 = G.mean().reset_index()

#%%
Return1 = Return1.iloc[:,[0,1,3]]
# %%
#prepare data for regression using pivot
DataForReqression = Return1.pivot(index='date', columns='CodeOfPerasianName', values='ReturnRate').fillna(0)
#DataForReqression0 = pd.merge(DataForReqression,Index, on = 'date', how = 'left')
# %%

# %%
DataForReqression.to_csv("C:/Users/masoumifard.kh/CodsOfprojects/Public/DataScienceInStockForecasting/ReturnModified.csv")

# %%

# %%
Return.columns
# %%
