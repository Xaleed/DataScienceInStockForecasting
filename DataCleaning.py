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
tickers = tse.download(symbols="all", write_to_csv=True,  include_jdate=True)


# In[3]:


path = r'C:\\Users\\masoumifard.kh\\CodsOfprojects\\Public\\PredictionOfPumpAndDump\\tickers_data'
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

big_frame


# In[ ]:




