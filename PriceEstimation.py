#!/usr/bin/env python
# coding: utf-8

# In[2600]:
import numpy as np
import pandas as pd
import jdatetime
from datetime import datetime
# In[2601]:
#Direction = "D:\\Data\\LossCalculation\\"
Direction = '/home/khaled/Project/Data/LossCalculate/'

# In[2602]:
DataForReqression0 = pd.read_csv(Direction+'ReturnModified.csv')
#DataForReqression0 = DataForReqression0[(DataForReqression0['ReturnIndexFaraBorse'] != -1) & (DataForReqression0['ReturnIndex'] != -1)]
# In[2605]:
DataForReqression0 = DataForReqression0.reset_index()
# In[2606]:
#extracting training data for regression model
FirstDayOfInterval = '1400-08-11'
LastDayOfInterval = '1400-10-20'
LastDayOfInterval2 = '1400-08-10'
def DateT(a):
    return(datetime.fromisoformat((a)))
def DateT1(a):
    return(pd.Timestamp(a))
DataForReqression0['GregorianDate'] = DataForReqression0['GregorianDate'].apply(lambda x: DateT(x))
gregorian_date = jdatetime.date(int(LastDayOfInterval[0:4]),int(LastDayOfInterval[5:7]),int(LastDayOfInterval[8:10])).togregorian()
#dataMerg = dataMerg.fillna(str(pd.Timestamp(gregorian_date)))
gregorian_date3 = jdatetime.date(int(LastDayOfInterval2[0:4]),int(LastDayOfInterval2[5:7]),int(LastDayOfInterval2[8:10])).togregorian()

DataForReqression1 = DataForReqression0[(DataForReqression0['GregorianDate'] > pd.Timestamp(pd.to_datetime(gregorian_date))) |
                                       (DataForReqression0['GregorianDate'] < pd.Timestamp(pd.to_datetime(gregorian_date3)))]

# In[2624]:
pip install statsmodels
# In[2607]:
## To use statsmodels for linear regression
import statsmodels.formula.api as smf
## To use sklearn for linear regression
#from sklearn.linear_model import LinearRegression

# In[2608]:
#Sellect independet varibles based on correlation with dependent variable, i.e x149.
q = list(DataForReqression.corr()['Stock149'].sort_values().index)
# adding ReturnIndexFaraBorse and ReturnIndex to independent variables if they are not in.
q.append('ReturnIndexFaraBorse') 
q.append('ReturnIndex') 
q.append('Stock149') 
q = np.unique(q)
DataForReqression2 = DataForReqression1[q]
# preparing variables for regression
q1 = list(DataForReqression2.corr()['Stock149'].sort_values().index)
q1.append('ReturnIndexFaraBorse') 
q1.append('ReturnIndex') 
q1 = np.unique(q1)
q1 = list(filter(lambda x : x != 'Stock149', q1))

b = 'Stock149 ~'
for a in q1:
    b = b + '+' + ' '+ a +' '

# In[2610]:
# ## Independent variable selection
# It is obvious that the number of the independent variable is too large and we should solve this challenge:
# In regression analysis, the magnitude of your coefficients is not necessarily related to their importance. The most common criteria to determine the importance of independent variables in regression analysis are p-values. Small p-values imply high levels of importance, whereas high p-values mean that a variable is not statistically significant. You should only use the magnitude of coefficients as a measure for feature importance when your model is penalizing variables. That is, when the optimization problem has L1 or L2 penalties, like lasso or ridge regressions. [For more details see this link](https://stackoverflow.com/questions/65439843/linear-regression-get-feature-importance-using-minmaxscaler-extremely-larg)
# In[2611]:
#remove some variable based on pvalue
slr_sm_model = smf.ols(b, data=DataForReqression2)
slr_sm_model_ko = slr_sm_model.fit()
print(slr_sm_model_ko.summary())
param_slr = slr_sm_model_ko.summary2()
sellect = param_slr.tables[1][param_slr.tables[1]['P>|t|']<0.99]
q = list(sellect.index)
# In[2612]:


param_slr.tables[1]['P>|t|']


# In[2613]:


q1 = list(sellect.index)
q1 = list(filter(lambda x : x != 'Intercept', q1))
b = 'Stock149 ~'
for a in q1:
    b = b + '+' + ' '+ a +' '
slr_sm_model = smf.ols(b, data=DataForReqression1)
a = slr_sm_model_ko.rsquared
while a > 0.75:
    q1 = list(filter(lambda x : x != 'Intercept', q1))
    b = 'Stock149 ~'
    for a in q1:
        b = b + '+' + ' '+ a +' '
    slr_sm_model = smf.ols(b, data=DataForReqression1)
    slr_sm_model_ko = slr_sm_model.fit()
    print(slr_sm_model_ko.summary()) 
    EEEE = slr_sm_model_ko.summary2()
    sellect = EEEE.tables[1][EEEE.tables[1]['P>|t|']<np.quantile(EEEE.tables[1]['P>|t|'], 0.65)]
    q1 = list(sellect.index)
    q1 = list(filter(lambda x : x != 'Intercept', q1))
    q1 = list(filter(lambda x : x != 'Stock149', q1))

    b = 'Stock149 ~'
    for a in q1:
        b = b + '+' + ' '+ a +' '
    slr_sm_model = smf.ols(b, data=DataForReqression1)
    a = slr_sm_model_ko.rsquared


# In[2614]:


len(q1)


# In[2615]:


gregorian_date1 = jdatetime.date(int(FirstDayOfInterval[0:4]),int(FirstDayOfInterval[5:7]),int(FirstDayOfInterval[8:10])).togregorian()

DataForReqression2 = DataForReqression0[(DataForReqression0['GregorianDate'] < pd.Timestamp(pd.to_datetime(gregorian_date)))
                                       &(DataForReqression0['GregorianDate'] > pd.Timestamp(pd.to_datetime(gregorian_date1)))]



# In[2616]:


pd.Timestamp(pd.to_datetime(gregorian_date))
q1 = list(filter(lambda x : x != 'Intercept', q1))
b = 'Stock149 ~'
for a in q1:
    b = b + '+' + ' '+ a +' '
slr_sm_model = smf.ols(b, data=DataForReqression1)

### Fit the model (statsmodels calculates beta_0 and beta_1 here)
slr_sm_model_ko = slr_sm_model.fit()
RES = slr_sm_model_ko.predict(DataForReqression2)


# In[2617]:


dd = {'ReturnEstimated' : RES, 'Date' : DataForReqression2['Date']}
#pd.DataFrame(dd).to_excel(Direction+"Return.xlsx")


# In[2618]:


np.prod(pd.DataFrame(dd)['ReturnEstimated']+1)-1


# In[2619]:


np.cumprod(pd.DataFrame(dd)['ReturnEstimated']+1)


# In[2620]:


pd.DataFrame(dd)


# # Predict actual result after model trained with MinMaxScaler LinearRegression
# 
# In the following we start with this [link](https://datascience.stackexchange.com/questions/114833/predict-actual-result-after-model-trained-with-minmaxscaler-linearregression)

# In[2622]:


from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

DataForReqression3 = DataForReqression1.drop(['Stock149', 'index', 'GregorianDate', 'Date'], axis=1)
DataForReqression3 

X =  DataForReqression1.drop(['Stock149', 'index', 'GregorianDate', 'Date'], axis=1)
y = DataForReqression1['Stock149']  



# always split between training and test set first
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)

# Then fit the scaling on the training set
# Convert Feature/Column with Scaler
scaler = MinMaxScaler()
# Note: the columns have already been selected
X_train_scaled = scaler.fit_transform(X_train)

# Calling LinearRegression
model = LinearRegression()

# Fit linearregression into training data
model = model.fit(X_train_scaled, y_train)

# Now we need to scale the test set features
X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)
# y has not been scaled so nothing else to do 

# Calculate MSE (Lower better)
mse = mean_squared_error(y_test, y_pred)
print("MSE of testing set:", mse)

# Calculate MAE
mae = mean_absolute_error(y_test, y_pred)
print("MAE of testing set:", mae)

# Calculate RMSE (Lower better)
rmse = np.sqrt(mse)
print("RMSE of testing set:", rmse)


# In[ ]:





# In[ ]:




