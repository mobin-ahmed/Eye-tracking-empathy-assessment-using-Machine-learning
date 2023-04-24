#!/usr/bin/env python
# coding: utf-8

# ## Import necessary libraries

# In[1]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

import glob
from sklearn.preprocessing import StandardScaler

from datetime import datetime


# ## Loading the questionary to marge the target variables to both group of the dataset
# <h1>.</h1> questionary holds extended empathy score after the intervention

# In[2]:


qu=pd.read_csv(r'C:\Users\Mobin Ahmed\Downloads\data science & decision making\projects/Questionnaire_datasetIB.csv', encoding='ISO-8859-1')



# ## Loading the control group of the dataset and adding column named 'Empathy Score' as target

# In[6]:


df=[]
x=0
v=0
for filename in glob.glob(r'C:\Users\Mobin Ahmed\Downloads\data science & decision making\projects\EyeT\EyeT/*.csv'):
#     print(filename)
#     print(type(filename))
#     break
    if(filename.find('letter')==-1):
        f=pd.read_csv(filename)
        f.insert(1, column = "Empathy Score", value = 0)
        for i in range(len(f)):
            if(int(filename[len(filename)-5])>=0 and int(filename[len(filename)-5])<=9):
                if(filename[len(filename)-6]!='_'):
                    if(int(filename[len(filename)-14])>=0 and int(filename[len(filename)-14])<=9):
                        if(filename[len(filename)-15]!='_'):
                            x=(int(filename[len(filename)-15])*10+int(filename[len(filename)-14]))
                            print(x)
                            x=x-1
                            f['Empathy Score'][i]=qu['Total Score extended'][x]
#                             v=v+1
                        else: 
                            x=(int(filename[len(filename)-14]))
                            print(x)
                            x=x-1
                          
                            f['Empathy Score'][i]=qu['Total Score extended'][x]
#                             v=v+1

                else:
            #             print('single num')
                    if(int(filename[len(filename)-13])>=0 and int(filename[len(filename)-13])<=9):
                        if(filename[len(filename)-14]!='_'):
            #                     print('double number 2')
                            x=(int(filename[len(filename)-14])*10+int(filename[len(filename)-13]))
                            print(x)
                            x=x-1
                      
                            f['Empathy Score'][i]=qu['Total Score extended'][x]
#                             v=v+1
                        else:
                            x=(int(filename[len(filename)-13]))
                            print(x)
                            x=x-1
                            
                            f['Empathy Score'][i]=qu['Total Score extended'][x]
#                             v=v+1
        f=np.array(f)
        df.append(f)



# ### Concating as Dataframe

# In[7]:


df=pd.concat([ pd.DataFrame(fl)
    for fl in df
               ])


# In[8]:


df.shape


# In[9]:


pd.set_option('display.max_columns', None)


# In[37]:


df.head()


# In[38]:


df.info()


# In[10]:


col=['Unnamed:0', 'Empathy Score', 'Recording timestamp', 'Computer timestamp', 'Sensor', 'Project name',
       'Export date', 'Participant name', 'Recording name', 'Recording date',
       'Recording date UTC', 'Recording start time',
       'Recording start time UTC', 'Recording duration', 'Timeline name',
       'Recording Fixation filter name', 'Recording software version',
       'Recording resolution height', 'Recording resolution width',
       'Recording monitor latency', 'Eyetracker timestamp', 'Event',
       'Event value', 'Gaze point X', 'Gaze point Y', 'Gaze point left X',
       'Gaze point left Y', 'Gaze point right X', 'Gaze point right Y',
       'Gaze direction left X', 'Gaze direction left Y',
       'Gaze direction left Z', 'Gaze direction right X',
       'Gaze direction right Y', 'Gaze direction right Z',
       'Pupil diameter left', 'Pupil diameter right', 'Validity left',
       'Validity right', 'Eye position left X (DACSmm)',
       'Eye position left Y (DACSmm)', 'Eye position left Z (DACSmm)',
       'Eye position right X (DACSmm)', 'Eye position right Y (DACSmm)',
       'Eye position right Z (DACSmm)', 'Gaze point left X (DACSmm)',
       'Gaze point left Y (DACSmm)', 'Gaze point right X (DACSmm)',
       'Gaze point right Y (DACSmm)', 'Gaze point X (MCSnorm)',
       'Gaze point Y (MCSnorm)', 'Gaze point left X (MCSnorm)',
       'Gaze point left Y (MCSnorm)', 'Gaze point right X (MCSnorm)',
       'Gaze point right Y (MCSnorm)', 'Presented Stimulus name',
       'Presented Media name', 'Presented Media width',
       'Presented Media height', 'Presented Media position X (DACSpx)',
       'Presented Media position Y (DACSpx)', 'Original Media width',
       'Original Media height', 'Eye movement type', 'Gaze event duration',
       'Eye movement type index', 'Fixation point X', 'Fixation point Y',
       'Fixation point X (MCSnorm)', 'Fixation point Y (MCSnorm)',
       'Mouse position X', 'Mouse position Y']


# In[11]:


df.columns=col



# In[12]:


### Unique values in the target variable

df['Empathy Score'].unique()

As the Empathy dataset contains many csv files of two sub group named test group and control group, here I am loading control group's csv files one by one using glob library. Then, concatinate them
in a single dataframe with target extended empathy score from questionary named QuestionariesIB.csv.
# ## Taking random samples for exploration and modelling

# In[162]:


# dtcnx=df.iloc[:int(df.shape[0]/2),]
# dtcnm=df.iloc[int(df.shape[0]/2)+1:,]


# In[310]:


# First time trying with 20000 random records from dataset both with exploration(dtcnx) and modelling(dtcnm)

# dtcnx=df.sample(20000)
# dtcnm=df.sample(20000)

# Second time trying with 100000 random records from dataset both with exploration(dtcnx) and modelling(dtcnm)

dtcnx=df.sample(100000)
dtcnm=df.sample(100000)


# In[15]:


dtcnx.shape


# ### Data exploration and visualization(Control group)

# In[55]:


## Printing some important aspects of data

print('Row: ', dtcnx.shape[0])
print('Columns: ', dtcnx.shape[1])
print('\nFeatures: ', dtcnx.columns.tolist())
print('\nMissing values: ', dtcnx.isnull().any())
print('\nUnique Values: ', dtcnx.nunique())



# In[59]:




# While observing all the features and its values of the dataset I can see some feature's value containing comma instead of dot.
# I really need to convert these commas to dots/points.

# In[166]:


dtcnx.info()


# In the dataset I can see there are categorial/objects variables and after number 19 of column index every column contains more or less missing values. Furthermore, in the dataset there are some features which are not relevent to the final task. So, I am going to remove these features from the dataset and select those are relevant.

# ### Now checking null values

# In[61]:


dtcnx.isna().sum()


# I can see that in the dataset here are almost every columns containing missing/null values.


# In[18]:


def cor(dataset, thres):
    col_corr=set()
    corr_matrix=dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j])>thres:
                colname=corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr


# In[19]:


corr_features=cor(dtcnx, 0.8)  ## I can see that there are nothing correlations among variables with threshold 0.8(80%)
len(set(corr_features))


# ## Preprocessing & cleaning of  control group

# In[311]:


## feature selection

## while checking correlation among variabls/features, it showed nothing correlated.

## But I am now selecting those features whose are really match pattern for this group, and really
## important for my task.


# In[312]:


feature_selection=['Empathy Score', 'Sensor',
       'Recording date', 'Recording duration',
        'Recording resolution height', 'Recording resolution width',
       'Recording monitor latency',
       'Gaze point X', 'Gaze point Y', 'Gaze point left X',
       'Gaze point left Y', 'Gaze point right X', 'Gaze point right Y',
       'Gaze direction left X', 'Gaze direction left Y',
       'Gaze direction left Z', 'Gaze direction right X',
       'Gaze direction right Y', 'Gaze direction right Z',
       'Pupil diameter left', 'Pupil diameter right', 'Validity left',
       'Validity right', 'Eye position left X (DACSmm)',
       'Eye position left Y (DACSmm)', 'Eye position left Z (DACSmm)',
       'Eye position right X (DACSmm)', 'Eye position right Y (DACSmm)',
       'Eye position right Z (DACSmm)', 'Gaze point left X (DACSmm)',
       'Gaze point left Y (DACSmm)', 'Gaze point right X (DACSmm)',
       'Gaze point right Y (DACSmm)', 'Gaze point X (MCSnorm)',
       'Gaze point Y (MCSnorm)', 'Gaze point left X (MCSnorm)',
       'Gaze point left Y (MCSnorm)', 'Gaze point right X (MCSnorm)',
       'Gaze point right Y (MCSnorm)', 'Presented Media width',
       'Presented Media height', 'Presented Media position X (DACSpx)',
       'Presented Media position Y (DACSpx)', 'Original Media width',
       'Original Media height', 'Eye movement type', 'Gaze event duration',
       'Eye movement type index', 'Fixation point X', 'Fixation point Y',
       'Fixation point X (MCSnorm)', 'Fixation point Y (MCSnorm)',
       'Mouse position X', 'Mouse position Y']


# In[313]:


len(feature_selection)


# In[314]:


dtcnm=dtcnm[feature_selection]
dtcnm.head(50)


# ### Converting feature's comma values to dot

# In[315]:


for i in dtcnm.columns:
    dtcnm[i] = dtcnm[i].astype(str).str.replace(',', '.')


# In[316]:


import math
dtcnm=dtcnm.replace('nan', math.nan)
dtcnm.head(50)


# In[317]:


numeric_features=['Empathy Score','Recording duration',
        'Recording resolution height', 'Recording resolution width',
       'Recording monitor latency',
       'Gaze point X', 'Gaze point Y', 'Gaze point left X',
       'Gaze point left Y', 'Gaze point right X', 'Gaze point right Y',
       'Gaze direction left X', 'Gaze direction left Y',
       'Gaze direction left Z', 'Gaze direction right X',
       'Gaze direction right Y', 'Gaze direction right Z',
       'Pupil diameter left', 'Pupil diameter right', 'Eye position left X (DACSmm)',
       'Eye position left Y (DACSmm)', 'Eye position left Z (DACSmm)',
       'Eye position right X (DACSmm)', 'Eye position right Y (DACSmm)',
       'Eye position right Z (DACSmm)', 'Gaze point left X (DACSmm)',
       'Gaze point left Y (DACSmm)', 'Gaze point right X (DACSmm)',
       'Gaze point right Y (DACSmm)', 'Gaze point X (MCSnorm)',
       'Gaze point Y (MCSnorm)', 'Gaze point left X (MCSnorm)',
       'Gaze point left Y (MCSnorm)', 'Gaze point right X (MCSnorm)',
       'Gaze point right Y (MCSnorm)', 'Presented Media width',
       'Presented Media height', 'Presented Media position X (DACSpx)',
       'Presented Media position Y (DACSpx)', 'Original Media width',
       'Original Media height', 'Gaze event duration',
       'Eye movement type index', 'Fixation point X', 'Fixation point Y',
       'Fixation point X (MCSnorm)', 'Fixation point Y (MCSnorm)',
       'Mouse position X', 'Mouse position Y']


# In[318]:


for i in numeric_features:
    dtcnm[i] = dtcnm[i].astype(float)
            




# In[319]:


dtcnm.info()


# In[320]:


dtcnm.head()


# ### Handling missing values of the featrues using the methods 'F-fill' and 'B-fill'

# In[321]:


for i in dtcnm.columns:
    dtcnm[i].ffill(inplace=True)


# In[322]:


for i in dtcnm.columns:
    dtcnm[i].bfill(inplace=True)


# In[323]:


dtcnm.info()


# In[324]:


dtcnm.head()


# ### Separating feature variables and the target variables

# In[325]:


dtcnm_y=dtcnm.iloc[:, 0]
x=dtcnm.iloc[:, 1:]


# In[326]:


x.head(50)


# In[327]:


dtcnm_y=dtcnm_y.astype(int)
dtcnm_y


# ### Using One Hot Encoder to convert categorical variables to numeric

# In[328]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
tr=ColumnTransformer(transformers=[
    ('tr', OneHotEncoder(sparse=False, drop='first'),['Sensor', 'Recording date', 'Validity left', 'Validity right', 'Eye movement type'])], 
                     remainder='passthrough')


# In[329]:


x=tr.fit_transform(x)


# In[330]:


x.shape


# In[331]:


x=pd.DataFrame(x)


# In[332]:


x.info()



# In[82]:


## Removing outliers using Z-score and capping


# In[333]:


for i in x.columns:
    upper_limit=x[i].mean()+3*x[i].std()
    lower_limit=x[i].mean()-3*x[i].std()
    
    x.loc[(x[i]>=upper_limit), i]=upper_limit
    x.loc[(x[i]<=lower_limit), i]=lower_limit




# ### Scaling the dataset using MinMaxScaler of the sklearn library
Here I see from the dataset, the dataset does not maintain proper scaling the values of some variables. So, now I need to
scale them to train the models properly.
# In[335]:


scaler=StandardScaler()
x=scaler.fit_transform(x)


# In[336]:


x=pd.DataFrame(x)


# ## Modelling part and Evaluation (Control group)

# In[ ]:





# My dataset is now a clean and processed dataset. The dataset now is ready to use in different ML models.
# I am going to use this dataset to different model and find out the best for real world test with main task.
# I will be using some techniques like train test split, cross validation, some ML techniques, some forcasting techniques.
# And I will work on the evaluation of the models results.

# In[337]:


## train test split
X_train, X_test, y_train, y_test=train_test_split(x, dtcnm_y, test_size=.3,random_state=40)


# In[338]:


X_train


# #### Model 1

# In[339]:


## Using Multiple linear Regression
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
cn_lr_model=lr.fit(X_train, y_train)
sc=cn_lr_model.score(X_train, y_train)
## using 10 fold cross validation on model
from sklearn.model_selection import cross_val_score
cros_val=cross_val_score(cn_lr_model, X_train, y_train, cv=10)
print('Cross validation scores', cros_val)
print('Mean cross validation of scores', cros_val.mean())

lr_pred=cn_lr_model.predict(X_test)



from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

score_r2=r2_score(lr_pred, y_test)
score_mae=mean_absolute_error(lr_pred, y_test)
print('R2 Score: ', score_r2)
print("The accuracy score of our model is ", sc)
print("The Mean Absolute Error of our Model is {}".format(round(score_mae, 2)))

from sklearn.metrics import mean_squared_error
print("The Mean Squared Error of our Model is {}".format(round(mean_squared_error(lr_pred, y_test), 2)))


# ##### Results while using 20000 random records from dataset to modelling part
# 
# Cross validation scores [0.7821064  0.79917144 0.81260007 0.78970735 0.80528669 0.7965341
#  0.79145768 0.80046563 0.78885857 0.81068451]
#  
# Mean cross validation of scores 0.797687245353328
# 
# R2 Score:  0.7400187388123709
# 
# The accuracy score of our model is  0.799932305548396
# 
# The Mean Absolute Error of our Model is 7.09
# 
# The Mean Squared Error of our Model is 85.35
# 
# 
# ##### Results while using 100000 random records from dataset to modelling part
# 
# Cross validation scores [0.79760472 0.79480774 0.79766517 0.79635573 0.79617614 0.79790275
#  0.7963607  0.80375658 0.7934591  0.79188611]
#  
# Mean cross validation of scores 0.796597474306951
# 
# R2 Score:  0.745463536722021
# 
# The accuracy score of our model is 0.7970392573622365
# 
# The Mean Absolute Error of our Model is 6.97
# 
# The Mean Squared Error of our Model is 82.95

# #### Model 2

# In[341]:


## Using Support Vector Regressor model
from sklearn.svm import SVR
svr=SVR(kernel='rbf')
cn_svr_model=svr.fit(X_train, y_train)
sc=cn_svr_model.score(X_train, y_train)
## using 10 fold cross validation on model
from sklearn.model_selection import cross_val_score
cros_val=cross_val_score(cn_svr_model, X_train, y_train, cv=10)
print('Cross validation scores', cros_val)
print('Mean cross validation of scores', cros_val.mean())

svr_pred=cn_svr_model.predict(X_test)



from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
score_r2=r2_score(svr_pred, y_test)
score_mae=mean_absolute_error(svr_pred, y_test)
print('R2 Score: ', score_r2)
print("The accuracy of our model is ", sc)
print("The Mean Absolute Error of our Model is {}".format(round(score_mae, 2)))

from sklearn.metrics import mean_squared_error
print("The Mean Squared Error of our Model is {}".format(round(mean_squared_error(svr_pred, y_test), 2)))


# ##### Results while using 20000 random records from dataset to modelling part
# 
# Cross validation scores [0.74787206 0.75756226 0.77091269 0.75444483 0.76074375 0.77497674
#  0.74844637 0.76965633 0.74298335 0.7741238 ]
#  
# Mean cross validation of scores 0.7601722182924401
# 
# R2 Score:  0.6465399453162952
# 
# The accuracy of our model is  0.7774435057460234
# 
# The Mean Absolute Error of our Model is 6.16
# 
# The Mean Squared Error of our Model is 95.1

# #### Model 3  (Decision Tree Regressor)

# In[342]:


## Using Decision tree regressor model

from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor()
cn_dtr_model=dtr.fit(X_train, y_train)
sc=cn_dtr_model.score(X_train, y_train)
## using 10 fold cross validation on model
from sklearn.model_selection import cross_val_score
cros_val=cross_val_score(cn_dtr_model, X_train, y_train, cv=10)
print('Cross validation scores', cros_val)
print('Mean cross validation of scores', cros_val.mean())

dtr_pred=cn_dtr_model.predict(X_test)



from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
score_r2=r2_score(dtr_pred, y_test)
score_mae=mean_absolute_error(dtr_pred, y_test)
print('R2 Score: ', score_r2)
print("The accuracy score of our model is ", sc)
print("The Mean Absolute Error of our Model is {}".format(round(score_mae, 2)))

from sklearn.metrics import mean_squared_error
print("The Mean Squared Error of our Model is {}".format(round(mean_squared_error(dtr_pred, y_test), 2)))


# ##### Results while using 20000 random records from dataset from modelling part
# 
# Cross validation scores [0.85817306 0.83243798 0.84924457 0.82337089 0.83019152 0.82513089
#  0.83650252 0.8132008  0.8294269  0.85617552]
#  
# Mean cross validation of scores 0.8353854653743757
# 
# R2 Score:  0.8215046300356605
# 
# The accuracy score of our model is  1.0
# 
# The Mean Absolute Error of our Model is 3.36
# 
# The Mean Squared Error of our Model is 71.74
# 
# 
# ##### Results while using 100000 random rows from modelling part
# 
# Cross validation scores [0.84568526 0.83932147 0.84370074 0.8410388  0.84226457 0.84915268
#  0.84563989 0.85058744 0.84180969 0.83753194]
#  
# Mean cross validation of scores 0.8436732495394171
# 
# R2 Score:  0.8492861576780535
# 
# The accuracy score of our model is  1.0
# 
# The Mean Absolute Error of our Model is 2.91
# 
# The Mean Squared Error of our Model is 61.56

# In[346]:


## Decision tree regressor worked best for control group with R2_score 0.8492 on 100000 data records

## saving the the best train model
import joblib
import pickle

filename="cn_dtr_model.joblib"
joblib.dump(cn_dtr_model, filename)


# # Loading test group of the dataset & marging the target variable named Empathy Score(extended) from questionary

# In[49]:


dff=[]
x=0
v=0
for filename in glob.glob(r'C:\Users\Mobin Ahmed\Downloads\data science & decision making\projects\EyeT\EyeT/*.csv'):
#     print(filename)
#     print(type(filename))
#     break
    if(filename.find('letter')!=-1):
        f=pd.read_csv(filename, nrows=3000)
        f.insert(1, column = "Empathy Score", value = 0)
        for i in range(len(f)):
            if(int(filename[len(filename)-5])>=0 and int(filename[len(filename)-5])<=9):
                if(filename[len(filename)-6]!='_'):
                    if(int(filename[len(filename)-14])>=0 and int(filename[len(filename)-14])<=9):
                        if(filename[len(filename)-15]!='_'):
                            x=(int(filename[len(filename)-15])*10+int(filename[len(filename)-14]))
#                             print(x)
                            x=x-1
                            f['Empathy Score'][i]=qu['Total Score extended'][x]
#                             v=v+1
                        else: 
                            x=(int(filename[len(filename)-14]))
#                             print(x)
                            x=x-1
                          
                            f['Empathy Score'][i]=qu['Total Score extended'][x]
#                             v=v+1

                else:
            #             print('single num')
                    if(int(filename[len(filename)-13])>=0 and int(filename[len(filename)-13])<=9):
                        if(filename[len(filename)-14]!='_'):
            #                     print('double number 2')
                            x=(int(filename[len(filename)-14])*10+int(filename[len(filename)-13]))
#                             print(x)
                            x=x-1
                      
                            f['Empathy Score'][i]=qu['Total Score extended'][x]
#                             v=v+1
                        else:
                            x=(int(filename[len(filename)-13]))
#                             print(x)
                            x=x-1
                            
                            f['Empathy Score'][i]=qu['Total Score extended'][x]
#                             v=v+1
        f=np.array(f)
        dff.append(f)



## concating as dataframe
dff=pd.concat([ pd.DataFrame(fl)
    for fl in dff
               ])


# In[50]:


dff.shape


# In[91]:


dff.head(50)


# In[179]:


dff.shape[0]/2


# In[348]:


dff.columns=col


# In[413]:


# First time trying with 20000 random records from dataset both with exploration(dttstx) and modelling(dttstm)

dttstx=dff.iloc[:int(dff.shape[0]/2), :]
dttstm=dff.iloc[(int(dff.shape[0]/2))+1:, :]

dttstx=dttstx.sample(20000)
dttstm=dttstm.sample(20000)



# Second time trying with 100000 random records from dataset both with exploration(dttstx) and modelling(dttstm)

# dttstx=dff.sample(100000)
# dttstm=dff.sample(100000)


# dttstx=dff
# dttstm=dff


# ### Data exploration and visualization(Test group)

# In[392]:



## Printing some important aspects of data

print('Row: ', dttstx.shape[0])
print('Columns: ', dttstx.shape[1])
print('\nFeatures: ', dttstx.columns.tolist())
print('\nMissing values: ', dttstx.isnull().any())
print('\nUnique Values: ', dttstx.nunique())



# In[393]:


dttstx['Empathy Score'].unique()


# In[96]:




# While observing all the features and its values of the dataset I can see some feature's value containing comma instead of dot.
# I really need to convert these commas to dots/points.


# In[125]:


# ### Now checking null values

dttstx.isna().sum()


# I can see that in the dataset here are almost every columns containing missing/null values.


# In[61]:




# ### Checking correlations among variables

def cor(dataset, thres):
    col_corr=set()
    corr_matrix=dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j])>thres:
                colname=corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr


corr_features=cor(dttstx, 0.8)  ## I can see that there are nothing correlations among variables with threshold 0.8(80%)
len(set(corr_features))


# ### Preprocessing & cleaning of  test group

# In[414]:


## feature selection

## while checking correlation among variabls/features, it showed nothing correlated.

## But I am now selecting those features whose are really match pattern for this group, and really
## important for my task. Same feature as I selected for control group before

dttstm=dttstm[feature_selection]


# In[415]:


# ### Converting feature's comma values to dot


for i in dttstm.columns:
    dttstm[i] = dttstm[i].astype(str).str.replace(',', '.')




# In[417]:



import math
dttstm=dttstm.replace('nan', math.nan)


# In[418]:



for i in numeric_features:
    dttstm[i] = dttstm[i].astype(float)



# In[419]:


# ### Handling missing values of the featrues using the methods 'F-fill' and 'B-fill'

for i in dttstm.columns:
    dttstm[i].ffill(inplace=True)

    
for i in dttstm.columns:
    dttstm[i].bfill(inplace=True)


# In[420]:


dttstm.head(50)


# In[421]:


# ### Separating feature variables and the target variables


dttstm_y=dttstm.iloc[:, 0]
xx=dttstm.iloc[:, 1:]


# In[422]:


xx.head()


# In[423]:


dttstm_y=dttstm_y.astype(int)
dttstm_y


# In[424]:


# ### Using One Hot Encoder to convert categorical variables to numeric

xx=tr.fit_transform(xx)


# In[425]:


xx=pd.DataFrame(xx)
xx.head()




## Removing outliers using Z-score and capping, in case found


for i in xx.columns:
    upper_limit=xx[i].mean()+3*xx[i].std()
    lower_limit=xx[i].mean()-3*xx[i].std()
    
    dttstm.loc[(xx[i]>=upper_limit), i]=upper_limit
    dttstm.loc[(xx[i]<=lower_limit), i]=lower_limit



# In[427]:


# ### Scaling the dataset using MinMaxScaler of the sklearn library

# Here I see from the dataset, the dataset does not maintain proper scaling the values of some variables. So, now I need to
# scale them to train the models properly.

scaler=StandardScaler()
xx=scaler.fit_transform(xx)


# In[428]:


xx=pd.DataFrame(xx)


# In[429]:


xx.head(50)


# ## Modelling part and Evaluation(Test group)

# In[430]:




# My dataset is now a clean and processed dataset. The dataset now is ready to use in different ML models.
# I am going to use this dataset to different model and find out the best for real world test with main task.
# I will be using some techniques like train test split, cross validation, some ML techniques, some forcasting techniques.
# And I will work on the evaluation of the models results.


## train test split
X_train, X_test, y_train, y_test=train_test_split(xx, dttstm_y, test_size=.3,random_state=40)


# #### Model 1(LR)

# In[431]:


## Using Multiple linear Regression
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
tst_lr_model=lr.fit(X_train, y_train)
sc=tst_lr_model.score(X_train, y_train)
## using 10 fold cross validation on model
from sklearn.model_selection import cross_val_score
cros_val=cross_val_score(tst_lr_model, X_train, y_train, cv=10)
print('Cross validation scores', cros_val)
print('Mean cross validation of scores', cros_val.mean())

lr_pred=tst_lr_model.predict(X_test)



from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
score_r2=r2_score(lr_pred, y_test)
score_mae=mean_absolute_error(lr_pred, y_test)
print('R2 Score: ', score_r2)
print("The accuracy score of our model is ", sc)
print("The Mean Absolute Error of our Model is {}".format(round(score_mae, 2)))

from sklearn.metrics import mean_squared_error
print("The Mean Squared Error of our Model is {}".format(round(mean_squared_error(lr_pred, y_test), 2)))


# ## While working with random 20000 records from 50% data of modelling part
# 
# Cross validation scores [0.55524679 0.53604502 0.56889078 0.54234096 0.55180701 0.55579731
#  0.5677466  0.55101198 0.57788126 0.57482264]
#  
# Mean cross validation of scores 0.5581590363277753
# 
# R2 Score:  0.19208582955104703
# 
# The accuracy of our model is  0.562028268440506
# 
# The Mean Absolute Error of our Model is 7.43
# 
# The Mean Squared Error of our Model is 85.98
# 
# 
# ## While working with random 100000 records from 50% data of modelling part
# 
# Cross validation scores [0.23099757 0.20871489 0.23247879 0.21700196 0.2260995  0.23133822
#  0.21424011 0.2361204  0.22007283 0.21183395]
#  
# Mean cross validation of scores 0.22288982208749583
# 
# R2 Score:  -2.4669848285498444
# 
# The accuracy score of our model is  0.22419225468510218
# 
# The Mean Absolute Error of our Model is 10.49
# 
# The Mean Squared Error of our Model is 230.49

# #### Model 2(SVR)

# In[388]:


## Using Support Vector Regressor model
from sklearn.svm import SVR
svr=SVR(kernel='rbf')
tst_svr_model=svr.fit(X_train, y_train)
sc=tst_svr_model.score(X_train, y_train)
## using 10 fold cross validation on model
from sklearn.model_selection import cross_val_score
cros_val=cross_val_score(tst_svr_model, X_train, y_train, cv=10)
print('Cross validation scores', cros_val)
print('Mean cross validation of scores', cros_val.mean())

svr_pred=tst_svr_model.predict(X_test)



from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
score_r2=r2_score(svr_pred, y_test)
score_mae=mean_absolute_error(svr_pred, y_test)
print('R2 Score: ', score_r2)
print("The accuracy score of our model is ", sc)
print("The Mean Absolute Error of our Model is {}".format(round(score_mae, 2)))

from sklearn.metrics import mean_squared_error
print("The Mean Squared Error of our Model is {}".format(round(mean_squared_error(svr_pred, y_test), 2)))


# ## While working with random 20000 records from 50% data of modelling part
# 
# Cross validation scores [0.41046727 0.41233265 0.47885456 0.43965481 0.4357067  0.44896521
#  0.438491   0.43772497 0.47461954 0.47771458]
#  
# Mean cross validation of scores 0.4454531298604847
# 
# R2 Score:  0.21568040358834706
# 
# The accuracy score of our model is  0.46296993780684714
# 
# The Mean Absolute Error of our Model is 6.18
# 
# The Mean Squared Error of our Model is 107.8

# #### Model 3(DTR)

# In[432]:


## Using Decision tree regressor model

from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor()
tst_dtr_model=dtr.fit(X_train, y_train)
sc=tst_dtr_model.score(X_train, y_train)
## using 10 fold cross validation on model
from sklearn.model_selection import cross_val_score
cros_val=cross_val_score(tst_dtr_model, X_train, y_train, cv=10)
print('Cross validation scores', cros_val)
print('Mean cross validation of scores', cros_val.mean())

dtr_pred=tst_dtr_model.predict(X_test)



from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
score_r2=r2_score(dtr_pred, y_test)
score_mae=mean_absolute_error(dtr_pred, y_test)
print('R2 Score: ', score_r2)
print("The accuracy score of our model is ", sc)
print("The Mean Absolute Error of our Model is {}".format(round(score_mae, 2)))

from sklearn.metrics import mean_squared_error
print("The Mean Squared Error of our Model is {}".format(round(mean_squared_error(dtr_pred, y_test), 2)))


# ## While working with random 20000 records from 50% data of modelling part
# 
# Cross validation scores [0.26208016 0.32653513 0.27662786 0.26409827 0.24776722 0.28932867
#  0.31844745 0.29352397 0.26646322 0.32804505]
#  
# Mean cross validation of scores 0.2872917010243576
# 
# R2 Score:  0.3024160422479224
# 
# The accuracy score of our model is  1.0
# 
# The Mean Absolute Error of our Model is 5.64
# 
# The Mean Squared Error of our Model is 137.8
# 
# 
# ## While working with random 100000 records from 50% data of modelling part
# 
# Cross validation scores [-0.50640467 -0.49598301 -0.46620347 -0.45172785 -0.48297866 -0.47122791
#  -0.49955945 -0.53532099 -0.47390204 -0.46114007]
#  
# Mean cross validation of scores -0.4844448111165519
# 
# R2 Score:  -0.4376022355000848
# 
# The accuracy score of our model is  1.0
# 
# The Mean Absolute Error of our Model is 10.65
# 
# The Mean Squared Error of our Model is 430.69

# In[433]:


## Decision tree regressor worked best aslo for test group with R2_score 0.3024 
#  on 100000 data records from 50% data of modelling part

## saving the the best train model
import joblib
import pickle

filename="tst_dtr_model.joblib"
joblib.dump(tst_dtr_model, filename)







#----------------------------------------------------------------------------------------
# ## References
Cites:

https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html

https://pandas.pydata.org/docs/user_guide/10min.html

https://scikit-learn.org/stable/getting_started.html

https://docs.python.org/3/tutorial/
# In[ ]:




