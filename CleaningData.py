
# coding: utf-8

# # import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn import linear_model
# from sklearn import ensemble
# import seaborn as sns
# # Ensures graphs to be displayed in ipynb
# %matplotlib inline 
# #loading train data
# train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
# # read data into dataframe
# titanic_df = pd.read_csv(train_url,header=0)  # Always use header=0 to read header of csv files
# titanic_df
# 
# 

# In[42]:

# Import Libraries 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import ensemble
import seaborn as sns
# Ensures graphs to be displayed in ipynb
get_ipython().magic(u'matplotlib inline')
#import train and test data
train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
test_url =  "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"

# read data into dataframe
titanic_df = pd.read_csv(train_url,header=0)  # Always use header=0 to read header of csv files
test = pd.read_csv(test_url,header = 0)


# In[43]:

train.isnull().sum()


# In[44]:

test.isnull().sum()


# In[45]:

#set null values in 'Embarked' to 'C'
_ = train.set_value(train.Embarked.isnull(), 'Embarked', 'C')


# In[46]:

#set null values in fare to 8.05, most common fare value
_ = test.set_value(test.Fare.isnull(), 'Fare', 8.05)


# In[47]:

#replace missing values of Cabin with U0
full = pd.concat([train, test], ignore_index=True)
_ = full.set_value(full.Cabin.isnull(), 'Cabin', 'U0')


# In[48]:



