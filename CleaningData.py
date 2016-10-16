

# Import Libraries 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Ensures graphs to be displayed in ipynb
get_ipython().magic(u'matplotlib inline')
#import train and test data
train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
test_url =  "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"

# read data into dataframe
titanic_df = pd.read_csv(train_url,header=0)  # Always use header=0 to read header of csv files
test = pd.read_csv(test_url,header = 0)

#print amount of missing values in train and test
train.isnull().sum()

test.isnull().sum()

#set null values in 'Embarked' to 'C'
_ = train.set_value(train.Embarked.isnull(), 'Embarked', 'C')

#set null values in fare to 8.05, most common fare value
_ = test.set_value(test.Fare.isnull(), 'Fare', 8.05)

#replace missing values of Cabin with U0
full = pd.concat([train, test], ignore_index=True)
_ = full.set_value(full.Cabin.isnull(), 'Cabin', 'U0')


#create feature called group = sibSP + parch + 1
full['Group_num'] = full.Parch + full.SibSp + 1

#create feautre called Group_size where Group_size>4 = 'L' and <4 = 'S'
#More people survived when size was between 2 and 4
full['Group_size'] = pd.Series('M', index=full.index)
_ = full.set_value(full.Group_num>4, 'Group_size', 'L')
_ = full.set_value(full.Group_num==1, 'Group_size', 'S')

#Gets a dictionary count of all the different titles
titles = dict()
for i in range(890):
	if titles.has_key(train.Title[int(i)]):
		names[train.Title[int(i)]] = names[train.Title[int(i)]] + 1
	else:
		names[train.Title[int(i)]] = 1

#Changes all titles to an integer value
train['TitleNum'] = 4
train['TitleNum'][train['Title'] == 'Mr'] = 0
train['TitleNum'][train['Title'] == 'Miss'] = 1
train['TitleNum'][train['Title'] == 'Ms'] = 1
train['TitleNum'][train['Title'] == 'Mrs'] = 2
train['TitleNum'][train['Title'] == 'Master'] = 3



