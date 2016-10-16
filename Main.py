
import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier

#import train and test data
train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
test_url =  "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"

#load train data in train_df
train_df = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv", header=0)      

#cleaning data..

#Replacing male with 0 and female with 1
train_df['Sex'][train_df['Sex']=='male'] == 0
train_df['Sex'][train_df['Sex']=='female'] == 1

#Imputing Embarked variable
train_df["Embarked"] = train_df["Embarked"].fillna("C")

#Converting Embarked classes to integers
train_df["Embarked"][train_df["Embarked"] == "S"] = 0
train_df["Embarked"][train_df["Embarked"] == "C"] = 1
train_df["Embarked"][train_df["Embarked"] == "Q"] = 2

# All the ages with no data -> make the median of all Ages
median_age = train_df['Age'].dropna().median()
if len(train_df.Age[ train_df.Age.isnull() ]) > 0:
    train_df.loc[ (train_df.Age.isnull()), 'Age'] = median_age

#load test data into test_df
test_df = pd.read_csv(test_url, header=0) 

#replace nan fare with median value
test_df.Fare[152] = test_df.Fare.median()     

#Replacing male with 0 and female with 1
test_df['Sex'][test_df['Sex']=='male'] == 0
test_df['Sex'][test_df['Sex']=='female'] == 1

#Imputing Embarked variable
test_df["Embarked"] = test_df["Embarked"].fillna("C")

#Converting Embarked classes to integers
test_df["Embarked"][test_df["Embarked"] == "S"] = 0
test_df["Embarked"][test_df["Embarked"] == "C"] = 1
test_df["Embarked"][test_df["Embarked"] == "Q"] = 2


# All the ages with no data -> make the median of all Ages
median_age = test_df['Age'].dropna().median()
if len(test_df.Age[ test_df.Age.isnull() ]) > 0:
    test_df.loc[ (test_df.Age.isnull()), 'Age'] = median_age

# All the missing Fares -> assume median of their respective class
if len(test_df.Fare[ test_df.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):                                              # loop 0 to 2
        median_fare[f] = test_df[ test_df.Pclass == f+1 ]['Fare'].dropna().median()
    for f in range(0,3):                                              # loop 0 to 2
        test_df.loc[ (test_df.Fare.isnull()) & (test_df.Pclass == f+1 ), 'Fare'] = median_fare[f]


# Convert back to a numpy array
train_data = train_df.values
test_data = test_df.values


print 'Training...'
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit( train_data[0::,1::], train_data[0::,0] )

print 'Predicting...'
output = forest.predict(test_data).astype(int)


predictions_file = open("myfirstforest.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'