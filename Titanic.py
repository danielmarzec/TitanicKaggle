import pandas as pd
import numpy as np 
from sklearn import tree

#loading train and test datasets
train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
test_url =  "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"

#creating training dataframe
train=pd.read_csv(train_url)
test=pd.read_csv(test_url)

#converting the male and female groups to integer form
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1

#Imputing Embarked variable
train["Embarked"] = train["Embarked"].fillna("S")

#Converting Embarked classes to integers
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

target = train["Survived"].values

#extract features for test and trains set
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values
test_features = train[["Pclass", "Sex", "Age", "Fare"]].values

#predict and print test set
my_prediction = my_tree_one.predict(test_features)
print(my_prediction)

#test solution size
print(my_solution.shape)

#write solution to a csv file
my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])


