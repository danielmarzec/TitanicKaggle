import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Load and create test and train data frames
train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)

test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url)

#Print the `head` of the train and test dataframes
print(train.head())
print(test.head())

#create child column
train["Child"]= float('NaN')

#Assign 1 to passengers under 18 and 0 to older 
train["Child"][train["Child"]<18] = 1
train["Child"][train["Child"]>=18] = 0

#converting male and female to integers
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1

#filling Embarked values
train["Embarked"] = train["Embarked"].fillna("C")

#Convert Embarked to Integers
# Convert the Embarked classes to integer form
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

#more data cleaning inserted here
#
##
###
####
#####
######

test.Fare[152] = test.Fare.median()

#Importing Features that we want 
features_forest = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
target = train["Survived"].values

# Building and fitting my_forest
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)
my_forest = forest.fit(features_forest, target)

# Print the score of the random fitted forest
print("Score of Random Forest: ")
print(my_forest.score(features_forest, target))

# Compute predictions on our test set features then print the length of the prediction vector
target = train["Survived"].values
test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
pred_forest = my_forest.predict(test_features)
print("Length of Prediction Vector: ")
print(len(pred_forest))

#Print features importances
print("my_tree_two feature importance: ")
print(my_tree_two.features_importances_)
print("my_forest feature importance: ")
print(my_forest.feature_importances_)

#Compute and print the mean accuracy score for both models
print("mean accuracry score for my_tree_two")
print(my_tree_two.score(features_two, target))
print("mean accuracry score for my_forest")
print(my_forest.score(features_forest,target))


