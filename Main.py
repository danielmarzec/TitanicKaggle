import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import HelpingFunctions as hf
import re


# Load and create test and train data frames
train_path = "train.csv"
train = pd.read_csv(train_path)

test_path = "test.csv"
test = pd.read_csv(test_path)

#create child column
train["Child"]= float('NaN')

#Gets a dictionary count of all the different titles
titles = dict()
for i in range(890):
	train['Title'] = hf.getTitle(train.Name[int(i)])

#Create TitleNum column
train['TitleNum'] = 4
train['TitleNum'][train['Title'] == 'Mr'] = 2
train['TitleNum'][train['Title'] == 'Miss'] = 1
train['TitleNum'][train['Title'] == 'Ms'] = 1
train['TitleNum'][train['Title'] == 'Mrs'] = 2
train['TitleNum'][train['Title'] == 'Master'] = 3

#assign masters to age = 10
train["Age"]= train['Age'].fillna(train['Age'].mean())

#Assign 1 to passengers under 18 and 0 to older 
train["Child"][train["Age"]<18] = 1
train["Child"][train["Age"]>=18] = 0


train["Age"][train["TitleNum"]==3]=1

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

'''
ticket_groups = dict()
for i in range(891):
	if ticket_groups.has_key(train.Ticket[i]):
		ticket_groups[train.Ticket[i]] = ticket_groups[train.Ticket[i]] + 1
	else:
		ticket_groups[train.Ticket[i]] = 1

train['Group'] = float('NaN')

for i in range(891):
	train.Group[i] = ticket_groups[train.Ticket[i]]

ticket_groups = dict()
for i in range(418):
	if ticket_groups.has_key(test.Ticket[i]):
		ticket_groups[test.Ticket[i]] = ticket_groups[test.Ticket[i]] + 1
	else:
		ticket_groups[test.Ticket[i]] = 1

test['Group'] = float('NaN')

for i in range(418):
	test.Group[i] = ticket_groups[test.Ticket[i]]
'''


#Gets a dictionary count of all the different titles
titles = dict()
for i in range(417):
	test['Title'] = hf.getTitle(test.Name[int(i)])

#Create TitleNum column
test['TitleNum'] = 4
test['TitleNum'][test['Title'] == 'Mr'] = 0
test['TitleNum'][test['Title'] == 'Miss'] = 1
test['TitleNum'][test['Title'] == 'Ms'] = 1
test['TitleNum'][test['Title'] == 'Mrs'] = 2
test['TitleNum'][test['Title'] == 'Master'] = 3


#filling Embarked values
test["Embarked"] = test["Embarked"].fillna("C")

#Convert Embarked to Integers
# Convert the Embarked classes to integer form
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2

test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1

test["Age"]= test['Age'].fillna(train['Age'].mean())

test['Child'] = float('NaN')
test["Child"][test["Age"]<11] = 1
test["Child"][test["Age"]>=11] = 0

#more data cleaning inserted here
#
##
###
####
#####
######


#create group column
train["Group"] = train['SibSp'] + train['Parch'] + 1
test["Group"]= test['SibSp'] + test['Parch'] + 1

#creating Family_size




#ended up not using this
#create feature called Group_size
train["Group_size"]= 2
train["Group_size"][train["Group"] == 1] = 1
train["Group_size"][train["Group"] > 5]= 3

test["Group_size"]= 2
test["Group_size"][test["Group"] == 1] = 1
test["Group_size"][test["Group"] > 5] = 3



test.Fare[152] = 14.4542

target = train["Survived"].values


#Importing Features that we want 
features_forest = train[["Pclass", "Age", "Sex","Fare"]].values
target = train["Survived"].values

# Building and fitting my_forest
forest = RandomForestClassifier(max_depth = 5, min_samples_split=4, n_estimators = 1500, random_state = 1)
my_forest = forest.fit(features_forest, target)

# Print the score of the random fitted forest
print("Score of Random Forest: ")
print(my_forest.score(features_forest, target))

# Compute predictions on our test set features then print the length of the prediction vector
target = train["Survived"].values
test_features = test[["Pclass", "Age", "Sex","Fare"]].values
pred_forest = my_forest.predict(test_features)
print("Length of Prediction Vector: ")
print(len(pred_forest))

#Print features importances
print("my_forest feature importance: ")
print(my_forest.feature_importances_)

#Compute and print the mean accuracy score for both models
print("mean accuracry score for my_forest")
print(my_forest.score(features_forest,target))

submission = pd.DataFrame({
        "PassengerId":test['PassengerId'],
        "Survived":pred_forest
    })
submission.to_csv('titanic.csv',index=False)
