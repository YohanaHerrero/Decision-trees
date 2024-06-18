from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import sklearn.tree as tree
from sklearn import preprocessing
from sklearn import metrics
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

#load data
my_data = pd.read_csv('drug200.csv', delimiter=",")

#define the variables
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
y = my_data["Drug"]

#Data preprocessing
#Because some variables are not numbers (eg BP, sex and cholesterol), I convert these features to numerical values 
#using the LabelEncoder() method. This converts the categorical variable into indicator variable (0 or 1).
from sklearn import preprocessing
indicator_sex = preprocessing.LabelEncoder()
indicator_sex.fit(['F','M'])
X[:,1] = indicator_sex.transform(X[:,1]) #a matrix with same rows and columns as table

indicator_BP = preprocessing.LabelEncoder()
indicator_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = indicator_BP.transform(X[:,2])

indicator_Chol = preprocessing.LabelEncoder()
indicator_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = indicator_Chol.transform(X[:,3]) 

#create train and test sets
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

#set up the decision tree
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4) #decision tree based on lowest entropy criterion
#fit the model to the training set
drugTree.fit(X_trainset,y_trainset)
#make predictions on the testing dataset
predTree = drugTree.predict(X_testset)
#compare predicted and real 
print(predTree[:5])
print(y_testset[:5])

#evaluate the accuracy of the model
print("DecisionTrees's Accuracy (between 0 and 1): ", metrics.accuracy_score(y_testset, predTree))

#visualize the tree
export_graphviz(drugTree, out_file='tree.dot', filled=True, feature_names=['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'])
