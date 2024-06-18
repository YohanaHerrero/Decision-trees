from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
import joblib
import pandas as pd

#load the data
df = pd.read_csv('music.csv')

#create the model that reproduces the data
model = DecisionTreeClassifier()
#we train the model with the input and output sets 
model.fit(X.values, y.values)

#make predictions
#E.g. predict which genre of music a 21 year old male likes (this we have in the original table)
model.predict([ [21,1] ])
#E.g. predict the same as above, together with a 22 year old woman and store it as "predictions"
predictions = model.predict([ [21,1], [22,0] ])

#measure the accuracy of the predictions
#20% of the data for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
#we train the model with the input and output TRAINING sets
model.fit(X_train.values, y_train.values)

#predictions with the X from TESTING, which will be compared to the testing output y_test to get the accuracy
predictions = model.predict( X_test.values )
accuracy = accuracy_score(y_test, predictions) #accuracy between 0 and 1

#If we need to run the above several times, we should save and load the predictive model (instead of running all every time)
import joblib
df = pd.read_csv('music.csv')
X = df.drop(columns=['genre']) 
y = df['genre']  
model = DecisionTreeClassifier()
model.fit(X.values, y.values)
#i give a name to the file where we wanna store the model
joblib.dump(model,'music-recommender.joblib') #creates a file in my directory

#I can now read that file with the model
model = joblib.load('music-recommender.joblib')
#make predictions
predictions = model.predict([ [21,1], [22,0] ])

#visualize decision trees
#feature names is our input column names, class names are the names of the output eg hiphop, classical, dance, etc (the unique list of y in alphabetical order)
tree.export_graphviz(model, out_file='music-recommender.dot', feature_names=['age', 'gender'], 
                     class_names=sorted(y.unique()), label='all', rounded=True, filled=True)
#the output file can be visualized with graphviz, an extension from the visual studio code 
