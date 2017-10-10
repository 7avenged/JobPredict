import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

input_file = "PATH AND NAME OF THE FILE.csv"
df = pd.read_csv(input_file, header = 0)
d = {'Y': 1, 'N': 0}   #create a list to put value of Y to 1 and of N to 0
df['Hired'] = df['Hired'].map(d)  #map the hired column's values to 0s and 1s(as they are already in Y and N 
df['Employed?'] = df['Employed?'].map(d)#map the employed column's values to 0s and 1s(as they are already in Y and N 
df['Top-tier school'] = df['Top-tier school'].map(d) #map the top-tier school column's values to 0s and 1s(as they are already in Y and N
df['Interned'] = df['Interned'].map(d) #map the interned column's values to 0s and 1s(as they are already in Y and N
d = {'BS': 0, 'MS': 1, 'PhD': 2}  #create a list to put value of BS to 0 MS to 1 phd to 2
df['Level of Education'] = df['Level of Education'].map(d)  ##map the Level of Education column's values to 0s,1s and 2s(as they are already #in  BS, MS and PhD
features = list(df.columns[:6])    #make a feature variable to take in the names of all the features(here from column 1 to 6)

y = df["Hired"]  #the column containing answers to whether people were hired or not, both in 0s and 1s
X = df[features]  #input feature column list with all the data
clf = tree.DecisionTreeClassifier() #the classifier is initialized
clf = clf.fit(X,y) #the classifier is run and trained

# to prevent overfitting of the training data, this training data can be divided into many decision trees in a forest

clf = RandomForestClassifier(n_estimators=10) #n_estimators is the number of trees wanted
clf = clf.fit(X, y) #the classifier is run and trained

#Predict employment of an employed 10-year veteran
print (clf.predict([[10, 1, 4, 0, 0, 1]]))  #all values of features for testing passed manually to predict the output
#print (clf.predict([[10, 0, 4, 0, 0, 0]]))
