import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X, y)
print (clf.predict([[10, 1, 4, 0, 0, 0]]))
print (clf.predict([[10, 0, 4, 0, 0, 0]]))
input_file = "PATH TO/PastHiredpeople.csv"
df = pd.read_csv(input_file, header = 0)
d = {'Y': 1, 'N': 0}
df['Hired'] = df['Hired'].map(d)
df['Employed?'] = df['Employed?'].map(d)
df['Top-tier school'] = df['Top-tier school'].map(d)
df['Interned'] = df['Interned'].map(d)
d = {'BS': 0, 'MS': 1, 'PhD': 2}
df['Level of Education'] = df['Level of Education'].map(d)
features = list(df.columns[:6])
y = df["Hired"]
X = df[features]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,y)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X, y)
print (clf.predict([[10, 1, 4, 0, 0, 0]]))
print (clf.predict([[10, 0, 4, 0, 0, 0]]))
