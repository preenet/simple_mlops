"""
@author: pree.t@cmu.ac.th
"""

import pickle # we use this lib for dumping the model to binary format

import pandas as pd # this is for create dataframe for easy manipulation
from sklearn.ensemble import RandomForestClassifier # i selected this as my meta- classifer
from sklearn.metrics import accuracy_score # just for evaluation sake
from sklearn.model_selection import train_test_split # for simplest holdout sampling meothd

df = pd.read_csv('iris.data') # we open the dataset and parse as dataframe


X = df.iloc[:,:-1] # here, we select all row, and until the column before the last one, because the last on is the label
y = df.iloc[:,-1] # we select all row


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=0)


classifier = RandomForestClassifier()
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)


score = accuracy_score(y_test,y_pred) # 0.75
print(score)

pickle_out = open("model_iris.pkl","wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()

# you can extend this code to find the optimal model by using cross_val_score or for loop
# that would iterate over the model parameters (e.g., number of tree, etc.)