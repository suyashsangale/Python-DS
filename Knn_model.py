# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 21:07:34 2019

@author: Suyash
"""

import os
os.getcwd()
os.chdir('C:\\Users\\Suyash\\Desktop\\NMIMS')
import pandas as pd
df=pd.read_csv("diabetes.csv")
df.head()
df.columns
X=df.drop(["diabetes"],1)
y=df["diabetes"]

from sklearn.neighbors import KNeighborsClassifier #Creating Knn model
knn_model = KNeighborsClassifier(metric='euclidean')
knn_model

from sklearn.model_selection import train_test_split as tts # Creating train test split
X_train,X_test,y_train,y_test= tts(X,y,test_size=0.3,random_state=42)

knn_model.fit(X_train,y_train) #Fitting the model into Train set
y_pred=knn_model.predict(X_test) #Calculating Y prediction using same model on x_test
print(list(y_pred))
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred) #Comparing y_test and y_pred

#Crossvalidation
