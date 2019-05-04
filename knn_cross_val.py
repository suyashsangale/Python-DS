# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 21:39:46 2019

@author: Suyash
"""
##### KNN using Cross-Validation##############
import pandas as pd
df=pd.read_csv("diabetes.csv")
df.columns
X=df.drop(["diabetes"],1)
y=df["diabetes"]

from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier(n_neighbors=5)
knn

from sklearn.model_selection import cross_val_score
cross_val_score(knn,X,y,cv=10).mean()
