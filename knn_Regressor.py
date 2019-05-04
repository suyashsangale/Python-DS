# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 22:34:58 2019

@author: Suyash
"""

import pandas as pd
df=pd.read_csv("diabetes.csv")
X=df.drop(["diabetes"],1)
y=df["diabetes"]

import numpy as np
params={"n_neighbors":np.arange(1,10)}


from sklearn.neighbors import KNeighborsRegressor
knn_reg=KNeighborsRegressor()

from sklearn.model_selection import GridSearchCV
knn_cv=GridSearchCV(knn_reg,param_grid=params)
knn_cv.fit(X,y)
knn_cv.best_params_
knn_cv.best_score_
