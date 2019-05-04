# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 22:47:50 2019

@author: Suyash
"""

import pandas as pd
df=pd.read_csv("gapminder.csv")
df.columns
import numpy as np
numerical_data=df.select_dtypes(include=np.number)
import seaborn as sns
sns.heatmap(numerical_data.corr(),annot=True)
features_by_filter_method=["HIV","GDP","BMI_female","child_mortality"]
df.head()

df.isnull().sum()
X=df.drop(["life"],1)
y=df["life"]
X=pd.get_dummies(X)
 from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
col_names=list(X) ## Because after MinMax Columns names are gone
X=scaler.fit_transform(X)
X=pd.DataFrame(X)
X.columns=col_names


X=X[features_by_filter_method]
###By train test split
from sklearn.model_selection import train_test_split as tts
X_train,X_test,y_train,y_test=tts(X,y,test_size=0.3,random_state=42)
from sklearn.neighbors import KNeighborsRegressor
knn_reg=KNeighborsRegressor()

knn_reg.fit(X_train,y_train)
y_pred=knn_reg.predict(X_test)

from sklearn.metrics import r2_score
r2_score(y_pred,y_test)


# By Cross-Validation
from sklearn.model_selection import cross_val_score
cross_val_score(knn_reg,X,y,cv=10).mean()

# By Grid Search
import numpy as np
params={"n_neighbors":np.arange(1,8)}
from sklearn.model_selection import GridSearchCV
knn_cv=GridSearchCV(knn_reg,param_grid=params,scoring="r2")
knn_cv.fit(X,y)
knn_cv.best_params_
knn_cv.best_estimator_
knn_cv.best_score_
