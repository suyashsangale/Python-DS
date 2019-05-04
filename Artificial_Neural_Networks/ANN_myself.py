
import os
os.chdir("D:\Deep_Learning_A_Z\Artificial_Neural_Networks")
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
#X = dataset.iloc[:, [2, 3]].values  Takes only 2 & 3
X = dataset.iloc[:,3:13].values  
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_1.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Lets make the ANN
# Import keras libraries and packages
import keras
from keras.models import Sequential  # Initialize Neural Network
from keras.layers import Dense #Build the layers of ANN

#Initalizing the ANN
classifier= Sequential()

# Adding input layer and first hidden layer
classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))
classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))

#Adding output layer
classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))

#Compling the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
 #Fitting the ANN to trainign set
 classifier.fit(X_train,y_train,batch_size=10,epochs=100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#New observation Prediction
new_prediction = classifier.predict(sc.transform(np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)


## Part4 - Evaluating, Implemnting & Tuning ANN
#Evaluating using cross validation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential  # Initialize Neural Network
from keras.layers import Dense
def build_classifier():
    classifier= Sequential()
    classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))
    classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))
    classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return classifier

classifier= KerasClassifier(build_fn= build_classifier,batch_size=10,epochs=100)
accuracies=cross_val_score(estimator= classifier,X = X_train,y = y_train,cv = 10)
mean = accuracies.mean()
varience = accuracies.std()

#Dropuout Regularization to avoid overfitting
from keras.layers import Dropout
#Initalizing the ANN
classifier= Sequential()
# Adding input layer with drop out 
classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))
classifier.add(Dropout(rate=0.1))
# Adding first hidden layer with drop out
classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))
classifier.add(Dropout(rate=0.1))




#Tuning ANN with GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential  # Initialize Neural Network
from keras.layers import Dense
def build_classifier(optimizer):
    classifier= Sequential()
    classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))
    classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))
    classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
    classifier.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
    return classifier

classifier= KerasClassifier(build_fn= build_classifier)


parameters = {'batch_size' : [25,32],
              'epochs' : [100,500],
              'optimizer': ['adam','rmsprop']}

grid_search = GridSearchCV(estimator=classifier,param_grid=parameters, scoring='accuracy',cv=10)
grid_search = grid_search.fit(X_train,y_train)
best_params = grid_search.best_params_
best_score = grid_search.best_score_
