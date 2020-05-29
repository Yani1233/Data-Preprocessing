#Data Preprocessing
#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('C:/Users/Arun Yani/Desktop/DATA/Part1-Data-Preprocessing/Section 3 - Data Preprocessing in Python/Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values
#Taking care of missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN' , strategy = 'mean' , axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])  

#Label encoder
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
Labelencoder_X = LabelEncoder()
X[:, 0] = Labelencoder_X.fit_transform(X[:, 0])
Onehotencoder = OneHotEncoder(categorical_features = [0])
X = Onehotencoder.fit_transform(X).toarray()
Labelencoder_Y = LabelEncoder()
Y = Labelencoder_Y.fit_transform(Y)

#splitting the dataset
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test  = sc_X.fit_transform(X_test)