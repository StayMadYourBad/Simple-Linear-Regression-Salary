# ---------------------------------------------------------------------------- #
#                Importing the necessary libraries and datasets                #
# ---------------------------------------------------------------------------- #

#Adding a shortcut to save character count
import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import  LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Importing Dataset

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1 ].values #Taking all the rows, all columns excluding the last column
y = dataset.iloc[:, -1].values #Taking the last column


# ---------------------------------------------------------------------------- #
#                      Splitting data into train and test                      #
# ---------------------------------------------------------------------------- #

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)