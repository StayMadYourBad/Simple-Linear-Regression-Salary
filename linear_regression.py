# ---------------------------------------------------------------------------- #
#                Importing the necessary libraries and datasets                #
# ---------------------------------------------------------------------------- #

#Adding a shortcut to save character count
import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Importing Dataset

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1 ].values #Taking all the rows, all columns excluding the last column
y = dataset.iloc[:, -1].values #Taking the last column


# ---------------------------------------------------------------------------- #
#                      Splitting data into train and test                      #
# ---------------------------------------------------------------------------- #

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)

# ---------------------------------------------------------------------------- #
#                         Training and Predicting Data                         #
# ---------------------------------------------------------------------------- #

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test) #y_pred is the y predicted values

# ---------------------------------------------------------------------------- #
#                       Visualizing the Training Data Set                      #
# ---------------------------------------------------------------------------- #

plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color = "blue")
plt.title("Salary vs Experience (Training Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()


# ---------------------------------------------------------------------------- #
#                         Visualizing the Test Data Set                        #
# ---------------------------------------------------------------------------- #

plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color = "blue")
plt.title("Salary vs Experience (Test Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()