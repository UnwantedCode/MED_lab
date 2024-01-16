import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from scipy.optimize import curve_fit

allData = pd.read_csv('./Lab2/OS/Data.csv')
allData['Cumulated_number_of_errors'] = allData['Number_of_errors'].cumsum()
allData['Cumulated_number_of_errors_normalized'] = allData['Cumulated_number_of_errors'] / allData['Cumulated_number_of_errors'].max()

print(allData)

x_train, x_test, y_train, y_test = train_test_split(allData['Month_number'].values.reshape(-1, 1), allData['Cumulated_number_of_errors_normalized'].values.reshape(-1, 1), test_size=0.2, random_state=0)

linearModel = LinearRegression()
linearModel.fit(x_train, y_train)

logisticModel = LogisticRegression()
# logisticModel.fit(x_train, y_train)

def logistic_function(x, a, b):
    return 1 / (1 + np.exp(-(a * x + b)))

params, covariance = curve_fit(logistic_function, allData['Month_number'].values, allData['Cumulated_number_of_errors_normalized'].values)

print(params)
print(logistic_function(allData['Month_number'], *params))

def ShowNumberOfErrors():
    plt.plot(allData['Month_number'], allData['Number_of_errors'], c='b')

    plt.xlabel('Nr miesiąca')
    plt.ylabel('Ilość błędów')
    plt.title('Wykaz ilości błędów w zależności od miesiąca')
    plt.grid(True)
    plt.legend(['Ilość błędów'])

    plt.show()

def ShowCumulatedNumberOfErrors():
    plt.plot(allData['Month_number'], allData['Cumulated_number_of_errors_normalized'], c='r')
    # plt.plot(allData['Month_number'], logisticModel.predict(allData['Month_number'].values.reshape(-1, 1)), c='g')
    plt.plot(allData['Month_number'], linearModel.predict(allData['Month_number'].values.reshape(-1, 1)), c='b')
    plt.plot(allData['Month_number'], logistic_function(allData['Month_number'], *params), c='y')

    plt.xlabel('Nr miesiąca')
    plt.ylabel('Ilość błędów')
    plt.title('Wykaz ilości błędów w zależności od miesiąca')
    plt.grid(True)
    plt.legend(['Ilość błędów', 'Ilość błędów (regresja logistyczna)', 'Ilość błędów (regresja liniowa)', 'Ilość błędów (regresja logistyczna z krzywą)'])

    plt.show()

ShowCumulatedNumberOfErrors()
