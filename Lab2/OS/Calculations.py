import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

allData = pd.read_csv('./Lab2/OS/Data.csv')
allData['Cumulated_number_of_errors'] = allData['Number_of_errors'].cumsum()
print('Dane')
print(allData)
print()

x_train, x_test, y_train, y_test = train_test_split(allData['Month_number'].values.reshape(-1, 1), allData['Cumulated_number_of_errors'].values.reshape(-1, 1), test_size=0.2, random_state=0)

linearModel = LinearRegression()
linearModel.fit(x_train, y_train)

scale = y_train.max() + 1
logitModel = LinearRegression()
logitModel.fit(x_train, np.log(y_train/(scale - y_train)))


def ShowCumulatedNumberOfErrors():
    plt.plot(allData['Month_number'], allData['Cumulated_number_of_errors'], c='r')
    plt.plot(allData['Month_number'], linearModel.predict(allData['Month_number'].values.reshape(-1, 1)), c='b')
    plt.plot(allData['Month_number'], scale / (1 + np.exp(-logitModel.predict(allData['Month_number'].values.reshape(-1, 1)))), c='g')

    plt.xlabel('Nr miesiąca')
    plt.ylabel('Ilość błędów')
    plt.title('Zsumowana ilości błędów w zależności od miesiąca')
    plt.grid(True)
    plt.legend(['Ilość błędów', 'Ilość błędów (regresja liniowa)', 'Ilość błędów (regresja logistyczna)'])

    plt.show()

ShowCumulatedNumberOfErrors()

meanSquaredErrorForLinear = np.mean((linearModel.predict(x_test) - y_test) ** 2)
meanSquaredErrorForLogit = np.mean((scale / (1 + np.exp(-logitModel.predict(x_test))) - y_test) ** 2)

print('Błąd średniokwadratowy dla regresji liniowej: ', meanSquaredErrorForLinear)
print('Błąd średniokwadratowy dla regresji logistycznej: ', meanSquaredErrorForLogit)

