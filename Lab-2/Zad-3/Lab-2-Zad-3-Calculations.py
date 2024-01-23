import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

allData = pd.read_csv('./Lab-2/Zad-3/Lab-2-Zad-3-Data.csv')
allData['Cumulated_number_of_errors'] = allData['Number_of_errors'].cumsum()
allData['temp_y_prim'] = allData['Number_of_errors'] / allData['Cumulated_number_of_errors'] 
print('Dane')
print(allData)
print()

x_train, x_test, y_train, y_test = train_test_split(allData[['Month_number', 'Number_of_errors', 'Cumulated_number_of_errors', 'temp_y_prim']], allData['Cumulated_number_of_errors'], test_size=0.01, random_state=42)

print(x_train)
print()

temp_linearModel = LinearRegression()
temp_linearModel.fit(x_train['Cumulated_number_of_errors'].values.reshape(-1, 1), x_train['temp_y_prim'])

# k, a, b
a = temp_linearModel.intercept_
k = -a/temp_linearModel.coef_[0]

print('a: ', a)
print('k: ', k)

temp_sum = np.sum( np.log(k/x_train['Cumulated_number_of_errors'] - 1))
print('Suma: ', temp_sum)

temp_sum2 = a*np.sum(x_train['Month_number'])
print('Suma2: ', temp_sum2)

b = np.exp((temp_sum + temp_sum2) / len(x_train['Month_number']))

print('b: ', b)

y = k / (1 + b * np.exp(-a * x_train['Month_number']))
print('y')
print(y)

# linear regression
linearModel = LinearRegression()
linearModel.fit(x_train['Month_number'].values.reshape(-1, 1), y_train)

# calculate errors
meanSquaredErrorForLinear = np.mean((linearModel.predict(x_test['Month_number'].values.reshape(-1, 1)) - y_test) ** 2)
print('Błąd średniokwadratowy dla regresji liniowej: ', meanSquaredErrorForLinear)

meanSquaredErrorForLogit = np.mean((k / (1 + b * np.exp(-a * x_test['Month_number'])) - y_test) ** 2)
print('Błąd średniokwadratowy dla regresji logistycznej: ', meanSquaredErrorForLogit)

def ShowCumulatedNumberOfErrors():
    plt.plot(allData['Month_number'], allData['Cumulated_number_of_errors'], c='r')
    plt.plot(allData['Month_number'], k / (1 + b * np.exp(-a * allData['Month_number'])), c='b')
    plt.plot(allData['Month_number'], linearModel.predict(allData['Month_number'].values.reshape(-1, 1)), c='g')

    plt.xlabel('Nr miesiąca')
    plt.ylabel('Ilość błędów')
    plt.title('Zsumowana ilości błędów w zależności od miesiąca')
    plt.grid(True)
    plt.legend(['Ilość błędów', 'Ilość błędów (regresja liniowa)'])

    plt.show()

ShowCumulatedNumberOfErrors()

