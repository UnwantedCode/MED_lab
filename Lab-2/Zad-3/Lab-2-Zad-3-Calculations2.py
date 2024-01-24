import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from test import find_parameters
from sklearn.linear_model import LinearRegression


allData = pd.read_csv('./Lab-2/Zad-3/Lab-2-Zad-3-Data.csv')
allData['Cumulated_number_of_errors'] = allData['Number_of_errors'].cumsum()
allData['y'] = allData['Cumulated_number_of_errors']

# szacowanie parametrów

iteration = 0

k = 100
a = 100
b = 100

def model_f(parameters):
    return parameters['k'] / (1 + parameters['b'] * np.exp(-parameters['a'] * parameters['Month_number']))

# print(allData.to_dict(orient='records')[2]['y'])

#print(model_f({'k': k, 'a': a, 'b': b}))
found_parameters = find_parameters({'k': allData['Cumulated_number_of_errors'].max(), 'a': 2, 'b': allData['Cumulated_number_of_errors'].max()}, model_f , allData.to_dict(orient='records'), iteration_limit=5)
print(found_parameters)

# k = 269.38
# a = 0.089
# b = 239.2



k = found_parameters['k']
a = found_parameters['a']
b = found_parameters['b']

meanSquaredErrorForLogit = np.mean((k / (1 + b * np.exp(-a * allData['Month_number'])) - allData['Cumulated_number_of_errors']) ** 2)
print('Błąd średniokwadratowy dla regresji logistycznej: ', meanSquaredErrorForLogit)

linearModel = LinearRegression()
linearModel.fit(allData['Month_number'].values.reshape(-1, 1), allData['Cumulated_number_of_errors'].values.reshape(-1, 1))
meanSquaredErrorForLinear = np.mean((linearModel.predict(allData['Month_number'].values.reshape(-1, 1)) - allData['Cumulated_number_of_errors'].values.reshape(-1, 1)) ** 2)
print('Błąd średniokwadratowy dla regresji liniowej: ', meanSquaredErrorForLinear)

allData['y_logisitic'] = k / (1 + b * np.exp(-a * allData['Month_number']))
allData['y_logisitic_error'] = (allData['y_logisitic'] - allData['Cumulated_number_of_errors']) ** 2
allData['y_logisitic_cumulated_error'] = allData['y_logisitic_error'].cumsum()
allData['y_linear'] = linearModel.predict(allData['Month_number'].values.reshape(-1, 1))
allData['y_linear_error'] = (allData['y_linear'] - allData['Cumulated_number_of_errors']) ** 2
allData['y_linear_cumulated_error'] = allData['y_linear_error'].cumsum()

print(allData.tail(10))

def ShowCumulatedErrors():
    plt.plot(allData['Month_number'], allData['y_logisitic_cumulated_error'], c='r')
    plt.plot(allData['Month_number'], allData['y_linear_cumulated_error'], c='g')

    plt.xlabel('Nr miesiąca')
    plt.ylabel('Suma błedu kwadratowego')
    plt.title('Zsumowana ilości błędów w zależności od miesiąca')
    plt.grid(True)
    plt.legend(['Ilość błędu (regresja logistyczna)', 'Ilość błędu (regresja liniowa)'])

    plt.show()

ShowCumulatedErrors()

def ShowCumulatedNumberOfErrors():
    plt.plot(allData['Month_number'], allData['Cumulated_number_of_errors'], c='r')
    plt.plot(allData['Month_number'], allData['y_logisitic'], c='b')
    plt.plot(allData['Month_number'], allData['y_linear'], c='g')

    plt.xlabel('Nr miesiąca')
    plt.ylabel('Ilość błędów')
    plt.title('Zsumowana ilości błędów w zależności od miesiąca')
    plt.grid(True)
    plt.legend(['Ilość błędów', 'Ilość błędów (regresja logistyczna)'])

    plt.show()

ShowCumulatedNumberOfErrors()
