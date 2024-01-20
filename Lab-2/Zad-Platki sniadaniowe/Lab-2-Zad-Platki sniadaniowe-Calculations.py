import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

allData = pd.read_csv('./Lab-2/Zad-Platki sniadaniowe/Lab-2-Zad-Platki sniadaniowe-Data.csv')
print('Wszystkie dane: ')
with pd.option_context('display.max_rows', None):
    print(allData)
print()

continiousVariables = ['kalorie', 'cukry', 'weglowodany', 'proteiny', 'tluszcz', 'sod', 'blonnik', 'potas']
categories = ['polka_1', 'polka_2', 'polka_3']

print('Obliczamy modele dla zmiennych ciągłych: ', continiousVariables)
print()

x_train, x_test, y_train, y_test = train_test_split(allData[continiousVariables], allData[categories], test_size=0.3)

logisticModel1 = LogisticRegression(max_iter=1000)
logisticModel1.fit(x_train, y_train['polka_1'])

print('Współczynniki regresji logistycznej dla polki 1: ', logisticModel1.coef_, logisticModel1.intercept_)
print()

# test model
y_pred = logisticModel1.predict(x_test)
print('Dokładność modelu: ', accuracy_score(y_test['polka_1'], y_pred))

print('Macierz pomyłek: ')
print(confusion_matrix(y_test['polka_1'], y_pred))
print()

print('Raport klasyfikacji: ')
print(classification_report(y_test['polka_1'], y_pred))
print()



logisticModel2 = LogisticRegression(max_iter=1000)
logisticModel2.fit(x_train, y_train['polka_2'])

print('Współczynniki regresji logistycznej dla polki 2: ', logisticModel2.coef_, logisticModel2.intercept_)
print()

# test model
y_pred = logisticModel2.predict(x_test)
print('Dokładność modelu: ', accuracy_score(y_test['polka_2'], y_pred))

print('Macierz pomyłek: ')
print(confusion_matrix(y_test['polka_2'], y_pred))
print()

print('Raport klasyfikacji: ')
print(classification_report(y_test['polka_2'], y_pred))
print()

logisticModel3 = LogisticRegression(max_iter=1000)
logisticModel3.fit(x_train, y_train['polka_3'])

print('Współczynniki regresji logistycznej dla polki 3: ', logisticModel3.coef_, logisticModel3.intercept_)
print()

# test model
y_pred = logisticModel3.predict(x_test)
print('Dokładność modelu: ', accuracy_score(y_test['polka_3'], y_pred))

print('Macierz pomyłek: ')
print(confusion_matrix(y_test['polka_3'], y_pred))
print()

print('Raport klasyfikacji: ')
print(classification_report(y_test['polka_3'], y_pred))
print()

# def calculateModel(variable, category):
#     model = LinearRegression()
#     model.fit(allData[variable].values.reshape(-1, 1), allData[category].values.reshape(-1, 1))
#     return model


# def ShowShelfNumber(variable, category):
#     model = calculateModel(variable, category)

#     plt.scatter(allData[variable], allData[category],  c='r')
#     plt.plot(allData[variable], model.predict(allData[variable].values.reshape(-1, 1)), c='b')
#     plt.xlabel('Ilość')
#     plt.ylabel('Czy na ' + category)
#     plt.title('Wykres dla zmiennej ' + variable)
#     plt.grid(True)
#     plt.legend(['Punkty', 'Regresja liniowa'])
#     plt.show()

# for variable in continiousVariables:
#     for category in categories:
#         ShowShelfNumber(variable, category)
