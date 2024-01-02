import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

allData = pd.read_csv('./Lab2/Marriage/Data.csv')
allData['Married'] = allData['State'].apply(lambda x: 1 if x == 'M' else 0)
allData['Free'] = allData['State'].apply(lambda x: 1 if x == 'W' else 0)

allYears = allData['Year'].unique()
allYears.sort()

x_train, x_test, y_train, y_test = train_test_split(allData['Year'].values.reshape(-1, 1), allData['Married'].values.reshape(-1, 1), test_size=0.2, random_state=0)

linearModel = LinearRegression()
linearModel.fit(x_train, y_train)

logisticModel = LogisticRegression()
logisticModel.fit(x_train, y_train)

cumulatedMarried = allData.groupby('Year')['Married'].sum().reset_index()
cumulatedFree = allData.groupby('Year')['Free'].sum().reset_index()

def ShowNumberOfMarried():
    plt.scatter(cumulatedMarried['Year'], cumulatedMarried['Married'], c='b')
    plt.scatter(cumulatedFree['Year'], cumulatedFree['Free'], c='r')

    plt.xlabel('Rok')
    plt.ylabel('Ilość studentów')
    plt.title('Wykaz ilości związków małżeńskich i stanu wolnego w zależności od roku studiów')
    plt.grid(True)
    plt.legend(['W związku małżeńskim', 'Stan wolny'])

    plt.show()

def ShowRegression():
    x_values = np.arange(0, 6, 0.01)
    plt.plot(x_values, linearModel.predict(x_values.reshape(-1, 1)), 'r-')
    plt.plot(x_values, logisticModel.predict(x_values.reshape(-1, 1)), 'b-')

    plt.xlabel('Rok')
    plt.ylabel('Stan cywilny')
    plt.title('Regresja dla stanu cywilnego w zależności od roku studiów')
    plt.grid(True)
    plt.legend(['Regresja liniowa', 'Regresja logistyczna'])

    plt.show()

def TestModel(model, modelName):
    print('Test modelu: ', modelName)

    y_pred = (model.predict(x_test) > 0.5).astype(int)

    accuracyScore = accuracy_score(y_test, y_pred)
    print('Dokładność: ', accuracyScore)

    confusionMatrix = confusion_matrix(y_test, y_pred)
    print('Macierz pomyłek: \n', confusionMatrix)

    classificationReport = classification_report(y_test, y_pred)
    print('Raport klasyfikacji: \n', classificationReport)

ShowNumberOfMarried()
ShowRegression()
TestModel(linearModel, 'Regresja liniowa')
TestModel(logisticModel, 'Regresja logistyczna')

