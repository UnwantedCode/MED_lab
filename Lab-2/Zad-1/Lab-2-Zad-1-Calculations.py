import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#M - Marriage
#W - Free

allData = pd.read_csv('./Lab-2/Zad-1/Lab-2-Zad-1-Data.csv')
print('Wszystkie dane: ')
print(allData)
print()


x_train, x_test, y_train, y_test = train_test_split(allData['Year'].values.reshape(-1, 1), allData['State'].values.reshape(-1, 1), test_size=0.3)

dataFromTrain = pd.DataFrame({'Year': x_train.reshape(-1), 'State': y_train.reshape(-1)})
print('Dane treningowe: ')
print(dataFromTrain)
print()

def CalculateLogit(x):
    if x == 1:
        x = 0.9999999999
    return np.log(x / (1 - x))

groupedByYearAndState = dataFromTrain.groupby(['Year', 'State']).size().reset_index(name='Count')
groupedByYearAndState['probability']= groupedByYearAndState['Count'] / groupedByYearAndState.groupby('Year')['Count'].transform('sum')
groupedByYearAndState['Logits'] = groupedByYearAndState['probability'].apply(CalculateLogit)
print('Dane treningowe pogrupowane po roku i stanie: ')
print(groupedByYearAndState)
print()



# groupedByYear = allData.groupby('Year').size().reset_index(name='Count')
# print(groupedByYear)
# print()

# groupedByYearAndState = allData.groupby(['Year', 'State']).size().reset_index(name='Count')
# groupedByYearAndState['probability']= groupedByYearAndState['Count'] / groupedByYearAndState.groupby('Year')['Count'].transform('sum')
# groupedByYearAndState['Logits'] = groupedByYearAndState['probability'].apply(lambda x: np.log(x / (1 - x)))
# print(groupedByYearAndState)
# print()

marriedByYear = groupedByYearAndState[groupedByYearAndState['State'] == 'M']
print('Prawdopodobieństwo małżeństwa w zależności od roku studiów: ')
print(marriedByYear)
print()

logitModel = LinearRegression()
logitModel.fit(marriedByYear['Year'].values.reshape(-1, 1), marriedByYear['Logits'].values.reshape(-1, 1))
print('Współczynniki regresji logistycznej: ', logitModel.coef_, logitModel.intercept_)
print()

def CalculateProbabilityUsingLogit(year):
    return 1 / (1 + np.exp(-(logitModel.intercept_ + logitModel.coef_ * year)))

print('Prawdopodobieństwo małżeństwa w 1 roku: ', CalculateProbabilityUsingLogit(1))
print('Prawdopodobieństwo małżeństwa w 5 roku: ', CalculateProbabilityUsingLogit(5))
print()

linearModel = LinearRegression()
linearModel.fit(marriedByYear['Year'].values.reshape(-1, 1), marriedByYear['probability'].values.reshape(-1, 1))
print('Współczynniki regresji liniowej: ', linearModel.coef_, linearModel.intercept_)
print()

def CalculateProbabilityUsingLinear(year):
    return linearModel.intercept_ + linearModel.coef_ * year

print('Prawdopodobieństwo małżeństwa w 1 roku: ', CalculateProbabilityUsingLinear(1))
print('Prawdopodobieństwo małżeństwa w 5 roku: ', CalculateProbabilityUsingLinear(5))
print()


def ShowRegression():
    x_values = np.arange(1, 5, 0.01)
    plt.plot(x_values, CalculateProbabilityUsingLogit(x_values.reshape(-1, 1)), 'b-')
    plt.plot(x_values, CalculateProbabilityUsingLinear(x_values.reshape(-1, 1)), 'r-')

    plt.xlabel('Rok')
    plt.ylabel('Prawdopodobieństwo małżeństwa')
    plt.title('Regresje dla prawdopodobieństwa małżeństwa w zależności od roku studiów')
    plt.grid(True)
    plt.legend(['Regresja logistyczna', 'Regresja liniowa'])

    plt.show()

ShowRegression()

dataFromTest = pd.DataFrame({'Year': x_test.reshape(-1), 'State': y_test.reshape(-1)})
print('Dane testowe: ')
print(dataFromTest)
print()

def TestModel(model, modelName):
    print('Test modelu: ', modelName)

    y_pred = (model.predict(x_test) > 0.5).astype(int)
    y_test_int = (y_test == 'M').astype(int)

    accuracyScore = accuracy_score(y_test_int, y_pred)
    print('Dokładność: ', accuracyScore)

TestModel(logitModel, 'Regresja logistyczna')
TestModel(linearModel, 'Regresja liniowa')




