import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

allData = pd.read_csv('Data/NumberOfUsers.csv')

def showData():
    showingData = allData.copy()
    showingData['Timestamp'] = pd.to_numeric(pd.to_datetime(showingData['Date']))

    plt.scatter(showingData['Date'], showingData['Users_in_mln'])

    coefficients = [model.coef_[0], model.intercept_]
    trendLine = np.polyval(coefficients, showingData['Timestamp'])
    plt.plot(showingData['Date'], trendLine, color='red')
    plt.xticks(showingData['Date'][::10])

    plt.show()

# Train model ---------------------------------------------------------------
def trainModel(yearFrom, yearTo):
    global model

    trainData = allData[(yearFrom <= allData['Year']) & (allData['Year'] <= yearTo)].copy()

    trainX = pd.to_numeric(pd.to_datetime(trainData['Date'])).values.reshape(-1, 1)
    trainValues = trainData['Users_in_mln'].values

    model = LinearRegression()
    model.fit(trainX, trainValues)

    print("Współczynnik determinacji: ", model.score(trainX, trainValues))

def testModel(yearFrom, yearTo):
    testData = allData[(yearFrom <= allData['Year']) & (allData['Year'] <= yearTo)].copy()

    testX = pd.to_numeric(pd.to_datetime(testData['Date'])).values.reshape(-1, 1)

    testData['GeneratedByModel'] = model.predict(testX)
    testData['Error'] = testData['Users_in_mln'] - testData['GeneratedByModel']
    testData['Error_squared'] = testData['Error'] ** 2

    print(testData)
    print()
    meanError = np.mean(testData['Error'])
    print("Błąd średni: ", meanError)
    meanAbsoluteError = np.mean(np.abs(testData['Error']))
    print("Błąd średni bezwzględny: ", meanAbsoluteError)
    meanSquareError = np.mean(testData['Error_squared'])
    print("Błąd średniokwadratowy: ", meanSquareError)
    print("Pierwiastek błędu średniokwadratowego: ", np.sqrt(meanSquareError))
    print("Odchylenie standardowe błedu: ", np.std(testData['Error']))
    print("Średni absolutny błąd procentowy: ", np.mean(np.abs(testData['Error'] / testData['Users_in_mln'])) * 100, "%")
    print()

print("Trening modelu 2008-2015")
trainModel(2008, 2015)
print()

showData()

# Check model ---------------------------------------------------------------
print("Testownie modelu 2016-2017:\n")
testModel(2016, 2017)

# Predict 2018-2020 ---------------------------------------------------------
print("Predykcja 2018-2020:\n")
testModel(2018, 2020)

# Predict 2021-2022 ---------------------------------------------------------
print("Predykcja 2021-2022:\n");
testModel(2021, 2022)

print("Trening modelu 2008-2017")
trainModel(2008, 2017)
print()

# Predict 2018-2020 ---------------------------------------------------------
print("Predykcja 2018-2020:\n")
testModel(2018, 2020)

print("Trening modelu 2008-2020")
trainModel(2008, 2020)
print()

# Predict 2021-2022 ---------------------------------------------------------
print("Predykcja 2021-2022:\n");
testModel(2021, 2022)

showData()