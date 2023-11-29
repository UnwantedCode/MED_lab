import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

incomeData = pd.read_csv('Data/Income.csv')
revenueData = pd.read_csv('Data/Revenue.csv')
numberOfEmployeesData = pd.read_csv('Data/NumberOfEmployees.csv')
numberOfUsersData = pd.read_csv('Data/NumberOfUsers.csv')
filtredNumberOfUsersData = numberOfUsersData[numberOfUsersData['Quarter'] == 4][['Year', 'Users_in_mln']].copy()

allData = incomeData.copy()
allData = pd.merge(allData, revenueData, how='outer', on='Year')
allData = pd.merge(allData, numberOfEmployeesData, how='outer', on='Year' )
allData = pd.merge(allData, filtredNumberOfUsersData[['Year', 'Users_in_mln']], how='outer', on='Year' )

# Train model ---------------------------------------------------------------
def TrainModels(yearFrom, yearTo, variableFrom, variableTo):
    global fit
    global exponentialModel
    global powerModel
    global polynomialModel
    global linearModel
    global independentVariableName
    global dependentVariableName

    independentVariableName = variableFrom
    dependentVariableName = variableTo

    print("Trenowanie modelu dla lat" , yearFrom , "-", yearTo, "ze zmienną objaśniającą: ", independentVariableName, "na zminną objaśnianą: ", dependentVariableName, "...\n")
    trainData = allData[(allData['Year'] >= yearFrom) & (allData['Year'] <= yearTo)].copy()

    trainX = trainData[independentVariableName].values
    trainValues = trainData[dependentVariableName].values
    fit = np.polyfit(trainX, np.log(trainValues), 1)

    exponentialModel = LinearRegression() 
    xForExponential = trainX.reshape(-1, 1)
    yForExponential = np.log(trainValues).reshape(-1, 1)
    exponentialModel.fit(xForExponential, yForExponential) # ln(y) = ln(a)x + ln(b) => y = a^x * b wykładniczy
    print("Współczynnik determinacji dla modelu wykładniczego: ", exponentialModel.score(xForExponential, yForExponential))

    powerModel = LinearRegression()
    xForPower = np.log(trainX).reshape(-1, 1)
    yForPower = np.log(trainValues).reshape(-1, 1)
    powerModel.fit(xForPower, yForPower) # ln(y) = a*ln(x) + ln(b) => y = x^a * b potęgowy
    print("Współczynnik determinacji dla modelu potęgowego: ", powerModel.score(xForPower, yForPower))

    polynomialModel = LinearRegression()
    x1 = trainX.reshape(-1, 1)
    x2 = np.power(trainX, 2).reshape(-1, 1)
    x3 = np.power(trainX, 3).reshape(-1, 1)
    x4 = np.power(trainX, 4).reshape(-1, 1)
    xForPolynomialModel = np.concatenate((x1, x2, x3, x4), axis=1)
    yForPolynomialModel = trainValues.reshape(-1, 1)
    polynomialModel.fit(xForPolynomialModel, yForPolynomialModel) # y = ax^4 + bx^3 + cx^2 + dx + e wielomianowy
    print("Współczynnik determinacji dla modelu wielomianowego: ", polynomialModel.score(xForPolynomialModel, yForPolynomialModel))
    print("Współczynniki dla modelu wielomianowego: ", polynomialModel.coef_)

    linearModel = LinearRegression()
    xForLinear = trainX.reshape(-1, 1)
    yForLinear = trainValues.reshape(-1, 1)
    linearModel.fit(xForLinear, yForLinear) # y = ax + b liniowy
    print("Współczynnik determinacji dla modelu liniowego: ", linearModel.score(xForLinear, yForLinear))

    trainData['Wykładniczy ręczny'] = np.exp(trainX * fit[0] + fit[1])
    trainData['Wykładniczy'] = np.exp(exponentialModel.predict(xForExponential))
    trainData['Potęgowy'] = np.exp(powerModel.predict(xForPower))
    trainData['Wielomianowy'] = polynomialModel.predict(xForPolynomialModel)
    trainData['Liniowy'] = linearModel.predict(xForLinear)

    print()
    print(trainData)
    print()

def TestModels(yearFrom, yearTo):
    print("Testowanie modelu dla lat" , yearFrom , "-", yearTo, "...\n")
    testData = allData[(yearFrom <= allData['Year']) & (allData['Year'] <= yearTo)].copy()
    testX = testData[independentVariableName].values
    testValues = testData[dependentVariableName].values
    testData['Wykładniczy ręczny'] = np.exp(testX * fit[0] + fit[1])
    testData['Wykładniczy'] = np.exp(exponentialModel.predict(testX.reshape(-1, 1)))
    testData['Potęgowy'] = np.exp(powerModel.predict(np.log(testX).reshape(-1, 1)))
    xt1 = testX.reshape(-1, 1)
    xt2 = np.power(testX, 2).reshape(-1, 1)
    xt3 = np.power(testX, 3).reshape(-1, 1)
    xt4 = np.power(testX, 4).reshape(-1, 1)
    X2 = np.concatenate((xt1, xt2, xt3, xt4), axis=1)
    testData['Wielomianowy'] = polynomialModel.predict(X2)
    testData['Liniowy'] = linearModel.predict(testX.reshape(-1, 1))
    testData['Error'] = testData[dependentVariableName] - testData['Wielomianowy']
    testData['Error_squared'] = testData['Error'] ** 2
    print(testData)
    print()
    
def TestFor(variableFrom, variableTo):
    TrainModels(2009, 2015, variableFrom, variableTo)
    TestModels(2016, 2018)

    TrainModels(2009, 2017, variableFrom, variableTo)
    TestModels(2018, 2020)

    TrainModels(2009, 2020, variableFrom, variableTo)
    TestModels(2021, 2022)

TestFor('Employees', 'Revenue_in_mln')





