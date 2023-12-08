import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from scipy.stats import f

incomeData = pd.read_csv('Data/Income.csv')
revenueData = pd.read_csv('Data/Revenue.csv')
numberOfEmployeesData = pd.read_csv('Data/NumberOfEmployees.csv')
numberOfUsersData = pd.read_csv('Data/NumberOfUsers.csv')
filtredNumberOfUsersData = numberOfUsersData[numberOfUsersData['Quarter'] == 4][['Year', 'Users_in_mln']].copy()

allData = incomeData.copy()
allData = pd.merge(allData, revenueData, how='outer', on='Year')
allData = pd.merge(allData, numberOfEmployeesData, how='outer', on='Year' )
allData = pd.merge(allData, filtredNumberOfUsersData[['Year', 'Users_in_mln']], how='outer', on='Year' )

alpha = 0.05

# Train model ---------------------------------------------------------------
def TrainModels(yearFrom, yearTo, variableFrom, variableTo):
    global fit
    global exponentialModel
    global powerModel
    global polynomialModel
    global squareModel
    global linearModel
    global independentVariableName
    global dependentVariableName

    independentVariableName = variableFrom
    dependentVariableName = variableTo

    print(f"Trenowanie modelu dla lat {yearFrom}-{yearTo} ze zmienną objaśniającą: '{independentVariableName}' na zminną objaśnianą: '{dependentVariableName}'...\n")
    trainData = allData[(allData['Year'] >= yearFrom) & (allData['Year'] <= yearTo)].copy()

    trainX = trainData[independentVariableName].values
    trainValues = trainData[dependentVariableName].values
    fit = np.polyfit(trainX, np.log(trainValues), 1)
    
    model = sm.OLS(trainValues, trainX).fit()
    degreeOfFreedom1 = model.df_model # liczba zmiennych objaśniających
    degreeOfFreedom2 = model.df_resid - 1 # liczba obserwacji - liczba zmiennych objaśniających - 1 bo w statsmodels.api nie liczy się stałej
    print(f'Stopnie swobody: {degreeOfFreedom1}, {degreeOfFreedom2}')
    critical_value = f.ppf(1 - alpha, degreeOfFreedom1, degreeOfFreedom2)
    print(f'Wartość krytyczna dla alpha {alpha}: {critical_value}')

    exponentialModel = LinearRegression()
    xForExponential = trainX.reshape(-1, 1)
    yForExponential = np.log(trainValues).reshape(-1, 1)
    exponentialModel.fit(xForExponential, yForExponential) # ln(y) = ln(a)x + ln(b) => y = a^x * b wykładniczy
    print("Model wykładniczy y = a^x * b zlinearyzowny do: ln(y) = ln(a)x + ln(b)")
    print("Współczynnik determinacji dla modelu wykładniczego R^2: ", exponentialModel.score(xForExponential, yForExponential))
    print("Współczynnik korelacji wielorakiej dla modelu wykładniczego R: ", np.sqrt(exponentialModel.score(xForExponential, yForExponential)))
    exponentialScore = exponentialModel.score(xForExponential, yForExponential)
    FValue = (exponentialScore / (1 - exponentialScore)) * ((len(trainX) - 2) / 1)
    print("Wartość F dla modelu wykładniczego: ", FValue)

    print()

    powerModel = LinearRegression()
    xForPower = np.log(trainX).reshape(-1, 1)
    yForPower = np.log(trainValues).reshape(-1, 1)
    powerModel.fit(xForPower, yForPower) # ln(y) = a*ln(x) + ln(b) => y = x^a * b potęgowy
    print("Model potęgowy y = x^a * b zlinearyzowny do: ln(y) = a*ln(x) + ln(b)")
    print("Współczynnik determinacji dla modelu potęgowego: ", powerModel.score(xForPower, yForPower))
    print("Współczynnik korelacji wielorakiej dla modelu potęgowego: ", np.sqrt(powerModel.score(xForPower, yForPower)))
    powerScore = powerModel.score(xForPower, yForPower)
    FValue = (powerScore / (1 - powerScore)) * ((len(trainX) - 2) / 1)
    print("Wartość F dla modelu potęgowego: ", FValue)
    print()

    polynomialModel = LinearRegression()
    x1 = trainX.reshape(-1, 1)
    x2 = np.power(trainX, 2).reshape(-1, 1)
    x3 = np.power(trainX, 3).reshape(-1, 1)
    x4 = np.power(trainX, 4).reshape(-1, 1)
    xForPolynomialModel = np.concatenate((x1, x2, x3, x4), axis=1)
    yForPolynomialModel = trainValues.reshape(-1, 1)
    polynomialModel.fit(xForPolynomialModel, yForPolynomialModel) # y = ax^4 + bx^3 + cx^2 + dx + e wielomianowy
    print("Model wielomianowy y = ax^4 + bx^3 + cx^2 + dx + e")
    print("Współczynnik determinacji dla modelu wielomianowego: ", polynomialModel.score(xForPolynomialModel, yForPolynomialModel))
    print("Współczynnik korleacji wielorakiej dla modelu wielomianowego: ", np.sqrt(polynomialModel.score(xForPolynomialModel, yForPolynomialModel)))
    print("Współczynniki dla modelu wielomianowego: ", polynomialModel.coef_)
    polynomialScore = polynomialModel.score(xForPolynomialModel, yForPolynomialModel)
    FValue = (polynomialScore / (1 - polynomialScore)) * ((len(trainX) - 2) / 1)
    print("Wartość F dla modelu wielomianowego: ", FValue)
    print()

    squareModel = LinearRegression()
    x1 = trainX.reshape(-1, 1)
    x2 = np.power(trainX, 2).reshape(-1, 1)
    xForSquareModel = np.concatenate((x1, x2), axis=1)
    yForSquareModel = trainValues.reshape(-1, 1)
    squareModel.fit(xForSquareModel, yForSquareModel) # y = ax^2 + bx + c kwadratowy
    print("Model kwadratowy y = ax^2 + bx + c")
    print("Współczynnik determinacji dla modelu kwadratowego: ", squareModel.score(xForSquareModel, yForSquareModel))
    print("Współczynnik korleacji wielorakiej dla modelu kwadratowego: ", np.sqrt(squareModel.score(xForSquareModel, yForSquareModel)))
    print("Współczynniki dla modelu kwadratowego: ", squareModel.coef_)
    squareScore = squareModel.score(xForSquareModel, yForSquareModel)
    FValue = (squareScore / (1 - squareScore)) * ((len(trainX) - 2) / 1)
    print("Wartość F dla modelu kwadratowego: ", FValue)
    print()

    linearModel = LinearRegression()
    xForLinear = trainX.reshape(-1, 1)
    yForLinear = trainValues.reshape(-1, 1)
    linearModel.fit(xForLinear, yForLinear) # y = ax + b liniowy
    print("Model liniowy y = ax + b")
    print("Współczynnik determinacji dla modelu liniowego: ", linearModel.score(xForLinear, yForLinear))
    print("Współczynnik korleacji wielorakiej dla modelu liniowego: ", np.sqrt(linearModel.score(xForLinear, yForLinear)))
    print("Współczynniki dla modelu liniowego: ", linearModel.coef_)
    linearScore = linearModel.score(xForLinear, yForLinear)
    FValue = (linearScore / (1 - linearScore)) * ((len(trainX) - 2) / 1)
    print("Wartość F dla modelu liniowego: ", FValue)
    print()

    trainData['Wykładniczy ręczny'] = np.exp(trainX * fit[0] + fit[1])
    trainData['Wykładniczy'] = np.exp(exponentialModel.predict(xForExponential))
    trainData['Potęgowy'] = np.exp(powerModel.predict(xForPower))
    trainData['Wielomianowy'] = polynomialModel.predict(xForPolynomialModel)
    trainData['Liniowy'] = linearModel.predict(xForLinear)

    print()
    print(trainData)
    print()

def TestModels(yearFrom, yearTo):
    print("Testowanie modelu (wyniki błędu dla różnych modeli) dla lat" , yearFrom , "-", yearTo, "...\n")
    testData = allData[(yearFrom <= allData['Year']) & (allData['Year'] <= yearTo)].copy()
    testX = testData[independentVariableName].values
    meanSquareErrorsForModels = []

    testData['Wykładniczy ręczny'] = np.exp(testX * fit[0] + fit[1])
    testData['Wykładniczy ręczny błąd'] = testData['Wykładniczy ręczny'] - testData[dependentVariableName]
    meanSquareError = np.mean(testData['Wykładniczy ręczny błąd'] ** 2)
    print("Średni błąd kwadratowy dla modelu wykładniczego ręcznego: ", meanSquareError)
    meanSquareErrorsForModels.append(("wykładniczy ręczny",meanSquareError))

    testData['Wykładniczy'] = np.exp(exponentialModel.predict(testX.reshape(-1, 1)))
    testData['Wykładniczy błąd'] = testData['Wykładniczy'] - testData[dependentVariableName]
    meanSquareError = np.mean(testData['Wykładniczy błąd'] ** 2)
    print("Średni błąd kwadratowy dla modelu wykładniczego: ", meanSquareError)
    meanSquareErrorsForModels.append(("wykładniczy",meanSquareError))

    testData['Potęgowy'] = np.exp(powerModel.predict(np.log(testX).reshape(-1, 1)))
    testData['Potęgowy błąd'] = testData['Potęgowy'] - testData[dependentVariableName]
    meanSquareError = np.mean(testData['Potęgowy błąd'] ** 2)
    print("Średni błąd kwadratowy dla modelu potęgowego: ", meanSquareError)
    meanSquareErrorsForModels.append(("potęgowy",meanSquareError))

    xt1 = testX.reshape(-1, 1)
    xt2 = np.power(testX, 2).reshape(-1, 1)
    xt3 = np.power(testX, 3).reshape(-1, 1)
    xt4 = np.power(testX, 4).reshape(-1, 1)
    X2 = np.concatenate((xt1, xt2, xt3, xt4), axis=1)
    testData['Wielomianowy'] = polynomialModel.predict(X2)
    testData['Wielomianowy błąd'] = testData['Wielomianowy'] - testData[dependentVariableName]
    meanSquareError = np.mean(testData['Wielomianowy błąd'] ** 2)
    print("Średni błąd kwadratowy dla modelu wielomianowego: ", meanSquareError)
    meanSquareErrorsForModels.append(("wielomianowy",meanSquareError))

    xt1 = testX.reshape(-1, 1)
    xt2 = np.power(testX, 2).reshape(-1, 1)
    X2 = np.concatenate((xt1, xt2), axis=1)
    testData['Kwadratowy'] = squareModel.predict(X2)
    testData['Kwadratowy błąd'] = testData['Kwadratowy'] - testData[dependentVariableName]
    meanSquareError = np.mean(testData['Kwadratowy błąd'] ** 2)
    print("Średni błąd kwadratowy dla modelu kwadratowego: ", meanSquareError)
    meanSquareErrorsForModels.append(("kwadratowy",meanSquareError))

    testData['Liniowy'] = linearModel.predict(testX.reshape(-1, 1))
    testData['Liniowy błąd'] = testData['Liniowy'] - testData[dependentVariableName]
    meanSquareError = np.mean(testData['Liniowy błąd'] ** 2)
    print("Średni błąd kwadratowy dla modelu liniowego: ", meanSquareError)
    meanSquareErrorsForModels.append(("liniowy",meanSquareError))

    print()
    bestModel = min(meanSquareErrorsForModels, key=lambda x: x[1])
    print(f"Najlepszy model: {bestModel[0]}, z błędem średnio kwadratowym: {bestModel[1]}")

    print()
    print(testData)
    print()
    
def TestFor(variableFrom, variableTo):
    TrainModels(2009, 2015, variableFrom, variableTo)
    TestModels(2016, 2018)

    TrainModels(2009, 2017, variableFrom, variableTo)
    TestModels(2018, 2020)

    TrainModels(2009, 2020, variableFrom, variableTo)
    TestModels(2021, 2022)

#TestFor('Employees', 'Income_in_mln')
#TestFor('Users_in_mln', 'Income_in_mln')
TestFor('Employees', 'Revenue_in_mln')





