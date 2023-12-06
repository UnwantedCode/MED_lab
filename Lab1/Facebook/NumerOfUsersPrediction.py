import pandas as pd
from sklearn.linear_model import LinearRegression

allData = pd.read_csv('Data/NumberOfUsers.csv')

# Train model ---------------------------------------------------------------
print("Trenowanie modelu...\n")
trainData = allData[(allData['Year'] < 2016) & (allData['Year'] >= 2008)].copy()

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

# Check model ---------------------------------------------------------------
print("Testownie modelu 2016-2017:\n")
testModel(2016, 2017)

# Predict 2018-2020 ---------------------------------------------------------
print("Predykcja 2018-2020:\n")
testModel(2018, 2020)

# Predict 2021-2022 ---------------------------------------------------------
print("Predykcja 2021-2022:\n");
testModel(2021, 2022)
