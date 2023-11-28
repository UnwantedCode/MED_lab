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

# Check model ---------------------------------------------------------------
print("Testownie modelu 2016-2017:\n")
testData = allData[(2016 <= allData['Year']) & (allData['Year'] < 2018)].copy()

testX = pd.to_numeric(pd.to_datetime(testData['Date'])).values.reshape(-1, 1)

testData['GeneratedByModel'] = model.predict(testX)
testData['Error'] = testData['Users_in_mln'] - testData['GeneratedByModel']
testData['Error_squared'] = testData['Error'] ** 2

print(testData)
print()

# Predict 2018-2020 ---------------------------------------------------------
print("Predykcja 2018-2020:\n")
predictData = allData[(2018 <= allData['Year']) & (allData['Year'] < 2021)].copy()

predictX = pd.to_numeric(pd.to_datetime(predictData['Date'])).values.reshape(-1, 1)

predictData['GeneratedByModel'] = model.predict(predictX)
predictData['Error'] = predictData['Users_in_mln'] - predictData['GeneratedByModel']
predictData['Error_squared'] = predictData['Error'] ** 2

print(predictData)
print()

# Predict 2021-2022 ---------------------------------------------------------
print("Predykcja 2021-2022:\n");
predictData = allData[(2021 <= allData['Year']) & (allData['Year'] < 2023)].copy()

predictX = pd.to_numeric(pd.to_datetime(predictData['Date'])).values.reshape(-1, 1)

predictData['GeneratedByModel'] = model.predict(predictX)
predictData['Error'] = predictData['Users_in_mln'] - predictData['GeneratedByModel']
predictData['Error_squared'] = predictData['Error'] ** 2

print(predictData)
print()
