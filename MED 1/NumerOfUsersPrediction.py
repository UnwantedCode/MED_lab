import pandas as pd
from sklearn.linear_model import LinearRegression

# Train model ---------------------------------------------------------------
print("Trenowanie modelu...\n")
trainData = pd.read_csv('Data/NumberOfUsers/train_data.csv')

trainX = pd.to_numeric(pd.to_datetime(trainData['Date'])).values.reshape(-1, 1)
trainValues = trainData['Users_in_mln'].values

model = LinearRegression()
model.fit(trainX, trainValues)

# Check model ---------------------------------------------------------------
print("Testownie modelu 2016-2017:\n");
testData = pd.read_csv('Data/NumberOfUsers/test_data_2016_2017.csv')

testX = pd.to_numeric(pd.to_datetime(testData['Date'])).values.reshape(-1, 1)

testData['Trend'] = model.predict(testX)
testData['Error'] = testData['Users_in_mln'] - testData['Trend']
testData['Error_squared'] = testData['Error'] ** 2

print(testData)
print("Współczynnik determinacji: ", model.score(testX, testData['Users_in_mln']))
print();

# Predict 2018-2020 ---------------------------------------------------------
print("Predykcja 2018-2020:\n");
predictData = pd.read_csv('Data/NumberOfUsers/predict_data_2018_2019_2020.csv')

predictX = pd.to_numeric(pd.to_datetime(predictData['Date'])).values.reshape(-1, 1)

predictData['Trend'] = model.predict(predictX)
predictData['Error'] = predictData['Users_in_mln'] - predictData['Trend']
predictData['Error_squared'] = predictData['Error'] ** 2

print(predictData)
print("Współczynnik determinacji: ", model.score(predictX, predictData['Users_in_mln']))
print();

# Predict 2021-2022 ---------------------------------------------------------
print("Predykcja 2021-2022:\n");
predictData = pd.read_csv('Data/NumberOfUsers/predict_data_2021_2022.csv')

predictX = pd.to_numeric(pd.to_datetime(predictData['Date'])).values.reshape(-1, 1)

predictData['Trend'] = model.predict(predictX)
predictData['Error'] = predictData['Users_in_mln'] - predictData['Trend']
predictData['Error_squared'] = predictData['Error'] ** 2

print(predictData)
print("Współczynnik determinacji: ", model.score(predictX, predictData['Users_in_mln']))
print();
