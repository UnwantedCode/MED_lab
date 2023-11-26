import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


# Train model ---------------------------------------------------------------
print("Trenowanie modelu...\n")
trainData = pd.read_csv('Data/Income/train_data.csv')

trainX = trainData['Year'].values
trainValues = trainData['Income_in_mln'].values

fit = np.polyfit(trainX, np.log(trainValues), 1)

model = LinearRegression() 
model.fit(trainX.reshape(-1, 1), np.log(trainValues).reshape(-1, 1)) # ln(y) = ln(a)x + ln(b) => y = a^x * b

secondModel = LinearRegression()
secondModel.fit(np.log(trainX).reshape(-1, 1), np.log(trainValues).reshape(-1, 1)) # ln(y) = a*ln(x) + ln(b) => y = x^a * b

thirdModel = LinearRegression()
x1 = trainX.reshape(-1, 1)
x2 = np.power(trainX, 2).reshape(-1, 1)
x3 = np.power(trainX, 3).reshape(-1, 1)
x4 = np.power(trainX, 4).reshape(-1, 1)
X = np.concatenate((x1, x2, x3, x4), axis=1)
print(X)
thirdModel.fit(X, trainValues.reshape(-1, 1)) # y = ax^4 + bx^3 + cx^2 + dx + e

trainData['Trend'] = np.exp(trainX * fit[0] + fit[1])
trainData['Trend2'] = np.exp(model.predict(trainX.reshape(-1, 1)))
trainData['XForTrand3'] = np.log(trainX).reshape(-1, 1)
trainData['Trend3'] = np.exp(secondModel.predict(np.log(trainX).reshape(-1, 1)))

trainData['Trend4'] = thirdModel.predict(X)
print(secondModel.coef_, secondModel.intercept_)
print(trainData)


# Check model ---------------------------------------------------------------
print("Testownie modelu 2016-2017:\n")
testData = pd.read_csv('Data/Income/test_data_2016_2017.csv')

testX = testData['Year'].values

testData['Trend'] = np.exp(testX * fit[0] + fit[1])
testData['Trend2'] = np.exp(model.predict(testX.reshape(-1, 1)))
testData['Trend3'] = np.exp(secondModel.predict(np.log(testX).reshape(-1, 1)))
xt1 = testX.reshape(-1, 1)
xt2 = np.power(testX, 2).reshape(-1, 1)
xt3 = np.power(testX, 3).reshape(-1, 1)
xt4 = np.power(testX, 4).reshape(-1, 1)
X2 = np.concatenate((xt1, xt2, xt3, xt4), axis=1)
print(X2)
testData['Trend4'] = thirdModel.predict(X2)
testData['Error'] = testData['Income_in_mln'] - testData['Trend']
testData['Error_squared'] = testData['Error'] ** 2

print(testData)
print("Współczynnik determinacji: ", thirdModel.score(X2, testData['Income_in_mln']))
# print();

