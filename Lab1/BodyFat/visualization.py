import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

allData = pd.read_csv('Data/BodyFat.csv')
print(allData['Pct.BF'])

# clear data


allData = allData[(allData['Pct.BF'] > 4) & (allData['Pct.BF'] < 35) & (allData['Abdomen'] * 0.0254 < 3)]

# Normalize to metric system

allData['Height'] = allData['Height'] * 0.0254
allData['Weight'] = allData['Weight'] * 0.45359237
allData['Neck'] = allData['Neck'] * 0.0254
allData['Chest'] = allData['Chest'] * 0.0254
allData['Abdomen'] = allData['Abdomen'] * 0.0254
allData['Waist'] = allData['Waist'] * 0.0254
allData['Hip'] = allData['Hip'] * 0.0254
allData['Thigh'] = allData['Thigh'] * 0.0254
allData['Knee'] = allData['Knee'] * 0.0254
allData['Ankle'] = allData['Ankle'] * 0.0254
allData['Bicep'] = allData['Bicep'] * 0.0254
allData['Forearm'] = allData['Forearm'] * 0.0254
allData['Wrist'] = allData['Wrist'] * 0.0254

# Add BMI

allData['BMI'] = allData['Weight'] / allData['Height'] ** 2




def showFor(columnName):
    plt.plot(allData[columnName], allData['Pct.BF'], 'o')

    plt.xlabel(columnName)
    plt.ylabel('Tkanka tłuszczowa [%]')
    plt.title('Wykres danych z pliku CSV')
    plt.grid(True)
    plt.legend(['Dane'])

    plt.show()

def normalizeColumn(columnName, newColumnName):
    allData[newColumnName] = allData[columnName] / allData['Height'] ** 2

def showNormalizedColumn(columnName):
    newColumnName = columnName + 'Normalized'
    normalizeColumn(columnName, newColumnName)
    showFor(newColumnName)

#normalizeColumn('Weight', 'WeightNormalized')
#normalizeColumn('Neck', 'NeckNormalized')
#normalizeColumn('Chest', 'ChestNormalized')
#normalizeColumn('Abdomen', 'AbdomenNormalized')
# normalizeColumn('Waist', 'WaistNormalized')
# normalizeColumn('Hip', 'HipNormalized')
# normalizeColumn('Thigh', 'ThighNormalized')
# normalizeColumn('Knee', 'KneeNormalized')
# normalizeColumn('Ankle', 'AnkleNormalized')
# normalizeColumn('Bicep', 'BicepNormalized')
#normalizeColumn('Forearm', 'ForearmNormalized')
#normalizeColumn('Wrist', 'WristNormalized')


def showCorrelationMatrix():
    correlationMatrix = allData.corr()

    correlationsSorted = correlationMatrix['Pct.BF'].abs().sort_values(ascending=False)
    print(correlationsSorted)

    sns.heatmap(correlationMatrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.show()

    sns.pairplot(allData, x_vars=['Pct.BF', 'BMI', 'Abdomen', 'Weight', 'Chest', 'Hip', 'Thigh', 'Waist'], y_vars='Pct.BF', kind='scatter')
    plt.show()

def showValuableColumns():
    valuableColumns = ['Pct.BF', 'BMI', 'Abdomen', 'Weight', 'Chest', 'Hip', 'Thigh', 'Waist']
    valuableData = allData[valuableColumns]

    sns.pairplot(valuableData)
    plt.show()

showCorrelationMatrix()

# showFor('Weight')
# showFor('Chest')
# showFor('Abdomen')
# showFor('Waist')
# showFor('Hip')
# showFor('Thigh')
# showFor('BMI')

