import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from test import find_parameters

allData = pd.read_csv('./Lab-2/Zad-Platki sniadaniowe/Lab-2-Zad-Platki sniadaniowe-Data.csv')
print('Wszystkie dane: ')
with pd.option_context('display.max_rows', None):
    print(allData)
print()

allData['kalorie'] = allData['kalorie'].apply(lambda x: x / allData['kalorie'].max())
allData['cukry'] = allData['cukry'].apply(lambda x: x / allData['cukry'].max())
allData['weglowodany'] = allData['weglowodany'].apply(lambda x: x / allData['weglowodany'].max())
allData['proteiny'] = allData['proteiny'].apply(lambda x: x / allData['proteiny'].max())
allData['tluszcz'] = allData['tluszcz'].apply(lambda x: x / allData['tluszcz'].max())
allData['sod'] = allData['sod'].apply(lambda x: x / allData['sod'].max())
allData['blonnik'] = allData['blonnik'].apply(lambda x: x / allData['blonnik'].max())
allData['potas'] = allData['potas'].apply(lambda x: x / allData['potas'].max())

allData['y'] = allData['polka_2'] 

x_train, x_test, y_train, y_test = train_test_split(allData[['kalorie', 'cukry', 'weglowodany', 'proteiny', 'tluszcz', 'sod', 'blonnik', 'potas', 'y']], allData['y'], test_size=0.3)



def model_f(parameters):
    gx = parameters['a'] * parameters['kalorie'] \
            + parameters['b'] * parameters['cukry'] \
            + parameters['c'] * parameters['weglowodany'] \
            + parameters['d'] * parameters['proteiny'] \
            + parameters['e'] * parameters['tluszcz'] \
            + parameters['f'] * parameters['sod'] \
            + parameters['g'] * parameters['blonnik'] \
            + parameters['h'] * parameters['potas'] \
            + parameters['i']
    p = 1 / (1 + np.exp(-gx))
    return p 

start_parameters = { 'a': 0.0, 'b': 0.0, 'c': 0.0, 'd': 0.0, 'e': 0.0, 'f': 0.0, 'g': 0.0, 'h': 0.0, 'i': 0.0 }
found_parameters = find_parameters(start_parameters, model_f, x_train.to_dict(orient='records'), iteration_limit=100)
print('Znalezione parametry: ', found_parameters)
print()

allData['gx'] = found_parameters['a'] * allData['kalorie'] \
            + found_parameters['b'] * allData['cukry'] \
            + found_parameters['c'] * allData['weglowodany'] \
            + found_parameters['d'] * allData['proteiny'] \
            + found_parameters['e'] * allData['tluszcz'] \
            + found_parameters['f'] * allData['sod'] \
            + found_parameters['g'] * allData['blonnik'] \
            + found_parameters['h'] * allData['potas'] \
            + found_parameters['i']

allData['p'] = allData['gx'].apply(lambda x: 1 / (1 + np.exp(-x)))

allData['y_logisitic'] = allData['p']

allData['prediction'] = allData['y_logisitic'].apply(lambda x: 1 if x > 0.5 else 0)
allData['correct'] = allData['prediction'] == allData['y']

number_of_correct = allData['correct'].sum()
number_of_all = allData['correct'].count()
print('Dokładność: ', number_of_correct / number_of_all)

testdf = pd.DataFrame(x_test, columns=['kalorie', 'cukry', 'weglowodany', 'proteiny', 'tluszcz', 'sod', 'blonnik', 'potas', 'y'])
testdf['gx'] = found_parameters['a'] * testdf['kalorie'] \
            + found_parameters['b'] * testdf['cukry'] \
            + found_parameters['c'] * testdf['weglowodany'] \
            + found_parameters['d'] * testdf['proteiny'] \
            + found_parameters['e'] * testdf['tluszcz'] \
            + found_parameters['f'] * testdf['sod'] \
            + found_parameters['g'] * testdf['blonnik'] \
            + found_parameters['h'] * testdf['potas'] \
            + found_parameters['i']

testdf['p'] = testdf['gx'].apply(lambda x: 1 / (1 + np.exp(-x)))
testdf['y_logisitic'] = testdf['p']
testdf['prediction'] = testdf['y_logisitic'].apply(lambda x: 1 if x > 0.5 else 0)
testdf['correct'] = testdf['prediction'] == testdf['y']

number_of_correct = testdf['correct'].sum()
number_of_all = testdf['correct'].count()
print('Dokładność dla testowych danych: ', number_of_correct / number_of_all)

traindf = pd.DataFrame(x_train, columns=['kalorie', 'cukry', 'weglowodany', 'proteiny', 'tluszcz', 'sod', 'blonnik', 'potas', 'y'])
traindf['gx'] = found_parameters['a'] * traindf['kalorie'] \
            + found_parameters['b'] * traindf['cukry'] \
            + found_parameters['c'] * traindf['weglowodany'] \
            + found_parameters['d'] * traindf['proteiny'] \
            + found_parameters['e'] * traindf['tluszcz'] \
            + found_parameters['f'] * traindf['sod'] \
            + found_parameters['g'] * traindf['blonnik'] \
            + found_parameters['h'] * traindf['potas'] \
            + found_parameters['i']

traindf['p'] = traindf['gx'].apply(lambda x: 1 / (1 + np.exp(-x)))
traindf['y_logisitic'] = traindf['p']
traindf['prediction'] = traindf['y_logisitic'].apply(lambda x: 1 if x > 0.5 else 0)
traindf['correct'] = traindf['prediction'] == traindf['y']

number_of_correct = traindf['correct'].sum()
number_of_all = traindf['correct'].count()
print('Dokładność dla treningowych danych: ', number_of_correct / number_of_all)

linearModel = LinearRegression()
linearModel.fit(traindf[['kalorie', 'cukry', 'weglowodany', 'proteiny', 'tluszcz', 'sod', 'blonnik', 'potas']], traindf['y'])

testdf['y_linear'] = linearModel.predict(testdf[['kalorie', 'cukry', 'weglowodany', 'proteiny', 'tluszcz', 'sod', 'blonnik', 'potas']])
testdf['linear_prediction'] = testdf['y_linear'].apply(lambda x: 1 if x > 0.5 else 0)
testdf['linear_correct'] = testdf['linear_prediction'] == testdf['y']

number_of_correct = testdf['linear_correct'].sum()
number_of_all = testdf['linear_correct'].count()
print('Dokładność dla testowych danych dla regresji liniowej: ', number_of_correct / number_of_all)






with pd.option_context('display.max_rows', None):
    print(allData)
print()


            

