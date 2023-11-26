import pandas as pd

excel_file_path = 'liczba_użytkowników_portalu_społecznościowego_facebook.xlsx'

# Wczytaj arkusz do obiektu DataFrame
initialData = pd.read_excel(excel_file_path, sheet_name='Statista')


initialData = initialData.iloc[:, :2]
initialData['Year'] = initialData['Kwartał'].apply(lambda x: int("20" + x.split("'")[1]))
initialData['Quarter'] = initialData['Kwartał'].apply(lambda x: int(x.split(" ")[0][1]))
initialData = initialData.drop('Kwartał', axis=1)
initialData['Date'] = pd.to_datetime(initialData['Year'].astype(str) + 'Q' + initialData['Quarter'].astype(str))
initialData.columns = [col if col != 'Liczba użytkowników w mln' else 'Users_in_mln' for col in initialData.columns]

initialData = initialData[["Date", "Year", "Quarter", "Users_in_mln"]]

train_data = initialData[initialData['Date'] < '2016-01-01']
test_data_2016_2017 = initialData[('2016-01-01' <= initialData['Date']) & (initialData['Date'] < '2018-01-01')]
predict_data_2018_2019_2020 = initialData[('2018-01-01' <= initialData['Date']) & (initialData['Date'] < '2021-01-01')]
predict_data_2021_2022 = initialData[('2021-01-01' <= initialData['Date']) & (initialData['Date'] < '2023-01-01')]

# Eksportuj do pliku CSV
initialData.to_csv('Data/NumberOfUsers/data.csv', index=False)
train_data.to_csv('Data/NumberOfUsers/train_data.csv', index=False)
test_data_2016_2017.to_csv('Data/NumberOfUsers/test_data_2016_2017.csv', index=False)
predict_data_2018_2019_2020.to_csv('Data/NumberOfUsers/predict_data_2018_2019_2020.csv', index=False)
predict_data_2021_2022.to_csv('Data/NumberOfUsers/predict_data_2021_2022.csv', index=False)


# Wyświetl wynik lub wykonaj inne operacje na wybranych kolumnach
print(initialData)
