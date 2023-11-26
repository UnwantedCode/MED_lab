import pandas as pd

excel_file_path = 'przychody_zysk_i_zatrudnienie_przedsiębiorstwa_facebook.xlsx'

# Wczytaj arkusz do obiektu DataFrame
initialData = pd.read_excel(excel_file_path, sheet_name='przychody_zysk_i_zatrudnienie_p')


initialData = initialData.iloc[:, :2]

initialData.columns = [col if col != 'Rok' else 'Year' for col in initialData.columns]
initialData.columns = [col if col != 'Przychód w mln $' else 'Income_in_mln' for col in initialData.columns]

train_data = initialData[(initialData['Year'] < 2016) & (initialData['Year'] >= 2008)]
test_data_2016_2017 = initialData[(2016 <= initialData['Year']) & (initialData['Year'] < 2018)]

# Eksportuj do pliku CSV
initialData.to_csv('Data/Income/data.csv', index=False)
train_data.to_csv('Data/Income/train_data.csv', index=False)
test_data_2016_2017.to_csv('Data/Income/test_data_2016_2017.csv', index=False)


# Wyświetl wynik lub wykonaj inne operacje na wybranych kolumnach
print(initialData)
