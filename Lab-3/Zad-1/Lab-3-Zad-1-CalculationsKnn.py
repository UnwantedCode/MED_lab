# using kNN
import pandas as pd

data = pd.read_csv('./Lab3/Matches/Data.csv')

data['SilaWiatruWartosc'] = data['SilaWiatru'].map({'silny': 1, 'słaby': 0.5, 'brak': 0})
data['ZachmurzenieWartosc'] = data['Zachmurzenie'].map({'słonecznie': 0, 'pochmurnie': 1})
data['OdczuwalnaTemperaturaWartosc'] = data['OdczuwalnaTemperatura'].map({'ciepło': 0.5, 'zimno': 0, 'gorąco': 1})

print('Dane:')
print(data)
print()

szukanaSilaWiatru = 1
szukaneZachmurzenie = 0
szukanaOdczuwalnaTemperatura = 0.5

print('Szukane dane:')
print('Sila wiatru:', szukanaSilaWiatru)
print('Zachmurzenie:', szukaneZachmurzenie)
print('Odczuwalna temperatura:', szukanaOdczuwalnaTemperatura)

k = 3

print('K:', k)
print()

def calculate_for_metric(metric, metricName):

    print('Metryka:', metricName)
    print()

    data['Odleglosc'] = metric(data)

    print('Dane z odlegloscia:')
    print(data)
    print()

    top_k_rows = data.nsmallest(k, 'Odleglosc')
    biggest_value_for_k_rows = top_k_rows['Odleglosc'].max()
    print('Najwieksza odleglosc:', biggest_value_for_k_rows)

    rows_with_value_at_most_k = data[data['Odleglosc'] <= biggest_value_for_k_rows]
    print('Najblizsze:', k, 'sasiadow:')
    print(rows_with_value_at_most_k)
    print()

    playedMatchCount = rows_with_value_at_most_k['ZagranoMecz'].value_counts()
    print('Rozegrane mecze w', k, 'najbliższych:')
    print(playedMatchCount)

    playedMatchResult = playedMatchCount.idxmax()
    print('Wynik:', playedMatchResult)
    print()

calculate_for_metric(lambda x: ((x['SilaWiatruWartosc'] - szukanaSilaWiatru) **2 + (x['ZachmurzenieWartosc'] - szukaneZachmurzenie) ** 2 + (x['OdczuwalnaTemperaturaWartosc'] - szukanaOdczuwalnaTemperatura) **2) ** 0.5, 'metryka euklidesowa')
calculate_for_metric(lambda x: abs(x['SilaWiatruWartosc'] - szukanaSilaWiatru) + abs(x['ZachmurzenieWartosc'] - szukaneZachmurzenie) + abs(x['OdczuwalnaTemperaturaWartosc'] - szukanaOdczuwalnaTemperatura), 'metryka taksowkowa')