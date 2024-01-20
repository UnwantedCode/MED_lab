import pandas as pd

print()

allData = pd.read_csv('./Lab-3/Zad-1/Lab-3-Zad-1-Data.csv')
# print('Wszystkie dane:')
# print(allData)
# print()

# using naive bayes theorem

matchPlayedCounts = allData['ZagranoMecz'].value_counts()
# print(matchPlayedCounts)
# print()

probabilityOfPlayMatch = matchPlayedCounts['tak'] / allData['ZagranoMecz'].count()
probabilityOfNotPlayMatch = matchPlayedCounts['nie'] / allData['ZagranoMecz'].count()
print('Prawdopodobienstwo zagrania meczu:', probabilityOfPlayMatch)
print('Prawdopodobienstwo nie zagrania meczu:', probabilityOfNotPlayMatch)
print()

playedMatch = allData[allData['ZagranoMecz'] == 'tak']
# print('Rozegrane mecze:')
# print(playedMatch)
# print()

playedMatchForWindCount = playedMatch['SilaWiatru'].value_counts()
# print('Rozegrane mecze wg sily wiatru:')
# print(playedMatchForWindCount)
# print()

probabilityOfStrongWindWhileMatch = playedMatchForWindCount['silny'] / playedMatch['SilaWiatru'].count()
print('Prawdopodobienstwo silnego wiatru podczas meczu:', probabilityOfStrongWindWhileMatch)
print()

playedMatchForCloudinessCount = playedMatch['Zachmurzenie'].value_counts()
# print('Rozegrane mecze wg zachmurzenia:')
# print(playedMatchForCloudinessCount)
# print()

probabilityOfSunnyWhileMatch = playedMatchForCloudinessCount['słonecznie'] / playedMatch['Zachmurzenie'].count()
print('Prawdopodobienstwo slonecznej pogody podczas meczu:', probabilityOfSunnyWhileMatch)
print()

playedMatchForTemperatureCount = playedMatch['OdczuwalnaTemperatura'].value_counts()
# print('Rozegrane mecze wg odczuwalnej temperatury:')
# print(playedMatchForTemperatureCount)
# print()

probabilityOfWarmWhileMatch = playedMatchForTemperatureCount['ciepło'] / playedMatch['OdczuwalnaTemperatura'].count()
print('Prawdopodobienstwo cieplej pogody podczas meczu:', probabilityOfWarmWhileMatch)
print()

notPlayedMatch = allData[allData['ZagranoMecz'] == 'nie']
# print('Nie rozegrane mecze:')
# print(notPlayedMatch)
# print()

notPlayedMatchForWindCount = notPlayedMatch['SilaWiatru'].value_counts()
# print('Nie rozegrane mecze wg sily wiatru:')
# print(notPlayedMatchForWindCount)
# print()

probabilityOfStrongWindWhileNotMatch = notPlayedMatchForWindCount['silny'] / notPlayedMatch['SilaWiatru'].count()
print('Prawdopodobienstwo silnego wiatru podczas nie rozegranego meczu:', probabilityOfStrongWindWhileNotMatch)
print()

notPlayedMatchForCloudinessCount = notPlayedMatch['Zachmurzenie'].value_counts()
# print('Nie rozegrane mecze wg zachmurzenia:')
# print(notPlayedMatchForCloudinessCount)
# print()

probabilityOfSunnyWhileNotMatch = notPlayedMatchForCloudinessCount['słonecznie'] / notPlayedMatch['Zachmurzenie'].count()
print('Prawdopodobienstwo slonecznej pogody podczas nie rozegranego meczu:', probabilityOfSunnyWhileNotMatch)
print()

notPlayedMatchForTemperatureCount = notPlayedMatch['OdczuwalnaTemperatura'].value_counts()
# print('Nie rozegrane mecze wg odczuwalnej temperatury:')
# print(notPlayedMatchForTemperatureCount)
# print()

probabilityOfWarmWhileNotMatch = notPlayedMatchForTemperatureCount['ciepło'] / notPlayedMatch['OdczuwalnaTemperatura'].count()
print('Prawdopodobienstwo cieplej pogody podczas nie rozegranego meczu:', probabilityOfWarmWhileNotMatch)
print()

productForPlayMatch = probabilityOfStrongWindWhileMatch * probabilityOfSunnyWhileMatch * probabilityOfWarmWhileMatch * probabilityOfPlayMatch
print('Iloczyn prawdopodobienstw dla zagrania meczu:', productForPlayMatch)
print()

productForNotPlayMatch = probabilityOfStrongWindWhileNotMatch * probabilityOfSunnyWhileNotMatch * probabilityOfWarmWhileNotMatch * probabilityOfNotPlayMatch
print('Iloczyn prawdopodobienstw dla nie zagrania meczu:', productForNotPlayMatch)
print()


print('Wynik:', 'tak' if productForPlayMatch > productForNotPlayMatch else 'nie')

print()

