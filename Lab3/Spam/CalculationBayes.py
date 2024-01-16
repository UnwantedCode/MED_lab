import pandas as pd


# pieniadz tak
# darmowy nie
# bogaty tak
# nieprzyzwoicie nie
# tajny tak


print()

allData = pd.read_csv('./Lab3/Spam/Data.csv')
print('Wszystkie dane:')
print(allData)
print()

# using naive bayes theorem

spamCounts = allData['spam'].value_counts()
print(spamCounts)
print()

probabilityOfSpam = spamCounts['tak'] / allData['spam'].count()
probabilityOfNotSpam = spamCounts['nie'] / allData['spam'].count()
print('Prawdopodobienstwo spamu:', probabilityOfSpam)
print('Prawdopodobienstwo nie spamu:', probabilityOfNotSpam)
print()

spam = allData[allData['spam'] == 'tak']
print('Spam:')
print(spam)
print()

spamForPieniadzCount = spam['pieniadz'].value_counts()
print('Spam wg pieniadza:')
print(spamForPieniadzCount)
print()

probabilityOfPieniadzWhileSpam = spamForPieniadzCount['tak'] / spam['pieniadz'].count()
print('Prawdopodobienstwo pieniadza podczas spamu:', probabilityOfPieniadzWhileSpam)
print()

spamForDarmowyCount = spam['darmowy'].value_counts()
print('Spam wg darmowego:')
print(spamForDarmowyCount)
print()

probabilityOfMissingDarmowyWhileSpam = spamForDarmowyCount['nie'] / spam['darmowy'].count()
print('Prawdopodobienstwo braku darmowego podczas spamu:', probabilityOfMissingDarmowyWhileSpam)
print()

spamForBogatyCount = spam['bogaty'].value_counts()
print('Spam wg bogatego:')
print(spamForBogatyCount)
print()

probabilityOfBogatyWhileSpam = spamForBogatyCount['tak'] / spam['bogaty'].count()
print('Prawdopodobienstwo bogatego podczas spamu:', probabilityOfBogatyWhileSpam)
print()

spamForNieprzyzwoicieCount = spam['nieprzyzwoicie'].value_counts()
print('Spam wg nieprzyzwoicie:')
print(spamForNieprzyzwoicieCount)
print()

probabilityOfMissingNieprzyzwoicieWhileSpam = spamForNieprzyzwoicieCount['nie'] / spam['nieprzyzwoicie'].count()
print('Prawdopodobienstwo braku nieprzyzwoicie podczas spamu:', probabilityOfMissingNieprzyzwoicieWhileSpam)
print()

spamForTajnyCount = spam['tajny'].value_counts()
print('Spam wg tajnego:')
print(spamForTajnyCount)
print()

probabilityOfTajnyWhileSpam = spamForTajnyCount['tak'] / spam['tajny'].count()
print('Prawdopodobienstwo tajnego podczas spamu:', probabilityOfTajnyWhileSpam)
print()

notSpam = allData[allData['spam'] == 'nie']
print('Nie spam:')
print(notSpam)
print()

notSpamForPieniadzCount = notSpam['pieniadz'].value_counts()
print('Nie spam wg pieniadza:')
print(notSpamForPieniadzCount)
print()

probabilityOfPieniadzWhileNotSpam = notSpamForPieniadzCount['tak'] / notSpam['pieniadz'].count()
print('Prawdopodobienstwo pieniadza podczas nie spamu:', probabilityOfPieniadzWhileNotSpam)
print()

notSpamForDarmowyCount = notSpam['darmowy'].value_counts()
print('Nie spam wg darmowego:')
print(notSpamForDarmowyCount)
print()

probabilityOfMissingDarmowyWhileNotSpam = notSpamForDarmowyCount['nie'] / notSpam['darmowy'].count()
print('Prawdopodobienstwo braku darmowego podczas nie spamu:', probabilityOfMissingDarmowyWhileNotSpam)
print()

notSpamForBogatyCount = notSpam['bogaty'].value_counts()
print('Nie spam wg bogatego:')
print(notSpamForBogatyCount)
print()

probabilityOfBogatyWhileNotSpam = notSpamForBogatyCount['tak'] / notSpam['bogaty'].count()
print('Prawdopodobienstwo bogatego podczas nie spamu:', probabilityOfBogatyWhileNotSpam)
print()

notSpamForNieprzyzwoicieCount = notSpam['nieprzyzwoicie'].value_counts()
print('Nie spam wg nieprzyzwoicie:')
print(notSpamForNieprzyzwoicieCount)
print()

probabilityOfMissingNieprzyzwoicieWhileNotSpam = notSpamForNieprzyzwoicieCount['nie'] / notSpam['nieprzyzwoicie'].count()
print('Prawdopodobienstwo braku nieprzyzwoicie podczas nie spamu:', probabilityOfMissingNieprzyzwoicieWhileNotSpam)
print()

notSpamForTajnyCount = notSpam['tajny'].value_counts()
print('Nie spam wg tajnego:')
print(notSpamForTajnyCount)
print()

probabilityOfTajnyWhileNotSpam = notSpamForTajnyCount['tak'] / notSpam['tajny'].count()
print('Prawdopodobienstwo tajnego podczas nie spamu:', probabilityOfTajnyWhileNotSpam)
print()

productForSpam = probabilityOfPieniadzWhileSpam * probabilityOfBogatyWhileSpam * probabilityOfTajnyWhileSpam * probabilityOfMissingDarmowyWhileSpam * probabilityOfMissingNieprzyzwoicieWhileSpam * probabilityOfSpam
print('Iloczyn prawdopodobienstw spamu:', productForSpam)
print()

productForNotSpam = probabilityOfPieniadzWhileNotSpam * probabilityOfBogatyWhileNotSpam * probabilityOfTajnyWhileNotSpam * probabilityOfMissingDarmowyWhileNotSpam * probabilityOfMissingNieprzyzwoicieWhileNotSpam * probabilityOfNotSpam
print('Iloczyn prawdopodobienstw nie spamu:', productForNotSpam)
print()

print('Wynik:', 'tak' if productForSpam > productForNotSpam else 'nie')
print()
