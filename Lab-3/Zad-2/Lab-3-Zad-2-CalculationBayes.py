import pandas as pd
from sklearn.model_selection import train_test_split


# pieniadz tak
# darmowy nie
# bogaty tak
# nieprzyzwoicie nie
# tajny tak


print()

allData = pd.read_csv('./Lab-3/Zad-2/Lab-3-Zad-2-Data.csv')
print('Wszystkie dane:')
print(allData)
print()

# using naive bayes theorem

spamCounts = allData['spam'].value_counts()
# print(spamCounts)
# print()

probabilityOfSpam = spamCounts['tak'] / allData['spam'].count()
probabilityOfNotSpam = spamCounts['nie'] / allData['spam'].count()
print('Prawdopodobienstwo spamu:', probabilityOfSpam)
print('Prawdopodobienstwo nie spamu:', probabilityOfNotSpam)
print()

spam = allData[allData['spam'] == 'tak']
# print('Spam:')
# print(spam)
# print()

spamForPieniadzCount = spam['pieniadz'].value_counts()
# print('Spam wg pieniadza:')
# print(spamForPieniadzCount)
# print()

probabilityOfPieniadzWhileSpam = spamForPieniadzCount['tak'] / spam['pieniadz'].count()
print('Prawdopodobienstwo pieniadza podczas spamu:', probabilityOfPieniadzWhileSpam)
print()

spamForDarmowyCount = spam['darmowy'].value_counts()
# print('Spam wg darmowego:')
# print(spamForDarmowyCount)
# print()

probabilityOfMissingDarmowyWhileSpam = spamForDarmowyCount['nie'] / spam['darmowy'].count()
print('Prawdopodobienstwo braku darmowego podczas spamu:', probabilityOfMissingDarmowyWhileSpam)
print()

spamForBogatyCount = spam['bogaty'].value_counts()
# print('Spam wg bogatego:')
# print(spamForBogatyCount)
# print()

probabilityOfBogatyWhileSpam = spamForBogatyCount['tak'] / spam['bogaty'].count()
print('Prawdopodobienstwo bogatego podczas spamu:', probabilityOfBogatyWhileSpam)
print()

spamForNieprzyzwoicieCount = spam['nieprzyzwoicie'].value_counts()
# print('Spam wg nieprzyzwoicie:')
# print(spamForNieprzyzwoicieCount)
# print()

probabilityOfMissingNieprzyzwoicieWhileSpam = spamForNieprzyzwoicieCount['nie'] / spam['nieprzyzwoicie'].count()
print('Prawdopodobienstwo braku nieprzyzwoicie podczas spamu:', probabilityOfMissingNieprzyzwoicieWhileSpam)
print()

spamForTajnyCount = spam['tajny'].value_counts()
# print('Spam wg tajnego:')
# print(spamForTajnyCount)
# print()

probabilityOfTajnyWhileSpam = spamForTajnyCount['tak'] / spam['tajny'].count()
print('Prawdopodobienstwo tajnego podczas spamu:', probabilityOfTajnyWhileSpam)
print()

notSpam = allData[allData['spam'] == 'nie']
# print('Nie spam:')
# print(notSpam)
# print()

notSpamForPieniadzCount = notSpam['pieniadz'].value_counts()
# print('Nie spam wg pieniadza:')
# print(notSpamForPieniadzCount)
# print()

probabilityOfPieniadzWhileNotSpam = notSpamForPieniadzCount['tak'] / notSpam['pieniadz'].count()
print('Prawdopodobienstwo pieniadza podczas nie spamu:', probabilityOfPieniadzWhileNotSpam)
print()

notSpamForDarmowyCount = notSpam['darmowy'].value_counts()
# print('Nie spam wg darmowego:')
# print(notSpamForDarmowyCount)
# print()

probabilityOfMissingDarmowyWhileNotSpam = notSpamForDarmowyCount['nie'] / notSpam['darmowy'].count()
print('Prawdopodobienstwo braku darmowego podczas nie spamu:', probabilityOfMissingDarmowyWhileNotSpam)
print()

notSpamForBogatyCount = notSpam['bogaty'].value_counts()
# print('Nie spam wg bogatego:')
# print(notSpamForBogatyCount)
# print()

probabilityOfBogatyWhileNotSpam = notSpamForBogatyCount['tak'] / notSpam['bogaty'].count()
print('Prawdopodobienstwo bogatego podczas nie spamu:', probabilityOfBogatyWhileNotSpam)
print()

notSpamForNieprzyzwoicieCount = notSpam['nieprzyzwoicie'].value_counts()
# print('Nie spam wg nieprzyzwoicie:')
# print(notSpamForNieprzyzwoicieCount)
# print()

probabilityOfMissingNieprzyzwoicieWhileNotSpam = notSpamForNieprzyzwoicieCount['nie'] / notSpam['nieprzyzwoicie'].count()
print('Prawdopodobienstwo braku nieprzyzwoicie podczas nie spamu:', probabilityOfMissingNieprzyzwoicieWhileNotSpam)
print()

notSpamForTajnyCount = notSpam['tajny'].value_counts()
# print('Nie spam wg tajnego:')
# print(notSpamForTajnyCount)
# print()

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


x_train, x_test, y_train, y_test = train_test_split(allData[['Nr','pieniadz', 'darmowy', 'bogaty', 'nieprzyzwoicie', 'tajny']], allData['spam'], test_size=0.2, random_state=1)

print('x_train:')
print(x_train)
print()

print('x_test:')
print(x_test)
print()

print('y_train:')
print(y_train)
print()

print('y_test:')
print(y_test)
print()

spamCounts = y_train.value_counts()
print('Wszystkie dane wg spamu:')
print(spamCounts)
print()

probabilityOfSpam = spamCounts['tak'] / y_train.count()
probabilityOfNotSpam = spamCounts['nie'] / y_train.count()
print('Prawdopodobienstwo spamu:', probabilityOfSpam)
print('Prawdopodobienstwo nie spamu:', probabilityOfNotSpam)
print()

spam = x_train[y_train == 'tak']
print('Spam:')
print(spam)
print()

notSpam = x_train[y_train == 'nie']
print('Nie spam:')
print(notSpam)
print()

n = 2
spamPieniadzValueCounts = spam['pieniadz'].value_counts()
notSpamPieniadzValueCounts = notSpam['pieniadz'].value_counts()
probabilityOfPieniadzInSpam = (spamPieniadzValueCounts['tak'] if 'tak' in spamPieniadzValueCounts else 0  + 1)/ (spam['pieniadz'].count() + n)
probabilityOfLackOfPieniadzInSpam = (spamPieniadzValueCounts['nie'] if 'nie' in spamPieniadzValueCounts else 0 + 1) / (spam['pieniadz'].count() + n)
probabilityOfPieniadzInNotSpam = (notSpamPieniadzValueCounts['tak'] if 'tak' in notSpamPieniadzValueCounts else 0 + 1) / (notSpam['pieniadz'].count() + n)
probabilityOfLackOfPieniadzInNotSpam = (notSpamPieniadzValueCounts['nie'] if 'nie' in notSpamPieniadzValueCounts else 0 + 1) / (notSpam['pieniadz'].count() + n)

spamDarmowyValueCounts = spam['darmowy'].value_counts()
notSpamDarmowyValueCounts = notSpam['darmowy'].value_counts()
probabilityOfDarmowyInSpam = (spamDarmowyValueCounts['tak'] if 'tak' in spamDarmowyValueCounts else 0 + 1) / (spam['darmowy'].count() + n)
probabilityOfLackOfDarmowyInSpam = (spamDarmowyValueCounts['nie'] if 'nie' in spamDarmowyValueCounts else 0 + 1) / (spam['darmowy'].count() + n)
probabilityOfDarmowyInNotSpam = (notSpamDarmowyValueCounts['tak'] if 'tak' in notSpamDarmowyValueCounts else 0 + 1) / (notSpam['darmowy'].count() + n)
probabilityOfLackOfDarmowyInNotSpam = (notSpamDarmowyValueCounts['nie'] if 'nie' in notSpamDarmowyValueCounts else 0 + 1) / (notSpam['darmowy'].count() + n)

spamBogatyValueCounts = spam['bogaty'].value_counts()
notSpamBogatyValueCounts = notSpam['bogaty'].value_counts()
probabilityOfBogatyInSpam = (spamBogatyValueCounts['tak'] if 'tak' in spamBogatyValueCounts else 0 + 1) / (spam['bogaty'].count() + n)
probabilityOfLackOfBogatyInSpam = (spamBogatyValueCounts['nie'] if 'nie' in spamBogatyValueCounts else 0 + 1) / (spam['bogaty'].count() + n)
probabilityOfBogatyInNotSpam = (notSpamBogatyValueCounts['tak'] if 'tak' in notSpamBogatyValueCounts else 0 + 1) / (notSpam['bogaty'].count() + n)
probabilityOfLackOfBogatyInNotSpam = (notSpamBogatyValueCounts['nie'] if 'nie' in notSpamBogatyValueCounts else 0 + 1) / (notSpam['bogaty'].count() + n)


spamNieprzyzwoicieValueCounts = spam['nieprzyzwoicie'].value_counts()
notSpamNieprzyzwoicieValueCounts = notSpam['nieprzyzwoicie'].value_counts()
probabilityOfNieprzyzwoicieInSpam = (spamNieprzyzwoicieValueCounts['tak'] if 'tak' in spamNieprzyzwoicieValueCounts else 0 + 1) / (spam['nieprzyzwoicie'].count() + n)
probabilityOfLackOfNieprzyzwoicieInSpam = (spamNieprzyzwoicieValueCounts['nie'] if 'nie' in spamNieprzyzwoicieValueCounts else 0 + 1) / (spam['nieprzyzwoicie'].count() + n)
probabilityOfNieprzyzwoicieInNotSpam = (notSpamNieprzyzwoicieValueCounts['tak'] if 'tak' in notSpamNieprzyzwoicieValueCounts else 0 + 1) / (notSpam['nieprzyzwoicie'].count() + n)
probabilityOfLackOfNieprzyzwoicieInNotSpam = (notSpamNieprzyzwoicieValueCounts['nie'] if 'nie' in notSpamNieprzyzwoicieValueCounts else 0 + 1) / (notSpam['nieprzyzwoicie'].count() + n)

spamTajnyValueCounts = spam['tajny'].value_counts()
notSpamTajnyValueCounts = notSpam['tajny'].value_counts()
probabilityOfTajnyInSpam = (spamTajnyValueCounts['tak'] if 'tak' in spamTajnyValueCounts else 0 + 1) / (spam['tajny'].count() + n)
probabilityOfLackOfTajnyInSpam = (spamTajnyValueCounts['nie'] if 'nie' in spamTajnyValueCounts else 0 + 1) / (spam['tajny'].count() + n)
probabilityOfTajnyInNotSpam = (notSpamTajnyValueCounts['tak'] if 'tak' in notSpamTajnyValueCounts else 0 + 1) / (notSpam['tajny'].count() + n)
probabilityOfLackOfTajnyInNotSpam = (notSpamTajnyValueCounts['nie'] if 'nie' in notSpamTajnyValueCounts else 0 + 1) / (notSpam['tajny'].count() + n)


test = pd.DataFrame(x_test)
test['spam'] = y_test
test['p_pieniadz'] = test['pieniadz'].apply(lambda x: probabilityOfPieniadzInSpam if x == 'tak' else probabilityOfLackOfPieniadzInSpam)
test['p_darmowy'] = test['darmowy'].apply(lambda x: probabilityOfDarmowyInSpam if x == 'tak' else probabilityOfLackOfDarmowyInSpam)
test['p_bogaty'] = test['bogaty'].apply(lambda x: probabilityOfBogatyInSpam if x == 'tak' else probabilityOfLackOfBogatyInSpam)
test['p_nieprzyzwoicie'] = test['nieprzyzwoicie'].apply(lambda x: probabilityOfNieprzyzwoicieInSpam if x == 'tak' else probabilityOfLackOfNieprzyzwoicieInSpam)
test['p_tajny'] = test['tajny'].apply(lambda x: probabilityOfTajnyInSpam if x == 'tak' else probabilityOfLackOfTajnyInSpam)
test['p_spam'] = test['p_pieniadz'] * test['p_darmowy'] * test['p_bogaty'] * test['p_nieprzyzwoicie'] * test['p_tajny'] * probabilityOfSpam

test['pn_pieniandz'] = test['pieniadz'].apply(lambda x: probabilityOfPieniadzInNotSpam if x == 'tak' else probabilityOfLackOfPieniadzInNotSpam)
test['pn_darmowy'] = test['darmowy'].apply(lambda x: probabilityOfDarmowyInNotSpam if x == 'tak' else probabilityOfLackOfDarmowyInNotSpam)
test['pn_bogaty'] = test['bogaty'].apply(lambda x: probabilityOfBogatyInNotSpam if x == 'tak' else probabilityOfLackOfBogatyInNotSpam)
test['pn_nieprzyzwoicie'] = test['nieprzyzwoicie'].apply(lambda x: probabilityOfNieprzyzwoicieInNotSpam if x == 'tak' else probabilityOfLackOfNieprzyzwoicieInNotSpam)
test['pn_tajny'] = test['tajny'].apply(lambda x: probabilityOfTajnyInNotSpam if x == 'tak' else probabilityOfLackOfTajnyInNotSpam)
test['pn_spam'] = test['pn_pieniandz'] * test['pn_darmowy'] * test['pn_bogaty'] * test['pn_nieprzyzwoicie'] * test['pn_tajny'] * probabilityOfNotSpam

test['wynik'] = test['p_spam'] > test['pn_spam']
test['spodziewany_wynik'] = test['spam'] == 'tak'


print(test[['Nr','pieniadz', 'darmowy', 'bogaty', 'nieprzyzwoicie', 'tajny', 'spam','wynik', 'spodziewany_wynik']])
