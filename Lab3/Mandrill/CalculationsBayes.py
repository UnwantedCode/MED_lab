from sklearn.model_selection import train_test_split
import pandas as pd

def preparePost(post):
    normalizedPost = ' ' + post
    return normalizedPost.upper().replace('#', ' ').replace('@', ' ').replace(',', ' ').replace('"', ' ').replace('?', ' ').replace('!', ' ').replace('.', ' ').replace(':', ' ').replace(';', ' ').replace(' - ', ' ').replace('&', ' ').replace('(', ' ').replace(')', ' ').replace('‘', ' ').replace('’', ' ').replace("//", ' ').replace("'RE ", ' ').replace(" TO ", ' ').replace(" A ", ' ').replace(" WITH ", ' ').replace(" THE ", ' ').replace(" OF ", ' ').replace(" IN ", ' ').replace(" AND ", ' ').replace(" FOR ", ' ').replace(" ON ", ' ').replace(" IS ", ' ').replace(" ARE ", ' ').replace(" I ", ' ').replace(" OR ", ' ').replace(" IT ", ' ').replace(" MY ", ' ').replace(" YOU ", ' ').replace(" ME ", ' ').replace(" HE ", ' ').replace(" SHE ", ' ').replace(" WE ", ' ').replace(" THEY ", ' ').replace(" OUR ", ' ').replace(" YOURS ", ' ').replace(" HIS ", ' ').replace(" HERS ", ' ').replace(" THEIR ", ' ').replace(" BY ", ' ').replace(" FROM ", ' ').replace(" AS ", ' ').replace(" AT ", ' ').replace(" THAT ", ' ').replace(" HAVE ", ' ').replace(" HAS ", ' ').replace(" IT ", ' ').replace(" EMAIL ", ' ').replace(" BUT ", ' ').replace(" NOT ", ' ').replace(" THIS ", ' ').replace(" SO ", ' ').replace(" IF ", ' ').replace(" AN ", ' ').replace(" ALL ", ' ').replace(" ABOUT ", ' ').replace(" WILL ", ' ').replace(" BE ", ' ').replace(" CAN ", ' ').replace(" MORE ", ' ').replace(" THAN ", ' ').replace(" WHEN ", ' ').replace(" WHAT ", ' ').replace(" HOW ", ' ').replace(" WHERE ", ' ').replace(" WHY ", ' ').replace(" HI ", ' ').replace(" NO ", ' ').replace("HTTP", '').replace("YOU'" , '').replace("’S", '').replace("DON'T", '').replace(" HERE ", ' ').replace(' I ', ' ').replace("I'D", '').replace("IT'S", '').replace("I'M", '').replace(" US ", ' ').replace(' JUST ', ' ').replace(' LIKE ', ' ').replace("...", ' ').replace("…", ' ').replace(" THERE ", ' ').replace(' YOUR ', ' ').replace(' u ', ' ').replace(' might ', ' ').replace(" DID ", ' ').replace(" DIDN'T ", ' ').replace(' LOOK ', ' ').replace(' LOOKS ', ' ').replace(' LOVE ', ' ').replace(' MANDRILL ' , ' ').replace('ING ', ' ').replace("N'T ", ' ').replace(' DO ', ' ').replace(' DOES ', ' ').replace(' BEEN ', ' ').replace(' GET ', ' ').replace(' GETS ', ' ').replace(' BY ', ' ').replace(' GOOD ', ' ').replace(' COME ', ' ').replace(' U ', ' ').replace(' SOME ', ' ')

mandrillPosts = pd.read_excel('./Lab3/Mandrill/Data.xlsx', sheet_name='dot. aplikacji Mandrill')
mandrillPosts['IsMandrill'] = 'yes'
print('Mandrill:')
print(mandrillPosts)
print()

notMandrillPosts = pd.read_excel('./Lab3/Mandrill/Data.xlsx', sheet_name='dot. innych')
notMandrillPosts['IsMandrill'] = 'no'
print('Nie Mandrill:')
print(notMandrillPosts)
print()

allPosts = pd.concat([mandrillPosts, notMandrillPosts])
print('Wszystkie posty:')
print(allPosts)
print()

x_train, x_test, y_train, y_test = train_test_split(allPosts['Post'], allPosts['IsMandrill'], test_size=0.2)

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

isMandrillValues = y_train.value_counts()
print('Wszystkie posty wg przypisania do mandrill:')
print(isMandrillValues)
print()

probabilityOfMandrill = isMandrillValues['yes'] / y_train.count()
probabilityOfNotMandrill = isMandrillValues['no'] / y_train.count()
print('Prawdopodobienstwo mandrilla:', probabilityOfMandrill)
print('Prawdopodobienstwo nie mandrilla:', probabilityOfNotMandrill)
print()

mandrillPosts = x_train[y_train == 'yes']
print('Mandrille:')
print(mandrillPosts)
print()

notMandrillPosts = x_train[y_train == 'no']
print('Nie mandrille:')
print(notMandrillPosts)
print()

mandrillWords = mandrillPosts.apply(preparePost).str.split(expand=True).stack().value_counts()
print('Mandrille wg slow:')
print(mandrillWords)
print()

notMandrillWords = notMandrillPosts.apply(preparePost).str.split(expand=True).stack().value_counts()
print('Nie mandrille wg slow:')
print(notMandrillWords)
print()

mandrillWordsCount = mandrillWords.count()
notMandrillWordsCount = notMandrillWords.count()
print('Liczba slow mandrilla:', mandrillWordsCount)
print('Liczba slow nie mandrilla:', notMandrillWordsCount)
print()

mandrillWordsProbability = (mandrillWords + 1) / (mandrillWordsCount + 2)
print('Prawdopodobienstwo slow mandrilla:')
print(mandrillWordsProbability)
print()

mandrillWordsProbabilityForZero = 1 / (mandrillWordsCount + 2)

notMandrillWordsProbability = (notMandrillWords + 1) / (notMandrillWordsCount + 2)
print('Prawdopodobienstwo slow nie mandrilla:')
print(notMandrillWordsProbability)
print()

notMandrillWordsProbabilityForZero = 1 / (notMandrillWordsCount + 2)

tests_mandrill = x_test[y_test == 'yes']
print('Testy mandrilla:')
print(tests_mandrill)
print()

tests_not_mandrill = x_test[y_test == 'no']
print('Testy nie mandrilla:')
print(tests_not_mandrill)
print()

def calculateProbabilityOfMandrillPost(post):
    words = preparePost(post).split()
    probability = 1
    for word in words:
        if word in mandrillWordsProbability:
            probability *= mandrillWordsProbability[word]
        else:
            probability *= mandrillWordsProbabilityForZero
    return probability * probabilityOfMandrill

def calculateProbabilityOfNotMandrillPost(post):
    words = preparePost(post).split()
    probability = 1
    for word in words:
        if word in notMandrillWordsProbability:
            probability *= notMandrillWordsProbability[word]
        else:
            probability *= notMandrillWordsProbabilityForZero
    return probability * probabilityOfNotMandrill

tests_mandrill_probability_dataFrame = pd.DataFrame(tests_mandrill, columns=['Post'])
tests_mandrill_probability_dataFrame['Probability'] = tests_mandrill_probability_dataFrame['Post'].apply(calculateProbabilityOfMandrillPost)
tests_mandrill_probability_dataFrame['ProbabilityNoMandrill'] = tests_mandrill_probability_dataFrame['Post'].apply(calculateProbabilityOfNotMandrillPost)
tests_mandrill_probability_dataFrame['IsMandrillPrediction'] = tests_mandrill_probability_dataFrame['Probability'] > tests_mandrill_probability_dataFrame['ProbabilityNoMandrill']
print('Testy mandrilla wg prawdopodobienstwa:')
print(tests_mandrill_probability_dataFrame)
print()

probability_find_mandrill = 100 * len(tests_mandrill_probability_dataFrame[tests_mandrill_probability_dataFrame['IsMandrillPrediction'] == True]) / len(tests_mandrill_probability_dataFrame)
print('Prawdopodobieństwo znalezienia dobrego:', probability_find_mandrill)
print()

tests_not_mandrill_probability_dataFrame = pd.DataFrame(tests_not_mandrill, columns=['Post'])
tests_not_mandrill_probability_dataFrame['Probability'] = tests_not_mandrill_probability_dataFrame['Post'].apply(calculateProbabilityOfMandrillPost)
tests_not_mandrill_probability_dataFrame['ProbabilityNoMandrill'] = tests_not_mandrill_probability_dataFrame['Post'].apply(calculateProbabilityOfNotMandrillPost)
tests_not_mandrill_probability_dataFrame['IsMandrillPrediction'] = tests_not_mandrill_probability_dataFrame['Probability'] > tests_not_mandrill_probability_dataFrame['ProbabilityNoMandrill']
print('Testy nie mandrilla wg prawdopodobienstwa:')
print(tests_not_mandrill_probability_dataFrame)
print()

probability_find_not_mandrill = 100 * len(tests_not_mandrill_probability_dataFrame[tests_not_mandrill_probability_dataFrame['IsMandrillPrediction'] == False]) / len(tests_not_mandrill_probability_dataFrame)
print('Prawdopodobieństwo znalezienia złego:', probability_find_not_mandrill)
print()

wordsInMandrillCount = mandrillPosts.apply(preparePost).str.split(expand=True).stack().value_counts()
wordsInMandrillCountDf = pd.DataFrame(wordsInMandrillCount, columns=['count'])
wordsInMandrillCountDf.to_csv('./Lab3/Mandrill/wordsInMandrillCount.csv')
print('Liczba slow w mandrillach:')
print(wordsInMandrillCountDf)
print()

wordsInNotMandrillCount = notMandrillPosts.apply(preparePost).str.split(expand=True).stack().value_counts()
wordsInNotMandrillCountDf = pd.DataFrame(wordsInNotMandrillCount, columns=['count'])
wordsInNotMandrillCountDf.to_csv('./Lab3/Mandrill/wordsInNotMandrillCount.csv')
print('Liczba slow w nie mandrillach:')
print(wordsInNotMandrillCountDf)
print()




