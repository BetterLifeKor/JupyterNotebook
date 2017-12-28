# Day_05_02_categorical.py
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
import pandas as pd
import numpy as np

pd.set_option('display.width', 1000)


def usage_dummies():
    print(pd.get_dummies(['a', 'b', 'c', 'a']))
    print(LabelBinarizer().fit_transform(['a', 'b', 'c', 'a']))

    df = pd.get_dummies(['a', 'b', 'c', 'a'])
    print(df.index)
    print(df.values)
    print(df.a)
    print('-' * 50)

    print(pd.get_dummies(['a', 'b', np.nan]))
    # print(LabelBinarizer().fit_transform(['a', 'b', np.nan])) # 에러

    print(pd.get_dummies(['a', 'b', np.nan], dummy_na=True))
    print('-' * 50)

    df = pd.DataFrame({'A': ['a', 'b', 'a'],
                       'B': ['b', 'a', 'c'],
                       'C': [1, 2, 3]})
    print(df)
    print(pd.get_dummies(df))
    print(pd.get_dummies(df, prefix=['c1', 'c2']))


def logistic_regression_1():
    url = 'Data/adult.txt'
    # url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
             'marital-status', 'occupation', 'relationship', 'race',
             'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
             'native-country', 'income']

    # age: continuous.
    # workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
    # fnlwgt: continuous.
    # education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
    # education-num: continuous.
    # marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
    # occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
    # relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
    # race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
    # sex: Female, Male.
    # capital-gain: continuous.
    # capital-loss: continuous.
    # hours-per-week: continuous.
    # native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

    adult = pd.read_csv(url, header=None, index_col=False, names=names)
    print(adult.shape)      # (32561, 15)

    # 숫자형 : 'age', 'hours-per-week'
    # 범주형 : 'workclass', 'education', 'occupation', 'sex', 'income'
    adult = adult[['age', 'workclass', 'education', 'occupation', 'sex',
                   'hours-per-week', 'income']]
    print(adult.shape)      # (32561, 7)
    print(adult.head(3))
    print()

    print('남여 갯수')
    print(adult.sex.value_counts())     # unique 갯수.

    # 문제
    # 남여 데이터갯수를 groupby 함수로 해결해 보세요.
    print(adult.groupby('sex').size())
    # print(adult.sex == ' Male')
    print('-' * 50)

    adult_dummies = pd.get_dummies(adult)
    print(adult_dummies.head(3))
    print()

    print(adult_dummies.columns)
    print(adult_dummies.columns.values)
    # ['age' 'hours-per-week' 'workclass_ ?' 'workclass_ Federal-gov'
    #  'workclass_ Local-gov' 'workclass_ Never-worked' 'workclass_ Private'
    #  'workclass_ Self-emp-inc' 'workclass_ Self-emp-not-inc' ...]
    print('-' * 50)

    # x = adult_dummies.loc[:, 'age': 'sex_ Male']
    x = adult_dummies.loc[:, :'sex_ Male']
    # y = adult_dummies.loc[:, 'income_ <=50K':]    # 2차원이라서 에러 (32561, 2)
    y = adult_dummies.loc[:, 'income_ <=50K']
    print(x.shape, y.shape)     # (32561, 44) (32561,)

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        random_state=0)

    print(type(x))          # <class 'pandas.core.frame.DataFrame'>
    print(type(x_train))    # <class 'pandas.core.frame.DataFrame'>

    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)

    print('train :', logreg.score(x_train, y_train))
    print(' test :', logreg.score(x_test, y_test))
    # train : 0.8138001638
    #  test : 0.808745854318


def logistic_regression_2():
    url = 'Data/adult.txt'
    names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
             'marital-status', 'occupation', 'relationship', 'race',
             'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
             'native-country', 'income']

    adult = pd.read_csv(url, header=None, index_col=False, names=names)
    # adult = adult[['age', 'workclass', 'education', 'occupation', 'sex',
    #                'hours-per-week', 'income']]

    age = LabelEncoder().fit_transform(adult.age)
    workclass = LabelEncoder().fit_transform(adult.workclass)
    education = LabelEncoder().fit_transform(adult.education)
    occupation = LabelEncoder().fit_transform(adult.occupation)
    sex = LabelEncoder().fit_transform(adult.sex)
    hours_per_week = LabelEncoder().fit_transform(adult['hours-per-week'])
    income = LabelEncoder().fit_transform(adult.income)

    print('workclass :', workclass)     # workclass : [7 6 4 ..., 4 4 5]

    # hstack을 사용하면 수평으로 연결. 결과는 (195366,).
    new_adult = np.vstack([age, workclass, education,
                           occupation, sex, hours_per_week])
    print('shape :', new_adult.shape)   # shape : (6, 32561)

    new_adult = new_adult.T
    print('shape :', new_adult.shape)   # shape : (32561, 6)

    x_train, x_test, y_train, y_test = train_test_split(new_adult, income,
                                                        random_state=0)

    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)

    print('train :', logreg.score(x_train, y_train))
    print(' test :', logreg.score(x_test, y_test))
    # train : 0.76122031122
    #  test : 0.757032305614


logistic_regression_2()
