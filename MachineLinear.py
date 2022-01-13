import sys
# !/usr/bin/python
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
from scipy.stats import expon
import seaborn as sns
import xlrd as xd
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

scale = StandardScaler()
import tensorflow as tf
import warnings

warnings.simplefilter("ignore")
import mysql.connector
# import pymongo
# import gym
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
# import kafka
from numpy import array
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import Row

class MachineLearLinear:
    def LinearReg1(self):
        desiredWidth=920
        pd.set_option("display.Width",desiredWidth)
        pd.set_option("display.max_columns",None)
        pd.set_option("display.max_rows",None)
        pd.set_option("display.precision",2)
        pd.options.display.max_columns=None

        np.set_printoptions(precision=2)
        dataset = pd.read_csv('C:\\Users\wrong\Desktop\SparkPythonDoBigDataAnalytics_Resources\churn_modelling.csv',
                              index_col=0)

        print(dataset.info())
        print("data describe")
        print(dataset.describe())
        print("data describe include")
        print(dataset.describe(include='all'))
        print("data size")
        print(dataset.size)
        print(dataset.columns)
        print(dataset.shape)

        print("duplicate")
        print(dataset.duplicated().value_counts())
        print("duplicate value counts")
        print(dataset.duplicated().value_counts())

        print("dataset nuniq")
        print(dataset.nunique())
        print("dataset size")
        print(dataset.size)

        print("number of dimensions")
        print(dataset.ndim)
        print("number of dtypes")
        print(dataset.dtypes)
        print("number of value counte")
        print(dataset.dtypes.value_counts())
        print("number of len")
        print(len(dataset))
        print(dataset.sample(4))

        print(dataset.head(3))

        print(dataset.tail(3))

        #check correlation between Creditscore and Exited
        print("dataset.isnull().values.sum()-----------------")
        print(dataset.isnull().values.sum())
        print("dataset.isnull().values.any()--------------")
        print(dataset.isnull().values.any())
        print("dataset.isnull().sum()--------------")
        print(dataset.isnull().sum())
        print("dataset.isnull().any()--------------")
        print(dataset.isnull().any())

        #rename data set
        dataset=dataset.rename(columns={"CreditScore":"CreditSc"})
        dataset=dataset.drop(columns=['CustomerId', 'Surname', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'])
        #dataset=dataset[['Exited','CreditSc']]
        print(dataset.head(3))

        X=dataset.iloc[:,:-1].values
        y=dataset.iloc[:,-1].values

        #y=np.arange(10000).reshape((10000,1))
        #print(y)
        #print(np.transpose(y, (10000, 1).shape))
        #print(np.transpose(y).shape)
        #print(np.arange(len(y)).reshape(10000,1))
        #print(y)
        #print(np.arange(len(y)).reshape(len(y)))

        #y=np.array(y)
        #t = np.transpose(y).reshape(len(y), 1)
        #print(t)
        #print(pd.DataFrame(t))

        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(missing_values=np.NaN, strategy='mean')
        X=imputer.fit_transform(X)

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)

        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train=sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        from sklearn.linear_model import LinearRegression, LogisticRegression
        regression = LinearRegression()
        regression.fit(X_train, y_train)

        y_pred = regression.predict(X_test)
        #slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
        #plt.scatter(X_train, y_train, color='red')
        #plt.plot(X_train, regression.predict(X_train), color='blue')
        #plt.show()
        #X=pd.DataFrame(X)
        #print(X.isnull().values.any())
        #print(X.isnull().values.sum())
        #Xx = np.array(X)
        #yy = np.array(y)
        #print(np.corrcoef(X, y))
        #plt.scatter(X,y)
        #plt.plot(X,y)
        #plt.show()
        #plt.xlabel('X')
        #plt.ylabel('y')
        #plt.title('Title')

        #Xd=pd.DataFrame(X)
        #yD=pd.DataFrame(y)

        #print(len(Xd))
        #print(len(yD))

        x=dataset.dropna()
        df = dataset.reset_index(drop=True)
        x=dataset["CreditSc"]
        y=dataset["Exited"]
        slope, intercept, r, p, std_err = stats.linregress(x, y)
        print(r)
        print("test")


MachineLearLinear.LinearReg1(MachineLearLinear)
