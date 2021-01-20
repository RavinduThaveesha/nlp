#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 22:26:03 2021

@author: casper
"""

"""
Data Reading
"""
import pandas as pd

#read news
#data source: https://www.kaggle.com/aaron7sun/stocknews
df = pd.read_csv('data/stock-news.csv', encoding='ISO-8859-1')


"""
Data Cleaning and Pre-processing
"""
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

#rename column names for easy of access
data = df.iloc[:,2:27]
columns = [i for i in range(1, 26)]
columns = [str(i) for i in columns]

data.columns = columns

#loop through all news and:
#remove stopwords
#lowercase 
#lemmatization

for index in list(df):
    df[index] = df[index].replace("[^a-zA-Z]", " ", regex=True)
    df[index] = df[index].str.lower()
#for index, row in data.iterrows():
    #row = row.str.replace("[^a-zA-Z]", " ", regex=True)
    #row = row.str.lower()
    #row = row.str.split()
    #row = [lemmatizer.lemmatize(word) for word in row if not word in stopwords.words('english')]

#for i in columns:
    #data = data[i].str.replace("[^a-zA-Z]", " ", regex=True)
    #data = data.str.lower()
    #data = data.str.split()
    #data = [lemmatizer.lemmatize(word) for word in data if not word in stopwords.words('english')]
    

"""
Train-Test Split
"""
#train = df[df['Date'] < '20150101']
#test = df[df['Date'] > '20141231']
