#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 18:40:51 2021

@author: casper
"""

"""
Data Reading
"""
import pandas as pd

#read messages
#data source: https://archive.ics.uci.edu/ml/datasets/sms+spam+collection#:~:text=UCI%20Machine%20Learning%20Repository%3A%20SMS%20Spam%20Collection%20Data%20Set&text=Abstract%3A%20The%20SMS%20Spam%20Collection,for%20mobile%20phone%20spam%20research.&text=Data%20Set%20Information%3A&text=The%20messages%20largely%20originate%20from,from%20students%20attending%20the%20University.
messages = pd.read_csv('data/sms-spam.csv', sep='\t', names=['label', 'message'])

"""
Data Cleaning and Pre-processing
"""
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
corpus = []

#loop through all messages and:
#remove stopwords
#lowercase 
#lemmatization
for i in range(len(messages)):
    message = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    message = message.lower()
    message = message.split()
    message = [lemmatizer.lemmatize(word) for word in message if not word in stopwords.words('english')]
    message = ' '.join(message)
    
    corpus.append(message)


"""
Creating Bag-of-Words model
"""
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000) #most frequent 5000 words
X = cv.fit_transform(corpus).toarray()

#convert dependant vraibale into dummy
Y = pd.get_dummies(messages['label'])
Y = Y.iloc[:,1].values


"""
Train-Test Split
"""
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


"""
Training the Model using Naive Bayes Classifier
"""
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB().fit(X_train, Y_train)

Y_predict = model.predict(X_test)


"""
Confusion Matrix 
"""
from sklearn.metrics import confusion_matrix
confusion_m = confusion_matrix(Y_test, Y_predict)

#calculate accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test, Y_predict)

