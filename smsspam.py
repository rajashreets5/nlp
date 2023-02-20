# -*- coding: utf-8 -*-
"""
Created on Fri May  6 11:50:05 2022

@author: Rajashree
"""
#Data cleanining
import nltk
nltk.download('stopwords')
import re
import pandas as pd
messeges = pd.read_csv(r"C:\Users\Rajashree\Downloads\SMSSpamCollection.txt", 
                       sep='\t', names = ['label', 'message'])



from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0, len(messeges)):
    review = re.sub('[^a-zA-Z]', ' ', messeges['message'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
#creating the bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2500)
X = cv.fit_transform(corpus).toarray()

y=pd.get_dummies(messeges['label'])
y = y.iloc[:,1].values

#train test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.20, random_state=0)

#Naive_byes classifier
from sklearn.naive_bayes import MultinomialNB
spam_detect_model=MultinomialNB().fit(X_train, y_train)

y_pred = spam_detect_model.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_n = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
accuracy= accuracy_score(y_test, y_pred)


