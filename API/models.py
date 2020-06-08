import pandas as pd
import numpy as np

data = pd.read_csv('news_processed.csv')




#splitting data into training and testing sets of x and y
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
tf = TfidfVectorizer(ngram_range = (1,3), min_df = 5)
xset = data.xdata
yset = data.REAL
xtrain, xtest, ytrain, ytest = train_test_split(xset, yset, test_size=0.33, random_state = 123)




#vectorizing text
xtraintf = tf.fit_transform(xtrain.values.astype(str))
xtesttf = tf.transform(xtest.values.astype(str))




#initializing and training model
from sklearn.linear_model import PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(random_state = 79).fit(xtraintf,ytrain)




#initializing and training model2
from sklearn.linear_model import PassiveAggressiveClassifier
pac2 = PassiveAggressiveClassifier(random_state = 123).fit(xtraintf,ytrain)




#Serializing and dumping model
import joblib
joblib.dump(tf, 'tfidfvectorizer.pkl')
joblib.dump(pac, 'model1.pkl')
joblib.dump(pac2, 'model2.pkl')
cols = list(data.columns)
cols.remove('REAL')
cols.remove('Unnamed: 0')
joblib.dump(cols, 'cols.pkl')
print ('Dumps Successful')