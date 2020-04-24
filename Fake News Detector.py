#From nltk, download 'averaged_perceptron_tagger', 'wordnet', 'stopwords'
import pandas as pd
import numpy as np
import sklearn as skl
data = pd.read_csv('G:/Users/Krunal/Desktop/news.csv')




#Original Data
data.head()




#Checking for null values
data.isnull().sum()




#No null values present.
#Value counts
data.label.value_counts()




#Categories are balanced.
#Defining preprocessing function
from nltk.corpus import wordnet

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def preprocessing(col):
    #Lower case
    lower = col.apply(str.lower)

    #Lemmatizing
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    lemmatized = lower.apply(lambda x: ' '.join(lemmatizer.lemmatize(word,get_wordnet_pos(word)) for word in str(x).split()))
    
    #removing stopwords and extra spaces
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    rem_stopwords = lemmatized.apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))
    
    #removing numbers
    rem_num = rem_stopwords.apply(lambda x: " ".join(x for x in x.split() if not x.isdigit()))
    
    #removing punctuations
    import string
    import re
    rem_punc = rem_num.str.replace('[^\w\s]','')
    
    #removing words of length 1
    rem_one = rem_punc.apply(lambda x: ' '.join([word for word in x.split() if len(word)>1]))

    
    return rem_one




#Processing data for vectorization
#news_processed was processed using the commands given below
#data = pd.concat([pd.get_dummies(data.label).drop('FAKE',axis=1),data.drop(['label','Unnamed: 0'],axis=1)],axis=1)
#data['xdata'] = data.title + " " + data.text
#data = data.drop(['title', 'text'], axis = 1)
#data['xdata'] = preprocessing(data['xdata'])
data = pd.read_csv('G:/Users/Krunal/Desktop/news_processed.csv')




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




#Predicting and calculating accuracy
ypred = pac.predict(xtesttf)
print(str(np.mean(ypred == ytest)*100) + "% accuracy")




#Generating confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(ytest, ypred)

