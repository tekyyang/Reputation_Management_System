# cross validation example: https://www.programcreek.com/python/example/75177/sklearn.cross_validation.cross_val_score


import sklearn
import ast


#---load the data

#load posi tweets
with open('/Users/yibingyang/Documents/final_thesis_project/Data/Twitter/after_preprocessing/positive_clean_tweets.txt', 'r') as f:
    posi_lines = f.readlines()  # read all the lines in the file
    posi_lines = [item.replace('\n', '') for item in posi_lines]

#load nega labels
with open('/Users/yibingyang/Documents/final_thesis_project/Data/Twitter/after_preprocessing/negative_clean_tweets.txt', 'r') as f:
    nega_lines = f.readlines()  # read all the lines in the file
    nega_lines = [item.replace('\n', '') for item in nega_lines]

posi_len = len(posi_lines) # 57796
nega_len = len(nega_lines) # 35490

# split to train/test dataset

positive_list = [ i for i in posi_lines[:nega_len] ] # take the same length as nega
negative_list = [ i for i in nega_lines ]

# print len(positive_list) # 35490
# print len(negative_list) # 35490

posi_labels = [1] * len(positive_list)
nega_labels = [-1] * len(negative_list)

tweets_tokens_list = positive_list + negative_list
tweets_list = []
for sentence in tweets_tokens_list:
    sentence = ast.literal_eval(sentence)
    new_sentence = ''
    for token in list(sentence):
        new_sentence = new_sentence + token + ' '
    tweets_list.append(new_sentence)

labels = posi_labels + nega_labels

X_train = tweets_list
Y_train = labels


# data resample
#---pending---#


# vector space building
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
tfidf_vectorizer = TfidfVectorizer()
X_tfidf_matrix = tfidf_vectorizer.fit_transform(X_train)
# print tfidf_vectorizer.get_feature_names() # vectors' name
print X_tfidf_matrix.shape


# feature selection
from sklearn.feature_selection import SelectKBest, chi2
ch2 = SelectKBest(chi2, k=5000)
chi_filter_training = ch2.fit_transform(X_tfidf_matrix.toarray(), np.asarray(Y_train))
# print chi_filter_training.shape # (70980, 5000)
# tfidf_df = pd.DataFrame(X_tfidf_matrix.toarray()) # turn to df


# Test dataset
# X_train, X_val, y_train, y_val = train_test_split( X, target, train_size = 0.75)
X_test = X_train[500:1000]
Y_test = Y_train[500:1000]
X_test_tfidf_matrix = tfidf_vectorizer.transform(X_test)
print X_test_tfidf_matrix.shape
# print X_test_tfidf_matrix.shape #(500, 1594)
X_test = ch2.transform(X_test_tfidf_matrix)
print X_test.shape


# ML algorithms
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import metrics

#NB
NB_clf = MultinomialNB().fit(chi_filter_training, Y_train)
# work on test data
from sklearn.metrics import accuracy_score
predicted_NB = NB_clf.predict(X_test)
print('NB accuracy: ' + str(accuracy_score(Y_test, predicted_NB)))


#LR
logreg = LogisticRegression().fit(chi_filter_training, Y_train)
# work on test data
from sklearn.metrics import accuracy_score
predicted_LG = logreg.predict(X_test)
print('NB accuracy: ' + str(accuracy_score(Y_test, predicted_LG)))


#SVM
SVM_clf = svm.SVC().fit(chi_filter_training, Y_train)
# work on test data
from sklearn.metrics import accuracy_score
predicted_svm = SVM_clf.predict(X_test)
print('NB accuracy: ' + str(accuracy_score(Y_test, predicted_svm)))



from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
