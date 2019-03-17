
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models import LsiModel
from gensim.models import HdpModel
import operator
from gensim.models import CoherenceModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from nltk import bigrams
#
# #
# # # print str(dt.datetime.time(dt.datetime.now())).split('.')[0].replace(':','_')
# # # startTime = time.time()
# # # time.sleep(5)
# # # print time.time() - startTime
# #
# # # list_test =[['hey', 'really', 'sorry', 'knocked', 'you', 'down', 'but', 'can', 'pick', 'you', 'up', 'at'], ['let', u'u', 'support', 'aldub', 'until', 'forever']]
# # # list_test = [' '.join(l) for l in list_test]
# # # posi_training_data_df = pd.DataFrame(list_test, columns=['tweets'])
# # # posi_training_data_df['label']=0
# # # l = pd.concat([posi_training_data_df, posi_training_data_df], ignore_index=True)
# # # print l['tweets'].tolist()
# # # print list_test
# # #
# # # string_test = '80'
# # # print string_test.isdigit()
# #
# # # 'This is the first document.',
# # # 'This document is the second document.',
# # # 'And this is the third one.',
# # # 'Is this the first document?',
# # # 'is'
# # # corpus = [
# # #     'is it is it is it mine mine',
# # #     'is it mine',
# # #     'is it mine true for me that sounds good',
# # #     'the weather is so good',
# # #     'food is what i need now'
# # # ]
# # # corpus = pd.DataFrame(corpus, columns=['text'])
# # #
# # # vectorizer = TfidfVectorizer()
# # # X = vectorizer.fit_transform(corpus['text'])
# # # # print X
# # # Y = [0,0,1]
# # # selector = SelectKBest(chi2, k=3)
# # # X_new = selector.fit_transform(X.toarray(), np.asarray(Y))
# # # print X_new
# # # '''
# # # [[0.         0.46979139 0.         0.         0.        ]
# # #  [0.         0.6876236  0.         0.53864762 0.        ]
# # #  [0.51184851 0.         0.51184851 0.         0.51184851]
# # #  [0.         0.46979139 0.         0.         0.        ]]
# # # '''
# # #
# #
# #
# # #print selector.get_support(indices=True).tolist() # [0, 1, 4, 5, 7]
# #
# # # feature_name_list_after_chi = [vectorizer.get_feature_names()[i] for i in selector.get_support(indices=True).tolist()]
# # #
# # # '''
# # # print vectorizer.get_feature_names()
# # # [u'for', u'good', u'is', u'it', u'me', u'mine', u'sounds', u'that', u'true']
# # #
# # # print selector.get_support()
# # # [False False False False False False  True  True  True]
# # #
# # # print selector.get_support(indices=True)
# # # [6 7 8]
# # #
# # # print selector.get_support(indices=True).tolist()
# # # [6, 7, 8]
# # # '''
# # #
# # # # print feature_name_list_after_chi #[u'sounds', u'that', u'true']
# # #
# # # X_new = X_new.tolist()
# # # l_f = []
# # # for l in X_new:
# # #     l_index = X_new.index(l)
# # #     t_l = []
# # #     for word_index in range(len(l)):
# # #         if X_new[l_index][word_index] <> 0:
# # #             t_l.append(feature_name_list_after_chi[word_index])
# # #     l_f.append(t_l)
# # #
# # # print l_f
# #
# #
#
# raw = [
#     ['is', 'it', 'mine', 'true', 'for', 'me', 'that', 'sounds', 'good'],
#     ['the', 'weather', 'is', 'so', 'good'],
#     ['food', 'is', 'what', 'i', 'need', 'now'],
#     ['food', 'is', 'what', 'i', 'need', 'now'],
#     ['food', 'is', 'what', 'i', 'need', 'now'],
#     ['food', 'is', 'what', 'i', 'need', 'now'],
#     ['food', 'is', 'what', 'i', 'need', 'now'],
#     ['food', 'is', 'what', 'i', 'need', 'now'],
#
# ]
# #
# dictionary = Dictionary(raw)  # Dictionary(16300 unique tokens: [u'jaesuk', u'sermersheim', u'headband', u'degenere', u'jetline']...)
# corpus = [dictionary.doc2bow(text) for text in raw]
#
# hdp = HdpModel(corpus=corpus, id2word=dictionary)
# print hdp.show_topics()
#
# hdp_topics=hdp.show_topics(formatted=False)
# print hdp_topics
#
# coherence_score = CoherenceModel(model=hdp, texts=raw, dictionary=dictionary).get_coherence()
# #
# print coherence_score
#
# raw_list = [' '.join(l) for l in raw]
# #
# # ---------method 1-----------
# raw_df = pd.DataFrame(raw_list, columns=['tweets'])
# vectorizer = TfidfVectorizer()
# X_tfidf_matrix = vectorizer.fit_transform(raw_df['tweets'])
#
# ch2 = SelectKBest(chi2, k=5)
# X_chi_matrix = ch2.fit_transform(X_tfidf_matrix.toarray(), np.asarray([0,0,1,1,1,0,0,0]))
#
# columns = [vectorizer.get_feature_names()[i] for i in ch2.get_support(indices=True).tolist()]
# values = X_chi_matrix
#
#
# test_df = pd.DataFrame(values, columns=columns)
# test_df2 = pd.DataFrame(index=range(test_df.shape[0]), columns=columns)
#
# for i in columns:
#     test_df2[i] = np.where(test_df[i]>0, i, None)
# l = test_df2.values.tolist()
# f_l=[]
# for sen in l:
#     new_sen = [x for x in sen if x <> None]
#     f_l.append(new_sen)
# # f_l=[l for l in f_l if l <> []]
# print f_l
#
# dictionary = Dictionary(f_l)  # Dictionary(16300 unique tokens: [u'jaesuk', u'sermersheim', u'headband', u'degenere', u'jetline']...)
# corpus = [dictionary.doc2bow(text) for text in f_l]
#
# print dictionary
# print corpus
#
#
# #
# #
# #
# # ch2 = SelectKBest(chi2, k=5)
# # X_chi_matrix = ch2.fit_transform(X_tfidf_matrix.toarray(), np.asarray([0,0,1,1,1,1,1,1]))
# #
# # X_chi_matrix_list = X_chi_matrix.tolist()
# #
# # final_list = []
# # for tweets_token_list in X_chi_matrix_list:
# #     inter_mediate_list = []
# #     index = -1
# #     for i in tweets_token_list:
# #         index+=1
# #         if i == 0.0:
# #             pass
# #         else:
# #             inter_mediate_list.append((index, i))
# #     if inter_mediate_list <> []:
# #         final_list.append(inter_mediate_list)
# #
# #
# # names = vectorizer.get_feature_names()
# # #[u'food', u'for', u'good', u'is', u'it', u'me', u'mine', u'need', u'now', u'so', u'sounds', u'that', u'the', u'true', u'weather', u'what']
# # index = -1
# # name_dic = {}
# # for i in names:
# #     index+=1
# #     name_dic.update({index:i})
#
#
# #
# # print name_dic
# # print final_list
# # dictionary = name_dic
# # corpus = final_list
# # print dictionary
# # print corpus
#
#
# # '''
# # [[0.29889729 0.         0.         0.35664612 0.        ]
# #  [0.42646808 0.5088644  0.5088644  0.         0.5088644 ]
# #  [0.         0.         0.         0.         0.        ]
# #  [0.         0.         0.         0.         0.        ]
# #  [0.         0.         0.         0.         0.        ]
# #  [0.         0.         0.         0.         0.        ]
# #  [0.         0.         0.         0.         0.        ]
# #  [0.         0.         0.         0.         0.        ]]
# # '''
#
#
# #---------method 2-------------#
# # dictionary = Dictionary(raw)  # Dictionary(16300 unique tokens: [u'jaesuk', u'sermersheim', u'headband', u'degenere', u'jetline']...)
# # corpus = [dictionary.doc2bow(text) for text in raw]  # [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 2)], [(9, 1), (10, 1)]
# # # print dictionary
# # # print corpus
# #
# #
# # lsi = LsiModel(corpus=corpus, num_topics=3, id2word=dictionary)
# # for idx in range(10):
# #     print("LSI Model:")
# #     print("Topic #%s:" % idx, lsi.print_topic(idx, 10))
# #
# # hdp = HdpModel(corpus=corpus, id2word=dictionary)
# # for idx in range(len(hdp.show_topics())):
# #     print("HDP Model:")
# #     print("Topic #%s:" % idx, hdp.print_topic(idx, len(hdp.show_topics())))
# # print("=" * 20)
# # #
# # # hdpmtopics = hdp.show_topics(num_topics=2, num_words=10, formatted=False)
# # # print hdpmtopics
# #
# # #
# # print corpus[0]
# # print hdp[corpus[0]]
#
#
#
#
#
#
# max_topic_num = 20
# BASE = 3
# if max_topic_num < BASE:
#     raise ValueError("Please enter limit > %d. You entered %d" % (BASE, max_topic_num))
# c_v = []
# lm_list = []
# for num_topics in range(BASE, max_topic_num):
#     lm = LdaModel(corpus=corpus, num_topics=3, id2word=dictionary)
#     lm_list.append(lm)
#     cm = CoherenceModel(texts=raw, model=lm, dictionary=dictionary, coherence='c_v')
#     c_v.append(cm.get_coherence())
# # print cm.get_coherence()
#
#
# # Show graph
# x = range(BASE, max_topic_num)
# plt.plot(x, c_v)
# plt.xlabel("num_topics")
# plt.ylabel("Coherence score")
# plt.legend(("c_v"), loc='best')
# plt.show()




# def rolling_mean(x, w):
#     """Compute a rolling mean of x
#     Right-aligned. Padded with NaNs on the front so the output is the same
#     size as x.
#     Parameters
#     ----------
#     x: Array.
#     w: Integer window size (number of elements).
#     Returns
#     -------
#     Rolling mean of x with window size w.
#     """
#     s = np.cumsum(np.insert(x, 0, 0))
#     print s
#
#     prefix = np.empty(w - 1)
#     print prefix
#
#     prefix.fill(np.nan)
#     print prefix
#
#     print s[w:]
#     print s[:-w]
#     print s[w:] - s[:-w]
#     print (s[w:] - s[:-w]) / float(w)
#
#     return np.hstack((prefix, (s[w:] - s[:-w]) / float(w)))  # right-aligned
#
# x = np.array([1,2,3,4,5,6])
# output = rolling_mean(x,3)
# print output

# class Foo(object):
#     def __init__(self, a, b):
#         self.frobnicate = a
#         self.b = b
#
#
# class Bar(Foo):
#     def __init__(self, a, c):
#         super(Bar, self).__init__(a, c)
#         self.b = 34
#         self.frazzle = c
#
# bar = Bar(1,2)
# print "frobnicate:", bar.frobnicate
# print "b:", bar.b
# print "frazzle:", bar.frazzle

# 
# date_horizon_list = [{'daily': [7, 30, 90, 180, 360, 720]}, {'30_days_aggr': [30, 60, 90, 120, 180, 240, 360, 720]},
#                      {'90_days_aggr': [90, 180, 270, 360, 450, 540, 630, 720]}]
# print date_horizon_list[0].values()

#
# x = 'global x'
#
# class TestVariable():
#
#     x = 'class x'
#
#     def layer_first(self):
#
#         y = 'layer first variable'
#
#         def layer_second():
#             print y
#             print x
#
#         layer_second()
#
#     def main(self):
#         self.layer_first()
#
# a = TestVariable()
# a.main()

import re
l =[u'zero', u'when', u'come', u'dog', u'house', u'feed', u'and', u'you_damn']
# bigram_token_list = []
# for i in l:
#     if '_' in i:
#         bigram_token_list.append(i)
# print bigram_token_list

# for i in range(0,9):
#     print '{0}\r'.format(i),

# x = input("How many top n features do you want to extract?")
# print x
a = (
        '__name__', '__force_rebuild__', '__metadata__', '__stabilization_window__',
        '__output_contract__', '__input_contracts__', '__monitor_level__', '__ignore_locks__',
        '__previous_output_contract__'
    )
b = ('test',)
print a+b