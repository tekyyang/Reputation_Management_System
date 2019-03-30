
import ast
import pandas as pd
import numpy as np

from gensim.corpora import Dictionary
from gensim.models import LdaModel, LsiModel, HdpModel, CoherenceModel
from gensim.models.phrases import Phrases

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import svm, linear_model, naive_bayes, ensemble
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import time
from matplotlib import pyplot as plt
import itertools

class topic_model_builder():

    def __init__(self, training_dataset_paths, test_dataset_path):
        '''
        :param input_training_data_path (String): the entire path of all the training data's paths. Could be parsed as a list. Separated by ','.
        :param input_test_data_path (String): same as above
        '''
        self.class_startTime = time.time()

        #read training datasets (multiple)
        for path in training_dataset_paths.split(','): #return a list of path
            path = path.strip()
            if 'posi_test' in path:
                with open(path, 'r') as f:
                    lines = f.readlines()  # read all the lines in the file
                    self.posi_training_data_list = [[ast.literal_eval(line.replace('\n', '') ),1] for line in lines]
                    self.posi_training_data_tokens = [item[0] for item in self.posi_training_data_list]
                    self.posi_training_data_labels = [item[1] for item in self.posi_training_data_list]
                    tweets_list = [' '.join(l) for l in self.posi_training_data_tokens]
                    self.posi_training_data_df = pd.DataFrame(tweets_list, columns=['tweets'])
                    self.posi_training_data_df['label'] = 0

            elif 'nega_test' in path:
                with open(path, 'r') as f:
                    lines = f.readlines()  # read all the lines in the file
                    self.nega_training_data_list = [[ast.literal_eval(line.replace('\n', '') ),0] for line in lines]
                    self.nega_training_data_tokens = [item[0] for item in self.nega_training_data_list]
                    self.nega_training_data_labels = [item[1] for item in self.nega_training_data_list]
                    tweets_list = [' '.join(l) for l in self.nega_training_data_tokens]
                    self.nega_training_data_df = pd.DataFrame(tweets_list, columns=['tweets'])
                    self.nega_training_data_df['label'] = 1

        #read test dataset (multiple)
        for path in test_dataset_path.split(','):  # return a list of path
            path = path.strip()
            if 'pos' in path:
                with open(path, 'r') as f:
                    lines = f.readlines()  # read all the lines in the file
                    self.posi_test_data_list = [[ast.literal_eval(line.replace('\n', '') ),1] for line in lines]
                    self.posi_test_data_tokens = [item[0] for item in self.posi_test_data_list]
                    self.posi_test_data_labels = [item[1] for item in self.posi_test_data_list]
                    tweets_list = [' '.join(l) for l in self.posi_test_data_tokens]
                    self.posi_test_data_df = pd.DataFrame(tweets_list, columns=['tweets'])
                    self.posi_test_data_df['label'] = 0

            elif 'neg' in path:
                with open(path, 'r') as f:
                    lines = f.readlines()  # read all the lines in the file
                    self.nega_test_data_list = [[ast.literal_eval(line.replace('\n', '') ),0] for line in lines]
                    self.nega_test_data_tokens = [item[0] for item in self.nega_test_data_list]
                    self.nega_test_data_labels = [item[1] for item in self.nega_test_data_list]
                    tweets_list = [' '.join(l) for l in self.nega_test_data_tokens]
                    self.nega_test_data_df = pd.DataFrame(tweets_list, columns=['tweets'])
                    self.nega_test_data_df['label'] = 1

    def to_string_list_tool(self, input, df_column_name=None, mode='df_to_token_list'):
        '''
        :param input: df or a list of token; If df, please provide the column name to be converted;
                      If df, please provide a df other than a series;
                      If list, please provide something like [['today', 'is', 'a', 'good', 'day'], [...]]
        :param tweet_column_name:
        :return: a list based on selected mode.
                 If df_to_token_list, return [['today', 'is', 'a', 'good', 'day'], ['...']]
                 If token_list_to_string_list, return ['today is a good day', '...']
        '''
        if mode=='df_to_token_list':
            assert isinstance(input, pd.DataFrame), 'not a dataframe!'
            tweet_list = input[df_column_name].tolist()
            tweet_list = [sen.split(' ') for sen in tweet_list]
            return tweet_list

        elif mode == 'df_to_string_list':
            tweet_list = input[df_column_name].tolist()
            return tweet_list

        elif mode=='token_list_to_string_list':
            assert isinstance(input, list), 'not a list!'
            tweet_list = [' '.join(l) for l in input]
            return tweet_list

        else:
            print 'Input is not df or list. Can not convert.'

    def data_resampling(self, posi_training_data_df, nega_training_data_df):
        '''
        To deal with data imbalance issue here.
        DataFrame.sample method to get random samples each class
        :param tweets_to_be_resampled (df): posi and nega tweets df waiting for be resampled
        :return (list): balanced posi and nega datasets list
        '''
        def random_upper_sampling(posi_training_data_df, nega_training_data_df):
            posi_training_data_df_under = posi_training_data_df.sample(len(nega_training_data_df))
            df_training_posi_resampled = posi_training_data_df_under
            df_training_nega_resampled = nega_training_data_df
            return df_training_posi_resampled, df_training_nega_resampled

        def random_under_sampling(posi_training_data_df, nega_training_data_df):
            nega_training_data_df_under = nega_training_data_df.sample(len(posi_training_data_df))
            df_training_posi_resampled = posi_training_data_df
            df_training_nega_resampled = nega_training_data_df_under
            return df_training_posi_resampled, df_training_nega_resampled

        def other_sampling_tech(df_train):
            pass

        def main(type='r_under_s'):
            if type=='r_under_s':
                df_training_posi_resampled, df_training_nega_resampled = random_upper_sampling(posi_training_data_df, nega_training_data_df)
                return df_training_posi_resampled, df_training_nega_resampled
            elif type=='r_upper_s':
                df_training_posi_resampled, df_training_nega_resampled = random_under_sampling(posi_training_data_df, nega_training_data_df)
                return df_training_posi_resampled, df_training_nega_resampled
            else:
                df_training_posi_resampled=None
                df_training_nega_resampled=None
                return df_training_posi_resampled, df_training_nega_resampled

        startTime = time.time()
        df_training_posi_resampled, df_training_nega_resampled=main()
        training_posi_resampled_token_list = self.to_string_list_tool(df_training_posi_resampled, 'tweets')
        training_nega_resampled_token_list = self.to_string_list_tool(df_training_nega_resampled, 'tweets')
        print '=== (1) Finish resampling! Taking ' + str(round((time.time() - startTime),4)) + 's ==='
        resampling_processing_time = str(round((time.time() - startTime), 4))
        print 'After resampling, now there are '+str(len(training_posi_resampled_token_list))+ ' positive tweets and '+str(len(training_nega_resampled_token_list))+' negative tweets!\n'
        return training_posi_resampled_token_list, training_nega_resampled_token_list, resampling_processing_time

    def bigram_or_unigram_extactor(self, posi_training_token_list, nega_training_token_list, min_count=20, mode='uni_and_bigram'):
        '''
        :param  posi_training_token_list, nega_training_token_list (token_list)
        :return: posi_unigran_bigram_training_token_list, posi_unigran_bigram_training_token_list (mode='uni_and_bigram')
                 or posi_training_token_list, nega_training_token_list (mode='unigram')
        '''
        startTime = time.time()
        training_list = posi_training_token_list + nega_training_token_list
        bigram = Phrases(training_list, min_count=min_count)

        # this will return something like
        # [[u'still', u'and', u'always', u'will_be', u'my_favorite', u'artist'],...]
        posi_bigram_training_token_list_with_unigram = [bigram[sent] for sent in posi_training_token_list]
        nega_bigram_training_token_list_with_unigram = [bigram[sent] for sent in nega_training_token_list]

        # picking only the bigrams
        posi_bigram_training_token_list = []
        for token_list in posi_bigram_training_token_list_with_unigram:
            bigram_token_list = []
            for token in token_list:
                if '_' in token:
                    bigram_token_list.append(token)
            posi_bigram_training_token_list.append(bigram_token_list)

        nega_bigram_training_token_list = []
        for token_list in nega_bigram_training_token_list_with_unigram:
            bigram_token_list = []
            for token in token_list:
                if '_' in token:
                    bigram_token_list.append(token)
            nega_bigram_training_token_list.append(bigram_token_list)

        # combine unigram and bigram
        posi_unigran_bigram_training_token_list = []
        for i in range(len(posi_bigram_training_token_list)):
            posi_unigran_bigram_training_token_list.append(posi_training_token_list[i]+posi_bigram_training_token_list[i])

        nega_unigran_bigram_training_token_list = []
        for i in range(len(nega_bigram_training_token_list)):
            nega_unigran_bigram_training_token_list.append(nega_training_token_list[i] + nega_bigram_training_token_list[i])

        print '=== (2) Finish bigram extraction! Mode is: '+ mode + '. Taking ' + str(round((time.time() - startTime),4)) + 's ==='

        feature_extraction_processing_time = str(round((time.time() - startTime), 4))

        if mode == 'uni_and_bigram':
            # print 'example: (there is no guarantee that a bigram would be shown tho :)'
            # print posi_unigran_bigram_training_token_list[:10]
            # print '\n'
            return posi_unigran_bigram_training_token_list, nega_unigran_bigram_training_token_list, feature_extraction_processing_time
        elif mode =='unigram':
            # print 'example:'
            # print posi_training_token_list[:10]
            # print '\n'
            return posi_training_token_list, nega_training_token_list, feature_extraction_processing_time
        else:
            print 'there is no such mode: '+mode+'!'

    def feature_selection(self, posi_training_token_list, nega_training_token_list, feature_represent_mode='tfidf', feature_selection_mode='chi2' ):
        '''
        :returns
        X_chi_matrix: the weighted matrix after feature selection
        feature_name_list_after_chi: a list of feature names after feature selection
        token_list_after_chi2: referred by name. Notice that it doesn't reduce the amount of tweets.
                               So positive vs negative tweets amounts are still 50:50
        '''
        startTime = time.time()
        posi_training_string_list=self.to_string_list_tool(posi_training_token_list, mode='token_list_to_string_list')
        nega_training_string_list=self.to_string_list_tool(nega_training_token_list, mode='token_list_to_string_list')

        X_train_string_list = posi_training_string_list + nega_training_string_list
        Y_train_list = [1] * len(posi_training_string_list) + [0] * len(nega_training_string_list)
        X_train_df = pd.DataFrame(X_train_string_list, columns=['text'])

        #--- feature representation from tfidf or word count ---#
        if feature_represent_mode == 'tfidf':
            # tf-idf vectorizer
            vectorizer = TfidfVectorizer()
            X_matrix = vectorizer.fit_transform(X_train_df['text'])

        else:
            # word count vectorizer
            vectorizer = CountVectorizer()
            X_matrix = vectorizer.fit_transform(X_train_df['text'])

        print '=== (3) Start feature selection! ==='
        print 'X_matrix.shape is (number of rows, number of columns):'
        print X_matrix.shape
        # n = input("How many top n features do you want to extract?")
        print 'Taking the top ' +str(5000)+ ' features...'
        top_n_feature = 5000

        #--- feature selection from chi2 or mutual info ---#
        if feature_selection_mode =='chi2':

            # ch2 selector
            ch2 = SelectKBest(chi2, k=top_n_feature)

            #get the feature matrix
            X_chi_matrix = ch2.fit_transform(X_matrix.toarray(), np.asarray(Y_train_list))
            '''
            [[0.         0.46979139 0.         0.         0.        ]
             [0.         0.6876236  0.         0.53864762 0.        ]
             [0.51184851 0.         0.51184851 0.         0.51184851]
             [0.         0.46979139 0.         0.         0.        ]]
            '''

            #get the feature names
            feature_name_list_after_chi = [vectorizer.get_feature_names()[i] for i in ch2.get_support(indices=True).tolist()]

            def get_token_list_after_feature_selection(X_chi_matrix, feature_name_list_after_chi):
                columns_names = feature_name_list_after_chi
                X_chi_matrix = X_chi_matrix
                conversion_df = pd.DataFrame(X_chi_matrix, columns=columns_names)
                conversion_df_2 = pd.DataFrame(index=range(conversion_df.shape[0]), columns=columns_names)

                for i in columns_names:
                    conversion_df_2[i] = np.where(conversion_df[i] > 0, i, None)
                l = conversion_df_2.values.tolist()

                token_list_after_chi2 = []
                for sen in l:
                    new_sen = [x for x in sen if x <> None]
                    token_list_after_chi2.append(new_sen)

                return token_list_after_chi2

            token_list_after_chi2=get_token_list_after_feature_selection(X_chi_matrix, feature_name_list_after_chi)

            print '=== (3) Finish feature selection! Feature representation mode is'+ feature_represent_mode +'. Feature selection mode is '+feature_selection_mode+'. Taking ' + str(round((time.time() - startTime),4)) + 's ==='
            feature_selection_processing_time = str(round((time.time() - startTime),4))
            print 'example output: X_chi_matrix, feature_name_list_after_chi, token_list_after_chi2'
            print X_chi_matrix[:2]
            print feature_name_list_after_chi[:2]
            print token_list_after_chi2[:2]
            print '\n'
            return X_chi_matrix,feature_name_list_after_chi, token_list_after_chi2, ch2, feature_selection_processing_time

        else:
            return [],[],[],[], ''

   # --- proposed classifier building start --- #

    def build_lda_lsi_model(self, token_list_after_feature_selection, min_topic_num = 3, max_topic_num = 30, model='lda'):
        """
        doc: https://radimrehurek.com/gensim/models/lsimodel.html
        Model Building
        """
        startTime = time.time()

        print '=== (4) Start building topic models! ==='

        dictionary = Dictionary(token_list_after_feature_selection)  # Dictionary(16300 unique tokens: [u'jaesuk', u'sermersheim', u'headband', u'degenere', u'jetline']...)
        corpus = [dictionary.doc2bow(text) for text in token_list_after_feature_selection]


        if max_topic_num < min_topic_num:
            raise ValueError("Please enter limit > %d. You entered %d" % (min_topic_num, max_topic_num))
        c_v = []
        lm_list = []
        print 'Start building '+model+' model!'
        for num_topics in range(min_topic_num, max_topic_num):
            if model=='lda':
                lm = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
            elif model=='lsi':
                lm = LsiModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
            else:
                lm=[]
                print 'please input the correct model name!'

            lm_list.append(lm)
            cm = CoherenceModel(model=lm, texts=token_list_after_feature_selection, dictionary=dictionary, coherence='c_v')
            c_v.append(cm.get_coherence())
            print 'finish building topic = {0}\r'.format(str(num_topics))

        print '=== (4) Finish ' + model + ' graph building! Taking '  + str(round((time.time() - startTime),4)) + 's ===\n'

        lad_lsi_processing_time = str(round((time.time() - startTime), 4))

        #--- show the graph for one single topic model ---#
        x = range(min_topic_num, max_topic_num)
        plt.plot(x, c_v)
        plt.xlabel("num_topics")
        plt.ylabel("Coherence score")
        plt.legend(("c_v"), loc='best')
        plt.show()

        highest_coherence_score = np.argmax(c_v)
        top_model = lm_list[highest_coherence_score]
        model_topics = top_model.show_topics(formatted=False)

        return top_model, model_topics, highest_coherence_score, dictionary, corpus, lad_lsi_processing_time

    def build_the_hdp_model(self, token_list_after_feature_selection):

        startTime = time.time()
        dictionary = Dictionary(token_list_after_feature_selection)  # Dictionary(16300 unique tokens: [u'jaesuk', u'sermersheim', u'headband', u'degenere', u'jetline']...)
        corpus = [dictionary.doc2bow(text) for text in token_list_after_feature_selection]

        hdp = HdpModel(corpus=corpus, id2word=dictionary)

        hdp_topics = hdp.show_topics(formatted=False)

        coherence_score = CoherenceModel(model=hdp, texts=token_list_after_feature_selection, dictionary=dictionary, coherence='c_v').get_coherence()

        print '=== (4) Finish hdp graph building! Taking ' + str(round((time.time() - startTime),4)) + 's ===\n'

        building_hdp_processing_time = str(round((time.time() - startTime), 4))

        return hdp, hdp_topics, coherence_score, dictionary, corpus, building_hdp_processing_time

    def evaluate_bar_graph(self, coherences, indices):
        """
        Function to plot bar graph to evaluate all the topic models we get.
        coherences: list of coherence values
        indices: Indices to be used to mark bars. Length of this and coherences should be equal.
        """
        assert len(coherences) == len(indices)
        n = len(coherences)
        x = np.arange(n)
        plt.bar(x, coherences, width=0.2, tick_label=indices, align='center')
        plt.xlabel('Models')
        plt.ylabel('Coherence Value')
        plt.show()

    def best_topic_model_selecion(self,lda_model, lsi_model, hdp_mode,
                                  lda_highest_coherence_score, lsi_highest_coherence_score, hdp_coherence_score,
                                  corpus_lda, corpus_lsi, corpus_hdp,
                                  dictionary_lda, dictionary_lsi, dictionary_hdp):
        if lda_highest_coherence_score > hdp_coherence_score and lda_highest_coherence_score > lsi_highest_coherence_score:
            return lda_model, 'lda', corpus_lda, dictionary_lda
        elif lsi_highest_coherence_score > hdp_coherence_score:
            return lsi_model, 'lsi', corpus_lsi, dictionary_lsi
        else:
            return hdp_mode, 'hdp', corpus_hdp, dictionary_hdp

    def get_tweet_topic_matrix_based_on_best_topic_model(self, best_topic_model, corpus):
        #https://groups.google.com/forum/#!topic/gensim/F4AWfh9yIhM

        best_topic_model.minimum_probability = 0.0 #prob smaller than this would be filtered
        doc_topic_collection_list = []
        for i in range(len(corpus)):
            doc_topic_collection_list.append([prob_tuple[1] for prob_tuple in best_topic_model[corpus[i]]])

        tweet_topic_distribution_df = pd.DataFrame.from_records(doc_topic_collection_list)
        # tweet_topic_distribution_df['Y'] = np.where(tweet_topic_distribution_df.index<=tweet_topic_distribution_df.shape[0]/2-1, 1, 0)
        return tweet_topic_distribution_df

    def collect_clustering_info(self, tweet_topic_distribution_df):

        startTime = time.time()
        print '=== (5) Start collecting clustering info... ===\n'


        # Run the Kmeans algorithm and get the index of data points clusters
        inertia_list = []
        lable_list = []
        model_list =[]
        list_k = list(range(1, 30))

        tweet_topic_distribution_df = tweet_topic_distribution_df.fillna(value = 0)

        for k in list_k:
            km = KMeans(n_clusters=k)
            km.fit(tweet_topic_distribution_df)
            inertia_list.append(km.inertia_)
            lable_list.append(km.labels_)
            model_list.append(km)

        # Plot sse against k
        plt.figure(figsize=(6, 6))
        plt.plot(list_k, inertia_list, '-o')
        plt.xlabel(r'Number of clusters *k*')
        plt.ylabel('Sum of squared distance')

        print '=== (5) Finish clustering graph building! Taking '  + str(round((time.time() - startTime),4)) + 's ===\n'

        collect_clustering_info_processing_time = str(round((time.time() - startTime), 4))

        return list_k, lable_list, model_list, collect_clustering_info_processing_time

    def add_clustering_info_to_df(self, tweet_topic_distribution_df, list_k, lable_list, model_list):

        startTime = time.time()

        print 'You can choose cluster number from '+str(list_k[0])+' to '+str(list_k[-1]) +'.'

        # test_n_list = []
        # while True:
        #     n = input("How many clusters do you want? Input number for checking details. Input -1 for exit.")
        #     if n in list_k and n <> -1:
        #         index_n = list_k.index(n)
        #         lable_n = lable_list[index_n]
        #         unique, counts = np.unique(lable_n, return_counts=True)
        #         print dict(zip(unique, counts))
        #         test_n_list.append(n)
        #     elif n not in list_k and n <> -1:
        #         print 'n is not in list_k!'
        #     else:
        #         break
        #
        # number_of_cluster = test_n_list[-1]
        # print 'choose n = {0}!'.format(n)

        number_of_cluster = 6

        index_n = list_k.index(number_of_cluster)
        selected_clustering_labels = lable_list[index_n]
        selected_kmeans_model = model_list[index_n]
        tweet_topic_distribution_with_cluster_df = tweet_topic_distribution_df
        tweet_topic_distribution_with_cluster_df['clustering_labels'] = selected_clustering_labels
        tweet_topic_distribution_with_cluster_df['Y'] = np.where(tweet_topic_distribution_df.index <= tweet_topic_distribution_df.shape[0] / 2 - 1, 1, 0)

        print '=== (6) Finish adding clustering into to df! Taking '  + str(round((time.time() - startTime),4)) + 's ===\n'

        add_clustering_info_to_df_processing_time = str(round((time.time() - startTime), 4))

        return tweet_topic_distribution_with_cluster_df, selected_kmeans_model, number_of_cluster, add_clustering_info_to_df_processing_time

    def classifier_building(self, tweet_topic_distribution_with_cluster_df, number_of_cluster, corpus, classifier = 'logistic_regression'):
        # for tweets in each cluster label, build a classifier for it
        # return the classifier

        startTime = time.time()

        vectorizer_clf_dict = {}
        for i in range(0,number_of_cluster):
            # index_list is a list of index for a certain cluster
            # Y_list is a list of Y labels for a certain cluster

            index_list = tweet_topic_distribution_with_cluster_df[tweet_topic_distribution_with_cluster_df['clustering_labels'] == i].index.tolist()
            Y_list = tweet_topic_distribution_with_cluster_df.Y[tweet_topic_distribution_with_cluster_df['clustering_labels'] == i].tolist()

            #take tweets and labels for each cluster
            tweets = []
            for k in index_list: #index list for a cluster
                tweets.append([str(k[0]) for k in corpus[k]]) #convert [(257,1), (...)] to [257, ...]. The final tweets look like [['7', '12', '13', '14', '15', ], ['7', '42', '43', '44'], ['7', '81'],...]
            labels = Y_list

            #text representation
            vectorizer = TfidfVectorizer()
            tweet_list = self.to_string_list_tool(tweets, mode='token_list_to_string_list')
            vectorizer.fit(tweet_list)
            X_train = vectorizer.transform(tweet_list).toarray()
            Y_train = labels

            if classifier == 'logistic_regression':
                clf = linear_model.LogisticRegression()
            else:
                clf = svm.SVC(kernel='linear', gamma='auto')
            clf.fit(X_train, Y_train)
            vectorizer_clf_dict.update({i:[vectorizer, clf]})

        print '=== (7) Finish classifier building! Taking ' + str(round((time.time() - startTime), 4)) + 's ===\n'

        classifier_building_processing_time = str(round((time.time() - startTime), 4))

        return vectorizer_clf_dict, classifier_building_processing_time

    def test_data_fit_in_model(self, vectorizer_clf_dict, best_topic_model, dictionary, selected_kmeans_model):

        startTime = time.time()

        # create corpus for test dataset based on the dictionary of the chosen model
        test_posi_corpus = [dictionary.doc2bow(text) for text in trail.posi_test_data_tokens if dictionary.doc2bow(text) <> []]  # for some tweet in test, the bow could be []
        test_nega_corpus = [dictionary.doc2bow(text) for text in trail.nega_test_data_tokens if dictionary.doc2bow(text) <> []]
        test_corpus = test_posi_corpus + test_nega_corpus

        # for each test data, use the best topic model we built to get the tweet, topic distribution
        # based on the distribution, find the corresponding cluster for it
        best_topic_model.minimum_probability = 0.0  # prob smaller than this would be filtered
        cluster_label_list = []  # generate label list for all test tweets
        for i in range(len(test_corpus)):
            test_tweet_prob_distribution = [prob_tuple[1] for prob_tuple in best_topic_model[test_corpus[i]]]
            cluster_label = selected_kmeans_model.predict(test_tweet_prob_distribution)
            cluster_label_list.append(cluster_label.tolist()[0])

        # apply the certian classifier on the test tweet
        Y_pred = []
        Y_test = len(test_posi_corpus) * [1] + len(test_nega_corpus) * [0]

        for i in range(len(cluster_label_list)):
            for j in vectorizer_clf_dict.keys():
                if cluster_label_list[i] == j:
                    # select the corresponding vectorizer and clf
                    vectorizer = vectorizer_clf_dict[j][0]  # vectorizer
                    clf = vectorizer_clf_dict[j][1]  # clf

                    x_raw_tweet = [str(k[0]) for k in test_corpus[i]] #['957', '1005', '1102', '1336', '2989']
                    x_raw_tweet = [' '.join(x_raw_tweet)] #['957 1005 1102 1336 2989']
                    x_test = vectorizer.transform(x_raw_tweet).toarray()
                    y_pred = clf.predict(x_test)
                    Y_pred = Y_pred + y_pred.tolist()

        print '=== (8) Finish test data fit in! Taking ' + str(round((time.time() - startTime), 4)) + 's ===\n'

        test_data_fit_in_processing_time = str(round((time.time() - startTime), 4))

        print '=== The overall program taking ' + str(round((time.time() - self.class_startTime), 4)) + 's! ===\n'

        return Y_test, Y_pred, cluster_label_list, test_data_fit_in_processing_time

    def evaluation(self, Y_test, Y_pred, cluster_label_list):
        print "=== (9) Performance evaluation moment! ==="

        print "Overall Performance - confusion matrix:"
        cm = confusion_matrix(Y_test, Y_pred)
        print cm
        np.set_printoptions(precision=2)
        plt.figure()
        self.plot_confusion_matrix(cm, classes=[0, 1], title='Confusion matrix')

        print "Overall Performance - classification report:"
        print(classification_report(Y_test, Y_pred))

        print "Overall Performance - accuracy score:"
        print(accuracy_score(Y_test, Y_pred))

        def generate_sub_class(Y_test, Y_pred, cluster_label_list):
            prepare_dict = {
                'Y_test': Y_test,
                'Y_pred': Y_pred,
                'Cluster_labels': cluster_label_list
            }

            # create a dataframe
            df = pd.DataFrame(prepare_dict)
            return df

        evaluation_df = generate_sub_class(Y_test, Y_pred, cluster_label_list)

        unique_cluster_labels = sorted(list(set(cluster_label_list)))
        for i in unique_cluster_labels:
            group_piece = evaluation_df[evaluation_df['Cluster_labels'] == i]
            group_Y_test = group_piece['Y_test']
            group_Y_pred = group_piece['Y_pred']

            print 'For group ' + str(i) + ' - confusion matrix:'
            cm = confusion_matrix(group_Y_test, group_Y_pred)
            print cm
            np.set_printoptions(precision=2)
            plt.figure()
            self.plot_confusion_matrix(cm, classes=[0, 1], title='Confusion matrix for group ' + str(i))

            print 'For group ' + str(i) + ' - classification report:'
            print(classification_report(group_Y_test, group_Y_pred))

            print 'For group ' + str(i) + ' - accuracy score:'
            print(accuracy_score(group_Y_test, group_Y_pred))


    # --- baseline classifier building start --- #

    def baseline_model_builder(self, token_list_after_feature_selection, mode = 'tfidf'):

        startTime = time.time()

        # Y train
        Y_train = [1] * (len(token_list_after_feature_selection)/2) + [0] * (len(token_list_after_feature_selection)/2)

        # build X train
        tweets_list_after_feature_selection = self.to_string_list_tool(token_list_after_feature_selection, mode='token_list_to_string_list')

        if mode == 'tfidf':
            vectorizer = TfidfVectorizer()
        else:
            vectorizer = CountVectorizer()

        vectorizer.fit(tweets_list_after_feature_selection)
        X_train = vectorizer.transform(tweets_list_after_feature_selection).toarray()

        # a list of baseline classifiers
        lg_clf = linear_model.LogisticRegression()
        nb_clf = naive_bayes.MultinomialNB()
        # svm_clf = svm.SVC()
        # rf_clf = ensemble.RandomForestClassifier()
        # b_clf = xgboost.XGBClassifier()

        # fit the training dataset on the classifier
        lg_clf.fit(X_train, Y_train)
        nb_clf.fit(X_train, Y_train)
        # svm_clf.fit(X_train, Y_train)
        # rf_clf.fit(X_train, Y_train)
        # b_clf.fit(X_train, Y_train)

        print '=== (10) Finish baseline classifier building! Taking ' + str(round((time.time() - startTime), 4)) + 's ===\n'
        baseline_classifier_building_processing_time = str(round((time.time() - startTime), 4))

        return vectorizer,\
                {
                    'logistic_regression':lg_clf,
                    'naive_bayes':nb_clf,
                    # 'svm': svm_clf,
                    # 'random_forest':rf_clf,
                    # 'boosting':b_clf
                }, \
               baseline_classifier_building_processing_time

    def baseline_test_data_fit_in_model(self, vectorizer, baseline_clf_dict):

        startTime = time.time()

        Y_test = len(self.posi_test_data_list) * [1] + len(self.nega_test_data_list) * [0]

        X_test_pre = self.to_string_list_tool(input=self.posi_test_data_tokens + self.nega_test_data_tokens, mode='token_list_to_string_list')

        X_test = vectorizer.transform(X_test_pre)

        baseline_result = {}
        for clf_name, clf in baseline_clf_dict.iteritems():
            y_pred = clf.predict(X_test).tolist()
            baseline_result.update({clf_name:y_pred})

        print '=== (11) Finish baseline test data fit in! Taking ' + str(round((time.time() - startTime), 4)) + 's ===\n'
        baseline_test_data_fit_in = str(round((time.time() - startTime), 4))

        return Y_test, baseline_result, baseline_test_data_fit_in

    def baseline_evaluation(self, Y_test, baseline_result):

        print "=== (12) Baseline performance evaluation! ==="

        for clf_name, Y_pred in baseline_result.iteritems():

            print clf_name + " - confusion matrix:"
            cm = confusion_matrix(Y_test, Y_pred)
            np.set_printoptions(precision=2)
            plt.figure()
            self.plot_confusion_matrix(cm, classes=[0, 1],
                                  title='Confusion matrix')

            print clf_name + " - classification report:"
            print(classification_report(Y_test, Y_pred))

            print clf_name + " - accuracy score:"
            print(accuracy_score(Y_test, Y_pred))

    def plot_confusion_matrix(self, cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')


trail = topic_model_builder(training_dataset_paths='/Users/yibingyang/Documents/final_thesis_project/Data/Twitter/after_preprocessing/posi_test.txt,\
                                                    /Users/yibingyang/Documents/final_thesis_project/Data/Twitter/after_preprocessing/nega_test.txt',
                            test_dataset_path='/Users/yibingyang/Documents/final_thesis_project/Data/Twitter/test_dataset/positive_test_tweets.txt,\
                                               /Users/yibingyang/Documents/final_thesis_project/Data/Twitter/test_dataset/negative_test_tweets.txt')


# ------ data preparation ------ #
training_posi_resampled_token_list, training_nega_resampled_token_list, resampling_processing_time = trail.data_resampling(trail.posi_training_data_df,trail.nega_training_data_df)
posi_bigram_training_token_list, nega_bigram_training_token_list, feature_extraction_processing_time = trail.bigram_or_unigram_extactor(training_posi_resampled_token_list, training_nega_resampled_token_list, min_count=20)
X_chi_matrix,feature_name_list_after_feature_selection, token_list_after_feature_selection, ch2, feature_selection_processing_time = trail.feature_selection(posi_bigram_training_token_list, nega_bigram_training_token_list)

# ------ build the three topic models: lda, lsi, hdp ------ #
top_lda_model, top_lda_model_topics, lda_highest_coherence_score, dictionary_lda, corpus_lda, lda_processing_time = trail.build_lda_lsi_model(token_list_after_feature_selection, min_topic_num=3, max_topic_num=30, model='lda')
top_lsi_model, top_lsi_model_topics, lsi_highest_coherence_score, dictionary_lsi, corpus_lsi, lsi_processing_time = trail.build_lda_lsi_model(token_list_after_feature_selection, min_topic_num=3, max_topic_num=30, model='lsi')
hdp_model,     hdp_topics,           hdp_coherence_score,         dictionary_hdp, corpus_hdp, hdp_processing_time = trail.build_the_hdp_model(token_list_after_feature_selection)

trail.evaluate_bar_graph([lda_highest_coherence_score, lsi_highest_coherence_score, hdp_coherence_score], ['LDA', 'LSI', 'HDP'])

best_topic_model, model_name, corpus, dictionary = trail.best_topic_model_selecion(top_lda_model, top_lsi_model, hdp_model,
                                                               lda_highest_coherence_score, lsi_highest_coherence_score, hdp_coherence_score,
                                                               corpus_lda, corpus_lsi, corpus_hdp,
                                                               dictionary_lda, dictionary_lsi, dictionary_hdp)

# ------ clustering based on tweet topic distribution ------ #
tweet_topic_distribution_df = trail.get_tweet_topic_matrix_based_on_best_topic_model(best_topic_model, corpus)
list_k, lable_list, model_list, collect_clustering_info_processing_time = trail.collect_clustering_info(tweet_topic_distribution_df)
tweet_topic_distribution_with_cluster_df, selected_kmeans_model, number_of_cluster, add_clustering_info_to_df_processing_time = trail.add_clustering_info_to_df(tweet_topic_distribution_df, list_k, lable_list, model_list)

# ------ classification ------ #
vectorizer_clf_dict, classifier_building_processing_time = trail.classifier_building(tweet_topic_distribution_with_cluster_df, number_of_cluster, corpus)
Y_test, Y_pred, cluster_label_list, test_data_fit_in_processing_time = trail.test_data_fit_in_model(vectorizer_clf_dict, best_topic_model, dictionary, selected_kmeans_model)
trail.evaluation(Y_test, Y_pred, cluster_label_list)

# ------ baseline clf ------ #
vectorizer, baseline_clf_dict, baseline_classifier_building_processing_time = trail.baseline_model_builder(token_list_after_feature_selection)
Y_test, baseline_result, baseline_test_data_fit_in = trail.baseline_test_data_fit_in_model(vectorizer, baseline_clf_dict)
trail.baseline_evaluation(Y_test, baseline_result)

# http://www.apnorton.com/blog/2016/12/19/Visualizing-Multidimensional-Data-in-Python/
# https://towardsdatascience.com/k-means-clustering-algorithm-applications-evaluation-methods-and-drawbacks-aa03e644b48a

# https://github.com/scipy/scipy/pull/8295/files
# https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
# https://markroxor.github.io/gensim/static/notebooks/gensim_news_classification.html notebook using 4 topic models
# https://medium.com/nanonets/topic-modeling-with-lsa-psla-lda-and-lda2vec-555ff65b0b05 Topic Modeling with LSA, PLSA, LDA & lda2Vec

# https://machinelearningmastery.com/feature-selection-machine-learning-python/ feature selection
# https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/
'''
potential improvements:
0) deal with [] after feature selection (don't need to)
1) add a baseline for comparison before running topic model (done)
2) evaluate the performance for each classifer to see if there is anything we can improve based on the topic model (done, needs to debug)
- add topic model visualization
- add clustering visualization 



- test something new based on the topic models
- add part-of-speech analysis in the dataset part
- something
'''


