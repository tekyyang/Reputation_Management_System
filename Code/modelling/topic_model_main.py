
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

from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

import time
from matplotlib import pyplot as plt
import itertools
import warnings
warnings.filterwarnings("ignore")


class topic_model_builder():

    def __init__(self, training_dataset_posi_paths, training_dataset_nega_paths, test_dataset_posi_path, test_dataset_nega_path, plot_flag=False):
        '''
        :param input_training_data_path (String): the entire path of all the training data's paths. Could be parsed as a list. Separated by ','.
        :param input_test_data_path (String): same as above
        '''
        self.class_startTime = time.time()

        #read training datasets

        with open(training_dataset_posi_paths, 'r') as f:
            lines = f.readlines()  # read all the lines in the file
            self.posi_training_data_list = [[ast.literal_eval(line.replace('\n', '') ),1] for line in lines]
            self.posi_training_data_tokens = [item[0] for item in self.posi_training_data_list]
            self.posi_training_data_labels = [item[1] for item in self.posi_training_data_list]
            tweets_list = [' '.join(l) for l in self.posi_training_data_tokens]
            self.posi_training_data_df = pd.DataFrame(tweets_list, columns=['tweets'])
            self.posi_training_data_df['label'] = 1

        with open(training_dataset_nega_paths, 'r') as f:
            lines = f.readlines()  # read all the lines in the file
            self.nega_training_data_list = [[ast.literal_eval(line.replace('\n', '') ),0] for line in lines]
            self.nega_training_data_tokens = [item[0] for item in self.nega_training_data_list]
            self.nega_training_data_labels = [item[1] for item in self.nega_training_data_list]
            tweets_list = [' '.join(l) for l in self.nega_training_data_tokens]
            self.nega_training_data_df = pd.DataFrame(tweets_list, columns=['tweets'])
            self.nega_training_data_df['label'] = 0

        #read test dataset

        with open(test_dataset_posi_path, 'r') as f:
            lines = f.readlines()  # read all the lines in the file
            self.posi_test_data_list = [[ast.literal_eval(line.replace('\n', '') ),1] for line in lines]
            self.posi_test_data_tokens = [item[0] for item in self.posi_test_data_list]
            self.posi_test_data_labels = [item[1] for item in self.posi_test_data_list]
            tweets_list = [' '.join(l) for l in self.posi_test_data_tokens]
            self.posi_test_data_df = pd.DataFrame(tweets_list, columns=['tweets'])
            self.posi_test_data_df['label'] = 1

        with open(test_dataset_nega_path, 'r') as f:
            lines = f.readlines()  # read all the lines in the file
            self.nega_test_data_list = [[ast.literal_eval(line.replace('\n', '') ),0] for line in lines]
            self.nega_test_data_tokens = [item[0] for item in self.nega_test_data_list]
            self.nega_test_data_labels = [item[1] for item in self.nega_test_data_list]
            tweets_list = [' '.join(l) for l in self.nega_test_data_tokens]
            self.nega_test_data_df = pd.DataFrame(tweets_list, columns=['tweets'])
            self.nega_test_data_df['label'] = 0

        self.plot_flag = plot_flag

    def to_string_list_tool(self, input, tweet_column_name_in_df=None, mode='df_to_token_list'):
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
            tweet_list = input[tweet_column_name_in_df].tolist()
            tweet_list = [sen.split(' ') for sen in tweet_list]
            return tweet_list

        elif mode == 'df_to_string_list':
            tweet_list = input[tweet_column_name_in_df].tolist()
            return tweet_list

        elif mode=='token_list_to_string_list':
            assert isinstance(input, list), 'not a list!'
            tweet_list = [' '.join(l) for l in input]
            return tweet_list

        else:
            print 'Input is not df or list. Can not convert.'

   # --- topic model build start --- #

    def prepare_data_for_topic_modelling(self, no_below=5, no_above=0.4):
        # dictionary defines the token features would be used later;  # Dictionary(16300 unique tokens: [u'jaesuk', u'sermersheim', u'headband', u'degenere', u'jetline']...)
        # corpus is the list that after tweets get truned into token_ids.  #[[(2, 1), (5, 1)],  [(12, 1), (16, 1)], ...]
        training_token_list = self.posi_training_data_tokens + self.nega_training_data_tokens
        dictionary = Dictionary(training_token_list)
        dictionary.filter_extremes(no_below=no_below, no_above=no_above)
        corpus = [dictionary.doc2bow(text) for text in training_token_list]
        return dictionary, corpus, training_token_list

    def build_lda_lsi_model(self, dictionary, corpus, training_token_list, min_topic_num = 3, max_topic_num = 30, model='lda'):
        """
        doc: https://radimrehurek.com/gensim/models/lsimodel.html
        """
        startTime = time.time()

        print '=== (1) Start building topic models! ==='

        if max_topic_num < min_topic_num:
            raise ValueError("Please enter limit > %d. You entered %d" % (min_topic_num, max_topic_num))
        c_v = []
        lm_list = []
        print 'Start building '+model+' model!'
        for num_topics in range(min_topic_num, max_topic_num+1):
            if model=='lda':
                lm = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
            elif model=='lsi':
                lm = LsiModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
            else:
                lm=[]
                print 'please input the correct model name!'

            lm_list.append(lm)
            cm = CoherenceModel(model=lm, texts=training_token_list, dictionary=dictionary, coherence='c_v')
            c_v.append(cm.get_coherence())
            # print '\rfinish building topic = {0}\r'.format(str(num_topics))

        print '=== (1) Finish ' + model + ' graph building! Taking '  + str(round((time.time() - startTime),4)) + 's ===\n'

        lad_lsi_processing_time = round((time.time() - startTime), 4)

        #--- show the graph for one single topic model ---#
        x = range(min_topic_num, max_topic_num+1)
        if self.plot_flag == True:
            plt.plot(x, c_v)
            plt.xlabel("num_topics")
            plt.ylabel("Coherence score")
            plt.legend(("c_v"), loc='best')
            plt.show()

        highest_coherence_score = np.argmax(c_v)
        top_model = lm_list[highest_coherence_score]
        model_topics = top_model.show_topics(formatted=False)

        return top_model, model_topics, highest_coherence_score, dictionary, corpus, lad_lsi_processing_time

    def build_the_hdp_model(self, dictionary, corpus, training_token_list):

        startTime = time.time()

        hdp = HdpModel(corpus=corpus, id2word=dictionary)

        hdp_topics = hdp.show_topics(formatted=False)

        coherence_score = CoherenceModel(model=hdp, texts=training_token_list, dictionary=dictionary, coherence='c_v').get_coherence()

        print '=== (1) Finish hdp graph building! Taking ' + str(round((time.time() - startTime),4)) + 's ===\n'

        building_hdp_processing_time = round((time.time() - startTime), 4)

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

        if self.plot_flag==True:
            plt.bar(x, coherences, width=0.2, tick_label=indices, align='center')
            plt.xlabel('Models')
            plt.ylabel('Coherence Value')
            plt.show()

    def best_topic_model_selecion(self,lda_model, lsi_model, hdp_mode,
                                  lda_highest_coherence_score, lsi_highest_coherence_score, hdp_coherence_score,
                                  corpus, dictionary):

        if lda_highest_coherence_score > hdp_coherence_score and lda_highest_coherence_score > lsi_highest_coherence_score:
            topic_model, topic_model_name = lda_model, 'lda'
        elif lsi_highest_coherence_score > hdp_coherence_score:
            topic_model, topic_model_name = lsi_model, 'lsi'
        else:
            topic_model, topic_model_name = hdp_mode, 'hdp'

        if self.plot_flag==True:
            import pyLDAvis
            vis_data = pyLDAvis.gensim.prepare(topic_model, corpus, dictionary)
            pyLDAvis.display(vis_data)

        return topic_model, topic_model_name, corpus, dictionary

    def data_resampling(self, mode='r_under_s'):
        '''
        To deal with data imbalance issue here.
        DataFrame.sample method to get random samples each class
        :param tweets_to_be_resampled (df): posi and nega tweets df waiting for be resampled
        :return (list): balanced posi and nega datasets list
        '''

        def random_upper_sampling(posi_training_data_df, nega_training_data_df):
            # pandas.DataFrame.sample
            if len(posi_training_data_df) > len(nega_training_data_df):
                posi_training_data_df_under = posi_training_data_df.sample(len(nega_training_data_df))
                return posi_training_data_df_under, nega_training_data_df

            elif len(posi_training_data_df) < len(nega_training_data_df):
                nega_training_data_df_under = nega_training_data_df.sample(len(posi_training_data_df))
                return posi_training_data_df, nega_training_data_df_under

            else:
                return posi_training_data_df, nega_training_data_df

        def random_under_sampling(posi_training_data_df, nega_training_data_df):
            # pandas.DataFrame.sample
            if len(posi_training_data_df) > len(nega_training_data_df):
                nega_training_data_df_upper = nega_training_data_df.sample(len(posi_training_data_df), replace=True)
                return posi_training_data_df, nega_training_data_df_upper

            elif len(posi_training_data_df) < len(nega_training_data_df):
                posi_training_data_df_upper = posi_training_data_df.sample(len(nega_training_data_df), replace=True)
                return posi_training_data_df_upper, nega_training_data_df

            else:
                return posi_training_data_df, nega_training_data_df

        def other_sampling_tech(df_train):
            pass

        def main(posi_training_data_df, nega_training_data_df, mode=mode):
            if mode == 'r_under_s':
                df_training_posi_resampled, df_training_nega_resampled = random_upper_sampling(posi_training_data_df,
                                                                                               nega_training_data_df)
                return df_training_posi_resampled, df_training_nega_resampled
            elif mode == 'r_upper_s':
                df_training_posi_resampled, df_training_nega_resampled = random_under_sampling(posi_training_data_df,
                                                                                               nega_training_data_df)
                return df_training_posi_resampled, df_training_nega_resampled
            else:
                return None, None

        startTime = time.time()
        df_training_posi_resampled, df_training_nega_resampled = main(self.posi_training_data_df,
                                                                      self.nega_training_data_df, mode=mode)
        training_posi_resampled_token_list = self.to_string_list_tool(df_training_posi_resampled, 'tweets')
        training_nega_resampled_token_list = self.to_string_list_tool(df_training_nega_resampled, 'tweets')
        print '=== (4) Finish resampling! Taking ' + str(round((time.time() - startTime), 4)) + 's ==='
        resampling_processing_time = round((time.time() - startTime), 4)
        print 'After resampling, now there are ' + str(
            len(training_posi_resampled_token_list)) + ' positive tweets and ' + str(
            len(training_nega_resampled_token_list)) + ' negative tweets!\n'
        return training_posi_resampled_token_list, training_nega_resampled_token_list, resampling_processing_time

    def apply_the_best_topic_model_on_tweets_to_get_tweet_topic_matrix(self, best_topic_model, dictionary,  training_posi_token_list, training_nega_token_list):
        #https://groups.google.com/forum/#!topic/gensim/F4AWfh9yIhM

        best_topic_model.minimum_probability = 0.0 #prob smaller than this would be filtered
        doc_topic_collection_list = []
        training_data_token = training_posi_token_list + training_nega_token_list
        training_data_corpus = [dictionary.doc2bow(tweet) for tweet in training_data_token]

        for i in range(len(training_data_corpus)):
            doc_topic_collection_list.append([prob_tuple[1] for prob_tuple in best_topic_model[training_data_corpus[i]]])

        tweet_topic_distribution_df = pd.DataFrame.from_records(doc_topic_collection_list)
        return tweet_topic_distribution_df

    def collect_clustering_info(self, tweet_topic_distribution_df, min_cluster_number = 1, max_cluster_number = 30):

        startTime = time.time()
        print '=== (2) Start collecting clustering info... ===\n'

        # Run the Kmeans algorithm and get the index of data points clusters
        inertia_list = []
        lable_list = []
        model_list =[]
        list_k = list(range(min_cluster_number, max_cluster_number+1))

        tweet_topic_distribution_df = tweet_topic_distribution_df.fillna(value = 0)

        for k in list_k:
            km = KMeans(n_clusters=k)
            km.fit(tweet_topic_distribution_df)
            inertia_list.append(km.inertia_)
            lable_list.append(km.labels_.tolist())
            model_list.append(km)

        if self.plot_flag == True:
            # Plot sse against k
            plt.figure(figsize=(6, 6))
            plt.plot(list_k, inertia_list, '-o')
            plt.xlabel(r'Number of clusters *k*')
            plt.ylabel('Sum of squared distance')

        print '=== (2) Finish clustering graph building! Taking '  + str(round((time.time() - startTime),4)) + 's ===\n'

        collect_clustering_info_processing_time = round((time.time() - startTime), 4)

        return list_k, lable_list, model_list, collect_clustering_info_processing_time

    def add_clustering_info_to_df(self, tweet_topic_distribution_df, list_k, lable_list, model_list, number_of_cluster = 6):

        startTime = time.time()

        index_n = list_k.index(number_of_cluster)
        selected_clustering_labels = lable_list[index_n]
        selected_kmeans_model = model_list[index_n]

        tweet_topic_distribution_with_cluster_df = tweet_topic_distribution_df.copy()
        tweet_topic_distribution_with_cluster_df['clustering_labels'] = selected_clustering_labels
        tweet_topic_distribution_with_cluster_df['Y'] = np.where(tweet_topic_distribution_df.index <= tweet_topic_distribution_df.shape[0] / 2 - 1, 1, 0)
        # tweet_topic_distribution_with_cluster_df['Y'] = [0]* len(self.posi_training_data_df) + [1]*len(self.nega_training_data_df)
        print '=== (3) Finish adding clustering into to df! Taking '  + str(round((time.time() - startTime),4)) + 's ===\n'
        print 'number of cluster is '+str(number_of_cluster) + '!'
        add_clustering_info_to_df_processing_time = round((time.time() - startTime), 4)

        return tweet_topic_distribution_with_cluster_df, selected_kmeans_model, number_of_cluster, add_clustering_info_to_df_processing_time

    # --- classifier building start --- #

    def ngram_extactor(self, posi_training_token_list, nega_training_token_list, bigram_min_count=20, threshold=10.0, mode='uni_and_bigram'):
        '''
        :param  posi_training_token_list,
                nega_training_token_list,
                bigram_min_count: Ignore all words and bigrams with total collected count lower than this value,
                threshold : Represent a score threshold for forming the phrases (higher means fewer phrases);
                            A phrase of words `a` followed by `b` is accepted if the score of the phrase is greater than threshold;
                            Heavily depends on concrete scoring-function, see the `scoring` parameter.
        :return: posi_unigran_bigram_training_token_list, posi_unigran_bigram_training_token_list (mode='uni_and_bigram');
                 or posi_training_token_list, nega_training_token_list (mode='unigram')
        '''
        startTime = time.time()
        training_list = posi_training_token_list + nega_training_token_list

        # unigram does't need to be processed

        # bigram extraction starts
        bigram = Phrases(training_list, min_count=bigram_min_count,
                         threshold=threshold)  # training list should be token list

        # if a bigram is captured, the original unigram will not be returned.
        # E.g. ['the','states'] --> ['the_states']
        # NOT ['the','states'] -/-> ['the','states','the_states']
        posi_bigram_training_token_list_with_unigram = [bigram[sent] for sent in posi_training_token_list]
        nega_bigram_training_token_list_with_unigram = [bigram[sent] for sent in nega_training_token_list]

        # picking only the bigrams ['the_states']
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
        # we get ['the','states','the_states'] from this
        posi_unigran_bigram_training_token_list = []
        for i in range(len(posi_bigram_training_token_list)):
            posi_unigran_bigram_training_token_list.append(
                posi_training_token_list[i] + posi_bigram_training_token_list[i])

        nega_unigran_bigram_training_token_list = []
        for i in range(len(nega_bigram_training_token_list)):
            nega_unigran_bigram_training_token_list.append(
                nega_training_token_list[i] + nega_bigram_training_token_list[i])

        print '=== (5) Finish bigram extraction! Mode is: ' + mode + '. Taking ' + str(
            round((time.time() - startTime), 4)) + 's ===\n'

        feature_extraction_processing_time = round((time.time() - startTime), 4)

        if mode == 'uni_and_bigram':
            return posi_unigran_bigram_training_token_list, nega_unigran_bigram_training_token_list, feature_extraction_processing_time
        elif mode == 'unigram':
            return posi_training_token_list, nega_training_token_list, feature_extraction_processing_time
        else:
            print 'there is no such mode: ' + mode + '!'

    def feature_selection(self, posi_training_token_list, nega_training_token_list, feature_represent_mode='tfidf',
                          feature_selection_mode='chi2', top_n_feature=5000):
        '''
        :returns
        X_chi_matrix: the weighted matrix after feature selection
        feature_name_list_after_chi: a list of feature names after feature selection
        token_list_after_chi2: referred by name. Notice that it doesn't reduce the amount of tweets.
        So positive vs negative tweets amounts are still 50:50
        '''
        startTime = time.time()
        posi_training_string_list = self.to_string_list_tool(posi_training_token_list, mode='token_list_to_string_list')
        nega_training_string_list = self.to_string_list_tool(nega_training_token_list, mode='token_list_to_string_list')

        X_train_string_list = posi_training_string_list + nega_training_string_list
        Y_train_list = [1] * len(posi_training_string_list) + [0] * len(nega_training_string_list)
        X_train_df = pd.DataFrame(X_train_string_list, columns=['text'])

        # --- feature representation from tfidf or word count ---#
        if feature_represent_mode == 'tfidf':
            # tf-idf vectorizer
            vectorizer = TfidfVectorizer()
            X_matrix = vectorizer.fit_transform(X_train_df['text'])

        elif feature_represent_mode == 'word_count':
            # word count vectorizer
            vectorizer = CountVectorizer()
            X_matrix = vectorizer.fit_transform(X_train_df['text'])

        else:
            vectorizer = []
            X_matrix = []
            print 'Please input the right mode!'

        print '=== (6) Start feature selection! ==='
        print 'X_matrix.shape is (number of rows, number of columns):'
        print X_matrix.shape
        print 'Taking the top ' + str(top_n_feature) + ' features...'

        # --- feature selection from chi2 or mutual info ---#
        if feature_selection_mode == 'chi2':

            # ch2 selector
            ch2 = SelectKBest(chi2, k=top_n_feature)

            # get the feature matrix
            X_chi_matrix = ch2.fit_transform(X_matrix.toarray(), np.asarray(Y_train_list))
            '''
            [[0.         0.46979139 0.         0.         0.        ]
             [0.         0.6876236  0.         0.53864762 0.        ]
             [0.51184851 0.         0.51184851 0.         0.51184851]
             [0.         0.46979139 0.         0.         0.        ]]
            '''

            print '=== (3) Finish feature selection! Feature representation mode is' + feature_represent_mode + '. Feature selection mode is ' + feature_selection_mode + '. Taking ' + str(
                round((time.time() - startTime), 4)) + 's ==='
            feature_selection_processing_time = round((time.time() - startTime), 4)

            return X_chi_matrix, vectorizer, ch2, feature_selection_processing_time

        else:
            return [], [], [], [], ''

    def lexicon_feature_prep(self):
        path = '/Users/yibingyang/Documents/thesis_project_new/Lexicon/SentiWordNet_3.0_clean.txt'
        with open(path, 'r') as f:
            lines = f.readlines() # read all the lines in

        # separate words apart based on \t
        lexicon_list = [line.strip().split('\t') for line in lines]

        # make the vertical words cluster horizonal
        split_list = []
        for record in lexicon_list:
            try:
                if len(record[4].split(' ')) == 1:
                    split_list.append(record)
                else:
                    for j in range(len(record[4].split(' '))):
                        split_list.append(
                            [record[0], record[1], record[2], record[3], record[4].split(' ')[j], record[5]])
            except:
                continue

        # get rid of the '#'
        split_list = [[record[0], record[1], record[2], record[3], record[4].split('#')[0], record[5]] for record in
                      split_list]

        # create dataframe
        df = pd.DataFrame(split_list, columns=['POS', 'ID', 'PosScore', 'NegScore', 'SynsetTerms', 'Gross'])
        df[['PosScore', 'NegScore']] = df[['PosScore', 'NegScore']].astype(np.float)
        df['NeuScore'] = 1 - df['PosScore'] - df['NegScore']
        # ps = PorterStemmer()
        # df['StemWord'] = [ps.stem(word) for word in df['SynsetTerms'].tolist()]
        wl = WordNetLemmatizer()
        df['LemmWord'] = [wl.lemmatize(word) for word in df['SynsetTerms'].tolist()]

        lexicon_df = df.groupby('LemmWord').agg({'PosScore': 'mean', 'NegScore': 'mean', 'NeuScore': 'mean'})

        return lexicon_df

    def build_all_features(self, posi_unigran_bigram_training_token_list, nega_unigran_bigram_training_token_list,
                           lexicon_df, read_path_from_file):
        import os
        if os.stat(read_path_from_file).st_size == 0:

            training_token_list = posi_unigran_bigram_training_token_list + nega_unigran_bigram_training_token_list

            # --- building lexicon features --- #
            LemmaWords = lexicon_df.index.tolist()
            score_list = []
            # tweet level
            for tweet in training_token_list:
                temp_list = [0, 0, 0, 0]  # for posi score, nega score, posi count, nega count
                overall_tweet_length = len(tweet)
                for word in tweet:
                    if word in LemmaWords:
                        posi_score = lexicon_df[lexicon_df.index == word]['PosScore'].tolist()[0]
                        nega_score = lexicon_df[lexicon_df.index == word]['NegScore'].tolist()[0]
                        temp_list[0] = temp_list[0] + posi_score
                        temp_list[1] = temp_list[1] + nega_score
                        temp_list[2] = (temp_list[2] + 1) if posi_score >= 0.3 else (temp_list[2] + 0)
                        temp_list[3] = (temp_list[3] + 1) if nega_score >= 0.3 else (temp_list[3] + 0)
                        # print word, posi_score, temp_list[2],temp_list[3], overall_tweet_length

                score_list.append({
                    # based on score
                    # --- sum of sentiment score
                    # --- sum of positive sentiment score
                    # --- sum of negative sentiment score
                    # --- the ratio of positive score to negative score
                    # based on count
                    # --- the ratio of positive words to all words; (score > 0.5)
                    # --- the ratio of positive words to all words; (score > 0.5)
                    'sum_senti_score': temp_list[0] - temp_list[1],
                    'sum_senti_pos_score': temp_list[0],
                    'sum_senti_neg_score': temp_list[1],
                    'pos_neg_score_ratio': (temp_list[0] / temp_list[1]) if temp_list[1] <> 0 else 0,
                    'pos_word_ratio': (
                                float(temp_list[2]) / float(overall_tweet_length)) if overall_tweet_length <> 0 else 0,
                    'neg_word_ratio': (
                                float(temp_list[3]) / float(overall_tweet_length)) if overall_tweet_length <> 0 else 0
                })

            lexicon_feature_df = pd.DataFrame.from_records(score_list)
            print 'finish lexicon features!'

            # --- building negation feature --- #
            negation_list = ['no', 'not', 'never', 'nobody', 'none', 'neither', 'nothing', 'nowhere', 'barely', 'rarely',
                             'seldom', 'hardly', 'scarcely']
            wl = WordNetLemmatizer()
            negation_lemma_list = [wl.lemmatize(word) for word in negation_list]
            negation_count_list = []
            for tweet in training_token_list:
                temp_score = 0
                for word in tweet:
                    if word in negation_lemma_list:
                        temp_score = temp_score + 1
                negation_count_list.append(temp_score % 2)  # the number of negaiton is odd or even
            print 'finish negation features!'
            # negation_count_list

            # --- building POS feature --- #
            tagged_features_list = []
            for tweet in training_token_list:
                tweet_length = len(tweet)
                try:
                    tagged_list = pos_tag(tweet)
                    temp_dict = {'noun': 0, 'verb': 0, 'adj': 0, 'adv': 0, 'pronoun': 0, 'other': 0, 'tweet_length': tweet_length}
                    for word_tuple in tagged_list:
                        word_tag = word_tuple[1]
                        if word_tag in ('NN', 'NNS', 'NNP', 'NNPS'):
                            temp_dict['noun'] += 1
                        elif word_tag in ('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'):
                            temp_dict['verb'] += 1
                        elif word_tag in ('JJ', 'JJR', 'JJS'):
                            temp_dict['adj'] += 1
                        elif word_tag in ('RB', 'RBR', 'RBS'):
                            temp_dict['adv'] += 1
                        elif word_tag in ('PRP', 'PRP$', 'WP', 'WP$'):
                            temp_dict['pronoun'] += 1
                        else:
                            temp_dict['other'] += 1
                except:
                    temp_dict = {'noun': 0, 'verb': 0, 'adj': 0, 'adv': 0, 'pronoun': 0, 'other': 0, 'tweet_length': 0}
                tagged_features_list.append(temp_dict)

            pos_feature_df = pd.DataFrame.from_records(tagged_features_list)
            pos_feature_df['adj_ratio'] = pos_feature_df['adj'] / pos_feature_df['tweet_length']
            pos_feature_df['adv_ratio'] = pos_feature_df['adv'] / pos_feature_df['tweet_length']
            pos_feature_df['noun_ratio'] = pos_feature_df['noun'] / pos_feature_df['tweet_length']
            pos_feature_df['pronoun_ratio'] = pos_feature_df['pronoun'] / pos_feature_df['tweet_length']
            pos_feature_df['verb_ratio'] = pos_feature_df['verb'] / pos_feature_df['tweet_length']
            pos_feature_df['other_ratio'] = pos_feature_df['other'] / pos_feature_df['tweet_length']
            pos_feature_df = pos_feature_df[
                ['adj_ratio', 'adv_ratio', 'noun_ratio', 'pronoun_ratio', 'verb_ratio', 'other_ratio']]
            print 'finish pos features!'

            return lexicon_feature_df, negation_count_list, pos_feature_df

        else:
            #write something like readline from file
            return [],[],[]

    def feature_combination(self, X_chi_matrix, lexicon_feature_df, negation_feature_df, pos_feature_df, mode):

        lexicon_feature_matrix = lexicon_feature_df[['sum_senti_score', 'sum_senti_pos_score', 'sum_senti_neg_score', 'pos_neg_score_ratio', 'pos_word_ratio','neg_word_ratio']].values
        negation_count_matrix = negation_feature_df[['negation_count']].values
        pos_feature_matrix = pos_feature_df[['adj_ratio', 'adv_ratio', 'noun_ratio', 'pronoun_ratio', 'verb_ratio', 'other_ratio']].values

        if mode == 'ngram_and_negation':
            X_train_matrix = np.concatenate((X_chi_matrix, negation_count_matrix), axis=1)
        elif mode == 'ngram_and_pos':
            X_train_matrix = np.concatenate((X_chi_matrix, pos_feature_matrix), axis=1)
        elif mode == 'ngram_and_lexicon':
            X_train_matrix = np.concatenate((X_chi_matrix, lexicon_feature_matrix), axis=1)
        elif mode == 'all_features':
            X_train_matrix = np.concatenate((X_chi_matrix, lexicon_feature_matrix), axis=1)
            X_train_matrix = np.concatenate((X_train_matrix, negation_count_matrix), axis=1)
            X_train_matrix = np.concatenate((X_train_matrix, pos_feature_matrix), axis=1)
        else:
            print 'not the right mode!'
            X_train_matrix = []

        return X_train_matrix

    def classifier_building(self, tweet_topic_distribution_with_cluster_df, number_of_cluster, X_train_matrix, classifier = 'logistic_regression'):
        # for tweets in each cluster label, build a classifier for it
        # return the classifier

        startTime = time.time()

        # --- build classifiers based on clusters --- #
        clf_dict = {}
        for i in range(0, number_of_cluster):
            # index_list is a list of index for a certain cluster
            # Y_list is a list of Y labels for a certain cluster
            index_list = tweet_topic_distribution_with_cluster_df[tweet_topic_distribution_with_cluster_df['clustering_labels'] == i].index.tolist()
            Y_list =     tweet_topic_distribution_with_cluster_df.Y[tweet_topic_distribution_with_cluster_df['clustering_labels'] == i].tolist()

            #take X_train and Y_train for each cluster
            X_train_for_selected_cluster = []
            for k in index_list: #index list for a cluster
                X_train_for_selected_cluster.append(X_train_matrix[k])
            Y_train = Y_list

            if classifier == 'logistic_regression':
                clf = linear_model.LogisticRegression()
            elif classifier == 'naive_bayes':
                clf = naive_bayes.MultinomialNB()
            else:
                clf = svm.SVC(kernel='linear', gamma='auto')

            clf.fit(X_train_for_selected_cluster, Y_train)
            clf_dict.update({i: clf})

        print '=== (7) Finish classifier building! Taking ' + str(round((time.time() - startTime), 4)) + 's ===\n'

        classifier_building_processing_time = round((time.time() - startTime), 4)

        return clf_dict, classifier_building_processing_time

    def test_data_fit_in_model(self, clf_dict, vectorizer, ch2, best_topic_model, dictionary, selected_kmeans_model, feature_mode):

        startTime = time.time()
        # add tracking number for test data
        # posi
        posi_test_data_tokens_with_index = []
        for i in range(len(self.posi_test_data_tokens)):
            posi_test_data_tokens_with_index.append((i, self.posi_test_data_tokens[i]))
        # nega
        nega_test_data_tokens_with_index = []
        for i in range(len(self.nega_test_data_tokens)):
            nega_test_data_tokens_with_index.append((i, self.nega_test_data_tokens[i]))

        # create corpus for test dataset based on the dictionary of the chosen topic model
        # posi
        test_posi_corpus, chosen_posi_index, unchosen_posi_index = [] ,[], []
        for item in posi_test_data_tokens_with_index:
            if dictionary.doc2bow(item[1]) <> []:
                try:
                    test_posi_corpus.append(dictionary.doc2bow(item[1]))
                    chosen_posi_index.append(item[0])
                except:
                    unchosen_posi_index.append(item[0])
            else:
                unchosen_posi_index.append(item[0])

        # nega
        test_nega_corpus, chosen_nega_index, unchosen_nega_index = [] ,[], []
        for item in nega_test_data_tokens_with_index:
            if dictionary.doc2bow(item[1]) <> []:
                try:
                    test_nega_corpus.append(dictionary.doc2bow(item[1]))
                    chosen_nega_index.append(item[0])
                except:
                    unchosen_nega_index.append(item[0])
            else:
                unchosen_nega_index.append(item[0])

        test_corpus = test_posi_corpus + test_nega_corpus

        # for each test tweet, use the best topic model we built to get the tweet, topic distribution
        # based on the distribution, find the corresponding cluster for it
        best_topic_model.minimum_probability = 0.0  # prob smaller than this would be filtered
        cluster_label_list = []  # generate label list for all test tweets
        for i in range(len(test_corpus)):
            test_tweet_prob_distribution = [prob_tuple[1] for prob_tuple in best_topic_model[test_corpus[i]]]
            cluster_label = selected_kmeans_model.predict([test_tweet_prob_distribution]).tolist()[0]
            cluster_label_list.append(cluster_label)

        X_test_posi_df = self.posi_test_data_df[self.posi_test_data_df.index.isin(chosen_posi_index)]
        X_test_nega_df = self.nega_test_data_df[self.nega_test_data_df.index.isin(chosen_nega_index)]

        X_test_df = pd.concat([X_test_posi_df, X_test_nega_df]).reset_index()
        X_test_df['cluster_label'] = cluster_label_list

        # --- test feature metrics --- #
        lexicon_feature_df = pd.read_csv('/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/intermediate/test_lexicon_features.csv',sep='\t')
        negation_feature_df = pd.read_csv('/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/intermediate/test_negation_features.csv',sep='\t')
        pos_feature_df = pd.read_csv('/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/intermediate/test_pos_features.csv', sep='\t')

        if feature_mode == 'ngram_and_negation':
            X_test_df = X_test_df.join(negation_feature_df,lsuffix='_left',rsuffix='_right')
        elif feature_mode == 'ngram_and_pos':
            X_test_df = X_test_df.join(pos_feature_df,lsuffix='_left',rsuffix='_right')
        elif feature_mode == 'ngram_and_lexicon':
            X_test_df = X_test_df.join(lexicon_feature_df,lsuffix='_left',rsuffix='_right')
        elif feature_mode == 'all_features':
           X_test_df = X_test_df.join(lexicon_feature_df,lsuffix='_left', rsuffix='_right').join(negation_feature_df,lsuffix='_left', rsuffix='_right').join(pos_feature_df,lsuffix='_left', rsuffix='_right')
        else:
            print 'this is not the supported feature mode for test data!'
            X_test_df = []

        restructured_X_test_df = None
        # apply the certian classifier on the test tweet
        for i in set(cluster_label_list):
            for j in clf_dict.keys():
                if i == j:
                    # select the corresponding vectorizer and clf
                    # vectorizer = vectorizer_clf_dict[j][0]  # vectorizer
                    clf = clf_dict[j]# clf

                    selected_cluster_piece_df = X_test_df[X_test_df['cluster_label'] == i]
                    selected_cluster_X_test = selected_cluster_piece_df['tweets'].tolist()

                    X_test_ngram = vectorizer.transform(selected_cluster_X_test).toarray()
                    X_test_matrix = ch2.transform(X_test_ngram)

                    if feature_mode == 'all_features':
                        lexicon_feature_matrix = selected_cluster_piece_df[['sum_senti_score', 'sum_senti_pos_score', 'sum_senti_neg_score', 'pos_neg_score_ratio', 'pos_word_ratio', 'neg_word_ratio']].values
                        negation_count_matrix = selected_cluster_piece_df[['negation_count']].values
                        pos_feature_matrix = selected_cluster_piece_df[['adj_ratio', 'adv_ratio', 'noun_ratio', 'pronoun_ratio', 'verb_ratio', 'other_ratio']].values

                        X_test_matrix = np.concatenate((X_test_matrix, lexicon_feature_matrix), axis=1)
                        X_test_matrix = np.concatenate((X_test_matrix, negation_count_matrix), axis=1)
                        X_test_matrix = np.concatenate((X_test_matrix, pos_feature_matrix), axis=1)

                    y_pred = clf.predict(X_test_matrix)
                    selected_cluster_piece_df['y_pred'] = y_pred
                    restructured_X_test_df = pd.concat([restructured_X_test_df,selected_cluster_piece_df]) if restructured_X_test_df is not None else selected_cluster_piece_df
                    print 'finish cluster ' + str(i)

        print '=== (8) Finish test data fit in! Taking ' + str(round((time.time() - startTime), 4)) + 's ===\n'

        test_data_fit_in_processing_time = round((time.time() - startTime), 4)

        print '=== The overall program taking ' + str(round((time.time() - self.class_startTime), 4)) + 's! ===\n'

        return restructured_X_test_df, cluster_label_list, test_data_fit_in_processing_time

    def evaluation(self,restructured_X_test_df):
        print "=== (9) Performance evaluation moment! ==="
        Y_test = restructured_X_test_df['label'].tolist()
        Y_pred = restructured_X_test_df['y_pred'].tolist()
        cluster_label_list = restructured_X_test_df['cluster_label'].tolist()

        print "Overall Performance - confusion matrix:"
        cm = confusion_matrix(Y_test, Y_pred)
        print cm
        np.set_printoptions(precision=2)
        plt.figure()
        self.plot_confusion_matrix(cm, classes=[1, 0], title='Confusion matrix')

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
            self.plot_confusion_matrix(cm, classes=[1, 0], title='Confusion matrix for group ' + str(i))

            print 'For group ' + str(i) + ' - classification report:'
            print(classification_report(group_Y_test, group_Y_pred))

            print 'For group ' + str(i) + ' - accuracy score:'
            print(accuracy_score(group_Y_test, group_Y_pred))

        return cm, classification_report(group_Y_test, group_Y_pred), accuracy_score(group_Y_test, group_Y_pred)

    # --- baseline classifier building start --- #

    def baseline_model_builder(self, X_chi_matrix):

        startTime = time.time()

        Y_train = [1] * (len(X_chi_matrix)/2) + [0] * (len(X_chi_matrix)/2)
        X_train = X_chi_matrix

        # a list of baseline classifiers
        lg_clf = linear_model.LogisticRegression()
        nb_clf = naive_bayes.MultinomialNB()
        # rf_clf = ensemble.RandomForestClassifier()
        # b_clf = xgboost.XGBClassifier()

        # fit the training dataset on the classifier

        lg_clf.fit(X_train, Y_train)
        nb_clf.fit(X_train, Y_train)
        # rf_clf.fit(X_train, Y_train)
        # b_clf.fit(X_train, Y_train)

        print '=== (10) Finish baseline classifier building! Taking ' + str(round((time.time() - startTime), 4)) + 's ===\n'
        baseline_classifier_building_processing_time = round((time.time() - startTime), 4)

        return {
                    'logistic_regression':lg_clf,
                    'naive_bayes':nb_clf,
                    # 'svm': svm_clf,
                    # 'random_forest':rf_clf,
                    # 'boosting':b_clf
                }, \
               baseline_classifier_building_processing_time

    def baseline_test_data_fit_in_model(self, vectorizer, ch2, baseline_clf_dict, restructured_X_test_df):

        startTime = time.time()

        X_test_pre = restructured_X_test_df['tweets'].tolist()

        X_test = vectorizer.transform(X_test_pre)
        X_test = ch2.transform(X_test)

        baseline_clf_name_list = baseline_clf_dict.keys()
        for clf_name, clf in baseline_clf_dict.iteritems():
            y_pred = clf.predict(X_test).tolist()
            restructured_X_test_df[clf_name] = y_pred

        print '=== (11) Finish baseline test data fit in! Taking ' + str(round((time.time() - startTime), 4)) + 's ===\n'
        baseline_test_data_fit_in_processing_time = round((time.time() - startTime), 4)

        return restructured_X_test_df, baseline_clf_name_list, baseline_test_data_fit_in_processing_time

    def baseline_evaluation(self, restructured_X_test_df, baseline_clf_name_list):

        print "=== (12) Baseline performance evaluation! ==="

        evaluation_dict = {}
        for clf_name in baseline_clf_name_list:
            Y_test = restructured_X_test_df['label']
            Y_pred = restructured_X_test_df[clf_name]

            print clf_name + " - confusion matrix:"
            cm = confusion_matrix(Y_test, Y_pred)
            np.set_printoptions(precision=2)
            plt.figure()
            self.plot_confusion_matrix(cm, classes=[1, 0],
                                  title='Confusion matrix')

            print clf_name + " - classification report:"
            print(classification_report(Y_test, Y_pred))

            print clf_name + " - accuracy score:"
            print(accuracy_score(Y_test, Y_pred))

            evaluation_dict.update({clf_name: [cm, classification_report(Y_test, Y_pred), accuracy_score(Y_test, Y_pred)]})
        return evaluation_dict


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

    def show_sample_tweets(self, restructured_X_test_df, cluster_label_list, head=15):

        cluster_index_set = set(cluster_label_list)
        for i in cluster_index_set:
            piece = restructured_X_test_df[restructured_X_test_df['cluster_label']==i]
            piece_correct = piece[piece['label'] == piece['y_pred']].head(head)
            piece_wrong = piece[piece['label'] <> piece['y_pred']].head(head)
            final_piece = pd.concat([piece_correct, piece_wrong]).reset_index()
            print 'sample tweets for cluster '+str(i)+' :'
            print final_piece[['tweets','label','y_pred','logistic_regression','naive_bayes','cluster_label']]

    def main(self,
             no_below=5, no_above=0.4,
             lda_min_topic_num=3, lda_max_topic_num=30,
             lsi_min_topic_num=3, lsi_max_topic_num=30,
             min_cluster_number=2, max_cluster_number=10, number_of_cluster=6, # don't need to revise this line and above
             resampling_mode='r_upper_s',
             feature_extraction_mode = 'uni_and_bigram', bigram_min_count=10, #uni_and_bigram; unigram
             feature_represent_mode='tfidf',feature_selection_mode='chi2', top_n_feature=20000,
             classifier = 'logistic_regression',
             show_sample_tweets_head=15,
             feature_mode = 'all_features'):

        startTime = time.time()

        # ------ build the three topic models: lda, lsi, hdp ------ #
        dictionary, corpus, training_token_list = self.prepare_data_for_topic_modelling(no_below=no_below, no_above=no_above)
        top_lda_model, top_lda_model_topics, lda_highest_coherence_score, dictionary_lda, corpus_lda, lda_processing_time = self.build_lda_lsi_model(dictionary, corpus, training_token_list, min_topic_num=lda_min_topic_num, max_topic_num=lda_max_topic_num, model='lda')
        top_lsi_model, top_lsi_model_topics, lsi_highest_coherence_score, dictionary_lsi, corpus_lsi, lsi_processing_time = self.build_lda_lsi_model(dictionary, corpus, training_token_list, min_topic_num=lsi_min_topic_num, max_topic_num=lsi_max_topic_num, model='lsi')
        hdp_model,     hdp_topics,           hdp_coherence_score,         dictionary_hdp, corpus_hdp, hdp_processing_time = self.build_the_hdp_model(dictionary, corpus, training_token_list)

        self.evaluate_bar_graph([lda_highest_coherence_score, lsi_highest_coherence_score, hdp_coherence_score], ['LDA', 'LSI', 'HDP'])

        best_topic_model, model_name, corpus, dictionary = self.best_topic_model_selecion(top_lda_model, top_lsi_model, hdp_model,
                                                                       lda_highest_coherence_score, lsi_highest_coherence_score, hdp_coherence_score,
                                                                                          corpus, dictionary)
        # ------ clustering based on tweet topic distribution ------ #
        training_posi_resampled_token_list, training_nega_resampled_token_list, resampling_processing_time = self.data_resampling(mode = resampling_mode)
        tweet_topic_distribution_df = self.apply_the_best_topic_model_on_tweets_to_get_tweet_topic_matrix(best_topic_model, dictionary, training_posi_resampled_token_list, training_nega_resampled_token_list)
        list_k, lable_list, model_list, collect_clustering_info_processing_time = self.collect_clustering_info(tweet_topic_distribution_df, min_cluster_number = min_cluster_number, max_cluster_number = max_cluster_number)
        tweet_topic_distribution_with_cluster_df, selected_kmeans_model, number_of_cluster, add_clustering_info_to_df_processing_time = self.add_clustering_info_to_df(tweet_topic_distribution_df, list_k, lable_list, model_list, number_of_cluster=number_of_cluster)

        # ------ classification ------ #
        posi_bigram_training_token_list, nega_bigram_training_token_list, feature_extraction_processing_time = self.ngram_extactor(training_posi_resampled_token_list, training_nega_resampled_token_list, mode=feature_extraction_mode, bigram_min_count=bigram_min_count)
        X_chi_matrix, vectorizer, ch2, feature_selection_processing_time = self.feature_selection(posi_bigram_training_token_list, nega_bigram_training_token_list, feature_represent_mode=feature_represent_mode, feature_selection_mode=feature_selection_mode, top_n_feature=top_n_feature)

        lexicon_feature_df = pd.read_csv('/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/intermediate/lexicon_features.csv', sep='\t')
        negation_feature_df = pd.read_csv('/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/intermediate/negation_features.csv', sep='\t')
        pos_feature_df = pd.read_csv('/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/intermediate/pos_features.csv', sep='\t')

        if feature_mode == 'ngram':
            X_train = X_chi_matrix
        elif feature_mode == 'ngram_and_negation':
            X_train = self.feature_combination(X_chi_matrix, lexicon_feature_df, negation_feature_df, pos_feature_df, mode=feature_mode)
            X_train = np.nan_to_num(X_train)
        elif feature_mode == 'ngram_and_pos':
            X_train = self.feature_combination(X_chi_matrix, lexicon_feature_df, negation_feature_df, pos_feature_df, mode=feature_mode)
            X_train = np.nan_to_num(X_train)
        elif feature_mode == 'ngram_and_lexicon':
            X_train = self.feature_combination(X_chi_matrix, lexicon_feature_df, negation_feature_df, pos_feature_df, mode=feature_mode)
            X_train = np.nan_to_num(X_train)
        elif feature_mode == 'all_features':
            X_train = self.feature_combination(X_chi_matrix, lexicon_feature_df, negation_feature_df, pos_feature_df, mode=feature_mode)
            X_train = np.nan_to_num(X_train)
        else:
            print 'please input the right mode!'
            # lexicon_df = self.lexicon_feature_prep()
            # lexicon_feature_df, negation_count_list, pos_feature_df = self.build_all_features(posi_bigram_training_token_list, nega_bigram_training_token_list, lexicon_df, read_path_from_file='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/intermediate/extended_features.txt')
            # X_train = self.feature_combination(X_chi_matrix, lexicon_feature_df, negation_count_list, pos_feature_df)
            X_train = []

        clf_dict, classifier_building_processing_time = self.classifier_building(tweet_topic_distribution_with_cluster_df, number_of_cluster, X_train, classifier = classifier)

        # ---------- test data ----------#
        restructured_X_test_df, cluster_label_list, test_data_fit_in_processing_time = self.test_data_fit_in_model(clf_dict, vectorizer, ch2, best_topic_model, dictionary, selected_kmeans_model, feature_mode)
        self.evaluation(restructured_X_test_df)

        # ------ baseline clf ------ #
        # taking X_chi_matrix, vectorizer from feature selection part
        baseline_clf_dict, baseline_classifier_building_processing_time = self.baseline_model_builder(X_chi_matrix)
        Y_test, baseline_result, baseline_test_data_fit_in_processing_time = self.baseline_test_data_fit_in_model(vectorizer, ch2, baseline_clf_dict, restructured_X_test_df)

        self.show_sample_tweets(restructured_X_test_df, cluster_label_list, head=show_sample_tweets_head)
        self.baseline_evaluation(Y_test, baseline_result)

        print 'Program running time: '+str(round((time.time() - startTime), 4))
        overall_time =  feature_extraction_processing_time + feature_selection_processing_time+\
                                              lda_processing_time + lsi_processing_time + hdp_processing_time+\
                                              collect_clustering_info_processing_time+add_clustering_info_to_df_processing_time+\
                                              classifier_building_processing_time+test_data_fit_in_processing_time+\
                                              baseline_classifier_building_processing_time+baseline_test_data_fit_in_processing_time
        print 'Overall processing time: '+str(overall_time)
        print '------- time for every step --------'
        # print 'resampling_processing_time: '+str(resampling_processing_time)
        print 'feature_extraction_processing_time: '+str(feature_extraction_processing_time)
        print 'feature_selection_processing_time: '+str(feature_selection_processing_time)
        print 'lda_processing_time :' + str(lda_processing_time)
        print 'lsi_processing_time :' + str(lsi_processing_time)
        print 'hdp_processing_time :' + str(hdp_processing_time)
        print 'collect_clustering_info_processing_time: '+str(collect_clustering_info_processing_time)
        print 'add_clustering_info_to_df_processing_time: '+str(add_clustering_info_to_df_processing_time)
        print 'classifier_building_processing_time: '+str(classifier_building_processing_time)
        print 'test_data_fit_in_processing_time: '+str(test_data_fit_in_processing_time)
        print 'baseline_classifier_building_processing_time: '+str(baseline_classifier_building_processing_time)
        print 'baseline_test_data_fit_in_processing_time: '+str(baseline_classifier_building_processing_time)


trail = topic_model_builder(
    training_dataset_posi_paths='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/positive_tweets_after_preprocessing_lemma_0407.txt',
    training_dataset_nega_paths='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/negative_tweets_after_preprocessing_lemma_0407.txt',
    # test_dataset_posi_path='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/positive_test_tweets_after_preprocessing.txt',
    # test_dataset_nega_path='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/negative_test_tweets_after_preprocessing.txt',
    test_dataset_posi_path='/Users/yibingyang/Documents/thesis_project_new/Data/E-Commerce/after_preprocessing/yelp_posi_after_preprocessing.txt',
    test_dataset_nega_path='/Users/yibingyang/Documents/thesis_project_new/Data/E-Commerce/after_preprocessing/yelp_nega_after_preprocessing.txt',
    plot_flag=False)

trail.main()

# http://www.apnorton.com/blog/2016/12/19/Visualizing-Multidimensional-Data-in-Python/
# https://towardsdatascience.com/k-means-clustering-algorithm-applications-evaluation-methods-and-drawbacks-aa03e644b48a

# https://github.com/scipy/scipy/pull/8295/files
# https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
# https://markroxor.github.io/gensim/static/notebooks/gensim_news_classification.html notebook using 4 topic models
# https://medium.com/nanonets/topic-modeling-with-lsa-psla-lda-and-lda2vec-555ff65b0b05 Topic Modeling with LSA, PLSA, LDA & lda2Vec

# https://machinelearningmastery.com/feature-selection-machine-learning-python/ feature selection
# https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/



