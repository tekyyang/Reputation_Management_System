from topic_model_main import topic_model_builder
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model, naive_bayes
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def test_init():
    test_instance = topic_model_builder(
        training_dataset_posi_paths='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_positive_tweets_after_preprocessing.txt',
        training_dataset_nega_paths='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_negative_tweets_after_preprocessing.txt',
        test_dataset_posi_path='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_positive_test_tweets_after_preprocessing.txt',
        test_dataset_nega_path='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_negative_test_tweets_after_preprocessing.txt')

    assert isinstance(test_instance.posi_training_data_list, list)
    assert isinstance(test_instance.posi_training_data_tokens, list)
    assert isinstance(test_instance.posi_training_data_labels, list)
    assert isinstance(test_instance.posi_training_data_df, DataFrame)

    assert test_instance.posi_training_data_list[0] == [['hey', 'really', 'sorry', 'knocked', 'you', 'down', 'but', 'can', 'pick', 'you', 'up', 'at'], 1]
    assert test_instance.posi_training_data_tokens[0] == ['hey', 'really', 'sorry', 'knocked', 'you', 'down', 'but', 'can', 'pick', 'you', 'up', 'at']
    assert test_instance.posi_training_data_labels[0] == 1
    assert test_instance.posi_training_data_df['tweets'][0] == 'hey really sorry knocked you down but can pick you up at'
    assert test_instance.posi_training_data_df['label'][0] == 0


def test_to_string_list_tool():
    test_instance = topic_model_builder(
        training_dataset_posi_paths='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_positive_tweets_after_preprocessing.txt',
        training_dataset_nega_paths='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_negative_tweets_after_preprocessing.txt',
        test_dataset_posi_path='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_positive_test_tweets_after_preprocessing.txt',
        test_dataset_nega_path='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_negative_test_tweets_after_preprocessing.txt')

    test_df = DataFrame(data={'tweet':['This is tweet one','This is tweet two'],
                              'others':[1,2] })

    test_token_list =[['This','is','tweet','one'],['This','is','tweet','two']]

    token_list = test_instance.to_string_list_tool(input=test_df, tweet_column_name_in_df='tweet', mode='df_to_token_list' )
    string_list_from_df = test_instance.to_string_list_tool(input=test_df, tweet_column_name_in_df='tweet', mode='df_to_string_list' )
    string_list_from_token_list = test_instance.to_string_list_tool(input=test_token_list, mode='token_list_to_string_list' )

    expected_token_list = [['This','is','tweet','one'],['This','is','tweet','two']]
    expected_string_list_from_df = ['This is tweet one','This is tweet two']
    expected_string_list_from_token_list = ['This is tweet one','This is tweet two']

    assert token_list == expected_token_list
    assert string_list_from_df == expected_string_list_from_df
    assert string_list_from_token_list == expected_string_list_from_token_list


def test_data_resampling():
    test_instance = topic_model_builder(
        training_dataset_posi_paths='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_positive_tweets_after_preprocessing.txt',
        training_dataset_nega_paths='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_negative_tweets_after_preprocessing.txt',
        test_dataset_posi_path='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_positive_test_tweets_after_preprocessing.txt',
        test_dataset_nega_path='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_negative_test_tweets_after_preprocessing.txt')

    test_posi_df = DataFrame(data={'tweets': ['This is tweet one', 'This is tweet two'],
                                             'others': [1, 2]})

    test_nega_df = DataFrame(data={'tweets': ['This is tweet one', 'This is tweet two', 'This is tweet three'],
                                             'others': [1, 2, 3]})

    training_posi_resampled_token_list, training_nega_resampled_token_list, resampling_processing_time = test_instance.data_resampling(test_posi_df, test_nega_df, mode = 'r_under_s')

    assert isinstance(training_posi_resampled_token_list, list)
    assert isinstance(training_nega_resampled_token_list, list)

    assert len(training_posi_resampled_token_list) == 2
    assert len(training_nega_resampled_token_list) == 2

    training_posi_resampled_token_list, training_nega_resampled_token_list, resampling_processing_time = test_instance.data_resampling(test_posi_df, test_nega_df, mode = 'r_upper_s')

    assert len(training_posi_resampled_token_list) == 3
    assert len(training_nega_resampled_token_list) == 3


def test_bigram_or_unigram_extactor():

    test_instance = topic_model_builder(
        training_dataset_posi_paths='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_positive_tweets_after_preprocessing.txt',
        training_dataset_nega_paths='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_negative_tweets_after_preprocessing.txt',
        test_dataset_posi_path='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_positive_test_tweets_after_preprocessing.txt',
        test_dataset_nega_path='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_negative_test_tweets_after_preprocessing.txt')

    posi_training_token_list = [[u'positive',u'tweet',u'one',u'pattern',u'here'],[u'the',u'second',u'token',u'pattern',u'here']]
    nega_training_token_list = [[u'negative',u'token',u'one',u'pattern',u'here'],[u'some',u'random',u'things',u'pattern',u'here']]

    # test_list = [['tweet','test','one','not','picked'],['one','pattern','here']]

    posi_unigran_training_token_list, nega_unigran_training_token_list, feature_extraction_processing_time = test_instance.bigram_or_unigram_extactor(posi_training_token_list, nega_training_token_list, bigram_min_count=3, mode='unigram')
    assert posi_unigran_training_token_list == posi_training_token_list
    assert nega_unigran_training_token_list == nega_training_token_list

    posi_training_token_list_with_unigram_and_bigram, nega_training_token_list_with_unigram_and_bigram, feature_extraction_processing_time = test_instance.bigram_or_unigram_extactor(posi_training_token_list, nega_training_token_list, bigram_min_count=2, threshold=1, mode='uni_and_bigram')
    assert posi_training_token_list_with_unigram_and_bigram == [[u'positive',u'tweet',u'one',u'pattern',u'here', u'pattern_here'],[u'the',u'second',u'token',u'pattern',u'here', u'pattern_here']]
    assert nega_training_token_list_with_unigram_and_bigram == [[u'negative',u'token',u'one',u'pattern',u'here', u'pattern_here'],[u'some',u'random',u'things',u'pattern',u'here', u'pattern_here']]


def test_feature_selection():
    test_instance = topic_model_builder(
        training_dataset_posi_paths='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_positive_tweets_after_preprocessing.txt',
        training_dataset_nega_paths='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_negative_tweets_after_preprocessing.txt',
        test_dataset_posi_path='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_positive_test_tweets_after_preprocessing.txt',
        test_dataset_nega_path='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_negative_test_tweets_after_preprocessing.txt')

    posi_training_token_list = [[u'positive', u'tweet', u'one', u'pattern', u'here'],[u'the', u'second', u'token', u'pattern', u'here']]
    nega_training_token_list = [[u'negative', u'token', u'one', u'pattern', u'here'],[u'some', u'random', u'things', u'pattern', u'here']]

    X_chi_matrix, feature_name_list_after_chi, token_list_after_chi2, ch2, feature_selection_processing_time=test_instance.feature_selection(posi_training_token_list, nega_training_token_list, top_n_feature = 5)

    assert X_chi_matrix.shape == (4,5)
    assert feature_name_list_after_chi == [u'negative', u'positive', u'second', u'the', u'tweet']
    assert token_list_after_chi2 == [[u'positive', u'tweet'], [u'second', u'the'], [u'negative'], []]


def test_build_lda_model():
    test_instance = topic_model_builder(
        training_dataset_posi_paths='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_positive_tweets_after_preprocessing.txt',
        training_dataset_nega_paths='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_negative_tweets_after_preprocessing.txt',
        test_dataset_posi_path='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_positive_test_tweets_after_preprocessing.txt',
        test_dataset_nega_path='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_negative_test_tweets_after_preprocessing.txt')

    token_list_after_feature_selection = [[u'positive', u'tweet', u'one', u'pattern', u'here'],[u'the', u'second', u'token', u'pattern', u'here'],
                                          [u'negative', u'token', u'one', u'pattern', u'here'],[u'some', u'random', u'things', u'pattern', u'here']]
    top_model, model_topics, highest_coherence_score, dictionary, corpus, lad_lsi_processing_time = test_instance.build_lda_lsi_model(token_list_after_feature_selection, min_topic_num = 2, max_topic_num = 5, model='lda')
    assert len(model_topics) <=5 and len(model_topics)>=2

    assert len(dictionary.id2token) == 12 # all the words ever appear in the token list
    assert dictionary.id2token[0] == 'here' and dictionary.id2token[1] == 'one' #{0: u'here', 1: u'one', 2: u'pattern', 3: u'positive', 4: u'tweet',...}
    assert dictionary.num_docs == 4

    assert len(corpus) == 4 # the length of the token list
    assert corpus[0] == [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]


def test_get_tweet_topic_matrix_based_on_best_topic_model():
    test_instance = topic_model_builder(
        training_dataset_posi_paths='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_positive_tweets_after_preprocessing.txt',
        training_dataset_nega_paths='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_negative_tweets_after_preprocessing.txt',
        test_dataset_posi_path='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_positive_test_tweets_after_preprocessing.txt',
        test_dataset_nega_path='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_negative_test_tweets_after_preprocessing.txt')

    token_list_after_feature_selection = [[u'positive', u'tweet', u'one', u'pattern', u'here'],[u'the', u'second', u'token', u'pattern', u'here'],
                                          [u'negative', u'token', u'one', u'pattern', u'here'],[u'some', u'random', u'things', u'pattern', u'here']]

    top_model, model_topics, highest_coherence_score, dictionary, corpus, lad_lsi_processing_time = test_instance.build_lda_lsi_model(token_list_after_feature_selection, min_topic_num=2, max_topic_num=5, model='lda')

    tweet_topic_distribution_df = test_instance.get_tweet_topic_matrix_based_on_best_topic_model(top_model, corpus)

    assert isinstance(tweet_topic_distribution_df, DataFrame)
    assert tweet_topic_distribution_df.shape == (4, 2)


def test_collect_clustering_info():
    test_instance = topic_model_builder(
        training_dataset_posi_paths='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_positive_tweets_after_preprocessing.txt',
        training_dataset_nega_paths='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_negative_tweets_after_preprocessing.txt',
        test_dataset_posi_path='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_positive_test_tweets_after_preprocessing.txt',
        test_dataset_nega_path='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_negative_test_tweets_after_preprocessing.txt')

    tweet_topic_distribution_df = DataFrame(data={0:[0.135607, 0.898807, 0.882122, 0.895495, 0.10, 0.20],
                                                  1:[0.864393, 0.101193, 0.117878, 0.104505, 0.90, 0.80]})

    list_k, lable_list, model_list, collect_clustering_info_processing_time = test_instance.collect_clustering_info(tweet_topic_distribution_df, min_cluster_number = 1, max_cluster_number = 3)

    assert list_k == [1,2,3]

    # assert sorted(lable_list) == [[0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 1, 1], [1, 0, 0, 0, 1, 2]]
    assert all([isinstance(model, KMeans) for model in model_list])


def test_add_clustering_info_to_df():

    test_instance = topic_model_builder(
        training_dataset_posi_paths='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_positive_tweets_after_preprocessing.txt',
        training_dataset_nega_paths='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_negative_tweets_after_preprocessing.txt',
        test_dataset_posi_path='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_positive_test_tweets_after_preprocessing.txt',
        test_dataset_nega_path='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_negative_test_tweets_after_preprocessing.txt')

    tweet_topic_distribution_df = DataFrame(data={0: [0.135607, 0.898807, 0.882122, 0.895495, 0.10, 0.20],
                                                  1: [0.864393, 0.101193, 0.117878, 0.104505, 0.90, 0.80]})

    list_k, lable_list, model_list, collect_clustering_info_processing_time = test_instance.collect_clustering_info(tweet_topic_distribution_df, min_cluster_number=1, max_cluster_number=3)

    tweet_topic_distribution_with_cluster_df, selected_kmeans_model, number_of_cluster, add_clustering_info_to_df_processing_time = test_instance.add_clustering_info_to_df(tweet_topic_distribution_df, list_k, lable_list, model_list, number_of_cluster=3)

    assert (tweet_topic_distribution_df[0] == tweet_topic_distribution_with_cluster_df[0]).tolist()
    assert (tweet_topic_distribution_df[1] == tweet_topic_distribution_with_cluster_df[1]).tolist()
    assert sorted(tweet_topic_distribution_with_cluster_df.columns.tolist()) == [0, 1, 'Y', 'clustering_labels']
    assert tweet_topic_distribution_with_cluster_df['Y'].tolist() == [1,1,1,0,0,0]
    assert len(tweet_topic_distribution_with_cluster_df['clustering_labels'])==6
    assert number_of_cluster == 3


def test_classifier_building():
    test_instance = topic_model_builder(
        training_dataset_posi_paths='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_positive_tweets_after_preprocessing.txt',
        training_dataset_nega_paths='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_negative_tweets_after_preprocessing.txt',
        test_dataset_posi_path='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_positive_test_tweets_after_preprocessing.txt',
        test_dataset_nega_path='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_negative_test_tweets_after_preprocessing.txt')

    token_list_after_feature_selection = [[u'positive', u'tweet', u'one', u'pattern', u'here'],
                                          [u'the', u'second', u'token', u'pattern', u'here'],
                                          [u'negative', u'token', u'one', u'pattern', u'here'],
                                          [u'some', u'random', u'things', u'pattern', u'here']]

    tweet_topic_distribution_with_cluster_df = DataFrame(data={0: [0.135607, 0.882122, 0.12, 0.895495],
                                                               1: [0.864393, 0.117878, 0.88, 0.104505],
                                                              'Y':[1,1,0,0],
                                                              'clustering_labels':[1,0,1,0]})

    vectorizer_clf_dict, classifier_building_processing_time = test_instance.classifier_building(tweet_topic_distribution_with_cluster_df, 2, token_list_after_feature_selection)

    assert vectorizer_clf_dict.keys() == [0,1] # the two topics
    assert isinstance(vectorizer_clf_dict[0][0], TfidfVectorizer)
    assert isinstance(vectorizer_clf_dict[0][1], linear_model.LogisticRegression)


def test_test_data_fit_in_model():
    test_instance = topic_model_builder(
        training_dataset_posi_paths='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_positive_tweets_after_preprocessing.txt',
        training_dataset_nega_paths='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_negative_tweets_after_preprocessing.txt',
        test_dataset_posi_path='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_positive_test_tweets_after_preprocessing.txt',
        test_dataset_nega_path='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_negative_test_tweets_after_preprocessing.txt')

    X_chi_matrix,feature_name_list_after_feature_selection, token_list_after_feature_selection, ch2, feature_selection_processing_time = test_instance.feature_selection(test_instance.posi_test_data_tokens, test_instance.nega_test_data_tokens, top_n_feature=150)

    top_model, model_topics, highest_coherence_score, dictionary, corpus, lad_lsi_processing_time = test_instance.build_lda_lsi_model(token_list_after_feature_selection, min_topic_num = 3, max_topic_num = 6, model='lda')

    tweet_topic_distribution_df = test_instance.get_tweet_topic_matrix_based_on_best_topic_model(top_model, corpus)

    list_k, lable_list, model_list, collect_clustering_info_processing_time = test_instance.collect_clustering_info(tweet_topic_distribution_df, min_cluster_number=2, max_cluster_number=10)

    tweet_topic_distribution_with_cluster_df, selected_kmeans_model, number_of_cluster, add_clustering_info_to_df_processing_time = test_instance.add_clustering_info_to_df(tweet_topic_distribution_df, list_k, lable_list, model_list, number_of_cluster=2)

    vectorizer_clf_dict, classifier_building_processing_time = test_instance.classifier_building(tweet_topic_distribution_with_cluster_df, number_of_cluster, token_list_after_feature_selection=token_list_after_feature_selection)

    restructured_X_test_df, test_data_fit_in_processing_time = test_instance.test_data_fit_in_model(vectorizer_clf_dict, top_model, dictionary, selected_kmeans_model)

    confusion_matrix, classification_report, accuracy_score = test_instance.evaluation(restructured_X_test_df)

    Y_test = restructured_X_test_df['label'].tolist()
    Y_pred = restructured_X_test_df['y_pred'].tolist()
    cluster_label_list = restructured_X_test_df['cluster_label'].tolist()

    assert len(Y_test) == len(Y_pred)
    assert len(Y_test) == len(cluster_label_list)

    assert isinstance(confusion_matrix, np.ndarray)
    assert isinstance(classification_report, unicode)
    assert isinstance(accuracy_score, np.float)


def test_baseline_model_builder():
    test_instance = topic_model_builder(
        training_dataset_posi_paths='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_positive_tweets_after_preprocessing.txt',
        training_dataset_nega_paths='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_negative_tweets_after_preprocessing.txt',
        test_dataset_posi_path='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_positive_test_tweets_after_preprocessing.txt',
        test_dataset_nega_path='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_negative_test_tweets_after_preprocessing.txt')

    token_list_after_feature_selection = [[u'positive', u'tweet', u'one', u'pattern', u'here'],
                                          [u'the', u'second', u'token', u'pattern', u'here'],
                                          [u'negative', u'token', u'one', u'pattern', u'here'],
                                          [u'some', u'random', u'things', u'pattern', u'here']]

    vectorizer, baseline_clf_dict, baseline_classifier_building_processing_time = test_instance.baseline_model_builder(token_list_after_feature_selection, mode = 'tfidf')

    assert isinstance(vectorizer, TfidfVectorizer)
    assert baseline_clf_dict.keys() == ['logistic_regression', 'naive_bayes']
    assert isinstance(baseline_clf_dict.values()[0], linear_model.LogisticRegression)
    assert isinstance(baseline_clf_dict.values()[1], naive_bayes.MultinomialNB)


def test_baseline_test_data_fit_in_model():
    test_instance = topic_model_builder(
        training_dataset_posi_paths='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_positive_tweets_after_preprocessing.txt',
        training_dataset_nega_paths='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_negative_tweets_after_preprocessing.txt',
        test_dataset_posi_path='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_positive_test_tweets_after_preprocessing.txt',
        test_dataset_nega_path='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_negative_test_tweets_after_preprocessing.txt')

    training_posi_resampled_token_list, training_nega_resampled_token_list, resampling_processing_time = test_instance.data_resampling(test_instance.posi_training_data_df, test_instance.nega_training_data_df)

    posi_bigram_training_token_list, nega_bigram_training_token_list, feature_extraction_processing_time = test_instance.bigram_or_unigram_extactor(training_posi_resampled_token_list, training_nega_resampled_token_list, mode='uni_and_bigram',bigram_min_count=3)

    X_chi_matrix, feature_name_list_after_feature_selection, token_list_after_feature_selection, ch2, feature_selection_processing_time = test_instance.feature_selection(posi_bigram_training_token_list, nega_bigram_training_token_list, top_n_feature=150)

    top_model, model_topics, highest_coherence_score, dictionary, corpus, lad_lsi_processing_time = test_instance.build_lda_lsi_model(token_list_after_feature_selection, min_topic_num=3, max_topic_num=6, model='lda')

    tweet_topic_distribution_df = test_instance.get_tweet_topic_matrix_based_on_best_topic_model(top_model, corpus)

    list_k, lable_list, model_list, collect_clustering_info_processing_time = test_instance.collect_clustering_info(tweet_topic_distribution_df, min_cluster_number=2, max_cluster_number=10)

    tweet_topic_distribution_with_cluster_df, selected_kmeans_model, number_of_cluster, add_clustering_info_to_df_processing_time = test_instance.add_clustering_info_to_df(tweet_topic_distribution_df, list_k, lable_list, model_list, number_of_cluster=2)

    vectorizer_clf_dict, classifier_building_processing_time = test_instance.classifier_building(tweet_topic_distribution_with_cluster_df, number_of_cluster,token_list_after_feature_selection=token_list_after_feature_selection)

    restructured_X_test_df, test_data_fit_in_processing_time = test_instance.test_data_fit_in_model(vectorizer_clf_dict, top_model, dictionary, selected_kmeans_model)

    vectorizer, baseline_clf_dict, baseline_classifier_building_processing_time = test_instance.baseline_model_builder(token_list_after_feature_selection, mode = 'tfidf')

    restructured_X_test_df, baseline_clf_name_list, baseline_test_data_fit_in_processing_time = test_instance.baseline_test_data_fit_in_model(vectorizer, baseline_clf_dict, restructured_X_test_df)

    evaluation_dict = test_instance.baseline_evaluation(restructured_X_test_df, baseline_clf_name_list)

    assert evaluation_dict.keys() == ['logistic_regression', 'naive_bayes']

    assert isinstance(evaluation_dict['logistic_regression'][0], np.ndarray) #cm
    assert isinstance(evaluation_dict['logistic_regression'][1], unicode) #classification_report
    assert isinstance(evaluation_dict['logistic_regression'][2], np.float) #accuracy_score

def test_show_sample_tweets():
    test_instance = topic_model_builder(
        training_dataset_posi_paths='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_positive_tweets_after_preprocessing.txt',
        training_dataset_nega_paths='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_negative_tweets_after_preprocessing.txt',
        test_dataset_posi_path='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_positive_test_tweets_after_preprocessing.txt',
        test_dataset_nega_path='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_negative_test_tweets_after_preprocessing.txt')
    df_prep =  {'tweets':              [u'tweet 1 here', u'tweet 2 here', u'tweet 3 here', u'tweet 4 here', u'tweet 5 here'],
                'label':               [1,1,1,0,0],
                'y_pred':              [1,1,0,0,0],
                'cluster_label':       [1,0,0,1,1],
                'logistic_regression': [1,1,1,1,1],
                'naive_bayes':         [0,0,0,0,0]}
    restructured_X_test_df = pd.DataFrame.from_dict(df_prep)
    test_instance.show_sample_tweets(restructured_X_test_df=restructured_X_test_df, cluster_label_list=[1,0,0,1,1])


def test_main():
    test_instance = topic_model_builder(
        training_dataset_posi_paths='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_positive_tweets_after_preprocessing.txt',
        training_dataset_nega_paths='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_negative_tweets_after_preprocessing.txt',
        test_dataset_posi_path='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_positive_test_tweets_after_preprocessing.txt',
        test_dataset_nega_path='/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/after_preprocessing/test_negative_test_tweets_after_preprocessing.txt')

    test_instance.main(
        feature_extraction_mode='uni_and_bigram', bigram_min_count=3,
        feature_represent_mode='tfidf', feature_selection_mode='chi2', top_n_feature=150,
        lda_min_topic_num=3, lda_max_topic_num=10,
        lsi_min_topic_num=3, lsi_max_topic_num=10,
        min_cluster_number=1, max_cluster_number=10,
        number_of_cluster=1,
        classifier='naive_bayes')


