from preprocessing_training_data import preprocessing
import json
import pandas as pd


def read_and_cleaning(folder_path,filename, tweet_column_num = 2, label_column_num = 1):
    url = folder_path +  filename
    tweets, labels = [], []
    with open(url, 'r') as f:
        lines = f.readlines()  # read all the lines in the file
        for line in lines:
            line_list = line.split(',')
            tweets.append(line_list[tweet_column_num])
            labels.append(line_list[label_column_num])
    return tweets, labels

def read_and_cleaning_via_df(folder_path,filename):
    url = folder_path +  filename
    df = pd.read_csv(url, names=['tweets', 'labels'])
    posi_tweets = df[df['labels']==0]['tweets'].tolist()
    nega_tweets = df[df['labels']==1]['tweets'].tolist()
    return posi_tweets, nega_tweets

def get_only_posi_nega_tweets_and_lables(tweets, labels):
    posi_index = [i for i, x in enumerate(labels) if x == "positive"]
    nega_index = [i for i, x in enumerate(labels) if x == "negative"]
    posi_tweets = [tweets[i] for i in posi_index]
    nega_tweets = [tweets[i] for i in nega_index]
    return posi_tweets, nega_tweets

def save_to_file(data_to_save, save_path):
    with open(save_path,'w') as f:
        for item in data_to_save:
            f.write("%s\n" % item)

#
# # --- start with using SemEval2017-task4-test-dataset --- #
# url = '/Users/yibingyang/Documents/final_thesis_project/Data/Twitter/test_dataset/'
# filename = 'SemEval2017-task4-test-dataset.txt'
# tweets, labels = read_and_cleaning(url, file)
#
# # --- get only posi and nega tweets and labels --- #
# posi_tweets, nega_tweets = get_only_posi_nega_tweets_and_lables(tweets, labels)

# # --- preprocessing --- #
# posi_tweets = preprocessing(posi_tweets).main()
# nega_tweets = preprocessing(nega_tweets).main()
#
# # --- save the result to file --- #
# save_to_file(posi_tweets, '/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/test_dataset/positive_test_tweets_after_preprocessing.txt')
# save_to_file(nega_tweets, '/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/test_dataset/negative_test_tweets_after_preprocessing.txt')


# --- start with yelp reviews --- #
url = '/Users/yibingyang/Documents/thesis_project_new/Data/E-Commerce/raw_data/'
filename = 'yelp-reviews.csv'
posi_tweets, nega_tweets = read_and_cleaning_via_df(url, filename)

# --- preprocessing --- #
posi_tweets = preprocessing(posi_tweets).main()
nega_tweets = preprocessing(nega_tweets).main()

# --- save the result to file --- #
save_to_file(posi_tweets, '/Users/yibingyang/Documents/thesis_project_new/Data/E-Commerce/after_preprocessing/yelp_posi_after_preprocessing.txt')
save_to_file(nega_tweets, '/Users/yibingyang/Documents/thesis_project_new/Data/E-Commerce/after_preprocessing/yelp_nega_after_preprocessing.txt')

