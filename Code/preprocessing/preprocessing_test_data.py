from preprocessing_training_data import preprocessing
import json


def read_and_cleaning(folder_path,filename):
    url = folder_path +  filename
    tweets, labels = [], []
    with open(url, 'r') as f:
        lines = f.readlines()  # read all the lines in the file
        for line in lines:
            line_list = line.split('\t')
            tweets.append(line_list[2])
            labels.append(line_list[1])
    return tweets, labels

def get_only_posi_nega_tweets_and_lables(tweets, labels):
    posi_index = [i for i, x in enumerate(labels) if x == "positive"]
    nega_index = [i for i, x in enumerate(labels) if x == "negative"]
    posi_tweets = [tweets[i] for i in posi_index]
    nega_tweets = [tweets[i] for i in nega_index]
    return posi_tweets, nega_tweets

def save_to_file(data_to_save, folder_path, filename):
    save_path = folder_path + filename
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
# save_to_file(posi_tweets, '/Users/yibingyang/Documents/final_thesis_project/Data/Twitter/test_dataset/', 'positive_test_tweets_after_preprocessing.txt')
# save_to_file(nega_tweets, '/Users/yibingyang/Documents/final_thesis_project/Data/Twitter/test_dataset/', 'negative_test_tweets_after_preprocessing.txt')
