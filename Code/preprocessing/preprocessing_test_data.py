from preprocessing_training_data import preprocessing
import json


def read_and_cleaning(folder_path,filename):
    url = folder_path + filename
    with open(url, 'r') as f:
        lines = f.readlines()  # read all the lines in the file
        tweets = [json.loads(line) for line in lines]  # load them as Python dict
        tweets_list = [tweet['text'] for tweet in tweets]
        return tweets_list


path = '/Users/yibingyang/Documents/final_thesis_project/Data/Twitter/'
iphone_file = 'iphone_stream_testdata_collection.json'
BeAVoter_file = 'BeAVoter.json'
MidtermElections_file = 'MidtermElections2018.json'

iphone_list = read_and_cleaning(path, iphone_file)

for i in iphone_list:
    print i + '\n'
    print '------------------------'

# def read_and_cleaning(folder_path,filename):
#     url = folder_path + filename
#     with open(url, 'r') as f:
#         lines = f.readlines()  # read all the lines in the file
#         tweets = [json.loads(line) for line in lines]  # load them as Python dict
#         tweets_list = [tweet['text'] for tweet in tweets]
#         return tweets_list
#
#
# path = '/Users/yibingyang/Documents/final_thesis_project/Data/Twitter/'
# iphone_file = 'iphone_stream_testdata_collection.json'
# BeAVoter_file = 'BeAVoter.json'
# MidtermElections_file = 'MidtermElections2018.json'
#
# iphone_list = read_and_cleaning(path, iphone_file)
#
# for i in iphone_list:
#     print i + '\n'
#     print '------------------------'



# a model that with noisy labels as initial labels, and can get evolved when user add some customized topics into it
# compare different model's performance and accuracy
# --- for each model, how's the accuracy when adding more data into the model and how's the increase of time
# find a performance (time) and accuracy rate
# use new data with emojis to replace old data periodically; and based on the rate, use the new topic related data to replace the old data

#---- start with using SemEval2017-task4-test-dataset ----#
url = '/Users/yibingyang/Documents/final_thesis_project/Data/Twitter/test_dataset/SemEval2017-task4-test-dataset.txt'
tweets = []
labels = []
with open(url, 'r') as f:
    lines = f.readlines()  # read all the lines in the file
    for line in lines:
        line_list = line.split('\t')
        tweets.append(line_list[2])
        labels.append(line_list[1])

def get_only_posi_nega_tweets_and_lables(tweets, labels):
    posi_index = [i for i, x in enumerate(labels) if x == "positive"]
    nega_index = [i for i, x in enumerate(labels) if x == "negative"]
    # posi_labels = [labels[i] for i in posi_index]
    # nega_labels = [labels[i] for i in nega_index]
    posi_tweets = [tweets[i] for i in posi_index]
    nega_tweets = [tweets[i] for i in nega_index]

    return posi_tweets, nega_tweets


posi_tweets, nega_tweets = get_only_posi_nega_tweets_and_lables(tweets, labels)
posi_tweets = preprocessing.processing_pipeline(posi_tweets)
nega_tweets = preprocessing.processing_pipeline(nega_tweets)

print posi_tweets[:10]
print nega_tweets[:10]
print len(posi_tweets)
print len(nega_tweets)


##save the result to file
with open('/Users/yibingyang/Documents/final_thesis_project/Data/Twitter/test_dataset/positive_test_tweets.txt', 'w') as f:
    for item in posi_tweets:
        f.write("%s\n" % item)

with open('/Users/yibingyang/Documents/final_thesis_project/Data/Twitter/test_dataset/negative_test_tweets.txt', 'w') as f:
    for item in nega_tweets:
        f.write("%s\n" % item)

