
#https://medium.com/@SeoJaeDuk/basic-data-cleaning-engineering-session-twitter-sentiment-data-b9376a91109b could refer to the reference

#read the files as dataframes



import pandas as pd
import json
from referring_list import cList,punc_str, positive_list, negative_list
import string
import re
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

def read_and_cleaning(folder_path,filename):
    url = folder_path + filename
    with open(url, 'r') as f:
        lines = f.readlines()  # read all the lines in the file
        tweets = [json.loads(line) for line in lines]  # load them as Python dict

        for tweet_dict in tweets:
            tweet_dict["hashtags"] = tweet_dict["entities"]["hashtags"]
            tweet_dict["hashtags_num"] = len(tweet_dict["entities"]["hashtags"])

            tweet_dict["symbols"] = tweet_dict["entities"]["symbols"]
            tweet_dict["symbols_num"] = len(tweet_dict["entities"]["symbols"])

            tweet_dict["urls"] = tweet_dict["entities"]["urls"]
            tweet_dict["urls_num"] = len(tweet_dict["entities"]["urls"])

            tweet_dict["user_mentions"] = tweet_dict["entities"]["user_mentions"]
            tweet_dict["user_mentions_num"] = len(tweet_dict["entities"]["user_mentions"])

        tweet_df = pd.DataFrame.from_dict(tweets) #it's okay
        tweet_df = tweet_df[[

            'id_str','retweeted','retweets',
            # time
            'created','time_zone','utc_offset',
            # location
            'coords','geo','loc','location',

            'hashtags','hashtags_num','symbols','symbols_num','urls','urls_num','user_mentions','user_mentions_num',

            # user
            'name','user_created','description','place','followers',

            'source','text',
        ]]
        return tweet_df

#---Preprocessing---#

#['That same old man is actually sitting a few 167018 @hyojinis Okey baby :) @BTS_twt #cool',
# 'Let's put an end to all the confusion. Find...']

# Remove Retweets ---skip cus this dataset has no retweets
# Drop duplicated tweets -- done previously
# Lowercase all the content -- done in f
# Negative contractions were expanded in place and converted to full form (e.g. don't -> do not) --- done in f
# (Maybe not, keep concise)Slang and contracted words were converted to their full form; for example, FYI became for your information.
# Drop the punctuations, symbols, emoticons and all other non-alphabet characters; --- done in f
# Drop @usernames and urls, turn hashtags to the word after # --- done in f, drop the # btw
# happpppppy will be replaced with happy (check if it is doable) --- done in f
# Stemming to deal with Typos and abbreviations and reduce the sparsity of the features.
# Stop words removal # Drop the 'RT' at the beginning

def lowercase(text_list):
    lower_case_list = [text.lower() for text in text_list]
    return lower_case_list

def expand_contractions(text_list):
    cList_keys = cList.keys() # take all keys as a list
    expand_tweet_list = []
    for tweet in text_list:
        for key in cList_keys:
            if key in tweet:
                tweet = string.replace(tweet, old=key, new=cList[key])
            else:
                tweet = tweet
        expand_tweet_list.append(tweet)
    return expand_tweet_list

def label_the_tweets(text_list):
    labeled_list = [0]
    len_list = len(text_list)
    pos_labeled_list = labeled_list * len_list
    neg_labeled_list = labeled_list * len_list
    for i in range(len_list):
        #calculate the pos ones
        for pos_emoji in positive_list:
            if pos_emoji in text_list[i]:
                pos_labeled_list[i] = pos_labeled_list[i] + 1
            else:
                pass
        #calculate the neg ones
        for neg_emoji in negative_list:
            if neg_emoji in text_list[i]:
                neg_labeled_list[i] = neg_labeled_list[i] - 1
            else:
                pass

    pos_labels = [1 if i>1 else i for i in pos_labeled_list]
    neg_labels = [-1 if i<-1 else i for i in neg_labeled_list]
    len_labels = len(pos_labels)
    combined_labels = [pos_labels[i]+neg_labels[i] for i in range(len_labels)]
    return combined_labels

def take_only_the_pos_neg_tweets_and_labels(text_list, combined_labels):
    import numpy as np
    labels = np.array(combined_labels)
    pos_one_labels_index = np.where(labels == 1)[0]
    neg_one_labels_index = np.where(labels == -1)[0]
    pos_text_list = [text_list[i] for i in pos_one_labels_index]
    neg_text_list = [text_list[i] for i in neg_one_labels_index]
    pos_labels = [combined_labels[i] for i in pos_one_labels_index]
    neg_labels = [combined_labels[i] for i in neg_one_labels_index]
    text_list = pos_text_list+neg_text_list
    labels = pos_labels + neg_labels
    return text_list, labels

def remove_tab_and_add_space(text_list):
    clean_list = []
    for tweet in text_list:
        tweet=string.replace(tweet, "\\", " \\") #get rid of \n
        tweet=string.replace(tweet, "\n", " ") #get rid of \n
        clean_list.append(tweet)
    return clean_list

def tokenize(text_list):
    tokenized_tweets = []
    for tweet in text_list:
        tokens = string.split(tweet,sep=' ')
        tokenized_tweets.append(tokens)
    return tokenized_tweets

def remove_punc_and_symbols(text_list):
    clean_list = []
    for tweet in text_list:
        tweet = [re.sub(r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', '', token) for token in tweet] # drop urls
        tweet = [re.sub(r'(?:@[\w_]+)', '', token) for token in tweet]                      # drop @mention
        tweet = [re.sub(r'\\u[a-z0-9]+', '', token) for token in tweet]                     # drop unicode emojis
        tweet = [re.sub(r'<[^>]+>', '', token) for token in tweet]                          # drop html tags
        tweet = [re.sub(r'(?:(?:\d+,?)+(?:\.?\d+)?)', '', token) for token in tweet]        # drop numbers
        tweet = [re.sub(r"([a-z]+[-]+[a-z])", '', token) for token in tweet]                # drop words with - and '
        tweet = [re.sub(r"([:=;][oO\-]?[D\)\]\(\]\\OpP])", '', token) for token in tweet]   # drop emoticons
        tweet = [re.sub(r"([\!\#\"\%\$\'\&\)\(\+\*\-\,\/\.\\\;\:\=\<\?\>\@\[\]\_\^\`\{\}\|\~])", '', token) for token in tweet]      # drop punctuations
        tweet = [token for token in tweet if not (token.startswith('\\') and len(token)==6)]      # drop \ud558
        tweet = [token for token in tweet if token != '']                                   # drop empty strings
        clean_list.append(tweet)
    return clean_list

def compressed_repetitive_words(text_list):
    clean_list = []
    for tweet in text_list:
        tweet = [re.sub(r'(.)\1+', r'\1\1', token) for token in tweet]
        clean_list.append(tweet)
    return clean_list

def stemming(text_list):
    ps = PorterStemmer()
    clean_list = []
    for tweet in text_list:
        tweet = [ps.stem(token) for token in tweet]
        clean_list.append(tweet)
    return clean_list

def lemmatizing(text_list):
    wl = WordNetLemmatizer()
    clean_list = []
    for tweet in text_list:
        tweet = [wl.lemmatize(token) for token in tweet]
        clean_list.append(tweet)
    return clean_list

def main(the_raw_tweets):
    text_list = lowercase(the_raw_tweets)
    text_list = expand_contractions(text_list)
    text_list = tokenize(text_list)
    text_list = remove_punc_and_symbols(text_list)
    text_list = compressed_repetitive_words(text_list)
    text_list = lemmatizing(text_list)
    return text_list

the_raw_tweets = ["RT @khairul_hafidz: Let us all start to have a #healthylifestyle\n\nDont be hasty, slowly but surely.\n\nIf I can, why cant you? \U0001f60a\n\n#cabarantur\u2026",
                    "If I can, why cant you? \U0001234 u0001f616",
                    "If I can, why cant you? \U0001f60a \u0001f609 u0001f616",
                    "If I can, why cant you?"
                  ]
text_list = lowercase(the_raw_tweets)
#['that same old man is actually sitting a few 167018 @hyojinis okey baby :) @bts_twt #cool', "let's put an end to all the confusion. find..."]

combined_labels = label_the_tweets(text_list)
#[1, -1, 0, 0]

text_list, labels = take_only_the_pos_neg_tweets_and_labels(text_list, combined_labels)
#['rt @khairul_hafidz: let us all start to have a #healthylifestyle\n\ndont be hasty, slowly but surely.\n\nif i can, why cant you? \\u0001f60a\n\n#cabarantur\\u2026', 'if i can, why cant you? \\u0001234 u0001f616'] [1, -1]

text_list = expand_contractions(text_list)
#['that same old man is actually sitting a few 167018 @hyojinis okey baby :) @bts_twt #cool', 'let us put an end to all the confusion. find...']

text_list = remove_tab_and_add_space(text_list)

text_list = tokenize(text_list)
#[['that', 'same', 'old', 'man', 'is', 'actually', 'sitting', 'a', 'few', '167018', '@hyojinis', 'okey', 'baby', ':)', '@bts_twt', '#cool'], ['let', 'us', 'put', 'an', 'end', 'to', 'all', 'the', 'confusion.', 'find...']]

text_list = remove_punc_and_symbols(text_list)
#[['that', 'same', 'old', 'man', 'is', 'actually', 'sitting', 'a', 'few', 'okey', 'baby', 'cool'], ['let', 'us', 'put', 'an', 'end', 'to', 'all', 'the', 'confusion', 'find', 'it', 'here', 'now']]

text_list = compressed_repetitive_words(text_list)
#[['happy', 'good', 'same', 'old', 'man', 'is', 'actually', 'sitting', 'a', 'few', 'okey', 'baby', 'cool'], ['let', 'us', 'put', 'an', 'end', 'to', 'all', 'the', 'confusion', 'find', 'it', 'here', 'now']]

text_list = stemming(text_list)
#[[u'happi', 'good', 'same', 'old', 'man', 'is', u'actual', u'sit', 'a', 'few', 'okey', u'babi', 'cool'], ['let', 'us', 'put', 'an', 'end', 'to', 'all', 'the', u'confus', 'find', 'it', 'here', 'now']]

text_list = lemmatizing(text_list)
#[['happy', 'good', 'same', 'old', 'man', 'is', 'actually', 'sitting', 'a', 'few', 'okey', 'baby', 'cool'], ['let', u'u', 'put', 'an', 'end', 'to', 'all', 'the', 'confusion', 'find', 'it', 'here', 'now']]


print text_list

# path = '/Users/yibingyang/Documents/final_thesis_project/Data/'
# file = 'positive_test.json'
# read_and_cleaning(path, file)