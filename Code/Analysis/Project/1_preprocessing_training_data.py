# -*- coding: utf-8 -*-


#https://medium.com/@SeoJaeDuk/basic-data-cleaning-engineering-session-twitter-sentiment-data-b9376a91109b could refer to the reference
#read the files as dataframes

#https://stackoverflow.com/questions/12944678/using-unicodedata-normalize-in-python-2-7
#how to get ride of emojis like \uxxxxxx in the tweets

#https://www.youtube.com/watch?v=5aJKKgSEUnY
#video about unicode


import pandas as pd
import json
from referring_list import cList, punc_str, positive_emoji_list, negative_emoji_list
import string
import re
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from unidecode import unidecode
import numpy as np
import time

class label_tweets():

    def __init__(self,folder_path, posi_filename, nega_filename):
        '''
        This function accept the 2 raw datasets(posi&nega tweets) and mainly do another round of cleaning based on the emoji
        contained in the tweets.
        1) The main idea here is to read posi and nega tweets from file (initial function);
        2) Label the tweets based on the emojis (label_the_tweets) and assign score 1 in posi list every time posi emoji appeared
           and -1 in nega list every time nega emoji appeared. Then set every number in posi list to 1 if the number >=1 and -1 if
           the number in nega list to -1 if the number <= -1. After that we have 2 list [1,0,1,1,1,0,1] and [-1,0,0,0,0,-1,0]. We
           want to add them up and only keep the ones not equal to 0 to discard all the tweets with both posi and nega emojis.
        3) In function take_only_the_pos_neg_tweets_and_labels, we only pick up the tweets with non-zero number.

        :param folder_path (String): the path of the folder store the posi and nega raw data acquired from Twitter API
        :param posi_filename (String): the exact filename for posi tweets dataset
        :param nega_filename (String): same as above for nega tweets dataset
        '''

        url_list = [folder_path + posi_filename, folder_path + nega_filename]

        for url in url_list:
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
                self.tweet_df = tweet_df[[

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

                if 'posi' in url:
                    self.posi_tweets_list = [tweet_dict['text'] for tweet_dict in tweets]
                    print 'There are ' + str(len(self.posi_tweets_list)) + ' positive tweets in raw data.'
                else:
                    self.nega_tweets_list = [tweet_dict['text'] for tweet_dict in tweets]
                    print 'There are ' + str(len(self.nega_tweets_list)) + ' negative tweets in raw data..'

    def label_the_tweets(self):  ##still need to find a way to parse unicode r'' means raw string, won't take / as escape sequence
        tweets_list_mixed = self.posi_tweets_list + self.nega_tweets_list
        labeled_list = [0]
        len_list = len(tweets_list_mixed)
        pos_labeled_list = labeled_list * len_list
        neg_labeled_list = labeled_list * len_list
        for i in range(len_list):
            # calculate the pos scores for all the tweets provided
            for pos_emoji in positive_emoji_list:
                if re.search(pos_emoji, tweets_list_mixed[i]) is not None:
                    pos_labeled_list[i] = pos_labeled_list[i] + 1
                else:
                    pass
            # calculate the neg scores for all the tweets provided
            for neg_emoji in negative_emoji_list:
                if neg_emoji in tweets_list_mixed[i]:
                    neg_labeled_list[i] = neg_labeled_list[i] - 1
                else:
                    pass

        pos_labels = [1 if i > 1 else i for i in pos_labeled_list]
        neg_labels = [-1 if i < -1 else i for i in neg_labeled_list]
        len_labels = len(pos_labels)
        combined_labels = [pos_labels[i] + neg_labels[i] for i in range(len_labels)]
        return tweets_list_mixed, combined_labels

    def take_only_the_pos_neg_tweets_and_labels(self, tweets_list_mixed, combined_labels):
        labels = np.array(combined_labels)
        pos_one_labels_index = np.where(labels == 1)[0]
        neg_one_labels_index = np.where(labels == -1)[0]

        pos_text_list = [tweets_list_mixed[i] for i in pos_one_labels_index]
        neg_text_list = [tweets_list_mixed[i] for i in neg_one_labels_index]

        return pos_text_list, neg_text_list

    def main(self):
        startTime = time.time()
        tweets_list_mixed, combined_labels = self.label_the_tweets()
        pos_text_list, neg_text_list = self.take_only_the_pos_neg_tweets_and_labels(tweets_list_mixed, combined_labels)
        print 'Labelling running time is: '+ str(time.time() - startTime)
        return pos_text_list, neg_text_list

class preprocessing():

    def __init__(self,after_labelled_text_list):
        '''
        #---Preprocessing---#

        ['That same old man is actually sitting a few 167018 @hyojinis Okey baby :) @BTS_twt #cool',
        ['Let's put an end to all the confusion. Find...']

        Remove Retweets ---skip cus this dataset has no retweets
        Drop duplicated tweets -- done previously
        Lowercase all the content -- done in f
        Negative contractions were expanded in place and converted to full form (e.g. don't -> do not) --- done in f
        (Maybe not, keep concise)Slang and contracted words were converted to their full form; for example, FYI became for your information.
        Drop the punctuations, symbols, emoticons and all other non-alphabet characters; --- done in f
        Drop @usernames and urls, turn hashtags to the word after # --- done in f, drop the # btw
        happpppppy will be replaced with happy (check if it is doable) --- done in f
        Stemming to deal with Typos and abbreviations and reduce the sparsity of the features.
        Stop words removal # Drop the 'RT' at the beginning

        :param after_labelled_text_list: The posi or nega text list beging processed after label_tweets
        '''
        self.text_list = after_labelled_text_list

    def lowercase(self, text_list):
        lower_case_list = [text.lower() for text in text_list]
        return lower_case_list


    # def convert_punc(text_list): ## convert this for the expand contract
    #     converted_list = [string.replace(unicode(tweet,'utf-8'),'’', '') for tweet in text_list]
    #     return converted_list

    def expand_contractions(self, text_list):  ##-----change to token level
        cList_keys = cList.keys()  # take all keys as a list
        expand_tweet_list = []
        for tweet in text_list:
            for key in cList_keys:
                if key in tweet:
                    tweet = string.replace(tweet, old=key, new=cList[key])
                else:
                    tweet = tweet
            expand_tweet_list.append(tweet)
        return expand_tweet_list


    def remove_tab(self, text_list):
        clean_list = []
        for tweet in text_list:
            tweet = string.replace(tweet, "\n", " ")  # get rid of \n
            clean_list.append(tweet)
        return clean_list


    def tokenize(self, text_list):
        tokenized_tweets = []
        for tweet in text_list:
            tokens = string.split(tweet, sep=' ')
            tokenized_tweets.append(tokens)
        return tokenized_tweets


    def remove_punc_and_symbols(self, text_list):
        clean_list = []
        for tweet in text_list:
            tweet = [re.sub(r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', '', token) for
                     token in tweet]  # drop urls
            tweet = [re.sub(r'(?:@[\w_]+)', '', token) for token in tweet]  # drop @mention
            tweet = [re.sub(r'<[^>]+>', '', token) for token in tweet]  # drop html tags
            tweet = [re.sub(r'(?:(?:\d+,?)+(?:\.?\d+)?)', '', token) for token in tweet]  # drop numbers
            tweet = [re.sub(r"([a-z]+[-]+[a-z])", '', token) for token in tweet]  # drop words with - and '
            tweet = [re.sub(r"([:=;][oO\-]?[D\)\]\(\]\\OpP])", '', token) for token in tweet]  # drop string emojis
            tweet = [string.replace(token, '\n', '') for token in tweet]  # drop \n
            tweet = [re.sub(r"([\!\#\"\%\$\'\&\)\(\+\*\-\,\/\.\;\:\=\<\?\>\@\[\]\\\_\^\`\{\}\|\~\+]+)", '', token) for token in tweet]  # drop punctuations
            temp_list = []
            for token in tweet:
                try:
                    token = re.search(r"([a-z]+[\'\-]*[a-z]+)",token).group(0) # take only the words with english letters
                    temp_list.append(token)
                except:
                    continue
            tweet = [unidecode(token) for token in temp_list] # drop unicode
            tweet = [token for token in tweet if not token == 'rt']  # drop rt
            tweet = [token for token in tweet if token != '']  # drop empty strings
            clean_list.append(tweet)
        return clean_list


    def compressed_repetitive_words(self, text_list):
        clean_list = []
        for tweet in text_list:
            tweet = [re.sub(r'(.)\1+', r'\1\1', token) for token in tweet]
            clean_list.append(tweet)
        return clean_list


    def stemming(self, text_list):
        ps = PorterStemmer()
        clean_list = []
        for tweet in text_list:
            tweet = [ps.stem(token) for token in tweet]
            clean_list.append(tweet)
        return clean_list


    def lemmatizing(self, text_list):
        wl = WordNetLemmatizer()
        clean_list = []
        for tweet in text_list:
            tweet = [wl.lemmatize(token) for token in tweet]
            clean_list.append(tweet)
        return clean_list


    def drop_duplicates(self, text_list):
        clean_list = []
        for i in text_list:
            if i not in clean_list:
                clean_list.append(i)
            else:
                continue
        return clean_list


    def main(self):
        startTime = time.time()
        text_list = self.lowercase(self.text_list)
        # tweet_list = convert_punc(text_list)
        text_list = self.expand_contractions(text_list)
        text_list = self.remove_tab(text_list)
        text_list = self.tokenize(text_list)
        text_list = self.remove_punc_and_symbols(text_list)
        text_list = self.compressed_repetitive_words(text_list)
        text_list = self.lemmatizing(text_list)
        text_list = self.drop_duplicates(text_list)
        if 'pos' in self.text_list:
            print 'Preprocesing running time for positive tweets is: ' + str(time.time() - startTime)
        else:
            print 'Preprocesing running time for negative tweets is: ' + str(time.time() - startTime)
        return text_list

class save_to_file():

    def __init__(self, save_path, save_filename, list_waiting_for_save):
        self.save_path = save_path
        self.save_filename = save_filename
        self.list_waiting_for_save = list_waiting_for_save

    def main(self):
        startTime = time.time()
        entire_path = self.save_path+self.save_filename
        with open(entire_path, 'w') as f:
            for item in self.list_waiting_for_save:
                f.write("%s\n" % item)
        print 'Saving to files running time is: ' + str(time.time() - startTime)


#-----labelling-----#
path = '/Users/yibingyang/Documents/final_thesis_project/Data/Twitter/raw_data/'
posi_filename = 'positive_test.json'
nega_filename = 'negative_test.json'
pos_text_list, neg_text_list = label_tweets(path,posi_filename,nega_filename).main()
print 'After labelling process, positive tweets are ' + str(len(pos_text_list)) + ' and negative ones are ' + str(len(neg_text_list))

#-----preprocessing-----#
posi_text_list = preprocessing(pos_text_list).main()
nega_text_list = preprocessing(neg_text_list).main()
print 'After preprocessing, positive tweets is '+str(len(posi_text_list))+ ' and negative tweets is '+str(len(nega_text_list))

#-----save the result to file-----#
save_path = '/Users/yibingyang/Documents/final_thesis_project/Data/Twitter/after_preprocessing/'
save_filename_posi = 'positive_clean_tweets.txt'
save_filename_nega = 'negative_clean_tweets.txt'
save_to_file(save_path,save_filename_posi, posi_text_list).main()
save_to_file(save_path,save_filename_nega, nega_text_list).main()
