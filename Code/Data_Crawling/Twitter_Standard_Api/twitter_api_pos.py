__author__ = 'yyb'

#real-time tweets flows
#http://docs.tweepy.org/en/latest/streaming_how_to.html#a-few-more-pointers
#https://stackoverflow.com/questions/37943800/stream-tweets-using-tweepy-python-using-emoji?rq=1 (helpful!)


import tweepy
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
import datetime
import pprint

consumer_key = 'S5Xu1dAbaWzpVXESPVzS1o5wl'
consumer_secret = 'xs2U9hnXuWDOpctU7fXHklTwVSPb5lZHfrNN2pIQleybEThh9I'
access_token = '3226613022-77lbqgNmmCt626HEh2WvykRbWQTLSJ6wQVxl7Ez'
access_secret = '96OTDjj8bFxnSNXHHiVX9ODr1uxF1noDCbtEPbVJlfDcl'

auth=OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_secret)
pp = pprint.PrettyPrinter(depth=6)

class MyListener(StreamListener):


    def on_status(self, status):
        if status.retweeted:
            return None

        id_str = status.id_str
        created = status.created_at
        text = status.text
        source = status.source
        entities = status.entities
        coords = status.coordinates
        geo = status.geo
        retweeted = status.retweeted
        retweets = status.retweet_count
        name = status.user.screen_name
        user_created = status.user.created_at
        time_zone = status.user.time_zone
        description = status.user.description
        loc = status.user.location
        utc_offset = status.user.utc_offset
        followers = status.user.followers_count
        location = status.user.location
        place = status.place

        tweet_dict = {
            "id_str" : id_str,
            "created" : str(created),
            "text" : text,
            "source" :  source,
            "entities" : entities,
            "coords": coords,
            "geo": geo,
            "retweets": retweets,
            "retweeted" : retweeted,
            "name": name,
            "user_created": str(user_created),
            "time_zone" : time_zone,
            "description" : description,
            "loc" : loc,
            "utc_offset" : utc_offset,
            "followers" : followers,
            "location" : location,
            "place": str(place)
        }

        import json
        json = json.dumps(tweet_dict) + '\n'
        f = open('/Users/yibingyang/Documents/final_thesis_project/Data/positive_test.json',"a+")
        f.write(json)
        f.close()

    # def on_data(self,data):
    #     try:
    #         with open('/Users/yibingyang/Documents/final_thesis_project/Data/positive_test.json','a')as f:
    #             f.write(data)
    #             return True
    #     except BaseException as e:
    #         print('Error on_data:%s'%str(e))

    def on_error(self,status):
        print(status)
        return True # Don't kill the stream

    def on_timeout(self):
        return True  # Don't kill the stream

##positive
twitter_stream = Stream(auth, MyListener())
twitter_stream.filter(
                      track=[u"\U0001F600",
                             u"\U0001F601",
                             u"\U0001F603",
                             u"\U0001F604",
                             u"\U0001F606",
                             u"\U0001F609",
                             u"\U0001F60A",
                             u"\U0001F60C",
                             u"\U0001F60B",
                             u"\U0001F60D",
                             u"\U0001F60E",
                             u"\U0001F60F",
                             u"\U0001F617",
                             u"\U0001F618",
                             u"\U0001F619",
                             u"\U0001F61A",
                             u"\U0001F62C",
                             u"\U0001F638",
                             u"\U0001F63A",
                             u"\U0001F63B",
                             u"\U0001F63C",
                             u"\U0001F63D"
                             ],
                      languages=['en']
                      )