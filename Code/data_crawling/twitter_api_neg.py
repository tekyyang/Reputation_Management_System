__author__ = 'yyb'

#real-time tweets flows
#http://docs.tweepy.org/en/latest/streaming_how_to.html#a-few-more-pointers
#https://stackoverflow.com/questions/37943800/stream-tweets-using-tweepy-python-using-emoji?rq=1 (helpful!)


from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
import pprint
from access_tokens import consumer_key, consumer_secret, access_token, access_secret

consumer_key = consumer_key
consumer_secret = consumer_secret
access_token = access_token
access_secret = access_secret

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
        # test path /Users/yibingyang/Documents/thesis_project_new/Data/Twitter/raw_data/test_negative_test.json
        # real path /Users/yibingyang/Documents/thesis_project_new/Data/Twitter/raw_data/negative_test.json
        f = open('/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/raw_data/negative_0407.json',"a+")
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

##negative
twitter_stream = Stream(auth, MyListener())
twitter_stream.filter(
                      track=[u"\U0001F612",
                             u"\U0001F613",
                             u"\U0001F614",
                             u"\U0001F615",
                             u"\U0001F616",
                             u"\U0001F61E",
                             u"\U0001F61F",
                             u"\U0001F620",
                             u"\U0001F621"
                             u"\U0001F622",
                             u"\U0001F623",
                             u"\U0001F624",
                             u"\U0001F625",
                             u"\U0001F626",
                             u"\U0001F627",
                             u"\U0001F628",
                             u"\U0001F629",
                             u"\U0001F62B",
                             u"\U0001F63E",
                             u"\U0001F63F"
                             ],
                      languages=['en']
                      )

