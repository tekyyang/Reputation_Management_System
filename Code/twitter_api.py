__author__ = 'yyb'

#real-time tweets flows

import tweepy
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
import datetime


consumer_key = 'PtnaBvuJxCsltqFjSGkGBC1lN'
consumer_secret = '6H7uDkccYIi5EbhCU5YBaTFSF3lgm682eeoWyUT5erSdvCJktX'
access_token = '3226613022-CNZsjaXuaVUSpdjE5lGPm68s031XlreiosX0vIl'
access_secret = '7ivtcI5itWV9qMMjYFpiwrJDDX0ZfMNZrv6HP8Q0X8lez'

auth=OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_secret)

class MyListener(StreamListener):
    def on_data(self,data):
        n=1
        try:
            with open('/Users/yibingyang/Documents/final_thesis/Weed/weed_kw.json','a')as f:
                print()
                f.write(data)
                n+=1
                return True
        except BaseException as e:
            print 'Error on_data:%S'%str(e)

        return True

    def on_error(self,status):
        return True


twitter_stream = Stream(auth, MyListener())
twitter_stream.filter(track=['#legalizationday'])