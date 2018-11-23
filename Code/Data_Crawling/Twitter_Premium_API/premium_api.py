#using python 3 in this file for this package is not supported in python 2
#https://github.com/twitterdev/search-tweets-python/issues/1
#The following code refers to http://benalexkeen.com/interacting-with-the-twitter-api-using-python/ neat!

from searchtweets import load_credentials
import base64
import requests

#
# consumer_key = "S5Xu1dAbaWzpVXESPVzS1o5wl"
# consumer_secret = "xs2U9hnXuWDOpctU7fXHklTwVSPb5lZHfrNN2pIQleybEThh9I"
# s = consumer_key+ ':' + consumer_secret
# b64_encoded_key = base64.b64encode(s.encode('utf-8'))
# b64_encoded_key = b64_encoded_key.decode('ascii')
# url = "https://api.twitter.com/oauth2/token"
# auth_headers = {
#   "Authorization":'Basic {}'.format(b64_encoded_key),
#   "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
#   "grant_type" : "client_credentials"
# }
#
# auth_data = {
#     'grant_type': 'client_credentials'
# }
#
# auth_resp = requests.post(url, headers=auth_headers, data=auth_data)
# access_token = auth_resp.json()['access_token']
# print(access_token)
# #AAAAAAAAAAAAAAAAAAAAAMk82QAAAAAAOBzeI7K05TV9b2XKJ%2FDt0UyGLkI%3D4Z9Cd9fPuDr6MtB25gdNj63kwCdXYYRctFBXl0DYVJm78JtDrb
# # print(load_credentials(filename="./twitter_keys.yaml",
# #                  yaml_key="search_tweets_api",
# #                  env_overwrite=False))

### The goal is to search for the previous 30 days records of the keywords for a branc





# base_url = 'https://api.twitter.com/'
# search_url = '{}1.1/search/tweets.json'.format(base_url)
# # '1.1/search/tweets.json'
# # '1.1/tweets/search/30day/my_env_name.json'
#
# access_token = 'AAAAAAAAAAAAAAAAAAAAAMk82QAAAAAAOBzeI7K05TV9b2XKJ%2FDt0UyGLkI%3D4Z9Cd9fPuDr6MtB25gdNj63kwCdXYYRctFBXl0DYVJm78JtDrb'
# search_headers = {
#     'Authorization': 'Bearer {}'.format(access_token)
# }
#
# search_params = {
#     'q': 'General Election',
#     # 'result_type': 'recent',
#     'count': 5
# }
#
# search_resp = requests.get(search_url, headers=search_headers, params=search_params)
# tweet_data = search_resp.json()
# print(tweet_data)

import pprint
from searchtweets import ResultStream, gen_rule_payload, load_credentials, collect_results

premium_search_args = load_credentials(filename="./twitter_keys.yaml",
                 yaml_key="search_tweets_30_day",
                 env_overwrite=False)


rule = gen_rule_payload("beyonce <🔥>",
                        results_per_call=2,
                        from_date="2018-11-01",
                        to_date="2018-11-10") # testing with a sandbox account

rs = ResultStream(rule_payload=rule,
                  max_results=5,
                  max_pages=1,
                  max_requests=2,
                  **premium_search_args)

tweets = list(rs.stream())


pp = pprint.PrettyPrinter(depth=6)
pp.pprint(tweets)














