# class a_test():
#     def __init__(self):
#         pass
#
#     def first_f(self):
#         sum_n = 1+1
#         return sum_n
#
#     def second_f(self):
#         value = self.first_f()
#         print value
#
#     def main(self):
#         self.second_f()
#
#
# a_test().main()


#
# text = '''
# <h3 class="r">
# <a href="/url?q=https://www.capebretonpost.com/business/shopify-opens-first-brick-and-mortar-spot-with-goal-of-helping-entrepreneurs-249100/&amp;sa=U&amp;ved=0ahUKEwj3ha7X_P7dAhWM44MKHX2nAhAQqQIIFygAMAA&amp;usg=AOvVaw2GO7YJpQuTG0KL18KEGTt4" target="_blank">
#  <b>
#   Shopify
#  </b>
#  opens first brick-and-mortar spot with goal of helping ...
# </a>
# </h3>
#
# <h3 class="r">
# <a href="22/url?q=https://www.capebretonpost.com/business/shopify-opens-first-brick-and-mortar-spot-with-goal-of-helping-entrepreneurs-249100/&amp;sa=U&amp;ved=0ahUKEwj3ha7X_P7dAhWM44MKHX2nAhAQqQIIFygAMAA&amp;usg=AOvVaw2GO7YJpQuTG0KL18KEGTt4" target="_blank">
#  <b>
#   Shopify
#  </b>
#  opens first brick-and-mortar spot with goal of helping ...
# </a>
# </h3>
# '''
# soup = BeautifulSoup(text, "html.parser")
#
# items = [i.get_text() for i in soup.find_all('h3', attrs={'class': 'r'})]
# print items



# items = [str(i.find('a').get('href')) for i in soup.find_all('h3', attrs={'class': 'r'})]
# items = [re.search(r'(http.+?)&amp', BeautifulSoup(i.find('a').get('href'))).group(1) for i in soup.find_all('h3', attrs={'class': 'r'})]

# items = soup.find_all('h3', attrs={'class': 'r'})
# for i in items:
#     r = i.find('a').get('href')
#     r = str(r)
#     r2 = re.search(r'(http.+?)&sa', r).group(1)
#     print r2

#
# href = "/url?q=https://www.capebretonpost.com/business/shopify-opens-first-brick-and-mortar-spot-with-goal-of-helping-entrepreneurs-249100/&amp;sa=U&amp;ved=0ahUKEwj3ha7X_P7dAhWM44MKHX2nAhAQqQIIFygAMAA&amp;usg=AOvVaw2GO7YJpQuTG0KL18KEGTt4"
# print type(href)
# result = re.search(r'(http.+?)&amp', href).group(1)
# print result
# drizzy
#
import requests
from bs4 import BeautifulSoup
import re
#
# url = 'https://www.capebretonpost.com/business/shopify-opens-first-brick-and-mortar-spot-with-goal-of-helping-entrepreneurs-249100/'
# r = requests.get(url)
# html = r.text
# result = re.findall(r'<p>(.+?)<\/p>', html)
# string = ''
# for i in range(0, len(result)):
#     string += result[i]
#
# print string.replace
# import datetime
# print str(datetime.datetime.now())[:10].replace('-','_')

# import requests
# url = 'https://www.google.ca/search?q=shopify&newwindow=1&rlz=1C5CHFA_enCA796CA796&tbs=qdr:d&tbm=nws&ei=mJG_W7jQO6PH_Qak_5T4BA&start=0&sa=N&biw=898&bih=1257&dpr=1'
# response = requests.get(url)
# html = response.text
# soup = BeautifulSoup(html, "html.parser")
# print soup.prettify()
# link = 'https://www.google.ca/search?q=legalization+day+canada&newwindow=1&rlz=1C5CHFA_enCA796CA796&tbs=qdr:d&tbm=nws&ei=mJG_W7jQO6PH_Qak_5T4BA&start=0&sa=N&biw=898&bih=1257&dpr=1'
# r = requests.get(link)
# html = r.text
#
#
# test = '''
# <h3 class="r dO0Ag"><a class="l lLrAF" href="https://www.cbc.ca/news/canada/british-columbia/anti-pot-legalization-1.4867712" onmousedown="return rwt(this,'','','','1','AOvVaw0509qRPhPaN6cdWsCESZJJ','','0ahUKEwjrpI2355DeAhVrs1QKHYArBZYQqQIIJigAMAA','','',event)" target="_blank">A 'dark <em>day</em> for <em>Canada</em>,' say anti-pot activists</a></h3>
# <h3 class="r dO0Ag"><a class="l lLrAF" href="https://www.cbc.ca/news/canada/british-columbia/anti-pot-legalization-1.4867712" onmousedown="return rwt(this,'','','','1','AOvVaw0509qRPhPaN6cdWsCESZJJ','','0ahUKEwjrpI2355DeAhVrs1QKHYArBZYQqQIIJigAMAA','','',event)" target="_blank">A 'dark <em>day</em> for <em>Canada</em>,' say anti-pot activists</a></h3>
# '''
# soup = BeautifulSoup(test, "html.parser")
# raw_links = soup.find_all('h3', attrs={'class': 'r'})
# for i in raw_links:
#     print i.find('a').get('href')
# raw_links = [re.search(r'(http.+?)&sa', i.find('a').get('href')).group(1) for i in soup.find_all('h3', attrs={'class': 'r'})]
# print raw_links
# print re.search(r'(http.+?)&sa' ,string).group(1)
#
# dic = {'a':1}
# print len(dic)


# cd_max='10/18/2018'
# print str.split(cd_max,'/')

#
# import re
# from bs4 import BeautifulSoup
# import urllib2
#
# link = 'https://www.furniture.ca/collections/furniture-living-room-sofas'
# r = urllib2.urlopen(link)
# html = r.read()
# soup = BeautifulSoup(html, "html.parser")
# print soup.prettify()
#
#
# for i in range(1,10):
#     locals()['cursor' + '_' + str(i+1)]=i+1
# print cursor_2
#
# import pandas as pd
# url = '/Users/yibingyang/Documents/5900X/ios_pr_test.json'
# df = pd.read_json(url,orient = 'records', lines = True)

#
# dict_test =\
# {'reviews':
# [
#                     {
#                       "author": {
#                         "login": "p-sun"
#                       },
#                       "bodyText": "",
#                       "comments": {
#                         "nodes": [
#                           {
#                             "bodyText": "Added this method that only returns true to make sure that the dev implementing will override canBecomeFirstResponder",
#                             "reactions": {
#                               "viewerHasReacted": False,
#                               "nodes": []
#                             }
#                           }
#                         ]
#                       },
#                       "createdAt": "2017-05-17T14:24:18Z",
#                       "state": "COMMENTED"
#                     },
#
#
#                     {
#                       "author": {
#                         "login": "Caiopia"
#                       },
#                       "bodyText": "",
#                       "comments": {
#                         "nodes": [
#                           {
#                             "bodyText": "One thing we should think about is that implementers of Copyable here might forget to override these functions and call the each of the overrideCopy(), overrideCanPerformAction(action) and overridecanBecomeFirstResponder()` functions inside them.\nWhile it's pretty simple for us to copy/paste since we know about it and can look at other implementations, I'm thinking it might be less straightforward for someone first seeing the Copyable protocol.",
#                             "reactions": {
#                               "viewerHasReacted": False,
#                               "nodes": []
#                             }
#                           }
#                         ]
#                       },
#                       "createdAt": "2017-05-18T03:43:42Z",
#                       "state": "COMMENTED"
#                     },
#
#
#
#                     {
#                       "author": {
#                         "login": "krbarnes"
#                       },
#                       "bodyText": "Lets have a discussion IRL. Trying to parse/discuss competing proposals in different PRs isn't a productive way to discuss this issue  #167 and #169",
#                       "comments": {
#                         "nodes": []
#                       },
#                       "createdAt": "2017-05-18T13:32:43Z",
#                       "state": "CHANGES_REQUESTED"
#                     }
#                   ]
# }
#
#
# for i in dict_test['reviews']:
#   for j in i['comments']['nodes']:
#     print j['bodyText']

# import pprint
# dict = \
#   {"shop_id": "8960700", "pod_id": "28", "shard_id": "28", "edge_location": "chi2", "request_served_from": "chi2",
#    "event_at": "2017-09-26T17:30:00Z", "processing_time_seconds_sum": "0 0 0 0 0 0 0 0 168 150 ",
#    "app_server_processing_time_seconds_sum": "NULL", "request_count": "119", "app_server_request_count": "119",
#    "load_balancer_request_count": "0", "_fact_key": "SRBjgtxXIeuZdextmv5Q5", "_audit_key": "71092818245607370846"}
# pprint.pprint(dict)


# text = 'iphone'
# print unicode(text)

# test_str = "i won't do that, he won't do that too"
# print test_str.replace("won't","will not")

# print len('\u2026')

a = [0]*5
print a