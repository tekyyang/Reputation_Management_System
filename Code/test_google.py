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

import re
from bs4 import BeautifulSoup
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


cd_max='10/18/2018'
print str.split(cd_max,'/')