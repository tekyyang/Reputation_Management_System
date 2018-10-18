import requests
from bs4 import BeautifulSoup
import re
import datetime


class google_crawler():

    def __init__(self,keyword_input,save_path,**kwargs):
        self.keyword = self.get_keyword(keyword_input)
        self.kwargs = kwargs # cd_min ='10/16/2018', cd_max='10/18/2018'
        self.save_path = save_path
        self.keyword_for_save_path = self.get_keyword_for_save_path(keyword_input)
        # print 'keyword: ' + self.keyword
        # print 'save path: ' + self.save_path
        # print 'keyword for save path: '+self.save_path+ self.keyword_for_save_path+".json"

    def get_keyword(self,keyword_input):
        keyword_input = keyword_input  #a string that contains one or more words speparated by space e.g. 'shopify toronto montreal'
        all_words = keyword_input.split()
        if len(all_words) == 1:
            keyword =  keyword_input
        else:
            keyword = all_words[0]
            for i in range(1, len(all_words)):
                keyword = keyword + '+' + all_words[i]
        return keyword

    def get_keyword_for_save_path(self,keyword_input):
        keyword_input = keyword_input  #a string that contains one or more words speparated by space e.g. 'shopify toronto montreal'
        all_words = keyword_input.split()
        if len(all_words) == 1:
            keyword = keyword_input
        else:
            keyword = all_words[0]
            for i in range(1, len(all_words)):
                keyword = keyword + '_' + all_words[i]
        if len(self.kwargs) == 0:
            keyword_for_save_path = keyword + '_' + str(datetime.datetime.now())[:10].replace('-','_')
        else:
            keyword_for_save_path = keyword + '_' + str.split(self.kwargs['cd_max'],'/')[2]+'_'+ str.split(self.kwargs['cd_max'],'/')[0]+'_'+ str.split(self.kwargs['cd_max'],'/')[1]
        return keyword_for_save_path

    def get_the_content_for_a_single_page(self, response):
        html = response.text
        soup = BeautifulSoup(html, "html.parser")
        titles = [i.get_text() for i in soup.find_all('h3', attrs={'class': 'r'})] #10
        presses = [i.get_text() for i in soup.find_all('div', attrs={'class': 'slp'})] #10
        brief_intros = [i.get_text() for i in soup.find_all('div', attrs={'class': 'st'})] #10
        # links = [re.search(r'(http.+?)&sa', i.find('a').get('href')).group(1) for i in soup.find_all('h3', attrs={'class': 'r'})]
        links = [i.find('a').get('href') for i in soup.find_all('h3', attrs={'class': 'r'})]
        contents = []
        for i in links:
            r = requests.get(i)
            html = r.text
            result = re.findall(r'<p>(.+?)<\/p>', html)
            string = ''
            for i in range(0, len(result)):
                string += result[i]
            contents.append(string)

        for i in range(0, len(titles)):
            dict = {'title':titles[i],
                    'press':presses[i],
                    'intro':brief_intros[i],
                    'link':links[i],
                    'content': contents[i]},

            import json
            json = json.dumps(dict)[1:-1] + '\n' #cause dumps add a bracket to the dictionary [...]; it's okay but when we try to read it as dataframe, it's better not have the bracket.
            f = open(self.save_path+self.keyword_for_save_path+".json", "a+")
            f.write(json)
            f.close()
        count_news_per_page = len(titles)
        return count_news_per_page
        #some of the contents are not written in English. We need to deal with it in later analysis


    def multiple_pages_crawling(self):
        page_indices = [0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200] #starts from 1:15
        # page_indices = [0,10,20,30,40,50,60,70,80,90,100] #starts from 1:15
        collected_news = 0
        for page_index in page_indices:
            if len(self.kwargs) == 0:
                page_url = 'https://www.google.ca/search?q='+self.keyword+'&newwindow=1&rlz=1C5CHFA_enCA796CA796&tbs=qdr:d&tbm=nws&ei=mJG_W7jQO6PH_Qak_5T4BA&start='+str(page_index)+'&sa=N&biw=898&bih=1257&dpr=1' #start from 0(the first page), increment as 10 (second page)
            else:
                # page_url = 'https://www.google.ca/search?q='+self.keyword+'&newwindow=1&rlz=1C5CHFA_enCA796CA796&tbs=cdr:1,cd_min:10/16/2018,cd_max:10/18/2018&tbm=nws&ei=Fu7IW7n5K4zNjwSUgpCwAw&start='+str(page_index)+'&sa=N&biw=1440&bih=718&dpr=2'
                page_url = 'https://www.google.ca/search?q='+self.keyword+'&newwindow=1&rlz=1C5CHFA_enCA796CA796&tbs=cdr:1,cd_min:'+ self.kwargs['cd_min'] +',cd_max:'+self.kwargs['cd_max']+'&tbm=nws&ei=Fu7IW7n5K4zNjwSUgpCwAw&start='+str(page_index)+'&sa=N&biw=1440&bih=718&dpr=2'
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT x.y; Win64; x64; rv:10.0) Gecko/20100101 Firefox/10.0'}
            response = requests.get(page_url, headers=headers)
            if str(response) == '<Response [200]>':
                count_news_per_page = self.get_the_content_for_a_single_page(response)
                collected_news = collected_news + count_news_per_page
                print "collected news equals to " + str(collected_news) + "!"
            elif str(response) == '<Response [503]>':
                print str(response)
                break
            else:
                print 'not 200 or 503, check the page!'
                break

    def main(self):
        self.multiple_pages_crawling()



# google_crawler('legalization day canada', '/Users/yibingyang/Documents/final_thesis_project/Data/Google/').main()
# google_crawler('legalization day canada', '/Users/yibingyang/Documents/final_thesis_project/Data/Google/', cd_min ='10/14/2018', cd_max='10/15/2018').main()
# google_crawler('cannabis canada', '/Users/yibingyang/Documents/final_thesis_project/Data/Google/').main()

keywords = ['legalization day canada', 'cannabis canada']
dates = [('10/13/2018','10/14/2018'),
         ('10/14/2018','10/15/2018'),
         ('10/15/2018','10/16/2018'),
         ('10/16/2018','10/17/2018')]
for keyword in keywords:
    for day_scope in dates:
        google_crawler(keyword, '/Users/yibingyang/Documents/final_thesis_project/Data/Google/WEED/',
                       cd_min=day_scope[0], cd_max=day_scope[1]).main()

