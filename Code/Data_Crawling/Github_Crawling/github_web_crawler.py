# import requests
# headers = {"Authorization": "token 59439ca09a4f8cb1543cfa5af7167961e57a8221", 'User-Agent':'Mozilla/5.0 (Windows NT x.y; Win64; x64; rv:10.0) Gecko/20100101 Firefox/10.0'}
# response = requests.post(url, headers=headers)
# print response

import requests
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import re
from bs4 import BeautifulSoup

url = 'https://github.com/'
browser = webdriver.Chrome(executable_path=r"/Users/yibingyang/Documents/final_thesis_project/Project/Code/chromedriver")
browser.implicitly_wait(10)
browser.get(url)
time.sleep(5)