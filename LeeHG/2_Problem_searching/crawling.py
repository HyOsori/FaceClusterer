import requests
from bs4 import BeautifulSoup
# from konlpy.tag import Okt
def get_html(headers, url) :
    html = ""
    resp = requests.get(url, headers = headers)
    if resp.status_code:
        print('url requests is OK')
        html = resp.text
    return html


URL = "https://search.naver.com/search.naver?sm=tab_hty.top&where=nexearch&ie=utf8&query="

print("Write the keyword")
keyword = input()

URL = URL + keyword
header = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36'}
html = get_html(header, URL)
soup = BeautifulSoup(html, 'html.parser')

cup = soup.find_all("div")
print(cup)