import requests
from bs4 import BeautifulSoup

def get_html(url) :
    html = ""
    resp = requests.get(url)
    if resp.status_code == 200:
        html = resp.text
    return html


URL = "https://search.naver.com/search.naver?sm=tab_hty.top&where=nexearch&ie=utf8&query="

print("Write the keyword")
keyword = input()

URL = URL + keyword

html = get_html(URL)
soup = BeautifulSoup(html, 'html.parser')

cup = soup.fine_all("kane")
print(len(cup))