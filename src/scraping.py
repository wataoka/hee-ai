import urllib
from bs4 import BeautifulSoup

URL = "https://ja.wikipedia.org/wiki/%E3%83%88%E3%83%AA%E3%83%93%E3%82%A2%E3%81%AE%E6%B3%89_%E3%80%9C%E7%B4%A0%E6%99%B4%E3%82%89%E3%81%97%E3%81%8D%E3%83%A0%E3%83%80%E7%9F%A5%E8%AD%98%E3%80%9C"

def get_text(tag):
    text = tag.text
    text = text.replace('\n', '')
    text = text.replace('[18]', '')
    text = text.replace('[19]', '')
    text = text.replace('[20]', '')
    text = text.replace('[21]', '')
    text = text.replace('[22]', '')
    return text

if __name__ == "__main__":

    html = urllib.request.urlopen(URL)
    soup = BeautifulSoup(html, 'html.parser')

    trivia_table = soup.find('table', attrs={'class': 'sortable'})

    trivias_list = []
    for i, line in enumerate(trivia_table.tbody):

        if i < 3:
            continue
        if line == '\n':
            continue
        
        id = line.find('th')
        content, hee, man_hee = line.find_all('td')

        id, content, hee, man_hee = map(get_text, [id, content, hee, man_hee])

        if hee == '?':
            continue
        
        trivias_list.append({'id': id, 'content': content, 'hee': int(hee), 'man_hee': int(man_hee)})
    
    print(trivias_list)