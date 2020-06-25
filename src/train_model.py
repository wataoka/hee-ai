import re
from tqdm import tqdm

import MeCab
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from data.trivias_list import trivias_list

# td: trivias_list
def normalize_hee(td):
    for i in range(len(td)):
        td[i]['norm_hee'] = td[i]['hee'] / td[i]['man_hee']
    return td

def wakati(text):
    node = tagger.parseToNode(text)
    l = []
    while node:
        if node.feature.split(',')[6] != '*':
            l.append(node.feature.split(',')[6])
        else:
            l.append(node.surface)
        node = node.next
    return ' '.join(l)

def preprocess(td):
    td = normalize_hee(td)
    for i in tqdm(range(len(td))):
        td[i]['wakati_content'] = wakati(td[i]['content'])
    return td


if __name__ == '__main__':

    # Mecab
    tagger = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
    tagger.parse('')

    trivias_list = preprocess(trivias_list)
    trivias_df = pd.DataFrame(trivias_list)

    wakati_contents_list = trivias_df['wakati_content'].values.tolist()

    word_vectorizer = TfidfVectorizer(max_features=50000)
    word_vectorizer.fit(wakati_contents_list)

    tfidf = word_vectorizer.transform(wakati_contents_list)
    train_df = pd.concat([
        trivias_df,
        pd.DataFrame(tfidf.toarray())
    ], axis=1)

    print(train_df.head())