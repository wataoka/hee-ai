import re
from tqdm import tqdm

import MeCab
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Mecab
tagger = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
tagger.parse('')

# lambda
re_hira = re.compile(r'^[あ-ん]+$')
re_kata = re.compile(r'[\u30A1-\u30F4]+')
re_kanj = re.compile(r'^[\u4E00-\u9FD0]+$')
re_eigo = re.compile(r'^[a-zA-Z]+$')
is_hira = lambda word: not re_hira.fullmatch(word) is None
is_kata = lambda word: not re_kata.fullmatch(word) is None
is_eigo = lambda word: not re_eigo.fullmatch(word) is None
is_kanj = lambda word: not re_kanj.fullmatch(word) is None

# tl: trivias_list
def normalize_hee(tl):
    for i in range(len(tl)):
        tl[i]['norm_hee'] = tl[i]['hee'] / tl[i]['man_hee']
    return tl

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

def preprocess(tl):
    tl = normalize_hee(tl)
    for i in tqdm(range(len(tl))):
        tl[i]['wakati_content'] = wakati(tl[i]['content'])
    return tl

def count_len(sentence):
    return len(sentence)
def count_word(sentence):
    return len(sentence.split(' '))
def count_kata(sentence):
    cnt = 0; total=0
    for word in sentence.split(' '):
        if word == '': continue
        total += 1
        if is_kata(word): cnt += 1
    return cnt/total
def count_hira(sentence):
    cnt = 0; total=0
    for word in sentence.split(' '):
        if word == '': continue
        total += 1
        if is_hira(word): cnt += 1
    return cnt/total
def count_eigo(sentence):
    cnt = 0; total=0
    for word in sentence.split(' '):
        if word == '': continue
        total += 1
        if is_eigo(word): cnt += 1
    return cnt/total
def count_kanj(sentence):
    cnt = 0; total=0
    for word in sentence.split(' '):
        if word == '': continue
        total += 1
        if is_kanj(word): cnt += 1
    return cnt/total

def get_features(trivias_list, content=None, mode='learn'):

    trivias_list = preprocess(trivias_list)
    trivias_df = pd.DataFrame(trivias_list)
    
    wakati_contents_list = trivias_df['wakati_content'].values.tolist()

    word_vectorizer = TfidfVectorizer(max_features=5)
    word_vectorizer.fit(wakati_contents_list)

    if mode == 'inference':
        content = [{'content': content, 'wakati_content': wakati(content)}]
        content_df = pd.DataFrame(content)

        wakati_content_list = content_df['wakati_content'].values.tolist()
        tfidf = word_vectorizer.transform(wakati_content_list)
        content_df = pd.concat([
            content_df,
            pd.DataFrame(tfidf.toarray())
        ], axis=1)
        num_len_df = content_df['wakati_content'].map(count_len)
        num_word_df = content_df['wakati_content'].map(count_word)
        num_hira_df = content_df['wakati_content'].map(count_hira)
        num_kata_df = content_df['wakati_content'].map(count_kata)
        num_eigo_df = content_df['wakati_content'].map(count_eigo)
        num_kanj_df = content_df['wakati_content'].map(count_kanj)
        content_df['num_len'] = num_len_df.values.tolist()
        content_df['num_word'] = num_word_df.values.tolist()
        content_df['num_hira'] = num_hira_df.values.tolist()
        content_df['num_kata'] = num_kata_df.values.tolist()
        content_df['num_eigo'] = num_eigo_df.values.tolist()
        content_df['num_kanj'] = num_kanj_df.values.tolist()

        content_df = content_df.drop('content', axis=1)
        content_df = content_df.drop('wakati_content', axis=1)

        return content_df


    tfidf = word_vectorizer.transform(wakati_contents_list)
    all_df = pd.concat([
        trivias_df,
        pd.DataFrame(tfidf.toarray())
    ], axis=1)

    num_len_df = all_df['wakati_content'].map(count_len)
    num_word_df = all_df['wakati_content'].map(count_word)
    num_hira_df = all_df['wakati_content'].map(count_hira)
    num_kata_df = all_df['wakati_content'].map(count_kata)
    num_eigo_df = all_df['wakati_content'].map(count_eigo)
    num_kanj_df = all_df['wakati_content'].map(count_kanj)
    all_df['num_len'] = num_len_df.values.tolist()
    all_df['num_word'] = num_word_df.values.tolist()
    all_df['num_hira'] = num_hira_df.values.tolist()
    all_df['num_kata'] = num_kata_df.values.tolist()
    all_df['num_eigo'] = num_eigo_df.values.tolist()
    all_df['num_kanj'] = num_kanj_df.values.tolist()

    if mode == 'learn':
        all_df = all_df.drop('id', axis=1)
        all_df = all_df.drop('hee', axis=1)
        all_df = all_df.drop('man_hee', axis=1)
        all_df = all_df.drop('content', axis=1)
        all_df = all_df.drop('wakati_content', axis=1)

    return all_df