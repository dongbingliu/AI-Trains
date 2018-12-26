#!usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:Jerry Liu
@file: WordToVec.py
@version:
@time: 2018/12/16 
@email:liudongbing8@163.com
@function：文字转换为数字编码
"""


class WordToVector():
    # 重要 one-hot 编码
    def oneHotWord2Vector(self, all_list):
        from sklearn.feature_extraction.text import CountVectorizer
        count_vect = CountVectorizer(analyzer="word", stop_words=self.stop_word, ngram_range=(1, 1), min_df=5,
                                     max_df=100,
                                     tokenizer=lambda doc: doc, lowercase=False)
        arrayVector = count_vect.fit_transform(all_list).toarray()
        print("oneHot arrayVector.shape = ", arrayVector.shape)
        return arrayVector

    # 重要 TF-IDF 编码 与 one_hot 编码二选一，推荐使用 TF-IDF

    def tfWord2Vector(self, all_list):
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf = TfidfVectorizer(ngram_range=(1, 1), min_df=5, stop_words=self.stop_word, tokenizer=lambda doc: doc,
                                lowercase=False)
        arrayVector = tfidf.fit_transform(all_list).toarray()
        print("TF-IDF arrayVector.shape = ", arrayVector.shape)
        # for key, value in tfidf.vocabulary_.items():
        #     print("key=", key, ";value=", value)
        return arrayVector

    def __init__(self):
        self.stop_word = []
        print("=========== init =======")
        print("WordtoVector main ===")
        # ONE-HOT 标签转化
        stop = open('datas/stop_words_english.txt', 'r', encoding='utf-8')
        # stop_word = [',','.','!','#']
        for ele in stop:
            stop_word = []
            ele = ele.strip()
            stop_word.append(ele)
