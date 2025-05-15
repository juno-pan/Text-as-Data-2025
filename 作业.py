#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 15 08:21:59 2025

@author: once
"""

#导入数据
import pandas as pd
text_ana= pd.read_excel('text_analysis_twitter_sample.xlsx')
text_ana
text_ana= text_ana.drop(['Unnamed: 0'],axis=1)
#drop去除这一列 axis=1 按列找 axis=0 按行找
text_ana

#数据预处理
import re
def preprocess_text(text):
    #转换大小写
    text=text.lower()
    #移除URL
    text=re.sub(r'http\S+','',text)
    #注意S要大写
    #移除停用词
    stop_words=set(['i','me','my','myself','to','com','http','in','of','for','and','with','the','on','is','this','you','we','be','it','your'])
    text=' '.join([word for word in text.split()if word not in stop_words])
    
    return text

text_ana['clean_text'] = text_ana['text'].apply(preprocess_text)

#构建词袋模型
from sklearn.feature_extraction.text import CountVectorizer

#初始化Countvectorizer
vectorizer = CountVectorizer()

#将文本数据转换为词袋特征矩阵
X=vectorizer.fit_transform(text_ana["clean_text"].apply(preprocess_text))

print('词袋特征矩阵的形状：', X.shape)

#展示词袋模型中的所有特征词
print('特征词列表：', vectorizer.get_feature_names_out()[:100])
#[:100]取前一百个

#词袋特征矩阵的形状： (500, 4509) (行，列)
#特征词列表： ['00' '000' '00am' ... 'zone' 'zvss30ftwn5' 'пенсионер']
#调用别人写好的函数CountVectorizer


#词频统计
#获取词汇表
feature_words=vectorizer.get_feature_names_out()

#计算词频统计
word_freq=dict(zip(feature_words, X.sum(axis=0).A1))
#zip:对应词语和词频 把words和按行sum的结果拉到一起
print ('特征词频词典', word_freq)

#观察词袋模型的词频计数结果
sorted (word_freq.items(),key=lambda x:x[1],reverse=True)
# x[1] 取建值对的第二位，reverse=True 从大到小排列 reverse=False 从小到大排列
print ('词袋模型的词频计数', word_freq.items())

#词云图
from wordcloud import WordCloud
import matplotlib.pyplot as plt

#根据词袋模型分析结果绘制词云图
wc = WordCloud(
    width=1200,height=800
    , max_words=200
    #, max_font_size=100
    , colormap='viridis'
    , background_color='white'
    #, font_path='arial.ttf'
). generate_from_frequencies(word_freq)

#显示词云图
plt.figure(figsize=(10,5))
plt.imshow(wc,interpolation='bilinear')
plt.axis('off')
#plt.savefig('wordcloud_twi.png', dpi=100)
plt.show()