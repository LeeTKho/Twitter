#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords

# reading csv files into pandas dataframes
hist_BK = pd.read_csv("historical_search_BK.csv")
hist_SI = pd.read_csv("historical_search_SI.csv")

# countvectorizing BK tweets

vectorizer = CountVectorizer(ngram_range=(1,3))

X = vectorizer.fit_transform(hist_BK['tweet'])

col_names = vectorizer.get_feature_names()

col_names.insert(0,'user_id')

user_ids = pd.DataFrame(hist_BK['user_id'],columns=['user_id'])

tweets = pd.DataFrame(X.toarray(),columns=vectorizer.get_feature_names())

# build df with BK user_ids and countvectorized tweets
df_BK = pd.concat([user_ids,tweets],axis=1)

# import list of general stopwords
stopwords = stopwords.words()

# create list of stopwords found in the BK tweets
stopwords_BK = list(set(df_BK.columns) & set(stopwords))

# remove stopwords from countvectorized BK tweets
df_BK = df_BK.drop(stopwords_BK,axis=1)


# In[4]:


df_BK.iloc[:,1:].sum(axis=0).sort_values(ascending=False)[:50]


# In[6]:


# countvectorizing SI tweets

vectorizer = CountVectorizer(ngram_range=(1,3))

X = vectorizer.fit_transform(hist_SI['tweet'])

col_names = vectorizer.get_feature_names()

col_names.insert(0,'user_id')

user_ids = pd.DataFrame(hist_SI['user_id'],columns=['user_id'])

tweets = pd.DataFrame(X.toarray(),columns=vectorizer.get_feature_names())

# build df with user_ids and countvectorized SI tweets
df_SI = pd.concat([user_ids,tweets],axis=1)

# create list of stopwords found in the SI tweets
stopwords_SI = list(set(df_SI.columns) & set(stopwords))

# remove stopwords from countvectorized SI tweets
df_SI = df_SI.drop(stopwords_SI,axis=1)


# In[7]:


df_SI.iloc[:,1:].sum(axis=0).sort_values(ascending=False)[:50]


# In[ ]:




