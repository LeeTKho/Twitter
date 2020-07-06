#!/usr/bin/env python
# coding: utf-8

# In[156]:


import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk


# In[157]:


# reading pickle files into pandas dataframes
BK_tweets = pd.read_pickle("./BK_tweets.pkl")
SI_tweets = pd.read_pickle("./SI_tweets.pkl")


# ### CountVectorizer analysis - Brooklyn

# In[96]:


from nltk.corpus import stopwords

BK_stopwords = ['www','https','https twitter','https twitter com','twitter',
                'pic twitter','pic twitter com','twitter com',
                'https www', 'https www instagram','instagram','instagram com',
                'www instagram','www instagram com','at the','in the','of the',
                'the protest','if you','this is','on the','to the','you re']

# filtering out tweets that got fewer than 10 likes to make processing easier
BK_tweets = BK_tweets[BK_tweets.nlikes>=10]

# countvectorizing BK tweets
vectorizer = CountVectorizer(ngram_range=(1,3))

X = vectorizer.fit_transform(BK_tweets['tweet'])

user_ids = list(BK_tweets['user_id'])

countvec_BK = pd.DataFrame(X.toarray(),columns=vectorizer.get_feature_names())

# build df with BK user_ids and countvectorized tweets
countvec_BK.insert(0, "user_id", user_ids, False) 

# import list of general stopwords
stopwords = stopwords.words()

# create list of stopwords found in the BK tweets
stopwords = list(set(countvec_BK.columns) & set(stopwords))+BK_stopwords

# remove stopwords from countvectorized BK tweets
countvec_BK = countvec_BK.drop(stopwords,axis=1)


# In[97]:


print(countvec_BK.shape)
countvec_BK.head()


# In[117]:


pd.options.display.max_colwidth = 300

BK_tweets[BK_tweets['tweet'].str.contains("officer")].tweet.shape


# In[127]:


pd.options.display.max_rows = 4000

print('Number of BK tweets: '+str(countvec_BK.shape[0]))
countvec_BK.iloc[:,1:].sum(axis=0).sort_values(ascending=False)[:100]


# ### CountVectorizer analysis - Staten Island

# In[161]:


from nltk.corpus import stopwords

SI_stopwords = ['twitter','twitter com','https','https twitter','https twitter com',
                'pic twitter','pic twitter com','in the','if you','you re','of the','petition http',
                'petition http chng','http chng','http chng it']

# countvectorizing SI tweets
vectorizer = CountVectorizer(ngram_range=(1,3))

X = vectorizer.fit_transform(SI_tweets['tweet'])

user_ids = list(SI_tweets['user_id'])

countvec_SI = pd.DataFrame(X.toarray(),columns=vectorizer.get_feature_names())

# build df with SI user_ids and countvectorized tweets
countvec_SI.insert(0, "user_id", user_ids, False) 

# import list of general stopwords
stopwords = stopwords.words()

# create list of stopwords found in the SI tweets
stopwords = list(set(countvec_SI.columns) & set(stopwords))+SI_stopwords

# remove stopwords from countvectorized SI tweets
countvec_SI = countvec_SI.drop(stopwords,axis=1)


# In[162]:


print(countvec_SI.shape)
countvec_SI.head()


# In[163]:


pd.options.display.max_rows = 4000

print('Number of SI tweets: '+str(countvec_SI.shape[0]))
countvec_SI.iloc[:,1:].sum(axis=0).sort_values(ascending=False)[:100]


# ### TF-IDF vectorizer analysis - Brooklyn

# In[85]:


from nltk.corpus import stopwords

BK_stopwords = ['www','https','https twitter','https twitter com','twitter',
                'pic twitter','pic twitter com','twitter com',
                'https www', 'https www instagram','instagram','instagram com',
                'www instagram','www instagram com','at the','in the','of the',
                'the protest','if you','this is','on the','to the']

# filtering out tweets that got fewer than 10 likes to make processing easier
BK_tweets = BK_tweets[BK_tweets.nlikes>=10]

# tfidf vectorizing BK tweets
vectorizer = TfidfVectorizer(ngram_range=(1,3))

X = vectorizer.fit_transform(BK_tweets['tweet'])

user_ids = list(BK_tweets['user_id'])

tfidf_BK = pd.DataFrame(X.toarray(),columns=vectorizer.get_feature_names())

# build df with BK user_ids and countvectorized tweets
tfidf_BK.insert(0, "user_id", user_ids, False) 

# import list of general stopwords
stopwords = stopwords.words()

# create list of stopwords found in the BK tweets
stopwords = list(set(tfidf_BK.columns) & set(stopwords))+BK_stopwords

# remove stopwords from countvectorized BK tweets
tfidf_BK = tfidf_BK.drop(stopwords,axis=1)


# In[86]:


tfidf_BK.head()


# In[93]:


pd.options.display.max_rows = 4000

print('Number of BK tweets: '+str(tfidf_BK.shape[0]))

# sorting keyword terms by average tfidf score (including zeros)
tfidf_BK.iloc[:,1:].mean(axis=0).sort_values(ascending=False)[:100]


# ### TF-IDF vectorizer analysis - Staten Island

# In[88]:


from nltk.corpus import stopwords

SI_stopwords = ['twitter','twitter com','https','https twitter','https twitter com',
                'pic twitter','pic twitter com','in the','if you','you re','of the']

# tfidf vectorizing BK tweets
vectorizer = TfidfVectorizer(ngram_range=(1,3))

X = vectorizer.fit_transform(SI_tweets['tweet'])

user_ids = list(SI_tweets['user_id'])

tfidf_SI = pd.DataFrame(X.toarray(),columns=vectorizer.get_feature_names())

# build df with BK user_ids and countvectorized tweets
tfidf_SI.insert(0, "user_id", user_ids, False) 

# import list of general stopwords
stopwords = stopwords.words()

# create list of stopwords found in the BK tweets
stopwords = list(set(tfidf_SI.columns) & set(stopwords))+SI_stopwords

# remove stopwords from countvectorized BK tweets
tfidf_SI = tfidf_SI.drop(stopwords,axis=1)


# In[89]:


tfidf_SI.head()


# In[95]:


pd.options.display.max_rows = 4000

print('Number of SI tweets: '+str(tfidf_SI.shape[0]))

# sorting keyword terms by average tfidf score (including zeros)
tfidf_SI.iloc[:,1:].mean(axis=0).sort_values(ascending=False)[:100]

