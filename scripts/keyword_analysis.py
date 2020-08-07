#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk


# In[2]:


# reading pickle files into pandas dataframes
BK_tweets = pd.read_pickle("./BK_tweets.pkl")
SI_tweets = pd.read_pickle("./SI_tweets.pkl")


# ### CountVectorizer analysis - Brooklyn

# In[4]:


from nltk.corpus import stopwords

BK_stopwords = ['www','https','https twitter','https twitter com','twitter',
                'pic twitter','pic twitter com','twitter com',
                'https www', 'https www instagram','instagram','instagram com',
                'www instagram','www instagram com','at the','in the','of the',
                'the protest','if you','this is','on the','to the','you re']

# filtering out tweets that got fewer than 10 likes to make processing easier
BK_tweets = BK_tweets[BK_tweets.nlikes>=1]

# countvectorizing BK tweets
vectorizer = CountVectorizer(ngram_range=(1,3))

X = vectorizer.fit_transform(BK_tweets['tweet'])

user_ids = list(BK_tweets['user_id'])

BK_countvec = pd.DataFrame(X.toarray(),columns=vectorizer.get_feature_names())

# build df with BK user_ids and countvectorized tweets
BK_countvec.insert(0, "user_id", user_ids, False) 

# import list of general stopwords
stopwords = stopwords.words()

# create list of stopwords found in the BK tweets
stopwords = list(set(BK_countvec.columns) & set(stopwords))+BK_stopwords

# remove stopwords from countvectorized BK tweets
BK_countvec = BK_countvec.drop(stopwords,axis=1)


# In[5]:


print(BK_countvec.shape)
BK_countvec.head()


# In[6]:


pd.options.display.max_rows = 4000

print('Number of BK tweets: '+str(BK_countvec.shape[0]))
BK_countvec.iloc[:,1:].sum(axis=0).sort_values(ascending=False)[:100]


# ### CountVectorizer analysis - Staten Island

# In[17]:


from nltk.corpus import stopwords

SI_stopwords = ['twitter','twitter com','https','https twitter','https twitter com',
                'pic twitter','pic twitter com','in the','if you','you re','of the','petition http',
                'petition http chng','http chng','http chng it','www','https www','instagram',
                'https www instagram','www instagram','www instagram com','http','instagram com']

# countvectorizing SI tweets
vectorizer = CountVectorizer(ngram_range=(1,3))

X = vectorizer.fit_transform(SI_tweets['tweet'])

user_ids = list(SI_tweets['user_id'])

SI_countvec = pd.DataFrame(X.toarray(),columns=vectorizer.get_feature_names())

# build df with SI user_ids and countvectorized tweets
SI_countvec.insert(0, "user_id", user_ids, False) 

# import list of general stopwords
stopwords = stopwords.words()

# create list of stopwords found in the SI tweets
stopwords = list(set(SI_countvec.columns) & set(stopwords))+SI_stopwords

# remove stopwords from countvectorized SI tweets
SI_countvec = SI_countvec.drop(stopwords,axis=1)


# In[18]:


print(SI_countvec.shape)
SI_countvec.head()


# In[19]:


pd.options.display.max_rows = 4000

print('Number of SI tweets: '+str(SI_countvec.shape[0]))
SI_countvec.iloc[:,1:].sum(axis=0).sort_values(ascending=False)[:100]


# In[20]:


pd.options.display.max_colwidth = 300

print('BK tweets')
print('number of times \'cop\' appears: '+str(BK_tweets[BK_tweets['tweet'].str.contains(" cop ")].tweet.shape[0]+
                                             BK_tweets[BK_tweets['tweet'].str.contains(" cops ")].tweet.shape[0]-
                                             BK_tweets[BK_tweets['tweet'].str.contains(" cop ")][BK_tweets['tweet'].str.contains(" cops ")].shape[0]))
print('number of times \'officer\' appears: '+str(BK_tweets[BK_tweets['tweet'].str.contains("officer")].tweet.shape[0]))

print('SI tweets')
print('number of times \'cop\' appears: '+str(SI_tweets[SI_tweets['tweet'].str.contains(" cop ")].tweet.shape[0]+
                                             SI_tweets[SI_tweets['tweet'].str.contains(" cops ")].tweet.shape[0]-
                                             SI_tweets[SI_tweets['tweet'].str.contains(" cop ")][SI_tweets['tweet'].str.contains(" cops ")].shape[0]))
print('number of times \'officer\' appears: '+str(SI_tweets[SI_tweets['tweet'].str.contains("officer")].tweet.shape[0]))


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

