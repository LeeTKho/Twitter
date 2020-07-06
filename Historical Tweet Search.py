#!/usr/bin/env python
# coding: utf-8

# In[1]:


import twint
import pandas as pd
import numpy as np
import nest_asyncio
import asyncio
nest_asyncio.apply()


# ### Store historical search to CSV file

# In[45]:


# list of search areas
searches = [["40.642502, -73.947316, 5mi","BK"],["40.556794, -74.109002, 5mi","SI"]]

# storing each search into csv file
for search in searches:
    c = twint.Config()
    c.Store_csv = True
    c.Geo = search[0]
    c.Output = "historical_search_"+search[1]+".csv"
    c.Search = "blm OR \"black lives matter\" OR police OR protest OR riot OR \"breonna taylor\" OR \"george floyd\"" #"blm" #search terms
    #c.Min_likes = 20
    c.Filter_retweets = True
    c.Since = "2020-05-26"
    c.Until = "2020-06-09"

    asyncio.set_event_loop(asyncio.new_event_loop())
    twint.run.Search(c)


# ### Store historical search to pandas dataframe

# In[17]:


# list of search areas
searches = [["40.642502, -73.947316, 5mi","BK"],["40.556794, -74.109002, 5mi","SI"]]

# storing each search into pandas df
for search in searches:
    c = twint.Config()
    c.Pandas = True
    c.Geo = search[0]
    c.Lang = "en"
    c.Search = "blm OR \"black lives matter\" OR police OR protest OR riot OR \"breonna taylor\" OR \"george floyd\""
    #c.Min_likes = 10
    c.Filter_retweets = True
    c.Since = "2020-05-26"
    c.Until = "2020-06-09"

    asyncio.set_event_loop(asyncio.new_event_loop())
    twint.run.Search(c)
    Tweets_df = twint.storage.panda.Tweets_df
    Tweets_df.to_pickle("./"+str(search[1])+"_tweets.pkl")
    print("number of "+str(search[1])+" tweets: "+str(Tweets_df.shape[0]))


# In[ ]:




