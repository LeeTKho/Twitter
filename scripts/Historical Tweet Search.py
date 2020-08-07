#!/usr/bin/env python
# coding: utf-8

# ### Borough searches
# Brooklyn: ['40.6451804, 73.9431988, 5.39mi','bk']
# 
# https://www.mapdevelopers.com/draw-circle-tool.php?circles=%5B%5B9135.23%2C40.7394143%2C-73.8470623%2C%22%23AAAAAA%22%2C%22%23000000%22%2C0.4%5D%5D
# 
# Queens: ['40.739414, -73.847062,5.68mi','qn'] 
# 
# https://www.mapdevelopers.com/draw-circle-tool.php?circles=%5B%5B9135.23%2C40.7394143%2C-73.8470623%2C%22%23AAAAAA%22%2C%22%23000000%22%2C0.4%5D%5D
# 
# Bronx: ['40.8474099, 73.8670223, 4.09mi','bx']
# 
# https://www.mapdevelopers.com/draw-circle-tool.php?circles=%5B%5B9135.23%2C40.7394143%2C-73.8470623%2C%22%23AAAAAA%22%2C%22%23000000%22%2C0.4%5D%5D
# 
# SI: ['40.5586358, 74.1347641, 6.90mi','si']
# 
# https://www.mapdevelopers.com/draw-circle-tool.php?circles=%5B%5B11109.24%2C40.5586358%2C-74.1347641%2C%22%23AAAAAA%22%2C%22%23000000%22%2C0.4%5D%5D
# 
# Manhattan: 
# ['40.7254044, 74.0030475, 0.88mi','mn1']
# ['40.7625644, 73.980033, 0.88mi','mn2']
# ['40.7953538, 73.9549461, 0.88mi','mn3']
# ['40.8243597, 73.9447894, 0.86mi','mn4']
# ['40.8448821, 73.9417006, 0.86mi','mn5']
# ['40.8633167, 73.9276253, 0.86mi','mn6']
# 
# https://www.mapdevelopers.com/draw-circle-tool.php?circles=%5B%5B2717.91%2C40.7254044%2C-74.0030475%2C%22%23AAAAAA%22%2C%22%23000000%22%2C0.4%5D%2C%5B2719.78%2C40.7625644%2C-73.980033%2C%22%23AAAAAA%22%2C%22%23000000%22%2C0.4%5D%2C%5B2719.78%2C40.7953538%2C-73.9549461%2C%22%23AAAAAA%22%2C%22%23000000%22%2C0.4%5D%5D
# 
# https://www.mapdevelopers.com/draw-circle-tool.php?circles=%5B%5B1389.37%2C40.8243597%2C-73.9447894%2C%22%23AAAAAA%22%2C%22%23000000%22%2C0.4%5D%2C%5B1384.03%2C40.8448821%2C-73.9417006%2C%22%23AAAAAA%22%2C%22%23000000%22%2C0.4%5D%2C%5B1384.03%2C40.8633167%2C-73.9276253%2C%22%23AAAAAA%22%2C%22%23000000%22%2C0.4%5D%5D
# 

# In[2]:


import twint
import pandas as pd
import numpy as np
import nest_asyncio
import asyncio
nest_asyncio.apply()


# In[15]:


bk = ['40.6451804, -73.9431988, 5.39mi','bk']
qn = ['40.739414, -73.847062, 5.68mi','qn']
bx = ['40.8474099, -73.8670223, 4.09mi','bx']
si = ['40.5586358, -74.1347641, 6.90mi','si']
mn1 = ['40.7254044, -74.0030475, 0.88mi','mn1']
mn2 = ['40.7625644, -73.980033, 0.88mi','mn2']
mn3 = ['40.7953538, -73.9549461, 0.88mi','mn3']
mn4 = ['40.8243597, -73.9447894, 0.86mi','mn4']
mn5 = ['40.8448821, -73.9417006, 0.86mi','mn5']
mn6 = ['40.8633167, -73.9276253, 0.86mi','mn6']


# ### Store historical search to CSV file

# In[45]:


# list of search areas
searches = [bk,sn,bx,mn1,mn2,mn3,mn4,mn5,mn6]

# storing each search into csv file
for search in searches:
    c = twint.Config()
    c.Store_csv = True
    c.Geo = search[0]
    c.Output = "historical_search_"+search[1]+".csv"
    c.Search = "blm OR \"black lives matter\" OR police OR protest OR riot OR \"breonna taylor\" OR \"george floyd\""
    #c.Min_likes = 20
    c.Filter_retweets = True
    c.Since = "2020-05-26"
    c.Until = "2020-06-09"

    asyncio.set_event_loop(asyncio.new_event_loop())
    twint.run.Search(c)


# ### Store historical search to pandas dataframe

# In[20]:


# list of search areas
searches = [bk,qn,bx,si,mn1,mn2,mn3,mn4,mn5,mn6]

# storing each search into pandas df
for search in searches:
    c = twint.Config()
    c.Pandas = True
    c.Geo = search[0]
    c.Search = "blm OR \"black lives matter\" OR police OR protest OR riot OR \"george floyd\""
    c.Filter_retweets = True
    c.Since = "2020-05-26"
    c.Until = "2020-06-16"

    asyncio.set_event_loop(asyncio.new_event_loop())
    twint.run.Search(c)
    tweets_df = twint.storage.panda.Tweets_df
    tweets_df.to_pickle("./"+str(search[1])+"_tweets.pkl")
    print("number of "+str(search[1])+" tweets: "+str(tweets_df.shape[0]))

