#!/usr/bin/env python
# coding: utf-8

# In[45]:


import twint
import pandas as pd
import numpy as np
import nest_asyncio
nest_asyncio.apply()

# list of search areas
searches = [["40.676302, -73.971787, 1mi","BK"],["40.588579, -74.143985, 3mi","SI"]]

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


# In[ ]:




