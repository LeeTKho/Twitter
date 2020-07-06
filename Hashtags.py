#!/usr/bin/env python
# coding: utf-8

# In[80]:


import twint
import pandas as pd
import numpy as np
import asyncio
import nest_asyncio
nest_asyncio.apply()


# In[81]:


# storing each search into csv file
c = twint.Config()
c.Lang = 'en'
c.Pandas = True
c.Search = "blm"
c.Min_likes = 50
c.Filter_retweets = True
c.Since = "2020-05-26"
c.Until = "2020-06-09"

asyncio.set_event_loop(asyncio.new_event_loop())
twint.run.Search(c)

data = twint.storage.panda.Tweets_df


# In[82]:


# filtering out tweets with fewer than 100 likes
data = data[data.nlikes>=100]


# In[83]:


data.shape


# In[84]:


data[['id','tweet','hashtags']].head(10)


# In[85]:


hashtags = data.astype('str')
hashtags = hashtags[hashtags.hashtags!="[]"].hashtags
hashtags.head()


# In[86]:


import itertools

hashtag_edges = []
for hash_list in hashtags:
    hash_list = hash_list.rstrip(']').lstrip('[').split(',')
    for n in range(len(hash_list)):
        hash_list[n] = hash_list[n].rstrip('\'').lstrip(' ').lstrip('\'#')
    combos = itertools.combinations(hash_list, 2)
    hashtag_edges = hashtag_edges+list(combos)
    
len(hashtag_edges)


# In[87]:


for n in range(len(hashtag_edges)):
    hashtag_edges[n] = tuple(sorted(hashtag_edges[n]))


# In[88]:


hashtag_edges


# In[89]:


import collections
counter=collections.Counter(hashtag_edges)

# filtering out edges with weights of less than 10
counter_fil = {key:val for key, val in counter.items() if val >10}
counter_fil


# In[90]:


weights = list(counter_fil.values())
edges = list(counter_fil.keys())

collections.Counter(weights)


# In[91]:


import matplotlib.pyplot as plt
import networkx as nx

G = nx.Graph()

for n in range(len(edges)):

    G.add_edge(edges[n][0], edges[n][1], weight=weights[n])
    

e10 = [(u, v) for (u, v, d) in G.edges(data=True) if (d['weight'] >4 and d['weight']<=20)]
e20 = [(u, v) for (u, v, d) in G.edges(data=True) if (d['weight'] >20 and d['weight']<=50)]
e30 = [(u, v) for (u, v, d) in G.edges(data=True) if (d['weight'] >50)]


#pos = nx.spring_layout(G)  # positions for all nodes
#pos = nx.circular_layout(G)
pos = nx.spring_layout(G)

# nodes
nx.draw_networkx_nodes(G, pos, node_color = 'r', node_size=100)

# edges
nx.draw_networkx_edges(G, pos, edgelist=e10,edge_color = 'r',width=1)
nx.draw_networkx_edges(G, pos, edgelist=e20,edge_color = 'g',width=3)#, alpha=0.5, edge_color='b', style='solid')
nx.draw_networkx_edges(G, pos, edgelist=e30,edge_color = 'b',width=5)

# labels
nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

fig = plt.figure(1,figsize=(15,15))
fig.set_figheight(15)
fig.set_figwidth(15)
plt.axis('off')
plt.show()


# In[ ]:




