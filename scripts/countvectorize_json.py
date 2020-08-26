import twint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import asyncio
import nest_asyncio
nest_asyncio.apply()
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import re
import sys
import collections
import itertools
import matplotlib.pyplot as plt
import networkx as nx
import graphviz
from community import community_louvain
import json
import logging
import os


boroughs = ['BK', 'BX', 'QN', 'SI', 'MN']

timestamp = sys.argv[1]

blm_stopwords = ['www','http','https','https twitter','https twitter com','twitter',
                'pic twitter','pic twitter com','twitter com',
                'https www', 'https www instagram','instagram','instagram com',
                'www instagram','www instagram com','at the','in the','of the',
                'the protest','if you','this is','on the','to the','you re','carrd','co',
                'on','out','with','is','our','too','so','my','igshid','status',
                'like','to be','would','let','via','for the','even','still','and the','back',
                'many','way','make','said','is the','think','say','see','go','going','know','time',
                'get','http chng','chng','chng it','peti712tion http','really','is not','you are',
                'via change','they are','much','we are','with the','want to','from the','they re','have to',
                'going to','the same','must','we re','it was','well','out of','are all','you can',
                'to do','will be','should be','start','saying','to get','com ca','in new']

for borough in boroughs:

    # select borough
    blm_data = pd.DataFrame(pd.read_pickle(f"./data/{timestamp}/{borough}_tweets.pkl"))

    # countvectorizing blm tweets
    print(f"-- {borough} CountVectorizer starting")
    vectorizer = CountVectorizer()

    X = vectorizer.fit_transform(blm_data['tweet'])

    user_ids = list(blm_data['user_id'])

    blm_countvec = pd.DataFrame(X.toarray(),columns=vectorizer.get_feature_names())

    # build df with blm user_ids and countvectorized tweets
    blm_countvec.insert(0, "user_id", user_ids, False)

    # import list of general stopwords
    nltk_stopwords = stopwords.words()

    # create list of stopwords found in the blm tweets
    all_stopwords = list(set(blm_countvec.columns) & set(nltk_stopwords)) +list(set(blm_countvec.columns) & set(blm_stopwords))

    # remove stopwords from countvectorized blm tweets
    blm_countvec = blm_countvec.drop(all_stopwords,axis=1)

    # get 100 top keywords
    top_blm_keywords = blm_countvec.iloc[:,1:].sum(axis=0).sort_values(ascending=False)[:100]

    # reading pickle files into pandas dataframes
    blm_tweets = blm_data['tweet']

    blm_tweets_clean = blm_tweets

    # cleaning tweets - removing non-alphabetic characters and converting to lowercase
    for n in range(blm_tweets.shape[0]):
        blm_tweets_clean.iloc[n] = re.sub(r'[^\w]', ' ', blm_tweets.iloc[n]).split(' ')
        blm_tweets_clean.iloc[n] = [i.lower() for i in blm_tweets.iloc[n]]

    print(f"-- Extracting top keywords from {borough} tweets")
    blm_keywords = [[]]*blm_tweets_clean.shape[0]
    blm_tweets_clean.iloc[0]

    for n in range(blm_tweets_clean.shape[0]):
        blm_keywords[n] = list(set(blm_tweets_clean.iloc[n]) & set(list(top_blm_keywords.index)))

    # determining connections between keywords (i.e. seeing which keywords were used together)
    print(f"-- Building {borough} edges")
    blm_keyword_edges = []
    for blm_keyword_list in blm_keywords:
        combos = itertools.combinations(blm_keyword_list, 2)
        blm_keyword_edges = blm_keyword_edges+list(combos)

    len(blm_keyword_edges)

    # creating list of all keyword connections to determine weight of edges
    for n in range(len(blm_keyword_edges)):
        blm_keyword_edges[n] = tuple(sorted(blm_keyword_edges[n]))

    # counting up keyword connections to determine weight of edges
    blm_counter=collections.Counter(blm_keyword_edges)

    # filtering out edges with weights of less than 10 (for bk, less than 20)
    blm_counter_fil = {key:val for key, val in blm_counter.items() if val >10}

    # getting final node list
    all_blm_keywords=list(set(list(sum(list(blm_counter_fil.keys()), ()))))
    len(all_blm_keywords)

    # creating list of weights and list of nodes for graph
    blm_weights = list(blm_counter_fil.values())
    blm_edges = list(blm_counter_fil.keys())

    g = nx.Graph()

    for n in range(len(blm_edges)):
        g.add_edge(blm_edges[n][0], blm_edges[n][1], weight=blm_weights[n])

    d = dict(g.degree)

    # creating groups using the Louvain method
    print(f"-- Grouping {borough} nodes")
    partition = community_louvain.best_partition(g)

    def get_index(keyword):
        return(all_blm_keywords.index(keyword)+1)

    blm_source = [int(i) for i in [get_index(edge[0]) for edge in blm_edges]]

    blm_target = [int(i) for i in [get_index(edge[1]) for edge in blm_edges]]

    # all blm keywords
    all_blm_keywords = list(d.keys())

    # blm node weights
    keyword_weight = [str(i) for i in list(d.values())]

    # blm node id_num
    id_num = [i+1 for i in range(len(all_blm_keywords))]

    # blm node group
    group = [str(i) for i in list(partition.values())]

    print("-- Saving to json")
    node_data = {'id':id_num,'name':all_blm_keywords,'value':keyword_weight,'group':group}
    node_df = pd.DataFrame(data=node_data)

    edge_data = {'source':blm_source,'target':blm_target,'weight':blm_weights}
    edge_df = pd.DataFrame(data=edge_data)

    node_json = node_df.to_json(orient='records')
    edge_json = edge_df.to_json(orient='records')

    # node_json = json.loads(node_df.to_json(orient='records'))
    # edge_json = json.loads(edge_df.to_json(orient='records'))

    with open(f'./network_graphs/data/{borough}_graph_uni_2.json', 'w') as f:
        json.dump({'nodes':node_json,'links':edge_json}, f)

    print(f"-- {borough} tweets json saved")
