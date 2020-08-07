import pandas as pd
import twint


def _fmt_search_terms(search_terms: list):
    return ' OR '.join([f'"{term}"' if ' ' in term else term for term in search_terms])


def make_twint_config(geo: str, search_terms: list, since: str, until: str) -> twint.Config:
    """
    :param geo: geocode for twitter search, eg "40.676302, -73.971787, 1mi"
    """
    c = twint.Config()
    c.Search = _fmt_search_terms(search_terms)
    c.Since = since
    c.Until = until
    c.Geo = geo
    c.Min_likes = 3
    c.Filter_retweets = True
    c.Pandas = True
    c.Hide_output = True
    return c


def search_tweets(geocode, search_terms, since, until) -> pd.DataFrame:
    twint.run.Search(make_twint_config(geocode, search_terms, since, until))
    return twint.storage.panda.Tweets_df.copy()
