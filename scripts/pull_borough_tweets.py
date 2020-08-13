import argparse
from datetime import datetime
import logging
import os
import time

import pandas as pd
import yaml
from yaml import Loader, Dumper

from twitter_utils import search_tweets

BOROUGH_GEOCODES = {
    'BK':  ['40.6451804, -73.9431988, 5.39mi'],
    'QN': ['40.739414, -73.847062, 5.68mi'],
    'BX:': ['40.8474099, -73.8670223, 4.09mi'],
    'SI': ['40.5586358, -74.1347641, 6.90mi'],
    'MN': ['40.7254044, -74.0030475, 0.88mi',
           '40.7625644, -73.980033, 0.88mi',
           '40.7953538, -73.9549461, 0.88mi',
           '40.8243597, -73.9447894, 0.86mi',
           '40.8448821, -73.9417006, 0.86mi',
           '40.8633167, -73.9276253, 0.86mi']
}

TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H%M%S')
SAVE_BASE_DIR = f"../data/{TIMESTAMP}"
# TWEETS_SAVE_BASE_PATH = f"{SAVE_BASE_DIR}/{{borough_code}}_tweets.csv"
TWEETS_SAVE_BASE_PATH = f"{SAVE_BASE_DIR}/{{borough_code}}_tweets.pkl"
CONFIG_SAVE_BASE_PATH = f"{SAVE_BASE_DIR}/search_config.yaml"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config_path',
        help='path to search term/date config file',
        default=None,
        type=str
    )
    return parser.parse_args()


def pull_data(args):
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    config = yaml.load(open(args.config_path), Loader=Loader)
    logging.info("Loaded Config:")
    logging.info(config)

    if not os.path.exists(SAVE_BASE_DIR):
        os.makedirs(SAVE_BASE_DIR)

    yaml.dump(config, open(CONFIG_SAVE_BASE_PATH, 'w'), Dumper=Dumper)
    logging.info(f"saved copy of config to `{CONFIG_SAVE_BASE_PATH}`")

    for borough_code, borough_geocodes in BOROUGH_GEOCODES.items():
        st = time.time()
        save_path = TWEETS_SAVE_BASE_PATH.format(borough_code=borough_code)
        logging.info(f"starting search for {borough_code}")
        tweets_df = pd.concat([search_tweets(geocode,
                                             search_terms=config['search_terms'],
                                             since=str(config['since']),
                                             until=str(config['until']))
                               for geocode in borough_geocodes])
        logging.info(f"-- pulled {len(tweets_df)} tweets")
        # tweets_df.to_csv(save_path, index=False)
        tweets_df.to_pickle(save_path)
        logging.info(f"-- saved tweets to `{save_path}`")
        run_time = time.time() - st
        logging.info(f"-- data pull took {round(run_time/60, 2)} minutes")


if __name__ == '__main__':
    args = parse_args()
    pull_data(args)
