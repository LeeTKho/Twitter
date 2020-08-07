# Twitter
Visualization tool for tweet patterns.

General time window: 5/26 - 6/16



Questions to answer:

1. How do twitter sentiments differ in police communities vs other parts of NYC?
2. Where are people tweeting the most about BLM?
3. Look at mini-events that occured within the time window and see which areas were most active. Also look at how these specific areas reacted differently to different mini-events.
  Some events to consider:
  - May 28 - First day of protests in NYC
  - June 1 - First day of pride month
  - June 2 - First day of NYC curfew
  - June 3 - Three other Minnesota officers charged with aiding & abetting Chauvin in murder of George Floyd; Chauvin's charged is raised from 3rd to 2nd degree murder
  - June 4 - George Floyd memorial is held in Minneapolis
  - June 5 - Breonna Taylor's birthday
4. BONUS - Hashtag network graphs!

## Usage

**Pulling Tweet Data**

First clone this repo and navigate to the project folder. In 
your terminal:
```bash
pip install -e .
python scripts/pull_borough_tweets.py search_config/config_1.yaml
```