import tweepy
import json
from tweepy import Cursor

with open("../data/raw/twitter_creds.json") as f:
    twitter_credentials = json.load(f)

auth = tweepy.OAuthHandler(twitter_credentials['consumer_key'], twitter_credentials['consumer_secret'])
auth.set_access_token(twitter_credentials['access_token'], twitter_credentials['access_token_secret'])
api = tweepy.API(auth)

def get_recent_tweets(user, n=20):
    all_tweets = []
    # q -> defines query for tweets
    for tweet in Cursor(api.search, q=f"@{user}", count=n, tweet_mode="extended").items():
        # append text and link of tweet
        all_tweets.append((tweet.full_text, f"https://twitter.com/{tweet.user.screen_name}/status/{tweet.id}"))
    return all_tweets