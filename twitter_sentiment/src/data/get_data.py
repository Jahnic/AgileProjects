import tweepy
import json
import time
from tweepy import Cursor

with open('../data/raw/twitter_creds.json') as f:
    twitter_credentials = json.load(f)

auth = tweepy.OAuthHandler(twitter_credentials['consumer_key'], twitter_credentials['consumer_secret'])
auth.set_access_token(twitter_credentials['access_token'], twitter_credentials['access_token_secret'])
api = tweepy.API(auth, wait_on_rate_limit=True,
                 wait_on_rate_limit_notify=True)


def get_recent_tweets(query, n=10):
    """
    Returns comprehensive list of information on recent tweets that fit query
    """
    all_tweets = []

    # tweepy status object
    # q -> defines query for tweets
    # extended mode for new 280 character tweet cap
    c = Cursor(api.search,
               q=f"@{query}",
               lang='en',
               count=n,
               tweet_mode="extended").items()

    # rate limit handling
    while True:
        try:
            tweet = c.next()
            # Insert into db
            all_tweets.append((
                tweet.full_text,
                f"https://twitter.com/{tweet.user.screen_name}/status/{tweet.id}",
                tweet.created_at, tweet.id_str, tweet.favorite_count, tweet.retweet_count,
                tweet.in_reply_to_user_id_str, tweet.user.name, tweet.user.screen_name,
                tweet.user.id_str, tweet.user.location, tweet.user.url, tweet.user.description,
                tweet.user.verified, tweet.user.followers_count, tweet.user.friends_count,
                tweet.user.favourites_count, tweet.user.statuses_count, tweet.user.listed_count,
                tweet.user.created_at, tweet.user.profile_image_url_https,
                tweet.user.default_profile, tweet.user.default_profile_image
            ))
        except tweepy.TweepError:
            time.sleep(60 * 15)
            continue
        except StopIteration:
            break

    return all_tweets
