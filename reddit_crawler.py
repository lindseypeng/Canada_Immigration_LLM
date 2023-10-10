import praw
import os
client_id = os.environ.get('REDDIT_CLIENT_ID')
client_secret = os.environ.get('REDDIT_CLIENT_SECRET')

print(len(client_id))
print(len(client_secret))

user_agent = 'chemengtodatasci'

# Authenticate with the Reddit API
reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)


# Set the subreddit and the flair you want to filter by
subreddit_name = 'ImmigrationCanada'
flair = 'Sponsorship'

# Connect to the subreddit
subreddit = reddit.subreddit(subreddit_name)

# Scrape the posts with the specified flair
filtered_posts = []
for submission in subreddit.new(limit=1000): # You can increase the limit if needed
    if submission.link_flair_text == flair:
        filtered_posts.append(submission)