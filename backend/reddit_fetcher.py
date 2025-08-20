# reddit_fetcher.py

import praw
import csv

reddit = praw.Reddit(
    client_id='NbmX9JgalunkU_NcRo0QBw',           # <--- YOUR client_id
    client_secret='0q-lsHutrnMHmytR4rnIWUn8dSshSg', # <--- YOUR secret
    user_agent='symptom_checker_bot by /u/UbKa123'  # <--- YOUR user_agent
)

def fetch_breast_cancer_posts(subreddit='breastcancer', limit=200):
    posts = []
    for submission in reddit.subreddit(subreddit).search('symptom', sort='new', limit=limit):
        posts.append({
            'title': submission.title,
            'selftext': submission.selftext,
            'id': submission.id
        })
    return posts

def save_posts(posts, filename='reddit_symptoms.csv'):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['id', 'title', 'selftext'])
        writer.writeheader()
        for post in posts:
            writer.writerow(post)

if __name__ == '__main__':
    posts = fetch_breast_cancer_posts()
    save_posts(posts)
    print(f"Saved {len(posts)} posts to reddit_symptoms.csv")
