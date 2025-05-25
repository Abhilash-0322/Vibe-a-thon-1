import pathway as pw
import tweepy
import json
import os
import time
import praw
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
import random
import os
from dotenv import load_dotenv

load_dotenv()
# Configuration
X_API_KEY = os.getenv("X_API_KEY")
X_API_SECRET = os.getenv("X_API_SECRET")
X_ACCESS_TOKEN = os.getenv("X_ACCESS_TOKEN")
X_ACCESS_TOKEN_SECRET = os.getenv("X_ACCESS_TOKEN_SECRET")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PRODUCT_QUERY = os.getenv("PRODUCT_QUERY")
TWEETS_DIR = os.getenv("TWEETS_DIR")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")

# Ensure tweets directory exists
if not os.path.exists(TWEETS_DIR):
    os.makedirs(TWEETS_DIR)

# Initialize Groq LLM
llm = ChatGroq(model="llama3-8b-8192", api_key=GROQ_API_KEY)

# Prompts
sentiment_prompt = PromptTemplate(
    input_variables=["text"],
    template="Classify the sentiment of this post as POSITIVE, NEGATIVE, or NEUTRAL: {text}\nReturn only the sentiment label."
)
qa_prompt = PromptTemplate(
    input_variables=["query", "context"],
    template="Based on the latest posts: {context}\nAnswer the query: {query}"
)
sentiment_chain = sentiment_prompt | llm
qa_chain = qa_prompt | llm

# Function to fetch X API v2 tweets
def fetch_x_tweets(query, max_tweets=10):
    try:
        client = tweepy.Client(
            consumer_key=X_API_KEY,
            consumer_secret=X_API_SECRET,
            access_token=X_ACCESS_TOKEN,
            access_token_secret=X_ACCESS_TOKEN_SECRET
        )
        tweets = client.search_recent_tweets(query=query, max_results=max_tweets, tweet_fields=["created_at"])
        if not tweets.data:
            return 0
        for i, tweet in enumerate(tweets.data):
            sentiment = sentiment_chain.invoke({"text": tweet.text}).content.strip()
            tweet_data = {
                "text": tweet.text,
                "created_at": str(tweet.created_at),
                "sentiment": sentiment
            }
            filename = f"{TWEETS_DIR}/tweet_{int(time.time())}_{i}.json"
            with open(filename, "w") as f:
                json.dump(tweet_data, f)
        return len(tweets.data)
    except Exception as e:
        print(f"X API error: {e}")
        return 0

# Function to fetch Reddit posts
def fetch_reddit_posts(query, max_posts=10):
    try:
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT
        )
        subreddit = reddit.subreddit("gadgets")
        posts = subreddit.search(query, limit=max_posts)
        count = 0
        for post in posts:
            sentiment = sentiment_chain.invoke({"text": post.title}).content.strip()
            post_data = {
                "text": post.title,
                "created_at": str(time.ctime(post.created_utc)),
                "sentiment": sentiment
            }
            filename = f"{TWEETS_DIR}/post_{int(time.time())}_{count}.json"
            with open(filename, "w") as f:
                json.dump(post_data, f)
            count += 1
        return count
    except Exception as e:
        print(f"Reddit API error: {e}")
        return 0

# Function to fetch data (X API first, then Reddit)
def fetch_data(query, max_items=10):
    count = fetch_x_tweets(query, max_tweets=max_items)
    if count == 0:
        print("Switching to Reddit API")
        count = fetch_reddit_posts(query, max_posts=max_items)
    return count

# Pathway schema
class PostSchema(pw.Schema):
    text: str
    created_at: str
    sentiment: str

# Pathway pipeline
data = pw.io.fs.read(
    TWEETS_DIR,
    format="json",
    schema=PostSchema,
    mode="streaming"
)

# Pathway-native retrieval
@pw.udf
def compute_score(text: str, query: str) -> int:
    query_words = query.lower().split()
    return sum(1 for word in query_words if word in text.lower())

def retrieve_posts(query, table, k=5):
    scored_table = table.select(*table, score=compute_score(table.text, pw.lift(query)))
    top_posts = scored_table.sort_by(scored_table.score, descending=True)[:k]
    result = top_posts.select(text=top_posts.text).to_pylist()
    return [{"text": row["text"]} for row in result]

# Function to compute sentiment distribution
def get_sentiment_distribution():
    posts = [json.load(open(f)) for f in os.listdir(TWEETS_DIR) if f.endswith(".json")]
    sentiments = [p["sentiment"] for p in posts]
    return pd.Series(sentiments).value_counts().to_dict()

# Main query loop
def query_loop():
    while True:
        # Fetch new data
        fetch_data(PRODUCT_QUERY, max_items=5)
        
        # Get user query
        query = input("Enter query (or 'quit' to exit): ")
        if query.lower() == "quit":
            break
            
        # Retrieve relevant posts
        retrieved = retrieve_posts(query, data, k=5)
        context = "\n".join([row["text"] for row in retrieved])
        
        # Generate response
        response = qa_chain.invoke({"query": query, "context": context}).content
        
        # Print sentiment distribution
        sentiment_counts = get_sentiment_distribution()
        print("\nSentiment Distribution:", sentiment_counts)
        print("Response:", response, "\n")

# Simulate real-time stream
def simulate_stream():
    while True:
        fetch_data(PRODUCT_QUERY, max_items=5)
        time.sleep(10)

if __name__ == "__main__":
    import threading
    threading.Thread(target=simulate_stream, daemon=True).start()
    pw.run(monitoring_level=pw.MonitoringLevel.NONE)
    query_loop()