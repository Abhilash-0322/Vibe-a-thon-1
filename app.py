# import pathway as pw
# from groq import Groq
# import praw
# import requests
# import json




# reddit = praw.Reddit(client_id="your_id", client_secret="your_secret", user_agent="hackathon")
# for submission in reddit.subreddit("gadgets").stream.submissions():
#     with open(f"posts/{submission.id}.json", "w") as f:
#         json.dump({"text": submission.title, "created_at": submission.created_utc}, f)

# class InputSchema(pw.Schema):
#     text: str
#     created_at: str
#     sentiment: str
# data = pw.io.fs.read("tweets_directory/", format="json", schema=InputSchema)

# response = requests.get("https://newsapi.org/v2/everything?q=iPhone+16&apiKey=your_key")
# for article in response.json()["articles"]:
#     with open(f"news/{article['publishedAt']}.json", "w") as f:
#         json.dump({"text": article["title"] + " " + article["description"], "created_at": article["publishedAt"]}, f)

# client = Groq(api_key="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")
# response = client.chat.completions.create(
#     model="llama3-8b-8192",
#     messages=[{"role": "user", "content": f"Summarize: {retrieved_tweets}"}]
# )



# import pathway as pw
# import tweepy
# import json
# import os
# import time
# from langchain_groq import ChatGroq
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnableSequence
# import gradio as gr
# import plotly.express as px
# import pandas as pd

# # Configuration
# X_API_KEY = "OKSF9af94PSq75XvYuPRWq9Ag"  # Your X API Key
# X_API_SECRET = "hdIQzjEe6e1uXOSREqIy8Q6h0WiGduONlEcGzepxVO023JB1NM"  # Your X API Secret
# X_ACCESS_TOKEN = "1211278825994145792-TywBGuuoHmqDMU4vuSmEfg6l36DgJC"  # Your Access Token
# X_ACCESS_TOKEN_SECRET = "VOfnYLe64IFaUvKkmjv1NWaVHRRRRuF6KkSiZrAG7VxLN"  # Your Access Token Secret
# GROQ_API_KEY = "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"  # Your Groq API Key
# PRODUCT_QUERY = "iPhone 16"  # Product to search for
# TWEETS_DIR = "tweets_data"  # Directory for storing tweets

# # Ensure tweets directory exists
# if not os.path.exists(TWEETS_DIR):
#     os.makedirs(TWEETS_DIR)

# # Initialize X API client
# auth = tweepy.OAuthHandler(X_API_KEY, X_API_SECRET)
# auth.set_access_token(X_ACCESS_TOKEN, X_ACCESS_TOKEN_SECRET)
# api = tweepy.API(auth, wait_on_rate_limit=True)

# # Initialize Groq LLM
# llm = ChatGroq(model="llama3-8b-8192", api_key=GROQ_API_KEY)

# # Prompt for sentiment analysis
# sentiment_prompt = PromptTemplate(
#     input_variables=["text"],
#     template="Classify the sentiment of this tweet as POSITIVE, NEGATIVE, or NEUTRAL: {text}\nReturn only the sentiment label."
# )

# # Prompt for Q&A
# qa_prompt = PromptTemplate(
#     input_variables=["query", "context"],
#     template="Based on the latest tweets: {context}\nAnswer the query: {query}"
# )

# # LangChain RunnableSequences
# sentiment_chain = sentiment_prompt | llm
# qa_chain = qa_prompt | llm

# # Function to fetch tweets and save to JSON with Groq-based sentiment
# def fetch_tweets(query, max_tweets=10):
#     tweets = api.search_tweets(q=query, count=max_tweets, tweet_mode="extended")
#     for tweet in tweets:
#         tweet_text = tweet.full_text if hasattr(tweet, "full_text") else tweet.text
#         # Use Groq to classify sentiment
#         sentiment = sentiment_chain.invoke({"text": tweet_text}).content.strip()
#         tweet_data = {
#             "text": tweet_text,
#             "created_at": str(tweet.created_at),
#             "sentiment": sentiment
#         }
#         filename = f"{TWEETS_DIR}/tweet_{tweet.id}.json"
#         with open(filename, "w") as f:
#             json.dump(tweet_data, f)
#     return len(tweets)

# # Pathway schema for tweets
# class TweetSchema(pw.Schema):
#     text: str
#     created_at: str
#     sentiment: str

# # Pathway pipeline to ingest tweets
# data = pw.io.fs.read(
#     TWEETS_DIR,
#     format="json",
#     schema=TweetSchema,
#     mode="streaming"
# )

# # Simple keyword-based retrieval function (replacing pw.ml.index)
# def retrieve_tweets(query, table, k=5):
#     # Convert Pathway table to list for filtering
#     tweets = table.select().to_pandas()
#     if tweets.empty:
#         return []
#     # Basic keyword matching for retrieval
#     query_words = query.lower().split()
#     tweets['score'] = tweets['text'].apply(
#         lambda x: sum(1 for word in query_words if word in x.lower())
#     )
#     # Sort by relevance and return top k
#     top_tweets = tweets.sort_values(by='score', ascending=False).head(k)
#     return top_tweets[['text']].to_dict('records')

# # Function to compute sentiment distribution
# def get_sentiment_distribution():
#     tweets = [json.load(open(f)) for f in os.listdir(TWEETS_DIR) if f.endswith(".json")]
#     sentiments = [t["sentiment"] for t in tweets]
#     sentiment_counts = pd.Series(sentiments).value_counts().to_dict()
#     return sentiment_counts

# # Gradio interface function
# def query_copilot(user_query):
#     # Fetch latest tweets
#     fetch_tweets(PRODUCT_QUERY, max_tweets=10)
    
#     # Retrieve relevant tweets using custom function
#     retrieved = retrieve_tweets(user_query, data, k=5)
#     context = "\n".join([row["text"] for row in retrieved])
    
#     # Generate response with Groq
#     response = qa_chain.invoke({"query": user_query, "context": context}).content
    
#     # Create sentiment chart
#     sentiment_counts = get_sentiment_distribution()
#     fig = px.pie(
#         values=list(sentiment_counts.values()),
#         names=list(sentiment_counts.keys()),
#         title="Sentiment Distribution"
#     )
    
#     return response, fig

# # Gradio interface
# with gr.Blocks() as demo:
#     gr.Markdown("# Real-Time Social Sentiment Copilot")
#     gr.Markdown(f"Query live sentiment and insights about {PRODUCT_QUERY}")
#     query_input = gr.Textbox(label="Ask a question")
#     output_text = gr.Textbox(label="Response")
#     output_plot = gr.Plot(label="Sentiment Distribution")
#     submit_button = gr.Button("Submit")
#     submit_button.click(
#         fn=query_copilot,
#         inputs=query_input,
#         outputs=[output_text, output_plot]
#     )

# # Simulate real-time tweet stream (for demo)
# def simulate_stream():
#     while True:
#         fetch_tweets(PRODUCT_QUERY, max_tweets=5)
#         time.sleep(10)  # Fetch new tweets every 10 seconds

# if __name__ == "__main__":
#     # Start streaming tweets in background
#     import threading
#     threading.Thread(target=simulate_stream, daemon=True).start()
    
#     # Start Pathway pipeline
#     pw.run()
    
#     # Launch Gradio interface
#     demo.launch()



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

# Configuration
X_API_KEY = "OKSF9af94PSq75XvYuPRWq9Ag"
X_API_SECRET = "hdIQzjEe6e1uXOSREqIy8Q6h0WiGduONlEcGzepxVO023JB1NM"
X_ACCESS_TOKEN = "1211278825994145792-TywBGuuoHmqDMU4vuSmEfg6l36DgJC"
X_ACCESS_TOKEN_SECRET = "VOfnYLe64IFaUvKkmjv1NWaVHRRRRuF6KkSiZrAG7VxLN"
GROQ_API_KEY = "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"
PRODUCT_QUERY = "iPhone 16"  # Product to search for
TWEETS_DIR = "tweets_data"
REDDIT_CLIENT_ID = "eSNwYwBx4vv9Tw0N1QKjaQ"  # Replace if using Reddit
REDDIT_CLIENT_SECRET = "KzO7WO21rRi0CFO-M5zy9nYAMkBeVg"  # Replace if using Reddit
REDDIT_USER_AGENT = "vibe-a-thon/1.0"

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