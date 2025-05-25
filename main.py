import pathway as pw
import json
import os
import time
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
import gradio as gr
import plotly.express as px
import pandas as pd
import random
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Your Groq API Key
PRODUCT_QUERY = "iPhone 16"  # Product for context
TWEETS_DIR = "tweets_data"  # Directory for storing tweets

# Ensure tweets directory exists
if not os.path.exists(TWEETS_DIR):
    os.makedirs(TWEETS_DIR)

# Initialize Groq LLM
llm = ChatGroq(model="llama3-8b-8192", api_key=GROQ_API_KEY)

# Prompt for sentiment analysis
sentiment_prompt = PromptTemplate(
    input_variables=["text"],
    template="Classify the sentiment of this tweet as POSITIVE, NEGATIVE, or NEUTRAL: {text}\nReturn only the sentiment label."
)

# Prompt for Q&A
qa_prompt = PromptTemplate(
    input_variables=["query", "context"],
    template="Based on the latest tweets: {context}\nAnswer the query: {query}"
)

# LangChain RunnableSequences
sentiment_chain = sentiment_prompt | llm
qa_chain = qa_prompt | llm

# Sample tweets for simulation
SAMPLE_TWEETS = [
    {"text": "iPhone 16 camera is insane! Best photos ever!", "created_at": "2025-05-25T18:00:00"},
    {"text": "iPhone 16 battery life is disappointing :(", "created_at": "2025-05-25T18:01:00"},
    {"text": "Love the new iPhone 16 design, so sleek!", "created_at": "2025-05-25T18:02:00"},
    {"text": "iPhone 16 is overpriced, not worth it.", "created_at": "2025-05-25T18:03:00"},
    {"text": "The iPhone 16 performance is top-notch!", "created_at": "2025-05-25T18:04:00"}
]

# Function to simulate tweet ingestion
def simulate_tweets(max_tweets=5):
    selected_tweets = random.sample(SAMPLE_TWEETS, min(max_tweets, len(SAMPLE_TWEETS)))
    for i, tweet in enumerate(selected_tweets):
        tweet_text = tweet["text"]
        # Use Groq to classify sentiment
        sentiment = sentiment_chain.invoke({"text": tweet_text}).content.strip()
        tweet_data = {
            "text": tweet_text,
            "created_at": tweet["created_at"],
            "sentiment": sentiment
        }
        filename = f"{TWEETS_DIR}/tweet_{int(time.time())}_{i}.json"
        with open(filename, "w") as f:
            json.dump(tweet_data, f)
    return len(selected_tweets)

# Pathway schema for tweets
class TweetSchema(pw.Schema):
    text: str
    created_at: str
    sentiment: str

# Pathway pipeline to ingest tweets
data = pw.io.fs.read(
    TWEETS_DIR,
    format="json",
    schema=TweetSchema,
    mode="streaming"
)

# Pathway-based retrieval function
def retrieve_tweets(query, table, k=5):
    # Split query into keywords
    query_words = query.lower().split()
    
    # Define a UDF to compute relevance score
    @pw.udf
    def compute_score(text: str) -> int:
        return sum(1 for word in query_words if word in text.lower())
    
    # Add score column to table
    scored_table = table.select(*table, score=compute_score(table.text))
    
    # Sort by score and limit to top k
    top_tweets = scored_table.sort_by(scored_table.score, descending=True)[:k]
    
    # Convert to list of dictionaries
    result = top_tweets.select(text=top_tweets.text).to_pylist()
    return [{"text": row["text"]} for row in result]

# Function to compute sentiment distribution
def get_sentiment_distribution():
    tweets = [json.load(open(f)) for f in os.listdir(TWEETS_DIR) if f.endswith(".json")]
    sentiments = [t["sentiment"] for t in tweets]
    sentiment_counts = pd.Series(sentiments).value_counts().to_dict()
    return sentiment_counts

# Gradio interface function
def query_copilot(user_query):
    # Simulate new tweets
    simulate_tweets(max_tweets=5)
    
    # Retrieve relevant tweets
    retrieved = retrieve_tweets(user_query, data, k=5)
    context = "\n".join([row["text"] for row in retrieved])
    
    # Generate response with Groq
    response = qa_chain.invoke({"query": user_query, "context": context}).content
    
    # Create sentiment chart
    sentiment_counts = get_sentiment_distribution()
    fig = px.pie(
        values=list(sentiment_counts.values()),
        names=list(sentiment_counts.keys()),
        title="Sentiment Distribution"
    )
    
    return response, fig

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Real-Time Social Sentiment Copilot")
    gr.Markdown(f"Query live sentiment and insights about {PRODUCT_QUERY}")
    query_input = gr.Textbox(label="Ask a question")
    output_text = gr.Textbox(label="Response")
    output_plot = gr.Plot(label="Sentiment Distribution")
    submit_button = gr.Button("Submit")
    submit_button.click(
        fn=query_copilot,
        inputs=query_input,
        outputs=[output_text, output_plot]
    )

# Simulate real-time tweet stream
def simulate_stream():
    while True:
        simulate_tweets(max_tweets=5)
        time.sleep(10)  # Add new tweets every 10 seconds

if __name__ == "__main__":
    # Start streaming tweets in background
    import threading
    threading.Thread(target=simulate_stream, daemon=True).start()
    
    # Start Pathway pipeline
    pw.run(monitoring_level=pw.MonitoringLevel.NONE)
    
    # Launch Gradio interface
    demo.launch()