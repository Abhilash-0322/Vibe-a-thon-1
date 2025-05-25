# Real-Time Social Sentiment Copilot

## Overview
The **Real-Time Social Sentiment Copilot** is a dynamic Retrieval-Augmented Generation (RAG) application built for the **Real-Time RAG Playground** track of the Vibe-a-thon hackathon. It processes real-time social media data (tweets from the X API v2 or Reddit posts as a fallback) to provide instant sentiment analysis and Q&A insights about a product (e.g., "iPhone 16"). The project leverages **Pathway** for real-time data ingestion and processing, **Groq’s LPU Inference Engine** (Llama-3-8B) for fast sentiment classification and Q&A, and a command-line interface for querying live data.

### Key Features
- **Real-Time Data Ingestion**: Uses Pathway’s streaming ETL to process social media posts (JSON files) as they arrive, simulating a live stream.
- **Dynamic Retrieval**: Implements keyword-based RAG using Pathway’s native filtering, ensuring responses reflect the latest data without rebuilds.
- **Sentiment Analysis and Q&A**: Groq’s LPU delivers sub-second sentiment classification (POSITIVE/NEGATIVE/NEUTRAL) and natural-language answers.
- **Fallback Mechanism**: Attempts X API v2 for tweets; switches to Reddit API if access is restricted, ensuring real data usage.
- **Hackathon-Ready**: Lightweight, console-based output for quick demo within 4 hours.

### Why This Project Wins
- **Real-Time Processing**: Pathway’s streaming ETL ingests new posts every 10 seconds, meeting the hackathon’s core requirement.
- **Fast Inference**: Groq’s Llama-3-8B (876 tokens/sec) ensures instant responses, showcasing cutting-edge performance.
- **Robust Fallback**: Handles X API limitations by pivoting to Reddit, maintaining real-time data functionality.
- **Creative Use Case**: Analyzing live product sentiment (e.g., iPhone 16) is engaging and relevant, with clear demo potential.

## Setup Instructions
### Prerequisites
- **Python**: 3.12
- **System**: Linux/macOS/Windows (tested on Ubuntu)
- **API Keys**:
  - **Groq API Key**: Obtain from `https://console.grok.ai`.
  - **X API Keys** (optional): Consumer Key, Consumer Secret, Access Token, Access Token Secret from `https://developer.x.com`.
  - **Reddit API Keys** (fallback): Client ID, Client Secret from `https://www.reddit.com/prefs/apps`.

### Installation
1. **Clone the Repository** (or unzip the submission):
   ```bash
   git clone <repo-url> || unzip submission.zip
   cd Vibe-a-thon-1


2. **Set Up Virtual Environment**
    python -m venv .venv
    source .venv/bin/activate

3. **Install Dependencies**
    pip install -r requirements.txt

4. **Configure Environment Variables**
    cp .env.example .env

5. **Edit .env with your API keys**
    nano .env

6. **Run The Application**
    python app.py

