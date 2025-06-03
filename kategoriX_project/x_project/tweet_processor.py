import tweepy
import pandas as pd
import joblib
from preprocessor import TextPreprocessor
import psycopg2
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Twitter API credentials
consumer_key = os.getenv('TWITTER_CONSUMER_KEY')
consumer_secret = os.getenv('TWITTER_CONSUMER_SECRET')
access_token = os.getenv('TWITTER_ACCESS_TOKEN')
access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')

# PostgreSQL connection parameters
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')

class TweetProcessor:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        # Initialize Twitter API client
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        self.api = tweepy.API(auth)
        
        # Load the trained model and vectorizer
        self.model = joblib.load('trained_model.joblib')
        self.vectorizer = joblib.load('tfidf_vectorizer.joblib')
        
        # Initialize database connection
        self.db_conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        self.cursor = self.db_conn.cursor()
        
        # Create table if not exists
        self.create_table()
        
    def create_table(self):
        """Create the predictions table if it doesn't exist"""
        create_table_query = """
        CREATE TABLE IF NOT EXISTS tbl_predict (
            id SERIAL PRIMARY KEY,
            tweet TEXT NOT NULL,
            kategori TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        self.cursor.execute(create_table_query)
        self.db_conn.commit()
    
    def fetch_tweets(self, query, count=100):
        """Fetch tweets using the Twitter API"""
        tweets = []
        try:
            for tweet in tweepy.Cursor(self.api.search_tweets, 
                                     q=query,
                                     lang="tr",
                                     tweet_mode="extended").items(count):
                tweets.append(tweet.full_text)
        except Exception as e:
            print(f"Error fetching tweets: {e}")
        return tweets
    
    def process_and_predict(self, tweets):
        """Process tweets and make predictions"""
        text_processor = TextPreprocessor()
        processed_tweets = [text_processor.preprocess_text(tweet) for tweet in tweets]
        vectorized_tweets = self.vectorizer.transform(processed_tweets)
        predictions = self.model.predict(vectorized_tweets)
        return predictions
    
    def save_to_database(self, tweets, predictions):
        """Save tweets and their predictions to PostgreSQL"""
        insert_query = """
        INSERT INTO tbl_predict (tweet, kategori)
        VALUES (%s, %s);
        """
        try:
            for tweet, prediction in zip(tweets, predictions):
                self.cursor.execute(insert_query, (tweet, prediction))
            self.db_conn.commit()
            print(f"{len(tweets)} tweet veritabanına kaydedildi.")
        except Exception as e:
            print(f"Veritabanına kaydetme hatası: {e}")
            self.db_conn.rollback()
    
    def close_connection(self):
        """Close database connection"""
        if hasattr(self, 'cursor') and self.cursor is not None:
            self.cursor.close()
        if hasattr(self, 'db_conn') and self.db_conn is not None:
            self.db_conn.close()
    
    def process_tweets(self, query, count=100):
        """Main method to orchestrate the entire process"""
        # Fetch tweets
        tweets = self.fetch_tweets(query, count)
        if not tweets:
            return
        
        # Process and predict
        predictions = self.process_and_predict(tweets)
        
        # Save to database
        self.save_to_database(tweets, predictions)
        
        return len(tweets)

if __name__ == "__main__":
    # Create .env file if it doesn't exist
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write("""TWITTER_CONSUMER_KEY=your_consumer_key
TWITTER_CONSUMER_SECRET=your_consumer_secret
TWITTER_ACCESS_TOKEN=your_access_token
TWITTER_ACCESS_TOKEN_SECRET=your_access_token_secret
DB_NAME=your_db_name
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_HOST=localhost
DB_PORT=5432""")
