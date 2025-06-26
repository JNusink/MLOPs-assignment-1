import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load the IMDB dataset
data = pd.read_csv('IMDB Dataset.csv')

# Prepare features and labels
X = data['review']
y = data['sentiment']

# Create and train the pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('nb', MultinomialNB())
])

# Train the model
pipeline.fit(X, y)

# Save the trained model
joblib.dump(pipeline, 'sentiment_model.pkl')