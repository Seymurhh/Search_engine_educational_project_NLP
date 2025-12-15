import arxiv
import pandas as pd
import re
import nltk
import gensim
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import streamlit as st

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

import os

@st.cache_data
def fetch_arxiv_papers(query="cat:cs.RO OR cat:cs.AI", max_results=100):
    """
    Fetches papers from arXiv based on the query.
    """
    print(f"Fetching up to {max_results} papers for query: {query}...")
    # Configure client with delay to respect rate limits
    client = arxiv.Client(
        page_size=100,
        delay_seconds=3,
        num_retries=3
    )
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    papers = []
    for result in client.results(search):
        papers.append({
            'title': result.title,
            'abstract': result.summary,
            'published': result.published,
            'categories': result.categories,
            'pdf_url': result.pdf_url
        })
    
    df = pd.DataFrame(papers)
    print(f"Fetched {len(df)} papers.")
    return df

@st.cache_data
def preprocess_text(text):
    """
    Preprocesses the text: lowercase, remove punctuation, remove stopwords, lemmatize.
    """
    # Lowercase
    text = text.lower()
    
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    # Add domain specific stopwords
    custom_stops = {'paper', 'method', 'result', 'proposed', 'approach', 'algorithm', 'model', 'system', 'using', 'based', 'show', 'study', 'new'}
    stop_words.update(custom_stops)
    
    tokens = [w for w in tokens if w not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    
    return tokens

def make_bigrams(texts):
    """
    Creates bigrams for a list of tokenized texts.
    """
    print("Generating Bigrams...")
    # Build the bigram models
    bigram = gensim.models.Phrases(texts, min_count=5, threshold=10)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    
    return [bigram_mod[doc] for doc in texts]

def save_to_csv(df, filepath="arxiv_dataset.csv"):
    """
    Saves the dataframe to a CSV file.
    """
    if os.path.exists(filepath):
        df.to_csv(filepath, index=False)
    else:
        df.to_csv(filepath, index=False)
    return filepath

def load_from_csv(filepath="arxiv_dataset.csv"):
    """
    Loads the dataframe from a CSV file.
    """
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        return pd.DataFrame()

if __name__ == "__main__":
    # Test the loader
    df = fetch_arxiv_papers(max_results=10)
    print(df.head())
    if not df.empty:
        print("\nExample processed abstract:")
        print(preprocess_text(df.iloc[0]['abstract']))
