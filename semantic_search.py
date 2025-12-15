import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch
from transformers import pipeline
import pandas as pd

class SemanticSearch:
    def __init__(self):
        self.model = self._load_model()
        self.paper_embeddings = None
        self.summarizer = self._load_summarizer()

    @st.cache_resource
    def _load_model(_self):
        """
        Loads the SentenceTransformer model. Cached by Streamlit.
        """
        return SentenceTransformer('all-MiniLM-L6-v2')
        
    @st.cache_resource
    def _load_summarizer(_self):
        # lightweight summarization model
        return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

    @st.cache_data
    def encode_papers(_self, papers_text):
        """
        Encodes a list of paper texts (e.g., abstracts) into embeddings.
        """
        return _self.model.encode(papers_text, convert_to_tensor=True)

    def search(self, query, paper_embeddings, papers_df, top_k=5):
        """
        Searches for the most relevant papers given a query.
        """
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        # Compute cosine similarity
        cos_scores = util.cos_sim(query_embedding, paper_embeddings)[0]
        
        # Find top_k results
        top_results = torch.topk(cos_scores, k=min(top_k, len(papers_df)))
        
        results = []
        for score, idx in zip(top_results[0], top_results[1]):
            paper = papers_df.iloc[idx.item()]
            results.append({
                'score': score.item(),
                'title': paper['title'],
                'abstract': paper['abstract'],
                'published': paper['published'],
                'pdf_url': paper['pdf_url']
            })
            
        return results

    def get_recommendations(self, papers, paper_index, k=5):
        """
        Finds similar papers to the one at paper_index.
        """
        # 'papers' is a list of paper objects/dicts, and paper_embeddings
        # has been pre-computed and stored in self.paper_embeddings
        if self.paper_embeddings is None:
            raise ValueError("Paper embeddings have not been computed. Call encode_papers first.")
            
        # the embedding of the target paper
        target_embedding = self.paper_embeddings[paper_index]
        
        # Compute cosine similarity with all papers
        similarities = util.cos_sim(target_embedding, self.paper_embeddings)[0]
        
        # Get top k+1 results (including itself)
        top_results = torch.topk(similarities, k=k+1)
        
        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            idx = int(idx)
            # Skip the paper itself (similarity ~ 1.0)
            if idx != paper_index:
                results.append({
                    'paper': papers[idx],
                    'score': float(score)
                })
                
        return results

    def summarize_text(self, text):
        """
        Summarizes the given text using the loaded summarization pipeline.
        """
        # Truncate text if it's too long for the model (approx limit)
        max_input_length = 1024
        if len(text) > max_input_length:
            text = text[:max_input_length]
            
        try:
            summary = self.summarizer(text, max_length=60, min_length=30, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            return f"Error generating summary: {str(e)}"
