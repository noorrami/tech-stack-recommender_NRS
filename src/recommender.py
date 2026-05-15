import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .preprocess import normalize_skill

class TechStackRecommender:
    def __init__(self, df, vectorizer_params=None):
        """
        df: DataFrame with columns 'skills_text', 'job_id', 'skills_list', 'url', 'job_title_from_url'
        vectorizer_params: dict for TfidfVectorizer (e.g., {'ngram_range': (1,3), 'max_features': 3000})
        """
        self.df = df
        if vectorizer_params is None:
            vectorizer_params = {
                'stop_words': 'english',
                'ngram_range': (1, 3),
                'max_features': 3000,
                'min_df': 1,
                'max_df': 0.9
            }
        self.vectorizer = TfidfVectorizer(**vectorizer_params)
        self.tfidf_matrix = self.vectorizer.fit_transform(df['skills_text'])

    def recommend(self, user_skills_str, top_n=5):
        """Return top_n job recommendations for given user skills."""
        # Normalize user input
        normalized = [normalize_skill(s) for s in user_skills_str.split()]
        user_text = ' '.join(normalized)
        user_vec = self.vectorizer.transform([user_text])
        similarities = cosine_similarity(user_vec, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[::-1][:top_n]
        recs = self.df.iloc[top_indices][['job_id', 'skills_list', 'job_title_from_url', 'url']].copy()
        recs['similarity'] = similarities[top_indices]
        return recs

    def get_vectorizer(self):
        return self.vectorizer

    def get_tfidf_matrix(self):
        return self.tfidf_matrix