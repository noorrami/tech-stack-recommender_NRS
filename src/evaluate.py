import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from .recommender import TechStackRecommender

def precision_at_k(recommender, df, k=5, test_ratio=0.3, num_trials=2):
    """Evaluate Precision@k using leave-some-out (hide test_ratio of skills)."""
    precisions = []
    for _ in range(num_trials):
        hits = 0
        total = 0
        for idx, skills in enumerate(df['skills_list']):
            if len(skills) < 2:
                continue
            n_test = max(1, int(len(skills) * test_ratio))
            test_indices = np.random.choice(len(skills), n_test, replace=False)
            train_skills = [skills[i] for i in range(len(skills)) if i not in test_indices]
            if not train_skills:
                continue
            train_str = ' '.join(train_skills)
            recs = recommender.recommend(train_str, top_n=k)
            if df.iloc[idx]['job_id'] in recs['job_id'].values:
                hits += 1
            total += 1
        precisions.append(hits / total if total > 0 else 0)
    return np.mean(precisions)

def overfitting_check(recommender, df, test_size=0.2):
    """Compare average similarity within training vs. between train-test."""
    train_idx, test_idx = train_test_split(df.index, test_size=test_size, random_state=42)
    train_tfidf = recommender.get_tfidf_matrix()[train_idx]
    test_tfidf = recommender.get_tfidf_matrix()[test_idx]
    train_sim = cosine_similarity(train_tfidf)
    np.fill_diagonal(train_sim, np.nan)
    mean_train_sim = np.nanmean(train_sim)
    cross_sim = cosine_similarity(train_tfidf, test_tfidf)
    mean_cross_sim = np.mean(cross_sim)
    gap = mean_train_sim - mean_cross_sim
    return mean_train_sim, mean_cross_sim, gap

def grid_search_tuning(df, param_grid, k=5, test_ratio=0.3, num_trials=2):
    """Simple grid search for TfidfVectorizer hyperparameters."""
    best_precision = -1
    best_params = None
    results = []
    from sklearn.feature_extraction.text import TfidfVectorizer
    for ngram_range in param_grid['ngram_range']:
        for max_features in param_grid['max_features']:
            for min_df in param_grid['min_df']:
                for max_df in param_grid['max_df']:
                    print(f"Testing: ngram={ngram_range}, max_features={max_features}, min_df={min_df}, max_df={max_df}")
                    vec_params = {
                        'stop_words': 'english',
                        'ngram_range': ngram_range,
                        'max_features': max_features,
                        'min_df': min_df,
                        'max_df': max_df
                    }
                    recommender = TechStackRecommender(df, vectorizer_params=vec_params)
                    p = precision_at_k(recommender, df, k=k, test_ratio=test_ratio, num_trials=num_trials)
                    results.append((vec_params, p))
                    if p > best_precision:
                        best_precision = p
                        best_params = vec_params
    return best_params, best_precision, results