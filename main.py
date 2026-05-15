import warnings
import pandas as pd
from src.preprocess import load_and_preprocess_data
from src.recommender import TechStackRecommender
from src.evaluate import precision_at_k, overfitting_check, grid_search_tuning
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

def main():
    print("="*60)
    print("Tech Stack Recommender - Project 3")
    print("="*60)

    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    df = load_and_preprocess_data('data/raw_skills.csv')
    print(f"Loaded {len(df)} job postings.")

    # Optional: grid search (comment out if time-consuming)
    # param_grid = {
    #     'ngram_range': [(1,1), (1,2), (1,3)],
    #     'max_features': [1000, 2000, 3000],
    #     'min_df': [1, 2],
    #     'max_df': [0.8, 0.9]
    # }
    # best_params, best_prec, _ = grid_search_tuning(df, param_grid, k=5)
    # print(f"Best params: {best_params} (Prec@5={best_prec:.4f})")

    # Use best known parameters from tuning
    best_params = {
        'stop_words': 'english',
        'ngram_range': (1, 3),
        'max_features': 3000,
        'min_df': 1,
        'max_df': 0.9
    }
    recommender = TechStackRecommender(df, vectorizer_params=best_params)

    # Evaluation
    print("\nEvaluating Precision@k...")
    for k in [1, 3, 5, 10]:
        p = precision_at_k(recommender, df, k=k, test_ratio=0.3, num_trials=2)
        print(f"Precision@{k}: {p:.4f}")

    # Overfitting check
    print("\nOverfitting check...")
    train_sim, cross_sim, gap = overfitting_check(recommender, df)
    print(f"Mean similarity within training: {train_sim:.4f}")
    print(f"Mean similarity train-test: {cross_sim:.4f}")
    print(f"Gap: {gap:.4f}")
    if abs(gap) < 0.2:
        print("✅ No significant overfitting.")
    else:
        print("⚠️ Potential overfitting detected.")

    # Interactive CLI
    print("\n" + "="*60)
    print("Interactive Job Recommender")
    print("="*60)
    while True:
        user_input = input("\nEnter your skills (space-separated, e.g., 'python sql tableau') or 'quit' to exit: ")
        if user_input.lower() == 'quit':
            break
        if not user_input.strip():
            print("Please enter some skills.")
            continue
        recs = recommender.recommend(user_input, top_n=5)
        print("\nTop 5 recommended jobs:")
        for _, row in recs.iterrows():
            print(f"\nJob {int(row['job_id'])} (Similarity: {row['similarity']:.4f})")
            print(f"  Title: {row['job_title_from_url'][:60]}")
            print(f"  Skills: {', '.join(row['skills_list'][:8])}")
            if 'url' in row:
                print(f"  URL: {row['url'][:80]}...")
        print("-"*50)

if __name__ == "__main__":
    main()