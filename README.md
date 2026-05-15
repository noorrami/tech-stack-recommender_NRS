# Tech Stack Recommender - Noor R Saad

Content-based job recommendation system using TF-IDF and Cosine Similarity.

## Features
- Loads job postings from `raw_skills.csv`
- Normalizes skills using synonym mapping and lemmatization
- Extracts job titles from URLs to enrich text
- Uses TF-IDF vectorization with optimal hyperparameters (ngram_range=(1,3), max_features=3000)
- Recommends top-N jobs based on user input skills
- Evaluates with Precision@k and checks for overfitting

## Installation
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## Usage
```bash
python main.py
```

## Project Structure
```
tech-stack-recommender/
├── data/                # raw_skills.csv
├── notebooks/           # analysis.ipynb
├── src/                 # source modules
│   ├── preprocess.py    # data cleaning & normalization
│   ├── recommender.py   # TF-IDF + cosine similarity
│   └── evaluate.py      # precision@k, overfitting, grid search
├── requirements.txt
├── README.md
└── main.py              # CLI entry point
```

## Evaluation Results
- Precision@1: ~0.78
- Precision@3: ~0.91
- Precision@5: ~0.95
- No overfitting detected.

