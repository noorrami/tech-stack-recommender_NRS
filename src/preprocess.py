import pandas as pd
import ast
import re
from nltk.stem import WordNetLemmatizer

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()
MAX_SKILL_WORDS = 5

# Synonym mapping dictionary
SKILL_SYNONYMS = {
    'ml': 'machine learning',
    'ai': 'artificial intelligence',
    'nlp': 'natural language processing',
    'dl': 'deep learning',
    'etl': 'extract transform load',
    'r': 'r programming',
    'python': 'python programming',
    'javascript': 'js',
    'c#': 'csharp',
    'c++': 'cpp',
    'aws': 'amazon web services',
    'azure': 'microsoft azure',
    'gcp': 'google cloud platform',
    'sql': 'structured query language',
    'nosql': 'non-relational database',
    'tableau': 'data visualization tableau',
    'power bi': 'powerbi',
    'spark': 'apache spark',
    'hadoop': 'apache hadoop',
    'scikit-learn': 'sklearn',
    'tensorflow': 'tf',
    'pytorch': 'torch'
}

def normalize_skill(skill_name: str) -> str:
    """Apply synonym mapping and lemmatization to a single skill."""
    skill_name = skill_name.strip().lower()
    if skill_name in SKILL_SYNONYMS:
        skill_name = SKILL_SYNONYMS[skill_name]
    lemmatized_words = [lemmatizer.lemmatize(word) for word in skill_name.split()]
    return ' '.join(lemmatized_words)

def extract_job_title_from_url(url: str) -> str:
    """Extract a cleaner job title from the Dice.com URL."""
    if pd.isna(url):
        return ''
    match = re.search(r'detail/([^/?]+)', url)
    if match:
        title = re.sub(r'-\d+$', '', match.group(1).replace('-', ' ')).strip().lower()
        if title in ['jobs', 'job']:
            return ''
        return title
    return ''

def parse_skills(skill_str):
    """Convert raw_skills string to a list of normalized skill phrases."""
    if pd.isna(skill_str):
        return []
    try:
        skills_list = ast.literal_eval(skill_str)
        if isinstance(skills_list, list):
            skills_list = [s.strip().lower() for s in skills_list 
                           if isinstance(s, str) and len(s.strip()) > 0 
                           and s.strip() not in ("''", '""')]
            skills_list = [normalize_skill(s) for s in skills_list]
            skills_list = [s for s in skills_list if len(s.split()) <= MAX_SKILL_WORDS]
            return skills_list
        else:
            return []
    except:
        words = re.findall(r'\b[a-zA-Z0-9#\+\.]+\b', skill_str)
        words = [normalize_skill(w) for w in words if len(w) > 1]
        words = [w for w in words if len(w.split()) <= MAX_SKILL_WORDS]
        return words

def load_and_preprocess_data(csv_path: str):
    """Load raw_skills.csv and return preprocessed DataFrame."""
    df = pd.read_csv(csv_path)
    df['skills_list'] = df['raw_skills'].apply(parse_skills)
    df = df[df['skills_list'].apply(len) > 0].reset_index(drop=True)
    df['job_id'] = df.index
    df['job_title_from_url'] = df['url'].apply(extract_job_title_from_url)
    df['skills_text'] = df.apply(
        lambda row: ' '.join(row['skills_list']) + 
                    (' ' + row['job_title_from_url'] if row['job_title_from_url'] else ''), 
        axis=1
    )
    return df