from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple
from .text_extraction_engine import TextExtractionEngine
from .text_processing_engine import TextProcessingEngine
from .tfidf_vectorizer import TFIDFVectorizer
from utils import build_token_set, calculate_cosine_similarity

def get_available_cvs(cvs_dir_path: str = "cvs") -> List[Path]:
    """Retrieve all supported CV files from the CVs directory."""
    cvs_dir = Path(cvs_dir_path)
    print(cvs_dir)
    if not cvs_dir.exists():
        return []
    return [p for p in cvs_dir.iterdir() if p.is_file() and p.suffix.lower() in (".pdf", ".txt")]

def compute_similarity_scores(jd_path: Path, cv_paths: List[Path]) -> List[Tuple[Path, float]]:
    """Compute cosine similarity scores between a job description and a list of CVs.
    
    Vectorization strategy: Continuous Bag of Words (CBOW) — raw term-count vectors
    built per (JD, CV) pair over their shared vocabulary.
    """
    extraction_engine = TextExtractionEngine()
    processing_engine = TextProcessingEngine()

    raw_jd = extraction_engine.extract(jd_path)
    
    raw_cvs = [extraction_engine.extract(cv_path) for cv_path in cv_paths]

    jd_tokens = processing_engine.process(raw_jd)
    cv_tokens_list = [processing_engine.process(raw_cv) for raw_cv in raw_cvs]

    jd_token_set = build_token_set(jd_tokens)
    cv_tokens_set = [build_token_set(cv_tokens) for cv_tokens in cv_tokens_list]

    master_token_sets = [jd_token_set.union(cv_tokens) for cv_tokens in cv_tokens_set]
    jd_counter = Counter(jd_tokens)

    results = []
    
    for i, cv_path in enumerate(cv_paths):
        cv_tokens = cv_tokens_list[i]
        cv_counter = Counter(cv_tokens)
        master_list = sorted(list(master_token_sets[i]))

        jd_vector = [jd_counter[token] for token in master_list]
        cv_vector = [cv_counter[token] for token in master_list]
        
        similarity = calculate_cosine_similarity(jd_vector, cv_vector)
        
        results.append((cv_path, similarity))
        
    return results


def compute_similarity_scores_tfidf(jd_path: Path, cv_paths: List[Path]) -> List[Tuple[Path, float]]:
    """Compute cosine similarity scores between a job description and a list of CVs.

    Vectorization strategy: TF-IDF — a single TFIDFVectorizer is fitted over the
    entire corpus (JD + all CVs) so that IDF weights reflect term rarity across
    the full document collection. Each document is then transformed into a TF-IDF
    vector and compared against the JD vector using cosine similarity.
    """
    extraction_engine = TextExtractionEngine()
    processing_engine = TextProcessingEngine()

    # Extract and tokenize all documents
    raw_jd = extraction_engine.extract(jd_path)
    raw_cvs = [extraction_engine.extract(cv_path) for cv_path in cv_paths]

    jd_tokens = processing_engine.process(raw_jd)
    cv_tokens_list = [processing_engine.process(raw_cv) for raw_cv in raw_cvs]

    # Build full corpus: JD is always the first document (index 0)
    corpus = [jd_tokens] + cv_tokens_list

    # Fit TF-IDF over the entire corpus so IDF weights are shared
    vectorizer = TFIDFVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Index 0 is the JD; indices 1..N are the CVs
    jd_vector = tfidf_matrix[0]

    results = []
    for i, cv_path in enumerate(cv_paths):
        cv_vector = tfidf_matrix[i + 1]
        similarity = calculate_cosine_similarity(jd_vector, cv_vector)
        results.append((cv_path, similarity))

    return results
