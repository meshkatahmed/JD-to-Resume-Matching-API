from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple
from .text_extraction_engine import TextExtractionEngine
from .text_processing_engine import TextProcessingEngine
from utils import build_token_set, calculate_cosine_similarity

def get_available_cvs(cvs_dir_path: str = "cvs") -> List[Path]:
    """Retrieve all supported CV files from the CVs directory."""
    cvs_dir = Path(cvs_dir_path)
    if not cvs_dir.exists():
        return []
    return [p for p in cvs_dir.iterdir() if p.is_file() and p.suffix.lower() in (".pdf", ".txt")]

def compute_similarity_scores(jd_path: Path, cvs: List[Path]) -> List[Tuple[Path, float]]:
    """Compute cosine similarity scores between a job description and a list of CVs."""
    extraction_engine = TextExtractionEngine()
    processing_engine = TextProcessingEngine()

    raw_jd = extraction_engine.extract(jd_path)
    raw_cvs = [extraction_engine.extract(cv) for cv in cvs]

    jd_tokens = processing_engine.process(raw_jd)
    cv_tokens_list = [processing_engine.process(raw_cv) for raw_cv in raw_cvs]

    jd_token_set = build_token_set(jd_tokens)
    cv_tokens_set = [build_token_set(cv_tokens) for cv_tokens in cv_tokens_list]

    master_token_sets = [jd_token_set.union(cv_tokens) for cv_tokens in cv_tokens_set]
    jd_counter = Counter(jd_tokens)

    results = []
    for i, cv_path in enumerate(cvs):
        cv_tokens = cv_tokens_list[i]
        cv_counter = Counter(cv_tokens)
        master_list = sorted(list(master_token_sets[i]))

        jd_vector = [jd_counter[token] for token in master_list]
        cv_vector = [cv_counter[token] for token in master_list]

        similarity = calculate_cosine_similarity(jd_vector, cv_vector)
        results.append((cv_path, similarity))
        
    return results
