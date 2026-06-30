"""
benchmark/scorers.py
--------------------
Scorer registry for the benchmark framework.

Each scorer is a plain Python function with the signature::

    def my_scorer(
        jd_text: str,
        resume_text: str,
        extraction_engine: TextExtractionEngine,
        processing_engine: TextProcessingEngine,
    ) -> float:
        ...
        return similarity   # float in [0.0, 1.0]

Register it with the ``@register("my_method_name")`` decorator and it
automatically becomes available via ``--methods my_method_name`` in the CLI.

Built-in methods
~~~~~~~~~~~~~~~~
- ``cbow``    — Raw term-count cosine similarity
- ``tfidf``   — TF-IDF cosine similarity (pairwise corpus)
- ``jaccard`` — Set intersection / union
- ``overlap`` — Szymkiewicz–Simpson (resume coverage of JD terms)
- ``bm25``    — BM25-inspired with TF saturation + length normalisation
- ``sbert``   — Sentence-BERT dense embeddings (all-MiniLM-L6-v2 by default)
                 Model is downloaded once on first use (~90 MB) and cached
                 in memory for the full benchmark run.
                 Change ``SBERT_MODEL_NAME`` below to swap the model.

Adding a new method
~~~~~~~~~~~~~~~~~~~
1. Write a scoring function that accepts (jd_text, resume_text,
   extraction_engine, processing_engine) and returns a float in [0, 1].
2. Decorate it with @register("your_method_name").
3. That's it — the runner and CLI pick it up automatically.

Example (add anywhere below the existing scorers)::

    @register("my_method")
    def _score_my_method(jd_text, resume_text, extraction_engine, processing_engine):
        # ... your logic ...
        return similarity_float   # must be in [0.0, 1.0]
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Callable

# ── Project imports (runner.py adds project root to sys.path first) ─────────
from services.text_extraction_engine import TextExtractionEngine
from services.text_processing_engine import TextProcessingEngine
from services.tfidf_vectorizer import TFIDFVectorizer
from utils import build_token_set, calculate_cosine_similarity

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# ScorerFn type alias — all scorers must match this signature.
ScorerFn = Callable[
    [str, str, TextExtractionEngine, TextProcessingEngine],
    float,
]

#: Global registry: method_name -> scorer function
REGISTRY: dict[str, ScorerFn] = {}


def register(name: str) -> Callable[[ScorerFn], ScorerFn]:
    """
    Decorator that registers a scorer function under *name*.

    Usage::

        @register("my_method")
        def _score_my_method(jd_text, resume_text, ee, pe):
            ...
            return 0.75
    """
    def decorator(fn: ScorerFn) -> ScorerFn:
        if name in REGISTRY:
            raise ValueError(
                f"A scorer named '{name}' is already registered. "
                "Choose a unique name or remove the old registration."
            )
        REGISTRY[name] = fn
        return fn
    return decorator


def available_methods() -> list[str]:
    """Return a sorted list of all registered method names."""
    return sorted(REGISTRY.keys())


# ---------------------------------------------------------------------------
# Built-in scorer 1: CBOW (Continuous Bag of Words)
# ---------------------------------------------------------------------------

@register("cbow")
def _score_cbow(
    jd_text: str,
    resume_text: str,
    extraction_engine: TextExtractionEngine,
    processing_engine: TextProcessingEngine,
) -> float:
    """
    Cosine similarity on raw term-count (CBOW) vectors.

    Builds a shared vocabulary from both documents, counts term frequencies,
    and computes cosine similarity. Fast and simple baseline.
    """
    jd_tokens     = processing_engine.process(extraction_engine.extract(jd_text))
    resume_tokens = processing_engine.process(extraction_engine.extract(resume_text))

    master_list = sorted(build_token_set(jd_tokens).union(build_token_set(resume_tokens)))

    jd_counter     = Counter(jd_tokens)
    resume_counter = Counter(resume_tokens)

    jd_vec     = [jd_counter[t]     for t in master_list]
    resume_vec = [resume_counter[t] for t in master_list]

    return calculate_cosine_similarity(jd_vec, resume_vec)


# ---------------------------------------------------------------------------
# Built-in scorer 2: TF-IDF (pairwise corpus)
# ---------------------------------------------------------------------------

@register("tfidf")
def _score_tfidf(
    jd_text: str,
    resume_text: str,
    extraction_engine: TextExtractionEngine,
    processing_engine: TextProcessingEngine,
) -> float:
    """
    TF-IDF cosine similarity.

    Fits the TFIDFVectorizer on a 2-document corpus (JD + resume) so that IDF
    weights reflect term rarity within this specific pair. Terms appearing in
    both documents get lower IDF weight; distinctive terms get higher weight.
    """
    jd_tokens     = processing_engine.process(extraction_engine.extract(jd_text))
    resume_tokens = processing_engine.process(extraction_engine.extract(resume_text))

    corpus = [jd_tokens, resume_tokens]
    vectorizer   = TFIDFVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    return calculate_cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])


# ---------------------------------------------------------------------------
# Built-in scorer 3: Jaccard Similarity
# ---------------------------------------------------------------------------

@register("jaccard")
def _score_jaccard(
    jd_text: str,
    resume_text: str,
    extraction_engine: TextExtractionEngine,
    processing_engine: TextProcessingEngine,
) -> float:
    """
    Jaccard similarity on token sets.

        J(A, B) = |A ∩ B| / |A ∪ B|

    Purely set-based — does not account for term frequency but is fast and
    interpretable. Useful as a lower-bound sanity check.
    """
    jd_set     = build_token_set(processing_engine.process(extraction_engine.extract(jd_text)))
    resume_set = build_token_set(processing_engine.process(extraction_engine.extract(resume_text)))

    intersection = len(jd_set & resume_set)
    union        = len(jd_set | resume_set)

    return intersection / union if union else 0.0


# ---------------------------------------------------------------------------
# Built-in scorer 4: Overlap Coefficient (resume coverage of JD)
# ---------------------------------------------------------------------------

@register("overlap")
def _score_overlap(
    jd_text: str,
    resume_text: str,
    extraction_engine: TextExtractionEngine,
    processing_engine: TextProcessingEngine,
) -> float:
    """
    Overlap coefficient (also called Szymkiewicz–Simpson coefficient).

        overlap(A, B) = |A ∩ B| / min(|A|, |B|)

    Measures what fraction of the *smaller* set's terms appear in the other.
    Because resumes are usually shorter than JDs, this effectively measures
    how well the resume covers the JD's vocabulary — a useful hiring signal.
    """
    jd_set     = build_token_set(processing_engine.process(extraction_engine.extract(jd_text)))
    resume_set = build_token_set(processing_engine.process(extraction_engine.extract(resume_text)))

    intersection = len(jd_set & resume_set)
    min_size     = min(len(jd_set), len(resume_set))

    return intersection / min_size if min_size else 0.0


# ---------------------------------------------------------------------------
# Built-in scorer 5: BM25-inspired score
# ---------------------------------------------------------------------------

@register("bm25")
def _score_bm25(
    jd_text: str,
    resume_text: str,
    extraction_engine: TextExtractionEngine,
    processing_engine: TextProcessingEngine,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """
    BM25-inspired similarity (query = JD tokens, document = resume tokens).

    BM25 is the industry-standard ranking function used by Elasticsearch and
    most search engines. It improves on raw TF by:
      - Saturating term frequency (k1 parameter) — more occurrences help, but
        with diminishing returns.
      - Penalising document length (b parameter) — longer resumes are not
        unfairly favoured.

    Because we have only one document (the resume), IDF is computed
    analytically: terms present in the resume get IDF = log(2) ≈ 0.693,
    absent terms get IDF = log(2/(0+1)) which simplifies here to treating
    only matched terms. The score is then normalised to [0, 1] by dividing
    by the maximum possible BM25 score for this query.

    Parameters
    ----------
    k1 : float
        Term frequency saturation. Typical range 1.2–2.0. Default 1.5.
    b : float
        Length normalisation. 0 = no normalisation, 1 = full. Default 0.75.
    """
    jd_tokens     = processing_engine.process(extraction_engine.extract(jd_text))
    resume_tokens = processing_engine.process(extraction_engine.extract(resume_text))

    if not jd_tokens or not resume_tokens:
        return 0.0

    # Build IDF from a minimal 2-doc corpus (JD + resume)
    N = 2
    jd_set     = set(jd_tokens)
    resume_set = set(resume_tokens)

    resume_counter  = Counter(resume_tokens)
    avg_dl          = len(resume_tokens)       # single document, so avgdl = dl
    dl              = len(resume_tokens)

    bm25_score   = 0.0
    max_possible = 0.0

    for term in set(jd_tokens):          # iterate unique JD terms (query terms)
        df  = (1 if term in jd_set else 0) + (1 if term in resume_set else 0)
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)

        # BM25 term score for the resume document
        tf = resume_counter.get(term, 0)
        tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
        bm25_score += idf * tf_norm

        # Max possible: term appears in resume with same frequency as JD
        tf_max = Counter(jd_tokens)[term]
        tf_max_norm = (tf_max * (k1 + 1)) / (tf_max + k1 * (1 - b + b * dl / avg_dl))
        max_possible += idf * tf_max_norm

    return bm25_score / max_possible if max_possible > 0 else 0.0


# ---------------------------------------------------------------------------
# Built-in scorer 6: Sentence-BERT (SBERT) — semantic dense embeddings
# ---------------------------------------------------------------------------

#: Swap this string to use a different Sentence-Transformers model, e.g.:
#:   "all-mpnet-base-v2"         — higher quality, ~420 MB
#:   "paraphrase-MiniLM-L3-v2"  — fastest, ~60 MB
#:   "multi-qa-MiniLM-L6-cos-v1"— tuned for QA / retrieval tasks
SBERT_MODEL_NAME: str = "all-MiniLM-L6-v2"

# Module-level cache so the model is downloaded + loaded only once per process,
# regardless of how many rows the benchmark processes.
_sbert_model_cache: dict[str, object] = {}


@register("sbert")
def _score_sbert(
    jd_text: str,
    resume_text: str,
    extraction_engine: TextExtractionEngine,
    processing_engine: TextProcessingEngine,   # not used — SBERT has its own tokenizer
) -> float:
    """
    Semantic similarity via Sentence-BERT dense embeddings.

    Unlike the lexical methods (CBOW, TF-IDF, BM25 …) that rely on shared
    vocabulary, SBERT encodes each document into a fixed-size dense vector
    using a fine-tuned BERT model. Two semantically similar texts — even
    without overlapping words — will have vectors close in cosine space.

    Model: ``all-MiniLM-L6-v2`` (default, ~90 MB)
      - 22 M parameters, 384-dim embeddings
      - Fine-tuned on 1B+ sentence pairs for semantic similarity
      - Runs on CPU in ~5–50 ms per pair (no GPU required)

    To use a different model, change ``SBERT_MODEL_NAME`` at the top of
    this file before running the benchmark.

    Notes
    -----
    - The ``processing_engine`` argument is intentionally ignored. SBERT
      has its own sub-word tokenizer (WordPiece) and performs best on
      natural-language text — not on lemmatized token lists.
    - The model is loaded lazily on the first call and cached for the
      rest of the benchmark run. Expect a one-time ~2–10 s delay on the
      first row.
    - Requires: ``pip install sentence-transformers``
    """
    try:
        from sentence_transformers import SentenceTransformer, util as st_util
    except ImportError as exc:
        raise ImportError(
            "The 'sbert' scorer requires the sentence-transformers package.\n"
            "Install it with:  pip install sentence-transformers"
        ) from exc

    # Lazy load + cache the model
    model_name = SBERT_MODEL_NAME
    if model_name not in _sbert_model_cache:
        print(f"\n  [sbert] Loading model '{model_name}' (first use — may download ~90 MB) …",
              flush=True)
        _sbert_model_cache[model_name] = SentenceTransformer(model_name)
        print("  [sbert] Model ready.\n", flush=True)

    model = _sbert_model_cache[model_name]

    # Extract raw text — let SBERT's own tokenizer handle it
    jd_raw     = extraction_engine.extract(jd_text)
    resume_raw = extraction_engine.extract(resume_text)

    # Encode both documents in a single batched call (faster than two calls)
    embeddings = model.encode(          # type: ignore[union-attr]
        [jd_raw, resume_raw],
        convert_to_tensor=True,
        show_progress_bar=False,
    )

    # Cosine similarity between the two embedding vectors, clamped to [0, 1]
    similarity = float(st_util.cos_sim(embeddings[0], embeddings[1]).item())
    return max(0.0, min(1.0, similarity))
