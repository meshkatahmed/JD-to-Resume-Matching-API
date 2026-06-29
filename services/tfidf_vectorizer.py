import math
from collections import Counter
from typing import List


class TFIDFVectorizer:
    """
    A pure-Python TF-IDF vectorizer (no external ML libraries required).

    Follows the sklearn-style smooth IDF formula to avoid zero-division on
    terms that appear in every document:

        TF(t, d)  = count(t, d) / |d|
        IDF(t)    = log((1 + N) / (1 + df(t))) + 1
        weight    = TF × IDF

    Usage
    -----
        vectorizer = TFIDFVectorizer()

        # Build vocabulary and IDF weights from the full corpus
        vectorizer.fit(corpus)          # corpus: list[list[str]]

        # Get a TF-IDF vector for a single document
        vec = vectorizer.transform(tokens)  # tokens: list[str] -> list[float]

        # Or do both in one call (returns one vector per document)
        vectors = vectorizer.fit_transform(corpus)
    """

    def __init__(self) -> None:
        self._vocabulary: List[str] = []          # sorted list of unique terms
        self._idf: dict[str, float] = {}          # term -> IDF weight
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, corpus: List[List[str]]) -> "TFIDFVectorizer":
        """
        Build the vocabulary and compute IDF weights from *corpus*.

        Parameters
        ----------
        corpus : list of token lists
            One token list per document (output of TextProcessingEngine.process).
        """
        N = len(corpus)
        if N == 0:
            raise ValueError("corpus must contain at least one document")

        # Collect all unique terms and their document frequencies
        df: Counter = Counter()
        all_terms: set[str] = set()

        for tokens in corpus:
            unique_in_doc = set(tokens)
            all_terms.update(unique_in_doc)
            for term in unique_in_doc:
                df[term] += 1

        # Sort vocabulary for deterministic vector ordering
        self._vocabulary = sorted(all_terms)

        # Compute smooth IDF for every term in the vocabulary
        self._idf = {
            term: math.log((1 + N) / (1 + df[term])) + 1
            for term in self._vocabulary
        }

        self._fitted = True
        return self

    def transform(self, tokens: List[str]) -> List[float]:
        """
        Return a TF-IDF vector for *tokens* using the fitted vocabulary/IDF.

        Parameters
        ----------
        tokens : list[str]
            Pre-processed tokens for a single document.

        Returns
        -------
        list[float]
            A dense vector aligned to the fitted vocabulary.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")

        total_terms = len(tokens)
        if total_terms == 0:
            return [0.0] * len(self._vocabulary)

        tf: Counter = Counter(tokens)

        vector: List[float] = []
        for term in self._vocabulary:
            term_tf = tf[term] / total_terms          # raw TF (0 if absent)
            term_idf = self._idf.get(term, 1.0)       # IDF from fit (fallback 1)
            vector.append(term_tf * term_idf)

        return vector

    def fit_transform(self, corpus: List[List[str]]) -> List[List[float]]:
        """
        Convenience method: fit on *corpus* then return one vector per document.

        Parameters
        ----------
        corpus : list of token lists

        Returns
        -------
        list[list[float]]
            TF-IDF matrix — one row per document, columns aligned to vocabulary.
        """
        self.fit(corpus)
        return [self.transform(tokens) for tokens in corpus]

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def vocabulary(self) -> List[str]:
        """The fitted vocabulary (sorted list of unique terms)."""
        return list(self._vocabulary)

    @property
    def idf_weights(self) -> dict:
        """The fitted IDF weights keyed by term."""
        return dict(self._idf)
