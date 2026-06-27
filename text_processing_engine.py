import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

class TextProcessingEngine:
    """
    Tokenizes and normalizes raw text into a clean list of words using nltk.

    Normalization pipeline:
      1. Lowercase
      2. Tokenize (nltk.word_tokenize)
      3. Drop punctuation / non-alphabetic tokens
      4. Drop stopwords (the, is, and, ...)
      5. Lemmatize (running -> run, companies -> company)
      6. Drop very short tokens (length < 2)
    """

    _NLTK_RESOURCES = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
    }

    def __init__(self, min_token_length: int = 2):
        self._ensure_nltk_resources()
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.min_token_length = min_token_length
        self._punct_table = str.maketrans("", "", string.punctuation)

    @classmethod
    def _ensure_nltk_resources(cls) -> None:
        """Download any missing nltk corpora/models quietly, only once."""
        for resource_path, package_name in cls._NLTK_RESOURCES.items():
            try:
                nltk.data.find(resource_path)
            except LookupError:
                nltk.download(package_name, quiet=True)

    def process(self, raw_text: str) -> list[str]:
        lowered = raw_text.lower()
        no_punct = lowered.translate(self._punct_table)
        tokens = word_tokenize(no_punct)

        clean_tokens = []
        for token in tokens:
            if not token.isalpha():
                continue
            if token in self.stop_words:
                continue
            if len(token) < self.min_token_length:
                continue
            lemma = self.lemmatizer.lemmatize(token)
            clean_tokens.append(lemma)

        return clean_tokens