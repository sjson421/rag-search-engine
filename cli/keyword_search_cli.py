import argparse
import string
from nltk.stem import PorterStemmer
from collections import defaultdict, Counter
import pickle
import math
import os

from loader import load_movies, load_stopwords
from constants.constants import BM25_K1, BM25_B

class InvertedIndex:
    def __init__(self):
        CACHE_DIR = "./cache"
        self.INDEX_PATH = os.path.join(CACHE_DIR, "index.pkl")
        self.DOCMAP_PATH = os.path.join(CACHE_DIR, "docmap.pkl")
        self.TERM_FREQ_PATH = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.DOC_LENGTHS_PATH = os.path.join(CACHE_DIR, "doc_lengths.pkl")

        self.index = defaultdict(set)
        self.docmap = {}
        self.term_freq = defaultdict(Counter)

        self.doc_lengths = {}

    def __add_document(self, doc_id, text):
        token_count = 0
        for token in tokenize_text(text):
            self.index[token].add(doc_id)
            self.term_freq[doc_id][token] += 1
            token_count += 1
        self.doc_lengths[doc_id] = token_count

    def get_documents(self, term):
        return sorted(self.index[term])

    def get_tf(self, doc_id: int, term: str) -> int:
        if doc_id not in self.term_freq:
            return 0
        return self.term_freq[doc_id][term]

    def build(self, movies):
        for movie in movies:
            id = movie["id"]
            self.docmap[id] = movie
            text = f"{movie["title"]} {movie["description"]}"
            self.__add_document(movie["id"], text)

    def save(self) -> None:
        files = [
            (self.INDEX_PATH, self.index),
            (self.DOCMAP_PATH, self.docmap),
            (self.TERM_FREQ_PATH, self.term_freq),
            (self.DOC_LENGTHS_PATH, self.doc_lengths),
        ]
        for file in files:
            self._save(*file)

    def _save(self, file_path: str, obj: object) -> None:
        with open(file_path, "wb") as file:
            pickle.dump(obj, file)

    def load(self):
        self.index = self._load(self.INDEX_PATH)
        self.docmap = self._load(self.DOCMAP_PATH)
        self.term_freq = self._load(self.TERM_FREQ_PATH)
        self.doc_lengths = self._load(self.DOC_LENGTHS_PATH)

    def _load(self, file_path: str) -> None:
        with open(file_path, "rb") as file:
            return pickle.load(file)

    def get_bm25_idf(self, term: str) -> float:
        df = len(self.get_documents(term))
        return math.log((len(self.docmap) - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(self, doc_id, term, k1, b):
        tf = self.get_tf(doc_id, term)
        length_norm = 1 - b + b * self.doc_lengths[doc_id] / self.__get_avg_doc_length()
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def bm25(self, doc_id, term, k1, b):
        return self.get_bm25_tf(doc_id, term, k1, b) * self.get_bm25_idf(term)
    
    def bm25_search(self, query, limit, k1, b) -> list[tuple]:
        terms = tokenize_text(query)
        scores = defaultdict(float)
        for term in terms:
            for doc_id in self.get_documents(term):
                scores[doc_id] += self.bm25(doc_id, term, k1, b)
        return sorted(scores.items(), key = lambda x: x[1], reverse=True)[:limit]

inverted_index = InvertedIndex()

def _modify_string(s: str) -> str:
    return s.lower().translate(str.maketrans("", "", string.punctuation))

movies = load_movies()
stopwords = load_stopwords()
for i, word in enumerate(stopwords):
    stopwords[i] = _modify_string(word)

stemmer = PorterStemmer()
def tokenize_text(s: str) -> list[str]:
    s = _modify_string(s)
    return [stemmer.stem(word) for word in s.split() if word not in stopwords]

def tokenize_word(word: str) -> str:
    if word in stopwords:
        raise ValueError("Can't tokenize stopword")
    word = stemmer.stem(_modify_string(word))
    if len(word.split()) != 1:
        raise ValueError("Got token with size unequal to 1")
    return word

def search_movies(query: str) -> list[str]:
    query_tokens = tokenize_text(query)
    res = []

    for t in query_tokens:
        ids = inverted_index.get_documents(t)
        for id in ids:
            movie = inverted_index.docmap[id]
            res.append(f"{movie["title"]} {id}")
            if len(res) == 5:
                return res
    return res

def get_tf(doc_id, term):
    return inverted_index.get_tf(doc_id, term)

def get_idf(term):
    return math.log((len(movies) + 1) / (len(inverted_index.get_documents(term)) + 1))

def get_tf_idf(doc_id, term):
    return get_tf(doc_id, term) * get_idf(term)

def get_bm25_idf_command(term: str):
    return inverted_index.get_bm25_idf(tokenize_word(term))

def get_bm25_tf_command(doc_id, term, k1=BM25_K1, b=BM25_B):
    return inverted_index.get_bm25_tf(doc_id, tokenize_word(term), k1, b)

def get_bm25search_command(query: str, limit: int) -> list[tuple]:
    return inverted_index.bm25_search(query, limit, BM25_K1, BM25_B)

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build inverted index")

    tf_parser = subparsers.add_parser("tf", help="Get the term frequency of the provided term in the given document")
    tf_parser.add_argument("document_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="The term to get the frequency of in the given doc")

    idf_parser = subparsers.add_parser("idf", help="Return inverse document frequency value of provided term")
    idf_parser.add_argument("term", type=str, help="The term to get IDF value for")

    tfidf_parser = subparsers.add_parser("tfidf", help="Return the TF-IDF score for a doc/term pair")
    tfidf_parser.add_argument("document_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="The term to get TF-IDF value for")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser(
    "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("document_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )
    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("limit", type=int, nargs="?", default=5, help="Search limit")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            inverted_index.load()
            movie_titles = search_movies(args.query)
            print("\n".join(movie_titles))
        case "build":
            inverted_index.build(movies)
            inverted_index.save()
        case "tf":
            inverted_index.load()
            term = tokenize_word(args.term)
            try:
                print(get_tf(args.document_id, term))
            except ValueError:
                print(0)
        case "idf":
            inverted_index.load()
            term = tokenize_word(args.term)
            idf_val = get_idf(term)
            print(f"Inverse document frequency of '{args.term}': {idf_val:.2f}")
        case "tfidf":
            inverted_index.load()
            term = tokenize_word(args.term)
            tf_idf = get_tf_idf(args.document_id, term)
            print(f"TF-IDF score of '{args.term}' in document '{args.document_id}': {tf_idf:.2f}")
        case "bm25idf":
            inverted_index.load()
            bm25idf = get_bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            inverted_index.load()
            bm25tf = get_bm25_tf_command(args.document_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}': {bm25tf:.2f}")
        case "bm25search":
            inverted_index.load()
            bm25 = get_bm25search_command(args.query, args.limit)
            for elem in bm25:
                title = inverted_index.docmap[elem[0]]["title"]
                print(f"({elem[0]}) {title} - Score: {elem[1]:.2f}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()