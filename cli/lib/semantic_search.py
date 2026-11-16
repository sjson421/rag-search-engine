import heapq
import json
import os
import re

import numpy as np
from loader import load_movies
from sentence_transformers import SentenceTransformer

CACHE_DIR = "./cache"

class SemanticSearch:
    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        if not text:
            raise ValueError("Invalid text for embedding")
        embedding = self.model.encode([text])
        return embedding[0]

    def build_embeddings(self, documents: list[dict]):
        self.documents = documents

        movies = []
        for movie in documents:
            self.document_map[movie["id"]] = movie
            movies.append(f"{movie['title']}: {movie['description']}")
        self.embeddings = self.model.encode(movies, show_progress_bar=True)

        movies_path = os.path.join(CACHE_DIR, "movie_embeddings.npy")
        np.save(movies_path, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]):
        self.documents = documents
        for movie in documents:
            self.document_map[movie["id"]] = movie
        movies_path = os.path.join(CACHE_DIR, "movie_embeddings.npy")
        if os.path.exists(movies_path):
            self.embeddings = np.load(movies_path)
            print(len(self.embeddings))
            print(len(documents))
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)

    def search(self, query, limit) -> list[dict]:
        if self.embeddings is None or self.documents is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embed = self.generate_embedding(query)
        sims = []

        for i, doc_embed in enumerate(self.embeddings):
            heapq.heappush(sims, (cosine_similarity(query_embed, doc_embed), self.documents[i]))
            if len(sims) > limit:
                heapq.heappop(sims)

        res = []
        for i in range(len(sims) - 1, -1, -1):
            sim = sims[i]
            res.append({
                "score": sim[0],
                "title": sim[1]["title"],
                "description": sim[1]["description"],
            })
        return res
        
class ChunkedSemanticSearch(SemanticSearch):
    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for movie in documents:
            self.document_map[movie["id"]] = movie

        chunks = []
        metadatas = []
        for i, doc in enumerate(documents):
            if not doc["description"]:
                continue
            doc_chunks = sem_chunk(doc["description"], 4, 1)
            chunks.extend(doc_chunks)
            
            for j in doc_chunks:
                metadatas.append({
                    "movie_idx": i,
                    "chunk_idx": j,
                    "total_chunks": len(chunks)
                })
        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)
        self.chunk_metadata = metadatas
        
        chunk_embeddings_path = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
        np.save(chunk_embeddings_path, self.chunk_embeddings)
        
        chunk_metadata_path = os.path.join(CACHE_DIR, "chunk_metadata.json")
        with open(chunk_metadata_path, "w") as f:
            json.dump({"chunks": self.chunk_metadata, "total_chunks": len(chunks)}, f, indent=2)
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for movie in documents:
            self.document_map[movie["id"]] = movie

        chunk_embeddings_path = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
        chunk_metadata_path = os.path.join(CACHE_DIR, "chunk_metadata.json")
        
        if os.path.exists(chunk_embeddings_path) and os.path.exists(chunk_metadata_path):
            self.chunk_embeddings = np.load(chunk_embeddings_path)
            with open(chunk_metadata_path) as f:
                self.chunk_metadata = json.load(f)["chunks"]
            return self.chunk_embeddings
        else:
            return self.build_chunk_embeddings(documents)
    
    def search_chunks(self, query: str, limit: int = 10):
        query_embed = self.generate_embedding(query)
        chunk_scores = []

        for i, chunk_embed in enumerate(self.chunk_embeddings):
            sim = cosine_similarity(chunk_embed, query_embed)
            metadata = self.chunk_metadata[i]
            chunk_scores.append({
                    "score": sim,
                    "chunk_idx": metadata["chunk_idx"],
                    "movie_idx": metadata["movie_idx"],
            })
        
        best = {}
        for score in chunk_scores:
            movie_idx = score["movie_idx"]
            if movie_idx not in best or score["score"] > best[movie_idx]:
                best[movie_idx] = score["score"]
        best = sorted(best.items(), key=lambda x: x[1], reverse=True)[:limit]

        res = []
        for movie_idx, score in best:
            doc = self.documents[movie_idx]
            res.append({
                "id": doc["id"],
                "title": doc["title"],
                "document": doc["description"][:100],
                "score": round(score, 3),
            })
        return res

def verify_model():
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")

def embed_text(text):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embedding():
    semantic_search = SemanticSearch()
    movies = load_movies()
    semantic_search.load_or_create_embeddings(movies)

    print(f"Number of docs:   {len(semantic_search.documents)}")
    print(
        f"Embeddings shape: {semantic_search.embeddings.shape[0]} vectors in {semantic_search.embeddings.shape[1]} dimensions"
    )

def embed_query_text(query):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def semantic_search(query, limit: int = 5):
    semantic_search = SemanticSearch()
    movies = load_movies()
    semantic_search.load_or_create_embeddings(movies)
    search_res = semantic_search.search(query, limit)
    for res in search_res:
        print(f"{res["title"]} (score: {res["score"]:.4f})")
        print(res["description"])

def sem_chunk(text:str, max_chunk_size: int, overlap: int = 0):
    text = text.strip()
    if text == "":
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)

    if len(sentences) == 1 and not sentences[0].endswith((".", "!", "?")):
        return sentences

    chunks = []
    step = max_chunk_size - overlap
    i = 0
    while i < len(sentences):
        chunk = [s.strip() for s in sentences[i : i + max_chunk_size]]
        chunk = [s for s in chunk if s]
        if chunk != "":
            chunks.append(" ".join(chunk))
        i += step
    return chunks

