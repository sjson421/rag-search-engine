from pathlib import Path

import torch
from loader import load_movies
from PIL import Image
from sentence_transformers import SentenceTransformer


class MultimodalSearch:
    def __init__(self, docs: list[dict], model_name="clip-ViT-B-32") -> None:
        self.model = SentenceTransformer(model_name)
        self.docs = docs
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in docs]
        self.text_embeddings = self.model.encode(self.texts, convert_to_tensor=True, show_progress_bar=True)

    def embed_image(self, image_path: Path):
        image = Image.open(image_path)
        return self.model.encode([image], convert_to_tensor=True)[0]

    def search_with_image(self, image_path: Path):
        image_embedding = self.embed_image(image_path)
        similarities = []

        for text in self.text_embeddings:
            similarities.append(torch.cosine_similarity(image_embedding, text, dim=0))
        res = sorted(
            [
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "description": doc["description"],
                    "score": sim,
                }
                for doc, sim in zip(self.docs, similarities)
            ],
            key=lambda x: x["score"],
            reverse=True,
        )[:5]
        return res


def verify_image_embedding(image_path: Path):
    search = MultimodalSearch([])
    embedding = search.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(image_path: Path):
    movies = load_movies()
    search = MultimodalSearch(movies)
    return search.search_with_image(image_path)

