import argparse

import lib.semantic_search as search
from lib.semantic_search import ChunkedSemanticSearch

from loader import load_movies

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify embedding model")

    embed_parser = subparsers.add_parser("embed_text", help="Embed a provided text")
    embed_parser.add_argument("text", type=str, help="The text to embed")

    subparsers.add_parser("verify_embeddings", help="Verify the embeddings of the movies")

    query_embed_parser = subparsers.add_parser("embed_query", help="Embed a provided query")
    query_embed_parser.add_argument("query", type=str, help="The query to embed")

    search_parser = subparsers.add_parser("search", help="Perform semantic search")
    search_parser.add_argument("query", type=str, help="The query to search for")
    search_parser.add_argument("--limit", type=int, default=5, help="The limit of query results")

    chunk_parser = subparsers.add_parser("chunk")
    chunk_parser.add_argument("text", type=str)
    chunk_parser.add_argument("--chunk-size", type=int, default=200)
    chunk_parser.add_argument("--overlap", type=int, default=0)
    
    sem_chunk_parser = subparsers.add_parser("semantic_chunk")
    sem_chunk_parser.add_argument("text", type=str)
    sem_chunk_parser.add_argument("--max-chunk-size", type=int, default=4)
    sem_chunk_parser.add_argument("--overlap", type=int, default=0)
    
    subparsers.add_parser("embed_chunks")
     
    search_chunked_parser = subparsers.add_parser("search_chunked")
    search_chunked_parser.add_argument("query", type=str)
    search_chunked_parser.add_argument("--limit", type=int, default=5)

    args = parser.parse_args()
    
    match args.command:
        case "verify":
            search.verify_model()
        case "embed_text":
            search.embed_text(args.text)
        case "verify_embeddings":
            search.verify_embedding()
        case "embed_query":
            search.embed_query_text(args.query)
        case "search":
            search.semantic_search(args.query, args.limit)
        case "chunk":
            text = args.text.split()
            chunks = []
            i = 0
            while i < len(text):
                j = i - args.overlap
                if j < 0:
                    j = 0
                chunk = text[j : i + args.chunk_size]
                chunks.append(" ".join(chunk))
                i += args.chunk_size
            
            print(f"Chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks):
                print(f"{i + 1}. {chunk}")
        case "semantic_chunk":
            chunks = search.sem_chunk(args.text, args.max_chunk_size, args.overlap)                      
            print(f"Semantically chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks):
                print(f"{i + 1}. {chunk}")
        case "embed_chunks":
            movies = load_movies()
            chunked_sem_search = ChunkedSemanticSearch()
            embeds = chunked_sem_search.load_or_create_chunk_embeddings(movies)
            print(f"Generated {len(embeds)} chunked embeddings")
        case "search_chunked":
            movies = load_movies()
            chunked_sem_search = ChunkedSemanticSearch()
            embeds = chunked_sem_search.load_or_create_chunk_embeddings(movies)
            search_res = chunked_sem_search.search_chunks(args.query, args.limit)
            for res in search_res:
                print(f"\n{res['id']}. {res['title']} (score: {res['score']:.4f})")
                print(f"   {res['document']}...")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
