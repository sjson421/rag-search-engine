import argparse

from lib.hybrid_search import HybridSearch
from loader import load_movies
from test_llm import LLMConnector


def llm_text(llm_connector: LLMConnector, prompt: str) -> str:
    response = llm_connector.send_message(prompt)
    return response.choices[0].message.content or ""

def print_search_results(results: list[dict]):
    print("Search Results:")
    for result in results:
        print(f"- {result['document']['title']}")
    print()

def rag(search: HybridSearch, llm_connector: LLMConnector, query: str):
    docs = search.rrf_search(query, 60, 5)

    prompt = f"""
        You are a RAG agent for Webflyx, a movie streaming service.
        Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
        Provide a comprehensive answer that addresses the user's query.

        Query: {query}

        Documents:
        {docs}

        Answer:
    """

    response = llm_text(llm_connector, prompt)
    print_search_results(docs)

    print("RAG Response:")
    print(response)

def summarize(search: HybridSearch, llm_connector: LLMConnector, query: str, limit: int):
    docs = search.rrf_search(query, 60, limit)

    prompt = f"""
        Provide information useful to the query below by synthesizing data from multiple search results in detail.

        The goal is to provide comprehensive information so that users know what their options are.
        Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

        This should be tailored to Webflyx users. Webflyx is a movie streaming service.

        Query: {query}

        Search results:
        {docs}

        Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
    """
    response = llm_text(llm_connector, prompt)
    print_search_results(docs)

    print("LLM Summary:")
    print(response)

def cite(search: HybridSearch, llm_connector: LLMConnector, query: str, limit: int):
    docs = search.rrf_search(query, 60, limit)
    prompt = f"""
        Answer the query below and give information based on the provided documents.

        The answer should be tailored to users of Webflyx, a movie streaming service.
        If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

        Query: {query}

        Documents:
        {docs}

        Instructions:
        - Provide a comprehensive answer that addresses the query
        - Cite sources in the format [1], [2], etc. when referencing information
        - If sources disagree, mention the different viewpoints
        - If the answer isn't in the provided documents, say "I don't have enough information"
        - Be direct and informative

        Answer:
    """

    response = llm_text(llm_connector, prompt)
    print_search_results(docs)
    print("LLM Answer:")
    print(response)

def question(search: HybridSearch, llm_connector: LLMConnector, question: str, limit: int):
    docs = search.rrf_search(question, 60, limit)
    prompt = f"""
        Answer the user's question based on the provided movies that are available on Webflyx, a streaming service.

        Question: {question}

        Documents:
        {docs}

        Instructions:
        - Answer questions directly and concisely
        - Be casual and conversational
        - Don't be cringe or hype-y
        - Talk like a normal person would in a chat conversation

        Answer:
        """
    response = llm_text(llm_connector, prompt)
    print_search_results(docs)
    print("Answer:")
    print(response)

def main() -> None:
    llm_connector = LLMConnector()
    movies = load_movies()
    search = HybridSearch(movies)
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarizer = subparsers.add_parser(
        "summarize", help="Summarize the search results using RRF search"
    )
    summarizer.add_argument("query", type=str, help="Search query for RRF search")
    summarizer.add_argument("--limit", type=int, default=5, help="Limit of RRF search")

    citer = subparsers.add_parser(
        "citations", help="Answer the provided query with citations"
    )
    citer.add_argument("query", type=str, help="Search query for RRF search")
    citer.add_argument("--limit", type=int, default=5, help="Limit of RRF search")

    questioner = subparsers.add_parser(
        "question", help="Ask a question"
    )
    questioner.add_argument("question", type=str, help="Question to ask")
    questioner.add_argument("--limit", type=int, default=5, help="Limit of RRF search")

    args = parser.parse_args()

    match args.command:
        case "rag":
            rag(search, llm_connector, args.query)
        case "summarize":
            summarize(search, llm_connector, args.query, args.limit)
        case "citations":
            cite(search, llm_connector, args.query, args.limit)
        case "question":
            question(search, llm_connector, args.question, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
