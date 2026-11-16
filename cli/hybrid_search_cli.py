import argparse
import json
import time

from lib.hybrid_search import HybridSearch
from loader import load_movies
from sentence_transformers import CrossEncoder
from test_llm import LLMConnector

Result = dict
Ranked = list[tuple[float | None, Result]]

ENHANCE_PROMPTS = {
    "spell": """
        Fix any spelling errors in the user-provided movie search query below.
        Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
        Preserve punctuation and capitalization unless a change is required for a typo fix.
        If there are no spelling errors, or if you're unsure, output the original query unchanged.
        Output only the final query text, nothing else.
        User query: "{query}"
    """,
    "rewrite": """
        Rewrite the user-provided movie search query below to be more specific and searchable.
        Consider:
        - Common movie knowledge (famous actors, popular films)
        - Genre conventions (horror = scary, animation = cartoon)
        - Keep the rewritten query concise (under 10 words)
        - It should be a Google-style search query, specific enough to yield relevant results
        - Don't use boolean logic

        Examples:
        - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
        - "movie about bear in london with marmalade" -> "Paddington London marmalade"
        - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

        If you cannot improve the query, output the original unchanged.
        Output only the rewritten query text, nothing else.

        User query: "{query}"
    """,
    "expand": """
        Expand the user-provided movie search query below with related terms.

        Add synonyms and related concepts that might appear in movie descriptions.
        Keep expansions relevant and focused.
        Output only the additional terms; they will be appended to the original query.

        Examples:
        - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
        - "action movie with bear" -> "action thriller bear chase fight adventure"
        - "comedy with bear" -> "comedy funny bear humor lighthearted"

        User query: "{query}"
    """,
}


def llm_text(llm_connector: LLMConnector, prompt: str) -> str:
    response = llm_connector.send_message(prompt)
    return response.choices[0].message.content or ""


def enhance_query(llm_connector: LLMConnector, method: str, query: str) -> str:
    answer = llm_text(llm_connector, ENHANCE_PROMPTS[method].format(query=query))
    if method == "expand":
        return f"{query} {answer}" if answer else query
    return answer


def format_results_for_prompt(results: list[Result]) -> str:
    """One line per result: '<id>: <title> - <description>'."""
    return "\n".join(
        f"{res['document']['id']}: {res['document']['title']} - "
        f"{res['document']['description']}"
        for res in results
    )


def print_result(i: int, res: Result, score_line: str | None = None) -> None:
    print(f"{i}. {res['document']['title']}")
    if score_line:
        print(f"   {score_line}")
    print(f"   RRF Score: {res['rrf']:.3f}")
    print(
        f"   BM25: {res.get('bm25', 0.0):.3f}, Semantic: {res.get('semantic', 0.0):.3f}"
    )
    print(f"   {res['document']['description'][:100]}...")


def print_ranked(ranked: Ranked, label: str | None = None) -> None:
    for i, (score, res) in enumerate(ranked, 1):
        score_line = None
        if label == "rank":
            score_line = f"Re-rank Rank: {i}"
        elif label and score is not None:
            score_line = f"{label} score: {score:.3f}/10"
        print_result(i, res, score_line)


def rerank_individual(
    llm_connector: LLMConnector, query: str, results: list[Result], limit: int
) -> Ranked:
    ranked: Ranked = []
    for res in results:
        time.sleep(3)
        doc = res["document"]
        answer = llm_text(
            llm_connector,
            f"""
            Rate how well this movie matches the search query.
            Query: "{query}"
            Movie: {doc.get("title", "")} - {doc.get("description", "")}

            Consider:
            - Direct relevance to query
            - User intent (what they're looking for)
            - Content appropriateness

            Rate 0-10 (10 = perfect match).
            Output ONLY the number in your response, no other text or explanation.
            """,
        )
        print(answer)
        ranked.append((int(answer or 0), res))
    return sorted(ranked, key=lambda x: x[0] or 0, reverse=True)[:limit]


def rerank_batch(
    llm_connector: LLMConnector, query: str, results: list[Result]
) -> Ranked:
    by_id = {res["document"]["id"]: res for res in results}
    answer = llm_text(
        llm_connector,
        f"""
        Rank the movies listed below by relevance to the following search query.

        Query: "{query}"

        Movies:
        {format_results_for_prompt(results)}

        Return the movie IDs in order of relevance, best match first.

        Your response must be a raw JSON array of integers, the movie IDs.
        Do not wrap the JSON in Markdown. Do not use a ```json code block.
        Do not include any explanatory text.

        For example:
        [75, 12, 34, 2, 1]
        """,
    )
    return [(None, by_id[doc_id]) for doc_id in json.loads(answer)]


def rerank_cross_encoder(
    cross_encoder: CrossEncoder, query: str, results: list[Result], limit: int
) -> Ranked:
    pairs = [
        [
            query,
            f"{res['document'].get('title', '')} - {res['document'].get('description', '')}",
        ]
        for res in results
    ]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
    return ranked[:limit]


def evaluate_results(
    llm_connector: LLMConnector, query: str, results: list[Result]
) -> list[int]:
    answer = llm_text(
        llm_connector,
        f"""
        Rate how relevant each result is to this query on a 0-3 scale:
        Query: "{query}"

        Results:
        {format_results_for_prompt(results)}

        Scale:
        - 3: Highly relevant
        - 2: Relevant
        - 1: Marginally relevant
        - 0: Not relevant

        Do NOT give any numbers other than 0, 1, 2, or 3.

        Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

        [2, 0, 3, 2, 0, 1]
        """,
    )
    return json.loads(answer or "[]")


def print_evaluation(results: list[Result], scores: list[int]) -> None:
    print("\nLLM evaluation:")
    for i, (res, score) in enumerate(zip(results, scores), 1):
        print(f"{i}. {res['document']['title']}: {score}/3")


def run_rrf_search(
    args: argparse.Namespace,
    hs: HybridSearch,
    llm_connector: LLMConnector,
    cross_encoder: CrossEncoder,
) -> None:
    print(f"Original query: {args.query}")
    query = args.query
    if args.enhance:
        query = enhance_query(llm_connector, args.enhance, args.query)
        print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{query}'")

    limit = args.limit * 5 if args.rerank_method else args.limit
    results = hs.rrf_search(query, args.k, limit)
    print(
        f"RRF search result titles: {', '.join(r['document']['title'] for r in results)}"
    )

    if args.rerank_method:
        print(
            f"Re-ranking the top {limit} results using the {args.rerank_method} method..."
        )
        print(f"Reciprocal Rank Fusion Results for '{args.query}' (k={args.k}):")
        match args.rerank_method:
            case "individual":
                ranked = rerank_individual(llm_connector, query, results, args.limit)
                label = "Re-rank"
            case "batch":
                ranked = rerank_batch(llm_connector, query, results)
                label = "rank"
            case _:
                ranked = rerank_cross_encoder(
                    cross_encoder, args.query, results, args.limit
                )
                label = "Cross Encoder"
    else:
        ranked = [(None, res) for res in results]
        label = None

    print_ranked(ranked, label)

    if args.evaluate:
        final_results = [res for _, res in ranked]
        scores = evaluate_results(llm_connector, query, final_results)
        print_evaluation(final_results, scores)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize")
    normalize_parser.add_argument("scores", nargs="*", type=float)

    weighted_search_parser = subparsers.add_parser("weighted-search")
    weighted_search_parser.add_argument("query", type=str)
    weighted_search_parser.add_argument("--alpha", default=0.5, type=float)
    weighted_search_parser.add_argument("--limit", default=5, type=int)

    rrf_search_parser = subparsers.add_parser("rrf-search")
    rrf_search_parser.add_argument("query", type=str)
    rrf_search_parser.add_argument("--k", default=60, type=int)
    rrf_search_parser.add_argument("--limit", default=5, type=int)
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_search_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
    )
    rrf_search_parser.add_argument(
        "--evaluate", action="store_true", help="Use an LLM to evaluate the results"
    )
    return parser


def run_normalize(hs: HybridSearch, scores: list[float]) -> None:
    for score in hs.normalize(scores):
        print(f"* {score:.4f}")


def run_weighted_search(hs: HybridSearch, args: argparse.Namespace) -> None:
    results = hs.weighted_search(args.query, args.alpha, args.limit)
    for i, res in enumerate(results, 1):
        print(f"{i}. {res['document']['title']}")
        print(f"   Hybrid Score: {res['hybrid']:.3f}")
        print(
            f"   BM25: {res.get('bm25', 0.0):.3f}, Semantic: {res.get('semantic', 0.0):.3f}"
        )
        print(f"   {res['document']['description'][:100]}...")


def main() -> None:
    args = build_parser().parse_args()
    if args.command is None:
        build_parser().print_help()
        return

    llm_connector = LLMConnector()
    hs = HybridSearch(load_movies())
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")

    match args.command:
        case "normalize":
            run_normalize(hs, args.scores)
        case "weighted-search":
            run_weighted_search(hs, args)
        case "rrf-search":
            run_rrf_search(args, hs, llm_connector, cross_encoder)


if __name__ == "__main__":
    main()
