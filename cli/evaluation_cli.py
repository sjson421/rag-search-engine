import argparse

from lib.hybrid_search import HybridSearch
from loader import load_golden_dataset, load_movies


def main() -> None:
    movies = load_movies()
    search = HybridSearch(movies)
    golden_dataset = load_golden_dataset()

    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    for test_case in golden_dataset:
        results = search.rrf_search(test_case["query"], 60, limit)
        titles = [res["document"]["title"] for res in results]
        found = 0
        for title in titles:
            found += title in test_case["relevant_docs"]
        precision = found / len(results)
        recall = found / len(test_case["relevant_docs"])

        f1 = 0 
        if precision + recall != 0:
            f1 = 2 * (precision * recall) / (precision + recall)

        print(f"- Query: {test_case['query']}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - Retrieved: {', '.join(titles)}")
        print(f"  - Relevant: {', '.join(test_case['relevant_docs'])}")
        print()


if __name__ == "__main__":
    main()
