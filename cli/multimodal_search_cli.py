import argparse
from pathlib import Path

import lib.multimodal_search


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verifier = subparsers.add_parser(
        "verify_image_embedding", help="Verify image embedding"
    )
    verifier.add_argument("image_path", type=Path, help="Path of the image")

    searcher = subparsers.add_parser("image_search", help="Search via image")
    searcher.add_argument("image_path", type=Path, help="Path of the image")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            lib.multimodal_search.verify_image_embedding(args.image_path)
        case "image_search":
            results = lib.multimodal_search.image_search_command(args.image_path)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']} (similarity: {res['score']:.3f})")
                print(f"  {res['description'][:100]}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
