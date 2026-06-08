import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gab.runner import resolve_input_path
from gab.vector_store import (
    DEFAULT_CHROMA_DIR,
    DEFAULT_COLLECTION,
    DEFAULT_EMBEDDING_MODEL,
    seed_cases,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed golden cases and upsert them into local Chroma."
    )
    parser.add_argument(
        "--golden",
        default="golden_sets/gen_alpha.json",
        help="Path to the golden set JSON file.",
    )
    parser.add_argument(
        "--dataset",
        default="gen_alpha",
        help="Dataset name stored in Chroma metadata.",
    )
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=DEFAULT_CHROMA_DIR,
        help="Persistent Chroma directory.",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help="Chroma collection name.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="SentenceTransformer model name.",
    )
    args = parser.parse_args()

    cases = json.loads(resolve_input_path(args.golden).read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("Golden set must be a JSON array of test cases.")

    count = seed_cases(
        cases=cases,
        dataset=args.dataset,
        chroma_dir=args.chroma_dir,
        collection_name=args.collection,
        embedding_model_name=args.embedding_model,
    )
    print(
        f"Seeded {count} cases into Chroma collection "
        f"{args.collection!r} at {args.chroma_dir}"
    )


if __name__ == "__main__":
    main()
