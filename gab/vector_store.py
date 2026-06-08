import json
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CHROMA_DIR = ROOT_DIR / ".chroma"
DEFAULT_COLLECTION = "gen_alpha_golden_cases"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def case_document(case: dict) -> str:
    criteria = "\n".join(f"- {criterion}" for criterion in case.get("criteria", []))
    return (
        f"Expression: {case.get('expression', '')}\n"
        f"Example usage: {case.get('context', '')}\n"
        f"Target audience: {case.get('target_audience', '')}\n"
        f"Criteria:\n{criteria}"
    )


def _metadata(case: dict, dataset: str) -> dict:
    return {
        "dataset": dataset,
        "case_id": case["id"],
        "expression": case.get("expression", ""),
        "context": case.get("context", ""),
        "target_audience": case.get("target_audience", ""),
        "criteria_json": json.dumps(case.get("criteria", [])),
    }


def _case_from_metadata(
    metadata: dict,
    document: str | None,
    distance: float | None,
) -> dict:
    return {
        "id": metadata["case_id"],
        "expression": metadata.get("expression", ""),
        "context": metadata.get("context", ""),
        "target_audience": metadata.get("target_audience", ""),
        "criteria": json.loads(metadata.get("criteria_json", "[]")),
        "document": document or "",
        "distance": distance,
    }


@lru_cache(maxsize=4)
def embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_texts(
    texts: list[str],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> list[list[float]]:
    vectors = embedding_model(model_name).encode(texts, normalize_embeddings=True)
    return [vector.tolist() for vector in vectors]


def chroma_client(chroma_dir: Path = DEFAULT_CHROMA_DIR) -> Any:
    import chromadb

    return chromadb.PersistentClient(path=str(chroma_dir))


def collection(
    chroma_dir: Path = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION,
) -> Any:
    return chroma_client(chroma_dir).get_or_create_collection(collection_name)


def seed_cases(
    cases: list[dict],
    dataset: str,
    chroma_dir: Path = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> int:
    docs = [case_document(case) for case in cases]
    ids = [f"{dataset}:{case['id']}" for case in cases]
    metadatas = [_metadata(case, dataset) for case in cases]
    embeddings = embed_texts(docs, embedding_model_name)
    collection(chroma_dir, collection_name).upsert(
        ids=ids,
        documents=docs,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    return len(cases)


def relevant_cases(
    query: str,
    dataset: str = "gen_alpha",
    top_k: int = 3,
    exclude_case_id: str | None = None,
    chroma_dir: Path = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> list[dict]:
    if top_k <= 0:
        return []

    n_results = top_k + 1 if exclude_case_id else top_k
    results = collection(chroma_dir, collection_name).query(
        query_embeddings=embed_texts([query], embedding_model_name),
        n_results=n_results,
        where={"dataset": dataset},
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    matches = []
    for metadata, document, distance in zip(
        metadatas,
        documents,
        distances,
        strict=False,
    ):
        if exclude_case_id and metadata.get("case_id") == exclude_case_id:
            continue
        matches.append(_case_from_metadata(metadata, document, distance))
        if len(matches) == top_k:
            break
    return matches
