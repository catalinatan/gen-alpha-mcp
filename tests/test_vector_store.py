from gab import vector_store


def _case(case_id, expression):
    return {
        "id": case_id,
        "expression": expression,
        "context": f"{expression} in context",
        "target_audience": "adult",
        "criteria": ["define correctly"],
    }


class _Collection:
    def __init__(self):
        self.upserts = []

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)

    def query(self, **kwargs):
        return {
            "documents": [["doc-1", "doc-2"]],
            "metadatas": [
                [
                    {
                        "dataset": "gen_alpha",
                        "case_id": "ga_001",
                        "expression": "no cap",
                        "context": "context 1",
                        "target_audience": "adult",
                        "criteria_json": '["define correctly"]',
                    },
                    {
                        "dataset": "gen_alpha",
                        "case_id": "ga_002",
                        "expression": "rizz",
                        "context": "context 2",
                        "target_audience": "adult",
                        "criteria_json": '["define charisma"]',
                    },
                ]
            ],
            "distances": [[0.1, 0.2]],
        }


def test_case_document_contains_searchable_fields():
    doc = vector_store.case_document(_case("ga_001", "no cap"))
    assert "Expression: no cap" in doc
    assert "Example usage: no cap in context" in doc
    assert "- define correctly" in doc


def test_seed_cases_upserts_documents(monkeypatch):
    fake_collection = _Collection()
    monkeypatch.setattr(
        vector_store,
        "collection",
        lambda *_args, **_kwargs: fake_collection,
    )
    monkeypatch.setattr(
        vector_store,
        "embed_texts",
        lambda texts, _model_name: [[0.1, 0.2] for _ in texts],
    )

    count = vector_store.seed_cases([_case("ga_001", "no cap")], dataset="gen_alpha")

    assert count == 1
    upsert = fake_collection.upserts[0]
    assert upsert["ids"] == ["gen_alpha:ga_001"]
    assert upsert["metadatas"][0]["case_id"] == "ga_001"
    assert upsert["embeddings"] == [[0.1, 0.2]]


def test_relevant_cases_queries_collection_and_excludes_current(monkeypatch):
    monkeypatch.setattr(
        vector_store,
        "collection",
        lambda *_args, **_kwargs: _Collection(),
    )
    monkeypatch.setattr(
        vector_store,
        "embed_texts",
        lambda texts, _model_name: [[0.3, 0.4] for _ in texts],
    )

    results = vector_store.relevant_cases(
        query="explain rizz",
        dataset="gen_alpha",
        top_k=1,
        exclude_case_id="ga_001",
    )

    assert len(results) == 1
    assert results[0]["id"] == "ga_002"
    assert results[0]["criteria"] == ["define charisma"]
