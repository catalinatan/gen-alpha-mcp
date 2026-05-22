import pytest
from server.mcp_server import dataset_path


@pytest.mark.parametrize("bad", ["../evil", "bad name", "bad/slash", "bad!name", ""])
def test_dataset_path_rejects_invalid(bad):
    with pytest.raises(ValueError):
        dataset_path(bad)


@pytest.mark.parametrize("good", ["gen_alpha", "my-set", "v1", "set123"])
def test_dataset_path_accepts_valid_name_format(good, tmp_path, monkeypatch):
    # Point GOLDEN_SETS_DIR at tmp_path so the path resolves safely,
    # then place a dummy file so is_relative_to passes.
    import server.mcp_server as srv
    monkeypatch.setattr(srv, "GOLDEN_SETS_DIR", tmp_path)
    (tmp_path / f"{good}.json").write_text("[]")
    p = dataset_path(good)
    assert p.name == f"{good}.json"
