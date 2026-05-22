import pytest
from gab.store import ResultStore


@pytest.fixture
def store(tmp_path):
    return ResultStore(db_path=tmp_path / "test.db")
