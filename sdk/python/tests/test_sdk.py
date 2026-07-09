"""Vexra Python SDK integration tests."""
import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import vexra


@pytest.fixture
def db():
    tmp = tempfile.NamedTemporaryFile(suffix=".vexra", delete=False)
    tmp.close()
    path = tmp.name
    db = vexra.Database(path)
    yield db
    db.close()
    try: os.unlink(path)
    except OSError: pass


def test_create_and_get_collection(db):
    col = db.create_collection("test", 3)
    assert col.name == "test"
    assert col.dimension == 3
    assert len(col) == 0
    assert repr(col) == "Collection(name='test', dim=3)"


def test_insert_and_search(db):
    col = db.create_collection("docs", 3)
    col.insert([1.0, 0.0, 0.0], id="a")
    col.insert([0.0, 1.0, 0.0], id="b")
    col.insert([0.0, 0.0, 1.0], id="c")
    assert len(col) == 3

    results = col.search([1.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0]["id"] == "a"
    assert results[0]["score"] < 1.0


def test_search_top_k_larger_than_count(db):
    col = db.create_collection("docs", 3)
    col.insert([1.0, 0.0, 0.0], id="a")
    results = col.search([1.0, 0.0, 0.0], top_k=10)
    assert len(results) == 1  # only 1 doc exists


def test_delete(db):
    col = db.create_collection("docs", 3)
    col.insert([1.0, 0.0, 0.0], id="x")
    assert len(col) == 1
    col.delete("x")
    assert len(col) == 0


def test_list_collections(db):
    db.create_collection("a", 3)
    db.create_collection("b", 5)
    names = db.list_collections()
    assert "a" in names
    assert "b" in names


def test_list_collections_empty(db):
    names = db.list_collections()
    assert names == []


def test_context_manager():
    tmp = tempfile.NamedTemporaryFile(suffix=".vexra", delete=False)
    tmp.close()
    path = tmp.name
    with vexra.Database(path) as db:
        db.create_collection("ctx", 3)
        assert "ctx" in db.list_collections()
    os.unlink(path)


def test_repr(db):
    col = db.create_collection("repr_test", 128)
    r = repr(col)
    assert "repr_test" in r
    assert "128" in r


def test_create_duplicate_fails(db):
    db.create_collection("dup", 3)
    with pytest.raises(RuntimeError):
        db.create_collection("dup", 3)


def test_large_dimension(db):
    col = db.create_collection("large", 1536)
    assert col.dimension == 1536
    v = [0.01] * 1536
    col.insert(v, id="big")
    assert len(col) == 1
    results = col.search(v, top_k=1)
    assert len(results) == 1
    assert results[0]["score"] < 1e-4


def test_many_inserts(db):
    col = db.create_collection("many", 4)
    for i in range(50):
        col.insert([float(i%10)/10, 0.5, 0.1, 0.9], id=f"doc_{i}")
    assert len(col) == 50
    results = col.search([0.5, 0.5, 0.1, 0.9], top_k=5)
    assert len(results) == 5


def test_close_reopen(db):
    """Insert, close DB, reopen, verify data persists."""
    path = db._db_path if hasattr(db, '_db_path') else None
    col = db.create_collection("persist", 3)
    col.insert([1.0, 0.0, 0.0], id="keep")
    col.insert([0.0, 1.0, 0.0], id="also")
    db.close()

    # Reopen
    db2 = vexra.Database(path) if path else db
    col2 = db2.get_collection("persist")
    assert len(col2) == 2
    results = col2.search([1.0, 0.0, 0.0], top_k=2)
    assert results[0]["id"] == "keep"
    if path: db2.close()
