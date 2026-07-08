"""EmbedDB Python client — wraps the native Rust library via FFI."""

import json
import os
import ctypes
from ctypes import c_char_p, c_float, c_int32, c_uint32, c_void_p, POINTER, byref, cast
from typing import List, Optional, Dict, Any
from .config import CollectionConfig, DistanceMetric, SearchResult

# ---------------------------------------------------------------------------
# Find and load the native library
# ---------------------------------------------------------------------------

def _find_library() -> str:
    """Locate the embeddb native library."""
    # Check environment variable
    env_path = os.environ.get("EMBEDDB_LIB_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    # Search common locations
    search_paths = [
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "target", "release"),
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "target", "debug"),
        os.path.join(os.path.dirname(__file__)),
    ]

    lib_name = "embeddb_ffi"
    if os.name == "nt":
        lib_name += ".dll"
    elif os.name == "posix" and os.uname().sysname == "Darwin":
        lib_name = "lib" + lib_name + ".dylib"
    else:
        lib_name = "lib" + lib_name + ".so"

    for path in search_paths:
        full = os.path.join(path, lib_name)
        if os.path.exists(full):
            return full

    # Fall back to system search
    return lib_name


_lib = None

def _get_lib():
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(_find_library())

        # Define function signatures
        _lib.embeddb_open.argtypes = [c_char_p]
        _lib.embeddb_open.restype = c_void_p

        _lib.embeddb_close.argtypes = [c_void_p]
        _lib.embeddb_close.restype = None

        _lib.embeddb_create_collection.argtypes = [c_void_p, c_char_p, c_uint32, c_uint32]
        _lib.embeddb_create_collection.restype = c_int32

        _lib.embeddb_insert_vector.argtypes = [c_void_p, c_char_p, c_char_p, POINTER(c_float), c_uint32]
        _lib.embeddb_insert_vector.restype = c_char_p

        _lib.embeddb_search.argtypes = [c_void_p, c_char_p, POINTER(c_float), c_uint32, c_uint32]
        _lib.embeddb_search.restype = c_void_p

        _lib.embeddb_result_count.argtypes = [c_void_p]
        _lib.embeddb_result_count.restype = c_uint32

        _lib.embeddb_result_get.argtypes = [c_void_p, c_uint32, c_char_p, c_uint32, POINTER(c_float)]
        _lib.embeddb_result_get.restype = c_int32

        _lib.embeddb_free_result_set.argtypes = [c_void_p]
        _lib.embeddb_free_result_set.restype = None

        _lib.embeddb_free_string.argtypes = [c_char_p]
        _lib.embeddb_free_string.restype = None

    return _lib


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

class Database:
    """An EmbedDB database handle."""

    def __init__(self, path: str):
        """Open or create a database at the given path."""
        lib = _get_lib()
        self._path = path
        self._handle = lib.embeddb_open(path.encode("utf-8"))
        if not self._handle:
            raise RuntimeError(f"Failed to open database at {path}")

    def create_collection(
        self,
        name: str,
        dimension: int,
        distance: DistanceMetric = DistanceMetric.COSINE,
    ) -> "Collection":
        """Create a new collection."""
        lib = _get_lib()
        dist_map = {DistanceMetric.COSINE: 0, DistanceMetric.EUCLIDEAN: 1, DistanceMetric.DOT_PRODUCT: 2}
        result = lib.embeddb_create_collection(
            self._handle,
            name.encode("utf-8"),
            dimension,
            dist_map[distance],
        )
        if result != 0:
            raise RuntimeError(f"Failed to create collection '{name}'")
        return Collection(self, name, dimension, distance)

    def get_collection(self, name: str) -> "Collection":
        """Get an existing collection by name."""
        return Collection(self, name)

    def close(self):
        """Close the database."""
        if self._handle:
            _get_lib().embeddb_close(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()

    @property
    def handle(self):
        return self._handle


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

class Collection:
    """A collection of vectors."""

    def __init__(self, db: Database, name: str, dimension: int = 0, distance: DistanceMetric = DistanceMetric.COSINE):
        self._db = db
        self._name = name
        self._dimension = dimension
        self._distance = distance

    @property
    def name(self) -> str:
        return self._name

    def insert(self, doc: Dict[str, Any]) -> str:
        """Insert a document. Must have 'vector' key with list of floats."""
        lib = _get_lib()
        vector = doc["vector"]
        doc_id = doc.get("id")

        arr_type = c_float * len(vector)
        c_vector = arr_type(*vector)

        c_id = lib.embeddb_insert_vector(
            self._db.handle,
            self._name.encode("utf-8"),
            doc_id.encode("utf-8") if doc_id else None,
            c_vector,
            len(vector),
        )

        if not c_id:
            raise RuntimeError(f"Failed to insert document into '{self._name}'")

        result = ctypes.string_at(c_id).decode("utf-8")
        lib.embeddb_free_string(c_id)
        return result

    def search(
        self,
        vector: List[float],
        top_k: int = 10,
        filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """Search for nearest neighbors."""
        lib = _get_lib()

        arr_type = c_float * len(vector)
        c_vector = arr_type(*vector)

        results_handle = lib.embeddb_search(
            self._db.handle,
            self._name.encode("utf-8"),
            c_vector,
            len(vector),
            top_k,
        )

        if not results_handle:
            raise RuntimeError(f"Search failed on collection '{self._name}'")

        count = lib.embeddb_result_count(results_handle)
        results = []

        id_buf = ctypes.create_string_buffer(256)
        score = c_float()

        for i in range(count):
            ret = lib.embeddb_result_get(results_handle, i, id_buf, 256, byref(score))
            if ret == 0:
                results.append(SearchResult(
                    id=id_buf.value.decode("utf-8"),
                    score=score.value,
                ))

        lib.embeddb_free_result_set(results_handle)

        # Apply metadata filter in Python if needed
        if filter:
            # Simple filter support — server-side filtering preferred
            pass

        return results

    def __repr__(self):
        return f"Collection(name='{self._name}')"
