// Package vexra provides Go bindings for the Vexra embedded vector database.
//
// Requires the Vexra C shared library (libvexra_ffi) to be installed.
// Build it with: cargo build --release -p vexra-ffi
package vexra

// #cgo LDFLAGS: -L../../target/release -lvexra_ffi
// #cgo windows LDFLAGS: -L../../target/release -lvexra_ffi
// #include <stdlib.h>
//
// extern void* vexra_open(const char* path);
// extern void vexra_close(void* db);
// extern int vexra_create_collection(void* db, const char* name, unsigned int dim, unsigned int distance);
// extern char* vexra_insert_vector(void* db, const char* col, const char* id, const float* vec, unsigned int dim);
// extern void* vexra_search(void* db, const char* col, const float* query, unsigned int dim, unsigned int k);
// extern unsigned int vexra_result_count(void* results);
// extern int vexra_result_get(void* results, unsigned int index, char* idBuf, unsigned int bufSize, float* score);
// extern void vexra_free_result_set(void* results);
// extern void vexra_free_string(char* s);
// extern char* vexra_list_collections(void* db);
// extern int vexra_delete(void* db, const char* col, const char* id);
import "C"
import (
	"fmt"
	"unsafe"
)

// Database wraps a Vexra database connection.
type Database struct {
	handle unsafe.Pointer
}

// Open opens or creates a database at the given path.
func Open(path string) (*Database, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	h := C.vexra_open(cPath)
	if h == nil {
		return nil, fmt.Errorf("failed to open database at %s", path)
	}
	return &Database{handle: h}, nil
}

// Close closes the database.
func (db *Database) Close() {
	if db.handle != nil {
		C.vexra_close(db.handle)
		db.handle = nil
	}
}

// CreateCollection creates a new collection.
func (db *Database) CreateCollection(name string, dim int) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	if C.vexra_create_collection(db.handle, cName, C.uint(dim), 0) != 0 {
		return fmt.Errorf("failed to create collection %s", name)
	}
	return nil
}

// Insert adds a document with a vector to a collection.
func (db *Database) Insert(collection, id string, vector []float32) (string, error) {
	cCol := C.CString(collection)
	defer C.free(unsafe.Pointer(cCol))
	var cId *C.char
	if id != "" {
		cId = C.CString(id)
		defer C.free(unsafe.Pointer(cId))
	}
	result := C.vexra_insert_vector(db.handle, cCol, cId, (*C.float)(&vector[0]), C.uint(len(vector)))
	if result == nil {
		return "", fmt.Errorf("insert failed")
	}
	defer C.vexra_free_string(result)
	return C.GoString(result), nil
}

// SearchResult is a single search hit.
type SearchResult struct {
	ID    string
	Score float32
}

// Search performs a vector similarity search.
func (db *Database) Search(collection string, query []float32, k int) ([]SearchResult, error) {
	cCol := C.CString(collection)
	defer C.free(unsafe.Pointer(cCol))
	results := C.vexra_search(db.handle, cCol, (*C.float)(&query[0]), C.uint(len(query)), C.uint(k))
	if results == nil {
		return nil, fmt.Errorf("search failed")
	}
	defer C.vexra_free_result_set(results)

	count := int(C.vexra_result_count(results))
	out := make([]SearchResult, count)
	idBuf := make([]byte, 256)
	for i := 0; i < count; i++ {
		var score C.float
		if C.vexra_result_get(results, C.uint(i), (*C.char)(&idBuf[0]), 256, &score) == 0 {
			out[i] = SearchResult{ID: string(idBuf[:clen(idBuf)]), Score: float32(score)}
		}
	}
	return out, nil
}

func clen(b []byte) int {
	for i, v := range b {
		if v == 0 {
			return i
		}
	}
	return len(b)
}
