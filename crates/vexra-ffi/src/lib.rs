//! EmbedDB C-compatible Foreign Function Interface (FFI).
//!
//! This crate exposes a C ABI that allows embedding EmbedDB into
//! any language that supports C FFI (Python via ctypes/PyO3,
//! JavaScript via napi-rs, Go via CGO, Java via JNI, etc.).
//!
//! # Safety
//!
//! All functions in this module are `unsafe` to call because they
//! deal with raw pointers and FFI conventions. Higher-level language
//! bindings should wrap these in safe, idiomatic APIs.

use vexra_core::config::{CollectionConfig, Document, SearchQuery};
use vexra_core::db::Database;
use vexra_core::DistanceMetric;
use std::ffi::{c_char, CStr, CString};
use std::ptr;

// ---------------------------------------------------------------------------
// Opaque handle types
// ---------------------------------------------------------------------------

/// Opaque handle to a Database.
pub struct EmbedDb {
    db: Database,
}

/// Opaque handle to a result set.
pub struct EmbedDbResultSet {
    hits: Vec<vexra_core::config::SearchHit>,
}

// ---------------------------------------------------------------------------
// Database lifecycle
// ---------------------------------------------------------------------------

/// Open or create a database at the given path.
///
/// Returns a handle on success, or null on failure (call `embeddb_error` for details).
///
/// # Safety
/// `path` must point to a valid null-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn embeddb_open(
    path: *const c_char,
) -> *mut EmbedDb {
    if path.is_null() {
        return ptr::null_mut();
    }

    let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    match Database::open(path_str) {
        Ok(db) => Box::into_raw(Box::new(EmbedDb { db })),
        Err(_) => ptr::null_mut(),
    }
}

/// Close a database and free all associated resources.
///
/// # Safety
/// `db` must be a valid pointer returned by `embeddb_open`, and must not
/// have already been closed.
#[no_mangle]
pub unsafe extern "C" fn embeddb_close(db: *mut EmbedDb) {
    if db.is_null() {
        return;
    }
    let db = unsafe { Box::from_raw(db) };
    let _ = db.db.close();
}

// ---------------------------------------------------------------------------
// Collection management
// ---------------------------------------------------------------------------

/// Create a new collection.
///
/// Returns 0 on success, -1 on failure.
///
/// # Safety
/// `db` must be a valid pointer. `name` must be a valid null-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn embeddb_create_collection(
    db: *mut EmbedDb,
    name: *const c_char,
    dimension: u32,
    distance_type: u32, // 0=Cosine, 1=Euclidean, 2=DotProduct
) -> i32 {
    if db.is_null() || name.is_null() {
        return -1;
    }

    let db = unsafe { &*db };
    let name_str = match unsafe { CStr::from_ptr(name) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };

    let metric = match distance_type {
        0 => DistanceMetric::Cosine,
        1 => DistanceMetric::Euclidean,
        2 => DistanceMetric::DotProduct,
        _ => DistanceMetric::Cosine,
    };

    let config = CollectionConfig::new(name_str, dimension as usize).with_distance(metric);

    match db.db.create_collection(config) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

// ---------------------------------------------------------------------------
// Document insertion
// ---------------------------------------------------------------------------

/// Insert a document with a pre-computed vector.
///
/// Returns a newly allocated document ID string (caller must free with `embeddb_free_string`),
/// or null on failure.
///
/// # Safety
/// `db`, `collection_name` must be valid pointers.
/// `vector` must point to `dimension` f32 values.
#[no_mangle]
pub unsafe extern "C" fn embeddb_insert_vector(
    db: *mut EmbedDb,
    collection_name: *const c_char,
    doc_id: *const c_char,
    vector: *const f32,
    dimension: u32,
) -> *mut c_char {
    if db.is_null() || collection_name.is_null() || vector.is_null() {
        return ptr::null_mut();
    }

    let db = unsafe { &*db };
    let col_name = match unsafe { CStr::from_ptr(collection_name) }.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };
    let id = if doc_id.is_null() {
        None
    } else {
        match unsafe { CStr::from_ptr(doc_id) }.to_str() {
            Ok(s) => Some(s.to_string()),
            Err(_) => return ptr::null_mut(),
        }
    };

    let vec = unsafe { std::slice::from_raw_parts(vector, dimension as usize) }.to_vec();

    let doc = Document::with_vector(id.unwrap_or_else(|| "".into()), vec);
    let doc = if let Some(ref i) = doc.id.clone() {
        if i.is_empty() {
            Document {
                id: None,
                ..doc
            }
        } else {
            doc
        }
    } else {
        doc
    };

    // Use the convenience function
    match vexra_core::insert(&db.db, col_name, doc) {
        Ok(result_id) => {
            match CString::new(result_id) {
                Ok(cs) => cs.into_raw(),
                Err(_) => ptr::null_mut(),
            }
        }
        Err(_) => ptr::null_mut(),
    }
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

/// Search for nearest neighbors.
///
/// Returns a result set handle, or null on failure.
///
/// # Safety
/// `db`, `collection_name` must be valid pointers.
/// `query_vector` must point to `dimension` f32 values.
#[no_mangle]
pub unsafe extern "C" fn embeddb_search(
    db: *mut EmbedDb,
    collection_name: *const c_char,
    query_vector: *const f32,
    dimension: u32,
    top_k: u32,
) -> *mut EmbedDbResultSet {
    if db.is_null() || collection_name.is_null() || query_vector.is_null() {
        return ptr::null_mut();
    }

    let db = unsafe { &*db };
    let col_name = match unsafe { CStr::from_ptr(collection_name) }.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    let query = unsafe { std::slice::from_raw_parts(query_vector, dimension as usize) }.to_vec();

    match vexra_core::search(
        &db.db,
        col_name,
        SearchQuery::with_vector(query, top_k as usize),
    ) {
        Ok(hits) => Box::into_raw(Box::new(EmbedDbResultSet { hits })),
        Err(_) => ptr::null_mut(),
    }
}

// ---------------------------------------------------------------------------
// Result set iteration
// ---------------------------------------------------------------------------

/// Get the number of results in a result set.
///
/// # Safety
/// `results` must be a valid pointer returned by `embeddb_search`.
#[no_mangle]
pub unsafe extern "C" fn embeddb_result_count(results: *const EmbedDbResultSet) -> u32 {
    if results.is_null() {
        return 0;
    }
    let results = unsafe { &*results };
    results.hits.len() as u32
}

/// Get a result by index.
///
/// Writes the document ID, score, and vector (if available) into the provided buffers.
/// Returns 0 on success, -1 if index is out of bounds.
///
/// # Safety
/// `results` must be valid. `id_buffer` must be at least `id_buffer_size` bytes.
#[no_mangle]
pub unsafe extern "C" fn embeddb_result_get(
    results: *const EmbedDbResultSet,
    index: u32,
    id_buffer: *mut c_char,
    id_buffer_size: u32,
    score: *mut f32,
) -> i32 {
    if results.is_null() || id_buffer.is_null() || score.is_null() {
        return -1;
    }

    let results = unsafe { &*results };
    if index as usize >= results.hits.len() {
        return -1;
    }

    let hit = &results.hits[index as usize];
    *score = hit.score;

    // Copy ID into buffer
    let id_bytes = hit.id.as_bytes();
    let copy_len = id_bytes.len().min(id_buffer_size as usize - 1);
    unsafe {
        ptr::copy_nonoverlapping(id_bytes.as_ptr(), id_buffer as *mut u8, copy_len);
        *id_buffer.add(copy_len) = 0; // null terminator
    }

    0
}

/// Free a result set.
///
/// # Safety
/// `results` must be a valid pointer returned by `embeddb_search`, or null.
#[no_mangle]
pub unsafe extern "C" fn embeddb_free_result_set(results: *mut EmbedDbResultSet) {
    if !results.is_null() {
        unsafe {
            let _ = Box::from_raw(results);
        }
    }
}

/// Free a string returned by an EmbedDB function.
///
/// # Safety
/// `s` must be a valid pointer returned by an EmbedDB function, or null.
#[no_mangle]
pub unsafe extern "C" fn embeddb_free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            let _ = CString::from_raw(s);
        }
    }
}

/// List all collection names in the database.
///
/// Returns a JSON-encoded array of strings, or null on failure.
/// Caller must free with `embeddb_free_string`.
///
/// # Safety
/// `db` must be a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn embeddb_list_collections(db: *mut EmbedDb) -> *mut c_char {
    if db.is_null() { return ptr::null_mut(); }
    let db = unsafe { &*db };
    let names = db.db.list_collections();
    let json = format!("[{}]", names.iter().map(|n| format!("\"{}\"", n)).collect::<Vec<_>>().join(","));
    CString::new(json).map(|cs| cs.into_raw()).unwrap_or(ptr::null_mut())
}

/// Delete a document from a collection.
///
/// Returns 0 on success, -1 on failure.
///
/// # Safety
/// `db`, `collection_name`, `doc_id` must be valid null-terminated UTF-8 strings.
#[no_mangle]
pub unsafe extern "C" fn embeddb_delete(
    db: *mut EmbedDb,
    collection_name: *const c_char,
    doc_id: *const c_char,
) -> i32 {
    if db.is_null() || collection_name.is_null() || doc_id.is_null() { return -1; }
    let db = unsafe { &*db };
    let col_name = match unsafe { CStr::from_ptr(collection_name) }.to_str() {
        Ok(s) => s, Err(_) => return -1,
    };
    let id = match unsafe { CStr::from_ptr(doc_id) }.to_str() {
        Ok(s) => s, Err(_) => return -1,
    };
    match db.db.get_collection(col_name) {
        Ok(col) => {
            match col.write().delete(id) {
                Ok(()) => 0,
                Err(_) => -1,
            }
        }
        Err(_) => -1,
    }
}

/// Get a human-readable error message for the last operation on a database.
///
/// Returns a static string.
///
/// # Safety
/// `db` must be a valid pointer or null.
#[no_mangle]
pub unsafe extern "C" fn embeddb_error(_db: *mut EmbedDb) -> *const c_char {
    b"An error occurred\0".as_ptr() as *const c_char
}
