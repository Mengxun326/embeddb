package com.vexra;

/**
 * Java bindings for the Vexra embedded vector database.
 *
 * Requires the native library (libvexra_ffi) to be on java.library.path.
 * Build it with: cargo build --release -p vexra-ffi
 */
public class Vexra implements AutoCloseable {
    private long handle;

    static { System.loadLibrary("vexra_ffi"); }

    private native long nativeOpen(String path);
    private native void nativeClose(long handle);
    private native int nativeCreateCollection(long handle, String name, int dim, int distance);
    private native String nativeInsert(long handle, String col, String id, float[] vector);
    private native long nativeSearch(long handle, String col, float[] query, int k);
    private native int nativeResultCount(long results);
    private native String nativeResultGet(long results, int index, float[] scoreOut);
    private native void nativeFreeResultSet(long results);

    public Vexra(String path) {
        handle = nativeOpen(path);
        if (handle == 0) throw new RuntimeException("Failed to open database: " + path);
    }

    public void createCollection(String name, int dim) {
        if (nativeCreateCollection(handle, name, dim, 0) != 0)
            throw new RuntimeException("Failed to create collection: " + name);
    }

    public String insert(String collection, String id, float[] vector) {
        String result = nativeInsert(handle, collection, id, vector);
        if (result == null) throw new RuntimeException("Insert failed");
        return result;
    }

    public SearchResult[] search(String collection, float[] query, int k) {
        long results = nativeSearch(handle, collection, query, k);
        if (results == 0) return new SearchResult[0];
        int count = nativeResultCount(results);
        SearchResult[] out = new SearchResult[count];
        for (int i = 0; i < count; i++) {
            float[] scoreHolder = new float[1];
            String id = nativeResultGet(results, i, scoreHolder);
            out[i] = new SearchResult(id, scoreHolder[0]);
        }
        nativeFreeResultSet(results);
        return out;
    }

    @Override public void close() { if (handle != 0) { nativeClose(handle); handle = 0; } }

    public static class SearchResult {
        public final String id;
        public final float score;
        public SearchResult(String id, float score) { this.id = id; this.score = score; }
    }
}
