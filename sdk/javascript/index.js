/**
 * EmbedDB JavaScript SDK
 *
 * Embedded vector database for Node.js applications.
 * Wraps the native Rust library via napi-rs bindings.
 *
 * @example
 * const embeddb = require('embeddb');
 * const db = new embeddb.Database('data.embeddb');
 * const col = db.createCollection('docs', { dimension: 384 });
 * col.insert({ id: 'doc1', vector: new Float32Array(384) });
 * const results = col.search({ vector: new Float32Array(384), topK: 10 });
 */

// Try to load native bindings, fall back to a helpful error
let native;
try {
  native = require('./native');
} catch (e) {
  native = null;
}

class Database {
  constructor(path, config = {}) {
    if (!native) {
      throw new Error(
        'EmbedDB native module not found. ' +
        'Build it with: cd sdk/javascript && npm run build-native'
      );
    }
    this._handle = native.open(path, config);
  }

  createCollection(name, config = {}) {
    const { dimension = 384, distance = 'cosine' } = config;
    native.createCollection(this._handle, name, dimension, distance);
    return new Collection(this._handle, name);
  }

  getCollection(name) {
    return new Collection(this._handle, name);
  }

  close() {
    if (this._handle) {
      native.close(this._handle);
      this._handle = null;
    }
  }
}

class Collection {
  constructor(dbHandle, name) {
    this._db = dbHandle;
    this._name = name;
  }

  get name() {
    return this._name;
  }

  insert(doc) {
    const { id, vector, metadata, text } = doc;
    return native.insertVector(this._db, this._name, id || null, vector, metadata || null);
  }

  search(query) {
    const { vector, topK = 10, filter = null } = query;
    return native.search(this._db, this._name, vector, topK, filter);
  }
}

module.exports = { Database, Collection };
