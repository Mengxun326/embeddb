/**
 * EmbedDB JavaScript SDK
 *
 * Embedded vector database for Node.js applications.
 * Uses native napi-rs bindings compiled from Rust.
 *
 * Build: cd sdk/javascript && npm run build  (requires @napi-rs/cli)
 *
 * @example
 * const { Database } = require('embeddb');
 * const db = new Database('data.embeddb');
 * db.createCollection('docs', 384, 'cosine');
 * db.insert('docs', 'doc1', new Float32Array(384), '{"title":"hello"}');
 * const results = db.search('docs', new Float32Array(384), 10);
 * db.close();
 */

let native;
try {
  // napi-rs builds to embeddb_js.{platform}.node
  native = require('./embeddb_js.win32-x64-msvc.node');
} catch (e1) {
  try { native = require('./embeddb_js'); }
  catch (e2) {
    native = null;
  }
}

class Database {
  constructor(path) {
    if (!native) {
      throw new Error(
        'EmbedDB native module not found. Build it with:\n' +
        '  cd sdk/javascript && npm install && npm run build'
      );
    }
    this._db = new native.EmbedDb(path);
  }

  createCollection(name, dimension, distance = 'cosine') {
    this._db.createCollection(name, dimension, distance);
  }

  insert(collection, id, vector, metadata) {
    return this._db.insert(collection, id || null, Array.from(vector), metadata || null);
  }

  search(collection, vector, topK = 10, filter = null) {
    return this._db.search(collection, Array.from(vector), topK, filter);
  }

  listCollections() {
    return this._db.listCollections();
  }

  close() {
    if (this._db) {
      this._db.close();
      this._db = null;
    }
  }
}

module.exports = { Database };
