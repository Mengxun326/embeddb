/** TypeScript definitions for EmbedDB JavaScript SDK */

export interface DatabaseConfig {
  /** Page size in bytes (default: 4096) */
  pageSize?: number;
  /** Max pages in memory cache (default: 16384) */
  cacheMaxPages?: number;
}

export interface CollectionConfig {
  /** Vector dimension */
  dimension: number;
  /** Distance metric: 'cosine' | 'euclidean' | 'dot' */
  distance?: 'cosine' | 'euclidean' | 'dot';
}

export interface Document {
  /** Optional document ID (auto-generated if omitted) */
  id?: string;
  /** The vector data (Float32Array or number[]) */
  vector: Float32Array | number[];
  /** Optional JSON metadata */
  metadata?: Record<string, any>;
  /** Optional text content */
  text?: string;
}

export interface SearchQuery {
  /** Query vector */
  vector: Float32Array | number[];
  /** Number of results (default: 10) */
  topK?: number;
  /** Metadata filter expression */
  filter?: string;
}

export interface SearchHit {
  /** Document ID */
  id: string;
  /** Similarity score */
  score: number;
  /** Document metadata (if available) */
  metadata?: Record<string, any>;
}

export class Database {
  constructor(path: string, config?: DatabaseConfig);
  createCollection(name: string, config: CollectionConfig): Collection;
  getCollection(name: string): Collection;
  close(): void;
}

export class Collection {
  get name(): string;
  insert(doc: Document): string;
  search(query: SearchQuery): SearchHit[];
}
