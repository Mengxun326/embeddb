//! BM25 sparse retrieval via Tantivy.
//!
//! Provides full-text indexing and search for document text/metadata.

use std::path::Path;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::*;
use tantivy::{doc, Index, IndexReader, IndexWriter, ReloadPolicy};
use tantivy::tokenizer::*;

/// A single BM25 search result.
#[derive(Debug, Clone)]
pub struct Bm25Hit {
    pub doc_id: String,
    pub score: f32,
}

/// BM25 full-text search index backed by Tantivy.
pub struct Bm25Index {
    index: Index,
    #[allow(dead_code)]
    schema: Schema,
    reader: IndexReader,
    writer: IndexWriter,
    id_field: Field,
    text_field: Field,
}

impl Bm25Index {
    /// Create or open a BM25 index at the given directory path.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let mut schema_builder = Schema::builder();
        let id_field = schema_builder.add_text_field("doc_id", STRING | STORED);
        let text_field = schema_builder.add_text_field("text", TEXT | STORED);
        let schema = schema_builder.build();

        let index_path = path.as_ref().join("bm25");
        std::fs::create_dir_all(&index_path)?;

        let index = if index_path.join("meta.json").exists() {
            Index::open_in_dir(&index_path)?
        } else {
            let index = Index::create_in_dir(&index_path, schema.clone())?;
            // Register English tokenizer
            index.tokenizers().register("default", TextAnalyzer::builder(
                SimpleTokenizer::default()
            ).filter(LowerCaser).build());
            index
        };

        let reader = index.reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()?;
        let writer = index.writer(50_000_000)?;

        Ok(Self { index, schema, reader, writer, id_field, text_field })
    }

    /// Add a document to the BM25 index.
    pub fn add_document(&self, doc_id: &str, text: &str) -> Result<u64, Box<dyn std::error::Error>> {
        let opstamp = self.writer.add_document(doc!(
            self.id_field => doc_id,
            self.text_field => text,
        ))?;
        Ok(opstamp)
    }

    /// Remove a document from the BM25 index.
    pub fn remove_document(&self, doc_id: &str) -> Result<(), Box<dyn std::error::Error>> {
        let term = tantivy::Term::from_field_text(self.id_field, doc_id);
        self.writer.delete_term(term);
        Ok(())
    }

    /// Commit pending writes.
    pub fn commit(&mut self) -> Result<u64, Box<dyn std::error::Error>> {
        Ok(self.writer.commit()?)
    }

    /// Search for documents matching the query text.
    pub fn search(&self, query_text: &str, top_k: usize) -> Result<Vec<Bm25Hit>, Box<dyn std::error::Error>> {
        self.reader.reload()?;
        let searcher = self.reader.searcher();

        let query_parser = QueryParser::for_index(&self.index, vec![self.text_field]);
        let query = query_parser.parse_query(query_text)?;

        let top_docs = searcher.search(&query, &TopDocs::with_limit(top_k))?;

        let mut hits = Vec::new();
        for (score, doc_address) in top_docs {
            let doc: tantivy::TantivyDocument = searcher.doc(doc_address)?;
            let doc_id = doc.get_first(self.id_field)
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();
            hits.push(Bm25Hit { doc_id, score });
        }
        Ok(hits)
    }
}
