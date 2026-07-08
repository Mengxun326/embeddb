//! EmbedDB CLI — Command-line interface for managing EmbedDB databases.

use clap::{Parser, Subcommand};
use embeddb_core::config::{CollectionConfig, Document, SearchQuery};
use embeddb_core::db::Database;
use embeddb_embedding::{Embedder, SimpleEmbedder};
use std::path::PathBuf;

#[tokio::main]
async fn main() {

/// EmbedDB CLI — SQLite for vectors.
#[derive(Parser)]
#[command(name = "embeddb")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(about = "Embedded vector database — one binary, one file, zero config")]
struct Cli {
    /// Path to the database file (default: ./data.embeddb)
    #[arg(short, long, default_value = "data.embeddb", global = true)]
    path: PathBuf,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize a new database
    Init {
        /// Page size in bytes
        #[arg(long, default_value = "4096")]
        page_size: u32,
    },

    /// Create a new collection
    CreateCollection {
        /// Collection name
        #[arg(short, long)]
        name: String,

        /// Vector dimension
        #[arg(short, long, default_value = "384")]
        dim: usize,

        /// Distance metric: cosine, euclidean, dot
        #[arg(short, long, default_value = "cosine")]
        distance: String,

        /// Index type: flat, hnsw
        #[arg(long, default_value = "flat")]
        index: String,
    },

    /// Insert a document into a collection
    Insert {
        /// Collection name
        #[arg(short, long)]
        collection: String,

        /// Document ID (auto-generated if not provided)
        #[arg(short, long)]
        id: Option<String>,

        /// Vector values as comma-separated f32 numbers (e.g. "0.1,0.2,0.3")
        #[arg(short, long, value_delimiter = ',')]
        vector: Option<Vec<f32>>,

        /// Metadata as JSON string
        #[arg(short, long)]
        meta: Option<String>,

        /// Text content (for future embedding support)
        #[arg(short, long)]
        text: Option<String>,

        /// Index type for auto-created collections: flat, hnsw
        #[arg(long, default_value = "flat")]
        index: String,
    },

    /// Search for similar vectors
    Search {
        /// Collection name
        #[arg(short, long)]
        collection: String,

        /// Query vector as comma-separated f32 numbers (or use --text for auto-embed)
        #[arg(short, long, value_delimiter = ',', required = false)]
        vector: Option<Vec<f32>>,

        /// Number of results (top-k)
        #[arg(short = 'k', long, default_value = "10")]
        top_k: usize,

        /// Metadata filter expression
        #[arg(short, long)]
        filter: Option<String>,

        /// Text query (auto-embed using SimpleEmbedder)
        #[arg(short, long)]
        text: Option<String>,

        /// Output format
        #[arg(long, default_value = "table")]
        format: String,
    },

    /// Show database information
    Info,

    /// Show collection statistics
    Stats {
        /// Collection name (omit for all collections)
        #[arg(short, long)]
        collection: Option<String>,
    },

    /// Start the web dashboard (Phase 2)
    Serve {
        /// Host address
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// Port number
        #[arg(long, default_value_t = 9020)]
        port: u16,
    },

    /// Delete a document from a collection
    Delete {
        /// Collection name
        #[arg(short, long)]
        collection: String,

        /// Document ID
        #[arg(short, long)]
        id: String,
    },
}

    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Init { page_size } => cmd_init(&cli.path, page_size),
        Commands::CreateCollection { name, dim, distance, index } => cmd_create_collection(&cli.path, &name, dim, &distance, &index),
        Commands::Insert {
            collection, id, vector, meta, text, index,
        } => cmd_insert(&cli.path, &collection, id, vector, meta, text, &index),
        Commands::Search {
            collection, vector, top_k, filter, text, format,
        } => cmd_search(&cli.path, &collection, vector, top_k, filter, text, &format),
        Commands::Info => cmd_info(&cli.path),
        Commands::Stats { collection } => cmd_stats(&cli.path, collection),
        Commands::Serve { host, port } => cmd_serve(&cli.path, &host, port).await,
        Commands::Delete { collection, id } => cmd_delete(&cli.path, &collection, &id),
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

// ---------------------------------------------------------------------------
// Command implementations
// ---------------------------------------------------------------------------

fn cmd_init(path: &std::path::Path, page_size: u32) -> Result<(), String> {
    let config = embeddb_core::config::DatabaseConfig {
        page_size,
        ..Default::default()
    };
    let db = Database::open_with_config(path, config).map_err(|e| e.to_string())?;
    println!("Initialized database at: {}", path.display());
    db.close().map_err(|e| e.to_string())?;
    Ok(())
}

fn cmd_create_collection(path: &std::path::Path, name: &str, dim: usize, distance: &str, index: &str) -> Result<(), String> {
    let db = Database::open(path).map_err(|e| e.to_string())?;
    let metric = match distance { "euclidean" => embeddb_core::DistanceMetric::Euclidean, "dot" => embeddb_core::DistanceMetric::DotProduct, _ => embeddb_core::DistanceMetric::Cosine, };
    let config = CollectionConfig { name: name.to_string(), dimension: dim, distance: metric, description: String::new(), data_root_page: 0, metadata_root_page: 0, index_type: index.to_string(), };
    db.create_collection(config).map_err(|e| e.to_string())?;
    println!("Created collection '{}' with dimension {}, distance={}, index={}", name, dim, distance, index);
    db.close().map_err(|e| e.to_string())?;
    Ok(())
}

fn cmd_insert(
    path: &std::path::Path,
    collection: &str,
    id: Option<String>,
    vector: Option<Vec<f32>>,
    meta: Option<String>,
    text: Option<String>,
    index: &str,
) -> Result<(), String> {
    let db = Database::open(path).map_err(|e| e.to_string())?;

    // Create collection if it doesn't exist (auto-detect dimension)
    if !db.collection_exists(collection) {
        let dim = vector.as_ref().map(|v| v.len()).unwrap_or(384);
        let mut config = CollectionConfig::new(collection, dim);
        config.index_type = index.to_string();
        db.create_collection(config).map_err(|e| e.to_string())?;
        println!("Created collection '{}' with dimension {}, index={}", collection, dim, index);
    }

    let metadata = meta
        .map(|m| serde_json::from_str(&m))
        .transpose()
        .map_err(|e| format!("Invalid JSON metadata: {}", e))?;

    let doc = Document {
        id,
        vector,
        metadata,
        text,
    };

    let doc_id = embeddb_core::insert(&db, collection, doc).map_err(|e| e.to_string())?;
    println!("Inserted document: {}", doc_id);

    db.close().map_err(|e| e.to_string())?;
    Ok(())
}

fn cmd_search(
    path: &std::path::Path,
    collection: &str,
    vector: Option<Vec<f32>>,
    top_k: usize,
    filter: Option<String>,
    text: Option<String>,
    format: &str,
) -> Result<(), String> {
    let db = Database::open(path).map_err(|e| e.to_string())?;

    // If --text provided, auto-embed using SimpleEmbedder
    let search_vector = if let Some(t) = text {
        let col = db.get_collection(collection).map_err(|e| format!("Cannot determine dimension: {}", e))?;
        let dim = col.read().dimension();
        let embedder = SimpleEmbedder::new(dim);
        embedder.embed(&t)
    } else if let Some(v) = vector {
        v
    } else {
        return Err("Either --vector or --text is required for search".into());
    };

    let query = SearchQuery::with_vector(search_vector, top_k);
    let query = if let Some(f) = filter {
        query.with_filter(f)
    } else {
        query
    };

    let hits = embeddb_core::search(&db, collection, query).map_err(|e| e.to_string())?;

    match format {
        "json" => {
            let output = serde_json::to_string_pretty(&hits).map_err(|e| e.to_string())?;
            println!("{}", output);
        }
        _ => {
            // Table format
            if hits.is_empty() {
                println!("No results found.");
            } else {
                println!("{:<5} {:<20} {:<15} {:<40}", "#", "ID", "Score", "Metadata");
                println!("{}", "-".repeat(85));
                for (i, hit) in hits.iter().enumerate() {
                    let meta_str = hit
                        .metadata
                        .as_ref()
                        .map(|m| m.to_string())
                        .unwrap_or_else(|| "-".to_string());
                    let meta_display = if meta_str.len() > 40 {
                        format!("{}...", &meta_str[..37])
                    } else {
                        meta_str
                    };
                    println!(
                        "{:<5} {:<20} {:<15.6} {:<40}",
                        i + 1,
                        truncate(&hit.id, 20),
                        hit.score,
                        meta_display
                    );
                }
            }
        }
    }

    db.close().map_err(|e| e.to_string())?;
    Ok(())
}

fn cmd_info(path: &std::path::Path) -> Result<(), String> {
    let db = Database::open(path).map_err(|e| e.to_string())?;
    let stats = db.stats().map_err(|e| e.to_string())?;

    println!("Database: {}", stats.path);
    println!("File size: {} bytes", stats.file_size);
    println!("Page size: {} bytes", stats.page_size);
    println!("Page count: {}", stats.page_count);
    println!("Collections: {}", stats.collection_count);
    println!();

    for col in &stats.collections {
        println!(
            "  {} | dim={} | metric={} | vectors={}",
            col.name, col.dimension, col.distance, col.vector_count
        );
    }

    db.close().map_err(|e| e.to_string())?;
    Ok(())
}

fn cmd_stats(
    path: &std::path::Path,
    collection: Option<String>,
) -> Result<(), String> {
    let db = Database::open(path).map_err(|e| e.to_string())?;
    let stats = db.stats().map_err(|e| e.to_string())?;

    if let Some(name) = collection {
        if let Some(col) = stats.collections.iter().find(|c| c.name == name) {
            println!("Collection: {}", col.name);
            println!("  Dimension: {}", col.dimension);
            println!("  Distance: {}", col.distance);
            println!("  Vectors: {}", col.vector_count);
            println!("  Metadata entries: {}", col.metadata_count);
        } else {
            return Err(format!("Collection '{}' not found", name));
        }
    } else {
        // Show all
        for col in &stats.collections {
            println!("{}: {} vectors (dim={}, {})", col.name, col.vector_count, col.dimension, col.distance);
        }
    }

    db.close().map_err(|e| e.to_string())?;
    Ok(())
}

async fn cmd_serve(path: &std::path::Path, host: &str, port: u16) -> Result<(), String> {
    let db_path = path.display().to_string();
    embeddb_server::serve(&db_path, host, port)
        .await
        .map_err(|e| e.to_string())
}

fn cmd_delete(path: &std::path::Path, collection: &str, id: &str) -> Result<(), String> {
    let db = Database::open(path).map_err(|e| e.to_string())?;

    let col = db
        .get_collection(collection)
        .map_err(|e| e.to_string())?;
    let mut col = col.write();
    col.delete(id).map_err(|e| e.to_string())?;

    println!("Deleted document: {}", id);

    db.close().map_err(|e| e.to_string())?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}
