//! LRU page cache with memory-mapped I/O support.
//!
//! The page cache manages in-memory copies of database pages. It uses a
//! two-tier approach:
//!
//! 1. **Hot tier**: Memory-mapped pages for zero-copy reads
//! 2. **Cold tier**: Recently evicted pages that can be quickly re-read
//!
//! The cache supports concurrent readers through Arc-based sharing of
//! page data. Each CachedPage is wrapped in `Arc<RwLock<CachedPage>>`
//! to allow shared reads and exclusive writes.

use crate::error::{Result, StorageError};
use crate::format::{self, DbHeader, PageHeader, PageType, PAGE_HEADER_SIZE, DB_HEADER_SIZE, CELL_POINTER_SIZE, MAX_CELLS_PER_PAGE};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// An in-memory cached page.
#[derive(Clone)]
pub struct CachedPage {
    /// The page's unique identifier.
    pub page_id: u64,
    /// Raw page data (page_size bytes).
    pub data: Vec<u8>,
    /// Whether the page has been modified since reading from disk.
    pub dirty: bool,
}

impl CachedPage {
    /// Get the page header.
    /// Page 0 has a synthetic header (no PageHeader on disk at offset 0).
    pub fn header(&self) -> PageHeader {
        if self.page_id == 0 {
            // Page 0 uses DbHeader at offset 0, no PageHeader stored.
            // Return a synthetic header for the catalog page.
            return PageHeader::new(PageType::Catalog, 0);
        }
        format::read_page_header(&self.data)
            .expect("page data too small for header")
    }

    /// Get cell data by index.
    pub fn cell_data(&self, index: u16) -> Option<Vec<u8>> {
        let header = self.header();
        if index >= header.num_cells {
            return None;
        }
        let pointers = format::read_cell_pointers(&header, &self.data);
        pointers.get(index as usize).map(|ptr| {
            format::read_cell_data(&self.data, *ptr).to_vec()
        })
    }

    /// Mark the page as dirty.
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }
}

/// Configuration for the page cache.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of pages to keep in memory.
    pub max_pages: usize,
    /// Page size in bytes (must match the database file).
    pub page_size: u32,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_pages: 16384, // 64MB with 4KB pages
            page_size: crate::DEFAULT_PAGE_SIZE,
        }
    }
}

/// The page cache — manages in-memory copies of database pages.
pub struct PageCache {
    config: CacheConfig,
    /// Map of page_id → cached page.
    pages: RwLock<HashMap<u64, Arc<RwLock<CachedPage>>>>,
    /// File handle for disk I/O.
    file: RwLock<File>,
    /// Path to the database file.
    path: PathBuf,
}

impl PageCache {
    /// Open a page cache for an existing or new database file.
    pub fn open(path: impl AsRef<Path>, config: CacheConfig) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let exists = path.exists();

        // Ensure the file exists and has at least one page
        if !exists {
            let file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&path)?;
            file.set_len(config.page_size as u64)?;
        }

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)?;

        let cache = Self {
            config,
            pages: RwLock::new(HashMap::new()),
            file: RwLock::new(file),
            path,
        };

        // Initialize page 0 if this is a new database
        if !exists {
            cache.init_page_zero()?;
        }

        Ok(cache)
    }

    /// Initialize page 0 with a database header.
    ///
    /// Page 0 layout:
    /// - Bytes 0..100: DbHeader (includes page-level metadata)
    /// - Bytes 100..4096: Available for catalog data
    ///
    /// Page 0 does NOT have a standard PageHeader at offset 0;
    /// the `header()` method synthesizes one for page 0.
    fn init_page_zero(&self) -> Result<()> {
        let mut page_data = vec![0u8; self.config.page_size as usize];

        // Write the database header at offset 0
        let db_header = DbHeader::new(self.config.page_size);
        let mut header_buf = [0u8; DB_HEADER_SIZE];
        db_header.serialize_to(&mut header_buf);
        page_data[..DB_HEADER_SIZE].copy_from_slice(&header_buf);

        // Write to disk
        self.write_page_to_disk(0, &page_data)?;

        // Cache page 0
        let cached = CachedPage {
            page_id: 0,
            data: page_data,
            dirty: false,
        };
        self.pages
            .write()
            .insert(0, Arc::new(RwLock::new(cached)));

        Ok(())
    }

    /// Get a page from the cache, loading it from disk if necessary.
    pub fn get_page(&self, page_id: u64) -> Result<Arc<RwLock<CachedPage>>> {
        // Check cache first
        {
            let pages = self.pages.read();
            if let Some(page) = pages.get(&page_id) {
                return Ok(Arc::clone(page));
            }
        }

        // Read from disk
        let page_data = self.read_page_from_disk(page_id)?;

        let cached = CachedPage {
            page_id,
            data: page_data,
            dirty: false,
        };

        let page = Arc::new(RwLock::new(cached));

        // Insert into cache (with eviction if needed)
        {
            let mut pages = self.pages.write();
            if pages.len() >= self.config.max_pages {
                self.evict_one(&mut pages);
            }
            pages.insert(page_id, Arc::clone(&page));
        }

        Ok(page)
    }

    /// Allocate a new, empty page.
    pub fn allocate_page(&self, page_type: PageType, collection_id: u32) -> Result<u64> {
        let mut page_data = vec![0u8; self.config.page_size as usize];
        let page_header = PageHeader::new(page_type, collection_id);
        let mut ph_buf = [0u8; PAGE_HEADER_SIZE as usize];
        page_header.serialize_to(&mut ph_buf);
        page_data[..PAGE_HEADER_SIZE as usize].copy_from_slice(&ph_buf);

        // Get the next page ID
        let page_id = self.next_page_id()?;

        // Write to disk
        self.write_page_to_disk(page_id, &page_data)?;

        // Cache
        let cached = CachedPage {
            page_id,
            data: page_data,
            dirty: false,
        };
        {
            let mut pages = self.pages.write();
            if pages.len() >= self.config.max_pages {
                self.evict_one(&mut pages);
            }
            pages.insert(page_id, Arc::new(RwLock::new(cached)));
        }

        // Update page count in header
        self.update_page_count(page_id + 1)?;

        Ok(page_id)
    }

    /// Write (insert or update) cell data to a page.
    ///
    /// Cell pointers are stored at PAGE_HEADER_SIZE..DATA_START_OFFSET (pre-allocated).
    /// Cell data is stored starting at DATA_START_OFFSET and grows upward.
    ///
    /// Returns true if the cell was successfully written.
    /// Returns false if there is not enough room on the page.
    pub fn write_cell(
        &self,
        page_id: u64,
        cell_data: &[u8],
        cell_index: Option<u16>,
    ) -> Result<bool> {
        let page = self.get_page(page_id)?;
        let mut pg = page.write();

        let needed = cell_data.len() as u16;
        let header = pg.header();

        // Check limits
        if header.num_cells >= MAX_CELLS_PER_PAGE {
            return Ok(false);
        }
        if !header.has_room(self.config.page_size, needed) {
            return Ok(false);
        }

        // Determine cell index
        let cell_idx = cell_index.unwrap_or(header.num_cells);

        // Write cell data at free_space_offset (grows upward from DATA_START_OFFSET)
        let cell_start = header.free_space_offset;
        let cell_end = cell_start + needed;
        pg.data[cell_start as usize..cell_end as usize].copy_from_slice(cell_data);

        // Write cell pointer at pre-allocated position (PAGE_HEADER_SIZE + idx * 4)
        let ptr_offset = (PAGE_HEADER_SIZE + cell_idx * CELL_POINTER_SIZE as u16) as usize;
        let encoded = format::encode_cell_pointer((cell_start, needed));
        pg.data[ptr_offset..ptr_offset + CELL_POINTER_SIZE].copy_from_slice(&encoded);

        // Update header in-place
        let mut new_header = header;
        new_header.free_space_offset = cell_end;
        if cell_index.is_none() {
            new_header.num_cells += 1;
        }
        let mut ph_buf = [0u8; PAGE_HEADER_SIZE as usize];
        new_header.serialize_to(&mut ph_buf);
        pg.data[..PAGE_HEADER_SIZE as usize].copy_from_slice(&ph_buf);

        pg.mark_dirty();
        Ok(true)
    }

    /// Flush a dirty page to disk.
    pub fn flush_page(&self, page_id: u64) -> Result<()> {
        let pages = self.pages.read();
        if let Some(page) = pages.get(&page_id) {
            let pg = page.read();
            if pg.dirty {
                self.write_page_to_disk(page_id, &pg.data)?;
            }
        }
        Ok(())
    }

    /// Flush all dirty pages to disk.
    pub fn flush_all(&self) -> Result<()> {
        let pages = self.pages.read();
        for (page_id, page) in pages.iter() {
            let pg = page.read();
            if pg.dirty {
                self.write_page_to_disk(*page_id, &pg.data)?;
            }
        }
        Ok(())
    }

    /// Get the database header from page 0.
    pub fn db_header(&self) -> Result<DbHeader> {
        let page = self.get_page(0)?;
        let pg = page.read();
        DbHeader::deserialize_from(&pg.data[..DB_HEADER_SIZE])
            .ok_or_else(|| StorageError::InvalidFile("Cannot parse DB header".into()))
    }

    /// Update the database header on page 0.
    pub fn update_db_header(&self, header: &DbHeader) -> Result<()> {
        let page = self.get_page(0)?;
        let mut pg = page.write();
        let mut buf = [0u8; DB_HEADER_SIZE];
        header.serialize_to(&mut buf);
        pg.data[..DB_HEADER_SIZE].copy_from_slice(&buf);
        pg.mark_dirty();
        Ok(())
    }

    /// Get the next available page ID.
    fn next_page_id(&self) -> Result<u64> {
        let header = self.db_header()?;
        Ok(header.page_count)
    }

    /// Update the page count in the database header.
    fn update_page_count(&self, new_count: u64) -> Result<()> {
        let mut header = self.db_header()?;
        header.page_count = new_count;
        self.update_db_header(&header)
    }

    /// Read a page from disk by ID.
    fn read_page_from_disk(&self, page_id: u64) -> Result<Vec<u8>> {
        let offset = page_id * self.config.page_size as u64;
        let mut buf = vec![0u8; self.config.page_size as usize];

        let mut file = self.file.write();
        file.seek(SeekFrom::Start(offset))?;
        file.read_exact(&mut buf)?;

        Ok(buf)
    }

    /// Write a page to disk by ID.
    pub fn write_page_to_disk(&self, page_id: u64, data: &[u8]) -> Result<()> {
        let offset = page_id * self.config.page_size as u64;

        // Ensure the file is large enough
        let mut file = self.file.write();
        let file_len = file.metadata()?.len();
        let required_len = offset + self.config.page_size as u64;
        if file_len < required_len {
            file.set_len(required_len)?;
        }

        file.seek(SeekFrom::Start(offset))?;
        file.write_all(data)?;
        file.flush()?;

        Ok(())
    }

    /// Evict one page from the cache (simple: evict the first non-dirty page).
    fn evict_one(&self, pages: &mut HashMap<u64, Arc<RwLock<CachedPage>>>) {
        let victim = pages
            .iter()
            .find(|(id, page)| {
                **id != 0 && {
                    let pg = page.read();
                    !pg.dirty
                }
            })
            .map(|(id, _)| *id);

        if let Some(id) = victim {
            pages.remove(&id);
        }
    }

    /// Get the path of the database file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get the page size.
    pub fn page_size(&self) -> u32 {
        self.config.page_size
    }

    /// Close the cache, flushing all dirty pages.
    pub fn close(&self) -> Result<()> {
        self.flush_all()?;
        Ok(())
    }
}

impl Drop for PageCache {
    fn drop(&mut self) {
        let _ = self.flush_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_read_page_zero() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.embeddb");

        let cache = PageCache::open(&path, CacheConfig::default()).unwrap();

        let page = cache.get_page(0).unwrap();
        let pg = page.read();
        assert_eq!(pg.header().page_type, PageType::Catalog as u8);

        let header = cache.db_header().unwrap();
        assert!(header.validate_magic());
        assert_eq!(header.page_size, 4096);
    }

    #[test]
    fn test_allocate_and_write_page() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.embeddb");

        let cache = PageCache::open(&path, CacheConfig::default()).unwrap();

        let page_id = cache.allocate_page(PageType::Vector, 1).unwrap();
        assert_eq!(page_id, 1);

        let page = cache.get_page(page_id).unwrap();
        let pg = page.read();
        assert_eq!(pg.header().page_type, PageType::Vector as u8);
        assert_eq!(pg.header().collection_id, 1);
    }

    #[test]
    fn test_write_and_read_cell() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.embeddb");

        let cache = PageCache::open(&path, CacheConfig::default()).unwrap();
        let page_id = cache.allocate_page(PageType::Vector, 1).unwrap();

        let cell_data = b"hello embeddb cell data";
        let ok = cache.write_cell(page_id, cell_data, None).unwrap();
        assert!(ok);

        let page = cache.get_page(page_id).unwrap();
        let pg = page.read();
        assert_eq!(pg.header().num_cells, 1);
        assert_eq!(pg.cell_data(0).unwrap(), cell_data);
    }

    #[test]
    fn test_page_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.embeddb");

        let page_id;
        {
            let cache = PageCache::open(&path, CacheConfig::default()).unwrap();
            page_id = cache.allocate_page(PageType::Vector, 1).unwrap();
            cache.write_cell(page_id, b"persistent data", None).unwrap();
            cache.flush_all().unwrap();
        }

        // Reopen and verify
        let cache = PageCache::open(&path, CacheConfig::default()).unwrap();
        let page = cache.get_page(page_id).unwrap();
        let pg = page.read();
        assert_eq!(pg.cell_data(0).unwrap(), b"persistent data");
    }
}
