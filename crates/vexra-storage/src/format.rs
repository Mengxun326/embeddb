//! On-disk page layout and type definitions.
//!
//! EmbedDB uses a SQLite-inspired paged file format. Each page is a fixed
//! size (default 4096 bytes) and consists of a header, cell pointer array,
//! free space, and cell data growing from opposite ends.
//!
//! # Page Layout
//!
//! ```text
//! ┌──────────────────┐
//! │ Page Header (16B) │  page_type, collection_id, free_space_offset,
//! │                   │  num_cells, crc32, flags
//! ├──────────────────┤
//! │ Cell Pointers     │  Array of (offset, len) pairs, growing forward
//! ├──────────────────┤
//! │ ... free space ...│
//! ├──────────────────┤
//! │ Cell Data         │  Variable-length data, growing backward
//! └──────────────────┘
//! ```

use serde::{Deserialize, Serialize};

/// Size of the page header in bytes (packed representation).
pub const PAGE_HEADER_SIZE: u16 = 16;

/// Size of the database file header in bytes.
pub const DB_HEADER_SIZE: usize = 100;

/// Cell pointer: (offset from page start, length in bytes).
pub type CellPointer = (u16, u16);

/// Size of a single cell pointer in bytes (two u16s).
pub const CELL_POINTER_SIZE: usize = 4;

/// Maximum number of cells per page.
pub const MAX_CELLS_PER_PAGE: u16 = 256;

/// Offset where cell data begins on a fresh page (after header + max cell pointers).
pub const DATA_START_OFFSET: u16 = PAGE_HEADER_SIZE + MAX_CELLS_PER_PAGE * CELL_POINTER_SIZE as u16;

// ---------------------------------------------------------------------------
// Page types
// ---------------------------------------------------------------------------

/// Identifies the type of data stored on a page.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PageType {
    /// Free / unallocated page.
    Free = 0x00,
    /// Vector data page: packed f32 vector arrays.
    Vector = 0x01,
    /// HNSW edge page: adjacency list fragments.
    HnswEdge = 0x02,
    /// Metadata page: JSON blobs and inverted index postings.
    Metadata = 0x03,
    /// Collection catalog page: B-tree internal or leaf node.
    Catalog = 0x04,
}

impl PageType {
    /// Try to convert a u8 into a PageType.
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0x00 => Some(Self::Free),
            0x01 => Some(Self::Vector),
            0x02 => Some(Self::HnswEdge),
            0x03 => Some(Self::Metadata),
            0x04 => Some(Self::Catalog),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Database header (first 100 bytes of page 0)
// ---------------------------------------------------------------------------

/// The database file header, stored in the first 100 bytes of the file.
///
/// Page 0 (first 4096 bytes) starts with the DB header, and the remaining
/// bytes on page 0 are unused (reserved for future use).
///
/// Binary layout (100 bytes):
/// - magic:          16 bytes (offset 0)
/// - format_version:  4 bytes (offset 16)
/// - page_size:       4 bytes (offset 20)
/// - page_count:      8 bytes (offset 24)
/// - catalog_root:    8 bytes (offset 32)
/// - wal_sequence:    8 bytes (offset 40)
/// - created_at:      8 bytes (offset 48)
/// - last_checkpoint: 8 bytes (offset 56)
/// - checksum:        4 bytes (offset 64)
/// - _reserved:       32 bytes (offset 68)
#[derive(Debug, Clone, Copy)]
pub struct DbHeader {
    /// Magic bytes: "EmbedDB v1\0" (11 bytes + 5 padding zeros).
    pub magic: [u8; 16],
    /// Format version number.
    pub format_version: u32,
    /// Page size in bytes (default 4096).
    pub page_size: u32,
    /// Total number of pages in the file.
    pub page_count: u64,
    /// Page ID of the first collection catalog page.
    pub catalog_root_page: u64,
    /// WAL sequence number (0 if no WAL).
    pub wal_sequence: u64,
    /// Creation timestamp (Unix epoch seconds).
    pub created_at: u64,
    /// Last checkpoint timestamp (Unix epoch seconds).
    pub last_checkpoint: u64,
    /// CRC32 checksum of the header (excluding this field).
    pub checksum: u32,
    /// Reserved for future use (padding to 100 bytes).
    pub _reserved: [u8; 32],
}

/// Pre-filled magic constant for DbHeader initialization.
const MAGIC_FILLED: [u8; 16] = {
    let mut m = [0u8; 16];
    m[0] = b'E';
    m[1] = b'm';
    m[2] = b'b';
    m[3] = b'e';
    m[4] = b'd';
    m[5] = b'D';
    m[6] = b'B';
    m[7] = b' ';
    m[8] = b'v';
    m[9] = b'1';
    m[10] = b'\0';
    m
};

impl Default for DbHeader {
    fn default() -> Self {
        Self {
            magic: MAGIC_FILLED,
            format_version: crate::FORMAT_VERSION,
            page_size: crate::DEFAULT_PAGE_SIZE,
            page_count: 1,
            catalog_root_page: 0,
            wal_sequence: 0,
            created_at: 0,
            last_checkpoint: 0,
            checksum: 0,
            _reserved: [0u8; 32],
        }
    }
}

impl DbHeader {
    /// Create a new header with the current timestamp and default values.
    pub fn new(page_size: u32) -> Self {
        Self {
            page_size,
            created_at: current_timestamp(),
            ..Default::default()
        }
    }

    /// Serialize to a byte buffer (must be exactly DB_HEADER_SIZE bytes).
    pub fn serialize_to(&self, buf: &mut [u8]) {
        debug_assert!(buf.len() >= DB_HEADER_SIZE);
        buf[..DB_HEADER_SIZE].fill(0);

        buf[0..16].copy_from_slice(&self.magic);
        buf[16..20].copy_from_slice(&self.format_version.to_le_bytes());
        buf[20..24].copy_from_slice(&self.page_size.to_le_bytes());
        buf[24..32].copy_from_slice(&self.page_count.to_le_bytes());
        buf[32..40].copy_from_slice(&self.catalog_root_page.to_le_bytes());
        buf[40..48].copy_from_slice(&self.wal_sequence.to_le_bytes());
        buf[48..56].copy_from_slice(&self.created_at.to_le_bytes());
        buf[56..64].copy_from_slice(&self.last_checkpoint.to_le_bytes());
        buf[64..68].copy_from_slice(&self.checksum.to_le_bytes());
        buf[68..100].copy_from_slice(&self._reserved);
    }

    /// Deserialize from a byte buffer (must be at least DB_HEADER_SIZE bytes).
    pub fn deserialize_from(buf: &[u8]) -> Option<Self> {
        if buf.len() < DB_HEADER_SIZE {
            return None;
        }

        let mut magic = [0u8; 16];
        magic.copy_from_slice(&buf[0..16]);

        let format_version = u32::from_le_bytes([buf[16], buf[17], buf[18], buf[19]]);
        let page_size = u32::from_le_bytes([buf[20], buf[21], buf[22], buf[23]]);
        let page_count = u64::from_le_bytes([
            buf[24], buf[25], buf[26], buf[27], buf[28], buf[29], buf[30], buf[31],
        ]);
        let catalog_root_page = u64::from_le_bytes([
            buf[32], buf[33], buf[34], buf[35], buf[36], buf[37], buf[38], buf[39],
        ]);
        let wal_sequence = u64::from_le_bytes([
            buf[40], buf[41], buf[42], buf[43], buf[44], buf[45], buf[46], buf[47],
        ]);
        let created_at = u64::from_le_bytes([
            buf[48], buf[49], buf[50], buf[51], buf[52], buf[53], buf[54], buf[55],
        ]);
        let last_checkpoint = u64::from_le_bytes([
            buf[56], buf[57], buf[58], buf[59], buf[60], buf[61], buf[62], buf[63],
        ]);
        let checksum = u32::from_le_bytes([buf[64], buf[65], buf[66], buf[67]]);

        let mut reserved = [0u8; 32];
        reserved.copy_from_slice(&buf[68..100]);

        Some(Self {
            magic,
            format_version,
            page_size,
            page_count,
            catalog_root_page,
            wal_sequence,
            created_at,
            last_checkpoint,
            checksum,
            _reserved: reserved,
        })
    }

    /// Validate the magic bytes.
    pub fn validate_magic(&self) -> bool {
        &self.magic[..11] == crate::MAGIC
    }

    /// Validate the format version.
    pub fn validate_version(&self) -> bool {
        self.format_version == crate::FORMAT_VERSION
    }

    /// Compute and set the CRC32 checksum over the header bytes.
    pub fn update_checksum(&mut self) {
        self.checksum = 0;
        let mut buf = [0u8; DB_HEADER_SIZE];
        self.serialize_to(&mut buf);
        // CRC over bytes 0..64 (excluding checksum at 64..68 and reserved at 68..100)
        self.checksum = crc32fast::hash(&buf[..64]);
    }

    /// Verify the CRC32 checksum.
    pub fn verify_checksum(&self) -> bool {
        let stored = self.checksum;
        let mut copy = *self;
        copy.checksum = 0;
        let mut buf = [0u8; DB_HEADER_SIZE];
        copy.serialize_to(&mut buf);
        crc32fast::hash(&buf[..64]) == stored
    }

    /// Size of the serialized header in bytes.
    pub fn serialized_size() -> usize {
        DB_HEADER_SIZE
    }
}

// ---------------------------------------------------------------------------
// Page header (first 16 bytes of every page)
// ---------------------------------------------------------------------------

/// The page header, stored at the beginning of every page.
///
/// Binary layout (exactly 16 bytes, no padding):
/// - page_type:        1 byte  (offset 0)
/// - collection_id:    4 bytes (offset 1)
/// - free_space_offset: 2 bytes (offset 5)
/// - num_cells:        2 bytes (offset 7)
/// - crc32:            4 bytes (offset 9)
/// - flags:            2 bytes (offset 13)
/// - _pad:             1 byte  (offset 15)
#[derive(Debug, Clone, Copy)]
pub struct PageHeader {
    /// Type of data on this page.
    pub page_type: u8,
    /// Collection ID this page belongs to (0 for catalog pages).
    pub collection_id: u32,
    /// Offset from page start to the beginning of free space.
    pub free_space_offset: u16,
    /// Number of cells stored on this page.
    pub num_cells: u16,
    /// CRC32 checksum of page data (excluding the header bytes).
    pub crc32: u32,
    /// Bit flags (bit 0: dirty, bit 1-15: reserved).
    pub flags: u16,
    /// Padding to 16 bytes.
    pub _pad: u8,
}

impl PageHeader {
    /// Create a new page header for the given page type.
    pub fn new(page_type: PageType, collection_id: u32) -> Self {
        Self {
            page_type: page_type as u8,
            collection_id,
            free_space_offset: DATA_START_OFFSET,
            num_cells: 0,
            crc32: 0,
            flags: 0,
            _pad: 0,
        }
    }

    /// Serialize to 16 bytes.
    pub fn serialize_to(&self, buf: &mut [u8]) {
        debug_assert!(buf.len() >= PAGE_HEADER_SIZE as usize);
        buf[0] = self.page_type;
        buf[1..5].copy_from_slice(&self.collection_id.to_le_bytes());
        buf[5..7].copy_from_slice(&self.free_space_offset.to_le_bytes());
        buf[7..9].copy_from_slice(&self.num_cells.to_le_bytes());
        buf[9..13].copy_from_slice(&self.crc32.to_le_bytes());
        buf[13..15].copy_from_slice(&self.flags.to_le_bytes());
        buf[15] = self._pad;
    }

    /// Deserialize from 16 bytes.
    pub fn deserialize_from(buf: &[u8]) -> Option<Self> {
        if buf.len() < PAGE_HEADER_SIZE as usize {
            return None;
        }

        Some(Self {
            page_type: buf[0],
            collection_id: u32::from_le_bytes([buf[1], buf[2], buf[3], buf[4]]),
            free_space_offset: u16::from_le_bytes([buf[5], buf[6]]),
            num_cells: u16::from_le_bytes([buf[7], buf[8]]),
            crc32: u32::from_le_bytes([buf[9], buf[10], buf[11], buf[12]]),
            flags: u16::from_le_bytes([buf[13], buf[14]]),
            _pad: buf[15],
        })
    }

    /// Check if the page has room for `needed` bytes of cell data.
    ///
    /// Cell pointers are pre-allocated at the top of the page (up to MAX_CELLS_PER_PAGE).
    /// Cell data grows upward from DATA_START_OFFSET. Available space is what remains.
    pub fn has_room(&self, page_size: u32, needed: u16) -> bool {
        if self.num_cells >= MAX_CELLS_PER_PAGE {
            return false;
        }
        self.free_space_offset as u32 + needed as u32 <= page_size
    }

    /// Get the free space available on this page.
    pub fn free_space(&self, page_size: u32) -> u16 {
        let used = self.free_space_offset as u32;
        if used >= page_size {
            return 0;
        }
        (page_size - used) as u16
    }

    /// Compute and set the CRC32 checksum for the page data.
    pub fn compute_crc(&mut self, page_data: &[u8]) {
        self.crc32 = 0;
        let crc = crc32fast::hash(&page_data[PAGE_HEADER_SIZE as usize..]);
        self.crc32 = crc;
    }

    /// Verify the CRC32 checksum.
    pub fn verify_crc(&self, page_data: &[u8]) -> bool {
        let expected = self.crc32;
        let mut copy = *self;
        copy.crc32 = 0;
        let mut header_buf = [0u8; PAGE_HEADER_SIZE as usize];
        copy.serialize_to(&mut header_buf);

        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&header_buf);
        hasher.update(&page_data[PAGE_HEADER_SIZE as usize..]);
        hasher.finalize() == expected
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Read a page header from a byte slice.
pub fn read_page_header(data: &[u8]) -> Option<PageHeader> {
    if data.len() < PAGE_HEADER_SIZE as usize {
        return None;
    }
    PageHeader::deserialize_from(&data[..PAGE_HEADER_SIZE as usize])
}

/// Write a page header to a mutable byte slice.
pub fn write_page_header(data: &mut [u8], header: &PageHeader) {
    if data.len() >= PAGE_HEADER_SIZE as usize {
        header.serialize_to(&mut data[..PAGE_HEADER_SIZE as usize]);
    }
}

/// Read cell pointers from a page as raw bytes.
/// Returns a byte slice containing (u16 offset, u16 len) pairs.
pub fn read_cell_pointers_raw<'a>(header: &PageHeader, data: &'a [u8]) -> &'a [u8] {
    let start = PAGE_HEADER_SIZE as usize;
    let end = start + header.num_cells as usize * CELL_POINTER_SIZE;
    if end > data.len() {
        return &[];
    }
    &data[start..end]
}

/// Decode cell pointers from raw bytes into (offset, length) pairs.
pub fn decode_cell_pointers(raw: &[u8]) -> Vec<CellPointer> {
    raw.chunks_exact(CELL_POINTER_SIZE)
        .map(|chunk| {
            let offset = u16::from_le_bytes([chunk[0], chunk[1]]);
            let len = u16::from_le_bytes([chunk[2], chunk[3]]);
            (offset, len)
        })
        .collect()
}

/// Encode a cell pointer to raw bytes.
pub fn encode_cell_pointer(ptr: CellPointer) -> [u8; CELL_POINTER_SIZE] {
    let mut buf = [0u8; CELL_POINTER_SIZE];
    buf[0..2].copy_from_slice(&ptr.0.to_le_bytes());
    buf[2..4].copy_from_slice(&ptr.1.to_le_bytes());
    buf
}

/// Read cell pointers from a page as decoded pairs.
pub fn read_cell_pointers(header: &PageHeader, data: &[u8]) -> Vec<CellPointer> {
    let raw = read_cell_pointers_raw(header, data);
    decode_cell_pointers(raw)
}

/// Get the data of a specific cell by its pointer.
pub fn read_cell_data<'a>(data: &'a [u8], ptr: CellPointer) -> &'a [u8] {
    let start = ptr.0 as usize;
    let end = start + ptr.1 as usize;
    if end > data.len() {
        return &[];
    }
    &data[start..end]
}

/// Get the current Unix timestamp in seconds.
pub fn current_timestamp() -> u64 {
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
    #[cfg(target_arch = "wasm32")]
    {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_header_serialize_roundtrip() {
        let header = PageHeader::new(PageType::Vector, 42);
        let mut buf = [0u8; 16];
        header.serialize_to(&mut buf);

        let decoded = PageHeader::deserialize_from(&buf).unwrap();
        assert_eq!(decoded.page_type, PageType::Vector as u8);
        assert_eq!(decoded.collection_id, 42);
        assert_eq!(decoded.free_space_offset, DATA_START_OFFSET);
        assert_eq!(decoded.num_cells, 0);
    }

    #[test]
    fn test_db_header_serialize_roundtrip() {
        let header = DbHeader::new(4096);
        let mut buf = [0u8; DB_HEADER_SIZE];
        header.serialize_to(&mut buf);

        let decoded = DbHeader::deserialize_from(&buf).unwrap();
        assert!(decoded.validate_magic());
        assert!(decoded.validate_version());
        assert_eq!(decoded.page_size, 4096);
        assert_eq!(decoded.page_count, 1);
    }

    #[test]
    fn test_db_header_checksum_roundtrip() {
        let mut header = DbHeader::new(4096);
        header.update_checksum();
        assert!(header.verify_checksum());

        // Corrupt the header
        header.format_version = 99;
        assert!(!header.verify_checksum());
    }

    #[test]
    fn test_page_header_has_room() {
        let header = PageHeader::new(PageType::Vector, 1);
        // Data starts at DATA_START_OFFSET = 1040, leaving 4096-1040=3056 bytes
        assert!(header.has_room(4096, 100));
        assert!(header.has_room(4096, 3000));
        assert!(!header.has_room(4096, 4000));
    }

    #[test]
    fn test_cell_pointer_encode_decode() {
        let ptr: CellPointer = (100, 50);
        let encoded = encode_cell_pointer(ptr);
        let decoded = decode_cell_pointers(&encoded);
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0], ptr);
    }
}
