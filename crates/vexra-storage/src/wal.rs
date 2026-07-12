//! Write-Ahead Log (WAL) implementation.
//!
//! The WAL provides crash-safe writes. All modifications are first appended
//! to the WAL file before being written to the main database file. This
//! guarantees that after a crash, the database can be recovered to a
//! consistent state by replaying the WAL.
//!
//! # WAL File Format
//!
//! ```text
//! WAL Header (32 bytes):
//!   magic:        u32 = 0x377F0683
//!   version:      u32 = 1
//!   page_size:    u32
//!   checkpoint_seq: u32
//!   salt1:        u32
//!   salt2:        u32
//!   checksum:     u64
//!
//! WAL Frame (24 + page_size bytes):
//!   page_number:   u32
//!   db_size_after: u64 (pages in DB after this frame is committed)
//!   salt1:         u32
//!   salt2:         u32
//!   checksum1:     u32
//!   checksum2:     u32
//!   page_data:     [u8; page_size]
//! ```

use crate::error::{Result, StorageError};
use parking_lot::Mutex;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// WAL magic number (inspired by SQLite's WAL magic).
pub const WAL_MAGIC: u32 = 0x377F_0683;

/// Current WAL format version.
pub const WAL_VERSION: u32 = 1;

/// Size of the WAL header in bytes.
pub const WAL_HEADER_SIZE: usize = 32;

/// Size of a single WAL frame header (without page data).
/// Layout: page_number(u32) + db_size_after(u64) + salt1(u32) + salt2(u32) + checksum1(u32) + checksum2(u32) = 28 bytes
pub const WAL_FRAME_HEADER_SIZE: usize = 28;

/// A single WAL frame: a modified page + metadata.
#[derive(Debug, Clone)]
pub struct WalFrame {
    /// Page number being modified.
    pub page_number: u32,
    /// Database size (in pages) after this frame is committed.
    pub db_size_after: u64,
    /// Full page data.
    pub page_data: Vec<u8>,
}

/// Write-Ahead Log manager.
pub struct WalManager {
    path: PathBuf,
    file: Mutex<Option<File>>,
    page_size: u32,
    sequence: Mutex<u64>,
    salt1: std::sync::atomic::AtomicU32,
    salt2: std::sync::atomic::AtomicU32,
    /// Frames written since last checkpoint.
    frame_count: Mutex<u64>,
    /// Auto-checkpoint threshold (frames).
    checkpoint_threshold: u64,
}

impl WalManager {
    /// Create a new WAL manager.
    pub fn new(db_path: impl AsRef<Path>, page_size: u32) -> Self {
        let wal_path = wal_path_from_db(db_path.as_ref());

        let salt1 = std::sync::atomic::AtomicU32::new(fastrand::u32(..));
        let salt2 = std::sync::atomic::AtomicU32::new(fastrand::u32(..));

        Self {
            path: wal_path,
            file: Mutex::new(None),
            page_size,
            sequence: Mutex::new(0),
            salt1,
            salt2,
            frame_count: Mutex::new(0),
            checkpoint_threshold: 1000,
        }
    }

    /// Open the WAL file, creating it if necessary.
    pub fn open(&self) -> Result<()> {
        let exists = self.path.exists();

        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&self.path)?;

        if !exists || file.metadata()?.len() == 0 {
            // Write WAL header
            self.write_wal_header(&mut file)?;
        } else {
            // Validate existing WAL header
            self.validate_wal_header(&mut file)?;
        }

        *self.file.lock() = Some(file);
        Ok(())
    }

    /// Append a frame to the WAL.
    ///
    /// This writes the modified page to the WAL file. The actual database
    /// file is NOT modified — that happens at checkpoint time.
    pub fn append_frame(&self, page_number: u32, page_data: &[u8], db_size_after: u64) -> Result<u64> {
        let mut file_guard = self.file.lock();
        let file = file_guard
            .as_mut()
            .ok_or_else(|| StorageError::Other("WAL not open".into()))?;

        // Seek to end of file
        file.seek(SeekFrom::End(0))?;

        // Get the next sequence number
        let mut seq = self.sequence.lock();
        *seq += 1;
        let frame_seq = *seq;

        // Build frame header
        let mut frame_header = [0u8; WAL_FRAME_HEADER_SIZE];

        // page_number (4 bytes)
        frame_header[0..4].copy_from_slice(&page_number.to_le_bytes());
        // db_size_after (8 bytes)
        frame_header[4..12].copy_from_slice(&db_size_after.to_le_bytes());
        // salt1 (4 bytes)
        frame_header[12..16].copy_from_slice(&self.salt1.load(std::sync::atomic::Ordering::Relaxed).to_le_bytes());
        // salt2 (4 bytes)
        frame_header[16..20].copy_from_slice(&self.salt2.load(std::sync::atomic::Ordering::Relaxed).to_le_bytes());

        // Compute checksums
        let s1 = self.salt1.load(std::sync::atomic::Ordering::Relaxed);
        let s2 = self.salt2.load(std::sync::atomic::Ordering::Relaxed);
        let (checksum1, checksum2) = Self::compute_frame_checksum(page_number, page_data, s1, s2, 0);

        frame_header[20..24].copy_from_slice(&checksum1.to_le_bytes());
        frame_header[24..28].copy_from_slice(&checksum2.to_le_bytes());

        // Write frame header + page data
        file.write_all(&frame_header)?;
        file.write_all(page_data)?;
        file.flush()?;

        // Update frame count
        let mut count = self.frame_count.lock();
        *count += 1;

        // Track frame count (auto-checkpoint handled by Database::maybe_checkpoint)
        drop(count);

        Ok(frame_seq)
    }

    /// Checkpoint: write all WAL frames back to the main database file.
    ///
    /// After a successful checkpoint, the WAL file is truncated.
    pub fn checkpoint(&self, db_path: &Path) -> Result<()> {
        let mut file_guard = self.file.lock();
        let file = file_guard
            .as_mut()
            .ok_or_else(|| StorageError::Other("WAL not open".into()))?;

        // Read all frames from the WAL
        let frames = self.read_all_frames(file)?;

        if frames.is_empty() {
            return Ok(());
        }

        // Open the database file for writing
        let mut db_file = OpenOptions::new().write(true).open(db_path)?;

        // Write each frame's page data to the database file
        for frame in &frames {
            let offset = frame.page_number as u64 * self.page_size as u64;
            db_file.seek(SeekFrom::Start(offset))?;
            db_file.write_all(&frame.page_data)?;
        }
        db_file.flush()?;

        // Truncate the WAL
        file.set_len(0)?;
        file.seek(SeekFrom::Start(0))?;
        self.write_wal_header(file)?;

        // Reset counters
        *self.sequence.lock() = 0;
        *self.frame_count.lock() = 0;

        Ok(())
    }

    /// Recover from WAL after a crash.
    ///
    /// Reads all frames from the WAL and applies them to the database file.
    /// Returns the number of frames replayed.
    pub fn recover(&self, db_path: &Path) -> Result<u64> {
        if !self.path.exists() {
            return Ok(0);
        }

        let mut wal_file = OpenOptions::new().read(true).write(true).open(&self.path)?;

        if wal_file.metadata()?.len() <= WAL_HEADER_SIZE as u64 {
            return Ok(0);
        }

        // Read and validate WAL header to extract the original salts
        let (header_salt1, header_salt2) = match self.read_wal_header(&mut wal_file) {
            Ok((s1, s2)) => {
                if s1 == 0 && s2 == 0 { return Ok(0); }
                (s1, s2)
            }
            Err(_) => {
                drop(wal_file);
                std::fs::remove_file(&self.path)?;
                return Ok(0);
            }
        };

        // Use the original salts from the WAL header for checksum validation
        self.salt1.store(header_salt1, std::sync::atomic::Ordering::Relaxed);
        self.salt2.store(header_salt2, std::sync::atomic::Ordering::Relaxed);

        // Read frames
        let frames = self.read_all_frames(&mut wal_file)?;
        let frame_count = frames.len() as u64;

        if frame_count == 0 {
            return Ok(0);
        }

        // Apply frames to database
        let mut db_file = OpenOptions::new().write(true).open(db_path)?;

        for frame in &frames {
            let offset = frame.page_number as u64 * self.page_size as u64;
            db_file.seek(SeekFrom::Start(offset))?;
            db_file.write_all(&frame.page_data)?;
        }
        db_file.flush()?;

        // Truncate WAL
        drop(db_file);
        let mut wal_file = OpenOptions::new().write(true).truncate(true).open(&self.path)?;
        wal_file.set_len(0)?;
        self.write_wal_header(&mut wal_file)?;

        Ok(frame_count)
    }

    /// Check if the WAL file exists.
    pub fn exists(&self) -> bool {
        self.path.exists()
    }

    /// Check if a checkpoint is needed (frame count exceeds threshold).
    pub fn needs_checkpoint(&self) -> bool {
        *self.frame_count.lock() >= self.checkpoint_threshold
    }

    /// Delete the WAL file (e.g., after a clean shutdown with full checkpoint).
    pub fn remove(&self) -> Result<()> {
        if self.path.exists() {
            std::fs::remove_file(&self.path)?;
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    fn write_wal_header(&self, file: &mut File) -> Result<()> {
        let mut header = [0u8; WAL_HEADER_SIZE];

        header[0..4].copy_from_slice(&WAL_MAGIC.to_le_bytes());
        header[4..8].copy_from_slice(&WAL_VERSION.to_le_bytes());
        header[8..12].copy_from_slice(&self.page_size.to_le_bytes());
        header[12..16].copy_from_slice(&(*self.sequence.lock() as u32).to_le_bytes());
        header[16..20].copy_from_slice(&self.salt1.load(std::sync::atomic::Ordering::Relaxed).to_le_bytes());
        header[20..24].copy_from_slice(&self.salt2.load(std::sync::atomic::Ordering::Relaxed).to_le_bytes());

        // Checksum over first 24 bytes
        let checksum = crc32fast::hash(&header[..24]) as u64;
        header[24..32].copy_from_slice(&checksum.to_le_bytes());

        file.seek(SeekFrom::Start(0))?;
        file.write_all(&header)?;
        file.flush()?;

        Ok(())
    }

    /// Read the WAL header and return (salt1, salt2). Used by recovery to preserve checksum context.
    fn read_wal_header(&self, file: &mut File) -> Result<(u32, u32)> {
        file.seek(SeekFrom::Start(0))?;
        let mut header = [0u8; WAL_HEADER_SIZE];
        file.read_exact(&mut header)?;

        let magic = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
        if magic != WAL_MAGIC {
            return Err(StorageError::WalCorrupted(format!("Invalid WAL magic: 0x{:08X}", magic)));
        }
        let version = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
        if version != WAL_VERSION {
            return Err(StorageError::WalCorrupted(format!("Unsupported WAL version: {}", version)));
        }
        let s1 = u32::from_le_bytes([header[16], header[17], header[18], header[19]]);
        let s2 = u32::from_le_bytes([header[20], header[21], header[22], header[23]]);
        Ok((s1, s2))
    }

    fn validate_wal_header(&self, file: &mut File) -> Result<()> {
        self.read_wal_header(file).map(|_| ())
    }

    fn read_all_frames(&self, file: &mut File) -> Result<Vec<WalFrame>> {
        let file_len = file.metadata()?.len();

        // Start reading frames after the header
        let mut offset = WAL_HEADER_SIZE as u64;
        let mut frames = Vec::new();

        while offset + WAL_FRAME_HEADER_SIZE as u64 <= file_len {
            file.seek(SeekFrom::Start(offset))?;

            let mut frame_header = [0u8; WAL_FRAME_HEADER_SIZE];
            file.read_exact(&mut frame_header)?;

            let page_number =
                u32::from_le_bytes([frame_header[0], frame_header[1], frame_header[2], frame_header[3]]);
            let db_size_after = u64::from_le_bytes([
                frame_header[4], frame_header[5], frame_header[6], frame_header[7],
                frame_header[8], frame_header[9], frame_header[10], frame_header[11],
            ]);

            // Read stored checksums for validation
            let stored_cs1 = u32::from_le_bytes([frame_header[20],frame_header[21],frame_header[22],frame_header[23]]);
            let stored_cs2 = u32::from_le_bytes([frame_header[24],frame_header[25],frame_header[26],frame_header[27]]);

            // Read page data
            let page_offset = offset + WAL_FRAME_HEADER_SIZE as u64;
            if page_offset + self.page_size as u64 > file_len {
                break; // Incomplete frame at end of file
            }

            file.seek(SeekFrom::Start(page_offset))?;
            let mut page_data = vec![0u8; self.page_size as usize];
            file.read_exact(&mut page_data)?;

            let s1 = self.salt1.load(std::sync::atomic::Ordering::Relaxed);
            let s2 = self.salt2.load(std::sync::atomic::Ordering::Relaxed);
            let (computed_cs1, computed_cs2) = Self::compute_frame_checksum(page_number, &page_data, s1, s2, 0);
            // Recompute with frame_seq=0 since append_frame uses the sequence counter
            // which is reset on checkpoint. Salts provide sufficient uniqueness.
            if computed_cs1 != stored_cs1 || computed_cs2 != stored_cs2 {
                log::warn!("WAL frame checksum mismatch at offset {}: page {}", offset, page_number);
                break; // Stop recovery here; data beyond this point may be corrupt
            }

            frames.push(WalFrame {
                page_number,
                db_size_after,
                page_data,
            });

            offset = page_offset + self.page_size as u64;
        }

        Ok(frames)
    }

    fn compute_frame_checksum(
        page_number: u32,
        page_data: &[u8],
        salt1: u32,
        salt2: u32,
        frame_seq: u32,
    ) -> (u32, u32) {
        let mut hasher1 = crc32fast::Hasher::new();
        let mut hasher2 = crc32fast::Hasher::new();

        // Checksum 1: page_number + salt1 + frame_seq + page data
        hasher1.update(&page_number.to_le_bytes());
        hasher1.update(&salt1.to_le_bytes());
        hasher1.update(&frame_seq.to_le_bytes());
        hasher1.update(page_data);

        // Checksum 2: page_number + salt2 + frame_seq + page data
        hasher2.update(&page_number.to_le_bytes());
        hasher2.update(&salt2.to_le_bytes());
        hasher2.update(&frame_seq.to_le_bytes());
        hasher2.update(page_data);

        (hasher1.finalize(), hasher2.finalize())
    }
}

/// Get the WAL file path for a given database path.
fn wal_path_from_db(db_path: &Path) -> PathBuf {
    let mut wal_name = db_path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();
    wal_name.push_str("-wal");
    db_path.with_file_name(wal_name)
}

impl Drop for WalManager {
    fn drop(&mut self) {
        // Don't delete the WAL on drop — it's needed for recovery.
        // A clean shutdown should call checkpoint() and remove() explicitly.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wal_create_and_open() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.embeddb");

        let wal = WalManager::new(&db_path, 4096);
        wal.open().unwrap();
        assert!(wal.exists());
    }

    #[test]
    fn test_wal_append_and_recover() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.embeddb");

        // Create a database file with a page worth of data
        let page_data = vec![0xABu8; 4096];
        {
            let mut file = File::create(&db_path).unwrap();
            file.write_all(&page_data).unwrap();
            file.flush().unwrap();
        }

        // Append a modified page to WAL
        {
            let wal = WalManager::new(&db_path, 4096);
            wal.open().unwrap();

            let modified_page = vec![0xCDu8; 4096];
            wal.append_frame(0, &modified_page, 1).unwrap();
            // Don't checkpoint — simulate crash
        }

        // Recover: the modified page should be applied
        {
            let wal = WalManager::new(&db_path, 4096);
            let recovered = wal.recover(&db_path).unwrap();
            assert_eq!(recovered, 1);

            // Verify the page was written
            let recovered_data = std::fs::read(&db_path).unwrap();
            assert_eq!(recovered_data[..4096], vec![0xCDu8; 4096][..]);
        }
    }

    #[test]
    fn test_wal_checkpoint() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.embeddb");

        // Create initial DB
        {
            let mut file = File::create(&db_path).unwrap();
            file.write_all(&vec![0u8; 4096]).unwrap();
        }

        let wal = WalManager::new(&db_path, 4096);
        wal.open().unwrap();

        let modified = vec![0x42u8; 4096];
        wal.append_frame(0, &modified, 1).unwrap();

        // Checkpoint
        wal.checkpoint(&db_path).unwrap();

        // Verify
        let data = std::fs::read(&db_path).unwrap();
        assert_eq!(data[..4096], modified[..]);
    }
}
