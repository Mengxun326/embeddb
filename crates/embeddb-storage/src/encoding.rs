//! Binary encoding and decoding helpers for page data.
//!
//! Provides utilities for reading and writing typed values to/from
//! byte buffers, used when serializing cell data within pages.

use bytemuck::Pod;

/// Write a value to a byte buffer at the given offset.
/// Returns the number of bytes written.
pub fn write_value<T: Pod>(buf: &mut [u8], offset: usize, value: &T) -> usize {
    let bytes = bytemuck::bytes_of(value);
    let end = offset + bytes.len();
    if end <= buf.len() {
        buf[offset..end].copy_from_slice(bytes);
    }
    bytes.len()
}

/// Read a value from a byte buffer at the given offset.
pub fn read_value<T: Pod>(buf: &[u8], offset: usize) -> Option<&T> {
    let size = std::mem::size_of::<T>();
    if offset + size > buf.len() {
        return None;
    }
    Some(bytemuck::from_bytes(&buf[offset..offset + size]))
}

/// Write a u32 in little-endian byte order.
pub fn write_u32_le(buf: &mut [u8], offset: usize, value: u32) {
    buf[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

/// Read a u32 in little-endian byte order.
pub fn read_u32_le(buf: &[u8], offset: usize) -> Option<u32> {
    if offset + 4 > buf.len() {
        return None;
    }
    Some(u32::from_le_bytes([
        buf[offset],
        buf[offset + 1],
        buf[offset + 2],
        buf[offset + 3],
    ]))
}

/// Write a u64 in little-endian byte order.
pub fn write_u64_le(buf: &mut [u8], offset: usize, value: u64) {
    buf[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

/// Read a u64 in little-endian byte order.
pub fn read_u64_le(buf: &[u8], offset: usize) -> Option<u64> {
    if offset + 8 > buf.len() {
        return None;
    }
    Some(u64::from_le_bytes([
        buf[offset],
        buf[offset + 1],
        buf[offset + 2],
        buf[offset + 3],
        buf[offset + 4],
        buf[offset + 5],
        buf[offset + 6],
        buf[offset + 7],
    ]))
}

/// Write an f32 array to a buffer as raw bytes.
pub fn write_f32_slice(buf: &mut [u8], offset: usize, values: &[f32]) -> usize {
    let bytes = bytemuck::cast_slice(values);
    let end = offset + bytes.len();
    if end <= buf.len() {
        buf[offset..end].copy_from_slice(bytes);
    }
    bytes.len()
}

/// Read an f32 slice from a buffer.
pub fn read_f32_slice(buf: &[u8], offset: usize, count: usize) -> &[f32] {
    let byte_len = count * std::mem::size_of::<f32>();
    let end = offset + byte_len;
    if end > buf.len() {
        return &[];
    }
    bytemuck::cast_slice(&buf[offset..end])
}

/// Compute how many f32 values can fit in `available` bytes.
pub fn f32_count_from_bytes(available: usize) -> usize {
    available / std::mem::size_of::<f32>()
}

/// Compute bytes needed for `count` f32 values.
pub fn bytes_for_f32(count: usize) -> usize {
    count * std::mem::size_of::<f32>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_read_u32() {
        let mut buf = [0u8; 16];
        write_u32_le(&mut buf, 0, 0xDEAD_BEEF);
        assert_eq!(read_u32_le(&buf, 0), Some(0xDEAD_BEEF));
    }

    #[test]
    fn test_write_read_u64() {
        let mut buf = [0u8; 16];
        write_u64_le(&mut buf, 0, 0x1234_5678_9ABC_DEF0);
        assert_eq!(read_u64_le(&buf, 0), Some(0x1234_5678_9ABC_DEF0));
    }

    #[test]
    fn test_write_read_f32_slice() {
        let mut buf = [0u8; 64];
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        write_f32_slice(&mut buf, 0, &values);
        let read: Vec<f32> = read_f32_slice(&buf, 0, 4).to_vec();
        assert_eq!(read, values);
    }

    #[test]
    fn test_bytes_for_f32() {
        assert_eq!(bytes_for_f32(100), 400);
        assert_eq!(f32_count_from_bytes(400), 100);
    }
}
