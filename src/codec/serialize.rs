//! Versioned binary file format with integrity checks.
//!
//! ## File Format Versions
//!
//! ### v1 (legacy): payload-only checksum
//! - magic[4] + version[4] + flags[4] + checksum[8] + payload_size[8] = 28 bytes
//! - Risk: attacker can modify header fields without detection
//!
//! ### v2 (deprecated): BROKEN — DO NOT USE
//!
//! ### v3 (current): fixed header layout
//! - magic[4] + version[4] + flags[4] + checksum_type[4] + header_crc[4] + checksum[8] + payload_size[8] = 36 bytes
//! - header_crc: CRC32 of (magic || version || flags || checksum_type || payload_size)
//! - checksum: CRC32 of payload
//! - Protection: both header fields and payload are integrity-checked

use std::io::{Read, Write};

use crate::codec::CodecError;
use crate::error::Result as DocResult;

/// Maximum allowed payload size to prevent OOM attacks.
pub const MAX_PAYLOAD_SIZE: usize = 256 * 1024 * 1024;

/// Magic constants for file type detection
pub const MAGIC_DELTA: [u8; 4] = *b"DELT";
pub const MAGIC_ZONE: [u8; 4] = *b"ZONE";
pub const MAGIC_BLOM: [u8; 4] = *b"BLOM";
pub const MAGIC_MASK: [u8; 4] = *b"MASK";
pub const MAGIC_GRAN: [u8; 4] = *b"GRAN";
pub const MAGIC_UPD: [u8; 4] = *b"UPDM";

/// Header sizes for each format version
pub const HEADER_SIZE_V1: usize = 28;
pub const HEADER_SIZE_V2: usize = 36; // DEPRECATED
pub const HEADER_SIZE_V3: usize = 36;
pub const HEADER_SIZE: usize = HEADER_SIZE_V3;

/// Checksum type constants
pub const CHECKSUM_PAYLOAD_ONLY: u32 = 0;
pub const CHECKSUM_HEADER_PLUS_PAYLOAD: u32 = 1;

/// Flag indicating extended checksum format
pub const FLAG_EXTENDED_CHECKSUM: u32 = 0x00000001;

/// File header for v1/v2/v3 formats.
///
/// ## v1 Format (28 bytes)
/// - magic[4] + version[4] + flags[4] + checksum[8] + payload_size[8]
///
/// ## v2 Format (36 bytes, DEPRECATED — BROKEN)
///
/// ## v3 Format (36 bytes)
/// - magic[4] + version[4] + flags[4] + checksum_type[4] + header_crc[4] + checksum[8] + payload_size[8]
///
/// WARNING: This struct uses manual byte layout via `as_bytes`/`from_bytes`.
/// The struct field order is optimized for the v3 layout, NOT `#[repr(C)]`.
#[derive(Debug, Clone, Copy)]
pub struct FileHeader {
    pub magic: [u8; 4],
    pub version: u32,
    pub flags: u32,
    pub checksum_type: u32,
    pub header_crc: u32,
    pub checksum: u64,
    pub payload_size: u64,
}

impl FileHeader {
    /// Current format version
    pub const CURRENT_VERSION: u32 = 3;

    /// Check if this header uses the extended checksum format (v3).
    pub fn is_extended_format(&self) -> bool {
        self.version >= 3 || (self.version == 2 && (self.flags & FLAG_EXTENDED_CHECKSUM) != 0)
    }

    pub fn validate_magic(&self, expected: [u8; 4]) -> Result<(), CodecError> {
        if self.magic != expected {
            return Err(CodecError::Decode(format!(
                "magic mismatch: expected {:4?}, got {:4?}",
                expected, self.magic
            )));
        }
        Ok(())
    }

    pub fn validate_version(&self) -> Result<(), CodecError> {
        if self.version > Self::CURRENT_VERSION {
            return Err(CodecError::Decode(format!(
                "unsupported version: {} (max: {})",
                self.version,
                Self::CURRENT_VERSION
            )));
        }
        if self.version == 2 {
            return Err(CodecError::Decode(
                "v2 format is deprecated and broken — upgrade to v3".into(),
            ));
        }
        Ok(())
    }

    /// Validate the header checksum (v3) if applicable.
    pub fn validate_header_checksum(&self) -> Result<(), CodecError> {
        if self.version == 1 {
            return Ok(());
        }
        if self.version == 2 {
            return Err(CodecError::Decode(
                "v2 format is deprecated and broken — upgrade to v3".into(),
            ));
        }

        let header_bytes = self.header_only_bytes_v3();
        let computed = crc32fast::hash(&header_bytes);
        if computed != self.header_crc {
            return Err(CodecError::Decode(format!(
                "header checksum mismatch: expected {:08x}, computed {:08x}",
                self.header_crc, computed
            )));
        }
        Ok(())
    }

    /// Compute CRC32 of the v3 header fields (for header_crc).
    /// Covers: magic + version + flags + checksum_type + payload_size = 24 bytes
    fn header_only_bytes_v3(&self) -> [u8; 24] {
        let mut buf = [0u8; 24];
        buf[0..4].copy_from_slice(&self.magic);
        buf[4..8].copy_from_slice(&self.version.to_le_bytes());
        buf[8..12].copy_from_slice(&self.flags.to_le_bytes());
        buf[12..16].copy_from_slice(&self.checksum_type.to_le_bytes());
        buf[16..24].copy_from_slice(&self.payload_size.to_le_bytes());
        buf
    }

    /// Compute CRC32 of the v3 header fields.
    pub fn compute_header_crc(&self) -> u32 {
        crc32fast::hash(&self.header_only_bytes_v3())
    }

    pub fn validate_checksum(&self, payload: &[u8]) -> Result<(), CodecError> {
        let computed = crc32fast::hash(payload) as u64;
        if computed != self.checksum {
            return Err(CodecError::Decode(format!(
                "checksum mismatch: expected {:016x}, computed {:016x}",
                self.checksum, computed
            )));
        }
        Ok(())
    }

    pub fn validate(&self, expected_magic: [u8; 4], payload: &[u8]) -> Result<(), CodecError> {
        self.validate_magic(expected_magic)?;
        self.validate_version()?;
        self.validate_header_checksum()?;
        self.validate_checksum(payload)?;
        Ok(())
    }

    /// Create a v1 FileHeader (legacy format, payload-only checksum).
    pub fn new_v1(magic: [u8; 4], payload_size: u64, checksum: u64) -> Self {
        Self {
            magic,
            version: 1,
            flags: 0,
            checksum_type: CHECKSUM_PAYLOAD_ONLY,
            header_crc: 0,
            checksum,
            payload_size,
        }
    }

    /// Create a v3 FileHeader with extended header integrity.
    pub fn new_v3(magic: [u8; 4], payload_size: u64, payload_checksum: u64) -> Self {
        let mut header = Self {
            magic,
            version: Self::CURRENT_VERSION,
            flags: FLAG_EXTENDED_CHECKSUM,
            checksum_type: CHECKSUM_HEADER_PLUS_PAYLOAD,
            header_crc: 0,
            checksum: payload_checksum,
            payload_size,
        };
        header.header_crc = header.compute_header_crc();
        header
    }

    pub fn write_to_file(&self, path: &std::path::Path, payload: &[u8]) -> DocResult<usize> {
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;
        file.write_all(&self.as_bytes())?;
        file.write_all(payload)?;
        file.flush()?;
        file.sync_all()?;
        Ok(self.as_bytes().len() + payload.len())
    }

    /// Parse from bytes, supporting v1 and v3 formats.
    /// v2 is explicitly rejected as broken.
    pub fn from_bytes(slice: &[u8]) -> Result<Self, CodecError> {
        if slice.len() < 8 {
            return Err(CodecError::Decode("header too short".into()));
        }

        let version = u32::from_le_bytes(
            slice[4..8]
                .try_into()
                .map_err(|_| CodecError::Decode("header: version: insufficient bytes".into()))?,
        );

        if version == 2 {
            return Err(CodecError::Decode(
                "v2 format is deprecated and broken — upgrade to v3".into(),
            ));
        }

        if version == 1 {
            return Self::from_bytes_v1(slice);
        }

        // version >= 3: use v3 format
        Self::from_bytes_v3(slice)
    }

    /// Parse v1 format: 28 bytes (magic + version + flags + checksum + payload_size)
    fn from_bytes_v1(slice: &[u8]) -> Result<Self, CodecError> {
        if slice.len() < HEADER_SIZE_V1 {
            return Err(CodecError::Decode(format!(
                "v1 header too short: {} < {} bytes",
                slice.len(),
                HEADER_SIZE_V1
            )));
        }
        let magic = slice[0..4].try_into().expect("v1 header: slice must have 4 bytes for magic");
        let version = u32::from_le_bytes(slice[4..8].try_into().expect("v1 header: slice must have 4 bytes for version"));
        let flags = u32::from_le_bytes(slice[8..12].try_into().expect("v1 header: slice must have 4 bytes for flags"));
        let checksum = u64::from_le_bytes(slice[12..20].try_into().expect("v1 header: slice must have 8 bytes for checksum"));
        let payload_size = u64::from_le_bytes(slice[20..28].try_into().expect("v1 header: slice must have 8 bytes for payload_size"));
        Ok(Self {
            magic,
            version,
            flags,
            checksum_type: CHECKSUM_PAYLOAD_ONLY,
            header_crc: 0,
            checksum,
            payload_size,
        })
    }

    /// Parse v3 format: 36 bytes
    /// Layout: magic[4] + version[4] + flags[4] + checksum_type[4] + header_crc[4] + checksum[8] + payload_size[8]
    fn from_bytes_v3(slice: &[u8]) -> Result<Self, CodecError> {
        if slice.len() < HEADER_SIZE_V3 {
            return Err(CodecError::Decode(format!(
                "v3 header too short: {} < {} bytes",
                slice.len(),
                HEADER_SIZE_V3
            )));
        }
        let magic = slice[0..4].try_into().expect("v3 header: slice must have 4 bytes for magic");
        let version = u32::from_le_bytes(slice[4..8].try_into().expect("v3 header: slice must have 4 bytes for version"));
        let flags = u32::from_le_bytes(slice[8..12].try_into().expect("v3 header: slice must have 4 bytes for flags"));
        let checksum_type = u32::from_le_bytes(slice[12..16].try_into().expect("v3 header: slice must have 4 bytes for checksum_type"));
        let header_crc = u32::from_le_bytes(slice[16..20].try_into().expect("v3 header: slice must have 4 bytes for header_crc"));
        let checksum = u64::from_le_bytes(slice[20..28].try_into().expect("v3 header: slice must have 8 bytes for checksum"));
        let payload_size = u64::from_le_bytes(slice[28..36].try_into().expect("v3 header: slice must have 8 bytes for payload_size"));
        Ok(Self {
            magic,
            version,
            flags,
            checksum_type,
            header_crc,
            checksum,
            payload_size,
        })
    }

    /// Serialize to bytes.
    pub fn as_bytes(&self) -> Vec<u8> {
        if self.version == 1 {
            self.as_bytes_v1()
        } else {
            self.as_bytes_v3()
        }
    }

    /// Serialize to v1 format bytes (28 bytes).
    /// Layout: magic[4] + version[4] + flags[4] + checksum[8] + payload_size[8]
    fn as_bytes_v1(&self) -> Vec<u8> {
        let mut buf = vec![0u8; HEADER_SIZE_V1];
        buf[0..4].copy_from_slice(&self.magic);
        buf[4..8].copy_from_slice(&self.version.to_le_bytes());
        buf[8..12].copy_from_slice(&self.flags.to_le_bytes());
        buf[12..20].copy_from_slice(&self.checksum.to_le_bytes());
        buf[20..28].copy_from_slice(&self.payload_size.to_le_bytes());
        buf
    }

    /// Serialize to v3 format bytes (36 bytes).
    /// Layout: magic[4] + version[4] + flags[4] + checksum_type[4] + header_crc[4] + checksum[8] + payload_size[8]
    fn as_bytes_v3(&self) -> Vec<u8> {
        let mut buf = vec![0u8; HEADER_SIZE_V3];
        buf[0..4].copy_from_slice(&self.magic);
        buf[4..8].copy_from_slice(&self.version.to_le_bytes());
        buf[8..12].copy_from_slice(&self.flags.to_le_bytes());
        buf[12..16].copy_from_slice(&self.checksum_type.to_le_bytes());
        buf[16..20].copy_from_slice(&self.header_crc.to_le_bytes());
        buf[20..28].copy_from_slice(&self.checksum.to_le_bytes());
        buf[28..36].copy_from_slice(&self.payload_size.to_le_bytes());
        buf
    }
}

/// Write data with a versioned, checksummed header using postcard.
/// Uses v3 format by default with extended header integrity protection.
pub fn write_with_header<T: serde::Serialize>(
    path: &std::path::Path,
    magic: [u8; 4],
    data: &T,
) -> DocResult<usize> {
    let payload =
        postcard::to_allocvec(data).map_err(|e| crate::RockDuckError::Codec(e.to_string()))?;
    let checksum = crc32fast::hash(&payload) as u64;

    let header = FileHeader::new_v3(magic, payload.len() as u64, checksum);

    header.write_to_file(path, &payload)
}

/// Read a file with header validation (supports both v1 and v3 formats).
pub fn read_with_header(path: &std::path::Path, expected_magic: [u8; 4]) -> DocResult<Vec<u8>> {
    let mut file = std::fs::File::open(path)?;
    let file_size = file.metadata().map(|m| m.len() as usize)?;

    // First, peek at the version to determine header size
    // Read minimum bytes to detect version (magic + version = 8 bytes)
    let mut version_peek = [0u8; 8];
    file.read_exact(&mut version_peek)?;
    let version = u32::from_le_bytes(version_peek[4..8].try_into().unwrap());

    let header_size = if version == 1 {
        HEADER_SIZE_V1
    } else if version == 2 {
        return Err(crate::RockDuckError::Codec(
            "v2 format is deprecated and broken".into(),
        ));
    } else {
        HEADER_SIZE_V3
    };

    // Read the rest of the header
    let header_buf = if header_size > 8 {
        let mut rest = vec![0u8; header_size - 8];
        file.read_exact(&mut rest)?;
        let mut full = version_peek.to_vec();
        full.extend(rest);
        full
    } else {
        version_peek.to_vec()
    };

    let header = FileHeader::from_bytes(&header_buf)?;

    header.validate_magic(expected_magic)?;
    header.validate_version()?;
    header.validate_header_checksum()?;

    let payload_size = header.payload_size as usize;

    if payload_size > MAX_PAYLOAD_SIZE {
        return Err(crate::RockDuckError::Codec(format!(
            "payload_size {} exceeds maximum allowed {} bytes (possible OOM attack)",
            payload_size, MAX_PAYLOAD_SIZE
        )));
    }

    if file_size != header_size + payload_size {
        return Err(crate::RockDuckError::Codec(format!(
            "file size mismatch: header says {} bytes payload, but file has {} bytes total",
            payload_size,
            file_size - header_size
        )));
    }

    let mut payload = vec![0u8; payload_size];
    file.read_exact(&mut payload)?;
    header.validate_checksum(&payload)?;

    Ok(payload)
}

/// Read a file with header validation and deserialize.
pub fn read_postcard<T: serde::de::DeserializeOwned>(
    path: &std::path::Path,
    expected_magic: [u8; 4],
) -> DocResult<T> {
    let payload = read_with_header(path, expected_magic)?;
    postcard::from_bytes(&payload).map_err(|e| crate::RockDuckError::Codec(e.to_string()))
}

/// Read header from a mmap slice without deserializing.
pub fn read_header_from_mmap(mmap_slice: &[u8]) -> Result<FileHeader, CodecError> {
    FileHeader::from_bytes(mmap_slice)
}

/// Validate and extract payload from a mmap slice (supports v1 and v3 formats).
pub fn extract_payload_from_mmap(
    mmap_slice: &[u8],
    expected_magic: [u8; 4],
) -> Result<&[u8], CodecError> {
    let header = read_header_from_mmap(mmap_slice)?;
    header.validate_magic(expected_magic)?;
    header.validate_version()?;
    header.validate_header_checksum()?;

    let payload_size = header.payload_size as usize;

    if payload_size > MAX_PAYLOAD_SIZE {
        return Err(CodecError::Decode(format!(
            "payload_size {} exceeds maximum allowed {} bytes (possible OOM attack)",
            payload_size, MAX_PAYLOAD_SIZE
        )));
    }

    let header_size = if header.is_extended_format() {
        HEADER_SIZE_V3
    } else {
        HEADER_SIZE_V1
    };
    if mmap_slice.len() < header_size + payload_size {
        return Err(CodecError::Decode(format!(
            "mmap truncated: header says {} payload bytes, but mmap has {} bytes total",
            payload_size,
            mmap_slice.len() - header_size
        )));
    }

    let payload = &mmap_slice[header_size..header_size + payload_size];
    header.validate_checksum(payload)?;
    Ok(payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_v1_v3_backward_compatibility() {
        let payload = b"test data";
        let checksum = crc32fast::hash(payload) as u64;

        // Test that v1 headers can still be read
        let v1_header = FileHeader::new_v1(*b"TEST", payload.len() as u64, checksum);
        assert_eq!(v1_header.version, 1);
        assert!(!v1_header.is_extended_format());

        // Test that v3 headers have extended protection
        let v3_header = FileHeader::new_v3(*b"TEST", payload.len() as u64, checksum);
        assert_eq!(v3_header.version, 3);
        assert!(v3_header.is_extended_format());

        // Header CRC should be valid for v3
        assert!(v3_header.validate_header_checksum().is_ok());
    }

    #[test]
    fn test_v3_header_roundtrip() {
        let magic = *b"TEST";
        let payload = b"hello world";
        let checksum = crc32fast::hash(payload) as u64;

        let header = FileHeader::new_v3(magic, payload.len() as u64, checksum);
        let bytes = header.as_bytes();

        // v3 format should be 36 bytes
        assert_eq!(bytes.len(), HEADER_SIZE_V3);

        // Verify layout: magic[0..4], version[4..8], flags[8..12],
        // checksum_type[12..16], header_crc[16..20], checksum[20..28], payload_size[28..36]
        assert_eq!(&bytes[0..4], b"TEST");
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 3);
        assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 1); // FLAG_EXTENDED_CHECKSUM

        // Round-trip parse
        let parsed = FileHeader::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.version, 3);
        assert!(parsed.is_extended_format());
        assert_eq!(parsed.payload_size, payload.len() as u64);
        assert_eq!(parsed.checksum, checksum);
        assert_eq!(parsed.header_crc, header.header_crc);
    }

    #[test]
    fn test_v2_rejected() {
        // Simulate a v2 header (broken format)
        let mut bytes = vec![0u8; HEADER_SIZE_V2];
        bytes[0..4].copy_from_slice(b"TEST");
        bytes[4..8].copy_from_slice(&2u32.to_le_bytes()); // version = 2
        bytes[8..12].copy_from_slice(&1u32.to_le_bytes()); // flags

        let result = FileHeader::from_bytes(&bytes);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("v2"));
    }

    #[test]
    fn test_detects_tampered_header() {
        let magic = *b"TEST";
        let payload = b"test data";
        let checksum = crc32fast::hash(payload) as u64;

        let mut header = FileHeader::new_v3(magic, payload.len() as u64, checksum);

        // Validate original header passes
        assert!(header.validate_header_checksum().is_ok());

        // Tamper with a header byte
        header.flags ^= 0xFF;
        assert!(header.validate_header_checksum().is_err());
    }

    #[test]
    fn test_v3_file_roundtrip() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let magic = *b"TEST";
        let payload = b"hello world from v3";
        let checksum = crc32fast::hash(payload) as u64;

        let header = FileHeader::new_v3(magic, payload.len() as u64, checksum);
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&header.as_bytes()).unwrap();
        file.write_all(payload).unwrap();
        file.flush().unwrap();

        let read_payload = read_with_header(file.path(), magic).unwrap();
        assert_eq!(&read_payload, payload);
    }

    #[test]
    fn test_v1_file_roundtrip() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let magic = *b"TEST";
        let payload = b"hello world from v1";
        let checksum = crc32fast::hash(payload) as u64;

        let header = FileHeader::new_v1(magic, payload.len() as u64, checksum);
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&header.as_bytes()).unwrap();
        file.write_all(payload).unwrap();
        file.flush().unwrap();

        let read_payload = read_with_header(file.path(), magic).unwrap();
        assert_eq!(&read_payload, payload);
    }
}
