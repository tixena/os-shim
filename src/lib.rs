//! Mockable abstraction layer over OS operations for testable Rust code.
//!
//! This crate provides a unified [`System`] trait for all external system interactions,
//! allowing for easy testing with mock implementations.
//!
//! # Implementations
//!
//! - [`RealSystem`](real::RealSystem): Production implementation using `std::env` and `std::fs`.
//! - [`MemorySystem`](mock::MemorySystem): In-memory implementation for fast, isolated unit tests.
//!
//! # Example
//!
//! ```
//! use os_shim::{System, mock::MemorySystem};
//! use std::path::Path;
//!
//! let system = MemorySystem::new()
//!     .with_env("HOME", "/home/user").unwrap()
//!     .with_file("/test/file.txt", b"Hello, world!").unwrap()
//!     .with_dir("/test/subdir").unwrap();
//!
//! assert_eq!(system.env_var("HOME").unwrap(), "/home/user");
//! assert!(system.exists(Path::new("/test/file.txt")).unwrap());
//! ```

#![expect(
    clippy::module_name_repetitions,
    reason = "trait name System is the canonical name for this abstraction"
)]

extern crate alloc;

#[cfg(feature = "zip")]
pub mod archive;
#[cfg(any(feature = "minio", feature = "s3"))]
pub mod bucket;
pub mod mock;
pub mod real;

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "Tests use unwrap for brevity; panics indicate test failure"
)]
mod parity_tests;

#[cfg(feature = "async")]
use core::pin::Pin;
use core::time::Duration;
use std::env::VarError;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// Entry from directory walking.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct WalkEntry {
    /// Whether the entry is a directory.
    pub is_dir: bool,
    /// Whether the entry is a file.
    pub is_file: bool,
    /// The path of the entry.
    pub path: PathBuf,
}

/// File metadata -- os-shim's own type, fully constructable by `MemorySystem`.
///
/// Replaces `std::fs::Metadata` which is opaque and cannot be
/// meaningfully constructed in mock implementations.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct FileMetadata {
    /// Whether the path is a directory.
    pub is_dir: bool,
    /// Whether the path is a regular file.
    pub is_file: bool,
    /// File size in bytes (0 for directories).
    pub len: u64,
    /// Last modification time.
    pub modified: SystemTime,
}

/// How `System::write` treats a file already at the path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum WriteMode {
    /// Writes the file only when nothing is at the path: `AlreadyExists`
    /// otherwise. The check and the creation are one step; the contents are
    /// written after it, so a reader can see the new file empty for a moment.
    CreateNew,
    /// Writes the file, replacing one already at the path.
    Replace,
}

/// Temporary directory handle that cleans up on drop.
///
/// This trait provides a system-agnostic way to create temporary directories
/// that are automatically cleaned up when the handle is dropped.
///
/// For `RealSystem`, this wraps `tempfile::TempDir` and uses real filesystem.
/// For `MemorySystem`, this manages an in-memory temporary directory.
pub trait TempDirHandle {
    /// Get the path to the temporary directory.
    fn path(&self) -> &Path;

    /// A pull-based zip stream of this directory's contents.
    ///
    /// The default body reports `Unsupported`, so an out-of-crate implementor
    /// keeps compiling; `RealSystem` and `MemorySystem` both override it.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The implementation does not support archiving itself.
    /// - The directory cannot be read.
    #[cfg(feature = "zip")]
    #[inline]
    fn to_zip_stream(&self) -> io::Result<Box<dyn Read>> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "this temporary directory handle cannot archive itself",
        ))
    }
}

/// Unified trait for system operations (environment + filesystem).
///
/// This trait abstracts all interactions with the operating system,
/// including environment variables and filesystem operations.
///
/// # Implementations
/// - `RealSystem`: Production implementation using `std::env` and `std::fs`.
/// - `MemorySystem`: Test implementation using in-memory storage.
pub trait System: Send + Sync {
    // ==================== Environment Operations ====================

    /// Canonicalize a path (resolve to absolute path).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The path cannot be canonicalized.
    fn canonicalize(&self, path: &Path) -> io::Result<PathBuf>;

    /// Copy a file from source to destination.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be copied.
    fn copy(&self, from: &Path, to: &Path) -> io::Result<u64>;

    /// Create a file for writing (returns a writable stream).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be created.
    fn create(&self, path: &Path) -> io::Result<Box<dyn Write + '_>>;

    /// Recursively create a directory and all parent directories.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The directory cannot be created.
    fn create_dir_all(&self, path: &Path) -> io::Result<()>;

    /// Create a temporary directory that is automatically cleaned up on drop.
    ///
    /// # Returns
    /// A handle to the temporary directory. The directory will be removed when
    /// the handle is dropped.
    ///
    /// # Note
    /// For `RealSystem`, this uses `tempfile::TempDir` on the real filesystem.
    /// For `MemorySystem`, this creates an in-memory temporary directory.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The temporary directory cannot be created.
    fn create_temp_dir(&self) -> io::Result<Box<dyn TempDirHandle>>;

    /// Get the current working directory.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The current working directory cannot be retrieved.
    fn current_dir(&self) -> io::Result<PathBuf>;

    /// Get the path to the current executable.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The executable path cannot be determined.
    fn current_exe(&self) -> io::Result<PathBuf>;

    /// Get an environment variable.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The environment variable cannot be retrieved.
    fn env_var(&self, key: &str) -> Result<String, VarError>;

    /// Check if a path exists.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The path cannot be checked.
    fn exists(&self, path: &Path) -> io::Result<bool>;

    /// Check if a path points to a directory.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The path cannot be checked.
    fn is_dir(&self, path: &Path) -> io::Result<bool>;

    /// Check if a path points to a file.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The path cannot be checked.
    fn is_file(&self, path: &Path) -> io::Result<bool>;

    /// Check if a process with the given PID is alive.
    #[cfg(feature = "process")]
    fn is_pid_alive(&self, pid: u32) -> bool;

    /// Get metadata for a path.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The path cannot be read.
    fn metadata(&self, path: &Path) -> io::Result<FileMetadata>;

    /// The current time: the machine's in `RealSystem`, the one a test set in
    /// `MemorySystem`.
    fn now(&self) -> SystemTime;

    /// Open a file for reading (returns a readable stream).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be opened.
    fn open(&self, path: &Path) -> io::Result<Box<dyn Read + '_>>;

    /// Open a file for appending (create if it does not exist).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be opened for appending.
    fn open_append(&self, path: &Path) -> io::Result<Box<dyn Write + '_>>;

    /// Read directory entries, returning paths of all entries.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The directory cannot be read.
    fn read_dir(&self, path: &Path) -> io::Result<Vec<PathBuf>>;

    /// Read entire file contents as a string.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be read.
    fn read_to_string(&self, path: &Path) -> io::Result<String>;

    /// Remove a directory and all its contents.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The directory cannot be removed.
    fn remove_dir_all(&self, path: &Path) -> io::Result<()>;

    /// Remove a file.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be removed.
    fn remove_file(&self, path: &Path) -> io::Result<()>;

    /// Removes the file at `path` only when its contents are `contents`:
    /// `Ok(true)` when it removed it, `Ok(false)` when the file holds other
    /// contents or another file took its place, and it stays.
    ///
    /// Two callers that remove the same file through this method never both
    /// remove it, and neither removes a file that another made at the path in
    /// the meantime, as long as files are made there only with
    /// `WriteMode::CreateNew`.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Nothing is at the path (`NotFound`), it is a folder (`IsADirectory`),
    ///   or a file is above it (`NotADirectory`).
    /// - The file cannot be read or removed.
    fn remove_file_if(&self, path: &Path, contents: &[u8]) -> io::Result<bool>;

    /// Rename or move a file or directory.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The source path does not exist.
    /// - The rename operation fails.
    fn rename(&self, from: &Path, to: &Path) -> io::Result<()>;

    /// Set an environment variable.
    fn set_env_var(&self, key: &str, value: &str);

    /// Waits `duration`: on the machine in `RealSystem`; in `MemorySystem`,
    /// it moves the clock `now` gives forward by `duration` and returns at once.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `MemorySystem`'s clock would overflow, or its state lock is poisoned.
    fn sleep(&self, duration: Duration) -> io::Result<()>;

    /// Waits `duration` without holding the thread: on tokio's timer in
    /// `RealSystem`; in `MemorySystem`, it moves the clock forward by
    /// `duration` when awaited.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `MemorySystem`'s clock would overflow, or its state lock is poisoned.
    ///
    /// # Panics
    ///
    /// In `RealSystem`, when awaited outside a tokio runtime with its timer on
    /// (`Builder::enable_time` or `Builder::enable_all`).
    #[cfg(feature = "async")]
    fn sleep_async(
        &self,
        duration: Duration,
    ) -> Pin<Box<dyn Future<Output = io::Result<()>> + Send + '_>>;

    /// A pull-based zip stream of `source`, held in memory.
    ///
    /// The stream is a reader over a finished archive, not a pipe fed while the
    /// archive is built: the first byte is available only once the last one is.
    /// Provided method: implementors get it for free.
    ///
    /// # Memory
    ///
    /// Peak usage is roughly twice the archive's size, since the archive is
    /// built in one buffer and handed to the reader in another. See
    /// [`to_zip_writer`](Self::to_zip_writer) for the ceiling that follows.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The source cannot be read.
    #[cfg(feature = "zip")]
    #[inline]
    fn to_zip_stream(&self, source: &Path) -> io::Result<Box<dyn Read>> {
        Ok(Box::new(io::Cursor::new(self.to_zip_vec(source)?)))
    }

    /// The zip archive of `source` as bytes.
    ///
    /// Provided method: implementors get it for free.
    ///
    /// # Memory
    ///
    /// Peak usage is roughly twice the archive's size: the returned `Vec` plus
    /// the buffer it was built in. See [`to_zip_writer`](Self::to_zip_writer)
    /// for the ceiling that follows from it.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The source cannot be read.
    #[cfg(feature = "zip")]
    #[inline]
    fn to_zip_vec(&self, source: &Path) -> io::Result<Vec<u8>> {
        let mut bytes = Vec::new();
        self.to_zip_writer(source, &mut bytes)?;
        Ok(bytes)
    }

    /// Write `source` into `sink` as a zip archive.
    ///
    /// A directory contributes its *contents*: archiving `A/` yields `file.txt`,
    /// not `A/file.txt`. Provided method: implementors get it for free.
    ///
    /// # Memory
    ///
    /// The whole archive is built in memory before any of it reaches `sink`, so
    /// peak usage is the archive's size plus a constant under 3 MiB. That is the
    /// compressed size: a tree of source text at roughly 5.5 to 1 costs far less
    /// than its bytes on disk, while already-compressed content costs slightly
    /// more than it. Keep archives under about 1 GiB unless the caller controls
    /// the machine — above that a small container or build runner is killed by
    /// the operating system rather than given an error it can handle.
    ///
    /// Buffering is not an implementation detail that can be tuned away. An
    /// archive written incrementally records each entry's size after the entry
    /// instead of in its header, and one shaped that way cannot be read back by
    /// [`MemorySystem::from_zip_stream`](crate::mock::MemorySystem::from_zip_stream).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The source cannot be read, or the sink cannot be written.
    #[cfg(feature = "zip")]
    #[inline]
    fn to_zip_writer(&self, source: &Path, sink: &mut dyn Write) -> io::Result<()> {
        archive::to_zip_writer(self, source, sink, archive::ZipOptions::default())
    }

    /// Recursively walk a directory, returning all entries.
    ///
    /// # Arguments
    /// * `path` - Root path to start walking from.
    /// * `follow_links` - Whether to follow symbolic links.
    /// * `hidden` - Whether to include hidden (dot-prefixed) entries. `true`
    ///   includes them; `false` excludes any entry under a dot-prefixed path
    ///   component.
    ///
    /// # Returns
    /// Vector of all entries found (files and directories), excluding the root itself.
    ///
    /// # Note
    /// For `RealSystem`, this respects .gitignore files using the `ignore` crate.
    /// For `MemorySystem`, this walks the in-memory filesystem.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The directory cannot be walked.
    fn walk_dir(&self, path: &Path, follow_links: bool, hidden: bool)
    -> io::Result<Vec<WalkEntry>>;

    /// Write bytes to a file, as `mode` says for a file already at the path.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be written.
    /// - `mode` is `WriteMode::CreateNew` and something is at the path
    ///   (`io::ErrorKind::AlreadyExists`).
    fn write(&self, path: &Path, contents: &[u8], mode: WriteMode) -> io::Result<()>;
}
