//! Mock system implementation for testing.

#![expect(
    clippy::module_name_repetitions,
    reason = "MockSystem is clearer than just Mock in the system module"
)]
#![expect(
    clippy::std_instead_of_alloc,
    reason = "I couldn't find that trait in the alloc crate"
)]
#![expect(
    clippy::std_instead_of_core,
    reason = "I couldn't find that trait in the core crate"
)]

use tracing::error;

use crate::{FileMetadata, System, TempDirHandle, WalkEntry};
use std::collections::{BTreeMap, BTreeSet};
use std::env::VarError;
use std::io::{self, Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::SystemTime;

/// Global counter for generating unique temp directory IDs.
static TEMP_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

/// In-memory implementation of System trait for testing.
///
/// `MockSystem` provides an in-memory filesystem and environment,
/// perfect for fast, isolated unit tests without side effects.
///
/// # Example
/// ```
/// use os_shim::{System, mock::MockSystem};
/// use std::path::Path;
///
/// let system = MockSystem::new()
///     .with_env("HOME", "/home/user").unwrap()
///     .with_file("/test/file.txt", b"Hello, world!").unwrap()
///     .with_dir("/test/subdir").unwrap();
///
/// assert_eq!(system.env_var("HOME").unwrap(), "/home/user");
/// assert!(system.exists(Path::new("/test/file.txt")).unwrap());
/// ```
#[derive(Clone)]
pub struct MockSystem {
    /// Shared mutable state protected by a read-write lock.
    state: Arc<RwLock<MockSystemState>>,
}

/// In-memory state backing the mock filesystem and environment.
struct MockSystemState {
    /// Current working directory path.
    current_dir: PathBuf,
    /// Path returned by `current_exe()`.
    current_exe: PathBuf,
    /// Set of directories that exist in the mock filesystem.
    dirs: BTreeSet<PathBuf>,
    /// Map of environment variable names to values.
    env_vars: BTreeMap<String, String>,
    /// Map of file paths to their byte contents.
    files: BTreeMap<PathBuf, Vec<u8>>,
    /// Per-file last-modification timestamps.
    modified: BTreeMap<PathBuf, SystemTime>,
    /// Set of PIDs considered alive (for `is_pid_alive` mock).
    #[cfg(feature = "process")]
    pids_alive: BTreeSet<u32>,
}

impl MockSystem {
    /// Ensure all ancestor directories exist for a given path.
    #[inline]
    fn ensure_parent_dirs(dirs: &mut BTreeSet<PathBuf>, path: &Path) {
        let mut ancestors = Vec::new();
        let mut current = path;

        // Collect all ancestors
        while let Some(parent) = current.parent() {
            ancestors.push(parent.to_path_buf());
            current = parent;
            if parent == Path::new("") || parent == Path::new("/") {
                break;
            }
        }

        // Insert all ancestors and the path itself
        for ancestor in ancestors {
            dirs.insert(ancestor);
        }
        dirs.insert(path.to_path_buf());
    }

    /// Create a new `MockSystem` with default state.
    #[must_use]
    #[inline]
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(MockSystemState {
                current_dir: PathBuf::from("/"),
                current_exe: PathBuf::from("/mock/exe"),
                dirs: BTreeSet::from([PathBuf::from("/")]),
                env_vars: BTreeMap::new(),
                files: BTreeMap::new(),
                modified: BTreeMap::new(),
                #[cfg(feature = "process")]
                pids_alive: BTreeSet::new(),
            })),
        }
    }

    /// Set the current working directory (builder pattern).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The internal lock cannot be acquired
    #[inline]
    pub fn with_current_dir<P: AsRef<Path>>(self, dir: P) -> io::Result<Self> {
        let mut state = self
            .state
            .write()
            .map_err(|err| io::Error::other(err.to_string()))?;
        state.current_dir = dir.as_ref().to_path_buf();
        drop(state);
        Ok(self)
    }

    /// Set the path returned by `current_exe()` (builder pattern).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The internal lock cannot be acquired
    #[inline]
    pub fn with_current_exe<P: AsRef<Path>>(self, path: P) -> io::Result<Self> {
        let mut state = self
            .state
            .write()
            .map_err(|err| io::Error::other(err.to_string()))?;
        state.current_exe = path.as_ref().to_path_buf();
        drop(state);
        Ok(self)
    }

    /// Add a directory (builder pattern).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The internal lock cannot be acquired
    #[inline]
    pub fn with_dir<P: AsRef<Path>>(self, path: P) -> io::Result<Self> {
        let path_buf = path.as_ref().to_path_buf();
        let mut state = self
            .state
            .write()
            .map_err(|err| io::Error::other(err.to_string()))?;
        Self::ensure_parent_dirs(&mut state.dirs, &path_buf);
        state.dirs.insert(path_buf);
        drop(state);
        Ok(self)
    }

    /// Set an environment variable (builder pattern).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The internal lock cannot be acquired
    #[inline]
    pub fn with_env(self, key: &str, value: &str) -> io::Result<Self> {
        let mut state = self
            .state
            .write()
            .map_err(|err| io::Error::other(err.to_string()))?;
        state.env_vars.insert(key.to_owned(), value.to_owned());
        drop(state);
        Ok(self)
    }

    /// Add a file with contents (builder pattern).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The internal lock cannot be acquired
    #[inline]
    pub fn with_file<P: AsRef<Path>>(self, path: P, contents: &[u8]) -> io::Result<Self> {
        let path_buf = path.as_ref().to_path_buf();
        let mut state = self
            .state
            .write()
            .map_err(|err| io::Error::other(err.to_string()))?;

        // Ensure parent directories exist
        if let Some(parent) = path_buf.parent() {
            Self::ensure_parent_dirs(&mut state.dirs, parent);
        }

        state.modified.insert(path_buf.clone(), SystemTime::now());
        state.files.insert(path_buf, contents.to_vec());
        drop(state);
        Ok(self)
    }

    /// Add a PID to the set of alive processes (builder pattern).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The internal lock cannot be acquired
    #[cfg(feature = "process")]
    #[inline]
    pub fn with_pid(self, pid: u32) -> io::Result<Self> {
        let mut state = self
            .state
            .write()
            .map_err(|err| io::Error::other(err.to_string()))?;
        state.pids_alive.insert(pid);
        drop(state);
        Ok(self)
    }
}

impl Default for MockSystem {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl System for MockSystem {
    #[inline]
    fn canonicalize(&self, path: &Path) -> io::Result<PathBuf> {
        // For mock, just return absolute path
        if path.is_absolute() {
            Ok(path.to_path_buf())
        } else {
            let current = self.current_dir()?;
            Ok(current.join(path))
        }
    }

    #[inline]
    #[expect(clippy::as_conversions, reason = "This is for usize to u64 conversion")]
    fn copy(&self, from: &Path, to: &Path) -> io::Result<u64> {
        let contents = {
            let state = self
                .state
                .read()
                .map_err(|err| io::Error::other(err.to_string()))?;
            state
                .files
                .get(from)
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::NotFound,
                        format!("Source file not found: {}", from.display()),
                    )
                })?
                .clone()
        };

        let size = contents.len() as u64;

        // Write to destination
        self.write(to, &contents)?;
        Ok(size)
    }

    #[inline]
    fn create(&self, path: &Path) -> io::Result<Box<dyn Write + '_>> {
        // For create, we need a writer that updates the mock filesystem
        // We'll use a custom writer that captures bytes
        Ok(Box::new(MockWriter {
            buffer: Vec::new(),
            path: path.to_path_buf(),
            system: self.clone(),
        }))
    }

    #[inline]
    fn create_dir_all(&self, path: &Path) -> io::Result<()> {
        let mut state = self
            .state
            .write()
            .map_err(|err| io::Error::other(err.to_string()))?;
        Self::ensure_parent_dirs(&mut state.dirs, path);
        drop(state);
        Ok(())
    }

    #[inline]
    fn create_temp_dir(&self) -> io::Result<Box<dyn TempDirHandle>> {
        // Generate unique temp directory ID
        let id = TEMP_DIR_COUNTER.fetch_add(1, Ordering::SeqCst);
        let temp_path = PathBuf::from(format!("/tmp/mock_{id}"));

        // Create the directory in the mock filesystem
        self.create_dir_all(&temp_path)?;

        Ok(Box::new(MockTempDir {
            path: temp_path,
            system: self.clone(),
        }))
    }

    #[inline]
    fn current_dir(&self) -> io::Result<PathBuf> {
        let state = self
            .state
            .read()
            .map_err(|err| io::Error::other(err.to_string()))?;
        Ok(state.current_dir.clone())
    }

    #[inline]
    fn current_exe(&self) -> io::Result<PathBuf> {
        let state = self
            .state
            .read()
            .map_err(|err| io::Error::other(err.to_string()))?;
        Ok(state.current_exe.clone())
    }

    #[inline]
    #[expect(clippy::map_err_ignore, reason = "This is for VarError")]
    fn env_var(&self, key: &str) -> Result<String, VarError> {
        let state = self.state.read().map_err(|_| VarError::NotPresent)?;
        state.env_vars.get(key).cloned().ok_or(VarError::NotPresent)
    }

    #[inline]
    fn exists(&self, path: &Path) -> io::Result<bool> {
        let state = self
            .state
            .read()
            .map_err(|err| io::Error::other(err.to_string()))?;
        Ok(state.files.contains_key(path) || state.dirs.contains(path))
    }

    #[inline]
    fn is_dir(&self, path: &Path) -> io::Result<bool> {
        let state = self
            .state
            .read()
            .map_err(|err| io::Error::other(err.to_string()))?;
        Ok(state.dirs.contains(path))
    }

    #[inline]
    fn is_file(&self, path: &Path) -> io::Result<bool> {
        let state = self
            .state
            .read()
            .map_err(|err| io::Error::other(err.to_string()))?;
        Ok(state.files.contains_key(path))
    }

    #[cfg(feature = "process")]
    #[inline]
    fn is_pid_alive(&self, pid: u32) -> bool {
        #[expect(
            clippy::expect_used,
            reason = "Lock poisoning in mock is unrecoverable"
        )]
        let state = self
            .state
            .read()
            .expect("MockSystem lock poisoned in is_pid_alive");
        state.pids_alive.contains(&pid)
    }

    #[inline]
    #[expect(
        clippy::as_conversions,
        reason = "usize to u64 conversion for file length"
    )]
    fn metadata(&self, path: &Path) -> io::Result<FileMetadata> {
        let state = self
            .state
            .read()
            .map_err(|err| io::Error::other(err.to_string()))?;

        let is_dir = state.dirs.contains(path);
        let is_file = state.files.contains_key(path);

        if !is_dir && !is_file {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("Path not found: {}", path.display()),
            ));
        }

        let len = state.files.get(path).map_or(0, |bytes| bytes.len() as u64);
        let modified = state
            .modified
            .get(path)
            .copied()
            .unwrap_or(SystemTime::UNIX_EPOCH);
        drop(state);

        Ok(FileMetadata {
            is_dir,
            is_file,
            len,
            modified,
        })
    }

    #[inline]
    fn open(&self, path: &Path) -> io::Result<Box<dyn Read + '_>> {
        let state = self
            .state
            .read()
            .map_err(|err| io::Error::other(err.to_string()))?;
        let bytes = state.files.get(path).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("File not found: {}", path.display()),
            )
        })?;
        let result = bytes.clone();
        drop(state);
        Ok(Box::new(Cursor::new(result)))
    }

    #[inline]
    fn open_append(&self, path: &Path) -> io::Result<Box<dyn Write + '_>> {
        Ok(Box::new(MockAppendWriter {
            buffer: Vec::new(),
            path: path.to_path_buf(),
            system: self.clone(),
        }))
    }

    #[inline]
    fn read_dir(&self, path: &Path) -> io::Result<Vec<PathBuf>> {
        let state = self
            .state
            .read()
            .map_err(|err| io::Error::other(err.to_string()))?;

        if !state.dirs.contains(path) {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("Directory not found: {}", path.display()),
            ));
        }

        let mut entries = Vec::new();

        // Find all direct children (files and directories)
        for file_path in state.files.keys() {
            if let Some(parent) = file_path.parent()
                && parent == path
            {
                entries.push(file_path.clone());
            }
        }

        for dir_path in &state.dirs {
            if let Some(parent) = dir_path.parent()
                && parent == path
                && dir_path != path
            {
                entries.push(dir_path.clone());
            }
        }

        drop(state);

        Ok(entries)
    }

    #[inline]
    fn read_to_string(&self, path: &Path) -> io::Result<String> {
        let state = self
            .state
            .read()
            .map_err(|err| io::Error::other(err.to_string()))?;
        let bytes = state.files.get(path).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("File not found: {}", path.display()),
            )
        })?;
        let result = bytes.clone();
        drop(state);
        String::from_utf8(result).map_err(|err| {
            io::Error::new(io::ErrorKind::InvalidData, format!("Invalid UTF-8: {err}"))
        })
    }

    #[inline]
    fn remove_dir_all(&self, path: &Path) -> io::Result<()> {
        let mut state = self
            .state
            .write()
            .map_err(|err| io::Error::other(err.to_string()))?;

        // Remove the directory
        state.dirs.remove(path);

        // Remove all files, subdirectories, and timestamps under this path
        state
            .files
            .retain(|file_path, _| !file_path.starts_with(path));
        state
            .modified
            .retain(|file_path, _| !file_path.starts_with(path));
        state
            .dirs
            .retain(|dir_path| !dir_path.starts_with(path) || dir_path == path);
        drop(state);
        Ok(())
    }

    #[inline]
    fn remove_file(&self, path: &Path) -> io::Result<()> {
        let mut state = self
            .state
            .write()
            .map_err(|err| io::Error::other(err.to_string()))?;

        if !state.files.contains_key(path) {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("File not found: {}", path.display()),
            ));
        }

        state.files.remove(path);
        state.modified.remove(path);
        drop(state);
        Ok(())
    }

    #[inline]
    fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        let mut state = self
            .state
            .write()
            .map_err(|err| io::Error::other(err.to_string()))?;

        // Handle file rename
        if let Some(contents) = state.files.remove(from) {
            // Ensure parent directories for destination exist
            if let Some(parent) = to.parent()
                && !state.dirs.contains(parent)
            {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("Parent directory does not exist: {}", parent.display()),
                ));
            }
            state.files.insert(to.to_path_buf(), contents);
            state.modified.remove(from);
            state.modified.insert(to.to_path_buf(), SystemTime::now());
            return Ok(());
        }

        // Handle directory rename
        if state.dirs.remove(from) {
            state.dirs.insert(to.to_path_buf());

            // Move all files and subdirectories under the old path
            let files_to_move: Vec<(PathBuf, Vec<u8>)> = state
                .files
                .iter()
                .filter(|&(path, _)| path.starts_with(from))
                .map(|(path, contents)| (path.clone(), contents.clone()))
                .collect();

            for (old_path, contents) in files_to_move {
                state.files.remove(&old_path);
                state.modified.remove(&old_path);
                let relative = old_path
                    .strip_prefix(from)
                    .map_err(|err| io::Error::other(err.to_string()))?;
                let new_path = to.join(relative);
                state.modified.insert(new_path.clone(), SystemTime::now());
                state.files.insert(new_path, contents);
            }

            let dirs_to_move: Vec<PathBuf> = state
                .dirs
                .iter()
                .filter(|path| path.starts_with(from))
                .cloned()
                .collect();

            for old_dir in dirs_to_move {
                state.dirs.remove(&old_dir);
                let relative = old_dir
                    .strip_prefix(from)
                    .map_err(|err| io::Error::other(err.to_string()))?;
                state.dirs.insert(to.join(relative));
            }

            drop(state);
            return Ok(());
        }

        drop(state);
        Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("Path not found: {}", from.display()),
        ))
    }

    #[inline]
    fn set_env_var(&self, key: &str, value: &str) {
        #[expect(
            clippy::expect_used,
            reason = "Lock poisoning in mock is unrecoverable"
        )]
        let mut state = self
            .state
            .write()
            .expect("MockSystem lock poisoned in set_env_var");
        state.env_vars.insert(key.to_owned(), value.to_owned());
    }

    #[inline]
    fn walk_dir(
        &self,
        path: &Path,
        _follow_links: bool,
        _hidden: bool,
    ) -> io::Result<Vec<WalkEntry>> {
        let state = self
            .state
            .read()
            .map_err(|err| io::Error::other(err.to_string()))?;

        if !state.dirs.contains(path) {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("Directory not found: {}", path.display()),
            ));
        }

        let mut entries = Vec::new();
        let mut to_visit: Vec<PathBuf> = vec![path.to_path_buf()];
        let mut visited: BTreeSet<PathBuf> = BTreeSet::new();

        while let Some(current) = to_visit.pop() {
            if !visited.insert(current.clone()) {
                continue; // Already visited
            }

            // Skip the root directory itself
            if current != path {
                entries.push(WalkEntry {
                    is_dir: state.dirs.contains(&current),
                    is_file: state.files.contains_key(&current),
                    path: current.clone(),
                });
            }

            // Find all direct children of current directory
            for dir in &state.dirs {
                if let Some(parent) = dir.parent()
                    && parent == current
                    && dir != &current
                {
                    to_visit.push(dir.clone());
                }
            }

            for file_path in state.files.keys() {
                if let Some(parent) = file_path.parent()
                    && parent == current
                {
                    entries.push(WalkEntry {
                        is_dir: false,
                        is_file: true,
                        path: file_path.clone(),
                    });
                }
            }
        }

        // Sort entries by path for deterministic output
        entries.sort_by(|left, right| left.path.cmp(&right.path));

        Ok(entries)
    }

    #[inline]
    fn write(&self, path: &Path, contents: &[u8]) -> io::Result<()> {
        let mut state = self
            .state
            .write()
            .map_err(|err| io::Error::other(err.to_string()))?;

        // Ensure parent directories exist
        if let Some(parent) = path.parent()
            && !state.dirs.contains(parent)
        {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("Parent directory does not exist: {}", parent.display()),
            ));
        }

        state.modified.insert(path.to_path_buf(), SystemTime::now());
        state.files.insert(path.to_path_buf(), contents.to_vec());
        drop(state);
        Ok(())
    }
}

/// Custom writer for `MockSystem` that writes to in-memory filesystem.
struct MockWriter {
    /// Accumulated bytes waiting to be flushed.
    buffer: Vec<u8>,
    /// Target file path in the mock filesystem.
    path: PathBuf,
    /// Reference to the parent mock system for writing.
    system: MockSystem,
}

#[expect(
    clippy::missing_trait_methods,
    reason = "Only implementing what I need"
)]
impl Write for MockWriter {
    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        self.system.write(&self.path, &self.buffer)?;
        Ok(())
    }

    #[inline]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.buffer.extend_from_slice(buf);
        Ok(buf.len())
    }
}

impl Drop for MockWriter {
    #[inline]
    fn drop(&mut self) {
        match self.flush() {
            Ok(()) => (),
            Err(err) => error!("Failed to flush mock writer: {err}"),
        }
    }
}

/// Append writer for `MockSystem` that extends existing file contents on flush.
struct MockAppendWriter {
    /// Accumulated bytes waiting to be appended.
    buffer: Vec<u8>,
    /// Target file path in the mock filesystem.
    path: PathBuf,
    /// Reference to the parent mock system for writing.
    system: MockSystem,
}

#[expect(
    clippy::missing_trait_methods,
    reason = "Only implementing what I need"
)]
impl Write for MockAppendWriter {
    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        let mut state = self
            .system
            .state
            .write()
            .map_err(|err| io::Error::other(err.to_string()))?;

        // Ensure parent directory exists
        if let Some(parent) = self.path.parent()
            && !state.dirs.contains(parent)
        {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("Parent directory does not exist: {}", parent.display()),
            ));
        }

        let entry = state
            .files
            .entry(self.path.clone())
            .or_insert_with(Vec::new);
        entry.extend_from_slice(&self.buffer);
        self.buffer.clear();
        state.modified.insert(self.path.clone(), SystemTime::now());
        drop(state);
        Ok(())
    }

    #[inline]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.buffer.extend_from_slice(buf);
        Ok(buf.len())
    }
}

impl Drop for MockAppendWriter {
    #[inline]
    fn drop(&mut self) {
        match self.flush() {
            Ok(()) => (),
            Err(err) => error!("Failed to flush mock append writer: {err}"),
        }
    }
}

/// Mock temporary directory handle that cleans up on drop.
#[non_exhaustive]
pub struct MockTempDir {
    /// Path to the temporary directory in the mock filesystem.
    path: PathBuf,
    /// Reference to the parent mock system for cleanup.
    system: MockSystem,
}

impl TempDirHandle for MockTempDir {
    #[inline]
    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for MockTempDir {
    #[inline]
    fn drop(&mut self) {
        // Remove the temporary directory from the mock filesystem when dropped
        match self.system.remove_dir_all(&self.path) {
            Ok(()) => (),
            Err(err) => error!("Failed to remove temporary directory: {err}"),
        }
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "Tests use unwrap for brevity; panics indicate test failure"
)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn metadata_returns_correct_len_for_file() {
        let system = MockSystem::new()
            .with_file("/test/file.txt", b"hello")
            .unwrap();

        let meta = system.metadata(Path::new("/test/file.txt")).unwrap();
        assert!(meta.is_file);
        assert!(!meta.is_dir);
        assert_eq!(meta.len, 5);
    }

    #[test]
    fn metadata_returns_dir_info() {
        let system = MockSystem::new().with_dir("/mydir").unwrap();

        let meta = system.metadata(Path::new("/mydir")).unwrap();
        assert!(meta.is_dir);
        assert!(!meta.is_file);
        assert_eq!(meta.len, 0);
    }

    #[test]
    fn metadata_not_found() {
        let system = MockSystem::new();
        let result = system.metadata(Path::new("/missing"));
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::NotFound);
    }

    #[test]
    fn metadata_modified_updates_on_write() {
        let system = MockSystem::new().with_dir("/test").unwrap();

        system.write(Path::new("/test/file.txt"), b"first").unwrap();
        let mtime1 = system
            .metadata(Path::new("/test/file.txt"))
            .unwrap()
            .modified;

        // Write again to update the modification time
        system
            .write(Path::new("/test/file.txt"), b"second")
            .unwrap();
        let mtime2 = system
            .metadata(Path::new("/test/file.txt"))
            .unwrap()
            .modified;

        assert!(mtime2 >= mtime1);
    }

    #[test]
    fn rename_file_moves_content() {
        let system = MockSystem::new()
            .with_file("/test/a.txt", b"content")
            .unwrap();

        system
            .rename(Path::new("/test/a.txt"), Path::new("/test/b.txt"))
            .unwrap();

        assert!(!system.exists(Path::new("/test/a.txt")).unwrap());
        assert!(system.exists(Path::new("/test/b.txt")).unwrap());
        assert_eq!(
            system.read_to_string(Path::new("/test/b.txt")).unwrap(),
            "content"
        );
    }

    #[test]
    fn rename_nonexistent_returns_error() {
        let system = MockSystem::new();
        let result = system.rename(Path::new("/missing"), Path::new("/other"));
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::NotFound);
    }

    #[test]
    fn open_append_creates_new_file() {
        let system = MockSystem::new().with_dir("/test").unwrap();

        let mut writer = system.open_append(Path::new("/test/new.txt")).unwrap();
        writer.write_all(b"hello").unwrap();
        writer.flush().unwrap();
        drop(writer);

        assert_eq!(
            system.read_to_string(Path::new("/test/new.txt")).unwrap(),
            "hello"
        );
    }

    #[test]
    fn open_append_extends_existing() {
        let system = MockSystem::new()
            .with_file("/test/file.txt", b"hello")
            .unwrap();

        let mut writer = system.open_append(Path::new("/test/file.txt")).unwrap();
        writer.write_all(b" world").unwrap();
        writer.flush().unwrap();
        drop(writer);

        assert_eq!(
            system.read_to_string(Path::new("/test/file.txt")).unwrap(),
            "hello world"
        );
    }

    #[test]
    fn current_exe_returns_configured_path() {
        let system = MockSystem::new()
            .with_current_exe("/usr/bin/vigil")
            .unwrap();

        assert_eq!(
            system.current_exe().unwrap(),
            PathBuf::from("/usr/bin/vigil")
        );
    }

    #[test]
    fn current_exe_returns_default() {
        let system = MockSystem::new();
        assert_eq!(system.current_exe().unwrap(), PathBuf::from("/mock/exe"));
    }

    #[test]
    fn set_env_var_updates_env() {
        let system = MockSystem::new();
        system.set_env_var("MY_KEY", "my_value");
        assert_eq!(system.env_var("MY_KEY").unwrap(), "my_value");
    }

    #[cfg(feature = "process")]
    #[test]
    fn is_pid_alive_checks_mock_pids() {
        let system = MockSystem::new().with_pid(1234).unwrap();

        assert!(system.is_pid_alive(1234));
        assert!(!system.is_pid_alive(9999));
    }
}
