//! In-memory system implementation for testing.

#![expect(
    clippy::std_instead_of_alloc,
    reason = "I couldn't find that trait in the alloc crate"
)]
#![expect(
    clippy::std_instead_of_core,
    reason = "I couldn't find that trait in the core crate"
)]

use tracing::error;

#[cfg(feature = "zip")]
use crate::archive::{load_zip, load_zip_seekable};
use crate::{FileMetadata, System, TempDirHandle, WalkEntry, WriteMode};
use std::collections::{BTreeMap, BTreeSet};
use std::env::VarError;
#[cfg(feature = "zip")]
use std::io::Seek;
use std::io::{self, Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, PoisonError, RwLock};
use std::time::{Duration, SystemTime};

/// Global counter for generating unique temp directory IDs.
static TEMP_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

/// In-memory implementation of System trait for testing.
///
/// `MemorySystem` provides an in-memory filesystem and environment,
/// perfect for fast, isolated unit tests without side effects.
///
/// # Example
/// ```
/// use os_shim::{System, mock::MemorySystem};
/// use std::path::Path;
///
/// let system = MemorySystem::new()
///     .with_env("HOME", "/home/user").unwrap()
///     .with_file("/test/file.txt", b"Hello, world!").unwrap()
///     .with_dir("/test/subdir").unwrap();
///
/// assert_eq!(system.env_var("HOME").unwrap(), "/home/user");
/// assert!(system.exists(Path::new("/test/file.txt")).unwrap());
/// ```
#[derive(Clone)]
pub struct MemorySystem {
    /// Shared mutable state protected by a read-write lock.
    state: Arc<RwLock<MemorySystemState>>,
}

/// In-memory state backing the mock filesystem and environment.
struct MemorySystemState {
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
    /// The time `now()` gives.
    now: SystemTime,
    /// Set of PIDs considered alive (for `is_pid_alive` mock).
    #[cfg(feature = "process")]
    pids_alive: BTreeSet<u32>,
}

impl MemorySystem {
    /// Moves the time `now()` gives forward by `by`.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The internal lock cannot be acquired
    /// - The time would overflow
    #[inline]
    pub fn advance(&self, by: Duration) -> io::Result<()> {
        let mut state = self
            .state
            .write()
            .map_err(|err| io::Error::other(err.to_string()))?;
        state.now = state.now.checked_add(by).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "Advancing the clock overflows SystemTime",
            )
        })?;
        drop(state);
        Ok(())
    }

    /// Append `chunk` to the file at `path`, creating it if this is the first
    /// write, without copying the whole file to do it.
    ///
    /// `MemoryWriter` uses this instead of buffering an entire file and handing
    /// it to `write`: that held the contents twice, once in the writer and once
    /// on the way into the map, which is a gigabyte of waste per gigabyte
    /// written.
    fn append_chunk(&self, path: &Path, chunk: &[u8], truncate: bool) -> io::Result<()> {
        let mut state = self
            .state
            .write()
            .map_err(|err| io::Error::other(err.to_string()))?;
        if let Some(parent) = path.parent()
            && !state.dirs.contains(parent)
        {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("Parent directory does not exist: {}", parent.display()),
            ));
        }
        let now = state.now;
        state.modified.insert(path.to_path_buf(), now);
        let slot = state.files.entry(path.to_path_buf()).or_default();
        if truncate {
            slot.clear();
        }
        slot.extend_from_slice(chunk);
        drop(state);
        Ok(())
    }

    /// Fail with `NotADirectory` when a proper ancestor of `path` is a file, as
    /// resolving `path` on disk does.
    fn check_no_file_above(files: &BTreeMap<PathBuf, Vec<u8>>, path: &Path) -> io::Result<()> {
        path.ancestors()
            .skip(1)
            .find(|ancestor| files.contains_key(*ancestor))
            .map_or(Ok(()), |file| {
                Err(io::Error::new(
                    io::ErrorKind::NotADirectory,
                    format!("Not a directory: {}", file.display()),
                ))
            })
    }

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

    /// Build a `MemorySystem` whose filesystem is the contents of a zip archive read
    /// from a seekable source.
    ///
    /// Entries land under `/`, as with [`from_zip_stream`](Self::from_zip_stream).
    /// They are found through the archive's central directory, so an archive whose
    /// entries carry data descriptors loads too.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The archive is malformed, or an entry cannot be written.
    #[cfg(feature = "zip")]
    #[inline]
    pub fn from_zip_reader<R>(source: R) -> io::Result<Self>
    where
        R: Read + Seek,
    {
        let system = Self::new();
        load_zip_seekable(&system, source, Path::new("/"))?;
        Ok(system)
    }

    /// Build a `MemorySystem` whose filesystem is the contents of a zip archive.
    ///
    /// Entries land under `/`, so archiving a folder and loading it back gives an
    /// in-memory filesystem holding that folder's contents at the root. The reader
    /// need not be seekable. An archive whose entries carry data descriptors is
    /// refused; [`from_zip_reader`](Self::from_zip_reader) reads it.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The archive is malformed, or an entry cannot be written.
    #[cfg(feature = "zip")]
    #[inline]
    pub fn from_zip_stream(source: &mut dyn Read) -> io::Result<Self> {
        let system = Self::new();
        load_zip(&system, source, Path::new("/"))?;
        Ok(system)
    }

    /// Create a new `MemorySystem` with default state.
    #[must_use]
    #[inline]
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(MemorySystemState {
                current_dir: PathBuf::from("/"),
                current_exe: PathBuf::from("/mock/exe"),
                dirs: BTreeSet::from([PathBuf::from("/")]),
                env_vars: BTreeMap::new(),
                files: BTreeMap::new(),
                modified: BTreeMap::new(),
                now: SystemTime::UNIX_EPOCH,
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
    pub fn with_current_dir<P>(self, dir: P) -> io::Result<Self>
    where
        P: AsRef<Path>,
    {
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
    pub fn with_current_exe<P>(self, path: P) -> io::Result<Self>
    where
        P: AsRef<Path>,
    {
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
    pub fn with_dir<P>(self, path: P) -> io::Result<Self>
    where
        P: AsRef<Path>,
    {
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
    pub fn with_file<P>(self, path: P, contents: &[u8]) -> io::Result<Self>
    where
        P: AsRef<Path>,
    {
        let path_buf = path.as_ref().to_path_buf();
        let mut state = self
            .state
            .write()
            .map_err(|err| io::Error::other(err.to_string()))?;

        // Ensure parent directories exist
        if let Some(parent) = path_buf.parent() {
            Self::ensure_parent_dirs(&mut state.dirs, parent);
        }

        let now = state.now;
        state.modified.insert(path_buf.clone(), now);
        state.files.insert(path_buf, contents.to_vec());
        drop(state);
        Ok(self)
    }

    /// The time `now()` gives (builder pattern).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The internal lock cannot be acquired
    #[inline]
    pub fn with_now(self, time: SystemTime) -> io::Result<Self> {
        let mut state = self
            .state
            .write()
            .map_err(|err| io::Error::other(err.to_string()))?;
        state.now = time;
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

impl Default for MemorySystem {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl System for MemorySystem {
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
            Self::check_no_file_above(&state.files, from)?;
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
        self.write(to, &contents, WriteMode::Replace)?;
        Ok(size)
    }

    #[inline]
    fn create(&self, path: &Path) -> io::Result<Box<dyn Write + '_>> {
        let state = self
            .state
            .read()
            .map_err(|err| io::Error::other(err.to_string()))?;
        Self::check_no_file_above(&state.files, path)?;
        if state.dirs.contains(path) {
            return Err(io::Error::new(
                io::ErrorKind::IsADirectory,
                format!("Is a directory: {}", path.display()),
            ));
        }
        drop(state);

        // For create, we need a writer that updates the mock filesystem
        // We'll use a custom writer that captures bytes
        Ok(Box::new(MemoryWriter {
            path: path.to_path_buf(),
            started: false,
            system: self.clone(),
        }))
    }

    #[inline]
    fn create_dir_all(&self, path: &Path) -> io::Result<()> {
        let mut state = self
            .state
            .write()
            .map_err(|err| io::Error::other(err.to_string()))?;
        if state.files.contains_key(path) {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("File exists: {}", path.display()),
            ));
        }
        Self::check_no_file_above(&state.files, path)?;
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

        Ok(Box::new(MemoryTempDir {
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
            .expect("MemorySystem lock poisoned in is_pid_alive");
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
        Self::check_no_file_above(&state.files, path)?;

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
    fn now(&self) -> SystemTime {
        self.state
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .now
    }

    #[inline]
    fn open(&self, path: &Path) -> io::Result<Box<dyn Read + '_>> {
        let state = self
            .state
            .read()
            .map_err(|err| io::Error::other(err.to_string()))?;
        Self::check_no_file_above(&state.files, path)?;
        if state.dirs.contains(path) {
            return Ok(Box::new(FolderReader));
        }
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
        let state = self
            .state
            .read()
            .map_err(|err| io::Error::other(err.to_string()))?;
        Self::check_no_file_above(&state.files, path)?;
        if state.dirs.contains(path) {
            return Err(io::Error::new(
                io::ErrorKind::IsADirectory,
                format!("Is a directory: {}", path.display()),
            ));
        }
        drop(state);

        Ok(Box::new(MemoryAppendWriter {
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
        Self::check_no_file_above(&state.files, path)?;

        if state.files.contains_key(path) {
            return Err(io::Error::new(
                io::ErrorKind::NotADirectory,
                format!("Not a directory: {}", path.display()),
            ));
        }

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
        Self::check_no_file_above(&state.files, path)?;
        if state.dirs.contains(path) {
            return Err(io::Error::new(
                io::ErrorKind::IsADirectory,
                format!("Is a directory: {}", path.display()),
            ));
        }
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
        Self::check_no_file_above(&state.files, path)?;

        if state.files.contains_key(path) {
            return Err(io::Error::new(
                io::ErrorKind::NotADirectory,
                format!("Not a directory: {}", path.display()),
            ));
        }

        // Match RealSystem (fs::remove_dir_all): a missing path is NotFound.
        if !state.dirs.remove(path) {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("Directory not found: {}", path.display()),
            ));
        }

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
        Self::check_no_file_above(&state.files, path)?;

        if state.dirs.contains(path) {
            return Err(io::Error::new(
                io::ErrorKind::IsADirectory,
                format!("Is a directory: {}", path.display()),
            ));
        }

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
        Self::check_no_file_above(&state.files, from)?;
        Self::check_no_file_above(&state.files, to)?;

        if state.files.contains_key(from) && state.dirs.contains(to) {
            return Err(io::Error::new(
                io::ErrorKind::IsADirectory,
                format!("Is a directory: {}", to.display()),
            ));
        }
        if state.dirs.contains(from) {
            if state.files.contains_key(to) {
                return Err(io::Error::new(
                    io::ErrorKind::NotADirectory,
                    format!("Not a directory: {}", to.display()),
                ));
            }
            if let Some(parent) = to.parent()
                && !state.dirs.contains(parent)
            {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("Parent directory does not exist: {}", parent.display()),
                ));
            }
        }

        // Handle file rename
        if state.files.contains_key(from) {
            // Ensure parent directories for destination exist
            if let Some(parent) = to.parent()
                && !state.dirs.contains(parent)
            {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("Parent directory does not exist: {}", parent.display()),
                ));
            }
            let contents = state.files.remove(from).unwrap_or_default();
            state.files.insert(to.to_path_buf(), contents);
            state.modified.remove(from);
            let now = state.now;
            state.modified.insert(to.to_path_buf(), now);
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

            let now = state.now;
            for (old_path, contents) in files_to_move {
                state.files.remove(&old_path);
                state.modified.remove(&old_path);
                let relative = old_path
                    .strip_prefix(from)
                    .map_err(|err| io::Error::other(err.to_string()))?;
                let new_path = to.join(relative);
                state.modified.insert(new_path.clone(), now);
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
            .expect("MemorySystem lock poisoned in set_env_var");
        state.env_vars.insert(key.to_owned(), value.to_owned());
    }

    #[inline]
    fn walk_dir(
        &self,
        path: &Path,
        // `follow_links` is a faithful no-op: neither the mock filesystem nor the
        // System trait models symlinks, so a symlink-free tree walks identically
        // for both flag values -- which is exactly what RealSystem does when no
        // symlinks are present. Kept to match the trait signature.
        _follow_links: bool,
        hidden: bool,
    ) -> io::Result<Vec<WalkEntry>> {
        let state = self
            .state
            .read()
            .map_err(|err| io::Error::other(err.to_string()))?;

        // Match RealSystem (ignore::WalkBuilder): walking a missing path yields no
        // entries rather than erroring.
        if !state.dirs.contains(path) {
            return Ok(Vec::new());
        }

        let mut entries = Vec::new();
        let mut to_visit: Vec<PathBuf> = vec![path.to_path_buf()];
        let mut visited: BTreeSet<PathBuf> = BTreeSet::new();

        while let Some(current) = to_visit.pop() {
            // Cycle guard: unreachable in practice since the mock has no symlinks,
            // so the directory graph is always a tree. Kept as a safety net.
            if !visited.insert(current.clone()) {
                continue;
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

        // Honor the trait's `hidden` semantics: `true` includes hidden entries,
        // `false` excludes any entry living under a dot-prefixed path component.
        if !hidden {
            entries.retain(|entry| {
                entry.path.strip_prefix(path).is_ok_and(|rel| {
                    !rel.components()
                        .any(|component| component.as_os_str().to_string_lossy().starts_with('.'))
                })
            });
        }

        // Sort entries by path for deterministic output
        entries.sort_by(|left, right| left.path.cmp(&right.path));

        Ok(entries)
    }

    #[inline]
    fn write(&self, path: &Path, contents: &[u8], mode: WriteMode) -> io::Result<()> {
        let mut state = self
            .state
            .write()
            .map_err(|err| io::Error::other(err.to_string()))?;
        Self::check_no_file_above(&state.files, path)?;

        // Ensure parent directories exist
        if let Some(parent) = path.parent()
            && !state.dirs.contains(parent)
        {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("Parent directory does not exist: {}", parent.display()),
            ));
        }

        if mode == WriteMode::CreateNew
            && (state.files.contains_key(path) || state.dirs.contains(path))
        {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("File exists: {}", path.display()),
            ));
        }

        if state.dirs.contains(path) {
            return Err(io::Error::new(
                io::ErrorKind::IsADirectory,
                format!("Is a directory: {}", path.display()),
            ));
        }

        let now = state.now;
        state.modified.insert(path.to_path_buf(), now);
        state.files.insert(path.to_path_buf(), contents.to_vec());
        drop(state);
        Ok(())
    }
}

/// What `open` returns for a folder: every read fails, as reading a folder does
/// on disk.
struct FolderReader;

impl Read for FolderReader {
    #[inline]
    fn read(&mut self, _buf: &mut [u8]) -> io::Result<usize> {
        Err(io::Error::new(
            io::ErrorKind::IsADirectory,
            "Is a directory",
        ))
    }
}

/// Custom writer for `MemorySystem` that writes to in-memory filesystem.
struct MemoryWriter {
    /// Target file path in the mock filesystem.
    path: PathBuf,
    /// Whether any chunk has reached the filesystem yet, so the first one
    /// truncates and later ones append.
    started: bool,
    /// Reference to the parent mock system for writing.
    system: MemorySystem,
}

#[expect(
    clippy::missing_trait_methods,
    reason = "Only implementing what I need"
)]
impl Write for MemoryWriter {
    /// Each chunk is already in the filesystem, so there is nothing held back.
    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        if self.started {
            return Ok(());
        }
        self.started = true;
        self.system.append_chunk(&self.path, &[], true)
    }

    #[inline]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.system.append_chunk(&self.path, buf, !self.started)?;
        self.started = true;
        Ok(buf.len())
    }
}

impl Drop for MemoryWriter {
    #[inline]
    fn drop(&mut self) {
        match self.flush() {
            Ok(()) => (),
            Err(err) => error!("Failed to flush mock writer: {err}"),
        }
    }
}

/// Append writer for `MemorySystem` that extends existing file contents on flush.
struct MemoryAppendWriter {
    /// Accumulated bytes waiting to be appended.
    buffer: Vec<u8>,
    /// Target file path in the mock filesystem.
    path: PathBuf,
    /// Reference to the parent mock system for writing.
    system: MemorySystem,
}

#[expect(
    clippy::missing_trait_methods,
    reason = "Only implementing what I need"
)]
impl Write for MemoryAppendWriter {
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
        let now = state.now;
        state.modified.insert(self.path.clone(), now);
        drop(state);
        Ok(())
    }

    #[inline]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.buffer.extend_from_slice(buf);
        Ok(buf.len())
    }
}

impl Drop for MemoryAppendWriter {
    #[inline]
    fn drop(&mut self) {
        match self.flush() {
            Ok(()) => (),
            Err(err) => error!("Failed to flush mock append writer: {err}"),
        }
    }
}

/// In-memory temporary directory handle that cleans up on drop.
#[non_exhaustive]
pub struct MemoryTempDir {
    /// Path to the temporary directory in the mock filesystem.
    path: PathBuf,
    /// Reference to the parent mock system for cleanup.
    system: MemorySystem,
}

impl TempDirHandle for MemoryTempDir {
    #[inline]
    fn path(&self) -> &Path {
        &self.path
    }

    #[cfg(feature = "zip")]
    #[inline]
    fn to_zip_stream(&self) -> io::Result<Box<dyn Read>> {
        self.system.to_zip_stream(&self.path)
    }
}

impl Drop for MemoryTempDir {
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
mod tests;
