//! [`MemorySystem`]: a [`System`] that lives entirely in memory.
//!
//! Write your code against `&dyn System` (or a generic `S: System`). Use
//! [`RealSystem`](crate::real::RealSystem) in production and `MemorySystem` in
//! tests. The test then needs no disk, no temporary folders, no real
//! environment variables, no real clock and no real processes.
//!
//! # What it holds
//!
//! One `MemorySystem` holds files, folders, environment variables, a current
//! folder, a current executable path, a clock and (with the `process` feature)
//! a set of live process ids. A new one starts with:
//!
//! - one folder, `/`;
//! - current folder `/` and current executable `/mock/exe`;
//! - no files and no environment variables;
//! - the clock at [`SystemTime::UNIX_EPOCH`];
//! - no live process ids.
//!
//! `Clone` gives a second handle to the same state, not a copy. Each
//! `MemorySystem` made by [`new`](MemorySystem::new) has its own state, so
//! tests that run in parallel do not see each other.
//!
//! The ways it differs from the real machine are listed under
//! [Differences to know about](#differences-to-know-about).
//!
//! # Set it up
//!
//! The `with_*` builders each take `self` and return `io::Result<Self>`. They
//! fail only on a poisoned lock or on a clash such as a file where a folder
//! must go. Chain them with `?`:
//!
//! ```
//! use os_shim::{System, mock::MemorySystem};
//! use std::path::Path;
//!
//! let system = MemorySystem::new()
//!     .with_env("HOME", "/home/user")?
//!     .with_file("/home/user/.app.toml", b"name = \"demo\"")?
//!     .with_current_dir("/home/user")?;
//!
//! // `with_file` made the parent folders.
//! assert!(system.is_dir(Path::new("/home/user"))?);
//! // A relative path is joined onto the current folder.
//! assert_eq!(
//!     system.read_to_string(&system.canonicalize(Path::new(".app.toml"))?)?,
//!     "name = \"demo\""
//! );
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Time
//!
//! The clock only moves when you move it. [`advance`](MemorySystem::advance)
//! and [`System::sleep`] both add to it and return at once. Every file takes
//! its modified time from the clock when it is written. A test for a lock that
//! expires after 60 seconds therefore runs in microseconds:
//!
//! ```
//! use os_shim::{System, mock::MemorySystem};
//! use std::path::Path;
//! use std::time::Duration;
//!
//! let system = MemorySystem::new().with_file("/app.lock", b"owner-1")?;
//! let taken_at = system.metadata(Path::new("/app.lock"))?.modified;
//!
//! system.sleep(Duration::from_secs(61))?;
//!
//! let age = system.now().duration_since(taken_at)?;
//! assert!(age > Duration::from_secs(60));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Errors match the real system
//!
//! The same call fails with the same [`io::ErrorKind`] on `MemorySystem` and on
//! `RealSystem`: `NotFound` for a missing path, `AlreadyExists` for
//! `WriteMode::CreateNew` onto an existing path, `IsADirectory` for a file call
//! on a folder, `NotADirectory` for a file used as a folder. The crate's parity
//! tests run the same operations on both and compare the answers, so code that
//! branches on the error kind behaves the same under test.
//!
//! # Differences to know about
//!
//! - There are no symbolic links. [`System::walk_dir`] ignores `follow_links`
//!   and has no `.gitignore` handling, which `RealSystem` has.
//! - [`System::canonicalize`] only joins a relative path onto the current
//!   folder. It does not check that the path exists.
//! - [`System::read_dir`] gives no order. [`System::walk_dir`] sorts by path.
//! - [`System::create_temp_dir`] gives `/tmp/mock_<n>` inside the memory
//!   filesystem, never a real folder.
//!
//! # Features
//!
//! - `process`: adds `MemorySystem::with_pid` and answers `System::is_pid_alive`.
//! - `async`: adds `System::sleep_async`, which advances the clock when awaited.
//! - `zip`: adds `MemorySystem::from_zip_reader` and `MemorySystem::from_zip_stream`.

#![expect(
    clippy::std_instead_of_alloc,
    reason = "I couldn't find that trait in the alloc crate"
)]
#![expect(
    clippy::std_instead_of_core,
    reason = "I couldn't find that trait in the core crate"
)]

#[cfg(feature = "async")]
use core::pin::Pin;
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

/// In-memory implementation of [`System`] for testing.
///
/// `MemorySystem` provides an in-memory filesystem, environment, clock and
/// process table, for fast, isolated unit tests without side effects. Build one
/// with [`new`](Self::new) and the `with_*` builders, then pass it where your
/// code takes `&dyn System`. Use [`advance`](Self::advance) or [`System::sleep`]
/// to move its clock.
///
/// Clones share state: `system.clone()` is a second handle to the same files,
/// not a snapshot. [`Default`] is the same as [`new`](Self::new).
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
///
/// # Shared state
///
/// ```
/// use os_shim::{System, WriteMode, mock::MemorySystem};
/// use std::path::Path;
///
/// let system = MemorySystem::new();
/// let handle = system.clone();
///
/// handle.write(Path::new("/note.txt"), b"hi", WriteMode::CreateNew)?;
///
/// assert_eq!(system.read_to_string(Path::new("/note.txt"))?, "hi");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors that match the disk
///
/// ```
/// use os_shim::{System, WriteMode, mock::MemorySystem};
/// use std::io::ErrorKind;
/// use std::path::Path;
///
/// let system = MemorySystem::new();
/// let path = Path::new("/app.lock");
///
/// system.write(path, b"first", WriteMode::CreateNew)?;
/// let second = system.write(path, b"second", WriteMode::CreateNew);
/// assert_eq!(second.unwrap_err().kind(), ErrorKind::AlreadyExists);
///
/// // Remove the lock only while it still holds our id.
/// assert!(!system.remove_file_if(path, b"someone else")?);
/// assert!(system.remove_file_if(path, b"first")?);
///
/// let gone = system.read_to_string(path);
/// assert_eq!(gone.unwrap_err().kind(), ErrorKind::NotFound);
/// # Ok::<(), Box<dyn std::error::Error>>(())
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
    /// This is the clock that [`System::sleep`] moves too. Files written after
    /// the call take the new time as their modified time. Files already
    /// written keep the time they had.
    ///
    /// ```
    /// use os_shim::{System, mock::MemorySystem};
    /// use std::path::Path;
    /// use std::time::{Duration, SystemTime};
    ///
    /// let system = MemorySystem::new().with_file("/old.txt", b"a")?;
    /// system.advance(Duration::from_secs(30))?;
    /// system.write(Path::new("/new.txt"), b"b", os_shim::WriteMode::Replace)?;
    ///
    /// let old = system.metadata(Path::new("/old.txt"))?.modified;
    /// let new = system.metadata(Path::new("/new.txt"))?.modified;
    /// assert_eq!(old, SystemTime::UNIX_EPOCH);
    /// assert_eq!(new, SystemTime::UNIX_EPOCH + Duration::from_secs(30));
    /// assert_eq!(system.now(), new);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The internal lock cannot be acquired
    /// - The time would overflow (`InvalidInput`)
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

    /// Fail a folder rename the way a real filesystem does, before anything moves.
    fn check_dir_rename(
        files: &BTreeMap<PathBuf, Vec<u8>>,
        dirs: &BTreeSet<PathBuf>,
        from: &Path,
        to: &Path,
    ) -> io::Result<()> {
        if files.contains_key(to) {
            return Err(io::Error::new(
                io::ErrorKind::NotADirectory,
                format!("Not a directory: {}", to.display()),
            ));
        }
        if let Some(parent) = to.parent()
            && !dirs.contains(parent)
        {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("Parent directory does not exist: {}", parent.display()),
            ));
        }
        if to == from {
            return Ok(());
        }
        if to.starts_with(from) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "Cannot move a folder into itself: {} to {}",
                    from.display(),
                    to.display()
                ),
            ));
        }
        if dirs.contains(to)
            && (files.keys().any(|path| path.parent() == Some(to))
                || dirs.iter().any(|path| path.parent() == Some(to)))
        {
            return Err(io::Error::new(
                io::ErrorKind::DirectoryNotEmpty,
                format!("Directory not empty: {}", to.display()),
            ));
        }
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

        while let Some(parent) = current.parent() {
            ancestors.push(parent.to_path_buf());
            current = parent;
            if parent == Path::new("") || parent == Path::new("/") {
                break;
            }
        }

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
    /// Available with the `zip` feature. A tree archived with
    /// [`System::to_zip_vec`] comes back with the folder's contents at the root.
    ///
    /// ```
    /// use os_shim::{System, mock::MemorySystem};
    /// use std::io::Cursor;
    /// use std::path::Path;
    ///
    /// let source = MemorySystem::new().with_file("/project/src/a.txt", b"hello")?;
    /// let bytes = source.to_zip_vec(Path::new("/project"))?;
    ///
    /// let loaded = MemorySystem::from_zip_reader(Cursor::new(bytes))?;
    /// assert_eq!(loaded.read_to_string(Path::new("/src/a.txt"))?, "hello");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
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
    /// Available with the `zip` feature.
    ///
    /// ```
    /// use os_shim::{System, mock::MemorySystem};
    /// use std::path::Path;
    ///
    /// let source = MemorySystem::new().with_file("/project/a.txt", b"hello")?;
    /// let mut archive = source.to_zip_stream(Path::new("/project"))?;
    ///
    /// let loaded = MemorySystem::from_zip_stream(&mut archive)?;
    /// assert_eq!(loaded.read_to_string(Path::new("/a.txt"))?, "hello");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
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
    ///
    /// The only folder is `/`. The current folder is `/` and the current
    /// executable is `/mock/exe`. There are no files, no environment variables
    /// and no live process ids. The clock reads [`SystemTime::UNIX_EPOCH`].
    ///
    /// ```
    /// use os_shim::{System, mock::MemorySystem};
    /// use std::path::{Path, PathBuf};
    /// use std::time::SystemTime;
    ///
    /// let system = MemorySystem::new();
    /// assert_eq!(system.current_dir()?, PathBuf::from("/"));
    /// assert_eq!(system.current_exe()?, PathBuf::from("/mock/exe"));
    /// assert_eq!(system.now(), SystemTime::UNIX_EPOCH);
    /// assert!(system.read_dir(Path::new("/"))?.is_empty());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
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
    /// The path is stored as given. The folder is not created and its
    /// existence is not checked. Relative paths given to
    /// [`System::canonicalize`] are joined onto it.
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
    /// The path is stored as given. No file is created and its existence is
    /// not checked.
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

    /// Add a directory and every folder above it (builder pattern).
    ///
    /// ```
    /// use os_shim::{System, mock::MemorySystem};
    /// use std::path::Path;
    ///
    /// let system = MemorySystem::new().with_dir("/srv/app/logs")?;
    /// assert!(system.is_dir(Path::new("/srv/app"))?);
    /// assert!(system.is_dir(Path::new("/srv/app/logs"))?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The internal lock cannot be acquired
    /// - The path is a file (`AlreadyExists`)
    /// - A path above it is a file (`NotADirectory`)
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
        if state.files.contains_key(&path_buf) {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("File exists: {}", path_buf.display()),
            ));
        }
        Self::check_no_file_above(&state.files, &path_buf)?;
        Self::ensure_parent_dirs(&mut state.dirs, &path_buf);
        state.dirs.insert(path_buf);
        drop(state);
        Ok(self)
    }

    /// Set an environment variable (builder pattern).
    ///
    /// The variable lives in this `MemorySystem` only. The process environment
    /// is not touched, so parallel tests do not collide. A later call with the
    /// same key replaces the value.
    ///
    /// ```
    /// use os_shim::{System, mock::MemorySystem};
    /// use std::env::VarError;
    ///
    /// let system = MemorySystem::new().with_env("MODE", "test")?;
    /// assert_eq!(system.env_var("MODE")?, "test");
    /// assert_eq!(system.env_var("MISSING"), Err(VarError::NotPresent));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
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
    /// Missing parent folders are created. An existing file at the path is
    /// replaced. The file's modified time is the clock at this moment, so call
    /// [`with_now`](Self::with_now) or [`advance`](Self::advance) first to give
    /// it another time.
    ///
    /// ```
    /// use os_shim::{System, mock::MemorySystem};
    /// use std::path::Path;
    /// use std::time::{Duration, SystemTime};
    ///
    /// let at = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000);
    /// let system = MemorySystem::new()
    ///     .with_now(at)?
    ///     .with_file("/data/in.csv", b"a,b\n1,2\n")?;
    ///
    /// let meta = system.metadata(Path::new("/data/in.csv"))?;
    /// assert_eq!(meta.len, 8);
    /// assert_eq!(meta.modified, at);
    /// assert!(system.is_dir(Path::new("/data"))?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The internal lock cannot be acquired
    /// - A path above it is a file (`NotADirectory`)
    /// - The path is a folder (`IsADirectory`)
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
        Self::check_no_file_above(&state.files, &path_buf)?;
        if state.dirs.contains(&path_buf) {
            return Err(io::Error::new(
                io::ErrorKind::IsADirectory,
                format!("Is a directory: {}", path_buf.display()),
            ));
        }

        if let Some(parent) = path_buf.parent() {
            Self::ensure_parent_dirs(&mut state.dirs, parent);
        }

        let now = state.now;
        state.modified.insert(path_buf.clone(), now);
        state.files.insert(path_buf, contents.to_vec());
        drop(state);
        Ok(self)
    }

    /// Set the clock to an absolute time (builder pattern).
    ///
    /// Without this the clock starts at [`SystemTime::UNIX_EPOCH`]. Files
    /// already created keep their earlier modified times.
    ///
    /// ```
    /// use os_shim::{System, mock::MemorySystem};
    /// use std::time::{Duration, SystemTime};
    ///
    /// let at = SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000);
    /// let system = MemorySystem::new().with_now(at)?;
    /// assert_eq!(system.now(), at);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
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
    /// Available with the `process` feature. [`System::is_pid_alive`] answers
    /// `true` for the ids added here and `false` for every other id. An id
    /// cannot be removed once added; build a new `MemorySystem` for the dead
    /// case.
    ///
    /// ```
    /// use os_shim::{System, mock::MemorySystem};
    ///
    /// let system = MemorySystem::new().with_pid(1234)?;
    /// assert!(system.is_pid_alive(1234));
    /// assert!(!system.is_pid_alive(4321));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
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
            if state.dirs.contains(from) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("Not a regular file: {}", from.display()),
                ));
            }
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

        self.write(to, &contents, WriteMode::Replace)?;
        Ok(size)
    }

    #[inline]
    fn create(&self, path: &Path) -> io::Result<Box<dyn Write + '_>> {
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
        state.files.insert(path.to_path_buf(), Vec::new());
        drop(state);

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
        let id = TEMP_DIR_COUNTER.fetch_add(1, Ordering::SeqCst);
        let temp_path = PathBuf::from(format!("/tmp/mock_{id}"));

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
        if let Some(parent) = path.parent()
            && !state.dirs.contains(parent)
        {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("Parent directory does not exist: {}", parent.display()),
            ));
        }
        if !state.files.contains_key(path) {
            let now = state.now;
            state.modified.insert(path.to_path_buf(), now);
            state.files.insert(path.to_path_buf(), Vec::new());
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
    fn remove_file_if(&self, path: &Path, contents: &[u8]) -> io::Result<bool> {
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

        let Some(held) = state.files.get(path) else {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("File not found: {}", path.display()),
            ));
        };
        if held.as_slice() != contents {
            return Ok(false);
        }

        state.files.remove(path);
        state.modified.remove(path);
        drop(state);
        Ok(true)
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
            Self::check_dir_rename(&state.files, &state.dirs, from, to)?;
        }

        if state.files.contains_key(from) {
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
            if let Some(modified) = state.modified.remove(from) {
                state.modified.insert(to.to_path_buf(), modified);
            }
            return Ok(());
        }

        if state.dirs.remove(from) {
            state.dirs.insert(to.to_path_buf());

            let files_to_move: Vec<(PathBuf, Vec<u8>)> = state
                .files
                .iter()
                .filter(|&(path, _)| path.starts_with(from))
                .map(|(path, contents)| (path.clone(), contents.clone()))
                .collect();

            for (old_path, contents) in files_to_move {
                state.files.remove(&old_path);
                let old_modified = state.modified.remove(&old_path);
                let relative = old_path
                    .strip_prefix(from)
                    .map_err(|err| io::Error::other(err.to_string()))?;
                let new_path = to.join(relative);
                if let Some(modified) = old_modified {
                    state.modified.insert(new_path.clone(), modified);
                }
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
    fn sleep(&self, duration: Duration) -> io::Result<()> {
        self.advance(duration)
    }

    #[cfg(feature = "async")]
    #[inline]
    fn sleep_async(
        &self,
        duration: Duration,
    ) -> Pin<Box<dyn Future<Output = io::Result<()>> + Send + '_>> {
        Box::pin(async move { self.advance(duration) })
    }

    #[inline]
    fn walk_dir(
        &self,
        path: &Path,
        // `follow_links` is a no-op: the mock filesystem has no symlinks, so a tree walks
        // the same for both flag values.
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

            if current != path {
                entries.push(WalkEntry {
                    is_dir: state.dirs.contains(&current),
                    is_file: state.files.contains_key(&current),
                    path: current.clone(),
                });
            }

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
///
/// It cannot be built directly. [`System::create_temp_dir`] on a
/// [`MemorySystem`] returns it as a `Box<dyn TempDirHandle>`. The folder is
/// `/tmp/mock_<n>` inside the memory filesystem, where `<n>` comes from a
/// counter shared by every `MemorySystem` in the process. Dropping the handle
/// removes the folder and everything in it.
///
/// ```
/// use os_shim::{System, TempDirHandle, mock::MemorySystem};
///
/// let system = MemorySystem::new();
/// let dir = system.create_temp_dir()?;
/// let path = dir.path().to_path_buf();
///
/// assert!(path.starts_with("/tmp"));
/// assert!(system.is_dir(&path)?);
///
/// drop(dir);
/// assert!(!system.exists(&path)?);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
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
