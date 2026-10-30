//! [`MemorySystem`]: a [`System`] that lives entirely in memory.
//!
//! Write your code against `&dyn System` (or a generic `S: System`). Use
//! [`RealSystem`](crate::real::RealSystem) in production and `MemorySystem` in
//! tests. The test then needs no disk, no temporary folders, no real
//! environment variables, no real clock and no real processes.
//!
//! # What it holds
//!
//! One `MemorySystem` holds a [`Bucket`](crate::bucket::Bucket) with its files
//! and folders, environment variables, a current folder, a current executable
//! path, a clock and (with the `process` feature) a set of live process ids.
//! `MemorySystem::new()` holds a [`MemoryBucket`](crate::bucket::MemoryBucket)
//! that stamps files from the system's own clock;
//! [`with_bucket`](MemorySystem::with_bucket) holds another bucket, which keeps
//! its own clock. A new one starts with:
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
//! must go. They are not async: a memory bucket never waits, so each builder
//! runs its bucket call to completion at once. Chain them with `?`:
//!
//! ```
//! use os_shim::{System, mock::MemorySystem};
//! use std::path::Path;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # tokio::runtime::Builder::new_current_thread().build()?.block_on(async {
//! let system = MemorySystem::new()
//!     .with_env("HOME", "/home/user")?
//!     .with_file("/home/user/.app.toml", b"name = \"demo\"")?
//!     .with_current_dir("/home/user")?;
//!
//! // `with_file` made the parent folders.
//! assert!(system.is_dir(Path::new("/home/user")).await?);
//! // A relative path is joined onto the current folder.
//! assert_eq!(system.read_to_string(Path::new(".app.toml")).await?, "name = \"demo\"");
//! # Ok(())
//! # })
//! # }
//! ```
//!
//! # Time
//!
//! The clock only moves when you move it. [`advance`](MemorySystem::advance)
//! and awaiting [`System::sleep`] both add to it and return at once. Every file takes
//! its modified time from the clock when it is written. A test for a lock that
//! expires after 60 seconds therefore runs in microseconds:
//!
//! ```
//! use os_shim::{System, mock::MemorySystem};
//! use std::path::Path;
//! use std::time::{Duration, SystemTime};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # tokio::runtime::Builder::new_current_thread().build()?.block_on(async {
//! let system = MemorySystem::new().with_file("/app.lock", b"owner-1")?;
//! let taken_at = system.metadata(Path::new("/app.lock")).await?.modified.unwrap_or(SystemTime::UNIX_EPOCH);
//!
//! system.sleep(Duration::from_secs(61)).await?;
//!
//! let age = system.now().duration_since(taken_at)?;
//! assert!(age > Duration::from_secs(60));
//! # Ok(())
//! # })
//! # }
//! ```
//!
//! # Errors match the real system
//!
//! The same call fails with the same [`ErrorKind`](crate::error::ErrorKind) on `MemorySystem` and on
//! `RealSystem`: `NotFound` for a missing path, `AlreadyExists` for
//! `Condition::Absent` onto an existing path, `IsADirectory` for a file call
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
//! - [`System::create_temp_dir`] gives `/tmp/mock_<n>` in the system's local
//!   storage, never a real folder. [`MemorySystem::new`] keeps that storage in
//!   the same tree as its other files; over another bucket it is a tree of its
//!   own, and the paths under it never reach the bucket.
//!
//! # Features
//!
//! - `process`: adds `MemorySystem::with_pid` and answers `System::is_pid_alive`.
//! - `zip`: adds `MemorySystem::from_zip_reader` and `MemorySystem::from_zip_stream`.

#[cfg(feature = "zip")]
use crate::archive::{load_zip, load_zip_seekable};
use crate::bucket::{Bucket, Condition, Guarantees, MemoryBucket, RevisionToken};
use crate::error::Result;
use crate::files::{self, LiveTemps, LocalTempDir, Machine};
use crate::{BoxFuture, FileMetadata, System, TempDirHandle, WalkEntry};
use alloc::collections::BTreeMap;
#[cfg(feature = "process")]
use alloc::collections::BTreeSet;
use alloc::sync::Arc;
use core::future::Future;
use core::pin::pin;
use core::result;
use core::sync::atomic::{AtomicU64, Ordering};
use core::task::{Context, Poll, Waker};
use core::time::Duration;
use futures_io::{AsyncRead, AsyncWrite};
use std::env::VarError;
use std::io;
#[cfg(feature = "zip")]
use std::io::{Read, Seek};
use std::path::{Path, PathBuf};
use std::sync::{PoisonError, RwLock};
use std::time::SystemTime;

/// Global counter for generating unique temp directory IDs.
static TEMP_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

/// In-memory implementation of [`System`] for testing.
///
/// `MemorySystem` provides an in-memory filesystem, environment, clock and
/// process table, for fast, isolated unit tests without side effects. Build one
/// with [`new`](Self::new) and the `with_*` builders, then pass it where your
/// code takes `&dyn System`. Use [`advance`](Self::advance) or await
/// [`System::sleep`] to move its clock.
///
/// Its files live in the bucket `B`, a [`MemoryBucket`] by default.
/// Clones share state: `system.clone()` is a second handle to the same files,
/// not a snapshot. [`Default`] is the same as [`new`](Self::new).
///
/// # Example
/// ```
/// use os_shim::{System, mock::MemorySystem};
/// use std::path::Path;
///
/// # fn main() -> std::io::Result<()> {
/// # tokio::runtime::Builder::new_current_thread().build()?.block_on(async {
/// let system = MemorySystem::new()
///     .with_env("HOME", "/home/user").unwrap()
///     .with_file("/test/file.txt", b"Hello, world!").unwrap()
///     .with_dir("/test/subdir").unwrap();
///
/// assert_eq!(system.env_var("HOME").unwrap(), "/home/user");
/// assert!(system.exists(Path::new("/test/file.txt")).await.unwrap());
/// # Ok(())
/// # })
/// # }
/// ```
///
/// # Shared state
///
/// ```
/// use os_shim::bucket::Condition;
/// use os_shim::{System, mock::MemorySystem};
/// use std::path::Path;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # tokio::runtime::Builder::new_current_thread().build()?.block_on(async {
/// let system = MemorySystem::new();
/// let handle = system.clone();
///
/// handle.write(Path::new("/note.txt"), b"hi", Condition::Absent).await?;
///
/// assert_eq!(system.read_to_string(Path::new("/note.txt")).await?, "hi");
/// # Ok(())
/// # })
/// # }
/// ```
///
/// # Errors that match the disk
///
/// ```
/// use os_shim::bucket::Condition;
/// use os_shim::error::ErrorKind;
/// use os_shim::{System, mock::MemorySystem};
/// use std::path::Path;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # tokio::runtime::Builder::new_current_thread().build()?.block_on(async {
/// let system = MemorySystem::new();
/// let path = Path::new("/app.lock");
///
/// let first = system.write(path, b"first", Condition::Absent).await?;
/// let second = system.write(path, b"second", Condition::Absent).await;
/// assert_eq!(second.unwrap_err().kind(), ErrorKind::AlreadyExists);
///
/// // Remove the lock only while it still has the token we were given.
/// assert!(system.remove_file_if(path, &first).await?);
///
/// let gone = system.read_to_string(path).await;
/// assert_eq!(gone.unwrap_err().kind(), ErrorKind::NotFound);
/// # Ok(())
/// # })
/// # }
/// ```
#[derive(Clone)]
pub struct MemorySystem<B = MemoryBucket> {
    /// The bucket that holds the files.
    bucket: B,
    /// The storage for the paths under a live temporary directory.
    local: MemoryBucket,
    /// Shared mutable state protected by a read-write lock.
    state: Arc<RwLock<MemorySystemState>>,
    /// The roots of the temporary directories whose handles are alive.
    temps: LiveTemps,
}

/// In-memory state backing the mock environment and clock.
struct MemorySystemState {
    /// Current working directory path.
    current_dir: PathBuf,
    /// Path returned by `current_exe()`.
    current_exe: PathBuf,
    /// Map of environment variable names to values.
    env_vars: BTreeMap<String, String>,
    /// The time `now()` gives.
    now: SystemTime,
    /// Set of PIDs considered alive (for `is_pid_alive` mock).
    #[cfg(feature = "process")]
    pids_alive: BTreeSet<u32>,
}

impl<B: Bucket> MemorySystem<B> {
    /// Moves the time `now()` gives forward by `by`.
    ///
    /// This is the clock that awaiting [`System::sleep`] moves too. Files written after
    /// the call take the new time as their modified time when the bucket is the
    /// system's own `MemoryBucket`. Files already written keep the time they had.
    ///
    /// ```
    /// use os_shim::bucket::Condition;
    /// use os_shim::{System, mock::MemorySystem};
    /// use std::path::Path;
    /// use std::time::{Duration, SystemTime};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # tokio::runtime::Builder::new_current_thread().build()?.block_on(async {
    /// let system = MemorySystem::new().with_file("/old.txt", b"a")?;
    /// system.advance(Duration::from_secs(30))?;
    /// system.write(Path::new("/new.txt"), b"b", Condition::None).await?;
    ///
    /// let old = system.metadata(Path::new("/old.txt")).await?.modified;
    /// let new = system.metadata(Path::new("/new.txt")).await?.modified;
    /// assert_eq!(old, Some(SystemTime::UNIX_EPOCH));
    /// assert_eq!(new, Some(SystemTime::UNIX_EPOCH + Duration::from_secs(30)));
    /// assert_eq!(Some(system.now()), new);
    /// # Ok(())
    /// # })
    /// # }
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

    /// A system over another bucket.
    ///
    /// The bucket keeps its own clock, so [`advance`](Self::advance) and
    /// [`with_now`](Self::with_now) move only the time `now()` gives.
    #[must_use]
    #[inline]
    pub fn with_bucket(bucket: B) -> Self {
        let state = fresh_state();
        let local = clocked_bucket(&state);
        Self {
            bucket,
            local,
            state,
            temps: LiveTemps::default(),
        }
    }

    /// Set the current working directory (builder pattern).
    ///
    /// The path is stored as given. The folder is not created and its
    /// existence is not checked. Relative paths given to
    /// the file operations are joined onto it.
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

impl MemorySystem<MemoryBucket> {
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
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # tokio::runtime::Builder::new_current_thread().build()?.block_on(async {
    /// let source = MemorySystem::new().with_file("/project/src/a.txt", b"hello")?;
    /// let bytes = source.to_zip_vec(Path::new("/project")).await?;
    ///
    /// let loaded = MemorySystem::from_zip_reader(Cursor::new(bytes))?;
    /// assert_eq!(loaded.read_to_string(Path::new("/src/a.txt")).await?, "hello");
    /// # Ok(())
    /// # })
    /// # }
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
        R: Read + Seek + Send,
    {
        let system = Self::new();
        ready_now(
            "load_zip_seekable",
            load_zip_seekable(&system, source, Path::new("/")),
        )?;
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
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # tokio::runtime::Builder::new_current_thread().build()?.block_on(async {
    /// let source = MemorySystem::new().with_file("/project/a.txt", b"hello")?;
    /// let mut archive = source.to_zip_stream(Path::new("/project")).await?;
    ///
    /// let loaded = MemorySystem::from_zip_stream(&mut archive)?;
    /// assert_eq!(loaded.read_to_string(Path::new("/a.txt")).await?, "hello");
    /// # Ok(())
    /// # })
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The archive is malformed, or an entry cannot be written.
    #[cfg(feature = "zip")]
    #[inline]
    pub fn from_zip_stream(source: &mut (dyn Read + Send)) -> io::Result<Self> {
        let system = Self::new();
        ready_now("load_zip", load_zip(&system, source, Path::new("/")))?;
        Ok(system)
    }

    /// Create a new `MemorySystem` with default state.
    ///
    /// The only folder is `/`. The current folder is `/` and the current
    /// executable is `/mock/exe`. There are no files, no environment variables
    /// and no live process ids. The clock reads [`SystemTime::UNIX_EPOCH`], and
    /// the bucket stamps files from it.
    ///
    /// ```
    /// use os_shim::{System, mock::MemorySystem};
    /// use std::path::{Path, PathBuf};
    /// use std::time::SystemTime;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # tokio::runtime::Builder::new_current_thread().build()?.block_on(async {
    /// let system = MemorySystem::new();
    /// assert_eq!(system.current_dir()?, PathBuf::from("/"));
    /// assert_eq!(system.current_exe()?, PathBuf::from("/mock/exe"));
    /// assert_eq!(system.now(), SystemTime::UNIX_EPOCH);
    /// assert!(system.read_dir(Path::new("/")).await?.is_empty());
    /// # Ok(())
    /// # })
    /// # }
    /// ```
    #[must_use]
    #[inline]
    pub fn new() -> Self {
        let state = fresh_state();
        let bucket = clocked_bucket(&state);
        Self {
            local: bucket.clone(),
            bucket,
            state,
            temps: LiveTemps::default(),
        }
    }

    /// Add a directory and every folder above it (builder pattern).
    ///
    /// ```
    /// use os_shim::{System, mock::MemorySystem};
    /// use std::path::Path;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # tokio::runtime::Builder::new_current_thread().build()?.block_on(async {
    /// let system = MemorySystem::new().with_dir("/srv/app/logs")?;
    /// assert!(system.is_dir(Path::new("/srv/app")).await?);
    /// assert!(system.is_dir(Path::new("/srv/app/logs")).await?);
    /// # Ok(())
    /// # })
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The path is a file (`AlreadyExists`)
    /// - A path above it is a file (`NotADirectory`)
    #[inline]
    pub fn with_dir<P>(self, path: P) -> io::Result<Self>
    where
        P: AsRef<Path>,
    {
        ready_now("create_folder", self.bucket.create_folder(path.as_ref()))?;
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
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # tokio::runtime::Builder::new_current_thread().build()?.block_on(async {
    /// let at = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000);
    /// let system = MemorySystem::new()
    ///     .with_now(at)?
    ///     .with_file("/data/in.csv", b"a,b\n1,2\n")?;
    ///
    /// let meta = system.metadata(Path::new("/data/in.csv")).await?;
    /// assert_eq!(meta.len, 8);
    /// assert_eq!(meta.modified, Some(at));
    /// assert!(system.is_dir(Path::new("/data")).await?);
    /// # Ok(())
    /// # })
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - A path above it is a file (`NotADirectory`)
    /// - The path is a folder (`IsADirectory`)
    #[inline]
    pub fn with_file<P>(self, path: P, contents: &[u8]) -> io::Result<Self>
    where
        P: AsRef<Path>,
    {
        let target = path.as_ref();
        match ready_now("put", self.bucket.put(target, contents, &Condition::None)) {
            Ok(_token) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                if let Some(parent) = target.parent() {
                    ready_now("create_folder", self.bucket.create_folder(parent))?;
                }
                ready_now("put", self.bucket.put(target, contents, &Condition::None))?;
            }
            Err(error) => return Err(error),
        }
        Ok(self)
    }
}

impl Default for MemorySystem<MemoryBucket> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Bucket> Machine for MemorySystem<B> {
    type Local = MemoryBucket;
    type Store = B;

    #[inline]
    fn local(&self) -> &MemoryBucket {
        &self.local
    }

    #[inline]
    fn store(&self) -> &B {
        &self.bucket
    }

    #[inline]
    fn temps(&self) -> &LiveTemps {
        &self.temps
    }

    #[inline]
    fn working_dir(&self) -> Result<PathBuf> {
        self.current_dir()
    }
}

impl<B: Bucket> System for MemorySystem<B> {
    #[inline]
    fn canonicalize<'system>(
        &'system self,
        path: &'system Path,
    ) -> BoxFuture<'system, Result<PathBuf>> {
        Box::pin(files::canonicalize(self, path))
    }

    #[inline]
    fn copy<'system>(
        &'system self,
        from: &'system Path,
        to: &'system Path,
    ) -> BoxFuture<'system, Result<u64>> {
        Box::pin(files::copy(self, from, to))
    }

    #[inline]
    fn create<'system>(
        &'system self,
        path: &'system Path,
    ) -> BoxFuture<'system, Result<Box<dyn AsyncWrite + Send + Unpin + 'system>>> {
        Box::pin(files::create(self, path))
    }

    #[inline]
    fn create_dir_all<'system>(
        &'system self,
        path: &'system Path,
    ) -> BoxFuture<'system, Result<()>> {
        Box::pin(files::create_dir_all(self, path))
    }

    #[inline]
    fn create_temp_dir<'system>(
        &'system self,
    ) -> BoxFuture<'system, Result<Box<dyn TempDirHandle + 'system>>> {
        Box::pin(async move {
            let id = TEMP_DIR_COUNTER.fetch_add(1, Ordering::SeqCst);
            let path = PathBuf::from(format!("/tmp/mock_{id}"));
            self.local.create_folder(&path).await?;
            let handle: Box<dyn TempDirHandle + 'system> = Box::new(LocalTempDir::new(self, path));
            Ok(handle)
        })
    }

    #[inline]
    fn current_dir(&self) -> Result<PathBuf> {
        let state = self
            .state
            .read()
            .map_err(|err| io::Error::other(err.to_string()))?;
        Ok(state.current_dir.clone())
    }

    #[inline]
    fn current_exe(&self) -> Result<PathBuf> {
        let state = self
            .state
            .read()
            .map_err(|err| io::Error::other(err.to_string()))?;
        Ok(state.current_exe.clone())
    }

    #[inline]
    #[expect(clippy::map_err_ignore, reason = "This is for VarError")]
    fn env_var(&self, key: &str) -> result::Result<String, VarError> {
        let state = self.state.read().map_err(|_| VarError::NotPresent)?;
        state.env_vars.get(key).cloned().ok_or(VarError::NotPresent)
    }

    #[inline]
    fn exists<'system>(&'system self, path: &'system Path) -> BoxFuture<'system, Result<bool>> {
        Box::pin(files::exists(self, path))
    }

    #[inline]
    fn guarantees(&self) -> Guarantees {
        self.bucket.guarantees()
    }

    #[inline]
    fn is_dir<'system>(&'system self, path: &'system Path) -> BoxFuture<'system, Result<bool>> {
        Box::pin(files::is_dir(self, path))
    }

    #[inline]
    fn is_file<'system>(&'system self, path: &'system Path) -> BoxFuture<'system, Result<bool>> {
        Box::pin(files::is_file(self, path))
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
    fn metadata<'system>(
        &'system self,
        path: &'system Path,
    ) -> BoxFuture<'system, Result<FileMetadata>> {
        Box::pin(files::metadata(self, path))
    }

    #[inline]
    fn now(&self) -> SystemTime {
        self.state
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .now
    }

    #[inline]
    fn open<'system>(
        &'system self,
        path: &'system Path,
    ) -> BoxFuture<'system, Result<Box<dyn AsyncRead + Send + Unpin + 'system>>> {
        Box::pin(files::open(self, path))
    }

    #[inline]
    fn open_append<'system>(
        &'system self,
        path: &'system Path,
    ) -> BoxFuture<'system, Result<Box<dyn AsyncWrite + Send + Unpin + 'system>>> {
        Box::pin(files::open_append(self, path))
    }

    #[inline]
    fn read<'system>(
        &'system self,
        path: &'system Path,
    ) -> BoxFuture<'system, Result<(Vec<u8>, RevisionToken)>> {
        Box::pin(files::read(self, path))
    }

    #[inline]
    fn read_dir<'system>(
        &'system self,
        path: &'system Path,
    ) -> BoxFuture<'system, Result<Vec<PathBuf>>> {
        Box::pin(files::read_dir(self, path))
    }

    #[inline]
    fn read_to_string<'system>(
        &'system self,
        path: &'system Path,
    ) -> BoxFuture<'system, Result<String>> {
        Box::pin(files::read_to_string(self, path))
    }

    #[inline]
    fn remove_dir_all<'system>(
        &'system self,
        path: &'system Path,
    ) -> BoxFuture<'system, Result<()>> {
        Box::pin(files::remove_dir_all(self, path))
    }

    #[inline]
    fn remove_file<'system>(&'system self, path: &'system Path) -> BoxFuture<'system, Result<()>> {
        Box::pin(files::remove_file(self, path))
    }

    #[inline]
    fn remove_file_if<'system>(
        &'system self,
        path: &'system Path,
        token: &'system RevisionToken,
    ) -> BoxFuture<'system, Result<bool>> {
        Box::pin(files::remove_file_if(self, path, token))
    }

    #[inline]
    fn rename<'system>(
        &'system self,
        from: &'system Path,
        to: &'system Path,
    ) -> BoxFuture<'system, Result<()>> {
        Box::pin(files::rename(self, from, to))
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
    fn sleep(&self, duration: Duration) -> BoxFuture<'_, Result<()>> {
        Box::pin(async move { Ok(self.advance(duration)?) })
    }

    #[inline]
    fn walk_dir<'system>(
        &'system self,
        path: &'system Path,
        follow_links: bool,
        hidden: bool,
    ) -> BoxFuture<'system, Result<Vec<WalkEntry>>> {
        Box::pin(files::walk_dir(self, path, follow_links, hidden))
    }

    #[inline]
    fn write<'system>(
        &'system self,
        path: &'system Path,
        contents: &'system [u8],
        condition: Condition,
    ) -> BoxFuture<'system, Result<RevisionToken>> {
        Box::pin(files::write(self, path, contents, condition))
    }
}

/// Runs `future`, which a memory bucket makes ready at once, in a single poll.
fn ready_now<T, F>(operation: &str, future: F) -> io::Result<T>
where
    F: Future<Output = Result<T>>,
{
    let mut pinned = pin!(future);
    match pinned
        .as_mut()
        .poll(&mut Context::from_waker(Waker::noop()))
    {
        Poll::Ready(outcome) => Ok(outcome?),
        Poll::Pending => Err(io::Error::other(format!(
            "{operation} did not finish at once"
        ))),
    }
}

/// A memory bucket that stamps files from the clock in `state`.
fn clocked_bucket(state: &Arc<RwLock<MemorySystemState>>) -> MemoryBucket {
    let clock_state = Arc::clone(state);
    MemoryBucket::new(Arc::new(move || {
        clock_state
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .now
    }))
}

/// The state of a new system.
fn fresh_state() -> Arc<RwLock<MemorySystemState>> {
    Arc::new(RwLock::new(MemorySystemState {
        current_dir: PathBuf::from("/"),
        current_exe: PathBuf::from("/mock/exe"),
        env_vars: BTreeMap::new(),
        now: SystemTime::UNIX_EPOCH,
        #[cfg(feature = "process")]
        pids_alive: BTreeSet::new(),
    }))
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "Tests use unwrap for brevity; panics indicate test failure"
)]
mod tests;
