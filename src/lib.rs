//! One trait, [`System`], between your code and the machine it runs on, with a
//! real implementation for production and an in-memory one for tests.
//!
//! `System` covers environment variables, the current directory and
//! executable, files, folders, metadata, temporary directories, the clock,
//! sleeping and (with features) process checks, zip archives and object
//! storage. Its file operations are async: each returns a boxed future, and
//! the [`Bucket`](bucket::Bucket) the system holds carries them out.
//! [`RealSystem`](real::RealSystem) holds the local disk and reads
//! `std::env`; [`MemorySystem`](mock::MemorySystem) holds a
//! [`MemoryBucket`](bucket::MemoryBucket) and a clock the test moves by hand.
//!
//! # The problem and who it is for
//!
//! Code that calls `std::fs` or `std::env` directly is hard to test. Tests
//! need temporary directories and cleanup, clash with each other through
//! shared files and environment variables, and wait on real clocks for
//! anything that expires. Use this crate if you write tools, daemons or
//! libraries that read config, walk project trees, take lock files, retry with
//! backoff or write archives, and you want tests that are fast, parallel and
//! exact. A coding agent can follow the same recipe: take `&dyn System` (or
//! `S: System`) in every function that touches the machine, and hand it
//! `MemorySystem` in tests.
//!
//! `MemorySystem` is checked against the real filesystem by tests that run the
//! same operations on both and compare answers, including the
//! [`std::io::ErrorKind`] of each failure.
//!
//! # The pieces
//!
//! | Piece | Real version | In-memory version | Feature |
//! | --- | --- | --- | --- |
//! | Environment, files, folders, metadata, temporary directories, clock, async sleep | [`RealSystem`](real::RealSystem), which sleeps on a thread of its own | [`MemorySystem`](mock::MemorySystem), which moves its clock | none |
//! | Compare-and-delete and create-new writes (lock files) | [`System::write`], [`System::remove_file_if`] | same methods | none |
//! | Files in another bucket | [`RealSystem::with_bucket`](real::RealSystem::with_bucket) | [`MemorySystem::with_bucket`](mock::MemorySystem::with_bucket) | none |
//! | Is a process alive | `kill(pid, 0)` on Unix | `MemorySystem::with_pid` | `process` |
//! | Zip a tree, unzip into a system | async `to_zip_*` methods, the `archive` module | `MemorySystem::from_zip_reader`, `MemorySystem::from_zip_stream` | `zip` |
//! | Files and folders in a bucket with conditional writes | `MinioBucket` (`MinIO`, `RustFS`), `S3Bucket` (AWS) in the `bucket` module | `MemoryBucket`, plus `BlockingBucket` for sync code | `minio` or `s3` for the clients |
//!
//! # What `System` offers
//!
//! - Environment: [`env_var`](System::env_var), [`set_env_var`](System::set_env_var),
//!   [`current_dir`](System::current_dir), [`current_exe`](System::current_exe).
//! - Files: [`read_to_string`](System::read_to_string),
//!   [`read`](System::read), [`write`](System::write), [`open`](System::open),
//!   [`create`](System::create), [`open_append`](System::open_append),
//!   [`copy`](System::copy), [`rename`](System::rename),
//!   [`remove_file`](System::remove_file),
//!   [`remove_file_if`](System::remove_file_if).
//! - Folders and walking: [`create_dir_all`](System::create_dir_all),
//!   [`read_dir`](System::read_dir), [`walk_dir`](System::walk_dir),
//!   [`remove_dir_all`](System::remove_dir_all).
//! - Metadata: [`exists`](System::exists), [`is_dir`](System::is_dir),
//!   [`is_file`](System::is_file), [`metadata`](System::metadata),
//!   [`canonicalize`](System::canonicalize).
//! - Temporary directories: [`create_temp_dir`](System::create_temp_dir).
//! - Clock and sleep: [`now`](System::now) and the async [`sleep`](System::sleep).
//! - Processes: `is_pid_alive`, with the `process` feature.
//! - Archives: `to_zip_vec`, `to_zip_stream` and `to_zip_writer`, with the
//!   `zip` feature.
//!
//! # Getting started
//!
//! Add the crate. No feature is on by default; turn on only the ones you need
//! (see [Features](#features)).
//!
//! ```toml
//! [dependencies]
//! os-shim = "0.1"
//!
//! # With the optional parts:
//! # os-shim = { version = "0.1", features = ["zip", "minio"] }
//! ```
//!
//! Write the code against [`System`]. Pass [`RealSystem`](real::RealSystem) in
//! `main` and [`MemorySystem`](mock::MemorySystem) in tests. A function can take
//! `&dyn System` or a generic `S: System`; both accept either implementation.
//! Every file operation is awaited.
//!
//! ```
//! use os_shim::error::{ErrorKind, Result};
//! use os_shim::{System, mock::MemorySystem, real::RealSystem};
//! use std::path::Path;
//!
//! async fn load_greeting(system: &dyn System) -> Result<String> {
//!     let home = system.env_var("HOME").map_err(std::io::Error::other)?;
//!     let path = Path::new(&home).join(".greeting");
//!     if system.exists(&path).await? {
//!         system.read_to_string(&path).await
//!     } else {
//!         Ok("hello".to_owned())
//!     }
//! }
//!
//! // The generic form takes the same systems as `&dyn System` does.
//! async fn greeting_length<S: System>(system: &S) -> Result<usize> {
//!     Ok(load_greeting(system).await?.len())
//! }
//!
//! // Production: `main` builds the real system once and passes it down.
//! async fn run() -> Result<()> {
//!     let system = RealSystem::new();
//!     println!("{}", load_greeting(&system).await?);
//!     println!("{}", greeting_length(&system).await?);
//!     Ok(())
//! }
//!
//! # fn main() -> std::io::Result<()> {
//! # tokio::runtime::Builder::new_current_thread().build()?.block_on(async {
//! // Test: no disk, no process-wide environment.
//! let with_file = MemorySystem::new()
//!     .with_env("HOME", "/home/ana").unwrap()
//!     .with_file("/home/ana/.greeting", b"hi there").unwrap();
//! assert_eq!(load_greeting(&with_file).await.unwrap(), "hi there");
//!
//! let without_file = MemorySystem::new()
//!     .with_env("HOME", "/home/ana").unwrap()
//!     .with_dir("/home/ana").unwrap();
//! assert_eq!(load_greeting(&without_file).await.unwrap(), "hello");
//! assert_eq!(greeting_length(&with_file).await.unwrap(), 8);
//!
//! let missing = without_file.read_to_string(Path::new("/nope")).await;
//! assert_eq!(missing.unwrap_err().kind(), ErrorKind::NotFound);
//! # Ok(())
//! # })
//! # }
//! ```
//!
//! # Converting existing code
//!
//! A coding agent or a person converts a function in four steps. Add a
//! `system: &dyn System` parameter. Replace each call in the table below with
//! the `System` method beside it, and make the function `async` to await the
//! file ones. Pass `system` on to every function that calls them. Call
//! `RealSystem::new()` once, in `main`.
//!
//! | Existing call | Use instead |
//! | --- | --- |
//! | `std::env::var(key)` | [`system.env_var(key)`](System::env_var) |
//! | `std::env::set_var(key, value)` | [`system.set_env_var(key, value)`](System::set_env_var) |
//! | `std::env::current_dir()`, `std::env::current_exe()` | [`system.current_dir()`](System::current_dir), [`system.current_exe()`](System::current_exe) |
//! | `std::fs::read_to_string(path)` | [`system.read_to_string(path)`](System::read_to_string) |
//! | `std::fs::write(path, bytes)` | [`system.write(path, bytes, Condition::None).await`](System::write) |
//! | `OpenOptions::new().create_new(true)` | [`system.write(path, bytes, Condition::Absent).await`](System::write) |
//! | `File::open(path)`, `File::create(path)` | [`system.open(path)`](System::open), [`system.create(path)`](System::create) |
//! | `std::fs::copy`, `rename`, `remove_file` | [`copy`](System::copy), [`rename`](System::rename), [`remove_file`](System::remove_file) |
//! | `std::fs::create_dir_all(path)` | [`system.create_dir_all(path)`](System::create_dir_all) |
//! | `std::fs::read_dir(path)` | [`system.read_dir(path)`](System::read_dir), which gives full paths |
//! | `walkdir` or `ignore::WalkBuilder` | [`system.walk_dir(path, follow_links, hidden)`](System::walk_dir) |
//! | `std::fs::metadata(path)` | [`system.metadata(path)`](System::metadata) |
//! | `Path::exists()`, `is_dir()`, `is_file()` | [`system.exists(path)?`](System::exists), [`is_dir`](System::is_dir), [`is_file`](System::is_file) |
//! | `std::fs::canonicalize(path)` | [`system.canonicalize(path)`](System::canonicalize) |
//! | `tempfile::tempdir()` | [`system.create_temp_dir()`](System::create_temp_dir) |
//! | `SystemTime::now()` | [`system.now()`](System::now) |
//! | `std::thread::sleep(duration)`, `tokio::time::sleep(duration)` | [`system.sleep(duration).await`](System::sleep) |
//!
//! Two things differ from the standard library. `Path::exists` hides errors
//! and returns `false`; `system.exists` returns a `Result<bool>`, so add
//! `?`. `Instant::now` has no equivalent: `System` models the wall clock
//! only, so time measured with `Instant` stays real.
//!
//! # Use cases
//!
//! ## Locks that expire, tested without waiting
//!
//! A lock file is made with [`Condition::Absent`](bucket::Condition::Absent),
//! which fails with `AlreadyExists` when someone else holds it, and the write
//! gives back the file's [`RevisionToken`]. The lock is stale when its age
//! passes a limit, and it is released with [`System::remove_file_if`], which
//! removes the file only while it still has the token the owner holds. The test
//! moves the clock with `advance` instead of sleeping.
//!
//! ```
//! use os_shim::bucket::{Condition, RevisionToken};
//! use os_shim::error::{ErrorKind, Result};
//! use os_shim::{System, mock::MemorySystem};
//! use std::path::Path;
//! use std::time::Duration;
//!
//! const LIMIT: Duration = Duration::from_secs(60);
//!
//! async fn try_lock(system: &dyn System, path: &Path, owner: &str) -> Result<Option<RevisionToken>> {
//!     match system.write(path, owner.as_bytes(), Condition::Absent).await {
//!         Ok(token) => Ok(Some(token)),
//!         Err(error) if error.kind() == ErrorKind::AlreadyExists => Ok(None),
//!         Err(error) => Err(error),
//!     }
//! }
//!
//! async fn is_stale(system: &dyn System, path: &Path) -> Result<bool> {
//!     let modified = system.metadata(path).await?.modified.unwrap_or(std::time::UNIX_EPOCH);
//!     Ok(system.now().duration_since(modified).is_ok_and(|age| age > LIMIT))
//! }
//!
//! # fn main() -> std::io::Result<()> {
//! # tokio::runtime::Builder::new_current_thread().build()?.block_on(async {
//! let system = MemorySystem::new().with_dir("/locks").unwrap();
//! let path = Path::new("/locks/job");
//!
//! let first = try_lock(&system, path, "worker-a").await.unwrap().unwrap();
//! assert!(try_lock(&system, path, "worker-b").await.unwrap().is_none());
//! assert!(!is_stale(&system, path).await.unwrap());
//!
//! system.advance(Duration::from_secs(61)).unwrap();
//! assert!(is_stale(&system, path).await.unwrap());
//!
//! // Only the holder of the current token can remove it.
//! let stale = RevisionToken::new("not-the-token");
//! assert!(!system.remove_file_if(path, &stale).await.unwrap());
//! assert!(system.remove_file_if(path, &first).await.unwrap());
//! assert!(try_lock(&system, path, "worker-b").await.unwrap().is_some());
//! # Ok(())
//! # })
//! # }
//! ```
//!
//! ## Retry and backoff with no real waiting
//!
//! Awaiting [`System::sleep`] waits on a real machine. On a `MemorySystem` it
//! moves the clock and returns at once, so a test can assert on how long the code would
//! have waited.
//!
//! ```
//! use os_shim::{System, mock::MemorySystem};
//! use std::io;
//! use std::time::Duration;
//!
//! async fn retry<T>(
//!     system: &dyn System,
//!     attempts: u32,
//!     mut operation: impl FnMut() -> io::Result<T>,
//! ) -> io::Result<T> {
//!     let mut delay = Duration::from_secs(1);
//!     let mut tried = 0;
//!     loop {
//!         match operation() {
//!             Ok(value) => return Ok(value),
//!             Err(error) => {
//!                 tried += 1;
//!                 if tried >= attempts {
//!                     return Err(error);
//!                 }
//!                 system.sleep(delay).await?;
//!                 delay *= 2;
//!             }
//!         }
//!     }
//! }
//!
//! # fn main() -> std::io::Result<()> {
//! # tokio::runtime::Builder::new_current_thread().build()?.block_on(async {
//! let system = MemorySystem::new();
//! let start = system.now();
//! let mut calls = 0;
//! let result = retry(&system, 5, || {
//!     calls += 1;
//!     if calls < 4 { Err(io::Error::other("busy")) } else { Ok(calls) }
//! })
//! .await;
//!
//! assert_eq!(result.unwrap(), 4);
//! let waited = system.now().duration_since(start).unwrap();
//! assert_eq!(waited, Duration::from_secs(1 + 2 + 4));
//! # Ok(())
//! # })
//! # }
//! ```
//!
//! ## Tests of tree-walking code, with no temporary directories
//!
//! Describe the tree in one expression. Each `MemorySystem` has its own state,
//! so tests run in parallel without clashing, and nothing is left to clean up.
//!
//! ```
//! use os_shim::error::Result;
//! use os_shim::{System, mock::MemorySystem};
//! use std::path::{Path, PathBuf};
//!
//! async fn find_configs(system: &dyn System, root: &Path) -> Result<Vec<PathBuf>> {
//!     let mut found: Vec<PathBuf> = system
//!         .walk_dir(root, false, false)
//!         .await?
//!         .into_iter()
//!         .filter(|entry| entry.is_file)
//!         .filter(|entry| entry.path.file_name().is_some_and(|name| name == "app.toml"))
//!         .map(|entry| entry.path)
//!         .collect();
//!     found.sort();
//!     Ok(found)
//! }
//!
//! # fn main() -> std::io::Result<()> {
//! # tokio::runtime::Builder::new_current_thread().build()?.block_on(async {
//! let system = MemorySystem::new()
//!     .with_file("/proj/app.toml", b"").unwrap()
//!     .with_file("/proj/api/app.toml", b"").unwrap()
//!     .with_file("/proj/api/main.rs", b"").unwrap()
//!     .with_file("/proj/.cache/app.toml", b"").unwrap();
//!
//! let found = find_configs(&system, Path::new("/proj")).await.unwrap();
//! assert_eq!(
//!     found,
//!     vec![PathBuf::from("/proj/api/app.toml"), PathBuf::from("/proj/app.toml")],
//! );
//! # Ok(())
//! # })
//! # }
//! ```
//!
//! ## Settings from the environment, with no pollution
//!
//! On a `MemorySystem`, [`System::set_env_var`] changes only that system. On a
//! [`RealSystem`](real::RealSystem) it changes the whole process, so a test
//! that sets variables there leaks them into every other test.
//!
//! ```
//! use os_shim::{System, mock::MemorySystem};
//!
//! fn log_level(system: &dyn System) -> String {
//!     system.env_var("LOG_LEVEL").unwrap_or_else(|_| "info".to_owned())
//! }
//!
//! let first = MemorySystem::new();
//! let second = MemorySystem::new();
//! first.set_env_var("LOG_LEVEL", "debug");
//!
//! assert_eq!(log_level(&first), "debug");
//! assert_eq!(log_level(&second), "info");
//! ```
//!
//! ## Zip a tree and load it back
//!
//! Available with the `zip` feature. [`System::to_zip_vec`] archives any
//! folder a system can read, and awaits. `MemorySystem::from_zip_reader` builds
//! a system from archive bytes, which makes a ready-made fixture for code that
//! reads archives.
//!
//! ```
//! # #[cfg(feature = "zip")]
//! # {
//! use os_shim::{System, mock::MemorySystem};
//! use std::io::Cursor;
//! use std::path::Path;
//!
//! # fn main() -> std::io::Result<()> {
//! # tokio::runtime::Builder::new_current_thread().build()?.block_on(async {
//! let site = MemorySystem::new()
//!     .with_file("/site/index.html", b"<h1>hi</h1>").unwrap()
//!     .with_file("/site/css/a.css", b"body {}").unwrap();
//!
//! let bytes = site.to_zip_vec(Path::new("/site")).await.unwrap();
//! let copy = MemorySystem::from_zip_reader(Cursor::new(bytes)).unwrap();
//!
//! // The folder's contents land at the root, not the folder itself.
//! assert_eq!(copy.read_to_string(Path::new("/index.html")).await.unwrap(), "<h1>hi</h1>");
//! assert_eq!(copy.read_to_string(Path::new("/css/a.css")).await.unwrap(), "body {}");
//! # Ok(())
//! # })
//! # }
//! # }
//! ```
//!
//! ## Stale locks left by dead processes
//!
//! Available with the `process` feature. A lock file can hold the id of the
//! process that took it. `is_pid_alive` says whether that process still runs.
//! `MemorySystem::with_pid` lists the ids a test treats as running.
//!
//! ```
//! # #[cfg(feature = "process")]
//! # {
//! use os_shim::{System, mock::MemorySystem};
//! use std::io;
//! use std::path::Path;
//!
//! async fn lock_is_stale(system: &dyn System, lock: &Path) -> io::Result<bool> {
//!     let pid: u32 = system.read_to_string(lock).await?.trim().parse().map_err(io::Error::other)?;
//!     Ok(!system.is_pid_alive(pid))
//! }
//!
//! # fn main() -> io::Result<()> {
//! # tokio::runtime::Builder::new_current_thread().build()?.block_on(async {
//! let system = MemorySystem::new()
//!     .with_file("/running.pid", b"1234").unwrap()
//!     .with_file("/dead.pid", b"999").unwrap()
//!     .with_pid(1234).unwrap();
//!
//! assert!(!lock_is_stale(&system, Path::new("/running.pid")).await.unwrap());
//! assert!(lock_is_stale(&system, Path::new("/dead.pid")).await.unwrap());
//! # Ok(())
//! # })
//! # }
//! # }
//! ```
//!
//! ## Conditional writes against object storage
//!
//! Available with the `minio` feature (`s3` for AWS). Write the
//! code against a generic `B: Bucket`. A put with `Condition::Absent` succeeds
//! for one caller only, which makes a claim on a key. Production builds a
//! `MinioBucket` from a `MinioConfig`; a test uses `MemoryBucket`, with no
//! server and no network. The `bucket` module has more, including updates
//! guarded by an entity tag.
//!
//! ```
//! # #[cfg(feature = "minio")]
//! # {
//! use os_shim::bucket::{Bucket, Condition, MemoryBucket, MinioBucket, MinioConfig};
//! use os_shim::error::{ErrorKind, StorageError};
//! use std::path::Path;
//!
//! async fn claim<B: Bucket>(bucket: &B, path: &Path, owner: &str) -> Result<bool, StorageError> {
//!     match bucket.put(path, owner.as_bytes(), &Condition::Absent).await {
//!         Ok(_) => Ok(true),
//!         Err(error) if error.kind() == ErrorKind::AlreadyExists => Ok(false),
//!         Err(error) => Err(error),
//!     }
//! }
//!
//! // Production: the same function against a MinIO-compatible server.
//! fn production() -> MinioBucket {
//!     MinioBucket::new(MinioConfig::new(
//!         "http://localhost:9000".to_owned(),
//!         "us-east-1".to_owned(),
//!         "jobs".to_owned(),
//!         "access-key".to_owned(),
//!         "secret-key".to_owned(),
//!     ))
//! }
//!
//! // Test: two callers race for one key.
//! let bucket = MemoryBucket::default();
//! let runtime = tokio::runtime::Builder::new_current_thread().build().unwrap();
//! runtime.block_on(async {
//!     bucket.create_folder(Path::new("/jobs")).await.unwrap();
//!     let lock = Path::new("/jobs/42");
//!     assert!(claim(&bucket, lock, "worker-a").await.unwrap());
//!     assert!(!claim(&bucket, lock, "worker-b").await.unwrap());
//!     assert_eq!(bucket.get(lock).await.unwrap().0, b"worker-a");
//! });
//! # }
//! ```
//!
//! # Differences from the real machine
//!
//! [`MemorySystem`](mock::MemorySystem) answers like [`RealSystem`](real::RealSystem)
//! for the cases tests cover, but these differ:
//!
//! - It has no symbolic links, so `follow_links` in `walk_dir` does nothing and
//!   nothing is skipped by `.gitignore` files, which `RealSystem` honours.
//! - `canonicalize` joins a relative path onto the current folder and does not
//!   check that the path exists. `RealSystem` resolves links and fails on a
//!   missing path.
//! - File permissions and ownership are not modelled. Metadata holds the kind,
//!   the size and the modification time.
//! - `set_env_var` changes only that system. `RealSystem` changes the whole
//!   process.
//! - `create_temp_dir` makes `/tmp/mock_<n>` in the system's local storage, never a real folder.
//! - The clock starts at `SystemTime::UNIX_EPOCH` and moves only when the test
//!   moves it. Awaiting `sleep` returns at once.
//! - A new system holds only the folder `/`, the current folder `/` and the
//!   executable path `/mock/exe`.
//!
//! The [`mock`] module lists the same differences beside the code that has them.
//!
//! # Features
//!
//! No feature is on by default. docs.rs builds with all of them.
//!
//! - `process`: adds `System::is_pid_alive` and `MemorySystem::with_pid`. Uses
//!   `libc`, so it is for Unix.
//! - `zip`: adds the `archive` module, the async `System::to_zip_stream`,
//!   `System::to_zip_vec`, `System::to_zip_writer`,
//!   `MemorySystem::from_zip_reader` and `MemorySystem::from_zip_stream`.
//! - `minio`: adds the `bucket` module with `MinioBucket`, for `MinIO`, `RustFS` and
//!   other S3-compatible servers, plus `MemoryBucket` and `BlockingBucket`.
//! - `s3`: adds the `bucket` module with `S3Bucket`, for AWS S3, plus
//!   `MemoryBucket` and `BlockingBucket`.
//!
//! # Modules
//!
//! - [`mock`]: [`MemorySystem`](mock::MemorySystem), the in-memory `System`.
//! - [`real`]: [`RealSystem`](real::RealSystem), the `System` for production.
//! - `archive` (feature `zip`): zip building and loading, including
//!   `ZipOptions` and `ZipStreamWriter`.
//! - `bucket`: the `Bucket` trait with its
//!   conditional `put` and `delete`, the real clients and `MemoryBucket`.

#![expect(
    clippy::module_name_repetitions,
    reason = "trait name System is the canonical name for this abstraction"
)]

extern crate alloc;

#[cfg(feature = "zip")]
pub mod archive;
pub mod bucket;
pub mod error;
mod files;
pub mod mock;
pub mod real;
mod streams;

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "Tests use unwrap for brevity; panics indicate test failure"
)]
mod parity_tests;

use core::future::Future;
use core::pin::Pin;
use core::result;
use core::time::Duration;
use futures_io::{AsyncRead, AsyncWrite};
use std::env::VarError;
#[cfg(feature = "zip")]
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use crate::bucket::{Condition, Guarantees, RevisionToken};
use crate::error::Result;

/// A boxed future: what keeps [`System`] usable as `&dyn System`.
pub type BoxFuture<'system, T> = Pin<Box<dyn Future<Output = T> + Send + 'system>>;

/// One entry found by [`System::walk_dir`].
///
/// Exactly one of `is_dir` and `is_file` is `true` for an ordinary file or
/// folder. In `RealSystem` they follow symbolic links, as `Path::is_dir` and
/// `Path::is_file` do.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct WalkEntry {
    pub is_dir: bool,
    pub is_file: bool,
    pub path: PathBuf,
}

impl WalkEntry {
    /// An entry from its three parts, for a [`System`] written outside this crate.
    ///
    /// ```
    /// use os_shim::WalkEntry;
    /// use std::path::{Path, PathBuf};
    ///
    /// let entry = WalkEntry::new(PathBuf::from("/org/templates/t1/v1.docx"), false, true);
    /// assert!(entry.is_file && !entry.is_dir);
    /// assert_eq!(entry.path, Path::new("/org/templates/t1/v1.docx"));
    /// ```
    #[inline]
    #[must_use]
    pub const fn new(path: PathBuf, is_dir: bool, is_file: bool) -> Self {
        Self {
            is_dir,
            is_file,
            path,
        }
    }
}

/// What [`System::metadata`] returns about a path.
///
/// This is os-shim's own type, not `std::fs::Metadata`, which cannot be built
/// by an in-memory system. `RealSystem` fills it from `std::fs::metadata`, so
/// it describes the target of a symbolic link, not the link.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct FileMetadata {
    pub is_dir: bool,
    pub is_file: bool,
    pub len: u64,
    /// `None` for a folder a store keeps no time for, such as a `MinIO` or S3 folder with no marker.
    pub modified: Option<SystemTime>,
    /// `None` where the token is not cheap to get: on disk, and for every folder.
    pub revision: Option<RevisionToken>,
}

impl FileMetadata {
    /// Metadata from its five parts, for a [`System`] written outside this crate.
    ///
    /// ```
    /// use os_shim::FileMetadata;
    /// use std::time::{Duration, UNIX_EPOCH};
    ///
    /// let modified = UNIX_EPOCH + Duration::from_millis(1_320_173_277_227);
    /// let metadata = FileMetadata::new(false, true, 1200, Some(modified), None);
    /// assert!(metadata.is_file && !metadata.is_dir);
    /// assert_eq!((metadata.len, metadata.modified), (1200, Some(modified)));
    /// ```
    #[inline]
    #[must_use]
    pub const fn new(
        is_dir: bool,
        is_file: bool,
        len: u64,
        modified: Option<SystemTime>,
        revision: Option<RevisionToken>,
    ) -> Self {
        Self {
            is_dir,
            is_file,
            len,
            modified,
            revision,
        }
    }
}

/// A temporary directory that is removed when the handle is dropped.
///
/// [`System::create_temp_dir`] returns one. Keep the handle alive for as long
/// as the directory is needed.
///
/// The directory is on local storage: the machine's disk for `RealSystem`, the
/// system's own memory for `MemorySystem`.
pub trait TempDirHandle {
    fn path(&self) -> &Path;

    /// A pull-based zip stream of this directory's contents.
    ///
    /// Available with the `zip` feature. It is the same archive as
    /// [`System::to_zip_stream`] gives for this directory's path.
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
    fn to_zip_stream(&self) -> BoxFuture<'_, Result<Box<dyn Read + Send>>> {
        Box::pin(async {
            Err(error::StorageError::from_kind(
                error::ErrorKind::Unsupported,
            ))
        })
    }
}

/// Everything your code asks of the machine it runs on.
///
/// Take `&dyn System` (or `S: System`) wherever code would call `std::env` or
/// `std::fs`. The file operations are async: each returns a [`BoxFuture`] of an
/// [`error::Result`], which is what keeps the trait usable as `&dyn System`.
/// A system holds a [`Bucket`](bucket::Bucket) and has the bucket carry out
/// every file operation, after it resolves a relative path against
/// [`current_dir`](Self::current_dir); [`guarantees`](Self::guarantees) passes
/// on what that bucket promises. The methods fall in these groups:
///
/// - Environment: `env_var`, `set_env_var`, `current_dir`, `current_exe`.
/// - Files: `create`, `open`, `open_append`, `read`, `read_to_string`, `write`,
///   `copy`, `rename`, `remove_file`, `remove_file_if`. `write` takes a
///   [`Condition`] and gives the new [`RevisionToken`], `read` gives the bytes
///   with their token, and `remove_file_if` removes a file only while it still
///   has a token. `open`, `create` and `open_append` return `futures-io`
///   streams.
/// - Folders: `create_dir_all`, `read_dir`, `walk_dir`, `remove_dir_all`.
/// - Queries: `exists`, `is_dir`, `is_file`, `metadata`, `canonicalize`.
/// - Temporary directories: `create_temp_dir`.
/// - Time: `now` and the async `sleep`.
/// - Processes: `is_pid_alive`, with the `process` feature.
/// - Archives: `to_zip_stream`, `to_zip_vec`, `to_zip_writer`, with the `zip`
///   feature. They are provided methods, so an implementor gets them for free.
///
/// # Implementations
/// - [`RealSystem`](real::RealSystem): Production implementation: the local
///   disk by default, and `std::env`.
/// - [`MemorySystem`](mock::MemorySystem): Test implementation using an
///   in-memory bucket and a clock the test controls.
///
/// Both give the same answers, including the same [`error::ErrorKind`] for the
/// same failure. The differences that remain are noted on the methods they
/// affect.
///
/// The trait needs `Send + Sync`, so one system can be shared between threads.
pub trait System: Send + Sync {
    /// Canonicalize a path (resolve to absolute path).
    ///
    /// `RealSystem` over the disk resolves symbolic links and fails when the
    /// path does not exist. `MemorySystem` has no symbolic links: it joins a
    /// relative path onto its current directory and checks nothing else.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The path cannot be canonicalized (on disk, it does not exist).
    fn canonicalize<'system>(
        &'system self,
        path: &'system Path,
    ) -> BoxFuture<'system, Result<PathBuf>>;

    /// Copy a file from source to destination, and return the number of bytes
    /// copied.
    ///
    /// An existing destination is replaced.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The source is missing (`NotFound`) or is a folder.
    /// - The file cannot be copied.
    fn copy<'system>(
        &'system self,
        from: &'system Path,
        to: &'system Path,
    ) -> BoxFuture<'system, Result<u64>>;

    /// Create a file for writing (returns a writable stream).
    ///
    /// A file already at the path is replaced. The parent folder must exist.
    /// Close the writer before reading the file back: a bucket may store the
    /// bytes only on close. A relative path is resolved first, and its writer
    /// stores the bytes when closed.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be created, for example because the parent folder is
    ///   missing (`NotFound`).
    fn create<'system>(
        &'system self,
        path: &'system Path,
    ) -> BoxFuture<'system, Result<Box<dyn AsyncWrite + Send + Unpin + 'system>>>;

    /// Recursively create a directory and all parent directories.
    ///
    /// Nothing happens when the directory already exists.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The directory cannot be created, for example because a file is at the
    ///   path or above it.
    fn create_dir_all<'system>(
        &'system self,
        path: &'system Path,
    ) -> BoxFuture<'system, Result<()>>;

    /// Create a temporary directory that is automatically cleaned up on drop.
    ///
    /// # Returns
    /// A handle to the temporary directory. The directory will be removed when
    /// the handle is dropped.
    ///
    /// # Note
    /// The directory is on local storage whatever bucket the system holds, so
    /// the path can go to a library or a child process. While the handle lives,
    /// every path under the directory goes to local storage; every other path
    /// goes to the bucket. `copy` and `rename` between the two move the bytes
    /// through streams, and `rename` removes the source.
    /// For `RealSystem`, this is a directory under the machine's temporary
    /// location, removed from disk on drop.
    /// For `MemorySystem`, this is a directory at a path like `/tmp/mock_7`,
    /// the number unique within the process, in the same tree as its other
    /// files when it was built by `new`.
    ///
    /// # Example
    ///
    /// ```
    /// use os_shim::{System, bucket::Condition, mock::MemorySystem};
    ///
    /// # fn main() -> std::io::Result<()> {
    /// # tokio::runtime::Builder::new_current_thread().build()?.block_on(async {
    /// let system = MemorySystem::new();
    /// let dir = system.create_temp_dir().await.unwrap();
    /// let file = dir.path().join("note.txt");
    /// system.write(&file, b"scratch", Condition::Absent).await.unwrap();
    /// assert!(system.exists(&file).await.unwrap());
    ///
    /// drop(dir);
    /// assert!(!system.exists(&file).await.unwrap());
    /// # Ok(())
    /// # })
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The temporary directory cannot be created.
    fn create_temp_dir<'system>(
        &'system self,
    ) -> BoxFuture<'system, Result<Box<dyn TempDirHandle + 'system>>>;

    /// Get the current working directory.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The current working directory cannot be retrieved.
    fn current_dir(&self) -> Result<PathBuf>;

    /// Get the path to the current executable.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The executable path cannot be determined.
    fn current_exe(&self) -> Result<PathBuf>;

    /// Get an environment variable.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The variable is not set (`VarError::NotPresent`).
    /// - The value is not valid Unicode (`VarError::NotUnicode`).
    fn env_var(&self, key: &str) -> result::Result<String, VarError>;

    /// Check if a path exists.
    ///
    /// A path under a file counts as missing.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The path cannot be checked.
    fn exists<'system>(&'system self, path: &'system Path) -> BoxFuture<'system, Result<bool>>;

    /// What the bucket this system holds promises.
    fn guarantees(&self) -> Guarantees;

    /// Check if a path points to a directory.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The path cannot be checked.
    fn is_dir<'system>(&'system self, path: &'system Path) -> BoxFuture<'system, Result<bool>>;

    /// Check if a path points to a file.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The path cannot be checked.
    fn is_file<'system>(&'system self, path: &'system Path) -> BoxFuture<'system, Result<bool>>;

    /// Check if a process with the given PID is alive.
    ///
    /// Available with the `process` feature. `RealSystem` sends signal 0 with
    /// `kill`, so it works on Unix only, and it answers `false` for a live
    /// process that belongs to another user. `MemorySystem` answers from the
    /// PIDs added with `MemorySystem::with_pid`.
    #[cfg(feature = "process")]
    fn is_pid_alive(&self, pid: u32) -> bool;

    /// Get metadata for a path: kind, size and last modification time.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Nothing is at the path (`NotFound`) or it cannot be read.
    fn metadata<'system>(
        &'system self,
        path: &'system Path,
    ) -> BoxFuture<'system, Result<FileMetadata>>;

    /// The current time: the machine's in `RealSystem`, the one a test set in
    /// `MemorySystem`.
    ///
    /// The `MemorySystem` clock starts at `SystemTime::UNIX_EPOCH`. Move it
    /// with `MemorySystem::with_now` or `MemorySystem::advance`, or by calling
    /// [`sleep`](Self::sleep). A file written there takes its modification
    /// time from this clock.
    fn now(&self) -> SystemTime;

    /// Open a file for reading (returns a readable stream).
    ///
    /// A relative path is resolved first, and its reader holds the whole file.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file does not exist (`NotFound`) or cannot be opened.
    fn open<'system>(
        &'system self,
        path: &'system Path,
    ) -> BoxFuture<'system, Result<Box<dyn AsyncRead + Send + Unpin + 'system>>>;

    /// Open a file for appending (create if it does not exist).
    ///
    /// The parent folder must exist. A bucket may hold the appended bytes until
    /// the writer is closed. A relative path is resolved first, and its writer
    /// appends the bytes when closed.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be opened for appending.
    fn open_append<'system>(
        &'system self,
        path: &'system Path,
    ) -> BoxFuture<'system, Result<Box<dyn AsyncWrite + Send + Unpin + 'system>>>;

    /// The bytes of the file at `path` and the token of the version read,
    /// together.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file is missing (`NotFound`), is a folder (`IsADirectory`) or
    ///   cannot be read.
    fn read<'system>(
        &'system self,
        path: &'system Path,
    ) -> BoxFuture<'system, Result<(Vec<u8>, RevisionToken)>>;

    /// Read directory entries, returning the full paths of its direct
    /// children, files and folders alike.
    ///
    /// The order is not specified; sort the result if it matters. For all
    /// levels use [`walk_dir`](Self::walk_dir).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The directory cannot be read: it is missing (`NotFound`) or is a file
    ///   (`NotADirectory`).
    fn read_dir<'system>(
        &'system self,
        path: &'system Path,
    ) -> BoxFuture<'system, Result<Vec<PathBuf>>>;

    /// Read entire file contents as a string.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file is missing (`NotFound`) or cannot be read.
    /// - The contents are not valid UTF-8 (`InvalidData`).
    fn read_to_string<'system>(
        &'system self,
        path: &'system Path,
    ) -> BoxFuture<'system, Result<String>>;

    /// Remove a directory and all its contents.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The directory is missing (`NotFound`), is a file (`NotADirectory`) or
    ///   cannot be removed.
    fn remove_dir_all<'system>(
        &'system self,
        path: &'system Path,
    ) -> BoxFuture<'system, Result<()>>;

    /// Remove a file.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file is missing (`NotFound`), is a folder or cannot be removed.
    fn remove_file<'system>(&'system self, path: &'system Path) -> BoxFuture<'system, Result<()>>;

    /// Removes the file only while it still has `token`: `Ok(true)` when it
    /// removed it, `Ok(false)` when the file has another token or another file
    /// took the path, and it stays.
    ///
    /// Two callers that remove the same file through this method never both
    /// remove it, and neither removes a file that another made at the path in
    /// the meantime, as long as files are made there only with
    /// `Condition::Absent`.
    ///
    /// This is how a lock file is released: the owner passes the token its own
    /// write returned, so it cannot remove a lock that someone else took after
    /// its own expired. On disk the check holds an advisory lock on the file,
    /// which binds only callers that cooperate through this method.
    ///
    /// # Example
    ///
    /// ```
    /// use os_shim::{System, bucket::Condition, mock::MemorySystem};
    /// use std::path::Path;
    ///
    /// # fn main() -> std::io::Result<()> {
    /// # tokio::runtime::Builder::new_current_thread().build()?.block_on(async {
    /// let system = MemorySystem::new();
    /// let lock = Path::new("/job.lock");
    /// let ours = system.write(lock, b"worker-a", Condition::Absent).await.unwrap();
    /// system.write(lock, b"worker-b", Condition::None).await.unwrap();
    ///
    /// assert!(!system.remove_file_if(lock, &ours).await.unwrap());
    /// assert!(system.exists(lock).await.unwrap());
    /// let (_bytes, current) = system.read(lock).await.unwrap();
    /// assert!(system.remove_file_if(lock, &current).await.unwrap());
    /// assert!(!system.exists(lock).await.unwrap());
    /// # Ok(())
    /// # })
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Nothing is at the path (`NotFound`), it is a folder (`IsADirectory`),
    ///   or a file is above it (`NotADirectory`).
    /// - The file cannot be read or removed.
    fn remove_file_if<'system>(
        &'system self,
        path: &'system Path,
        token: &'system RevisionToken,
    ) -> BoxFuture<'system, Result<bool>>;

    /// Rename or move a file or directory.
    ///
    /// A file renamed onto an existing file replaces it.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The source path does not exist.
    /// - The rename operation fails.
    fn rename<'system>(
        &'system self,
        from: &'system Path,
        to: &'system Path,
    ) -> BoxFuture<'system, Result<()>>;

    /// Set an environment variable.
    ///
    /// `RealSystem` changes the environment of the whole process, which is
    /// sound only while no other thread reads the environment. `MemorySystem`
    /// changes only its own variables. The method has no error path.
    ///
    /// # Example
    ///
    /// ```
    /// use os_shim::{System, mock::MemorySystem};
    ///
    /// let system = MemorySystem::new();
    /// system.set_env_var("MODE", "fast");
    /// assert_eq!(system.env_var("MODE").unwrap(), "fast");
    /// ```
    fn set_env_var(&self, key: &str, value: &str);

    /// Waits `duration`: on a thread of its own in `RealSystem`, which needs no
    /// runtime; in `MemorySystem`, it moves the clock forward by `duration`
    /// when awaited.
    ///
    /// # Example
    ///
    /// ```
    /// use os_shim::{System, mock::MemorySystem};
    /// use std::time::Duration;
    ///
    /// # fn main() -> std::io::Result<()> {
    /// # tokio::runtime::Builder::new_current_thread().build()?.block_on(async {
    /// let system = MemorySystem::new();
    /// let before = system.now();
    /// system.sleep(Duration::from_secs(30)).await.unwrap();
    /// let waited = system.now().duration_since(before).unwrap();
    /// assert_eq!(waited, Duration::from_secs(30));
    /// # Ok(())
    /// # })
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `MemorySystem`'s clock would overflow, or its state lock is poisoned.
    /// - `RealSystem` cannot start the thread.
    fn sleep(&self, duration: Duration) -> BoxFuture<'_, Result<()>>;

    /// A pull-based zip stream of `source`, held in memory.
    ///
    /// Available with the `zip` feature. Hidden entries (names that start with
    /// a dot) are included; to leave them out call
    /// [`archive::to_zip_writer`] with a `ZipOptions`.
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
    fn to_zip_stream<'system>(
        &'system self,
        source: &'system Path,
    ) -> BoxFuture<'system, Result<Box<dyn Read + Send>>> {
        Box::pin(async move {
            let bytes = self.to_zip_vec(source).await?;
            let reader: Box<dyn Read + Send> = Box::new(Cursor::new(bytes));
            Ok(reader)
        })
    }

    /// The zip archive of `source` as bytes.
    ///
    /// Available with the `zip` feature. Hidden entries are included; see
    /// [`to_zip_stream`](Self::to_zip_stream).
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
    fn to_zip_vec<'system>(
        &'system self,
        source: &'system Path,
    ) -> BoxFuture<'system, Result<Vec<u8>>> {
        Box::pin(async move {
            let mut bytes = Vec::new();
            self.to_zip_writer(source, &mut bytes).await?;
            Ok(bytes)
        })
    }

    /// Write `source` into `sink` as a zip archive.
    ///
    /// A directory contributes its *contents*: archiving `A/` yields `file.txt`,
    /// not `A/file.txt`. A file as `source` gives one entry named after the
    /// file. Provided method: implementors get it for free.
    ///
    /// Available with the `zip` feature. Hidden entries are included; the
    /// `archive` module's `to_zip_writer` function takes a `ZipOptions` that
    /// can leave them out.
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
    /// - The source is neither a file nor a folder (`NotFound`).
    #[cfg(feature = "zip")]
    #[inline]
    fn to_zip_writer<'system>(
        &'system self,
        source: &'system Path,
        sink: &'system mut (dyn Write + Send),
    ) -> BoxFuture<'system, Result<()>> {
        Box::pin(archive::to_zip_writer(
            self,
            source,
            sink,
            archive::ZipOptions::default(),
        ))
    }

    /// Recursively walk a directory, returning all entries.
    ///
    /// The order of the result is not specified; sort it if it matters.
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
    /// Over the disk, this respects `.gitignore` files using the `ignore` crate.
    /// Over the memory bucket, `follow_links` does nothing and nothing is
    /// skipped except by `hidden`; there are no symbolic links and no ignore
    /// files.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The directory cannot be walked.
    fn walk_dir<'system>(
        &'system self,
        path: &'system Path,
        follow_links: bool,
        hidden: bool,
    ) -> BoxFuture<'system, Result<Vec<WalkEntry>>>;

    /// Write bytes to a file under `condition` and give the new token.
    ///
    /// The parent folder must exist. With [`Condition::Absent`] this takes a
    /// lock file: one caller gets `Ok`, the others `AlreadyExists`.
    ///
    /// # Example
    ///
    /// ```
    /// use os_shim::bucket::Condition;
    /// use os_shim::error::ErrorKind;
    /// use os_shim::{System, mock::MemorySystem};
    /// use std::path::Path;
    ///
    /// # fn main() -> std::io::Result<()> {
    /// # tokio::runtime::Builder::new_current_thread().build()?.block_on(async {
    /// let system = MemorySystem::new();
    /// let path = Path::new("/state.txt");
    ///
    /// let first = system.write(path, b"one", Condition::Absent).await.unwrap();
    /// let second = system.write(path, b"two", Condition::Absent).await;
    /// assert_eq!(second.unwrap_err().kind(), ErrorKind::AlreadyExists);
    ///
    /// let third = system.write(path, b"three", Condition::Matches(first)).await.unwrap();
    /// let (bytes, token) = system.read(path).await.unwrap();
    /// assert_eq!((bytes.as_slice(), token), (&b"three"[..], third));
    /// # Ok(())
    /// # })
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be written.
    /// - `condition` is `Condition::Absent` and something is at the path
    ///   (`ErrorKind::AlreadyExists`).
    /// - `condition` is `Condition::Matches` and the file has another token
    ///   (`ErrorKind::PreconditionFailed`).
    fn write<'system>(
        &'system self,
        path: &'system Path,
        contents: &'system [u8],
        condition: Condition,
    ) -> BoxFuture<'system, Result<RevisionToken>>;
}
