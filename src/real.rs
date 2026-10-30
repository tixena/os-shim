//! The [`System`] for production: [`RealSystem`] runs on the real machine.
//!
//! Pass it wherever code takes `&dyn System`. Tests pass
//! [`MemorySystem`](crate::mock::MemorySystem) instead.
//!
//! # Use it in `main`
//!
//! Build the system once in `main` and pass a reference down. A function can
//! take `&dyn System` or a generic `S: System`; both accept `&RealSystem`.
//!
//! ```
//! use os_shim::{System, real::RealSystem};
//! use std::io;
//! use std::path::Path;
//!
//! fn file_size(system: &dyn System, path: &Path) -> io::Result<u64> {
//!     Ok(system.metadata(path)?.len)
//! }
//!
//! fn has_config<S: System>(system: &S) -> io::Result<bool> {
//!     system.exists(Path::new("app.toml"))
//! }
//!
//! // The body of `main` in a program that uses the real machine.
//! fn run() -> io::Result<()> {
//!     let system = RealSystem::new();
//!     if has_config(&system)? {
//!         println!("{} bytes", file_size(&system, Path::new("app.toml"))?);
//!     }
//!     Ok(())
//! }
//! ```

use crate::{FileMetadata, System, TempDirHandle, WalkEntry, WriteMode};
#[cfg(feature = "async")]
use core::pin::Pin;
use core::time::Duration;
use ignore::WalkBuilder;
use same_file::Handle;
use std::env;
use std::env::VarError;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::SystemTime;
use tempfile::TempDir;
#[cfg(feature = "async")]
use tokio::time;

/// Production implementation of [`System`].
///
/// This implementation directly delegates to the standard library's
/// environment and filesystem functions. It's a zero-cost abstraction
/// that provides no overhead in production.
///
/// It holds no state, so it is `Copy` and every value is the same machine.
/// What to know before using it:
///
/// - [`walk_dir`](System::walk_dir) respects `.gitignore` files.
/// - [`set_env_var`](System::set_env_var) changes the environment of the whole
///   process.
/// - [`sleep`](System::sleep) blocks the thread; `sleep_async` (feature
///   `async`) needs a tokio runtime with its timer on.
/// - [`create_temp_dir`](System::create_temp_dir) makes a real directory,
///   handed back as a [`RealTempDir`].
/// - [`is_pid_alive`](System::is_pid_alive) (feature `process`) works on Unix
///   only.
///
/// # Example
///
/// ```
/// use os_shim::{System, real::RealSystem};
///
/// fn is_unconfigured(system: &dyn System) -> bool {
///     system.env_var("OS_SHIM_DOC_EXAMPLE_UNSET").is_err()
/// }
///
/// let system = RealSystem::new();
/// let copy = system;
/// assert!(is_unconfigured(&system));
/// assert!(is_unconfigured(&copy));
/// assert!(is_unconfigured(&RealSystem::default()));
/// ```
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct RealSystem;

impl RealSystem {
    /// Create a new `RealSystem`. It is the same as [`RealSystem::default`].
    #[must_use]
    #[inline]
    pub const fn new() -> Self {
        Self
    }
}

impl Default for RealSystem {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl System for RealSystem {
    #[inline]
    fn canonicalize(&self, path: &Path) -> io::Result<PathBuf> {
        fs::canonicalize(path)
    }

    #[inline]
    fn copy(&self, from: &Path, to: &Path) -> io::Result<u64> {
        fs::copy(from, to)
    }

    #[inline]
    fn create(&self, path: &Path) -> io::Result<Box<dyn Write + '_>> {
        let file = fs::File::create(path)?;
        Ok(Box::new(file))
    }

    #[inline]
    fn create_dir_all(&self, path: &Path) -> io::Result<()> {
        fs::create_dir_all(path)
    }

    #[inline]
    fn create_temp_dir(&self) -> io::Result<Box<dyn TempDirHandle>> {
        let temp_dir = TempDir::new()?;
        Ok(Box::new(RealTempDir { inner: temp_dir }))
    }

    #[inline]
    fn current_dir(&self) -> io::Result<PathBuf> {
        env::current_dir()
    }

    #[inline]
    fn current_exe(&self) -> io::Result<PathBuf> {
        env::current_exe()
    }

    #[inline]
    fn env_var(&self, key: &str) -> Result<String, VarError> {
        env::var(key)
    }

    #[inline]
    fn exists(&self, path: &Path) -> io::Result<bool> {
        Ok(path.exists())
    }

    #[inline]
    fn is_dir(&self, path: &Path) -> io::Result<bool> {
        Ok(path.is_dir())
    }

    #[inline]
    fn is_file(&self, path: &Path) -> io::Result<bool> {
        Ok(path.is_file())
    }

    #[cfg(feature = "process")]
    #[inline]
    fn is_pid_alive(&self, pid: u32) -> bool {
        // SAFETY: Sending signal 0 to a process does not affect the process;
        // it only checks whether the process exists.
        unsafe { libc::kill(pid.cast_signed(), 0) == 0 }
    }

    #[inline]
    fn metadata(&self, path: &Path) -> io::Result<FileMetadata> {
        let meta = fs::metadata(path)?;
        Ok(FileMetadata {
            is_dir: meta.is_dir(),
            is_file: meta.is_file(),
            len: meta.len(),
            modified: Some(meta.modified()?),
            revision: None,
        })
    }

    #[inline]
    fn now(&self) -> SystemTime {
        SystemTime::now()
    }

    #[inline]
    fn open(&self, path: &Path) -> io::Result<Box<dyn Read + '_>> {
        let file = fs::File::open(path)?;
        Ok(Box::new(file))
    }

    #[inline]
    fn open_append(&self, path: &Path) -> io::Result<Box<dyn Write + '_>> {
        let file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        Ok(Box::new(file))
    }

    #[inline]
    fn read_dir(&self, path: &Path) -> io::Result<Vec<PathBuf>> {
        fs::read_dir(path)?
            .map(|entry| entry.map(|dir_entry| dir_entry.path()))
            .collect()
    }

    #[inline]
    fn read_to_string(&self, path: &Path) -> io::Result<String> {
        fs::read_to_string(path)
    }

    #[inline]
    fn remove_dir_all(&self, path: &Path) -> io::Result<()> {
        fs::remove_dir_all(path)
    }

    #[inline]
    fn remove_file(&self, path: &Path) -> io::Result<()> {
        fs::remove_file(path)
    }

    #[inline]
    fn remove_file_if(&self, path: &Path, contents: &[u8]) -> io::Result<bool> {
        remove_opened_if(fs::File::open(path)?, path, contents)
    }

    #[inline]
    fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        fs::rename(from, to)
    }

    #[inline]
    fn set_env_var(&self, key: &str, value: &str) {
        // SAFETY: This is safe when called from a single-threaded context or when
        // the caller ensures no other threads are reading environment variables.
        unsafe {
            env::set_var(key, value);
        }
    }

    #[inline]
    fn sleep(&self, duration: Duration) -> io::Result<()> {
        thread::sleep(duration);
        Ok(())
    }

    #[cfg(feature = "async")]
    #[inline]
    fn sleep_async(
        &self,
        duration: Duration,
    ) -> Pin<Box<dyn Future<Output = io::Result<()>> + Send + '_>> {
        Box::pin(async move {
            time::sleep(duration).await;
            Ok(())
        })
    }

    #[inline]
    fn walk_dir(
        &self,
        path: &Path,
        follow_links: bool,
        hidden: bool,
    ) -> io::Result<Vec<WalkEntry>> {
        let mut entries = Vec::new();

        // `hidden` here means "include hidden files", but `WalkBuilder::hidden(true)` excludes
        // them, so the flag is forwarded negated.
        for result in WalkBuilder::new(path)
            .follow_links(follow_links)
            .hidden(!hidden)
            .build()
            .skip(1)
        {
            let entry = result.map_err(io::Error::other)?;
            let entry_path = entry.path().to_path_buf();

            entries.push(WalkEntry {
                is_dir: entry_path.is_dir(),
                is_file: entry_path.is_file(),
                path: entry_path.clone(),
            });
        }

        Ok(entries)
    }

    #[inline]
    fn write(&self, path: &Path, contents: &[u8], mode: WriteMode) -> io::Result<()> {
        match mode {
            WriteMode::CreateNew => fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)?
                .write_all(contents),
            WriteMode::Replace => fs::write(path, contents),
        }
    }
}

/// The temporary directory [`RealSystem::create_temp_dir`] makes, removed from
/// disk when dropped.
///
/// It is reached through the [`TempDirHandle`] it is returned as, and cannot be
/// built directly.
pub struct RealTempDir {
    /// The underlying `tempfile::TempDir` that manages cleanup on drop.
    inner: TempDir,
}

impl TempDirHandle for RealTempDir {
    #[inline]
    fn path(&self) -> &Path {
        self.inner.path()
    }

    #[cfg(feature = "zip")]
    #[inline]
    fn to_zip_stream(&self) -> io::Result<Box<dyn Read>> {
        RealSystem::new().to_zip_stream(self.inner.path())
    }
}

/// The steps of `remove_file_if` after it opens `path` as `file`.
fn remove_opened_if(file: fs::File, path: &Path, contents: &[u8]) -> io::Result<bool> {
    file.lock()?;
    let mut held = Vec::new();
    (&file).read_to_end(&mut held)?;
    if held != contents {
        return Ok(false);
    }
    let opened = Handle::from_file(file)?;
    let still_named = match Handle::from_path(path) {
        Ok(named) => named == opened,
        Err(error) if error.kind() == io::ErrorKind::NotFound => false,
        Err(error) => return Err(error),
    };
    if !still_named {
        return Ok(false);
    }
    fs::remove_file(path)?;
    Ok(true)
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "Tests use unwrap for brevity; panics indicate test failure"
)]
mod tests;
