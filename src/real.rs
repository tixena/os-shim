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
//! use os_shim::error::Result;
//! use os_shim::{System, real::RealSystem};
//! use std::path::Path;
//!
//! async fn file_size(system: &dyn System, path: &Path) -> Result<u64> {
//!     Ok(system.metadata(path).await?.len)
//! }
//!
//! async fn has_config<S: System>(system: &S) -> Result<bool> {
//!     system.exists(Path::new("app.toml")).await
//! }
//!
//! // The body of `main` in a program that uses the real machine.
//! async fn run() -> Result<()> {
//!     let system = RealSystem::new();
//!     if has_config(&system).await? {
//!         println!("{} bytes", file_size(&system, Path::new("app.toml")).await?);
//!     }
//!     Ok(())
//! }
//! ```

use crate::bucket::disk::DiskBucket;
use crate::bucket::{Bucket, Condition, Guarantees, RevisionToken};
use crate::error::Result;
use crate::files::{self, Machine};
use crate::{BoxFuture, FileMetadata, System, TempDirHandle, WalkEntry};
use core::result;
use core::time::Duration;
use futures_io::{AsyncRead, AsyncWrite};
use std::env;
use std::env::VarError;
#[cfg(feature = "zip")]
use std::io::Read;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::SystemTime;
use tempfile::TempDir;
#[cfg(feature = "async")]
use tokio::time;

/// Production implementation of [`System`]: the machine's environment,
/// processes and clock, with its files in a bucket, the local disk by default.
///
/// Environment, clock and process calls go straight to the standard library;
/// every file operation is carried out by the bucket `B`. `RealSystem::new()`
/// holds the disk through [`DiskBucket`], and
/// [`with_bucket`](Self::with_bucket) holds any other bucket.
/// What to know before using it:
///
/// - [`walk_dir`](System::walk_dir) respects `.gitignore` files over the disk.
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
/// let copy = system.clone();
/// assert!(is_unconfigured(&system));
/// assert!(is_unconfigured(&copy));
/// ```
#[derive(Debug, Clone, Default)]
pub struct RealSystem<B = DiskBucket> {
    /// The bucket that holds the files.
    bucket: B,
}

impl RealSystem<DiskBucket> {
    /// The machine with its files on the local disk.
    #[must_use]
    #[inline]
    pub const fn new() -> Self {
        Self {
            bucket: DiskBucket::new(),
        }
    }
}

impl<B: Bucket> RealSystem<B> {
    /// The machine with its files in `bucket`.
    #[must_use]
    #[inline]
    pub const fn with_bucket(bucket: B) -> Self {
        Self { bucket }
    }
}

impl<B: Bucket> Machine for RealSystem<B> {
    type Store = B;

    #[inline]
    fn store(&self) -> &B {
        &self.bucket
    }

    #[inline]
    fn working_dir(&self) -> Result<PathBuf> {
        Ok(env::current_dir()?)
    }
}

impl<B: Bucket> System for RealSystem<B> {
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
        Box::pin(async {
            let handle: Box<dyn TempDirHandle + 'system> = Box::new(RealTempDir {
                inner: TempDir::new()?,
            });
            Ok(handle)
        })
    }

    #[inline]
    fn current_dir(&self) -> Result<PathBuf> {
        Ok(env::current_dir()?)
    }

    #[inline]
    fn current_exe(&self) -> Result<PathBuf> {
        Ok(env::current_exe()?)
    }

    #[inline]
    fn env_var(&self, key: &str) -> result::Result<String, VarError> {
        env::var(key)
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
        // SAFETY: Sending signal 0 to a process does not affect the process;
        // it only checks whether the process exists.
        unsafe { libc::kill(pid.cast_signed(), 0) == 0 }
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
        SystemTime::now()
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
        // SAFETY: This is safe when called from a single-threaded context or when
        // the caller ensures no other threads are reading environment variables.
        unsafe {
            env::set_var(key, value);
        }
    }

    #[inline]
    fn sleep(&self, duration: Duration) -> Result<()> {
        thread::sleep(duration);
        Ok(())
    }

    #[cfg(feature = "async")]
    #[inline]
    fn sleep_async(&self, duration: Duration) -> BoxFuture<'_, Result<()>> {
        Box::pin(async move {
            time::sleep(duration).await;
            Ok(())
        })
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
    fn to_zip_stream(&self) -> BoxFuture<'_, Result<Box<dyn Read + Send>>> {
        Box::pin(async move {
            let system = RealSystem::new();
            system.to_zip_stream(self.inner.path()).await
        })
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "Tests use unwrap for brevity; panics indicate test failure"
)]
mod tests;
