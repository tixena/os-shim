//! Files and folders in one bucket of an object store, written and deleted
//! under conditions, with an in-memory twin for tests.
//!
//! Available in every build; `MinioBucket` needs the `minio` feature,
//! `S3Bucket` the `s3` feature, and `BlockingBucket` either of them.
//!
//! # What it is
//!
//! The module reads, writes, appends to, copies, renames, deletes and lists
//! files and folders by absolute path, and reads a file's kind, size,
//! modification time and revision token without its bytes. Every put and
//! delete takes a [`Condition`]:
//!
//! - [`Condition::None`]: no check. A put replaces, a delete removes.
//! - [`Condition::Absent`]: the put succeeds only if nothing is at the path.
//!   A failed `Absent` is [`ErrorKind::AlreadyExists`].
//! - [`Condition::Matches`]: the put or delete succeeds only if the file has
//!   this [`RevisionToken`]. A failed `Matches` is
//!   [`ErrorKind::PreconditionFailed`].
//!
//! Every failure is a [`StorageError`] whose kind
//! means the same on every implementation.
//!
//! # The pieces
//!
//! - `Bucket`: the asynchronous trait. Its methods return
//!   `impl Future + Send`, so it cannot be a trait object; take a generic
//!   `B: Bucket`. Only `delete`, `get`, `guarantees`, `head`, `list` and `put`
//!   are required; the other methods have bodies a backend overrides when its
//!   storage has a native form.
//! - `MinioBucket` and `MinioConfig` (feature `minio`): a client for `MinIO`,
//!   `RustFS` and other servers that speak its API. Path-style addressing.
//!   `MinioBucket::ensure` creates the bucket when it is missing.
//! - `S3Bucket` and `S3Config` (feature `s3`): a client for AWS S3.
//!   `S3Bucket::ensure` creates the bucket when it is missing: private, in the
//!   client's region, encrypted with a KMS key when one is given.
//! - [`MemoryBucket`]: the same interface over files and folders in memory. No
//!   server, no network. Clones share the same files.
//! - [`BlockingBucket`]: wraps any `Bucket` for code that is not async. It owns
//!   a tokio runtime.
//!
//! # When to use it
//!
//! - **Claim a lock or lease.** Put with `Condition::Absent`; only one caller
//!   wins. Renew or release it with the token the put returned.
//! - **Edit a shared document.** Get, change the bytes, put with
//!   `Condition::Matches(token)`. On `PreconditionFailed`, get again and
//!   retry. No update is lost.
//! - **Test with no server.** Write the code against a generic `B: Bucket`,
//!   run it on `MinioBucket` or `S3Bucket` in production and on
//!   [`MemoryBucket`] in tests. A test can play two clients with two clones.
//!
//! # Example
//!
//! Claim a key, then release it only while nobody else has changed it:
//!
//! ```
//! use os_shim::bucket::{Bucket as _, Condition, Depth, MemoryBucket};
//! use os_shim::error::ErrorKind;
//! use std::path::Path;
//! use std::sync::{Arc, Mutex};
//! use std::time::{Duration, SystemTime};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let now = Arc::new(Mutex::new(SystemTime::UNIX_EPOCH + Duration::from_secs(1_000)));
//! let read = Arc::clone(&now);
//! let bucket = MemoryBucket::new(Arc::new(move || *read.lock().unwrap()));
//! let runtime = tokio::runtime::Builder::new_current_thread().build()?;
//! runtime.block_on(async {
//!     bucket.create_folder(Path::new("/jobs")).await?;
//!     let lock = Path::new("/jobs/42.lock");
//!     let ours = bucket.put(lock, b"worker-a", &Condition::Absent).await?;
//!     let theirs = bucket.put(lock, b"worker-b", &Condition::Absent).await;
//!     assert_eq!(theirs.unwrap_err().kind(), ErrorKind::AlreadyExists);
//!
//!     *now.lock().unwrap() += Duration::from_secs(60);
//!     bucket.append(lock, b" renewed").await?;
//!     let stale = bucket.delete(lock, &Condition::Matches(ours)).await;
//!     assert_eq!(stale.unwrap_err().kind(), ErrorKind::PreconditionFailed);
//!
//!     let (bytes, token) = bucket.get(lock).await?;
//!     assert_eq!(bytes, b"worker-a renewed");
//!     bucket.delete(lock, &Condition::Matches(token)).await?;
//!     assert!(bucket.list(Path::new("/jobs"), Depth::Children).await?.is_empty());
//!     Ok::<_, os_shim::error::StorageError>(())
//! })?;
//! # Ok(())
//! # }
//! ```
//!
//! # Things that bite
//!
//! - The clients never retry. A network failure or a 5xx is an
//!   [`ErrorKind::Other`] error and the caller decides what to do.
//! - Credentials are a fixed access key and secret key. There is no session
//!   token and no environment or profile lookup.
//! - Revision tokens differ between implementations. A token from
//!   [`MemoryBucket`] means nothing to a real store, and the reverse.
//! - [`BlockingBucket`] panics when called from inside a tokio runtime.

#[macro_use]
mod interface;

mod defaults;
pub mod disk;
mod memory;
mod paths;

use self::defaults::{
    BufferedWriter, BytesReader, Target, append_by_rewrite, copy_by_rewrite, rename_by_copy,
    walk_by_list,
};
#[cfg(any(feature = "minio", feature = "s3"))]
use self::paths::key_of;
use self::paths::lexical;
use crate::error::{ErrorKind, Result, StorageError};
use crate::{FileMetadata, WalkEntry};
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::sync::Arc;
#[cfg(feature = "s3")]
use aws_sdk_s3::config::Builder as ConfigBuilder;
#[cfg(feature = "s3")]
use aws_sdk_s3::types::{
    BucketCannedAcl, BucketLocationConstraint, CreateBucketConfiguration, ServerSideEncryption,
    ServerSideEncryptionByDefault, ServerSideEncryptionConfiguration, ServerSideEncryptionRule,
};
#[cfg(any(feature = "minio", feature = "s3"))]
use aws_sdk_s3::{
    Client, Config,
    config::{BehaviorVersion, Credentials, Region, retry::RetryConfig},
    error::{ProvideErrorMetadata, SdkError},
    primitives::ByteStream,
};
#[cfg(any(feature = "minio", feature = "s3"))]
use core::error::Error;
#[cfg(any(feature = "minio", feature = "s3"))]
use core::fmt;
use core::future::ready;
#[cfg(any(feature = "minio", feature = "s3"))]
use core::iter;
use futures_io::{AsyncRead, AsyncWrite};
#[cfg(any(feature = "minio", feature = "s3"))]
use std::io;
use std::path::{Path, PathBuf};
use std::sync::RwLock;
use std::time::SystemTime;
#[cfg(any(feature = "minio", feature = "s3"))]
use tokio::runtime::{Builder, Runtime};

declare_bucket!();

/// A value a backend changes every time a file's bytes change: an entity tag on `MinIO` and S3,
/// a counter in memory, a SHA-256 hash of the bytes on disk.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RevisionToken(String);

impl RevisionToken {
    /// The token's text, as the backend gave it.
    #[inline]
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// A token from a backend's own value; for a `Bucket` written outside this crate.
    #[inline]
    #[must_use]
    pub fn new<S>(value: S) -> Self
    where
        S: Into<String>,
    {
        Self(value.into())
    }
}

/// What a write or a delete waits for.
///
/// Import the variants by path (`Condition::None`), not with `use
/// Condition::*`, where `None` hides `Option::None`. The enum is
/// `non_exhaustive`: a `match` outside this crate needs a `_` arm.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Condition {
    /// Nothing is at the path. A failed `Absent` is `AlreadyExists`.
    Absent,
    /// The file still has this token. A failed `Matches` is `PreconditionFailed`.
    Matches(RevisionToken),
    /// No check: a write replaces, a delete removes.
    None,
}

/// How far a listing reaches below a folder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Depth {
    /// Every entry at every level below the folder.
    All,
    /// The folder's direct children only.
    Children,
}

/// The clock a bucket stamps modification times from.
pub type Clock = Arc<dyn Fn() -> SystemTime + Send + Sync>;

/// One promise a backend can make.
///
/// For the disk, "atomic" means within one file system and "in one step" means
/// among callers of os-shim, under the advisory lock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Guarantee {
    /// An append happens as one step: concurrent appends neither interleave nor lose bytes.
    AppendIsAtomic,
    /// An append writes only the new bytes, not the whole file again.
    AppendWritesOnlyNewBytes,
    /// A delete under `Matches` checks the token and removes the file in one step.
    ConditionalDeleteInOneStep,
    /// A put under `Matches` checks the token and writes the file in one step.
    ConditionalWriteInOneStep,
    /// A put under `Absent` checks the path and creates the file in one step.
    CreateOnlyWriteInOneStep,
    /// A writer dropped without `close` keeps the bytes written to it.
    DroppedWriterKeepsBytes,
    /// A folder with nothing in it exists.
    EmptyFoldersExist,
    /// A rename that fails leaves the destination as it was.
    FailedRenameKeepsDestination,
    /// A file and a folder never share a path.
    FileAndFolderNeverShareAPath,
    /// Renaming a file happens as one step.
    FileRenameIsAtomic,
    /// Renaming a folder happens as one step.
    FolderRenameIsAtomic,
    /// The futures of the bucket never wait: each is ready the first time it is polled.
    FuturesNeverWait,
    /// A put checks that the parent folder exists and writes the file in one step.
    ParentCheckInOneStep,
    /// A reader never sees a file half written.
    ReadersNeverSeePartialWrites,
    /// `walk` leaves out what a `.gitignore` file names.
    WalkHonorsGitignore,
}

impl Guarantee {
    /// The bit this guarantee takes in a [`Guarantees`]: a distinct power of two per variant.
    const fn bit(self) -> u16 {
        match self {
            Self::AppendIsAtomic => 0x0001,
            Self::AppendWritesOnlyNewBytes => 0x0002,
            Self::ConditionalDeleteInOneStep => 0x0004,
            Self::ConditionalWriteInOneStep => 0x0008,
            Self::CreateOnlyWriteInOneStep => 0x0010,
            Self::DroppedWriterKeepsBytes => 0x0020,
            Self::EmptyFoldersExist => 0x0040,
            Self::FailedRenameKeepsDestination => 0x0080,
            Self::FileAndFolderNeverShareAPath => 0x0100,
            Self::FileRenameIsAtomic => 0x0200,
            Self::FolderRenameIsAtomic => 0x0400,
            Self::FuturesNeverWait => 0x0800,
            Self::ParentCheckInOneStep => 0x1000,
            Self::ReadersNeverSeePartialWrites => 0x2000,
            Self::WalkHonorsGitignore => 0x4000,
        }
    }
}

/// The promises a backend makes, as a set; `System::guarantees` passes on its bucket's.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Guarantees {
    /// One bit per [`Guarantee`].
    bits: u16,
}

impl Guarantees {
    /// Whether the set holds `guarantee`.
    #[inline]
    #[must_use]
    pub const fn contains(self, guarantee: Guarantee) -> bool {
        self.bits & guarantee.bit() != 0
    }

    /// The empty set: no promise.
    #[inline]
    #[must_use]
    pub const fn none() -> Self {
        Self { bits: 0 }
    }

    /// This set with `guarantee` added.
    #[inline]
    #[must_use]
    pub const fn with(self, guarantee: Guarantee) -> Self {
        Self {
            bits: self.bits | guarantee.bit(),
        }
    }
}

/// Files and folders in memory, for tests that reach no disk and no server.
///
/// It follows the same rules as the real clients: [`Condition`] and
/// [`ErrorKind`] mean the same, and a condition is checked and applied in one
/// step. Make one with [`MemoryBucket::new`] and a clock the test moves, or
/// with `MemoryBucket::default()` on the machine's clock. Clones share the
/// same files, so a test can play two clients with two clones.
///
/// Its revision tokens come from one counter for the whole bucket, so a file
/// deleted and created again never gets a token it had before. `/` is always a
/// folder, and a put needs its parent folder.
///
/// ```
/// use os_shim::bucket::{Bucket as _, Condition, Depth, MemoryBucket};
/// use os_shim::error::ErrorKind;
/// use std::path::Path;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let runtime = tokio::runtime::Builder::new_current_thread().build()?;
/// runtime.block_on(async {
///     let bucket = MemoryBucket::default();
///     let config = Path::new("/config.json");
///
///     let token = bucket.put(config, b"{\"v\":1}", &Condition::Absent).await?;
///     let again = bucket.put(config, b"{\"v\":2}", &Condition::Absent).await;
///     assert_eq!(again.unwrap_err().kind(), ErrorKind::AlreadyExists);
///
///     let (_bytes, seen) = bucket.get(config).await?;
///     assert_eq!(seen, token);
///
///     bucket.put(config, b"{\"v\":2}", &Condition::Matches(token.clone())).await?;
///     let stale = bucket.delete(config, &Condition::Matches(token)).await;
///     assert_eq!(stale.unwrap_err().kind(), ErrorKind::PreconditionFailed);
///
///     assert_eq!(bucket.list(Path::new("/"), Depth::Children).await?.len(), 1);
///     Ok::<_, os_shim::error::StorageError>(())
/// })?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct MemoryBucket {
    /// The clock that stamps modification times.
    clock: Clock,
    /// The files, the folders and the revision counter.
    state: Arc<RwLock<MemoryState>>,
}

/// The tree of a [`MemoryBucket`].
struct MemoryState {
    /// The files, by path.
    files: BTreeMap<PathBuf, MemoryFile>,
    /// The folders, by path. `/` is always one.
    folders: BTreeSet<PathBuf>,
    /// The next revision; one counter for the whole bucket, so a token is never handed out twice.
    next_revision: u64,
}

/// One file of a [`MemoryBucket`].
struct MemoryFile {
    /// The bytes.
    bytes: Vec<u8>,
    /// The time of the last write.
    modified: SystemTime,
    /// The revision of the last write.
    revision: u64,
}

impl Default for MemoryBucket {
    /// A bucket on the machine's clock.
    #[inline]
    fn default() -> Self {
        Self::new(Arc::new(SystemTime::now))
    }
}

/// A bucket for synchronous code: each call runs on a tokio runtime it owns.
///
/// A call from code already inside a tokio runtime panics; such code uses
/// the asynchronous interface directly.
///
/// Wrap [`MemoryBucket`], `MinioBucket` or `S3Bucket`. The methods take the
/// same arguments and return the same results as the `Bucket` trait, without
/// the futures, and it has no streams. It cannot be cloned and gives no access
/// to the bucket inside.
///
/// ```
/// use os_shim::bucket::{BlockingBucket, Condition, MemoryBucket};
/// use std::path::Path;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let bucket = BlockingBucket::new(MemoryBucket::default())?;
/// bucket.put(Path::new("/config.json"), b"{}", &Condition::Absent)?;
/// assert_eq!(bucket.get(Path::new("/config.json"))?.0, b"{}");
/// # Ok(())
/// # }
/// ```
#[cfg(any(feature = "minio", feature = "s3"))]
#[derive(Debug)]
pub struct BlockingBucket<B> {
    /// The bucket each call reaches.
    bucket: B,
    /// The current-thread runtime each call blocks on.
    runtime: Runtime,
}

#[cfg(any(feature = "minio", feature = "s3"))]
impl<B: Bucket> BlockingBucket<B> {
    /// Add `more` at the end of the file, creating it when missing, and give the new token.
    ///
    /// # Errors
    ///
    /// The errors of `Bucket::append`.
    ///
    /// # Panics
    ///
    /// Panics when called from inside a tokio runtime.
    #[inline]
    pub fn append(&self, path: &Path, more: &[u8]) -> Result<RevisionToken> {
        self.runtime.block_on(self.bucket.append(path, more))
    }

    /// The absolute form of `path`.
    ///
    /// # Errors
    ///
    /// The errors of `Bucket::canonicalize`.
    ///
    /// # Panics
    ///
    /// Panics when called from inside a tokio runtime.
    #[inline]
    pub fn canonicalize(&self, path: &Path) -> Result<PathBuf> {
        self.runtime.block_on(self.bucket.canonicalize(path))
    }

    /// Copy a file and give the number of bytes copied.
    ///
    /// # Errors
    ///
    /// The errors of `Bucket::copy`.
    ///
    /// # Panics
    ///
    /// Panics when called from inside a tokio runtime.
    #[inline]
    pub fn copy(&self, from: &Path, to: &Path) -> Result<u64> {
        self.runtime.block_on(self.bucket.copy(from, to))
    }

    /// Create `path` and every missing folder above it.
    ///
    /// # Errors
    ///
    /// The errors of `Bucket::create_folder`.
    ///
    /// # Panics
    ///
    /// Panics when called from inside a tokio runtime.
    #[inline]
    pub fn create_folder(&self, path: &Path) -> Result<()> {
        self.runtime.block_on(self.bucket.create_folder(path))
    }

    /// Remove the file at `path` when `condition` holds.
    ///
    /// # Errors
    ///
    /// The errors of `Bucket::delete`.
    ///
    /// # Panics
    ///
    /// Panics when called from inside a tokio runtime.
    #[inline]
    pub fn delete(&self, path: &Path, condition: &Condition) -> Result<()> {
        self.runtime.block_on(self.bucket.delete(path, condition))
    }

    /// The bytes at `path` and the token of that version.
    ///
    /// # Errors
    ///
    /// The errors of `Bucket::get`.
    ///
    /// # Panics
    ///
    /// Panics when called from inside a tokio runtime.
    #[inline]
    pub fn get(&self, path: &Path) -> Result<(Vec<u8>, RevisionToken)> {
        self.runtime.block_on(self.bucket.get(path))
    }

    /// Kind, size, modification time and token of what is at `path`, without its bytes.
    ///
    /// # Errors
    ///
    /// The errors of `Bucket::head`.
    ///
    /// # Panics
    ///
    /// Panics when called from inside a tokio runtime.
    #[inline]
    pub fn head(&self, path: &Path) -> Result<FileMetadata> {
        self.runtime.block_on(self.bucket.head(path))
    }

    /// The entries below `folder`, files and folders, to `depth`.
    ///
    /// # Errors
    ///
    /// The errors of `Bucket::list`.
    ///
    /// # Panics
    ///
    /// Panics when called from inside a tokio runtime.
    #[inline]
    pub fn list(&self, folder: &Path, depth: Depth) -> Result<Vec<WalkEntry>> {
        self.runtime.block_on(self.bucket.list(folder, depth))
    }

    /// A blocking bucket over `bucket`.
    ///
    /// Each one builds its own current-thread runtime, so make it once and
    /// reuse it.
    ///
    /// # Errors
    ///
    /// Returns an error if the runtime cannot be built.
    #[inline]
    pub fn new(bucket: B) -> io::Result<Self> {
        let runtime = Builder::new_current_thread().enable_all().build()?;
        Ok(Self { bucket, runtime })
    }

    /// Write `bytes` at `path` when `condition` holds, and give the new token.
    ///
    /// # Errors
    ///
    /// The errors of `Bucket::put`.
    ///
    /// # Panics
    ///
    /// Panics when called from inside a tokio runtime.
    #[inline]
    pub fn put(&self, path: &Path, bytes: &[u8], condition: &Condition) -> Result<RevisionToken> {
        self.runtime
            .block_on(self.bucket.put(path, bytes, condition))
    }

    /// Remove `path` and everything below it.
    ///
    /// # Errors
    ///
    /// The errors of `Bucket::remove_folder`.
    ///
    /// # Panics
    ///
    /// Panics when called from inside a tokio runtime.
    #[inline]
    pub fn remove_folder(&self, path: &Path) -> Result<()> {
        self.runtime.block_on(self.bucket.remove_folder(path))
    }

    /// Move a file or a folder.
    ///
    /// # Errors
    ///
    /// The errors of `Bucket::rename`.
    ///
    /// # Panics
    ///
    /// Panics when called from inside a tokio runtime.
    #[inline]
    pub fn rename(&self, from: &Path, to: &Path) -> Result<()> {
        self.runtime.block_on(self.bucket.rename(from, to))
    }

    /// Every entry below `folder`.
    ///
    /// # Errors
    ///
    /// The errors of `Bucket::walk`.
    ///
    /// # Panics
    ///
    /// Panics when called from inside a tokio runtime.
    #[inline]
    pub fn walk(&self, folder: &Path, follow_links: bool, hidden: bool) -> Result<Vec<WalkEntry>> {
        self.runtime
            .block_on(self.bucket.walk(folder, follow_links, hidden))
    }
}

/// A bucket reached through AWS's S3 SDK, which `MinioBucket` and `S3Bucket` wrap.
#[cfg(any(feature = "minio", feature = "s3"))]
#[derive(Debug)]
struct SdkBucket {
    /// The bucket every operation reaches.
    bucket: String,
    /// Whether a conditional delete reads the object's tag before it sends the delete.
    check_before_delete: bool,
    /// The SDK client every request goes through.
    client: Client,
    /// What every key starts with, before its first `/`; empty for none.
    prefix: String,
}

#[cfg(any(feature = "minio", feature = "s3"))]
impl SdkBucket {
    /// Whether the bucket exists: `HEAD` on it answers 200, or 404 when it is missing.
    async fn bucket_exists(&self) -> Result<bool> {
        match self.client.head_bucket().bucket(&self.bucket).send().await {
            Ok(_) => Ok(true),
            Err(error)
                if error
                    .raw_response()
                    .map(|response| response.status().as_u16())
                    == Some(404) =>
            {
                Ok(false)
            }
            Err(error) => Err(classify(error)),
        }
    }

    /// The entity tag of the object at `key`, or `None` when no object is there.
    async fn current_tag(&self, key: &str) -> Result<Option<String>> {
        match self
            .client
            .head_object()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await
            .map_err(classify)
        {
            Ok(head) => Ok(head.e_tag().map(str::to_owned)),
            Err(error) if error.kind() == ErrorKind::NotFound => self.object_is_missing().await,
            Err(error) => Err(error),
        }
    }

    /// Delete the file at `path` when `condition` holds.
    async fn delete(&self, path: &Path, condition: &Condition) -> Result<()> {
        let key = key_of(&self.prefix, path)?;
        let mut request = self.client.delete_object().bucket(&self.bucket).key(&key);
        match condition.clone() {
            Condition::None => {}
            Condition::Absent => {
                return match self.current_tag(&key).await? {
                    None => Ok(()),
                    Some(_) => Err(StorageError::from_kind(ErrorKind::AlreadyExists)),
                };
            }
            Condition::Matches(token) => {
                if self.check_before_delete
                    && self.current_tag(&key).await?.as_deref() != Some(token.as_str())
                {
                    return Err(StorageError::from_kind(ErrorKind::PreconditionFailed));
                }
                request = request.if_match(token.as_str());
            }
        }
        match request.send().await.map_err(classify) {
            Ok(_) => Ok(()),
            Err(error)
                if error.kind() == ErrorKind::NotFound
                    && matches!(*condition, Condition::Matches(_)) =>
            {
                Err(StorageError::from_kind(ErrorKind::PreconditionFailed))
            }
            Err(error) => Err(error),
        }
    }

    /// The bytes and token of the file at `path`.
    async fn get(&self, path: &Path) -> Result<(Vec<u8>, RevisionToken)> {
        let key = key_of(&self.prefix, path)?;
        let output = self
            .client
            .get_object()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await
            .map_err(classify)?;
        let etag = output.e_tag().map(str::to_owned).ok_or_else(missing_etag)?;
        let collected = output
            .body
            .collect()
            .await
            .map_err(|error| StorageError::new(ErrorKind::Other, error))?;
        Ok((collected.into_bytes().to_vec(), RevisionToken::new(etag)))
    }

    /// The metadata of the file at `path`, from a `HEAD` on it.
    async fn head(&self, path: &Path) -> Result<FileMetadata> {
        let key = key_of(&self.prefix, path)?;
        match self
            .client
            .head_object()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await
            .map_err(classify)
        {
            Ok(head) => {
                let etag = head.e_tag().map(str::to_owned).ok_or_else(missing_etag)?;
                let len = head
                    .content_length()
                    .and_then(|length| u64::try_from(length).ok())
                    .ok_or_else(|| {
                        StorageError::new(
                            ErrorKind::Other,
                            "the answer carries no Content-Length header",
                        )
                    })?;
                let modified = head
                    .last_modified()
                    .and_then(|time| SystemTime::try_from(*time).ok());
                Ok(FileMetadata::new(
                    false,
                    true,
                    len,
                    modified,
                    Some(RevisionToken::new(etag)),
                ))
            }
            Err(error) if error.kind() == ErrorKind::NotFound => {
                self.object_is_missing().await?;
                Err(StorageError::from_kind(ErrorKind::NotFound))
            }
            Err(error) => Err(error),
        }
    }

    /// The entries below `folder`, every page of the listing read.
    async fn list(&self, folder: &Path, depth: Depth) -> Result<Vec<WalkEntry>> {
        let base = lexical(folder)?;
        let key = key_of(&self.prefix, &base)?;
        let listed = if key.is_empty() {
            key
        } else {
            format!("{key}/")
        };
        let mut entries = Vec::new();
        let mut folders = BTreeSet::new();
        let mut pages = self
            .client
            .list_objects_v2()
            .bucket(&self.bucket)
            .prefix(&listed)
            .into_paginator()
            .send();
        while let Some(answer) = pages.next().await {
            let page = answer.map_err(classify)?;
            for relative in page
                .contents()
                .iter()
                .filter_map(|object| object.key())
                .filter_map(|object_key| object_key.strip_prefix(listed.as_str()))
                .filter(|relative| !relative.is_empty())
            {
                match (depth, relative.split_once('/')) {
                    (Depth::Children, Some((name, _rest))) => {
                        if folders.insert(name.to_owned()) {
                            entries.push(WalkEntry::new(base.join(name), true, false));
                        }
                    }
                    (Depth::All | Depth::Children, _) if relative.ends_with('/') => {}
                    (Depth::All | Depth::Children, _) => {
                        entries.push(WalkEntry::new(base.join(relative), false, true));
                    }
                }
            }
        }
        Ok(entries)
    }

    /// `Ok(None)` when the bucket exists, since a `HEAD` 404 carries no S3 code to say which of
    /// the object and the bucket is missing.
    async fn object_is_missing(&self) -> Result<Option<String>> {
        if self.bucket_exists().await? {
            return Ok(None);
        }
        Err(StorageError::new(
            ErrorKind::Other,
            format!("404: the bucket {} does not exist", self.bucket),
        ))
    }

    /// Write `bytes` at `path` when `condition` holds, and give the new token.
    async fn put(&self, path: &Path, bytes: &[u8], condition: &Condition) -> Result<RevisionToken> {
        let key = key_of(&self.prefix, path)?;
        let unconditional = self
            .client
            .put_object()
            .bucket(&self.bucket)
            .key(key)
            .body(ByteStream::from(bytes.to_vec()));
        let request = match condition.clone() {
            Condition::None => unconditional,
            Condition::Absent => unconditional.if_none_match("*"),
            Condition::Matches(token) => unconditional.if_match(token.as_str()),
        };
        match request.send().await.map_err(classify) {
            Ok(output) => output
                .e_tag()
                .map(RevisionToken::new)
                .ok_or_else(missing_etag),
            Err(error)
                if error.kind() == ErrorKind::PreconditionFailed
                    && matches!(*condition, Condition::Absent) =>
            {
                Err(StorageError::from_kind(ErrorKind::AlreadyExists))
            }
            Err(error)
                if error.kind() == ErrorKind::NotFound
                    && matches!(*condition, Condition::Matches(_)) =>
            {
                Err(StorageError::from_kind(ErrorKind::PreconditionFailed))
            }
            Err(error) => Err(error),
        }
    }
}

/// Where a `MinIO`-compatible server is, and the keys that sign every request.
///
/// Available with the `minio` feature. The arguments of [`new`](Self::new) are
/// all strings, so keep their order straight: endpoint, region, bucket,
/// access key, secret key. `Debug` prints the secret key as `<redacted>`.
#[cfg(feature = "minio")]
#[derive(Clone)]
#[non_exhaustive]
pub struct MinioConfig {
    pub access_key: String,
    pub bucket: String,
    pub endpoint: String,
    /// What every key starts with, before its first `/`.
    pub prefix: Option<String>,
    pub region: String,
    pub secret_key: String,
}

#[cfg(feature = "minio")]
impl MinioConfig {
    /// A configuration from its five parts, with no key prefix.
    #[inline]
    #[must_use]
    pub const fn new(
        endpoint: String,
        region: String,
        bucket: String,
        access_key: String,
        secret_key: String,
    ) -> Self {
        Self {
            access_key,
            bucket,
            endpoint,
            prefix: None,
            region,
            secret_key,
        }
    }

    /// The same configuration with every key under `prefix`; a trailing `/` is dropped.
    #[inline]
    #[must_use]
    pub fn with_prefix<S>(mut self, prefix: S) -> Self
    where
        S: Into<String>,
    {
        self.prefix = Some(prefix.into().trim_end_matches('/').to_owned());
        self
    }
}

#[cfg(feature = "minio")]
impl fmt::Debug for MinioConfig {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MinioConfig")
            .field("access_key", &self.access_key)
            .field("bucket", &self.bucket)
            .field("endpoint", &self.endpoint)
            .field("prefix", &self.prefix)
            .field("region", &self.region)
            .field("secret_key", &"<redacted>")
            .finish()
    }
}

/// A client for a server that speaks `MinIO`'s API: `MinIO`, `RustFS` and the like.
///
/// A delete with a token condition reads the object's tag first, because
/// `MinIO`'s server ignores `If-Match` on a delete, and sends `If-Match` as well.
/// The read and the delete are two requests, so a write between them can slip through
/// on servers that ignore `If-Match`.
///
/// Available with the `minio` feature. It uses path-style addressing, a fixed
/// access key and secret key, and never retries. It implements `Bucket`; wrap
/// it in [`BlockingBucket`] for synchronous code.
#[cfg(feature = "minio")]
#[derive(Debug)]
pub struct MinioBucket(
    /// The SDK bucket every call goes through.
    SdkBucket,
);

#[cfg(feature = "minio")]
impl MinioBucket {
    /// Create the bucket when it does not exist.
    ///
    /// # Errors
    ///
    /// `Other` when the store cannot be reached, or refuses to say whether the
    /// bucket exists or to create it.
    #[inline]
    pub async fn ensure(&self) -> Result<()> {
        if self.0.bucket_exists().await? {
            return Ok(());
        }
        self.0
            .client
            .create_bucket()
            .bucket(&self.0.bucket)
            .send()
            .await
            .map(drop)
            .map_err(classify)
    }

    /// A client for `config.bucket` on `config.endpoint`, with path-style addressing and no retries.
    #[inline]
    #[must_use]
    pub fn new(config: MinioConfig) -> Self {
        let credentials =
            Credentials::new(config.access_key, config.secret_key, None, None, "os-shim");
        let sdk = Config::builder()
            .behavior_version(BehaviorVersion::latest())
            .region(Region::new(config.region))
            .credentials_provider(credentials)
            .retry_config(RetryConfig::disabled())
            .endpoint_url(config.endpoint)
            .force_path_style(true)
            .build();
        Self(SdkBucket {
            bucket: config.bucket,
            check_before_delete: true,
            client: Client::from_conf(sdk),
            prefix: config.prefix.unwrap_or_default(),
        })
    }
}

#[cfg(feature = "minio")]
impl Bucket for MinioBucket {
    #[inline]
    fn delete(
        &self,
        path: &Path,
        condition: &Condition,
    ) -> impl Future<Output = Result<()>> + Send {
        self.0.delete(path, condition)
    }

    #[inline]
    fn get(&self, path: &Path) -> impl Future<Output = Result<(Vec<u8>, RevisionToken)>> + Send {
        self.0.get(path)
    }

    #[inline]
    fn guarantees(&self) -> Guarantees {
        sdk_guarantees(false)
    }

    #[inline]
    fn head(&self, path: &Path) -> impl Future<Output = Result<FileMetadata>> + Send {
        self.0.head(path)
    }

    #[inline]
    fn list(
        &self,
        folder: &Path,
        depth: Depth,
    ) -> impl Future<Output = Result<Vec<WalkEntry>>> + Send {
        self.0.list(folder, depth)
    }

    #[inline]
    fn put(
        &self,
        path: &Path,
        bytes: &[u8],
        condition: &Condition,
    ) -> impl Future<Output = Result<RevisionToken>> + Send {
        self.0.put(path, bytes, condition)
    }
}

/// The AWS S3 bucket to reach, and the keys that sign every request.
///
/// Available with the `s3` feature. The order of the arguments of
/// [`new`](Self::new) is region, bucket, access key, secret key. `Debug`
/// prints the secret key as `<redacted>`.
#[cfg(feature = "s3")]
#[derive(Clone)]
#[non_exhaustive]
pub struct S3Config {
    pub access_key: String,
    pub bucket: String,
    /// What every key starts with, before its first `/`.
    pub prefix: Option<String>,
    pub region: String,
    pub secret_key: String,
}

#[cfg(feature = "s3")]
impl S3Config {
    /// A configuration from its four parts, with no key prefix.
    #[inline]
    #[must_use]
    pub const fn new(
        region: String,
        bucket: String,
        access_key: String,
        secret_key: String,
    ) -> Self {
        Self {
            access_key,
            bucket,
            prefix: None,
            region,
            secret_key,
        }
    }

    /// The same configuration with every key under `prefix`; a trailing `/` is dropped.
    #[inline]
    #[must_use]
    pub fn with_prefix<S>(mut self, prefix: S) -> Self
    where
        S: Into<String>,
    {
        self.prefix = Some(prefix.into().trim_end_matches('/').to_owned());
        self
    }
}

#[cfg(feature = "s3")]
impl fmt::Debug for S3Config {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("S3Config")
            .field("access_key", &self.access_key)
            .field("bucket", &self.bucket)
            .field("prefix", &self.prefix)
            .field("region", &self.region)
            .field("secret_key", &"<redacted>")
            .finish()
    }
}

/// A client for an AWS S3 bucket, over AWS's SDK.
///
/// A delete with a token condition sends `If-Match` and lets AWS check it.
///
/// Available with the `s3` feature. It uses a fixed access key and secret key
/// and never retries. It implements `Bucket`; wrap it in [`BlockingBucket`] for
/// synchronous code.
#[cfg(feature = "s3")]
#[derive(Debug)]
pub struct S3Bucket(
    /// The SDK bucket every call goes through.
    SdkBucket,
);

#[cfg(feature = "s3")]
impl S3Bucket {
    /// Create the bucket when it does not exist: private, in the client's
    /// region, and encrypted by default with `kms_key_arn` when one is given.
    ///
    /// # Errors
    ///
    /// `Other` when AWS cannot be reached, or refuses to say whether the bucket
    /// exists, to create it or to set its encryption.
    #[inline]
    pub async fn ensure(&self, kms_key_arn: Option<&str>) -> Result<()> {
        if self.0.bucket_exists().await? {
            return Ok(());
        }
        let mut create = self
            .0
            .client
            .create_bucket()
            .bucket(&self.0.bucket)
            .acl(BucketCannedAcl::Private);
        // A bucket outside us-east-1 names its region, or AWS refuses it.
        if let Some(region) = self.0.client.config().region().map(Region::as_ref)
            && region != "us-east-1"
        {
            create = create.create_bucket_configuration(
                CreateBucketConfiguration::builder()
                    .location_constraint(BucketLocationConstraint::from(region))
                    .build(),
            );
        }
        create.send().await.map_err(classify)?;
        let Some(key_arn) = kms_key_arn else {
            return Ok(());
        };
        let by_default = ServerSideEncryptionByDefault::builder()
            .sse_algorithm(ServerSideEncryption::AwsKms)
            .kms_master_key_id(key_arn)
            .build()
            .map_err(|error| StorageError::new(ErrorKind::Other, error))?;
        let encryption = ServerSideEncryptionConfiguration::builder()
            .rules(
                ServerSideEncryptionRule::builder()
                    .apply_server_side_encryption_by_default(by_default)
                    .bucket_key_enabled(true)
                    .build(),
            )
            .build()
            .map_err(|error| StorageError::new(ErrorKind::Other, error))?;
        self.0
            .client
            .put_bucket_encryption()
            .bucket(&self.0.bucket)
            .server_side_encryption_configuration(encryption)
            .send()
            .await
            .map(drop)
            .map_err(classify)
    }

    /// A client for `config.bucket` in `config.region`, with no retries.
    #[inline]
    #[must_use]
    pub fn new(config: S3Config) -> Self {
        let prefix = config.prefix.clone().unwrap_or_default();
        Self::with_sdk_config(
            config.bucket.clone(),
            prefix,
            Self::sdk_config(config).build(),
        )
    }

    /// The SDK configuration for `config`, before any endpoint is set.
    fn sdk_config(config: S3Config) -> ConfigBuilder {
        let credentials =
            Credentials::new(config.access_key, config.secret_key, None, None, "os-shim");
        Config::builder()
            .behavior_version(BehaviorVersion::latest())
            .region(Region::new(config.region))
            .credentials_provider(credentials)
            .retry_config(RetryConfig::disabled())
    }

    /// A client for `bucket`, with every key under `prefix`, over the SDK configuration `sdk`.
    fn with_sdk_config(bucket: String, prefix: String, sdk: Config) -> Self {
        Self(SdkBucket {
            bucket,
            check_before_delete: false,
            client: Client::from_conf(sdk),
            prefix,
        })
    }
}

#[cfg(feature = "s3")]
impl Bucket for S3Bucket {
    #[inline]
    fn delete(
        &self,
        path: &Path,
        condition: &Condition,
    ) -> impl Future<Output = Result<()>> + Send {
        self.0.delete(path, condition)
    }

    #[inline]
    fn get(&self, path: &Path) -> impl Future<Output = Result<(Vec<u8>, RevisionToken)>> + Send {
        self.0.get(path)
    }

    #[inline]
    fn guarantees(&self) -> Guarantees {
        sdk_guarantees(true)
    }

    #[inline]
    fn head(&self, path: &Path) -> impl Future<Output = Result<FileMetadata>> + Send {
        self.0.head(path)
    }

    #[inline]
    fn list(
        &self,
        folder: &Path,
        depth: Depth,
    ) -> impl Future<Output = Result<Vec<WalkEntry>>> + Send {
        self.0.list(folder, depth)
    }

    #[inline]
    fn put(
        &self,
        path: &Path,
        bytes: &[u8],
        condition: &Condition,
    ) -> impl Future<Output = Result<RevisionToken>> + Send {
        self.0.put(path, bytes, condition)
    }
}

/// An SDK failure that reads as what the answer or the transport said.
#[cfg(any(feature = "minio", feature = "s3"))]
#[derive(Debug)]
struct SdkFailure {
    /// The status, S3 code and message of the answer, or the transport's causes.
    message: String,
    /// The SDK's own error.
    source: Box<dyn Error + Send + Sync>,
}

#[cfg(any(feature = "minio", feature = "s3"))]
impl fmt::Display for SdkFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

#[cfg(any(feature = "minio", feature = "s3"))]
impl Error for SdkFailure {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(self.source.as_ref())
    }
}

/// The error for an SDK failure: 404 is `NotFound` unless its S3 code is `NoSuchBucket`, 412 is
/// `PreconditionFailed`, anything else is `Other`.
#[cfg(any(feature = "minio", feature = "s3"))]
fn classify<E>(error: SdkError<E>) -> StorageError
where
    E: Error + ProvideErrorMetadata + Send + Sync + 'static,
{
    match error
        .raw_response()
        .map(|response| response.status().as_u16())
    {
        Some(404)
            if error
                .as_service_error()
                .and_then(ProvideErrorMetadata::code)
                != Some("NoSuchBucket") =>
        {
            StorageError::from_kind(ErrorKind::NotFound)
        }
        Some(412) => StorageError::from_kind(ErrorKind::PreconditionFailed),
        _ => StorageError::new(
            ErrorKind::Other,
            SdkFailure {
                message: describe(&error),
                source: Box::new(error),
            },
        ),
    }
}

/// What `error` says: the status, S3 code and message of a service's answer, or else each cause in turn.
#[cfg(any(feature = "minio", feature = "s3"))]
fn describe<E>(error: &SdkError<E>) -> String
where
    E: Error + ProvideErrorMetadata + 'static,
{
    if let (Some(service), Some(response)) = (error.as_service_error(), error.raw_response()) {
        let status = response.status().as_u16();
        match (service.code(), service.message()) {
            (Some(code), Some(text)) => format!("{status} {code}: {text}"),
            (Some(code), None) => format!("{status} {code}"),
            (None, Some(text)) => format!("{status}: {text}"),
            (None, None) => status.to_string(),
        }
    } else {
        let top: &(dyn Error + 'static) = error;
        iter::successors(Some(top), |&cause| cause.source())
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(": ")
    }
}

/// The error for an answer that should carry an `ETag` header and does not.
#[cfg(any(feature = "minio", feature = "s3"))]
fn missing_etag() -> StorageError {
    StorageError::new(ErrorKind::Other, "the answer carries no ETag header")
}

/// What the SDK clients promise: only the one-step writes, and the delete when `S3` checks `If-Match`.
#[cfg(any(feature = "minio", feature = "s3"))]
const fn sdk_guarantees(conditional_delete: bool) -> Guarantees {
    let writes = Guarantees::none()
        .with(Guarantee::ConditionalWriteInOneStep)
        .with(Guarantee::CreateOnlyWriteInOneStep);
    if conditional_delete {
        writes.with(Guarantee::ConditionalDeleteInOneStep)
    } else {
        writes
    }
}

#[cfg(test)]
mod core_tests;

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "Tests use unwrap for brevity; panics indicate test failure"
)]
mod tests;

#[cfg(any(minio_tests, s3_tests))]
#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "Tests use unwrap for brevity; panics indicate test failure"
)]
mod parity_tests;
