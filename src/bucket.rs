//! Whole objects in one bucket of an S3-compatible object store, written and
//! deleted under conditions, with an in-memory twin for tests.
//!
//! Available with the `minio` or the `s3` feature. The `Bucket` trait can be
//! named and implemented only with the `async` feature as well.
//!
//! # What it is
//!
//! The module reads, writes, deletes and lists whole objects by key, and reads
//! an object's size and tag without its bytes. It has no ranges, no streaming,
//! no custom metadata, no copy and no versions. Every put and delete takes a
//! [`Condition`]:
//!
//! - [`Condition::None`]: no check. A put replaces, a delete removes.
//! - [`Condition::Absent`]: the put succeeds only if no object is at the key.
//! - [`Condition::ETag`]: the put or delete succeeds only if the object at the
//!   key has this entity tag.
//!
//! A put returns the new entity tag. A get returns the bytes with their tag
//! ([`Object`]), a head the size with the tag ([`Head`]). A failed condition is an [`ObjectError`] whose
//! [`kind`](ObjectError::kind) is [`ObjectErrorKind::PreconditionFailed`], the
//! same on every implementation.
//!
//! # The pieces
//!
//! - `Bucket`: the asynchronous trait with `get`, `head`, `put`, `delete` and
//!   `list`.
//!   Its methods return `impl Future + Send`, so it cannot be a trait object;
//!   take a generic `B: Bucket`.
//! - `MinioBucket` and `MinioConfig` (feature `minio`): a client for `MinIO`,
//!   `RustFS` and other servers that speak its API. Path-style addressing.
//!   `MinioBucket::ensure` creates the bucket when it is missing.
//! - `S3Bucket` and `S3Config` (feature `s3`): a client for AWS S3.
//!   `S3Bucket::ensure` creates the bucket when it is missing: private, in the
//!   client's region, encrypted with a KMS key when one is given.
//! - [`MemoryBucket`]: the same interface over a map in memory. No server, no
//!   network. Clones share the same objects.
//! - [`BlockingBucket`]: wraps any `Bucket` for code that is not async. It owns
//!   a tokio runtime.
//!
//! # When to use it
//!
//! - **Claim a lock or lease.** Put with `Condition::Absent`; only one caller
//!   wins. Renew or release it with the tag the put returned.
//! - **Edit a shared document.** Get, change the bytes, put with
//!   `Condition::ETag(object.etag)`. On `PreconditionFailed`, get again and
//!   retry. No update is lost.
//! - **Test with no server.** Write the code against a generic `B: Bucket`,
//!   run it on `MinioBucket` or `S3Bucket` in production and on
//!   [`MemoryBucket`] in tests. A test can play two clients with two clones.
//!
//! # Example
//!
//! Claim a key, then update it only while nobody else has changed it:
//!
//! ```
//! use os_shim::bucket::{BlockingBucket, Condition, MemoryBucket, ObjectErrorKind};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let shared = MemoryBucket::default();
//! let ours = BlockingBucket::new(shared.clone())?;
//! let theirs = BlockingBucket::new(shared)?;
//!
//! // Create if absent: the first caller wins, the second is refused.
//! let first_tag = ours.put("jobs/42", b"claimed by ours", &Condition::Absent)?;
//! let refused = theirs.put("jobs/42", b"claimed by theirs", &Condition::Absent);
//! assert_eq!(
//!     refused.unwrap_err().kind(),
//!     ObjectErrorKind::PreconditionFailed
//! );
//!
//! // Update if unchanged: the tag from the get says what we saw.
//! let seen = ours.get("jobs/42")?;
//! assert_eq!(seen.etag, first_tag);
//! ours.put("jobs/42", b"done", &Condition::ETag(seen.etag.clone()))?;
//!
//! // The same tag is now stale, so a second update with it is refused.
//! let stale = theirs.put("jobs/42", b"overwrite", &Condition::ETag(seen.etag));
//! assert_eq!(
//!     stale.unwrap_err().kind(),
//!     ObjectErrorKind::PreconditionFailed
//! );
//! assert_eq!(theirs.get("jobs/42")?.bytes, b"done");
//! # Ok(())
//! # }
//! ```
//!
//! # Things that bite
//!
//! - The clients never retry. A network failure or a 5xx is an
//!   [`ObjectErrorKind::Other`] error and the caller decides what to do.
//! - Credentials are a fixed access key and secret key. There is no session
//!   token and no environment or profile lookup.
//! - Entity tags differ between implementations. A tag from [`MemoryBucket`]
//!   means nothing to a real store, and the reverse.
//! - [`BlockingBucket`] panics when called from inside a tokio runtime.

#[macro_use]
mod interface;

use alloc::collections::BTreeMap;
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
use core::error::Error;
use core::fmt;
use core::future::ready;
use core::hash::{Hash as _, Hasher as _};
#[cfg(any(feature = "minio", feature = "s3"))]
use core::iter;
#[cfg(not(feature = "async"))]
use interface::Bucket;
use std::hash::DefaultHasher;
use std::io;
use std::sync::{PoisonError, RwLock};
use tokio::runtime::{Builder, Runtime};

#[cfg(feature = "async")]
declare_bucket!();

/// What a store says about one object without sending its bytes.
///
/// Returned by `head`. It cannot be built outside this crate.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Head {
    /// The `ETag` exactly as the store gave it, quotes included.
    pub etag: String,
    pub size: u64,
}

/// One object: its bytes and the entity tag the store gave it.
///
/// Returned by `get`. It cannot be built outside this crate. Pass `etag` back
/// in [`Condition::ETag`] to write or delete only if the object has not changed
/// since you read it.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Object {
    pub bytes: Vec<u8>,
    /// The `ETag` exactly as the store gave it, quotes included.
    pub etag: String,
}

/// What a put or a delete waits for.
///
/// Import the variants by path (`Condition::None`), not with `use
/// Condition::*`, where `None` hides `Option::None`. The enum is
/// `non_exhaustive`: a `match` outside this crate needs a `_` arm.
///
/// `Absent` on a delete deletes nothing: it succeeds on an empty key and fails
/// with `PreconditionFailed` if an object is there. `ETag` on an empty key fails
/// with `PreconditionFailed`, not `NotFound`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Condition {
    /// No object is at the key: `If-None-Match: *`.
    Absent,
    /// The object at the key has this entity tag: `If-Match`.
    ETag(String),
    /// Nothing: a put replaces and a delete removes whatever is there.
    None,
}

/// Which way a bucket operation failed.
///
/// Read it with [`ObjectError::kind`]. The enum is `non_exhaustive`: a `match`
/// outside this crate needs a `_` arm.
///
/// Only `get` and `head` return `NotFound`; a missing bucket is `Other`. Nothing is
/// retried for you, so a transient `Other` is yours to retry. After
/// `PreconditionFailed` the object is unchanged: get it again and decide.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ObjectErrorKind {
    /// No object at the key: HTTP 404.
    NotFound,
    /// Any other failure: the network, a missing bucket, a 403, a 5xx, an answer that cannot be read.
    Other,
    /// The condition does not hold: HTTP 412.
    PreconditionFailed,
}

/// Why a bucket operation failed.
///
/// Branch on [`kind`](Self::kind). The only public way to make one is
/// `From<io::Error>`, which gives kind [`ObjectErrorKind::Other`]. `Display`
/// shows the underlying error of an `Other` failure, else a fixed phrase for
/// the kind.
#[derive(Debug)]
pub struct ObjectError {
    /// Which way it failed.
    kind: ObjectErrorKind,
    /// The underlying failure of an `Other` error.
    source: Option<io::Error>,
}

impl ObjectError {
    /// Which way the operation failed.
    #[inline]
    #[must_use]
    pub const fn kind(&self) -> ObjectErrorKind {
        self.kind
    }

    /// A `NotFound` error.
    pub(crate) const fn not_found() -> Self {
        Self {
            kind: ObjectErrorKind::NotFound,
            source: None,
        }
    }

    /// A `PreconditionFailed` error.
    pub(crate) const fn precondition_failed() -> Self {
        Self {
            kind: ObjectErrorKind::PreconditionFailed,
            source: None,
        }
    }
}

impl From<io::Error> for ObjectError {
    #[inline]
    fn from(error: io::Error) -> Self {
        Self {
            kind: ObjectErrorKind::Other,
            source: Some(error),
        }
    }
}

impl fmt::Display for ObjectError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(error) = self.source.as_ref() {
            return write!(f, "{error}");
        }
        match self.kind {
            ObjectErrorKind::NotFound => f.write_str("no object at the key"),
            ObjectErrorKind::Other => f.write_str("the operation failed"),
            ObjectErrorKind::PreconditionFailed => f.write_str("the condition does not hold"),
        }
    }
}

impl Error for ObjectError {
    #[inline]
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source.as_ref().map(|error| {
            let source: &(dyn Error + 'static) = error;
            source
        })
    }
}

/// Objects in memory, for tests that reach no server.
///
/// It follows the same rules as the real clients: [`Condition`] and
/// [`ObjectErrorKind`] mean the same, and a condition is checked and applied in
/// one step. Make one with `MemoryBucket::default()`. Clones share the same
/// objects, so a test can play two clients with two clones.
///
/// Its entity tag is a hash of the bytes in double quotes, so the same bytes
/// give the same tag, and putting identical bytes again does not change it.
/// Tags from a real store are not comparable. `list` returns keys sorted.
///
/// ```
/// use os_shim::bucket::{BlockingBucket, Condition, MemoryBucket, ObjectErrorKind};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let bucket = BlockingBucket::new(MemoryBucket::default())?;
///
/// let tag = bucket.put("config.json", b"{\"v\":1}", &Condition::Absent)?;
/// assert!(bucket
///     .put("config.json", b"{\"v\":2}", &Condition::Absent)
///     .is_err());
///
/// let object = bucket.get("config.json")?;
/// assert_eq!(object.etag, tag);
///
/// bucket.put("config.json", b"{\"v\":2}", &Condition::ETag(tag.clone()))?;
/// let stale = bucket.delete("config.json", &Condition::ETag(tag));
/// assert_eq!(
///     stale.unwrap_err().kind(),
///     ObjectErrorKind::PreconditionFailed
/// );
///
/// assert_eq!(bucket.list("config")?, ["config.json"]);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Default)]
pub struct MemoryBucket {
    /// The bytes at each key.
    objects: Arc<RwLock<BTreeMap<String, Vec<u8>>>>,
}

impl MemoryBucket {
    /// Delete the object at `key` when `condition` holds.
    fn delete_now(&self, key: &str, condition: &Condition) -> Result<(), ObjectError> {
        let mut objects = self.objects.write().unwrap_or_else(PoisonError::into_inner);
        if !holds(condition, objects.get(key)) {
            return Err(ObjectError::precondition_failed());
        }
        objects.remove(key);
        drop(objects);
        Ok(())
    }

    /// The object at `key`.
    fn get_now(&self, key: &str) -> Result<Object, ObjectError> {
        let bytes = self
            .objects
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .get(key)
            .cloned()
            .ok_or_else(ObjectError::not_found)?;
        let etag = tag_of(&bytes);
        Ok(Object { bytes, etag })
    }

    /// The size and tag of the object at `key`, read without copying its bytes.
    fn head_now(&self, key: &str) -> Result<Head, ObjectError> {
        let (etag, length) = self
            .objects
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .get(key)
            .map(|bytes| (tag_of(bytes), bytes.len()))
            .ok_or_else(ObjectError::not_found)?;
        Ok(Head {
            etag,
            size: u64::try_from(length).map_err(io::Error::other)?,
        })
    }

    /// Every key that starts with `prefix`, in order.
    fn list_now(&self, prefix: &str) -> Vec<String> {
        let objects = self.objects.read().unwrap_or_else(PoisonError::into_inner);
        objects
            .keys()
            .filter(|key| key.starts_with(prefix))
            .cloned()
            .collect()
    }

    /// Write `bytes` at `key` when `condition` holds, and give their tag.
    fn put_now(
        &self,
        key: &str,
        bytes: &[u8],
        condition: &Condition,
    ) -> Result<String, ObjectError> {
        let mut objects = self.objects.write().unwrap_or_else(PoisonError::into_inner);
        if !holds(condition, objects.get(key)) {
            return Err(ObjectError::precondition_failed());
        }
        objects.insert(key.to_owned(), bytes.to_vec());
        drop(objects);
        Ok(tag_of(bytes))
    }
}

impl Bucket for MemoryBucket {
    #[inline]
    fn delete(
        &self,
        key: &str,
        condition: &Condition,
    ) -> impl Future<Output = Result<(), ObjectError>> + Send {
        ready(self.delete_now(key, condition))
    }

    #[inline]
    fn get(&self, key: &str) -> impl Future<Output = Result<Object, ObjectError>> + Send {
        ready(self.get_now(key))
    }

    #[inline]
    fn head(&self, key: &str) -> impl Future<Output = Result<Head, ObjectError>> + Send {
        ready(self.head_now(key))
    }

    #[inline]
    fn list(&self, prefix: &str) -> impl Future<Output = Result<Vec<String>, ObjectError>> + Send {
        ready(Ok(self.list_now(prefix)))
    }

    #[inline]
    fn put(
        &self,
        key: &str,
        bytes: &[u8],
        condition: &Condition,
    ) -> impl Future<Output = Result<String, ObjectError>> + Send {
        ready(self.put_now(key, bytes, condition))
    }
}

/// A bucket for synchronous code: each call runs on a tokio runtime it owns.
///
/// A call from code already inside a tokio runtime panics; such code uses
/// the asynchronous interface directly.
///
/// Wrap [`MemoryBucket`], `MinioBucket` or `S3Bucket`. The methods take the
/// same arguments and return the same results as the `Bucket` trait, without
/// the futures. It cannot be cloned and gives no access to the bucket inside.
#[derive(Debug)]
pub struct BlockingBucket<B> {
    /// The bucket each call reaches.
    bucket: B,
    /// The current-thread runtime each call blocks on.
    runtime: Runtime,
}

impl<B: Bucket> BlockingBucket<B> {
    /// Delete the object at `key` when `condition` holds.
    ///
    /// # Errors
    ///
    /// `PreconditionFailed` when `condition` does not hold, the object unchanged.
    /// `Other` when the store cannot be reached or refuses.
    ///
    /// # Panics
    ///
    /// Panics when called from inside a tokio runtime.
    #[inline]
    pub fn delete(&self, key: &str, condition: &Condition) -> Result<(), ObjectError> {
        self.runtime.block_on(self.bucket.delete(key, condition))
    }

    /// The object at `key`.
    ///
    /// # Errors
    ///
    /// `NotFound` when no object is at `key`.
    /// `Other` when the store cannot be reached or refuses.
    ///
    /// # Panics
    ///
    /// Panics when called from inside a tokio runtime.
    #[inline]
    pub fn get(&self, key: &str) -> Result<Object, ObjectError> {
        self.runtime.block_on(self.bucket.get(key))
    }

    /// The size and entity tag of the object at `key`, without its bytes.
    ///
    /// # Errors
    ///
    /// `NotFound` when no object is at `key`.
    /// `Other` when the store cannot be reached or the bucket is missing.
    ///
    /// # Panics
    ///
    /// Panics when called from inside a tokio runtime.
    #[inline]
    pub fn head(&self, key: &str) -> Result<Head, ObjectError> {
        self.runtime.block_on(self.bucket.head(key))
    }

    /// Every key that starts with `prefix`, every page of the listing read.
    ///
    /// # Errors
    ///
    /// `Other` when a page cannot be read.
    ///
    /// # Panics
    ///
    /// Panics when called from inside a tokio runtime.
    #[inline]
    pub fn list(&self, prefix: &str) -> Result<Vec<String>, ObjectError> {
        self.runtime.block_on(self.bucket.list(prefix))
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

    /// Write `bytes` whole at `key` when `condition` holds, and give the new entity tag.
    ///
    /// # Errors
    ///
    /// `PreconditionFailed` when `condition` does not hold, the object unchanged.
    /// `Other` when the store cannot be reached or refuses.
    ///
    /// # Panics
    ///
    /// Panics when called from inside a tokio runtime.
    #[inline]
    pub fn put(
        &self,
        key: &str,
        bytes: &[u8],
        condition: &Condition,
    ) -> Result<String, ObjectError> {
        self.runtime
            .block_on(self.bucket.put(key, bytes, condition))
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
}

#[cfg(any(feature = "minio", feature = "s3"))]
impl SdkBucket {
    /// Whether the bucket exists: `HEAD` on it answers 200, or 404 when it is missing.
    async fn bucket_exists(&self) -> Result<bool, ObjectError> {
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
    async fn current_tag(&self, key: &str) -> Result<Option<String>, ObjectError> {
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
            Err(error) if error.kind() == ObjectErrorKind::NotFound => {
                self.object_is_missing().await
            }
            Err(error) => Err(error),
        }
    }

    /// Delete the object at `key` when `condition` holds.
    async fn delete(&self, key: &str, condition: &Condition) -> Result<(), ObjectError> {
        let mut request = self.client.delete_object().bucket(&self.bucket).key(key);
        match condition.clone() {
            Condition::None => {}
            Condition::Absent => {
                return match self.current_tag(key).await? {
                    None => Ok(()),
                    Some(_) => Err(ObjectError::precondition_failed()),
                };
            }
            Condition::ETag(tag) => {
                if self.check_before_delete && self.current_tag(key).await?.as_ref() != Some(&tag) {
                    return Err(ObjectError::precondition_failed());
                }
                request = request.if_match(tag);
            }
        }
        match request.send().await.map_err(classify) {
            Ok(_) => Ok(()),
            Err(error)
                if error.kind() == ObjectErrorKind::NotFound
                    && matches!(*condition, Condition::ETag(_)) =>
            {
                Err(ObjectError::precondition_failed())
            }
            Err(error) => Err(error),
        }
    }

    /// The object at `key`.
    async fn get(&self, key: &str) -> Result<Object, ObjectError> {
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
            .map_err(|error| ObjectError::from(io::Error::other(error)))?;
        Ok(Object {
            bytes: collected.into_bytes().to_vec(),
            etag,
        })
    }

    /// The size and tag of the object at `key`, from a `HEAD` on it.
    async fn head(&self, key: &str) -> Result<Head, ObjectError> {
        match self
            .client
            .head_object()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await
            .map_err(classify)
        {
            Ok(head) => Ok(Head {
                etag: head.e_tag().map(str::to_owned).ok_or_else(missing_etag)?,
                size: head
                    .content_length()
                    .and_then(|length| u64::try_from(length).ok())
                    .ok_or_else(|| {
                        ObjectError::from(io::Error::other(
                            "the answer carries no Content-Length header",
                        ))
                    })?,
            }),
            Err(error) if error.kind() == ObjectErrorKind::NotFound => {
                self.object_is_missing().await?;
                Err(ObjectError::not_found())
            }
            Err(error) => Err(error),
        }
    }

    /// Every key that starts with `prefix`, every page of the listing read.
    async fn list(&self, prefix: &str) -> Result<Vec<String>, ObjectError> {
        let mut keys = Vec::new();
        let mut pages = self
            .client
            .list_objects_v2()
            .bucket(&self.bucket)
            .prefix(prefix)
            .into_paginator()
            .send();
        while let Some(answer) = pages.next().await {
            let page = answer.map_err(classify)?;
            keys.extend(
                page.contents()
                    .iter()
                    .filter_map(|object| object.key().map(str::to_owned)),
            );
        }
        Ok(keys)
    }

    /// `Ok(None)` when the bucket exists, since a `HEAD` 404 carries no S3 code to say which of
    /// the object and the bucket is missing.
    async fn object_is_missing(&self) -> Result<Option<String>, ObjectError> {
        if self.bucket_exists().await? {
            return Ok(None);
        }
        Err(ObjectError::from(io::Error::other(format!(
            "404: the bucket {} does not exist",
            self.bucket
        ))))
    }

    /// Write `bytes` whole at `key` when `condition` holds, and give the new entity tag.
    async fn put(
        &self,
        key: &str,
        bytes: &[u8],
        condition: &Condition,
    ) -> Result<String, ObjectError> {
        let unconditional = self
            .client
            .put_object()
            .bucket(&self.bucket)
            .key(key)
            .body(ByteStream::from(bytes.to_vec()));
        let request = match condition.clone() {
            Condition::None => unconditional,
            Condition::Absent => unconditional.if_none_match("*"),
            Condition::ETag(tag) => unconditional.if_match(tag),
        };
        match request.send().await.map_err(classify) {
            Ok(output) => output.e_tag().map(str::to_owned).ok_or_else(missing_etag),
            Err(error)
                if error.kind() == ObjectErrorKind::NotFound
                    && matches!(*condition, Condition::ETag(_)) =>
            {
                Err(ObjectError::precondition_failed())
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
    pub region: String,
    pub secret_key: String,
}

#[cfg(feature = "minio")]
impl MinioConfig {
    /// A configuration from its five parts.
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
            region,
            secret_key,
        }
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
            .field("region", &self.region)
            .field("secret_key", &"<redacted>")
            .finish()
    }
}

/// A client for a server that speaks `MinIO`'s API: `MinIO`, `RustFS` and the like.
///
/// A delete with an entity-tag condition reads the object's tag first, because
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
    pub async fn ensure(&self) -> Result<(), ObjectError> {
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
        })
    }
}

#[cfg(feature = "minio")]
impl Bucket for MinioBucket {
    #[inline]
    fn delete(
        &self,
        key: &str,
        condition: &Condition,
    ) -> impl Future<Output = Result<(), ObjectError>> + Send {
        self.0.delete(key, condition)
    }

    #[inline]
    fn get(&self, key: &str) -> impl Future<Output = Result<Object, ObjectError>> + Send {
        self.0.get(key)
    }

    #[inline]
    fn head(&self, key: &str) -> impl Future<Output = Result<Head, ObjectError>> + Send {
        self.0.head(key)
    }

    #[inline]
    fn list(&self, prefix: &str) -> impl Future<Output = Result<Vec<String>, ObjectError>> + Send {
        self.0.list(prefix)
    }

    #[inline]
    fn put(
        &self,
        key: &str,
        bytes: &[u8],
        condition: &Condition,
    ) -> impl Future<Output = Result<String, ObjectError>> + Send {
        self.0.put(key, bytes, condition)
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
    pub region: String,
    pub secret_key: String,
}

#[cfg(feature = "s3")]
impl S3Config {
    /// A configuration from its four parts.
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
            region,
            secret_key,
        }
    }
}

#[cfg(feature = "s3")]
impl fmt::Debug for S3Config {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("S3Config")
            .field("access_key", &self.access_key)
            .field("bucket", &self.bucket)
            .field("region", &self.region)
            .field("secret_key", &"<redacted>")
            .finish()
    }
}

/// A client for an AWS S3 bucket, over AWS's SDK.
///
/// A delete with an entity-tag condition sends `If-Match` and lets AWS check it.
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
    pub async fn ensure(&self, kms_key_arn: Option<&str>) -> Result<(), ObjectError> {
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
            .map_err(|error| ObjectError::from(io::Error::other(error)))?;
        let encryption = ServerSideEncryptionConfiguration::builder()
            .rules(
                ServerSideEncryptionRule::builder()
                    .apply_server_side_encryption_by_default(by_default)
                    .bucket_key_enabled(true)
                    .build(),
            )
            .build()
            .map_err(|error| ObjectError::from(io::Error::other(error)))?;
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
        Self::with_sdk_config(config.bucket.clone(), Self::sdk_config(config).build())
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

    /// A client for `bucket` over the SDK configuration `sdk`.
    fn with_sdk_config(bucket: String, sdk: Config) -> Self {
        Self(SdkBucket {
            bucket,
            check_before_delete: false,
            client: Client::from_conf(sdk),
        })
    }
}

#[cfg(feature = "s3")]
impl Bucket for S3Bucket {
    #[inline]
    fn delete(
        &self,
        key: &str,
        condition: &Condition,
    ) -> impl Future<Output = Result<(), ObjectError>> + Send {
        self.0.delete(key, condition)
    }

    #[inline]
    fn get(&self, key: &str) -> impl Future<Output = Result<Object, ObjectError>> + Send {
        self.0.get(key)
    }

    #[inline]
    fn head(&self, key: &str) -> impl Future<Output = Result<Head, ObjectError>> + Send {
        self.0.head(key)
    }

    #[inline]
    fn list(&self, prefix: &str) -> impl Future<Output = Result<Vec<String>, ObjectError>> + Send {
        self.0.list(prefix)
    }

    #[inline]
    fn put(
        &self,
        key: &str,
        bytes: &[u8],
        condition: &Condition,
    ) -> impl Future<Output = Result<String, ObjectError>> + Send {
        self.0.put(key, bytes, condition)
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
fn classify<E>(error: SdkError<E>) -> ObjectError
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
            ObjectError::not_found()
        }
        Some(412) => ObjectError::precondition_failed(),
        _ => ObjectError::from(io::Error::other(SdkFailure {
            message: describe(&error),
            source: Box::new(error),
        })),
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

/// Whether `condition` holds for the object now at the key, if any.
fn holds(condition: &Condition, current: Option<&Vec<u8>>) -> bool {
    match (condition.clone(), current) {
        (Condition::None, _) | (Condition::Absent, None) => true,
        (Condition::ETag(tag), Some(bytes)) => tag == tag_of(bytes),
        (Condition::Absent, Some(_)) | (Condition::ETag(_), None) => false,
    }
}

/// The error for an answer that should carry an `ETag` header and does not.
#[cfg(any(feature = "minio", feature = "s3"))]
fn missing_etag() -> ObjectError {
    ObjectError::from(io::Error::other("the answer carries no ETag header"))
}

/// The entity tag of `bytes`: sixteen hex digits in double quotes, the same for the same bytes.
fn tag_of(bytes: &[u8]) -> String {
    let mut hasher = DefaultHasher::new();
    bytes.hash(&mut hasher);
    format!("\"{:016x}\"", hasher.finish())
}

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
