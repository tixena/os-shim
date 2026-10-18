//! Whole objects in one bucket of an object store: an asynchronous interface,
//! an in-memory bucket for tests, and a synchronous wrapper.

#[macro_use]
mod interface;

use alloc::collections::BTreeMap;
use alloc::sync::Arc;
#[cfg(feature = "minio")]
use aws_sdk_s3::{
    Client, Config,
    config::{BehaviorVersion, Credentials, Region, retry::RetryConfig},
    error::SdkError,
    primitives::ByteStream,
};
use core::error::Error;
use core::fmt;
use core::future::ready;
use core::hash::{Hash as _, Hasher as _};
#[cfg(not(feature = "async"))]
use interface::Bucket;
use std::hash::DefaultHasher;
use std::io;
use std::sync::{PoisonError, RwLock};
use tokio::runtime::{Builder, Runtime};

#[cfg(feature = "async")]
declare_bucket!();

/// One object: its bytes and the entity tag the store gave it.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Object {
    /// The object's content.
    pub bytes: Vec<u8>,
    /// The `ETag` exactly as the store gave it, quotes included.
    pub etag: String,
}

/// What a put or a delete waits for.
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ObjectErrorKind {
    /// No object at the key: HTTP 404.
    NotFound,
    /// Any other failure: the network, a 403, a 5xx, an answer that cannot be read.
    Other,
    /// The condition does not hold: HTTP 412.
    PreconditionFailed,
}

/// Why a bucket operation failed.
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
    #[inline]
    pub fn delete(&self, key: &str, condition: &Condition) -> Result<(), ObjectError> {
        self.runtime.block_on(self.bucket.delete(key, condition))
    }

    /// The object at `key`.
    ///
    /// # Errors
    ///
    /// `NotFound` when no object is at `key`.
    #[inline]
    pub fn get(&self, key: &str) -> Result<Object, ObjectError> {
        self.runtime.block_on(self.bucket.get(key))
    }

    /// Every key that starts with `prefix`, every page of the listing read.
    ///
    /// # Errors
    ///
    /// `Other` when a page cannot be read.
    #[inline]
    pub fn list(&self, prefix: &str) -> Result<Vec<String>, ObjectError> {
        self.runtime.block_on(self.bucket.list(prefix))
    }

    /// A blocking bucket over `bucket`.
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

/// A bucket reached through AWS's S3 SDK, which `MinioBucket` wraps.
#[cfg(feature = "minio")]
#[derive(Debug)]
struct SdkBucket {
    /// The bucket every operation reaches.
    bucket: String,
    /// Whether a conditional delete reads the object's tag before it sends the delete.
    check_before_delete: bool,
    /// The SDK client every request goes through.
    client: Client,
}

#[cfg(feature = "minio")]
impl SdkBucket {
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
            Err(error) if error.kind() == ObjectErrorKind::NotFound => Ok(None),
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
        let output = request.send().await.map_err(classify)?;
        output.e_tag().map(str::to_owned).ok_or_else(missing_etag)
    }
}

/// Where a `MinIO`-compatible server is, and the keys that sign every request.
#[cfg(feature = "minio")]
#[derive(Clone)]
#[non_exhaustive]
pub struct MinioConfig {
    /// The access key.
    pub access_key: String,
    /// The bucket every operation reaches.
    pub bucket: String,
    /// The server's address: scheme, host and port.
    pub endpoint: String,
    /// The region requests are signed for, such as `us-east-1`.
    pub region: String,
    /// The secret key; `Debug` prints it as `<redacted>`.
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
#[cfg(feature = "minio")]
#[derive(Debug)]
pub struct MinioBucket(
    /// The SDK bucket every call goes through.
    SdkBucket,
);

#[cfg(feature = "minio")]
impl MinioBucket {
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

/// The error for an SDK failure: 404 is `NotFound`, 412 is `PreconditionFailed`, anything else is `Other`.
#[cfg(feature = "minio")]
fn classify<E>(error: SdkError<E>) -> ObjectError
where
    E: Error + Send + Sync + 'static,
{
    match error
        .raw_response()
        .map(|response| response.status().as_u16())
    {
        Some(404) => ObjectError::not_found(),
        Some(412) => ObjectError::precondition_failed(),
        _ => ObjectError::from(io::Error::other(error)),
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
#[cfg(feature = "minio")]
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

#[cfg(minio_tests)]
#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "Tests use unwrap for brevity; panics indicate test failure"
)]
mod parity_tests;
