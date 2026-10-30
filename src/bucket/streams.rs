//! Streams over the S3 API: ranged reads and multipart uploads, holding at most one part in memory.

use super::paths::key_of;
use super::{Condition, RevisionToken, SdkBucket, classify, for_condition, missing_etag};
use crate::error::{ErrorKind, Result, StorageError};
use aws_sdk_s3::{
    primitives::ByteStream,
    types::{CompletedMultipartUpload, CompletedPart},
};
use core::future::Future;
use core::mem;
use core::pin::Pin;
use core::task::{Context, Poll, ready};
use futures_io::{AsyncRead, AsyncWrite};
use std::io;
use std::path::Path;
use tracing::error;

/// The size of each ranged read and of each multipart part: 8 MiB, the AWS command-line tool's default.
const PART_BYTES: u64 = 8 * 1024 * 1024;

/// `PART_BYTES` as a buffer length.
const PART_LEN: usize = 8 * 1024 * 1024;

/// The smallest object S3 accepts as a copied part that is not the last.
const COPY_MIN_BYTES: u64 = 5 * 1024 * 1024;

/// A request in flight.
type Flight<'bucket, T> = Pin<Box<dyn Future<Output = Result<T>> + Send + 'bucket>>;

/// What a request of a [`MultipartWriter`] achieved.
enum Step {
    /// The upload is complete or the object is stored.
    Finished,
    /// A part is stored, in the upload with this id.
    Part {
        /// The part, as `CompleteMultipartUpload` lists it.
        part: Box<CompletedPart>,
        /// The upload the part belongs to.
        upload_id: String,
    },
}

/// Reads an object in ranges of `PART_BYTES`, one range in memory at a time.
pub struct RangedReader<'bucket> {
    /// The bucket the object is in.
    bucket: &'bucket SdkBucket,
    /// The entity tag every range must still match.
    etag: String,
    /// The object's key.
    key: String,
    /// The object's length.
    length: u64,
    /// The next byte to request.
    next: u64,
    /// How much of `range` was served.
    offset: usize,
    /// The range being fetched.
    pending: Option<Flight<'bucket, Vec<u8>>>,
    /// The range being served.
    range: Vec<u8>,
}

impl AsyncRead for RangedReader<'_> {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<io::Result<usize>> {
        let this = &mut *self;
        loop {
            if let Some(rest) = this
                .range
                .get(this.offset..)
                .filter(|rest| !rest.is_empty())
            {
                let count = rest.len().min(buf.len());
                if let (Some(target), Some(source)) = (buf.get_mut(..count), rest.get(..count)) {
                    target.copy_from_slice(source);
                }
                this.offset = this.offset.saturating_add(count);
                return Poll::Ready(Ok(count));
            }
            if let Some(pending) = this.pending.as_mut() {
                let outcome = ready!(pending.as_mut().poll(cx));
                this.pending = None;
                let bytes = outcome.map_err(io::Error::from)?;
                if bytes.is_empty() {
                    return Poll::Ready(Err(io::ErrorKind::UnexpectedEof.into()));
                }
                this.range = bytes;
                this.offset = 0;
            } else if this.next >= this.length {
                return Poll::Ready(Ok(0));
            } else {
                let last = this.length.min(this.next.saturating_add(PART_BYTES));
                this.pending = Some(Box::pin(fetch_range(
                    this.bucket,
                    this.key.clone(),
                    this.etag.clone(),
                    this.next,
                    last.saturating_sub(1),
                )));
                this.next = last;
            }
        }
    }
}

/// Uploads in parts of `PART_BYTES`; `close` completes the upload.
pub struct MultipartWriter<'bucket> {
    /// The bucket the object goes to.
    bucket: &'bucket SdkBucket,
    /// The bytes not yet sent, at most `PART_BYTES`.
    buffer: Vec<u8>,
    /// Whether no request is left to make: closed, or failed.
    closed: bool,
    /// What the request that makes the object visible waits for.
    condition: Condition,
    /// Whether a request failed.
    failed: bool,
    /// The object's key.
    key: String,
    /// The parts stored so far.
    parts: Vec<CompletedPart>,
    /// The request in flight.
    pending: Option<Flight<'bucket, Step>>,
    /// The multipart upload, once the first part is sent.
    upload_id: Option<String>,
}

impl<'bucket> MultipartWriter<'bucket> {
    /// A writer whose first bytes are `buffer`, over an upload that already holds `parts`, if any.
    fn new(
        bucket: &'bucket SdkBucket,
        key: String,
        condition: Condition,
        buffer: Vec<u8>,
        upload: Option<(String, Vec<CompletedPart>)>,
    ) -> Self {
        let (upload_id, parts) = match upload {
            Some((id, parts)) => (Some(id), parts),
            None => (None, Vec::new()),
        };
        Self {
            bucket,
            buffer,
            closed: false,
            condition,
            failed: false,
            key,
            parts,
            pending: None,
            upload_id,
        }
    }

    /// Drive the request in flight and record its outcome.
    fn poll_pending(&mut self, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let Some(pending) = self.pending.as_mut() else {
            return Poll::Ready(Ok(()));
        };
        let outcome = ready!(pending.as_mut().poll(cx));
        self.pending = None;
        match outcome {
            Ok(Step::Finished) => self.closed = true,
            Ok(Step::Part { part, upload_id }) => {
                self.parts.push(*part);
                self.upload_id = Some(upload_id);
            }
            Err(error) => {
                self.closed = true;
                self.failed = true;
                return Poll::Ready(Err(error.into()));
            }
        }
        Poll::Ready(Ok(()))
    }

    /// Start the request that stores the bytes left and makes the object visible.
    fn start_finish(&mut self) {
        let bucket = self.bucket;
        let key = self.key.clone();
        let upload_id = self.upload_id.take();
        let parts = mem::take(&mut self.parts);
        let tail = mem::take(&mut self.buffer);
        let condition = self.condition.clone();
        self.pending = Some(Box::pin(async move {
            finish_upload(bucket, key, upload_id, parts, tail, &condition).await
        }));
    }

    /// Start sending the full buffer as the next part.
    fn start_part(&mut self) {
        let bucket = self.bucket;
        let key = self.key.clone();
        let upload_id = self.upload_id.clone();
        let number = part_number(self.parts.len());
        let body = mem::take(&mut self.buffer);
        self.pending = Some(Box::pin(async move {
            let (id, part) = send_part(bucket, &key, upload_id, number, body).await?;
            Ok(Step::Part {
                part: Box::new(part),
                upload_id: id,
            })
        }));
    }
}

impl AsyncWrite for MultipartWriter<'_> {
    fn poll_close(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let this = &mut *self;
        loop {
            ready!(this.poll_pending(cx))?;
            if this.closed {
                return Poll::Ready(if this.failed {
                    Err(io::Error::other("the upload failed"))
                } else {
                    Ok(())
                });
            }
            this.start_finish();
        }
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        self.poll_pending(cx)
    }

    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let this = &mut *self;
        loop {
            ready!(this.poll_pending(cx))?;
            if this.closed {
                return Poll::Ready(Err(io::Error::other("the stream is closed")));
            }
            if this.buffer.len() < PART_LEN {
                break;
            }
            this.start_part();
        }
        let room = PART_LEN.saturating_sub(this.buffer.len());
        let count = room.min(buf.len());
        if let Some(chunk) = buf.get(..count) {
            this.buffer.extend_from_slice(chunk);
        }
        Poll::Ready(Ok(count))
    }
}

impl Drop for MultipartWriter<'_> {
    fn drop(&mut self) {
        if !self.closed {
            error!(key = %self.key, "a write stream was dropped without close; its bytes are not stored");
        }
    }
}

/// Send `AbortMultipartUpload`; a failure is not reported, since the caller already has an error.
async fn abort(bucket: &SdkBucket, key: &str, upload_id: &str) {
    drop(
        bucket
            .client
            .abort_multipart_upload()
            .bucket(&bucket.bucket)
            .key(key)
            .upload_id(upload_id)
            .send()
            .await,
    );
}

/// A writer that appends to the file at `path`, keeping the bytes it has and adding the new ones.
pub async fn append_stream<'bucket>(
    bucket: &'bucket SdkBucket,
    path: &Path,
) -> Result<Box<dyn AsyncWrite + Send + Unpin + 'bucket>> {
    Ok(Box::new(append_writer(bucket, path).await?))
}

/// The writer `append_stream` boxes.
async fn append_writer<'bucket>(
    bucket: &'bucket SdkBucket,
    path: &Path,
) -> Result<MultipartWriter<'bucket>> {
    bucket.check_put(path).await?;
    let key = key_of(&bucket.prefix, path)?;
    let current = match bucket.head(path).await {
        Ok(found) => found,
        Err(error) if error.kind() == ErrorKind::NotFound => {
            return Ok(MultipartWriter::new(
                bucket,
                key,
                Condition::Absent,
                Vec::new(),
                None,
            ));
        }
        Err(error) => return Err(error),
    };
    let tag = current.revision.ok_or_else(missing_etag)?;
    let condition = Condition::Matches(tag.clone());
    if current.len < COPY_MIN_BYTES {
        let buffer = match current.len.checked_sub(1) {
            Some(last) => {
                fetch_range(bucket, key.clone(), tag.as_str().to_owned(), 0, last).await?
            }
            None => Vec::new(),
        };
        return Ok(MultipartWriter::new(bucket, key, condition, buffer, None));
    }
    let upload_id = create_upload(bucket, &key).await?;
    match copy_part(bucket, &key, &upload_id, &tag).await {
        Ok(part) => Ok(MultipartWriter::new(
            bucket,
            key,
            condition,
            Vec::new(),
            Some((upload_id, vec![part])),
        )),
        Err(error) => {
            abort(bucket, &key, &upload_id).await;
            Err(error)
        }
    }
}

/// `CompleteMultipartUpload` under `condition`.
async fn complete(
    bucket: &SdkBucket,
    key: &str,
    upload_id: &str,
    parts: Vec<CompletedPart>,
    condition: &Condition,
) -> Result<()> {
    let request = bucket
        .client
        .complete_multipart_upload()
        .bucket(&bucket.bucket)
        .key(key)
        .upload_id(upload_id)
        .multipart_upload(
            CompletedMultipartUpload::builder()
                .set_parts(Some(parts))
                .build(),
        );
    let conditioned = match condition.clone() {
        Condition::Absent => request.if_none_match("*"),
        Condition::Matches(token) => request.if_match(token.as_str()),
        Condition::None => request,
    };
    conditioned
        .send()
        .await
        .map_err(classify)
        .map(drop)
        .map_err(|error| for_condition(error, condition))
}

/// `UploadPartCopy` of the whole object at `key` as part 1, if it still has the tag `tag`.
async fn copy_part(
    bucket: &SdkBucket,
    key: &str,
    upload_id: &str,
    tag: &RevisionToken,
) -> Result<CompletedPart> {
    let copied = bucket
        .client
        .upload_part_copy()
        .bucket(&bucket.bucket)
        .key(key)
        .upload_id(upload_id)
        .part_number(1)
        .copy_source(super::percent_encode(&format!("{}/{key}", bucket.bucket)))
        .copy_source_if_match(tag.as_str())
        .send()
        .await
        .map_err(classify)?;
    let etag = copied
        .copy_part_result()
        .and_then(|result| result.e_tag())
        .ok_or_else(missing_etag)?;
    Ok(CompletedPart::builder().part_number(1).e_tag(etag).build())
}

/// `CreateMultipartUpload` for `key`, giving the upload's id.
async fn create_upload(bucket: &SdkBucket, key: &str) -> Result<String> {
    bucket
        .client
        .create_multipart_upload()
        .bucket(&bucket.bucket)
        .key(key)
        .send()
        .await
        .map_err(classify)?
        .upload_id()
        .map(str::to_owned)
        .ok_or_else(|| StorageError::new(ErrorKind::Other, "the answer carries no upload id"))
}

/// The bytes `from` to `to`, both included, of the object at `key` if it still has the tag `etag`.
async fn fetch_range(
    bucket: &SdkBucket,
    key: String,
    etag: String,
    from: u64,
    to: u64,
) -> Result<Vec<u8>> {
    let output = bucket
        .client
        .get_object()
        .bucket(&bucket.bucket)
        .key(key)
        .if_match(etag)
        .range(format!("bytes={from}-{to}"))
        .send()
        .await
        .map_err(classify)?;
    let collected = output
        .body
        .collect()
        .await
        .map_err(|error| StorageError::new(ErrorKind::Other, error))?;
    Ok(collected.into_bytes().to_vec())
}

/// Store `tail` and make the object visible: one `PutObject` when no upload started, else the
/// last part and `CompleteMultipartUpload`. A failure after the upload started aborts it.
async fn finish_upload(
    bucket: &SdkBucket,
    key: String,
    upload_id: Option<String>,
    mut parts: Vec<CompletedPart>,
    tail: Vec<u8>,
    condition: &Condition,
) -> Result<Step> {
    let Some(id) = upload_id else {
        bucket.put_key(&key, tail, condition).await?;
        return Ok(Step::Finished);
    };
    if !tail.is_empty() {
        let number = part_number(parts.len());
        match send_part(bucket, &key, Some(id.clone()), number, tail).await {
            Ok((_id, part)) => parts.push(part),
            Err(error) => return Err(error),
        }
    }
    if let Err(error) = complete(bucket, &key, &id, parts, condition).await {
        abort(bucket, &key, &id).await;
        return Err(error);
    }
    Ok(Step::Finished)
}

/// A reader over the file at `path`, in ranges of `PART_BYTES`.
pub async fn read_stream<'bucket>(
    bucket: &'bucket SdkBucket,
    path: &Path,
) -> Result<Box<dyn AsyncRead + Send + Unpin + 'bucket>> {
    let found = bucket.head(path).await?;
    if found.is_dir {
        return Err(StorageError::from_kind(ErrorKind::IsADirectory));
    }
    let etag = found.revision.ok_or_else(missing_etag)?;
    Ok(Box::new(RangedReader {
        bucket,
        etag: etag.as_str().to_owned(),
        key: key_of(&bucket.prefix, path)?,
        length: found.len,
        next: 0,
        offset: 0,
        pending: None,
        range: Vec::new(),
    }))
}

/// `UploadPart` of `body`, starting the upload first when `upload_id` is `None`. A failure
/// aborts the upload.
async fn send_part(
    bucket: &SdkBucket,
    key: &str,
    upload_id: Option<String>,
    number: i32,
    body: Vec<u8>,
) -> Result<(String, CompletedPart)> {
    let id = match upload_id {
        Some(id) => id,
        None => create_upload(bucket, key).await?,
    };
    let sent = bucket
        .client
        .upload_part()
        .bucket(&bucket.bucket)
        .key(key)
        .upload_id(&id)
        .part_number(number)
        .body(ByteStream::from(body))
        .send()
        .await
        .map_err(classify)
        .and_then(|output| {
            output
                .e_tag()
                .map(|etag| {
                    CompletedPart::builder()
                        .part_number(number)
                        .e_tag(etag)
                        .build()
                })
                .ok_or_else(missing_etag)
        });
    match sent {
        Ok(part) => Ok((id, part)),
        Err(error) => {
            abort(bucket, key, &id).await;
            Err(error)
        }
    }
}

/// A writer that stores its bytes at `path` under `condition` when closed.
pub async fn write_stream<'bucket>(
    bucket: &'bucket SdkBucket,
    path: &Path,
    condition: Condition,
) -> Result<Box<dyn AsyncWrite + Send + Unpin + 'bucket>> {
    bucket.check_put(path).await?;
    let key = key_of(&bucket.prefix, path)?;
    Ok(Box::new(MultipartWriter::new(
        bucket,
        key,
        condition,
        Vec::new(),
        None,
    )))
}

/// The number of the part after `stored` parts.
fn part_number(stored: usize) -> i32 {
    i32::try_from(stored.saturating_add(1)).unwrap_or(i32::MAX)
}
