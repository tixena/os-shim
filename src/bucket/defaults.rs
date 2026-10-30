//! The bodies the `Bucket` trait's default methods run, over the required methods only.

use super::paths::lexical;
use super::{Bucket, Condition, Depth, RevisionToken};
use crate::WalkEntry;
use crate::error::{ErrorKind, Result, StorageError};
use core::future::Future;
use core::mem;
use core::pin::Pin;
use core::task::{Context, Poll};
use futures_io::{AsyncRead, AsyncWrite};
use std::io;
use std::path::{Path, PathBuf};

/// What a [`BufferedWriter`] does with its bytes when it is closed.
#[derive(Clone)]
pub enum Target {
    /// `append` them to the file.
    Append,
    /// `put` them as the whole file, under the condition.
    Put(Condition),
}

/// A reader over bytes already in memory.
pub struct BytesReader {
    /// The bytes.
    bytes: Vec<u8>,
    /// How many of them were read already.
    position: usize,
}

impl BytesReader {
    /// A reader that gives `bytes` from the start.
    pub const fn new(bytes: Vec<u8>) -> Self {
        Self { bytes, position: 0 }
    }
}

impl AsyncRead for BytesReader {
    fn poll_read(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<io::Result<usize>> {
        let remaining = self.bytes.get(self.position..).unwrap_or_default();
        let count = remaining.len().min(buf.len());
        if let (Some(target), Some(source)) = (buf.get_mut(..count), remaining.get(..count)) {
            target.copy_from_slice(source);
        }
        self.position = self.position.saturating_add(count);
        Poll::Ready(Ok(count))
    }
}

/// The future that stores a [`BufferedWriter`]'s bytes.
type Store<'bucket> = Pin<Box<dyn Future<Output = Result<()>> + Send + 'bucket>>;

/// A writer that keeps its bytes and hands them to the bucket when it is closed.
pub struct BufferedWriter<'bucket, B>
where
    B: ?Sized,
{
    /// The bucket that stores them.
    bucket: &'bucket B,
    /// The bytes written so far.
    buffer: Vec<u8>,
    /// Whether `close` has finished.
    done: bool,
    /// The file they go to.
    path: &'bucket Path,
    /// The store in progress, once `close` has started it.
    store: Option<Store<'bucket>>,
    /// How they are stored.
    target: Target,
}

impl<'bucket, B> BufferedWriter<'bucket, B>
where
    B: ?Sized,
{
    /// A writer over `path` that stores as `target` says.
    pub const fn new(bucket: &'bucket B, path: &'bucket Path, target: Target) -> Self {
        Self {
            buffer: Vec::new(),
            bucket,
            done: false,
            path,
            store: None,
            target,
        }
    }
}

impl<B> AsyncWrite for BufferedWriter<'_, B>
where
    B: Bucket + ?Sized,
{
    fn poll_close(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let this = &mut *self;
        if this.done {
            return Poll::Ready(Ok(()));
        }
        if this.store.is_none() {
            let bucket = this.bucket;
            let path = this.path;
            let bytes = mem::take(&mut this.buffer);
            let target = this.target.clone();
            this.store = Some(Box::pin(async move {
                match target {
                    Target::Append => bucket.append(path, &bytes).await.map(drop),
                    Target::Put(condition) => bucket.put(path, &bytes, &condition).await.map(drop),
                }
            }));
        }
        let Some(store) = this.store.as_mut() else {
            return Poll::Ready(Ok(()));
        };
        match store.as_mut().poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(outcome) => {
                this.done = true;
                this.store = None;
                Poll::Ready(outcome.map_err(io::Error::from))
            }
        }
    }

    fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn poll_write(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        self.buffer.extend_from_slice(buf);
        Poll::Ready(Ok(buf.len()))
    }
}

/// Read the file, then write it back with `more` at the end under the token read.
pub async fn append_by_rewrite<B>(bucket: &B, path: &Path, more: &[u8]) -> Result<RevisionToken>
where
    B: Bucket + ?Sized,
{
    match bucket.get(path).await {
        Ok((mut bytes, token)) => {
            bytes.extend_from_slice(more);
            bucket.put(path, &bytes, &Condition::Matches(token)).await
        }
        Err(error) if error.kind() == ErrorKind::NotFound => {
            bucket.put(path, more, &Condition::Absent).await
        }
        Err(error) => Err(error),
    }
}

/// Read the file, write it at `to`, and give its length.
pub async fn copy_by_rewrite<B>(bucket: &B, from: &Path, to: &Path) -> Result<u64>
where
    B: Bucket + ?Sized,
{
    let (bytes, _token) = bucket.get(from).await?;
    bucket.put(to, &bytes, &Condition::None).await?;
    Ok(u64::try_from(bytes.len()).map_err(io::Error::other)?)
}

/// Move a file or, when `from` is a folder, everything in it.
pub async fn rename_by_copy<B>(bucket: &B, from: &Path, to: &Path) -> Result<()>
where
    B: Bucket + ?Sized,
{
    if bucket.head(from).await?.is_dir {
        rename_folder(bucket, from, to).await
    } else {
        rename_file(bucket, from, to).await
    }
}

/// Write the file at `to`, then delete `from` under the token read; a failed delete removes the copy.
async fn rename_file<B>(bucket: &B, from: &Path, to: &Path) -> Result<()>
where
    B: Bucket + ?Sized,
{
    let (bytes, token) = bucket.get(from).await?;
    bucket.put(to, &bytes, &Condition::None).await?;
    if let Err(error) = bucket.delete(from, &Condition::Matches(token)).await {
        drop(bucket.delete(to, &Condition::None).await);
        return Err(error);
    }
    Ok(())
}

/// Create the folders of `from` under `to`, move its files one by one, then remove `from`.
async fn rename_folder<B>(bucket: &B, from: &Path, to: &Path) -> Result<()>
where
    B: Bucket + ?Sized,
{
    let source = lexical(from)?;
    let target = lexical(to)?;
    bucket.create_folder(&target).await?;
    let entries = bucket.list(&source, Depth::All).await?;
    for entry in entries.iter().filter(|entry| entry.is_dir) {
        bucket
            .create_folder(&moved(&source, &target, entry)?)
            .await?;
    }
    for entry in entries.iter().filter(|entry| entry.is_file) {
        rename_file(bucket, &entry.path, &moved(&source, &target, entry)?).await?;
    }
    bucket.remove_folder(&source).await
}

/// Where `entry`, found below `source`, goes below `target`.
fn moved(source: &Path, target: &Path, entry: &WalkEntry) -> Result<PathBuf> {
    let relative = entry
        .path
        .strip_prefix(source)
        .map_err(|error| StorageError::new(ErrorKind::Other, error))?;
    Ok(target.join(relative))
}

/// Every entry below `folder`, leaving out those under a dot-prefixed component unless `hidden`.
pub async fn walk_by_list<B>(bucket: &B, folder: &Path, hidden: bool) -> Result<Vec<WalkEntry>>
where
    B: Bucket + ?Sized,
{
    let base = lexical(folder)?;
    let mut entries = bucket.list(&base, Depth::All).await?;
    if !hidden {
        entries.retain(|entry| !is_hidden(&base, &entry.path));
    }
    Ok(entries)
}

/// Whether `path` is under a dot-prefixed component below `base`.
pub fn is_hidden(base: &Path, path: &Path) -> bool {
    path.strip_prefix(base).is_ok_and(|relative| {
        relative
            .components()
            .any(|component| component.as_os_str().to_string_lossy().starts_with('.'))
    })
}
