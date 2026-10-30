//! The file operations of `System`, run on the bucket a system holds.

use crate::bucket::{Bucket, Condition, Depth, RevisionToken};
use crate::error::{ErrorKind, Result, StorageError};
use crate::{BoxFuture, FileMetadata, WalkEntry};
use alloc::borrow::Cow;
use core::mem;
use core::pin::Pin;
use core::task::{Context, Poll};
use futures_io::{AsyncRead, AsyncWrite};
use std::io;
use std::path::{Path, PathBuf};

/// A reader stream a system hands out.
pub type Reader<'system> = Box<dyn AsyncRead + Send + Unpin + 'system>;

/// A writer stream a system hands out.
pub type Writer<'system> = Box<dyn AsyncWrite + Send + Unpin + 'system>;

/// A system that keeps its files in a bucket and resolves relative paths against a folder.
pub trait Machine: Send + Sync {
    /// The bucket that holds the files.
    type Store: Bucket;

    /// The bucket that holds the files.
    fn store(&self) -> &Self::Store;

    /// The folder a relative path is resolved against.
    fn working_dir(&self) -> Result<PathBuf>;
}

/// What a closed [`OwnedWriter`] does with its bytes.
enum Target {
    /// Append them to the file.
    Append,
    /// Store them as the whole file, replacing what is there.
    Replace,
}

/// A reader over bytes in memory.
struct OwnedReader {
    /// The bytes.
    bytes: Vec<u8>,
    /// How many were read already.
    position: usize,
}

impl AsyncRead for OwnedReader {
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

/// The store in progress of an [`OwnedWriter`].
type Store<'system> = BoxFuture<'system, Result<()>>;

/// A writer that owns its path, keeps its bytes and hands them to the bucket when closed.
struct OwnedWriter<'system, B> {
    /// The bucket that stores the bytes.
    bucket: &'system B,
    /// The bytes written so far.
    buffer: Vec<u8>,
    /// Whether `poll_close` has finished.
    done: bool,
    /// The file the bytes go to.
    path: PathBuf,
    /// The store, once `poll_close` has started it.
    store: Option<Store<'system>>,
    /// How the bytes are stored.
    target: Target,
}

impl<'system, B: Bucket> OwnedWriter<'system, B> {
    /// A writer over `path` that stores as `target` says.
    const fn new(bucket: &'system B, path: PathBuf, target: Target) -> Self {
        Self {
            bucket,
            buffer: Vec::new(),
            done: false,
            path,
            store: None,
            target,
        }
    }
}

impl<B: Bucket> AsyncWrite for OwnedWriter<'_, B> {
    fn poll_close(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let this = &mut *self;
        if this.done {
            return Poll::Ready(Ok(()));
        }
        if this.store.is_none() {
            let bucket = this.bucket;
            let path = mem::take(&mut this.path);
            let bytes = mem::take(&mut this.buffer);
            let append = matches!(this.target, Target::Append);
            this.store = Some(Box::pin(async move {
                if append {
                    bucket.append(&path, &bytes).await.map(drop)
                } else {
                    bucket.put(&path, &bytes, &Condition::None).await.map(drop)
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

/// `path` as the bucket sees it: absolute, a relative path joined onto the working folder.
fn resolve<'system, M>(machine: &M, path: &'system Path) -> Result<Cow<'system, Path>>
where
    M: Machine,
{
    if path.is_absolute() {
        Ok(Cow::Borrowed(path))
    } else {
        Ok(Cow::Owned(machine.working_dir()?.join(path)))
    }
}

/// What is at `path`, or `None` when nothing is.
async fn find<M>(machine: &M, path: &Path) -> Result<Option<FileMetadata>>
where
    M: Machine,
{
    let resolved = resolve(machine, path)?;
    match machine.store().head(&resolved).await {
        Ok(found) => Ok(Some(found)),
        Err(error) if matches!(error.kind(), ErrorKind::NotFound | ErrorKind::NotADirectory) => {
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

/// The absolute form of `path`.
pub async fn canonicalize<M>(machine: &M, path: &Path) -> Result<PathBuf>
where
    M: Machine,
{
    machine.store().canonicalize(&resolve(machine, path)?).await
}

/// Copy the file at `from` to `to` and give the bytes copied.
pub async fn copy<M>(machine: &M, from: &Path, to: &Path) -> Result<u64>
where
    M: Machine,
{
    let source = resolve(machine, from)?;
    let target = resolve(machine, to)?;
    machine.store().copy(&source, &target).await
}

/// A writer that replaces the file at `path`.
pub async fn create<'system, M>(machine: &'system M, path: &'system Path) -> Result<Writer<'system>>
where
    M: Machine,
{
    match resolve(machine, path)? {
        Cow::Borrowed(absolute) => {
            machine
                .store()
                .write_stream(absolute, Condition::None)
                .await
        }
        Cow::Owned(absolute) => Ok(Box::new(OwnedWriter::new(
            machine.store(),
            absolute,
            Target::Replace,
        ))),
    }
}

/// Create `path` and the folders above it.
pub async fn create_dir_all<M>(machine: &M, path: &Path) -> Result<()>
where
    M: Machine,
{
    machine
        .store()
        .create_folder(&resolve(machine, path)?)
        .await
}

/// Whether anything is at `path`.
pub async fn exists<M>(machine: &M, path: &Path) -> Result<bool>
where
    M: Machine,
{
    Ok(find(machine, path).await?.is_some())
}

/// Whether a folder is at `path`.
pub async fn is_dir<M>(machine: &M, path: &Path) -> Result<bool>
where
    M: Machine,
{
    Ok(find(machine, path).await?.is_some_and(|found| found.is_dir))
}

/// Whether a file is at `path`.
pub async fn is_file<M>(machine: &M, path: &Path) -> Result<bool>
where
    M: Machine,
{
    Ok(find(machine, path)
        .await?
        .is_some_and(|found| found.is_file))
}

/// Kind, size, modification time and token of what is at `path`.
pub async fn metadata<M>(machine: &M, path: &Path) -> Result<FileMetadata>
where
    M: Machine,
{
    machine.store().head(&resolve(machine, path)?).await
}

/// A reader over the file at `path`.
pub async fn open<'system, M>(machine: &'system M, path: &'system Path) -> Result<Reader<'system>>
where
    M: Machine,
{
    match resolve(machine, path)? {
        Cow::Borrowed(absolute) => machine.store().read_stream(absolute).await,
        Cow::Owned(absolute) => {
            let (bytes, _token) = machine.store().get(&absolute).await?;
            Ok(Box::new(OwnedReader { bytes, position: 0 }))
        }
    }
}

/// A writer that appends to the file at `path`, creating it when missing.
pub async fn open_append<'system, M>(
    machine: &'system M,
    path: &'system Path,
) -> Result<Writer<'system>>
where
    M: Machine,
{
    match resolve(machine, path)? {
        Cow::Borrowed(absolute) => machine.store().append_stream(absolute).await,
        Cow::Owned(absolute) => Ok(Box::new(OwnedWriter::new(
            machine.store(),
            absolute,
            Target::Append,
        ))),
    }
}

/// The bytes at `path` and the token of that version.
pub async fn read<M>(machine: &M, path: &Path) -> Result<(Vec<u8>, RevisionToken)>
where
    M: Machine,
{
    machine.store().get(&resolve(machine, path)?).await
}

/// The full paths of the direct children of the folder at `path`.
pub async fn read_dir<M>(machine: &M, path: &Path) -> Result<Vec<PathBuf>>
where
    M: Machine,
{
    let resolved = resolve(machine, path)?;
    let store = machine.store();
    if !store.head(&resolved).await?.is_dir {
        return Err(StorageError::from_kind(ErrorKind::NotADirectory));
    }
    let entries = store.list(&resolved, Depth::Children).await?;
    Ok(entries.into_iter().map(|entry| entry.path).collect())
}

/// The contents of the file at `path` as text.
pub async fn read_to_string<M>(machine: &M, path: &Path) -> Result<String>
where
    M: Machine,
{
    let (bytes, _token) = read(machine, path).await?;
    String::from_utf8(bytes).map_err(|error| StorageError::new(ErrorKind::InvalidData, error))
}

/// Remove the folder at `path` and everything in it.
pub async fn remove_dir_all<M>(machine: &M, path: &Path) -> Result<()>
where
    M: Machine,
{
    machine
        .store()
        .remove_folder(&resolve(machine, path)?)
        .await
}

/// Remove the file at `path`; `NotFound` when nothing is there.
pub async fn remove_file<M>(machine: &M, path: &Path) -> Result<()>
where
    M: Machine,
{
    let resolved = resolve(machine, path)?;
    let store = machine.store();
    store.head(&resolved).await?;
    store.delete(&resolved, &Condition::None).await
}

/// Remove the file at `path` while it still has `token`; `NotFound` when nothing is there.
pub async fn remove_file_if<M>(machine: &M, path: &Path, token: &RevisionToken) -> Result<bool>
where
    M: Machine,
{
    let resolved = resolve(machine, path)?;
    let store = machine.store();
    store.head(&resolved).await?;
    let condition = Condition::Matches(token.clone());
    match store.delete(&resolved, &condition).await {
        Ok(()) => Ok(true),
        Err(error) if error.kind() == ErrorKind::PreconditionFailed => Ok(false),
        Err(error) => Err(error),
    }
}

/// Move the file or folder at `from` to `to`.
pub async fn rename<M>(machine: &M, from: &Path, to: &Path) -> Result<()>
where
    M: Machine,
{
    let source = resolve(machine, from)?;
    let target = resolve(machine, to)?;
    machine.store().rename(&source, &target).await
}

/// Every entry below `path`.
pub async fn walk_dir<M>(
    machine: &M,
    path: &Path,
    follow_links: bool,
    hidden: bool,
) -> Result<Vec<WalkEntry>>
where
    M: Machine,
{
    machine
        .store()
        .walk(&resolve(machine, path)?, follow_links, hidden)
        .await
}

/// Write `contents` at `path` under `condition` and give the new token.
pub async fn write<M>(
    machine: &M,
    path: &Path,
    contents: &[u8],
    condition: Condition,
) -> Result<RevisionToken>
where
    M: Machine,
{
    machine
        .store()
        .put(&resolve(machine, path)?, contents, &condition)
        .await
}
