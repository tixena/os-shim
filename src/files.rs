//! The file operations of `System`, run on the bucket a system holds, or on local storage while a temporary directory lives.

use crate::bucket::{Bucket, Condition, Depth, RevisionToken};
use crate::error::{ErrorKind, Result, StorageError};
use crate::streams::{close, read_chunk, write_all};
use crate::{BoxFuture, FileMetadata, System, TempDirHandle, WalkEntry};
use alloc::borrow::Cow;
use alloc::collections::BTreeSet;
use alloc::sync::Arc;
use core::future::Future;
use core::mem;
use core::pin::{Pin, pin};
use core::task::{Context, Poll, Waker};
use futures_io::{AsyncRead, AsyncWrite};
use std::io;
#[cfg(feature = "zip")]
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};
use tracing::error;

/// Run `$body` with `$bucket` bound to whichever storage `$pick` holds.
macro_rules! on {
    ($pick:expr, $bucket:ident => $body:expr) => {
        match $pick {
            Pick::Local($bucket) => $body,
            Pick::Store($bucket) => $body,
        }
    };
}

/// Bytes moved per pass when a file crosses from one storage to the other.
const COPY_CHUNK_BYTES: usize = 0x0001_0000;

/// A reader stream a system hands out.
pub type Reader<'system> = Box<dyn AsyncRead + Send + Unpin + 'system>;

/// A writer stream a system hands out.
pub type Writer<'system> = Box<dyn AsyncWrite + Send + Unpin + 'system>;

/// The roots of the temporary directories whose handles are alive.
pub type LiveTemps = Arc<Mutex<BTreeSet<PathBuf>>>;

/// A system that keeps its files in a bucket, and the paths under a live temporary directory on local storage.
pub trait Machine: Send + Sync {
    /// The local storage the temporary directories live on.
    type Local: Bucket;

    /// The bucket that holds every other file.
    type Store: Bucket;

    /// The local storage the temporary directories live on.
    fn local(&self) -> &Self::Local;

    /// The bucket that holds every other file.
    fn store(&self) -> &Self::Store;

    /// The roots of the temporary directories whose handles are alive.
    fn temps(&self) -> &LiveTemps;

    /// The folder a relative path is resolved against.
    fn working_dir(&self) -> Result<PathBuf>;
}

/// The storage a path lives on.
enum Pick<'storage, S, L> {
    /// Local storage, for a path under a live temporary directory.
    Local(&'storage L),
    /// The bucket, for every other path.
    Store(&'storage S),
}

impl<S, L> Clone for Pick<'_, S, L> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<S, L> Copy for Pick<'_, S, L> {}

impl<'storage, S: Bucket, L: Bucket> Pick<'storage, S, L> {
    /// Add `more` at the end of the file at `path`.
    async fn append(self, path: &Path, more: &[u8]) -> Result<RevisionToken> {
        on!(self, bucket => bucket.append(path, more).await)
    }

    /// A writer that appends to the file at `path` when closed.
    async fn append_stream(self, path: &'storage Path) -> Result<Writer<'storage>> {
        on!(self, bucket => bucket.append_stream(path).await)
    }

    /// The absolute form of `path`.
    async fn canonicalize(self, path: &Path) -> Result<PathBuf> {
        on!(self, bucket => bucket.canonicalize(path).await)
    }

    /// Copy a file inside this storage.
    async fn copy(self, from: &Path, to: &Path) -> Result<u64> {
        on!(self, bucket => bucket.copy(from, to).await)
    }

    /// Create `path` and the folders above it.
    async fn create_folder(self, path: &Path) -> Result<()> {
        on!(self, bucket => bucket.create_folder(path).await)
    }

    /// Remove the file at `path` when `condition` holds.
    async fn delete(self, path: &Path, condition: &Condition) -> Result<()> {
        on!(self, bucket => bucket.delete(path, condition).await)
    }

    /// The bytes at `path` and their token.
    async fn get(self, path: &Path) -> Result<(Vec<u8>, RevisionToken)> {
        on!(self, bucket => bucket.get(path).await)
    }

    /// What is at `path`.
    async fn head(self, path: &Path) -> Result<FileMetadata> {
        on!(self, bucket => bucket.head(path).await)
    }

    /// The entries below `folder`, to `depth`.
    async fn list(self, folder: &Path, depth: Depth) -> Result<Vec<WalkEntry>> {
        on!(self, bucket => bucket.list(folder, depth).await)
    }

    /// Write `bytes` at `path` when `condition` holds.
    async fn put(self, path: &Path, bytes: &[u8], condition: &Condition) -> Result<RevisionToken> {
        on!(self, bucket => bucket.put(path, bytes, condition).await)
    }

    /// A reader over the file at `path`.
    async fn read_stream(self, path: &'storage Path) -> Result<Reader<'storage>> {
        on!(self, bucket => bucket.read_stream(path).await)
    }

    /// Remove `path` and everything below it.
    async fn remove_folder(self, path: &Path) -> Result<()> {
        on!(self, bucket => bucket.remove_folder(path).await)
    }

    /// Move a file or a folder inside this storage.
    async fn rename(self, from: &Path, to: &Path) -> Result<()> {
        on!(self, bucket => bucket.rename(from, to).await)
    }

    /// Every entry below `folder`.
    async fn walk(self, folder: &Path, follow_links: bool, hidden: bool) -> Result<Vec<WalkEntry>> {
        on!(self, bucket => bucket.walk(folder, follow_links, hidden).await)
    }

    /// A writer that stores its bytes at `path` under `condition` when closed.
    async fn write_stream(
        self,
        path: &'storage Path,
        condition: Condition,
    ) -> Result<Writer<'storage>> {
        on!(self, bucket => bucket.write_stream(path, condition).await)
    }
}

/// What a closed [`OwnedWriter`] does with its bytes.
enum Target {
    /// Append them to the file.
    Append,
    /// Store them as the whole file, replacing what is there.
    Replace,
}

/// A path resolved, and the storage that holds it.
type Located<'system, M> = (
    Cow<'system, Path>,
    Pick<'system, <M as Machine>::Store, <M as Machine>::Local>,
);

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
struct OwnedWriter<'system, S, L> {
    /// The bytes written so far.
    buffer: Vec<u8>,
    /// Whether `poll_close` has finished.
    done: bool,
    /// The file the bytes go to.
    path: PathBuf,
    /// The storage that stores the bytes.
    storage: Pick<'system, S, L>,
    /// The store, once `poll_close` has started it.
    store: Option<Store<'system>>,
    /// How the bytes are stored.
    target: Target,
}

impl<'system, S: Bucket, L: Bucket> OwnedWriter<'system, S, L> {
    /// A writer over `path` that stores as `target` says.
    const fn new(storage: Pick<'system, S, L>, path: PathBuf, target: Target) -> Self {
        Self {
            storage,
            buffer: Vec::new(),
            done: false,
            path,
            store: None,
            target,
        }
    }
}

impl<S: Bucket, L: Bucket> AsyncWrite for OwnedWriter<'_, S, L> {
    fn poll_close(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let this = &mut *self;
        if this.done {
            return Poll::Ready(Ok(()));
        }
        if this.store.is_none() {
            let storage = this.storage;
            let path = mem::take(&mut this.path);
            let bytes = mem::take(&mut this.buffer);
            let append = matches!(this.target, Target::Append);
            this.store = Some(Box::pin(async move {
                if append {
                    storage.append(&path, &bytes).await.map(drop)
                } else {
                    storage.put(&path, &bytes, &Condition::None).await.map(drop)
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

/// A temporary directory on local storage, removed when the handle drops.
pub struct LocalTempDir<'system, M: Machine> {
    /// The folder, below the machine's local storage.
    path: PathBuf,
    /// The system that made the folder.
    system: &'system M,
}

impl<'system, M: Machine> LocalTempDir<'system, M> {
    /// A handle for `path`, which becomes a live temporary directory.
    pub fn new(system: &'system M, path: PathBuf) -> Self {
        system
            .temps()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(path.clone());
        Self { path, system }
    }
}

impl<M: Machine + System> TempDirHandle for LocalTempDir<'_, M> {
    #[inline]
    fn path(&self) -> &Path {
        &self.path
    }

    #[cfg(feature = "zip")]
    #[inline]
    fn to_zip_stream(&self) -> BoxFuture<'_, Result<Box<dyn Read + Send>>> {
        self.system.to_zip_stream(&self.path)
    }
}

impl<M: Machine> Drop for LocalTempDir<'_, M> {
    fn drop(&mut self) {
        match run_without_runtime(self.system.local().remove_folder(&self.path)) {
            Some(Ok(())) => {}
            Some(Err(failure)) => error!(
                path = %self.path.display(),
                "temporary directory not removed: {failure}"
            ),
            None => error!(
                path = %self.path.display(),
                "local storage waited, which it never should"
            ),
        }
        self.system
            .temps()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(&self.path);
    }
}

/// `path` as the bucket sees it: absolute, a relative path joined onto the working folder.
fn resolve<'path, M>(machine: &M, path: &'path Path) -> Result<Cow<'path, Path>>
where
    M: Machine,
{
    if path.is_absolute() {
        Ok(Cow::Borrowed(path))
    } else {
        Ok(Cow::Owned(machine.working_dir()?.join(path)))
    }
}

/// The storage that holds `resolved`: local while a temporary directory above it lives.
fn route<'system, M>(machine: &'system M, resolved: &Path) -> Pick<'system, M::Store, M::Local>
where
    M: Machine,
{
    let under_temp = machine
        .temps()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .iter()
        .any(|root| resolved.starts_with(root));
    if under_temp {
        Pick::Local(machine.local())
    } else {
        Pick::Store(machine.store())
    }
}

/// `path` resolved, and the storage that holds it.
fn locate<'system, M>(machine: &'system M, path: &'system Path) -> Result<Located<'system, M>>
where
    M: Machine,
{
    let resolved = resolve(machine, path)?;
    let storage = route(machine, &resolved);
    Ok((resolved, storage))
}

/// What is at `path`, or `None` when nothing is.
async fn find<M>(machine: &M, path: &Path) -> Result<Option<FileMetadata>>
where
    M: Machine,
{
    let (resolved, storage) = locate(machine, path)?;
    match storage.head(&resolved).await {
        Ok(found) => Ok(Some(found)),
        Err(error) if matches!(error.kind(), ErrorKind::NotFound | ErrorKind::NotADirectory) => {
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

/// Copy the file at `from` in `source` to `to` in `target` through streams.
async fn copy_across<S, L>(
    source: Pick<'_, S, L>,
    from: &Path,
    target: Pick<'_, S, L>,
    to: &Path,
) -> Result<u64>
where
    S: Bucket,
    L: Bucket,
{
    let mut reader = source.read_stream(from).await?;
    let mut writer = target.write_stream(to, Condition::None).await?;
    let mut buffer = vec![0_u8; COPY_CHUNK_BYTES];
    let mut copied: u64 = 0;
    loop {
        let filled = read_chunk(&mut *reader, &mut buffer).await?;
        let Some(chunk) = buffer.get(..filled).filter(|chunk| !chunk.is_empty()) else {
            break;
        };
        write_all(&mut *writer, chunk).await?;
        copied = copied.saturating_add(u64::try_from(filled).unwrap_or(u64::MAX));
    }
    close(&mut *writer).await?;
    Ok(copied)
}

/// Move the file or folder at `from` in `source` to `to` in `target` through streams.
async fn rename_across<S, L>(
    source: Pick<'_, S, L>,
    from: &Path,
    target: Pick<'_, S, L>,
    to: &Path,
) -> Result<()>
where
    S: Bucket,
    L: Bucket,
{
    if !source.head(from).await?.is_dir {
        copy_across(source, from, target, to).await?;
        return source.delete(from, &Condition::None).await;
    }
    let mut entries = source.list(from, Depth::All).await?;
    entries.sort_by(|left, right| left.path.cmp(&right.path));
    target.create_folder(to).await?;
    for entry in entries {
        let relative = entry.path.strip_prefix(from).map_err(io::Error::other)?;
        let destination = to.join(relative);
        if entry.is_dir {
            target.create_folder(&destination).await?;
        } else {
            copy_across(source, &entry.path, target, &destination).await?;
        }
    }
    source.remove_folder(from).await
}

/// Run `future` in a single poll, with no runtime to wake it; `None` when it waits.
pub fn run_without_runtime<T, F>(future: F) -> Option<T>
where
    F: Future<Output = T>,
{
    let mut pinned = pin!(future);
    match pinned
        .as_mut()
        .poll(&mut Context::from_waker(Waker::noop()))
    {
        Poll::Ready(outcome) => Some(outcome),
        Poll::Pending => None,
    }
}

/// The absolute form of `path`.
pub async fn canonicalize<M>(machine: &M, path: &Path) -> Result<PathBuf>
where
    M: Machine,
{
    let (resolved, storage) = locate(machine, path)?;
    storage.canonicalize(&resolved).await
}

/// Copy the file at `from` to `to` and give the bytes copied.
pub async fn copy<M>(machine: &M, from: &Path, to: &Path) -> Result<u64>
where
    M: Machine,
{
    let (source, from_storage) = locate(machine, from)?;
    let (target, to_storage) = locate(machine, to)?;
    match (from_storage, to_storage) {
        (Pick::Local(_), Pick::Local(_)) | (Pick::Store(_), Pick::Store(_)) => {
            from_storage.copy(&source, &target).await
        }
        _ => copy_across(from_storage, &source, to_storage, &target).await,
    }
}

/// A writer that replaces the file at `path`.
pub async fn create<'system, M>(machine: &'system M, path: &'system Path) -> Result<Writer<'system>>
where
    M: Machine,
{
    let (resolved, storage) = locate(machine, path)?;
    match resolved {
        Cow::Borrowed(absolute) => storage.write_stream(absolute, Condition::None).await,
        Cow::Owned(absolute) => Ok(Box::new(OwnedWriter::new(
            storage,
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
    let (resolved, storage) = locate(machine, path)?;
    storage.create_folder(&resolved).await
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
    let (resolved, storage) = locate(machine, path)?;
    storage.head(&resolved).await
}

/// A reader over the file at `path`.
pub async fn open<'system, M>(machine: &'system M, path: &'system Path) -> Result<Reader<'system>>
where
    M: Machine,
{
    let (resolved, storage) = locate(machine, path)?;
    match resolved {
        Cow::Borrowed(absolute) => storage.read_stream(absolute).await,
        Cow::Owned(absolute) => {
            let (bytes, _token) = storage.get(&absolute).await?;
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
    let (resolved, storage) = locate(machine, path)?;
    match resolved {
        Cow::Borrowed(absolute) => storage.append_stream(absolute).await,
        Cow::Owned(absolute) => Ok(Box::new(OwnedWriter::new(
            storage,
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
    let (resolved, storage) = locate(machine, path)?;
    storage.get(&resolved).await
}

/// The full paths of the direct children of the folder at `path`.
pub async fn read_dir<M>(machine: &M, path: &Path) -> Result<Vec<PathBuf>>
where
    M: Machine,
{
    let (resolved, storage) = locate(machine, path)?;
    if !storage.head(&resolved).await?.is_dir {
        return Err(StorageError::from_kind(ErrorKind::NotADirectory));
    }
    let entries = storage.list(&resolved, Depth::Children).await?;
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
    let (resolved, storage) = locate(machine, path)?;
    storage.remove_folder(&resolved).await
}

/// Remove the file at `path`; `NotFound` when nothing is there.
pub async fn remove_file<M>(machine: &M, path: &Path) -> Result<()>
where
    M: Machine,
{
    let (resolved, storage) = locate(machine, path)?;
    storage.head(&resolved).await?;
    storage.delete(&resolved, &Condition::None).await
}

/// Remove the file at `path` while it still has `token`; `NotFound` when nothing is there.
pub async fn remove_file_if<M>(machine: &M, path: &Path, token: &RevisionToken) -> Result<bool>
where
    M: Machine,
{
    let (resolved, storage) = locate(machine, path)?;
    storage.head(&resolved).await?;
    let condition = Condition::Matches(token.clone());
    match storage.delete(&resolved, &condition).await {
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
    let (source, from_storage) = locate(machine, from)?;
    let (target, to_storage) = locate(machine, to)?;
    match (from_storage, to_storage) {
        (Pick::Local(_), Pick::Local(_)) | (Pick::Store(_), Pick::Store(_)) => {
            from_storage.rename(&source, &target).await
        }
        _ => rename_across(from_storage, &source, to_storage, &target).await,
    }
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
    let (resolved, storage) = locate(machine, path)?;
    storage.walk(&resolved, follow_links, hidden).await
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
    let (resolved, storage) = locate(machine, path)?;
    storage.put(&resolved, contents, &condition).await
}
