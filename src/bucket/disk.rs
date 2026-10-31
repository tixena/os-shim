//! Files on the machine's disk through `std::fs`, as a `Bucket`.

use super::{Bucket, Condition, Depth, Guarantee, Guarantees, RevisionToken};
use crate::error::{ErrorKind, Result, StorageError};
use crate::{BoxFuture, FileMetadata, WalkEntry};
use core::future::poll_fn;
use core::pin::Pin;
use core::task::{Context, Poll};
use futures_io::{AsyncRead, AsyncWrite};
use ignore::WalkBuilder;
use same_file::Handle;
use sha2::{Digest as _, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read as _, Seek as _, Write as _};
use std::path::{Path, PathBuf};

/// The local disk as a bucket: every path is handed to the operating system as it is.
///
/// Its futures do their `std::fs` call when first polled and never wait, so they need no
/// runtime; a call holds its thread for its whole length, as a synchronous call does.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct DiskBucket;

/// A disk reader: each poll does its `std::fs` read at once.
struct DiskReader(File);

/// A disk writer: each poll writes its bytes at once, so a dropped writer keeps what it wrote.
struct DiskWriter(File);

impl DiskBucket {
    /// The disk, rooted at `/`, as `RealSystem` sees it today.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl AsyncRead for DiskReader {
    fn poll_read(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<io::Result<usize>> {
        Poll::Ready(self.0.read(buf))
    }
}

impl AsyncWrite for DiskWriter {
    fn poll_close(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(self.0.flush())
    }

    fn poll_flush(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(self.0.flush())
    }

    fn poll_write(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        Poll::Ready(self.0.write(buf))
    }
}

impl Bucket for DiskBucket {
    #[inline]
    fn append<'call>(
        &'call self,
        path: &'call Path,
        more: &'call [u8],
    ) -> BoxFuture<'call, Result<RevisionToken>> {
        Box::pin(on_first_poll(move || append_now(path, more)))
    }

    #[inline]
    fn append_stream<'bucket, 'call>(
        &'bucket self,
        path: &'call Path,
    ) -> BoxFuture<'call, Result<Box<dyn AsyncWrite + Send + Unpin + 'bucket>>>
    where
        'bucket: 'call,
    {
        Box::pin(on_first_poll(move || {
            let file = OpenOptions::new().create(true).append(true).open(path)?;
            let writer: Box<dyn AsyncWrite + Send + Unpin + 'bucket> = Box::new(DiskWriter(file));
            Ok(writer)
        }))
    }

    #[inline]
    fn canonicalize<'call>(&'call self, path: &'call Path) -> BoxFuture<'call, Result<PathBuf>> {
        Box::pin(on_first_poll(move || Ok(fs::canonicalize(path)?)))
    }

    #[inline]
    fn copy<'call>(
        &'call self,
        from: &'call Path,
        to: &'call Path,
    ) -> BoxFuture<'call, Result<u64>> {
        Box::pin(on_first_poll(move || Ok(fs::copy(from, to)?)))
    }

    #[inline]
    fn create_folder<'call>(&'call self, path: &'call Path) -> BoxFuture<'call, Result<()>> {
        Box::pin(on_first_poll(move || Ok(fs::create_dir_all(path)?)))
    }

    #[inline]
    fn delete<'call>(
        &'call self,
        path: &'call Path,
        condition: &'call Condition,
    ) -> BoxFuture<'call, Result<()>> {
        Box::pin(on_first_poll(move || {
            refuse_folder(path)?;
            match condition.clone() {
                Condition::Absent => match fs::symlink_metadata(path) {
                    Ok(_) => Err(StorageError::from_kind(ErrorKind::AlreadyExists)),
                    Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
                    Err(error) => Err(error.into()),
                },
                Condition::Matches(wanted) => delete_matching(path, &wanted),
                Condition::None => match fs::remove_file(path) {
                    Err(error) if error.kind() != io::ErrorKind::NotFound => Err(error.into()),
                    _ => Ok(()),
                },
            }
        }))
    }

    #[inline]
    fn get<'call>(
        &'call self,
        path: &'call Path,
    ) -> BoxFuture<'call, Result<(Vec<u8>, RevisionToken)>> {
        Box::pin(on_first_poll(move || {
            let bytes = fs::read(path)?;
            let token = token_of(&bytes);
            Ok((bytes, token))
        }))
    }

    /// What the disk promises, with two limits.
    ///
    /// A rename is atomic only within one file system: `std::fs::rename` does not work across
    /// mount points. The conditional checks are one step only among callers of os-shim, under
    /// the advisory lock.
    #[inline]
    fn guarantees(&self) -> Guarantees {
        const DISK: Guarantees = Guarantees::none()
            .with(Guarantee::AppendWritesOnlyNewBytes)
            .with(Guarantee::ConditionalDeleteInOneStep)
            .with(Guarantee::ConditionalWriteInOneStep)
            .with(Guarantee::CreateOnlyWriteInOneStep)
            .with(Guarantee::DroppedWriterKeepsBytes)
            .with(Guarantee::EmptyFoldersExist)
            .with(Guarantee::FailedRenameKeepsDestination)
            .with(Guarantee::FileAndFolderNeverShareAPath)
            .with(Guarantee::FileRenameIsAtomic)
            .with(Guarantee::FolderRenameIsAtomic)
            .with(Guarantee::FuturesNeverWait)
            .with(Guarantee::ParentCheckInOneStep)
            .with(Guarantee::RenameKeepsModificationTime)
            .with(Guarantee::WalkHonorsGitignore);
        DISK
    }

    #[inline]
    fn head<'call>(&'call self, path: &'call Path) -> BoxFuture<'call, Result<FileMetadata>> {
        Box::pin(on_first_poll(move || {
            let metadata = fs::metadata(path)?;
            Ok(FileMetadata::new(
                metadata.is_dir(),
                metadata.is_file(),
                metadata.len(),
                Some(metadata.modified()?),
                None,
            ))
        }))
    }

    #[inline]
    fn list<'call>(
        &'call self,
        folder: &'call Path,
        depth: Depth,
    ) -> BoxFuture<'call, Result<Vec<WalkEntry>>> {
        Box::pin(on_first_poll(move || list_now(folder, depth)))
    }

    #[inline]
    fn put<'call>(
        &'call self,
        path: &'call Path,
        bytes: &'call [u8],
        condition: &'call Condition,
    ) -> BoxFuture<'call, Result<RevisionToken>> {
        Box::pin(on_first_poll(move || {
            refuse_folder(path)?;
            match condition.clone() {
                Condition::Absent => create_new_with(path, bytes),
                Condition::Matches(wanted) => {
                    open_for_replace(path, &wanted)?.write_all(bytes)?;
                    Ok(token_of(bytes))
                }
                Condition::None => {
                    fs::write(path, bytes)?;
                    Ok(token_of(bytes))
                }
            }
        }))
    }

    #[inline]
    fn read_stream<'bucket, 'call>(
        &'bucket self,
        path: &'call Path,
    ) -> BoxFuture<'call, Result<Box<dyn AsyncRead + Send + Unpin + 'bucket>>>
    where
        'bucket: 'call,
    {
        Box::pin(on_first_poll(move || {
            let reader: Box<dyn AsyncRead + Send + Unpin + 'bucket> =
                Box::new(DiskReader(File::open(path)?));
            Ok(reader)
        }))
    }

    #[inline]
    fn remove_folder<'call>(&'call self, path: &'call Path) -> BoxFuture<'call, Result<()>> {
        Box::pin(on_first_poll(move || Ok(fs::remove_dir_all(path)?)))
    }

    #[inline]
    fn rename<'call>(
        &'call self,
        from: &'call Path,
        to: &'call Path,
    ) -> BoxFuture<'call, Result<()>> {
        Box::pin(on_first_poll(move || match fs::rename(from, to) {
            Err(error) if error.kind() == io::ErrorKind::DirectoryNotEmpty => {
                Err(StorageError::new(ErrorKind::AlreadyExists, error))
            }
            other => Ok(other?),
        }))
    }

    #[inline]
    fn walk<'call>(
        &'call self,
        folder: &'call Path,
        follow_links: bool,
        hidden: bool,
    ) -> BoxFuture<'call, Result<Vec<WalkEntry>>> {
        Box::pin(on_first_poll(move || {
            walk_now(folder, follow_links, hidden)
        }))
    }

    #[inline]
    fn write_stream<'bucket, 'call>(
        &'bucket self,
        path: &'call Path,
        condition: Condition,
    ) -> BoxFuture<'call, Result<Box<dyn AsyncWrite + Send + Unpin + 'bucket>>>
    where
        'bucket: 'call,
    {
        Box::pin(on_first_poll(move || {
            refuse_folder(path)?;
            let file = match condition.clone() {
                Condition::Absent => OpenOptions::new().write(true).create_new(true).open(path)?,
                Condition::Matches(wanted) => open_for_replace(path, &wanted)?,
                Condition::None => File::create(path)?,
            };
            let writer: Box<dyn AsyncWrite + Send + Unpin + 'bucket> = Box::new(DiskWriter(file));
            Ok(writer)
        }))
    }
}

/// A future that runs `call` when first polled and is ready at once.
fn on_first_poll<T, F>(mut call: F) -> impl Future<Output = T> + Send
where
    F: FnMut() -> T + Send,
    T: Send,
{
    poll_fn(move |_cx| Poll::Ready(call()))
}

/// The disk's revision token: the SHA-256 hash of the bytes, in lowercase hexadecimal.
fn token_of(bytes: &[u8]) -> RevisionToken {
    let hex: String = Sha256::digest(bytes)
        .iter()
        .flat_map(|byte| [byte >> 4_u8, byte & 0x0f_u8])
        .filter_map(|nibble| char::from_digit(u32::from(nibble), 16))
        .collect();
    RevisionToken::new(hex)
}

/// `IsADirectory` when `path` names a folder, whatever the condition of the call.
fn refuse_folder(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(found) if found.is_dir() => Err(StorageError::from_kind(ErrorKind::IsADirectory)),
        Err(error) if error.kind() != io::ErrorKind::NotFound => Err(error.into()),
        _ => Ok(()),
    }
}

/// The error of a condition that does not hold.
const fn stale() -> StorageError {
    StorageError::from_kind(ErrorKind::PreconditionFailed)
}

/// Open `path` with `options`.
///
/// A missing file is stale when its parent folder exists, and `NotFound` otherwise.
fn open_existing(options: &OpenOptions, path: &Path) -> Result<File> {
    match options.open(path) {
        Ok(opened) => Ok(opened),
        Err(error)
            if error.kind() == io::ErrorKind::NotFound
                && path.parent().is_some_and(Path::is_dir) =>
        {
            Err(stale())
        }
        Err(error) => Err(error.into()),
    }
}

/// Take the advisory lock on `file` and check that its bytes have `wanted`.
fn lock_and_compare(file: &File, wanted: &RevisionToken) -> Result<()> {
    file.lock()?;
    let mut held = Vec::new();
    io::Read::read_to_end(&mut &*file, &mut held)?;
    if token_of(&held) == *wanted {
        Ok(())
    } else {
        Err(stale())
    }
}

/// Open `path` with `options`, take the advisory lock, and give the file when its bytes have `wanted`.
fn open_matching(options: &OpenOptions, path: &Path, wanted: &RevisionToken) -> Result<File> {
    let file = open_existing(options, path)?;
    lock_and_compare(&file, wanted)?;
    Ok(file)
}

/// The file at `path`, locked, checked against `wanted`, emptied and ready to be written whole.
fn open_for_replace(path: &Path, wanted: &RevisionToken) -> Result<File> {
    let mut file = open_matching(OpenOptions::new().read(true).write(true), path, wanted)?;
    file.set_len(0)?;
    file.rewind()?;
    Ok(file)
}

/// Remove the file at `path` when its bytes have `wanted` and `path` still names the file that was read.
fn delete_matching(path: &Path, wanted: &RevisionToken) -> Result<()> {
    let file = open_existing(OpenOptions::new().read(true), path)?;
    delete_opened(file, path, wanted)
}

/// The steps of `delete_matching` after it opened `path` as `file`: lock, compare, check the path
/// still names `file`, remove.
fn delete_opened(file: File, path: &Path, wanted: &RevisionToken) -> Result<()> {
    lock_and_compare(&file, wanted)?;
    let opened = Handle::from_file(file)?;
    let still_named = match Handle::from_path(path) {
        Ok(named) => named == opened,
        Err(error) if error.kind() == io::ErrorKind::NotFound => false,
        Err(error) => return Err(error.into()),
    };
    if !still_named {
        return Err(stale());
    }
    fs::remove_file(path)?;
    Ok(())
}

/// The file at `path`, created and written with `bytes` when nothing is there.
fn create_new_with(path: &Path, bytes: &[u8]) -> Result<RevisionToken> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?
        .write_all(bytes)?;
    Ok(token_of(bytes))
}

/// Add `more` at the end of the file at `path`, and give the token of the whole file.
fn append_now(path: &Path, more: &[u8]) -> Result<RevisionToken> {
    let mut file = OpenOptions::new()
        .read(true)
        .create(true)
        .append(true)
        .open(path)?;
    file.write_all(more)?;
    drop(file);
    Ok(token_of(&fs::read(path)?))
}

/// The entry for `path`, kinds read the way `Path::is_dir` and `Path::is_file` do.
fn entry_of(path: PathBuf) -> WalkEntry {
    let is_dir = path.is_dir();
    let is_file = path.is_file();
    WalkEntry::new(path, is_dir, is_file)
}

/// The entries below `folder` to `depth`, sorted by path, with no ignore rules.
fn list_now(folder: &Path, depth: Depth) -> Result<Vec<WalkEntry>> {
    let mut entries = Vec::new();
    let mut pending = vec![folder.to_path_buf()];
    while let Some(current) = pending.pop() {
        let items = match fs::read_dir(&current) {
            Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
            other => other?,
        };
        for item in items {
            let found = item?;
            if depth == Depth::All && found.file_type()?.is_dir() {
                pending.push(found.path());
            }
            entries.push(entry_of(found.path()));
        }
    }
    entries.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(entries)
}

/// The entries below `folder` that `.gitignore` files leave in, sorted by path.
fn walk_now(folder: &Path, follow_links: bool, hidden: bool) -> Result<Vec<WalkEntry>> {
    let mut entries = Vec::new();
    for result in WalkBuilder::new(folder)
        .follow_links(follow_links)
        .hidden(!hidden)
        .build()
        .skip(1)
    {
        let entry = result.map_err(io::Error::other)?;
        entries.push(entry_of(entry.path().to_path_buf()));
    }
    entries.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(entries)
}

#[cfg(test)]
mod tests;
