//! `MemoryBucket`'s operations: the file tree of `MemorySystem`, behind one lock.

use super::defaults::{BytesReader, is_hidden};
use super::paths::lexical;
use super::{
    Bucket, Clock, Condition, Depth, Guarantee, Guarantees, MemoryBucket, MemoryFile, MemoryState,
    RevisionToken,
};
use crate::error::{ErrorKind, Result, StorageError};
use crate::{BoxFuture, FileMetadata, WalkEntry};
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::sync::Arc;
use core::fmt;
use core::future::ready;
use core::mem;
use core::pin::Pin;
use core::task::{Context, Poll};
use futures_io::{AsyncRead, AsyncWrite};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::time::SystemTime;

/// A writer that stores its bytes at its path when closed, and when dropped without a close.
struct MemoryWriter {
    /// The bucket that stores them.
    bucket: MemoryBucket,
    /// The bytes written so far.
    buffer: Vec<u8>,
    /// The file they go to.
    path: PathBuf,
    /// The condition to write the whole file under, or `None` to append.
    put_under: Option<Condition>,
    /// Whether the bytes are stored already.
    stored: bool,
}

impl MemoryWriter {
    /// Store the bytes, once.
    fn finish(&mut self) -> Result<()> {
        if self.stored {
            return Ok(());
        }
        self.stored = true;
        match self.put_under.as_ref() {
            None => self.bucket.append_now(&self.path, &self.buffer).map(drop),
            Some(condition) => self
                .bucket
                .put_now(&self.path, &self.buffer, condition)
                .map(drop),
        }
    }
}

impl Drop for MemoryWriter {
    fn drop(&mut self) {
        drop(self.finish());
    }
}

impl AsyncWrite for MemoryWriter {
    fn poll_close(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(self.finish().map_err(io::Error::from))
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

impl fmt::Debug for MemoryBucket {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = self.read();
        f.debug_struct("MemoryBucket")
            .field("files", &state.files.len())
            .field("folders", &state.folders.len())
            .finish()
    }
}

impl MemoryState {
    /// Fail with `NotADirectory` when a proper ancestor of `path` is a file.
    fn check_no_file_above(&self, path: &Path) -> Result<()> {
        path.ancestors()
            .skip(1)
            .find(|ancestor| self.files.contains_key(*ancestor))
            .map_or(Ok(()), |file| {
                Err(StorageError::new(
                    ErrorKind::NotADirectory,
                    format!("{} is a file", file.display()),
                ))
            })
    }

    /// Fail unless a file can be written at `path`: no file above it, no folder at it, a parent folder.
    fn check_writable(&self, path: &Path) -> Result<()> {
        self.check_no_file_above(path)?;
        if self.folders.contains(path) {
            return Err(StorageError::from_kind(ErrorKind::IsADirectory));
        }
        match path.parent() {
            Some(parent) if self.folders.contains(parent) => Ok(()),
            _ => Err(StorageError::from_kind(ErrorKind::NotFound)),
        }
    }

    /// The next revision, which no file has had before.
    const fn next_revision(&mut self) -> u64 {
        let revision = self.next_revision;
        self.next_revision = self.next_revision.saturating_add(1);
        revision
    }

    /// Store `bytes` at `path`, which `check_writable` accepted, with a new revision and `modified`.
    fn store(&mut self, path: &Path, bytes: Vec<u8>, modified: SystemTime) -> RevisionToken {
        let revision = self.next_revision();
        self.files.insert(
            path.to_path_buf(),
            MemoryFile {
                bytes,
                modified,
                revision,
            },
        );
        token(revision)
    }
}

impl MemoryBucket {
    /// Add `more` to the file at `path`, creating it when missing.
    fn append_now(&self, path: &Path, more: &[u8]) -> Result<RevisionToken> {
        let mut state = self.write();
        state.check_writable(path)?;
        let modified = (self.clock)();
        let mut bytes = state
            .files
            .get(path)
            .map_or_else(Vec::new, |file| file.bytes.clone());
        bytes.extend_from_slice(more);
        let token = state.store(path, bytes, modified);
        drop(state);
        Ok(token)
    }

    /// Copy the file at `from` to `to`.
    fn copy_now(&self, from: &Path, to: &Path) -> Result<u64> {
        let mut state = self.write();
        state.check_no_file_above(from)?;
        if state.folders.contains(from) {
            return Err(StorageError::from_kind(ErrorKind::InvalidInput));
        }
        let bytes = state
            .files
            .get(from)
            .map(|file| file.bytes.clone())
            .ok_or_else(|| StorageError::from_kind(ErrorKind::NotFound))?;
        state.check_writable(to)?;
        let length = u64::try_from(bytes.len()).map_err(io::Error::other)?;
        let modified = (self.clock)();
        state.store(to, bytes, modified);
        drop(state);
        Ok(length)
    }

    /// Create `path` and every folder above it.
    fn create_folder_now(&self, path: &Path) -> Result<()> {
        let mut state = self.write();
        state.check_no_file_above(path)?;
        if state.files.contains_key(path) {
            return Err(StorageError::from_kind(ErrorKind::AlreadyExists));
        }
        state
            .folders
            .extend(path.ancestors().map(Path::to_path_buf));
        drop(state);
        Ok(())
    }

    /// Delete the file at `path` when `condition` holds.
    fn delete_now(&self, path: &Path, condition: &Condition) -> Result<()> {
        let mut state = self.write();
        state.check_no_file_above(path)?;
        if state.folders.contains(path) {
            return Err(StorageError::from_kind(ErrorKind::IsADirectory));
        }
        holds(condition, state.files.get(path))?;
        state.files.remove(path);
        drop(state);
        Ok(())
    }

    /// The bytes and token at `path`.
    fn get_now(&self, path: &Path) -> Result<(Vec<u8>, RevisionToken)> {
        let state = self.read();
        state.check_no_file_above(path)?;
        if state.folders.contains(path) {
            return Err(StorageError::from_kind(ErrorKind::IsADirectory));
        }
        let found = state
            .files
            .get(path)
            .map(|file| (file.bytes.clone(), token(file.revision)));
        drop(state);
        found.ok_or_else(|| StorageError::from_kind(ErrorKind::NotFound))
    }

    /// The metadata of what is at `path`.
    fn head_now(&self, path: &Path) -> Result<FileMetadata> {
        let state = self.read();
        state.check_no_file_above(path)?;
        if state.folders.contains(path) {
            return Ok(FileMetadata::new(true, false, 0, None, None));
        }
        let file = state
            .files
            .get(path)
            .ok_or_else(|| StorageError::from_kind(ErrorKind::NotFound))?;
        let length = u64::try_from(file.bytes.len()).map_err(io::Error::other)?;
        let metadata = FileMetadata::new(
            false,
            true,
            length,
            Some(file.modified),
            Some(token(file.revision)),
        );
        drop(state);
        Ok(metadata)
    }

    /// The entries below `folder`, to `depth`, sorted by path.
    fn list_now(&self, folder: &Path, depth: Depth) -> Result<Vec<WalkEntry>> {
        let state = self.read();
        state.check_no_file_above(folder)?;
        if state.files.contains_key(folder) {
            return Err(StorageError::from_kind(ErrorKind::NotADirectory));
        }
        let within = |path: &Path| match depth {
            Depth::All => path != folder && path.starts_with(folder),
            Depth::Children => path.parent() == Some(folder),
        };
        let mut entries: Vec<WalkEntry> = state
            .folders
            .iter()
            .filter(|path| within(path))
            .map(|path| WalkEntry::new(path.clone(), true, false))
            .chain(
                state
                    .files
                    .keys()
                    .filter(|path| within(path))
                    .map(|path| WalkEntry::new(path.clone(), false, true)),
            )
            .collect();
        drop(state);
        entries.sort_by(|left, right| left.path.cmp(&right.path));
        Ok(entries)
    }

    /// A bucket whose files take their modification time from `clock`.
    #[inline]
    #[must_use]
    pub fn new(clock: Clock) -> Self {
        Self {
            clock,
            state: Arc::new(RwLock::new(MemoryState {
                files: BTreeMap::new(),
                folders: BTreeSet::from([PathBuf::from("/")]),
                next_revision: 1,
            })),
        }
    }

    /// Write `bytes` at `path` when `condition` holds.
    fn put_now(&self, path: &Path, bytes: &[u8], condition: &Condition) -> Result<RevisionToken> {
        let mut state = self.write();
        state.check_writable(path)?;
        holds(condition, state.files.get(path))?;
        let modified = (self.clock)();
        let token = state.store(path, bytes.to_vec(), modified);
        drop(state);
        Ok(token)
    }

    /// The lock for reading the tree, whether or not a writer panicked.
    fn read(&self) -> RwLockReadGuard<'_, MemoryState> {
        self.state.read().unwrap_or_else(PoisonError::into_inner)
    }

    /// Remove `path` and everything below it.
    fn remove_folder_now(&self, path: &Path) -> Result<()> {
        let mut state = self.write();
        state.check_no_file_above(path)?;
        if state.files.contains_key(path) {
            return Err(StorageError::from_kind(ErrorKind::NotADirectory));
        }
        if !state.folders.contains(path) {
            return Err(StorageError::from_kind(ErrorKind::NotFound));
        }
        state.files.retain(|file, _| !file.starts_with(path));
        state
            .folders
            .retain(|folder| !folder.starts_with(path) || folder.parent().is_none());
        drop(state);
        Ok(())
    }

    /// Move the file or the folder at `from` to `to`.
    fn rename_now(&self, from: &Path, to: &Path) -> Result<()> {
        let mut state = self.write();
        state.check_no_file_above(from)?;
        state.check_no_file_above(to)?;
        let outcome = if state.files.contains_key(from) {
            rename_file_now(&mut state, from, to)
        } else if state.folders.contains(from) {
            rename_folder_now(&mut state, from, to)
        } else {
            Err(StorageError::from_kind(ErrorKind::NotFound))
        };
        drop(state);
        outcome
    }

    /// The lock for changing the tree, whether or not a writer panicked.
    fn write(&self) -> RwLockWriteGuard<'_, MemoryState> {
        self.state.write().unwrap_or_else(PoisonError::into_inner)
    }

    /// A writer to `path` that appends, or writes the whole file under `put_under`.
    fn writer<'bucket>(
        &self,
        path: &Path,
        put_under: Option<Condition>,
    ) -> Result<Box<dyn AsyncWrite + Send + Unpin + 'bucket>> {
        Ok(Box::new(MemoryWriter {
            bucket: self.clone(),
            buffer: Vec::new(),
            path: lexical(path)?,
            put_under,
            stored: false,
        }))
    }
}

impl Bucket for MemoryBucket {
    #[inline]
    fn append<'call>(
        &'call self,
        path: &'call Path,
        more: &'call [u8],
    ) -> BoxFuture<'call, Result<RevisionToken>> {
        Box::pin(ready(
            lexical(path).and_then(|resolved| self.append_now(&resolved, more)),
        ))
    }

    #[inline]
    fn append_stream<'bucket, 'call>(
        &'bucket self,
        path: &'call Path,
    ) -> BoxFuture<'call, Result<Box<dyn AsyncWrite + Send + Unpin + 'bucket>>>
    where
        'bucket: 'call,
    {
        Box::pin(ready(self.writer(path, None)))
    }

    #[inline]
    fn canonicalize<'call>(&'call self, path: &'call Path) -> BoxFuture<'call, Result<PathBuf>> {
        Box::pin(ready(lexical(path)))
    }

    #[inline]
    fn copy<'call>(
        &'call self,
        from: &'call Path,
        to: &'call Path,
    ) -> BoxFuture<'call, Result<u64>> {
        Box::pin(ready(
            both(from, to).and_then(|(source, target)| self.copy_now(&source, &target)),
        ))
    }

    #[inline]
    fn create_folder<'call>(&'call self, path: &'call Path) -> BoxFuture<'call, Result<()>> {
        Box::pin(ready(
            lexical(path).and_then(|resolved| self.create_folder_now(&resolved)),
        ))
    }

    #[inline]
    fn delete<'call>(
        &'call self,
        path: &'call Path,
        condition: &'call Condition,
    ) -> BoxFuture<'call, Result<()>> {
        Box::pin(ready(
            lexical(path).and_then(|resolved| self.delete_now(&resolved, condition)),
        ))
    }

    #[inline]
    fn get<'call>(
        &'call self,
        path: &'call Path,
    ) -> BoxFuture<'call, Result<(Vec<u8>, RevisionToken)>> {
        Box::pin(ready(
            lexical(path).and_then(|resolved| self.get_now(&resolved)),
        ))
    }

    #[inline]
    fn guarantees(&self) -> Guarantees {
        const PROMISED: Guarantees = Guarantees::none()
            .with(Guarantee::AppendIsAtomic)
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
            .with(Guarantee::ReadersNeverSeePartialWrites)
            .with(Guarantee::RenameKeepsModificationTime);
        PROMISED
    }

    #[inline]
    fn head<'call>(&'call self, path: &'call Path) -> BoxFuture<'call, Result<FileMetadata>> {
        Box::pin(ready(
            lexical(path).and_then(|resolved| self.head_now(&resolved)),
        ))
    }

    #[inline]
    fn list<'call>(
        &'call self,
        folder: &'call Path,
        depth: Depth,
    ) -> BoxFuture<'call, Result<Vec<WalkEntry>>> {
        Box::pin(ready(
            lexical(folder).and_then(|resolved| self.list_now(&resolved, depth)),
        ))
    }

    #[inline]
    fn put<'call>(
        &'call self,
        path: &'call Path,
        bytes: &'call [u8],
        condition: &'call Condition,
    ) -> BoxFuture<'call, Result<RevisionToken>> {
        Box::pin(ready(
            lexical(path).and_then(|resolved| self.put_now(&resolved, bytes, condition)),
        ))
    }

    #[inline]
    fn read_stream<'bucket, 'call>(
        &'bucket self,
        path: &'call Path,
    ) -> BoxFuture<'call, Result<Box<dyn AsyncRead + Send + Unpin + 'bucket>>>
    where
        'bucket: 'call,
    {
        let outcome = lexical(path)
            .and_then(|resolved| self.get_now(&resolved))
            .map(|(bytes, _token)| {
                let reader: Box<dyn AsyncRead + Send + Unpin + 'bucket> =
                    Box::new(BytesReader::new(bytes));
                reader
            });
        Box::pin(ready(outcome))
    }

    #[inline]
    fn remove_folder<'call>(&'call self, path: &'call Path) -> BoxFuture<'call, Result<()>> {
        Box::pin(ready(
            lexical(path).and_then(|resolved| self.remove_folder_now(&resolved)),
        ))
    }

    #[inline]
    fn rename<'call>(
        &'call self,
        from: &'call Path,
        to: &'call Path,
    ) -> BoxFuture<'call, Result<()>> {
        Box::pin(ready(
            both(from, to).and_then(|(source, target)| self.rename_now(&source, &target)),
        ))
    }

    #[inline]
    fn walk<'call>(
        &'call self,
        folder: &'call Path,
        _follow_links: bool,
        hidden: bool,
    ) -> BoxFuture<'call, Result<Vec<WalkEntry>>> {
        Box::pin(ready(lexical(folder).and_then(|base| {
            let mut entries = self.list_now(&base, Depth::All)?;
            if !hidden {
                entries.retain(|entry| !is_hidden(&base, &entry.path));
            }
            Ok(entries)
        })))
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
        Box::pin(ready(self.writer(path, Some(condition))))
    }
}

/// Both paths in their lexical form.
fn both(from: &Path, to: &Path) -> Result<(PathBuf, PathBuf)> {
    Ok((lexical(from)?, lexical(to)?))
}

/// The token of `revision`.
fn token(revision: u64) -> RevisionToken {
    RevisionToken::new(revision.to_string())
}

/// `Ok` when `condition` holds for the file now at the path, if any.
fn holds(condition: &Condition, current: Option<&MemoryFile>) -> Result<()> {
    match (condition.clone(), current) {
        (Condition::Absent, Some(_)) => Err(StorageError::from_kind(ErrorKind::AlreadyExists)),
        (Condition::Matches(wanted), Some(file)) if wanted == token(file.revision) => Ok(()),
        (Condition::Matches(_), _) => Err(StorageError::from_kind(ErrorKind::PreconditionFailed)),
        _ => Ok(()),
    }
}

/// Move the file at `from`, which exists, to `to`.
fn rename_file_now(state: &mut MemoryState, from: &Path, to: &Path) -> Result<()> {
    if state.folders.contains(to) {
        return Err(StorageError::from_kind(ErrorKind::IsADirectory));
    }
    match to.parent() {
        Some(parent) if state.folders.contains(parent) => {}
        _ => return Err(StorageError::from_kind(ErrorKind::NotFound)),
    }
    if let Some(file) = state.files.remove(from) {
        state.files.insert(to.to_path_buf(), file);
    }
    Ok(())
}

/// Move the folder at `from`, which exists, and everything below it to `to`.
fn rename_folder_now(state: &mut MemoryState, from: &Path, to: &Path) -> Result<()> {
    if state.files.contains_key(to) {
        return Err(StorageError::from_kind(ErrorKind::NotADirectory));
    }
    match to.parent() {
        Some(parent) if state.folders.contains(parent) => {}
        _ => return Err(StorageError::from_kind(ErrorKind::NotFound)),
    }
    if to == from {
        return Ok(());
    }
    if from.parent().is_none() || to.starts_with(from) {
        return Err(StorageError::from_kind(ErrorKind::InvalidInput));
    }
    let occupied = state
        .files
        .keys()
        .chain(state.folders.iter())
        .any(|path| path.parent() == Some(to));
    if occupied {
        return Err(StorageError::from_kind(ErrorKind::AlreadyExists));
    }
    let relocate = |path: &Path| to.join(path.strip_prefix(from).unwrap_or(path));
    let files = mem::take(&mut state.files);
    state.files = files
        .into_iter()
        .map(|(path, file)| {
            if path.starts_with(from) {
                (relocate(&path), file)
            } else {
                (path, file)
            }
        })
        .collect::<BTreeMap<_, _>>();
    let folders = mem::take(&mut state.folders);
    state.folders = folders
        .into_iter()
        .map(|path| {
            if path.starts_with(from) {
                relocate(&path)
            } else {
                path
            }
        })
        .collect::<BTreeSet<_>>();
    state.folders.insert(to.to_path_buf());
    Ok(())
}
