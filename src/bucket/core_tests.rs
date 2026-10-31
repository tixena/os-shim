//! Tests that run in every build, with no client feature: `MemoryBucket`, and the trait's default
//! bodies on a bucket that implements only what the trait requires.

use super::{Bucket, Clock, Condition, Depth, Guarantee, Guarantees, MemoryBucket, RevisionToken};
use crate::error::{ErrorKind, Result, StorageError};
use crate::{BoxFuture, FileMetadata, WalkEntry};
use alloc::sync::Arc;
use core::error::Error;
use core::future::{poll_fn, ready};
use core::pin::Pin;
use core::result;
use core::sync::atomic::{AtomicU64, Ordering};
use core::time::Duration;
use futures_io::{AsyncRead, AsyncWrite};
use std::io;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

macro_rules! forward_required {
    ($name:ident { $($early:tt)* } { $($late:tt)* }) => {
        impl Bucket for $name {
            $($early)*

            fn delete<'call>(&'call self, path: &'call Path, condition: &'call Condition) -> BoxFuture<'call, Result<()>> {
Box::pin(self.0.delete(path, condition))
}

            fn get<'call>(&'call self, path: &'call Path) -> BoxFuture<'call, Result<(Vec<u8>, RevisionToken)>> {
Box::pin(self.0.get(path))
}

            fn guarantees(&self) -> Guarantees {
                self.0.guarantees()
            }

            fn head<'call>(&'call self, path: &'call Path) -> BoxFuture<'call, Result<FileMetadata>> {
Box::pin(self.0.head(path))
}

            fn list<'call>(&'call self, folder: &'call Path, depth: Depth) -> BoxFuture<'call, Result<Vec<WalkEntry>>> {
Box::pin(self.0.list(folder, depth))
}

            fn put<'call>(&'call self, path: &'call Path, bytes: &'call [u8], condition: &'call Condition) -> BoxFuture<'call, Result<RevisionToken>> {
Box::pin(self.0.put(path, bytes, condition))
}

            $($late)*
        }
    };
}

/// What a scenario saw, line by line, or why it could not go on.
pub(super) type Observed = result::Result<Vec<String>, Box<dyn Error>>;

/// A bucket whose `list` is refused, so the default `reachable` has an error to give.
struct ListRefused(MemoryBucket);

impl Bucket for ListRefused {
    fn delete<'call>(
        &'call self,
        path: &'call Path,
        condition: &'call Condition,
    ) -> BoxFuture<'call, Result<()>> {
        Box::pin(self.0.delete(path, condition))
    }

    fn get<'call>(
        &'call self,
        path: &'call Path,
    ) -> BoxFuture<'call, Result<(Vec<u8>, RevisionToken)>> {
        Box::pin(self.0.get(path))
    }

    fn guarantees(&self) -> Guarantees {
        self.0.guarantees()
    }

    fn head<'call>(&'call self, path: &'call Path) -> BoxFuture<'call, Result<FileMetadata>> {
        Box::pin(self.0.head(path))
    }

    fn list<'call>(
        &'call self,
        _folder: &'call Path,
        _depth: Depth,
    ) -> BoxFuture<'call, Result<Vec<WalkEntry>>> {
        Box::pin(ready(Err(StorageError::from_kind(ErrorKind::Unavailable))))
    }

    fn put<'call>(
        &'call self,
        path: &'call Path,
        bytes: &'call [u8],
        condition: &'call Condition,
    ) -> BoxFuture<'call, Result<RevisionToken>> {
        Box::pin(self.0.put(path, bytes, condition))
    }
}

/// A bucket that implements only the required methods, so every other method is the trait's default.
struct RequiredOnly(MemoryBucket);

forward_required!(RequiredOnly {} {});

/// The required methods and the two folder methods, which have no useful default.
struct WithFolders(MemoryBucket);

forward_required!(WithFolders {
    fn create_folder<'call>(&'call self, path: &'call Path) -> BoxFuture<'call, Result<()>> {
Box::pin(self.0.create_folder(path))
}
} {
    fn remove_folder<'call>(&'call self, path: &'call Path) -> BoxFuture<'call, Result<()>> {
Box::pin(self.0.remove_folder(path))
}
});

/// A bucket whose file changes just before a delete under a token, so the token is stale.
struct ChangesBeforeDelete(MemoryBucket);

impl Bucket for ChangesBeforeDelete {
    fn delete<'call>(
        &'call self,
        path: &'call Path,
        condition: &'call Condition,
    ) -> BoxFuture<'call, Result<()>> {
        Box::pin(async move {
            if matches!(*condition, Condition::Matches(_)) {
                self.0.put(path, b"changed", &Condition::None).await?;
            }
            self.0.delete(path, condition).await
        })
    }

    fn get<'call>(
        &'call self,
        path: &'call Path,
    ) -> BoxFuture<'call, Result<(Vec<u8>, RevisionToken)>> {
        Box::pin(self.0.get(path))
    }

    fn guarantees(&self) -> Guarantees {
        self.0.guarantees()
    }

    fn head<'call>(&'call self, path: &'call Path) -> BoxFuture<'call, Result<FileMetadata>> {
        Box::pin(self.0.head(path))
    }

    fn list<'call>(
        &'call self,
        folder: &'call Path,
        depth: Depth,
    ) -> BoxFuture<'call, Result<Vec<WalkEntry>>> {
        Box::pin(self.0.list(folder, depth))
    }

    fn put<'call>(
        &'call self,
        path: &'call Path,
        bytes: &'call [u8],
        condition: &'call Condition,
    ) -> BoxFuture<'call, Result<RevisionToken>> {
        Box::pin(self.0.put(path, bytes, condition))
    }
}

/// Assert that a scenario saw exactly `expected`.
pub(super) fn assert_observed(observed: Observed, expected: &[&str]) {
    assert_eq!(
        observed.map_err(|error| error.to_string()),
        Ok(expected.iter().map(ToString::to_string).collect())
    );
}

/// A clock the test moves, in seconds after the Unix epoch.
fn moving_clock() -> (Clock, Arc<AtomicU64>) {
    let seconds = Arc::new(AtomicU64::new(1_000));
    let read = Arc::clone(&seconds);
    let clock: Clock = Arc::new(move || {
        SystemTime::UNIX_EPOCH
            .checked_add(Duration::from_secs(read.load(Ordering::SeqCst)))
            .unwrap_or(SystemTime::UNIX_EPOCH)
    });
    (clock, seconds)
}

/// What an outcome looks like with its tokens hidden: `ok`, or the kind of its error.
pub(super) fn shape<T>(outcome: &Result<T>) -> String {
    match outcome.as_ref() {
        Ok(_) => "ok".to_owned(),
        Err(error) => format!("{:?}", error.kind()),
    }
}

/// `path` with the folder `root` standing for `/`.
fn rooted(path: &Path, root: &Path) -> PathBuf {
    Path::new("/").join(path.strip_prefix(root).unwrap_or(path))
}

/// The paths of `entries`, each shown with `root` standing for `/`.
fn paths(entries: &[WalkEntry], root: &Path) -> Vec<String> {
    entries
        .iter()
        .map(|entry| rooted(&entry.path, root).display().to_string())
        .collect()
}

/// Every byte `reader` gives.
async fn read_all(reader: &mut (dyn AsyncRead + Send + Unpin + '_)) -> io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    let mut chunk = [0_u8; 4];
    loop {
        let count = poll_fn(|cx| Pin::new(&mut *reader).poll_read(cx, &mut chunk)).await?;
        if count == 0 {
            return Ok(bytes);
        }
        bytes.extend_from_slice(chunk.get(..count).unwrap_or_default());
    }
}

/// Give `bytes` to `writer`.
async fn write_all(
    writer: &mut (dyn AsyncWrite + Send + Unpin + '_),
    bytes: &[u8],
) -> io::Result<()> {
    let mut rest = bytes;
    while !rest.is_empty() {
        let count = poll_fn(|cx| Pin::new(&mut *writer).poll_write(cx, rest)).await?;
        rest = rest.get(count..).unwrap_or_default();
    }
    Ok(())
}

/// Close `writer`.
async fn close(writer: &mut (dyn AsyncWrite + Send + Unpin + '_)) -> io::Result<()> {
    poll_fn(|cx| Pin::new(&mut *writer).poll_close(cx)).await
}

/// Files: put, append, copy, rename and the heads and gets that show them.
pub(super) async fn file_operations<B>(bucket: &B, root: &Path) -> Observed
where
    B: Bucket,
{
    let folder = root.join("d");
    let first = folder.join("a.txt");
    let second = folder.join("b.txt");
    let mut lines = Vec::new();
    bucket.create_folder(&folder).await?;
    lines.push(shape(&bucket.put(&first, b"one", &Condition::None).await));
    lines.push(shape(&bucket.append(&first, b"-two").await));
    lines.push(shape(&bucket.append(&second, b"new").await));
    lines.push(format!("get {:?}", bucket.get(&first).await?.0));
    lines.push(format!("get {:?}", bucket.get(&second).await?.0));
    lines.push(format!(
        "copy {:?}",
        bucket.copy(&first, &folder.join("c.txt")).await?
    ));
    lines.push(shape(&bucket.copy(&folder.join("none"), &second).await));
    lines.push(shape(&bucket.rename(&first, &folder.join("e.txt")).await));
    lines.push(shape(&bucket.rename(&first, &folder.join("f.txt")).await));
    lines.push(shape(&bucket.head(&first).await));
    lines.push(format!(
        "renamed {:?}",
        bucket.get(&folder.join("e.txt")).await?.0
    ));
    Ok(lines)
}

/// Folders: a subtree renamed, walked with and without hidden entries, and removed.
pub(super) async fn folder_operations<B>(bucket: &B, root: &Path) -> Observed
where
    B: Bucket,
{
    let folder = root.join("d");
    let moved = root.join("moved");
    let mut lines = Vec::new();
    bucket.create_folder(&folder.join("sub/deeper")).await?;
    bucket
        .put(&folder.join("sub/deeper/x.txt"), b"x", &Condition::None)
        .await?;
    bucket
        .put(&folder.join(".hidden"), b"h", &Condition::None)
        .await?;
    lines.push(shape(&bucket.rename(&folder, &moved).await));
    lines.push(format!("old {}", shape(&bucket.head(&folder).await)));
    lines.push(format!(
        "moved {:?}",
        bucket.get(&moved.join("sub/deeper/x.txt")).await?.0
    ));
    for hidden in [true, false] {
        let walked = bucket.walk(&moved, false, hidden).await?;
        lines.push(format!("walk hidden={hidden} {:?}", paths(&walked, root)));
    }
    let resolved = bucket
        .canonicalize(&moved.join("sub/./deeper/../deeper"))
        .await?;
    lines.push(format!("canonicalize {:?}", rooted(&resolved, root)));
    lines.push(shape(&bucket.canonicalize(Path::new("relative")).await));
    lines.push(shape(&bucket.remove_folder(&moved).await));
    lines.push(shape(&bucket.head(&moved).await));
    Ok(lines)
}

/// Streams: a reader, a writer closed, a second create-only writer refused at open or close, and an appender.
pub(super) async fn stream_operations<B>(bucket: &B, root: &Path) -> Observed
where
    B: Bucket,
{
    let source = root.join("s.txt");
    let target = root.join("streamed.txt");
    let mut lines = Vec::new();
    bucket.put(&source, b"abcdefghij", &Condition::None).await?;
    let mut reader = bucket.read_stream(&source).await?;
    lines.push(format!("read {:?}", read_all(&mut *reader).await?));
    drop(reader);
    let mut writer = bucket.write_stream(&target, Condition::Absent).await?;
    write_all(&mut *writer, b"stream").await?;
    close(&mut *writer).await?;
    drop(writer);
    lines.push(format!("streamed {:?}", bucket.get(&target).await?.0));
    let refused = match bucket.write_stream(&target, Condition::Absent).await {
        Err(error) => Some(error.kind()),
        Ok(mut again) => {
            write_all(&mut *again, b"late").await?;
            close(&mut *again)
                .await
                .map_err(StorageError::from)
                .err()
                .map(|error| error.kind())
        }
    };
    lines.push(format!("second {refused:?}"));
    let mut tail = bucket.append_stream(&target).await?;
    write_all(&mut *tail, b"+tail").await?;
    close(&mut *tail).await?;
    drop(tail);
    lines.push(format!("appended {:?}", bucket.get(&target).await?.0));
    Ok(lines)
}

/// Files, folders and streams, one after the other on the same bucket.
pub(super) async fn every_operation<B>(bucket: &B, root: &Path) -> Observed
where
    B: Bucket,
{
    let mut lines = file_operations(bucket, root).await?;
    lines.extend(folder_operations(bucket, root).await?);
    lines.extend(stream_operations(bucket, root).await?);
    Ok(lines)
}

/// Two callers race for a lock; the second is refused and the first bytes stay.
pub(super) async fn lock_taken_twice<B>(bucket: &B, root: &Path) -> Observed
where
    B: Bucket,
{
    let jobs = root.join("jobs");
    bucket.create_folder(&jobs).await?;
    let lock = jobs.join("42.lock");
    let mut lines = vec![shape(
        &bucket.put(&lock, b"worker-a", &Condition::Absent).await,
    )];
    lines.push(shape(
        &bucket.put(&lock, b"worker-b", &Condition::Absent).await,
    ));
    lines.push(format!("{:?}", bucket.get(&lock).await?.0));
    Ok(lines)
}

/// A release with a token from before an append is refused; the current token releases.
pub(super) async fn stale_release<B>(bucket: &B, root: &Path) -> Observed
where
    B: Bucket,
{
    let jobs = root.join("jobs");
    bucket.create_folder(&jobs).await?;
    let lock = jobs.join("42.lock");
    let ours = bucket.put(&lock, b"worker-a", &Condition::Absent).await?;
    bucket.append(&lock, b" renewed").await?;
    let mut lines = vec![shape(
        &bucket.delete(&lock, &Condition::Matches(ours)).await,
    )];
    let (bytes, token) = bucket.get(&lock).await?;
    lines.push(format!("{bytes:?}"));
    lines.push(shape(
        &bucket.delete(&lock, &Condition::Matches(token)).await,
    ));
    lines.push(shape(&bucket.get(&lock).await));
    Ok(lines)
}

/// Each condition on a path with nothing at it.
pub(super) async fn conditions_on_a_missing_file<B>(bucket: &B, root: &Path) -> Observed
where
    B: Bucket,
{
    let path = root.join("none.md");
    let stale = RevisionToken::new("1");
    Ok(vec![
        shape(&bucket.delete(&path, &Condition::None).await),
        shape(&bucket.delete(&path, &Condition::Absent).await),
        shape(
            &bucket
                .delete(&path, &Condition::Matches(stale.clone()))
                .await,
        ),
        shape(&bucket.put(&path, b"x", &Condition::Matches(stale)).await),
    ])
}

/// The tokens of a path deleted and written again, of two paths with the same bytes, and of an append.
async fn tokens_never_repeat() -> Observed {
    let bucket = MemoryBucket::default();
    let path = Path::new("/a.md");
    let first = bucket.put(path, b"x", &Condition::None).await?;
    bucket.delete(path, &Condition::None).await?;
    let second = bucket.put(path, b"x", &Condition::None).await?;
    let other = bucket
        .put(Path::new("/b.md"), b"x", &Condition::None)
        .await?;
    let third = bucket.append(path, b"y").await?;
    let all = [first, second, other, third];
    Ok(all
        .iter()
        .enumerate()
        .map(|(index, token)| {
            format!(
                "{} repeats: {}",
                token.as_str(),
                all.iter()
                    .skip(index.saturating_add(1))
                    .any(|later| later == token)
            )
        })
        .collect())
}

/// A put under a missing parent, a file, a folder and the root; folder operations on a file.
pub(super) async fn parent_rules<B>(bucket: &B, root: &Path) -> Observed
where
    B: Bucket,
{
    let folder = root.join("folder");
    let file = root.join("file");
    bucket.create_folder(&folder).await?;
    bucket.put(&file, b"x", &Condition::None).await?;
    let mut lines = Vec::new();
    for target in [
        root.join("missing/a"),
        file.join("a"),
        folder.clone(),
        root.to_path_buf(),
    ] {
        lines.push(shape(&bucket.put(&target, b"y", &Condition::None).await));
    }
    lines.push(shape(&bucket.create_folder(&file.join("sub")).await));
    lines.push(shape(&bucket.remove_folder(&file).await));
    lines.push(shape(&bucket.copy(&folder, &root.join("copy")).await));
    lines.push(shape(&bucket.get(Path::new("relative")).await));
    lines.push(shape(&bucket.get(Path::new("/../a")).await));
    Ok(lines)
}

/// Two writes with the clock moved by 60 seconds between them.
async fn clock_stamps() -> Observed {
    let (clock, seconds) = moving_clock();
    let bucket = MemoryBucket::new(clock);
    let path = Path::new("/a.md");
    bucket.put(path, b"x", &Condition::None).await?;
    let before = bucket.head(path).await?.modified;
    seconds.fetch_add(60, Ordering::SeqCst);
    bucket.append(path, b"y").await?;
    let after = bucket.head(path).await?.modified;
    let difference = after
        .zip(before)
        .and_then(|(later, earlier)| later.duration_since(earlier).ok());
    Ok(vec![
        format!("{difference:?}"),
        format!("root {:?}", bucket.head(Path::new("/")).await?.modified),
    ])
}

/// A subtree renamed under one lock.
pub(super) async fn folder_rename<B>(bucket: &B, root: &Path) -> Observed
where
    B: Bucket,
{
    let source = root.join("a");
    let target = root.join("c");
    bucket.create_folder(&source.join("b")).await?;
    bucket
        .put(&source.join("b/x"), b"1", &Condition::None)
        .await?;
    bucket
        .put(&source.join("y"), b"2", &Condition::None)
        .await?;
    bucket.rename(&source, &target).await?;
    Ok(vec![
        format!("{:?}", bucket.get(&target.join("b/x")).await?.0),
        format!("{:?}", bucket.get(&target.join("y")).await?.0),
        shape(&bucket.head(&source).await),
        format!("{}", bucket.head(&target.join("b")).await?.is_dir),
    ])
}

/// A listing at each depth, of a file, and of a folder that is not there.
pub(super) async fn listings<B>(bucket: &B, root: &Path) -> Observed
where
    B: Bucket,
{
    let folder = root.join("a");
    bucket.create_folder(&folder.join("b")).await?;
    bucket
        .put(&folder.join("b/x"), b"1", &Condition::None)
        .await?;
    bucket
        .put(&folder.join("y"), b"2", &Condition::None)
        .await?;
    Ok(vec![
        format!(
            "{:?}",
            paths(&bucket.list(&folder, Depth::Children).await?, root)
        ),
        format!(
            "{:?}",
            paths(&bucket.list(&folder, Depth::All).await?, root)
        ),
        shape(&bucket.list(&folder.join("y"), Depth::All).await),
        format!(
            "{:?}",
            paths(&bucket.list(&root.join("none"), Depth::All).await?, root)
        ),
    ])
}

/// A clone sees what the other writes.
async fn clones() -> Observed {
    let ours = MemoryBucket::default();
    let theirs = ours.clone();
    ours.put(Path::new("/a.md"), b"x", &Condition::None).await?;
    Ok(vec![
        format!("{:?}", theirs.get(Path::new("/a.md")).await?.0),
        format!("{ours:?}"),
    ])
}

/// A writer dropped without `close` keeps the bytes it was given when the bucket promises it.
pub(super) async fn dropped_writers<B>(bucket: &B, root: &Path) -> Observed
where
    B: Bucket,
{
    let path = root.join("a.md");
    let mut writer = bucket.write_stream(&path, Condition::Absent).await?;
    write_all(&mut *writer, b"kept").await?;
    drop(writer);
    Ok(vec![format!("native {:?}", bucket.get(&path).await?.0)])
}

/// A writer dropped without `close` on a bucket with the default writer.
async fn dropped_default_writer() -> Observed {
    let path = Path::new("/a.md");
    let defaults = RequiredOnly(MemoryBucket::default());
    let mut lost = defaults.write_stream(path, Condition::Absent).await?;
    write_all(&mut *lost, b"lost").await?;
    drop(lost);
    Ok(vec![format!(
        "default {}",
        shape(&defaults.get(path).await)
    )])
}

/// A rename whose source changed between the read and the delete: the copy is removed again.
async fn failed_default_rename() -> Observed {
    let bucket = ChangesBeforeDelete(MemoryBucket::default());
    let from = Path::new("/from");
    let to = Path::new("/to");
    bucket.put(from, b"x", &Condition::None).await?;
    Ok(vec![
        shape(&bucket.rename(from, to).await),
        format!("{:?}", bucket.get(from).await?.0),
        shape(&bucket.head(to).await),
    ])
}

/// The folder defaults of a bucket that implements only what the trait requires.
async fn folder_defaults() -> Observed {
    let bucket = RequiredOnly(MemoryBucket::default());
    Ok(vec![
        shape(&bucket.create_folder(Path::new("/d")).await),
        shape(&bucket.remove_folder(Path::new("/d")).await),
    ])
}

/// A file renamed by the default body.
async fn default_rename() -> Observed {
    let memory = MemoryBucket::default();
    let bucket = RequiredOnly(memory.clone());
    let from = Path::new("/from");
    let to = Path::new("/to");
    memory.put(from, b"x", &Condition::None).await?;
    let renamed = bucket.rename(from, to).await;
    Ok(vec![
        shape(&renamed),
        shape(&bucket.head(from).await),
        shape(&bucket.head(to).await),
    ])
}

#[tokio::test(flavor = "current_thread")]
async fn a_lock_taken_twice_is_already_exists_and_keeps_the_first_bytes() {
    assert_observed(
        lock_taken_twice(&MemoryBucket::default(), Path::new("/")).await,
        &[
            "ok",
            "AlreadyExists",
            "[119, 111, 114, 107, 101, 114, 45, 97]",
        ],
    );
}

#[tokio::test(flavor = "current_thread")]
async fn a_stale_release_is_precondition_failed_and_the_file_stays() {
    assert_observed(
        stale_release(&MemoryBucket::default(), Path::new("/")).await,
        &[
            "PreconditionFailed",
            "[119, 111, 114, 107, 101, 114, 45, 97, 32, 114, 101, 110, 101, 119, 101, 100]",
            "ok",
            "NotFound",
        ],
    );
}

#[tokio::test(flavor = "current_thread")]
async fn conditions_on_a_missing_file_are_checked() {
    assert_observed(
        conditions_on_a_missing_file(&MemoryBucket::default(), Path::new("/")).await,
        &["ok", "ok", "PreconditionFailed", "PreconditionFailed"],
    );
}

#[tokio::test(flavor = "current_thread")]
async fn a_token_is_never_handed_out_twice() {
    assert_observed(
        tokens_never_repeat().await,
        &[
            "1 repeats: false",
            "2 repeats: false",
            "3 repeats: false",
            "4 repeats: false",
        ],
    );
}

#[tokio::test(flavor = "current_thread")]
async fn the_parent_rules() {
    assert_observed(
        parent_rules(&MemoryBucket::default(), Path::new("/")).await,
        &[
            "NotFound",
            "NotADirectory",
            "IsADirectory",
            "IsADirectory",
            "NotADirectory",
            "NotADirectory",
            "InvalidInput",
            "InvalidInput",
            "InvalidInput",
        ],
    );
}

#[tokio::test(flavor = "current_thread")]
async fn every_write_takes_its_time_from_the_clock() {
    assert_observed(clock_stamps().await, &["Some(60s)", "root None"]);
}

#[tokio::test(flavor = "current_thread")]
async fn renaming_a_folder_moves_its_subtree() {
    assert_observed(
        folder_rename(&MemoryBucket::default(), Path::new("/")).await,
        &["[49]", "[50]", "NotFound", "true"],
    );
}

#[tokio::test(flavor = "current_thread")]
async fn a_listing_reaches_the_depth_it_is_given() {
    assert_observed(
        listings(&MemoryBucket::default(), Path::new("/")).await,
        &[
            "[\"/a/b\", \"/a/y\"]",
            "[\"/a/b\", \"/a/b/x\", \"/a/y\"]",
            "NotADirectory",
            "[]",
        ],
    );
}

#[tokio::test(flavor = "current_thread")]
async fn clones_share_the_files() {
    assert_observed(
        clones().await,
        &["[120]", "MemoryBucket { files: 1, folders: 1 }"],
    );
}

#[tokio::test(flavor = "current_thread")]
async fn a_write_stream_dropped_without_close_stores_its_bytes_only_when_native() {
    let native = dropped_writers(&MemoryBucket::default(), Path::new("/")).await;
    let default = dropped_default_writer().await;
    assert_observed(
        native.and_then(|mut lines| {
            default.map(|more| {
                lines.extend(more);
                lines
            })
        }),
        &["native [107, 101, 112, 116]", "default NotFound"],
    );
}

#[tokio::test(flavor = "current_thread")]
async fn the_folder_defaults_are_unsupported() {
    assert_observed(folder_defaults().await, &["Unsupported", "Unsupported"]);
}

#[tokio::test(flavor = "current_thread")]
async fn the_default_rename_moves_a_file() {
    assert_observed(default_rename().await, &["ok", "NotFound", "ok"]);
}

#[tokio::test(flavor = "current_thread")]
async fn a_failed_delete_in_the_default_rename_removes_the_copy() {
    assert_observed(
        failed_default_rename().await,
        &[
            "PreconditionFailed",
            "[99, 104, 97, 110, 103, 101, 100]",
            "NotFound",
        ],
    );
}

#[tokio::test(flavor = "current_thread")]
async fn the_defaults_agree_with_the_memory_bucket() {
    let native = every_operation(&MemoryBucket::default(), Path::new("/"))
        .await
        .map_err(|error| error.to_string());
    let defaults = every_operation(&WithFolders(MemoryBucket::default()), Path::new("/"))
        .await
        .map_err(|error| error.to_string());
    assert_eq!(defaults, native);
    assert_eq!(native.map(|lines| lines.len()), Ok(24));
}

#[tokio::test(flavor = "current_thread")]
async fn the_memory_bucket_promises_what_it_does() {
    let promised = MemoryBucket::default().guarantees();
    assert!(promised.contains(Guarantee::FuturesNeverWait));
    assert!(promised.contains(Guarantee::DroppedWriterKeepsBytes));
    assert!(promised.contains(Guarantee::RenameKeepsModificationTime));
    assert!(!promised.contains(Guarantee::WalkHonorsGitignore));
}

#[test]
fn a_set_holds_each_of_the_sixteen_guarantees_and_only_those_it_was_given() {
    let every = [
        Guarantee::AppendIsAtomic,
        Guarantee::AppendWritesOnlyNewBytes,
        Guarantee::ConditionalDeleteInOneStep,
        Guarantee::ConditionalWriteInOneStep,
        Guarantee::CreateOnlyWriteInOneStep,
        Guarantee::DroppedWriterKeepsBytes,
        Guarantee::EmptyFoldersExist,
        Guarantee::FailedRenameKeepsDestination,
        Guarantee::FileAndFolderNeverShareAPath,
        Guarantee::FileRenameIsAtomic,
        Guarantee::FolderRenameIsAtomic,
        Guarantee::FuturesNeverWait,
        Guarantee::ParentCheckInOneStep,
        Guarantee::ReadersNeverSeePartialWrites,
        Guarantee::RenameKeepsModificationTime,
        Guarantee::WalkHonorsGitignore,
    ];
    let all = every
        .iter()
        .fold(Guarantees::none(), |set, guarantee| set.with(*guarantee));
    assert_eq!(every.map(|guarantee| all.contains(guarantee)), [true; 16]);
    for alone in every {
        let set = Guarantees::none().with(alone);
        assert_eq!(
            every.map(|guarantee| set.contains(guarantee)),
            every.map(|guarantee| guarantee == alone)
        );
    }
}

#[tokio::test(flavor = "current_thread")]
async fn reachable_by_default_answers_ok_when_list_answers() -> Result<()> {
    RequiredOnly(MemoryBucket::default()).reachable().await
}

#[tokio::test(flavor = "current_thread")]
async fn reachable_by_default_gives_the_error_of_list() {
    let answer = ListRefused(MemoryBucket::default()).reachable().await;
    assert_eq!(
        answer.map_err(|error| error.kind()),
        Err(ErrorKind::Unavailable)
    );
}
