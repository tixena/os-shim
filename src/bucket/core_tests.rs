//! Tests that run in every build, with no client feature: `MemoryBucket`, and the trait's default
//! bodies on a bucket that implements only what the trait requires.

use super::{Bucket, Clock, Condition, Depth, Guarantee, Guarantees, MemoryBucket, RevisionToken};
use crate::error::{Result, StorageError};
use crate::{FileMetadata, WalkEntry};
use alloc::sync::Arc;
use core::error::Error;
use core::future::poll_fn;
use core::pin::Pin;
use core::result;
use core::sync::atomic::{AtomicU64, Ordering};
use core::time::Duration;
use futures_io::{AsyncRead, AsyncWrite};
use std::io;
use std::path::Path;
use std::time::SystemTime;

macro_rules! forward_required {
    ($name:ident { $($early:tt)* } { $($late:tt)* }) => {
        impl Bucket for $name {
            $($early)*

            fn delete(
                &self,
                path: &Path,
                condition: &Condition,
            ) -> impl Future<Output = Result<()>> + Send {
                self.0.delete(path, condition)
            }

            fn get(
                &self,
                path: &Path,
            ) -> impl Future<Output = Result<(Vec<u8>, RevisionToken)>> + Send {
                self.0.get(path)
            }

            fn guarantees(&self) -> Guarantees {
                self.0.guarantees()
            }

            fn head(&self, path: &Path) -> impl Future<Output = Result<FileMetadata>> + Send {
                self.0.head(path)
            }

            fn list(
                &self,
                folder: &Path,
                depth: Depth,
            ) -> impl Future<Output = Result<Vec<WalkEntry>>> + Send {
                self.0.list(folder, depth)
            }

            fn put(
                &self,
                path: &Path,
                bytes: &[u8],
                condition: &Condition,
            ) -> impl Future<Output = Result<RevisionToken>> + Send {
                self.0.put(path, bytes, condition)
            }

            $($late)*
        }
    };
}

/// What a scenario saw, line by line, or why it could not go on.
type Observed = result::Result<Vec<String>, Box<dyn Error>>;

/// A bucket that implements only the required methods, so every other method is the trait's default.
struct RequiredOnly(MemoryBucket);

forward_required!(RequiredOnly {} {});

/// The required methods and the two folder methods, which have no useful default.
struct WithFolders(MemoryBucket);

forward_required!(WithFolders {
    fn create_folder(&self, path: &Path) -> impl Future<Output = Result<()>> + Send {
        self.0.create_folder(path)
    }
} {
    fn remove_folder(&self, path: &Path) -> impl Future<Output = Result<()>> + Send {
        self.0.remove_folder(path)
    }
});

/// A bucket whose file changes just before a delete under a token, so the token is stale.
struct ChangesBeforeDelete(MemoryBucket);

impl Bucket for ChangesBeforeDelete {
    async fn delete(&self, path: &Path, condition: &Condition) -> Result<()> {
        if matches!(*condition, Condition::Matches(_)) {
            self.0.put(path, b"changed", &Condition::None).await?;
        }
        self.0.delete(path, condition).await
    }

    fn get(&self, path: &Path) -> impl Future<Output = Result<(Vec<u8>, RevisionToken)>> + Send {
        self.0.get(path)
    }

    fn guarantees(&self) -> Guarantees {
        self.0.guarantees()
    }

    fn head(&self, path: &Path) -> impl Future<Output = Result<FileMetadata>> + Send {
        self.0.head(path)
    }

    fn list(
        &self,
        folder: &Path,
        depth: Depth,
    ) -> impl Future<Output = Result<Vec<WalkEntry>>> + Send {
        self.0.list(folder, depth)
    }

    fn put(
        &self,
        path: &Path,
        bytes: &[u8],
        condition: &Condition,
    ) -> impl Future<Output = Result<RevisionToken>> + Send {
        self.0.put(path, bytes, condition)
    }
}

/// Assert that a scenario saw exactly `expected`.
fn assert_observed(observed: Observed, expected: &[&str]) {
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
fn shape<T>(outcome: &Result<T>) -> String {
    match outcome.as_ref() {
        Ok(_) => "ok".to_owned(),
        Err(error) => format!("{:?}", error.kind()),
    }
}

/// The paths of `entries`.
fn paths(entries: &[WalkEntry]) -> Vec<String> {
    entries
        .iter()
        .map(|entry| entry.path.display().to_string())
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
async fn file_operations<B>(bucket: &B) -> Observed
where
    B: Bucket,
{
    let first = Path::new("/d/a.txt");
    let second = Path::new("/d/b.txt");
    let mut lines = Vec::new();
    bucket.create_folder(Path::new("/d")).await?;
    lines.push(shape(&bucket.put(first, b"one", &Condition::None).await));
    lines.push(shape(&bucket.append(first, b"-two").await));
    lines.push(shape(&bucket.append(second, b"new").await));
    lines.push(format!("get {:?}", bucket.get(first).await?.0));
    lines.push(format!("get {:?}", bucket.get(second).await?.0));
    lines.push(format!(
        "copy {:?}",
        bucket.copy(first, Path::new("/d/c.txt")).await?
    ));
    lines.push(shape(&bucket.copy(Path::new("/d/none"), second).await));
    lines.push(shape(&bucket.rename(first, Path::new("/d/e.txt")).await));
    lines.push(shape(&bucket.rename(first, Path::new("/d/f.txt")).await));
    lines.push(shape(&bucket.head(first).await));
    lines.push(format!(
        "renamed {:?}",
        bucket.get(Path::new("/d/e.txt")).await?.0
    ));
    Ok(lines)
}

/// Folders: a subtree renamed, walked with and without hidden entries, and removed.
async fn folder_operations<B>(bucket: &B) -> Observed
where
    B: Bucket,
{
    let mut lines = Vec::new();
    bucket.create_folder(Path::new("/d/sub/deeper")).await?;
    bucket
        .put(Path::new("/d/sub/deeper/x.txt"), b"x", &Condition::None)
        .await?;
    bucket
        .put(Path::new("/d/.hidden"), b"h", &Condition::None)
        .await?;
    lines.push(shape(
        &bucket.rename(Path::new("/d"), Path::new("/moved")).await,
    ));
    lines.push(format!(
        "old {}",
        shape(&bucket.head(Path::new("/d")).await)
    ));
    lines.push(format!(
        "moved {:?}",
        bucket.get(Path::new("/moved/sub/deeper/x.txt")).await?.0
    ));
    for hidden in [true, false] {
        let walked = bucket.walk(Path::new("/moved"), false, hidden).await?;
        lines.push(format!("walk hidden={hidden} {:?}", paths(&walked)));
    }
    lines.push(format!(
        "canonicalize {:?}",
        bucket.canonicalize(Path::new("/a/./b/../c")).await?
    ));
    lines.push(shape(&bucket.canonicalize(Path::new("relative")).await));
    lines.push(shape(&bucket.remove_folder(Path::new("/moved")).await));
    lines.push(shape(&bucket.head(Path::new("/moved")).await));
    Ok(lines)
}

/// Streams: a reader, a writer closed, a second close, and an appender.
async fn stream_operations<B>(bucket: &B) -> Observed
where
    B: Bucket,
{
    let source = Path::new("/s.txt");
    let target = Path::new("/streamed.txt");
    let mut lines = Vec::new();
    bucket.put(source, b"abcdefghij", &Condition::None).await?;
    let mut reader = bucket.read_stream(source).await?;
    lines.push(format!("read {:?}", read_all(&mut *reader).await?));
    drop(reader);
    let mut writer = bucket.write_stream(target, Condition::Absent).await?;
    write_all(&mut *writer, b"stream").await?;
    close(&mut *writer).await?;
    drop(writer);
    lines.push(format!("streamed {:?}", bucket.get(target).await?.0));
    let mut again = bucket.write_stream(target, Condition::Absent).await?;
    write_all(&mut *again, b"late").await?;
    let refused = close(&mut *again).await.map_err(StorageError::from);
    lines.push(format!(
        "second {:?}",
        refused.err().map(|error| error.kind())
    ));
    drop(again);
    let mut tail = bucket.append_stream(target).await?;
    write_all(&mut *tail, b"+tail").await?;
    close(&mut *tail).await?;
    drop(tail);
    lines.push(format!("appended {:?}", bucket.get(target).await?.0));
    Ok(lines)
}

/// Files, folders and streams, one after the other on the same bucket.
async fn every_operation<B>(bucket: &B) -> Observed
where
    B: Bucket,
{
    let mut lines = file_operations(bucket).await?;
    lines.extend(folder_operations(bucket).await?);
    lines.extend(stream_operations(bucket).await?);
    Ok(lines)
}

/// Two callers race for a lock; the second is refused and the first bytes stay.
async fn lock_taken_twice() -> Observed {
    let bucket = MemoryBucket::default();
    bucket.create_folder(Path::new("/jobs")).await?;
    let lock = Path::new("/jobs/42.lock");
    let mut lines = vec![shape(
        &bucket.put(lock, b"worker-a", &Condition::Absent).await,
    )];
    lines.push(shape(
        &bucket.put(lock, b"worker-b", &Condition::Absent).await,
    ));
    lines.push(format!("{:?}", bucket.get(lock).await?.0));
    Ok(lines)
}

/// A release with a token from before an append is refused; the current token releases.
async fn stale_release() -> Observed {
    let bucket = MemoryBucket::default();
    bucket.create_folder(Path::new("/jobs")).await?;
    let lock = Path::new("/jobs/42.lock");
    let ours = bucket.put(lock, b"worker-a", &Condition::Absent).await?;
    bucket.append(lock, b" renewed").await?;
    let mut lines = vec![shape(&bucket.delete(lock, &Condition::Matches(ours)).await)];
    let (bytes, token) = bucket.get(lock).await?;
    lines.push(format!("{bytes:?}"));
    lines.push(shape(
        &bucket.delete(lock, &Condition::Matches(token)).await,
    ));
    lines.push(shape(&bucket.get(lock).await));
    Ok(lines)
}

/// Each condition on a path with nothing at it.
async fn conditions_on_a_missing_file() -> Observed {
    let bucket = MemoryBucket::default();
    let path = Path::new("/none.md");
    let stale = RevisionToken::new("1");
    Ok(vec![
        shape(&bucket.delete(path, &Condition::None).await),
        shape(&bucket.delete(path, &Condition::Absent).await),
        shape(
            &bucket
                .delete(path, &Condition::Matches(stale.clone()))
                .await,
        ),
        shape(&bucket.put(path, b"x", &Condition::Matches(stale)).await),
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
async fn parent_rules() -> Observed {
    let bucket = MemoryBucket::default();
    bucket.create_folder(Path::new("/folder")).await?;
    bucket
        .put(Path::new("/file"), b"x", &Condition::None)
        .await?;
    let mut lines = Vec::new();
    for target in ["/missing/a", "/file/a", "/folder", "/"] {
        lines.push(shape(
            &bucket.put(Path::new(target), b"y", &Condition::None).await,
        ));
    }
    lines.push(shape(&bucket.create_folder(Path::new("/file/sub")).await));
    lines.push(shape(&bucket.remove_folder(Path::new("/file")).await));
    lines.push(shape(
        &bucket.copy(Path::new("/folder"), Path::new("/copy")).await,
    ));
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
async fn folder_rename() -> Observed {
    let bucket = MemoryBucket::default();
    bucket.create_folder(Path::new("/a/b")).await?;
    bucket
        .put(Path::new("/a/b/x"), b"1", &Condition::None)
        .await?;
    bucket
        .put(Path::new("/a/y"), b"2", &Condition::None)
        .await?;
    bucket.rename(Path::new("/a"), Path::new("/c")).await?;
    Ok(vec![
        format!("{:?}", bucket.get(Path::new("/c/b/x")).await?.0),
        format!("{:?}", bucket.get(Path::new("/c/y")).await?.0),
        shape(&bucket.head(Path::new("/a")).await),
        format!("{}", bucket.head(Path::new("/c/b")).await?.is_dir),
    ])
}

/// A listing at each depth, of a file, and of a folder that is not there.
async fn listings() -> Observed {
    let bucket = MemoryBucket::default();
    bucket.create_folder(Path::new("/a/b")).await?;
    bucket
        .put(Path::new("/a/b/x"), b"1", &Condition::None)
        .await?;
    bucket
        .put(Path::new("/a/y"), b"2", &Condition::None)
        .await?;
    Ok(vec![
        format!(
            "{:?}",
            paths(&bucket.list(Path::new("/a"), Depth::Children).await?)
        ),
        format!(
            "{:?}",
            paths(&bucket.list(Path::new("/a"), Depth::All).await?)
        ),
        shape(&bucket.list(Path::new("/a/y"), Depth::All).await),
        format!(
            "{:?}",
            paths(&bucket.list(Path::new("/none"), Depth::All).await?)
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

/// A writer dropped without `close`, on the memory bucket and on one with the default writer.
async fn dropped_writers() -> Observed {
    let path = Path::new("/a.md");
    let native = MemoryBucket::default();
    let mut writer = native.write_stream(path, Condition::Absent).await?;
    write_all(&mut *writer, b"kept").await?;
    drop(writer);
    let defaults = RequiredOnly(MemoryBucket::default());
    let mut lost = defaults.write_stream(path, Condition::Absent).await?;
    write_all(&mut *lost, b"lost").await?;
    drop(lost);
    Ok(vec![
        format!("native {:?}", native.get(path).await?.0),
        format!("default {}", shape(&defaults.get(path).await)),
    ])
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
        lock_taken_twice().await,
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
        stale_release().await,
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
        conditions_on_a_missing_file().await,
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
        parent_rules().await,
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
    assert_observed(folder_rename().await, &["[49]", "[50]", "NotFound", "true"]);
}

#[tokio::test(flavor = "current_thread")]
async fn a_listing_reaches_the_depth_it_is_given() {
    assert_observed(
        listings().await,
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
    assert_observed(
        dropped_writers().await,
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
    let native = every_operation(&MemoryBucket::default())
        .await
        .map_err(|error| error.to_string());
    let defaults = every_operation(&WithFolders(MemoryBucket::default()))
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
    assert!(!promised.contains(Guarantee::WalkHonorsGitignore));
}
