//! Tests for `DiskBucket`: task 02's scenarios run over a temporary folder and over the memory
//! bucket, and the behaviour only the disk has.

use super::{DiskBucket, delete_opened, token_of};
use crate::WalkEntry;
use crate::bucket::core_tests::{
    Observed, assert_observed, conditions_on_a_missing_file, dropped_writers, every_operation,
    file_operations, folder_operations, folder_rename, listings, lock_taken_twice, parent_rules,
    shape, stale_release, stream_operations,
};
use crate::bucket::{Bucket as _, Condition, Depth, Guarantee, MemoryBucket, RevisionToken};
use core::error::Error;
use core::future::{Future, poll_fn};
use core::pin::{Pin, pin};
use core::task::{Context, Poll, Waker};
use futures_io::{AsyncRead, AsyncWrite};
use std::fs::{self, File};
use std::io::{self, Write as _};
use std::path::Path;
use std::thread;
use tempfile::TempDir;

/// The result of `future` after one poll with no runtime; a future that is not ready is an error.
fn ready<F>(future: F) -> Result<F::Output, Box<dyn Error>>
where
    F: Future,
{
    let mut pinned = pin!(future);
    let mut context = Context::from_waker(Waker::noop());
    match pinned.as_mut().poll(&mut context) {
        Poll::Ready(value) => Ok(value),
        Poll::Pending => Err("the future was not ready on its first poll".into()),
    }
}

/// A scenario run on a disk folder.
async fn on_disk<S>(scenario: S) -> Observed
where
    S: for<'bucket, 'folder> AsyncFn(&'bucket DiskBucket, &'folder Path) -> Observed,
{
    let folder = TempDir::new()?;
    let root = fs::canonicalize(folder.path())?;
    scenario(&DiskBucket::new(), &root).await
}

/// A scenario run in memory under a folder with the same relative layout.
async fn in_memory<S>(scenario: S) -> Observed
where
    S: for<'bucket, 'folder> AsyncFn(&'bucket MemoryBucket, &'folder Path) -> Observed,
{
    let bucket = MemoryBucket::default();
    let root = Path::new("/root");
    bucket.create_folder(root).await?;
    scenario(&bucket, root).await
}

/// Assert that the disk saw what the memory bucket saw, except that the line at each index in
/// `differing` is the disk's own line.
fn assert_same_as_memory(disk: Observed, memory: Observed, differing: &[(usize, &str)]) {
    let expected = memory.map(|mut lines| {
        for &(index, line) in differing {
            if let Some(slot) = lines.get_mut(index) {
                line.clone_into(slot);
            }
        }
        lines
    });
    assert_eq!(
        disk.map_err(|error| error.to_string()),
        expected.map_err(|error| error.to_string())
    );
}

/// Every byte `reader` gives, one poll per read.
fn read_all(reader: &mut (dyn AsyncRead + Send + Unpin + '_)) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut bytes = Vec::new();
    let mut chunk = [0_u8; 4];
    loop {
        let count = ready(poll_fn(|cx| {
            Pin::new(&mut *reader).poll_read(cx, &mut chunk)
        }))??;
        if count == 0 {
            return Ok(bytes);
        }
        bytes.extend_from_slice(chunk.get(..count).unwrap_or_default());
    }
}

/// Give `bytes` to `writer`, one poll per write, and close it.
fn write_and_close(
    writer: &mut (dyn AsyncWrite + Send + Unpin + '_),
    bytes: &[u8],
) -> Result<(), Box<dyn Error>> {
    let mut rest = bytes;
    while !rest.is_empty() {
        let count = ready(poll_fn(|cx| Pin::new(&mut *writer).poll_write(cx, rest)))??;
        rest = rest.get(count..).unwrap_or_default();
    }
    ready(poll_fn(|cx| Pin::new(&mut *writer).poll_close(cx)))??;
    Ok(())
}

/// The paths of `entries` below `root`, sorted.
fn relative(entries: &[WalkEntry], root: &Path) -> Vec<String> {
    let mut shown: Vec<String> = entries
        .iter()
        .map(|entry| {
            entry
                .path
                .strip_prefix(root)
                .unwrap_or(&entry.path)
                .display()
                .to_string()
        })
        .collect();
    shown.sort();
    shown
}

#[tokio::test(flavor = "current_thread")]
async fn file_operations_give_the_memory_buckets_results() {
    assert_same_as_memory(
        on_disk(file_operations).await,
        in_memory(file_operations).await,
        &[],
    );
}

// Differs at line 6: memory refuses a relative path as InvalidInput by its lexical rules; the
// disk hands every path to std::fs, which looks it up from the working folder.
#[tokio::test(flavor = "current_thread")]
async fn folder_operations_give_the_memory_buckets_results() {
    assert_same_as_memory(
        on_disk(folder_operations).await,
        in_memory(folder_operations).await,
        &[(6, "NotFound")],
    );
}

#[tokio::test(flavor = "current_thread")]
async fn stream_operations_give_the_memory_buckets_results() {
    assert_same_as_memory(
        on_disk(stream_operations).await,
        in_memory(stream_operations).await,
        &[],
    );
}

// Differs at line 17, the relative path of folder_operations, as above.
#[tokio::test(flavor = "current_thread")]
async fn every_operation_gives_the_memory_buckets_results() {
    assert_same_as_memory(
        on_disk(every_operation).await,
        in_memory(every_operation).await,
        &[(17, "NotFound")],
    );
}

#[tokio::test(flavor = "current_thread")]
async fn a_lock_taken_twice_gives_the_memory_buckets_results() {
    assert_same_as_memory(
        on_disk(lock_taken_twice).await,
        in_memory(lock_taken_twice).await,
        &[],
    );
}

#[tokio::test(flavor = "current_thread")]
async fn a_stale_release_gives_the_memory_buckets_results() {
    assert_same_as_memory(
        on_disk(stale_release).await,
        in_memory(stale_release).await,
        &[],
    );
}

#[tokio::test(flavor = "current_thread")]
async fn conditions_on_a_missing_file_give_the_memory_buckets_results() {
    assert_same_as_memory(
        on_disk(conditions_on_a_missing_file).await,
        in_memory(conditions_on_a_missing_file).await,
        &[],
    );
}

#[tokio::test(flavor = "current_thread")]
async fn the_parent_rules_give_the_memory_buckets_results() {
    assert_same_as_memory(
        on_disk(parent_rules).await,
        in_memory(parent_rules).await,
        &[(7, "NotFound"), (8, "NotFound")],
    );
}

#[tokio::test(flavor = "current_thread")]
async fn renaming_a_folder_gives_the_memory_buckets_results() {
    assert_same_as_memory(
        on_disk(folder_rename).await,
        in_memory(folder_rename).await,
        &[],
    );
}

#[tokio::test(flavor = "current_thread")]
async fn a_listing_gives_the_memory_buckets_results() {
    assert_same_as_memory(on_disk(listings).await, in_memory(listings).await, &[]);
}

#[tokio::test(flavor = "current_thread")]
async fn a_dropped_writer_gives_the_memory_buckets_results() {
    assert_same_as_memory(
        on_disk(dropped_writers).await,
        in_memory(dropped_writers).await,
        &[],
    );
}

#[tokio::test(flavor = "current_thread")]
async fn a_missing_file_and_folder_are_not_errors_to_delete_and_list() {
    async fn scenario(bucket: &DiskBucket, root: &Path) -> Observed {
        let none = root.join("none");
        Ok(vec![
            shape(&bucket.delete(&none, &Condition::None).await),
            shape(&bucket.list(&none, Depth::All).await),
            shape(&bucket.list(&none, Depth::Children).await),
            shape(&bucket.remove_folder(&none).await),
        ])
    }
    assert_observed(on_disk(scenario).await, &["ok", "ok", "ok", "NotFound"]);
}

#[tokio::test(flavor = "current_thread")]
async fn a_folder_is_not_deleted_under_any_condition() {
    async fn scenario(bucket: &DiskBucket, root: &Path) -> Observed {
        let folder = root.join("folder");
        bucket.create_folder(&folder).await?;
        let token = RevisionToken::new("1");
        Ok(vec![
            shape(&bucket.delete(&folder, &Condition::None).await),
            shape(&bucket.delete(&folder, &Condition::Absent).await),
            shape(&bucket.delete(&folder, &Condition::Matches(token)).await),
            format!("{}", folder.is_dir()),
        ])
    }
    assert_observed(
        on_disk(scenario).await,
        &["IsADirectory", "IsADirectory", "IsADirectory", "true"],
    );
}

#[tokio::test(flavor = "current_thread")]
async fn a_folder_is_not_written_under_any_condition() {
    async fn scenario(bucket: &DiskBucket, root: &Path) -> Observed {
        let folder = root.join("folder");
        bucket.create_folder(&folder).await?;
        let token = RevisionToken::new("1");
        Ok(vec![
            shape(&bucket.put(&folder, b"x", &Condition::None).await),
            shape(&bucket.put(&folder, b"x", &Condition::Absent).await),
            shape(
                &bucket
                    .put(&folder, b"x", &Condition::Matches(token.clone()))
                    .await,
            ),
            shape(&bucket.write_stream(&folder, Condition::None).await),
            shape(&bucket.write_stream(&folder, Condition::Absent).await),
            shape(
                &bucket
                    .write_stream(&folder, Condition::Matches(token))
                    .await,
            ),
            format!("{}", folder.is_dir()),
        ])
    }
    assert_observed(
        on_disk(scenario).await,
        &[
            "IsADirectory",
            "IsADirectory",
            "IsADirectory",
            "IsADirectory",
            "IsADirectory",
            "IsADirectory",
            "true",
        ],
    );
}

#[tokio::test(flavor = "current_thread")]
async fn a_folder_renamed_onto_a_non_empty_folder_is_already_exists() {
    async fn scenario(bucket: &DiskBucket, root: &Path) -> Observed {
        for name in ["d2", "e2"] {
            bucket.create_folder(&root.join(name)).await?;
        }
        bucket
            .put(&root.join("d2").join("a.md"), b"a", &Condition::None)
            .await?;
        bucket
            .put(&root.join("e2").join("x.md"), b"x", &Condition::None)
            .await?;
        let refused = bucket
            .rename(&root.join("d2"), &root.join("e2"))
            .await
            .err()
            .ok_or("the rename was not refused")?;
        let source = refused
            .source()
            .and_then(|inner| inner.downcast_ref::<io::Error>())
            .ok_or("the refusal has no io error as its source")?;
        Ok(vec![
            format!("{:?}", refused.kind()),
            format!("{:?}", source.kind()),
            format!("{}", root.join("d2").join("a.md").is_file()),
            format!("{}", root.join("e2").join("x.md").is_file()),
        ])
    }
    assert_observed(
        on_disk(scenario).await,
        &["AlreadyExists", "DirectoryNotEmpty", "true", "true"],
    );
}

#[tokio::test(flavor = "current_thread")]
async fn a_lock_is_released_only_with_the_current_content_hash() {
    async fn scenario(bucket: &DiskBucket, root: &Path) -> Observed {
        let lock = root.join("job.lock");
        let ours = bucket.put(&lock, b"worker-a", &Condition::Absent).await?;
        let mut lines = vec![ours.as_str().to_owned()];
        lines.push(shape(
            &bucket.put(&lock, b"worker-b", &Condition::Absent).await,
        ));
        bucket.append(&lock, b" renewed").await?;
        lines.push(shape(
            &bucket.delete(&lock, &Condition::Matches(ours)).await,
        ));
        lines.push(format!("{}", lock.exists()));
        let (bytes, token) = bucket.get(&lock).await?;
        lines.push(format!("{bytes:?}"));
        lines.push(token.as_str().to_owned());
        bucket.delete(&lock, &Condition::Matches(token)).await?;
        lines.push(format!("{}", lock.exists()));
        Ok(lines)
    }
    assert_observed(
        on_disk(scenario).await,
        &[
            "6a65e237ae44c42895b5049031466616173469f513b00240e9db90df66906a49",
            "AlreadyExists",
            "PreconditionFailed",
            "true",
            "[119, 111, 114, 107, 101, 114, 45, 97, 32, 114, 101, 110, 101, 119, 101, 100]",
            "fe84ff74f874e1bbdb82aea30ff8d02f16ac2bc9f1ebb36367debb1847d63076",
            "false",
        ],
    );
}

#[tokio::test(flavor = "current_thread")]
async fn equal_bytes_have_equal_tokens() {
    async fn scenario(bucket: &DiskBucket, root: &Path) -> Observed {
        let path = root.join("a.md");
        let first = bucket.put(&path, b"x", &Condition::None).await?;
        bucket.put(&path, b"y", &Condition::None).await?;
        let back = bucket.put(&path, b"x", &Condition::None).await?;
        Ok(vec![
            format!("{}", first == back),
            format!("{}", first == token_of(b"x")),
            format!("{:?}", bucket.head(&path).await?.revision),
        ])
    }
    assert_observed(on_disk(scenario).await, &["true", "true", "None"]);
}

#[test]
fn a_put_waits_for_the_lock_and_then_compares() {
    fn scenario(root: &Path) -> Observed {
        let bucket = DiskBucket::new();
        let path = root.join("job.lock");
        let ours = ready(bucket.put(&path, b"worker-a", &Condition::Absent))??;
        let mut held = fs::OpenOptions::new().read(true).write(true).open(&path)?;
        held.lock()?;
        let kind = thread::scope(|scope| -> Result<String, Box<dyn Error>> {
            let attempt = scope.spawn(|| {
                ready(bucket.put(&path, b"worker-c", &Condition::Matches(ours.clone())))
                    .map(|outcome| shape(&outcome))
                    .map_err(|error| error.to_string())
            });
            held.set_len(0)?;
            held.write_all(b"worker-b")?;
            held.unlock()?;
            Ok(attempt.join().map_err(|_panic| "the put panicked")??)
        })?;
        Ok(vec![kind, format!("{:?}", fs::read(&path)?)])
    }
    let folder = TempDir::new();
    assert_observed(
        folder
            .map_err(Into::into)
            .and_then(|dir| scenario(dir.path())),
        &[
            "PreconditionFailed",
            "[119, 111, 114, 107, 101, 114, 45, 98]",
        ],
    );
}

#[test]
fn a_delete_leaves_a_different_file_made_after_the_open() {
    fn scenario(root: &Path) -> Observed {
        let path = root.join("job.lock");
        fs::write(&path, b"worker-a")?;
        let opened = File::open(&path)?;
        fs::remove_file(&path)?;
        fs::write(&path, b"worker-b")?;
        let outcome = delete_opened(opened, &path, &token_of(b"worker-a"));
        Ok(vec![
            format!("{:?}", outcome.map_err(|error| error.kind())),
            format!("{:?}", fs::read(&path)?),
        ])
    }
    let folder = TempDir::new();
    assert_observed(
        folder
            .map_err(Into::into)
            .and_then(|dir| scenario(dir.path())),
        &[
            "Err(PreconditionFailed)",
            "[119, 111, 114, 107, 101, 114, 45, 98]",
        ],
    );
}

#[test]
fn a_delete_leaves_a_file_made_after_the_open_with_the_same_bytes() {
    fn scenario(root: &Path) -> Observed {
        let path = root.join("job.lock");
        fs::write(&path, b"worker-a")?;
        let opened = File::open(&path)?;
        fs::remove_file(&path)?;
        fs::write(&path, b"worker-a")?;
        let outcome = delete_opened(opened, &path, &token_of(b"worker-a"));
        Ok(vec![
            format!("{:?}", outcome.map_err(|error| error.kind())),
            format!("{}", path.exists()),
        ])
    }
    let folder = TempDir::new();
    assert_observed(
        folder
            .map_err(Into::into)
            .and_then(|dir| scenario(dir.path())),
        &["Err(PreconditionFailed)", "true"],
    );
}

#[test]
fn a_delete_leaves_the_path_empty_when_the_file_was_removed_after_the_open() {
    fn scenario(root: &Path) -> Observed {
        let path = root.join("job.lock");
        fs::write(&path, b"worker-a")?;
        let opened = File::open(&path)?;
        fs::remove_file(&path)?;
        let outcome = delete_opened(opened, &path, &token_of(b"worker-a"));
        Ok(vec![
            format!("{:?}", outcome.map_err(|error| error.kind())),
            format!("{}", path.exists()),
        ])
    }
    let folder = TempDir::new();
    assert_observed(
        folder
            .map_err(Into::into)
            .and_then(|dir| scenario(dir.path())),
        &["Err(PreconditionFailed)", "false"],
    );
}

#[test]
fn a_write_stream_under_a_token_checks_it_first() {
    fn scenario(root: &Path) -> Observed {
        let bucket = DiskBucket::new();
        let path = root.join("a.txt");
        let token = ready(bucket.put(&path, b"old", &Condition::None))??;
        let refused =
            ready(bucket.write_stream(&path, Condition::Matches(RevisionToken::new("0"))))?;
        let mut lines = vec![shape(&refused)];
        let mut writer = ready(bucket.write_stream(&path, Condition::Matches(token)))??;
        write_and_close(&mut *writer, b"new")?;
        drop(writer);
        lines.push(format!("{:?}", fs::read(&path)?));
        Ok(lines)
    }
    let folder = TempDir::new();
    assert_observed(
        folder
            .map_err(Into::into)
            .and_then(|dir| scenario(dir.path())),
        &["PreconditionFailed", "[110, 101, 119]"],
    );
}

#[test]
fn walk_leaves_out_what_gitignore_names_and_list_keeps_it() {
    fn scenario(root: &Path) -> Observed {
        let bucket = DiskBucket::new();
        fs::create_dir_all(root.join(".git"))?;
        fs::create_dir_all(root.join("target"))?;
        fs::write(root.join(".gitignore"), "target/\n*.log\n")?;
        fs::write(root.join("target/out.bin"), b"x")?;
        fs::write(root.join("a.log"), b"x")?;
        fs::write(root.join("a.rs"), b"x")?;
        let walked = ready(bucket.walk(root, false, true))??;
        let listed = ready(bucket.list(root, Depth::All))??;
        Ok(vec![
            format!("{:?}", relative(&walked, root)),
            format!("{:?}", relative(&listed, root)),
        ])
    }
    let folder = TempDir::new();
    assert_observed(
        folder
            .map_err(Into::into)
            .and_then(|dir| scenario(dir.path())),
        &[
            "[\".git\", \".gitignore\", \"a.rs\"]",
            "[\".git\", \".gitignore\", \"a.log\", \"a.rs\", \"target\", \"target/out.bin\"]",
        ],
    );
}

#[test]
fn every_method_is_ready_on_its_first_poll_with_no_runtime() {
    fn scenario(root: &Path) -> Observed {
        let bucket = DiskBucket::new();
        let first = root.join("a");
        let second = root.join("b");
        ready(bucket.create_folder(&root.join("d")))??;
        ready(bucket.put(&first, b"x", &Condition::Absent))??;
        ready(bucket.append(&first, b"y"))??;
        ready(bucket.copy(&first, &second))??;
        ready(bucket.rename(&second, &root.join("c")))??;
        ready(bucket.head(&first))??;
        ready(bucket.get(&first))??;
        ready(bucket.list(root, Depth::Children))??;
        ready(bucket.walk(root, false, true))??;
        ready(bucket.canonicalize(&first))??;
        let mut reader = ready(bucket.read_stream(&first))??;
        let read = read_all(&mut *reader)?;
        let scratch = root.join("w");
        let mut writer = ready(bucket.write_stream(&scratch, Condition::None))??;
        write_and_close(&mut *writer, b"w")?;
        let mut tail = ready(bucket.append_stream(&scratch))??;
        write_and_close(&mut *tail, b"+")?;
        let token = ready(bucket.get(&scratch))??.1;
        ready(bucket.delete(&scratch, &Condition::Matches(token)))??;
        ready(bucket.remove_folder(&root.join("d")))??;
        ready(bucket.delete(&first, &Condition::None))??;
        Ok(vec![format!("{read:?}")])
    }
    let folder = TempDir::new();
    assert_observed(
        folder
            .map_err(Into::into)
            .and_then(|dir| scenario(dir.path())),
        &["[120, 121]"],
    );
}

#[test]
fn a_write_stream_dropped_without_close_keeps_its_bytes() {
    fn scenario(root: &Path) -> Observed {
        let bucket = DiskBucket::new();
        let path = root.join("a.txt");
        let mut writer = ready(bucket.write_stream(&path, Condition::None))??;
        ready(poll_fn(|cx| Pin::new(&mut *writer).poll_write(cx, b"kept")))??;
        drop(writer);
        Ok(vec![format!("{:?}", fs::read(&path)?)])
    }
    let folder = TempDir::new();
    assert_observed(
        folder
            .map_err(Into::into)
            .and_then(|dir| scenario(dir.path())),
        &["[107, 101, 112, 116]"],
    );
}

#[cfg(unix)]
#[test]
fn a_refused_open_is_permission_denied() {
    use std::os::unix::fs::PermissionsExt as _;

    fn scenario(root: &Path) -> Observed {
        let bucket = DiskBucket::new();
        let path = root.join("secret");
        fs::write(&path, b"x")?;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o000))?;
        if File::open(&path).is_ok() {
            return Ok(vec!["PermissionDenied".to_owned(); 2]);
        }
        Ok(vec![
            shape(&ready(bucket.get(&path))?),
            shape(&ready(bucket.put(&path, b"y", &Condition::None))?),
        ])
    }
    let folder = TempDir::new();
    assert_observed(
        folder
            .map_err(Into::into)
            .and_then(|dir| scenario(dir.path())),
        &["PermissionDenied", "PermissionDenied"],
    );
}

#[cfg(unix)]
#[test]
fn head_and_canonicalize_follow_a_symbolic_link() {
    use std::os::unix::fs::symlink;

    fn scenario(root: &Path) -> Observed {
        let bucket = DiskBucket::new();
        let target = root.join("target.txt");
        let link = root.join("link.txt");
        fs::write(&target, b"four")?;
        symlink(&target, &link)?;
        let head = ready(bucket.head(&link))??;
        Ok(vec![
            format!("{} {} {}", head.is_file, head.is_dir, head.len),
            format!("{}", head.modified.is_some()),
            format!(
                "{}",
                ready(bucket.canonicalize(&link))?? == fs::canonicalize(&target)?
            ),
        ])
    }
    let folder = TempDir::new();
    assert_observed(
        folder
            .map_err(Into::into)
            .and_then(|dir| scenario(dir.path())),
        &["true false 4", "true", "true"],
    );
}

#[cfg(unix)]
#[test]
fn a_name_that_is_not_utf8_is_written_and_read() {
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt as _;

    fn scenario(root: &Path) -> Observed {
        let bucket = DiskBucket::new();
        let path = root.join(OsStr::from_bytes(b"\xff.md"));
        ready(bucket.put(&path, b"x", &Condition::None))??;
        Ok(vec![format!("{:?}", ready(bucket.get(&path))??.0)])
    }
    let folder = TempDir::new();
    assert_observed(
        folder
            .map_err(Into::into)
            .and_then(|dir| scenario(dir.path())),
        &["[120]"],
    );
}

#[test]
fn the_disk_promises_what_it_does() {
    let promised = DiskBucket::new().guarantees();
    let kept = [
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
        Guarantee::RenameKeepsModificationTime,
        Guarantee::WalkHonorsGitignore,
    ];
    assert_eq!(
        kept.map(|guarantee| promised.contains(guarantee)),
        [true; 14]
    );
    assert!(!promised.contains(Guarantee::AppendIsAtomic));
    assert!(!promised.contains(Guarantee::ReadersNeverSeePartialWrites));
}
