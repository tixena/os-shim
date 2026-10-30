//! Behavioral tests for `RealSystem` driven against real temp dirs.
//!
//! Every `System` method on `RealSystem` is exercised here. Tests are isolated
//! via `create_temp_dir` and unique env keys so they do not depend on the repo
//! working directory or leak state between runs.

use super::RealSystem;
use crate::System as _;
use crate::bucket::{Bucket as _, Condition, Depth, MemoryBucket};
use crate::error::ErrorKind;
use crate::files::run_without_runtime;
use crate::streams::{close, read_to_end, write_all};
#[cfg(slow_tests)]
use alloc::sync::Arc;
#[cfg(slow_tests)]
use alloc::task::Wake;
#[cfg(slow_tests)]
use core::future::Future;
#[cfg(slow_tests)]
use core::pin::pin;
#[cfg(slow_tests)]
use core::task::{Context, Poll, Waker};
use core::time::Duration;
use std::env::VarError;
use std::fs;
use std::path::Path;
#[cfg(slow_tests)]
use std::thread::{self, Thread};
#[cfg(slow_tests)]
use std::time::Instant as MachineInstant;
use std::time::SystemTime;

#[cfg(slow_tests)]
struct ParkedThread(Thread);

#[cfg(slow_tests)]
impl Wake for ParkedThread {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}

#[tokio::test(flavor = "current_thread")]
async fn write_then_read_to_string_roundtrips() {
    let system = RealSystem::new();
    let tmp = system.create_temp_dir().await.unwrap();
    let file = tmp.path().join("hello.txt");

    system
        .write(&file, b"hello world", Condition::None)
        .await
        .unwrap();
    assert_eq!(system.read_to_string(&file).await.unwrap(), "hello world");
}

#[tokio::test(flavor = "current_thread")]
async fn write_truncates_existing_file() {
    let system = RealSystem::new();
    let tmp = system.create_temp_dir().await.unwrap();
    let file = tmp.path().join("file.txt");

    system
        .write(&file, b"first contents", Condition::None)
        .await
        .unwrap();
    system
        .write(&file, b"second", Condition::None)
        .await
        .unwrap();
    assert_eq!(system.read_to_string(&file).await.unwrap(), "second");
}

#[tokio::test(flavor = "current_thread")]
async fn copy_duplicates_file_and_returns_len() {
    let system = RealSystem::new();
    let tmp = system.create_temp_dir().await.unwrap();
    let src = tmp.path().join("src.txt");
    let dest = tmp.path().join("dest.txt");

    system
        .write(&src, b"payload", Condition::None)
        .await
        .unwrap();
    let copied = system.copy(&src, &dest).await.unwrap();

    assert_eq!(copied, 7);
    assert_eq!(system.read_to_string(&dest).await.unwrap(), "payload");
    assert!(system.exists(&src).await.unwrap());
}

#[tokio::test(flavor = "current_thread")]
async fn create_returns_writable_stream() {
    let system = RealSystem::new();
    let tmp = system.create_temp_dir().await.unwrap();
    let file = tmp.path().join("created.txt");

    let mut writer = system.create(&file).await.unwrap();
    write_all(&mut *writer, b"streamed").await.unwrap();
    close(&mut *writer).await.unwrap();
    drop(writer);

    assert_eq!(system.read_to_string(&file).await.unwrap(), "streamed");
}

#[tokio::test(flavor = "current_thread")]
async fn create_truncates_existing_file() {
    let system = RealSystem::new();
    let tmp = system.create_temp_dir().await.unwrap();
    let file = tmp.path().join("file.txt");

    system
        .write(&file, b"long original contents", Condition::None)
        .await
        .unwrap();
    let mut writer = system.create(&file).await.unwrap();
    write_all(&mut *writer, b"short").await.unwrap();
    close(&mut *writer).await.unwrap();
    drop(writer);

    assert_eq!(system.read_to_string(&file).await.unwrap(), "short");
}

#[tokio::test(flavor = "current_thread")]
async fn create_dir_all_builds_nested_and_is_idempotent() {
    let system = RealSystem::new();
    let tmp = system.create_temp_dir().await.unwrap();
    let nested = tmp.path().join("a").join("b").join("c");

    system.create_dir_all(&nested).await.unwrap();
    assert!(system.is_dir(&nested).await.unwrap());

    system.create_dir_all(&nested).await.unwrap();
    assert!(system.is_dir(&nested).await.unwrap());
}

#[tokio::test(flavor = "current_thread")]
async fn create_temp_dir_exists_then_cleans_up_on_drop() {
    let system = RealSystem::new();
    let tmp = system.create_temp_dir().await.unwrap();
    let path = tmp.path().to_path_buf();

    assert!(system.is_dir(&path).await.unwrap());
    drop(tmp);
    assert!(!system.exists(&path).await.unwrap());
}

#[tokio::test(flavor = "current_thread")]
async fn current_dir_is_absolute_and_exists() {
    let system = RealSystem::new();
    let dir = system.current_dir().unwrap();

    assert!(dir.is_absolute());
    assert!(system.is_dir(&dir).await.unwrap());
}

#[tokio::test(flavor = "current_thread")]
async fn current_exe_points_at_existing_file() {
    let system = RealSystem::new();
    let exe = system.current_exe().unwrap();

    assert!(exe.is_absolute());
    assert!(system.is_file(&exe).await.unwrap());
}

#[test]
fn env_var_roundtrips_and_reports_missing() {
    let system = RealSystem::new();
    let key = "OS_SHIM_REAL_TEST_ENV_ROUNDTRIP";

    system.set_env_var(key, "configured");
    assert_eq!(system.env_var(key).unwrap(), "configured");

    assert_eq!(
        system.env_var("OS_SHIM_REAL_TEST_DEFINITELY_UNSET"),
        Err(VarError::NotPresent)
    );
}

#[tokio::test(flavor = "current_thread")]
async fn exists_is_dir_is_file_classify_paths() {
    let system = RealSystem::new();
    let tmp = system.create_temp_dir().await.unwrap();
    let file = tmp.path().join("file.txt");
    let dir = tmp.path().join("dir");
    let missing = tmp.path().join("missing");

    system.write(&file, b"x", Condition::None).await.unwrap();
    system.create_dir_all(&dir).await.unwrap();

    assert!(system.exists(&file).await.unwrap());
    assert!(system.is_file(&file).await.unwrap());
    assert!(!system.is_dir(&file).await.unwrap());

    assert!(system.exists(&dir).await.unwrap());
    assert!(system.is_dir(&dir).await.unwrap());
    assert!(!system.is_file(&dir).await.unwrap());

    assert!(!system.exists(&missing).await.unwrap());
    assert!(!system.is_file(&missing).await.unwrap());
    assert!(!system.is_dir(&missing).await.unwrap());
}

#[cfg(feature = "process")]
#[test]
fn is_pid_alive_true_for_self_false_for_reaped_child() {
    use std::process::{self, Command};

    let system = RealSystem::new();

    assert!(system.is_pid_alive(process::id()));

    let mut child = Command::new("sh").arg("-c").arg("exit 0").spawn().unwrap();
    let child_pid = child.id();
    child.wait().unwrap();

    assert!(!system.is_pid_alive(child_pid));
}

#[tokio::test(flavor = "current_thread")]
async fn metadata_reports_len_for_known_and_empty_files() {
    let system = RealSystem::new();
    let tmp = system.create_temp_dir().await.unwrap();
    let known = tmp.path().join("known.txt");
    let empty = tmp.path().join("empty.txt");

    system
        .write(&known, b"12345", Condition::None)
        .await
        .unwrap();
    system.write(&empty, b"", Condition::None).await.unwrap();

    let known_meta = system.metadata(&known).await.unwrap();
    assert!(known_meta.is_file);
    assert!(!known_meta.is_dir);
    assert_eq!(known_meta.len, 5);

    assert_eq!(system.metadata(&empty).await.unwrap().len, 0);
}

#[tokio::test(flavor = "current_thread")]
async fn metadata_missing_path_is_not_found() {
    let system = RealSystem::new();
    let tmp = system.create_temp_dir().await.unwrap();
    let missing = tmp.path().join("missing");

    let err = system.metadata(&missing).await.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::NotFound);
}

#[test]
fn now_is_after_the_unix_epoch() {
    let now = RealSystem::new().now();

    now.duration_since(SystemTime::UNIX_EPOCH).unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn open_reads_existing_contents() {
    let system = RealSystem::new();
    let tmp = system.create_temp_dir().await.unwrap();
    let file = tmp.path().join("read.txt");

    system
        .write(&file, b"readable", Condition::None)
        .await
        .unwrap();
    let mut reader = system.open(&file).await.unwrap();
    let contents = String::from_utf8(read_to_end(&mut *reader).await.unwrap()).unwrap();

    assert_eq!(contents, "readable");
}

#[tokio::test(flavor = "current_thread")]
async fn open_missing_file_is_not_found() {
    let system = RealSystem::new();
    let tmp = system.create_temp_dir().await.unwrap();
    let missing = tmp.path().join("missing.txt");

    let kind = system.open(&missing).await.err().map(|err| err.kind());
    assert_eq!(kind, Some(ErrorKind::NotFound));
}

#[tokio::test(flavor = "current_thread")]
async fn open_append_appends_and_creates() {
    let system = RealSystem::new();
    let tmp = system.create_temp_dir().await.unwrap();
    let file = tmp.path().join("log.txt");

    let mut first = system.open_append(&file).await.unwrap();
    write_all(&mut *first, b"one").await.unwrap();
    close(&mut *first).await.unwrap();
    drop(first);

    let mut second = system.open_append(&file).await.unwrap();
    write_all(&mut *second, b"-two").await.unwrap();
    close(&mut *second).await.unwrap();
    drop(second);

    assert_eq!(system.read_to_string(&file).await.unwrap(), "one-two");
}

#[tokio::test(flavor = "current_thread")]
async fn read_dir_lists_direct_children() {
    let system = RealSystem::new();
    let tmp = system.create_temp_dir().await.unwrap();
    let root = tmp.path();

    system
        .write(&root.join("a.txt"), b"a", Condition::None)
        .await
        .unwrap();
    system
        .write(&root.join("b.txt"), b"b", Condition::None)
        .await
        .unwrap();
    system.create_dir_all(&root.join("sub")).await.unwrap();

    let mut names: Vec<String> = system
        .read_dir(root)
        .await
        .unwrap()
        .iter()
        .map(|path| {
            path.strip_prefix(root)
                .unwrap()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    names.sort();

    assert_eq!(names, vec!["a.txt", "b.txt", "sub"]);
}

#[tokio::test(flavor = "current_thread")]
async fn remove_file_deletes_then_missing_is_not_found() {
    let system = RealSystem::new();
    let tmp = system.create_temp_dir().await.unwrap();
    let file = tmp.path().join("doomed.txt");

    system.write(&file, b"bye", Condition::None).await.unwrap();
    system.remove_file(&file).await.unwrap();
    assert!(!system.exists(&file).await.unwrap());

    let err = system.remove_file(&file).await.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::NotFound);
}

#[tokio::test(flavor = "current_thread")]
async fn remove_dir_all_removes_tree_then_missing_is_not_found() {
    let system = RealSystem::new();
    let tmp = system.create_temp_dir().await.unwrap();
    let dir = tmp.path().join("tree");

    system.create_dir_all(&dir.join("nested")).await.unwrap();
    system
        .write(&dir.join("nested").join("file.txt"), b"x", Condition::None)
        .await
        .unwrap();

    system.remove_dir_all(&dir).await.unwrap();
    assert!(!system.exists(&dir).await.unwrap());

    let err = system.remove_dir_all(&dir).await.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::NotFound);
}

#[tokio::test(flavor = "current_thread")]
async fn rename_moves_file() {
    let system = RealSystem::new();
    let tmp = system.create_temp_dir().await.unwrap();
    let src = tmp.path().join("src.txt");
    let dest = tmp.path().join("dest.txt");

    system
        .write(&src, b"content", Condition::None)
        .await
        .unwrap();
    system.rename(&src, &dest).await.unwrap();

    assert!(!system.exists(&src).await.unwrap());
    assert_eq!(system.read_to_string(&dest).await.unwrap(), "content");
}

#[tokio::test(flavor = "current_thread")]
async fn rename_over_existing_target_overwrites() {
    let system = RealSystem::new();
    let tmp = system.create_temp_dir().await.unwrap();
    let src = tmp.path().join("src.txt");
    let dest = tmp.path().join("dest.txt");

    system.write(&src, b"new", Condition::None).await.unwrap();
    system.write(&dest, b"old", Condition::None).await.unwrap();
    system.rename(&src, &dest).await.unwrap();

    assert_eq!(system.read_to_string(&dest).await.unwrap(), "new");
}

#[tokio::test(flavor = "current_thread")]
async fn rename_missing_source_is_not_found() {
    let system = RealSystem::new();
    let tmp = system.create_temp_dir().await.unwrap();
    let src = tmp.path().join("missing.txt");
    let dest = tmp.path().join("dest.txt");

    let err = system.rename(&src, &dest).await.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::NotFound);
}

#[tokio::test(flavor = "current_thread")]
async fn canonicalize_resolves_existing_path() {
    let system = RealSystem::new();
    let tmp = system.create_temp_dir().await.unwrap();
    let file = tmp.path().join("real.txt");
    system.write(&file, b"x", Condition::None).await.unwrap();

    let canonical = system.canonicalize(&file).await.unwrap();
    assert!(canonical.is_absolute());
    assert!(canonical.ends_with("real.txt"));
}

#[tokio::test(flavor = "current_thread")]
async fn canonicalize_missing_path_is_not_found() {
    let system = RealSystem::new();
    let tmp = system.create_temp_dir().await.unwrap();
    let missing = tmp.path().join("missing");

    let err = system.canonicalize(&missing).await.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::NotFound);
}

async fn relative_walk_names(
    system: &RealSystem,
    root: &Path,
    follow_links: bool,
    hidden: bool,
) -> Vec<String> {
    let mut names: Vec<String> = system
        .walk_dir(root, follow_links, hidden)
        .await
        .unwrap()
        .iter()
        .map(|entry| {
            entry
                .path
                .strip_prefix(root)
                .unwrap()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    names.sort();
    names
}

#[tokio::test(flavor = "current_thread")]
async fn walk_dir_includes_nested_entries() {
    let system = RealSystem::new();
    let tmp = system.create_temp_dir().await.unwrap();
    let root = tmp.path();

    system
        .write(&root.join("top.txt"), b"t", Condition::None)
        .await
        .unwrap();
    system.create_dir_all(&root.join("sub")).await.unwrap();
    system
        .write(&root.join("sub").join("inner.txt"), b"i", Condition::None)
        .await
        .unwrap();

    let names = relative_walk_names(&system, root, false, false).await;
    assert_eq!(names, vec!["sub", "sub/inner.txt", "top.txt"]);
}

// `hidden` true includes dotfiles and false excludes them, as the trait documents.
#[tokio::test(flavor = "current_thread")]
async fn walk_dir_hidden_true_includes_dotfiles() {
    let system = RealSystem::new();
    let tmp = system.create_temp_dir().await.unwrap();
    let root = tmp.path();

    system
        .write(&root.join("visible.txt"), b"v", Condition::None)
        .await
        .unwrap();
    system
        .write(&root.join(".secret"), b"s", Condition::None)
        .await
        .unwrap();

    let names = relative_walk_names(&system, root, false, true).await;
    assert_eq!(names, vec![".secret", "visible.txt"]);
}

#[tokio::test(flavor = "current_thread")]
async fn walk_dir_hidden_false_excludes_dotfiles() {
    let system = RealSystem::new();
    let tmp = system.create_temp_dir().await.unwrap();
    let root = tmp.path();

    system
        .write(&root.join("visible.txt"), b"v", Condition::None)
        .await
        .unwrap();
    system
        .write(&root.join(".secret"), b"s", Condition::None)
        .await
        .unwrap();

    let names = relative_walk_names(&system, root, false, false).await;
    assert_eq!(names, vec!["visible.txt"]);
}

#[tokio::test(flavor = "current_thread")]
async fn walk_dir_missing_path_yields_no_entries() {
    let system = RealSystem::new();
    let tmp = system.create_temp_dir().await.unwrap();
    let missing = tmp.path().join("missing");

    let entries = system.walk_dir(&missing, false, false).await.unwrap();
    assert!(entries.is_empty());
}

#[cfg(unix)]
#[tokio::test(flavor = "current_thread")]
async fn walk_dir_follow_links_true_descends_symlinked_dir() {
    use std::os::unix::fs::symlink;

    let system = RealSystem::new();
    let tmp = system.create_temp_dir().await.unwrap();
    let root = tmp.path();

    let target = root.join("target");
    system.create_dir_all(&target).await.unwrap();
    system
        .write(&target.join("inside.txt"), b"x", Condition::None)
        .await
        .unwrap();

    let walk_root = root.join("walk");
    system.create_dir_all(&walk_root).await.unwrap();
    symlink(&target, walk_root.join("link")).unwrap();

    let names = relative_walk_names(&system, &walk_root, true, false).await;
    assert!(names.iter().any(|name| name == "link/inside.txt"));
}

#[cfg(unix)]
#[tokio::test(flavor = "current_thread")]
async fn walk_dir_follow_links_false_does_not_descend_symlinked_dir() {
    use std::os::unix::fs::symlink;

    let system = RealSystem::new();
    let tmp = system.create_temp_dir().await.unwrap();
    let root = tmp.path();

    let target = root.join("target");
    system.create_dir_all(&target).await.unwrap();
    system
        .write(&target.join("inside.txt"), b"x", Condition::None)
        .await
        .unwrap();

    let walk_root = root.join("walk");
    system.create_dir_all(&walk_root).await.unwrap();
    symlink(&target, walk_root.join("link")).unwrap();

    let names = relative_walk_names(&system, &walk_root, false, false).await;
    assert!(names.iter().any(|name| name == "link"));
    assert!(!names.iter().any(|name| name == "link/inside.txt"));
}

#[test]
fn a_sleep_polled_once_is_pending_without_waiting() {
    let system = RealSystem::new();

    let outcome = run_without_runtime(system.sleep(Duration::from_secs(3600)));

    assert!(outcome.is_none());
}

#[cfg(slow_tests)]
fn block_on_parking<F>(future: F) -> F::Output
where
    F: Future,
{
    let mut pinned = pin!(future);
    let waker = Waker::from(Arc::new(ParkedThread(thread::current())));
    let mut context = Context::from_waker(&waker);
    loop {
        if let Poll::Ready(output) = pinned.as_mut().poll(&mut context) {
            return output;
        }
        thread::park();
    }
}

#[cfg(slow_tests)]
#[tokio::test(flavor = "current_thread")]
async fn sleep_waits_at_least_the_duration_on_a_current_thread_runtime() {
    let system = RealSystem::new();
    let start = MachineInstant::now();

    system.sleep(Duration::from_millis(50)).await.unwrap();

    assert!(start.elapsed() >= Duration::from_millis(50));
}

#[cfg(slow_tests)]
#[test]
fn sleep_waits_at_least_the_duration_on_an_executor_that_is_not_tokio() {
    let system = RealSystem::new();
    let start = MachineInstant::now();

    block_on_parking(system.sleep(Duration::from_millis(50))).unwrap();

    assert!(start.elapsed() >= Duration::from_millis(50));
}

#[tokio::test(flavor = "current_thread")]
async fn a_temporary_directory_stays_on_the_machine() {
    let bucket = MemoryBucket::default();
    let system = RealSystem::with_bucket(bucket.clone());
    let dir = system.create_temp_dir().await.unwrap();
    let scratch = dir.path().join("note.txt");

    system
        .write(&scratch, b"scratch", Condition::None)
        .await
        .unwrap();

    assert!(scratch.exists());
    assert!(
        bucket
            .list(Path::new("/"), Depth::All)
            .await
            .unwrap()
            .is_empty()
    );

    drop(dir);
    assert!(!scratch.exists());
}

#[tokio::test(flavor = "current_thread")]
async fn a_path_outside_a_temporary_directory_goes_to_the_bucket() {
    let bucket = MemoryBucket::default();
    let system = RealSystem::with_bucket(bucket.clone());
    let _dir = system.create_temp_dir().await.unwrap();

    system
        .write(Path::new("/kept.txt"), b"kept", Condition::None)
        .await
        .unwrap();

    assert_eq!(bucket.get(Path::new("/kept.txt")).await.unwrap().0, b"kept");
}

#[tokio::test(flavor = "current_thread")]
async fn renaming_out_of_a_temporary_directory_moves_the_file_into_the_bucket() {
    let bucket = MemoryBucket::default();
    let system = RealSystem::with_bucket(bucket.clone());
    let dir = system.create_temp_dir().await.unwrap();
    let scratch = dir.path().join("note.txt");
    system
        .write(&scratch, b"scratch", Condition::None)
        .await
        .unwrap();

    system
        .rename(&scratch, Path::new("/moved.txt"))
        .await
        .unwrap();

    assert_eq!(
        bucket.get(Path::new("/moved.txt")).await.unwrap().0,
        b"scratch"
    );
    assert!(!scratch.exists());
}

#[tokio::test(flavor = "current_thread")]
async fn renaming_a_folder_out_of_a_temporary_directory_moves_every_file() {
    let bucket = MemoryBucket::default();
    let system = RealSystem::with_bucket(bucket.clone());
    let dir = system.create_temp_dir().await.unwrap();
    let folder = dir.path().join("tree");
    system.create_dir_all(&folder.join("inner")).await.unwrap();
    system
        .write(&folder.join("inner/a.txt"), b"a", Condition::None)
        .await
        .unwrap();

    system.rename(&folder, Path::new("/tree")).await.unwrap();

    assert_eq!(
        bucket.get(Path::new("/tree/inner/a.txt")).await.unwrap().0,
        b"a"
    );
    assert!(!folder.exists());
}

#[tokio::test(flavor = "current_thread")]
async fn copying_into_a_temporary_directory_leaves_the_source_in_the_bucket() {
    let bucket = MemoryBucket::default();
    let system = RealSystem::with_bucket(bucket.clone());
    system
        .write(Path::new("/source.txt"), b"source", Condition::None)
        .await
        .unwrap();
    let dir = system.create_temp_dir().await.unwrap();
    let copy = dir.path().join("copy.txt");

    let copied = system.copy(Path::new("/source.txt"), &copy).await.unwrap();

    assert_eq!(copied, 6);
    assert_eq!(fs::read(&copy).unwrap(), b"source");
    assert_eq!(
        bucket.get(Path::new("/source.txt")).await.unwrap().0,
        b"source"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn dropping_a_handle_removes_its_directory_and_its_files() {
    let system = RealSystem::new();
    let dir = system.create_temp_dir().await.unwrap();
    let root = dir.path().to_path_buf();
    system.create_dir_all(&root.join("deep")).await.unwrap();
    system
        .write(&root.join("deep/a.txt"), b"a", Condition::None)
        .await
        .unwrap();

    drop(dir);

    assert!(!root.exists());
}
