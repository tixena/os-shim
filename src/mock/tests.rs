use super::*;
use crate::bucket::{Condition, Depth, MemoryBucket};
use crate::error::ErrorKind;
use crate::streams::{close, read_to_end, write_all};
use core::sync::atomic::AtomicUsize;
use std::path::Path;

/// A bucket over memory that counts the whole-file reads and the streamed reads it serves.
struct CountingBucket {
    /// How many times `get` ran.
    gets: Arc<AtomicUsize>,
    /// The bucket that holds the files.
    inner: MemoryBucket,
    /// How many times `read_stream` ran.
    streams: Arc<AtomicUsize>,
}

impl Bucket for CountingBucket {
    fn delete(
        &self,
        path: &Path,
        condition: &Condition,
    ) -> impl Future<Output = Result<()>> + Send {
        self.inner.delete(path, condition)
    }

    fn get(&self, path: &Path) -> impl Future<Output = Result<(Vec<u8>, RevisionToken)>> + Send {
        self.gets.fetch_add(1, Ordering::SeqCst);
        self.inner.get(path)
    }

    fn guarantees(&self) -> Guarantees {
        self.inner.guarantees()
    }

    fn head(&self, path: &Path) -> impl Future<Output = Result<FileMetadata>> + Send {
        self.inner.head(path)
    }

    fn list(
        &self,
        folder: &Path,
        depth: Depth,
    ) -> impl Future<Output = Result<Vec<WalkEntry>>> + Send {
        self.inner.list(folder, depth)
    }

    fn put(
        &self,
        path: &Path,
        bytes: &[u8],
        condition: &Condition,
    ) -> impl Future<Output = Result<RevisionToken>> + Send {
        self.inner.put(path, bytes, condition)
    }

    fn read_stream<'bucket>(
        &'bucket self,
        path: &Path,
    ) -> impl Future<Output = Result<Box<dyn AsyncRead + Send + Unpin + 'bucket>>> + Send {
        self.streams.fetch_add(1, Ordering::SeqCst);
        self.inner.read_stream(path)
    }
}

#[tokio::test(flavor = "current_thread")]
async fn metadata_returns_correct_len_for_file() {
    let system = MemorySystem::new()
        .with_file("/test/file.txt", b"hello")
        .unwrap();

    let meta = system.metadata(Path::new("/test/file.txt")).await.unwrap();
    assert!(meta.is_file);
    assert!(!meta.is_dir);
    assert_eq!(meta.len, 5);
}

#[tokio::test(flavor = "current_thread")]
async fn metadata_returns_dir_info() {
    let system = MemorySystem::new().with_dir("/mydir").unwrap();

    let meta = system.metadata(Path::new("/mydir")).await.unwrap();
    assert!(meta.is_dir);
    assert!(!meta.is_file);
    assert_eq!(meta.len, 0);
}

#[tokio::test(flavor = "current_thread")]
async fn metadata_not_found() {
    let system = MemorySystem::new();
    let result = system.metadata(Path::new("/missing")).await;
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().kind(), ErrorKind::NotFound);
}

#[tokio::test(flavor = "current_thread")]
async fn metadata_modified_updates_on_write() {
    let start = SystemTime::UNIX_EPOCH
        .checked_add(Duration::from_secs(1_700_000_000))
        .unwrap();
    let system = MemorySystem::new()
        .with_dir("/test")
        .unwrap()
        .with_now(start)
        .unwrap();

    system
        .write(Path::new("/test/file.txt"), b"first", Condition::None)
        .await
        .unwrap();
    let mtime1 = system
        .metadata(Path::new("/test/file.txt"))
        .await
        .unwrap()
        .modified;

    system.advance(Duration::from_secs(1)).unwrap();
    system
        .write(Path::new("/test/file.txt"), b"second", Condition::None)
        .await
        .unwrap();
    let mtime2 = system
        .metadata(Path::new("/test/file.txt"))
        .await
        .unwrap()
        .modified;

    assert_eq!(mtime1, Some(start));
    assert_eq!(mtime2, start.checked_add(Duration::from_secs(1)));
}

#[test]
fn now_starts_at_the_unix_epoch() {
    assert_eq!(MemorySystem::new().now(), SystemTime::UNIX_EPOCH);
}

#[test]
fn with_now_sets_the_time() {
    let time = SystemTime::UNIX_EPOCH
        .checked_add(Duration::from_secs(1_700_000_000))
        .unwrap();
    let system = MemorySystem::new().with_now(time).unwrap();

    assert_eq!(system.now(), time);
}

#[test]
fn advance_moves_the_time_forward() {
    let time = SystemTime::UNIX_EPOCH
        .checked_add(Duration::from_secs(1_700_000_000))
        .unwrap();
    let system = MemorySystem::new().with_now(time).unwrap();

    system.advance(Duration::from_secs(5)).unwrap();

    assert_eq!(
        system.now(),
        time.checked_add(Duration::from_secs(5)).unwrap()
    );
}

#[test]
fn advance_that_overflows_errors_and_keeps_the_time() {
    let system = MemorySystem::new();

    let err = system.advance(Duration::MAX).unwrap_err();

    assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    assert_eq!(system.now(), SystemTime::UNIX_EPOCH);
}

#[tokio::test(flavor = "current_thread")]
async fn rename_file_moves_content() {
    let system = MemorySystem::new()
        .with_file("/test/a.txt", b"content")
        .unwrap();

    system
        .rename(Path::new("/test/a.txt"), Path::new("/test/b.txt"))
        .await
        .unwrap();

    assert!(!system.exists(Path::new("/test/a.txt")).await.unwrap());
    assert!(system.exists(Path::new("/test/b.txt")).await.unwrap());
    assert_eq!(
        system
            .read_to_string(Path::new("/test/b.txt"))
            .await
            .unwrap(),
        "content"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn rename_keeps_the_modification_time() {
    let start = SystemTime::UNIX_EPOCH
        .checked_add(Duration::from_secs(1_700_000_000))
        .unwrap();
    let system = MemorySystem::new()
        .with_dir("/test")
        .unwrap()
        .with_now(start)
        .unwrap();
    system
        .write(Path::new("/test/m.md"), b"m", Condition::None)
        .await
        .unwrap();
    system.advance(Duration::from_secs(60)).unwrap();

    system
        .rename(Path::new("/test/m.md"), Path::new("/test/n.md"))
        .await
        .unwrap();

    assert_eq!(
        system
            .metadata(Path::new("/test/n.md"))
            .await
            .unwrap()
            .modified,
        Some(start)
    );
}

#[tokio::test(flavor = "current_thread")]
async fn rename_of_a_folder_keeps_its_files_modification_times() {
    let start = SystemTime::UNIX_EPOCH
        .checked_add(Duration::from_secs(1_700_000_000))
        .unwrap();
    let system = MemorySystem::new()
        .with_dir("/test/g")
        .unwrap()
        .with_now(start)
        .unwrap();
    system
        .write(Path::new("/test/g/k.md"), b"k", Condition::None)
        .await
        .unwrap();
    system.advance(Duration::from_secs(60)).unwrap();

    system
        .rename(Path::new("/test/g"), Path::new("/test/h"))
        .await
        .unwrap();

    assert_eq!(
        system
            .metadata(Path::new("/test/h/k.md"))
            .await
            .unwrap()
            .modified,
        Some(start)
    );
}

#[tokio::test(flavor = "current_thread")]
async fn rename_nonexistent_returns_error() {
    let system = MemorySystem::new();
    let result = system
        .rename(Path::new("/missing"), Path::new("/other"))
        .await;
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().kind(), ErrorKind::NotFound);
}

#[tokio::test(flavor = "current_thread")]
async fn open_append_creates_new_file() {
    let system = MemorySystem::new().with_dir("/test").unwrap();

    let mut writer = system
        .open_append(Path::new("/test/new.txt"))
        .await
        .unwrap();
    write_all(&mut *writer, b"hello").await.unwrap();
    close(&mut *writer).await.unwrap();
    drop(writer);

    assert_eq!(
        system
            .read_to_string(Path::new("/test/new.txt"))
            .await
            .unwrap(),
        "hello"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn open_append_extends_existing() {
    let system = MemorySystem::new()
        .with_file("/test/file.txt", b"hello")
        .unwrap();

    let mut writer = system
        .open_append(Path::new("/test/file.txt"))
        .await
        .unwrap();
    write_all(&mut *writer, b" world").await.unwrap();
    close(&mut *writer).await.unwrap();
    drop(writer);

    assert_eq!(
        system
            .read_to_string(Path::new("/test/file.txt"))
            .await
            .unwrap(),
        "hello world"
    );
}

#[test]
fn current_exe_returns_configured_path() {
    let system = MemorySystem::new()
        .with_current_exe("/usr/bin/vigil")
        .unwrap();

    assert_eq!(
        system.current_exe().unwrap(),
        PathBuf::from("/usr/bin/vigil")
    );
}

#[test]
fn current_exe_returns_default() {
    let system = MemorySystem::new();
    assert_eq!(system.current_exe().unwrap(), PathBuf::from("/mock/exe"));
}

#[test]
fn set_env_var_updates_env() {
    let system = MemorySystem::new();
    system.set_env_var("MY_KEY", "my_value");
    assert_eq!(system.env_var("MY_KEY").unwrap(), "my_value");
}

#[cfg(feature = "process")]
#[test]
fn is_pid_alive_checks_mock_pids() {
    let system = MemorySystem::new().with_pid(1234).unwrap();

    assert!(system.is_pid_alive(1234));
    assert!(!system.is_pid_alive(9999));
}

#[tokio::test(flavor = "current_thread")]
async fn canonicalize_passes_absolute_through_and_joins_relative() {
    let system = MemorySystem::new().with_current_dir("/base").unwrap();

    assert_eq!(
        system
            .canonicalize(Path::new("/already/absolute"))
            .await
            .unwrap(),
        PathBuf::from("/already/absolute")
    );
    assert_eq!(
        system.canonicalize(Path::new("rel/path")).await.unwrap(),
        PathBuf::from("/base/rel/path")
    );
}

#[tokio::test(flavor = "current_thread")]
async fn create_temp_dir_exists_then_cleaned_up_on_drop() {
    let system = MemorySystem::new();
    let temp = system.create_temp_dir().await.unwrap();
    let path = temp.path().to_path_buf();

    assert!(system.is_dir(&path).await.unwrap());
    drop(temp);
    assert!(!system.exists(&path).await.unwrap());
}

#[tokio::test(flavor = "current_thread")]
async fn read_to_string_rejects_invalid_utf8() {
    let system = MemorySystem::new()
        .with_file("/bad.bin", &[0xff, 0xfe, 0x00])
        .unwrap();

    let err = system
        .read_to_string(Path::new("/bad.bin"))
        .await
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidData);
}

#[test]
fn with_current_dir_sets_current_dir() {
    let system = MemorySystem::new().with_current_dir("/custom/cwd").unwrap();
    assert_eq!(system.current_dir().unwrap(), PathBuf::from("/custom/cwd"));
}

#[test]
fn with_env_sets_env_var() {
    let system = MemorySystem::new().with_env("CONFIGURED", "yes").unwrap();
    assert_eq!(system.env_var("CONFIGURED").unwrap(), "yes");
}

#[tokio::test(flavor = "current_thread")]
async fn with_dir_over_a_file_fails_and_keeps_the_file() {
    let system = MemorySystem::new().with_file("/v/a.md", b"a").unwrap();

    let kind = system
        .clone()
        .with_dir("/v/a.md")
        .err()
        .map(|err| err.kind());

    assert_eq!(kind, Some(io::ErrorKind::AlreadyExists));
    assert!(system.is_file(Path::new("/v/a.md")).await.unwrap());
    assert!(!system.is_dir(Path::new("/v/a.md")).await.unwrap());
}

#[tokio::test(flavor = "current_thread")]
async fn with_dir_below_a_file_fails_and_keeps_the_file() {
    let system = MemorySystem::new().with_file("/v/a.md", b"a").unwrap();

    let kind = system
        .clone()
        .with_dir("/v/a.md/sub")
        .err()
        .map(|err| err.kind());

    assert_eq!(kind, Some(io::ErrorKind::NotADirectory));
    assert!(system.is_file(Path::new("/v/a.md")).await.unwrap());
    assert!(!system.is_dir(Path::new("/v/a.md")).await.unwrap());
    assert!(!system.exists(Path::new("/v/a.md/sub")).await.unwrap());
}

#[tokio::test(flavor = "current_thread")]
async fn with_file_below_a_file_fails_and_keeps_the_file() {
    let system = MemorySystem::new().with_file("/v/a.md", b"a").unwrap();

    let kind = system
        .clone()
        .with_file("/v/a.md/b.md", b"b")
        .err()
        .map(|err| err.kind());

    assert_eq!(kind, Some(io::ErrorKind::NotADirectory));
    assert!(system.is_file(Path::new("/v/a.md")).await.unwrap());
    assert!(!system.is_dir(Path::new("/v/a.md")).await.unwrap());
    assert!(!system.exists(Path::new("/v/a.md/b.md")).await.unwrap());
}

#[tokio::test(flavor = "current_thread")]
async fn with_file_over_a_folder_fails_and_keeps_the_folder() {
    let system = MemorySystem::new().with_dir("/v/d").unwrap();

    let kind = system
        .clone()
        .with_file("/v/d", b"d")
        .err()
        .map(|err| err.kind());

    assert_eq!(kind, Some(io::ErrorKind::IsADirectory));
    assert!(system.is_dir(Path::new("/v/d")).await.unwrap());
    assert!(!system.is_file(Path::new("/v/d")).await.unwrap());
}

#[tokio::test(flavor = "current_thread")]
async fn default_matches_new() {
    let system = MemorySystem::default();
    assert_eq!(system.current_dir().unwrap(), PathBuf::from("/"));
    assert!(system.is_dir(Path::new("/")).await.unwrap());
}

fn epoch_plus(seconds: u64) -> SystemTime {
    SystemTime::UNIX_EPOCH
        .checked_add(Duration::from_secs(seconds))
        .unwrap()
}

#[tokio::test(flavor = "current_thread")]
async fn sleep_moves_the_clock_forward_by_the_duration() {
    let system = MemorySystem::new().with_now(epoch_plus(100)).unwrap();

    system.sleep(Duration::from_secs(2)).await.unwrap();

    assert_eq!(system.now(), epoch_plus(102));
}

#[tokio::test(flavor = "current_thread")]
async fn sleeping_twice_adds_the_durations() {
    let system = MemorySystem::new().with_now(epoch_plus(100)).unwrap();

    system.sleep(Duration::from_secs(2)).await.unwrap();
    system.sleep(Duration::from_secs(3)).await.unwrap();

    assert_eq!(system.now(), epoch_plus(105));
}

#[tokio::test(flavor = "current_thread")]
async fn sleep_that_overflows_the_clock_is_invalid_input_and_leaves_it_unchanged() {
    let near_end = epoch_plus(1_000);
    let system = MemorySystem::new().with_now(near_end).unwrap();

    let err = system.sleep(Duration::MAX).await.unwrap_err();

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(system.now(), near_end);
}

#[tokio::test(flavor = "current_thread")]
async fn files_written_after_a_sleep_take_the_moved_clock_as_modified() {
    let system = MemorySystem::new()
        .with_dir("/test")
        .unwrap()
        .with_now(epoch_plus(100))
        .unwrap();

    system.sleep(Duration::from_secs(2)).await.unwrap();
    system
        .write(Path::new("/test/file.txt"), b"x", Condition::None)
        .await
        .unwrap();

    let meta = system.metadata(Path::new("/test/file.txt")).await.unwrap();
    assert_eq!(meta.modified, Some(epoch_plus(102)));
}

#[tokio::test(flavor = "current_thread")]
async fn sleep_through_dyn_system_moves_the_clock_when_awaited() {
    let system = MemorySystem::new().with_now(epoch_plus(100)).unwrap();
    let shared: &dyn System = &system;

    let pending = shared.sleep(Duration::from_secs(3));
    assert_eq!(system.now(), epoch_plus(100));
    pending.await.unwrap();

    assert_eq!(system.now(), epoch_plus(103));
}

#[tokio::test(flavor = "current_thread")]
async fn sleep_dropped_unawaited_leaves_the_clock_where_it_was() {
    let system = MemorySystem::new().with_now(epoch_plus(100)).unwrap();

    drop(system.sleep(Duration::from_secs(3)));

    assert_eq!(system.now(), epoch_plus(100));
}

#[tokio::test(flavor = "current_thread")]
async fn a_relative_path_is_resolved_against_the_current_dir() {
    let system = MemorySystem::new()
        .with_dir("/work")
        .unwrap()
        .with_current_dir("/work")
        .unwrap();

    system
        .write(Path::new("a.txt"), b"relative", Condition::Absent)
        .await
        .unwrap();

    assert!(system.exists(Path::new("/work/a.txt")).await.unwrap());
    assert_eq!(
        system.read_to_string(Path::new("a.txt")).await.unwrap(),
        "relative"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn streams_on_a_relative_path_land_at_the_resolved_path() {
    let system = MemorySystem::new()
        .with_dir("/work")
        .unwrap()
        .with_current_dir("/work")
        .unwrap();

    let mut created = system.create(Path::new("c.txt")).await.unwrap();
    write_all(&mut *created, b"created").await.unwrap();
    close(&mut *created).await.unwrap();
    drop(created);
    let mut appended = system.open_append(Path::new("c.txt")).await.unwrap();
    write_all(&mut *appended, b"+more").await.unwrap();
    close(&mut *appended).await.unwrap();
    drop(appended);
    let mut opened = system.open(Path::new("c.txt")).await.unwrap();
    let bytes = read_to_end(&mut *opened).await.unwrap();

    assert_eq!(bytes, b"created+more");
    assert_eq!(
        system
            .read_to_string(Path::new("/work/c.txt"))
            .await
            .unwrap(),
        "created+more"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn opening_a_relative_path_streams_the_file_without_reading_it_whole() {
    let gets = Arc::new(AtomicUsize::new(0));
    let streams = Arc::new(AtomicUsize::new(0));
    let inner = MemoryBucket::default();
    inner.create_folder(Path::new("/work")).await.unwrap();
    inner
        .put(Path::new("/work/big.bin"), b"bytes", &Condition::None)
        .await
        .unwrap();
    let system = MemorySystem::with_bucket(CountingBucket {
        inner,
        gets: Arc::clone(&gets),
        streams: Arc::clone(&streams),
    })
    .with_current_dir("/work")
    .unwrap();

    let mut reader = system.open(Path::new("big.bin")).await.unwrap();
    let bytes = read_to_end(&mut *reader).await.unwrap();

    assert_eq!(bytes, b"bytes");
    assert_eq!(streams.load(Ordering::SeqCst), 1);
    assert_eq!(gets.load(Ordering::SeqCst), 0);
}

#[tokio::test(flavor = "current_thread")]
async fn a_stale_token_leaves_the_file() {
    let system = MemorySystem::new();
    let path = Path::new("/lock");
    let first = system.write(path, b"one", Condition::Absent).await.unwrap();
    system.write(path, b"two", Condition::None).await.unwrap();

    assert!(!system.remove_file_if(path, &first).await.unwrap());

    assert_eq!(system.read_to_string(path).await.unwrap(), "two");
}

#[tokio::test(flavor = "current_thread")]
async fn guarantees_are_the_buckets() {
    assert_eq!(
        MemorySystem::new().guarantees(),
        MemoryBucket::default().guarantees()
    );
    assert_eq!(
        MemorySystem::with_bucket(MemoryBucket::default()).guarantees(),
        MemoryBucket::default().guarantees()
    );
}

#[tokio::test(flavor = "current_thread")]
async fn a_system_over_another_bucket_keeps_the_bucket_clock() {
    let bucket = MemoryBucket::new(Arc::new(|| SystemTime::UNIX_EPOCH + Duration::from_secs(7)));
    let system = MemorySystem::with_bucket(bucket);
    system.advance(Duration::from_secs(60)).unwrap();

    system
        .write(Path::new("/f"), b"x", Condition::Absent)
        .await
        .unwrap();

    assert_eq!(
        system.metadata(Path::new("/f")).await.unwrap().modified,
        Some(SystemTime::UNIX_EPOCH + Duration::from_secs(7))
    );
    assert_eq!(
        system.now(),
        SystemTime::UNIX_EPOCH + Duration::from_secs(60)
    );
}

#[tokio::test(flavor = "current_thread")]
async fn a_temporary_folder_lives_in_the_bucket_and_goes_with_its_handle() {
    let system = MemorySystem::new();
    let bucket = system.bucket.clone();

    let handle = system.create_temp_dir().await.unwrap();
    let path = handle.path().to_path_buf();
    assert!(bucket.head(&path).await.unwrap().is_dir);

    drop(handle);
    assert_eq!(
        bucket.head(&path).await.unwrap_err().kind(),
        ErrorKind::NotFound
    );
}

#[tokio::test(flavor = "current_thread")]
async fn a_temporary_directory_of_a_system_over_a_bucket_stays_out_of_the_bucket() {
    let bucket = MemoryBucket::default();
    let system = MemorySystem::with_bucket(bucket.clone());
    let dir = system.create_temp_dir().await.unwrap();
    let scratch = dir.path().join("note.txt");

    system
        .write(&scratch, b"scratch", Condition::None)
        .await
        .unwrap();

    assert!(system.exists(&scratch).await.unwrap());
    bucket.head(&scratch).await.unwrap_err();
    assert!(
        bucket
            .list(Path::new("/"), Depth::All)
            .await
            .unwrap()
            .is_empty()
    );

    drop(dir);
    assert!(!system.exists(&scratch).await.unwrap());
}

#[tokio::test(flavor = "current_thread")]
async fn a_temporary_directory_of_a_new_system_is_in_the_tree_of_its_other_files() {
    let system = MemorySystem::new();
    let dir = system.create_temp_dir().await.unwrap();
    let scratch = dir.path().join("note.txt");

    system
        .write(&scratch, b"scratch", Condition::None)
        .await
        .unwrap();

    assert!(system.is_dir(Path::new("/tmp")).await.unwrap());
    assert_eq!(system.read_to_string(&scratch).await.unwrap(), "scratch");
}

#[tokio::test(flavor = "current_thread")]
async fn renaming_out_of_a_temporary_directory_moves_the_file_into_the_bucket() {
    let bucket = MemoryBucket::default();
    let system = MemorySystem::with_bucket(bucket.clone());
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
    assert!(!system.exists(&scratch).await.unwrap());
}
