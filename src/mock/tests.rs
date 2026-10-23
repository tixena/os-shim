use super::*;
use std::path::Path;

#[test]
fn metadata_returns_correct_len_for_file() {
    let system = MemorySystem::new()
        .with_file("/test/file.txt", b"hello")
        .unwrap();

    let meta = system.metadata(Path::new("/test/file.txt")).unwrap();
    assert!(meta.is_file);
    assert!(!meta.is_dir);
    assert_eq!(meta.len, 5);
}

#[test]
fn metadata_returns_dir_info() {
    let system = MemorySystem::new().with_dir("/mydir").unwrap();

    let meta = system.metadata(Path::new("/mydir")).unwrap();
    assert!(meta.is_dir);
    assert!(!meta.is_file);
    assert_eq!(meta.len, 0);
}

#[test]
fn metadata_not_found() {
    let system = MemorySystem::new();
    let result = system.metadata(Path::new("/missing"));
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().kind(), io::ErrorKind::NotFound);
}

#[test]
fn metadata_modified_updates_on_write() {
    let start = SystemTime::UNIX_EPOCH
        .checked_add(Duration::from_secs(1_700_000_000))
        .unwrap();
    let system = MemorySystem::new()
        .with_dir("/test")
        .unwrap()
        .with_now(start)
        .unwrap();

    system
        .write(Path::new("/test/file.txt"), b"first", WriteMode::Replace)
        .unwrap();
    let mtime1 = system
        .metadata(Path::new("/test/file.txt"))
        .unwrap()
        .modified;

    system.advance(Duration::from_secs(1)).unwrap();
    system
        .write(Path::new("/test/file.txt"), b"second", WriteMode::Replace)
        .unwrap();
    let mtime2 = system
        .metadata(Path::new("/test/file.txt"))
        .unwrap()
        .modified;

    assert_eq!(mtime1, start);
    assert_eq!(mtime2, mtime1.checked_add(Duration::from_secs(1)).unwrap());
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

#[test]
fn rename_file_moves_content() {
    let system = MemorySystem::new()
        .with_file("/test/a.txt", b"content")
        .unwrap();

    system
        .rename(Path::new("/test/a.txt"), Path::new("/test/b.txt"))
        .unwrap();

    assert!(!system.exists(Path::new("/test/a.txt")).unwrap());
    assert!(system.exists(Path::new("/test/b.txt")).unwrap());
    assert_eq!(
        system.read_to_string(Path::new("/test/b.txt")).unwrap(),
        "content"
    );
}

#[test]
fn rename_keeps_the_modification_time() {
    let start = SystemTime::UNIX_EPOCH
        .checked_add(Duration::from_secs(1_700_000_000))
        .unwrap();
    let system = MemorySystem::new()
        .with_dir("/test")
        .unwrap()
        .with_now(start)
        .unwrap();
    system
        .write(Path::new("/test/m.md"), b"m", WriteMode::Replace)
        .unwrap();
    system.advance(Duration::from_secs(60)).unwrap();

    system
        .rename(Path::new("/test/m.md"), Path::new("/test/n.md"))
        .unwrap();

    assert_eq!(
        system.metadata(Path::new("/test/n.md")).unwrap().modified,
        start
    );
}

#[test]
fn rename_of_a_folder_keeps_its_files_modification_times() {
    let start = SystemTime::UNIX_EPOCH
        .checked_add(Duration::from_secs(1_700_000_000))
        .unwrap();
    let system = MemorySystem::new()
        .with_dir("/test/g")
        .unwrap()
        .with_now(start)
        .unwrap();
    system
        .write(Path::new("/test/g/k.md"), b"k", WriteMode::Replace)
        .unwrap();
    system.advance(Duration::from_secs(60)).unwrap();

    system
        .rename(Path::new("/test/g"), Path::new("/test/h"))
        .unwrap();

    assert_eq!(
        system.metadata(Path::new("/test/h/k.md")).unwrap().modified,
        start
    );
}

#[test]
fn rename_nonexistent_returns_error() {
    let system = MemorySystem::new();
    let result = system.rename(Path::new("/missing"), Path::new("/other"));
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().kind(), io::ErrorKind::NotFound);
}

#[test]
fn open_append_creates_new_file() {
    let system = MemorySystem::new().with_dir("/test").unwrap();

    let mut writer = system.open_append(Path::new("/test/new.txt")).unwrap();
    writer.write_all(b"hello").unwrap();
    writer.flush().unwrap();
    drop(writer);

    assert_eq!(
        system.read_to_string(Path::new("/test/new.txt")).unwrap(),
        "hello"
    );
}

#[test]
fn open_append_extends_existing() {
    let system = MemorySystem::new()
        .with_file("/test/file.txt", b"hello")
        .unwrap();

    let mut writer = system.open_append(Path::new("/test/file.txt")).unwrap();
    writer.write_all(b" world").unwrap();
    writer.flush().unwrap();
    drop(writer);

    assert_eq!(
        system.read_to_string(Path::new("/test/file.txt")).unwrap(),
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

#[test]
fn canonicalize_passes_absolute_through_and_joins_relative() {
    let system = MemorySystem::new().with_current_dir("/base").unwrap();

    assert_eq!(
        system.canonicalize(Path::new("/already/absolute")).unwrap(),
        PathBuf::from("/already/absolute")
    );
    assert_eq!(
        system.canonicalize(Path::new("rel/path")).unwrap(),
        PathBuf::from("/base/rel/path")
    );
}

#[test]
fn create_temp_dir_exists_then_cleaned_up_on_drop() {
    let system = MemorySystem::new();
    let temp = system.create_temp_dir().unwrap();
    let path = temp.path().to_path_buf();

    assert!(system.is_dir(&path).unwrap());
    drop(temp);
    assert!(!system.exists(&path).unwrap());
}

#[test]
fn read_to_string_rejects_invalid_utf8() {
    let system = MemorySystem::new()
        .with_file("/bad.bin", &[0xff, 0xfe, 0x00])
        .unwrap();

    let err = system.read_to_string(Path::new("/bad.bin")).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidData);
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

#[test]
fn with_dir_over_a_file_fails_and_keeps_the_file() {
    let system = MemorySystem::new().with_file("/v/a.md", b"a").unwrap();

    let kind = system
        .clone()
        .with_dir("/v/a.md")
        .err()
        .map(|err| err.kind());

    assert_eq!(kind, Some(io::ErrorKind::AlreadyExists));
    assert!(system.is_file(Path::new("/v/a.md")).unwrap());
    assert!(!system.is_dir(Path::new("/v/a.md")).unwrap());
}

#[test]
fn with_dir_below_a_file_fails_and_keeps_the_file() {
    let system = MemorySystem::new().with_file("/v/a.md", b"a").unwrap();

    let kind = system
        .clone()
        .with_dir("/v/a.md/sub")
        .err()
        .map(|err| err.kind());

    assert_eq!(kind, Some(io::ErrorKind::NotADirectory));
    assert!(system.is_file(Path::new("/v/a.md")).unwrap());
    assert!(!system.is_dir(Path::new("/v/a.md")).unwrap());
    assert!(!system.exists(Path::new("/v/a.md/sub")).unwrap());
}

#[test]
fn with_file_below_a_file_fails_and_keeps_the_file() {
    let system = MemorySystem::new().with_file("/v/a.md", b"a").unwrap();

    let kind = system
        .clone()
        .with_file("/v/a.md/b.md", b"b")
        .err()
        .map(|err| err.kind());

    assert_eq!(kind, Some(io::ErrorKind::NotADirectory));
    assert!(system.is_file(Path::new("/v/a.md")).unwrap());
    assert!(!system.is_dir(Path::new("/v/a.md")).unwrap());
    assert!(!system.exists(Path::new("/v/a.md/b.md")).unwrap());
}

#[test]
fn with_file_over_a_folder_fails_and_keeps_the_folder() {
    let system = MemorySystem::new().with_dir("/v/d").unwrap();

    let kind = system
        .clone()
        .with_file("/v/d", b"d")
        .err()
        .map(|err| err.kind());

    assert_eq!(kind, Some(io::ErrorKind::IsADirectory));
    assert!(system.is_dir(Path::new("/v/d")).unwrap());
    assert!(!system.is_file(Path::new("/v/d")).unwrap());
}

#[test]
fn default_matches_new() {
    let system = MemorySystem::default();
    assert_eq!(system.current_dir().unwrap(), PathBuf::from("/"));
    assert!(system.is_dir(Path::new("/")).unwrap());
}

fn epoch_plus(seconds: u64) -> SystemTime {
    SystemTime::UNIX_EPOCH
        .checked_add(Duration::from_secs(seconds))
        .unwrap()
}

#[test]
fn sleep_moves_the_clock_forward_by_the_duration() {
    let system = MemorySystem::new().with_now(epoch_plus(100)).unwrap();

    system.sleep(Duration::from_secs(2)).unwrap();

    assert_eq!(system.now(), epoch_plus(102));
}

#[test]
fn sleeping_twice_adds_the_durations() {
    let system = MemorySystem::new().with_now(epoch_plus(100)).unwrap();

    system.sleep(Duration::from_secs(2)).unwrap();
    system.sleep(Duration::from_secs(3)).unwrap();

    assert_eq!(system.now(), epoch_plus(105));
}

#[test]
fn sleep_that_overflows_the_clock_is_invalid_input_and_leaves_it_unchanged() {
    let near_end = epoch_plus(1_000);
    let system = MemorySystem::new().with_now(near_end).unwrap();

    let err = system.sleep(Duration::MAX).unwrap_err();

    assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    assert_eq!(system.now(), near_end);
}

#[test]
fn files_written_after_a_sleep_take_the_moved_clock_as_modified() {
    let system = MemorySystem::new()
        .with_dir("/test")
        .unwrap()
        .with_now(epoch_plus(100))
        .unwrap();

    system.sleep(Duration::from_secs(2)).unwrap();
    system
        .write(Path::new("/test/file.txt"), b"x", WriteMode::Replace)
        .unwrap();

    let meta = system.metadata(Path::new("/test/file.txt")).unwrap();
    assert_eq!(meta.modified, epoch_plus(102));
}

#[cfg(feature = "async")]
#[tokio::test(flavor = "current_thread")]
async fn sleep_async_through_dyn_system_moves_the_clock_when_awaited() {
    let system = MemorySystem::new().with_now(epoch_plus(100)).unwrap();
    let shared: &dyn System = &system;

    let pending = shared.sleep_async(Duration::from_secs(3));
    assert_eq!(system.now(), epoch_plus(100));
    pending.await.unwrap();

    assert_eq!(system.now(), epoch_plus(103));
}

#[cfg(feature = "async")]
#[tokio::test(flavor = "current_thread")]
async fn sleep_async_dropped_unawaited_leaves_the_clock_where_it_was() {
    let system = MemorySystem::new().with_now(epoch_plus(100)).unwrap();

    drop(system.sleep_async(Duration::from_secs(3)));

    assert_eq!(system.now(), epoch_plus(100));
}

#[cfg(feature = "async")]
#[tokio::test(flavor = "current_thread")]
async fn sleep_async_that_overflows_the_clock_is_invalid_input_and_leaves_it_unchanged() {
    let near_end = epoch_plus(1_000);
    let system = MemorySystem::new().with_now(near_end).unwrap();

    let err = system.sleep_async(Duration::MAX).await.unwrap_err();

    assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    assert_eq!(system.now(), near_end);
}
