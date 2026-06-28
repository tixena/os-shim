use super::*;
use std::path::Path;

#[test]
fn metadata_returns_correct_len_for_file() {
    let system = MockSystem::new()
        .with_file("/test/file.txt", b"hello")
        .unwrap();

    let meta = system.metadata(Path::new("/test/file.txt")).unwrap();
    assert!(meta.is_file);
    assert!(!meta.is_dir);
    assert_eq!(meta.len, 5);
}

#[test]
fn metadata_returns_dir_info() {
    let system = MockSystem::new().with_dir("/mydir").unwrap();

    let meta = system.metadata(Path::new("/mydir")).unwrap();
    assert!(meta.is_dir);
    assert!(!meta.is_file);
    assert_eq!(meta.len, 0);
}

#[test]
fn metadata_not_found() {
    let system = MockSystem::new();
    let result = system.metadata(Path::new("/missing"));
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().kind(), io::ErrorKind::NotFound);
}

#[test]
fn metadata_modified_updates_on_write() {
    let system = MockSystem::new().with_dir("/test").unwrap();

    system.write(Path::new("/test/file.txt"), b"first").unwrap();
    let mtime1 = system
        .metadata(Path::new("/test/file.txt"))
        .unwrap()
        .modified;

    // Write again to update the modification time
    system
        .write(Path::new("/test/file.txt"), b"second")
        .unwrap();
    let mtime2 = system
        .metadata(Path::new("/test/file.txt"))
        .unwrap()
        .modified;

    assert!(mtime2 >= mtime1);
}

#[test]
fn rename_file_moves_content() {
    let system = MockSystem::new()
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
fn rename_nonexistent_returns_error() {
    let system = MockSystem::new();
    let result = system.rename(Path::new("/missing"), Path::new("/other"));
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().kind(), io::ErrorKind::NotFound);
}

#[test]
fn open_append_creates_new_file() {
    let system = MockSystem::new().with_dir("/test").unwrap();

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
    let system = MockSystem::new()
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
    let system = MockSystem::new()
        .with_current_exe("/usr/bin/vigil")
        .unwrap();

    assert_eq!(
        system.current_exe().unwrap(),
        PathBuf::from("/usr/bin/vigil")
    );
}

#[test]
fn current_exe_returns_default() {
    let system = MockSystem::new();
    assert_eq!(system.current_exe().unwrap(), PathBuf::from("/mock/exe"));
}

#[test]
fn set_env_var_updates_env() {
    let system = MockSystem::new();
    system.set_env_var("MY_KEY", "my_value");
    assert_eq!(system.env_var("MY_KEY").unwrap(), "my_value");
}

#[cfg(feature = "process")]
#[test]
fn is_pid_alive_checks_mock_pids() {
    let system = MockSystem::new().with_pid(1234).unwrap();

    assert!(system.is_pid_alive(1234));
    assert!(!system.is_pid_alive(9999));
}
