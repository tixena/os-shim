//! Memory<->Real parity harness.
//!
//! These tests run the SAME operation sequences against `MemorySystem` and
//! `RealSystem` and assert they agree on observable results (return values,
//! `io::ErrorKind` on failure, `metadata().len`, directory contents). A mock
//! that silently diverges from real is the dangerous failure mode -- green
//! tests, broken production -- so this harness is the highest-value guard.
//!
//! Known, currently-accepted divergences are pinned explicitly at the bottom of
//! this file (see the `divergence_*` tests) and tracked as separate issues so a
//! reviewer can decide whether to converge the behaviors.

use crate::mock::MemorySystem;
use crate::real::RealSystem;
use crate::{System, WriteMode};
use std::env::VarError;
use std::io::{self, Read as _, Write as _};
use std::path::{Path, PathBuf};

/// Assert both results failed with the same `io::ErrorKind`.
fn assert_both_err_kind<L, R>(left: io::Result<L>, right: io::Result<R>, kind: io::ErrorKind) {
    assert_eq!(left.err().map(|err| err.kind()), Some(kind));
    assert_eq!(right.err().map(|err| err.kind()), Some(kind));
}

/// Build the same standard tree under `root` on a system.
fn populate(system: &dyn System, root: &Path) {
    system.create_dir_all(root).unwrap();
    system
        .write(&root.join("alpha.txt"), b"alpha", WriteMode::Replace)
        .unwrap();
    system
        .write(&root.join("empty.txt"), b"", WriteMode::Replace)
        .unwrap();
    system.create_dir_all(&root.join("sub")).unwrap();
    system
        .write(
            &root.join("sub").join("beta.txt"),
            b"beta",
            WriteMode::Replace,
        )
        .unwrap();
}

/// Sort and strip `root` from a list of paths into comparable relative names.
fn relative_names(paths: &[PathBuf], root: &Path) -> Vec<String> {
    let mut names: Vec<String> = paths
        .iter()
        .map(|path| {
            path.strip_prefix(root)
                .unwrap()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    names.sort();
    names
}

#[test]
fn parity_write_read_roundtrip() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    let real_file = tmp.path().join("file.txt");
    real.write(&real_file, b"payload", WriteMode::Replace)
        .unwrap();

    let mock = MemorySystem::new().with_dir("/work").unwrap();
    let mock_file = Path::new("/work/file.txt");
    mock.write(mock_file, b"payload", WriteMode::Replace)
        .unwrap();

    assert_eq!(
        real.read_to_string(&real_file).unwrap(),
        mock.read_to_string(mock_file).unwrap()
    );
}

#[test]
fn parity_metadata_len_for_known_and_empty_files() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    real.write(&tmp.path().join("known.txt"), b"12345", WriteMode::Replace)
        .unwrap();
    real.write(&tmp.path().join("empty.txt"), b"", WriteMode::Replace)
        .unwrap();

    let mock = MemorySystem::new().with_dir("/work").unwrap();
    mock.write(Path::new("/work/known.txt"), b"12345", WriteMode::Replace)
        .unwrap();
    mock.write(Path::new("/work/empty.txt"), b"", WriteMode::Replace)
        .unwrap();

    let real_known = real.metadata(&tmp.path().join("known.txt")).unwrap();
    let mock_known = mock.metadata(Path::new("/work/known.txt")).unwrap();
    assert_eq!(real_known.len, mock_known.len);
    assert_eq!(real_known.is_file, mock_known.is_file);
    assert_eq!(real_known.is_dir, mock_known.is_dir);

    assert_eq!(
        real.metadata(&tmp.path().join("empty.txt")).unwrap().len,
        mock.metadata(Path::new("/work/empty.txt")).unwrap().len
    );
}

#[test]
fn parity_copy_returns_len_and_contents() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    real.write(&tmp.path().join("src.txt"), b"payload", WriteMode::Replace)
        .unwrap();
    let real_len = real
        .copy(&tmp.path().join("src.txt"), &tmp.path().join("dest.txt"))
        .unwrap();

    let mock = MemorySystem::new().with_dir("/work").unwrap();
    mock.write(Path::new("/work/src.txt"), b"payload", WriteMode::Replace)
        .unwrap();
    let mock_len = mock
        .copy(Path::new("/work/src.txt"), Path::new("/work/dest.txt"))
        .unwrap();

    assert_eq!(real_len, mock_len);
    assert_eq!(
        real.read_to_string(&tmp.path().join("dest.txt")).unwrap(),
        mock.read_to_string(Path::new("/work/dest.txt")).unwrap()
    );
}

#[test]
fn parity_exists_is_dir_is_file() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    populate(&real, tmp.path());

    let mock = MemorySystem::new();
    populate(&mock, Path::new("/work"));

    let real_file = tmp.path().join("alpha.txt");
    let mock_file = Path::new("/work/alpha.txt");
    assert_eq!(
        real.is_file(&real_file).unwrap(),
        mock.is_file(mock_file).unwrap()
    );
    assert_eq!(
        real.is_dir(&real_file).unwrap(),
        mock.is_dir(mock_file).unwrap()
    );

    let real_dir = tmp.path().join("sub");
    let mock_dir = Path::new("/work/sub");
    assert_eq!(
        real.is_dir(&real_dir).unwrap(),
        mock.is_dir(mock_dir).unwrap()
    );
    assert_eq!(
        real.is_file(&real_dir).unwrap(),
        mock.is_file(mock_dir).unwrap()
    );

    let real_missing = tmp.path().join("nope");
    let mock_missing = Path::new("/work/nope");
    assert_eq!(
        real.exists(&real_missing).unwrap(),
        mock.exists(mock_missing).unwrap()
    );
}

#[test]
fn parity_read_missing_file_is_not_found() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    let mock = MemorySystem::new().with_dir("/work").unwrap();

    assert_both_err_kind(
        real.read_to_string(&tmp.path().join("missing.txt")),
        mock.read_to_string(Path::new("/work/missing.txt")),
        io::ErrorKind::NotFound,
    );
}

#[test]
fn parity_remove_file_missing_is_not_found() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    let mock = MemorySystem::new().with_dir("/work").unwrap();

    assert_both_err_kind(
        real.remove_file(&tmp.path().join("missing.txt")),
        mock.remove_file(Path::new("/work/missing.txt")),
        io::ErrorKind::NotFound,
    );
}

#[test]
fn parity_rename_missing_source_is_not_found() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    let mock = MemorySystem::new().with_dir("/work").unwrap();

    assert_both_err_kind(
        real.rename(&tmp.path().join("a"), &tmp.path().join("b")),
        mock.rename(Path::new("/work/a"), Path::new("/work/b")),
        io::ErrorKind::NotFound,
    );
}

#[test]
fn parity_copy_missing_parent_is_not_found() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    real.write(&tmp.path().join("src.txt"), b"x", WriteMode::Replace)
        .unwrap();

    let mock = MemorySystem::new()
        .with_file("/work/src.txt", b"x")
        .unwrap();

    assert_both_err_kind(
        real.copy(
            &tmp.path().join("src.txt"),
            &tmp.path().join("no").join("dir").join("dest.txt"),
        ),
        mock.copy(
            Path::new("/work/src.txt"),
            Path::new("/work/no/dir/dest.txt"),
        ),
        io::ErrorKind::NotFound,
    );
}

#[test]
fn parity_rename_over_existing_overwrites() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    real.write(&tmp.path().join("src.txt"), b"new", WriteMode::Replace)
        .unwrap();
    real.write(&tmp.path().join("dest.txt"), b"old", WriteMode::Replace)
        .unwrap();
    real.rename(&tmp.path().join("src.txt"), &tmp.path().join("dest.txt"))
        .unwrap();

    let mock = MemorySystem::new().with_dir("/work").unwrap();
    mock.write(Path::new("/work/src.txt"), b"new", WriteMode::Replace)
        .unwrap();
    mock.write(Path::new("/work/dest.txt"), b"old", WriteMode::Replace)
        .unwrap();
    mock.rename(Path::new("/work/src.txt"), Path::new("/work/dest.txt"))
        .unwrap();

    assert_eq!(
        real.read_to_string(&tmp.path().join("dest.txt")).unwrap(),
        mock.read_to_string(Path::new("/work/dest.txt")).unwrap()
    );
    assert_eq!(
        real.exists(&tmp.path().join("src.txt")).unwrap(),
        mock.exists(Path::new("/work/src.txt")).unwrap()
    );
}

#[test]
fn parity_read_dir_lists_same_children() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    populate(&real, tmp.path());

    let mock = MemorySystem::new();
    populate(&mock, Path::new("/work"));

    let real_names = relative_names(&real.read_dir(tmp.path()).unwrap(), tmp.path());
    let mock_names = relative_names(
        &mock.read_dir(Path::new("/work")).unwrap(),
        Path::new("/work"),
    );

    assert_eq!(real_names, mock_names);
    assert_eq!(real_names, vec!["alpha.txt", "empty.txt", "sub"]);
}

#[test]
fn parity_open_append_appends() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    real.write(&tmp.path().join("log.txt"), b"head", WriteMode::Replace)
        .unwrap();
    let mut real_writer = real.open_append(&tmp.path().join("log.txt")).unwrap();
    real_writer.write_all(b"-tail").unwrap();
    real_writer.flush().unwrap();
    drop(real_writer);

    let mock = MemorySystem::new()
        .with_file("/work/log.txt", b"head")
        .unwrap();
    let mut mock_writer = mock.open_append(Path::new("/work/log.txt")).unwrap();
    mock_writer.write_all(b"-tail").unwrap();
    mock_writer.flush().unwrap();
    drop(mock_writer);

    assert_eq!(
        real.read_to_string(&tmp.path().join("log.txt")).unwrap(),
        mock.read_to_string(Path::new("/work/log.txt")).unwrap()
    );
}

#[test]
fn parity_open_reads_same_contents() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    real.write(
        &tmp.path().join("file.txt"),
        b"shared bytes",
        WriteMode::Replace,
    )
    .unwrap();

    let mock = MemorySystem::new()
        .with_file("/work/file.txt", b"shared bytes")
        .unwrap();

    let mut real_contents = String::new();
    real.open(&tmp.path().join("file.txt"))
        .unwrap()
        .read_to_string(&mut real_contents)
        .unwrap();
    let mut mock_contents = String::new();
    mock.open(Path::new("/work/file.txt"))
        .unwrap()
        .read_to_string(&mut mock_contents)
        .unwrap();

    assert_eq!(real_contents, mock_contents);
}

#[test]
fn parity_write_truncates() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    real.write(
        &tmp.path().join("file.txt"),
        b"long original",
        WriteMode::Replace,
    )
    .unwrap();
    real.write(&tmp.path().join("file.txt"), b"short", WriteMode::Replace)
        .unwrap();

    let mock = MemorySystem::new().with_dir("/work").unwrap();
    mock.write(
        Path::new("/work/file.txt"),
        b"long original",
        WriteMode::Replace,
    )
    .unwrap();
    mock.write(Path::new("/work/file.txt"), b"short", WriteMode::Replace)
        .unwrap();

    assert_eq!(
        real.read_to_string(&tmp.path().join("file.txt")).unwrap(),
        mock.read_to_string(Path::new("/work/file.txt")).unwrap()
    );
}

#[test]
fn parity_write_create_new_writes_missing_file() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    let real_file = tmp.path().join("file.lock");
    real.write(&real_file, b"payload", WriteMode::CreateNew)
        .unwrap();

    let mock = MemorySystem::new().with_dir("/work").unwrap();
    let mock_file = Path::new("/work/file.lock");
    mock.write(mock_file, b"payload", WriteMode::CreateNew)
        .unwrap();

    assert_eq!(real.read_to_string(&real_file).unwrap(), "payload");
    assert_eq!(mock.read_to_string(mock_file).unwrap(), "payload");
}

#[test]
fn parity_write_create_new_over_file_is_already_exists() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    let real_file = tmp.path().join("file.lock");
    real.write(&real_file, b"first", WriteMode::Replace)
        .unwrap();

    let mock = MemorySystem::new().with_dir("/work").unwrap();
    let mock_file = Path::new("/work/file.lock");
    mock.write(mock_file, b"first", WriteMode::Replace).unwrap();

    assert_both_err_kind(
        real.write(&real_file, b"second", WriteMode::CreateNew),
        mock.write(mock_file, b"second", WriteMode::CreateNew),
        io::ErrorKind::AlreadyExists,
    );
    assert_eq!(real.read_to_string(&real_file).unwrap(), "first");
    assert_eq!(mock.read_to_string(mock_file).unwrap(), "first");
}

#[test]
fn parity_write_create_new_over_dir_is_already_exists() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    let real_dir = tmp.path().join("sub");
    real.create_dir_all(&real_dir).unwrap();

    let mock = MemorySystem::new().with_dir("/work/sub").unwrap();
    let mock_dir = Path::new("/work/sub");

    assert_both_err_kind(
        real.write(&real_dir, b"x", WriteMode::CreateNew),
        mock.write(mock_dir, b"x", WriteMode::CreateNew),
        io::ErrorKind::AlreadyExists,
    );
    assert!(real.is_dir(&real_dir).unwrap());
    assert!(mock.is_dir(mock_dir).unwrap());
}

#[test]
fn parity_write_replace_over_dir_is_a_directory() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    let real_dir = tmp.path().join("sub");
    real.create_dir_all(&real_dir).unwrap();

    let mock = MemorySystem::new().with_dir("/work/sub").unwrap();
    let mock_dir = Path::new("/work/sub");

    assert_both_err_kind(
        real.write(&real_dir, b"x", WriteMode::Replace),
        mock.write(mock_dir, b"x", WriteMode::Replace),
        io::ErrorKind::IsADirectory,
    );
    assert!(real.is_dir(&real_dir).unwrap());
    assert!(mock.is_dir(mock_dir).unwrap());
    assert!(!mock.is_file(mock_dir).unwrap());
}

#[test]
fn parity_copy_onto_dir_is_a_directory() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    let real_dir = tmp.path().join("sub");
    real.create_dir_all(&real_dir).unwrap();
    real.write(&tmp.path().join("src.txt"), b"x", WriteMode::Replace)
        .unwrap();

    let mock = MemorySystem::new().with_dir("/work/sub").unwrap();
    let mock_dir = Path::new("/work/sub");
    mock.write(Path::new("/work/src.txt"), b"x", WriteMode::Replace)
        .unwrap();

    assert_both_err_kind(
        real.copy(&tmp.path().join("src.txt"), &real_dir),
        mock.copy(Path::new("/work/src.txt"), mock_dir),
        io::ErrorKind::IsADirectory,
    );
    assert!(real.is_dir(&real_dir).unwrap());
    assert!(mock.is_dir(mock_dir).unwrap());
    assert!(!mock.is_file(mock_dir).unwrap());
}

#[test]
fn parity_write_create_new_missing_parent_is_not_found() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    let mock = MemorySystem::new().with_dir("/work").unwrap();

    assert_both_err_kind(
        real.write(
            &tmp.path().join("no").join("file.lock"),
            b"x",
            WriteMode::CreateNew,
        ),
        mock.write(Path::new("/work/no/file.lock"), b"x", WriteMode::CreateNew),
        io::ErrorKind::NotFound,
    );
}

#[test]
fn parity_path_below_a_file_is_not_a_directory() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    let mock = MemorySystem::new().with_dir("/work").unwrap();
    real.write(&tmp.path().join("a.md"), b"a", WriteMode::Replace)
        .unwrap();
    mock.write(Path::new("/work/a.md"), b"a", WriteMode::Replace)
        .unwrap();

    assert_both_err_kind(
        real.read_to_string(&tmp.path().join("a.md/b.md")),
        mock.read_to_string(Path::new("/work/a.md/b.md")),
        io::ErrorKind::NotADirectory,
    );
    assert_both_err_kind(
        real.write(&tmp.path().join("a.md/b.md"), b"b", WriteMode::Replace),
        mock.write(Path::new("/work/a.md/b.md"), b"b", WriteMode::Replace),
        io::ErrorKind::NotADirectory,
    );
    assert_both_err_kind(
        real.remove_file(&tmp.path().join("a.md/b.md")),
        mock.remove_file(Path::new("/work/a.md/b.md")),
        io::ErrorKind::NotADirectory,
    );
    assert_both_err_kind(
        real.create_dir_all(&tmp.path().join("a.md/b")),
        mock.create_dir_all(Path::new("/work/a.md/b")),
        io::ErrorKind::NotADirectory,
    );
    assert_both_err_kind(
        real.open(&tmp.path().join("a.md/b.md")),
        mock.open(Path::new("/work/a.md/b.md")),
        io::ErrorKind::NotADirectory,
    );
    assert_both_err_kind(
        real.metadata(&tmp.path().join("a.md/b.md")),
        mock.metadata(Path::new("/work/a.md/b.md")),
        io::ErrorKind::NotADirectory,
    );
    assert_both_err_kind(
        real.create(&tmp.path().join("a.md/b.md")),
        mock.create(Path::new("/work/a.md/b.md")),
        io::ErrorKind::NotADirectory,
    );
    assert_both_err_kind(
        real.open_append(&tmp.path().join("a.md/b.md")),
        mock.open_append(Path::new("/work/a.md/b.md")),
        io::ErrorKind::NotADirectory,
    );
    assert_both_err_kind(
        real.read_dir(&tmp.path().join("a.md/b")),
        mock.read_dir(Path::new("/work/a.md/b")),
        io::ErrorKind::NotADirectory,
    );
    assert_both_err_kind(
        real.remove_dir_all(&tmp.path().join("a.md/b")),
        mock.remove_dir_all(Path::new("/work/a.md/b")),
        io::ErrorKind::NotADirectory,
    );
    assert_both_err_kind(
        real.rename(&tmp.path().join("a.md/b.md"), &tmp.path().join("c.md")),
        mock.rename(Path::new("/work/a.md/b.md"), Path::new("/work/c.md")),
        io::ErrorKind::NotADirectory,
    );
    assert_both_err_kind(
        real.rename(&tmp.path().join("a.md"), &tmp.path().join("a.md/b.md")),
        mock.rename(Path::new("/work/a.md"), Path::new("/work/a.md/b.md")),
        io::ErrorKind::NotADirectory,
    );
    assert_both_err_kind(
        real.copy(&tmp.path().join("a.md/b.md"), &tmp.path().join("c.md")),
        mock.copy(Path::new("/work/a.md/b.md"), Path::new("/work/c.md")),
        io::ErrorKind::NotADirectory,
    );
    assert_both_err_kind(
        real.copy(&tmp.path().join("a.md"), &tmp.path().join("a.md/b.md")),
        mock.copy(Path::new("/work/a.md"), Path::new("/work/a.md/b.md")),
        io::ErrorKind::NotADirectory,
    );
    assert_eq!(mock.read_to_string(Path::new("/work/a.md")).unwrap(), "a");
    assert!(!mock.is_dir(Path::new("/work/a.md")).unwrap());
    assert!(!mock.exists(Path::new("/work/a.md/b.md")).unwrap());
}

#[test]
fn parity_create_dir_all_over_a_file_already_exists() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    let mock = MemorySystem::new().with_dir("/work").unwrap();
    real.write(&tmp.path().join("a.md"), b"a", WriteMode::Replace)
        .unwrap();
    mock.write(Path::new("/work/a.md"), b"a", WriteMode::Replace)
        .unwrap();

    assert_both_err_kind(
        real.create_dir_all(&tmp.path().join("a.md")),
        mock.create_dir_all(Path::new("/work/a.md")),
        io::ErrorKind::AlreadyExists,
    );
    assert!(!mock.is_dir(Path::new("/work/a.md")).unwrap());
}

#[test]
fn parity_create_dir_all_idempotent_nested() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    let real_nested = tmp.path().join("a").join("b").join("c");
    real.create_dir_all(&real_nested).unwrap();
    real.create_dir_all(&real_nested).unwrap();

    let mock = MemorySystem::new();
    let mock_nested = Path::new("/work/a/b/c");
    mock.create_dir_all(mock_nested).unwrap();
    mock.create_dir_all(mock_nested).unwrap();

    assert_eq!(
        real.is_dir(&real_nested).unwrap(),
        mock.is_dir(mock_nested).unwrap()
    );
}

#[test]
fn parity_env_var_roundtrip_and_missing() {
    let real = RealSystem::new();
    let mock = MemorySystem::new();
    let key = "OS_SHIM_PARITY_ENV_KEY";

    real.set_env_var(key, "value");
    mock.set_env_var(key, "value");
    assert_eq!(real.env_var(key).unwrap(), mock.env_var(key).unwrap());

    assert_eq!(
        real.env_var("OS_SHIM_PARITY_DEFINITELY_UNSET"),
        Err::<String, VarError>(VarError::NotPresent)
    );
    assert_eq!(
        mock.env_var("OS_SHIM_PARITY_DEFINITELY_UNSET"),
        Err::<String, VarError>(VarError::NotPresent)
    );
}

#[test]
fn parity_walk_dir_visible_tree() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    populate(&real, tmp.path());

    let mock = MemorySystem::new();
    populate(&mock, Path::new("/work"));

    let real_walk: Vec<PathBuf> = real
        .walk_dir(tmp.path(), false, false)
        .unwrap()
        .into_iter()
        .map(|entry| entry.path)
        .collect();
    let mock_walk: Vec<PathBuf> = mock
        .walk_dir(Path::new("/work"), false, false)
        .unwrap()
        .into_iter()
        .map(|entry| entry.path)
        .collect();

    let real_names = relative_names(&real_walk, tmp.path());
    let mock_names = relative_names(&mock_walk, Path::new("/work"));

    assert_eq!(real_names, mock_names);
    assert_eq!(
        real_names,
        vec!["alpha.txt", "empty.txt", "sub", "sub/beta.txt"]
    );
}

#[test]
fn parity_create_writes_via_returned_stream() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    let real_file = tmp.path().join("made.txt");
    let mut real_writer = real.create(&real_file).unwrap();
    real_writer.write_all(b"streamed").unwrap();
    real_writer.flush().unwrap();
    drop(real_writer);

    let mock = MemorySystem::new().with_dir("/work").unwrap();
    let mock_file = Path::new("/work/made.txt");
    let mut mock_writer = mock.create(mock_file).unwrap();
    mock_writer.write_all(b"streamed").unwrap();
    mock_writer.flush().unwrap();
    drop(mock_writer);

    assert_eq!(
        real.read_to_string(&real_file).unwrap(),
        mock.read_to_string(mock_file).unwrap()
    );
}

#[test]
fn parity_copy_missing_source_is_not_found() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    let mock = MemorySystem::new().with_dir("/work").unwrap();

    assert_both_err_kind(
        real.copy(
            &tmp.path().join("missing.txt"),
            &tmp.path().join("dest.txt"),
        ),
        mock.copy(Path::new("/work/missing.txt"), Path::new("/work/dest.txt")),
        io::ErrorKind::NotFound,
    );
}

#[test]
fn parity_open_missing_file_is_not_found() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    let mock = MemorySystem::new().with_dir("/work").unwrap();

    assert_both_err_kind(
        real.open(&tmp.path().join("missing.txt")),
        mock.open(Path::new("/work/missing.txt")),
        io::ErrorKind::NotFound,
    );
}

#[test]
fn parity_read_dir_missing_is_not_found() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    let mock = MemorySystem::new().with_dir("/work").unwrap();

    assert_both_err_kind(
        real.read_dir(&tmp.path().join("missing")),
        mock.read_dir(Path::new("/work/missing")),
        io::ErrorKind::NotFound,
    );
}

#[test]
fn parity_rename_dest_parent_missing_is_not_found() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    real.write(&tmp.path().join("src.txt"), b"x", WriteMode::Replace)
        .unwrap();

    let mock = MemorySystem::new()
        .with_file("/work/src.txt", b"x")
        .unwrap();

    assert_both_err_kind(
        real.rename(
            &tmp.path().join("src.txt"),
            &tmp.path().join("no").join("dest.txt"),
        ),
        mock.rename(Path::new("/work/src.txt"), Path::new("/work/no/dest.txt")),
        io::ErrorKind::NotFound,
    );
    assert_eq!(
        real.read_to_string(&tmp.path().join("src.txt")).unwrap(),
        "x"
    );
    assert_eq!(
        mock.read_to_string(Path::new("/work/src.txt")).unwrap(),
        "x"
    );
}

#[test]
fn parity_rename_file_onto_dir_is_a_directory() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    real.write(&tmp.path().join("a.md"), b"a", WriteMode::Replace)
        .unwrap();
    real.create_dir_all(&tmp.path().join("d")).unwrap();

    let mock = MemorySystem::new()
        .with_file("/work/a.md", b"a")
        .unwrap()
        .with_dir("/work/d")
        .unwrap();

    assert_both_err_kind(
        real.rename(&tmp.path().join("a.md"), &tmp.path().join("d")),
        mock.rename(Path::new("/work/a.md"), Path::new("/work/d")),
        io::ErrorKind::IsADirectory,
    );
    assert!(real.is_file(&tmp.path().join("a.md")).unwrap());
    assert!(mock.is_file(Path::new("/work/a.md")).unwrap());
    assert!(real.is_dir(&tmp.path().join("d")).unwrap());
    assert!(mock.is_dir(Path::new("/work/d")).unwrap());
    assert!(!real.is_file(&tmp.path().join("d")).unwrap());
    assert!(!mock.is_file(Path::new("/work/d")).unwrap());
}

#[test]
fn parity_rename_dir_onto_file_is_not_a_directory() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    real.write(&tmp.path().join("a.md"), b"a", WriteMode::Replace)
        .unwrap();
    real.create_dir_all(&tmp.path().join("d")).unwrap();

    let mock = MemorySystem::new()
        .with_file("/work/a.md", b"a")
        .unwrap()
        .with_dir("/work/d")
        .unwrap();

    assert_both_err_kind(
        real.rename(&tmp.path().join("d"), &tmp.path().join("a.md")),
        mock.rename(Path::new("/work/d"), Path::new("/work/a.md")),
        io::ErrorKind::NotADirectory,
    );
    assert!(real.is_file(&tmp.path().join("a.md")).unwrap());
    assert!(mock.is_file(Path::new("/work/a.md")).unwrap());
    assert!(!real.is_dir(&tmp.path().join("a.md")).unwrap());
    assert!(!mock.is_dir(Path::new("/work/a.md")).unwrap());
    assert!(real.is_dir(&tmp.path().join("d")).unwrap());
    assert!(mock.is_dir(Path::new("/work/d")).unwrap());
}

#[test]
fn parity_rename_dir_to_missing_parent_is_not_found() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    real.create_dir_all(&tmp.path().join("d")).unwrap();

    let mock = MemorySystem::new().with_dir("/work/d").unwrap();

    assert_both_err_kind(
        real.rename(&tmp.path().join("d"), &tmp.path().join("nope").join("d3")),
        mock.rename(Path::new("/work/d"), Path::new("/work/nope/d3")),
        io::ErrorKind::NotFound,
    );
    assert!(real.exists(&tmp.path().join("d")).unwrap());
    assert!(mock.exists(Path::new("/work/d")).unwrap());
    assert!(!real.exists(&tmp.path().join("nope").join("d3")).unwrap());
    assert!(!mock.exists(Path::new("/work/nope/d3")).unwrap());
}

#[test]
fn parity_open_append_parent_missing_is_not_found() {
    // Real surfaces the missing parent on open; Memory surfaces it on flush. Both
    // fail with NotFound before any bytes land -- that is the observable parity.
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    let real_result = real
        .open_append(&tmp.path().join("no").join("log.txt"))
        .and_then(|mut writer| {
            writer.write_all(b"x")?;
            writer.flush()
        });

    let mock = MemorySystem::new().with_dir("/work").unwrap();
    let mock_result = mock
        .open_append(Path::new("/work/no/log.txt"))
        .and_then(|mut writer| {
            writer.write_all(b"x")?;
            writer.flush()
        });

    assert_both_err_kind(real_result, mock_result, io::ErrorKind::NotFound);
}

#[test]
fn parity_remove_file_success_then_missing() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    real.write(&tmp.path().join("doomed.txt"), b"x", WriteMode::Replace)
        .unwrap();
    real.remove_file(&tmp.path().join("doomed.txt")).unwrap();

    let mock = MemorySystem::new()
        .with_file("/work/doomed.txt", b"x")
        .unwrap();
    mock.remove_file(Path::new("/work/doomed.txt")).unwrap();

    assert_eq!(
        real.exists(&tmp.path().join("doomed.txt")).unwrap(),
        mock.exists(Path::new("/work/doomed.txt")).unwrap()
    );
    assert_both_err_kind(
        real.remove_file(&tmp.path().join("doomed.txt")),
        mock.remove_file(Path::new("/work/doomed.txt")),
        io::ErrorKind::NotFound,
    );
}

#[test]
fn parity_rename_directory_moves_subtree() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    real.create_dir_all(&tmp.path().join("from").join("nested"))
        .unwrap();
    real.write(
        &tmp.path().join("from").join("a.txt"),
        b"a",
        WriteMode::Replace,
    )
    .unwrap();
    real.write(
        &tmp.path().join("from").join("nested").join("b.txt"),
        b"b",
        WriteMode::Replace,
    )
    .unwrap();
    real.rename(&tmp.path().join("from"), &tmp.path().join("to"))
        .unwrap();

    let mock = MemorySystem::new();
    mock.create_dir_all(Path::new("/work/from/nested")).unwrap();
    mock.write(Path::new("/work/from/a.txt"), b"a", WriteMode::Replace)
        .unwrap();
    mock.write(
        Path::new("/work/from/nested/b.txt"),
        b"b",
        WriteMode::Replace,
    )
    .unwrap();
    mock.rename(Path::new("/work/from"), Path::new("/work/to"))
        .unwrap();

    // Old tree is gone, new tree carries the same relative contents on both.
    assert_eq!(
        real.exists(&tmp.path().join("from")).unwrap(),
        mock.exists(Path::new("/work/from")).unwrap()
    );
    let real_walk: Vec<PathBuf> = real
        .walk_dir(&tmp.path().join("to"), false, false)
        .unwrap()
        .into_iter()
        .map(|entry| entry.path)
        .collect();
    let mock_walk: Vec<PathBuf> = mock
        .walk_dir(Path::new("/work/to"), false, false)
        .unwrap()
        .into_iter()
        .map(|entry| entry.path)
        .collect();
    assert_eq!(
        relative_names(&real_walk, &tmp.path().join("to")),
        relative_names(&mock_walk, Path::new("/work/to"))
    );
    assert_eq!(
        real.read_to_string(&tmp.path().join("to").join("nested").join("b.txt"))
            .unwrap(),
        mock.read_to_string(Path::new("/work/to/nested/b.txt"))
            .unwrap()
    );
}

/// `remove_dir_all` on a missing path: both return `NotFound` (converged; was a
/// divergence where Memory silently returned Ok -- shim-cyh).
#[test]
fn parity_remove_dir_all_missing_is_not_found() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    let mock = MemorySystem::new().with_dir("/work").unwrap();

    assert_both_err_kind(
        real.remove_dir_all(&tmp.path().join("missing")),
        mock.remove_dir_all(Path::new("/work/missing")),
        io::ErrorKind::NotFound,
    );
}

/// `remove_dir_all` on a populated tree evicts the root and every descendant on
/// both impls. The missing-path case only exercises the early `NotFound` return;
/// this pins the actual subtree-removal path where Memory could silently diverge.
#[test]
fn parity_remove_dir_all_removes_populated_tree() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    let real_root = tmp.path().join("tree");
    populate(&real, &real_root);

    let mock = MemorySystem::new();
    let mock_root = Path::new("/work/tree");
    populate(&mock, mock_root);

    real.remove_dir_all(&real_root).unwrap();
    mock.remove_dir_all(mock_root).unwrap();

    assert!(!real.exists(&real_root).unwrap());
    assert!(!mock.exists(mock_root).unwrap());
    for relative in ["alpha.txt", "empty.txt", "sub", "sub/beta.txt"] {
        assert_eq!(
            real.exists(&real_root.join(relative)).unwrap(),
            mock.exists(&mock_root.join(relative)).unwrap(),
            "mock/real disagree on {relative:?} after remove_dir_all"
        );
        assert!(!mock.exists(&mock_root.join(relative)).unwrap());
    }
}

/// `walk_dir` on a missing path: both yield an empty list (converged; Memory used
/// to return `NotFound` -- shim-977).
#[test]
fn parity_walk_dir_missing_yields_empty() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    let mock = MemorySystem::new().with_dir("/work").unwrap();

    assert!(
        real.walk_dir(&tmp.path().join("missing"), false, false)
            .unwrap()
            .is_empty()
    );
    assert!(
        mock.walk_dir(Path::new("/work/missing"), false, false)
            .unwrap()
            .is_empty()
    );
}

/// The `hidden` flag honors its documented meaning on both impls: `true`
/// includes dotfiles, `false` excludes them (converged; Memory used to ignore the
/// flag and both impls used the inverse semantics -- shim-977).
#[test]
fn parity_walk_dir_hidden_flag_includes_dotfiles() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    real.write(&tmp.path().join("visible.txt"), b"v", WriteMode::Replace)
        .unwrap();
    real.write(&tmp.path().join(".secret"), b"s", WriteMode::Replace)
        .unwrap();

    let mock = MemorySystem::new().with_dir("/work").unwrap();
    mock.write(Path::new("/work/visible.txt"), b"v", WriteMode::Replace)
        .unwrap();
    mock.write(Path::new("/work/.secret"), b"s", WriteMode::Replace)
        .unwrap();

    let walk_names = |hidden: bool| {
        let real_names = relative_names(
            &real
                .walk_dir(tmp.path(), false, hidden)
                .unwrap()
                .into_iter()
                .map(|entry| entry.path)
                .collect::<Vec<_>>(),
            tmp.path(),
        );
        let mock_names = relative_names(
            &mock
                .walk_dir(Path::new("/work"), false, hidden)
                .unwrap()
                .into_iter()
                .map(|entry| entry.path)
                .collect::<Vec<_>>(),
            Path::new("/work"),
        );
        assert_eq!(real_names, mock_names);
        real_names
    };

    // hidden = true includes the dotfile; hidden = false excludes it.
    assert_eq!(walk_names(true), vec![".secret", "visible.txt"]);
    assert_eq!(walk_names(false), vec!["visible.txt"]);
}

/// `follow_links` is a faithful no-op on the mock (no symlink modeling). On a
/// symlink-free tree both impls yield identical entries for either flag value,
/// so mock and real agree -- shim-977.
#[test]
fn parity_walk_dir_follow_links_agrees_without_symlinks() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    populate(&real, tmp.path());

    let mock = MemorySystem::new();
    populate(&mock, Path::new("/work"));

    let walk_names = |follow_links: bool| {
        let real_names = relative_names(
            &real
                .walk_dir(tmp.path(), follow_links, false)
                .unwrap()
                .into_iter()
                .map(|entry| entry.path)
                .collect::<Vec<_>>(),
            tmp.path(),
        );
        let mock_names = relative_names(
            &mock
                .walk_dir(Path::new("/work"), follow_links, false)
                .unwrap()
                .into_iter()
                .map(|entry| entry.path)
                .collect::<Vec<_>>(),
            Path::new("/work"),
        );
        assert_eq!(real_names, mock_names);
        real_names
    };

    // Both flag values yield the same tree on both impls.
    assert_eq!(walk_names(true), walk_names(false));
    assert_eq!(
        walk_names(false),
        vec!["alpha.txt", "empty.txt", "sub", "sub/beta.txt"]
    );
}
