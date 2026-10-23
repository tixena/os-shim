//! Memory<->Real parity harness.
//!
//! These tests run the SAME operation sequences against `MemorySystem` and
//! `RealSystem` and assert they agree on observable results (return values,
//! `io::ErrorKind` on failure, `metadata().len`, directory contents). A mock
//! that silently diverges from real is the dangerous failure mode -- green
//! tests, broken production -- so this harness is the highest-value guard.

use crate::mock::MemorySystem;
use crate::real::RealSystem;
use crate::{System, WriteMode};
use core::time::Duration;
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
fn parity_create_over_dir_is_a_directory() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    let real_dir = tmp.path().join("sub");
    real.create_dir_all(&real_dir).unwrap();

    let mock = MemorySystem::new().with_dir("/work/sub").unwrap();
    let mock_dir = Path::new("/work/sub");

    assert_both_err_kind(
        real.create(&real_dir),
        mock.create(mock_dir),
        io::ErrorKind::IsADirectory,
    );
    assert!(real.is_dir(&real_dir).unwrap());
    assert!(mock.is_dir(mock_dir).unwrap());
    assert!(!mock.is_file(mock_dir).unwrap());
}

#[test]
fn parity_open_append_over_dir_is_a_directory() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    let real_dir = tmp.path().join("sub");
    real.create_dir_all(&real_dir).unwrap();

    let mock = MemorySystem::new().with_dir("/work/sub").unwrap();
    let mock_dir = Path::new("/work/sub");

    assert_both_err_kind(
        real.open_append(&real_dir),
        mock.open_append(mock_dir),
        io::ErrorKind::IsADirectory,
    );
    assert!(real.is_dir(&real_dir).unwrap());
    assert!(mock.is_dir(mock_dir).unwrap());
    assert!(!mock.is_file(mock_dir).unwrap());
}

#[test]
fn parity_copy_from_a_dir_is_invalid_input() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    let real_dir = tmp.path().join("d");
    let real_to = tmp.path().join("c.md");
    real.create_dir_all(&real_dir).unwrap();

    let mock = MemorySystem::new().with_dir("/work/d").unwrap();
    let mock_dir = Path::new("/work/d");
    let mock_to = Path::new("/work/c.md");

    assert_both_err_kind(
        real.copy(&real_dir, &real_to),
        mock.copy(mock_dir, mock_to),
        io::ErrorKind::InvalidInput,
    );
    assert!(real.is_dir(&real_dir).unwrap());
    assert!(mock.is_dir(mock_dir).unwrap());
    assert!(!real.exists(&real_to).unwrap());
    assert!(!mock.exists(mock_to).unwrap());
}

#[test]
fn parity_create_makes_the_file_when_opened() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    let real_path = tmp.path().join("new.md");
    let mock = MemorySystem::new().with_dir("/work").unwrap();
    let mock_path = Path::new("/work/new.md");

    let real_writer = real.create(&real_path).unwrap();
    let mock_writer = mock.create(mock_path).unwrap();

    assert!(real.exists(&real_path).unwrap());
    assert!(mock.exists(mock_path).unwrap());
    assert_both_err_kind(
        real.create_dir_all(&real_path),
        mock.create_dir_all(mock_path),
        io::ErrorKind::AlreadyExists,
    );
    drop(real_writer);
    drop(mock_writer);
}

#[test]
fn parity_create_empties_an_existing_file_when_opened() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    let real_path = tmp.path().join("old.md");
    real.write(&real_path, b"old", WriteMode::Replace).unwrap();
    let mock = MemorySystem::new().with_dir("/work").unwrap();
    let mock_path = Path::new("/work/old.md");
    mock.write(mock_path, b"old", WriteMode::Replace).unwrap();

    let real_writer = real.create(&real_path).unwrap();
    let mock_writer = mock.create(mock_path).unwrap();

    assert_eq!(real.read_to_string(&real_path).unwrap(), "");
    assert_eq!(mock.read_to_string(mock_path).unwrap(), "");
    drop(real_writer);
    drop(mock_writer);
}

#[test]
fn parity_open_append_makes_the_file_when_opened() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    let real_path = tmp.path().join("log.md");
    let mock = MemorySystem::new().with_dir("/work").unwrap();
    let mock_path = Path::new("/work/log.md");

    let real_writer = real.open_append(&real_path).unwrap();
    let mock_writer = mock.open_append(mock_path).unwrap();

    assert!(real.exists(&real_path).unwrap());
    assert!(mock.exists(mock_path).unwrap());
    assert_eq!(real.read_to_string(&real_path).unwrap(), "");
    assert_eq!(mock.read_to_string(mock_path).unwrap(), "");
    assert_both_err_kind(
        real.create_dir_all(&real_path),
        mock.create_dir_all(mock_path),
        io::ErrorKind::AlreadyExists,
    );
    drop(real_writer);
    drop(mock_writer);
}

#[test]
fn parity_open_append_keeps_an_existing_file_when_opened() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    let real_path = tmp.path().join("log.md");
    real.write(&real_path, b"old", WriteMode::Replace).unwrap();
    let mock = MemorySystem::new().with_dir("/work").unwrap();
    let mock_path = Path::new("/work/log.md");
    mock.write(mock_path, b"old", WriteMode::Replace).unwrap();

    let mut real_writer = real.open_append(&real_path).unwrap();
    let mut mock_writer = mock.open_append(mock_path).unwrap();

    assert_eq!(real.read_to_string(&real_path).unwrap(), "old");
    assert_eq!(mock.read_to_string(mock_path).unwrap(), "old");
    real_writer.write_all(b"+new").unwrap();
    real_writer.flush().unwrap();
    mock_writer.write_all(b"+new").unwrap();
    mock_writer.flush().unwrap();
    assert_eq!(real.read_to_string(&real_path).unwrap(), "old+new");
    assert_eq!(mock.read_to_string(mock_path).unwrap(), "old+new");
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
fn parity_read_dir_of_a_file_is_not_a_directory() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    let mock = MemorySystem::new().with_dir("/work").unwrap();
    real.write(&tmp.path().join("a.md"), b"a", WriteMode::Replace)
        .unwrap();
    mock.write(Path::new("/work/a.md"), b"a", WriteMode::Replace)
        .unwrap();

    assert_both_err_kind(
        real.read_dir(&tmp.path().join("a.md")),
        mock.read_dir(Path::new("/work/a.md")),
        io::ErrorKind::NotADirectory,
    );
    assert_eq!(real.read_to_string(&tmp.path().join("a.md")).unwrap(), "a");
    assert_eq!(mock.read_to_string(Path::new("/work/a.md")).unwrap(), "a");
    assert!(!mock.is_dir(Path::new("/work/a.md")).unwrap());
}

#[test]
fn parity_remove_dir_all_of_a_file_is_not_a_directory() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    let mock = MemorySystem::new().with_dir("/work").unwrap();
    real.write(&tmp.path().join("a.md"), b"a", WriteMode::Replace)
        .unwrap();
    mock.write(Path::new("/work/a.md"), b"a", WriteMode::Replace)
        .unwrap();

    assert_both_err_kind(
        real.remove_dir_all(&tmp.path().join("a.md")),
        mock.remove_dir_all(Path::new("/work/a.md")),
        io::ErrorKind::NotADirectory,
    );
    assert_eq!(real.read_to_string(&tmp.path().join("a.md")).unwrap(), "a");
    assert_eq!(mock.read_to_string(Path::new("/work/a.md")).unwrap(), "a");
    assert!(!mock.is_dir(Path::new("/work/a.md")).unwrap());
}

#[test]
fn parity_remove_file_of_a_dir_is_a_directory() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    let real_dir = tmp.path().join("d");
    real.create_dir_all(&real_dir).unwrap();

    let mock = MemorySystem::new().with_dir("/work/d").unwrap();
    let mock_dir = Path::new("/work/d");

    assert_both_err_kind(
        real.remove_file(&real_dir),
        mock.remove_file(mock_dir),
        io::ErrorKind::IsADirectory,
    );
    assert!(real.is_dir(&real_dir).unwrap());
    assert!(mock.is_dir(mock_dir).unwrap());
    assert!(!mock.is_file(mock_dir).unwrap());
}

#[test]
fn parity_read_to_string_of_a_dir_is_a_directory() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    let real_dir = tmp.path().join("d");
    real.create_dir_all(&real_dir).unwrap();

    let mock = MemorySystem::new().with_dir("/work/d").unwrap();
    let mock_dir = Path::new("/work/d");

    assert_both_err_kind(
        real.read_to_string(&real_dir),
        mock.read_to_string(mock_dir),
        io::ErrorKind::IsADirectory,
    );
    assert!(real.is_dir(&real_dir).unwrap());
    assert!(mock.is_dir(mock_dir).unwrap());
    assert!(!mock.is_file(mock_dir).unwrap());
}

#[test]
fn parity_open_of_a_dir_fails_on_read_with_is_a_directory() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    let real_dir = tmp.path().join("d");
    real.create_dir_all(&real_dir).unwrap();

    let mock = MemorySystem::new().with_dir("/work/d").unwrap();
    let mock_dir = Path::new("/work/d");

    let mut real_bytes = Vec::new();
    let mut mock_bytes = Vec::new();
    assert_both_err_kind(
        real.open(&real_dir).unwrap().read_to_end(&mut real_bytes),
        mock.open(mock_dir).unwrap().read_to_end(&mut mock_bytes),
        io::ErrorKind::IsADirectory,
    );
    assert!(real.is_dir(&real_dir).unwrap());
    assert!(mock.is_dir(mock_dir).unwrap());
    assert!(!mock.is_file(mock_dir).unwrap());
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
fn parity_rename_dir_onto_non_empty_dir_is_directory_not_empty() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    let root = tmp.path();
    real.create_dir_all(&root.join("d2")).unwrap();
    real.create_dir_all(&root.join("e2")).unwrap();
    real.write(&root.join("d2").join("a.md"), b"a", WriteMode::Replace)
        .unwrap();
    real.write(&root.join("e2").join("x.md"), b"x", WriteMode::Replace)
        .unwrap();

    let mock = MemorySystem::new()
        .with_file("/work/d2/a.md", b"a")
        .unwrap()
        .with_file("/work/e2/x.md", b"x")
        .unwrap();

    assert_both_err_kind(
        real.rename(&root.join("d2"), &root.join("e2")),
        mock.rename(Path::new("/work/d2"), Path::new("/work/e2")),
        io::ErrorKind::DirectoryNotEmpty,
    );
    assert!(real.is_file(&root.join("d2").join("a.md")).unwrap());
    assert!(mock.is_file(Path::new("/work/d2/a.md")).unwrap());
    assert!(real.is_file(&root.join("e2").join("x.md")).unwrap());
    assert!(mock.is_file(Path::new("/work/e2/x.md")).unwrap());
}

#[test]
fn parity_rename_dir_onto_empty_dir_replaces_it() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    let root = tmp.path();
    real.create_dir_all(&root.join("d1")).unwrap();
    real.create_dir_all(&root.join("e1")).unwrap();
    real.write(&root.join("d1").join("a.md"), b"a", WriteMode::Replace)
        .unwrap();

    let mock = MemorySystem::new()
        .with_file("/work/d1/a.md", b"a")
        .unwrap()
        .with_dir("/work/e1")
        .unwrap();

    real.rename(&root.join("d1"), &root.join("e1")).unwrap();
    mock.rename(Path::new("/work/d1"), Path::new("/work/e1"))
        .unwrap();

    assert!(real.is_file(&root.join("e1").join("a.md")).unwrap());
    assert!(mock.is_file(Path::new("/work/e1/a.md")).unwrap());
    assert!(!real.exists(&root.join("d1")).unwrap());
    assert!(!mock.exists(Path::new("/work/d1")).unwrap());
}

#[test]
fn parity_rename_dir_into_itself_is_invalid_input() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    let root = tmp.path();
    real.create_dir_all(&root.join("d3")).unwrap();
    real.write(&root.join("d3").join("f.md"), b"f", WriteMode::Replace)
        .unwrap();

    let mock = MemorySystem::new()
        .with_file("/work/d3/f.md", b"f")
        .unwrap();

    assert_both_err_kind(
        real.rename(&root.join("d3"), &root.join("d3").join("sub")),
        mock.rename(Path::new("/work/d3"), Path::new("/work/d3/sub")),
        io::ErrorKind::InvalidInput,
    );
    assert!(real.is_dir(&root.join("d3")).unwrap());
    assert!(mock.is_dir(Path::new("/work/d3")).unwrap());
    assert_eq!(
        real.read_to_string(&root.join("d3").join("f.md")).unwrap(),
        "f"
    );
    assert_eq!(
        mock.read_to_string(Path::new("/work/d3/f.md")).unwrap(),
        "f"
    );
    assert!(!real.exists(&root.join("d3").join("sub")).unwrap());
    assert!(!mock.exists(Path::new("/work/d3/sub")).unwrap());
}

#[test]
fn parity_rename_dir_to_sibling_with_shared_prefix_succeeds() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    let root = tmp.path();
    real.create_dir_all(&root.join("d3")).unwrap();
    real.write(&root.join("d3").join("f.md"), b"f", WriteMode::Replace)
        .unwrap();

    let mock = MemorySystem::new()
        .with_file("/work/d3/f.md", b"f")
        .unwrap();

    real.rename(&root.join("d3"), &root.join("d3x")).unwrap();
    mock.rename(Path::new("/work/d3"), Path::new("/work/d3x"))
        .unwrap();

    assert!(real.is_file(&root.join("d3x").join("f.md")).unwrap());
    assert!(mock.is_file(Path::new("/work/d3x/f.md")).unwrap());
    assert!(!real.exists(&root.join("d3")).unwrap());
    assert!(!mock.exists(Path::new("/work/d3")).unwrap());
}

#[test]
fn parity_rename_keeps_the_modification_time() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    let root = tmp.path();
    real.write(&root.join("m.md"), b"m", WriteMode::Replace)
        .unwrap();
    let real_before = real.metadata(&root.join("m.md")).unwrap().modified;

    let mock = MemorySystem::new().with_dir("/work").unwrap();
    mock.write(Path::new("/work/m.md"), b"m", WriteMode::Replace)
        .unwrap();
    let mock_before = mock.metadata(Path::new("/work/m.md")).unwrap().modified;
    mock.advance(Duration::from_secs(60)).unwrap();

    real.rename(&root.join("m.md"), &root.join("n.md")).unwrap();
    mock.rename(Path::new("/work/m.md"), Path::new("/work/n.md"))
        .unwrap();

    assert_eq!(
        real.metadata(&root.join("n.md")).unwrap().modified,
        real_before
    );
    assert_eq!(
        mock.metadata(Path::new("/work/n.md")).unwrap().modified,
        mock_before
    );
}

#[test]
fn parity_open_append_parent_missing_is_not_found() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    let mock = MemorySystem::new().with_dir("/work").unwrap();

    assert_both_err_kind(
        real.open_append(&tmp.path().join("no").join("log.txt")),
        mock.open_append(Path::new("/work/no/log.txt")),
        io::ErrorKind::NotFound,
    );
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
/// divergence where Memory silently returned Ok).
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
/// to return `NotFound`).
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
/// flag and both impls used the inverse semantics).
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
/// so mock and real agree.
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

#[test]
fn parity_remove_file_if_same_contents_removes() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    let real_file = tmp.path().join("lock");
    real.write(&real_file, b"a", WriteMode::CreateNew).unwrap();
    let mock = MemorySystem::new().with_dir("/work").unwrap();
    let mock_file = Path::new("/work/lock");
    mock.write(mock_file, b"a", WriteMode::CreateNew).unwrap();

    assert!(real.remove_file_if(&real_file, b"a").unwrap());
    assert!(mock.remove_file_if(mock_file, b"a").unwrap());
    assert!(!real.exists(&real_file).unwrap());
    assert!(!mock.exists(mock_file).unwrap());
}

#[test]
fn parity_remove_file_if_other_contents_keeps() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    let real_file = tmp.path().join("lock");
    real.write(&real_file, b"a", WriteMode::CreateNew).unwrap();
    let mock = MemorySystem::new().with_dir("/work").unwrap();
    let mock_file = Path::new("/work/lock");
    mock.write(mock_file, b"a", WriteMode::CreateNew).unwrap();

    assert!(!real.remove_file_if(&real_file, b"b").unwrap());
    assert!(!mock.remove_file_if(mock_file, b"b").unwrap());
    assert_eq!(real.read_to_string(&real_file).unwrap(), "a");
    assert_eq!(mock.read_to_string(mock_file).unwrap(), "a");
}

#[test]
fn parity_remove_file_if_missing_is_not_found() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    let mock = MemorySystem::new().with_dir("/work").unwrap();

    assert_both_err_kind(
        real.remove_file_if(&tmp.path().join("missing"), b"a"),
        mock.remove_file_if(Path::new("/work/missing"), b"a"),
        io::ErrorKind::NotFound,
    );
}

#[test]
fn parity_remove_file_if_missing_parent_is_not_found() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    let mock = MemorySystem::new().with_dir("/work").unwrap();

    assert_both_err_kind(
        real.remove_file_if(&tmp.path().join("missing/lock"), b"a"),
        mock.remove_file_if(Path::new("/work/missing/lock"), b"a"),
        io::ErrorKind::NotFound,
    );
}

#[test]
fn parity_remove_file_if_of_a_dir_is_a_directory() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    let real_dir = tmp.path().join("d");
    real.create_dir_all(&real_dir).unwrap();
    let mock = MemorySystem::new().with_dir("/work/d").unwrap();

    assert_both_err_kind(
        real.remove_file_if(&real_dir, b"a"),
        mock.remove_file_if(Path::new("/work/d"), b"a"),
        io::ErrorKind::IsADirectory,
    );
}

#[test]
fn parity_remove_file_if_below_a_file_is_not_a_directory() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    let mock = MemorySystem::new().with_dir("/work").unwrap();
    real.write(&tmp.path().join("a.md"), b"a", WriteMode::Replace)
        .unwrap();
    mock.write(Path::new("/work/a.md"), b"a", WriteMode::Replace)
        .unwrap();

    assert_both_err_kind(
        real.remove_file_if(&tmp.path().join("a.md/b.md"), b"a"),
        mock.remove_file_if(Path::new("/work/a.md/b.md"), b"a"),
        io::ErrorKind::NotADirectory,
    );
}
