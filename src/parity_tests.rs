//! Memory<->Real parity harness.
//!
//! These tests run the SAME operation sequences against `MemorySystem` and
//! `RealSystem` and assert they agree on observable results (return values,
//! `io::ErrorKind` on failure, `metadata().len`, directory contents). A mock
//! that silently diverges from real is the dangerous failure mode -- green
//! tests, broken production -- so this harness is the highest-value guard.

use crate::System;
use crate::bucket::{Condition, RevisionToken};
use crate::error::{ErrorKind, Result};
use crate::mock::MemorySystem;
use crate::real::RealSystem;
use crate::streams::{close, read_to_end, write_all};
use core::time::Duration;
use std::env::VarError;
use std::path::{Path, PathBuf};

/// Assert both results failed with the same `ErrorKind`.
fn assert_both_err_kind<L, R>(left: Result<L>, right: Result<R>, kind: ErrorKind) {
    assert_eq!(left.err().map(|err| err.kind()), Some(kind));
    assert_eq!(right.err().map(|err| err.kind()), Some(kind));
}

/// Create the file at `path` and close its writer.
async fn create_and_close(system: &dyn System, path: &Path) -> Result<()> {
    let mut writer = system.create(path).await?;
    Ok(close(&mut *writer).await?)
}

/// Open the file at `path` for appending and close its writer.
async fn append_and_close(system: &dyn System, path: &Path) -> Result<()> {
    let mut writer = system.open_append(path).await?;
    Ok(close(&mut *writer).await?)
}

/// Open the file at `path` and read it to the end.
async fn read_through_open(system: &dyn System, path: &Path) -> Result<Vec<u8>> {
    let mut reader = system.open(path).await?;
    Ok(read_to_end(&mut *reader).await?)
}

/// Build the same standard tree under `root` on a system.
async fn populate(system: &dyn System, root: &Path) {
    system.create_dir_all(root).await.unwrap();
    system
        .write(&root.join("alpha.txt"), b"alpha", Condition::None)
        .await
        .unwrap();
    system
        .write(&root.join("empty.txt"), b"", Condition::None)
        .await
        .unwrap();
    system.create_dir_all(&root.join("sub")).await.unwrap();
    system
        .write(&root.join("sub").join("beta.txt"), b"beta", Condition::None)
        .await
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

#[tokio::test(flavor = "current_thread")]
async fn parity_write_read_roundtrip() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    let real_file = tmp.path().join("file.txt");
    real.write(&real_file, b"payload", Condition::None)
        .await
        .unwrap();

    let mock = MemorySystem::new().with_dir("/work").unwrap();
    let mock_file = Path::new("/work/file.txt");
    mock.write(mock_file, b"payload", Condition::None)
        .await
        .unwrap();

    assert_eq!(
        real.read_to_string(&real_file).await.unwrap(),
        mock.read_to_string(mock_file).await.unwrap()
    );
}

#[tokio::test(flavor = "current_thread")]
async fn parity_metadata_len_for_known_and_empty_files() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    real.write(&tmp.path().join("known.txt"), b"12345", Condition::None)
        .await
        .unwrap();
    real.write(&tmp.path().join("empty.txt"), b"", Condition::None)
        .await
        .unwrap();

    let mock = MemorySystem::new().with_dir("/work").unwrap();
    mock.write(Path::new("/work/known.txt"), b"12345", Condition::None)
        .await
        .unwrap();
    mock.write(Path::new("/work/empty.txt"), b"", Condition::None)
        .await
        .unwrap();

    let real_known = real.metadata(&tmp.path().join("known.txt")).await.unwrap();
    let mock_known = mock.metadata(Path::new("/work/known.txt")).await.unwrap();
    assert_eq!(real_known.len, mock_known.len);
    assert_eq!(real_known.is_file, mock_known.is_file);
    assert_eq!(real_known.is_dir, mock_known.is_dir);

    assert_eq!(
        real.metadata(&tmp.path().join("empty.txt"))
            .await
            .unwrap()
            .len,
        mock.metadata(Path::new("/work/empty.txt"))
            .await
            .unwrap()
            .len
    );
}

#[tokio::test(flavor = "current_thread")]
async fn parity_copy_returns_len_and_contents() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    real.write(&tmp.path().join("src.txt"), b"payload", Condition::None)
        .await
        .unwrap();
    let real_len = real
        .copy(&tmp.path().join("src.txt"), &tmp.path().join("dest.txt"))
        .await
        .unwrap();

    let mock = MemorySystem::new().with_dir("/work").unwrap();
    mock.write(Path::new("/work/src.txt"), b"payload", Condition::None)
        .await
        .unwrap();
    let mock_len = mock
        .copy(Path::new("/work/src.txt"), Path::new("/work/dest.txt"))
        .await
        .unwrap();

    assert_eq!(real_len, mock_len);
    assert_eq!(
        real.read_to_string(&tmp.path().join("dest.txt"))
            .await
            .unwrap(),
        mock.read_to_string(Path::new("/work/dest.txt"))
            .await
            .unwrap()
    );
}

#[tokio::test(flavor = "current_thread")]
async fn parity_exists_is_dir_is_file() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    populate(&real, tmp.path()).await;

    let mock = MemorySystem::new();
    populate(&mock, Path::new("/work")).await;

    let real_file = tmp.path().join("alpha.txt");
    let mock_file = Path::new("/work/alpha.txt");
    assert_eq!(
        real.is_file(&real_file).await.unwrap(),
        mock.is_file(mock_file).await.unwrap()
    );
    assert_eq!(
        real.is_dir(&real_file).await.unwrap(),
        mock.is_dir(mock_file).await.unwrap()
    );

    let real_dir = tmp.path().join("sub");
    let mock_dir = Path::new("/work/sub");
    assert_eq!(
        real.is_dir(&real_dir).await.unwrap(),
        mock.is_dir(mock_dir).await.unwrap()
    );
    assert_eq!(
        real.is_file(&real_dir).await.unwrap(),
        mock.is_file(mock_dir).await.unwrap()
    );

    let real_missing = tmp.path().join("nope");
    let mock_missing = Path::new("/work/nope");
    assert_eq!(
        real.exists(&real_missing).await.unwrap(),
        mock.exists(mock_missing).await.unwrap()
    );
}

#[tokio::test(flavor = "current_thread")]
async fn parity_read_missing_file_is_not_found() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    let mock = MemorySystem::new().with_dir("/work").unwrap();

    assert_both_err_kind(
        real.read_to_string(&tmp.path().join("missing.txt")).await,
        mock.read_to_string(Path::new("/work/missing.txt")).await,
        ErrorKind::NotFound,
    );
}

#[tokio::test(flavor = "current_thread")]
async fn parity_remove_file_missing_is_not_found() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    let mock = MemorySystem::new().with_dir("/work").unwrap();

    assert_both_err_kind(
        real.remove_file(&tmp.path().join("missing.txt")).await,
        mock.remove_file(Path::new("/work/missing.txt")).await,
        ErrorKind::NotFound,
    );
}

#[tokio::test(flavor = "current_thread")]
async fn parity_rename_missing_source_is_not_found() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    let mock = MemorySystem::new().with_dir("/work").unwrap();

    assert_both_err_kind(
        real.rename(&tmp.path().join("a"), &tmp.path().join("b"))
            .await,
        mock.rename(Path::new("/work/a"), Path::new("/work/b"))
            .await,
        ErrorKind::NotFound,
    );
}

#[tokio::test(flavor = "current_thread")]
async fn parity_copy_missing_parent_is_not_found() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    real.write(&tmp.path().join("src.txt"), b"x", Condition::None)
        .await
        .unwrap();

    let mock = MemorySystem::new()
        .with_file("/work/src.txt", b"x")
        .unwrap();

    assert_both_err_kind(
        real.copy(
            &tmp.path().join("src.txt"),
            &tmp.path().join("no").join("dir").join("dest.txt"),
        )
        .await,
        mock.copy(
            Path::new("/work/src.txt"),
            Path::new("/work/no/dir/dest.txt"),
        )
        .await,
        ErrorKind::NotFound,
    );
}

#[tokio::test(flavor = "current_thread")]
async fn parity_rename_over_existing_overwrites() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    real.write(&tmp.path().join("src.txt"), b"new", Condition::None)
        .await
        .unwrap();
    real.write(&tmp.path().join("dest.txt"), b"old", Condition::None)
        .await
        .unwrap();
    real.rename(&tmp.path().join("src.txt"), &tmp.path().join("dest.txt"))
        .await
        .unwrap();

    let mock = MemorySystem::new().with_dir("/work").unwrap();
    mock.write(Path::new("/work/src.txt"), b"new", Condition::None)
        .await
        .unwrap();
    mock.write(Path::new("/work/dest.txt"), b"old", Condition::None)
        .await
        .unwrap();
    mock.rename(Path::new("/work/src.txt"), Path::new("/work/dest.txt"))
        .await
        .unwrap();

    assert_eq!(
        real.read_to_string(&tmp.path().join("dest.txt"))
            .await
            .unwrap(),
        mock.read_to_string(Path::new("/work/dest.txt"))
            .await
            .unwrap()
    );
    assert_eq!(
        real.exists(&tmp.path().join("src.txt")).await.unwrap(),
        mock.exists(Path::new("/work/src.txt")).await.unwrap()
    );
}

#[tokio::test(flavor = "current_thread")]
async fn parity_read_dir_lists_same_children() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    populate(&real, tmp.path()).await;

    let mock = MemorySystem::new();
    populate(&mock, Path::new("/work")).await;

    let real_names = relative_names(&real.read_dir(tmp.path()).await.unwrap(), tmp.path());
    let mock_names = relative_names(
        &mock.read_dir(Path::new("/work")).await.unwrap(),
        Path::new("/work"),
    );

    assert_eq!(real_names, mock_names);
    assert_eq!(real_names, vec!["alpha.txt", "empty.txt", "sub"]);
}

#[tokio::test(flavor = "current_thread")]
async fn parity_open_append_appends() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    real.write(&tmp.path().join("log.txt"), b"head", Condition::None)
        .await
        .unwrap();
    let real_log = tmp.path().join("log.txt");
    let mut real_writer = real.open_append(&real_log).await.unwrap();
    write_all(&mut *real_writer, b"-tail").await.unwrap();
    close(&mut *real_writer).await.unwrap();
    drop(real_writer);

    let mock = MemorySystem::new()
        .with_file("/work/log.txt", b"head")
        .unwrap();
    let mut mock_writer = mock.open_append(Path::new("/work/log.txt")).await.unwrap();
    write_all(&mut *mock_writer, b"-tail").await.unwrap();
    close(&mut *mock_writer).await.unwrap();
    drop(mock_writer);

    assert_eq!(
        real.read_to_string(&tmp.path().join("log.txt"))
            .await
            .unwrap(),
        mock.read_to_string(Path::new("/work/log.txt"))
            .await
            .unwrap()
    );
}

#[tokio::test(flavor = "current_thread")]
async fn parity_open_reads_same_contents() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    real.write(
        &tmp.path().join("file.txt"),
        b"shared bytes",
        Condition::None,
    )
    .await
    .unwrap();

    let mock = MemorySystem::new()
        .with_file("/work/file.txt", b"shared bytes")
        .unwrap();

    let real_contents = read_through_open(&real, &tmp.path().join("file.txt"))
        .await
        .unwrap();
    let mock_contents = read_through_open(&mock, Path::new("/work/file.txt"))
        .await
        .unwrap();

    assert_eq!(real_contents, mock_contents);
}

#[tokio::test(flavor = "current_thread")]
async fn parity_write_truncates() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    real.write(
        &tmp.path().join("file.txt"),
        b"long original",
        Condition::None,
    )
    .await
    .unwrap();
    real.write(&tmp.path().join("file.txt"), b"short", Condition::None)
        .await
        .unwrap();

    let mock = MemorySystem::new().with_dir("/work").unwrap();
    mock.write(
        Path::new("/work/file.txt"),
        b"long original",
        Condition::None,
    )
    .await
    .unwrap();
    mock.write(Path::new("/work/file.txt"), b"short", Condition::None)
        .await
        .unwrap();

    assert_eq!(
        real.read_to_string(&tmp.path().join("file.txt"))
            .await
            .unwrap(),
        mock.read_to_string(Path::new("/work/file.txt"))
            .await
            .unwrap()
    );
}

#[tokio::test(flavor = "current_thread")]
async fn parity_write_create_new_writes_missing_file() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    let real_file = tmp.path().join("file.lock");
    real.write(&real_file, b"payload", Condition::Absent)
        .await
        .unwrap();

    let mock = MemorySystem::new().with_dir("/work").unwrap();
    let mock_file = Path::new("/work/file.lock");
    mock.write(mock_file, b"payload", Condition::Absent)
        .await
        .unwrap();

    assert_eq!(real.read_to_string(&real_file).await.unwrap(), "payload");
    assert_eq!(mock.read_to_string(mock_file).await.unwrap(), "payload");
}

#[tokio::test(flavor = "current_thread")]
async fn parity_write_create_new_over_file_is_already_exists() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    let real_file = tmp.path().join("file.lock");
    real.write(&real_file, b"first", Condition::None)
        .await
        .unwrap();

    let mock = MemorySystem::new().with_dir("/work").unwrap();
    let mock_file = Path::new("/work/file.lock");
    mock.write(mock_file, b"first", Condition::None)
        .await
        .unwrap();

    assert_both_err_kind(
        real.write(&real_file, b"second", Condition::Absent).await,
        mock.write(mock_file, b"second", Condition::Absent).await,
        ErrorKind::AlreadyExists,
    );
    assert_eq!(real.read_to_string(&real_file).await.unwrap(), "first");
    assert_eq!(mock.read_to_string(mock_file).await.unwrap(), "first");
}

#[tokio::test(flavor = "current_thread")]
async fn parity_write_absent_over_a_dir_fails_and_keeps_the_dir() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    let real_dir = tmp.path().join("sub");
    real.create_dir_all(&real_dir).await.unwrap();

    let mock = MemorySystem::new().with_dir("/work/sub").unwrap();
    let mock_dir = Path::new("/work/sub");

    let real_kind = real
        .write(&real_dir, b"x", Condition::Absent)
        .await
        .err()
        .map(|err| err.kind());
    let mock_kind = mock
        .write(mock_dir, b"x", Condition::Absent)
        .await
        .err()
        .map(|err| err.kind());
    assert_eq!(
        (real_kind, mock_kind),
        (Some(ErrorKind::IsADirectory), Some(ErrorKind::IsADirectory))
    );
    assert!(real.is_dir(&real_dir).await.unwrap());
    assert!(mock.is_dir(mock_dir).await.unwrap());
}

#[tokio::test(flavor = "current_thread")]
async fn parity_write_replace_over_dir_is_a_directory() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    let real_dir = tmp.path().join("sub");
    real.create_dir_all(&real_dir).await.unwrap();

    let mock = MemorySystem::new().with_dir("/work/sub").unwrap();
    let mock_dir = Path::new("/work/sub");

    assert_both_err_kind(
        real.write(&real_dir, b"x", Condition::None).await,
        mock.write(mock_dir, b"x", Condition::None).await,
        ErrorKind::IsADirectory,
    );
    assert!(real.is_dir(&real_dir).await.unwrap());
    assert!(mock.is_dir(mock_dir).await.unwrap());
    assert!(!mock.is_file(mock_dir).await.unwrap());
}

#[tokio::test(flavor = "current_thread")]
async fn parity_create_over_dir_is_a_directory() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    let real_dir = tmp.path().join("sub");
    real.create_dir_all(&real_dir).await.unwrap();

    let mock = MemorySystem::new().with_dir("/work/sub").unwrap();
    let mock_dir = Path::new("/work/sub");

    assert_both_err_kind(
        create_and_close(&real, &real_dir).await,
        create_and_close(&mock, mock_dir).await,
        ErrorKind::IsADirectory,
    );
    assert!(real.is_dir(&real_dir).await.unwrap());
    assert!(mock.is_dir(mock_dir).await.unwrap());
    assert!(!mock.is_file(mock_dir).await.unwrap());
}

#[tokio::test(flavor = "current_thread")]
async fn parity_open_append_over_dir_is_a_directory() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    let real_dir = tmp.path().join("sub");
    real.create_dir_all(&real_dir).await.unwrap();

    let mock = MemorySystem::new().with_dir("/work/sub").unwrap();
    let mock_dir = Path::new("/work/sub");

    assert_both_err_kind(
        append_and_close(&real, &real_dir).await,
        append_and_close(&mock, mock_dir).await,
        ErrorKind::IsADirectory,
    );
    assert!(real.is_dir(&real_dir).await.unwrap());
    assert!(mock.is_dir(mock_dir).await.unwrap());
    assert!(!mock.is_file(mock_dir).await.unwrap());
}

#[tokio::test(flavor = "current_thread")]
async fn parity_copy_from_a_dir_is_invalid_input() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    let real_dir = tmp.path().join("d");
    let real_to = tmp.path().join("c.md");
    real.create_dir_all(&real_dir).await.unwrap();

    let mock = MemorySystem::new().with_dir("/work/d").unwrap();
    let mock_dir = Path::new("/work/d");
    let mock_to = Path::new("/work/c.md");

    assert_both_err_kind(
        real.copy(&real_dir, &real_to).await,
        mock.copy(mock_dir, mock_to).await,
        ErrorKind::InvalidInput,
    );
    assert!(real.is_dir(&real_dir).await.unwrap());
    assert!(mock.is_dir(mock_dir).await.unwrap());
    assert!(!real.exists(&real_to).await.unwrap());
    assert!(!mock.exists(mock_to).await.unwrap());
}

#[tokio::test(flavor = "current_thread")]
async fn parity_create_makes_the_file_when_closed() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    let real_path = tmp.path().join("new.md");
    let mock = MemorySystem::new().with_dir("/work").unwrap();
    let mock_path = Path::new("/work/new.md");

    let mut real_writer = real.create(&real_path).await.unwrap();
    let mut mock_writer = mock.create(mock_path).await.unwrap();
    close(&mut *real_writer).await.unwrap();
    close(&mut *mock_writer).await.unwrap();

    assert!(real.exists(&real_path).await.unwrap());
    assert!(mock.exists(mock_path).await.unwrap());
    assert_both_err_kind(
        real.create_dir_all(&real_path).await,
        mock.create_dir_all(mock_path).await,
        ErrorKind::AlreadyExists,
    );
    drop(real_writer);
    drop(mock_writer);
}

#[tokio::test(flavor = "current_thread")]
async fn parity_create_empties_an_existing_file_when_closed() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    let real_path = tmp.path().join("old.md");
    real.write(&real_path, b"old", Condition::None)
        .await
        .unwrap();
    let mock = MemorySystem::new().with_dir("/work").unwrap();
    let mock_path = Path::new("/work/old.md");
    mock.write(mock_path, b"old", Condition::None)
        .await
        .unwrap();

    let mut real_writer = real.create(&real_path).await.unwrap();
    let mut mock_writer = mock.create(mock_path).await.unwrap();
    close(&mut *real_writer).await.unwrap();
    close(&mut *mock_writer).await.unwrap();

    assert_eq!(real.read_to_string(&real_path).await.unwrap(), "");
    assert_eq!(mock.read_to_string(mock_path).await.unwrap(), "");
    drop(real_writer);
    drop(mock_writer);
}

#[tokio::test(flavor = "current_thread")]
async fn parity_open_append_makes_the_file_when_closed() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    let real_path = tmp.path().join("log.md");
    let mock = MemorySystem::new().with_dir("/work").unwrap();
    let mock_path = Path::new("/work/log.md");

    let mut real_writer = real.open_append(&real_path).await.unwrap();
    let mut mock_writer = mock.open_append(mock_path).await.unwrap();
    close(&mut *real_writer).await.unwrap();
    close(&mut *mock_writer).await.unwrap();

    assert!(real.exists(&real_path).await.unwrap());
    assert!(mock.exists(mock_path).await.unwrap());
    assert_eq!(real.read_to_string(&real_path).await.unwrap(), "");
    assert_eq!(mock.read_to_string(mock_path).await.unwrap(), "");
    assert_both_err_kind(
        real.create_dir_all(&real_path).await,
        mock.create_dir_all(mock_path).await,
        ErrorKind::AlreadyExists,
    );
    drop(real_writer);
    drop(mock_writer);
}

#[tokio::test(flavor = "current_thread")]
async fn parity_open_append_keeps_an_existing_file_when_opened() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    let real_path = tmp.path().join("log.md");
    real.write(&real_path, b"old", Condition::None)
        .await
        .unwrap();
    let mock = MemorySystem::new().with_dir("/work").unwrap();
    let mock_path = Path::new("/work/log.md");
    mock.write(mock_path, b"old", Condition::None)
        .await
        .unwrap();

    let mut real_writer = real.open_append(&real_path).await.unwrap();
    let mut mock_writer = mock.open_append(mock_path).await.unwrap();

    assert_eq!(real.read_to_string(&real_path).await.unwrap(), "old");
    assert_eq!(mock.read_to_string(mock_path).await.unwrap(), "old");
    write_all(&mut *real_writer, b"+new").await.unwrap();
    close(&mut *real_writer).await.unwrap();
    write_all(&mut *mock_writer, b"+new").await.unwrap();
    close(&mut *mock_writer).await.unwrap();
    assert_eq!(real.read_to_string(&real_path).await.unwrap(), "old+new");
    assert_eq!(mock.read_to_string(mock_path).await.unwrap(), "old+new");
}

#[tokio::test(flavor = "current_thread")]
async fn parity_copy_onto_dir_is_a_directory() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    let real_dir = tmp.path().join("sub");
    real.create_dir_all(&real_dir).await.unwrap();
    real.write(&tmp.path().join("src.txt"), b"x", Condition::None)
        .await
        .unwrap();

    let mock = MemorySystem::new().with_dir("/work/sub").unwrap();
    let mock_dir = Path::new("/work/sub");
    mock.write(Path::new("/work/src.txt"), b"x", Condition::None)
        .await
        .unwrap();

    assert_both_err_kind(
        real.copy(&tmp.path().join("src.txt"), &real_dir).await,
        mock.copy(Path::new("/work/src.txt"), mock_dir).await,
        ErrorKind::IsADirectory,
    );
    assert!(real.is_dir(&real_dir).await.unwrap());
    assert!(mock.is_dir(mock_dir).await.unwrap());
    assert!(!mock.is_file(mock_dir).await.unwrap());
}

#[tokio::test(flavor = "current_thread")]
async fn parity_write_create_new_missing_parent_is_not_found() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    let mock = MemorySystem::new().with_dir("/work").unwrap();

    assert_both_err_kind(
        real.write(
            &tmp.path().join("no").join("file.lock"),
            b"x",
            Condition::Absent,
        )
        .await,
        mock.write(Path::new("/work/no/file.lock"), b"x", Condition::Absent)
            .await,
        ErrorKind::NotFound,
    );
}

#[tokio::test(flavor = "current_thread")]
async fn parity_path_below_a_file_is_not_a_directory() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    let mock = MemorySystem::new().with_dir("/work").unwrap();
    real.write(&tmp.path().join("a.md"), b"a", Condition::None)
        .await
        .unwrap();
    mock.write(Path::new("/work/a.md"), b"a", Condition::None)
        .await
        .unwrap();

    assert_both_err_kind(
        real.read_to_string(&tmp.path().join("a.md/b.md")).await,
        mock.read_to_string(Path::new("/work/a.md/b.md")).await,
        ErrorKind::NotADirectory,
    );
    assert_both_err_kind(
        real.write(&tmp.path().join("a.md/b.md"), b"b", Condition::None)
            .await,
        mock.write(Path::new("/work/a.md/b.md"), b"b", Condition::None)
            .await,
        ErrorKind::NotADirectory,
    );
    assert_both_err_kind(
        real.remove_file(&tmp.path().join("a.md/b.md")).await,
        mock.remove_file(Path::new("/work/a.md/b.md")).await,
        ErrorKind::NotADirectory,
    );
    assert_both_err_kind(
        real.create_dir_all(&tmp.path().join("a.md/b")).await,
        mock.create_dir_all(Path::new("/work/a.md/b")).await,
        ErrorKind::NotADirectory,
    );
    assert_both_err_kind(
        real.open(&tmp.path().join("a.md/b.md")).await,
        mock.open(Path::new("/work/a.md/b.md")).await,
        ErrorKind::NotADirectory,
    );
    assert_both_err_kind(
        real.metadata(&tmp.path().join("a.md/b.md")).await,
        mock.metadata(Path::new("/work/a.md/b.md")).await,
        ErrorKind::NotADirectory,
    );
    assert_both_err_kind(
        create_and_close(&real, &tmp.path().join("a.md/b.md")).await,
        create_and_close(&mock, Path::new("/work/a.md/b.md")).await,
        ErrorKind::NotADirectory,
    );
    assert_both_err_kind(
        append_and_close(&real, &tmp.path().join("a.md/b.md")).await,
        append_and_close(&mock, Path::new("/work/a.md/b.md")).await,
        ErrorKind::NotADirectory,
    );
    assert_both_err_kind(
        real.read_dir(&tmp.path().join("a.md/b")).await,
        mock.read_dir(Path::new("/work/a.md/b")).await,
        ErrorKind::NotADirectory,
    );
    assert_both_err_kind(
        real.remove_dir_all(&tmp.path().join("a.md/b")).await,
        mock.remove_dir_all(Path::new("/work/a.md/b")).await,
        ErrorKind::NotADirectory,
    );
    assert_both_err_kind(
        real.rename(&tmp.path().join("a.md/b.md"), &tmp.path().join("c.md"))
            .await,
        mock.rename(Path::new("/work/a.md/b.md"), Path::new("/work/c.md"))
            .await,
        ErrorKind::NotADirectory,
    );
    assert_both_err_kind(
        real.rename(&tmp.path().join("a.md"), &tmp.path().join("a.md/b.md"))
            .await,
        mock.rename(Path::new("/work/a.md"), Path::new("/work/a.md/b.md"))
            .await,
        ErrorKind::NotADirectory,
    );
    assert_both_err_kind(
        real.copy(&tmp.path().join("a.md/b.md"), &tmp.path().join("c.md"))
            .await,
        mock.copy(Path::new("/work/a.md/b.md"), Path::new("/work/c.md"))
            .await,
        ErrorKind::NotADirectory,
    );
    assert_both_err_kind(
        real.copy(&tmp.path().join("a.md"), &tmp.path().join("a.md/b.md"))
            .await,
        mock.copy(Path::new("/work/a.md"), Path::new("/work/a.md/b.md"))
            .await,
        ErrorKind::NotADirectory,
    );
    assert_eq!(
        mock.read_to_string(Path::new("/work/a.md")).await.unwrap(),
        "a"
    );
    assert!(!mock.is_dir(Path::new("/work/a.md")).await.unwrap());
    assert!(!mock.exists(Path::new("/work/a.md/b.md")).await.unwrap());
}

#[tokio::test(flavor = "current_thread")]
async fn parity_create_dir_all_over_a_file_already_exists() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    let mock = MemorySystem::new().with_dir("/work").unwrap();
    real.write(&tmp.path().join("a.md"), b"a", Condition::None)
        .await
        .unwrap();
    mock.write(Path::new("/work/a.md"), b"a", Condition::None)
        .await
        .unwrap();

    assert_both_err_kind(
        real.create_dir_all(&tmp.path().join("a.md")).await,
        mock.create_dir_all(Path::new("/work/a.md")).await,
        ErrorKind::AlreadyExists,
    );
    assert!(!mock.is_dir(Path::new("/work/a.md")).await.unwrap());
}

#[tokio::test(flavor = "current_thread")]
async fn parity_read_dir_of_a_file_is_not_a_directory() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    let mock = MemorySystem::new().with_dir("/work").unwrap();
    real.write(&tmp.path().join("a.md"), b"a", Condition::None)
        .await
        .unwrap();
    mock.write(Path::new("/work/a.md"), b"a", Condition::None)
        .await
        .unwrap();

    assert_both_err_kind(
        real.read_dir(&tmp.path().join("a.md")).await,
        mock.read_dir(Path::new("/work/a.md")).await,
        ErrorKind::NotADirectory,
    );
    assert_eq!(
        real.read_to_string(&tmp.path().join("a.md")).await.unwrap(),
        "a"
    );
    assert_eq!(
        mock.read_to_string(Path::new("/work/a.md")).await.unwrap(),
        "a"
    );
    assert!(!mock.is_dir(Path::new("/work/a.md")).await.unwrap());
}

#[tokio::test(flavor = "current_thread")]
async fn parity_remove_dir_all_of_a_file_is_not_a_directory() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    let mock = MemorySystem::new().with_dir("/work").unwrap();
    real.write(&tmp.path().join("a.md"), b"a", Condition::None)
        .await
        .unwrap();
    mock.write(Path::new("/work/a.md"), b"a", Condition::None)
        .await
        .unwrap();

    assert_both_err_kind(
        real.remove_dir_all(&tmp.path().join("a.md")).await,
        mock.remove_dir_all(Path::new("/work/a.md")).await,
        ErrorKind::NotADirectory,
    );
    assert_eq!(
        real.read_to_string(&tmp.path().join("a.md")).await.unwrap(),
        "a"
    );
    assert_eq!(
        mock.read_to_string(Path::new("/work/a.md")).await.unwrap(),
        "a"
    );
    assert!(!mock.is_dir(Path::new("/work/a.md")).await.unwrap());
}

#[tokio::test(flavor = "current_thread")]
async fn parity_remove_file_of_a_dir_is_a_directory() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    let real_dir = tmp.path().join("d");
    real.create_dir_all(&real_dir).await.unwrap();

    let mock = MemorySystem::new().with_dir("/work/d").unwrap();
    let mock_dir = Path::new("/work/d");

    assert_both_err_kind(
        real.remove_file(&real_dir).await,
        mock.remove_file(mock_dir).await,
        ErrorKind::IsADirectory,
    );
    assert!(real.is_dir(&real_dir).await.unwrap());
    assert!(mock.is_dir(mock_dir).await.unwrap());
    assert!(!mock.is_file(mock_dir).await.unwrap());
}

#[tokio::test(flavor = "current_thread")]
async fn parity_read_to_string_of_a_dir_is_a_directory() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    let real_dir = tmp.path().join("d");
    real.create_dir_all(&real_dir).await.unwrap();

    let mock = MemorySystem::new().with_dir("/work/d").unwrap();
    let mock_dir = Path::new("/work/d");

    assert_both_err_kind(
        real.read_to_string(&real_dir).await,
        mock.read_to_string(mock_dir).await,
        ErrorKind::IsADirectory,
    );
    assert!(real.is_dir(&real_dir).await.unwrap());
    assert!(mock.is_dir(mock_dir).await.unwrap());
    assert!(!mock.is_file(mock_dir).await.unwrap());
}

#[tokio::test(flavor = "current_thread")]
async fn parity_open_of_a_dir_fails_on_read_with_is_a_directory() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    let real_dir = tmp.path().join("d");
    real.create_dir_all(&real_dir).await.unwrap();

    let mock = MemorySystem::new().with_dir("/work/d").unwrap();
    let mock_dir = Path::new("/work/d");

    assert_both_err_kind(
        read_through_open(&real, &real_dir).await,
        read_through_open(&mock, mock_dir).await,
        ErrorKind::IsADirectory,
    );
    assert!(real.is_dir(&real_dir).await.unwrap());
    assert!(mock.is_dir(mock_dir).await.unwrap());
    assert!(!mock.is_file(mock_dir).await.unwrap());
}

#[tokio::test(flavor = "current_thread")]
async fn parity_create_dir_all_idempotent_nested() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    let real_nested = tmp.path().join("a").join("b").join("c");
    real.create_dir_all(&real_nested).await.unwrap();
    real.create_dir_all(&real_nested).await.unwrap();

    let mock = MemorySystem::new();
    let mock_nested = Path::new("/work/a/b/c");
    mock.create_dir_all(mock_nested).await.unwrap();
    mock.create_dir_all(mock_nested).await.unwrap();

    assert_eq!(
        real.is_dir(&real_nested).await.unwrap(),
        mock.is_dir(mock_nested).await.unwrap()
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

#[tokio::test(flavor = "current_thread")]
async fn parity_walk_dir_visible_tree() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    populate(&real, tmp.path()).await;

    let mock = MemorySystem::new();
    populate(&mock, Path::new("/work")).await;

    let real_walk: Vec<PathBuf> = real
        .walk_dir(tmp.path(), false, false)
        .await
        .unwrap()
        .into_iter()
        .map(|entry| entry.path)
        .collect();
    let mock_walk: Vec<PathBuf> = mock
        .walk_dir(Path::new("/work"), false, false)
        .await
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

#[tokio::test(flavor = "current_thread")]
async fn parity_create_writes_via_returned_stream() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    let real_file = tmp.path().join("made.txt");
    let mut real_writer = real.create(&real_file).await.unwrap();
    write_all(&mut *real_writer, b"streamed").await.unwrap();
    close(&mut *real_writer).await.unwrap();
    drop(real_writer);

    let mock = MemorySystem::new().with_dir("/work").unwrap();
    let mock_file = Path::new("/work/made.txt");
    let mut mock_writer = mock.create(mock_file).await.unwrap();
    write_all(&mut *mock_writer, b"streamed").await.unwrap();
    close(&mut *mock_writer).await.unwrap();
    drop(mock_writer);

    assert_eq!(
        real.read_to_string(&real_file).await.unwrap(),
        mock.read_to_string(mock_file).await.unwrap()
    );
}

#[tokio::test(flavor = "current_thread")]
async fn parity_copy_missing_source_is_not_found() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    let mock = MemorySystem::new().with_dir("/work").unwrap();

    assert_both_err_kind(
        real.copy(
            &tmp.path().join("missing.txt"),
            &tmp.path().join("dest.txt"),
        )
        .await,
        mock.copy(Path::new("/work/missing.txt"), Path::new("/work/dest.txt"))
            .await,
        ErrorKind::NotFound,
    );
}

#[tokio::test(flavor = "current_thread")]
async fn parity_open_missing_file_is_not_found() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    let mock = MemorySystem::new().with_dir("/work").unwrap();

    assert_both_err_kind(
        real.open(&tmp.path().join("missing.txt")).await,
        mock.open(Path::new("/work/missing.txt")).await,
        ErrorKind::NotFound,
    );
}

#[tokio::test(flavor = "current_thread")]
async fn parity_read_dir_missing_is_not_found() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    let mock = MemorySystem::new().with_dir("/work").unwrap();

    assert_both_err_kind(
        real.read_dir(&tmp.path().join("missing")).await,
        mock.read_dir(Path::new("/work/missing")).await,
        ErrorKind::NotFound,
    );
}

#[tokio::test(flavor = "current_thread")]
async fn parity_rename_dest_parent_missing_is_not_found() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    real.write(&tmp.path().join("src.txt"), b"x", Condition::None)
        .await
        .unwrap();

    let mock = MemorySystem::new()
        .with_file("/work/src.txt", b"x")
        .unwrap();

    assert_both_err_kind(
        real.rename(
            &tmp.path().join("src.txt"),
            &tmp.path().join("no").join("dest.txt"),
        )
        .await,
        mock.rename(Path::new("/work/src.txt"), Path::new("/work/no/dest.txt"))
            .await,
        ErrorKind::NotFound,
    );
    assert_eq!(
        real.read_to_string(&tmp.path().join("src.txt"))
            .await
            .unwrap(),
        "x"
    );
    assert_eq!(
        mock.read_to_string(Path::new("/work/src.txt"))
            .await
            .unwrap(),
        "x"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn parity_rename_file_onto_dir_is_a_directory() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    real.write(&tmp.path().join("a.md"), b"a", Condition::None)
        .await
        .unwrap();
    real.create_dir_all(&tmp.path().join("d")).await.unwrap();

    let mock = MemorySystem::new()
        .with_file("/work/a.md", b"a")
        .unwrap()
        .with_dir("/work/d")
        .unwrap();

    assert_both_err_kind(
        real.rename(&tmp.path().join("a.md"), &tmp.path().join("d"))
            .await,
        mock.rename(Path::new("/work/a.md"), Path::new("/work/d"))
            .await,
        ErrorKind::IsADirectory,
    );
    assert!(real.is_file(&tmp.path().join("a.md")).await.unwrap());
    assert!(mock.is_file(Path::new("/work/a.md")).await.unwrap());
    assert!(real.is_dir(&tmp.path().join("d")).await.unwrap());
    assert!(mock.is_dir(Path::new("/work/d")).await.unwrap());
    assert!(!real.is_file(&tmp.path().join("d")).await.unwrap());
    assert!(!mock.is_file(Path::new("/work/d")).await.unwrap());
}

#[tokio::test(flavor = "current_thread")]
async fn parity_rename_dir_onto_file_is_not_a_directory() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    real.write(&tmp.path().join("a.md"), b"a", Condition::None)
        .await
        .unwrap();
    real.create_dir_all(&tmp.path().join("d")).await.unwrap();

    let mock = MemorySystem::new()
        .with_file("/work/a.md", b"a")
        .unwrap()
        .with_dir("/work/d")
        .unwrap();

    assert_both_err_kind(
        real.rename(&tmp.path().join("d"), &tmp.path().join("a.md"))
            .await,
        mock.rename(Path::new("/work/d"), Path::new("/work/a.md"))
            .await,
        ErrorKind::NotADirectory,
    );
    assert!(real.is_file(&tmp.path().join("a.md")).await.unwrap());
    assert!(mock.is_file(Path::new("/work/a.md")).await.unwrap());
    assert!(!real.is_dir(&tmp.path().join("a.md")).await.unwrap());
    assert!(!mock.is_dir(Path::new("/work/a.md")).await.unwrap());
    assert!(real.is_dir(&tmp.path().join("d")).await.unwrap());
    assert!(mock.is_dir(Path::new("/work/d")).await.unwrap());
}

#[tokio::test(flavor = "current_thread")]
async fn parity_rename_dir_to_missing_parent_is_not_found() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    real.create_dir_all(&tmp.path().join("d")).await.unwrap();

    let mock = MemorySystem::new().with_dir("/work/d").unwrap();

    assert_both_err_kind(
        real.rename(&tmp.path().join("d"), &tmp.path().join("nope").join("d3"))
            .await,
        mock.rename(Path::new("/work/d"), Path::new("/work/nope/d3"))
            .await,
        ErrorKind::NotFound,
    );
    assert!(real.exists(&tmp.path().join("d")).await.unwrap());
    assert!(mock.exists(Path::new("/work/d")).await.unwrap());
    assert!(
        !real
            .exists(&tmp.path().join("nope").join("d3"))
            .await
            .unwrap()
    );
    assert!(!mock.exists(Path::new("/work/nope/d3")).await.unwrap());
}

#[tokio::test(flavor = "current_thread")]
async fn parity_rename_dir_onto_non_empty_dir_fails_and_keeps_both() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    let root = tmp.path();
    real.create_dir_all(&root.join("d2")).await.unwrap();
    real.create_dir_all(&root.join("e2")).await.unwrap();
    real.write(&root.join("d2").join("a.md"), b"a", Condition::None)
        .await
        .unwrap();
    real.write(&root.join("e2").join("x.md"), b"x", Condition::None)
        .await
        .unwrap();

    let mock = MemorySystem::new()
        .with_file("/work/d2/a.md", b"a")
        .unwrap()
        .with_file("/work/e2/x.md", b"x")
        .unwrap();

    let real_kind = real
        .rename(&root.join("d2"), &root.join("e2"))
        .await
        .err()
        .map(|err| err.kind());
    let mock_kind = mock
        .rename(Path::new("/work/d2"), Path::new("/work/e2"))
        .await
        .err()
        .map(|err| err.kind());
    assert_eq!(
        (real_kind, mock_kind),
        (
            Some(ErrorKind::AlreadyExists),
            Some(ErrorKind::AlreadyExists)
        )
    );
    assert!(real.is_file(&root.join("d2").join("a.md")).await.unwrap());
    assert!(mock.is_file(Path::new("/work/d2/a.md")).await.unwrap());
    assert!(real.is_file(&root.join("e2").join("x.md")).await.unwrap());
    assert!(mock.is_file(Path::new("/work/e2/x.md")).await.unwrap());
}

#[tokio::test(flavor = "current_thread")]
async fn parity_rename_dir_onto_empty_dir_replaces_it() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    let root = tmp.path();
    real.create_dir_all(&root.join("d1")).await.unwrap();
    real.create_dir_all(&root.join("e1")).await.unwrap();
    real.write(&root.join("d1").join("a.md"), b"a", Condition::None)
        .await
        .unwrap();

    let mock = MemorySystem::new()
        .with_file("/work/d1/a.md", b"a")
        .unwrap()
        .with_dir("/work/e1")
        .unwrap();

    real.rename(&root.join("d1"), &root.join("e1"))
        .await
        .unwrap();
    mock.rename(Path::new("/work/d1"), Path::new("/work/e1"))
        .await
        .unwrap();

    assert!(real.is_file(&root.join("e1").join("a.md")).await.unwrap());
    assert!(mock.is_file(Path::new("/work/e1/a.md")).await.unwrap());
    assert!(!real.exists(&root.join("d1")).await.unwrap());
    assert!(!mock.exists(Path::new("/work/d1")).await.unwrap());
}

#[tokio::test(flavor = "current_thread")]
async fn parity_rename_dir_into_itself_is_invalid_input() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    let root = tmp.path();
    real.create_dir_all(&root.join("d3")).await.unwrap();
    real.write(&root.join("d3").join("f.md"), b"f", Condition::None)
        .await
        .unwrap();

    let mock = MemorySystem::new()
        .with_file("/work/d3/f.md", b"f")
        .unwrap();

    assert_both_err_kind(
        real.rename(&root.join("d3"), &root.join("d3").join("sub"))
            .await,
        mock.rename(Path::new("/work/d3"), Path::new("/work/d3/sub"))
            .await,
        ErrorKind::InvalidInput,
    );
    assert!(real.is_dir(&root.join("d3")).await.unwrap());
    assert!(mock.is_dir(Path::new("/work/d3")).await.unwrap());
    assert_eq!(
        real.read_to_string(&root.join("d3").join("f.md"))
            .await
            .unwrap(),
        "f"
    );
    assert_eq!(
        mock.read_to_string(Path::new("/work/d3/f.md"))
            .await
            .unwrap(),
        "f"
    );
    assert!(!real.exists(&root.join("d3").join("sub")).await.unwrap());
    assert!(!mock.exists(Path::new("/work/d3/sub")).await.unwrap());
}

#[tokio::test(flavor = "current_thread")]
async fn parity_rename_dir_to_sibling_with_shared_prefix_succeeds() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    let root = tmp.path();
    real.create_dir_all(&root.join("d3")).await.unwrap();
    real.write(&root.join("d3").join("f.md"), b"f", Condition::None)
        .await
        .unwrap();

    let mock = MemorySystem::new()
        .with_file("/work/d3/f.md", b"f")
        .unwrap();

    real.rename(&root.join("d3"), &root.join("d3x"))
        .await
        .unwrap();
    mock.rename(Path::new("/work/d3"), Path::new("/work/d3x"))
        .await
        .unwrap();

    assert!(real.is_file(&root.join("d3x").join("f.md")).await.unwrap());
    assert!(mock.is_file(Path::new("/work/d3x/f.md")).await.unwrap());
    assert!(!real.exists(&root.join("d3")).await.unwrap());
    assert!(!mock.exists(Path::new("/work/d3")).await.unwrap());
}

#[tokio::test(flavor = "current_thread")]
async fn parity_rename_keeps_the_modification_time() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    let root = tmp.path();
    real.write(&root.join("m.md"), b"m", Condition::None)
        .await
        .unwrap();
    let real_before = real.metadata(&root.join("m.md")).await.unwrap().modified;

    let mock = MemorySystem::new().with_dir("/work").unwrap();
    mock.write(Path::new("/work/m.md"), b"m", Condition::None)
        .await
        .unwrap();
    let mock_before = mock
        .metadata(Path::new("/work/m.md"))
        .await
        .unwrap()
        .modified;
    mock.advance(Duration::from_secs(60)).unwrap();

    real.rename(&root.join("m.md"), &root.join("n.md"))
        .await
        .unwrap();
    mock.rename(Path::new("/work/m.md"), Path::new("/work/n.md"))
        .await
        .unwrap();

    assert_eq!(
        real.metadata(&root.join("n.md")).await.unwrap().modified,
        real_before
    );
    assert_eq!(
        mock.metadata(Path::new("/work/n.md"))
            .await
            .unwrap()
            .modified,
        mock_before
    );
}

#[tokio::test(flavor = "current_thread")]
async fn parity_open_append_parent_missing_is_not_found() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    let mock = MemorySystem::new().with_dir("/work").unwrap();

    assert_both_err_kind(
        append_and_close(&real, &tmp.path().join("no").join("log.txt")).await,
        append_and_close(&mock, Path::new("/work/no/log.txt")).await,
        ErrorKind::NotFound,
    );
}

#[tokio::test(flavor = "current_thread")]
async fn parity_remove_file_success_then_missing() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    real.write(&tmp.path().join("doomed.txt"), b"x", Condition::None)
        .await
        .unwrap();
    real.remove_file(&tmp.path().join("doomed.txt"))
        .await
        .unwrap();

    let mock = MemorySystem::new()
        .with_file("/work/doomed.txt", b"x")
        .unwrap();
    mock.remove_file(Path::new("/work/doomed.txt"))
        .await
        .unwrap();

    assert_eq!(
        real.exists(&tmp.path().join("doomed.txt")).await.unwrap(),
        mock.exists(Path::new("/work/doomed.txt")).await.unwrap()
    );
    assert_both_err_kind(
        real.remove_file(&tmp.path().join("doomed.txt")).await,
        mock.remove_file(Path::new("/work/doomed.txt")).await,
        ErrorKind::NotFound,
    );
}

#[tokio::test(flavor = "current_thread")]
async fn parity_rename_directory_moves_subtree() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    real.create_dir_all(&tmp.path().join("from").join("nested"))
        .await
        .unwrap();
    real.write(
        &tmp.path().join("from").join("a.txt"),
        b"a",
        Condition::None,
    )
    .await
    .unwrap();
    real.write(
        &tmp.path().join("from").join("nested").join("b.txt"),
        b"b",
        Condition::None,
    )
    .await
    .unwrap();
    real.rename(&tmp.path().join("from"), &tmp.path().join("to"))
        .await
        .unwrap();

    let mock = MemorySystem::new();
    mock.create_dir_all(Path::new("/work/from/nested"))
        .await
        .unwrap();
    mock.write(Path::new("/work/from/a.txt"), b"a", Condition::None)
        .await
        .unwrap();
    mock.write(Path::new("/work/from/nested/b.txt"), b"b", Condition::None)
        .await
        .unwrap();
    mock.rename(Path::new("/work/from"), Path::new("/work/to"))
        .await
        .unwrap();

    assert_eq!(
        real.exists(&tmp.path().join("from")).await.unwrap(),
        mock.exists(Path::new("/work/from")).await.unwrap()
    );
    let real_walk: Vec<PathBuf> = real
        .walk_dir(&tmp.path().join("to"), false, false)
        .await
        .unwrap()
        .into_iter()
        .map(|entry| entry.path)
        .collect();
    let mock_walk: Vec<PathBuf> = mock
        .walk_dir(Path::new("/work/to"), false, false)
        .await
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
            .await
            .unwrap(),
        mock.read_to_string(Path::new("/work/to/nested/b.txt"))
            .await
            .unwrap()
    );
}

/// `remove_dir_all` on a missing path returns `NotFound` on both.
#[tokio::test(flavor = "current_thread")]
async fn parity_remove_dir_all_missing_is_not_found() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    let mock = MemorySystem::new().with_dir("/work").unwrap();

    assert_both_err_kind(
        real.remove_dir_all(&tmp.path().join("missing")).await,
        mock.remove_dir_all(Path::new("/work/missing")).await,
        ErrorKind::NotFound,
    );
}

/// `remove_dir_all` on a populated tree removes the root and every descendant on both.
#[tokio::test(flavor = "current_thread")]
async fn parity_remove_dir_all_removes_populated_tree() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    let real_root = tmp.path().join("tree");
    populate(&real, &real_root).await;

    let mock = MemorySystem::new();
    let mock_root = Path::new("/work/tree");
    populate(&mock, mock_root).await;

    real.remove_dir_all(&real_root).await.unwrap();
    mock.remove_dir_all(mock_root).await.unwrap();

    assert!(!real.exists(&real_root).await.unwrap());
    assert!(!mock.exists(mock_root).await.unwrap());
    for relative in ["alpha.txt", "empty.txt", "sub", "sub/beta.txt"] {
        assert_eq!(
            real.exists(&real_root.join(relative)).await.unwrap(),
            mock.exists(&mock_root.join(relative)).await.unwrap(),
            "mock/real disagree on {relative:?} after remove_dir_all"
        );
        assert!(!mock.exists(&mock_root.join(relative)).await.unwrap());
    }
}

/// `walk_dir` on a missing path yields an empty list on both.
#[tokio::test(flavor = "current_thread")]
async fn parity_walk_dir_missing_yields_empty() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    let mock = MemorySystem::new().with_dir("/work").unwrap();

    assert!(
        real.walk_dir(&tmp.path().join("missing"), false, false)
            .await
            .unwrap()
            .is_empty()
    );
    assert!(
        mock.walk_dir(Path::new("/work/missing"), false, false)
            .await
            .unwrap()
            .is_empty()
    );
}

/// `hidden` true includes dotfiles and false excludes them, on both.
#[tokio::test(flavor = "current_thread")]
async fn parity_walk_dir_hidden_flag_includes_dotfiles() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    real.write(&tmp.path().join("visible.txt"), b"v", Condition::None)
        .await
        .unwrap();
    real.write(&tmp.path().join(".secret"), b"s", Condition::None)
        .await
        .unwrap();

    let mock = MemorySystem::new().with_dir("/work").unwrap();
    mock.write(Path::new("/work/visible.txt"), b"v", Condition::None)
        .await
        .unwrap();
    mock.write(Path::new("/work/.secret"), b"s", Condition::None)
        .await
        .unwrap();

    let walk_names = async |hidden: bool| {
        let real_names = relative_names(
            &real
                .walk_dir(tmp.path(), false, hidden)
                .await
                .unwrap()
                .into_iter()
                .map(|entry| entry.path)
                .collect::<Vec<_>>(),
            tmp.path(),
        );
        let mock_names = relative_names(
            &mock
                .walk_dir(Path::new("/work"), false, hidden)
                .await
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
    assert_eq!(walk_names(true).await, vec![".secret", "visible.txt"]);
    assert_eq!(walk_names(false).await, vec!["visible.txt"]);
}

/// On a tree without symlinks, `follow_links` yields the same entries on both.
#[tokio::test(flavor = "current_thread")]
async fn parity_walk_dir_follow_links_agrees_without_symlinks() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    populate(&real, tmp.path()).await;

    let mock = MemorySystem::new();
    populate(&mock, Path::new("/work")).await;

    let walk_names = async |follow_links: bool| {
        let real_names = relative_names(
            &real
                .walk_dir(tmp.path(), follow_links, false)
                .await
                .unwrap()
                .into_iter()
                .map(|entry| entry.path)
                .collect::<Vec<_>>(),
            tmp.path(),
        );
        let mock_names = relative_names(
            &mock
                .walk_dir(Path::new("/work"), follow_links, false)
                .await
                .unwrap()
                .into_iter()
                .map(|entry| entry.path)
                .collect::<Vec<_>>(),
            Path::new("/work"),
        );
        assert_eq!(real_names, mock_names);
        real_names
    };

    assert_eq!(walk_names(true).await, walk_names(false).await);
    assert_eq!(
        walk_names(false).await,
        vec!["alpha.txt", "empty.txt", "sub", "sub/beta.txt"]
    );
}

#[tokio::test(flavor = "current_thread")]
async fn parity_remove_file_if_current_token_removes() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    let real_file = tmp.path().join("lock");
    let real_token = real
        .write(&real_file, b"a", Condition::Absent)
        .await
        .unwrap();
    let mock = MemorySystem::new().with_dir("/work").unwrap();
    let mock_file = Path::new("/work/lock");
    let mock_token = mock
        .write(mock_file, b"a", Condition::Absent)
        .await
        .unwrap();

    assert!(real.remove_file_if(&real_file, &real_token).await.unwrap());
    assert!(mock.remove_file_if(mock_file, &mock_token).await.unwrap());
    assert!(!real.exists(&real_file).await.unwrap());
    assert!(!mock.exists(mock_file).await.unwrap());
}

#[tokio::test(flavor = "current_thread")]
async fn parity_remove_file_if_stale_token_keeps() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    let real_file = tmp.path().join("lock");
    let real_stale = real
        .write(&real_file, b"a", Condition::Absent)
        .await
        .unwrap();
    real.write(&real_file, b"b", Condition::None).await.unwrap();
    let mock = MemorySystem::new().with_dir("/work").unwrap();
    let mock_file = Path::new("/work/lock");
    let mock_stale = mock
        .write(mock_file, b"a", Condition::Absent)
        .await
        .unwrap();
    mock.write(mock_file, b"b", Condition::None).await.unwrap();

    assert!(!real.remove_file_if(&real_file, &real_stale).await.unwrap());
    assert!(!mock.remove_file_if(mock_file, &mock_stale).await.unwrap());
    assert_eq!(real.read_to_string(&real_file).await.unwrap(), "b");
    assert_eq!(mock.read_to_string(mock_file).await.unwrap(), "b");
}

#[tokio::test(flavor = "current_thread")]
async fn parity_remove_file_if_missing_is_not_found() {
    let token = RevisionToken::new("any");
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    let mock = MemorySystem::new().with_dir("/work").unwrap();

    assert_both_err_kind(
        real.remove_file_if(&tmp.path().join("missing"), &token)
            .await,
        mock.remove_file_if(Path::new("/work/missing"), &token)
            .await,
        ErrorKind::NotFound,
    );
}

#[tokio::test(flavor = "current_thread")]
async fn parity_remove_file_if_missing_parent_is_not_found() {
    let token = RevisionToken::new("any");
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    let mock = MemorySystem::new().with_dir("/work").unwrap();

    assert_both_err_kind(
        real.remove_file_if(&tmp.path().join("missing/lock"), &token)
            .await,
        mock.remove_file_if(Path::new("/work/missing/lock"), &token)
            .await,
        ErrorKind::NotFound,
    );
}

#[tokio::test(flavor = "current_thread")]
async fn parity_remove_file_if_of_a_dir_is_a_directory() {
    let token = RevisionToken::new("any");
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    let real_dir = tmp.path().join("d");
    real.create_dir_all(&real_dir).await.unwrap();
    let mock = MemorySystem::new().with_dir("/work/d").unwrap();

    assert_both_err_kind(
        real.remove_file_if(&real_dir, &token).await,
        mock.remove_file_if(Path::new("/work/d"), &token).await,
        ErrorKind::IsADirectory,
    );
}

#[tokio::test(flavor = "current_thread")]
async fn parity_remove_file_if_below_a_file_is_not_a_directory() {
    let token = RevisionToken::new("any");
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    let mock = MemorySystem::new().with_dir("/work").unwrap();
    real.write(&tmp.path().join("a.md"), b"a", Condition::None)
        .await
        .unwrap();
    mock.write(Path::new("/work/a.md"), b"a", Condition::None)
        .await
        .unwrap();

    assert_both_err_kind(
        real.remove_file_if(&tmp.path().join("a.md/b.md"), &token)
            .await,
        mock.remove_file_if(Path::new("/work/a.md/b.md"), &token)
            .await,
        ErrorKind::NotADirectory,
    );
}

async fn release_lock(system: &dyn System, root: &Path) -> Result<()> {
    let jobs = root.join("jobs");
    let lock = jobs.join("42.lock");
    system.create_dir_all(&jobs).await?;
    let ours = system.write(&lock, b"worker-a", Condition::Absent).await?;
    let theirs = system.write(&lock, b"worker-b", Condition::Absent).await;
    assert_eq!(theirs.unwrap_err().kind(), ErrorKind::AlreadyExists);
    let (bytes, token) = system.read(&lock).await?;
    assert_eq!((bytes.as_slice(), &token), (&b"worker-a"[..], &ours));
    assert!(system.remove_file_if(&lock, &ours).await?);
    assert!(!system.exists(&lock).await?);
    Ok(())
}

#[tokio::test(flavor = "current_thread")]
async fn the_same_code_runs_on_memory_and_on_disk() {
    release_lock(&MemorySystem::new(), Path::new("/"))
        .await
        .unwrap();
    let folder = tempfile::tempdir().unwrap();
    release_lock(&RealSystem::new(), folder.path())
        .await
        .unwrap();
}

async fn read_back(system: &dyn System, root: &Path) -> (Vec<u8>, bool, u64) {
    system.create_dir_all(root).await.unwrap();
    let file = root.join("a.txt");
    let written = system
        .write(&file, b"bytes", Condition::Absent)
        .await
        .unwrap();
    let (bytes, token) = system.read(&file).await.unwrap();
    let len = system.metadata(&file).await.unwrap().len;
    (bytes, token == written, len)
}

async fn rewrite_then_release_with_the_first_token(system: &dyn System, root: &Path) -> String {
    let lock = root.join("job.lock");
    let first = system
        .write(&lock, b"one", Condition::Absent)
        .await
        .unwrap();
    system.write(&lock, b"two", Condition::None).await.unwrap();
    assert!(!system.remove_file_if(&lock, &first).await.unwrap());
    system.read_to_string(&lock).await.unwrap()
}

#[tokio::test(flavor = "current_thread")]
async fn boxed_systems_answer_the_same_through_dyn_system() {
    let folder = tempfile::tempdir().unwrap();
    let memory: Box<dyn System> = Box::new(MemorySystem::new());
    let real: Box<dyn System> = Box::new(RealSystem::new());

    let from_memory = read_back(&*memory, Path::new("/work")).await;
    let from_disk = read_back(&*real, folder.path()).await;

    assert_eq!(from_memory, (b"bytes".to_vec(), true, 5));
    assert_eq!(from_memory, from_disk);
}

#[tokio::test(flavor = "current_thread")]
async fn parity_remove_file_if_after_a_rewrite_keeps_the_new_file() {
    let folder = tempfile::tempdir().unwrap();

    let from_memory =
        rewrite_then_release_with_the_first_token(&MemorySystem::new(), Path::new("/")).await;
    let from_disk =
        rewrite_then_release_with_the_first_token(&RealSystem::new(), folder.path()).await;

    assert_eq!(from_memory, "two");
    assert_eq!(from_disk, "two");
}

#[tokio::test(flavor = "current_thread")]
async fn parity_read_dir_of_a_missing_folder_and_of_a_file() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().await.unwrap();
    real.write(&tmp.path().join("f.txt"), b"f", Condition::None)
        .await
        .unwrap();
    let mock = MemorySystem::new().with_file("/work/f.txt", b"f").unwrap();

    assert_both_err_kind(
        real.read_dir(&tmp.path().join("missing")).await,
        mock.read_dir(Path::new("/work/missing")).await,
        ErrorKind::NotFound,
    );
    assert_both_err_kind(
        real.read_dir(&tmp.path().join("f.txt")).await,
        mock.read_dir(Path::new("/work/f.txt")).await,
        ErrorKind::NotADirectory,
    );
}
