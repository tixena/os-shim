//! Memory<->Real parity harness.
//!
//! These tests run the SAME operation sequences against `MemorySystem` and
//! `RealSystem` and assert they agree on observable results (return values,
//! `io::ErrorKind` on failure, `metadata().len`, directory contents). A mock
//! that silently diverges from real is the dangerous failure mode -- green
//! tests, broken production -- so this harness is the highest-value guard.

#[macro_use]
mod sequences;

parity_sequence!(copy_from_a_dir_is_invalid_input);
parity_sequence!(copy_missing_parent_is_not_found);
parity_sequence!(copy_missing_source_is_not_found);
parity_sequence!(copy_onto_dir_is_a_directory);
parity_sequence!(copy_returns_len_and_contents);
parity_sequence!(create_dir_all_idempotent_nested);
parity_sequence!(create_dir_all_over_a_file_already_exists);
parity_sequence!(create_empties_an_existing_file_when_closed);
parity_sequence!(create_makes_the_file_when_closed);
parity_sequence!(create_over_dir_is_a_directory);
parity_sequence!(create_writes_via_returned_stream);
parity_sequence!(exists_is_dir_is_file);
parity_sequence!(folders_and_moves_below_a_file_are_refused);
parity_sequence!(metadata_len_for_known_and_empty_files);
parity_sequence!(open_append_appends);
parity_sequence!(open_append_keeps_an_existing_file_when_opened);
parity_sequence!(open_append_makes_the_file_when_closed);
parity_sequence!(open_append_over_dir_is_a_directory);
parity_sequence!(open_append_parent_missing_is_not_found);
parity_sequence!(open_missing_file_is_not_found);
parity_sequence!(open_of_a_dir_fails_on_read_with_is_a_directory);
parity_sequence!(open_reads_same_contents);
parity_sequence!(path_below_a_file_is_not_a_directory);
parity_sequence!(read_dir_lists_same_children);
parity_sequence!(read_dir_missing_is_not_found);
parity_sequence!(read_dir_of_a_file_is_not_a_directory);
parity_sequence!(read_dir_of_a_missing_folder_and_of_a_file);
parity_sequence!(read_missing_file_is_not_found);
parity_sequence!(read_to_string_of_a_dir_is_a_directory);
parity_sequence!(remove_dir_all_missing_is_not_found);
parity_sequence!(remove_dir_all_of_a_file_is_not_a_directory);
parity_sequence!(remove_dir_all_removes_populated_tree);
parity_sequence!(remove_file_if_below_a_file_is_not_a_directory);
parity_sequence!(remove_file_if_current_token_removes);
parity_sequence!(remove_file_if_missing_is_not_found);
parity_sequence!(remove_file_if_missing_parent_is_not_found);
parity_sequence!(remove_file_if_of_a_dir_is_a_directory);
parity_sequence!(remove_file_if_stale_token_keeps);
parity_sequence!(remove_file_missing_is_not_found);
parity_sequence!(remove_file_of_a_dir_is_a_directory);
parity_sequence!(remove_file_success_then_missing);
parity_sequence!(rename_dest_parent_missing_is_not_found);
parity_sequence!(rename_dir_into_itself_is_invalid_input);
parity_sequence!(rename_dir_onto_empty_dir_replaces_it);
parity_sequence!(rename_dir_onto_file_is_not_a_directory);
parity_sequence!(rename_dir_onto_non_empty_dir_fails_and_keeps_both);
parity_sequence!(rename_dir_to_missing_parent_is_not_found);
parity_sequence!(rename_dir_to_sibling_with_shared_prefix_succeeds);
parity_sequence!(rename_directory_moves_subtree);
parity_sequence!(rename_file_onto_dir_is_a_directory);
parity_sequence!(rename_keeps_the_modification_time);
parity_sequence!(rename_missing_source_is_not_found);
parity_sequence!(rename_over_existing_overwrites);
parity_sequence!(walk_dir_follow_links_agrees_without_symlinks);
parity_sequence!(walk_dir_hidden_flag_includes_dotfiles);
parity_sequence!(walk_dir_missing_yields_empty);
parity_sequence!(walk_dir_visible_tree);
parity_sequence!(write_absent_over_a_dir_fails_and_keeps_the_dir);
parity_sequence!(write_create_new_missing_parent_is_not_found);
parity_sequence!(write_create_new_over_file_is_already_exists);
parity_sequence!(write_create_new_writes_missing_file);
parity_sequence!(write_read_roundtrip);
parity_sequence!(write_replace_over_dir_is_a_directory);
parity_sequence!(write_truncates);

use crate::System;
#[cfg(any(minio_tests, s3_tests))]
use crate::bucket::Bucket as _;
use crate::bucket::{Condition, Guarantee, RevisionToken};
#[cfg(minio_tests)]
use crate::bucket::{MinioBucket, MinioConfig};
#[cfg(s3_tests)]
use crate::bucket::{S3Bucket, S3Config};
use crate::error::{ErrorKind, Result};
use crate::mock::MemorySystem;
use crate::real::RealSystem;
use crate::streams::{close, read_to_end, write_all};
use core::time::Duration;
#[cfg(any(minio_tests, s3_tests))]
use std::env;
use std::env::VarError;
use std::path::{Path, PathBuf};
#[cfg(any(minio_tests, s3_tests))]
use std::process;
#[cfg(minio_tests)]
use std::sync::OnceLock;
#[cfg(minio_tests)]
use std::thread;
#[cfg(minio_tests)]
use tokio::runtime::Builder;

#[cfg(any(minio_tests, s3_tests))]
/// A key prefix no other run or sequence shares.
fn run_prefix(name: &str) -> String {
    format!("os-shim-tests/{}-system-{name}", process::id())
}

#[cfg(minio_tests)]
/// The server's address and keys from the environment, with every key under a prefix of its own.
fn server_config(name: &str) -> MinioConfig {
    MinioConfig::new(
        env::var("OS_SHIM_MINIO_ENDPOINT").unwrap(),
        "us-east-1".to_owned(),
        "os-shim-tests".to_owned(),
        env::var("OS_SHIM_MINIO_ACCESS_KEY").unwrap(),
        env::var("OS_SHIM_MINIO_SECRET_KEY").unwrap(),
    )
    .with_prefix(run_prefix(name))
}

#[cfg(minio_tests)]
/// Create the server's bucket once per test binary, on a runtime of its own.
fn ensure_server_bucket(config: &MinioConfig) {
    static CREATED: OnceLock<()> = OnceLock::new();
    CREATED.get_or_init(|| {
        let bucket = MinioBucket::new(config.clone());
        thread::scope(|scope| {
            scope
                .spawn(|| {
                    Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .unwrap()
                        .block_on(bucket.ensure())
                        .unwrap();
                })
                .join()
                .unwrap();
        });
    });
}

#[cfg(minio_tests)]
/// Run `sequence` on the server with `/` as its root, then remove the prefix it wrote under.
async fn run_on_the_server(name: &str, sequence: impl AsyncFn(&dyn System, &Path)) {
    let config = server_config(name);
    ensure_server_bucket(&config);
    let system = RealSystem::with_bucket(MinioBucket::new(config.clone()));
    sequence(&system, Path::new("/")).await;
    drop(MinioBucket::new(config).remove_folder(Path::new("/")).await);
}

#[cfg(s3_tests)]
/// The value of an environment variable the S3 tests need.
fn aws_variable(name: &str) -> String {
    env::var(name).unwrap()
}

#[cfg(s3_tests)]
/// The AWS bucket the environment names, with every key under a prefix of its own.
fn aws_config(name: &str) -> S3Config {
    S3Config::new(
        aws_variable("OS_SHIM_S3_REGION"),
        aws_variable("OS_SHIM_S3_BUCKET"),
        aws_variable("AWS_ACCESS_KEY_ID"),
        aws_variable("AWS_SECRET_ACCESS_KEY"),
    )
    .with_prefix(run_prefix(name))
}

#[cfg(s3_tests)]
/// Run `sequence` on AWS with `/` as its root, then remove the prefix it wrote under.
async fn run_on_aws(name: &str, sequence: impl AsyncFn(&dyn System, &Path)) {
    let system = RealSystem::with_bucket(S3Bucket::new(aws_config(name)));
    sequence(&system, Path::new("/")).await;
    drop(
        S3Bucket::new(aws_config(name))
            .remove_folder(Path::new("/"))
            .await,
    );
}

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

async fn write_read_roundtrip(real: &dyn System, root: &Path) {
    let real_file = root.join("file.txt");
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

async fn metadata_len_for_known_and_empty_files(real: &dyn System, root: &Path) {
    real.write(&root.join("known.txt"), b"12345", Condition::None)
        .await
        .unwrap();
    real.write(&root.join("empty.txt"), b"", Condition::None)
        .await
        .unwrap();

    let mock = MemorySystem::new().with_dir("/work").unwrap();
    mock.write(Path::new("/work/known.txt"), b"12345", Condition::None)
        .await
        .unwrap();
    mock.write(Path::new("/work/empty.txt"), b"", Condition::None)
        .await
        .unwrap();

    let real_known = real.metadata(&root.join("known.txt")).await.unwrap();
    let mock_known = mock.metadata(Path::new("/work/known.txt")).await.unwrap();
    assert_eq!(real_known.len, mock_known.len);
    assert_eq!(real_known.is_file, mock_known.is_file);
    assert_eq!(real_known.is_dir, mock_known.is_dir);

    assert_eq!(
        real.metadata(&root.join("empty.txt")).await.unwrap().len,
        mock.metadata(Path::new("/work/empty.txt"))
            .await
            .unwrap()
            .len
    );
}

async fn copy_returns_len_and_contents(real: &dyn System, root: &Path) {
    real.write(&root.join("src.txt"), b"payload", Condition::None)
        .await
        .unwrap();
    let real_len = real
        .copy(&root.join("src.txt"), &root.join("dest.txt"))
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
        real.read_to_string(&root.join("dest.txt")).await.unwrap(),
        mock.read_to_string(Path::new("/work/dest.txt"))
            .await
            .unwrap()
    );
}

async fn exists_is_dir_is_file(real: &dyn System, root: &Path) {
    populate(real, root).await;

    let mock = MemorySystem::new();
    populate(&mock, Path::new("/work")).await;

    let real_file = root.join("alpha.txt");
    let mock_file = Path::new("/work/alpha.txt");
    assert_eq!(
        real.is_file(&real_file).await.unwrap(),
        mock.is_file(mock_file).await.unwrap()
    );
    assert_eq!(
        real.is_dir(&real_file).await.unwrap(),
        mock.is_dir(mock_file).await.unwrap()
    );

    let real_dir = root.join("sub");
    let mock_dir = Path::new("/work/sub");
    assert_eq!(
        real.is_dir(&real_dir).await.unwrap(),
        mock.is_dir(mock_dir).await.unwrap()
    );
    assert_eq!(
        real.is_file(&real_dir).await.unwrap(),
        mock.is_file(mock_dir).await.unwrap()
    );

    let real_missing = root.join("nope");
    let mock_missing = Path::new("/work/nope");
    assert_eq!(
        real.exists(&real_missing).await.unwrap(),
        mock.exists(mock_missing).await.unwrap()
    );
}

async fn read_missing_file_is_not_found(real: &dyn System, root: &Path) {
    let mock = MemorySystem::new().with_dir("/work").unwrap();

    assert_both_err_kind(
        real.read_to_string(&root.join("missing.txt")).await,
        mock.read_to_string(Path::new("/work/missing.txt")).await,
        ErrorKind::NotFound,
    );
}

async fn remove_file_missing_is_not_found(real: &dyn System, root: &Path) {
    let mock = MemorySystem::new().with_dir("/work").unwrap();

    assert_both_err_kind(
        real.remove_file(&root.join("missing.txt")).await,
        mock.remove_file(Path::new("/work/missing.txt")).await,
        ErrorKind::NotFound,
    );
}

async fn rename_missing_source_is_not_found(real: &dyn System, root: &Path) {
    let mock = MemorySystem::new().with_dir("/work").unwrap();

    assert_both_err_kind(
        real.rename(&root.join("a"), &root.join("b")).await,
        mock.rename(Path::new("/work/a"), Path::new("/work/b"))
            .await,
        ErrorKind::NotFound,
    );
}

async fn copy_missing_parent_is_not_found(real: &dyn System, root: &Path) {
    real.write(&root.join("src.txt"), b"x", Condition::None)
        .await
        .unwrap();

    let mock = MemorySystem::new()
        .with_file("/work/src.txt", b"x")
        .unwrap();

    assert_both_err_kind(
        real.copy(
            &root.join("src.txt"),
            &root.join("no").join("dir").join("dest.txt"),
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

async fn rename_over_existing_overwrites(real: &dyn System, root: &Path) {
    real.write(&root.join("src.txt"), b"new", Condition::None)
        .await
        .unwrap();
    real.write(&root.join("dest.txt"), b"old", Condition::None)
        .await
        .unwrap();
    real.rename(&root.join("src.txt"), &root.join("dest.txt"))
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
        real.read_to_string(&root.join("dest.txt")).await.unwrap(),
        mock.read_to_string(Path::new("/work/dest.txt"))
            .await
            .unwrap()
    );
    assert_eq!(
        real.exists(&root.join("src.txt")).await.unwrap(),
        mock.exists(Path::new("/work/src.txt")).await.unwrap()
    );
}

async fn read_dir_lists_same_children(real: &dyn System, root: &Path) {
    populate(real, root).await;

    let mock = MemorySystem::new();
    populate(&mock, Path::new("/work")).await;

    let real_names = relative_names(&real.read_dir(root).await.unwrap(), root);
    let mock_names = relative_names(
        &mock.read_dir(Path::new("/work")).await.unwrap(),
        Path::new("/work"),
    );

    assert_eq!(real_names, mock_names);
    assert_eq!(real_names, vec!["alpha.txt", "empty.txt", "sub"]);
}

async fn open_append_appends(real: &dyn System, root: &Path) {
    real.write(&root.join("log.txt"), b"head", Condition::None)
        .await
        .unwrap();
    let real_log = root.join("log.txt");
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
        real.read_to_string(&root.join("log.txt")).await.unwrap(),
        mock.read_to_string(Path::new("/work/log.txt"))
            .await
            .unwrap()
    );
}

async fn open_reads_same_contents(real: &dyn System, root: &Path) {
    real.write(&root.join("file.txt"), b"shared bytes", Condition::None)
        .await
        .unwrap();

    let mock = MemorySystem::new()
        .with_file("/work/file.txt", b"shared bytes")
        .unwrap();

    let real_contents = read_through_open(real, &root.join("file.txt"))
        .await
        .unwrap();
    let mock_contents = read_through_open(&mock, Path::new("/work/file.txt"))
        .await
        .unwrap();

    assert_eq!(real_contents, mock_contents);
}

async fn write_truncates(real: &dyn System, root: &Path) {
    real.write(&root.join("file.txt"), b"long original", Condition::None)
        .await
        .unwrap();
    real.write(&root.join("file.txt"), b"short", Condition::None)
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
        real.read_to_string(&root.join("file.txt")).await.unwrap(),
        mock.read_to_string(Path::new("/work/file.txt"))
            .await
            .unwrap()
    );
}

async fn write_create_new_writes_missing_file(real: &dyn System, root: &Path) {
    let real_file = root.join("file.lock");
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

async fn write_create_new_over_file_is_already_exists(real: &dyn System, root: &Path) {
    let real_file = root.join("file.lock");
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

async fn write_absent_over_a_dir_fails_and_keeps_the_dir(real: &dyn System, root: &Path) {
    let real_dir = root.join("sub");
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

async fn write_replace_over_dir_is_a_directory(real: &dyn System, root: &Path) {
    let real_dir = root.join("sub");
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

async fn create_over_dir_is_a_directory(real: &dyn System, root: &Path) {
    let real_dir = root.join("sub");
    real.create_dir_all(&real_dir).await.unwrap();

    let mock = MemorySystem::new().with_dir("/work/sub").unwrap();
    let mock_dir = Path::new("/work/sub");

    assert_both_err_kind(
        create_and_close(real, &real_dir).await,
        create_and_close(&mock, mock_dir).await,
        ErrorKind::IsADirectory,
    );
    assert!(real.is_dir(&real_dir).await.unwrap());
    assert!(mock.is_dir(mock_dir).await.unwrap());
    assert!(!mock.is_file(mock_dir).await.unwrap());
}

async fn open_append_over_dir_is_a_directory(real: &dyn System, root: &Path) {
    let real_dir = root.join("sub");
    real.create_dir_all(&real_dir).await.unwrap();

    let mock = MemorySystem::new().with_dir("/work/sub").unwrap();
    let mock_dir = Path::new("/work/sub");

    assert_both_err_kind(
        append_and_close(real, &real_dir).await,
        append_and_close(&mock, mock_dir).await,
        ErrorKind::IsADirectory,
    );
    assert!(real.is_dir(&real_dir).await.unwrap());
    assert!(mock.is_dir(mock_dir).await.unwrap());
    assert!(!mock.is_file(mock_dir).await.unwrap());
}

async fn copy_from_a_dir_is_invalid_input(real: &dyn System, root: &Path) {
    let real_dir = root.join("d");
    let real_to = root.join("c.md");
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

async fn create_makes_the_file_when_closed(real: &dyn System, root: &Path) {
    let real_path = root.join("new.md");
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

async fn create_empties_an_existing_file_when_closed(real: &dyn System, root: &Path) {
    let real_path = root.join("old.md");
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

async fn open_append_makes_the_file_when_closed(real: &dyn System, root: &Path) {
    let real_path = root.join("log.md");
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

async fn open_append_keeps_an_existing_file_when_opened(real: &dyn System, root: &Path) {
    let real_path = root.join("log.md");
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

async fn copy_onto_dir_is_a_directory(real: &dyn System, root: &Path) {
    let real_dir = root.join("sub");
    real.create_dir_all(&real_dir).await.unwrap();
    real.write(&root.join("src.txt"), b"x", Condition::None)
        .await
        .unwrap();

    let mock = MemorySystem::new().with_dir("/work/sub").unwrap();
    let mock_dir = Path::new("/work/sub");
    mock.write(Path::new("/work/src.txt"), b"x", Condition::None)
        .await
        .unwrap();

    assert_both_err_kind(
        real.copy(&root.join("src.txt"), &real_dir).await,
        mock.copy(Path::new("/work/src.txt"), mock_dir).await,
        ErrorKind::IsADirectory,
    );
    assert!(real.is_dir(&real_dir).await.unwrap());
    assert!(mock.is_dir(mock_dir).await.unwrap());
    assert!(!mock.is_file(mock_dir).await.unwrap());
}

async fn write_create_new_missing_parent_is_not_found(real: &dyn System, root: &Path) {
    let mock = MemorySystem::new().with_dir("/work").unwrap();

    assert_both_err_kind(
        real.write(&root.join("no").join("file.lock"), b"x", Condition::Absent)
            .await,
        mock.write(Path::new("/work/no/file.lock"), b"x", Condition::Absent)
            .await,
        ErrorKind::NotFound,
    );
}

async fn path_below_a_file_is_not_a_directory(real: &dyn System, root: &Path) {
    let mock = MemorySystem::new().with_dir("/work").unwrap();
    real.write(&root.join("a.md"), b"a", Condition::None)
        .await
        .unwrap();
    mock.write(Path::new("/work/a.md"), b"a", Condition::None)
        .await
        .unwrap();

    assert_both_err_kind(
        real.read_to_string(&root.join("a.md/b.md")).await,
        mock.read_to_string(Path::new("/work/a.md/b.md")).await,
        ErrorKind::NotADirectory,
    );
    assert_both_err_kind(
        real.write(&root.join("a.md/b.md"), b"b", Condition::None)
            .await,
        mock.write(Path::new("/work/a.md/b.md"), b"b", Condition::None)
            .await,
        ErrorKind::NotADirectory,
    );
    assert_both_err_kind(
        real.remove_file(&root.join("a.md/b.md")).await,
        mock.remove_file(Path::new("/work/a.md/b.md")).await,
        ErrorKind::NotADirectory,
    );
    assert_both_err_kind(
        real.create_dir_all(&root.join("a.md/b")).await,
        mock.create_dir_all(Path::new("/work/a.md/b")).await,
        ErrorKind::NotADirectory,
    );
    assert_both_err_kind(
        real.open(&root.join("a.md/b.md")).await,
        mock.open(Path::new("/work/a.md/b.md")).await,
        ErrorKind::NotADirectory,
    );
    assert_both_err_kind(
        real.metadata(&root.join("a.md/b.md")).await,
        mock.metadata(Path::new("/work/a.md/b.md")).await,
        ErrorKind::NotADirectory,
    );
    assert_both_err_kind(
        create_and_close(real, &root.join("a.md/b.md")).await,
        create_and_close(&mock, Path::new("/work/a.md/b.md")).await,
        ErrorKind::NotADirectory,
    );
    assert_both_err_kind(
        append_and_close(real, &root.join("a.md/b.md")).await,
        append_and_close(&mock, Path::new("/work/a.md/b.md")).await,
        ErrorKind::NotADirectory,
    );
}

async fn folders_and_moves_below_a_file_are_refused(real: &dyn System, root: &Path) {
    let mock = MemorySystem::new().with_dir("/work").unwrap();
    real.write(&root.join("a.md"), b"a", Condition::None)
        .await
        .unwrap();
    mock.write(Path::new("/work/a.md"), b"a", Condition::None)
        .await
        .unwrap();

    assert_both_err_kind(
        real.read_dir(&root.join("a.md/b")).await,
        mock.read_dir(Path::new("/work/a.md/b")).await,
        ErrorKind::NotADirectory,
    );
    assert_both_err_kind(
        real.remove_dir_all(&root.join("a.md/b")).await,
        mock.remove_dir_all(Path::new("/work/a.md/b")).await,
        ErrorKind::NotADirectory,
    );
    assert_both_err_kind(
        real.rename(&root.join("a.md/b.md"), &root.join("c.md"))
            .await,
        mock.rename(Path::new("/work/a.md/b.md"), Path::new("/work/c.md"))
            .await,
        ErrorKind::NotADirectory,
    );
    assert_both_err_kind(
        real.rename(&root.join("a.md"), &root.join("a.md/b.md"))
            .await,
        mock.rename(Path::new("/work/a.md"), Path::new("/work/a.md/b.md"))
            .await,
        ErrorKind::NotADirectory,
    );
    assert_both_err_kind(
        real.copy(&root.join("a.md/b.md"), &root.join("c.md")).await,
        mock.copy(Path::new("/work/a.md/b.md"), Path::new("/work/c.md"))
            .await,
        ErrorKind::NotADirectory,
    );
    assert_both_err_kind(
        real.copy(&root.join("a.md"), &root.join("a.md/b.md")).await,
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

async fn create_dir_all_over_a_file_already_exists(real: &dyn System, root: &Path) {
    let mock = MemorySystem::new().with_dir("/work").unwrap();
    real.write(&root.join("a.md"), b"a", Condition::None)
        .await
        .unwrap();
    mock.write(Path::new("/work/a.md"), b"a", Condition::None)
        .await
        .unwrap();

    assert_both_err_kind(
        real.create_dir_all(&root.join("a.md")).await,
        mock.create_dir_all(Path::new("/work/a.md")).await,
        ErrorKind::AlreadyExists,
    );
    assert!(!mock.is_dir(Path::new("/work/a.md")).await.unwrap());
}

async fn read_dir_of_a_file_is_not_a_directory(real: &dyn System, root: &Path) {
    let mock = MemorySystem::new().with_dir("/work").unwrap();
    real.write(&root.join("a.md"), b"a", Condition::None)
        .await
        .unwrap();
    mock.write(Path::new("/work/a.md"), b"a", Condition::None)
        .await
        .unwrap();

    assert_both_err_kind(
        real.read_dir(&root.join("a.md")).await,
        mock.read_dir(Path::new("/work/a.md")).await,
        ErrorKind::NotADirectory,
    );
    assert_eq!(real.read_to_string(&root.join("a.md")).await.unwrap(), "a");
    assert_eq!(
        mock.read_to_string(Path::new("/work/a.md")).await.unwrap(),
        "a"
    );
    assert!(!mock.is_dir(Path::new("/work/a.md")).await.unwrap());
}

async fn remove_dir_all_of_a_file_is_not_a_directory(real: &dyn System, root: &Path) {
    let mock = MemorySystem::new().with_dir("/work").unwrap();
    real.write(&root.join("a.md"), b"a", Condition::None)
        .await
        .unwrap();
    mock.write(Path::new("/work/a.md"), b"a", Condition::None)
        .await
        .unwrap();

    assert_both_err_kind(
        real.remove_dir_all(&root.join("a.md")).await,
        mock.remove_dir_all(Path::new("/work/a.md")).await,
        ErrorKind::NotADirectory,
    );
    assert_eq!(real.read_to_string(&root.join("a.md")).await.unwrap(), "a");
    assert_eq!(
        mock.read_to_string(Path::new("/work/a.md")).await.unwrap(),
        "a"
    );
    assert!(!mock.is_dir(Path::new("/work/a.md")).await.unwrap());
}

async fn remove_file_of_a_dir_is_a_directory(real: &dyn System, root: &Path) {
    let real_dir = root.join("d");
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

async fn read_to_string_of_a_dir_is_a_directory(real: &dyn System, root: &Path) {
    let real_dir = root.join("d");
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

async fn open_of_a_dir_fails_on_read_with_is_a_directory(real: &dyn System, root: &Path) {
    let real_dir = root.join("d");
    real.create_dir_all(&real_dir).await.unwrap();

    let mock = MemorySystem::new().with_dir("/work/d").unwrap();
    let mock_dir = Path::new("/work/d");

    assert_both_err_kind(
        read_through_open(real, &real_dir).await,
        read_through_open(&mock, mock_dir).await,
        ErrorKind::IsADirectory,
    );
    assert!(real.is_dir(&real_dir).await.unwrap());
    assert!(mock.is_dir(mock_dir).await.unwrap());
    assert!(!mock.is_file(mock_dir).await.unwrap());
}

async fn create_dir_all_idempotent_nested(real: &dyn System, root: &Path) {
    let real_nested = root.join("a").join("b").join("c");
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

async fn walk_dir_visible_tree(real: &dyn System, root: &Path) {
    populate(real, root).await;

    let mock = MemorySystem::new();
    populate(&mock, Path::new("/work")).await;

    let real_walk: Vec<PathBuf> = real
        .walk_dir(root, false, false)
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

    let real_names = relative_names(&real_walk, root);
    let mock_names = relative_names(&mock_walk, Path::new("/work"));

    assert_eq!(real_names, mock_names);
    assert_eq!(
        real_names,
        vec!["alpha.txt", "empty.txt", "sub", "sub/beta.txt"]
    );
}

async fn create_writes_via_returned_stream(real: &dyn System, root: &Path) {
    let real_file = root.join("made.txt");
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

async fn copy_missing_source_is_not_found(real: &dyn System, root: &Path) {
    let mock = MemorySystem::new().with_dir("/work").unwrap();

    assert_both_err_kind(
        real.copy(&root.join("missing.txt"), &root.join("dest.txt"))
            .await,
        mock.copy(Path::new("/work/missing.txt"), Path::new("/work/dest.txt"))
            .await,
        ErrorKind::NotFound,
    );
}

async fn open_missing_file_is_not_found(real: &dyn System, root: &Path) {
    let mock = MemorySystem::new().with_dir("/work").unwrap();

    assert_both_err_kind(
        real.open(&root.join("missing.txt")).await,
        mock.open(Path::new("/work/missing.txt")).await,
        ErrorKind::NotFound,
    );
}

async fn read_dir_missing_is_not_found(real: &dyn System, root: &Path) {
    let mock = MemorySystem::new().with_dir("/work").unwrap();

    assert_both_err_kind(
        real.read_dir(&root.join("missing")).await,
        mock.read_dir(Path::new("/work/missing")).await,
        ErrorKind::NotFound,
    );
}

async fn rename_dest_parent_missing_is_not_found(real: &dyn System, root: &Path) {
    real.write(&root.join("src.txt"), b"x", Condition::None)
        .await
        .unwrap();

    let mock = MemorySystem::new()
        .with_file("/work/src.txt", b"x")
        .unwrap();

    assert_both_err_kind(
        real.rename(&root.join("src.txt"), &root.join("no").join("dest.txt"))
            .await,
        mock.rename(Path::new("/work/src.txt"), Path::new("/work/no/dest.txt"))
            .await,
        ErrorKind::NotFound,
    );
    assert_eq!(
        real.read_to_string(&root.join("src.txt")).await.unwrap(),
        "x"
    );
    assert_eq!(
        mock.read_to_string(Path::new("/work/src.txt"))
            .await
            .unwrap(),
        "x"
    );
}

async fn rename_file_onto_dir_is_a_directory(real: &dyn System, root: &Path) {
    real.write(&root.join("a.md"), b"a", Condition::None)
        .await
        .unwrap();
    real.create_dir_all(&root.join("d")).await.unwrap();

    let mock = MemorySystem::new()
        .with_file("/work/a.md", b"a")
        .unwrap()
        .with_dir("/work/d")
        .unwrap();

    assert_both_err_kind(
        real.rename(&root.join("a.md"), &root.join("d")).await,
        mock.rename(Path::new("/work/a.md"), Path::new("/work/d"))
            .await,
        ErrorKind::IsADirectory,
    );
    assert!(real.is_file(&root.join("a.md")).await.unwrap());
    assert!(mock.is_file(Path::new("/work/a.md")).await.unwrap());
    assert!(real.is_dir(&root.join("d")).await.unwrap());
    assert!(mock.is_dir(Path::new("/work/d")).await.unwrap());
    assert!(!real.is_file(&root.join("d")).await.unwrap());
    assert!(!mock.is_file(Path::new("/work/d")).await.unwrap());
}

async fn rename_dir_onto_file_is_not_a_directory(real: &dyn System, root: &Path) {
    real.write(&root.join("a.md"), b"a", Condition::None)
        .await
        .unwrap();
    real.create_dir_all(&root.join("d")).await.unwrap();

    let mock = MemorySystem::new()
        .with_file("/work/a.md", b"a")
        .unwrap()
        .with_dir("/work/d")
        .unwrap();

    assert_both_err_kind(
        real.rename(&root.join("d"), &root.join("a.md")).await,
        mock.rename(Path::new("/work/d"), Path::new("/work/a.md"))
            .await,
        ErrorKind::NotADirectory,
    );
    assert!(real.is_file(&root.join("a.md")).await.unwrap());
    assert!(mock.is_file(Path::new("/work/a.md")).await.unwrap());
    assert!(!real.is_dir(&root.join("a.md")).await.unwrap());
    assert!(!mock.is_dir(Path::new("/work/a.md")).await.unwrap());
    assert!(real.is_dir(&root.join("d")).await.unwrap());
    assert!(mock.is_dir(Path::new("/work/d")).await.unwrap());
}

async fn rename_dir_to_missing_parent_is_not_found(real: &dyn System, root: &Path) {
    real.create_dir_all(&root.join("d")).await.unwrap();

    let mock = MemorySystem::new().with_dir("/work/d").unwrap();

    assert_both_err_kind(
        real.rename(&root.join("d"), &root.join("nope").join("d3"))
            .await,
        mock.rename(Path::new("/work/d"), Path::new("/work/nope/d3"))
            .await,
        ErrorKind::NotFound,
    );
    assert!(real.exists(&root.join("d")).await.unwrap());
    assert!(mock.exists(Path::new("/work/d")).await.unwrap());
    assert!(!real.exists(&root.join("nope").join("d3")).await.unwrap());
    assert!(!mock.exists(Path::new("/work/nope/d3")).await.unwrap());
}

async fn rename_dir_onto_non_empty_dir_fails_and_keeps_both(real: &dyn System, root: &Path) {
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

async fn rename_dir_onto_empty_dir_replaces_it(real: &dyn System, root: &Path) {
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

async fn rename_dir_into_itself_is_invalid_input(real: &dyn System, root: &Path) {
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

async fn rename_dir_to_sibling_with_shared_prefix_succeeds(real: &dyn System, root: &Path) {
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

async fn rename_keeps_the_modification_time(real: &dyn System, root: &Path) {
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

    let real_after = real.metadata(&root.join("n.md")).await.unwrap().modified;
    if real
        .guarantees()
        .contains(Guarantee::RenameKeepsModificationTime)
    {
        assert_eq!(real_after, real_before);
    } else {
        assert!(real_after.is_some());
    }
    assert_eq!(
        mock.metadata(Path::new("/work/n.md"))
            .await
            .unwrap()
            .modified,
        mock_before
    );
}

async fn open_append_parent_missing_is_not_found(real: &dyn System, root: &Path) {
    let mock = MemorySystem::new().with_dir("/work").unwrap();

    assert_both_err_kind(
        append_and_close(real, &root.join("no").join("log.txt")).await,
        append_and_close(&mock, Path::new("/work/no/log.txt")).await,
        ErrorKind::NotFound,
    );
}

async fn remove_file_success_then_missing(real: &dyn System, root: &Path) {
    real.write(&root.join("doomed.txt"), b"x", Condition::None)
        .await
        .unwrap();
    real.remove_file(&root.join("doomed.txt")).await.unwrap();

    let mock = MemorySystem::new()
        .with_file("/work/doomed.txt", b"x")
        .unwrap();
    mock.remove_file(Path::new("/work/doomed.txt"))
        .await
        .unwrap();

    assert_eq!(
        real.exists(&root.join("doomed.txt")).await.unwrap(),
        mock.exists(Path::new("/work/doomed.txt")).await.unwrap()
    );
    assert_both_err_kind(
        real.remove_file(&root.join("doomed.txt")).await,
        mock.remove_file(Path::new("/work/doomed.txt")).await,
        ErrorKind::NotFound,
    );
}

async fn rename_directory_moves_subtree(real: &dyn System, root: &Path) {
    real.create_dir_all(&root.join("from").join("nested"))
        .await
        .unwrap();
    real.write(&root.join("from").join("a.txt"), b"a", Condition::None)
        .await
        .unwrap();
    real.write(
        &root.join("from").join("nested").join("b.txt"),
        b"b",
        Condition::None,
    )
    .await
    .unwrap();
    real.rename(&root.join("from"), &root.join("to"))
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
        real.exists(&root.join("from")).await.unwrap(),
        mock.exists(Path::new("/work/from")).await.unwrap()
    );
    let real_walk: Vec<PathBuf> = real
        .walk_dir(&root.join("to"), false, false)
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
        relative_names(&real_walk, &root.join("to")),
        relative_names(&mock_walk, Path::new("/work/to"))
    );
    assert_eq!(
        real.read_to_string(&root.join("to").join("nested").join("b.txt"))
            .await
            .unwrap(),
        mock.read_to_string(Path::new("/work/to/nested/b.txt"))
            .await
            .unwrap()
    );
}

/// `remove_dir_all` on a missing path returns `NotFound` on both.
async fn remove_dir_all_missing_is_not_found(real: &dyn System, root: &Path) {
    let mock = MemorySystem::new().with_dir("/work").unwrap();

    assert_both_err_kind(
        real.remove_dir_all(&root.join("missing")).await,
        mock.remove_dir_all(Path::new("/work/missing")).await,
        ErrorKind::NotFound,
    );
}

/// `remove_dir_all` on a populated tree removes the root and every descendant on both.
async fn remove_dir_all_removes_populated_tree(real: &dyn System, root: &Path) {
    let real_root = root.join("tree");
    populate(real, &real_root).await;

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
async fn walk_dir_missing_yields_empty(real: &dyn System, root: &Path) {
    let mock = MemorySystem::new().with_dir("/work").unwrap();

    assert!(
        real.walk_dir(&root.join("missing"), false, false)
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
async fn walk_dir_hidden_flag_includes_dotfiles(real: &dyn System, root: &Path) {
    real.write(&root.join("visible.txt"), b"v", Condition::None)
        .await
        .unwrap();
    real.write(&root.join(".secret"), b"s", Condition::None)
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
                .walk_dir(root, false, hidden)
                .await
                .unwrap()
                .into_iter()
                .map(|entry| entry.path)
                .collect::<Vec<_>>(),
            root,
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
async fn walk_dir_follow_links_agrees_without_symlinks(real: &dyn System, root: &Path) {
    populate(real, root).await;

    let mock = MemorySystem::new();
    populate(&mock, Path::new("/work")).await;

    let walk_names = async |follow_links: bool| {
        let real_names = relative_names(
            &real
                .walk_dir(root, follow_links, false)
                .await
                .unwrap()
                .into_iter()
                .map(|entry| entry.path)
                .collect::<Vec<_>>(),
            root,
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

async fn remove_file_if_current_token_removes(real: &dyn System, root: &Path) {
    let real_file = root.join("lock");
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

async fn remove_file_if_stale_token_keeps(real: &dyn System, root: &Path) {
    let real_file = root.join("lock");
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

async fn remove_file_if_missing_is_not_found(real: &dyn System, root: &Path) {
    let token = RevisionToken::new("any");
    let mock = MemorySystem::new().with_dir("/work").unwrap();

    assert_both_err_kind(
        real.remove_file_if(&root.join("missing"), &token).await,
        mock.remove_file_if(Path::new("/work/missing"), &token)
            .await,
        ErrorKind::NotFound,
    );
}

async fn remove_file_if_missing_parent_is_not_found(real: &dyn System, root: &Path) {
    let token = RevisionToken::new("any");
    let mock = MemorySystem::new().with_dir("/work").unwrap();

    assert_both_err_kind(
        real.remove_file_if(&root.join("missing/lock"), &token)
            .await,
        mock.remove_file_if(Path::new("/work/missing/lock"), &token)
            .await,
        ErrorKind::NotFound,
    );
}

async fn remove_file_if_of_a_dir_is_a_directory(real: &dyn System, root: &Path) {
    let token = RevisionToken::new("any");
    let real_dir = root.join("d");
    real.create_dir_all(&real_dir).await.unwrap();
    let mock = MemorySystem::new().with_dir("/work/d").unwrap();

    assert_both_err_kind(
        real.remove_file_if(&real_dir, &token).await,
        mock.remove_file_if(Path::new("/work/d"), &token).await,
        ErrorKind::IsADirectory,
    );
}

async fn remove_file_if_below_a_file_is_not_a_directory(real: &dyn System, root: &Path) {
    let token = RevisionToken::new("any");
    let mock = MemorySystem::new().with_dir("/work").unwrap();
    real.write(&root.join("a.md"), b"a", Condition::None)
        .await
        .unwrap();
    mock.write(Path::new("/work/a.md"), b"a", Condition::None)
        .await
        .unwrap();

    assert_both_err_kind(
        real.remove_file_if(&root.join("a.md/b.md"), &token).await,
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

async fn read_dir_of_a_missing_folder_and_of_a_file(real: &dyn System, root: &Path) {
    real.write(&root.join("f.txt"), b"f", Condition::None)
        .await
        .unwrap();
    let mock = MemorySystem::new().with_file("/work/f.txt", b"f").unwrap();

    assert_both_err_kind(
        real.read_dir(&root.join("missing")).await,
        mock.read_dir(Path::new("/work/missing")).await,
        ErrorKind::NotFound,
    );
    assert_both_err_kind(
        real.read_dir(&root.join("f.txt")).await,
        mock.read_dir(Path::new("/work/f.txt")).await,
        ErrorKind::NotADirectory,
    );
}

#[cfg(minio_tests)]
#[tokio::test(flavor = "current_thread")]
async fn the_same_code_runs_on_the_server() {
    run_on_the_server("release_lock", async |system: &dyn System, root: &Path| {
        release_lock(system, root).await.unwrap();
    })
    .await;
}

#[cfg(minio_tests)]
#[tokio::test(flavor = "current_thread")]
async fn boxed_systems_answer_the_same_through_dyn_system_on_the_server() {
    run_on_the_server("read_back", async |system: &dyn System, root: &Path| {
        let memory: Box<dyn System> = Box::new(MemorySystem::new());
        assert_eq!(
            read_back(system, root).await,
            read_back(&*memory, Path::new("/work")).await
        );
    })
    .await;
}

#[cfg(minio_tests)]
#[tokio::test(flavor = "current_thread")]
async fn parity_remove_file_if_after_a_rewrite_keeps_the_new_file_on_the_server() {
    run_on_the_server(
        "rewrite_then_release",
        async |system: &dyn System, root: &Path| {
            let from_memory =
                rewrite_then_release_with_the_first_token(&MemorySystem::new(), Path::new("/"))
                    .await;
            assert_eq!(
                rewrite_then_release_with_the_first_token(system, root).await,
                from_memory
            );
        },
    )
    .await;
}

#[cfg(s3_tests)]
#[tokio::test(flavor = "current_thread")]
async fn the_same_code_runs_on_aws() {
    run_on_aws("release_lock", async |system: &dyn System, root: &Path| {
        release_lock(system, root).await.unwrap();
    })
    .await;
}

#[cfg(s3_tests)]
#[tokio::test(flavor = "current_thread")]
async fn boxed_systems_answer_the_same_through_dyn_system_on_aws() {
    run_on_aws("read_back", async |system: &dyn System, root: &Path| {
        let memory: Box<dyn System> = Box::new(MemorySystem::new());
        assert_eq!(
            read_back(system, root).await,
            read_back(&*memory, Path::new("/work")).await
        );
    })
    .await;
}

#[cfg(s3_tests)]
#[tokio::test(flavor = "current_thread")]
async fn parity_remove_file_if_after_a_rewrite_keeps_the_new_file_on_aws() {
    run_on_aws(
        "rewrite_then_release",
        async |system: &dyn System, root: &Path| {
            let from_memory =
                rewrite_then_release_with_the_first_token(&MemorySystem::new(), Path::new("/"))
                    .await;
            assert_eq!(
                rewrite_then_release_with_the_first_token(system, root).await,
                from_memory
            );
        },
    )
    .await;
}
