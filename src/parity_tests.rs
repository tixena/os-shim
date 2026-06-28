//! Mock<->Real parity harness.
//!
//! These tests run the SAME operation sequences against `MockSystem` and
//! `RealSystem` and assert they agree on observable results (return values,
//! `io::ErrorKind` on failure, `metadata().len`, directory contents). A mock
//! that silently diverges from real is the dangerous failure mode -- green
//! tests, broken production -- so this harness is the highest-value guard.
//!
//! Known, currently-accepted divergences are pinned explicitly at the bottom of
//! this file (see the `divergence_*` tests) and tracked as separate issues so a
//! reviewer can decide whether to converge the behaviors.

use crate::System;
use crate::mock::MockSystem;
use crate::real::RealSystem;
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
    system.write(&root.join("alpha.txt"), b"alpha").unwrap();
    system.write(&root.join("empty.txt"), b"").unwrap();
    system.create_dir_all(&root.join("sub")).unwrap();
    system
        .write(&root.join("sub").join("beta.txt"), b"beta")
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
    real.write(&real_file, b"payload").unwrap();

    let mock = MockSystem::new().with_dir("/work").unwrap();
    let mock_file = Path::new("/work/file.txt");
    mock.write(mock_file, b"payload").unwrap();

    assert_eq!(
        real.read_to_string(&real_file).unwrap(),
        mock.read_to_string(mock_file).unwrap()
    );
}

#[test]
fn parity_metadata_len_for_known_and_empty_files() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    real.write(&tmp.path().join("known.txt"), b"12345").unwrap();
    real.write(&tmp.path().join("empty.txt"), b"").unwrap();

    let mock = MockSystem::new().with_dir("/work").unwrap();
    mock.write(Path::new("/work/known.txt"), b"12345").unwrap();
    mock.write(Path::new("/work/empty.txt"), b"").unwrap();

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
    real.write(&tmp.path().join("src.txt"), b"payload").unwrap();
    let real_len = real
        .copy(&tmp.path().join("src.txt"), &tmp.path().join("dest.txt"))
        .unwrap();

    let mock = MockSystem::new().with_dir("/work").unwrap();
    mock.write(Path::new("/work/src.txt"), b"payload").unwrap();
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

    let mock = MockSystem::new();
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
    let mock = MockSystem::new().with_dir("/work").unwrap();

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
    let mock = MockSystem::new().with_dir("/work").unwrap();

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
    let mock = MockSystem::new().with_dir("/work").unwrap();

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
    real.write(&tmp.path().join("src.txt"), b"x").unwrap();

    let mock = MockSystem::new().with_file("/work/src.txt", b"x").unwrap();

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
    real.write(&tmp.path().join("src.txt"), b"new").unwrap();
    real.write(&tmp.path().join("dest.txt"), b"old").unwrap();
    real.rename(&tmp.path().join("src.txt"), &tmp.path().join("dest.txt"))
        .unwrap();

    let mock = MockSystem::new().with_dir("/work").unwrap();
    mock.write(Path::new("/work/src.txt"), b"new").unwrap();
    mock.write(Path::new("/work/dest.txt"), b"old").unwrap();
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

    let mock = MockSystem::new();
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
    real.write(&tmp.path().join("log.txt"), b"head").unwrap();
    let mut real_writer = real.open_append(&tmp.path().join("log.txt")).unwrap();
    real_writer.write_all(b"-tail").unwrap();
    real_writer.flush().unwrap();
    drop(real_writer);

    let mock = MockSystem::new()
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
    real.write(&tmp.path().join("file.txt"), b"shared bytes")
        .unwrap();

    let mock = MockSystem::new()
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
    real.write(&tmp.path().join("file.txt"), b"long original")
        .unwrap();
    real.write(&tmp.path().join("file.txt"), b"short").unwrap();

    let mock = MockSystem::new().with_dir("/work").unwrap();
    mock.write(Path::new("/work/file.txt"), b"long original")
        .unwrap();
    mock.write(Path::new("/work/file.txt"), b"short").unwrap();

    assert_eq!(
        real.read_to_string(&tmp.path().join("file.txt")).unwrap(),
        mock.read_to_string(Path::new("/work/file.txt")).unwrap()
    );
}

#[test]
fn parity_create_dir_all_idempotent_nested() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    let real_nested = tmp.path().join("a").join("b").join("c");
    real.create_dir_all(&real_nested).unwrap();
    real.create_dir_all(&real_nested).unwrap();

    let mock = MockSystem::new();
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
    let mock = MockSystem::new();
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

    let mock = MockSystem::new();
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

// ============================ Known divergences ============================
// The following tests pin behavior where Mock and Real currently DISAGREE.
// They assert each implementation's actual behavior so any future change is
// caught, and each is tracked by a separate issue for a human to converge.

/// `remove_dir_all` on a missing path: Real returns `NotFound`, Mock returns Ok.
#[test]
fn divergence_remove_dir_all_missing() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    let real_err = real
        .remove_dir_all(&tmp.path().join("missing"))
        .unwrap_err();
    assert_eq!(real_err.kind(), io::ErrorKind::NotFound);

    let mock = MockSystem::new().with_dir("/work").unwrap();
    mock.remove_dir_all(Path::new("/work/missing")).unwrap();
}

/// `walk_dir` on a missing path: Real yields an empty list, Mock returns `NotFound`.
#[test]
fn divergence_walk_dir_missing() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    assert!(
        real.walk_dir(&tmp.path().join("missing"), false, false)
            .unwrap()
            .is_empty()
    );

    let mock = MockSystem::new().with_dir("/work").unwrap();
    let mock_err = mock
        .walk_dir(Path::new("/work/missing"), false, false)
        .unwrap_err();
    assert_eq!(mock_err.kind(), io::ErrorKind::NotFound);
}

/// The `hidden` flag is honored by Real (`hidden = true` excludes dotfiles) but
/// ignored entirely by Mock (dotfiles are always included regardless).
#[test]
fn divergence_walk_dir_hidden_flag() {
    let real = RealSystem::new();
    let tmp = real.create_temp_dir().unwrap();
    real.write(&tmp.path().join("visible.txt"), b"v").unwrap();
    real.write(&tmp.path().join(".secret"), b"s").unwrap();
    let real_names = relative_names(
        &real
            .walk_dir(tmp.path(), false, true)
            .unwrap()
            .into_iter()
            .map(|entry| entry.path)
            .collect::<Vec<_>>(),
        tmp.path(),
    );
    assert_eq!(real_names, vec!["visible.txt"]);

    let mock = MockSystem::new().with_dir("/work").unwrap();
    mock.write(Path::new("/work/visible.txt"), b"v").unwrap();
    mock.write(Path::new("/work/.secret"), b"s").unwrap();
    let mock_names = relative_names(
        &mock
            .walk_dir(Path::new("/work"), false, true)
            .unwrap()
            .into_iter()
            .map(|entry| entry.path)
            .collect::<Vec<_>>(),
        Path::new("/work"),
    );
    assert_eq!(mock_names, vec![".secret", "visible.txt"]);
}
