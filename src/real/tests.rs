//! Behavioral tests for `RealSystem` driven against real temp dirs.
//!
//! Every `System` method on `RealSystem` is exercised here. Tests are isolated
//! via `create_temp_dir` and unique env keys so they do not depend on the repo
//! working directory or leak state between runs.

use super::RealSystem;
use crate::System as _;
use std::env::VarError;
use std::io::{self, Read as _, Write as _};
use std::path::Path;

#[test]
fn write_then_read_to_string_roundtrips() {
    let system = RealSystem::new();
    let tmp = system.create_temp_dir().unwrap();
    let file = tmp.path().join("hello.txt");

    system.write(&file, b"hello world").unwrap();
    assert_eq!(system.read_to_string(&file).unwrap(), "hello world");
}

#[test]
fn write_truncates_existing_file() {
    let system = RealSystem::new();
    let tmp = system.create_temp_dir().unwrap();
    let file = tmp.path().join("file.txt");

    system.write(&file, b"first contents").unwrap();
    system.write(&file, b"second").unwrap();
    assert_eq!(system.read_to_string(&file).unwrap(), "second");
}

#[test]
fn copy_duplicates_file_and_returns_len() {
    let system = RealSystem::new();
    let tmp = system.create_temp_dir().unwrap();
    let src = tmp.path().join("src.txt");
    let dest = tmp.path().join("dest.txt");

    system.write(&src, b"payload").unwrap();
    let copied = system.copy(&src, &dest).unwrap();

    assert_eq!(copied, 7);
    assert_eq!(system.read_to_string(&dest).unwrap(), "payload");
    assert!(system.exists(&src).unwrap());
}

#[test]
fn create_returns_writable_stream() {
    let system = RealSystem::new();
    let tmp = system.create_temp_dir().unwrap();
    let file = tmp.path().join("created.txt");

    let mut writer = system.create(&file).unwrap();
    writer.write_all(b"streamed").unwrap();
    writer.flush().unwrap();
    drop(writer);

    assert_eq!(system.read_to_string(&file).unwrap(), "streamed");
}

#[test]
fn create_truncates_existing_file() {
    let system = RealSystem::new();
    let tmp = system.create_temp_dir().unwrap();
    let file = tmp.path().join("file.txt");

    system.write(&file, b"long original contents").unwrap();
    let mut writer = system.create(&file).unwrap();
    writer.write_all(b"short").unwrap();
    writer.flush().unwrap();
    drop(writer);

    assert_eq!(system.read_to_string(&file).unwrap(), "short");
}

#[test]
fn create_dir_all_builds_nested_and_is_idempotent() {
    let system = RealSystem::new();
    let tmp = system.create_temp_dir().unwrap();
    let nested = tmp.path().join("a").join("b").join("c");

    system.create_dir_all(&nested).unwrap();
    assert!(system.is_dir(&nested).unwrap());

    system.create_dir_all(&nested).unwrap();
    assert!(system.is_dir(&nested).unwrap());
}

#[test]
fn create_temp_dir_exists_then_cleans_up_on_drop() {
    let system = RealSystem::new();
    let tmp = system.create_temp_dir().unwrap();
    let path = tmp.path().to_path_buf();

    assert!(system.is_dir(&path).unwrap());
    drop(tmp);
    assert!(!system.exists(&path).unwrap());
}

#[test]
fn current_dir_is_absolute_and_exists() {
    let system = RealSystem::new();
    let dir = system.current_dir().unwrap();

    assert!(dir.is_absolute());
    assert!(system.is_dir(&dir).unwrap());
}

#[test]
fn current_exe_points_at_existing_file() {
    let system = RealSystem::new();
    let exe = system.current_exe().unwrap();

    assert!(exe.is_absolute());
    assert!(system.is_file(&exe).unwrap());
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

#[test]
fn exists_is_dir_is_file_classify_paths() {
    let system = RealSystem::new();
    let tmp = system.create_temp_dir().unwrap();
    let file = tmp.path().join("file.txt");
    let dir = tmp.path().join("dir");
    let missing = tmp.path().join("missing");

    system.write(&file, b"x").unwrap();
    system.create_dir_all(&dir).unwrap();

    assert!(system.exists(&file).unwrap());
    assert!(system.is_file(&file).unwrap());
    assert!(!system.is_dir(&file).unwrap());

    assert!(system.exists(&dir).unwrap());
    assert!(system.is_dir(&dir).unwrap());
    assert!(!system.is_file(&dir).unwrap());

    assert!(!system.exists(&missing).unwrap());
    assert!(!system.is_file(&missing).unwrap());
    assert!(!system.is_dir(&missing).unwrap());
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

#[test]
fn metadata_reports_len_for_known_and_empty_files() {
    let system = RealSystem::new();
    let tmp = system.create_temp_dir().unwrap();
    let known = tmp.path().join("known.txt");
    let empty = tmp.path().join("empty.txt");

    system.write(&known, b"12345").unwrap();
    system.write(&empty, b"").unwrap();

    let known_meta = system.metadata(&known).unwrap();
    assert!(known_meta.is_file);
    assert!(!known_meta.is_dir);
    assert_eq!(known_meta.len, 5);

    assert_eq!(system.metadata(&empty).unwrap().len, 0);
}

#[test]
fn metadata_missing_path_is_not_found() {
    let system = RealSystem::new();
    let tmp = system.create_temp_dir().unwrap();
    let missing = tmp.path().join("missing");

    let err = system.metadata(&missing).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::NotFound);
}

#[test]
fn open_reads_existing_contents() {
    let system = RealSystem::new();
    let tmp = system.create_temp_dir().unwrap();
    let file = tmp.path().join("read.txt");

    system.write(&file, b"readable").unwrap();
    let mut reader = system.open(&file).unwrap();
    let mut contents = String::new();
    reader.read_to_string(&mut contents).unwrap();

    assert_eq!(contents, "readable");
}

#[test]
fn open_missing_file_is_not_found() {
    let system = RealSystem::new();
    let tmp = system.create_temp_dir().unwrap();
    let missing = tmp.path().join("missing.txt");

    let kind = system.open(&missing).err().map(|err| err.kind());
    assert_eq!(kind, Some(io::ErrorKind::NotFound));
}

#[test]
fn open_append_appends_and_creates() {
    let system = RealSystem::new();
    let tmp = system.create_temp_dir().unwrap();
    let file = tmp.path().join("log.txt");

    // open_append on a missing file creates it.
    let mut first = system.open_append(&file).unwrap();
    first.write_all(b"one").unwrap();
    first.flush().unwrap();
    drop(first);

    // A second append extends rather than truncates.
    let mut second = system.open_append(&file).unwrap();
    second.write_all(b"-two").unwrap();
    second.flush().unwrap();
    drop(second);

    assert_eq!(system.read_to_string(&file).unwrap(), "one-two");
}

#[test]
fn read_dir_lists_direct_children() {
    let system = RealSystem::new();
    let tmp = system.create_temp_dir().unwrap();
    let root = tmp.path();

    system.write(&root.join("a.txt"), b"a").unwrap();
    system.write(&root.join("b.txt"), b"b").unwrap();
    system.create_dir_all(&root.join("sub")).unwrap();

    let mut names: Vec<String> = system
        .read_dir(root)
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

#[test]
fn remove_file_deletes_then_missing_is_not_found() {
    let system = RealSystem::new();
    let tmp = system.create_temp_dir().unwrap();
    let file = tmp.path().join("doomed.txt");

    system.write(&file, b"bye").unwrap();
    system.remove_file(&file).unwrap();
    assert!(!system.exists(&file).unwrap());

    let err = system.remove_file(&file).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::NotFound);
}

#[test]
fn remove_dir_all_removes_tree_then_missing_is_not_found() {
    let system = RealSystem::new();
    let tmp = system.create_temp_dir().unwrap();
    let dir = tmp.path().join("tree");

    system.create_dir_all(&dir.join("nested")).unwrap();
    system
        .write(&dir.join("nested").join("file.txt"), b"x")
        .unwrap();

    system.remove_dir_all(&dir).unwrap();
    assert!(!system.exists(&dir).unwrap());

    let err = system.remove_dir_all(&dir).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::NotFound);
}

#[test]
fn rename_moves_file() {
    let system = RealSystem::new();
    let tmp = system.create_temp_dir().unwrap();
    let src = tmp.path().join("src.txt");
    let dest = tmp.path().join("dest.txt");

    system.write(&src, b"content").unwrap();
    system.rename(&src, &dest).unwrap();

    assert!(!system.exists(&src).unwrap());
    assert_eq!(system.read_to_string(&dest).unwrap(), "content");
}

#[test]
fn rename_over_existing_target_overwrites() {
    let system = RealSystem::new();
    let tmp = system.create_temp_dir().unwrap();
    let src = tmp.path().join("src.txt");
    let dest = tmp.path().join("dest.txt");

    system.write(&src, b"new").unwrap();
    system.write(&dest, b"old").unwrap();
    system.rename(&src, &dest).unwrap();

    assert_eq!(system.read_to_string(&dest).unwrap(), "new");
}

#[test]
fn rename_missing_source_is_not_found() {
    let system = RealSystem::new();
    let tmp = system.create_temp_dir().unwrap();
    let src = tmp.path().join("missing.txt");
    let dest = tmp.path().join("dest.txt");

    let err = system.rename(&src, &dest).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::NotFound);
}

#[test]
fn canonicalize_resolves_existing_path() {
    let system = RealSystem::new();
    let tmp = system.create_temp_dir().unwrap();
    let file = tmp.path().join("real.txt");
    system.write(&file, b"x").unwrap();

    let canonical = system.canonicalize(&file).unwrap();
    assert!(canonical.is_absolute());
    assert!(canonical.ends_with("real.txt"));
}

#[test]
fn canonicalize_missing_path_is_not_found() {
    let system = RealSystem::new();
    let tmp = system.create_temp_dir().unwrap();
    let missing = tmp.path().join("missing");

    let err = system.canonicalize(&missing).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::NotFound);
}

fn relative_walk_names(
    system: RealSystem,
    root: &Path,
    follow_links: bool,
    hidden: bool,
) -> Vec<String> {
    let mut names: Vec<String> = system
        .walk_dir(root, follow_links, hidden)
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

#[test]
fn walk_dir_includes_nested_entries() {
    let system = RealSystem::new();
    let tmp = system.create_temp_dir().unwrap();
    let root = tmp.path();

    system.write(&root.join("top.txt"), b"t").unwrap();
    system.create_dir_all(&root.join("sub")).unwrap();
    system
        .write(&root.join("sub").join("inner.txt"), b"i")
        .unwrap();

    let names = relative_walk_names(system, root, false, false);
    assert_eq!(names, vec!["sub", "sub/inner.txt", "top.txt"]);
}

// The `hidden` flag is forwarded to `ignore::WalkBuilder::hidden`, where `true`
// means "ignore hidden files". So `hidden = true` EXCLUDES dotfiles and
// `hidden = false` INCLUDES them -- the inverse of the parameter's name. These
// two tests pin that observed behavior. See shim issue on the inverted flag.
#[test]
fn walk_dir_hidden_true_excludes_dotfiles() {
    let system = RealSystem::new();
    let tmp = system.create_temp_dir().unwrap();
    let root = tmp.path();

    system.write(&root.join("visible.txt"), b"v").unwrap();
    system.write(&root.join(".secret"), b"s").unwrap();

    let names = relative_walk_names(system, root, false, true);
    assert_eq!(names, vec!["visible.txt"]);
}

#[test]
fn walk_dir_hidden_false_includes_dotfiles() {
    let system = RealSystem::new();
    let tmp = system.create_temp_dir().unwrap();
    let root = tmp.path();

    system.write(&root.join("visible.txt"), b"v").unwrap();
    system.write(&root.join(".secret"), b"s").unwrap();

    let names = relative_walk_names(system, root, false, false);
    assert_eq!(names, vec![".secret", "visible.txt"]);
}

#[test]
fn walk_dir_missing_path_yields_no_entries() {
    let system = RealSystem::new();
    let tmp = system.create_temp_dir().unwrap();
    let missing = tmp.path().join("missing");

    let entries = system.walk_dir(&missing, false, false).unwrap();
    assert!(entries.is_empty());
}

#[cfg(unix)]
#[test]
fn walk_dir_follow_links_true_descends_symlinked_dir() {
    use std::os::unix::fs::symlink;

    let system = RealSystem::new();
    let tmp = system.create_temp_dir().unwrap();
    let root = tmp.path();

    let target = root.join("target");
    system.create_dir_all(&target).unwrap();
    system.write(&target.join("inside.txt"), b"x").unwrap();

    let walk_root = root.join("walk");
    system.create_dir_all(&walk_root).unwrap();
    symlink(&target, walk_root.join("link")).unwrap();

    let names = relative_walk_names(system, &walk_root, true, false);
    assert!(names.iter().any(|name| name == "link/inside.txt"));
}

#[cfg(unix)]
#[test]
fn walk_dir_follow_links_false_does_not_descend_symlinked_dir() {
    use std::os::unix::fs::symlink;

    let system = RealSystem::new();
    let tmp = system.create_temp_dir().unwrap();
    let root = tmp.path();

    let target = root.join("target");
    system.create_dir_all(&target).unwrap();
    system.write(&target.join("inside.txt"), b"x").unwrap();

    let walk_root = root.join("walk");
    system.create_dir_all(&walk_root).unwrap();
    symlink(&target, walk_root.join("link")).unwrap();

    let names = relative_walk_names(system, &walk_root, false, false);
    assert!(names.iter().any(|name| name == "link"));
    assert!(!names.iter().any(|name| name == "link/inside.txt"));
}
