//! Tests for zip archive production and loading.

use super::{ZipOptions, ZipStreamWriter, to_zip_writer};
use crate::bucket::{Condition, Guarantees, RevisionToken};
use crate::error::{ErrorKind, Result};
use crate::mock::MemorySystem;
use crate::real::RealSystem;
#[cfg(slow_tests)]
use crate::streams::read_chunk;
use crate::streams::{close, write_all};
use crate::{BoxFuture, FileMetadata, System, TempDirHandle, WalkEntry};
use core::result;
use core::time::Duration;
use futures_io::{AsyncRead, AsyncWrite};
use std::env::VarError;
#[cfg(slow_tests)]
use std::fs::File;
use std::io::{self, Cursor, Read as _, Write, pipe};
use std::path::{Path, PathBuf};
use std::process::Command;
#[cfg(slow_tests)]
use std::process::Stdio;
use std::thread;
use std::time::SystemTime;
use time::OffsetDateTime;
use tokio::runtime::Builder;
use zip::ZipArchive;
use zip::write::{SimpleFileOptions, StreamWriter, ZipWriter};

/// Bytes in the large entry the zip64 pin archives: four gibibytes and one byte,
/// one past the largest length a non-zip64 ZIP header can express.
#[cfg(slow_tests)]
const LARGE_ENTRY_BYTES: u64 = 0x0001_0000_0001;

/// Bytes moved per pass while the large fixture is generated.
#[cfg(slow_tests)]
const LARGE_ENTRY_CHUNK_BYTES: usize = 0x0010_0000;

/// Bytes a local file header occupies ahead of its variable-length file name.
const LOCAL_HEADER_FIXED_BYTES: usize = 30;

/// The four bytes opening a zip64 extended information extra field: header id
/// `0x0001` and a sixteen-byte payload length, each little-endian.
const ZIP64_EXTRA_FIELD_HEADER: [u8; 4] = [0x01, 0x00, 0x10, 0x00];

/// A sink that can only be written to, counting what it receives.
struct CountingSink {
    /// Bytes received so far.
    count: usize,
}

/// A `System` implementor writing only the methods the trait required before the
/// zip methods arrived, proving an out-of-crate implementor gains them for free.
struct MinimalSystem {
    /// The filesystem every required method delegates to.
    inner: MemorySystem,
}

/// A `TempDirHandle` implementor writing only `path`, so it keeps the default
/// `to_zip_stream` body.
struct PlainTempDir {
    /// The path the handle reports.
    path: PathBuf,
}

impl System for MinimalSystem {
    fn canonicalize<'system>(
        &'system self,
        path: &'system Path,
    ) -> BoxFuture<'system, Result<PathBuf>> {
        self.inner.canonicalize(path)
    }

    fn copy<'system>(
        &'system self,
        from: &'system Path,
        to: &'system Path,
    ) -> BoxFuture<'system, Result<u64>> {
        self.inner.copy(from, to)
    }

    fn create<'system>(
        &'system self,
        path: &'system Path,
    ) -> BoxFuture<'system, Result<Box<dyn AsyncWrite + Send + Unpin + 'system>>> {
        self.inner.create(path)
    }

    fn create_dir_all<'system>(
        &'system self,
        path: &'system Path,
    ) -> BoxFuture<'system, Result<()>> {
        self.inner.create_dir_all(path)
    }

    fn create_temp_dir<'system>(
        &'system self,
    ) -> BoxFuture<'system, Result<Box<dyn TempDirHandle + 'system>>> {
        self.inner.create_temp_dir()
    }

    fn current_dir(&self) -> Result<PathBuf> {
        self.inner.current_dir()
    }

    fn current_exe(&self) -> Result<PathBuf> {
        self.inner.current_exe()
    }

    fn env_var(&self, key: &str) -> result::Result<String, VarError> {
        self.inner.env_var(key)
    }

    fn exists<'system>(&'system self, path: &'system Path) -> BoxFuture<'system, Result<bool>> {
        self.inner.exists(path)
    }

    fn guarantees(&self) -> Guarantees {
        self.inner.guarantees()
    }

    fn is_dir<'system>(&'system self, path: &'system Path) -> BoxFuture<'system, Result<bool>> {
        self.inner.is_dir(path)
    }

    fn is_file<'system>(&'system self, path: &'system Path) -> BoxFuture<'system, Result<bool>> {
        self.inner.is_file(path)
    }

    #[cfg(feature = "process")]
    fn is_pid_alive(&self, pid: u32) -> bool {
        self.inner.is_pid_alive(pid)
    }

    fn metadata<'system>(
        &'system self,
        path: &'system Path,
    ) -> BoxFuture<'system, Result<FileMetadata>> {
        self.inner.metadata(path)
    }

    fn now(&self) -> SystemTime {
        self.inner.now()
    }

    fn open<'system>(
        &'system self,
        path: &'system Path,
    ) -> BoxFuture<'system, Result<Box<dyn AsyncRead + Send + Unpin + 'system>>> {
        self.inner.open(path)
    }

    fn open_append<'system>(
        &'system self,
        path: &'system Path,
    ) -> BoxFuture<'system, Result<Box<dyn AsyncWrite + Send + Unpin + 'system>>> {
        self.inner.open_append(path)
    }

    fn read<'system>(
        &'system self,
        path: &'system Path,
    ) -> BoxFuture<'system, Result<(Vec<u8>, RevisionToken)>> {
        self.inner.read(path)
    }

    fn read_dir<'system>(
        &'system self,
        path: &'system Path,
    ) -> BoxFuture<'system, Result<Vec<PathBuf>>> {
        self.inner.read_dir(path)
    }

    fn read_to_string<'system>(
        &'system self,
        path: &'system Path,
    ) -> BoxFuture<'system, Result<String>> {
        self.inner.read_to_string(path)
    }

    fn remove_dir_all<'system>(
        &'system self,
        path: &'system Path,
    ) -> BoxFuture<'system, Result<()>> {
        self.inner.remove_dir_all(path)
    }

    fn remove_file<'system>(&'system self, path: &'system Path) -> BoxFuture<'system, Result<()>> {
        self.inner.remove_file(path)
    }

    fn remove_file_if<'system>(
        &'system self,
        path: &'system Path,
        token: &'system RevisionToken,
    ) -> BoxFuture<'system, Result<bool>> {
        self.inner.remove_file_if(path, token)
    }

    fn rename<'system>(
        &'system self,
        from: &'system Path,
        to: &'system Path,
    ) -> BoxFuture<'system, Result<()>> {
        self.inner.rename(from, to)
    }

    fn set_env_var(&self, key: &str, value: &str) {
        self.inner.set_env_var(key, value);
    }

    fn sleep(&self, duration: Duration) -> BoxFuture<'_, Result<()>> {
        self.inner.sleep(duration)
    }

    fn walk_dir<'system>(
        &'system self,
        path: &'system Path,
        follow_links: bool,
        hidden: bool,
    ) -> BoxFuture<'system, Result<Vec<WalkEntry>>> {
        self.inner.walk_dir(path, follow_links, hidden)
    }

    fn write<'system>(
        &'system self,
        path: &'system Path,
        contents: &'system [u8],
        condition: Condition,
    ) -> BoxFuture<'system, Result<RevisionToken>> {
        self.inner.write(path, contents, condition)
    }
}

impl Write for CountingSink {
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }

    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.count = self.count.checked_add(buf.len()).unwrap();
        Ok(buf.len())
    }
}

impl TempDirHandle for PlainTempDir {
    fn path(&self) -> &Path {
        &self.path
    }
}

/// Bytes of the entry named `name`.
fn entry_bytes(archive: &[u8], name: &str) -> Vec<u8> {
    let mut reader = ZipArchive::new(Cursor::new(archive.to_vec())).unwrap();
    let mut entry = reader.by_name(name).unwrap();
    let mut bytes = Vec::new();
    entry.read_to_end(&mut bytes).unwrap();
    bytes
}

/// Entry names in the order the archive stores them.
fn entry_names(archive: &[u8]) -> Vec<String> {
    let mut reader = ZipArchive::new(Cursor::new(archive.to_vec())).unwrap();
    (0..reader.len())
        .map(|index| reader.by_index(index).unwrap().name().unwrap().into_owned())
        .collect()
}

/// The year stamped on the entry named `name`.
fn entry_year(archive: &[u8], name: &str) -> u16 {
    let mut reader = ZipArchive::new(Cursor::new(archive.to_vec())).unwrap();
    reader
        .by_name(name)
        .unwrap()
        .last_modified()
        .unwrap()
        .year()
}

/// The extra field bytes carried by the *local* header of the entry named `name`.
///
/// The reader strips a zip64 extra field out of the field it hands back, so the
/// bytes are cut from the archive itself, between the end of the local header's
/// file name and the start of the entry's data.
fn local_extra_field(archive: &[u8], name: &str) -> Vec<u8> {
    let mut reader = ZipArchive::new(Cursor::new(archive.to_vec())).unwrap();
    let entry = reader.by_name(name).unwrap();
    let extra_start = usize::try_from(entry.header_start())
        .unwrap()
        .checked_add(LOCAL_HEADER_FIXED_BYTES)
        .and_then(|offset| offset.checked_add(entry.name_raw().len()))
        .unwrap();
    let data_start = usize::try_from(entry.data_start().unwrap()).unwrap();
    archive
        .get(extra_start..data_start)
        .map(<[u8]>::to_vec)
        .unwrap()
}

/// An in-memory filesystem holding the tree the archive tests are written against.
fn memory_project() -> MemorySystem {
    MemorySystem::new()
        .with_dir("/proj/.git")
        .unwrap()
        .with_dir("/proj/empty")
        .unwrap()
        .with_file("/proj/.gitignore", b"target/\n")
        .unwrap()
        .with_file("/proj/README.md", b"hi\n")
        .unwrap()
        .with_file("/proj/src/main.rs", b"fn main() {}\n")
        .unwrap()
        .with_file("/proj/target/artifact.bin", b"\x00\x01\x02")
        .unwrap()
}

#[test]
fn default_options_include_hidden() {
    assert!(ZipOptions::default().include_hidden);
}

#[test]
fn with_hidden_false_excludes_hidden() {
    assert!(!ZipOptions::default().with_hidden(false).include_hidden);
}

#[test]
fn with_hidden_true_restores_the_default() {
    let options = ZipOptions::default().with_hidden(false).with_hidden(true);
    assert!(options.include_hidden);
}

#[tokio::test(flavor = "current_thread")]
async fn directory_contributes_its_contents_at_the_top_level() {
    let system = memory_project();
    let mut archive = Vec::new();
    to_zip_writer(
        &system,
        Path::new("/proj"),
        &mut archive,
        ZipOptions::default(),
    )
    .await
    .unwrap();

    assert_eq!(
        entry_names(&archive),
        vec![
            ".git/".to_owned(),
            "empty/".to_owned(),
            "src/".to_owned(),
            "target/".to_owned(),
            ".gitignore".to_owned(),
            "README.md".to_owned(),
            "src/main.rs".to_owned(),
            "target/artifact.bin".to_owned(),
        ]
    );
    assert_eq!(entry_bytes(&archive, "README.md"), b"hi\n");
    assert_eq!(entry_bytes(&archive, "src/main.rs"), b"fn main() {}\n");
}

#[tokio::test(flavor = "current_thread")]
async fn empty_directory_survives_as_an_explicit_entry() {
    let system = memory_project();
    let mut archive = Vec::new();
    to_zip_writer(
        &system,
        Path::new("/proj"),
        &mut archive,
        ZipOptions::default(),
    )
    .await
    .unwrap();

    assert!(entry_names(&archive).contains(&"empty/".to_owned()));
}

#[tokio::test(flavor = "current_thread")]
async fn hidden_entries_are_dropped_when_the_option_is_off() {
    let system = memory_project();
    let mut archive = Vec::new();
    to_zip_writer(
        &system,
        Path::new("/proj"),
        &mut archive,
        ZipOptions::default().with_hidden(false),
    )
    .await
    .unwrap();

    let names = entry_names(&archive);
    assert!(!names.contains(&".gitignore".to_owned()));
    assert!(!names.contains(&".git/".to_owned()));
    assert!(names.contains(&"README.md".to_owned()));
}

#[tokio::test(flavor = "current_thread")]
async fn single_file_yields_exactly_one_bare_entry() {
    let system = MemorySystem::new()
        .with_file("/notes/notes.txt", b"jotted")
        .unwrap();
    let mut archive = Vec::new();
    to_zip_writer(
        &system,
        Path::new("/notes/notes.txt"),
        &mut archive,
        ZipOptions::default(),
    )
    .await
    .unwrap();

    assert_eq!(entry_names(&archive), vec!["notes.txt".to_owned()]);
    assert_eq!(entry_bytes(&archive, "notes.txt"), b"jotted");
}

#[tokio::test(flavor = "current_thread")]
async fn missing_source_reports_not_found() {
    let system = MemorySystem::new();
    let mut archive = Vec::new();
    let result = to_zip_writer(
        &system,
        Path::new("/absent"),
        &mut archive,
        ZipOptions::default(),
    )
    .await;

    assert_eq!(result.unwrap_err().kind(), ErrorKind::NotFound);
}

#[tokio::test(flavor = "current_thread")]
async fn directory_timestamp_older_than_the_zip_floor_is_clamped_up() {
    // `MemorySystem` stamps unwritten directories with the Unix epoch, before any date a
    // ZIP entry can carry.
    let system = MemorySystem::new().with_dir("/proj/empty").unwrap();
    let mut archive = Vec::new();
    to_zip_writer(
        &system,
        Path::new("/proj"),
        &mut archive,
        ZipOptions::default(),
    )
    .await
    .unwrap();

    assert_eq!(entry_year(&archive, "empty/"), 1980);
}

#[tokio::test(flavor = "current_thread")]
async fn entry_carries_the_source_modification_time() {
    let system = RealSystem::new();
    let temp = system.create_temp_dir().await.unwrap();
    let file = temp.path().join("fresh.txt");
    system.write(&file, b"now", Condition::None).await.unwrap();

    let mut archive = Vec::new();
    to_zip_writer(&system, &file, &mut archive, ZipOptions::default())
        .await
        .unwrap();

    let modified = system.metadata(&file).await.unwrap().modified.unwrap();
    let expected = OffsetDateTime::from(modified);
    let expected_year = u16::try_from(expected.year()).unwrap();
    assert_eq!(entry_year(&archive, "fresh.txt"), expected_year);
    assert!(expected_year > 1980);
}

#[tokio::test(flavor = "current_thread")]
async fn gitignored_file_inside_a_git_repository_is_archived() {
    let system = RealSystem::new();
    let temp = system.create_temp_dir().await.unwrap();
    let root = temp.path().join("repo");
    system.create_dir_all(&root.join(".git")).await.unwrap();
    system
        .write(&root.join(".gitignore"), b"target/\n", Condition::None)
        .await
        .unwrap();
    system.create_dir_all(&root.join("target")).await.unwrap();
    system
        .write(&root.join("target/artifact.bin"), b"built", Condition::None)
        .await
        .unwrap();

    let mut archive = Vec::new();
    to_zip_writer(&system, &root, &mut archive, ZipOptions::default())
        .await
        .unwrap();

    let names = entry_names(&archive);
    assert!(names.contains(&"target/artifact.bin".to_owned()));
    assert!(names.contains(&".gitignore".to_owned()));
    assert_eq!(entry_bytes(&archive, "target/artifact.bin"), b"built");
}

#[cfg(unix)]
#[tokio::test(flavor = "current_thread")]
async fn symlink_back_to_an_ancestor_terminates_as_an_empty_directory() {
    use std::os::unix::fs::symlink;

    let system = RealSystem::new();
    let temp = system.create_temp_dir().await.unwrap();
    let root = temp.path().join("root");
    system.create_dir_all(&root.join("inner")).await.unwrap();
    symlink(&root, root.join("inner/loop")).unwrap();

    let mut archive = Vec::new();
    to_zip_writer(&system, &root, &mut archive, ZipOptions::default())
        .await
        .unwrap();

    assert_eq!(
        entry_names(&archive),
        vec!["inner/".to_owned(), "inner/loop/".to_owned()]
    );
}

#[tokio::test(flavor = "current_thread")]
async fn real_and_memory_agree_on_names_and_contents() {
    let real = RealSystem::new();
    let temp = real.create_temp_dir().await.unwrap();
    let root = temp.path().join("proj");
    real.create_dir_all(&root.join("empty")).await.unwrap();
    real.create_dir_all(&root.join("src")).await.unwrap();
    real.write(&root.join(".hidden"), b"dot", Condition::None)
        .await
        .unwrap();
    real.write(&root.join("README.md"), b"hi\n", Condition::None)
        .await
        .unwrap();
    real.write(
        &root.join("src/main.rs"),
        b"fn main() {}\n",
        Condition::None,
    )
    .await
    .unwrap();

    let memory = MemorySystem::new()
        .with_dir("/proj/empty")
        .unwrap()
        .with_file("/proj/.hidden", b"dot")
        .unwrap()
        .with_file("/proj/README.md", b"hi\n")
        .unwrap()
        .with_file("/proj/src/main.rs", b"fn main() {}\n")
        .unwrap();

    let mut from_real = Vec::new();
    to_zip_writer(&real, &root, &mut from_real, ZipOptions::default())
        .await
        .unwrap();
    let mut from_memory = Vec::new();
    to_zip_writer(
        &memory,
        Path::new("/proj"),
        &mut from_memory,
        ZipOptions::default(),
    )
    .await
    .unwrap();

    assert_eq!(entry_names(&from_real), entry_names(&from_memory));
    for name in ["README.md", "src/main.rs", ".hidden"] {
        assert_eq!(
            entry_bytes(&from_real, name),
            entry_bytes(&from_memory, name)
        );
    }
}

#[tokio::test(flavor = "current_thread")]
async fn info_zip_reports_a_clean_archive() {
    let system = RealSystem::new();
    let temp = system.create_temp_dir().await.unwrap();
    let root = temp.path().join("proj");
    system.create_dir_all(&root.join("empty")).await.unwrap();
    system.create_dir_all(&root.join("src")).await.unwrap();
    system
        .write(&root.join("README.md"), b"hi\n", Condition::None)
        .await
        .unwrap();
    system
        .write(
            &root.join("src/main.rs"),
            b"fn main() {}\n",
            Condition::None,
        )
        .await
        .unwrap();

    let bundle = temp.path().join("proj.zip");
    let archive = system.to_zip_vec(&root).await.unwrap();
    let mut sink = system.create(&bundle).await.unwrap();
    write_all(&mut *sink, &archive).await.unwrap();
    close(&mut *sink).await.unwrap();

    // `unzip` may not be installed; the reader-based tests cover the archive then.
    let Ok(output) = Command::new("unzip").arg("-t").arg(&bundle).output() else {
        return;
    };
    assert!(output.status.success());
}

#[tokio::test(flavor = "current_thread")]
async fn the_three_zip_shapes_agree_byte_for_byte() {
    let system = memory_project();
    let source = Path::new("/proj");

    let mut from_writer = Vec::new();
    system
        .to_zip_writer(source, &mut from_writer)
        .await
        .unwrap();
    let from_vec = system.to_zip_vec(source).await.unwrap();
    let mut from_stream = Vec::new();
    system
        .to_zip_stream(source)
        .await
        .unwrap()
        .read_to_end(&mut from_stream)
        .unwrap();

    assert_eq!(from_writer, from_vec);
    assert_eq!(from_vec, from_stream);
    assert!(entry_names(&from_vec).contains(&"README.md".to_owned()));
}

#[tokio::test(flavor = "current_thread")]
async fn zip_methods_are_reachable_through_trait_objects() {
    let owned = memory_project();
    let borrowed: &dyn System = &owned;
    let boxed: Box<dyn System> = Box::new(memory_project());

    let from_borrowed = borrowed.to_zip_vec(Path::new("/proj")).await.unwrap();
    let from_boxed = boxed.to_zip_vec(Path::new("/proj")).await.unwrap();

    assert_eq!(entry_names(&from_borrowed), entry_names(&from_boxed));
    assert_eq!(entry_bytes(&from_borrowed, "README.md"), b"hi\n");
    assert_eq!(entry_bytes(&from_boxed, "src/main.rs"), b"fn main() {}\n");
}

#[tokio::test(flavor = "current_thread")]
async fn a_type_with_only_the_required_methods_gains_the_zip_methods() {
    let system = MinimalSystem {
        inner: memory_project(),
    };

    let archive = system.to_zip_vec(Path::new("/proj")).await.unwrap();

    assert!(entry_names(&archive).contains(&"src/main.rs".to_owned()));
    assert_eq!(entry_bytes(&archive, "README.md"), b"hi\n");
}

#[tokio::test(flavor = "current_thread")]
async fn real_temp_dir_archives_its_own_contents() {
    let system = RealSystem::new();
    let temp = system.create_temp_dir().await.unwrap();
    system
        .write(&temp.path().join("inside.txt"), b"kept", Condition::None)
        .await
        .unwrap();

    let mut archive = Vec::new();
    temp.to_zip_stream()
        .await
        .unwrap()
        .read_to_end(&mut archive)
        .unwrap();

    assert_eq!(entry_names(&archive), vec!["inside.txt".to_owned()]);
    assert_eq!(entry_bytes(&archive, "inside.txt"), b"kept");
}

#[tokio::test(flavor = "current_thread")]
async fn memory_temp_dir_archives_its_own_contents() {
    let system = MemorySystem::new();
    let temp = system.create_temp_dir().await.unwrap();
    system
        .write(&temp.path().join("inside.txt"), b"kept", Condition::None)
        .await
        .unwrap();

    let mut archive = Vec::new();
    temp.to_zip_stream()
        .await
        .unwrap()
        .read_to_end(&mut archive)
        .unwrap();

    assert_eq!(entry_names(&archive), vec!["inside.txt".to_owned()]);
    assert_eq!(entry_bytes(&archive, "inside.txt"), b"kept");
}

#[tokio::test(flavor = "current_thread")]
async fn a_handle_without_an_override_reports_unsupported() {
    let handle = PlainTempDir {
        path: PathBuf::from("/nowhere"),
    };

    let kind = handle.to_zip_stream().await.err().map(|error| error.kind());

    assert_eq!(kind, Some(ErrorKind::Unsupported));
}

#[tokio::test(flavor = "current_thread")]
async fn a_real_tree_round_trips_into_an_in_memory_filesystem() {
    let real = RealSystem::new();
    let temp = real.create_temp_dir().await.unwrap();
    let root = temp.path().join("proj");
    real.create_dir_all(&root.join("empty")).await.unwrap();
    real.create_dir_all(&root.join("src")).await.unwrap();
    real.write(&root.join("README.md"), b"hi\n", Condition::None)
        .await
        .unwrap();
    real.write(
        &root.join("src/main.rs"),
        b"fn main() {}\n",
        Condition::None,
    )
    .await
    .unwrap();

    let mut stream = real.to_zip_stream(&root).await.unwrap();
    let loaded = MemorySystem::from_zip_stream(&mut stream).unwrap();

    assert_eq!(
        loaded
            .read_to_string(Path::new("/README.md"))
            .await
            .unwrap(),
        "hi\n"
    );
    assert_eq!(
        loaded
            .read_to_string(Path::new("/src/main.rs"))
            .await
            .unwrap(),
        "fn main() {}\n"
    );
    assert!(loaded.is_dir(Path::new("/empty")).await.unwrap());
}

#[tokio::test(flavor = "current_thread")]
async fn a_loaded_filesystem_can_be_edited_and_archived_again() {
    let source = memory_project();
    let mut stream = source.to_zip_stream(Path::new("/proj")).await.unwrap();
    let loaded = MemorySystem::from_zip_stream(&mut stream).unwrap();

    loaded.remove_dir_all(Path::new("/target")).await.unwrap();
    loaded
        .write(Path::new("/added.txt"), b"new", Condition::None)
        .await
        .unwrap();

    let republished = loaded.to_zip_vec(Path::new("/")).await.unwrap();

    let names = entry_names(&republished);
    assert!(names.contains(&"added.txt".to_owned()));
    assert!(!names.contains(&"target/".to_owned()));
    assert!(!names.contains(&"target/artifact.bin".to_owned()));
    assert_eq!(entry_bytes(&republished, "src/main.rs"), b"fn main() {}\n");
}

#[tokio::test(flavor = "current_thread")]
async fn an_unseekable_pipe_is_a_valid_source() {
    let real = RealSystem::new();
    let temp = real.create_temp_dir().await.unwrap();
    let root = temp.path().join("proj");
    real.create_dir_all(&root.join("src")).await.unwrap();
    real.write(&root.join("README.md"), b"hi\n", Condition::None)
        .await
        .unwrap();
    real.write(
        &root.join("src/main.rs"),
        b"fn main() {}\n",
        Condition::None,
    )
    .await
    .unwrap();

    let (mut reader, mut writer) = pipe().unwrap();
    let producer = thread::spawn(move || {
        let runtime = Builder::new_current_thread().build().unwrap();
        runtime
            .block_on(RealSystem::new().to_zip_writer(&root, &mut writer))
            .unwrap();
        drop(writer);
    });

    let loaded = MemorySystem::from_zip_stream(&mut reader).unwrap();
    producer.join().unwrap();

    assert_eq!(
        loaded
            .read_to_string(Path::new("/README.md"))
            .await
            .unwrap(),
        "hi\n"
    );
    assert_eq!(
        loaded
            .read_to_string(Path::new("/src/main.rs"))
            .await
            .unwrap(),
        "fn main() {}\n"
    );
}

#[test]
fn an_entry_escaping_the_destination_is_refused() {
    let mut buffer = Cursor::new(Vec::<u8>::new());
    let mut writer = ZipWriter::new(&mut buffer);
    writer
        .start_file("../escape.txt", SimpleFileOptions::default())
        .unwrap();
    writer.write_all(b"nope").unwrap();
    writer.finish().unwrap();

    let mut source = Cursor::new(buffer.into_inner());
    let result = MemorySystem::from_zip_stream(&mut source);

    let message = result.err().map(|error| error.to_string()).unwrap();
    assert!(message.contains("escapes the destination"));
    assert!(message.contains("../escape.txt"));
}

/// An archive written to a sink that cannot seek: every file entry's sizes
/// follow its data in a data descriptor, as in files from streaming exporters.
fn data_descriptor_archive() -> Vec<u8> {
    let mut writer = ZipWriter::new_stream(Vec::<u8>::new());
    writer
        .add_directory("word/", SimpleFileOptions::default())
        .unwrap();
    writer
        .start_file("word/document.xml", SimpleFileOptions::default())
        .unwrap();
    writer.write_all(b"<w:document/>").unwrap();
    writer
        .start_file("[Content_Types].xml", SimpleFileOptions::default())
        .unwrap();
    writer.write_all(b"<Types/>").unwrap();
    StreamWriter::into_inner(writer.finish().unwrap())
}

#[test]
fn a_data_descriptor_archive_is_refused_by_the_stream_reader() {
    let archive = data_descriptor_archive();
    // Byte 6 of a local header holds its flags; bit 3 defers sizes, which a directory entry
    // has none to defer, so the check reads the first file's header.
    let header = ZipArchive::new(Cursor::new(archive.as_slice()))
        .unwrap()
        .by_name("word/document.xml")
        .unwrap()
        .header_start();
    let flags = usize::try_from(header)
        .ok()
        .and_then(|start| start.checked_add(6))
        .and_then(|index| archive.get(index).copied())
        .unwrap();
    assert_eq!(
        flags & 0b1000,
        0b1000,
        "the fixture's first file must defer its sizes"
    );

    let message = MemorySystem::from_zip_stream(&mut Cursor::new(archive))
        .err()
        .map(|error| error.to_string())
        .unwrap();

    assert!(message.contains("The file length is not available in the local header"));
}

#[tokio::test(flavor = "current_thread")]
async fn a_data_descriptor_archive_loads_from_a_seekable_reader() {
    let loaded = MemorySystem::from_zip_reader(Cursor::new(data_descriptor_archive())).unwrap();

    assert_eq!(
        loaded
            .read_to_string(Path::new("/word/document.xml"))
            .await
            .unwrap(),
        "<w:document/>"
    );
    assert_eq!(
        loaded
            .read_to_string(Path::new("/[Content_Types].xml"))
            .await
            .unwrap(),
        "<Types/>"
    );
    assert!(loaded.is_dir(Path::new("/word")).await.unwrap());
}

/// The repeated text of the large entry in the streamed archive.
fn streamed_large_entry() -> String {
    "<w:p>row</w:p>".repeat(0x1_0000)
}

/// Three entries, one empty, one of about a mebibyte and one small, written through
/// a `ZipStreamWriter`.
fn streamed_archive() -> Vec<u8> {
    let mut zip = ZipStreamWriter::new(Vec::<u8>::new());
    zip.start_entry("empty.xml").unwrap();
    zip.start_entry("word/large.xml").unwrap();
    zip.write_all(streamed_large_entry().as_bytes()).unwrap();
    zip.start_entry("[Content_Types].xml").unwrap();
    zip.write_all(b"<Types/>").unwrap();
    zip.finish().unwrap()
}

/// The two bytes at `offset` in the local header of every entry.
fn local_header_bytes(archive: &[u8], offset: usize) -> Vec<Vec<u8>> {
    let mut reader = ZipArchive::new(Cursor::new(archive.to_vec())).unwrap();
    (0..reader.len())
        .map(|index| {
            let start = usize::try_from(reader.by_index(index).unwrap().header_start())
                .unwrap()
                .checked_add(offset)
                .unwrap();
            archive
                .get(start..start.checked_add(2).unwrap())
                .map(<[u8]>::to_vec)
                .unwrap()
        })
        .collect()
}

#[tokio::test(flavor = "current_thread")]
async fn a_streamed_archive_loads_with_every_entry_intact() {
    let loaded = MemorySystem::from_zip_reader(Cursor::new(streamed_archive())).unwrap();

    assert_eq!(
        loaded
            .read_to_string(Path::new("/empty.xml"))
            .await
            .unwrap(),
        ""
    );
    assert_eq!(
        loaded
            .read_to_string(Path::new("/word/large.xml"))
            .await
            .unwrap(),
        streamed_large_entry()
    );
    assert_eq!(
        loaded
            .read_to_string(Path::new("/[Content_Types].xml"))
            .await
            .unwrap(),
        "<Types/>"
    );
}

#[test]
fn every_streamed_entry_defers_its_sizes_to_a_data_descriptor() {
    let flags = local_header_bytes(&streamed_archive(), 6);

    assert_eq!(flags.len(), 3);
    for word in flags {
        assert_eq!(word.first().copied().unwrap() & 0b1000, 0b1000);
    }
}

#[test]
fn every_streamed_entry_is_deflated() {
    let methods = local_header_bytes(&streamed_archive(), 8);

    assert_eq!(methods, [[8, 0], [8, 0], [8, 0]]);
}

#[test]
fn a_streamed_archive_is_refused_by_the_stream_reader() {
    let message = MemorySystem::from_zip_stream(&mut Cursor::new(streamed_archive()))
        .err()
        .map(|error| error.to_string())
        .unwrap();

    assert!(message.contains("The file length is not available in the local header"));
}

#[test]
fn a_streamed_archive_is_written_to_a_sink_that_only_writes() {
    let mut zip = ZipStreamWriter::new(CountingSink { count: 0 });
    zip.start_entry("word/document.xml").unwrap();
    zip.write_all(b"<w:document/>").unwrap();

    let sink = zip.finish().unwrap();

    assert!(sink.count > b"<w:document/>".len());
}

#[tokio::test(flavor = "current_thread")]
async fn a_streamed_archive_with_no_entries_loads_as_empty() {
    let bytes = ZipStreamWriter::new(Vec::<u8>::new()).finish().unwrap();

    let loaded = MemorySystem::from_zip_reader(Cursor::new(bytes)).unwrap();

    assert_eq!(loaded.read_dir(Path::new("/")).await.unwrap().len(), 0);
}

#[tokio::test(flavor = "current_thread")]
async fn the_seekable_reader_loads_what_the_writer_produces() {
    let archive = memory_project()
        .to_zip_vec(Path::new("/proj"))
        .await
        .unwrap();

    let from_reader = MemorySystem::from_zip_reader(Cursor::new(archive.clone()))
        .unwrap()
        .to_zip_vec(Path::new("/"))
        .await
        .unwrap();
    let from_stream = MemorySystem::from_zip_stream(&mut Cursor::new(archive.clone()))
        .unwrap()
        .to_zip_vec(Path::new("/"))
        .await
        .unwrap();

    // Names and contents only: a loaded file is stamped with its load time.
    for reloaded in [&from_reader, &from_stream] {
        assert_eq!(entry_names(reloaded), entry_names(&archive));
        for name in entry_names(&archive) {
            assert_eq!(entry_bytes(reloaded, &name), entry_bytes(&archive, &name));
        }
    }
}

#[test]
fn the_seekable_reader_refuses_an_entry_escaping_the_destination() {
    let mut buffer = Cursor::new(Vec::<u8>::new());
    let mut writer = ZipWriter::new(&mut buffer);
    writer
        .start_file("../escape.txt", SimpleFileOptions::default())
        .unwrap();
    writer.write_all(b"nope").unwrap();
    writer.finish().unwrap();

    let message = MemorySystem::from_zip_reader(Cursor::new(buffer.into_inner()))
        .err()
        .map(|error| error.to_string())
        .unwrap();

    assert!(message.contains("escapes the destination"));
    assert!(message.contains("../escape.txt"));
}

#[test]
fn bytes_that_are_not_an_archive_report_an_error_from_a_seekable_reader() {
    let source = Cursor::new(b"this is not a zip archive at all".to_vec());

    assert!(MemorySystem::from_zip_reader(source).is_err());
}

#[test]
fn bytes_that_are_not_an_archive_report_an_error() {
    let mut source = Cursor::new(b"this is not a zip archive at all".to_vec());

    assert!(MemorySystem::from_zip_stream(&mut source).is_err());
}

#[tokio::test(flavor = "current_thread")]
async fn every_file_entry_carries_a_zip64_local_header() {
    // The zip64 extra field lets an entry declare a length past 0xFFFFFFFF; `file_options`
    // sets `large_file(true)`, so every file carries it.
    let system = memory_project();
    let mut archive = Vec::new();
    to_zip_writer(
        &system,
        Path::new("/proj"),
        &mut archive,
        ZipOptions::default(),
    )
    .await
    .unwrap();

    let files: Vec<String> = entry_names(&archive)
        .into_iter()
        .filter(|name| !name.ends_with('/'))
        .collect();
    assert_eq!(files.len(), 4);
    for name in files {
        let extra = local_extra_field(&archive, &name);
        assert!(
            extra.starts_with(&ZIP64_EXTRA_FIELD_HEADER),
            "entry {name} has no zip64 extra field in its local header: {extra:?}"
        );
    }
}

#[tokio::test(flavor = "current_thread")]
async fn directory_entries_stay_off_the_zip64_path() {
    // `add_directory` never revisits its local header, so a zip64 extra field there keeps
    // `u64::MAX` placeholder sizes and a streaming reader waits for data that never comes.
    let system = memory_project();
    let mut archive = Vec::new();
    to_zip_writer(
        &system,
        Path::new("/proj"),
        &mut archive,
        ZipOptions::default(),
    )
    .await
    .unwrap();

    let directories: Vec<String> = entry_names(&archive)
        .into_iter()
        .filter(|name| name.ends_with('/'))
        .collect();
    assert_eq!(directories.len(), 4);
    for name in directories {
        assert_eq!(local_extra_field(&archive, &name), Vec::new());
    }
}

/// The digest at the head of a `sha256sum` output line.
#[cfg(slow_tests)]
fn digest_of(output: &[u8]) -> String {
    let text = String::from_utf8(output.to_vec()).unwrap();
    text.split_whitespace().next().unwrap().to_owned()
}

/// SHA-256 of the entry `name` as Info-ZIP's `unzip -p` streams it out.
#[cfg(slow_tests)]
fn sha256_of_extracted_entry(archive: &Path, name: &str) -> String {
    let mut extractor = Command::new("unzip")
        .arg("-p")
        .arg(archive)
        .arg(name)
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let stream = extractor.stdout.take().unwrap();
    let digest = Command::new("sha256sum")
        .stdin(Stdio::from(stream))
        .output()
        .unwrap();
    assert!(
        extractor.wait().unwrap().success(),
        "unzip -p could not extract {name}"
    );
    assert!(
        digest.status.success(),
        "sha256sum failed on the extracted entry"
    );
    digest_of(&digest.stdout)
}

/// SHA-256 of the file at `path`, as `sha256sum` computes it.
#[cfg(slow_tests)]
fn sha256_of_file(path: &Path) -> String {
    let output = Command::new("sha256sum").arg(path).output().unwrap();
    assert!(
        output.status.success(),
        "sha256sum failed on {}",
        path.display()
    );
    digest_of(&output.stdout)
}

/// Write `bytes` of `/dev/urandom` to `path`, streaming through a fixed buffer.
///
/// Random data, because deflate must not be able to hide four gibibytes behind a
/// small compressed length: that would leave the zip64 path untouched.
#[cfg(slow_tests)]
async fn write_incompressible_file(system: &RealSystem, path: &Path, bytes: u64) {
    let mut entropy = system.open(Path::new("/dev/urandom")).await.unwrap();
    let mut sink = system.create(path).await.unwrap();
    let mut buffer = vec![0_u8; LARGE_ENTRY_CHUNK_BYTES];
    let mut remaining = bytes;
    while remaining != 0 {
        let want = usize::try_from(remaining)
            .unwrap_or(LARGE_ENTRY_CHUNK_BYTES)
            .min(LARGE_ENTRY_CHUNK_BYTES);
        let (slot, _rest) = buffer.split_at_mut(want);
        let mut filled = 0;
        while filled < want {
            let (_done, tail) = slot.split_at_mut(filled);
            let count = read_chunk(&mut *entropy, tail).await.unwrap();
            assert_ne!(count, 0, "/dev/urandom ended");
            filled = filled.saturating_add(count);
        }
        write_all(&mut *sink, slot).await.unwrap();
        remaining = remaining.saturating_sub(u64::try_from(want).unwrap());
    }
    close(&mut *sink).await.unwrap();
}

/// An entry past 32 bits archives with zip64 and survives the round trip byte for byte.
///
/// Needs a disk-backed `TMPDIR` and double-digit gibibytes of headroom.
#[cfg(slow_tests)]
#[tokio::test(flavor = "current_thread")]
async fn an_entry_above_four_gibibytes_archives_and_extracts() {
    let real = RealSystem::new();
    let temp = real.create_temp_dir().await.unwrap();
    let source = temp.path().join("big.bin");
    write_incompressible_file(&real, &source, LARGE_ENTRY_BYTES).await;
    assert_eq!(real.metadata(&source).await.unwrap().len, LARGE_ENTRY_BYTES);

    let destination = temp.path().join("big.zip");
    let mut sink = File::create(&destination).unwrap();
    to_zip_writer(&real, &source, &mut sink, ZipOptions::default())
        .await
        .unwrap();
    drop(sink);

    let tested = Command::new("unzip")
        .arg("-t")
        .arg(&destination)
        .output()
        .unwrap();
    assert!(
        tested.status.success(),
        "unzip -t rejected the archive: {}{}",
        String::from_utf8_lossy(&tested.stdout),
        String::from_utf8_lossy(&tested.stderr)
    );

    assert_eq!(
        sha256_of_extracted_entry(&destination, "big.bin"),
        sha256_of_file(&source)
    );
}

#[tokio::test(flavor = "current_thread")]
async fn minimal_system_passes_sleep_to_its_inner_system() {
    let inner = MemorySystem::new();
    let system = MinimalSystem { inner };

    system.sleep(Duration::from_secs(2)).await.unwrap();

    assert_eq!(
        system.now(),
        SystemTime::UNIX_EPOCH
            .checked_add(Duration::from_secs(2))
            .unwrap()
    );
}
