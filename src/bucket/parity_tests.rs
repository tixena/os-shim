//! Parity of `MemoryBucket` with a real server: each sequence runs against both
//! and the two transcripts must be equal. `minio_tests` runs them against a
//! MinIO-compatible server, `s3_tests` against an AWS bucket.

use super::{BlockingBucket, Bucket, Condition, Depth, MemoryBucket, RevisionToken};
#[cfg(minio_tests)]
use super::{MinioBucket, MinioConfig};
#[cfg(s3_tests)]
use super::{S3Bucket, S3Config};
use crate::error::Result;
#[cfg(minio_tests)]
use crate::error::{ErrorKind, StorageError};
use crate::{FileMetadata, WalkEntry};
#[cfg(any(minio_tests, s3_tests))]
use aws_sdk_s3::primitives::ByteStream;
#[cfg(s3_tests)]
use aws_sdk_s3::types::BucketLocationConstraint;
use aws_sdk_s3::{
    Client, Config,
    config::{BehaviorVersion, Credentials, Region, retry::RetryConfig},
    error::SdkError,
};
use core::result;
#[cfg(minio_tests)]
use futures_util::{AsyncReadExt as _, AsyncWriteExt as _};
use std::env;
use std::path::{Path, PathBuf};
use std::process;
#[cfg(minio_tests)]
use std::sync::OnceLock;
#[cfg(minio_tests)]
use std::thread;
use tokio::runtime::{Builder, Runtime};

#[cfg(minio_tests)]
/// The server's address and keys from the environment, for the bucket `os-shim-tests`.
fn config() -> MinioConfig {
    MinioConfig::new(
        env::var("OS_SHIM_MINIO_ENDPOINT").unwrap(),
        "us-east-1".to_owned(),
        "os-shim-tests".to_owned(),
        env::var("OS_SHIM_MINIO_ACCESS_KEY").unwrap(),
        env::var("OS_SHIM_MINIO_SECRET_KEY").unwrap(),
    )
}

#[cfg(minio_tests)]
/// An SDK client configured the way `MinioBucket::new` configures its own.
fn client(config: &MinioConfig) -> Client {
    let credentials = Credentials::new(
        config.access_key.clone(),
        config.secret_key.clone(),
        None,
        None,
        "os-shim",
    );
    let sdk = Config::builder()
        .behavior_version(BehaviorVersion::latest())
        .region(Region::new(config.region.clone()))
        .credentials_provider(credentials)
        .retry_config(RetryConfig::disabled())
        .endpoint_url(config.endpoint.clone())
        .force_path_style(true)
        .build();
    Client::from_conf(sdk)
}

/// A runtime for the tests that call the SDK directly.
fn runtime() -> Runtime {
    Builder::new_current_thread().enable_all().build().unwrap()
}

#[cfg(minio_tests)]
/// The configuration of a server whose bucket exists, made by `ensure` once per test binary on a thread of its own.
fn ready_config() -> MinioConfig {
    static CREATED: OnceLock<()> = OnceLock::new();
    let config = config();
    CREATED.get_or_init(|| {
        thread::scope(|scope| {
            scope
                .spawn(|| {
                    runtime()
                        .block_on(MinioBucket::new(config.clone()).ensure())
                        .unwrap();
                })
                .join()
                .unwrap();
        });
    });
    config
}

/// The memory side of a parity test.
fn memory() -> BlockingBucket<MemoryBucket> {
    BlockingBucket::new(MemoryBucket::default()).unwrap()
}

#[cfg(minio_tests)]
/// The server side of a parity test.
fn server() -> BlockingBucket<MinioBucket> {
    BlockingBucket::new(MinioBucket::new(ready_config())).unwrap()
}

/// What an answer looks like with its token hidden: `tag`, the shown value, or the kind of its error.
fn shape<T>(answer: &Result<T>, show: impl Fn(&T) -> String) -> String {
    match answer.as_ref() {
        Ok(value) => show(value),
        Err(error) => format!("{:?}", error.kind()),
    }
}

/// The line for a put's answer.
fn put_line(answer: &Result<RevisionToken>) -> String {
    shape(answer, |_| "tag".to_owned())
}

/// The line for a get's answer.
fn get_line(answer: &Result<(Vec<u8>, RevisionToken)>) -> String {
    shape(answer, |found| format!("bytes {:?}", found.0))
}

/// The line for a head's answer.
fn head_line(answer: &Result<FileMetadata>) -> String {
    shape(answer, |head| {
        format!(
            "file {} dir {} size {} modified {} token {}",
            head.is_file,
            head.is_dir,
            head.len,
            head.modified.is_some(),
            head.revision.is_some()
        )
    })
}

/// The line for a delete's answer.
fn delete_line(answer: &Result<()>) -> String {
    shape(answer, |&()| "deleted".to_owned())
}

/// The line for a head's answer without the modification time, which a folder has only on a server.
fn kind_line(answer: &Result<FileMetadata>) -> String {
    shape(answer, |head| {
        format!(
            "file {} dir {} size {} token {}",
            head.is_file,
            head.is_dir,
            head.len,
            head.revision.is_some()
        )
    })
}

/// The line for the answer of an operation that gives nothing back.
fn done_line(answer: &Result<()>) -> String {
    shape(answer, |&()| "done".to_owned())
}

/// The line for a copy's answer.
fn copy_line(answer: &Result<u64>) -> String {
    shape(answer, |length| format!("copied {length} bytes"))
}

/// The path of `name` in the folder `base`.
fn at(base: &str, name: &str) -> PathBuf {
    PathBuf::from(format!("{base}/{name}"))
}

/// The line for a listing's answer: the paths below `base`, sorted, folders ending in `/`.
fn list_line(answer: &Result<Vec<WalkEntry>>, base: &str) -> String {
    shape(answer, |entries| {
        let mut shown: Vec<String> = entries
            .iter()
            .map(|entry| {
                let relative = entry.path.strip_prefix(base).unwrap_or(&entry.path);
                format!(
                    "{}{}",
                    relative.display(),
                    if entry.is_dir { "/" } else { "" }
                )
            })
            .collect();
        shown.sort();
        format!("paths {shown:?}")
    })
}

/// Print the server's transcript and assert it equals the memory one.
fn assert_agree(name: &str, memory_transcript: &[String], server_transcript: &[String]) {
    eprintln!("{name} on the server:\n{}", server_transcript.join("\n"));
    assert_eq!(memory_transcript, server_transcript);
}

/// Create the folder.
fn ensure_folder<B>(bucket: &BlockingBucket<B>, folder: &str)
where
    B: Bucket,
{
    bucket.create_folder(Path::new(folder)).unwrap();
}

/// A put, a get and the token the put returned.
fn put_and_get<B>(bucket: &BlockingBucket<B>, base: &str) -> Vec<String>
where
    B: Bucket,
{
    ensure_folder(bucket, base);
    let path = PathBuf::from(format!("{base}/note.md"));
    let mut transcript = Vec::new();
    let put = bucket.put(&path, b"# Note\n", &Condition::None);
    transcript.push(put_line(&put));
    let got = bucket.get(&path);
    transcript.push(get_line(&got));
    transcript.push(format!(
        "get token equals put token: {}",
        got.unwrap().1 == put.unwrap()
    ));
    transcript
}

/// A head of a path that holds nothing, then of the same path after a put, against the put's token.
fn head_after_put<B>(bucket: &BlockingBucket<B>, base: &str) -> Vec<String>
where
    B: Bucket,
{
    ensure_folder(bucket, base);
    let path = PathBuf::from(format!("{base}/note.md"));
    let missing = bucket.head(&path);
    let put = bucket.put(&path, b"12345", &Condition::None);
    let found = bucket.head(&path);
    vec![
        head_line(&missing),
        put_line(&put),
        head_line(&found),
        format!(
            "head token equals put token: {}",
            found.unwrap().revision == Some(put.unwrap())
        ),
    ]
}

/// Puts and deletes under `Absent` and under current and stale tokens.
fn conditional_writes<B>(bucket: &BlockingBucket<B>, base: &str) -> Vec<String>
where
    B: Bucket,
{
    ensure_folder(bucket, base);
    let path = PathBuf::from(format!("{base}/task.md"));
    let mut transcript = Vec::new();
    let created = bucket.put(&path, b"# Task\n", &Condition::Absent);
    transcript.push(put_line(&created));
    transcript.push(put_line(&bucket.put(
        &path,
        b"# Other\n",
        &Condition::Absent,
    )));
    let first = created.unwrap();
    let replaced = bucket.put(&path, b"# Done\n", &Condition::Matches(first.clone()));
    transcript.push(put_line(&replaced));
    transcript.push(put_line(&bucket.put(
        &path,
        b"# Late\n",
        &Condition::Matches(first.clone()),
    )));
    transcript.push(get_line(&bucket.get(&path)));
    transcript.push(delete_line(
        &bucket.delete(&path, &Condition::Matches(first)),
    ));
    transcript.push(delete_line(
        &bucket.delete(&path, &Condition::Matches(replaced.unwrap())),
    ));
    transcript.push(get_line(&bucket.get(&path)));
    transcript
}

/// Deletes of a path that holds nothing, under each condition.
fn missing_path_deletes<B>(bucket: &BlockingBucket<B>, base: &str) -> Vec<String>
where
    B: Bucket,
{
    ensure_folder(bucket, base);
    let path = PathBuf::from(format!("{base}/none.md"));
    vec![
        delete_line(&bucket.delete(&path, &Condition::None)),
        delete_line(&bucket.delete(
            &path,
            &Condition::Matches(RevisionToken::new("\"0123456789abcdef\"")),
        )),
        delete_line(&bucket.delete(&path, &Condition::Absent)),
    ]
}

/// A delete of a path that holds a file, under `Absent`, and what is left.
fn absent_delete_over_a_file<B>(bucket: &BlockingBucket<B>, base: &str) -> Vec<String>
where
    B: Bucket,
{
    ensure_folder(bucket, base);
    let path = PathBuf::from(format!("{base}/kept.md"));
    let put = bucket.put(&path, b"kept", &Condition::None);
    vec![
        put_line(&put),
        delete_line(&bucket.delete(&path, &Condition::Absent)),
        get_line(&bucket.get(&path)),
    ]
}

/// A put under a stale token at a path that holds nothing, and a get of that path.
fn missing_path_tokened_put<B>(bucket: &BlockingBucket<B>, base: &str) -> Vec<String>
where
    B: Bucket,
{
    ensure_folder(bucket, base);
    let path = PathBuf::from(format!("{base}/none.md"));
    vec![
        put_line(&bucket.put(
            &path,
            b"# Task\n",
            &Condition::Matches(RevisionToken::new("\"stale\"")),
        )),
        get_line(&bucket.get(&path)),
    ]
}

/// Tokens of different bytes at one path, and of the same bytes at two paths.
fn rewritten_tokens<B>(bucket: &BlockingBucket<B>, base: &str) -> Vec<String>
where
    B: Bucket,
{
    ensure_folder(bucket, base);
    let one = bucket.put(
        &PathBuf::from(format!("{base}/one.md")),
        b"first",
        &Condition::None,
    );
    let again = bucket.put(
        &PathBuf::from(format!("{base}/one.md")),
        b"second",
        &Condition::None,
    );
    let one_token = one.unwrap();
    vec![format!(
        "rewrite changes the token: {}",
        one_token != again.unwrap()
    )]
}

/// A listing of 1,001 files, more than one page holds.
fn long_listing<B>(bucket: &BlockingBucket<B>, base: &str) -> Vec<String>
where
    B: Bucket,
{
    ensure_folder(bucket, base);
    let expected: Vec<String> = (0_u32..=1000_u32)
        .map(|index| format!("{base}/{index:04}"))
        .collect();
    for path in &expected {
        bucket.put(Path::new(path), b"x", &Condition::None).unwrap();
    }
    let listed = bucket.list(Path::new(base), Depth::All).unwrap();
    let count = listed.len();
    let mut paths: Vec<String> = listed
        .iter()
        .map(|entry| entry.path.display().to_string())
        .collect();
    paths.sort();
    vec![
        format!("entries {count}"),
        format!("every path once: {}", paths == expected),
    ]
}

/// A listing below a folder nothing is in.
fn empty_listing<B>(bucket: &BlockingBucket<B>, base: &str) -> Vec<String>
where
    B: Bucket,
{
    ensure_folder(bucket, base);
    let folder = format!("{base}/nothing-here");
    vec![list_line(
        &bucket.list(Path::new(&folder), Depth::All),
        &folder,
    )]
}

/// Children of a folder with a file and a subfolder holding a file.
fn children_listing<B>(bucket: &BlockingBucket<B>, base: &str) -> Vec<String>
where
    B: Bucket,
{
    ensure_folder(bucket, &format!("{base}/sub"));
    bucket
        .put(
            &PathBuf::from(format!("{base}/top.md")),
            b"t",
            &Condition::None,
        )
        .unwrap();
    bucket
        .put(
            &PathBuf::from(format!("{base}/sub/deep.md")),
            b"d",
            &Condition::None,
        )
        .unwrap();
    vec![list_line(
        &bucket.list(Path::new(base), Depth::Children),
        base,
    )]
}

/// Folders made with their parents, headed, listed and removed.
fn folders<B>(bucket: &BlockingBucket<B>, base: &str) -> Vec<String>
where
    B: Bucket,
{
    let nested = at(base, "x/y");
    let file = at(base, "x/y/f.md");
    let mut transcript = vec![done_line(&bucket.create_folder(&nested))];
    transcript.push(kind_line(&bucket.head(&at(base, "x"))));
    transcript.push(kind_line(&bucket.head(&nested)));
    transcript.push(put_line(&bucket.put(&file, b"f", &Condition::Absent)));
    transcript.push(list_line(&bucket.list(Path::new(base), Depth::All), base));
    transcript.push(list_line(
        &bucket.list(&at(base, "x"), Depth::Children),
        &format!("{base}/x"),
    ));
    transcript.push(done_line(&bucket.remove_folder(&at(base, "x"))));
    transcript.push(kind_line(&bucket.head(&at(base, "x"))));
    transcript.push(done_line(&bucket.remove_folder(&at(base, "x"))));
    transcript.push(done_line(&bucket.remove_folder(&file)));
    drop(bucket.remove_folder(Path::new(base)));
    transcript
}

/// A file where a folder is wanted, and a folder where a file is wanted.
fn file_and_folder_conflict<B>(bucket: &BlockingBucket<B>, base: &str) -> Vec<String>
where
    B: Bucket,
{
    ensure_folder(bucket, base);
    let file = at(base, "a");
    let mut transcript = vec![put_line(&bucket.put(&file, b"file", &Condition::None))];
    transcript.push(put_line(&bucket.put(
        &at(base, "a/b"),
        b"under",
        &Condition::None,
    )));
    transcript.push(done_line(&bucket.create_folder(&at(base, "a/b"))));
    transcript.push(done_line(&bucket.create_folder(&file)));
    transcript.push(list_line(&bucket.list(&file, Depth::Children), base));
    transcript.push(list_line(&bucket.list(&file, Depth::All), base));
    transcript.push(done_line(&bucket.remove_folder(&file)));
    transcript.push(kind_line(&bucket.head(&file)));
    ensure_folder(bucket, &format!("{base}/c"));
    transcript.push(put_line(&bucket.put(
        &at(base, "c"),
        b"file",
        &Condition::None,
    )));
    transcript.push(kind_line(&bucket.head(&at(base, "c"))));
    transcript.push(list_line(
        &bucket.list(Path::new(base), Depth::Children),
        base,
    ));
    drop(bucket.remove_folder(Path::new(base)));
    transcript
}

/// Copies of a file, of a missing file, of a folder, and into a missing folder and over a folder.
fn copies<B>(bucket: &BlockingBucket<B>, base: &str) -> Vec<String>
where
    B: Bucket,
{
    ensure_folder(bucket, &format!("{base}/dir"));
    let source = at(base, "source.md");
    let mut transcript = vec![put_line(&bucket.put(&source, b"hello", &Condition::None))];
    let copied = bucket.copy(&source, &at(base, "copy.md"));
    transcript.push(copy_line(&copied));
    transcript.push(get_line(&bucket.get(&at(base, "copy.md"))));
    transcript.push(get_line(&bucket.get(&source)));
    transcript.push(copy_line(
        &bucket.copy(&at(base, "nothing.md"), &at(base, "to.md")),
    ));
    transcript.push(copy_line(&bucket.copy(&source, &at(base, "missing/to.md"))));
    transcript.push(copy_line(&bucket.copy(&source, &at(base, "dir"))));
    transcript.push(copy_line(
        &bucket.copy(&at(base, "dir"), &at(base, "to.md")),
    ));
    drop(bucket.remove_folder(Path::new(base)));
    transcript
}

/// Renames of a file: over nothing, over a file, of a missing file, into a missing folder, over a folder.
fn file_renames<B>(bucket: &BlockingBucket<B>, base: &str) -> Vec<String>
where
    B: Bucket,
{
    ensure_folder(bucket, &format!("{base}/dir"));
    let first = at(base, "first.md");
    let second = at(base, "second.md");
    let mut transcript = vec![put_line(&bucket.put(&first, b"one", &Condition::None))];
    transcript.push(done_line(&bucket.rename(&first, &second)));
    transcript.push(get_line(&bucket.get(&first)));
    transcript.push(get_line(&bucket.get(&second)));
    transcript.push(put_line(&bucket.put(&first, b"two", &Condition::None)));
    transcript.push(done_line(&bucket.rename(&first, &second)));
    transcript.push(get_line(&bucket.get(&second)));
    transcript.push(done_line(&bucket.rename(&first, &second)));
    transcript.push(put_line(&bucket.put(&first, b"three", &Condition::None)));
    transcript.push(done_line(
        &bucket.rename(&first, &at(base, "missing/to.md")),
    ));
    transcript.push(done_line(&bucket.rename(&first, &at(base, "dir"))));
    transcript.push(get_line(&bucket.get(&first)));
    drop(bucket.remove_folder(Path::new(base)));
    transcript
}

/// A rename of a folder with files, a subfolder and an empty folder in it.
fn folder_rename<B>(bucket: &BlockingBucket<B>, base: &str) -> Vec<String>
where
    B: Bucket,
{
    ensure_folder(bucket, &format!("{base}/s/inner/empty"));
    for name in ["s/one.md", "s/two.md", "s/inner/three.md"] {
        bucket
            .put(&at(base, name), name.as_bytes(), &Condition::None)
            .unwrap();
    }
    let mut transcript = vec![done_line(&bucket.rename(&at(base, "s"), &at(base, "t")))];
    transcript.push(kind_line(&bucket.head(&at(base, "s"))));
    transcript.push(kind_line(&bucket.head(&at(base, "t"))));
    transcript.push(list_line(&bucket.list(Path::new(base), Depth::All), base));
    transcript.push(get_line(&bucket.get(&at(base, "t/inner/three.md"))));
    transcript.push(done_line(
        &bucket.rename(&at(base, "nothing"), &at(base, "u")),
    ));
    transcript.push(done_line(
        &bucket.rename(&at(base, "t"), &at(base, "t/inside")),
    ));
    drop(bucket.remove_folder(Path::new(base)));
    transcript
}

/// Puts and appends whose parent folder is missing, is a file, or exists.
fn parent_check<B>(bucket: &BlockingBucket<B>, base: &str) -> Vec<String>
where
    B: Bucket,
{
    ensure_folder(bucket, base);
    let mut transcript = vec![put_line(&bucket.put(
        &at(base, "nothing/f.md"),
        b"x",
        &Condition::None,
    ))];
    transcript.push(put_line(&bucket.append(&at(base, "nothing/f.md"), b"x")));
    transcript.push(put_line(&bucket.put(
        &at(base, "plain"),
        b"x",
        &Condition::None,
    )));
    transcript.push(put_line(&bucket.put(
        &at(base, "plain/f.md"),
        b"x",
        &Condition::None,
    )));
    transcript.push(put_line(&bucket.append(&at(base, "plain/f.md"), b"x")));
    transcript.push(put_line(&bucket.append(&at(base, "log.md"), b"one")));
    transcript.push(put_line(&bucket.append(&at(base, "log.md"), b" two")));
    transcript.push(get_line(&bucket.get(&at(base, "log.md"))));
    drop(bucket.remove_folder(Path::new(base)));
    transcript
}

/// Relative paths and `..` above the root are refused before any request.
fn refused_paths<B>(bucket: &BlockingBucket<B>) -> Vec<String>
where
    B: Bucket,
{
    vec![
        get_line(&bucket.get(Path::new("relative/note.md"))),
        get_line(&bucket.get(Path::new("/../note.md"))),
    ]
}

/// The status an SDK answer carried, or `accepted` when it succeeded.
fn status<T, E>(answer: &result::Result<T, SdkError<E>>) -> String {
    match answer.as_ref() {
        Ok(_) => "accepted".to_owned(),
        Err(error) => format!(
            "status {:?}",
            error
                .raw_response()
                .map(|response| response.status().as_u16())
        ),
    }
}

#[cfg(minio_tests)]
#[test]
fn put_and_get_agrees_with_the_server() {
    assert_agree(
        "put and get",
        &put_and_get(&memory(), "/a"),
        &put_and_get(&server(), "/put-and-get"),
    );
}

#[cfg(minio_tests)]
#[test]
fn head_after_put_agrees_with_the_server() {
    assert_agree(
        "head after put",
        &head_after_put(&memory(), "/a"),
        &head_after_put(&server(), "/head-after-put"),
    );
}

#[cfg(minio_tests)]
#[test]
fn conditional_writes_agrees_with_the_server() {
    assert_agree(
        "conditional writes",
        &conditional_writes(&memory(), "/a"),
        &conditional_writes(&server(), "/conditional-writes"),
    );
}

#[cfg(minio_tests)]
#[test]
fn missing_path_deletes_agrees_with_the_server() {
    assert_agree(
        "missing path deletes",
        &missing_path_deletes(&memory(), "/a"),
        &missing_path_deletes(&server(), "/missing-path-deletes"),
    );
}

#[cfg(minio_tests)]
#[test]
fn absent_delete_over_a_file_agrees_with_the_server() {
    assert_agree(
        "absent delete over a file",
        &absent_delete_over_a_file(&memory(), "/a"),
        &absent_delete_over_a_file(&server(), "/absent-delete-over-a-file"),
    );
}

#[cfg(minio_tests)]
#[test]
fn missing_path_tokened_put_agrees_with_the_server() {
    assert_agree(
        "missing path tokened put",
        &missing_path_tokened_put(&memory(), "/a"),
        &missing_path_tokened_put(&server(), "/missing-path-tokened-put"),
    );
}

#[cfg(minio_tests)]
#[test]
fn rewritten_tokens_agrees_with_the_server() {
    assert_agree(
        "rewritten tokens",
        &rewritten_tokens(&memory(), "/a"),
        &rewritten_tokens(&server(), "/rewritten-tokens"),
    );
}

#[cfg(minio_tests)]
#[test]
fn long_listing_agrees_with_the_server() {
    assert_agree(
        "long listing",
        &long_listing(&memory(), "/a"),
        &long_listing(&server(), "/long-listing"),
    );
}

#[cfg(minio_tests)]
#[test]
fn empty_listing_agrees_with_the_server() {
    assert_agree(
        "empty listing",
        &empty_listing(&memory(), "/a"),
        &empty_listing(&server(), "/empty-listing"),
    );
}

#[cfg(minio_tests)]
#[test]
fn children_listing_agrees_with_the_server() {
    assert_agree(
        "children listing",
        &children_listing(&memory(), "/a"),
        &children_listing(&server(), "/children-listing"),
    );
}

#[cfg(minio_tests)]
#[test]
fn folders_agrees_with_the_server() {
    assert_agree(
        "folders",
        &folders(&memory(), "/a"),
        &folders(&server(), "/folders"),
    );
}

#[cfg(minio_tests)]
#[test]
fn file_and_folder_conflict_agrees_with_the_server() {
    assert_agree(
        "file and folder conflict",
        &file_and_folder_conflict(&memory(), "/a"),
        &file_and_folder_conflict(&server(), "/file-and-folder-conflict"),
    );
}

#[cfg(minio_tests)]
#[test]
fn copies_agrees_with_the_server() {
    assert_agree(
        "copies",
        &copies(&memory(), "/a"),
        &copies(&server(), "/copies"),
    );
}

#[cfg(minio_tests)]
#[test]
fn file_renames_agrees_with_the_server() {
    assert_agree(
        "file renames",
        &file_renames(&memory(), "/a"),
        &file_renames(&server(), "/file-renames"),
    );
}

#[cfg(minio_tests)]
#[test]
fn folder_rename_agrees_with_the_server() {
    assert_agree(
        "folder rename",
        &folder_rename(&memory(), "/a"),
        &folder_rename(&server(), "/folder-rename"),
    );
}

#[cfg(minio_tests)]
#[test]
fn parent_check_agrees_with_the_server() {
    assert_agree(
        "parent check",
        &parent_check(&memory(), "/a"),
        &parent_check(&server(), "/parent-check"),
    );
}

#[cfg(minio_tests)]
#[test]
fn refused_paths_agree_with_the_server() {
    assert_agree(
        "refused paths",
        &refused_paths(&memory()),
        &refused_paths(&server()),
    );
}

#[cfg(minio_tests)]
#[test]
fn observe_a_raw_delete_with_a_wrong_if_match() {
    let config = ready_config();
    let bucket = server();
    let key = "observe-raw-delete/note.md";
    let path = Path::new("/observe-raw-delete/note.md");
    ensure_folder(&bucket, "/observe-raw-delete");
    bucket.put(path, b"kept", &Condition::None).unwrap();
    let answer = runtime().block_on(
        client(&config)
            .delete_object()
            .bucket(&config.bucket)
            .key(key)
            .if_match("\"0123456789abcdef0123456789abcdef\"")
            .send(),
    );
    eprintln!("raw delete with a wrong If-Match: {}", status(&answer));
    eprintln!("object afterwards: {}", get_line(&bucket.get(path)));
}

#[cfg(minio_tests)]
#[test]
fn observe_a_put_with_the_tag_stripped_of_its_quotes() {
    let bucket = server();
    let path = Path::new("/observe-unquoted-tag/note.md");
    ensure_folder(&bucket, "/observe-unquoted-tag");
    let tag = bucket.put(path, b"first", &Condition::None).unwrap();
    let answer = bucket.put(
        path,
        b"second",
        &Condition::Matches(RevisionToken::new(tag.as_str().trim_matches('"'))),
    );
    eprintln!("put with an unquoted tag: {}", put_line(&answer));
    eprintln!("object afterwards: {}", get_line(&bucket.get(path)));
}

/// A server bucket whose folder `name` exists and holds nothing.
#[cfg(minio_tests)]
async fn empty_folder(name: &str) -> (MinioBucket, PathBuf) {
    let bucket = MinioBucket::new(ready_config());
    let folder = PathBuf::from(format!("/{name}"));
    drop(bucket.remove_folder(&folder).await);
    bucket.create_folder(&folder).await.unwrap();
    (bucket, folder)
}

/// The bytes of `reader`, read to its end through a small buffer.
#[cfg(minio_tests)]
async fn drain(reader: &mut (dyn futures_io::AsyncRead + Send + Unpin + '_)) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut buffer = vec![0_u8; 64 * 1024];
    loop {
        let read = reader.read(&mut buffer).await.unwrap();
        if read == 0 {
            return bytes;
        }
        bytes.extend(buffer.iter().take(read));
    }
}

#[cfg(minio_tests)]
#[test]
fn a_20_mib_stream_is_written_and_read_back_on_the_server() {
    runtime().block_on(async {
        let (bucket, folder) = empty_folder("stream-round-trip").await;
        let path = folder.join("20mib.bin");
        let mut writer = bucket.write_stream(&path, Condition::Absent).await.unwrap();
        for index in 0..20_u8 {
            writer.write_all(&vec![index; 1024 * 1024]).await.unwrap();
        }
        writer.close().await.unwrap();
        assert_eq!(bucket.head(&path).await.unwrap().len, 20 * 1024 * 1024);

        let mut reader = bucket.read_stream(&path).await.unwrap();
        let bytes = drain(&mut *reader).await;
        assert_eq!(bytes.len(), 20 * 1024 * 1024);
        for (index, chunk) in bytes.chunks(1024 * 1024).enumerate() {
            assert!(
                chunk.iter().all(|&byte| usize::from(byte) == index),
                "{index}"
            );
        }
    });
}

#[cfg(minio_tests)]
#[test]
fn a_stream_write_under_absent_on_an_existing_key_is_already_exists_on_the_server() {
    runtime().block_on(async {
        let (bucket, folder) = empty_folder("stream-absent").await;
        let path = folder.join("taken.bin");
        bucket.put(&path, b"first", &Condition::None).await.unwrap();

        let mut small = bucket.write_stream(&path, Condition::Absent).await.unwrap();
        small.write_all(b"second").await.unwrap();
        let refused_small = StorageError::from(small.close().await.unwrap_err());
        assert_eq!(refused_small.kind(), ErrorKind::AlreadyExists);

        let mut large = bucket.write_stream(&path, Condition::Absent).await.unwrap();
        large.write_all(&vec![1_u8; 9 * 1024 * 1024]).await.unwrap();
        let refused_large = StorageError::from(large.close().await.unwrap_err());
        assert_eq!(refused_large.kind(), ErrorKind::AlreadyExists);
        assert_eq!(bucket.get(&path).await.unwrap().0, b"first");
    });
}

#[cfg(minio_tests)]
#[test]
fn a_stream_write_under_a_stale_tag_is_precondition_failed_on_the_server() {
    runtime().block_on(async {
        let (bucket, folder) = empty_folder("stream-matches").await;
        let path = folder.join("tagged.bin");
        let stale = bucket.put(&path, b"first", &Condition::None).await.unwrap();
        bucket
            .put(&path, b"second", &Condition::None)
            .await
            .unwrap();

        let mut small = bucket
            .write_stream(&path, Condition::Matches(stale.clone()))
            .await
            .unwrap();
        small.write_all(b"third").await.unwrap();
        let refused_small = StorageError::from(small.close().await.unwrap_err());
        assert_eq!(refused_small.kind(), ErrorKind::PreconditionFailed);

        let mut large = bucket
            .write_stream(&path, Condition::Matches(stale))
            .await
            .unwrap();
        large.write_all(&vec![1_u8; 9 * 1024 * 1024]).await.unwrap();
        let refused_large = StorageError::from(large.close().await.unwrap_err());
        assert_eq!(refused_large.kind(), ErrorKind::PreconditionFailed);
        assert_eq!(bucket.get(&path).await.unwrap().0, b"second");
    });
}

#[cfg(minio_tests)]
#[test]
fn an_append_stream_keeps_the_old_bytes_first_on_the_server() {
    runtime().block_on(async {
        let (bucket, folder) = empty_folder("stream-append").await;
        for (name, old_len) in [("six.bin", 6 * 1024 * 1024), ("small.bin", 3)] {
            let path = folder.join(name);
            bucket
                .put(&path, &vec![1_u8; old_len], &Condition::None)
                .await
                .unwrap();
            let mut writer = bucket.append_stream(&path).await.unwrap();
            writer.write_all(&vec![2_u8; 1024 * 1024]).await.unwrap();
            writer.close().await.unwrap();
            let (bytes, _token) = bucket.get(&path).await.unwrap();
            assert_eq!(bytes.len(), old_len + 1024 * 1024, "{name}");
            assert!(bytes.iter().take(old_len).all(|&byte| byte == 1), "{name}");
            assert!(bytes.iter().skip(old_len).all(|&byte| byte == 2), "{name}");
        }
        let missing = folder.join("missing.bin");
        let mut writer = bucket.append_stream(&missing).await.unwrap();
        writer.write_all(b"new").await.unwrap();
        writer.close().await.unwrap();
        assert_eq!(bucket.get(&missing).await.unwrap().0, b"new");
    });
}

#[cfg(minio_tests)]
#[test]
fn a_write_stream_dropped_without_close_stores_nothing_on_the_server() {
    runtime().block_on(async {
        let (bucket, folder) = empty_folder("stream-dropped").await;
        let small = folder.join("small.bin");
        let large = folder.join("large.bin");
        let mut first = bucket.write_stream(&small, Condition::None).await.unwrap();
        first.write_all(b"lost").await.unwrap();
        drop(first);
        let mut second = bucket.write_stream(&large, Condition::None).await.unwrap();
        second
            .write_all(&vec![1_u8; 9 * 1024 * 1024])
            .await
            .unwrap();
        second.flush().await.unwrap();
        drop(second);
        for path in [small, large] {
            assert_eq!(
                bucket.head(&path).await.unwrap_err().kind(),
                ErrorKind::NotFound
            );
        }
    });
}

/// Not a parity sequence, since `MemoryBucket` never holds a file and a folder at one path: keys
/// written straight to the server make `/a` an object and a folder, and the file wins everywhere.
#[cfg(minio_tests)]
#[test]
fn a_file_and_a_folder_made_by_keys_show_the_file_on_the_server() {
    let config = ready_config();
    let bucket = server();
    let sdk = client(&config);
    ensure_folder(&bucket, "/conflict-made");
    for key in ["conflict-made/a", "conflict-made/a/b"] {
        runtime()
            .block_on(
                sdk.put_object()
                    .bucket(&config.bucket)
                    .key(key)
                    .body(ByteStream::from_static(b"x"))
                    .send(),
            )
            .unwrap();
    }
    let folder = Path::new("/conflict-made");
    let transcript = vec![
        kind_line(&bucket.head(&at("/conflict-made", "a"))),
        list_line(&bucket.list(folder, Depth::Children), "/conflict-made"),
        list_line(&bucket.list(folder, Depth::All), "/conflict-made"),
        shape(&bucket.walk(folder, false, true), |entries| {
            format!("walked {}", entries.len())
        }),
        get_line(&bucket.get(&at("/conflict-made", "a/b"))),
    ];
    drop(bucket.remove_folder(folder));
    eprintln!(
        "file and folder by keys on the server:\n{}",
        transcript.join("\n")
    );
    assert_eq!(
        transcript,
        [
            "file true dir false size 1 token true",
            "paths [\"a\"]",
            "paths [\"a\"]",
            "walked 1",
            "bytes [120]",
        ]
    );
}

/// Not a parity sequence, since `MemoryBucket` has no markers: a folder `create_folder` made has
/// the time of its marker, and a folder implied by a key below it has none.
#[cfg(minio_tests)]
#[test]
fn a_made_folder_has_a_time_and_an_implied_folder_has_none_on_the_server() {
    let config = ready_config();
    let bucket = server();
    ensure_folder(&bucket, "/folder-times/made");
    runtime()
        .block_on(
            client(&config)
                .put_object()
                .bucket(&config.bucket)
                .key("folder-times/implied/z")
                .body(ByteStream::from_static(b"x"))
                .send(),
        )
        .unwrap();
    let made = bucket.head(Path::new("/folder-times/made"));
    let implied = bucket.head(Path::new("/folder-times/implied"));
    let transcript = vec![head_line(&made), head_line(&implied)];
    drop(bucket.remove_folder(Path::new("/folder-times")));
    eprintln!("folder times on the server:\n{}", transcript.join("\n"));
    assert_eq!(
        transcript,
        [
            "file false dir true size 0 modified true token false",
            "file false dir true size 0 modified false token false",
        ]
    );
}

/// Not a parity sequence, since `MemoryBucket` has no missing-bucket state: every call to
/// a bucket the server does not have is `Other`, and names the server's answer.
#[cfg(minio_tests)]
#[test]
fn a_missing_bucket_is_other_on_the_server() {
    let mut config = config();
    "os-shim-no-such-bucket".clone_into(&mut config.bucket);
    let bucket = BlockingBucket::new(MinioBucket::new(config)).unwrap();
    let path = Path::new("/missing-bucket/note.md");
    let got = bucket.get(path);
    let transcript = vec![
        get_line(&got),
        head_line(&bucket.head(path)),
        put_line(&bucket.put(path, b"# Note\n", &Condition::None)),
        put_line(&bucket.put(
            path,
            b"# Note\n",
            &Condition::Matches(RevisionToken::new("\"stale\"")),
        )),
        list_line(
            &bucket.list(Path::new("/missing-bucket"), Depth::All),
            "/missing-bucket",
        ),
    ];
    eprintln!("missing bucket on the server:\n{}", transcript.join("\n"));
    assert_eq!(transcript, ["Other", "Other", "Other", "Other", "Other"]);
    let message = got.unwrap_err().to_string();
    assert!(message.starts_with("404 NoSuchBucket"), "{message}");
}

#[cfg(minio_tests)]
#[test]
fn a_delete_on_a_missing_bucket_is_other_on_the_server() {
    let mut config = config();
    "os-shim-no-such-bucket".clone_into(&mut config.bucket);
    let bucket = BlockingBucket::new(MinioBucket::new(config)).unwrap();
    let path = Path::new("/missing-bucket/note.md");
    for condition in [
        Condition::Absent,
        Condition::Matches(RevisionToken::new("\"x\"")),
    ] {
        let error = bucket.delete(path, &condition).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Other, "{condition:?}");
        assert!(
            error.to_string().contains("os-shim-no-such-bucket"),
            "{error}"
        );
    }
}

/// Not a parity sequence, since `MemoryBucket` has no bucket to create: `ensure` creates a
/// missing bucket, then leaves it as it is, and the bucket takes objects.
#[cfg(minio_tests)]
#[test]
fn ensure_creates_a_missing_bucket_on_the_server() {
    let mut config = config();
    config.bucket = format!("os-shim-ensure-{}", process::id());
    let sdk = client(&config);
    let bucket = BlockingBucket::new(MinioBucket::new(config.clone())).unwrap();
    let key = "ensure/note.md";
    let path = Path::new("/ensure/note.md");
    let before = bucket.get(path);
    let created = bucket.runtime.block_on(bucket.bucket.ensure());
    let again = bucket.runtime.block_on(bucket.bucket.ensure());
    let folder = bucket.create_folder(Path::new("/ensure"));
    let put = bucket.put(path, b"# Note\n", &Condition::None);
    let got = bucket.get(path);
    let removed = bucket
        .runtime
        .block_on(sdk.delete_object().bucket(&config.bucket).key(key).send());
    let unmarked = bucket.runtime.block_on(
        sdk.delete_object()
            .bucket(&config.bucket)
            .key("ensure/")
            .send(),
    );
    let deleted = bucket
        .runtime
        .block_on(sdk.delete_bucket().bucket(&config.bucket).send());
    let transcript = vec![
        get_line(&before),
        shape(&created, |&()| "ensured".to_owned()),
        shape(&again, |&()| "ensured".to_owned()),
        done_line(&folder),
        put_line(&put),
        get_line(&got),
        status(&removed),
        status(&unmarked),
        status(&deleted),
    ];
    eprintln!("ensure on the server:\n{}", transcript.join("\n"));
    assert_eq!(
        transcript,
        [
            "Other",
            "ensured",
            "ensured",
            "done",
            "tag",
            "bytes [35, 32, 78, 111, 116, 101, 10]",
            "accepted",
            "accepted",
            "accepted"
        ]
    );
}

/// The value of an environment variable the S3 tests need.
#[cfg(s3_tests)]
fn aws_variable(name: &str) -> String {
    env::var(name).unwrap()
}

/// The AWS bucket the environment names.
#[cfg(s3_tests)]
fn aws() -> BlockingBucket<S3Bucket> {
    BlockingBucket::new(S3Bucket::new(S3Config::new(
        aws_variable("OS_SHIM_S3_REGION"),
        aws_variable("OS_SHIM_S3_BUCKET"),
        aws_variable("AWS_ACCESS_KEY_ID"),
        aws_variable("AWS_SECRET_ACCESS_KEY"),
    )))
    .unwrap()
}

/// An SDK client for the AWS bucket the environment names.
#[cfg(s3_tests)]
fn aws_client() -> Client {
    aws_client_in(&aws_variable("OS_SHIM_S3_REGION"))
}

/// An SDK client for AWS in `region`, with the keys the environment names.
#[cfg(s3_tests)]
fn aws_client_in(region: &str) -> Client {
    let credentials = Credentials::new(
        aws_variable("AWS_ACCESS_KEY_ID"),
        aws_variable("AWS_SECRET_ACCESS_KEY"),
        None,
        None,
        "os-shim",
    );
    let sdk = Config::builder()
        .behavior_version(BehaviorVersion::latest())
        .region(Region::new(region.to_owned()))
        .credentials_provider(credentials)
        .retry_config(RetryConfig::disabled())
        .build();
    Client::from_conf(sdk)
}

/// A key prefix no other run or sequence shares.
#[cfg(s3_tests)]
fn run_prefix(name: &str) -> String {
    format!("/os-shim-tests/{}-{name}", process::id())
}

/// Delete every key under `prefix`.
#[cfg(s3_tests)]
fn remove_prefix(bucket: &BlockingBucket<S3Bucket>, prefix: &str) {
    drop(bucket.remove_folder(Path::new(prefix)));
}

/// The transcript of `sequence` on AWS under a prefix of its own, removed before this returns.
#[cfg(s3_tests)]
fn aws_transcript(
    name: &str,
    sequence: impl Fn(&BlockingBucket<S3Bucket>, &str) -> Vec<String>,
) -> Vec<String> {
    let bucket = aws();
    let prefix = run_prefix(name);
    let transcript = sequence(&bucket, &prefix);
    remove_prefix(&bucket, &prefix);
    transcript
}

#[cfg(s3_tests)]
#[test]
fn put_and_get_agrees_with_aws() {
    assert_agree(
        "put and get",
        &put_and_get(&memory(), "/a"),
        &aws_transcript("put-and-get", put_and_get),
    );
}

#[cfg(s3_tests)]
#[test]
fn head_after_put_agrees_with_aws() {
    assert_agree(
        "head after put",
        &head_after_put(&memory(), "/a"),
        &aws_transcript("head-after-put", head_after_put),
    );
}

#[cfg(s3_tests)]
#[test]
fn conditional_writes_agrees_with_aws() {
    assert_agree(
        "conditional writes",
        &conditional_writes(&memory(), "/a"),
        &aws_transcript("conditional-writes", conditional_writes),
    );
}

#[cfg(s3_tests)]
#[test]
fn missing_path_deletes_agrees_with_aws() {
    assert_agree(
        "missing path deletes",
        &missing_path_deletes(&memory(), "/a"),
        &aws_transcript("missing-path-deletes", missing_path_deletes),
    );
}

#[cfg(s3_tests)]
#[test]
fn absent_delete_over_a_file_agrees_with_aws() {
    assert_agree(
        "absent delete over a file",
        &absent_delete_over_a_file(&memory(), "/a"),
        &aws_transcript("absent-delete-over-a-file", absent_delete_over_a_file),
    );
}

#[cfg(s3_tests)]
#[test]
fn missing_path_tokened_put_agrees_with_aws() {
    assert_agree(
        "missing path tokened put",
        &missing_path_tokened_put(&memory(), "/a"),
        &aws_transcript("missing-path-tokened-put", missing_path_tokened_put),
    );
}

#[cfg(s3_tests)]
#[test]
fn rewritten_tokens_agrees_with_aws() {
    assert_agree(
        "rewritten tokens",
        &rewritten_tokens(&memory(), "/a"),
        &aws_transcript("rewritten-tokens", rewritten_tokens),
    );
}

#[cfg(s3_tests)]
#[test]
fn long_listing_agrees_with_aws() {
    assert_agree(
        "long listing",
        &long_listing(&memory(), "/a"),
        &aws_transcript("long-listing", long_listing),
    );
}

#[cfg(s3_tests)]
#[test]
fn empty_listing_agrees_with_aws() {
    assert_agree(
        "empty listing",
        &empty_listing(&memory(), "/a"),
        &aws_transcript("empty-listing", empty_listing),
    );
}

#[cfg(s3_tests)]
#[test]
fn children_listing_agrees_with_aws() {
    assert_agree(
        "children listing",
        &children_listing(&memory(), "/a"),
        &aws_transcript("children-listing", children_listing),
    );
}

#[cfg(s3_tests)]
#[test]
fn folders_agrees_with_aws() {
    assert_agree(
        "folders",
        &folders(&memory(), "/a"),
        &aws_transcript("folders", folders),
    );
}

#[cfg(s3_tests)]
#[test]
fn file_and_folder_conflict_agrees_with_aws() {
    assert_agree(
        "file and folder conflict",
        &file_and_folder_conflict(&memory(), "/a"),
        &aws_transcript("file-and-folder-conflict", file_and_folder_conflict),
    );
}

#[cfg(s3_tests)]
#[test]
fn copies_agrees_with_aws() {
    assert_agree(
        "copies",
        &copies(&memory(), "/a"),
        &aws_transcript("copies", copies),
    );
}

#[cfg(s3_tests)]
#[test]
fn file_renames_agrees_with_aws() {
    assert_agree(
        "file renames",
        &file_renames(&memory(), "/a"),
        &aws_transcript("file-renames", file_renames),
    );
}

#[cfg(s3_tests)]
#[test]
fn folder_rename_agrees_with_aws() {
    assert_agree(
        "folder rename",
        &folder_rename(&memory(), "/a"),
        &aws_transcript("folder-rename", folder_rename),
    );
}

#[cfg(s3_tests)]
#[test]
fn parent_check_agrees_with_aws() {
    assert_agree(
        "parent check",
        &parent_check(&memory(), "/a"),
        &aws_transcript("parent-check", parent_check),
    );
}

#[cfg(s3_tests)]
#[test]
fn refused_paths_agree_with_aws() {
    assert_agree(
        "refused paths",
        &refused_paths(&memory()),
        &refused_paths(&aws()),
    );
}

/// `ensure` on the bucket the environment names, which exists, succeeds and creates nothing.
#[cfg(s3_tests)]
#[test]
fn ensure_leaves_an_existing_bucket_on_aws() {
    let bucket = aws();
    bucket.runtime.block_on(bucket.bucket.ensure(None)).unwrap();
}

/// Not a parity sequence, since `MemoryBucket` has no bucket to create: `ensure` on a bucket that
/// does not exist creates it in the client's region, private, with the key as its default
/// encryption. Every call runs before the first `unwrap`, so a failure leaves no bucket behind.
#[cfg(s3_tests)]
#[test]
fn ensure_creates_a_private_encrypted_bucket_on_aws() {
    let region = aws_variable("OS_SHIM_S3_ENSURE_REGION");
    let key_arn = aws_variable("OS_SHIM_S3_KMS_KEY_ARN");
    let name = format!("os-shim-ensure-{}", process::id());
    let bucket = S3Bucket::new(S3Config::new(
        region.clone(),
        name.clone(),
        aws_variable("AWS_ACCESS_KEY_ID"),
        aws_variable("AWS_SECRET_ACCESS_KEY"),
    ));
    let client = aws_client_in(&region);
    let runtime = runtime();
    let created = runtime.block_on(bucket.ensure(Some(&key_arn)));
    let again = runtime.block_on(bucket.ensure(Some(&key_arn)));
    let located = runtime.block_on(client.get_bucket_location().bucket(&name).send());
    let encrypted = runtime.block_on(client.get_bucket_encryption().bucket(&name).send());
    let deleted = runtime.block_on(client.delete_bucket().bucket(&name).send());

    created.unwrap();
    again.unwrap();
    let location = located.unwrap();
    let encryption = encrypted.unwrap();
    deleted.unwrap();

    let rule = encryption
        .server_side_encryption_configuration()
        .unwrap()
        .rules()
        .first()
        .unwrap();
    let by_default = rule.apply_server_side_encryption_by_default().unwrap();
    let constraint = location
        .location_constraint()
        .map(BucketLocationConstraint::as_str);
    eprintln!(
        "location: {}\nalgorithm: {}\nbucket key: {}",
        constraint.unwrap_or("none"),
        by_default.sse_algorithm().as_str(),
        rule.bucket_key_enabled() == Some(true)
    );
    assert_eq!(constraint, Some(region.as_str()));
    assert_eq!(by_default.sse_algorithm().as_str(), "aws:kms");
    assert_eq!(by_default.kms_master_key_id(), Some(key_arn.as_str()));
    assert_eq!(rule.bucket_key_enabled(), Some(true));
}

#[cfg(s3_tests)]
#[test]
fn observe_a_raw_delete_with_a_wrong_if_match_on_aws() {
    let bucket = aws();
    let prefix = run_prefix("observe-raw-delete");
    let path = PathBuf::from(format!("{prefix}/note.md"));
    ensure_folder(&bucket, &prefix);
    let key = path
        .display()
        .to_string()
        .trim_start_matches('/')
        .to_owned();
    bucket.put(&path, b"kept", &Condition::None).unwrap();
    let answer = runtime().block_on(
        aws_client()
            .delete_object()
            .bucket(aws_variable("OS_SHIM_S3_BUCKET"))
            .key(&key)
            .if_match("\"0123456789abcdef0123456789abcdef\"")
            .send(),
    );
    eprintln!("raw delete with a wrong If-Match: {}", status(&answer));
    eprintln!("object afterwards: {}", get_line(&bucket.get(&path)));
    remove_prefix(&bucket, &prefix);
}

#[cfg(s3_tests)]
#[test]
fn observe_a_put_with_the_tag_stripped_of_its_quotes_on_aws() {
    let bucket = aws();
    let prefix = run_prefix("observe-unquoted-tag");
    let path = PathBuf::from(format!("{prefix}/note.md"));
    ensure_folder(&bucket, &prefix);
    let tag = bucket.put(&path, b"first", &Condition::None).unwrap();
    let answer = bucket.put(
        &path,
        b"second",
        &Condition::Matches(RevisionToken::new(tag.as_str().trim_matches('"'))),
    );
    eprintln!("put with an unquoted tag: {}", put_line(&answer));
    eprintln!("object afterwards: {}", get_line(&bucket.get(&path)));
    remove_prefix(&bucket, &prefix);
}

/// Every object written, copied or renamed has a modification time when read back.
fn modification_times<B>(bucket: &BlockingBucket<B>, base: &str) -> Vec<String>
where
    B: Bucket,
{
    ensure_folder(bucket, base);
    let note = at(base, "note.md");
    let empty = at(base, "empty.md");
    let copied = at(base, "copied.md");
    let moved = at(base, "moved.md");
    bucket.put(&note, b"# Note\n", &Condition::None).unwrap();
    bucket.put(&empty, b"", &Condition::None).unwrap();
    bucket.copy(&note, &copied).unwrap();
    bucket.rename(&copied, &moved).unwrap();
    [note, empty, moved]
        .iter()
        .map(|path| {
            format!(
                "{} modified {}",
                path.display(),
                bucket.head(path).unwrap().modified.is_some()
            )
        })
        .collect()
}

#[cfg(any(minio_tests, s3_tests))]
#[test]
fn every_object_has_a_modification_time() {
    #[cfg(minio_tests)]
    {
        let bucket = server();
        let transcript = modification_times(&bucket, "/modification-times");
        drop(bucket.remove_folder(Path::new("/modification-times")));
        assert_eq!(
            transcript,
            [
                "/modification-times/note.md modified true",
                "/modification-times/empty.md modified true",
                "/modification-times/moved.md modified true",
            ]
        );
    }
    #[cfg(s3_tests)]
    {
        let bucket = aws();
        let prefix = run_prefix("modification-times");
        let transcript = modification_times(&bucket, &prefix);
        remove_prefix(&bucket, &prefix);
        assert_eq!(
            transcript,
            [
                format!("{prefix}/note.md modified true"),
                format!("{prefix}/empty.md modified true"),
                format!("{prefix}/moved.md modified true"),
            ]
        );
    }
}

/// A file and a folder under one path: puts at `a` and `a/b` both succeed, and `/a` reads as the file.
#[cfg(s3_tests)]
#[test]
fn a_file_and_a_folder_share_a_path_on_aws() {
    let bucket = aws();
    let sdk = aws_client();
    let prefix = run_prefix("file-and-folder");
    ensure_folder(&bucket, &prefix);
    let key_base = prefix.trim_start_matches('/').to_owned();
    let puts: Vec<bool> = [format!("{key_base}/a"), format!("{key_base}/a/b")]
        .iter()
        .map(|key| {
            runtime()
                .block_on(
                    sdk.put_object()
                        .bucket(aws_variable("OS_SHIM_S3_BUCKET"))
                        .key(key)
                        .body(ByteStream::from_static(b"x"))
                        .send(),
                )
                .is_ok()
        })
        .collect();
    let transcript = vec![
        kind_line(&bucket.head(&at(&prefix, "a"))),
        get_line(&bucket.get(&at(&prefix, "a/b"))),
    ];
    remove_prefix(&bucket, &prefix);
    assert_eq!(puts, [true, true]);
    assert_eq!(
        transcript,
        ["file true dir false size 1 token true", "bytes [120]"]
    );
}
