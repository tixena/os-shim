//! Tests for the blocking wrapper, the key rules and the `MinIO` and AWS S3 clients.

#![cfg(any(feature = "minio", feature = "s3"))]

use super::paths::key_of;
use super::streams::{COPY_MAX_BYTES, COPY_MIN_BYTES, copy_ranges};
use super::{
    BlockingBucket, Bucket as _, Condition, Depth, Guarantee, MemoryBucket, RevisionToken,
};
#[cfg(feature = "minio")]
use super::{MinioBucket, MinioConfig};
#[cfg(feature = "s3")]
use super::{S3Bucket, S3Config};
use crate::WalkEntry;
use crate::error::{ErrorKind, StorageError};
use alloc::sync::Arc;
#[cfg(feature = "minio")]
use aws_sdk_s3::{error::SdkError, operation::get_object::GetObjectError};
#[cfg(feature = "minio")]
use core::sync::atomic::{AtomicUsize, Ordering};
#[cfg(feature = "minio")]
use core::time::Duration;
#[cfg(feature = "minio")]
use core::{error::Error, iter};
#[cfg(feature = "minio")]
use futures_io::AsyncRead;
#[cfg(feature = "minio")]
use futures_util::{AsyncReadExt as _, AsyncWriteExt as _};
#[cfg(feature = "minio")]
use std::io;
use std::path::Path;
#[cfg(feature = "minio")]
use std::time::UNIX_EPOCH;
#[cfg(any(feature = "minio", feature = "s3"))]
use std::{
    io::{BufRead as _, BufReader, Read as _, Write as _},
    net::{TcpListener, TcpStream},
    sync::Mutex,
    thread,
};
#[cfg(feature = "minio")]
use tracing::{
    Event, Level, Metadata, Subscriber,
    span::{Attributes, Id, Record},
    subscriber::with_default,
};

/// The guarantees every SDK client holds, whatever the store.
#[cfg(any(feature = "minio", feature = "s3"))]
const SDK_GUARANTEES: [Guarantee; 3] = [
    Guarantee::ConditionalWriteInOneStep,
    Guarantee::CreateOnlyWriteInOneStep,
    Guarantee::EmptyFoldersExist,
];

/// Every guarantee there is.
#[cfg(any(feature = "minio", feature = "s3"))]
const EVERY_GUARANTEE: [Guarantee; 15] = [
    Guarantee::AppendIsAtomic,
    Guarantee::AppendWritesOnlyNewBytes,
    Guarantee::ConditionalDeleteInOneStep,
    Guarantee::ConditionalWriteInOneStep,
    Guarantee::CreateOnlyWriteInOneStep,
    Guarantee::DroppedWriterKeepsBytes,
    Guarantee::EmptyFoldersExist,
    Guarantee::FailedRenameKeepsDestination,
    Guarantee::FileAndFolderNeverShareAPath,
    Guarantee::FileRenameIsAtomic,
    Guarantee::FolderRenameIsAtomic,
    Guarantee::FuturesNeverWait,
    Guarantee::ParentCheckInOneStep,
    Guarantee::ReadersNeverSeePartialWrites,
    Guarantee::WalkHonorsGitignore,
];

/// Eight MiB, the size of a part and of a range.
#[cfg(feature = "minio")]
const PART: usize = 8 * 1024 * 1024;

/// The path of the stream tests' object.
#[cfg(feature = "minio")]
const BIG: &str = "/big.bin";

/// A list of lines the canned server records, shared with the test.
type Log = Arc<Mutex<Vec<String>>>;

/// A subscriber that counts the events logged at error level.
#[cfg(feature = "minio")]
struct CountsErrors(Arc<AtomicUsize>);

#[cfg(feature = "minio")]
impl Subscriber for CountsErrors {
    fn enabled(&self, _metadata: &Metadata<'_>) -> bool {
        true
    }

    fn enter(&self, _span: &Id) {}

    fn event(&self, event: &Event<'_>) {
        if *event.metadata().level() == Level::ERROR {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn exit(&self, _span: &Id) {}

    fn new_span(&self, _span: &Attributes<'_>) -> Id {
        Id::from_u64(1)
    }

    fn record(&self, _span: &Id, _values: &Record<'_>) {}

    fn record_follows_from(&self, _span: &Id, _follows: &Id) {}
}

/// The paths of `entries`.
fn paths(entries: &[WalkEntry]) -> Vec<String> {
    entries
        .iter()
        .map(|entry| entry.path.display().to_string())
        .collect()
}

fn blocking() -> BlockingBucket<MemoryBucket> {
    BlockingBucket::new(MemoryBucket::default()).unwrap()
}

#[test]
fn blocking_put_then_get_gives_the_bytes_and_the_put_token() {
    let bucket = blocking();
    let path = Path::new("/a.md");
    let token = bucket.put(path, b"x", &Condition::None).unwrap();
    assert_eq!(bucket.get(path).unwrap(), (b"x".to_vec(), token));
}

#[test]
fn blocking_wraps_every_whole_file_operation() {
    let bucket = blocking();
    bucket.create_folder(Path::new("/d/e")).unwrap();
    bucket
        .put(Path::new("/d/a.md"), b"x", &Condition::None)
        .unwrap();
    bucket.append(Path::new("/d/a.md"), b"y").unwrap();
    assert_eq!(
        bucket
            .copy(Path::new("/d/a.md"), Path::new("/d/b.md"))
            .unwrap(),
        2
    );
    bucket
        .rename(Path::new("/d/b.md"), Path::new("/d/e/c.md"))
        .unwrap();
    assert_eq!(
        paths(&bucket.list(Path::new("/d"), Depth::Children).unwrap()),
        ["/d/a.md", "/d/e"]
    );
    assert_eq!(
        paths(&bucket.walk(Path::new("/d"), false, true).unwrap()),
        ["/d/a.md", "/d/e", "/d/e/c.md"]
    );
    assert!(bucket.head(Path::new("/d/e")).unwrap().is_dir);
    assert_eq!(
        bucket.canonicalize(Path::new("/d/./e/../a.md")).unwrap(),
        Path::new("/d/a.md")
    );
    bucket
        .delete(Path::new("/d/a.md"), &Condition::None)
        .unwrap();
    bucket.remove_folder(Path::new("/d")).unwrap();
    assert_eq!(
        bucket.head(Path::new("/d")).unwrap_err().kind(),
        ErrorKind::NotFound
    );
}

#[tokio::test(flavor = "current_thread")]
#[should_panic(expected = "Cannot start a runtime from within a runtime")]
async fn blocking_inside_a_runtime_panics() {
    let bucket = blocking();
    drop(bucket.get(Path::new("/a.md")));
}

#[test]
fn the_key_rules_give_the_designs_table() {
    let key = |prefix: &str, path: &str| key_of(prefix, Path::new(path)).unwrap();
    assert_eq!(key("", "/notes/a.md"), "notes/a.md");
    assert_eq!(key("tenant-7", "/notes/a.md"), "tenant-7/notes/a.md");
    assert_eq!(key("", "/notes/./b/../a.md"), "notes/a.md");
    for refused in ["/../a.md", "notes/a.md", "a.md", "/notes/../../a.md"] {
        let error = key_of("", Path::new(refused)).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidInput, "{refused}");
    }
}

#[cfg(unix)]
#[test]
fn a_path_that_is_not_utf8_has_no_key() {
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt as _;

    let path = Path::new(OsStr::from_bytes(b"/notes/\xff.md"));
    let error: StorageError = key_of("", path).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidInput);
}

#[cfg(windows)]
#[test]
fn a_windows_path_keeps_its_drive_in_the_key() {
    assert_eq!(
        key_of("", Path::new(r"C:\Users\a.md")).unwrap(),
        "C:/Users/a.md"
    );
}

/// Serve `answers` in order, one per request, and record each request's method and conditional headers.
#[cfg(any(feature = "minio", feature = "s3"))]
fn canned(answers: Vec<String>) -> (String, Arc<Mutex<Vec<String>>>) {
    let (endpoint, seen, _targets) = canned_with_targets(answers);
    (endpoint, seen)
}

/// Like `canned`, and also record the target of each request, as `METHOD /path`.
#[cfg(any(feature = "minio", feature = "s3"))]
fn canned_with_targets(answers: Vec<String>) -> (String, Log, Log) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let targets = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&seen);
    let target_log = Arc::clone(&targets);
    thread::spawn(move || serve(&listener, answers, &log, &target_log));
    (endpoint, seen, targets)
}

/// Answer each request on every connection with the next answer, until none is left.
#[cfg(any(feature = "minio", feature = "s3"))]
fn serve(
    listener: &TcpListener,
    answers: Vec<String>,
    log: &Mutex<Vec<String>>,
    targets: &Mutex<Vec<String>>,
) {
    let mut pending = answers.into_iter();
    for incoming in listener.incoming() {
        let mut stream = incoming.unwrap();
        let mut reader = BufReader::new(stream.try_clone().unwrap());
        let mut request_line = String::new();
        while reader.read_line(&mut request_line).unwrap() > 0 {
            targets.lock().unwrap().push(
                request_line
                    .split(' ')
                    .take(2)
                    .collect::<Vec<_>>()
                    .join(" "),
            );
            log.lock()
                .unwrap()
                .push(read_request(&request_line, &mut reader));
            request_line.clear();
            let Some(answer) = pending.next() else {
                return;
            };
            stream.write_all(answer.as_bytes()).unwrap();
        }
    }
}

/// The rest of the request that starts with `request_line`, read off `reader`, as its method, its
/// conditional headers, and its canned ACL, copy source, range, part length and location constraint
/// when it carries them.
#[cfg(any(feature = "minio", feature = "s3"))]
fn read_request(request_line: &str, reader: &mut BufReader<TcpStream>) -> String {
    let mut length = 0_u64;
    let mut if_match = String::from("-");
    let mut if_none_match = None;
    let mut acl = None;
    let mut copy_source = None;
    let mut range = None;
    let mut copy_range = None;
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        let Some((name, value)) = line.trim_end().split_once(':') else {
            break;
        };
        match name.to_ascii_lowercase().as_str() {
            "content-length" => length = value.trim().parse().unwrap(),
            "if-match" => value.trim().clone_into(&mut if_match),
            "if-none-match" => if_none_match = Some(value.trim().to_owned()),
            "x-amz-acl" => acl = Some(value.trim().to_owned()),
            "x-amz-copy-source" => copy_source = Some(value.trim().to_owned()),
            "x-amz-copy-source-range" => copy_range = Some(value.trim().to_owned()),
            "range" => range = Some(value.trim().to_owned()),
            _ => {}
        }
    }
    let mut body = Vec::new();
    reader.take(length).read_to_end(&mut body).unwrap();
    let location = String::from_utf8_lossy(&body)
        .split_once("<LocationConstraint>")
        .and_then(|(_, rest)| rest.split_once("</LocationConstraint>"))
        .map(|(region, _)| region.to_owned());
    let method = request_line.split(' ').next().unwrap();
    let mut parts = vec![method.to_owned(), format!("if-match={if_match}")];
    if request_line.contains("partNumber=")
        && request_line.starts_with("PUT")
        && copy_source.is_none()
    {
        parts.push(format!("len={length}"));
    }
    parts.extend(if_none_match.map(|value| format!("if-none-match={value}")));
    parts.extend(acl.map(|value| format!("acl={value}")));
    parts.extend(copy_source.map(|value| format!("copy-source={value}")));
    parts.extend(copy_range.map(|value| format!("copy-range={value}")));
    parts.extend(range.map(|value| format!("range={value}")));
    parts.extend(location.map(|region| format!("location={region}")));
    parts.join(" ")
}

/// An HTTP answer with `status`, the header lines in `headers` and `body`.
#[cfg(any(feature = "minio", feature = "s3"))]
fn answer(status: &str, headers: &str, body: &str) -> String {
    format!(
        "HTTP/1.1 {status}\r\n{headers}Content-Length: {}\r\n\r\n{body}",
        body.len()
    )
}

/// A 200 answer to a `HEAD`: the header lines in `headers`, which declare the length themselves, and no body.
#[cfg(any(feature = "minio", feature = "s3"))]
fn head_answer(headers: &str) -> String {
    format!("HTTP/1.1 200 OK\r\n{headers}\r\n")
}

/// A 404 answer whose S3 error body carries `code` and `message`.
#[cfg(any(feature = "minio", feature = "s3"))]
fn not_found_answer(code: &str, message: &str) -> String {
    let body = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><Error><Code>{code}</Code><Message>{message}</Message></Error>"
    );
    answer("404 Not Found", "Content-Type: application/xml\r\n", &body)
}

/// The 404 answer for a bucket that does not exist.
#[cfg(any(feature = "minio", feature = "s3"))]
fn no_such_bucket() -> String {
    not_found_answer("NoSuchBucket", "The specified bucket does not exist.")
}

/// A `ListObjectsV2` answer with `key`, truncated with the token `next` when it is given.
#[cfg(any(feature = "minio", feature = "s3"))]
fn page(key: &str, next: Option<&str>) -> String {
    let truncation = next.map_or_else(
        || "<IsTruncated>false</IsTruncated>".to_owned(),
        |token| {
            format!(
                "<IsTruncated>true</IsTruncated><NextContinuationToken>{token}</NextContinuationToken>"
            )
        },
    );
    let body = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><ListBucketResult><Name>archie</Name><Prefix>vault/</Prefix>{truncation}<Contents><Key>{key}</Key></Contents></ListBucketResult>"
    );
    answer("200 OK", "Content-Type: application/xml\r\n", &body)
}

/// A `ListObjectsV2` answer with no key.
#[cfg(any(feature = "minio", feature = "s3"))]
fn empty_page() -> String {
    answer(
        "200 OK",
        "Content-Type: application/xml\r\n",
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><ListBucketResult><Name>archie</Name><IsTruncated>false</IsTruncated></ListBucketResult>",
    )
}

/// A `ListObjectsV2` answer holding `keys` as objects, with `LastModified` on the first, and
/// `folders` as common prefixes.
#[cfg(feature = "minio")]
fn listing(keys: &[&str], folders: &[&str]) -> String {
    let contents = keys
        .iter()
        .map(|key| {
            format!(
                "<Contents><Key>{key}</Key><LastModified>1970-01-01T00:16:40.000Z</LastModified></Contents>"
            )
        })
        .collect::<Vec<_>>()
        .concat();
    let prefixes = folders
        .iter()
        .map(|folder| format!("<CommonPrefixes><Prefix>{folder}</Prefix></CommonPrefixes>"))
        .collect::<Vec<_>>()
        .concat();
    answer(
        "200 OK",
        "Content-Type: application/xml\r\n",
        &format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?><ListBucketResult><Name>archie</Name><IsTruncated>false</IsTruncated>{contents}{prefixes}</ListBucketResult>"
        ),
    )
}

/// A `CopyObject` answer carrying the entity tag `tag`, which is quoted.
#[cfg(any(feature = "minio", feature = "s3"))]
fn copied(tag: &str) -> String {
    answer(
        "200 OK",
        "Content-Type: application/xml\r\n",
        &format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?><CopyObjectResult><ETag>&quot;{tag}&quot;</ETag></CopyObjectResult>"
        ),
    )
}

/// The requests the canned server has seen so far.
#[cfg(any(feature = "minio", feature = "s3"))]
fn requests(seen: &Mutex<Vec<String>>) -> Vec<String> {
    seen.lock().unwrap().clone()
}

/// The configuration every scenario uses, against `endpoint`.
#[cfg(feature = "minio")]
fn minio_config(endpoint: String) -> MinioConfig {
    MinioConfig::new(
        endpoint,
        "us-east-1".to_owned(),
        "archie".to_owned(),
        "os-shim".to_owned(),
        "os-shim-secret".to_owned(),
    )
}

/// A blocking `MinIO` client for `endpoint`.
#[cfg(feature = "minio")]
fn minio(endpoint: String) -> BlockingBucket<MinioBucket> {
    BlockingBucket::new(MinioBucket::new(minio_config(endpoint))).unwrap()
}

#[cfg(feature = "minio")]
#[test]
fn minio_get_answered_404_is_not_found() {
    let (endpoint, seen) = canned(vec![answer("404 Not Found", "", ""), empty_page()]);
    let error = minio(endpoint).get(Path::new("/task.md")).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert_eq!(requests(&seen), ["GET if-match=-", "GET if-match=-"]);
}

#[cfg(feature = "minio")]
#[test]
fn minio_put_with_a_stale_tag_answered_412_is_precondition_failed() {
    let (endpoint, seen) = canned(vec![
        empty_page(),
        answer("412 Precondition Failed", "", ""),
    ]);
    let error = minio(endpoint)
        .put(
            Path::new("/task.md"),
            b"x",
            &Condition::Matches(RevisionToken::new("\"a\"")),
        )
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::PreconditionFailed);
    assert_eq!(requests(&seen), ["GET if-match=-", "PUT if-match=\"a\""]);
}

#[cfg(feature = "minio")]
#[test]
fn minio_put_with_absent_answered_412_is_already_exists_and_sends_if_none_match_star() {
    let (endpoint, seen) = canned(vec![
        empty_page(),
        answer("412 Precondition Failed", "", ""),
    ]);
    let error = minio(endpoint)
        .put(Path::new("/task.md"), b"x", &Condition::Absent)
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::AlreadyExists);
    assert_eq!(
        requests(&seen),
        ["GET if-match=-", "PUT if-match=- if-none-match=*"]
    );
}

#[cfg(feature = "minio")]
#[test]
fn minio_put_gives_the_tag_the_server_answered() {
    let (endpoint, _seen) = canned(vec![empty_page(), answer("200 OK", "ETag: \"b\"\r\n", "")]);
    let tag = minio(endpoint)
        .put(Path::new("/task.md"), b"x", &Condition::None)
        .unwrap();
    assert_eq!(tag, RevisionToken::new("\"b\""));
}

#[cfg(feature = "minio")]
#[test]
fn minio_get_gives_the_bytes_and_the_tag() {
    let (endpoint, _seen) = canned(vec![answer("200 OK", "ETag: \"b\"\r\n", "x")]);
    let (bytes, token) = minio(endpoint).get(Path::new("/task.md")).unwrap();
    assert_eq!(bytes, b"x");
    assert_eq!(token, RevisionToken::new("\"b\""));
}

#[cfg(feature = "minio")]
#[test]
fn minio_delete_with_a_stale_tag_fails_without_deleting() {
    let (endpoint, seen) = canned(vec![
        answer("200 OK", "ETag: \"b\"\r\n", ""),
        answer("200 OK", "ETag: \"b\"\r\n", ""),
    ]);
    let error = minio(endpoint)
        .delete(
            Path::new("/task.md"),
            &Condition::Matches(RevisionToken::new("\"a\"")),
        )
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::PreconditionFailed);
    assert_eq!(requests(&seen), ["HEAD if-match=-", "HEAD if-match=-"]);
}

#[cfg(feature = "minio")]
#[test]
fn minio_delete_with_the_current_tag_sends_if_match() {
    let (endpoint, seen) = canned(vec![
        answer("200 OK", "ETag: \"a\"\r\n", ""),
        answer("204 No Content", "", ""),
    ]);
    minio(endpoint)
        .delete(
            Path::new("/task.md"),
            &Condition::Matches(RevisionToken::new("\"a\"")),
        )
        .unwrap();
    assert_eq!(
        requests(&seen),
        ["HEAD if-match=-", "DELETE if-match=\"a\""]
    );
}

#[cfg(feature = "minio")]
#[test]
fn minio_delete_with_a_tag_of_a_missing_object_fails_without_deleting() {
    let (endpoint, seen) = canned(vec![
        answer("404 Not Found", "", ""),
        answer("200 OK", "", ""),
        answer("404 Not Found", "", ""),
        answer("200 OK", "", ""),
        empty_page(),
    ]);
    let error = minio(endpoint)
        .delete(
            Path::new("/task.md"),
            &Condition::Matches(RevisionToken::new("\"a\"")),
        )
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::PreconditionFailed);
    assert_eq!(
        requests(&seen),
        [
            "HEAD if-match=-",
            "HEAD if-match=-",
            "HEAD if-match=-",
            "HEAD if-match=-",
            "GET if-match=-"
        ]
    );
}

#[cfg(feature = "minio")]
#[test]
fn minio_get_of_a_folder_is_a_directory() {
    let (endpoint, _seen) = canned(vec![
        answer("404 Not Found", "", ""),
        listing(&["d/f.md"], &[]),
    ]);
    let error = minio(endpoint).get(Path::new("/d")).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::IsADirectory);
}

#[cfg(feature = "minio")]
#[test]
fn minio_delete_of_a_folder_is_a_directory_and_sends_no_delete() {
    let (endpoint, seen) = canned(vec![
        answer("404 Not Found", "", ""),
        answer("200 OK", "", ""),
        listing(&["d/f.md"], &[]),
    ]);
    let error = minio(endpoint)
        .delete(Path::new("/d"), &Condition::None)
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::IsADirectory);
    assert_eq!(
        requests(&seen),
        ["HEAD if-match=-", "HEAD if-match=-", "GET if-match=-"]
    );
}

#[cfg(feature = "minio")]
#[test]
fn minio_delete_with_no_condition_checks_the_object_then_sends_one_delete() {
    let (endpoint, seen) = canned(vec![
        answer("200 OK", "ETag: \"a\"\r\n", ""),
        answer("204 No Content", "", ""),
    ]);
    minio(endpoint)
        .delete(Path::new("/task.md"), &Condition::None)
        .unwrap();
    assert_eq!(requests(&seen), ["HEAD if-match=-", "DELETE if-match=-"]);
}

#[cfg(feature = "minio")]
#[test]
fn minio_get_answered_503_is_unavailable_after_one_request() {
    let unavailable = answer("503 Service Unavailable", "", "");
    let (endpoint, seen) = canned(vec![unavailable.clone(), unavailable.clone(), unavailable]);
    let error = minio(endpoint).get(Path::new("/task.md")).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unavailable);
    assert_eq!(requests(&seen), ["GET if-match=-"]);
}

#[cfg(feature = "minio")]
#[test]
fn minio_list_reads_every_page() {
    let (endpoint, seen) = canned(vec![
        page("vault/a.md", Some("next")),
        page("vault/b.md", None),
    ]);
    let entries = minio(endpoint)
        .list(Path::new("/vault"), Depth::All)
        .unwrap();
    assert_eq!(paths(&entries), ["/vault/a.md", "/vault/b.md"]);
    assert_eq!(requests(&seen), ["GET if-match=-", "GET if-match=-"]);
}

#[cfg(feature = "minio")]
#[test]
fn minio_list_of_a_file_is_not_a_directory_at_both_depths() {
    for depth in [Depth::Children, Depth::All] {
        let (endpoint, seen) = canned(vec![empty_page(), answer("200 OK", "ETag: \"a\"\r\n", "")]);
        let error = minio(endpoint)
            .list(Path::new("/vault/a.md"), depth)
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::NotADirectory);
        assert_eq!(requests(&seen), ["GET if-match=-", "HEAD if-match=-"]);
    }
}

#[cfg(feature = "minio")]
#[test]
fn minio_list_of_a_missing_folder_is_empty_after_one_head() {
    let (endpoint, seen) = canned(vec![
        empty_page(),
        not_found_answer("NoSuchKey", "The specified key does not exist."),
        answer("200 OK", "", ""),
    ]);
    let entries = minio(endpoint)
        .list(Path::new("/vault/nothing"), Depth::Children)
        .unwrap();
    assert!(entries.is_empty());
    assert_eq!(
        requests(&seen),
        ["GET if-match=-", "HEAD if-match=-", "HEAD if-match=-"]
    );
}

#[cfg(feature = "minio")]
#[test]
fn minio_list_with_entries_costs_no_head() {
    let (endpoint, seen) = canned(vec![page("vault/a.md", None)]);
    minio(endpoint)
        .list(Path::new("/vault"), Depth::All)
        .unwrap();
    assert_eq!(requests(&seen), ["GET if-match=-"]);
}

#[cfg(feature = "minio")]
#[test]
fn minio_config_debug_keeps_the_secret_key_out() {
    let config = minio_config("http://127.0.0.1:9000".to_owned());
    let shown = format!("{config:?}");
    assert!(shown.contains("<redacted>"));
    assert!(!shown.contains("os-shim-secret"));
}

#[cfg(feature = "minio")]
#[test]
fn minio_get_with_nothing_listening_is_unavailable_naming_the_dispatch_failure_and_its_cause() {
    let error = minio("http://127.0.0.1:9".to_owned())
        .get(Path::new("/task.md"))
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unavailable);
    let message = error.to_string();
    assert!(message.starts_with("dispatch failure: "));
    assert!(message.contains("Connection refused"));
}

#[cfg(feature = "minio")]
#[test]
fn minio_get_answered_403_is_permission_denied_naming_the_status_the_code_and_the_message() {
    let body = "<?xml version=\"1.0\" encoding=\"UTF-8\"?><Error><Code>SignatureDoesNotMatch</Code><Message>The request signature we calculated does not match the signature you provided.</Message></Error>";
    let (endpoint, _seen) = canned(vec![answer(
        "403 Forbidden",
        "Content-Type: application/xml\r\n",
        body,
    )]);
    let error = minio(endpoint).get(Path::new("/task.md")).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::PermissionDenied);
    assert_eq!(
        error.to_string(),
        "403 SignatureDoesNotMatch: The request signature we calculated does not match the signature you provided."
    );
    assert!(
        iter::successors(error.source(), |&cause| cause.source())
            .any(<dyn Error>::is::<SdkError<GetObjectError>>)
    );
}

#[cfg(feature = "minio")]
#[test]
fn minio_get_answered_404_no_such_bucket_is_other_naming_the_answer() {
    let (endpoint, _seen) = canned(vec![no_such_bucket()]);
    let error = minio(endpoint).get(Path::new("/task.md")).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(
        error.to_string(),
        "404 NoSuchBucket: The specified bucket does not exist."
    );
}

#[cfg(feature = "minio")]
#[test]
fn minio_get_answered_404_no_such_key_is_not_found() {
    let (endpoint, _seen) = canned(vec![
        not_found_answer("NoSuchKey", "The specified key does not exist."),
        empty_page(),
    ]);
    let error = minio(endpoint).get(Path::new("/task.md")).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
}

#[cfg(feature = "minio")]
#[test]
fn minio_delete_absent_of_a_missing_key_in_an_existing_bucket_succeeds() {
    let (endpoint, seen) = canned(vec![
        answer("404 Not Found", "", ""),
        answer("200 OK", "", ""),
        empty_page(),
    ]);
    minio(endpoint)
        .delete(Path::new("/task.md"), &Condition::Absent)
        .unwrap();
    assert_eq!(
        requests(&seen),
        ["HEAD if-match=-", "HEAD if-match=-", "GET if-match=-"]
    );
}

#[cfg(feature = "minio")]
#[test]
fn minio_delete_absent_in_a_missing_bucket_is_other_naming_the_bucket() {
    let (endpoint, seen) = canned(vec![
        answer("404 Not Found", "", ""),
        answer("404 Not Found", "", ""),
    ]);
    let error = minio(endpoint)
        .delete(Path::new("/task.md"), &Condition::Absent)
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(error.to_string(), "404: the bucket archie does not exist");
    assert_eq!(requests(&seen), ["HEAD if-match=-", "HEAD if-match=-"]);
}

#[cfg(feature = "minio")]
#[test]
fn minio_delete_with_a_tag_in_a_missing_bucket_is_other() {
    let (endpoint, _seen) = canned(vec![
        answer("404 Not Found", "", ""),
        answer("404 Not Found", "", ""),
    ]);
    let error = minio(endpoint)
        .delete(
            Path::new("/task.md"),
            &Condition::Matches(RevisionToken::new("\"a\"")),
        )
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Other);
}

#[cfg(feature = "minio")]
#[test]
fn minio_put_with_a_tag_answered_404_no_such_bucket_is_other() {
    let (endpoint, seen) = canned(vec![empty_page(), no_such_bucket()]);
    let error = minio(endpoint)
        .put(
            Path::new("/task.md"),
            b"x",
            &Condition::Matches(RevisionToken::new("\"a\"")),
        )
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(requests(&seen), ["GET if-match=-", "PUT if-match=\"a\""]);
}

#[cfg(feature = "minio")]
#[test]
fn minio_head_gives_the_size_and_the_tag_from_one_head() {
    let (endpoint, seen) = canned(vec![head_answer("ETag: \"a\"\r\nContent-Length: 5\r\n")]);
    let metadata = minio(endpoint).head(Path::new("/task.md")).unwrap();
    assert_eq!(
        (
            metadata.is_file,
            metadata.is_dir,
            metadata.len,
            metadata.modified
        ),
        (true, false, 5, None)
    );
    assert_eq!(metadata.revision, Some(RevisionToken::new("\"a\"")));
    assert_eq!(requests(&seen), ["HEAD if-match=-"]);
}

#[cfg(feature = "minio")]
#[test]
fn minio_head_of_a_missing_object_in_an_existing_bucket_is_not_found() {
    let (endpoint, seen) = canned(vec![
        answer("404 Not Found", "", ""),
        answer("200 OK", "", ""),
        empty_page(),
    ]);
    let error = minio(endpoint).head(Path::new("/task.md")).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert_eq!(
        requests(&seen),
        ["HEAD if-match=-", "HEAD if-match=-", "GET if-match=-"]
    );
}

#[cfg(feature = "minio")]
#[test]
fn minio_head_in_a_missing_bucket_is_other_naming_the_bucket() {
    let (endpoint, seen) = canned(vec![
        answer("404 Not Found", "", ""),
        answer("404 Not Found", "", ""),
    ]);
    let error = minio(endpoint).head(Path::new("/task.md")).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(error.to_string(), "404: the bucket archie does not exist");
    assert_eq!(requests(&seen), ["HEAD if-match=-", "HEAD if-match=-"]);
}

#[cfg(feature = "minio")]
#[test]
fn minio_head_without_an_etag_is_other() {
    let (endpoint, _seen) = canned(vec![head_answer("Content-Length: 5\r\n")]);
    let error = minio(endpoint).head(Path::new("/task.md")).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(error.to_string(), "the answer carries no ETag header");
}

#[cfg(feature = "minio")]
#[test]
fn minio_head_without_a_content_length_is_other() {
    let (endpoint, _seen) = canned(vec![head_answer("ETag: \"a\"\r\n")]);
    let error = minio(endpoint).head(Path::new("/task.md")).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(
        error.to_string(),
        "the answer carries no Content-Length header"
    );
}

/// A 403 answer whose S3 error body carries `AccessDenied`.
#[cfg(any(feature = "minio", feature = "s3"))]
fn access_denied() -> String {
    answer(
        "403 Forbidden",
        "Content-Type: application/xml\r\n",
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><Error><Code>AccessDenied</Code><Message>Access Denied.</Message></Error>",
    )
}

#[cfg(feature = "minio")]
#[test]
fn minio_ensure_on_an_existing_bucket_sends_one_head() {
    let (endpoint, seen) = canned(vec![answer("200 OK", "", "")]);
    let bucket = minio(endpoint);
    bucket.runtime.block_on(bucket.bucket.ensure()).unwrap();
    assert_eq!(requests(&seen), ["HEAD if-match=-"]);
}

#[cfg(feature = "minio")]
#[test]
fn minio_ensure_creates_a_missing_bucket_with_no_acl_and_no_location() {
    let (endpoint, seen) = canned(vec![
        answer("404 Not Found", "", ""),
        answer("200 OK", "", ""),
    ]);
    let bucket = minio(endpoint);
    bucket.runtime.block_on(bucket.bucket.ensure()).unwrap();
    assert_eq!(requests(&seen), ["HEAD if-match=-", "PUT if-match=-"]);
}

#[cfg(feature = "minio")]
#[test]
fn minio_ensure_with_the_head_refused_is_permission_denied_and_creates_nothing() {
    let (endpoint, seen) = canned(vec![access_denied()]);
    let bucket = minio(endpoint);
    let error = bucket.runtime.block_on(bucket.bucket.ensure()).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::PermissionDenied);
    assert_eq!(requests(&seen), ["HEAD if-match=-"]);
}

#[cfg(feature = "minio")]
#[test]
fn minio_ensure_with_the_create_refused_is_permission_denied_naming_the_answer() {
    let (endpoint, seen) = canned(vec![answer("404 Not Found", "", ""), access_denied()]);
    let bucket = minio(endpoint);
    let error = bucket.runtime.block_on(bucket.bucket.ensure()).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::PermissionDenied);
    assert_eq!(error.to_string(), "403 AccessDenied: Access Denied.");
    assert_eq!(requests(&seen), ["HEAD if-match=-", "PUT if-match=-"]);
}

/// The configuration every AWS S3 scenario uses.
#[cfg(feature = "s3")]
fn s3_config() -> S3Config {
    S3Config::new(
        "us-east-1".to_owned(),
        "archie".to_owned(),
        "os-shim".to_owned(),
        "os-shim-secret".to_owned(),
    )
}

/// A blocking AWS S3 client that reaches the canned server at `endpoint`.
#[cfg(feature = "s3")]
fn s3(endpoint: &str) -> BlockingBucket<S3Bucket> {
    s3_in(endpoint, "us-east-1")
}

/// A blocking AWS S3 client in `region` that reaches the canned server at `endpoint`.
#[cfg(feature = "s3")]
fn s3_in(endpoint: &str, region: &str) -> BlockingBucket<S3Bucket> {
    let mut config = s3_config();
    region.clone_into(&mut config.region);
    let sdk = S3Bucket::sdk_config(config)
        .endpoint_url(endpoint)
        .force_path_style(true)
        .build();
    BlockingBucket::new(S3Bucket::with_sdk_config(
        "archie".to_owned(),
        String::new(),
        sdk,
    ))
    .unwrap()
}

#[cfg(feature = "s3")]
#[test]
fn s3_delete_with_the_current_tag_sends_one_delete_with_if_match() {
    let (endpoint, seen) = canned(vec![answer("204 No Content", "", "")]);
    s3(&endpoint)
        .delete(
            Path::new("/task.md"),
            &Condition::Matches(RevisionToken::new("\"a\"")),
        )
        .unwrap();
    assert_eq!(requests(&seen), ["DELETE if-match=\"a\""]);
}

#[cfg(feature = "s3")]
#[test]
fn s3_delete_with_a_stale_tag_answered_412_is_precondition_failed() {
    let (endpoint, seen) = canned(vec![
        answer("412 Precondition Failed", "", ""),
        answer("200 OK", "ETag: \"b\"\r\n", ""),
    ]);
    let error = s3(&endpoint)
        .delete(
            Path::new("/task.md"),
            &Condition::Matches(RevisionToken::new("\"a\"")),
        )
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::PreconditionFailed);
    assert_eq!(
        requests(&seen),
        ["DELETE if-match=\"a\"", "HEAD if-match=-"]
    );
}

#[cfg(feature = "s3")]
#[test]
fn s3_delete_with_a_tag_of_a_missing_object_is_precondition_failed() {
    let (endpoint, seen) = canned(vec![
        answer("404 Not Found", "", ""),
        answer("404 Not Found", "", ""),
        answer("200 OK", "", ""),
        empty_page(),
    ]);
    let error = s3(&endpoint)
        .delete(
            Path::new("/task.md"),
            &Condition::Matches(RevisionToken::new("\"a\"")),
        )
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::PreconditionFailed);
    assert_eq!(
        requests(&seen),
        [
            "DELETE if-match=\"a\"",
            "HEAD if-match=-",
            "HEAD if-match=-",
            "GET if-match=-"
        ]
    );
}

#[cfg(feature = "s3")]
#[test]
fn s3_delete_with_a_tag_answered_404_no_such_bucket_is_other() {
    let (endpoint, seen) = canned(vec![no_such_bucket()]);
    let error = s3(&endpoint)
        .delete(
            Path::new("/task.md"),
            &Condition::Matches(RevisionToken::new("\"a\"")),
        )
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(requests(&seen), ["DELETE if-match=\"a\""]);
}

#[cfg(feature = "s3")]
#[test]
fn s3_delete_with_no_condition_checks_the_object_then_sends_one_delete() {
    let (endpoint, seen) = canned(vec![
        answer("200 OK", "ETag: \"a\"\r\n", ""),
        answer("204 No Content", "", ""),
    ]);
    s3(&endpoint)
        .delete(Path::new("/task.md"), &Condition::None)
        .unwrap();
    assert_eq!(requests(&seen), ["HEAD if-match=-", "DELETE if-match=-"]);
}

#[cfg(feature = "s3")]
#[test]
fn s3_get_answered_404_is_not_found() {
    let (endpoint, _seen) = canned(vec![answer("404 Not Found", "", ""), empty_page()]);
    let error = s3(&endpoint).get(Path::new("/task.md")).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
}

#[cfg(feature = "s3")]
#[test]
fn s3_put_with_absent_answered_412_is_already_exists() {
    let (endpoint, _seen) = canned(vec![
        empty_page(),
        answer("412 Precondition Failed", "", ""),
    ]);
    let error = s3(&endpoint)
        .put(Path::new("/task.md"), b"x", &Condition::Absent)
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::AlreadyExists);
}

#[cfg(feature = "s3")]
#[test]
fn s3_get_answered_503_is_unavailable_after_one_request() {
    let unavailable = answer("503 Service Unavailable", "", "");
    let (endpoint, seen) = canned(vec![unavailable.clone(), unavailable.clone(), unavailable]);
    let error = s3(&endpoint).get(Path::new("/task.md")).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unavailable);
    assert_eq!(requests(&seen), ["GET if-match=-"]);
}

#[cfg(feature = "s3")]
#[test]
fn s3_head_gives_the_size_and_the_tag_from_one_head() {
    let (endpoint, seen) = canned(vec![head_answer("ETag: \"a\"\r\nContent-Length: 5\r\n")]);
    let metadata = s3(&endpoint).head(Path::new("/task.md")).unwrap();
    assert_eq!(
        (
            metadata.is_file,
            metadata.is_dir,
            metadata.len,
            metadata.modified
        ),
        (true, false, 5, None)
    );
    assert_eq!(metadata.revision, Some(RevisionToken::new("\"a\"")));
    assert_eq!(requests(&seen), ["HEAD if-match=-"]);
}

#[cfg(feature = "s3")]
#[test]
fn s3_head_of_a_missing_object_in_an_existing_bucket_is_not_found() {
    let (endpoint, seen) = canned(vec![
        answer("404 Not Found", "", ""),
        answer("200 OK", "", ""),
        empty_page(),
    ]);
    let error = s3(&endpoint).head(Path::new("/task.md")).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert_eq!(
        requests(&seen),
        ["HEAD if-match=-", "HEAD if-match=-", "GET if-match=-"]
    );
}

#[cfg(feature = "s3")]
#[test]
fn s3_ensure_on_an_existing_bucket_sends_one_head() {
    let (endpoint, seen) = canned(vec![answer("200 OK", "", "")]);
    let bucket = s3(&endpoint);
    bucket
        .runtime
        .block_on(
            bucket
                .bucket
                .ensure(Some("arn:aws:kms:us-east-1:111122223333:key/a")),
        )
        .unwrap();
    assert_eq!(requests(&seen), ["HEAD if-match=-"]);
}

#[cfg(feature = "s3")]
#[test]
fn s3_ensure_with_a_key_creates_a_private_bucket_then_sets_its_encryption() {
    let (endpoint, seen) = canned(vec![
        answer("404 Not Found", "", ""),
        answer("200 OK", "", ""),
        answer("200 OK", "", ""),
    ]);
    let bucket = s3(&endpoint);
    bucket
        .runtime
        .block_on(
            bucket
                .bucket
                .ensure(Some("arn:aws:kms:us-east-1:111122223333:key/a")),
        )
        .unwrap();
    assert_eq!(
        requests(&seen),
        [
            "HEAD if-match=-",
            "PUT if-match=- acl=private",
            "PUT if-match=-"
        ]
    );
}

#[cfg(feature = "s3")]
#[test]
fn s3_ensure_without_a_key_outside_us_east_1_names_the_region_and_sets_no_encryption() {
    let (endpoint, seen) = canned(vec![
        answer("404 Not Found", "", ""),
        answer("200 OK", "", ""),
    ]);
    let bucket = s3_in(&endpoint, "eu-west-1");
    bucket.runtime.block_on(bucket.bucket.ensure(None)).unwrap();
    assert_eq!(
        requests(&seen),
        [
            "HEAD if-match=-",
            "PUT if-match=- acl=private location=eu-west-1"
        ]
    );
}

#[cfg(feature = "s3")]
#[test]
fn s3_ensure_with_the_encryption_refused_is_permission_denied_naming_the_answer() {
    let (endpoint, seen) = canned(vec![
        answer("404 Not Found", "", ""),
        answer("200 OK", "", ""),
        access_denied(),
    ]);
    let bucket = s3(&endpoint);
    let error = bucket
        .runtime
        .block_on(
            bucket
                .bucket
                .ensure(Some("arn:aws:kms:us-east-1:111122223333:key/a")),
        )
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::PermissionDenied);
    assert_eq!(error.to_string(), "403 AccessDenied: Access Denied.");
    assert_eq!(requests(&seen).len(), 3);
}

#[cfg(feature = "s3")]
#[test]
fn s3_config_debug_keeps_the_secret_key_out() {
    let shown = format!("{:?}", s3_config());
    assert!(shown.contains("<redacted>"));
    assert!(!shown.contains("os-shim-secret"));
}

#[cfg(feature = "minio")]
#[test]
fn minio_put_under_a_prefix_puts_every_key_under_it() {
    let (endpoint, _seen, targets) = canned_with_targets(vec![
        empty_page(),
        page("tenant-7/notes/x.md", None),
        answer("200 OK", "ETag: \"b\"\r\n", ""),
    ]);
    let config = minio_config(endpoint).with_prefix("tenant-7/");
    assert_eq!(config.prefix.as_deref(), Some("tenant-7"));
    BlockingBucket::new(MinioBucket::new(config))
        .unwrap()
        .put(Path::new("/notes/a.md"), b"x", &Condition::None)
        .unwrap();
    assert_eq!(
        *targets.lock().unwrap(),
        [
            "GET /archie/?list-type=2&max-keys=1&prefix=tenant-7%2Fnotes%2Fa.md%2F",
            "GET /archie/?list-type=2&max-keys=1&prefix=tenant-7%2Fnotes%2F",
            "PUT /archie/tenant-7/notes/a.md?x-id=PutObject"
        ]
    );
}

#[cfg(feature = "minio")]
#[test]
fn minio_delete_absent_over_an_object_is_already_exists() {
    let (endpoint, seen) = canned(vec![
        answer("200 OK", "ETag: \"a\"\r\n", ""),
        answer("200 OK", "ETag: \"a\"\r\n", ""),
    ]);
    let error = minio(endpoint)
        .delete(Path::new("/task.md"), &Condition::Absent)
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::AlreadyExists);
    assert_eq!(requests(&seen), ["HEAD if-match=-", "HEAD if-match=-"]);
}

#[cfg(feature = "minio")]
#[test]
fn minio_head_reads_the_modification_time_from_last_modified() {
    let (endpoint, _seen) = canned(vec![head_answer(
        "ETag: \"a\"\r\nContent-Length: 5\r\nLast-Modified: Thu, 01 Jan 1970 00:16:40 GMT\r\n",
    )]);
    let modified = minio(endpoint)
        .head(Path::new("/task.md"))
        .unwrap()
        .modified;
    assert_eq!(modified, UNIX_EPOCH.checked_add(Duration::from_secs(1_000)));
}

#[cfg(feature = "minio")]
#[test]
fn minio_list_of_children_groups_the_keys_by_the_next_component() {
    let (endpoint, _seen, targets) = canned_with_targets(vec![page("vault/a.md", None)]);
    let entries = minio(endpoint)
        .list(Path::new("/vault"), Depth::Children)
        .unwrap();
    assert_eq!(paths(&entries), ["/vault/a.md"]);
    assert!(entries.iter().all(|entry| entry.is_file && !entry.is_dir));
    assert_eq!(targets.lock().unwrap().len(), 1);
}

#[cfg(feature = "s3")]
#[test]
fn s3_put_under_a_prefix_puts_every_key_under_it() {
    let (endpoint, _seen, targets) = canned_with_targets(vec![
        empty_page(),
        page("tenant-7/notes/x.md", None),
        answer("200 OK", "ETag: \"b\"\r\n", ""),
    ]);
    let config = s3_config().with_prefix("tenant-7/");
    let prefix = config.prefix.clone().unwrap_or_default();
    let sdk = S3Bucket::sdk_config(config)
        .endpoint_url(endpoint)
        .force_path_style(true)
        .build();
    BlockingBucket::new(S3Bucket::with_sdk_config("archie".to_owned(), prefix, sdk))
        .unwrap()
        .put(Path::new("/notes/a.md"), b"x", &Condition::None)
        .unwrap();
    assert_eq!(
        *targets.lock().unwrap(),
        [
            "GET /archie/?list-type=2&max-keys=1&prefix=tenant-7%2Fnotes%2Fa.md%2F",
            "GET /archie/?list-type=2&max-keys=1&prefix=tenant-7%2Fnotes%2F",
            "PUT /archie/tenant-7/notes/a.md?x-id=PutObject"
        ]
    );
}

#[cfg(feature = "s3")]
#[test]
fn s3_delete_absent_over_an_object_is_already_exists() {
    let (endpoint, _seen) = canned(vec![
        answer("200 OK", "ETag: \"a\"\r\n", ""),
        answer("200 OK", "ETag: \"a\"\r\n", ""),
    ]);
    let error = s3(&endpoint)
        .delete(Path::new("/task.md"), &Condition::Absent)
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::AlreadyExists);
}

#[cfg(feature = "minio")]
#[test]
fn minio_create_folder_writes_a_marker_for_every_missing_level_highest_first() {
    let missing = || answer("404 Not Found", "", "");
    let (endpoint, _seen, targets) = canned_with_targets(vec![
        empty_page(),
        empty_page(),
        missing(),
        answer("200 OK", "", ""),
        answer("200 OK", "", ""),
        missing(),
        answer("200 OK", "", ""),
        answer("200 OK", "", ""),
    ]);
    minio(endpoint).create_folder(Path::new("/a/b")).unwrap();
    assert_eq!(
        *targets.lock().unwrap(),
        [
            "GET /archie/?list-type=2&max-keys=1&prefix=a%2Fb%2F",
            "GET /archie/?list-type=2&max-keys=1&prefix=a%2F",
            "HEAD /archie/a",
            "HEAD /archie/",
            "PUT /archie/a/?x-id=PutObject",
            "HEAD /archie/a/b",
            "HEAD /archie/",
            "PUT /archie/a/b/?x-id=PutObject",
        ]
    );
}

#[cfg(feature = "minio")]
#[test]
fn minio_create_folder_over_an_object_at_the_path_is_already_exists_and_writes_nothing() {
    let (endpoint, _seen, targets) = canned_with_targets(vec![
        empty_page(),
        head_answer("ETag: \"a\"\r\nContent-Length: 1\r\n"),
    ]);
    let error = minio(endpoint).create_folder(Path::new("/a")).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::AlreadyExists);
    assert_eq!(
        *targets.lock().unwrap(),
        [
            "GET /archie/?list-type=2&max-keys=1&prefix=a%2F",
            "HEAD /archie/a"
        ]
    );
}

#[cfg(feature = "minio")]
#[test]
fn minio_create_folder_below_an_object_is_not_a_directory() {
    let (endpoint, seen) = canned(vec![
        empty_page(),
        empty_page(),
        head_answer("ETag: \"a\"\r\nContent-Length: 1\r\n"),
    ]);
    let error = minio(endpoint)
        .create_folder(Path::new("/a/b"))
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotADirectory);
    assert_eq!(
        requests(&seen),
        ["GET if-match=-", "GET if-match=-", "HEAD if-match=-"]
    );
}

#[cfg(feature = "minio")]
#[test]
fn minio_create_folder_skips_a_level_that_has_keys_below_it() {
    let (endpoint, _seen, targets) = canned_with_targets(vec![listing(&["a/f.md"], &[])]);
    minio(endpoint).create_folder(Path::new("/a")).unwrap();
    assert_eq!(targets.lock().unwrap().len(), 1);
}

#[cfg(feature = "minio")]
#[test]
fn minio_remove_folder_deletes_every_key_and_marker_below_it() {
    let (endpoint, _seen, targets) = canned_with_targets(vec![
        listing(&["x/", "x/f.md", "x/y/"], &[]),
        answer("204 No Content", "", ""),
        answer("204 No Content", "", ""),
        answer("204 No Content", "", ""),
    ]);
    minio(endpoint).remove_folder(Path::new("/x")).unwrap();
    assert_eq!(
        *targets.lock().unwrap(),
        [
            "GET /archie/?list-type=2&prefix=x%2F",
            "DELETE /archie/x/?x-id=DeleteObject",
            "DELETE /archie/x/f.md?x-id=DeleteObject",
            "DELETE /archie/x/y/?x-id=DeleteObject",
        ]
    );
}

#[cfg(feature = "minio")]
#[test]
fn minio_remove_folder_with_no_marker_and_no_key_is_not_found() {
    let (endpoint, _seen) = canned(vec![
        empty_page(),
        answer("404 Not Found", "", ""),
        answer("200 OK", "", ""),
        empty_page(),
    ]);
    let error = minio(endpoint).remove_folder(Path::new("/x")).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
}

#[cfg(feature = "minio")]
#[test]
fn minio_remove_folder_of_an_object_with_nothing_below_is_not_a_directory() {
    let (endpoint, seen) = canned(vec![
        empty_page(),
        head_answer("ETag: \"a\"\r\nContent-Length: 1\r\n"),
    ]);
    let error = minio(endpoint).remove_folder(Path::new("/x")).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotADirectory);
    assert_eq!(requests(&seen), ["GET if-match=-", "HEAD if-match=-"]);
}

#[cfg(feature = "minio")]
#[test]
fn minio_list_of_children_sends_the_delimiter_and_reads_folders_from_the_common_prefixes() {
    let (endpoint, _seen, targets) = canned_with_targets(vec![listing(
        &["vault/", "vault/b.md", "vault/a"],
        &["vault/sub/", "vault/a/"],
    )]);
    let entries = minio(endpoint)
        .list(Path::new("/vault"), Depth::Children)
        .unwrap();
    assert_eq!(
        *targets.lock().unwrap(),
        ["GET /archie/?list-type=2&delimiter=%2F&prefix=vault%2F"]
    );
    assert_eq!(paths(&entries), ["/vault/a", "/vault/b.md", "/vault/sub"]);
    let kinds: Vec<(bool, bool)> = entries
        .iter()
        .map(|entry| (entry.is_dir, entry.is_file))
        .collect();
    assert_eq!(kinds, [(false, true), (false, true), (true, false)]);
}

#[cfg(feature = "minio")]
#[test]
fn minio_list_of_everything_turns_markers_and_implied_levels_into_folders_once() {
    let (endpoint, _seen, targets) = canned_with_targets(vec![listing(
        &[
            "vault/",
            "vault/sub/",
            "vault/sub/f.md",
            "vault/deep/er/g.md",
            "vault/a.md",
        ],
        &[],
    )]);
    let entries = minio(endpoint)
        .list(Path::new("/vault"), Depth::All)
        .unwrap();
    assert_eq!(
        *targets.lock().unwrap(),
        ["GET /archie/?list-type=2&prefix=vault%2F"]
    );
    let shown: Vec<String> = entries
        .iter()
        .map(|entry| {
            format!(
                "{}{}",
                entry.path.display(),
                if entry.is_dir { "/" } else { "" }
            )
        })
        .collect();
    assert_eq!(
        shown,
        [
            "/vault/a.md",
            "/vault/deep/",
            "/vault/deep/er/",
            "/vault/deep/er/g.md",
            "/vault/sub/",
            "/vault/sub/f.md",
        ]
    );
}

#[cfg(feature = "minio")]
#[test]
fn minio_list_of_everything_gives_the_file_and_nothing_below_it_when_a_file_and_a_folder_share_a_path()
 {
    let (endpoint, _seen) = canned(vec![listing(
        &["vault/a", "vault/a/", "vault/a/b", "vault/a/c/d"],
        &[],
    )]);
    let entries = minio(endpoint)
        .list(Path::new("/vault"), Depth::All)
        .unwrap();
    assert_eq!(paths(&entries), ["/vault/a"]);
    assert!(entries.iter().all(|entry| entry.is_file));
}

#[cfg(feature = "minio")]
#[test]
fn minio_head_of_a_folder_with_a_marker_gives_the_marker_time_and_no_revision() {
    let (endpoint, seen) = canned(vec![
        answer("404 Not Found", "", ""),
        answer("200 OK", "", ""),
        listing(&["x/", "x/f.md"], &[]),
    ]);
    let metadata = minio(endpoint).head(Path::new("/x")).unwrap();
    assert_eq!(
        (metadata.is_dir, metadata.is_file, metadata.len),
        (true, false, 0)
    );
    assert_eq!(metadata.revision, None);
    assert_eq!(
        metadata.modified,
        UNIX_EPOCH.checked_add(Duration::from_secs(1_000))
    );
    assert_eq!(
        requests(&seen),
        ["HEAD if-match=-", "HEAD if-match=-", "GET if-match=-"]
    );
}

#[cfg(feature = "minio")]
#[test]
fn minio_head_of_a_folder_with_keys_and_no_marker_has_no_time() {
    let (endpoint, _seen) = canned(vec![
        answer("404 Not Found", "", ""),
        answer("200 OK", "", ""),
        listing(&["x/f.md"], &[]),
    ]);
    let metadata = minio(endpoint).head(Path::new("/x")).unwrap();
    assert_eq!((metadata.is_dir, metadata.modified), (true, None));
    assert_eq!(metadata.revision, None);
}

#[cfg(feature = "minio")]
#[test]
fn minio_head_of_the_root_is_a_folder_without_a_request() {
    let (endpoint, seen) = canned(vec![]);
    let metadata = minio(endpoint).head(Path::new("/")).unwrap();
    assert!(metadata.is_dir && !metadata.is_file);
    assert_eq!(requests(&seen), Vec::<String>::new());
}

#[cfg(feature = "minio")]
#[test]
fn minio_put_with_keys_below_the_path_is_a_directory_and_sends_no_put() {
    let (endpoint, seen) = canned(vec![listing(&["task.md/f"], &[])]);
    let error = minio(endpoint)
        .put(Path::new("/task.md"), b"x", &Condition::None)
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::IsADirectory);
    assert_eq!(requests(&seen), ["GET if-match=-"]);
}

#[cfg(feature = "minio")]
#[test]
fn minio_put_into_a_missing_parent_is_not_found_and_sends_no_put() {
    let (endpoint, seen) = canned(vec![
        empty_page(),
        empty_page(),
        answer("404 Not Found", "", ""),
        answer("200 OK", "", ""),
    ]);
    let error = minio(endpoint)
        .put(Path::new("/d/task.md"), b"x", &Condition::None)
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert_eq!(
        requests(&seen),
        [
            "GET if-match=-",
            "GET if-match=-",
            "HEAD if-match=-",
            "HEAD if-match=-"
        ]
    );
}

#[cfg(feature = "minio")]
#[test]
fn minio_put_under_an_object_is_not_a_directory() {
    let (endpoint, _seen) = canned(vec![
        empty_page(),
        empty_page(),
        head_answer("ETag: \"a\"\r\nContent-Length: 1\r\n"),
    ]);
    let error = minio(endpoint)
        .put(Path::new("/d/task.md"), b"x", &Condition::None)
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotADirectory);
}

#[cfg(feature = "minio")]
#[test]
fn minio_append_to_a_missing_parent_is_not_found() {
    let (endpoint, seen) = canned(vec![
        answer("404 Not Found", "", ""),
        answer("404 Not Found", "", ""),
        answer("200 OK", "", ""),
        empty_page(),
        empty_page(),
        empty_page(),
        answer("404 Not Found", "", ""),
        answer("200 OK", "", ""),
    ]);
    let error = minio(endpoint)
        .append(Path::new("/d/log.md"), b"x")
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert_eq!(requests(&seen).len(), 8);
}

#[cfg(feature = "minio")]
#[test]
fn minio_copy_sends_copy_object_and_gives_the_length_of_the_source() {
    let (endpoint, seen, targets) = canned_with_targets(vec![
        head_answer("ETag: \"a\"\r\nContent-Length: 5\r\n"),
        empty_page(),
        copied("c"),
    ]);
    let length = minio(endpoint)
        .copy(Path::new("/from me.md"), Path::new("/to.md"))
        .unwrap();
    assert_eq!(length, 5);
    assert_eq!(
        requests(&seen).last().map(String::as_str),
        Some("PUT if-match=- copy-source=archie/from%20me.md")
    );
    assert_eq!(
        targets.lock().unwrap().last().map(String::as_str),
        Some("PUT /archie/to.md?x-id=CopyObject")
    );
}

#[cfg(feature = "minio")]
#[test]
fn minio_copy_of_a_missing_source_is_not_found_and_copies_nothing() {
    let (endpoint, seen) = canned(vec![
        answer("404 Not Found", "", ""),
        answer("200 OK", "", ""),
        empty_page(),
    ]);
    let error = minio(endpoint)
        .copy(Path::new("/from.md"), Path::new("/to.md"))
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert!(requests(&seen).iter().all(|line| !line.starts_with("PUT")));
}

#[cfg(feature = "minio")]
#[test]
fn minio_rename_of_a_file_copies_then_deletes_the_source_under_its_tag() {
    let (endpoint, seen) = canned(vec![
        head_answer("ETag: \"a\"\r\nContent-Length: 5\r\n"),
        empty_page(),
        copied("c"),
        answer("200 OK", "ETag: \"a\"\r\n", ""),
        answer("204 No Content", "", ""),
    ]);
    minio(endpoint)
        .rename(Path::new("/from.md"), Path::new("/to.md"))
        .unwrap();
    assert_eq!(
        requests(&seen),
        [
            "HEAD if-match=-",
            "GET if-match=-",
            "PUT if-match=- copy-source=archie/from.md",
            "HEAD if-match=-",
            "DELETE if-match=\"a\"",
        ]
    );
}

#[cfg(feature = "minio")]
#[test]
fn minio_rename_whose_delete_finds_another_tag_removes_the_copy_again() {
    let (endpoint, seen) = canned(vec![
        head_answer("ETag: \"a\"\r\nContent-Length: 5\r\n"),
        empty_page(),
        copied("c"),
        answer("200 OK", "ETag: \"z\"\r\n", ""),
        answer("200 OK", "ETag: \"c\"\r\n", ""),
        answer("204 No Content", "", ""),
    ]);
    let error = minio(endpoint)
        .rename(Path::new("/from.md"), Path::new("/to.md"))
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::PreconditionFailed);
    assert_eq!(
        requests(&seen).last().map(String::as_str),
        Some("DELETE if-match=\"c\"")
    );
}

#[cfg(feature = "s3")]
#[test]
fn s3_rename_whose_delete_is_answered_412_deletes_the_copy_and_returns_the_failure() {
    let (endpoint, seen, targets) = canned_with_targets(vec![
        head_answer("ETag: \"a\"\r\nContent-Length: 5\r\n"),
        empty_page(),
        copied("c"),
        answer("412 Precondition Failed", "", ""),
        answer("204 No Content", "", ""),
    ]);
    let error = s3(&endpoint)
        .rename(Path::new("/from.md"), Path::new("/to.md"))
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::PreconditionFailed);
    assert_eq!(
        requests(&seen),
        [
            "HEAD if-match=-",
            "GET if-match=-",
            "PUT if-match=- copy-source=archie/from.md",
            "DELETE if-match=\"a\"",
            "DELETE if-match=\"c\"",
        ]
    );
    assert_eq!(
        targets.lock().unwrap().last().map(String::as_str),
        Some("DELETE /archie/to.md?x-id=DeleteObject")
    );
}

#[cfg(feature = "minio")]
#[test]
fn minio_rename_of_a_folder_moves_every_key_and_marker_below_it() {
    let missing = || answer("404 Not Found", "", "");
    let (endpoint, _seen, targets) = canned_with_targets(vec![
        missing(),
        answer("200 OK", "", ""),
        listing(&["s/", "s/f.md"], &[]),
        missing(),
        answer("200 OK", "", ""),
        empty_page(),
        empty_page(),
        missing(),
        answer("200 OK", "", ""),
        answer("200 OK", "", ""),
        listing(&["s/", "s/f.md"], &[]),
        head_answer("ETag: \"m\"\r\nContent-Length: 0\r\n"),
        copied("m"),
        head_answer("ETag: \"m\"\r\n"),
        answer("204 No Content", "", ""),
        head_answer("ETag: \"f\"\r\nContent-Length: 1\r\n"),
        copied("f"),
        head_answer("ETag: \"f\"\r\n"),
        answer("204 No Content", "", ""),
    ]);
    minio(endpoint)
        .rename(Path::new("/s"), Path::new("/t"))
        .unwrap();
    let log = targets.lock().unwrap().clone();
    let copies: Vec<&String> = log
        .iter()
        .filter(|line| line.contains("CopyObject"))
        .collect();
    assert_eq!(
        copies,
        [
            "PUT /archie/t/?x-id=CopyObject",
            "PUT /archie/t/f.md?x-id=CopyObject"
        ]
    );
    let deletes: Vec<&String> = log
        .iter()
        .filter(|line| line.starts_with("DELETE"))
        .collect();
    assert_eq!(
        deletes,
        [
            "DELETE /archie/s/?x-id=DeleteObject",
            "DELETE /archie/s/f.md?x-id=DeleteObject"
        ]
    );
}

#[cfg(feature = "minio")]
#[test]
fn minio_rename_of_a_folder_into_itself_is_invalid_input() {
    let (endpoint, _seen) = canned(vec![
        answer("404 Not Found", "", ""),
        answer("200 OK", "", ""),
        listing(&["s/f.md"], &[]),
        answer("404 Not Found", "", ""),
        answer("200 OK", "", ""),
        answer("404 Not Found", "", ""),
        answer("200 OK", "", ""),
        listing(&["s/f.md"], &[]),
    ]);
    let error = minio(endpoint)
        .rename(Path::new("/s"), Path::new("/s/inside"))
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidInput);
}

#[cfg(feature = "minio")]
#[test]
fn minio_answers_500_502_and_504_are_unavailable_and_403_is_permission_denied() {
    for status in [
        "500 Internal Server Error",
        "502 Bad Gateway",
        "504 Gateway Timeout",
    ] {
        let (endpoint, _seen) = canned(vec![answer(status, "", "")]);
        let error = minio(endpoint).get(Path::new("/task.md")).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unavailable, "{status}");
    }
    let (endpoint, _seen) = canned(vec![access_denied()]);
    let error = minio(endpoint).get(Path::new("/task.md")).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::PermissionDenied);
}

#[cfg(feature = "minio")]
#[test]
fn minio_guarantees_hold_the_one_step_writes_and_empty_folders_only() {
    let held = MinioBucket::new(minio_config("http://127.0.0.1:9".to_owned())).guarantees();
    for guarantee in EVERY_GUARANTEE {
        assert_eq!(
            held.contains(guarantee),
            SDK_GUARANTEES.contains(&guarantee),
            "{guarantee:?}"
        );
    }
}

#[cfg(feature = "s3")]
#[test]
fn s3_guarantees_add_the_conditional_delete_to_the_minio_ones() {
    let held = S3Bucket::new(s3_config()).guarantees();
    for guarantee in EVERY_GUARANTEE {
        assert_eq!(
            held.contains(guarantee),
            SDK_GUARANTEES.contains(&guarantee)
                || guarantee == Guarantee::ConditionalDeleteInOneStep,
            "{guarantee:?}"
        );
    }
}

/// A `CreateMultipartUpload` answer for the upload `u1`.
#[cfg(feature = "minio")]
fn upload_started() -> String {
    answer(
        "200 OK",
        "Content-Type: application/xml\r\n",
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><InitiateMultipartUploadResult><Bucket>archie</Bucket><Key>big.bin</Key><UploadId>u1</UploadId></InitiateMultipartUploadResult>",
    )
}

/// An `UploadPart` answer carrying the entity tag `tag`.
#[cfg(feature = "minio")]
fn part_stored(tag: &str) -> String {
    answer("200 OK", &format!("ETag: \"{tag}\"\r\n"), "")
}

/// A `CompleteMultipartUpload` answer.
#[cfg(feature = "minio")]
fn upload_completed() -> String {
    answer(
        "200 OK",
        "Content-Type: application/xml\r\n",
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><CompleteMultipartUploadResult><Bucket>archie</Bucket><Key>big.bin</Key><ETag>&quot;done&quot;</ETag></CompleteMultipartUploadResult>",
    )
}

/// An `UploadPartCopy` answer carrying the entity tag `tag`.
#[cfg(feature = "minio")]
fn part_copied(tag: &str) -> String {
    answer(
        "200 OK",
        "Content-Type: application/xml\r\n",
        &format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?><CopyPartResult><ETag>&quot;{tag}&quot;</ETag></CopyPartResult>"
        ),
    )
}

/// A `HEAD` answer for an object of `length` bytes with the entity tag `t`.
#[cfg(feature = "minio")]
fn sized(length: usize) -> String {
    head_answer(&format!("ETag: \"t\"\r\nContent-Length: {length}\r\n"))
}

/// A ranged `GET` answer with `length` bytes.
#[cfg(feature = "minio")]
fn range_of(length: usize) -> String {
    answer(
        "206 Partial Content",
        "ETag: \"t\"\r\n",
        &"x".repeat(length),
    )
}

/// Read `reader` to its end and count the bytes.
#[cfg(feature = "minio")]
async fn count(reader: &mut (dyn AsyncRead + Send + Unpin + '_)) -> io::Result<usize> {
    let mut total = 0_usize;
    let mut buffer = vec![0_u8; 64 * 1024];
    loop {
        let read = reader.read(&mut buffer).await?;
        if read == 0 {
            return Ok(total);
        }
        total = total.saturating_add(read);
    }
}

#[cfg(feature = "minio")]
#[tokio::test(flavor = "current_thread")]
async fn minio_write_stream_closed_under_a_part_sends_one_put() {
    let (endpoint, seen) = canned(vec![empty_page(), answer("200 OK", "ETag: \"b\"\r\n", "")]);
    let bucket = MinioBucket::new(minio_config(endpoint));
    let mut writer = bucket
        .write_stream(Path::new(BIG), Condition::Absent)
        .await
        .unwrap();
    writer.write_all(b"abc").await.unwrap();
    writer.close().await.unwrap();
    assert_eq!(
        requests(&seen),
        ["GET if-match=-", "PUT if-match=- if-none-match=*"]
    );
}

#[cfg(feature = "minio")]
#[tokio::test(flavor = "current_thread")]
async fn minio_write_stream_of_exactly_a_part_sends_one_put() {
    let (endpoint, seen) = canned(vec![empty_page(), answer("200 OK", "ETag: \"b\"\r\n", "")]);
    let bucket = MinioBucket::new(minio_config(endpoint));
    let mut writer = bucket
        .write_stream(Path::new(BIG), Condition::None)
        .await
        .unwrap();
    writer.write_all(&vec![7_u8; PART]).await.unwrap();
    writer.close().await.unwrap();
    assert_eq!(requests(&seen), ["GET if-match=-", "PUT if-match=-"]);
}

#[cfg(feature = "minio")]
#[tokio::test(flavor = "current_thread")]
async fn minio_write_stream_of_20_mib_sends_three_parts_and_completes_under_absent() {
    let (endpoint, seen, targets) = canned_with_targets(vec![
        empty_page(),
        upload_started(),
        part_stored("p1"),
        part_stored("p2"),
        part_stored("p3"),
        upload_completed(),
    ]);
    let bucket = MinioBucket::new(minio_config(endpoint));
    let mut writer = bucket
        .write_stream(Path::new(BIG), Condition::Absent)
        .await
        .unwrap();
    let chunk = vec![7_u8; 1024 * 1024];
    for _ in 0_u8..20 {
        writer.write_all(&chunk).await.unwrap();
    }
    writer.close().await.unwrap();
    assert_eq!(
        requests(&seen),
        [
            "GET if-match=-",
            "POST if-match=-",
            "PUT if-match=- len=8388608",
            "PUT if-match=- len=8388608",
            "PUT if-match=- len=4194304",
            "POST if-match=- if-none-match=*"
        ]
    );
    let sent = requests(&targets);
    assert!(
        sent.iter().any(|target| target.contains("?uploads")),
        "{sent:?}"
    );
    for number in 1_u8..=3 {
        assert!(
            sent.iter()
                .any(|target| target.contains(&format!("partNumber={number}"))
                    && target.contains("uploadId=u1")),
            "{sent:?}"
        );
    }
    assert!(sent.last().unwrap().starts_with("POST"), "{sent:?}");
    assert!(sent.last().unwrap().contains("uploadId=u1"), "{sent:?}");
}

#[cfg(feature = "minio")]
#[tokio::test(flavor = "current_thread")]
async fn minio_write_stream_completes_under_the_tag_of_a_matches_condition() {
    let (endpoint, seen) = canned(vec![
        empty_page(),
        upload_started(),
        part_stored("p1"),
        part_stored("p2"),
        upload_completed(),
    ]);
    let bucket = MinioBucket::new(minio_config(endpoint));
    let mut writer = bucket
        .write_stream(
            Path::new(BIG),
            Condition::Matches(RevisionToken::new("\"old\"")),
        )
        .await
        .unwrap();
    writer.write_all(&vec![7_u8; PART + 1]).await.unwrap();
    writer.close().await.unwrap();
    assert_eq!(requests(&seen).last().unwrap(), "POST if-match=\"old\"");
}

#[cfg(feature = "minio")]
#[tokio::test(flavor = "current_thread")]
async fn minio_write_stream_with_a_part_refused_aborts_the_upload_before_the_error() {
    let (endpoint, seen, targets) = canned_with_targets(vec![
        empty_page(),
        upload_started(),
        answer("503 Service Unavailable", "", ""),
        answer("204 No Content", "", ""),
    ]);
    let bucket = MinioBucket::new(minio_config(endpoint));
    let mut writer = bucket
        .write_stream(Path::new(BIG), Condition::None)
        .await
        .unwrap();
    let error = writer.write_all(&vec![7_u8; PART + 1]).await.unwrap_err();
    assert_eq!(StorageError::from(error).kind(), ErrorKind::Unavailable);
    assert_eq!(
        requests(&seen),
        [
            "GET if-match=-",
            "POST if-match=-",
            "PUT if-match=- len=8388608",
            "DELETE if-match=-"
        ]
    );
    assert!(
        requests(&targets).last().unwrap().contains("uploadId=u1"),
        "{targets:?}"
    );
}

#[cfg(feature = "minio")]
#[tokio::test(flavor = "current_thread")]
async fn minio_write_stream_under_absent_on_an_existing_key_aborts_and_is_already_exists() {
    let (endpoint, seen) = canned(vec![
        empty_page(),
        upload_started(),
        part_stored("p1"),
        part_stored("p2"),
        answer("412 Precondition Failed", "", ""),
        answer("204 No Content", "", ""),
    ]);
    let bucket = MinioBucket::new(minio_config(endpoint));
    let mut writer = bucket
        .write_stream(Path::new(BIG), Condition::Absent)
        .await
        .unwrap();
    writer.write_all(&vec![7_u8; PART + 1]).await.unwrap();
    let error = writer.close().await.unwrap_err();
    assert_eq!(StorageError::from(error).kind(), ErrorKind::AlreadyExists);
    assert_eq!(requests(&seen).last().unwrap(), "DELETE if-match=-");
}

#[cfg(feature = "minio")]
#[tokio::test(flavor = "current_thread")]
async fn minio_write_stream_under_a_stale_tag_answered_412_is_precondition_failed() {
    let (endpoint, _seen) = canned(vec![
        empty_page(),
        answer("412 Precondition Failed", "", ""),
    ]);
    let bucket = MinioBucket::new(minio_config(endpoint));
    let mut writer = bucket
        .write_stream(
            Path::new(BIG),
            Condition::Matches(RevisionToken::new("\"old\"")),
        )
        .await
        .unwrap();
    writer.write_all(b"abc").await.unwrap();
    let error = writer.close().await.unwrap_err();
    assert_eq!(
        StorageError::from(error).kind(),
        ErrorKind::PreconditionFailed
    );
}

#[cfg(feature = "minio")]
#[tokio::test(flavor = "current_thread")]
async fn minio_write_stream_runs_the_parent_check_first() {
    let (endpoint, seen) = canned(vec![
        empty_page(),
        empty_page(),
        not_found_answer("NoSuchKey", "missing"),
        empty_page(),
    ]);
    let bucket = MinioBucket::new(minio_config(endpoint));
    let error = bucket
        .write_stream(Path::new("/missing/big.bin"), Condition::None)
        .await
        .err()
        .unwrap();
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert!(
        requests(&seen)
            .iter()
            .all(|line| line.starts_with("GET") || line.starts_with("HEAD")),
        "{seen:?}"
    );
}

#[cfg(feature = "minio")]
#[tokio::test(flavor = "current_thread")]
async fn minio_read_stream_of_20_mib_sends_one_head_and_three_ranges() {
    let (endpoint, seen) = canned(vec![
        sized(20 * 1024 * 1024),
        range_of(PART),
        range_of(PART),
        range_of(4 * 1024 * 1024),
    ]);
    let bucket = MinioBucket::new(minio_config(endpoint));
    let mut reader = bucket.read_stream(Path::new(BIG)).await.unwrap();
    assert_eq!(count(&mut *reader).await.unwrap(), 20 * 1024 * 1024);
    assert_eq!(
        requests(&seen),
        [
            "HEAD if-match=-",
            "GET if-match=\"t\" range=bytes=0-8388607",
            "GET if-match=\"t\" range=bytes=8388608-16777215",
            "GET if-match=\"t\" range=bytes=16777216-20971519"
        ]
    );
}

#[cfg(feature = "minio")]
#[tokio::test(flavor = "current_thread")]
async fn minio_read_stream_of_an_empty_object_sends_no_range() {
    let (endpoint, seen) = canned(vec![sized(0)]);
    let bucket = MinioBucket::new(minio_config(endpoint));
    let mut reader = bucket.read_stream(Path::new(BIG)).await.unwrap();
    assert_eq!(count(&mut *reader).await.unwrap(), 0);
    assert_eq!(requests(&seen), ["HEAD if-match=-"]);
}

#[cfg(feature = "minio")]
#[tokio::test(flavor = "current_thread")]
async fn minio_read_stream_of_an_object_replaced_mid_read_is_precondition_failed() {
    let (endpoint, _seen) = canned(vec![
        sized(20 * 1024 * 1024),
        range_of(PART),
        answer("412 Precondition Failed", "", ""),
    ]);
    let bucket = MinioBucket::new(minio_config(endpoint));
    let mut reader = bucket.read_stream(Path::new(BIG)).await.unwrap();
    let error = count(&mut *reader).await.unwrap_err();
    assert_eq!(
        StorageError::from(error).kind(),
        ErrorKind::PreconditionFailed
    );
}

#[cfg(feature = "minio")]
#[tokio::test(flavor = "current_thread")]
async fn minio_read_stream_of_a_missing_object_is_not_found() {
    let (endpoint, _seen) = canned(vec![
        answer("404 Not Found", "", ""),
        empty_page(),
        empty_page(),
    ]);
    let bucket = MinioBucket::new(minio_config(endpoint));
    let error = bucket.read_stream(Path::new(BIG)).await.err().unwrap();
    assert_eq!(error.kind(), ErrorKind::NotFound);
}

#[cfg(feature = "minio")]
#[tokio::test(flavor = "current_thread")]
async fn minio_append_stream_to_a_6_mib_object_copies_it_as_the_first_part() {
    let (endpoint, seen, targets) = canned_with_targets(vec![
        empty_page(),
        sized(6 * 1024 * 1024),
        upload_started(),
        part_copied("c1"),
        part_stored("p2"),
        upload_completed(),
    ]);
    let bucket = MinioBucket::new(minio_config(endpoint));
    let mut writer = bucket.append_stream(Path::new(BIG)).await.unwrap();
    writer.write_all(&vec![1_u8; 1024 * 1024]).await.unwrap();
    writer.close().await.unwrap();
    assert_eq!(
        requests(&seen),
        [
            "GET if-match=-",
            "HEAD if-match=-",
            "POST if-match=-",
            "PUT if-match=- copy-source=archie/big.bin copy-range=bytes=0-6291455",
            "PUT if-match=- len=1048576",
            "POST if-match=\"t\""
        ]
    );
    let sent = requests(&targets);
    for number in 1_u8..=2 {
        assert!(
            sent.iter()
                .any(|target| target.contains(&format!("partNumber={number}"))),
            "{sent:?}"
        );
    }
}

#[test]
fn copy_ranges_cover_every_byte_once_within_the_part_limits() {
    const GIB: u64 = 1024 * 1024 * 1024;
    for length in [
        COPY_MIN_BYTES,
        COPY_MAX_BYTES,
        COPY_MAX_BYTES + 1,
        12 * GIB,
        10_000 * COPY_MAX_BYTES,
    ] {
        let ranges = copy_ranges(length);
        assert!(ranges.len() <= 10_000, "{length}");
        let mut next = 0_u64;
        for (first, last) in ranges {
            assert_eq!(first, next, "{length}");
            let size = last.saturating_sub(first).saturating_add(1);
            assert!(
                (COPY_MIN_BYTES..=COPY_MAX_BYTES).contains(&size),
                "{length}"
            );
            next = last.saturating_add(1);
        }
        assert_eq!(next, length, "{length}");
    }
    assert_eq!(copy_ranges(COPY_MAX_BYTES), [(0, COPY_MAX_BYTES - 1)]);
    assert_eq!(
        copy_ranges(12 * GIB),
        [
            (0, 0xFFFF_FFFF),
            (0x1_0000_0000, 0x1_FFFF_FFFF),
            (0x2_0000_0000, 0x2_FFFF_FFFF)
        ]
    );
}

#[cfg(feature = "minio")]
#[tokio::test(flavor = "current_thread")]
async fn minio_append_stream_to_a_12_gib_object_copies_it_in_three_ranged_parts() {
    let (endpoint, seen) = canned(vec![
        empty_page(),
        sized(12 * 1024 * 1024 * 1024),
        upload_started(),
        part_copied("c1"),
        part_copied("c2"),
        part_copied("c3"),
        part_stored("p4"),
        upload_completed(),
    ]);
    let bucket = MinioBucket::new(minio_config(endpoint));
    let mut writer = bucket.append_stream(Path::new(BIG)).await.unwrap();
    writer.write_all(&vec![1_u8; 1024 * 1024]).await.unwrap();
    writer.close().await.unwrap();
    assert_eq!(
        requests(&seen),
        [
            "GET if-match=-",
            "HEAD if-match=-",
            "POST if-match=-",
            "PUT if-match=- copy-source=archie/big.bin copy-range=bytes=0-4294967295",
            "PUT if-match=- copy-source=archie/big.bin copy-range=bytes=4294967296-8589934591",
            "PUT if-match=- copy-source=archie/big.bin copy-range=bytes=8589934592-12884901887",
            "PUT if-match=- len=1048576",
            "POST if-match=\"t\""
        ]
    );
}

#[cfg(feature = "minio")]
#[tokio::test(flavor = "current_thread")]
async fn minio_append_stream_with_the_second_range_refused_aborts_and_is_precondition_failed() {
    let (endpoint, seen) = canned(vec![
        empty_page(),
        sized(12 * 1024 * 1024 * 1024),
        upload_started(),
        part_copied("c1"),
        answer("412 Precondition Failed", "", ""),
        answer("204 No Content", "", ""),
    ]);
    let bucket = MinioBucket::new(minio_config(endpoint));
    let error = bucket.append_stream(Path::new(BIG)).await.err().unwrap();
    assert_eq!(error.kind(), ErrorKind::PreconditionFailed);
    assert_eq!(requests(&seen).last().unwrap(), "DELETE if-match=-");
    assert_eq!(requests(&seen).len(), 6);
}

#[cfg(feature = "minio")]
#[tokio::test(flavor = "current_thread")]
async fn minio_append_stream_to_a_small_object_reads_it_and_puts_under_its_tag() {
    let (endpoint, seen) = canned(vec![
        empty_page(),
        sized(3),
        range_of(3),
        answer("200 OK", "ETag: \"n\"\r\n", ""),
    ]);
    let bucket = MinioBucket::new(minio_config(endpoint));
    let mut writer = bucket.append_stream(Path::new(BIG)).await.unwrap();
    writer.write_all(b"def").await.unwrap();
    writer.close().await.unwrap();
    assert_eq!(
        requests(&seen),
        [
            "GET if-match=-",
            "HEAD if-match=-",
            "GET if-match=\"t\" range=bytes=0-2",
            "PUT if-match=\"t\""
        ]
    );
}

#[cfg(feature = "minio")]
#[tokio::test(flavor = "current_thread")]
async fn minio_append_stream_to_a_missing_object_puts_under_absent() {
    let (endpoint, seen) = canned(vec![
        empty_page(),
        answer("404 Not Found", "", ""),
        empty_page(),
        empty_page(),
        answer("200 OK", "ETag: \"n\"\r\n", ""),
    ]);
    let bucket = MinioBucket::new(minio_config(endpoint));
    let mut writer = bucket.append_stream(Path::new(BIG)).await.unwrap();
    writer.write_all(b"def").await.unwrap();
    writer.close().await.unwrap();
    assert_eq!(
        requests(&seen).last().unwrap(),
        "PUT if-match=- if-none-match=*"
    );
}

#[cfg(feature = "minio")]
#[tokio::test(flavor = "current_thread")]
async fn minio_write_stream_dropped_without_close_logs_one_error_and_sends_nothing() {
    let (endpoint, seen) = canned(vec![empty_page()]);
    let bucket = MinioBucket::new(minio_config(endpoint));
    let mut writer = bucket
        .write_stream(Path::new(BIG), Condition::None)
        .await
        .unwrap();
    writer.write_all(b"abc").await.unwrap();
    let logged = Arc::new(AtomicUsize::new(0));
    with_default(CountsErrors(Arc::clone(&logged)), || drop(writer));
    assert_eq!(logged.load(Ordering::SeqCst), 1);
    assert_eq!(requests(&seen), ["GET if-match=-"]);
}

#[cfg(feature = "minio")]
#[tokio::test(flavor = "current_thread")]
async fn minio_write_stream_closed_logs_no_error() {
    let (endpoint, _seen) = canned(vec![empty_page(), answer("200 OK", "ETag: \"b\"\r\n", "")]);
    let bucket = MinioBucket::new(minio_config(endpoint));
    let mut writer = bucket
        .write_stream(Path::new(BIG), Condition::None)
        .await
        .unwrap();
    writer.close().await.unwrap();
    let logged = Arc::new(AtomicUsize::new(0));
    with_default(CountsErrors(Arc::clone(&logged)), || drop(writer));
    assert_eq!(logged.load(Ordering::SeqCst), 0);
}
