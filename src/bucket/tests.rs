//! Tests for the blocking wrapper, the key rules and the `MinIO` and AWS S3 clients.

#![cfg(any(feature = "minio", feature = "s3"))]

use super::paths::key_of;
use super::{BlockingBucket, Condition, Depth, MemoryBucket, RevisionToken};
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
use core::time::Duration;
#[cfg(feature = "minio")]
use core::{error::Error, iter};
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

/// A list of lines the canned server records, shared with the test.
type Log = Arc<Mutex<Vec<String>>>;

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
/// conditional headers, and its canned ACL and location constraint when it carries them.
#[cfg(any(feature = "minio", feature = "s3"))]
fn read_request(request_line: &str, reader: &mut BufReader<TcpStream>) -> String {
    let mut length = 0_u64;
    let mut if_match = String::from("-");
    let mut if_none_match = None;
    let mut acl = None;
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
    parts.extend(if_none_match.map(|value| format!("if-none-match={value}")));
    parts.extend(acl.map(|value| format!("acl={value}")));
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
#[cfg(feature = "minio")]
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
    let (endpoint, seen) = canned(vec![answer("404 Not Found", "", "")]);
    let error = minio(endpoint).get(Path::new("/task.md")).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert_eq!(requests(&seen), ["GET if-match=-"]);
}

#[cfg(feature = "minio")]
#[test]
fn minio_put_with_a_stale_tag_answered_412_is_precondition_failed() {
    let (endpoint, seen) = canned(vec![answer("412 Precondition Failed", "", "")]);
    let error = minio(endpoint)
        .put(
            Path::new("/task.md"),
            b"x",
            &Condition::Matches(RevisionToken::new("\"a\"")),
        )
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::PreconditionFailed);
    assert_eq!(requests(&seen), ["PUT if-match=\"a\""]);
}

#[cfg(feature = "minio")]
#[test]
fn minio_put_with_absent_answered_412_is_already_exists_and_sends_if_none_match_star() {
    let (endpoint, seen) = canned(vec![answer("412 Precondition Failed", "", "")]);
    let error = minio(endpoint)
        .put(Path::new("/task.md"), b"x", &Condition::Absent)
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::AlreadyExists);
    assert_eq!(requests(&seen), ["PUT if-match=- if-none-match=*"]);
}

#[cfg(feature = "minio")]
#[test]
fn minio_put_gives_the_tag_the_server_answered() {
    let (endpoint, _seen) = canned(vec![answer("200 OK", "ETag: \"b\"\r\n", "")]);
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
    let (endpoint, seen) = canned(vec![answer("200 OK", "ETag: \"b\"\r\n", "")]);
    let error = minio(endpoint)
        .delete(
            Path::new("/task.md"),
            &Condition::Matches(RevisionToken::new("\"a\"")),
        )
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::PreconditionFailed);
    assert_eq!(requests(&seen), ["HEAD if-match=-"]);
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
fn minio_delete_with_no_condition_sends_one_delete() {
    let (endpoint, seen) = canned(vec![answer("204 No Content", "", "")]);
    minio(endpoint)
        .delete(Path::new("/task.md"), &Condition::None)
        .unwrap();
    assert_eq!(requests(&seen), ["DELETE if-match=-"]);
}

#[cfg(feature = "minio")]
#[test]
fn minio_get_answered_503_is_other_after_one_request() {
    let unavailable = answer("503 Service Unavailable", "", "");
    let (endpoint, seen) = canned(vec![unavailable.clone(), unavailable.clone(), unavailable]);
    let error = minio(endpoint).get(Path::new("/task.md")).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Other);
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
fn minio_config_debug_keeps_the_secret_key_out() {
    let config = minio_config("http://127.0.0.1:9000".to_owned());
    let shown = format!("{config:?}");
    assert!(shown.contains("<redacted>"));
    assert!(!shown.contains("os-shim-secret"));
}

#[cfg(feature = "minio")]
#[test]
fn minio_get_with_nothing_listening_is_other_naming_the_dispatch_failure_and_its_cause() {
    let error = minio("http://127.0.0.1:9".to_owned())
        .get(Path::new("/task.md"))
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Other);
    let message = error.to_string();
    assert!(message.starts_with("dispatch failure: "));
    assert!(message.contains("Connection refused"));
}

#[cfg(feature = "minio")]
#[test]
fn minio_get_answered_403_names_the_status_the_code_and_the_message() {
    let body = "<?xml version=\"1.0\" encoding=\"UTF-8\"?><Error><Code>SignatureDoesNotMatch</Code><Message>The request signature we calculated does not match the signature you provided.</Message></Error>";
    let (endpoint, _seen) = canned(vec![answer(
        "403 Forbidden",
        "Content-Type: application/xml\r\n",
        body,
    )]);
    let error = minio(endpoint).get(Path::new("/task.md")).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Other);
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
    let (endpoint, _seen) = canned(vec![not_found_answer(
        "NoSuchKey",
        "The specified key does not exist.",
    )]);
    let error = minio(endpoint).get(Path::new("/task.md")).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
}

#[cfg(feature = "minio")]
#[test]
fn minio_delete_absent_of_a_missing_key_in_an_existing_bucket_succeeds() {
    let (endpoint, seen) = canned(vec![
        answer("404 Not Found", "", ""),
        answer("200 OK", "", ""),
    ]);
    minio(endpoint)
        .delete(Path::new("/task.md"), &Condition::Absent)
        .unwrap();
    assert_eq!(requests(&seen), ["HEAD if-match=-", "HEAD if-match=-"]);
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
    let (endpoint, seen) = canned(vec![no_such_bucket()]);
    let error = minio(endpoint)
        .put(
            Path::new("/task.md"),
            b"x",
            &Condition::Matches(RevisionToken::new("\"a\"")),
        )
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(requests(&seen), ["PUT if-match=\"a\""]);
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
    ]);
    let error = minio(endpoint).head(Path::new("/task.md")).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert_eq!(requests(&seen), ["HEAD if-match=-", "HEAD if-match=-"]);
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
fn minio_ensure_with_the_head_refused_is_other_and_creates_nothing() {
    let (endpoint, seen) = canned(vec![access_denied()]);
    let bucket = minio(endpoint);
    let error = bucket.runtime.block_on(bucket.bucket.ensure()).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(requests(&seen), ["HEAD if-match=-"]);
}

#[cfg(feature = "minio")]
#[test]
fn minio_ensure_with_the_create_refused_is_other_naming_the_answer() {
    let (endpoint, seen) = canned(vec![answer("404 Not Found", "", ""), access_denied()]);
    let bucket = minio(endpoint);
    let error = bucket.runtime.block_on(bucket.bucket.ensure()).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Other);
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
    let (endpoint, seen) = canned(vec![answer("412 Precondition Failed", "", "")]);
    let error = s3(&endpoint)
        .delete(
            Path::new("/task.md"),
            &Condition::Matches(RevisionToken::new("\"a\"")),
        )
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::PreconditionFailed);
    assert_eq!(requests(&seen), ["DELETE if-match=\"a\""]);
}

#[cfg(feature = "s3")]
#[test]
fn s3_delete_with_a_tag_of_a_missing_object_is_precondition_failed() {
    let (endpoint, seen) = canned(vec![answer("404 Not Found", "", "")]);
    let error = s3(&endpoint)
        .delete(
            Path::new("/task.md"),
            &Condition::Matches(RevisionToken::new("\"a\"")),
        )
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::PreconditionFailed);
    assert_eq!(requests(&seen), ["DELETE if-match=\"a\""]);
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
fn s3_delete_with_no_condition_sends_one_delete() {
    let (endpoint, seen) = canned(vec![answer("204 No Content", "", "")]);
    s3(&endpoint)
        .delete(Path::new("/task.md"), &Condition::None)
        .unwrap();
    assert_eq!(requests(&seen), ["DELETE if-match=-"]);
}

#[cfg(feature = "s3")]
#[test]
fn s3_get_answered_404_is_not_found() {
    let (endpoint, _seen) = canned(vec![answer("404 Not Found", "", "")]);
    let error = s3(&endpoint).get(Path::new("/task.md")).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
}

#[cfg(feature = "s3")]
#[test]
fn s3_put_with_absent_answered_412_is_already_exists() {
    let (endpoint, _seen) = canned(vec![answer("412 Precondition Failed", "", "")]);
    let error = s3(&endpoint)
        .put(Path::new("/task.md"), b"x", &Condition::Absent)
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::AlreadyExists);
}

#[cfg(feature = "s3")]
#[test]
fn s3_get_answered_503_is_other_after_one_request() {
    let unavailable = answer("503 Service Unavailable", "", "");
    let (endpoint, seen) = canned(vec![unavailable.clone(), unavailable.clone(), unavailable]);
    let error = s3(&endpoint).get(Path::new("/task.md")).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Other);
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
    ]);
    let error = s3(&endpoint).head(Path::new("/task.md")).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert_eq!(requests(&seen), ["HEAD if-match=-", "HEAD if-match=-"]);
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
fn s3_ensure_with_the_encryption_refused_is_other_naming_the_answer() {
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
    assert_eq!(error.kind(), ErrorKind::Other);
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
    let (endpoint, _seen, targets) =
        canned_with_targets(vec![answer("200 OK", "ETag: \"b\"\r\n", "")]);
    let config = minio_config(endpoint).with_prefix("tenant-7/");
    assert_eq!(config.prefix.as_deref(), Some("tenant-7"));
    BlockingBucket::new(MinioBucket::new(config))
        .unwrap()
        .put(Path::new("/notes/a.md"), b"x", &Condition::None)
        .unwrap();
    assert_eq!(
        *targets.lock().unwrap(),
        ["PUT /archie/tenant-7/notes/a.md?x-id=PutObject"]
    );
}

#[cfg(feature = "minio")]
#[test]
fn minio_delete_absent_over_an_object_is_already_exists() {
    let (endpoint, seen) = canned(vec![answer("200 OK", "ETag: \"a\"\r\n", "")]);
    let error = minio(endpoint)
        .delete(Path::new("/task.md"), &Condition::Absent)
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::AlreadyExists);
    assert_eq!(requests(&seen), ["HEAD if-match=-"]);
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
    let (endpoint, _seen, targets) =
        canned_with_targets(vec![answer("200 OK", "ETag: \"b\"\r\n", "")]);
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
        ["PUT /archie/tenant-7/notes/a.md?x-id=PutObject"]
    );
}

#[cfg(feature = "s3")]
#[test]
fn s3_delete_absent_over_an_object_is_already_exists() {
    let (endpoint, _seen) = canned(vec![answer("200 OK", "ETag: \"a\"\r\n", "")]);
    let error = s3(&endpoint)
        .delete(Path::new("/task.md"), &Condition::Absent)
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::AlreadyExists);
}
