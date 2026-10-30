//! Tests for the in-memory bucket, the blocking wrapper, the bucket error and the `MinIO` and AWS S3 clients.

#![cfg(any(feature = "minio", feature = "s3"))]

use super::{
    BlockingBucket, Bucket, Condition, Head, MemoryBucket, Object, ObjectError, ObjectErrorKind,
};
#[cfg(feature = "minio")]
use super::{MinioBucket, MinioConfig};
#[cfg(feature = "s3")]
use super::{S3Bucket, S3Config};
#[cfg(any(feature = "minio", feature = "s3"))]
use alloc::sync::Arc;
#[cfg(feature = "minio")]
use aws_sdk_s3::{error::SdkError, operation::get_object::GetObjectError};
#[cfg(not(feature = "minio"))]
use core::error::Error as _;
#[cfg(feature = "minio")]
use core::{error::Error, iter};
use std::io;
#[cfg(any(feature = "minio", feature = "s3"))]
use std::{
    io::{BufRead as _, BufReader, Read as _, Write as _},
    net::{TcpListener, TcpStream},
    sync::Mutex,
    thread,
};

/// A bucket that implements only the required methods, so its `head` is the trait's default body.
struct RequiredOnly(MemoryBucket);

impl Bucket for RequiredOnly {
    fn delete(
        &self,
        key: &str,
        condition: &Condition,
    ) -> impl Future<Output = Result<(), ObjectError>> + Send {
        self.0.delete(key, condition)
    }

    fn get(&self, key: &str) -> impl Future<Output = Result<Object, ObjectError>> + Send {
        self.0.get(key)
    }

    fn list(&self, prefix: &str) -> impl Future<Output = Result<Vec<String>, ObjectError>> + Send {
        self.0.list(prefix)
    }

    fn put(
        &self,
        key: &str,
        bytes: &[u8],
        condition: &Condition,
    ) -> impl Future<Output = Result<String, ObjectError>> + Send {
        self.0.put(key, bytes, condition)
    }
}

fn blocking() -> BlockingBucket<MemoryBucket> {
    BlockingBucket::new(MemoryBucket::default()).unwrap()
}

#[test]
fn put_then_get_gives_the_bytes_and_the_put_tag() {
    let bucket = blocking();
    let tag = bucket.put("vault/a.md", b"x", &Condition::None).unwrap();
    let object = bucket.get("vault/a.md").unwrap();
    assert_eq!(object.bytes, b"x");
    assert_eq!(object.etag, tag);
}

#[tokio::test(flavor = "current_thread")]
async fn async_put_then_get_gives_the_bytes_and_the_put_tag() {
    let bucket = MemoryBucket::default();
    let tag = bucket
        .put("vault/a.md", b"x", &Condition::None)
        .await
        .unwrap();
    let object = bucket.get("vault/a.md").await.unwrap();
    assert_eq!(object.bytes, b"x");
    assert_eq!(object.etag, tag);
}

#[test]
fn get_of_a_missing_key_is_not_found() {
    let error = blocking().get("vault/none.md").unwrap_err();
    assert_eq!(error.kind(), ObjectErrorKind::NotFound);
}

#[tokio::test(flavor = "current_thread")]
async fn async_get_of_a_missing_key_is_not_found() {
    let error = MemoryBucket::default()
        .get("vault/none.md")
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ObjectErrorKind::NotFound);
}

#[test]
fn head_gives_the_size_and_the_put_tag() {
    let bucket = blocking();
    let tag = bucket
        .put("org/a.docx", b"PK\x03\x04", &Condition::None)
        .unwrap();
    assert_eq!(
        bucket.head("org/a.docx").unwrap(),
        Head { etag: tag, size: 4 }
    );
}

#[tokio::test(flavor = "current_thread")]
async fn async_head_gives_the_size_and_the_put_tag() {
    let bucket = MemoryBucket::default();
    let tag = bucket
        .put("org/a.docx", b"PK\x03\x04", &Condition::None)
        .await
        .unwrap();
    assert_eq!(
        bucket.head("org/a.docx").await.unwrap(),
        Head { etag: tag, size: 4 }
    );
}

#[test]
fn head_of_a_missing_key_is_not_found() {
    let error = blocking().head("org/b.docx").unwrap_err();
    assert_eq!(error.kind(), ObjectErrorKind::NotFound);
}

#[test]
fn default_head_gives_the_size_and_the_tag_get_gives() {
    let bucket = BlockingBucket::new(RequiredOnly(MemoryBucket::default())).unwrap();
    bucket
        .put("org/a.docx", b"12345", &Condition::None)
        .unwrap();
    let object = bucket.get("org/a.docx").unwrap();
    assert_eq!(
        bucket.head("org/a.docx").unwrap(),
        Head {
            etag: object.etag,
            size: 5
        }
    );
    let error = bucket.head("org/b.docx").unwrap_err();
    assert_eq!(error.kind(), ObjectErrorKind::NotFound);
}

#[test]
fn absent_over_an_object_fails_and_keeps_it() {
    let bucket = blocking();
    bucket.put("vault/a.md", b"x", &Condition::None).unwrap();
    let error = bucket
        .put("vault/a.md", b"y", &Condition::Absent)
        .unwrap_err();
    assert_eq!(error.kind(), ObjectErrorKind::PreconditionFailed);
    assert_eq!(bucket.get("vault/a.md").unwrap().bytes, b"x");
}

#[tokio::test(flavor = "current_thread")]
async fn async_absent_over_an_object_fails_and_keeps_it() {
    let bucket = MemoryBucket::default();
    bucket
        .put("vault/a.md", b"x", &Condition::None)
        .await
        .unwrap();
    let error = bucket
        .put("vault/a.md", b"y", &Condition::Absent)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ObjectErrorKind::PreconditionFailed);
    assert_eq!(bucket.get("vault/a.md").await.unwrap().bytes, b"x");
}

#[test]
fn matching_tag_replaces_the_object() {
    let bucket = blocking();
    let first = bucket.put("vault/a.md", b"x", &Condition::None).unwrap();
    let second = bucket
        .put("vault/a.md", b"y", &Condition::ETag(first.clone()))
        .unwrap();
    assert_ne!(second, first);
    assert_eq!(bucket.get("vault/a.md").unwrap().bytes, b"y");
}

#[tokio::test(flavor = "current_thread")]
async fn async_matching_tag_replaces_the_object() {
    let bucket = MemoryBucket::default();
    let first = bucket
        .put("vault/a.md", b"x", &Condition::None)
        .await
        .unwrap();
    let second = bucket
        .put("vault/a.md", b"y", &Condition::ETag(first.clone()))
        .await
        .unwrap();
    assert_ne!(second, first);
    assert_eq!(bucket.get("vault/a.md").await.unwrap().bytes, b"y");
}

#[test]
fn stale_tag_on_put_fails_and_keeps_the_object() {
    let bucket = blocking();
    let stale = bucket.put("vault/a.md", b"x", &Condition::None).unwrap();
    bucket
        .put("vault/a.md", b"y", &Condition::ETag(stale.clone()))
        .unwrap();
    let error = bucket
        .put("vault/a.md", b"z", &Condition::ETag(stale))
        .unwrap_err();
    assert_eq!(error.kind(), ObjectErrorKind::PreconditionFailed);
    assert_eq!(bucket.get("vault/a.md").unwrap().bytes, b"y");
}

#[test]
fn same_bytes_give_the_same_quoted_hex_tag() {
    let bucket = blocking();
    let first = bucket.put("vault/a.md", b"x", &Condition::None).unwrap();
    let second = bucket.put("vault/b.md", b"x", &Condition::None).unwrap();
    assert_eq!(first, second);
    let digits = first
        .strip_prefix('"')
        .and_then(|rest| rest.strip_suffix('"'))
        .unwrap();
    assert_eq!(digits.len(), 16);
    assert!(
        digits
            .chars()
            .all(|digit| digit.is_ascii_digit() || ('a'..='f').contains(&digit))
    );
}

#[test]
fn conditional_delete_needs_the_current_tag() {
    let bucket = blocking();
    let stale = bucket.put("vault/a.md", b"x", &Condition::None).unwrap();
    let current = bucket
        .put("vault/a.md", b"y", &Condition::ETag(stale.clone()))
        .unwrap();
    let error = bucket
        .delete("vault/a.md", &Condition::ETag(stale))
        .unwrap_err();
    assert_eq!(error.kind(), ObjectErrorKind::PreconditionFailed);
    bucket
        .delete("vault/a.md", &Condition::ETag(current))
        .unwrap();
    let missing = bucket.get("vault/a.md").unwrap_err();
    assert_eq!(missing.kind(), ObjectErrorKind::NotFound);
}

#[test]
fn delete_of_a_missing_key_fails_only_under_a_tag() {
    let bucket = blocking();
    bucket.delete("vault/none.md", &Condition::None).unwrap();
    let error = bucket
        .delete("vault/none.md", &Condition::ETag("\"0\"".to_owned()))
        .unwrap_err();
    assert_eq!(error.kind(), ObjectErrorKind::PreconditionFailed);
    bucket.delete("vault/none.md", &Condition::Absent).unwrap();
}

#[test]
fn absent_delete_over_an_object_fails_and_keeps_it() {
    let bucket = blocking();
    bucket.put("vault/a.md", b"x", &Condition::None).unwrap();
    let error = bucket.delete("vault/a.md", &Condition::Absent).unwrap_err();
    assert_eq!(error.kind(), ObjectErrorKind::PreconditionFailed);
    assert_eq!(bucket.get("vault/a.md").unwrap().bytes, b"x");
}

#[test]
fn listing_gives_every_key_under_the_prefix() {
    let bucket = blocking();
    for key in ["vault/a.md", "vault/b/c.md", "other.md"] {
        bucket.put(key, b"x", &Condition::None).unwrap();
    }
    assert_eq!(
        bucket.list("vault/").unwrap(),
        ["vault/a.md", "vault/b/c.md"]
    );
}

#[test]
fn listing_with_no_match_is_empty() {
    let bucket = blocking();
    bucket.put("vault/a.md", b"x", &Condition::None).unwrap();
    assert_eq!(bucket.list("none/").unwrap(), Vec::<String>::new());
}

#[test]
fn error_text_and_source() {
    let not_found = ObjectError::not_found();
    let failed = ObjectError::precondition_failed();
    let other = ObjectError::from(io::Error::other("boom"));
    assert_eq!(not_found.to_string(), "no object at the key");
    assert_eq!(failed.to_string(), "the condition does not hold");
    assert_eq!(other.to_string(), "boom");
    assert_eq!(other.kind(), ObjectErrorKind::Other);
    assert!(not_found.source().is_none());
    assert!(failed.source().is_none());
    assert_eq!(other.source().unwrap().to_string(), "boom");
}

#[tokio::test(flavor = "current_thread")]
#[should_panic(expected = "Cannot start a runtime from within a runtime")]
async fn blocking_inside_a_runtime_panics() {
    let bucket = blocking();
    drop(bucket.get("vault/a.md"));
}

/// Serve `answers` in order, one per request, and record each request's method and conditional headers.
#[cfg(any(feature = "minio", feature = "s3"))]
fn canned(answers: Vec<String>) -> (String, Arc<Mutex<Vec<String>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&seen);
    thread::spawn(move || serve(&listener, answers, &log));
    (endpoint, seen)
}

/// Answer each request on every connection with the next answer, until none is left.
#[cfg(any(feature = "minio", feature = "s3"))]
fn serve(listener: &TcpListener, answers: Vec<String>, log: &Mutex<Vec<String>>) {
    let mut pending = answers.into_iter();
    for incoming in listener.incoming() {
        let mut stream = incoming.unwrap();
        let mut reader = BufReader::new(stream.try_clone().unwrap());
        let mut request_line = String::new();
        while reader.read_line(&mut request_line).unwrap() > 0 {
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
    let error = minio(endpoint).get("task.md").unwrap_err();
    assert_eq!(error.kind(), ObjectErrorKind::NotFound);
    assert_eq!(requests(&seen), ["GET if-match=-"]);
}

#[cfg(feature = "minio")]
#[test]
fn minio_put_with_a_stale_tag_answered_412_is_precondition_failed() {
    let (endpoint, seen) = canned(vec![answer("412 Precondition Failed", "", "")]);
    let error = minio(endpoint)
        .put("task.md", b"x", &Condition::ETag("\"a\"".to_owned()))
        .unwrap_err();
    assert_eq!(error.kind(), ObjectErrorKind::PreconditionFailed);
    assert_eq!(requests(&seen), ["PUT if-match=\"a\""]);
}

#[cfg(feature = "minio")]
#[test]
fn minio_put_with_absent_sends_if_none_match_star() {
    let (endpoint, seen) = canned(vec![answer("412 Precondition Failed", "", "")]);
    let error = minio(endpoint)
        .put("task.md", b"x", &Condition::Absent)
        .unwrap_err();
    assert_eq!(error.kind(), ObjectErrorKind::PreconditionFailed);
    assert_eq!(requests(&seen), ["PUT if-match=- if-none-match=*"]);
}

#[cfg(feature = "minio")]
#[test]
fn minio_put_gives_the_tag_the_server_answered() {
    let (endpoint, _seen) = canned(vec![answer("200 OK", "ETag: \"b\"\r\n", "")]);
    let tag = minio(endpoint)
        .put("task.md", b"x", &Condition::None)
        .unwrap();
    assert_eq!(tag, "\"b\"");
}

#[cfg(feature = "minio")]
#[test]
fn minio_get_gives_the_bytes_and_the_tag() {
    let (endpoint, _seen) = canned(vec![answer("200 OK", "ETag: \"b\"\r\n", "x")]);
    let object = minio(endpoint).get("task.md").unwrap();
    assert_eq!(object.bytes, b"x");
    assert_eq!(object.etag, "\"b\"");
}

#[cfg(feature = "minio")]
#[test]
fn minio_delete_with_a_stale_tag_fails_without_deleting() {
    let (endpoint, seen) = canned(vec![answer("200 OK", "ETag: \"b\"\r\n", "")]);
    let error = minio(endpoint)
        .delete("task.md", &Condition::ETag("\"a\"".to_owned()))
        .unwrap_err();
    assert_eq!(error.kind(), ObjectErrorKind::PreconditionFailed);
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
        .delete("task.md", &Condition::ETag("\"a\"".to_owned()))
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
        .delete("task.md", &Condition::ETag("\"a\"".to_owned()))
        .unwrap_err();
    assert_eq!(error.kind(), ObjectErrorKind::PreconditionFailed);
    assert_eq!(requests(&seen), ["HEAD if-match=-", "HEAD if-match=-"]);
}

#[cfg(feature = "minio")]
#[test]
fn minio_delete_with_no_condition_sends_one_delete() {
    let (endpoint, seen) = canned(vec![answer("204 No Content", "", "")]);
    minio(endpoint).delete("task.md", &Condition::None).unwrap();
    assert_eq!(requests(&seen), ["DELETE if-match=-"]);
}

#[cfg(feature = "minio")]
#[test]
fn minio_get_answered_503_is_other_after_one_request() {
    let unavailable = answer("503 Service Unavailable", "", "");
    let (endpoint, seen) = canned(vec![unavailable.clone(), unavailable.clone(), unavailable]);
    let error = minio(endpoint).get("task.md").unwrap_err();
    assert_eq!(error.kind(), ObjectErrorKind::Other);
    assert_eq!(requests(&seen), ["GET if-match=-"]);
}

#[cfg(feature = "minio")]
#[test]
fn minio_list_reads_every_page() {
    let (endpoint, seen) = canned(vec![
        page("vault/a.md", Some("next")),
        page("vault/b.md", None),
    ]);
    let keys = minio(endpoint).list("vault/").unwrap();
    assert_eq!(keys, ["vault/a.md", "vault/b.md"]);
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
        .get("task.md")
        .unwrap_err();
    assert_eq!(error.kind(), ObjectErrorKind::Other);
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
    let error = minio(endpoint).get("task.md").unwrap_err();
    assert_eq!(error.kind(), ObjectErrorKind::Other);
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
    let error = minio(endpoint).get("task.md").unwrap_err();
    assert_eq!(error.kind(), ObjectErrorKind::Other);
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
    let error = minio(endpoint).get("task.md").unwrap_err();
    assert_eq!(error.kind(), ObjectErrorKind::NotFound);
}

#[cfg(feature = "minio")]
#[test]
fn minio_delete_absent_of_a_missing_key_in_an_existing_bucket_succeeds() {
    let (endpoint, seen) = canned(vec![
        answer("404 Not Found", "", ""),
        answer("200 OK", "", ""),
    ]);
    minio(endpoint)
        .delete("task.md", &Condition::Absent)
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
        .delete("task.md", &Condition::Absent)
        .unwrap_err();
    assert_eq!(error.kind(), ObjectErrorKind::Other);
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
        .delete("task.md", &Condition::ETag("\"a\"".to_owned()))
        .unwrap_err();
    assert_eq!(error.kind(), ObjectErrorKind::Other);
}

#[cfg(feature = "minio")]
#[test]
fn minio_put_with_a_tag_answered_404_no_such_bucket_is_other() {
    let (endpoint, seen) = canned(vec![no_such_bucket()]);
    let error = minio(endpoint)
        .put("task.md", b"x", &Condition::ETag("\"a\"".to_owned()))
        .unwrap_err();
    assert_eq!(error.kind(), ObjectErrorKind::Other);
    assert_eq!(requests(&seen), ["PUT if-match=\"a\""]);
}

#[cfg(feature = "minio")]
#[test]
fn minio_head_gives_the_size_and_the_tag_from_one_head() {
    let (endpoint, seen) = canned(vec![head_answer("ETag: \"a\"\r\nContent-Length: 5\r\n")]);
    assert_eq!(
        minio(endpoint).head("task.md").unwrap(),
        Head {
            etag: "\"a\"".to_owned(),
            size: 5
        }
    );
    assert_eq!(requests(&seen), ["HEAD if-match=-"]);
}

#[cfg(feature = "minio")]
#[test]
fn minio_head_of_a_missing_object_in_an_existing_bucket_is_not_found() {
    let (endpoint, seen) = canned(vec![
        answer("404 Not Found", "", ""),
        answer("200 OK", "", ""),
    ]);
    let error = minio(endpoint).head("task.md").unwrap_err();
    assert_eq!(error.kind(), ObjectErrorKind::NotFound);
    assert_eq!(requests(&seen), ["HEAD if-match=-", "HEAD if-match=-"]);
}

#[cfg(feature = "minio")]
#[test]
fn minio_head_in_a_missing_bucket_is_other_naming_the_bucket() {
    let (endpoint, seen) = canned(vec![
        answer("404 Not Found", "", ""),
        answer("404 Not Found", "", ""),
    ]);
    let error = minio(endpoint).head("task.md").unwrap_err();
    assert_eq!(error.kind(), ObjectErrorKind::Other);
    assert_eq!(error.to_string(), "404: the bucket archie does not exist");
    assert_eq!(requests(&seen), ["HEAD if-match=-", "HEAD if-match=-"]);
}

#[cfg(feature = "minio")]
#[test]
fn minio_head_without_an_etag_is_other() {
    let (endpoint, _seen) = canned(vec![head_answer("Content-Length: 5\r\n")]);
    let error = minio(endpoint).head("task.md").unwrap_err();
    assert_eq!(error.kind(), ObjectErrorKind::Other);
    assert_eq!(error.to_string(), "the answer carries no ETag header");
}

#[cfg(feature = "minio")]
#[test]
fn minio_head_without_a_content_length_is_other() {
    let (endpoint, _seen) = canned(vec![head_answer("ETag: \"a\"\r\n")]);
    let error = minio(endpoint).head("task.md").unwrap_err();
    assert_eq!(error.kind(), ObjectErrorKind::Other);
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
    assert_eq!(error.kind(), ObjectErrorKind::Other);
    assert_eq!(requests(&seen), ["HEAD if-match=-"]);
}

#[cfg(feature = "minio")]
#[test]
fn minio_ensure_with_the_create_refused_is_other_naming_the_answer() {
    let (endpoint, seen) = canned(vec![answer("404 Not Found", "", ""), access_denied()]);
    let bucket = minio(endpoint);
    let error = bucket.runtime.block_on(bucket.bucket.ensure()).unwrap_err();
    assert_eq!(error.kind(), ObjectErrorKind::Other);
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
    BlockingBucket::new(S3Bucket::with_sdk_config("archie".to_owned(), sdk)).unwrap()
}

#[cfg(feature = "s3")]
#[test]
fn s3_delete_with_the_current_tag_sends_one_delete_with_if_match() {
    let (endpoint, seen) = canned(vec![answer("204 No Content", "", "")]);
    s3(&endpoint)
        .delete("task.md", &Condition::ETag("\"a\"".to_owned()))
        .unwrap();
    assert_eq!(requests(&seen), ["DELETE if-match=\"a\""]);
}

#[cfg(feature = "s3")]
#[test]
fn s3_delete_with_a_stale_tag_answered_412_is_precondition_failed() {
    let (endpoint, seen) = canned(vec![answer("412 Precondition Failed", "", "")]);
    let error = s3(&endpoint)
        .delete("task.md", &Condition::ETag("\"a\"".to_owned()))
        .unwrap_err();
    assert_eq!(error.kind(), ObjectErrorKind::PreconditionFailed);
    assert_eq!(requests(&seen), ["DELETE if-match=\"a\""]);
}

#[cfg(feature = "s3")]
#[test]
fn s3_delete_with_a_tag_of_a_missing_object_is_precondition_failed() {
    let (endpoint, seen) = canned(vec![answer("404 Not Found", "", "")]);
    let error = s3(&endpoint)
        .delete("task.md", &Condition::ETag("\"a\"".to_owned()))
        .unwrap_err();
    assert_eq!(error.kind(), ObjectErrorKind::PreconditionFailed);
    assert_eq!(requests(&seen), ["DELETE if-match=\"a\""]);
}

#[cfg(feature = "s3")]
#[test]
fn s3_delete_with_a_tag_answered_404_no_such_bucket_is_other() {
    let (endpoint, seen) = canned(vec![no_such_bucket()]);
    let error = s3(&endpoint)
        .delete("task.md", &Condition::ETag("\"a\"".to_owned()))
        .unwrap_err();
    assert_eq!(error.kind(), ObjectErrorKind::Other);
    assert_eq!(requests(&seen), ["DELETE if-match=\"a\""]);
}

#[cfg(feature = "s3")]
#[test]
fn s3_delete_with_no_condition_sends_one_delete() {
    let (endpoint, seen) = canned(vec![answer("204 No Content", "", "")]);
    s3(&endpoint).delete("task.md", &Condition::None).unwrap();
    assert_eq!(requests(&seen), ["DELETE if-match=-"]);
}

#[cfg(feature = "s3")]
#[test]
fn s3_get_answered_404_is_not_found() {
    let (endpoint, _seen) = canned(vec![answer("404 Not Found", "", "")]);
    let error = s3(&endpoint).get("task.md").unwrap_err();
    assert_eq!(error.kind(), ObjectErrorKind::NotFound);
}

#[cfg(feature = "s3")]
#[test]
fn s3_put_with_absent_answered_412_is_precondition_failed() {
    let (endpoint, _seen) = canned(vec![answer("412 Precondition Failed", "", "")]);
    let error = s3(&endpoint)
        .put("task.md", b"x", &Condition::Absent)
        .unwrap_err();
    assert_eq!(error.kind(), ObjectErrorKind::PreconditionFailed);
}

#[cfg(feature = "s3")]
#[test]
fn s3_get_answered_503_is_other_after_one_request() {
    let unavailable = answer("503 Service Unavailable", "", "");
    let (endpoint, seen) = canned(vec![unavailable.clone(), unavailable.clone(), unavailable]);
    let error = s3(&endpoint).get("task.md").unwrap_err();
    assert_eq!(error.kind(), ObjectErrorKind::Other);
    assert_eq!(requests(&seen), ["GET if-match=-"]);
}

#[cfg(feature = "s3")]
#[test]
fn s3_head_gives_the_size_and_the_tag_from_one_head() {
    let (endpoint, seen) = canned(vec![head_answer("ETag: \"a\"\r\nContent-Length: 5\r\n")]);
    assert_eq!(
        s3(&endpoint).head("task.md").unwrap(),
        Head {
            etag: "\"a\"".to_owned(),
            size: 5
        }
    );
    assert_eq!(requests(&seen), ["HEAD if-match=-"]);
}

#[cfg(feature = "s3")]
#[test]
fn s3_head_of_a_missing_object_in_an_existing_bucket_is_not_found() {
    let (endpoint, seen) = canned(vec![
        answer("404 Not Found", "", ""),
        answer("200 OK", "", ""),
    ]);
    let error = s3(&endpoint).head("task.md").unwrap_err();
    assert_eq!(error.kind(), ObjectErrorKind::NotFound);
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
    assert_eq!(error.kind(), ObjectErrorKind::Other);
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
