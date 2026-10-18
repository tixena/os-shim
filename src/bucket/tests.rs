//! Tests for the in-memory bucket, the blocking wrapper and the bucket error.

use super::{BlockingBucket, Bucket as _, Condition, MemoryBucket, ObjectError, ObjectErrorKind};
use core::error::Error as _;
use std::io;

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
