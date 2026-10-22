//! Parity of `MemoryBucket` with a real server: each sequence runs against both
//! and the two transcripts must be equal. `minio_tests` runs them against a
//! MinIO-compatible server, `s3_tests` against an AWS bucket.

use super::{
    BlockingBucket, Bucket, Condition, MemoryBucket, Object, ObjectError, ObjectErrorKind,
};
#[cfg(minio_tests)]
use super::{MinioBucket, MinioConfig};
#[cfg(s3_tests)]
use super::{S3Bucket, S3Config};
#[cfg(minio_tests)]
use aws_sdk_s3::operation::create_bucket::CreateBucketError;
use aws_sdk_s3::{
    Client, Config,
    config::{BehaviorVersion, Credentials, Region, retry::RetryConfig},
    error::SdkError,
};
use std::env;
#[cfg(s3_tests)]
use std::process;
#[cfg(minio_tests)]
use std::sync::OnceLock;
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
/// The configuration of a server whose bucket exists, created once per test binary.
fn ready_config() -> MinioConfig {
    static CREATED: OnceLock<()> = OnceLock::new();
    let config = config();
    CREATED.get_or_init(|| {
        let answer = runtime().block_on(
            client(&config)
                .create_bucket()
                .bucket(&config.bucket)
                .send(),
        );
        if let Err(error) = answer {
            assert!(
                error
                    .as_service_error()
                    .is_some_and(CreateBucketError::is_bucket_already_owned_by_you),
                "the bucket cannot be created: {error:?}"
            );
        }
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

/// What an answer looks like with its tag hidden: `tag`, the shown value, `NotFound`, `PreconditionFailed` or `Other`.
fn shape<T>(answer: &Result<T, ObjectError>, show: impl Fn(&T) -> String) -> String {
    match answer.as_ref() {
        Ok(value) => show(value),
        Err(error) => match error.kind() {
            ObjectErrorKind::NotFound => "NotFound".to_owned(),
            ObjectErrorKind::PreconditionFailed => "PreconditionFailed".to_owned(),
            ObjectErrorKind::Other => "Other".to_owned(),
        },
    }
}

/// The line for a put's answer.
fn put_line(answer: &Result<String, ObjectError>) -> String {
    shape(answer, |_| "tag".to_owned())
}

/// The line for a get's answer.
fn get_line(answer: &Result<Object, ObjectError>) -> String {
    shape(answer, |object| format!("bytes {:?}", object.bytes))
}

/// The line for a delete's answer.
fn delete_line(answer: &Result<(), ObjectError>) -> String {
    shape(answer, |&()| "deleted".to_owned())
}

/// Print the server's transcript and assert it equals the memory one.
fn assert_agree(name: &str, memory_transcript: &[String], server_transcript: &[String]) {
    eprintln!("{name} on the server:\n{}", server_transcript.join("\n"));
    assert_eq!(memory_transcript, server_transcript);
}

/// A put, a get and the tag the put returned.
fn put_and_get<B>(bucket: &BlockingBucket<B>, prefix: &str) -> Vec<String>
where
    B: Bucket,
{
    let key = format!("{prefix}note.md");
    let mut transcript = Vec::new();
    let put = bucket.put(&key, b"# Note\n", &Condition::None);
    transcript.push(put_line(&put));
    let got = bucket.get(&key);
    transcript.push(get_line(&got));
    transcript.push(format!(
        "get tag equals put tag: {}",
        got.unwrap().etag == put.unwrap()
    ));
    transcript
}

/// Puts and deletes under `Absent` and under current and stale tags.
fn conditional_writes<B>(bucket: &BlockingBucket<B>, prefix: &str) -> Vec<String>
where
    B: Bucket,
{
    let key = format!("{prefix}task.md");
    let mut transcript = Vec::new();
    let created = bucket.put(&key, b"# Task\n", &Condition::Absent);
    transcript.push(put_line(&created));
    transcript.push(put_line(&bucket.put(
        &key,
        b"# Other\n",
        &Condition::Absent,
    )));
    let first = created.unwrap();
    let replaced = bucket.put(&key, b"# Done\n", &Condition::ETag(first.clone()));
    transcript.push(put_line(&replaced));
    transcript.push(put_line(&bucket.put(
        &key,
        b"# Late\n",
        &Condition::ETag(first.clone()),
    )));
    transcript.push(get_line(&bucket.get(&key)));
    transcript.push(delete_line(&bucket.delete(&key, &Condition::ETag(first))));
    transcript.push(delete_line(
        &bucket.delete(&key, &Condition::ETag(replaced.unwrap())),
    ));
    transcript.push(get_line(&bucket.get(&key)));
    transcript
}

/// Deletes of a key that holds nothing, under each condition.
fn missing_key_deletes<B>(bucket: &BlockingBucket<B>, prefix: &str) -> Vec<String>
where
    B: Bucket,
{
    let key = format!("{prefix}none.md");
    vec![
        delete_line(&bucket.delete(&key, &Condition::None)),
        delete_line(&bucket.delete(&key, &Condition::ETag("\"0123456789abcdef\"".to_owned()))),
        delete_line(&bucket.delete(&key, &Condition::Absent)),
    ]
}

/// A put under a stale tag at a key that holds nothing, and a get of that key.
#[cfg(minio_tests)]
fn missing_key_tagged_put<B>(bucket: &BlockingBucket<B>, prefix: &str) -> Vec<String>
where
    B: Bucket,
{
    let key = format!("{prefix}none.md");
    vec![
        put_line(&bucket.put(&key, b"# Task\n", &Condition::ETag("\"stale\"".to_owned()))),
        get_line(&bucket.get(&key)),
    ]
}

/// Tags of the same bytes at two keys, and of different bytes.
fn same_bytes_tags<B>(bucket: &BlockingBucket<B>, prefix: &str) -> Vec<String>
where
    B: Bucket,
{
    let one = bucket.put(&format!("{prefix}one.md"), b"same", &Condition::None);
    let two = bucket.put(&format!("{prefix}two.md"), b"same", &Condition::None);
    let other = bucket.put(&format!("{prefix}other.md"), b"different", &Condition::None);
    let one_tag = one.unwrap();
    vec![
        format!("same bytes, same tag: {}", one_tag == two.unwrap()),
        format!("other bytes, same tag: {}", one_tag == other.unwrap()),
    ]
}

/// A listing of 1,001 keys, more than one page holds.
fn long_listing<B>(bucket: &BlockingBucket<B>, prefix: &str) -> Vec<String>
where
    B: Bucket,
{
    let expected: Vec<String> = (0_u32..=1000_u32)
        .map(|index| format!("{prefix}{index:04}"))
        .collect();
    for key in &expected {
        bucket.put(key, b"x", &Condition::None).unwrap();
    }
    let mut listed = bucket.list(prefix).unwrap();
    let count = listed.len();
    listed.sort();
    vec![
        format!("keys {count}"),
        format!("every key once: {}", listed == expected),
    ]
}

/// A listing under a prefix no key has.
fn empty_listing<B>(bucket: &BlockingBucket<B>, prefix: &str) -> Vec<String>
where
    B: Bucket,
{
    let listed = bucket.list(&format!("{prefix}nothing-here/"));
    vec![format!("keys {:?}", listed.unwrap())]
}

/// The status an SDK answer carried, or `accepted` when it succeeded.
fn status<T, E>(answer: &Result<T, SdkError<E>>) -> String {
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
fn put_and_get_agree_with_the_server() {
    assert_agree(
        "put and get",
        &put_and_get(&memory(), "a/"),
        &put_and_get(&server(), "put-and-get/"),
    );
}

#[cfg(minio_tests)]
#[test]
fn conditional_writes_agree_with_the_server() {
    assert_agree(
        "conditional writes",
        &conditional_writes(&memory(), "a/"),
        &conditional_writes(&server(), "conditional-writes/"),
    );
}

#[cfg(minio_tests)]
#[test]
fn missing_key_deletes_agree_with_the_server() {
    assert_agree(
        "missing key deletes",
        &missing_key_deletes(&memory(), "a/"),
        &missing_key_deletes(&server(), "missing-key-deletes/"),
    );
}

#[cfg(minio_tests)]
#[test]
fn missing_key_tagged_put_agrees_with_the_server() {
    assert_agree(
        "missing key tagged put",
        &missing_key_tagged_put(&memory(), "a/"),
        &missing_key_tagged_put(&server(), "missing-key-tagged-put/"),
    );
}

#[cfg(minio_tests)]
#[test]
fn same_bytes_tags_agree_with_the_server() {
    assert_agree(
        "same bytes tags",
        &same_bytes_tags(&memory(), "a/"),
        &same_bytes_tags(&server(), "same-bytes-tags/"),
    );
}

#[cfg(minio_tests)]
#[test]
fn long_listing_agrees_with_the_server() {
    assert_agree(
        "long listing",
        &long_listing(&memory(), "a/"),
        &long_listing(&server(), "long-listing/"),
    );
}

#[cfg(minio_tests)]
#[test]
fn empty_listing_agrees_with_the_server() {
    assert_agree(
        "empty listing",
        &empty_listing(&memory(), "a/"),
        &empty_listing(&server(), "empty-listing/"),
    );
}

#[cfg(minio_tests)]
#[test]
fn observe_a_raw_delete_with_a_wrong_if_match() {
    let config = ready_config();
    let bucket = server();
    let key = "observe-raw-delete/note.md";
    bucket.put(key, b"kept", &Condition::None).unwrap();
    let answer = runtime().block_on(
        client(&config)
            .delete_object()
            .bucket(&config.bucket)
            .key(key)
            .if_match("\"0123456789abcdef0123456789abcdef\"")
            .send(),
    );
    eprintln!("raw delete with a wrong If-Match: {}", status(&answer));
    eprintln!("object afterwards: {}", get_line(&bucket.get(key)));
}

#[cfg(minio_tests)]
#[test]
fn observe_a_put_with_the_tag_stripped_of_its_quotes() {
    let bucket = server();
    let key = "observe-unquoted-tag/note.md";
    let tag = bucket.put(key, b"first", &Condition::None).unwrap();
    let answer = bucket.put(
        key,
        b"second",
        &Condition::ETag(tag.trim_matches('"').to_owned()),
    );
    eprintln!("put with an unquoted tag: {}", put_line(&answer));
    eprintln!("object afterwards: {}", get_line(&bucket.get(key)));
}

/// Not a parity sequence, since `MemoryBucket` has no missing-bucket state: every call to
/// a bucket the server does not have is `Other`, and names the server's answer.
#[cfg(minio_tests)]
#[test]
fn a_missing_bucket_is_other_on_the_server() {
    let mut config = config();
    "os-shim-no-such-bucket".clone_into(&mut config.bucket);
    let bucket = BlockingBucket::new(MinioBucket::new(config)).unwrap();
    let key = "missing-bucket/note.md";
    let got = bucket.get(key);
    let transcript = vec![
        get_line(&got),
        put_line(&bucket.put(key, b"# Note\n", &Condition::None)),
        put_line(&bucket.put(key, b"# Note\n", &Condition::ETag("\"stale\"".to_owned()))),
        shape(&bucket.list("missing-bucket/"), |keys| {
            format!("keys {keys:?}")
        }),
    ];
    eprintln!("missing bucket on the server:\n{}", transcript.join("\n"));
    assert_eq!(transcript, ["Other", "Other", "Other", "Other"]);
    let message = got.unwrap_err().to_string();
    assert!(message.starts_with("404 NoSuchBucket"), "{message}");
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
    let credentials = Credentials::new(
        aws_variable("AWS_ACCESS_KEY_ID"),
        aws_variable("AWS_SECRET_ACCESS_KEY"),
        None,
        None,
        "os-shim",
    );
    let sdk = Config::builder()
        .behavior_version(BehaviorVersion::latest())
        .region(Region::new(aws_variable("OS_SHIM_S3_REGION")))
        .credentials_provider(credentials)
        .retry_config(RetryConfig::disabled())
        .build();
    Client::from_conf(sdk)
}

/// A key prefix no other run or sequence shares.
#[cfg(s3_tests)]
fn run_prefix(name: &str) -> String {
    format!("os-shim-tests/{}-{name}/", process::id())
}

/// Delete every key under `prefix`.
#[cfg(s3_tests)]
fn remove_prefix(bucket: &BlockingBucket<S3Bucket>, prefix: &str) {
    for key in bucket.list(prefix).unwrap() {
        bucket.delete(&key, &Condition::None).unwrap();
    }
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
fn put_and_get_agree_with_aws() {
    assert_agree(
        "put and get",
        &put_and_get(&memory(), "a/"),
        &aws_transcript("put-and-get", put_and_get),
    );
}

#[cfg(s3_tests)]
#[test]
fn conditional_writes_agree_with_aws() {
    assert_agree(
        "conditional writes",
        &conditional_writes(&memory(), "a/"),
        &aws_transcript("conditional-writes", conditional_writes),
    );
}

#[cfg(s3_tests)]
#[test]
fn missing_key_deletes_agree_with_aws() {
    assert_agree(
        "missing key deletes",
        &missing_key_deletes(&memory(), "a/"),
        &aws_transcript("missing-key-deletes", missing_key_deletes),
    );
}

#[cfg(s3_tests)]
#[test]
fn same_bytes_tags_agree_with_aws() {
    assert_agree(
        "same bytes tags",
        &same_bytes_tags(&memory(), "a/"),
        &aws_transcript("same-bytes-tags", same_bytes_tags),
    );
}

#[cfg(s3_tests)]
#[test]
fn long_listing_agrees_with_aws() {
    assert_agree(
        "long listing",
        &long_listing(&memory(), "a/"),
        &aws_transcript("long-listing", long_listing),
    );
}

#[cfg(s3_tests)]
#[test]
fn empty_listing_agrees_with_aws() {
    assert_agree(
        "empty listing",
        &empty_listing(&memory(), "a/"),
        &aws_transcript("empty-listing", empty_listing),
    );
}

#[cfg(s3_tests)]
#[test]
fn observe_a_raw_delete_with_a_wrong_if_match_on_aws() {
    let bucket = aws();
    let prefix = run_prefix("observe-raw-delete");
    let key = format!("{prefix}note.md");
    bucket.put(&key, b"kept", &Condition::None).unwrap();
    let answer = runtime().block_on(
        aws_client()
            .delete_object()
            .bucket(aws_variable("OS_SHIM_S3_BUCKET"))
            .key(&key)
            .if_match("\"0123456789abcdef0123456789abcdef\"")
            .send(),
    );
    eprintln!("raw delete with a wrong If-Match: {}", status(&answer));
    eprintln!("object afterwards: {}", get_line(&bucket.get(&key)));
    remove_prefix(&bucket, &prefix);
}

#[cfg(s3_tests)]
#[test]
fn observe_a_put_with_the_tag_stripped_of_its_quotes_on_aws() {
    let bucket = aws();
    let prefix = run_prefix("observe-unquoted-tag");
    let key = format!("{prefix}note.md");
    let tag = bucket.put(&key, b"first", &Condition::None).unwrap();
    let answer = bucket.put(
        &key,
        b"second",
        &Condition::ETag(tag.trim_matches('"').to_owned()),
    );
    eprintln!("put with an unquoted tag: {}", put_line(&answer));
    eprintln!("object afterwards: {}", get_line(&bucket.get(&key)));
    remove_prefix(&bucket, &prefix);
}
