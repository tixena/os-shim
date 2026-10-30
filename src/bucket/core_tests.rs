//! Tests that run in every build, with no client feature.

use super::{Bucket, Condition, Head, MemoryBucket, Object, ObjectError, ObjectErrorKind};
use core::future::ready;

struct Empty;

impl Bucket for Empty {
    fn delete(
        &self,
        _key: &str,
        _condition: &Condition,
    ) -> impl Future<Output = Result<(), ObjectError>> + Send {
        ready(Ok(()))
    }

    fn get(&self, _key: &str) -> impl Future<Output = Result<Object, ObjectError>> + Send {
        ready(Err(ObjectError::not_found()))
    }

    fn list(&self, _prefix: &str) -> impl Future<Output = Result<Vec<String>, ObjectError>> + Send {
        ready(Ok(Vec::new()))
    }

    fn put(
        &self,
        _key: &str,
        _bytes: &[u8],
        _condition: &Condition,
    ) -> impl Future<Output = Result<String, ObjectError>> + Send {
        ready(Ok(String::new()))
    }
}

#[tokio::test]
async fn bucket_is_implementable_for_a_local_type() {
    let head: Result<Head, ObjectError> = Empty.head("k").await;
    assert_eq!(
        head.err().map(|error| error.kind()),
        Some(ObjectErrorKind::NotFound)
    );
    assert_eq!(Empty.list("").await.ok(), Some(Vec::new()));
    assert_eq!(
        MemoryBucket::default().list("").await.ok(),
        Some(Vec::new())
    );
}
