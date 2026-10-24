//! The `Bucket` trait's single declaration: invoked at `os_shim::bucket` with the
//! `async` feature, and in this private module without it, where it cannot be named.

#[cfg(not(feature = "async"))]
use super::{Condition, Object, ObjectError};

/// Declares the `Bucket` trait where it is invoked.
macro_rules! declare_bucket {
    () => {
        /// Whole objects in one bucket, each put and delete under an optional condition.
        ///
        /// Keys are opaque strings. Objects are read and written whole: no
        /// ranges, no streaming. Write code against a generic `B: Bucket`, use
        /// `MinioBucket` or `S3Bucket` in production and `MemoryBucket` in tests.
        /// The methods return `impl Future + Send`, so the trait cannot be a
        /// trait object. It can be named and implemented only with the `async`
        /// feature. Code that is not async wraps a bucket in `BlockingBucket`.
        pub trait Bucket: Send + Sync {
            /// Delete the object at `key` when `condition` holds.
            ///
            /// With `Condition::None` the object is removed, and a missing key
            /// is not an error. With `Condition::ETag` the object is removed
            /// only if its tag matches. With `Condition::Absent` nothing is
            /// deleted: it succeeds if the key is empty.
            ///
            /// # Errors
            ///
            /// `PreconditionFailed` when `condition` does not hold, the object
            /// unchanged. That includes `Condition::ETag` on a missing key.
            /// `Other` when the store cannot be reached or refuses.
            fn delete(
                &self,
                key: &str,
                condition: &Condition,
            ) -> impl Future<Output = Result<(), ObjectError>> + Send;

            /// The object at `key`.
            ///
            /// # Errors
            ///
            /// `NotFound` when no object is at `key`.
            /// `Other` when the store cannot be reached, the bucket is missing
            /// or the answer cannot be read.
            fn get(&self, key: &str) -> impl Future<Output = Result<Object, ObjectError>> + Send;

            /// Every key that starts with `prefix`, every page of the listing read.
            ///
            /// An empty prefix lists every key. The order is not promised;
            /// `MemoryBucket` sorts, a real store answers in its own order.
            ///
            /// # Errors
            ///
            /// `Other` when a page cannot be read.
            fn list(
                &self,
                prefix: &str,
            ) -> impl Future<Output = Result<Vec<String>, ObjectError>> + Send;

            /// Write `bytes` whole at `key` when `condition` holds, and give the new entity tag.
            ///
            /// Use `Condition::Absent` to create the object only if the key is
            /// free, and `Condition::ETag` to replace it only if it is
            /// unchanged since you read it.
            ///
            /// # Errors
            ///
            /// `PreconditionFailed` when `condition` does not hold, the object
            /// unchanged. That includes `Condition::ETag` on a missing key.
            /// `Other` when the store cannot be reached or refuses.
            fn put(
                &self,
                key: &str,
                bytes: &[u8],
                condition: &Condition,
            ) -> impl Future<Output = Result<String, ObjectError>> + Send;
        }
    };
}

#[cfg(not(feature = "async"))]
declare_bucket!();
