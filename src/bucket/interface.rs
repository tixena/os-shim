//! The `Bucket` trait's single declaration: invoked at `os_shim::bucket` with the
//! `async` feature, and in this private module without it, where it cannot be named.

#[cfg(not(feature = "async"))]
use super::{Condition, Object, ObjectError};

/// Declares the `Bucket` trait where it is invoked.
macro_rules! declare_bucket {
    () => {
        /// Whole objects in one bucket, each put and delete under an optional condition.
        pub trait Bucket: Send + Sync {
            /// Delete the object at `key` when `condition` holds.
            ///
            /// # Errors
            ///
            /// `PreconditionFailed` when `condition` does not hold, the object unchanged.
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
            fn get(&self, key: &str) -> impl Future<Output = Result<Object, ObjectError>> + Send;

            /// Every key that starts with `prefix`, every page of the listing read.
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
            /// # Errors
            ///
            /// `PreconditionFailed` when `condition` does not hold, the object unchanged.
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
