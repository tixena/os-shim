//! The `Bucket` trait's single declaration. A macro places it at `os_shim::bucket` without a `pub use`.

/// Declares the `Bucket` trait where it is invoked.
macro_rules! declare_bucket {
    () => {
        /// Files and folders in one bucket, each write and delete under an optional condition.
        ///
        /// Paths are absolute: the `System` resolves a relative path before it
        /// reaches a bucket. A file's bytes are read and written whole, or
        /// through the stream methods. Write code against a generic
        /// `B: Bucket`, use `MinioBucket` or `S3Bucket` in production and
        /// `MemoryBucket` in tests. Hold an `Arc<dyn Bucket>` where the backend is
        /// chosen at run time. Code that is not async wraps a bucket in
        /// `BlockingBucket`. Each method returns a boxed future, which it
        /// allocates once per call.
        ///
        /// A backend overrides the methods that have a body when its storage
        /// has a native form of them.
        ///
        /// ```
        /// use os_shim::bucket::{Bucket, Condition, MemoryBucket};
        /// use std::path::Path;
        /// use std::sync::Arc;
        ///
        /// struct Service {
        ///     store: Arc<dyn Bucket>,
        /// }
        ///
        /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
        /// let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        /// runtime.block_on(async {
        ///     let service = Service {
        ///         store: Arc::new(MemoryBucket::default()),
        ///     };
        ///     let licence = Path::new("/licencia.pdf");
        ///     service.store.put(licence, b"%PDF-1.7", &Condition::Absent).await?;
        ///     assert_eq!(service.store.get(licence).await?.0, b"%PDF-1.7");
        ///     Ok(())
        /// })
        /// # }
        /// ```
        pub trait Bucket: Send + Sync {
            /// Add `more` at the end of the file, creating it when missing, and give the new token.
            ///
            /// Default: read, then write the whole file back under its token.
            ///
            /// # Errors
            ///
            /// `PreconditionFailed` when the file changed between the read and the write.
            /// The errors of `get` and `put` otherwise.
            #[inline]
            fn append<'call>(
                &'call self,
                path: &'call Path,
                more: &'call [u8],
            ) -> BoxFuture<'call, Result<RevisionToken>> {
                Box::pin(append_by_rewrite(self, path, more))
            }

            /// A writer that appends its bytes to `path` when closed.
            ///
            /// Default: buffers, then `append` on `close`; bytes written to a
            /// writer that is dropped without `close` are lost.
            ///
            /// # Errors
            ///
            /// The errors of `append` surface from `close`.
            #[inline]
            fn append_stream<'bucket, 'call>(
                &'bucket self,
                path: &'call Path,
            ) -> BoxFuture<'call, Result<Box<dyn AsyncWrite + Send + Unpin + 'bucket>>>
            where
                'bucket: 'call,
            {
                let writer: Box<dyn AsyncWrite + Send + Unpin + 'bucket> =
                    Box::new(BufferedWriter::new(self, path, Target::Append));
                Box::pin(ready(Ok(writer)))
            }

            /// The absolute form of `path`.
            ///
            /// Default: lexical, `.` and `..` removed; the disk overrides it with `fs::canonicalize`.
            ///
            /// # Errors
            ///
            /// `InvalidInput` for a relative path, and for `..` above the root.
            #[inline]
            fn canonicalize<'call>(
                &'call self,
                path: &'call Path,
            ) -> BoxFuture<'call, Result<PathBuf>> {
                Box::pin(ready(lexical(path)))
            }

            /// Copy a file and give the number of bytes copied.
            ///
            /// Default: `get`, then `put` with `Condition::None`.
            ///
            /// # Errors
            ///
            /// The errors of `get` and `put`.
            #[inline]
            fn copy<'call>(
                &'call self,
                from: &'call Path,
                to: &'call Path,
            ) -> BoxFuture<'call, Result<u64>> {
                Box::pin(copy_by_rewrite(self, from, to))
            }

            /// Create `path` and every missing folder above it, as `create_dir_all`.
            ///
            /// Default: `Unsupported`.
            ///
            /// # Errors
            ///
            /// `Unsupported` by default. `NotADirectory` when a file is above `path`.
            #[inline]
            fn create_folder<'call>(
                &'call self,
                _path: &'call Path,
            ) -> BoxFuture<'call, Result<()>> {
                Box::pin(ready(Err(StorageError::from_kind(ErrorKind::Unsupported))))
            }

            /// Remove the file at `path` when `condition` holds.
            ///
            /// # Errors
            ///
            /// `AlreadyExists` when `condition` is `Absent` and a file is there.
            /// `PreconditionFailed` when `condition` is `Matches` and the file has
            /// another token or is missing. `IsADirectory` for a folder.
            fn delete<'call>(
                &'call self,
                path: &'call Path,
                condition: &'call Condition,
            ) -> BoxFuture<'call, Result<()>>;

            /// The bytes at `path` and the token of that version, together.
            ///
            /// # Errors
            ///
            /// `NotFound` when nothing is at `path`. `IsADirectory` for a folder.
            fn get<'call>(
                &'call self,
                path: &'call Path,
            ) -> BoxFuture<'call, Result<(Vec<u8>, RevisionToken)>>;

            /// What this backend promises.
            fn guarantees(&self) -> Guarantees;

            /// Kind, size, modification time and token of what is at `path`, without its bytes.
            ///
            /// # Errors
            ///
            /// `NotFound` when nothing is at `path`.
            fn head<'call>(
                &'call self,
                path: &'call Path,
            ) -> BoxFuture<'call, Result<FileMetadata>>;

            /// The entries below `folder`, files and folders, to `depth`.
            ///
            /// # Errors
            ///
            /// `NotADirectory` when `folder` is a file.
            fn list<'call>(
                &'call self,
                folder: &'call Path,
                depth: Depth,
            ) -> BoxFuture<'call, Result<Vec<WalkEntry>>>;

            /// Write `bytes` at `path` when `condition` holds, and give the new token.
            ///
            /// The parent folder must exist.
            ///
            /// # Errors
            ///
            /// `AlreadyExists` when `condition` is `Absent` and a file is there.
            /// `PreconditionFailed` when `condition` is `Matches` and the file has
            /// another token or is missing. `NotFound` for a missing parent folder,
            /// `NotADirectory` for a path under a file, `IsADirectory` for a folder.
            fn put<'call>(
                &'call self,
                path: &'call Path,
                bytes: &'call [u8],
                condition: &'call Condition,
            ) -> BoxFuture<'call, Result<RevisionToken>>;

            /// A reader over the file at `path`.
            ///
            /// Default: `get`, then a reader over the bytes.
            ///
            /// # Errors
            ///
            /// The errors of `get`.
            #[inline]
            fn read_stream<'bucket, 'call>(
                &'bucket self,
                path: &'call Path,
            ) -> BoxFuture<'call, Result<Box<dyn AsyncRead + Send + Unpin + 'bucket>>>
            where
                'bucket: 'call,
            {
                Box::pin(async move {
                    let (bytes, _token) = self.get(path).await?;
                    let reader: Box<dyn AsyncRead + Send + Unpin + 'bucket> =
                        Box::new(BytesReader::new(bytes));
                    Ok(reader)
                })
            }

            /// Remove `path` and everything below it, as `remove_dir_all`.
            ///
            /// Default: `Unsupported`.
            ///
            /// # Errors
            ///
            /// `Unsupported` by default. `NotFound` when no folder is at `path`.
            #[inline]
            fn remove_folder<'call>(
                &'call self,
                _path: &'call Path,
            ) -> BoxFuture<'call, Result<()>> {
                Box::pin(ready(Err(StorageError::from_kind(ErrorKind::Unsupported))))
            }

            /// Move a file or a folder.
            ///
            /// Default: copy, then delete the source under its token; on a failed
            /// delete the copy is removed again. A folder moves file by file.
            ///
            /// # Errors
            ///
            /// The errors of the operations the default is made of.
            #[inline]
            fn rename<'call>(
                &'call self,
                from: &'call Path,
                to: &'call Path,
            ) -> BoxFuture<'call, Result<()>> {
                Box::pin(rename_by_copy(self, from, to))
            }

            /// Every entry below `folder`.
            ///
            /// Default: `list` with `Depth::All`, without entries under a
            /// dot-prefixed component unless `hidden`; `follow_links` does nothing.
            ///
            /// # Errors
            ///
            /// The errors of `list`.
            #[inline]
            fn walk<'call>(
                &'call self,
                folder: &'call Path,
                _follow_links: bool,
                hidden: bool,
            ) -> BoxFuture<'call, Result<Vec<WalkEntry>>> {
                Box::pin(walk_by_list(self, folder, hidden))
            }

            /// A writer that stores its bytes at `path` under `condition` when closed.
            ///
            /// Default: buffers, then `put` on `close`; bytes written to a writer
            /// that is dropped without `close` are lost.
            ///
            /// # Errors
            ///
            /// The errors of `put` surface from `close`.
            #[inline]
            fn write_stream<'bucket, 'call>(
                &'bucket self,
                path: &'call Path,
                condition: Condition,
            ) -> BoxFuture<'call, Result<Box<dyn AsyncWrite + Send + Unpin + 'bucket>>>
            where
                'bucket: 'call,
            {
                let writer: Box<dyn AsyncWrite + Send + Unpin + 'bucket> =
                    Box::new(BufferedWriter::new(self, path, Target::Put(condition)));
                Box::pin(ready(Ok(writer)))
            }
        }
    };
}
