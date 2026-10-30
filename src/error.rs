//! The error every storage operation of os-shim returns, with a kind that means the same on
//! every backend.

use core::error::Error as StdError;
use core::fmt;
use core::result;
use std::io;

/// The `io::ErrorKind`s with a kind of their own.
const IO_KINDS: [(io::ErrorKind, ErrorKind); 15] = [
    (io::ErrorKind::NotFound, ErrorKind::NotFound),
    (io::ErrorKind::AlreadyExists, ErrorKind::AlreadyExists),
    (io::ErrorKind::PermissionDenied, ErrorKind::PermissionDenied),
    (io::ErrorKind::InvalidInput, ErrorKind::InvalidInput),
    (io::ErrorKind::InvalidData, ErrorKind::InvalidData),
    (io::ErrorKind::IsADirectory, ErrorKind::IsADirectory),
    (io::ErrorKind::NotADirectory, ErrorKind::NotADirectory),
    (io::ErrorKind::Unsupported, ErrorKind::Unsupported),
    (io::ErrorKind::TimedOut, ErrorKind::Unavailable),
    (io::ErrorKind::ConnectionRefused, ErrorKind::Unavailable),
    (io::ErrorKind::ConnectionReset, ErrorKind::Unavailable),
    (io::ErrorKind::ConnectionAborted, ErrorKind::Unavailable),
    (io::ErrorKind::NotConnected, ErrorKind::Unavailable),
    (io::ErrorKind::HostUnreachable, ErrorKind::Unavailable),
    (io::ErrorKind::NetworkUnreachable, ErrorKind::Unavailable),
];

/// Which way a storage operation failed, the same on every backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ErrorKind {
    /// A write under `Condition::Absent` found something at the path.
    AlreadyExists,
    /// Bytes that are not what the operation reads, such as text that is not UTF-8.
    InvalidData,
    /// An argument the backend cannot take, such as a path that is not UTF-8 on a bucket with string keys.
    InvalidInput,
    /// A file operation on a folder.
    IsADirectory,
    /// A folder operation on a file.
    NotADirectory,
    /// Nothing is at the path.
    NotFound,
    /// Any other failure.
    Other,
    /// The file or the request was refused.
    PermissionDenied,
    /// A write or a delete under `Condition::Matches` found another revision token.
    PreconditionFailed,
    /// The store cannot be reached, or the request timed out.
    Unavailable,
    /// The backend does not offer the operation.
    Unsupported,
}

impl ErrorKind {
    /// The fixed phrase `Display` shows when there is no source.
    const fn phrase(self) -> &'static str {
        match self {
            Self::AlreadyExists => "something is already at the path",
            Self::InvalidData => "the data is not valid",
            Self::InvalidInput => "the argument is not valid",
            Self::IsADirectory => "the path is a directory",
            Self::NotADirectory => "the path is not a directory",
            Self::NotFound => "nothing is at the path",
            Self::Other => "the operation failed",
            Self::PermissionDenied => "permission denied",
            Self::PreconditionFailed => "the condition does not hold",
            Self::Unavailable => "the store is unavailable",
            Self::Unsupported => "the backend does not offer the operation",
        }
    }

    /// The `io::ErrorKind` this kind maps to.
    const fn to_io(self) -> io::ErrorKind {
        match self {
            Self::AlreadyExists => io::ErrorKind::AlreadyExists,
            Self::InvalidData => io::ErrorKind::InvalidData,
            Self::InvalidInput => io::ErrorKind::InvalidInput,
            Self::IsADirectory => io::ErrorKind::IsADirectory,
            Self::NotADirectory => io::ErrorKind::NotADirectory,
            Self::NotFound => io::ErrorKind::NotFound,
            Self::PermissionDenied => io::ErrorKind::PermissionDenied,
            Self::Unsupported => io::ErrorKind::Unsupported,
            Self::Other | Self::PreconditionFailed | Self::Unavailable => io::ErrorKind::Other,
        }
    }
}

/// Why a storage operation failed: its kind, and the underlying failure when there is one.
#[derive(Debug)]
pub struct StorageError {
    /// Which way it failed.
    kind: ErrorKind,
    /// The underlying failure, when there is one.
    source: Option<Box<dyn StdError + Send + Sync>>,
}

impl StorageError {
    /// An error of `kind` with no underlying failure.
    #[inline]
    #[must_use]
    pub const fn from_kind(kind: ErrorKind) -> Self {
        Self { kind, source: None }
    }

    /// Which way the operation failed.
    #[inline]
    #[must_use]
    pub const fn kind(&self) -> ErrorKind {
        self.kind
    }

    /// An error of `kind` caused by `source`.
    #[inline]
    pub fn new<S>(kind: ErrorKind, source: S) -> Self
    where
        S: Into<Box<dyn StdError + Send + Sync>>,
    {
        Self {
            kind,
            source: Some(source.into()),
        }
    }
}

impl fmt::Display for StorageError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.source.as_ref() {
            Some(source) => write!(f, "{source}"),
            None => f.write_str(self.kind.phrase()),
        }
    }
}

impl StdError for StorageError {
    #[inline]
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.source.as_ref().map(|boxed| {
            let inner: &(dyn StdError + 'static) = boxed.as_ref();
            inner
        })
    }
}

impl From<io::Error> for StorageError {
    #[inline]
    fn from(error: io::Error) -> Self {
        let carried = error
            .get_ref()
            .and_then(|inner| inner.downcast_ref::<Self>())
            .map(|ours| ours.kind);
        let kind = carried.unwrap_or_else(|| {
            let io_kind = error.kind();
            IO_KINDS
                .iter()
                .find(|pair| pair.0 == io_kind)
                .map_or(ErrorKind::Other, |pair| pair.1)
        });
        Self {
            kind,
            source: Some(Box::new(error)),
        }
    }
}

impl From<StorageError> for io::Error {
    #[inline]
    fn from(error: StorageError) -> Self {
        Self::new(error.kind.to_io(), error)
    }
}

/// The result of a storage operation.
pub type Result<T> = result::Result<T, StorageError>;

#[cfg(test)]
mod tests;
