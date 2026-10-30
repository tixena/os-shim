use super::{ErrorKind, StorageError};
use core::error::Error as _;
use std::io;

const ALL: [ErrorKind; 11] = [
    ErrorKind::AlreadyExists,
    ErrorKind::InvalidData,
    ErrorKind::InvalidInput,
    ErrorKind::IsADirectory,
    ErrorKind::NotADirectory,
    ErrorKind::NotFound,
    ErrorKind::Other,
    ErrorKind::PermissionDenied,
    ErrorKind::PreconditionFailed,
    ErrorKind::Unavailable,
    ErrorKind::Unsupported,
];

#[test]
fn io_kinds_map_to_their_kind() {
    let cases = [
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
        (io::ErrorKind::BrokenPipe, ErrorKind::Other),
    ];
    for (from, expected) in cases {
        assert_eq!(
            StorageError::from(io::Error::from(from)).kind(),
            expected,
            "{from:?}"
        );
    }
}

#[test]
fn every_kind_survives_a_round_trip_through_io() {
    for kind in ALL {
        let through_io = io::Error::from(StorageError::from_kind(kind));
        assert_eq!(StorageError::from(through_io).kind(), kind, "{kind:?}");
    }
}

#[test]
fn error_to_io_uses_the_mapped_io_kind() {
    let cases = [
        (ErrorKind::AlreadyExists, io::ErrorKind::AlreadyExists),
        (ErrorKind::InvalidData, io::ErrorKind::InvalidData),
        (ErrorKind::InvalidInput, io::ErrorKind::InvalidInput),
        (ErrorKind::IsADirectory, io::ErrorKind::IsADirectory),
        (ErrorKind::NotADirectory, io::ErrorKind::NotADirectory),
        (ErrorKind::NotFound, io::ErrorKind::NotFound),
        (ErrorKind::Other, io::ErrorKind::Other),
        (ErrorKind::PermissionDenied, io::ErrorKind::PermissionDenied),
        (ErrorKind::PreconditionFailed, io::ErrorKind::Other),
        (ErrorKind::Unavailable, io::ErrorKind::Other),
        (ErrorKind::Unsupported, io::ErrorKind::Unsupported),
    ];
    for (kind, expected) in cases {
        assert_eq!(
            io::Error::from(StorageError::from_kind(kind)).kind(),
            expected
        );
    }
}

#[test]
fn display_shows_the_source_else_a_phrase() {
    let with_source = StorageError::new(ErrorKind::Other, "disk on fire");
    assert_eq!(with_source.to_string(), "disk on fire");
    assert!(with_source.source().is_some());

    let bare = StorageError::from_kind(ErrorKind::NotFound);
    assert_eq!(bare.to_string(), "nothing is at the path");
    assert!(bare.source().is_none());
}

#[test]
fn precondition_failed_through_io_keeps_its_kind() {
    let stale = StorageError::new(ErrorKind::PreconditionFailed, "the revision token changed");
    let through_io: io::Error = stale.into();
    assert_eq!(through_io.kind(), io::ErrorKind::Other);
    assert_eq!(
        StorageError::from(through_io).kind(),
        ErrorKind::PreconditionFailed
    );
}
