//! Lexical path rules every bucket shares, and the keys of buckets with string keys.

use crate::error::{ErrorKind, Result, StorageError};
use std::path::{Component, Path, PathBuf};

/// The absolute form of `path` with `.` and `..` removed, with no file system consulted.
///
/// # Errors
///
/// `InvalidInput` for a path with no root and for `..` above the root.
pub fn lexical(path: &Path) -> Result<PathBuf> {
    let mut resolved = PathBuf::new();
    let mut depth = 0_usize;
    let mut rooted = false;
    for component in path.components() {
        match component {
            Component::Prefix(_) => resolved.push(component),
            Component::RootDir => {
                resolved.push(component);
                rooted = true;
            }
            Component::CurDir => {}
            Component::ParentDir => {
                if depth == 0 {
                    return Err(invalid(format!("{} goes above the root", path.display())));
                }
                resolved.pop();
                depth = depth.saturating_sub(1);
            }
            Component::Normal(name) => {
                resolved.push(name);
                depth = depth.saturating_add(1);
            }
        }
    }
    if !rooted {
        return Err(invalid(format!(
            "{} is relative; a bucket takes absolute paths",
            path.display()
        )));
    }
    Ok(resolved)
}

/// The key of `path` under `prefix`: its components after the root, joined by `/`.
///
/// # Errors
///
/// `InvalidInput` for a relative path, `..` above the root and a component that is not UTF-8.
#[cfg(any(feature = "minio", feature = "s3"))]
pub fn key_of(prefix: &str, path: &Path) -> Result<String> {
    let mut parts: Vec<&str> = Vec::new();
    if !prefix.is_empty() {
        parts.push(prefix);
    }
    let resolved = lexical(path)?;
    let mut names = Vec::new();
    for component in resolved.components() {
        match component {
            Component::Prefix(shown) => names.push(
                shown
                    .as_os_str()
                    .to_str()
                    .ok_or_else(|| not_utf8(path))?
                    .to_owned(),
            ),
            Component::Normal(name) => {
                names.push(name.to_str().ok_or_else(|| not_utf8(path))?.to_owned());
            }
            Component::RootDir | Component::CurDir | Component::ParentDir => {}
        }
    }
    parts.extend(names.iter().map(String::as_str));
    Ok(parts.join("/"))
}

/// An `InvalidInput` error saying `message`.
fn invalid(message: String) -> StorageError {
    StorageError::new(ErrorKind::InvalidInput, message)
}

/// The `InvalidInput` error for a path with a component that is not UTF-8.
#[cfg(any(feature = "minio", feature = "s3"))]
fn not_utf8(path: &Path) -> StorageError {
    invalid(format!("{} is not UTF-8", path.display()))
}
