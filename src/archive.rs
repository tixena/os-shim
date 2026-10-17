//! Zip archive support for the [`System`] trait: a tree in, a zip stream out, and back.

use crate::System;
use alloc::collections::BTreeSet;
use std::io::{self, Cursor, Read, Seek, Write};
use std::path::{Component, Path, PathBuf};
use time::{OffsetDateTime, PrimitiveDateTime};
use zip::read::{ZipFile, read_zipfile_from_stream};
use zip::write::{SimpleFileOptions, StreamWriter, ZipWriter};
use zip::{CompressionMethod, DateTime, ZipArchive};

/// Bytes held by the copy buffer (64 KiB), heap-allocated because a stack array
/// this size trips `clippy::large_stack_arrays`.
const COPY_BUFFER_BYTES: usize = 0x1_0000;

/// First year a ZIP timestamp can express.
const ZIP_EPOCH_YEAR: i32 = 1980;

/// Options controlling archive construction.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct ZipOptions {
    /// Whether dot-prefixed entries are archived.
    pub include_hidden: bool,
}

impl ZipOptions {
    /// Set whether dot-prefixed entries are archived (builder pattern).
    #[must_use]
    #[inline]
    pub const fn with_hidden(mut self, include_hidden: bool) -> Self {
        self.include_hidden = include_hidden;
        self
    }
}

impl Default for ZipOptions {
    #[inline]
    fn default() -> Self {
        Self {
            include_hidden: true,
        }
    }
}

/// A zip writer that writes one entry at a time into a sink that cannot seek.
///
/// Each entry is deflated and its sizes follow its data in a data descriptor,
/// so the archive loads through [`load_zip_seekable`] and not through
/// [`load_zip`], which needs the sizes in the local header.
pub struct ZipStreamWriter<W: Write> {
    /// The zip writer over the sink.
    zip: ZipWriter<StreamWriter<W>>,
}

impl<W: Write> ZipStreamWriter<W> {
    /// Write the central directory and give the sink back.
    ///
    /// # Errors
    ///
    /// Returns an error if the sink fails.
    #[inline]
    pub fn finish(self) -> io::Result<W> {
        self.zip
            .finish()
            .map(StreamWriter::into_inner)
            .map_err(io::Error::other)
    }

    /// A writer over `sink`, with no entry started.
    #[must_use]
    #[inline]
    pub fn new(sink: W) -> Self {
        Self {
            zip: ZipWriter::new_stream(sink),
        }
    }

    /// End the entry being written, if any, and start the next; the bytes
    /// written next are its content.
    ///
    /// # Errors
    ///
    /// Returns an error if the name has no zip representation or the sink fails.
    #[inline]
    pub fn start_entry(&mut self, name: &str) -> io::Result<()> {
        let options = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
        self.zip.start_file(name, options).map_err(io::Error::other)
    }
}

impl<W: Write> Write for ZipStreamWriter<W> {
    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        self.zip.flush()
    }

    #[inline]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.zip.write(buf)
    }
}

/// Stream the file at `path` into `sink` through a heap-allocated buffer.
fn copy_into<S>(system: &S, path: &Path, sink: &mut dyn Write) -> io::Result<()>
where
    S: System + ?Sized,
{
    let mut reader = system.open(path)?;
    let mut buffer = vec![0_u8; COPY_BUFFER_BYTES];
    loop {
        let filled = reader.read(&mut buffer)?;
        if filled == 0 {
            break;
        }
        let chunk = buffer
            .get(..filled)
            .ok_or_else(|| io::Error::other("Read reported more bytes than the buffer holds"))?;
        sink.write_all(chunk)?;
    }
    Ok(())
}

/// Archive entry name for `path`, relative to `source`.
fn entry_name(source: &Path, path: &Path) -> io::Result<String> {
    let relative = path.strip_prefix(source).map_err(io::Error::other)?;
    let mut name = String::new();
    let mut first = true;
    for component in relative.components() {
        let Component::Normal(part) = component else {
            return Err(io::Error::other(format!(
                "Unsupported path component in {}",
                path.display()
            )));
        };
        let text = part.to_str().ok_or_else(|| {
            io::Error::other(format!("Non-UTF-8 path component in {}", path.display()))
        })?;
        if !first {
            name.push('/');
        }
        name.push_str(text);
        first = false;
    }
    Ok(name)
}

/// Per-entry zip options carrying the source's real modification time.
///
/// A ZIP timestamp is a DOS date: 1980-01-01 to 2107-12-31, two-second
/// resolution. Anything outside that clamps to the nearest end rather than
/// failing the archive -- the in-memory filesystem stamps un-written directories
/// with the Unix epoch, which predates the format's floor by a decade.
fn entry_options<S>(system: &S, path: &Path) -> io::Result<SimpleFileOptions>
where
    S: System + ?Sized,
{
    let modified = system.metadata(path)?.modified;
    let offset = OffsetDateTime::from(modified);
    let civil = PrimitiveDateTime::new(offset.date(), offset.time());
    let stamp = DateTime::try_from(civil).unwrap_or_else(|_ignored| {
        if civil.year() < ZIP_EPOCH_YEAR {
            zip_floor()
        } else {
            zip_ceiling()
        }
    });
    Ok(SimpleFileOptions::default().last_modified_time(stamp))
}

/// Archive entry name for a lone file: its final component and nothing else.
fn file_name_of(path: &Path) -> io::Result<String> {
    let name = path
        .file_name()
        .ok_or_else(|| io::Error::other(format!("Path has no file name: {}", path.display())))?;
    name.to_str()
        .map(str::to_owned)
        .ok_or_else(|| io::Error::other(format!("Non-UTF-8 file name in {}", path.display())))
}

/// Per-entry zip options for a file, pinned to zip64.
///
/// `large_file(true)` goes on every file, not on files a size check picks out.
/// Deflate can make incompressible data larger than its source and the zip crate
/// rejects an oversized compressed length separately from an oversized
/// uncompressed one, so any threshold needs a margin, and a wrong margin fails
/// only on multi-gigabyte inputs nobody exercises routinely. The cost is a
/// 20-byte zip64 extra field per file and a version-needed-to-extract of 4.5.
///
/// Directory entries keep the plain options. `ZipWriter::add_directory` marks
/// the entry finished the moment it is opened, so the writer never revisits the
/// local header to replace the zip64 placeholder sizes it stamped there --
/// `u64::MAX` for both -- and a streaming reader then waits for that many bytes
/// of entry data and hits end of file. A directory carries no data anyway.
fn file_options<S>(system: &S, path: &Path) -> io::Result<SimpleFileOptions>
where
    S: System + ?Sized,
{
    Ok(entry_options(system, path)?.large_file(true))
}

/// Whether `path`'s final component is dot-prefixed.
fn is_hidden(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.starts_with('.'))
}

/// Read a zip archive from `source` and materialise its entries under `destination`.
///
/// The reader need not be seekable, so a pipe is a valid source. An entry whose
/// sizes follow its data in a data descriptor cannot be sized from a stream and is
/// refused; [`load_zip_seekable`] reads such archives.
///
/// # Errors
///
/// Returns an error if the archive is malformed, an entry name escapes the
/// destination, or an entry cannot be written.
#[inline]
pub fn load_zip<S>(system: &S, mut source: &mut dyn Read, destination: &Path) -> io::Result<()>
where
    S: System + ?Sized,
{
    system.create_dir_all(destination)?;
    while let Some(mut entry) = read_zipfile_from_stream(&mut source).map_err(io::Error::other)? {
        materialise_entry(system, &mut entry, destination)?;
    }
    Ok(())
}

/// Read a zip archive from a seekable `source` and materialise its entries under `destination`.
///
/// Entries are found through the central directory at the end of the archive,
/// so an entry whose sizes follow its data in a data descriptor loads too;
/// [`load_zip`] refuses those, because a stream offers nothing past the local
/// header to size the entry by.
///
/// # Errors
///
/// Returns an error if the archive is malformed, an entry name escapes the
/// destination, or an entry cannot be written.
#[inline]
pub fn load_zip_seekable<S, R>(system: &S, source: R, destination: &Path) -> io::Result<()>
where
    S: System + ?Sized,
    R: Read + Seek,
{
    let mut archive = ZipArchive::new(source).map_err(io::Error::other)?;
    system.create_dir_all(destination)?;
    for index in 0..archive.len() {
        let mut entry = archive.by_index(index).map_err(io::Error::other)?;
        materialise_entry(system, &mut entry, destination)?;
    }
    Ok(())
}

/// Write one archive entry under `destination`: a directory, or a file and its parents.
fn materialise_entry<S, R>(
    system: &S,
    entry: &mut ZipFile<'_, R>,
    destination: &Path,
) -> io::Result<()>
where
    S: System + ?Sized,
    R: Read + ?Sized,
{
    let Some(relative) = entry.enclosed_name() else {
        return Err(io::Error::other(format!(
            "Archive entry escapes the destination: {}",
            String::from_utf8_lossy(entry.name_raw())
        )));
    };
    let target = destination.join(relative);
    if entry.is_dir() {
        return system.create_dir_all(&target);
    }
    if let Some(parent) = target.parent() {
        system.create_dir_all(parent)?;
    }
    // Streamed rather than read whole: `read_to_end` would hold the entry's
    // full *uncompressed* length, which for a real filesystem is a gigabyte
    // of resident memory to write a gigabyte file. `create` hands back a
    // sink, so the entry moves through a fixed buffer instead.
    let mut sink = system.create(&target)?;
    io::copy(entry, &mut sink)?;
    sink.flush()
}

/// Write `source` into `sink` as a zip archive.
///
/// A directory contributes its *contents*: archiving `A/` yields `file.txt`, not
/// `A/file.txt`. A lone file yields exactly one entry, named after the file.
///
/// # Errors
///
/// Returns an error if:
/// - `source` does not exist.
/// - The source tree cannot be read, or holds a name with no ZIP representation.
/// - The sink cannot be written.
#[inline]
pub fn to_zip_writer<S>(
    system: &S,
    source: &Path,
    sink: &mut dyn Write,
    options: ZipOptions,
) -> io::Result<()>
where
    S: System + ?Sized,
{
    // Seekable, not streaming: `ZipWriter::new_stream` puts each entry's size in a
    // trailing data descriptor instead of its local header, and an archive shaped
    // that way cannot be read back by `zip::read::read_zipfile_from_stream` -- it
    // fails with "The file length is not available in the local header". Building
    // into a rewindable buffer keeps the round trip through a pipe possible, and
    // is also what makes Info-ZIP's `unzip -t` report a clean archive.
    let mut buffer = Cursor::new(Vec::<u8>::new());
    let mut writer = ZipWriter::new(&mut buffer);

    if system.is_dir(source)? {
        let (dirs, files) = walk(system, source, options)?;
        for dir in &dirs {
            writer
                .add_directory(entry_name(source, dir)?, entry_options(system, dir)?)
                .map_err(io::Error::other)?;
        }
        for file in &files {
            writer
                .start_file(entry_name(source, file)?, file_options(system, file)?)
                .map_err(io::Error::other)?;
            copy_into(system, file, &mut writer)?;
        }
    } else if system.is_file(source)? {
        writer
            .start_file(file_name_of(source)?, file_options(system, source)?)
            .map_err(io::Error::other)?;
        copy_into(system, source, &mut writer)?;
    } else {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("Archive source not found: {}", source.display()),
        ));
    }

    writer.finish().map_err(io::Error::other)?;
    sink.write_all(&buffer.into_inner())?;
    Ok(())
}

/// Collect the directories and files under `source`, guarding against symlink cycles.
fn walk<S>(
    system: &S,
    source: &Path,
    options: ZipOptions,
) -> io::Result<(Vec<PathBuf>, Vec<PathBuf>)>
where
    S: System + ?Sized,
{
    let mut stack: Vec<PathBuf> = vec![source.to_path_buf()];
    let mut dirs: Vec<PathBuf> = Vec::new();
    let mut files: Vec<PathBuf> = Vec::new();
    // A directory that canonicalizes to somewhere already walked is a symlink
    // back into the tree; descending into it would recurse until the kernel
    // refuses the path depth.
    let mut seen: BTreeSet<PathBuf> = BTreeSet::new();
    seen.insert(system.canonicalize(source)?);

    while let Some(current) = stack.pop() {
        let mut children = system.read_dir(&current)?;
        children.sort();
        for child in children {
            if !options.include_hidden && is_hidden(&child) {
                continue;
            }
            if system.is_dir(&child)? {
                dirs.push(child.clone());
                if seen.insert(system.canonicalize(&child)?) {
                    stack.push(child);
                }
            } else {
                files.push(child);
            }
        }
    }

    dirs.sort();
    files.sort();
    Ok((dirs, files))
}

/// The latest instant a ZIP timestamp can express: 2107-12-31 23:59:58.
fn zip_ceiling() -> DateTime {
    DateTime::from_date_and_time(2107, 12, 31, 23, 59, 58).unwrap_or_default()
}

/// The earliest instant a ZIP timestamp can express: 1980-01-01 00:00:00.
fn zip_floor() -> DateTime {
    DateTime::default()
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "Tests use unwrap for brevity; panics indicate test failure"
)]
mod tests;
