//! Reading and writing `futures-io` streams to the end.

use core::future::poll_fn;
use core::pin::Pin;
use futures_io::{AsyncRead, AsyncWrite};
use std::io;

/// The next bytes of `reader` in `buffer`, and how many came.
pub async fn read_chunk(
    reader: &mut (dyn AsyncRead + Send + Unpin),
    buffer: &mut [u8],
) -> io::Result<usize> {
    poll_fn(|cx| Pin::new(&mut *reader).poll_read(cx, buffer)).await
}

/// All of `bytes` into `writer`.
pub async fn write_all(
    writer: &mut (dyn AsyncWrite + Send + Unpin),
    mut bytes: &[u8],
) -> io::Result<()> {
    while !bytes.is_empty() {
        let written = poll_fn(|cx| Pin::new(&mut *writer).poll_write(cx, bytes)).await?;
        if written == 0 {
            return Err(io::ErrorKind::WriteZero.into());
        }
        bytes = bytes.get(written..).unwrap_or_default();
    }
    Ok(())
}

/// Close `writer`, which is when a bucket stores what it was given.
pub async fn close(writer: &mut (dyn AsyncWrite + Send + Unpin)) -> io::Result<()> {
    poll_fn(|cx| Pin::new(&mut *writer).poll_close(cx)).await
}

/// Every byte `reader` has left.
#[cfg(test)]
pub async fn read_to_end(reader: &mut (dyn AsyncRead + Send + Unpin)) -> io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    let mut buffer = vec![0_u8; 4096];
    loop {
        let filled = read_chunk(reader, &mut buffer).await?;
        let Some(chunk) = buffer.get(..filled).filter(|chunk| !chunk.is_empty()) else {
            return Ok(bytes);
        };
        bytes.extend_from_slice(chunk);
    }
}
