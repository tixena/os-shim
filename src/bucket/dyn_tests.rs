//! `Bucket` behind `dyn`: every method on each backend every build has, and `Arc<dyn Bucket>` where a bucket goes.

use super::disk::DiskBucket;
use super::{Bucket, Condition, Depth, Guarantee, MemoryBucket};
use crate::System as _;
use crate::error::{ErrorKind, Result};
use crate::mock::MemorySystem;
use crate::real::RealSystem;
use crate::streams::{close, read_to_end, write_all};
use alloc::sync::Arc;
use std::path::Path;

/// `create_folder`, `put`, `get`, `append` and `head`, leaving a 9-byte file at `folder/licencia.pdf`.
async fn writes_and_reads(store: &dyn Bucket, folder: &Path) -> Result<()> {
    let file = folder.join("licencia.pdf");
    store.create_folder(folder).await?;
    let first = store.put(&file, b"%PDF-1.7", &Condition::Absent).await?;
    let refused = store.put(&file, b"other", &Condition::Absent).await;
    assert_eq!(
        refused.map(drop).map_err(|error| error.kind()),
        Err(ErrorKind::AlreadyExists)
    );
    assert_eq!(store.get(&file).await?, (b"%PDF-1.7".to_vec(), first));
    store.append(&file, b"+").await?;
    assert_eq!(store.head(&file).await?.len, 9);
    Ok(())
}

/// `copy`, `rename`, `canonicalize`, `list` and `walk`, leaving two files in `folder`.
async fn moves_and_lists(store: &dyn Bucket, folder: &Path) -> Result<()> {
    let file = folder.join("licencia.pdf");
    let copy = folder.join("copia.pdf");
    assert_eq!(store.copy(&file, &copy).await?, 9);
    store.rename(&copy, &folder.join("movida.pdf")).await?;
    let dotted = folder.join("./licencia.pdf");
    assert_eq!(
        store.canonicalize(&dotted).await?,
        store.canonicalize(&file).await?
    );
    assert_eq!(store.list(folder, Depth::Children).await?.len(), 2);
    assert_eq!(store.walk(folder, false, false).await?.len(), 2);
    Ok(())
}

/// Write through each stream and read the bytes back; `delete` removes a file and `remove_folder` the rest.
async fn streams_and_removals(store: &dyn Bucket, folder: &Path) -> Result<()> {
    let fresh = folder.join("nuevo.pdf");
    let mut writer = store.write_stream(&fresh, Condition::Absent).await?;
    write_all(&mut *writer, b"written").await?;
    close(&mut *writer).await?;
    drop(writer);
    let mut appender = store.append_stream(&fresh).await?;
    write_all(&mut *appender, b"+more").await?;
    close(&mut *appender).await?;
    drop(appender);
    let mut reader = store.read_stream(&fresh).await?;
    assert_eq!(read_to_end(&mut *reader).await?, b"written+more");
    drop(reader);
    store
        .delete(&folder.join("movida.pdf"), &Condition::None)
        .await?;
    store.remove_folder(folder).await?;
    let gone = store.get(&fresh).await;
    assert_eq!(
        gone.map(drop).map_err(|error| error.kind()),
        Err(ErrorKind::NotFound)
    );
    Ok(())
}

/// Every method but `guarantees`, through `dyn`, in `folder`.
pub(super) async fn every_method(store: &dyn Bucket, folder: &Path) -> Result<()> {
    writes_and_reads(store, folder).await?;
    moves_and_lists(store, folder).await?;
    streams_and_removals(store, folder).await
}

#[tokio::test(flavor = "current_thread")]
async fn memory_answers_through_dyn() -> Result<()> {
    let store: Arc<dyn Bucket> = Arc::new(MemoryBucket::default());
    every_method(store.as_ref(), Path::new("/documents")).await
}

#[tokio::test(flavor = "current_thread")]
async fn disk_answers_through_dyn() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let store: Arc<dyn Bucket> = Arc::new(DiskBucket::new());
    every_method(store.as_ref(), &tmp.path().join("documents")).await
}

#[test]
fn guarantees_are_the_backends() {
    let store: Arc<dyn Bucket> = Arc::new(MemoryBucket::default());
    assert!(store.guarantees().contains(Guarantee::FolderRenameIsAtomic));
}

/// A `RealSystem` and a `MemorySystem` that share `store` read each other's files.
async fn systems_share_a_dyn_bucket(store: Arc<dyn Bucket>) -> Result<()> {
    let file = Path::new("/licencia.pdf");
    let real = RealSystem::with_bucket(Arc::clone(&store));
    real.write(file, b"%PDF-1.7", Condition::Absent).await?;
    let memory = MemorySystem::with_bucket(store);
    assert_eq!(memory.read(file).await?.0, b"%PDF-1.7");
    Ok(())
}

#[tokio::test(flavor = "current_thread")]
async fn a_system_holds_a_dyn_bucket() -> Result<()> {
    systems_share_a_dyn_bucket(Arc::new(MemoryBucket::default())).await
}
