# os-shim

[![XERJ Corpus Hub contributor](https://hub.xerj.org/contributors/eburgos/badge.svg)](https://hub.xerj.org/contributors/eburgos)

A mockable layer between Rust code and the machine it runs on. Code written against os-shim's traits runs on the real system in production and on an in-memory twin in tests, with no filesystem, server or clock to set up.

| Piece | Real implementation | In-memory twin | Feature |
|---|---|---|---|
| Environment, files, folders, temporary directories (`System`) | `RealSystem` | `MemorySystem` | always |
| Whether a process is alive (`System::is_pid_alive`) | `RealSystem` | `MemorySystem::with_pid` | `process` |
| Zip archives of a tree, and trees loaded from archives | `archive`, `System::to_zip_*` | the same, over `MemorySystem` | `zip` |
| Files and folders in a bucket (`Bucket`) | `DiskBucket`, and `MinioBucket`, `S3Bucket` for S3-compatible stores | `MemoryBucket` | always; `minio`, `s3` for the two clients |

No feature is on by default. os-shim needs Rust 1.89 or newer; the `minio` and `s3` features need 1.94.1, because of the AWS crates.

## The system

```rust
use os_shim::System;
use os_shim::mock::MemorySystem;
use std::path::Path;

async fn config_path(system: &dyn System) -> os_shim::error::Result<String> {
    let home = system.env_var("HOME").map_err(std::io::Error::other)?;
    system.read_to_string(&Path::new(&home).join(".app.toml")).await
}

let system = MemorySystem::new()
    .with_env("HOME", "/home/user")?
    .with_file("/home/user/.app.toml", b"debug = true")?;
assert_eq!(config_path(&system).await?, "debug = true");
```

In production the same function takes `&os_shim::real::RealSystem::new()`, which reads `std::env` and keeps its files on the local disk. Every file operation is async and is carried out by the bucket the system holds: `RealSystem::with_bucket` and `MemorySystem::with_bucket` hold any other. `System` covers environment variables, the current directory and executable, reading, writing, appending, copying, renaming and removing files, removing a file only while it still has the revision token the caller read, folders and their walks, metadata, temporary directories that are removed on drop and live on local storage whatever the bucket is, the current time, and an async wait that `RealSystem` spends on a thread of its own, with no runtime, and `MemorySystem` answers by moving its clock. `MemorySystem`'s clock starts at the Unix epoch; `with_now` sets it, `advance` moves it forward, and each file `MemorySystem` writes takes its modification time from it.

## Zip archives (`zip`)

```rust
use os_shim::System;
use os_shim::mock::MemorySystem;
use std::io::Cursor;
use std::path::Path;

let system = MemorySystem::new().with_file("/project/src/main.rs", b"fn main() {}")?;
let archive = system.to_zip_vec(Path::new("/project")).await?;

let loaded = MemorySystem::from_zip_reader(Cursor::new(archive))?;
assert_eq!(loaded.read_to_string(Path::new("/src/main.rs")).await?, "fn main() {}");
```

`System::to_zip_stream`, `to_zip_vec` and `to_zip_writer` archive a tree. `archive::load_zip` loads an archive from a stream, `archive::load_zip_seekable` from a seekable reader, including archives whose entries carry data descriptors. `archive::ZipStreamWriter` writes an archive one entry at a time into a sink that cannot seek. The `System` methods and the two `load_zip` functions are async and stream through a fixed buffer; `MemorySystem::from_zip_reader` and `from_zip_stream` are synchronous.

## Buckets

A `Bucket` holds files and folders, each put and delete under an optional condition. `Bucket`, `DiskBucket` and `MemoryBucket` are in every build. `MinioBucket` needs the `minio` feature, `S3Bucket` needs `s3`, and `BlockingBucket` needs either.

| `Condition` | A put or delete goes ahead only if |
|---|---|
| `None` | always |
| `Absent` | nothing is at the path |
| `Matches(token)` | the file at the path still has this revision token |

A failed `Absent` is an error whose `kind()` is `ErrorKind::AlreadyExists`, a failed `Matches` is `ErrorKind::PreconditionFailed`, and a missing file is `ErrorKind::NotFound`.

```rust
use os_shim::bucket::{BlockingBucket, Condition, MemoryBucket};
use os_shim::error::ErrorKind;
use std::path::Path;

let bucket = BlockingBucket::new(MemoryBucket::default())?;
let task = Path::new("/task.md");

let first = bucket.put(task, b"# Task\n", &Condition::Absent)?;
let refused = bucket.put(task, b"# Other\n", &Condition::Absent);
assert_eq!(refused.unwrap_err().kind(), ErrorKind::AlreadyExists);

let second = bucket.put(task, b"# Done\n", &Condition::Matches(first.clone()))?;
let stale = bucket.delete(task, &Condition::Matches(first));
assert_eq!(stale.unwrap_err().kind(), ErrorKind::PreconditionFailed);

bucket.delete(task, &Condition::Matches(second))?;
assert_eq!(bucket.get(task).unwrap_err().kind(), ErrorKind::NotFound);
```

The clients reach a server through AWS's SDK, `aws-sdk-s3`, and never retry a request:

- `MinioBucket` (`minio`) talks to a server that speaks MinIO's API, such as MinIO or RustFS, with path-style addressing. Some of these servers ignore `If-Match` on a delete, so a conditional delete reads the object's tag first and sends `If-Match` as well.
- `S3Bucket` (`s3`) talks to AWS S3, which checks `If-Match` on a delete itself.

On both, a folder is an empty object whose key ends in `/`. `create_folder` writes one for each missing level, and a put needs its parent folder, so every put also sends list requests. A folder another program made only by writing keys below it disappears with its last key. When a file and a folder share a path, the file wins: listings leave the folder out.

`read_stream`, `write_stream` and `append_stream` move a file in parts of 8 MiB, holding one part in memory at a time. A written file appears when its writer closes; a writer dropped without `close` logs an error and stores nothing. `copy` and `rename` run on the server.

```rust
use os_shim::bucket::{BlockingBucket, MinioBucket, MinioConfig};

let config = MinioConfig::new(
    "http://127.0.0.1:9000".to_owned(),
    "us-east-1".to_owned(),
    "notes".to_owned(),
    std::env::var("AWS_ACCESS_KEY_ID")?,
    std::env::var("AWS_SECRET_ACCESS_KEY")?,
);
let bucket = BlockingBucket::new(MinioBucket::new(config))?;
```

`S3Config::new(region, bucket, access_key, secret_key)` builds an `S3Bucket` the same way. `with_prefix` on either configuration puts every key under a prefix. Both configurations print the secret key as `<redacted>` in `Debug`. `ensure` on either client creates the bucket when it does not exist; `S3Bucket::ensure` makes it private and takes an optional KMS key to encrypt it with.

`BlockingBucket` runs each call on a tokio runtime of its own, so synchronous code needs no runtime; calling it from inside a tokio runtime panics. The `Bucket` trait itself is public and asynchronous code awaits the clients directly. A bucket example:

```rust
use os_shim::bucket::{Bucket as _, Condition, MemoryBucket};
use std::path::Path;

let bucket = MemoryBucket::default();
bucket.create_folder(Path::new("/notes")).await?;
let token = bucket.put(Path::new("/notes/a.md"), b"a", &Condition::None).await?;
assert_eq!(bucket.get(Path::new("/notes/a.md")).await?.1, token);
```

A program that picks its backend at run time holds an `Arc<dyn Bucket>`.

### What each backend promises

Backends differ in what they can promise. `guarantees()` on a `Bucket` or a `System` gives the set this one holds:

```rust
use os_shim::bucket::{Bucket as _, Guarantee, MemoryBucket};

let bucket = MemoryBucket::default();
assert!(bucket.guarantees().contains(Guarantee::FolderRenameIsAtomic));
```

`DiskBucket` and `MemoryBucket` hold most of them, such as `FileRenameIsAtomic`, `DroppedWriterKeepsBytes` and `RenameKeepsModificationTime`. On the disk, a rename is atomic only within one file system, and the conditional checks are one step only among callers of os-shim, under an advisory lock. `MinioBucket` and `S3Bucket` hold `ConditionalWriteInOneStep`, `CreateOnlyWriteInOneStep` and `EmptyFoldersExist`, and `S3Bucket` also holds `ConditionalDeleteInOneStep`. A rename on either client is a copy and then a delete, so it is not atomic and the file takes the time of the copy.

## Tests

The in-memory twins are held to their real counterparts by parity tests: the same sequence of operations runs against both and must give the same answers. The `System` sequences run on the real filesystem in the ordinary test run. On a server or on AWS, they and the bucket sequences run only when asked:

| Recipe | Runs |
|---|---|
| `just all` | formatting, clippy and tests for every combination of features, and the build |
| `just test-slow` | the tests too expensive for every run, such as a four-gibibyte zip64 archive |
| `just test-minio` | the parity tests on `MinioBucket`, against a RustFS server in a container it starts and removes (needs docker) |
| `just test-s3` | the parity tests on `S3Bucket`, against the AWS bucket named by `OS_SHIM_S3_REGION` and `OS_SHIM_S3_BUCKET`, with `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`; one test creates and deletes a bucket in `OS_SHIM_S3_ENSURE_REGION`, encrypted with `OS_SHIM_S3_KMS_KEY_ARN` |

## License

MIT
