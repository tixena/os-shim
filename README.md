# os-shim

A mockable layer between Rust code and the machine it runs on. Code written against os-shim's traits runs on the real system in production and on an in-memory twin in tests, with no filesystem, server or clock to set up.

| Piece | Real implementation | In-memory twin | Feature |
|---|---|---|---|
| Environment, files, folders, temporary directories (`System`) | `RealSystem` | `MemorySystem` | always |
| A wait that does not hold the thread (`System::sleep_async`) | `RealSystem`, on tokio's timer | `MemorySystem` | `async` |
| Whether a process is alive (`System::is_pid_alive`) | `RealSystem` | `MemorySystem::with_pid` | `process` |
| Zip archives of a tree, and trees loaded from archives | `archive`, `System::to_zip_*` | the same, over `MemorySystem` | `zip` |
| Objects in a bucket of an S3-compatible store | `MinioBucket`, `S3Bucket` | `MemoryBucket` | `minio`, `s3` |

No feature is on by default.

## The system

```rust
use os_shim::System;
use os_shim::mock::MemorySystem;
use std::path::Path;

fn config_path(system: &dyn System) -> std::io::Result<String> {
    let home = system.env_var("HOME").map_err(std::io::Error::other)?;
    system.read_to_string(&Path::new(&home).join(".app.toml"))
}

let system = MemorySystem::new()
    .with_env("HOME", "/home/user")?
    .with_file("/home/user/.app.toml", b"debug = true")?;
assert_eq!(config_path(&system)?, "debug = true");
```

In production the same function takes `&os_shim::real::RealSystem::new()`, which delegates to `std::env` and `std::fs`. `System` covers environment variables, the current directory and executable, reading, writing, appending, copying, renaming and removing files, removing a file only while it holds what the caller read, folders and their walks, metadata, temporary directories that are removed on drop, the current time, and a wait that `MemorySystem` answers by moving its clock. `MemorySystem`'s clock starts at the Unix epoch; `with_now` sets it, `advance` moves it forward, and each file `MemorySystem` writes takes its modification time from it.

## Zip archives (`zip`)

```rust
use os_shim::System;
use os_shim::mock::MemorySystem;
use std::io::Cursor;
use std::path::Path;

let system = MemorySystem::new().with_file("/project/src/main.rs", b"fn main() {}")?;
let archive = system.to_zip_vec(Path::new("/project"))?;

let loaded = MemorySystem::from_zip_reader(Cursor::new(archive))?;
assert_eq!(loaded.read_to_string(Path::new("/src/main.rs"))?, "fn main() {}");
```

`System::to_zip_stream`, `to_zip_vec` and `to_zip_writer` archive a tree. `archive::load_zip` loads an archive from a stream, `archive::load_zip_seekable` from a seekable reader, including archives whose entries carry data descriptors. `archive::ZipStreamWriter` writes an archive one entry at a time into a sink that cannot seek.

## Buckets (`minio`, `s3`, `async`)

Whole objects in one bucket, each put and delete under an optional condition:

| `Condition` | A put or delete goes ahead only if |
|---|---|
| `None` | always |
| `Absent` | no object is at the key |
| `ETag(tag)` | the object at the key still has this entity tag |

A failed condition is an error whose `kind()` is `ObjectErrorKind::PreconditionFailed`; a missing object is `ObjectErrorKind::NotFound`.

```rust
use os_shim::bucket::{BlockingBucket, Condition, MemoryBucket, ObjectErrorKind};

let bucket = BlockingBucket::new(MemoryBucket::default())?;

let first = bucket.put("notes/task.md", b"# Task\n", &Condition::Absent)?;
let refused = bucket.put("notes/task.md", b"# Other\n", &Condition::Absent);
assert_eq!(refused.unwrap_err().kind(), ObjectErrorKind::PreconditionFailed);

let second = bucket.put("notes/task.md", b"# Done\n", &Condition::ETag(first.clone()))?;
let stale = bucket.delete("notes/task.md", &Condition::ETag(first));
assert_eq!(stale.unwrap_err().kind(), ObjectErrorKind::PreconditionFailed);

bucket.delete("notes/task.md", &Condition::ETag(second))?;
assert_eq!(bucket.get("notes/task.md").unwrap_err().kind(), ObjectErrorKind::NotFound);
```

The clients reach a server through AWS's SDK, `aws-sdk-s3`, and never retry a request:

- `MinioBucket` (`minio`) talks to a server that speaks MinIO's API, such as MinIO or RustFS, with path-style addressing. Some of these servers ignore `If-Match` on a delete, so a conditional delete reads the object's tag first and sends `If-Match` as well.
- `S3Bucket` (`s3`) talks to AWS S3, which checks `If-Match` on a delete itself.

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

`S3Config::new(region, bucket, access_key, secret_key)` builds an `S3Bucket` the same way. Both configurations print the secret key as `<redacted>` in `Debug`.

`BlockingBucket` runs each call on a tokio runtime of its own, so synchronous code needs no runtime; calling it from inside a tokio runtime panics. With the `async` feature, the `Bucket` trait itself is public and asynchronous code awaits the clients directly; it also turns on `System::sleep_async`, which `RealSystem` awaits on tokio's timer, inside a tokio runtime with its timer on. A bucket example:

```rust
use os_shim::bucket::{Bucket as _, Condition, MemoryBucket};

let bucket = MemoryBucket::default();
let tag = bucket.put("notes/a.md", b"a", &Condition::None).await?;
assert_eq!(bucket.get("notes/a.md").await?.etag, tag);
```

`Bucket` is not `dyn`-compatible: a program that picks its backend at run time builds the concrete client in each branch.

## Tests

The in-memory twins are held to their real counterparts by parity tests: the same sequence of operations runs against both and must give the same answers. `MemorySystem` is checked against the real filesystem in the ordinary test run. The bucket parity tests need a server and run only when asked:

| Recipe | Runs |
|---|---|
| `just all` | formatting, clippy and tests for every combination of features, and the build |
| `just test-slow` | the tests too expensive for every run, such as a four-gibibyte zip64 archive |
| `just test-minio` | `MinioBucket` against a RustFS server in a container it starts and removes (needs docker) |
| `just test-s3` | `S3Bucket` against the AWS bucket named by `OS_SHIM_S3_REGION`, `OS_SHIM_S3_BUCKET`, `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` |

## License

MIT
