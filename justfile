# os-shim project justfile
# Run `just --list` to see all available commands

# Default recipe - show help
default:
    @just --list

# Run code formatting
fmt:
    @echo "Formatting code..."
    cargo fmt --all

# Check code formatting without making changes
fmt-check:
    @echo "Checking code formatting..."
    cargo fmt --all -- --check

# Run linting with clippy (lint config lives in Cargo.toml [lints.clippy]).
# `-D warnings` promotes rustc's own warn-level lints -- unused imports, dead
# fields, unfulfilled lint expectations -- to errors. The [lints.clippy] table
# only governs clippy's groups, so without this a plain rustc warning passes the
# gate.
lint:
    @echo "Running clippy lints..."
    cargo clippy --all-targets --all-features -- -D warnings

# Run all tests
test:
    @echo "Running tests..."
    cargo test --all-features

# Lint every feature combination that ships. `lint` alone only covers
# --all-features, so a warning that appears only without a feature slips past it.
lint-features:
    @echo "Linting each feature combination..."
    cargo clippy --all-targets --no-default-features -- -D warnings
    cargo clippy --all-targets --features process -- -D warnings
    cargo clippy --all-targets --features zip -- -D warnings
    cargo clippy --all-targets --features minio -- -D warnings
    cargo clippy --all-targets --features s3 -- -D warnings
    cargo clippy --all-targets --features async -- -D warnings
    cargo clippy --all-targets --features process,zip -- -D warnings
    cargo clippy --all-targets --features process,minio -- -D warnings
    cargo clippy --all-targets --features process,s3 -- -D warnings
    cargo clippy --all-targets --features process,async -- -D warnings
    cargo clippy --all-targets --features zip,minio -- -D warnings
    cargo clippy --all-targets --features zip,s3 -- -D warnings
    cargo clippy --all-targets --features zip,async -- -D warnings
    cargo clippy --all-targets --features minio,s3 -- -D warnings
    cargo clippy --all-targets --features minio,async -- -D warnings
    cargo clippy --all-targets --features s3,async -- -D warnings
    cargo clippy --all-targets --features process,zip,minio -- -D warnings
    cargo clippy --all-targets --features process,zip,s3 -- -D warnings
    cargo clippy --all-targets --features process,zip,async -- -D warnings
    cargo clippy --all-targets --features process,minio,s3 -- -D warnings
    cargo clippy --all-targets --features process,minio,async -- -D warnings
    cargo clippy --all-targets --features process,s3,async -- -D warnings
    cargo clippy --all-targets --features zip,minio,s3 -- -D warnings
    cargo clippy --all-targets --features zip,minio,async -- -D warnings
    cargo clippy --all-targets --features zip,s3,async -- -D warnings
    cargo clippy --all-targets --features minio,s3,async -- -D warnings
    cargo clippy --all-targets --features process,zip,minio,s3 -- -D warnings
    cargo clippy --all-targets --features process,zip,minio,async -- -D warnings
    cargo clippy --all-targets --features process,zip,s3,async -- -D warnings
    cargo clippy --all-targets --features process,minio,s3,async -- -D warnings
    cargo clippy --all-targets --features zip,minio,s3,async -- -D warnings
    cargo clippy --all-targets --features process,zip,minio,s3,async -- -D warnings

# Run tests across every feature combination that ships
test-features:
    @echo "Testing each feature combination..."
    cargo test --no-default-features
    cargo test --features process
    cargo test --features zip
    cargo test --features minio
    cargo test --features s3
    cargo test --features async
    cargo test --features process,zip
    cargo test --features process,minio
    cargo test --features process,s3
    cargo test --features process,async
    cargo test --features zip,minio
    cargo test --features zip,s3
    cargo test --features zip,async
    cargo test --features minio,s3
    cargo test --features minio,async
    cargo test --features s3,async
    cargo test --features process,zip,minio
    cargo test --features process,zip,s3
    cargo test --features process,zip,async
    cargo test --features process,minio,s3
    cargo test --features process,minio,async
    cargo test --features process,s3,async
    cargo test --features zip,minio,s3
    cargo test --features zip,minio,async
    cargo test --features zip,s3,async
    cargo test --features minio,s3,async
    cargo test --features process,zip,minio,s3
    cargo test --features process,zip,minio,async
    cargo test --features process,zip,s3,async
    cargo test --features process,minio,s3,async
    cargo test --features zip,minio,s3,async
    cargo test --features process,zip,minio,s3,async

# The opt-in expensive tests compile only under `cfg(slow_tests)`, which no other
# recipe sets.
slow_tests := "--all-features --config 'build.rustflags=[\"--cfg\", \"slow_tests\"]'"

# Lint, then run, the opt-in expensive tests under the same flags. Wants a
# disk-backed TMPDIR and double-digit gibibytes of headroom; takes minutes, not
# seconds.
test-slow:
    @echo "Linting expensive tests..."
    cargo clippy --all-targets {{slow_tests}} -- -D warnings
    @echo "Running expensive tests..."
    cargo test {{slow_tests}} -- --nocapture

# The `minio` client's tests compile only under `cfg(minio_tests)`, which no other
# recipe sets.
minio_tests := "--features minio --config 'build.rustflags=[\"--cfg\", \"minio_tests\"]'"

# Lint, then run, the `minio` client's tests against a RustFS server in a
# container of their own on port 9100, removed afterwards. Needs docker.
test-minio:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Linting MinIO tests..."
    cargo clippy --all-targets {{minio_tests}} -- -D warnings
    docker run -d --rm --name os-shim-rustfs -p 9100:9000 \
        -e RUSTFS_ACCESS_KEY=os-shim -e RUSTFS_SECRET_KEY=os-shim-secret \
        rustfs/rustfs:1.0.0-beta.8 > /dev/null
    trap 'docker stop os-shim-rustfs > /dev/null' EXIT
    for _ in $(seq 1 50); do
        curl -s -o /dev/null http://127.0.0.1:9100/ && break
        sleep 0.2
    done
    echo "Running MinIO tests against RustFS..."
    OS_SHIM_MINIO_ENDPOINT=http://127.0.0.1:9100 \
    OS_SHIM_MINIO_ACCESS_KEY=os-shim \
    OS_SHIM_MINIO_SECRET_KEY=os-shim-secret \
        cargo test {{minio_tests}} bucket::parity_tests -- --nocapture

# The `s3` client's tests compile only under `cfg(s3_tests)`, which no other
# recipe sets.
s3_tests := "--features s3 --config 'build.rustflags=[\"--cfg\", \"s3_tests\"]'"

# Lint, then run, the `s3` client's tests against the AWS bucket the environment
# names. Writes under a prefix of its own and removes what it wrote. One test
# creates a bucket in OS_SHIM_S3_ENSURE_REGION, encrypted with the KMS key
# OS_SHIM_S3_KMS_KEY_ARN, and deletes it.
test-s3:
    #!/usr/bin/env bash
    set -euo pipefail
    missing=()
    for name in OS_SHIM_S3_REGION OS_SHIM_S3_BUCKET OS_SHIM_S3_ENSURE_REGION OS_SHIM_S3_KMS_KEY_ARN AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY; do
        [ -n "${!name:-}" ] || missing+=("$name")
    done
    if [ "${#missing[@]}" -gt 0 ]; then
        echo "test-s3 needs: ${missing[*]}" >&2
        exit 1
    fi
    echo "Linting S3 tests..."
    cargo clippy --all-targets {{s3_tests}} -- -D warnings
    echo "Running S3 tests against AWS..."
    cargo test {{s3_tests}} bucket::parity_tests -- --nocapture

# Run tests with output
test-verbose:
    @echo "Running tests (verbose)..."
    cargo test --all-features -- --nocapture

# Generate code coverage report (requires cargo-llvm-cov)
test-coverage:
    @echo "Generating code coverage report..."
    cargo llvm-cov --workspace --all-features

# Generate code coverage HTML report (requires cargo-llvm-cov)
test-coverage-html:
    @echo "Generating code coverage HTML report..."
    cargo llvm-cov --workspace --all-features --html
    @echo "Coverage report generated in target/llvm-cov/html/"

# Build in debug mode
build:
    @echo "Building..."
    cargo build --all-features

# Build optimized release
build-release:
    @echo "Building release..."
    cargo build --release --all-features

# Check for security vulnerabilities
audit:
    @echo "Checking for security vulnerabilities..."
    cargo audit

# Clean build artifacts
clean:
    @echo "Cleaning build artifacts..."
    cargo clean

# Generate documentation
docs:
    @echo "Generating documentation..."
    cargo doc --all-features --no-deps

# Full pipeline (standardized `all` entry point across tixena repos).
all: fmt-check lint lint-features test-features build
    @echo "All checks completed successfully!"
