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
    cargo clippy --all-targets --features zip,process -- -D warnings

# Run tests across every feature combination that ships
test-features:
    @echo "Testing each feature combination..."
    cargo test --no-default-features
    cargo test --features process
    cargo test --features zip
    cargo test --features zip,process

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
