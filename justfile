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

# Run linting with clippy (lint config lives in Cargo.toml [lints.clippy])
lint:
    @echo "Running clippy lints..."
    cargo clippy --all-targets --all-features

# Run all tests
test:
    @echo "Running tests..."
    cargo test --all-features

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
all: fmt-check lint test build
    @echo "All checks completed successfully!"
