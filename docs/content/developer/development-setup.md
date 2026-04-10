# Development Setup

If you want to contribute to AutoAgents or build from source, follow these additional steps:

### Additional Prerequisites

- **LeftHook** - Git hooks manager for code quality
- **Cargo Tarpaulin** - Test coverage tool (optional)
- **Python 3.12+** - required for Python bindings development
- **maturin** - build backend for PyO3 packages
- **uv** - Python package manager

### Installing LeftHook

LeftHook is essential for maintaining code quality and is required for development.

**macOS (using Homebrew):**
```bash
brew install lefthook
```

**Linux (Ubuntu/Debian):**
```bash
# using npm
npm install -g lefthook
```

### Clone and Setup Repository

```bash
# Clone the repository
git clone https://github.com/liquidos-ai/AutoAgents.git
cd AutoAgents

# Install Git hooks using lefthook
lefthook install

# Build the project
cargo build --release

# Run tests to verify setup
cargo test --all-features
```

### Python Bindings Development Setup

```bash
# From repository root
 uv venv --python=3.12
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -U pip maturin pytest pytest-asyncio pytest-cov
```

Build and install all CPU packages using the repository Makefile:

```bash
make python-bindings-build

# With CUDA (requires CUDA Toolkit 12.x)
make python-bindings-build-cuda
```

See [Python Bindings](../getting-started/python-bindings.md) for the full
local testing guide including protocol events and GPU variants.

### Installing Additional Development Tools

```bash
# For test coverage (optional)
cargo install cargo-tarpaulin

# Install docs dependencies
cd docs
npm install

# For security auditing (recommended)
cargo install cargo-audit
```

## System Dependencies

### macOS

```bash
# Install Xcode command line tools (if not already installed)
xcode-select --install

# Install additional dependencies via Homebrew
brew install pkg-config openssl
```

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    curl \
    git
```

### Windows

Install the following:

1. **Visual Studio Build Tools** or **Visual Studio Community** with C++ build tools
2. **Git for Windows**
3. **Windows Subsystem for Linux (WSL)** - recommended for better compatibility

## Verification

After installation, verify everything is working:

```bash
# Check Rust installation
cargo --version
rustc --version

# Check lefthook installation (for development)
lefthook --version

# Build AutoAgents
cd AutoAgents
cargo build --all-features

# Run tests
cargo test --all-features

# Check git hooks are installed (for development)
lefthook run pre-commit
```

## Git Hooks (Development)

The project uses LeftHook to manage Git hooks that ensure code quality:

### Pre-commit Hooks
- **Formatting**: `cargo fmt --check` - Ensures consistent code formatting
- **Linting**: `cargo clippy --all-features --all-targets -- -D warnings` - Catches common mistakes
- **Testing**: `cargo test --all-features` - Runs the test suite
- **Type Checking**: `cargo check --all-features --all-targets` - Validates compilation

### Pre-push Hooks
- **Full Testing**: `cargo test --all-features --release` - Comprehensive test suite
- **Documentation**: `cargo doc --all-features --no-deps` - Ensures docs build correctly

## Running Tests with Coverage

```bash
# Install tarpaulin if not already installed
cargo install cargo-tarpaulin
rustup component add llvm-tools-preview

# Run tests with coverage
make coverage-rust
```


## Documentation

Build the docs locally with Docusaurus:

```bash
cd docs
npm start
```
