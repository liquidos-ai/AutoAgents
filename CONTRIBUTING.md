# Contributing to AutoAgents

Thank you for your interest in contributing to AutoAgents! We welcome contributions from the community and are excited to see what you build.

When contributing to this repository, please first discuss the change you wish to make via issue, email, or any other method with the owners of this repository before making a change.

Please note we have a [Code of Conduct](CODE_OF_CONDUCT.md), please follow it in all your interactions with the project.

## Getting Started

### Development Setup

Please follow the installation instructions in our [README.md](README.md#-installation) to set up your development environment. This includes:

1. Installing Rust and Cargo
2. Installing LeftHook for Git hooks
3. Cloning the repository and setting up the project

### Prerequisites

- Rust (latest stable recommended)
- Cargo package manager  
- LeftHook for Git hooks management
- Cargo Tarpaulin for test coverage (optional)

```bash
# Install lefthook if not already installed
brew install lefthook  # macOS
# or
npm install -g lefthook  # Cross-platform

# Clone and setup
git clone https://github.com/liquidos-ai/AutoAgents.git
cd AutoAgents
lefthook install
cargo build --release
cargo test --all-features
```

## Development Workflow

### Code Quality

This project uses LeftHook to manage Git hooks that ensure code quality:

- **Formatting**: `cargo fmt --check` ensures consistent code formatting
- **Linting**: `cargo clippy -- -D warnings` catches common mistakes and enforces best practices  
- **Testing**: `cargo test --features full` runs the test suite

These hooks run automatically before commits. You can also run them manually:

```bash
# Format code
cargo fmt

# Run lints
cargo clippy --all-features

# Run tests
cargo test --all-features

# Run all checks (same as pre-commit hook)
lefthook run pre-commit
```

### Testing

```bash
# Run all tests
cargo test --all-features

# Run with coverage
cargo install cargo-tarpaulin
cargo tarpaulin --all-features --out html
```

## Pull Request Process

1. Fork the repository and create your feature branch from `main`
2. Ensure any install or build dependencies are removed before the end of the layer when doing a build
3. Make your changes and ensure they follow our coding standards
4. Add tests for any new functionality
5. Update the README.md with details of changes to the interface, including new environment variables, exposed ports, useful file locations and container parameters
6. Update version numbers in any example files and the README.md to the new version that this Pull Request would represent. We use [SemVer](http://semver.org/) versioning
7. Ensure all tests pass and pre-commit hooks succeed
8. Create a pull request targeting the `main` branch
9. You may merge the Pull Request once you have the sign-off of two other developers, or if you do not have permission to do that, you may request the second reviewer to merge it for you

### Before Submitting

- [ ] All tests pass (`cargo test --all-features`)
- [ ] Code is properly formatted (`cargo fmt`)
- [ ] No linting errors (`cargo clippy --all-features`)
- [ ] Documentation is updated
- [ ] Pre-commit hooks pass (`lefthook run pre-commit`)
