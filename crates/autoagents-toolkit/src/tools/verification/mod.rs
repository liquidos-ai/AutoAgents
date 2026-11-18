//! Verification tools for coding agents.
//!
//! [`RunTests`] runs a test command chosen by the developer (never by the model) and returns
//! a signed [treeproof](https://crates.io/crates/treeproof) receipt bound to the git tree of the
//! workspace: an in-toto test-result attestation that says this exact tree passed (or failed)
//! this exact command. [`CheckVerified`] tells the agent whether a passing receipt exists for
//! the workspace as it is now, so "the tests pass" is something the agent looks up instead of
//! something it asserts.
//!
//! Runs with 0 tests, an unrecognised runner, a failing test or a non-zero exit are `FAILED`.
//! Passing results are cached by tree, command, toolchain versions and lockfiles, and shared
//! across worktrees, so repeated calls on an unchanged tree return at once.

mod check_verified;
mod run_tests;

pub use check_verified::CheckVerified;
pub use run_tests::RunTests;

use autoagents::core::tool::ToolCallError;

pub(crate) fn runtime_error(message: impl Into<String>) -> ToolCallError {
    ToolCallError::RuntimeError(message.into().into())
}
