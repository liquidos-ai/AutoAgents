//! Helpers for [`Environment`](autoagents::core::environment::Environment) lifecycle
//! in long-running examples.
//!
//! `Environment::wait()` is safe inside `tokio::select!`: if another branch wins,
//! the join handle is restored so [`shutdown_environment`] can still drain runtimes.

use autoagents::core::environment::Environment;
use autoagents::core::runtime::RuntimeError;
use std::time::Duration;
use tokio::task::JoinError;

/// Shut down the environment and log failures instead of discarding them.
pub async fn shutdown_environment(environment: &mut Environment) {
    if let Err(err) = environment.shutdown().await {
        eprintln!("Environment shutdown failed: {err}");
    }
}

fn log_wait_result(result: Result<Result<(), RuntimeError>, JoinError>) {
    match result {
        Ok(Ok(())) => {}
        Ok(Err(err)) => eprintln!("Environment run failed: {err}"),
        Err(err) => eprintln!("Environment run task join failed: {err}"),
    }
}

/// Wait for the managed run task to finish, or shut down gracefully on Ctrl+C.
pub async fn wait_or_ctrl_c(environment: &mut Environment, success_message: &str) {
    tokio::select! {
        result = environment.wait() => {
            log_wait_result(result);
            println!("{success_message}");
        }
        _ = tokio::signal::ctrl_c() => {
            println!("\nCtrl+C detected. Shutting down...");
            shutdown_environment(environment).await;
        }
    }
}

/// Wait for the managed run task to finish, or shut down on Ctrl+C or after `timeout`.
pub async fn wait_ctrl_c_or_timeout(
    environment: &mut Environment,
    timeout: Duration,
    success_message: &str,
    timeout_message: &str,
) {
    tokio::select! {
        result = environment.wait() => {
            log_wait_result(result);
            println!("{success_message}");
        }
        _ = tokio::time::sleep(timeout) => {
            println!("{timeout_message}");
            shutdown_environment(environment).await;
        }
        _ = tokio::signal::ctrl_c() => {
            println!("\nCtrl+C detected. Shutting down...");
            shutdown_environment(environment).await;
        }
    }
}
