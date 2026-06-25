use std::future::Future;
use std::time::Duration;

use tokio::time;

use super::McpError;

/// Apply a timeout to an async MCP operation.
pub async fn with_timeout<T, F, E>(
    duration: Duration,
    operation: &'static str,
    fut: F,
) -> Result<T, McpError>
where
    F: Future<Output = Result<T, E>>,
    E: Into<McpError>,
{
    match time::timeout(duration, fut).await {
        Ok(result) => result.map_err(Into::into),
        Err(_) => Err(McpError::Timeout {
            operation,
            millis: duration.as_millis() as u64,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn with_timeout_returns_error_on_elapsed() {
        let err = with_timeout(Duration::from_millis(10), "test_op", async {
            time::sleep(Duration::from_secs(1)).await;
            Ok::<(), McpError>(())
        })
        .await
        .unwrap_err();

        assert!(matches!(
            err,
            McpError::Timeout {
                operation: "test_op",
                millis: 10
            }
        ));
    }

    #[tokio::test]
    async fn with_timeout_returns_ok_before_deadline() {
        let value = with_timeout(Duration::from_secs(1), "test_op", async {
            Ok::<i32, McpError>(7)
        })
        .await
        .unwrap();
        assert_eq!(value, 7);
    }
}
