use autoagents::async_trait;
use autoagents::core::tool::{ToolCallError, ToolRuntime};
use autoagents::prelude::ToolT;
use autoagents_derive::{ToolInput, tool};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Serialize, Deserialize, ToolInput)]
pub struct BinaryMathArgs {
    #[input(description = "The left-hand integer operand")]
    pub left: i64,
    #[input(description = "The right-hand integer operand")]
    pub right: i64,
}

#[tool(
    name = "AddNumbers",
    description = "Add two integers and return the result",
    input = BinaryMathArgs,
    output = i64,
)]
#[derive(Default)]
pub struct AddNumbers;

#[async_trait]
impl ToolRuntime for AddNumbers {
    async fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        let args: BinaryMathArgs = serde_json::from_value(args)?;
        Ok(Value::from(args.left + args.right))
    }
}

#[tool(
    name = "MultiplyNumbers",
    description = "Multiply two integers and return the result",
    input = BinaryMathArgs,
    output = i64,
)]
#[derive(Default)]
pub struct MultiplyNumbers;

#[async_trait]
impl ToolRuntime for MultiplyNumbers {
    async fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        let args: BinaryMathArgs = serde_json::from_value(args)?;
        Ok(Value::from(args.left * args.right))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn binary_math_tools_execute_expected_results() {
        let sum = AddNumbers
            .execute(json!({"left": 7, "right": 5}))
            .await
            .expect("addition should succeed");
        assert_eq!(sum, json!(12));

        let product = MultiplyNumbers
            .execute(json!({"left": 7, "right": 5}))
            .await
            .expect("multiplication should succeed");
        assert_eq!(product, json!(35));
    }
}
