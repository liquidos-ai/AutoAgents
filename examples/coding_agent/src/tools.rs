use autoagents::async_trait;
use autoagents::core::tool::{ToolCallError, ToolInputT, ToolRuntime, ToolT};
use autoagents_derive::{ToolInput, tool};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use walkdir::WalkDir;

#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct GrepArgs {
    #[input(description = "Regular expression pattern to search for")]
    pattern: String,
    #[input(description = "File glob pattern to search in (e.g., '*.rs')")]
    file_pattern: String,
    #[input(description = "Base directory to search in")]
    base_dir: String,
}

#[tool(
    name = "GrepTool",
    description = "Search for content in files using regex patterns",
    input = GrepArgs,
)]
pub struct GrepTool {}

#[async_trait]
impl ToolRuntime for GrepTool {
    async fn execute(&self, args: serde_json::Value) -> Result<serde_json::Value, ToolCallError> {
        let args: GrepArgs = serde_json::from_value(args)?;
        println!("🔎 Grepping for: {} in {}", args.pattern, args.file_pattern);

        let regex = Regex::new(&args.pattern)
            .map_err(|e| ToolCallError::RuntimeError(format!("Invalid regex: {}", e).into()))?;

        let base_path = Path::new(&args.base_dir);
        if !base_path.exists() {
            return Err(ToolCallError::RuntimeError(
                format!("Directory {} does not exist", args.base_dir).into(),
            ));
        }

        let file_pattern = glob::Pattern::new(&args.file_pattern).map_err(|e| {
            ToolCallError::RuntimeError(format!("Invalid file pattern: {}", e).into())
        })?;

        let mut results = Vec::new();
        let max_results = 50;

        for entry in WalkDir::new(&args.base_dir)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if results.len() >= max_results {
                break;
            }

            let path = entry.path();
            if path.is_file() {
                let relative_path = path.strip_prefix(&args.base_dir).unwrap_or(path);
                if file_pattern.matches_path(relative_path)
                    && let Ok(content) = fs::read_to_string(path)
                {
                    for (line_num, line) in content.lines().enumerate() {
                        if regex.is_match(line) {
                            results.push(format!(
                                "{}:{}: {}",
                                relative_path.display(),
                                line_num + 1,
                                line.trim()
                            ));
                            if results.len() >= max_results {
                                break;
                            }
                        }
                    }
                }
            }
        }

        if results.is_empty() {
            Ok("No matches found.".to_string().into())
        } else {
            Ok(format!(
                "Found {} matches (showing up to {}):\n{}",
                results.len(),
                max_results,
                results.join("\n")
            )
            .into())
        }
    }
}

#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct AnalyzeCodeArgs {
    #[input(description = "Path to the file or directory to analyze")]
    path: String,
    #[input(description = "Type of analysis: 'structure', 'complexity', 'dependencies'")]
    analysis_type: String,
}

#[tool(
    name = "AnalyzeCodeTool",
    description = "Analyze code structure, complexity, or dependencies",
    input = AnalyzeCodeArgs,
)]
pub struct AnalyzeCodeTool {}

#[async_trait]
impl ToolRuntime for AnalyzeCodeTool {
    async fn execute(&self, args: serde_json::Value) -> Result<serde_json::Value, ToolCallError> {
        let args: AnalyzeCodeArgs = serde_json::from_value(args)?;
        println!("🔬 Analyzing code: {} ({})", args.path, args.analysis_type);

        let path = Path::new(&args.path);
        if !path.exists() {
            return Err(ToolCallError::RuntimeError(
                format!("Path {} does not exist", args.path).into(),
            ));
        }

        match args.analysis_type.as_str() {
            "structure" => Ok(analyze_structure(path)?.into()),
            "complexity" => Ok(analyze_complexity(path)?.into()),
            "dependencies" => Ok(analyze_dependencies(path)?.into()),
            _ => Err(ToolCallError::RuntimeError(
                "Invalid analysis type. Choose 'structure', 'complexity', or 'dependencies'".into(),
            )),
        }
    }
}

fn analyze_structure(path: &Path) -> Result<String, ToolCallError> {
    let mut file_count = 0;
    let mut dir_count = 0;
    let mut total_lines = 0;
    let mut extensions: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

    if path.is_file() {
        file_count = 1;
        if let Ok(content) = fs::read_to_string(path) {
            total_lines = content.lines().count();
        }
        if let Some(ext) = path.extension() {
            *extensions
                .entry(ext.to_string_lossy().to_string())
                .or_insert(0) += 1;
        }
    } else {
        for entry in WalkDir::new(path).into_iter().filter_map(|e| e.ok()) {
            let entry_path = entry.path();
            if entry_path.is_file() {
                file_count += 1;
                if let Ok(content) = fs::read_to_string(entry_path) {
                    total_lines += content.lines().count();
                }
                if let Some(ext) = entry_path.extension() {
                    *extensions
                        .entry(ext.to_string_lossy().to_string())
                        .or_insert(0) += 1;
                }
            } else if entry_path.is_dir() {
                dir_count += 1;
            }
        }
    }

    let mut ext_summary = String::new();
    for (ext, count) in extensions.iter() {
        ext_summary.push_str(&format!("\n  .{}: {} files", ext, count));
    }

    Ok(format!(
        "Code Structure Analysis:\n\
        - Files: {}\n\
        - Directories: {}\n\
        - Total lines: {}\n\
        - File types:{}",
        file_count, dir_count, total_lines, ext_summary
    ))
}

fn analyze_complexity(_path: &Path) -> Result<String, ToolCallError> {
    // Simplified complexity analysis
    Ok(
        "Complexity analysis: This is a placeholder. In a real implementation, \
        this would calculate cyclomatic complexity, function lengths, and other metrics."
            .to_string(),
    )
}

fn analyze_dependencies(_path: &Path) -> Result<String, ToolCallError> {
    // Simplified dependency analysis
    Ok(
        "Dependency analysis: This is a placeholder. In a real implementation, \
        this would parse import statements and analyze module dependencies."
            .to_string(),
    )
}
